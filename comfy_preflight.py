#!/usr/bin/env python3
"""comfy-preflight: static pre-flight checker for ComfyUI workflows.

Analyzes a workflow JSON (UI or API format) WITHOUT ComfyUI installed and
reports:
  - missing custom node packs (curated map + optional ComfyUI Registry lookup)
  - model files the workflow needs and their target folders
  - a rough VRAM estimate with a verdict for your GPU size

Usage:
    python comfy_preflight.py workflow.json
    python comfy_preflight.py workflow.json --vram 6 --markdown > report.md

The tool only ever READS the workflow you point it at; all output goes to
stdout, so saving a report is just shell redirection.

Estimates are heuristics for fp16 checkpoints on NVIDIA cards - treat them
as order-of-magnitude guidance, not benchmarks.

Single-file, stdlib only. Python >= 3.9.
"""

import argparse
import json
import os
import re
import sys
import urllib.parse
import urllib.request

__version__ = "0.1.0"
REGISTRY_HOST = "https://api.comfy.org"
# class names are identifiers, not free text: allowlist before they reach a URL
_CLASS_NAME_RE = re.compile(r"^[A-Za-z0-9_ .:\-]{1,160}$")

MODEL_EXTS = (".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".sft")


def resolve_read_path(path):
    """Resolve the workflow path to read; reject parent-dir segments outright."""
    raw_segs = str(path).replace("\\", "/").split("/")
    if os.path.pardir in raw_segs:
        raise SystemExit("error: workflow path must not contain parent-dir segments")
    resolved = os.path.realpath(os.path.abspath(path))
    if not os.path.isfile(resolved):
        raise SystemExit("error: workflow file not found: %s" % path)
    return resolved


# --- Which core (built-in) ComfyUI nodes exist. Anything not listed here and
# --- not in PACK_MAP is reported as "needs a custom node pack".
CORE_NODES = set("""
CheckpointLoader CheckpointLoaderSimple unCLIPCheckpointLoader ImageOnlyCheckpointLoader
CheckpointLoaderNF4 UNETLoader DualCLIPLoader TripleCLIPLoader QuadrupleCLIPLoader
CLIPLoader CLIPVisionLoader VAELoader LoraLoader LoraLoaderModelOnly DiffusersLoader
GLIGENLoader ControlNetLoader DiffControlNetLoader StyleModelLoader UpscaleModelLoader
CLIPTextEncode CLIPTextEncodeSDXL CLIPTextEncodeSDXLRefiner CLIPTextEncodeSD3
CLIPTextEncodeFlux CLIPTextEncodeHunyuanDiT CLIPSetLastLayer CLIPVisionEncode
ConditioningCombine ConditioningConcat ConditioningAverage ConditioningZeroOut
ConditioningSetArea ConditioningSetAreaPercentage ConditioningSetAreaStrength
ConditioningSetMask ConditioningSetTimestepRange unCLIPConditioning StyleModelApply
GLIGENTextBoxApply ControlNetApply ControlNetApplyAdvanced ControlNetApplySD3
SetUnionControlNetType InpaintModelConditioning FluxGuidance FluxKontextImageScale
EmptyLatentImage EmptyLatentImageLarge EmptySD3LatentImage EmptyHunyuanLatentVideo
EmptyMochiLatentVideo EmptyLTXVLatentVideo EmptyCosmosLatentVideo LatentUpscale
LatentUpscaleBy LatentFromBatch RepeatLatentBatch LatentComposite LatentBlend
LatentFlip LatentRotate LatentCrop SetLatentNoiseMask LatentAdd LatentSubtract
LatentMultiply LatentInterpolate LatentBatch LatentBatchSeedBehavior LatentApplyOperation
KSampler KSamplerAdvanced SamplerCustom SamplerCustomAdvanced KSamplerSelect
SamplerEulerAncestral SamplerDPMPP_2M_SDE SamplerDPMPP_3M_SDE
BasicScheduler KarrasScheduler ExponentialScheduler PolyexponentialScheduler
LaplaceScheduler VPScheduler SGDUniformScheduler BetaSamplingScheduler
AlignYourStepsScheduler GITSScheduler SplitSigmas FlipSigmas SetFirstSigma
CFGGuider DualCFGGuider BasicGuider PerpNegGuider RandomNoise DisableNoise AddNoise
RescaleCFG ModelSamplingDiscrete ModelSamplingContinuousEDM ModelSamplingFlux
ModelSamplingSD3 ModelSamplingAuraFlow ModelSamplingLTXV ModelSamplingStableCascade
LoadImage LoadImageMask LoadImageOutput SaveImage PreviewImage
SaveAnimatedWEBP SaveAnimatedPNG SaveImageWebsocket LoadVideo LoadVideoUpload
CreateVideo SaveVideo GetVideoComponents ImageUpscale ImageUpscaleWithModel
ImageScale ImageScaleBy ImageInvert ImageBatch ImagePadForOutpaint ImageCrop
ImageBlur ImageQuantize ImageSharpen ImageColorToMask ImageToMask MaskToImage
ImageCompositeMasked ImageBlend ImageBlendV2 ImageMaskComposite MaskComposite
FeatherMask GrowMask GrowMaskWithBlur InvertMask SolidMask ThresholdMask CropMask
EmptyImage LoadAudio SaveAudio PreviewAudio RecordAudio
DifferentialDiffusion PatchSageAttention
Note MarkdownNote PrimitiveNode PrimitiveInt PrimitiveFloat PrimitiveString
PrimitiveStringMultiline PrimitiveBoolean Reroute ImageReroute
VAEDecode VAEDecodeTiled VAEEncode VAEEncodeTiled VAEEncodeForInpaint
VAEDecodeBatched SVD_img2vid_Conditioning SVD_img2vid VideoLinearCFGGuidance
""".split())

# --- Curated map: well-known custom node class -> (pack name, repo slug).
# --- Kept small and high-confidence; the Registry API (online mode) covers the rest.
PACK_MAP = {
    "UnetLoaderGGUF": ("ComfyUI-GGUF", "city96/ComfyUI-GGUF"),
    "UnetLoaderGGUFAdvanced": ("ComfyUI-GGUF", "city96/ComfyUI-GGUF"),
    "CLIPLoaderGGUF": ("ComfyUI-GGUF", "city96/ComfyUI-GGUF"),
    "DualCLIPLoaderGGUF": ("ComfyUI-GGUF", "city96/ComfyUI-GGUF"),
    "TripleCLIPLoaderGGUF": ("ComfyUI-GGUF", "city96/ComfyUI-GGUF"),
    "IPAdapter": ("ComfyUI_IPAdapter_plus", "cubiq/ComfyUI_IPAdapter_plus"),
    "IPAdapterAdvanced": ("ComfyUI_IPAdapter_plus", "cubiq/ComfyUI_IPAdapter_plus"),
    "IPAdapterModelLoader": ("ComfyUI_IPAdapter_plus", "cubiq/ComfyUI_IPAdapter_plus"),
    "InstantID": ("ComfyUI_InstantID", "cubiq/ComfyUI_InstantID"),
    "ApplyInstantID": ("ComfyUI_InstantID", "cubiq/ComfyUI_InstantID"),
    "FaceDetailer": ("ComfyUI-Impact-Pack", "ltdrdata/ComfyUI-Impact-Pack"),
    "DetailerForEach": ("ComfyUI-Impact-Pack", "ltdrdata/ComfyUI-Impact-Pack"),
    "ImpactWildcardProcessor": ("ComfyUI-Impact-Pack", "ltdrdata/ComfyUI-Impact-Pack"),
    "UltralyticsDetectorProvider": ("ComfyUI-Impact-Subpack", "ltdrdata/ComfyUI-Impact-Subpack"),
    "ReActorFaceSwap": ("comfyui-reactor", "Gourieff/comfyui-reactor"),
    "ReActorFastFaceSwap": ("comfyui-reactor", "Gourieff/comfyui-reactor"),
    "UltimateSDUpscale": ("ComfyUI_UltimateSDUpscale", "ssitu/ComfyUI_UltimateSDUpscale"),
    "UltimateSDUpscaleNoUpscale": ("ComfyUI_UltimateSDUpscale", "ssitu/ComfyUI_UltimateSDUpscale"),
    "AnimateDiffLoaderGen1": ("ComfyUI-AnimateDiff-Evolved", "Kosinkadink/ComfyUI-AnimateDiff-Evolved"),
    "ADE_AnimateDiffLoaderGen1": ("ComfyUI-AnimateDiff-Evolved", "Kosinkadink/ComfyUI-AnimateDiff-Evolved"),
    "VideoCombine": ("ComfyUI-VideoHelperSuite", "Kosinkadink/ComfyUI-VideoHelperSuite"),
    "VHS_VideoCombine": ("ComfyUI-VideoHelperSuite", "Kosinkadink/ComfyUI-VideoHelperSuite"),
    "MiDaS-DepthMapPreprocessor": ("comfyui_controlnet_aux", "Fannovel16/comfyui_controlnet_aux"),
    "CannyEdgePreprocessor": ("comfyui_controlnet_aux", "Fannovel16/comfyui_controlnet_aux"),
    "OpenPosePreprocessor": ("comfyui_controlnet_aux", "Fannovel16/comfyui_controlnet_aux"),
    "DWPreprocessor": ("comfyui_controlnet_aux", "Fannovel16/comfyui_controlnet_aux"),
    "TilePreprocessor": ("comfyui_controlnet_aux", "Fannovel16/comfyui_controlnet_aux"),
    "DepthAnythingV2Preprocessor": ("comfyui_controlnet_aux", "Fannovel16/comfyui_controlnet_aux"),
    "LayerUtility: ImageScale": ("ComfyUI_LayerStyle", "chflame163/ComfyUI_LayerStyle"),
    "WanVideo Sampler": ("ComfyUI-WanVideoWrapper", "kijai/ComfyUI-WanVideoWrapper"),
    "WanVideoModelLoader": ("ComfyUI-WanVideoWrapper", "kijai/ComfyUI-WanVideoWrapper"),
    "Florence2ModelLoader": ("ComfyUI-Florence2", "kijai/ComfyUI-Florence2"),
    "SUPIR_upscale": ("ComfyUI-SUPIR", "kijai/ComfyUI-SUPIR"),
    "LoraLoaderBlockWeight": ("ComfyUI-Inspire-Pack", "ltdrdata/ComfyUI-Inspire-Pack"),
}

# --- Loader classes whose widget order we know, for precise model extraction.
# --- slot name -> canonical model folder
SLOT_DIRS = {
    "ckpt_name": "checkpoints",
    "vae_name": "vae",
    "lora_name": "loras",
    "unet_name": "diffusion_models",
    "diffusion_model": "diffusion_models",
    "clip_name": "text_encoders",
    "clip_name1": "text_encoders",
    "clip_name2": "text_encoders",
    "clip_name3": "text_encoders",
    "clip_name4": "text_encoders",
    "control_net_name": "controlnet",
    "control_net": "controlnet",
    "model_name": "upscale_models",
    "clip_vision": "clip_vision",
    "style_model_name": "style_models",
    "gligen_name": "gligen",
    "ipadapter": "ipadapter",
    "instantid": "instantid",
    "insightface": "insightface",
}
# loader class -> ordered widget slot names (UI format widgets_values)
WIDGET_MAP = {
    "CheckpointLoaderSimple": ["ckpt_name"],
    "CheckpointLoader": ["ckpt_name", "config_name"],
    "ImageOnlyCheckpointLoader": ["ckpt_name"],
    "unCLIPCheckpointLoader": ["ckpt_name"],
    "CheckpointLoaderNF4": ["ckpt_name"],
    "LoraLoader": ["lora_name", "strength_model", "strength_clip"],
    "LoraLoaderModelOnly": ["lora_name", "strength_model"],
    "VAELoader": ["vae_name"],
    "CLIPLoader": ["clip_name", "type"],
    "DualCLIPLoader": ["clip_name1", "clip_name2", "type"],
    "TripleCLIPLoader": ["clip_name1", "clip_name2", "clip_name3"],
    "QuadrupleCLIPLoader": ["clip_name1", "clip_name2", "clip_name3", "clip_name4"],
    "CLIPVisionLoader": ["clip_name"],
    "StyleModelLoader": ["style_model_name"],
    "UpscaleModelLoader": ["model_name"],
    "ControlNetLoader": ["control_net_name"],
    "DiffControlNetLoader": ["control_net_name"],
    "GLIGENLoader": ["gligen_name"],
    "UNETLoader": ["unet_name", "weight_dtype"],
    "EmptyLatentImage": ["width", "height", "batch"],
    "EmptyLatentImageLarge": ["width", "height", "batch"],
    "EmptySD3LatentImage": ["width", "height", "batch"],
    "EmptyHunyuanLatentVideo": ["width", "height", "length", "batch"],
}
LATENT_CLASSES = {"EmptyLatentImage", "EmptyLatentImageLarge", "EmptySD3LatentImage",
                  "EmptyHunyuanLatentVideo", "EmptyLTXVLatentVideo", "EmptyMochiLatentVideo"}

# --- Model families for VRAM estimation. weights = fp16 bytes on disk, GB.
FAMILIES = {
    "sd15":  {"weights": 2.4,  "res": (512, 512),  "act": 1.0, "lora": 0.10, "label": "SD 1.5"},
    "sdxl":  {"weights": 6.9,  "res": (1024, 1024), "act": 2.0, "lora": 0.25, "label": "SDXL"},
    "sd35m": {"weights": 7.5,  "res": (1024, 1024), "act": 2.2, "lora": 0.25, "label": "SD 3.5 Medium"},
    "sd35l": {"weights": 21.0, "res": (1024, 1024), "act": 2.6, "lora": 0.25, "label": "SD 3.5 Large"},
    "flux":  {"weights": 23.8, "res": (1024, 1024), "act": 2.6, "lora": 0.30, "label": "FLUX.1 (fp16)"},
    "qwen":  {"weights": 40.0, "res": (1328, 1328), "act": 3.5, "lora": 0.30, "label": "Qwen-Image"},
}
QUANT_DISCOUNT = {"q2": 0.16, "q3": 0.22, "q4": 0.30, "q5": 0.38, "q6": 0.45, "q8": 0.52}


def detect_family(names):
    """Guess model family from checkpoint/unet filenames (and loader hints)."""
    joined = " ".join(names).lower()
    if "flux" in joined:
        return "flux"
    if "qwen" in joined:
        return "qwen"
    if "sd3" in joined or "sd_3" in joined:
        return "sd35l" if "large" in joined else "sd35m"
    if ("sdxl" in joined or "sd_xl" in joined or "playground" in joined
            or re.search(r"illustrious|noobai|pony", joined)
            or re.search(r"\bxl\b", joined)):
        return "sdxl"
    if "sd15" in joined or "v1-5" in joined or "sd_v1" in joined or "1.5" in joined:
        return "sd15"
    return None


def quant_factor(name):
    """Return (factor, label) for quantized checkpoints, (1.0, '') otherwise."""
    low = name.lower()
    if "gguf" in low or re.search(r"q\d[_ ]?k", low):
        m = re.search(r"q(\d)", low)
        level = "q" + m.group(1) if m else "q4"
        return QUANT_DISCOUNT.get(level, 0.35), level.upper() + " (GGUF-style)"
    if "fp8" in low or "e4m3" in low or "e5m2" in low:
        return 0.55, "fp8"
    if "nf4" in low:
        return 0.35, "NF4"
    return 1.0, ""


def parse_workflow(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    nodes = []  # {class_type, widgets(list), inputs(dict), disabled(bool)}
    if isinstance(data, dict) and isinstance(data.get("nodes"), list):
        fmt = "ui"
        for n in data["nodes"]:
            nodes.append({
                "class_type": n.get("type", "?"),
                "widgets": n.get("widgets_values") or [],
                "inputs": {},
                "disabled": n.get("mode") in (2, 4),
            })
    elif isinstance(data, dict):
        fmt = "api"
        for nid, n in data.items():
            if isinstance(n, dict) and "class_type" in n:
                nodes.append({
                    "class_type": n["class_type"],
                    "widgets": [],
                    "inputs": n.get("inputs") or {},
                    "disabled": False,
                })
    else:
        raise SystemExit("error: not a ComfyUI workflow JSON (no nodes found)")
    if not nodes:
        raise SystemExit("error: workflow contains no nodes")
    return fmt, nodes


def extract_models(nodes):
    """Return models [(filename, folder)], loras count, latent dict, dtype hint."""
    models, loras, latent = [], 0, None
    dtype_hint = ""
    seen = set()

    def add(fname, folder):
        """Returns True if fname counts as a lora."""
        is_lora = False
        if fname and str(fname).lower().endswith(MODEL_EXTS):
            key = (fname, folder)
            if key not in seen:
                seen.add(key)
                models.append(key)
            is_lora = (folder == "loras") or ("lora" in str(fname).lower())
        return is_lora

    for n in nodes:
        if n["disabled"]:
            continue
        cls, widgets, inputs = n["class_type"], n["widgets"], n["inputs"]
        slots = WIDGET_MAP.get(cls)
        if slots:  # UI format: positional widgets
            for slot, val in zip(slots, widgets):
                if not isinstance(val, str):
                    if slot in ("width", "height", "batch") and isinstance(val, (int, float)):
                        if latent is None:
                            latent = {"width": 1024, "height": 1024, "batch": 1}
                        latent[slot] = int(val)
                    continue
                if slot == "weight_dtype" and val not in ("default", "disabled"):
                    dtype_hint = val
                if add(val, SLOT_DIRS.get(slot, "unknown")):
                    loras += 1
        # API format / generic: scan named inputs
        for key, val in inputs.items():
            if key == "weight_dtype" and isinstance(val, str) and val not in ("default", "disabled"):
                dtype_hint = val
                continue
            if isinstance(val, str) and val.lower().endswith(MODEL_EXTS):
                if add(val, SLOT_DIRS.get(key, "unknown")):
                    loras += 1
        # heuristic for unknown UI nodes: any string widget that looks like a file
        if slots is None:
            for val in widgets:
                if isinstance(val, str) and val.lower().endswith(MODEL_EXTS):
                    if add(val, "unknown"):
                        loras += 1
    if latent is None:
        latent = {"width": 1024, "height": 1024, "batch": 1}
    return models, loras, latent, dtype_hint


def find_missing(nodes, offline):
    """Return list of (class_name, pack_info_or_None) for non-core nodes."""
    missing = []
    seen = set()
    for n in nodes:
        if n["disabled"]:
            continue
        cls = n["class_type"]
        if cls in CORE_NODES or cls in seen or cls.startswith("Reroute"):
            continue
        seen.add(cls)
        info = PACK_MAP.get(cls)
        if info is None and not offline:
            info = registry_lookup(cls)
        missing.append((cls, info))
    return missing


_registry_cache = {}


def registry_lookup(class_name):
    """Query the ComfyUI Registry for a pack matching this node class name.

    Hardened: class name is allowlisted, URL is pinned to the registry host,
    and no redirects are followed.
    """
    if class_name in _registry_cache:
        return _registry_cache[class_name]
    info = None
    if not _CLASS_NAME_RE.match(class_name):
        info = ("unresolved (unusual node name)", "")
    else:
        url = REGISTRY_HOST + "/nodes/search?query=" + urllib.parse.quote(class_name, safe="")
        if urllib.parse.urlsplit(url).netloc != "api.comfy.org":
            info = ("unresolved (bad registry url)", "")
        else:
            try:
                req = urllib.request.Request(
                    url, headers={"User-Agent": "comfy-preflight/" + __version__})
                # block redirects: the registry should answer directly
                class _NoRedirect(urllib.request.HTTPRedirectHandler):
                    def redirect_request(self, *a, **k):
                        return None
                opener = urllib.request.build_opener(_NoRedirect)
                with opener.open(req, timeout=8) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                results = data.get("nodes") or []
                if results:
                    top = results[0]
                    repo = (top.get("publisher") or {}).get("source_code_repo") or top.get("repository") or ""
                    info = (top.get("name", "?"), repo)
            except Exception:
                info = ("unresolved (registry unreachable?)", "")
    _registry_cache[class_name] = info
    return info


def estimate_vram(nodes, models, loras, latent, family_override=None, dtype_hint=""):
    """Return dict with label, weights, act, low, high, notes[]."""
    notes = []
    loader_files = []
    unet_only = any(n["class_type"] == "UNETLoader" for n in nodes if not n["disabled"])
    for fname, folder in models:
        if folder in ("checkpoints", "diffusion_models", "unknown"):
            if "vae" not in fname.lower():
                loader_files.append(fname)
    if not loader_files:
        # text-encoder-only or loader-less workflow: treat as tiny
        return {"family": "unknown", "label": "no checkpoint/unet found",
                "weights": 0.0, "act": 0.0, "low": 0.5, "high": 1.0,
                "notes": ["No checkpoint or unet loader detected - estimate may be meaningless."]}
    fam_key = family_override or detect_family(loader_files) or "sdxl"
    if family_override is None and detect_family(loader_files) is None:
        notes.append("Family unrecognized; assuming SDXL-class weights. Pass --family to override.")
    fam = FAMILIES[fam_key]
    main_file = loader_files[0]
    factor, quant_label = quant_factor(main_file)
    if factor == 1.0 and "fp8" in dtype_hint.lower():
        factor, quant_label = 0.55, "fp8 (loader weight_dtype)"
    weights = fam["weights"] * factor
    if quant_label:
        notes.append("%s quantized weights: '%s' counts %.1f GB instead of %.1f GB fp16."
                     % (quant_label, main_file, weights, fam["weights"]))
    if unet_only:
        # separated architecture: unet + separate text encoder(s) + vae
        clip_files = [f for f, d in models if d == "text_encoders"]
        clip_gguf = any(quant_factor(c)[0] < 1.0 for c in clip_files)
        if clip_gguf:
            weights += 2.9
            notes.append("Quantized separate text encoders detected (+2.9 GB).")
        elif any("t5" in c.lower() for c in clip_files):
            weights += 5.2
            notes.append("Separate T5 text encoder (+5.2 GB fp16).")
        elif clip_files:
            weights += 1.7
        weights += 0.3  # vae
    if loras:
        notes.append("%d LoRA(s): +%.1f GB." % (loras, loras * fam["lora"]))
    dw, dh = fam["res"]
    w, h, batch = latent["width"], latent["height"], latent["batch"]
    act = fam["act"] * ((w * h) / (dw * dh)) ** 1.5 * batch
    if w * h > dw * dh:
        notes.append("Latent %dx%d is above the family default %dx%d: activation scaled x%.2f."
                     % (w, h, dw, dh, ((w * h) / (dw * dh)) ** 1.5))
    if batch > 1:
        notes.append("Batch size %d multiplies activation memory." % batch)
    low = weights * 0.55 + act * 0.9 + 0.4
    high = weights * 1.0 + act * 1.6 + 0.7
    return {"family": fam_key, "label": fam["label"], "weights": weights, "act": act,
            "low": low, "high": high, "notes": notes}


def verdict(est, vram):
    low, high, w = est["low"], est["high"], est["weights"]
    tips = []
    if high <= vram * 0.92:
        return "OK", "Estimated peak fits in %.1f GB." % vram, tips
    if w * 0.55 > vram:
        return "NO", "Weights alone cannot fit %.1f GB even with aggressive offload." % vram, \
               ["Use a quantized model (GGUF Q4 or fp8) or a smaller base model."]
    if low <= vram * 0.95:
        tips += ["Run with --lowvram (or --novram if ComfyUI still OOMs).",
                 "Use Tiled VAE decode for high resolutions."]
        return "TIGHT", "Only fits with offloading; expect slower generation.", tips
    tips += ["Try --lowvram and Tiled VAE.", "Consider a GGUF/fp8 quantized checkpoint."]
    return "MAYBE", "Above comfortable headroom for %.1f GB; risky." % vram, tips


def fmt_report(path, fmt, nodes, missing, models, est, vd, vram, latent):
    badge = {"OK": "[OK]", "TIGHT": "[TIGHT]", "MAYBE": "[MAYBE]", "NO": "[NO]"}[vd[0]]
    lines = []
    lines.append("# comfy-preflight report")
    lines.append("")
    lines.append("- Workflow: `%s` (%s format, %d active nodes)" % (
        path, fmt, len([n for n in nodes if not n["disabled"]])))
    lines.append("- Target VRAM: %.1f GB" % vram)
    lines.append("")
    lines.append("## Verdict: %s %s" % (badge, vd[0]))
    lines.append("")
    lines.append(vd[1])
    lines.append("")
    lines.append("## VRAM estimate (heuristic, fp16 unless noted)")
    lines.append("")
    lines.append("| Component | Estimate |")
    lines.append("|---|---|")
    lines.append("| Family | %s |" % est["label"])
    lines.append("| Weights (after quant discount) | %.1f GB |" % est["weights"])
    lines.append("| Activation @ %dx%d x%d | %.1f GB |" % (
        latent["width"], latent["height"], latent["batch"], est["act"]))
    lines.append("| Range | low %.1f GB (offload) - high %.1f GB (all resident) |" % (
        est["low"], est["high"]))
    lines.append("")
    for note in est["notes"]:
        lines.append("- Note: %s" % note)
    for tip in vd[2]:
        lines.append("- Tip: %s" % tip)
    lines.append("")
    lines.append("## Missing nodes (%d)" % len(missing))
    lines.append("")
    if missing:
        lines.append("| Node class | Likely pack | Source |")
        lines.append("|---|---|---|")
        for cls, info in missing:
            if info:
                lines.append("| `%s` | %s | %s |" % (cls, info[0], info[1] or "-"))
            else:
                lines.append("| `%s` | core? (not in bundled list; may be a newer core node) | - |" % cls)
    else:
        lines.append("None - all node classes are known core nodes.")
    lines.append("")
    lines.append("## Models required (%d)" % len(models))
    lines.append("")
    if models:
        lines.append("| File | Folder under ComfyUI/models/ |")
        lines.append("|---|---|")
        for fname, folder in models:
            lines.append("| `%s` | %s |" % (fname, folder))
    else:
        lines.append("None detected.")
    lines.append("")
    lines.append("---")
    lines.append("*Estimates are heuristics, not benchmarks. Actual usage depends on "
                 "ComfyUI version, attention backend and offload behavior.*")
    return "\n".join(lines)


def report_text(path, fmt, nodes, missing, models, est, vd, vram, latent):
    out = []
    out.append("comfy-preflight %s - %s" % (__version__, os.path.basename(path)))
    out.append("format=%s  active_nodes=%d  target_vram=%.1fGB" % (
        fmt, len([n for n in nodes if not n["disabled"]]), vram))
    out.append("")
    out.append("VERDICT: %-6s %s" % (vd[0], vd[1]))
    out.append("VRAM est: %.1f - %.1f GB (weights %.1f + act %.1f @ %dx%d x%d)  family=%s" % (
        est["low"], est["high"], est["weights"], est["act"],
        latent["width"], latent["height"], latent["batch"], est["label"]))
    for note in est["notes"]:
        out.append("  note: %s" % note)
    for tip in vd[2]:
        out.append("  tip:  %s" % tip)
    out.append("")
    out.append("Missing nodes (%d):" % len(missing))
    if not missing:
        out.append("  none")
    for cls, info in missing:
        if info:
            out.append("  - %-34s -> %s  (%s)" % (cls, info[0], info[1] or "repo n/a"))
        else:
            out.append("  - %-34s -> unknown (not core, not in curated map)" % cls)
    out.append("")
    out.append("Models required (%d):" % len(models))
    if not models:
        out.append("  none")
    for fname, folder in models:
        out.append("  - %-46s -> models/%s" % (fname, folder))
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="comfy-preflight",
        description="Static pre-flight checker for ComfyUI workflows: missing nodes, "
                    "required models, rough VRAM feasibility - before installing anything. "
                    "All output goes to stdout; redirect to save a report.")
    ap.add_argument("workflow", help="path to workflow.json (UI or API format)")
    ap.add_argument("--vram", type=float, default=6.0, metavar="GB",
                    help="your GPU VRAM in GB (default: 6)")
    ap.add_argument("--family", choices=sorted(FAMILIES), default=None,
                    help="override model family detection")
    ap.add_argument("--offline", action="store_true",
                    help="skip ComfyUI Registry API lookups")
    ap.add_argument("--markdown", action="store_true", help="emit markdown report")
    ap.add_argument("--version", action="version", version="comfy-preflight " + __version__)
    args = ap.parse_args(argv)

    workflow = resolve_read_path(args.workflow)
    fmt, nodes = parse_workflow(workflow)
    models, loras, latent, dtype_hint = extract_models(nodes)
    missing = find_missing(nodes, args.offline)
    est = estimate_vram(nodes, models, loras, latent,
                        family_override=args.family, dtype_hint=dtype_hint)
    vd = verdict(est, args.vram)

    if args.markdown:
        print(fmt_report(args.workflow, fmt, nodes, missing, models, est, vd, args.vram, latent))
    else:
        print(report_text(args.workflow, fmt, nodes, missing, models, est, vd, args.vram, latent))
    return 0 if vd[0] != "NO" else 1


if __name__ == "__main__":
    sys.exit(main())
