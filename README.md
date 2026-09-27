# comfy-preflight

**Know before you load.** A static pre-flight checker for ComfyUI workflows:
find missing custom nodes, required model files, and whether the whole thing
fits in your VRAM — *without installing anything or launching ComfyUI*.

You downloaded a 40-node workflow JSON from Civitai. You have a 6 GB GPU.
Instead of installing four node packs, downloading 15 GB of models and then
watching it OOM at step 3, run:

```bash
python comfy_preflight.py workflow.json --vram 6
```

```
comfy-preflight 0.1.0 - flux_dev_fp8.json
format=api  active_nodes=9  target_vram=6.0GB

VERDICT: NO     Weights alone cannot fit 6.0 GB even with aggressive offload.
VRAM est: 11.7 - 21.2 GB (weights 16.3 + act 2.6 @ 1024x1024 x1)  family=FLUX.1 (fp16)
  note: fp8 (loader weight_dtype) quantized weights: 'flux1-dev.safetensors' counts 13.1 GB instead of 23.8 GB fp16.
  tip:  Use a quantized model (GGUF Q4 or fp8) or a smaller base model.

Missing nodes (0):
  none

Models required (4):
  - flux1-dev.safetensors                          -> models/diffusion_models
  - clip_l.safetensors                             -> models/text_encoders
  - t5xxl_fp8_e4m3fn.safetensors                   -> models/text_encoders
  - ae.safetensors                                 -> models/vae
```

## What it does

| Check | How |
|---|---|
| Workflow format | Both UI format (saved from the ComfyUI canvas) and API format |
| Missing nodes | Bundled core-node list + curated map of popular packs; unknown classes are looked up in the [ComfyUI Registry API](https://api.comfy.org) (offline mode: `--offline`) |
| Required models | Loader widgets/inputs parsed by slot name; each file mapped to its folder under `ComfyUI/models/` |
| VRAM estimate | Model family (SD1.5 / SDXL / SD3.5 / FLUX / Qwen-Image) + quantization (GGUF, fp8, NF4 — from filename or loader `weight_dtype`) + latent size/batch + LoRA count |
| Verdict | OK / TIGHT / MAYBE / NO against your `--vram` budget, with actionable tips |

The VRAM report shows its own arithmetic (weights, activation, discounts) so
you can sanity-check it instead of trusting a black box.

## Install

No install needed — single file, Python 3.9+, stdlib only:

```bash
python comfy_preflight.py <workflow.json> [--vram 6] [--markdown] [--offline] [--family sdxl]
```

Or via pip:

```bash
pip install .
comfy-preflight workflow.json
```

Reports go to stdout; save one with redirection:

```bash
python comfy_preflight.py workflow.json --markdown > report.md
```

## How the VRAM estimate works

Rough, deliberate, and calibrated for fp16 inference on NVIDIA cards:

- **Weights** per family (SD 1.5 ≈ 2.4 GB, SDXL ≈ 6.9 GB, FLUX fp16 ≈ 23.8 GB…),
  discounted ~0.3–0.55x for GGUF/fp8/NF4 quantization detected in the filename
  or the loader's `weight_dtype`.
- **Activation** from latent resolution relative to the family's native
  resolution (scaled ~pixel^1.5) and batch size.
- **Range**: `low` assumes aggressive offloading (~0.55x weights resident),
  `high` assumes everything resident plus decode spikes.
- Verdict thresholds leave headroom; a "TIGHT" workflow typically needs
  `--lowvram`/offloading, "NO" means even the weights can't fit.

These are heuristics, not benchmarks. Real usage varies with attention backend,
ComfyUI version and smart-offload behavior. Treat the range as
order-of-magnitude guidance.

## Limitations (v0.1)

- The bundled core-node list can lag behind new ComfyUI releases; a false
  "missing" report for a brand-new core node is possible (it will say so).
- Registry lookups are keyword search, so the suggested pack is a *candidate*,
  not gospel. The curated map (top ~35 packs) is authoritative where it hits.
- Estimation covers txt2img-class graphs; multi-pass (hires-fix, detailer
  chains) currently counts the largest single pass.

## Roadmap

- [ ] `--comfyui PATH`: diff against a real ComfyUI install (installed packs, actual model files present)
- [ ] Multi-pass graphs (hires-fix, upscale chains) as combined peak estimate
- [ ] Community-calibrated activation table (contributions welcome)
- [ ] Batch mode: scan a folder of workflow JSONs

## License

MIT
