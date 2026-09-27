# Community VRAM benchmarks

`data.json` collects **real measured** peak-VRAM numbers for ComfyUI runs,
so comfy-preflight's estimates can be calibrated against reality instead of
pure heuristics.

## Format

Each entry in `entries`:

```json
{
  "gpu": "RTX 3060 Laptop",
  "vram_total_gb": 6,
  "model": "sd_xl_base_1.0.safetensors",
  "family": "sdxl",
  "resolution": [1024, 1024],
  "batch": 1,
  "mode": "normal",
  "peak_vram_gb": 6.1,
  "measured_with": "nvidia-smi dmon",
  "date": "2026-09-27",
  "notes": ""
}
```

Field rules:

- `family`: one of `sd15`, `sdxl`, `sd35m`, `sd35l`, `flux`, `qwen`
- `mode`: `normal` | `lowvram` | `medvram` | `novram` (ComfyUI launch flags used)
- `resolution`: `[width, height]` of the latent (the EmptyLatentImage values)
- `peak_vram_gb`: the highest GPU memory in use during generation, GB
- `measured_with`: how you measured (`nvidia-smi`, Task Manager,
  `torch.cuda.max_memory_allocated` - say which; allocated vs reserved differ)
- duplicate `(gpu, model, resolution, batch, mode)` entries will be rejected

## Contributing

1. Fork, add your entry to `entries` in `data.json`
2. Run `python benchmarks/validate.py` - it must print `OK`
3. Open a PR

Entries are used to calibrate the estimation table in `comfy_preflight.py`
over time. Measurements are community data, attributed only to "the
benchmark file" - do not include personal info.
