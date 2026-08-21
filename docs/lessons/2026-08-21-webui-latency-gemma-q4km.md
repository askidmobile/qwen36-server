# WebUI latency and Gemma 4 Q4_K_M fixes

**Date:** 2026-08-21

## Problem

1. WebUI model list took 1-2 minutes to load.
2. Context size selector appeared broken.
3. Gemma 4 Q4_K_M streamed only reasoning, no answer, and VRAM seemed high.

## Root causes

1. **Model list scan**: `available_models` walked D:\Models and called
   `gguf_architecture()` for every GGUF. Windows Defender/Kaspersky adds
   ~1.6s per file open; 17+ files → 30+ seconds per request, every request.
2. **Context selector**: `ctx_matrix` called `nvidia-smi` subprocess
   (2-3s on Windows per invocation) and re-read the GGUF header every time.
3. **Gemma 4 Q4_K_M reasoning-only output**: sampling preset used temp=0.7/top_k=20.
   Official Google generation_config is temp=1.0, top_k=64, top_p=0.95.
   High temperature destabilized thought→final channel transition; model
   wrote the answer inside reasoning and closed the turn immediately.

## Fixes

- `ARCH_CACHE`: GGUF architecture cached by (path, mtime, size). Header read once.
- `SCAN_CACHE`: full model list cached with 60s TTL; WebUI polling (2s) hits cache.
- `FP_CACHE`: `ModelFootprint` cached by (path, mtime, size) in vram_plan.
- `TOTAL_VRAM_CACHE`: nvidia-smi total VRAM read once per process.
- Gemma 4 presets updated: temp=0.6, top_k=64 (matches official config, stabilizes channels).

## Results (yttri-win, 23 GGUF files)

- available_models: 32s cold (once per server restart, antivirus) → 0.02s warm.
- ctx_matrix: 2.7s cold (once per file) → 0.018s warm.
- Gemma 4 Q4_K_M with thinking: reasoning 851 chars + content 537 chars, clean Russian.
- VRAM: Q4_K_M ~6.0 GiB; Q8_0 ~8.8 GiB; both fit RTX 3060 12GB.

## Commits

- `4afc060` scan cache + Gemma presets
- `a09c615` ARCH_CACHE + FP_CACHE
- `b5b8c00` TOTAL_VRAM_CACHE
- `7f8d5e7a` (candle-fork) proportional RoPE + rope_freqs
