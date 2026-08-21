# Gemma 4 family capability is not runtime capability

**Date:** 2026-08-21

## Symptom

WebUI allowed attaching an image to `gemma-4-e4b-it`, then chat returned:

```text
HTTP 503 component_unavailable: multimodal requests require batched engine
```

## Root cause

Gemma 4 model family is multimodal, and sibling `mmproj-gemma-4-E4B-it-BF16.gguf` exists. Active production route nevertheless used serialized quantized `CandleEngine`, whose graph loaded only text GGUF and explicitly rejected media. WebUI assumed family/model-file capability instead of querying effective active runtime capability.

## Investigation

A native Candle experiment connected:

- Gemma4V BF16 mmproj GGUF (1411 tensors, ~946 MiB);
- existing Candle Gemma 4 vision tower;
- dynamic image preprocessing (40–280 visual tokens);
- `<|image>` / repeated `<|image|>` / `<image|>` protocol;
- quantized text embedding injection;
- calibration clamps and PLE padding-row semantics from llama.cpp oracle.

GPU load and prefill succeeded (`image_count=1`, `visual_tokens=114`, ~6.7 GiB VRAM), but semantic gate failed: model repeatedly answered that no image was provided. Experimental code was saved outside repositories and rolled back. Parser/load success was not accepted as runtime support.

## Production fix

Added effective runtime capability methods to `Engine`. `/v1/models` intersects profile capabilities with active graph capabilities. For text-only routes:

- `capabilities.vision=false`, `capabilities.video=false`;
- WebUI attachment controls disabled with clear title and `text-only` badge;
- `/v1/media` rejects before storing upload with explicit text-only/mmproj-runtime message.

Qwen/Ornith batched route reports vision/video only when a validated vision component path is loaded.

## Gate

Gemma Q4_K_M live:

```text
/v1/models vision=false video=false
/v1/media HTTP 503 before upload storage
Chrome attachDisabled=true fileDisabled=true
Current model card: Q4_K_M · ctx 8192 · slots 1 · text-only
```

## References

- `src/engine.rs`
- `src/engine_swap.rs`
- `src/engine_batched.rs`
- `src/api/openai.rs`
- `src/api/media.rs`
- `web/index.html`
- commit `f09cde9`
