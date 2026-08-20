# GGUF runtime support gate

**Date:** 2026-08-21
**Source:** Gemma 4 GGUF / CUDA 13.2 production deployment

## What happened

WebUI model filtering was removed and `gemma4` was declared supported after adding a new source module. Live switch still failed in `Qwen35BatchAdapter` with `unsupported architecture: 'gemma4'`. Initial module was not connected to server dispatch, did not compile cleanly, and did not implement complete Gemma 4 graph.

## Root cause

GGUF container parsing, architecture graph support, tokenizer support, server dispatch, and API protocol handling are separate gates. Passing metadata scan or displaying a model does not prove runtime support. Gemma 4 additionally required mixed sliding/global attention, shared KV layers, per-layer embeddings, SPM-style BPE, native chat-template compatibility, Gemma thought channels, and model-specific stop tokens.

## Fix

Support is now accepted only after all gates pass:

1. architecture-specific model loader is selected from `general.architecture`;
2. clean local compile and Windows CUDA release build pass;
3. real GGUF loads on target GPU;
4. prefill/decode show GPU utilization and expected VRAM residency;
5. non-stream and SSE return clean content/reasoning fields;
6. model switch releases old VRAM and waits for real adapter readiness;
7. switch back to production model succeeds;
8. commit, push, and live binary are verified independently.

`GPU_ONLY=1` now rejects non-CUDA device and partial `GPU_LAYERS`; Qwen token embeddings cannot silently remain on CPU. CUDA 13.2 is pinned explicitly in build and task launch scripts.

## Verified results

- Qwen3.6 35B-A3B IQ2_XXS: GPU max 99%, about 11.2 GiB VRAM.
- Gemma 4 E4B Q4_K_M: GPU max 100%, about 6.1 GiB VRAM after switch cleanup.
- Ornith 1.5 9B Q6_K: GPU max 100%, about 7.9 GiB VRAM, production ctx 131072 / slots 4.
- CUDA toolkit: 13.2 V13.2.78; clean build compiled 18/18 Candle PTX and 53/53 FlashAttention kernels.

## Remaining scope

`gpt-oss-20b-MXFP4.gguf` is not covered by this implementation. Candle fork has partial MXFP4 CUDA template code but no instantiated MXFP4 storage dispatch and no GPT-OSS graph runtime. It must not be considered runtime-supported until the same gate passes.

## References

- `candle-transformers/src/models/quantized_gemma4.rs`
- `qwen35-batch/src/real/tokenizer.rs`
- `qwen35-batch/src/real/model_weights.rs`
- `src/engine.rs`
- `src/api/admin.rs`
- `src/api/openai.rs`
- `src/chat_template.rs`
- candle commit `6afc5ceb`
- server commit `6b67c3a`
