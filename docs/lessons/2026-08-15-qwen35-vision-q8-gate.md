# Qwen3.5 Vision Q8 runtime gate

**Date:** 2026-08-15
**Source:** Phase 5 of full Qwen3.5-4B multimodal/MTP plan

## What happened

Native Candle Vision runtime loaded full mixed-Q8 artifact and completed CUDA image/video forward. End-to-end multimodal prefill plus decode also worked against Text Q4_K_M. Full-Q8 and BF16 Vision produced identical first four greedy tokens for English OCR, Russian OCR and video, with final-logit cosine above `0.996`.

Mandatory embedding gate still failed:

```text
English: cosine=0.9980 nRMSE=0.0634 PASS
Russian: cosine=0.9940 nRMSE=0.1115 FAIL
Video:   cosine=0.9897 nRMSE=0.1432 FAIL
thresholds: cosine>=0.995, nRMSE<=0.10
```

## Root cause and experiments

Failure is quantization sensitivity, not processor mismatch: Rust and pinned Transformers patches remained exact. Official source, converter output and BF16 artifact hashes matched. Runtime BF16 reference also stayed close to official Transformers BF16 for English/Russian.

Tested mixed profiles without touching `current`:

- merger-only Q8;
- first 12 or 16 Vision blocks Q8;
- attention-only Q8;
- attention-out-only Q8;
- FFN-down-only Q8;
- full blocks Q8 with BF16 merger.

Nearest memory-reducing profile (`blocks 0..15 Q8`, tail eight blocks and merger BF16) passed English/Russian embedding and all tested logits, but video remained `cosine=0.9900`, `nRMSE=0.1411`. No profile met every mandatory gate.

## Runtime fixes retained

- Quantized Conv3D split uses metadata-only `QTensor::reshape`; processor temporal layout is channel-major and slices are strided.
- Multimodal prefill keeps cache positions as token indices and sends separate T/H/W MRoPE tables.
- Decode now passes separate cache and RoPE positions. Using adjusted RoPE position as KV offset was incorrect.
- First `reset_first` prefill preserves already-installed media and `decode_rope_delta`.
- Production Vision load remains fail-closed; BF16 path is explicit correctness oracle only.

## Outcome

Phase 5 code and CUDA functional path retained. Numerical promotion remains blocked. No Vision candidate published and model `current` unchanged.

## Reports

```text
D:\Projects\yttri-inference\bench\qwen35-artifacts\phase5-cuda-gate.json
D:\Projects\yttri-inference\bench\qwen35-artifacts\phase5-embedding-gate.json
D:\Projects\yttri-inference\bench\qwen35-artifacts\phase5-mixed-profile-experiment.json
```
