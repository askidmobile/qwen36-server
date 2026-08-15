# Qwen3.5 Vision mixed-Q8 runtime gate

**Date:** 2026-08-15
**Source:** Phase 5 of full Qwen3.5-4B multimodal/MTP plan

## What happened

Native Candle Vision runtime loaded full mixed-Q8 artifact and completed CUDA image/video forward. End-to-end multimodal prefill plus decode also worked against Text Q4_K_M. Uniform Q8_0 over all eligible Vision matrices produced identical first four greedy tokens for English OCR, Russian OCR and video, but mandatory embedding gate failed:

```text
English: cosine=0.9980 nRMSE=0.0634 PASS
Russian: cosine=0.9940 nRMSE=0.1115 FAIL
Video:   cosine=0.9897 nRMSE=0.1432 FAIL
thresholds: cosine>=0.995, nRMSE<=0.10
```

## Root cause

Processor mismatch was excluded: Rust and pinned Transformers patches remained exact. Error came from cumulative Q8 quantization through sensitive Vision layers. Final logits masked much of this drift, so argmax/token equality alone was insufficient.

## Fix

Measured tensor groups against same BF16 Vision and Text Q4 backbone. Promoted minimal policy that reduces weights while preserving embedding and final-logit gates:

```text
Q8_0: v.blk.1..19.ffn_down.weight
BF16: remaining eligible Vision matrices
F32/BF16: norms, biases, patch projection and position embeddings
```

CUDA gate after policy:

```text
embedding cosine min:  0.99946
embedding nRMSE max:   0.03296
final-logit cosine min: 0.99629
four greedy tokens:    exact for English/Russian/video
```

## Runtime fixes retained

- Quantized Conv3D split uses metadata-only `QTensor::reshape`; processor temporal layout is channel-major and slices are strided.
- Multimodal prefill keeps cache positions as token indices and sends separate T/H/W MRoPE tables.
- Decode passes separate cache and RoPE positions. Using adjusted RoPE position as KV offset was incorrect.
- First `reset_first` prefill preserves already-installed media and `decode_rope_delta`.
- Production Vision load is fail-closed against exact promoted dtype policy; BF16 path is explicit correctness oracle only.

## Outcome

Phase 5 passed. Candidate remains staged; `current` unchanged until full ten-phase release.

## Reports

```text
D:\Projects\yttri-inference\bench\qwen35-artifacts\phase5-cuda-gate-v2.json
D:\Projects\yttri-inference\bench\qwen35-artifacts\phase5-embedding-gate.json
D:\Projects\yttri-inference\bench\qwen35-artifacts\phase5-mixed-profile-experiment.json
```
