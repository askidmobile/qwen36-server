# Gemma 4 long-context prefill stalled before first SSE token

**Date:** 2026-08-21

## Symptom

WebUI appeared to stop after sending roughly 8K context to Gemma 4 Q8_0. Server still consumed GPU memory and showed load.

## Root cause

Live monitor showed the request was still running before first token:

```text
GPU utilization: 100%
VRAM: 11,989 / 12,114 MiB
first generated token: none
```

`Config::batch_size` existed (`BATCH_SIZE=2048`), but serialized `CandleEngine` ignored it. `run_generation()` sent the entire prompt to one `model.forward()`. Gemma attention therefore allocated quadratic matrices for all prompt tokens at once, saturated RTX 3060 VRAM, and entered WDDM pressure before any SSE delta.

## Fix

- Prefill prompt through sequential KV-continuation chunks.
- Cap Candle prefill chunks at 1024 tokens because this route has no validated FlashAttention long-prefill path.
- Preserve absolute `index_pos` across chunks.
- Clear completed KV state and trim CUDA default mempool after each serialized request (`PREFIX_CACHE_MIB=0`).
- Show `обработка контекста…` in WebUI until first delta instead of an empty frozen bubble.

## Gates

Direct realistic context gate, Gemma Q8_0:

```text
prompt_tokens: 15,638
first delta: 11.4s
total: 12.7s
completion_tokens: 20
finish: stop
answer: correct
VRAM after request: 8,465 MiB
```

Chrome DevTools WebUI gate:

```text
during prefill: sendDisabled=true, "обработка контекста…"
prompt_tokens: 18,184
visible answer: "Проект называется Qwen3.6 27B (инференс на candle)."
total: 15.7s
sendDisabled after: false
console errors: 0
```

A repeated-word synthetic prompt generated immediate `<turn|>` after prefill. It was rejected as quality gate because repetition is pathological; realistic documentation context generated correctly.

## Commits

- `51bdb97` chunk long prompt prefill and WebUI state
- `c405313` release KV and CUDA pool after request
