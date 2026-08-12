# yttri-win stability and memory interpretation

**Date:** 2026-08-12  
**Source:** long CUDA stability run on RTX 3060 12 GB

## What happened

Task Manager showed roughly 11 GB private memory while running Qwen3.6-35B-A3B IQ2_XXS. This looked like model weights had paged into system RAM. A four-slot 8K stability run was also needed to verify state isolation and memory stability.

## Root cause

Windows WDDM accounts CUDA GPU allocations in process private commit. `PrivateMemorySize64` therefore is not physical system-RAM residency. Relevant counters are `GPU Process Memory/Dedicated Usage`, `Shared Usage`, and process working set.

`QWEN36_TRACE=0` was also interpreted as enabled because code checked environment-variable presence instead of a truthy value. Per-token trace made logs noisy and added avoidable overhead.

Long generation exposed context-boundary and API-edge issues: requested output was not clamped after exact prompt tokenization, `max_tokens=0` reached engine as HTTP 500, and `truncated` was true even when no history message was removed.

## Fix

- Treat only `1`, `true`, `yes`, and `on` as enabled trace values.
- Clamp output after exact tokenization to `context_length - prompt_tokens`.
- Reject zero output limits with HTTP 400 in all three APIs.
- Set `truncated=true` only after actual history removal.
- Drop disconnected queued requests through `mpsc::Sender::is_closed()` before GPU work.
- Keep prefix cache rejected until snapshot/logits restore has parity tests.

## Validation

Qwen3.6-35B-A3B IQ2_XXS, `ctx=8192`, `slots=4`:

- Four concurrent identical requests completed `8140` tokens each (`52 + 8140 = 8192`).
- Four outputs were bit-exact by content hash.
- Four SSE streams had terminal `finish_reason=length` and `[DONE]`.
- Zero malformed events, duplicate generated lines, or client errors.
- Elapsed: 1525 seconds.
- GPU shared system RAM stayed at 78 MiB.
- Server answered post-run smoke in about 1.4 seconds.
- Disconnected queued request produced zero output and was not admitted after slots freed.

## References

- `src/engine.rs`
- `src/engine_batched.rs`
- `src/api/openai.rs`
- `src/api/responses.rs`
- `src/api/anthropic.rs`
- `scripts/stability_smoke.sh`
- `tests/stability_plan.md`
- candle fork `qwen35-batch/src/scheduler.rs`
