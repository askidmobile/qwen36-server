# Qwen3.5 media trust boundary

**Date:** 2026-08-15
**Source:** Phase 3 implementation for full Qwen3.5-4B multimodal/MTP plan

## What happened

Media intake combines four separate trust boundaries: authenticated temporary storage, public HTTPS fetch, untrusted codec execution and helper result validation. First Windows helper build linked CUDA because server crate features propagate into binaries; resulting helper failed without CUDA runtime DLL despite doing no GPU work.

## Root cause

- Helper shares package dependency graph, so `--features cuda` pulls CUDA link requirements even though helper code has no CUDA calls.
- Owner labels are not safe identity material; upload ownership must derive from actual API key without storing key.
- Reserving temp bytes after file write permits concurrent uploads to oversubscribe global budget.
- Closing Windows Job Object immediately after assignment kills child because kill-on-close is enabled.

## Fix

- Build helper without CUDA features; package it beside CUDA server.
- Derive `OwnerDigest = SHA256(API key)` during constant-time auth and store only digest.
- Reserve pending count and encoded bytes atomically before exclusive `.part` creation; RAII releases failed reservations.
- Keep Job Object handle alive through child completion; limits include kill-on-close, process count, job memory and CPU time.
- Validate helper paths, RGB byte product, total output, SHA-256, finite metadata and `audio_processed=false` before accepting output.
- HTTPS fetch disables proxy/redirect/decompression, pins validated public addresses and revalidates every manual redirect.

## References

- `src/media/store.rs`
- `src/media/fetch.rs`
- `src/media/helper.rs`
- `src/bin/qwen36-media-helper.rs`
- `tests/media_test.rs`
- `tests/helper_test.rs`
