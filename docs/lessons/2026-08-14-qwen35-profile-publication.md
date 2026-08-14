# Qwen3.5 profile publication foundation

**Date:** 2026-08-14
**Source:** Phase 2 implementation for full Qwen3.5-4B multimodal/MTP plan

## What happened

Profile support needed two failure classes with different behavior: mandatory Text corruption must block startup, while optional Vision/MTP corruption must preserve Text and remove capability claims. Windows publication also needed a crash-safe `current` change without mutating releases.

## Root cause

A manifest boolean cannot prove capability. Paths, hashes, ABI, quant policy, runtime siblings and processor files all affect whether component is usable. Plain PowerShell `Move-Item` does not express required write-through replacement semantics.

## Fix

- `ResolvedProfile` derives capabilities only after containment, size, SHA-256, ABI and contract validation.
- Mandatory Text/runtime server/gates fail profile; optional Vision/MTP/media runtime become `error`/unavailable.
- `Config` validates `QWEN36_PROFILE` once and reuses resolved profile; absent profile keeps legacy text-only `QWEN36_MODEL` behavior.
- Publisher flushes temporary pointer then calls `MoveFileExW(MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)`.
- Launcher resolves and hashes exact server sibling from immutable release.
- Synthetic Windows test proves failed gate leaves current pointer byte-identical and rollback reuses validated release.

## References

- `src/profile.rs`
- `scripts/qwen35_release.ps1`
- `scripts/qwen35_run_current.ps1`
- `tests/profile_test.rs`
