# Qwen3.5 processor parity

**Date:** 2026-08-15

**Source:** Phase 4 implementation

## What happened

Generic bicubic resize looked compatible but could not prove `≤1e-5` tensor parity. Pinned Qwen3.5 uses fast Torchvision processing on decoded RGB8. Video prompt expansion also adds timestamp-separated vision spans whose MRoPE groups differ from one contiguous video span.

## Root cause

Torchvision CPU bicubic antialias on RGB8 computes separable Keys cubic weights, quantizes each axis to signed 16-bit coefficients, rounds after horizontal pass, then rounds after vertical pass. Float bicubic or `image::imageops::resize` can differ before normalization.

Pinned Qwen3.5 video processing expands every temporal pair as `<N.N seconds><|vision_start|>...<|vision_end|>`. Position construction therefore splits `video_grid_thw[T,H,W]` into `T` grids `[1,H,W]`. Cache position remains token-index based; decode RoPE uses `token_position + decode_rope_delta`.

## Fix

- Port exact uint8 antialiased bicubic coefficient and two-pass rounding behavior.
- Keep patch packing order identical to pinned reshape/permute.
- Build `mm_token_type_ids`, T/H/W positions and decode delta from expanded markers.
- Compare Rust and pinned Transformers from JSON probes; fail closed on any exact-field mismatch or patch error above `1e-5`.
- Run pinned Transformers in isolated `yttri-win` tool environment. Do not mix Homebrew and user Python wheels through a macOS `--system-site-packages` venv; mixed `pyarrow/libarrow/protobuf` caused process abort during import.

## References

- `qwen35-batch/src/real/multimodal.rs`
- `qwen35-batch/tests/multimodal_processor.rs`
- `tools/qwen35-processor-reference.py`
- `tools/qwen35-artifacts.py`
