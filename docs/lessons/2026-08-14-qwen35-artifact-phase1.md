# Qwen3.5 artifact inventory phase 1

**Date:** 2026-08-14
**Source:** Phase 1 implementation for full Qwen3.5-4B multimodal/MTP plan

## What happened

Deterministic multimodal fixtures initially changed SHA-256 on every generation despite identical pixels and byte lengths. Pinned llama.cpp patch also failed through `git apply` after clean checkout. Windows CUDA inspector build failed with `nvcc fatal: Cannot find compiler 'cl.exe' in PATH`.

## Root cause

- ImageMagick embedded current PNG timestamps. Visual determinism did not imply byte determinism.
- Unified patch paths already contained `conversion/`; script also ran `git apply` from wrong effective prefix.
- CUDA build shell did not initialize MSVC Build Tools environment.
- ImageMagick `-orient RightTop` did not write EXIF orientation in generated JPEG.

## Fix

- Strip metadata, exclude PNG date/time chunks, set fixed filesystem timestamps, and compare regenerated manifest byte-for-byte.
- Apply converter patch with `git apply --directory=conversion`; hard-reset pinned checkout before each preparation, then verify exact HEAD.
- Run Windows CUDA build after `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat`.
- Insert minimal EXIF APP1 orientation segment directly, then verify decoder sees `RightTop`.
- Keep source inventory audit independent from model conversion: exact public index proves `426 + 297 + 15 = 738`; physical output audit runs after artifacts exist.

## References

- `tools/generate_multimodal_fixtures.py`
- `tools/llama-qwen35-map-report.patch`
- `tools/qwen35-artifacts.py`
- `scripts/qwen35_artifacts.ps1`
- `D:\Projects\yttri-inference\bench\qwen35-artifacts\source-inventory.json`
