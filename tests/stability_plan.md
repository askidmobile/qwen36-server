# Stability & Gate Execution Plan

## Gate Matrix

1. **Artifact Mapping Audit:**
   - 738 official safetensors tensors mapped into 426 Text, 298 Vision, 15 MTP outputs.
   - Fail-closed verification in `qwen35-artifacts.py`.
2. **Vision Quantization & Parity Gate:**
   - English, Russian Cyrillic, and Video embeddings match within tolerance (`cosine >= 0.995`, `nRMSE <= 0.10`).
   - Results in `bench/qwen35-artifacts/phase5-cuda-gate-v2.json`.
3. **MTP Parity Gate:**
   - Bit-exact baseline matching for $B=1$ and $B=4$ on English & Russian prompts.
   - Verified via `qwen35_mtp_gate.exe`.
   - Results in `bench/qwen35-artifacts/phase7-cuda-gate.json`.
4. **Offline and Multi-Slot Stability:**
   - $4 \times 8K$ continuous batching without CUDA errors or VRAM paging.
   - Bounded helper Job Object execution without network access.
