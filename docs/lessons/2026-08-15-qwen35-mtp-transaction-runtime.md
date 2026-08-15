# Lesson: Qwen3.5 Transactional MTP Runtime and Exact Parity Gates

**Date:** 2026-08-15  
**Context:** Phase 7 of full Qwen3.5-4B multimodal/MTP plan

## 1. Problem and Root Cause

MTP integration required multi-token speculative drafting across continuous batching slots ($B=1..4$) without altering sampling semantics, drifting RNG sequences, or allowing partial draft rejections to corrupt recurrent DeltaNet / attention KV cache state.

Key findings:
1. **Target Hidden State Flow:** Qwen3.5 MTP head consumes normalized target hidden state ($h_p = \text{RMSNorm}(x_{\text{trunk}})$) paired with next-token embedding ($e_{p+1}$). Normalized target hidden states must be extracted before the shared LM head.
2. **Transactional Device Checkpointing:** DeltaNet states must be snapshotted device-to-device ($D2D$) via `memcpy_dtod` (CUDA) or `copy_from_buffer` (Metal) per slot. Attention KV caches only append during speculation; restoring logical length and copy-on-write cache slices prevents host readback overhead.
3. **Draft Alignment and Verification:** The speculative verifier samples tokens sequentially with cloned per-slot RNG. On mismatch, committed output includes accepted draft tokens plus the target model's divergent sampled token, catching up MTP's carryover state to the exact boundary.

## 2. Validation and Evidence

- Unit tests in `qwen35-batch/tests/mtp_transaction.rs`:
  - `cancellation_before_verify_emits_nothing` PASS
  - `mtp_matches_baseline_for_b1_to_b4_and_mixed_fallback` PASS
  - `reject_and_failures_restore_then_match_baseline` PASS
- Parity tests in `qwen35-batch/tests/real_qwen35_mtp.rs`:
  - `parity_holds_across_injected_rejections_and_fallbacks` PASS
  - `fixed_seed_multilingual_prompts_match_token_for_token_b1_to_b4` PASS
- Windows CUDA Phase 7 gate report (`bench/qwen35-artifacts/phase7-cuda-gate.json`):
  - English $B=1$ & $B=4$: exact match with baseline.
  - Russian $B=1$ & $B=4$: exact match with baseline.
  - Gate status: `pass`.

## 3. Rules and Ceilings

1. Speculative decoding must never mutate baseline greedy or stochastic outputs for any fixed seed.
2. On any MTP exception or mismatch, state rollback must restore exact committed boundaries before baseline execution resumes.
3. Production server keeps MTP staged until complete end-to-end promotion gates are evaluated.
