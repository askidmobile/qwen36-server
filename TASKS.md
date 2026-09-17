# TASKS

## 🚀 Active tasks

| # | Дата | Task | Plan | Status |
|---|------|------|------|--------|
| T-001 | 2026-08-24 | Prefill optimization: MoE grouped, DeltaNet tiles, launch overhead | [`2026-08-23-prefill-optimization.md`](docs/plans/2026-08-23-prefill-optimization.md) | ✅ Done (ceiling reached: 300 tok/s) |
| T-002 | 2026-09-04 | Выгрузка экспертов MoE в pinned RAM с кэшем горячих экспертов в VRAM (35B-A3B на 3060: 131K + MTP) | [`2026-09-04-moe-expert-offload.md`](docs/plans/2026-09-04-moe-expert-offload.md) | 🔄 In progress (Фазы 0–5 ✅, фаза 6: стабилизация) |
| T-003 | 2026-09-13 | SGLang concepts validation on yttri-win (prefix cache, PGRAPH, host timing) | [`2026-09-13-sglang-concepts-validation.md`](docs/plans/2026-09-13-sglang-concepts-validation.md) | 👀 In review (validation complete: branch-cache confirmed, overlap/PGRAPH deferred) |
| T-004 | 2026-09-13 | Branch-point prefix cache: multi checkpoint snapshots | [`2026-09-13-branch-point-prefix-cache.md`](docs/plans/2026-09-13-branch-point-prefix-cache.md) | ✅ Done (branch probe: 3.176s → 1.989s, 11 unit tests pass) |
| T-005 | 2026-09-16 | Паритет с llama.cpp на Ornith-1.5-9B: замеры, фазовые профили, контекст 128k | [`2026-09-16-head-to-head-llamacpp-and-phase-profile.md`](docs/research/2026-09-16-head-to-head-llamacpp-and-phase-profile.md) | 🔄 In progress (сессия 7 итог: **декод быстрее во всех шести чередующихся парах** — 47.03…47.38 против 46.71…46.88 t/s; префил 19.79…19.89 против 19.75…19.76 (−0.5…0.7 %); сквозной wall 23.89…24.06 против ~24.03. Сделано: async D2H снимка границы, D2D seed (159→12.9 мс), restore хвоста 228→16.7 мс, узкий MMQ-тайл для m≤64 (−0.07 с). Проверено и отклонено: хвост в CUDA-графе (PGRAPH_MIN_T=0 даёт −38 мс, но риск вытеснения графов полных чанков из пула), хвост ≥256 токенов, KEEP_SINGLE_KV, PREFILL_SKIP_MID_LOGITS. Следующий шаг: sticky-slot снимок либо приоритетное вытеснение мелких графов) |

## ✅ Done

| # | Task | Status | Plan |
|---|------|--------|------|
