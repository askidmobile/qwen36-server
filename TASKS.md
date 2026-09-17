# TASKS

## 🚀 Active tasks

| # | Дата | Task | Plan | Status |
|---|------|------|------|--------|
| T-001 | 2026-08-24 | Prefill optimization: MoE grouped, DeltaNet tiles, launch overhead | [`2026-08-23-prefill-optimization.md`](docs/plans/2026-08-23-prefill-optimization.md) | ✅ Done (ceiling reached: 300 tok/s) |
| T-002 | 2026-09-04 | Выгрузка экспертов MoE в pinned RAM с кэшем горячих экспертов в VRAM (35B-A3B на 3060: 131K + MTP) | [`2026-09-04-moe-expert-offload.md`](docs/plans/2026-09-04-moe-expert-offload.md) | 🔄 In progress (Фазы 0–5 ✅, фаза 6: стабилизация) |
| T-003 | 2026-09-13 | SGLang concepts validation on yttri-win (prefix cache, PGRAPH, host timing) | [`2026-09-13-sglang-concepts-validation.md`](docs/plans/2026-09-13-sglang-concepts-validation.md) | 👀 In review (validation complete: branch-cache confirmed, overlap/PGRAPH deferred) |
| T-004 | 2026-09-13 | Branch-point prefix cache: multi checkpoint snapshots | [`2026-09-13-branch-point-prefix-cache.md`](docs/plans/2026-09-13-branch-point-prefix-cache.md) | ✅ Done (branch probe: 3.176s → 1.989s, 11 unit tests pass) |
| T-005 | 2026-09-16 | Паритет с llama.cpp на Ornith-1.5-9B: замеры, фазовые профили, контекст 128k | [`2026-09-16-head-to-head-llamacpp-and-phase-profile.md`](docs/research/2026-09-16-head-to-head-llamacpp-and-phase-profile.md) | 🔄 In progress (сессия 7: декод 47.3 против 46.6…46.9 — быстрее llama.cpp; префил 19.74…19.85 против 19.57…19.69. Инструментирован префикс-путь (`[pcap]`/`[pfin]`): D2H снимка границы 189.7 мс, seed финального чанка 159.3 мс. Seed переведён на D2D (`seed_slot_batched_from_device` + `seed_slot_cuda_state_from_single`) — TTFT 19.739/19.851 против 19.802/19.904 и декод +0.45 t/s, откат `QWEN36_D2D_SEED=0`; PPL 2.4680 не сдвинулся. Остаток: D2H снимка границы (+ чекпоинт 16k) — следующий шаг q8-снимок или sticky-slot путь) |

## ✅ Done

| # | Task | Status | Plan |
|---|------|--------|------|
