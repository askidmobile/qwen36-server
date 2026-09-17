# TASKS

## 🚀 Active tasks

| # | Дата | Task | Plan | Status |
|---|------|------|------|--------|
| T-001 | 2026-08-24 | Prefill optimization: MoE grouped, DeltaNet tiles, launch overhead | [`2026-08-23-prefill-optimization.md`](docs/plans/2026-08-23-prefill-optimization.md) | ✅ Done (ceiling reached: 300 tok/s) |
| T-002 | 2026-09-04 | Выгрузка экспертов MoE в pinned RAM с кэшем горячих экспертов в VRAM (35B-A3B на 3060: 131K + MTP) | [`2026-09-04-moe-expert-offload.md`](docs/plans/2026-09-04-moe-expert-offload.md) | 🔄 In progress (Фазы 0–5 ✅, фаза 6: стабилизация) |
| T-003 | 2026-09-13 | SGLang concepts validation on yttri-win (prefix cache, PGRAPH, host timing) | [`2026-09-13-sglang-concepts-validation.md`](docs/plans/2026-09-13-sglang-concepts-validation.md) | 👀 In review (validation complete: branch-cache confirmed, overlap/PGRAPH deferred) |
| T-004 | 2026-09-13 | Branch-point prefix cache: multi checkpoint snapshots | [`2026-09-13-branch-point-prefix-cache.md`](docs/plans/2026-09-13-branch-point-prefix-cache.md) | ✅ Done (branch probe: 3.176s → 1.989s, 11 unit tests pass) |
| T-005 | 2026-09-16 | Паритет с llama.cpp на Ornith-1.5-9B: замеры, фазовые профили, контекст 128k | [`2026-09-16-head-to-head-llamacpp-and-phase-profile.md`](docs/research/2026-09-16-head-to-head-llamacpp-and-phase-profile.md) | 🔄 In progress (сессия 6: декод **47.04…47.38 против 46.88…46.80** — быстрее llama.cpp; префил с выключенным кешем 19.63…19.68 = паритет, с включённым 19.78…19.90 = −0.8…1.2 %, и это ровно цена портативного снапшота prefix-кеша: `snap+seed` хвостового чанка 201 мс + `restore` 228 мс. Снятие снапшота ушло в отдельный поток (`HostSnapshotWorker`), D2H больше не держит первый токен: цена кеша упала с 0.25…0.35 до 0.15…0.20 с. Проверено и не помогает: `KEEP_SINGLE_KV=1`, тайлы префила 0/6/8, `PREFILL_SKIP_MID_LOGITS=1`. Следующий шаг: снимать снимок на границе последнего полного чанка и не разрывать владение состоянием) |

## ✅ Done

| # | Task | Status | Plan |
|---|------|--------|------|
