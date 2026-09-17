# TASKS

## 🚀 Active tasks

| # | Дата | Task | Plan | Status |
|---|------|------|------|--------|
| T-001 | 2026-08-24 | Prefill optimization: MoE grouped, DeltaNet tiles, launch overhead | [`2026-08-23-prefill-optimization.md`](docs/plans/2026-08-23-prefill-optimization.md) | ✅ Done (ceiling reached: 300 tok/s) |
| T-002 | 2026-09-04 | Выгрузка экспертов MoE в pinned RAM с кэшем горячих экспертов в VRAM (35B-A3B на 3060: 131K + MTP) | [`2026-09-04-moe-expert-offload.md`](docs/plans/2026-09-04-moe-expert-offload.md) | 🔄 In progress (Фазы 0–5 ✅, фаза 6: стабилизация) |
| T-003 | 2026-09-13 | SGLang concepts validation on yttri-win (prefix cache, PGRAPH, host timing) | [`2026-09-13-sglang-concepts-validation.md`](docs/plans/2026-09-13-sglang-concepts-validation.md) | 👀 In review (validation complete: branch-cache confirmed, overlap/PGRAPH deferred) |
| T-004 | 2026-09-13 | Branch-point prefix cache: multi checkpoint snapshots | [`2026-09-13-branch-point-prefix-cache.md`](docs/plans/2026-09-13-branch-point-prefix-cache.md) | ✅ Done (branch probe: 3.176s → 1.989s, 11 unit tests pass) |
| T-005 | 2026-09-16 | Паритет с llama.cpp на Ornith-1.5-9B: замеры, фазовые профили, контекст 128k | [`2026-09-16-head-to-head-llamacpp-and-phase-profile.md`](docs/research/2026-09-16-head-to-head-llamacpp-and-phase-profile.md) | 🔄 In progress (сессия 7: декод **+0.58…0.83 %** (47.1 против 46.8 t/s), префил −0.5 %. §77 показал: **без обвязки кеша наш префил быстрее** (19.55…19.67 против 19.65…19.75), GPU во время префила занят непрерывно, а платим мы 0.12…0.23 с за портативный кеш, из которых ~0.09…0.12 с — хвостовой чанк 24 токена. Следующий шаг: (A) checkpoint-offset в ядрах delta/conv1d, чтобы снимать границу внутри большого чанка, либо (B) класть в кеш полный промпт с логитами и разрешить попадание ровно в длину) |

## ✅ Done

| # | Task | Status | Plan |
|---|------|--------|------|
