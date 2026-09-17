# TASKS

## 🚀 Active tasks

| # | Дата | Task | Plan | Status |
|---|------|------|------|--------|
| T-001 | 2026-08-24 | Prefill optimization: MoE grouped, DeltaNet tiles, launch overhead | [`2026-08-23-prefill-optimization.md`](docs/plans/2026-08-23-prefill-optimization.md) | ✅ Done (ceiling reached: 300 tok/s) |
| T-002 | 2026-09-04 | Выгрузка экспертов MoE в pinned RAM с кэшем горячих экспертов в VRAM (35B-A3B на 3060: 131K + MTP) | [`2026-09-04-moe-expert-offload.md`](docs/plans/2026-09-04-moe-expert-offload.md) | 🔄 In progress (Фазы 0–5 ✅, фаза 6: стабилизация) |
| T-003 | 2026-09-13 | SGLang concepts validation on yttri-win (prefix cache, PGRAPH, host timing) | [`2026-09-13-sglang-concepts-validation.md`](docs/plans/2026-09-13-sglang-concepts-validation.md) | 👀 In review (validation complete: branch-cache confirmed, overlap/PGRAPH deferred) |
| T-004 | 2026-09-13 | Branch-point prefix cache: multi checkpoint snapshots | [`2026-09-13-branch-point-prefix-cache.md`](docs/plans/2026-09-13-branch-point-prefix-cache.md) | ✅ Done (branch probe: 3.176s → 1.989s, 11 unit tests pass) |
| T-005 | 2026-09-16 | Паритет с llama.cpp на Ornith-1.5-9B: замеры, фазовые профили, контекст 128k | [`2026-09-16-head-to-head-llamacpp-and-phase-profile.md`](docs/research/2026-09-16-head-to-head-llamacpp-and-phase-profile.md) | 🔄 In progress (сессия 7: **декод +0.83 %** (47.17 против 46.78 t/s, быстрее во всех трёх парах), префил −0.67 % (19.886 против 19.753). Узкий MMQ-тайл для m≤64 (`MMQ_SMALL_TILE`, дефолт вкл.) дал −0.07 с к префилу; PPL с хвостом 2.4721 и tok_hash совпали, но жадная выдача 64 токенов разошлась (порядок накопления в хвосте → редкие перевороты top-1). Сняты D2H снимка границы (186 мс → воркер) и seed (159 → 12.9 мс D2D). Остаток — хвостовой чанк 24 токена (122 мс GPU) как цена выравненной границы кеша; следующий шаг sticky-slot снимок) |

## ✅ Done

| # | Task | Status | Plan |
|---|------|--------|------|
