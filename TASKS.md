# TASKS

## 🚀 Active tasks

| # | Дата | Task | Plan | Status |
|---|------|------|------|--------|
| T-001 | 2026-08-24 | Prefill optimization: MoE grouped, DeltaNet tiles, launch overhead | [`2026-08-23-prefill-optimization.md`](docs/plans/2026-08-23-prefill-optimization.md) | ✅ Done (ceiling reached: 300 tok/s) |
| T-002 | 2026-09-04 | Выгрузка экспертов MoE в pinned RAM с кэшем горячих экспертов в VRAM (35B-A3B на 3060: 131K + MTP) | [`2026-09-04-moe-expert-offload.md`](docs/plans/2026-09-04-moe-expert-offload.md) | 🔄 In progress (Фазы 0–5 ✅, фаза 6: стабилизация) |
| T-003 | 2026-09-13 | SGLang concepts validation on yttri-win (prefix cache, PGRAPH, host timing) | [`2026-09-13-sglang-concepts-validation.md`](docs/plans/2026-09-13-sglang-concepts-validation.md) | 👀 In review (validation complete: branch-cache confirmed, overlap/PGRAPH deferred) |
| T-004 | 2026-09-13 | Branch-point prefix cache: multi checkpoint snapshots | [`2026-09-13-branch-point-prefix-cache.md`](docs/plans/2026-09-13-branch-point-prefix-cache.md) | ✅ Done (branch probe: 3.176s → 1.989s, 11 unit tests pass) |
| T-005 | 2026-09-16 | Паритет с llama.cpp на Ornith-1.5-9B: замеры, фазовые профили, контекст 128k | [`2026-09-16-head-to-head-llamacpp-and-phase-profile.md`](docs/research/2026-09-16-head-to-head-llamacpp-and-phase-profile.md) | 🔄 In progress (сессия 7: три чередующиеся пары — **декод 47.03 против 46.82 t/s (+0.45 %)**, префил 19.837 против 19.70 (−0.7 %). Сняты два «тяжёлых» места префикс-пути: D2H снимка границы 186 мс → в воркер (`[pcap] host=async`), seed рекуррентного состояния 159 мс → D2D 12.9 мс (`QWEN36_D2D_SEED=0` — откат); побочно restore хвоста 228 → 16.7 мс. Проверено: trim 0.0 мс (159.8 мс — это реальная работа GPU хвоста), PPL 2.4680 без изменений, кеш-хит 0.968 с. Остаток: эффективность хвостового чанка (m=24 при тайле 64/128) — следующий шаг sticky-slot снимок или q8-снимок) |

## ✅ Done

| # | Task | Status | Plan |
|---|------|--------|------|
