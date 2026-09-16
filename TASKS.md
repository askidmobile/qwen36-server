# TASKS

## 🚀 Active tasks

| # | Дата | Task | Plan | Status |
|---|------|------|------|--------|
| T-001 | 2026-08-24 | Prefill optimization: MoE grouped, DeltaNet tiles, launch overhead | [`2026-08-23-prefill-optimization.md`](docs/plans/2026-08-23-prefill-optimization.md) | ✅ Done (ceiling reached: 300 tok/s) |
| T-002 | 2026-09-04 | Выгрузка экспертов MoE в pinned RAM с кэшем горячих экспертов в VRAM (35B-A3B на 3060: 131K + MTP) | [`2026-09-04-moe-expert-offload.md`](docs/plans/2026-09-04-moe-expert-offload.md) | 🔄 In progress (Фазы 0–5 ✅, фаза 6: стабилизация) |
| T-003 | 2026-09-13 | SGLang concepts validation on yttri-win (prefix cache, PGRAPH, host timing) | [`2026-09-13-sglang-concepts-validation.md`](docs/plans/2026-09-13-sglang-concepts-validation.md) | 👀 In review (validation complete: branch-cache confirmed, overlap/PGRAPH deferred) |
| T-004 | 2026-09-13 | Branch-point prefix cache: multi checkpoint snapshots | [`2026-09-13-branch-point-prefix-cache.md`](docs/plans/2026-09-13-branch-point-prefix-cache.md) | ✅ Done (branch probe: 3.176s → 1.989s, 11 unit tests pass) |
| T-005 | 2026-09-16 | Паритет с llama.cpp на Ornith-1.5-9B: замеры, фазовые профили, контекст 128k | [`2026-09-16-head-to-head-llamacpp-and-phase-profile.md`](docs/research/2026-09-16-head-to-head-llamacpp-and-phase-profile.md) | 🔄 In progress (сессия 3: префил **19.65–19.71 с** против 19.52–19.54 у llama (−0.9 %), декод 30k 45.3–45.7 против 46.9–47.1 t/s (**−3.2 %**), короткий декод быстрее; dp4a-ветка MMQ отвергнута и выключена (§54); покернельное сравнение и протокол против шума (§55, §55.1); декод = **745 ядер на шаг**, 1.8 мс/шаг в зазорах (§56); сэмплер частот показал, что упирается в 170 Вт llama, а не мы (§57); следующий шаг: слияние микрокернеллов декода — квантование активаций в MMVQ, rmsnorm+badd, delta-prep) |

## ✅ Done

| # | Task | Status | Plan |
|---|------|--------|------|
