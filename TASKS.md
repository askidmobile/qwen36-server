# TASKS

## 🚀 Active tasks

| # | Дата | Task | Plan | Status |
|---|------|------|------|--------|
| T-001 | 2026-08-24 | Prefill optimization: MoE grouped, DeltaNet tiles, launch overhead | [`2026-08-23-prefill-optimization.md`](docs/plans/2026-08-23-prefill-optimization.md) | ✅ Done (ceiling reached: 300 tok/s) |
| T-002 | 2026-09-04 | Выгрузка экспертов MoE в pinned RAM с кэшем горячих экспертов в VRAM (35B-A3B на 3060: 131K + MTP) | [`2026-09-04-moe-expert-offload.md`](docs/plans/2026-09-04-moe-expert-offload.md) | 🔄 In progress (Фазы 0–5 ✅, фаза 6: стабилизация) |
| T-003 | 2026-09-13 | SGLang concepts validation on yttri-win (prefix cache, PGRAPH, host timing) | [`2026-09-13-sglang-concepts-validation.md`](docs/plans/2026-09-13-sglang-concepts-validation.md) | 👀 In review (validation complete: branch-cache confirmed, overlap/PGRAPH deferred) |
| T-004 | 2026-09-13 | Branch-point prefix cache: multi checkpoint snapshots | [`2026-09-13-branch-point-prefix-cache.md`](docs/plans/2026-09-13-branch-point-prefix-cache.md) | ✅ Done (branch probe: 3.176s → 1.989s, 11 unit tests pass) |
| T-005 | 2026-09-16 | Паритет с llama.cpp на Ornith-1.5-9B: замеры, фазовые профили, контекст 128k | [`2026-09-16-head-to-head-llamacpp-and-phase-profile.md`](docs/research/2026-09-16-head-to-head-llamacpp-and-phase-profile.md) | 🔄 In progress (сессия 3: dp4a-ветка MMQ для малых батчей собрана, сверена с эталоном и **отвергнута** — на sm_86 в 1.3–1.9× медленнее mma, по умолчанию выключена (§54); снят покернельный nsys-профиль на том же промпте, что у llama: наши ядра 9835 мс против 8566, разрыв — MMQ +459, DeltaNet +362, копии q/k/v +358 мс (§55); свежая сверка на 30232 токенах: префил 19.96 против 19.52 с (−2.2 %), декод 30k 45.4 против 47.0 t/s (−3.3 %), короткий декод по-прежнему быстрее; следующие шаги по весу: раскладка q/k/v под FA2 одним проходом, DeltaNet-ядро, порт архитектуры MMQ) |

## ✅ Done

| # | Task | Status | Plan |
|---|------|--------|------|
