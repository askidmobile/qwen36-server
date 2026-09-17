# TASKS

## 🚀 Active tasks

| # | Дата | Task | Plan | Status |
|---|------|------|------|--------|
| T-001 | 2026-08-24 | Prefill optimization: MoE grouped, DeltaNet tiles, launch overhead | [`2026-08-23-prefill-optimization.md`](docs/plans/2026-08-23-prefill-optimization.md) | ✅ Done (ceiling reached: 300 tok/s) |
| T-002 | 2026-09-04 | Выгрузка экспертов MoE в pinned RAM с кэшем горячих экспертов в VRAM (35B-A3B на 3060: 131K + MTP) | [`2026-09-04-moe-expert-offload.md`](docs/plans/2026-09-04-moe-expert-offload.md) | 🔄 In progress (Фазы 0–5 ✅, фаза 6: стабилизация) |
| T-003 | 2026-09-13 | SGLang concepts validation on yttri-win (prefix cache, PGRAPH, host timing) | [`2026-09-13-sglang-concepts-validation.md`](docs/plans/2026-09-13-sglang-concepts-validation.md) | 👀 In review (validation complete: branch-cache confirmed, overlap/PGRAPH deferred) |
| T-004 | 2026-09-13 | Branch-point prefix cache: multi checkpoint snapshots | [`2026-09-13-branch-point-prefix-cache.md`](docs/plans/2026-09-13-branch-point-prefix-cache.md) | ✅ Done (branch probe: 3.176s → 1.989s, 11 unit tests pass) |
| T-005 | 2026-09-16 | Паритет с llama.cpp на Ornith-1.5-9B: замеры, фазовые профили, контекст 128k | [`2026-09-16-head-to-head-llamacpp-and-phase-profile.md`](docs/research/2026-09-16-head-to-head-llamacpp-and-phase-profile.md) | 🔄 In progress (сессия 5: **декод впервые быстрее llama.cpp — 47.04…47.38 против 46.83…46.88 t/s на 30k**; причина найдена: `n_blocks_per_split` в split-KV считался от окна пула (65536), а не от реальной длины KV, из-за чего 7 CTA из 14 работали вхолостую; вторая правка — эвристика сплитов «одна волна по 2 блока на SM» (14 вместо 63) после развёртки `fa_decode_probe` 1k…91k. Stream-k в MMQ проверен и закрыт отрицательно (19.915 против 19.828 с). Префил 19.86–19.95 против 19.68 у llama при включённом prefix-кеше; с `PREFIX_CACHE_MIB=0` — 19.66, то есть остаток это цена снятия снапшота. PPL 2.4680 / tok_hash совпал) |

## ✅ Done

| # | Task | Status | Plan |
|---|------|--------|------|
