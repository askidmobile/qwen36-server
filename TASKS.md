# TASKS

## 🚀 Active tasks

| # | Дата | Task | Plan | Status |
|---|------|------|------|--------|
| T-001 | 2026-08-24 | Prefill optimization: MoE grouped, DeltaNet tiles, launch overhead | [`2026-08-23-prefill-optimization.md`](docs/plans/2026-08-23-prefill-optimization.md) | ✅ Done (ceiling reached: 300 tok/s) |
| T-002 | 2026-09-04 | Выгрузка экспертов MoE в pinned RAM с кэшем горячих экспертов в VRAM (35B-A3B на 3060: 131K + MTP) | [`2026-09-04-moe-expert-offload.md`](docs/plans/2026-09-04-moe-expert-offload.md) | 🔄 In progress (Фазы 0–5 ✅, фаза 6: стабилизация) |
| T-003 | 2026-09-13 | SGLang concepts validation on yttri-win (prefix cache, PGRAPH, host timing) | [`2026-09-13-sglang-concepts-validation.md`](docs/plans/2026-09-13-sglang-concepts-validation.md) | 👀 In review (validation complete: branch-cache confirmed, overlap/PGRAPH deferred) |
| T-004 | 2026-09-13 | Branch-point prefix cache: multi checkpoint snapshots | [`2026-09-13-branch-point-prefix-cache.md`](docs/plans/2026-09-13-branch-point-prefix-cache.md) | ✅ Done (branch probe: 3.176s → 1.989s, 11 unit tests pass) |
| T-005 | 2026-09-16 | Паритет с llama.cpp на Ornith-1.5-9B: замеры, фазовые профили, контекст 128k | [`2026-09-16-head-to-head-llamacpp-and-phase-profile.md`](docs/research/2026-09-16-head-to-head-llamacpp-and-phase-profile.md) | 🔄 In progress (сессия 8: реализован вариант (B) — полный промпт в записи кеша с логитами (`PREFIX_CACHE_FULL_HIT=1`, по умолчанию выкл.): попадание 1.75 с против 21.07 с холодного, но **восстановление состояния не fully** — два попадания c одним снимком дают разные ответы, поэтому флаг выключен. Найден корень «недетерминизма»: `SAMPLING_LOCK=1` игнорирует клиентскую temperature, из-за чего прежние сравнения жадных ответов были бессмысленны; с `SAMPLING_LOCK=0` прод-путь детерминирован (3/3 одинаковых хеша), а cold-ответы с хвост-сплитом и без него совпали побитово — сегодняшние префил-правки численно нейтральны. Следующий шаг: найти утечку состояния между запросами в слоте и включить (B)) |

## ✅ Done

| # | Task | Status | Plan |
|---|------|--------|------|
