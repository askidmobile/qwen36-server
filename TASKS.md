# TASKS

## 🚀 Active tasks

| # | Дата | Task | Plan | Status |
|---|------|------|------|--------|
| T-001 | 2026-08-24 | Prefill optimization: MoE grouped, DeltaNet tiles, launch overhead | [`2026-08-23-prefill-optimization.md`](docs/plans/2026-08-23-prefill-optimization.md) | ✅ Done (ceiling reached: 300 tok/s) |
| T-002 | 2026-09-04 | Выгрузка экспертов MoE в pinned RAM с кэшем горячих экспертов в VRAM (35B-A3B на 3060: 131K + MTP) | [`2026-09-04-moe-expert-offload.md`](docs/plans/2026-09-04-moe-expert-offload.md) | 🔄 In progress (Фазы 0–5 ✅, фаза 6: стабилизация) |
| T-003 | 2026-09-13 | SGLang concepts validation on yttri-win (prefix cache, PGRAPH, host timing) | [`2026-09-13-sglang-concepts-validation.md`](docs/plans/2026-09-13-sglang-concepts-validation.md) | 👀 In review (validation complete: branch-cache confirmed, overlap/PGRAPH deferred) |
| T-004 | 2026-09-13 | Branch-point prefix cache: multi checkpoint snapshots | [`2026-09-13-branch-point-prefix-cache.md`](docs/plans/2026-09-13-branch-point-prefix-cache.md) | ✅ Done (branch probe: 3.176s → 1.989s, 11 unit tests pass) |
| T-005 | 2026-09-16 | Паритет с llama.cpp на Ornith-1.5-9B: замеры, фазовые профили, контекст 128k | [`2026-09-16-head-to-head-llamacpp-and-phase-profile.md`](docs/research/2026-09-16-head-to-head-llamacpp-and-phase-profile.md) | 🔄 In progress (сессия 9: контрольное плечо `PREFIX_CACHE_MIN_TOKENS` показало, что **присутствие кеша бесплатно, а снятие снимка стоит ~150 мс**, из которых в фазах видно только 16 мс (`[pcap]`), а остальное — в межшаговой части; исключены дренаж токена (идёт до блока кеша), хранение записи и клон (9.2 мс). Прod: `FULL_HIT=0` (полная запись добавляет ещё +150 мс), холод 19.66…19.83, повтор 0.95 с; пять пар против llama.cpp: декод **+0.81 %**, префил −0.5 %. Следующий шаг — развести перенос снимка на отдельный стрим с pinned-буфером либо отложить снятие на момент после выдачи ответа) |

## ✅ Done

| # | Task | Status | Plan |
|---|------|--------|------|
