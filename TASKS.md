# TASKS

## 🚀 Active tasks

| # | Дата | Task | Plan | Status |
|---|------|------|------|--------|
| T-001 | 2026-08-24 | Prefill optimization: MoE grouped, DeltaNet tiles, launch overhead | [`2026-08-23-prefill-optimization.md`](docs/plans/2026-08-23-prefill-optimization.md) | ✅ Done (ceiling reached: 300 tok/s) |
| T-002 | 2026-09-04 | Выгрузка экспертов MoE в pinned RAM с кэшем горячих экспертов в VRAM (35B-A3B на 3060: 131K + MTP) | [`2026-09-04-moe-expert-offload.md`](docs/plans/2026-09-04-moe-expert-offload.md) | 🔄 In progress (Фазы 0–5 ✅, фаза 6: стабилизация) |
| T-003 | 2026-09-13 | SGLang concepts validation on yttri-win (prefix cache, PGRAPH, host timing) | [`2026-09-13-sglang-concepts-validation.md`](docs/plans/2026-09-13-sglang-concepts-validation.md) | 👀 In review (validation complete: branch-cache confirmed, overlap/PGRAPH deferred) |
| T-004 | 2026-09-13 | Branch-point prefix cache: multi checkpoint snapshots | [`2026-09-13-branch-point-prefix-cache.md`](docs/plans/2026-09-13-branch-point-prefix-cache.md) | ✅ Done (branch probe: 3.176s → 1.989s, 11 unit tests pass) |
| T-005 | 2026-09-16 | Паритет с llama.cpp на Ornith-1.5-9B: замеры, фазовые профили, контекст 128k | [`2026-09-16-head-to-head-llamacpp-and-phase-profile.md`](docs/research/2026-09-16-head-to-head-llamacpp-and-phase-profile.md) | ✅ **Цель достигнута (§80): префил и декод быстрее llama.cpp.** Пять чередующихся пар: префил 19.661 против 19.716 с (**+0.28 %**, быстрее во всех парах), декод 47.11 против 46.81 t/s (**+0.64 %**). Ключ — pool-backed записи кеша (`PREFIX_CACHE_POOL_BACKED=1`): вместо копии KV (~1 ГиБ) запись хранит ссылку на строки пула (поколение слота + длина), размер записи 50 МиБ; устаревшие записи отвергаются проверкой `pool_backed_valid`. Проверено: попадание даёт тот же sha, что холодный прогон; чужая запись отвергается; PPL 2.4680 без изменений; повторный запрос 0.66…0.68 с против 0.95 с. Остаток на будущее: hit-путь можно ускорить (rehydrate ~1 ГиБ device→device), декод-разрыв расширять дальше |
| T-006 | 2026-09-17 | Профиль CTX=128000 на 12 ГБ (int8-пул, paged-префил без графов) | [`2026-09-16-head-to-head-llamacpp-and-phase-profile.md`](docs/research/2026-09-16-head-to-head-llamacpp-and-phase-profile.md) §81 | 🟡 **128k работает и измерен**: 126 917-токенный промпт — префил 155.96 с (813 t/s), декод 24.65 t/s; 100k — 110.78 с / 27.88 t/s; 50k — 31.42 с / 39.48 t/s. Рабочий набор: `CTX=128000`, `KV_POOL_Q8=1`, `GRAPH_WINDOW=131072`, `PGRAPH=on`, `PGRAPH_MIN_T=1000000` (графы префила не захватываем — replay в 4…17 раз медленнее eager из-за WDDM), `PREFILL_CHUNK=8192`. Тупики: `PGRAPH=off` → OOM (single-slot F16-кэш 4 ГБ); захват графов префила → чанк 186…189 с; `PREFILL_CHUNK=16384` → OOM на ~127k. Открыто: попадание в prefix cache на холодном слоте с int8-пулом падает (`Q8 KV snapshot … restore_slot_state`, model_weights.rs:8217-8223); декод −11 % к F16-пулу (dequant в paged-внимании) |

## ✅ Done

| # | Task | Status | Plan |
|---|------|--------|------|
