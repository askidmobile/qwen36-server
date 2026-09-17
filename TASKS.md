# TASKS

## 🚀 Active tasks

| # | Дата | Task | Plan | Status |
|---|------|------|------|--------|
| T-001 | 2026-08-24 | Prefill optimization: MoE grouped, DeltaNet tiles, launch overhead | [`2026-08-23-prefill-optimization.md`](docs/plans/2026-08-23-prefill-optimization.md) | ✅ Done (ceiling reached: 300 tok/s) |
| T-002 | 2026-09-04 | Выгрузка экспертов MoE в pinned RAM с кэшем горячих экспертов в VRAM (35B-A3B на 3060: 131K + MTP) | [`2026-09-04-moe-expert-offload.md`](docs/plans/2026-09-04-moe-expert-offload.md) | 🔄 In progress (Фазы 0–5 ✅, фаза 6: стабилизация) |
| T-003 | 2026-09-13 | SGLang concepts validation on yttri-win (prefix cache, PGRAPH, host timing) | [`2026-09-13-sglang-concepts-validation.md`](docs/plans/2026-09-13-sglang-concepts-validation.md) | 👀 In review (validation complete: branch-cache confirmed, overlap/PGRAPH deferred) |
| T-004 | 2026-09-13 | Branch-point prefix cache: multi checkpoint snapshots | [`2026-09-13-branch-point-prefix-cache.md`](docs/plans/2026-09-13-branch-point-prefix-cache.md) | ✅ Done (branch probe: 3.176s → 1.989s, 11 unit tests pass) |
| T-005 | 2026-09-16 | Паритет с llama.cpp на Ornith-1.5-9B: замеры, фазовые профили, контекст 128k | [`2026-09-16-head-to-head-llamacpp-and-phase-profile.md`](docs/research/2026-09-16-head-to-head-llamacpp-and-phase-profile.md) | 🔄 In progress (сессия 4: **префил −2.4 % (20.00 против 19.54 с), декод 30k −2.0 % (45.98 против 46.92 t/s)** — лучший декод за все сессии, разрыв сократился с −3.2 %; порт Q-in-regs в пейдженный split-KV заработал после фикса смещения `sK` (smem 64→32 КБ, 3 CTA/SM, PPL 2.4680 эталонный), закрыты как бесполезные: смена раскладки KV-пула, число сплитов, фьюжн add+rmsnorm (−40 узлов графа, ноль эффекта), tail-split=0; отладочная инфраструктура: `CANDLE_FLASH_ATTN_MINIMAL=1` (сборка FA2 14–16 мин вместо 49) и пробник `fa_decode_probe` (проверка ядра за 2 с); остаток разрыва — дизайн ядер: MMQ (префил, +0.7 с на 30k, нужен конфигурируемый тайл из свежего llama.cpp) и split-KV (декод, фиксированная цена ~0.2 мкс на CTA)) |

## ✅ Done

| # | Task | Status | Plan |
|---|------|--------|------|
