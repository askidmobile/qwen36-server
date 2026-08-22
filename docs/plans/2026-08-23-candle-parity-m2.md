# Plan: candle-parity M2 (спека docs/specs/2026-08-23-candle-parity-m2.md)

**Дата:** 2026-08-23. Порядок: P1 → P0 → W1 → P2 → P4 → P3 → P5.

## Фаза P1 — Инкрементальное f16-зеркало KV (ВЫПОЛНЕНА, результат отрицательный на 12 GB)

**Итог (2026-08-23):** механизм реализован полностью и корректен, но на
RTX 3060 12 GB даёт регрессию: упреждающие зеркала (480 MiB на слот 0 при
cap 24K) оставляют <100 MiB на транзиенты префилла → всё уходит в WDDM shared.
Дефолт ВЫКЛЮЧЕН; включение — `QWEN36_KV_MIRROR_TOKENS` (+`..._PREPARE` для
упреждающего размещения). Целесообразно на картах ≥24 GB или при гарантированно
одном активном длинном запросе. Коммит форка `6496242e`.
Настоящее решение остаточного разрыва (11 vs 36 ток/с) — q8-aware FA2 kernel,
перенесено в перспективу после P5.

Файлы: `candle-fork/qwen35-batch/src/real/model_weights.rs`.

| # | Задача | Детали |
|---|---|---|
| 1.1 | Структуры | `struct KvMirror { k: Tensor, v: Tensor, valid_tokens: usize }` (flat [cap*row] f16). В `GatedAttentionLayer`: `kv_mirror: Vec<Option<KvMirror>>` по decode_capacity(). В `ModelWeights`: `kv_mirror_budget: Arc<AtomicI64>` (элементы), init из `QWEN36_KV_MIRROR_MIB` (default 640) |
| 1.2 | ensure_mirror(slot, kv_len) | None→аллокация cap=max(need,4096) с проверкой бюджета (fetch_sub, при отказе вернуть False); рост кэша → realloc ×2 c копией старого; возвращает bool «зеркало активно» |
| 1.3 | top-up | while valid<kv_len: чанк ≤4096 через `prefix_chunk_f16(valid,cs)` → `slice_set` во flat; valid+=cs |
| 1.4 | Инвалидации | `shift_and_append` → valid=0 (полная перезаливка); конец `seed_slot_batched` → valid=0; рост q8-кэша НЕ инвалидирует (логическое содержимое то же) |
| 1.5 | Хот-путь | порядок: mirror → shared scratch (Stage-1 остаётся fallback) → f16_prefix. Budget передаётся `&Arc<AtomicI64>` из hidden_inner (клон до цикла блоков) |
| 1.6 | Снятие дебаг-принтов | `[attn-decode-b]`, `[adb]` уже сняты; оставить одноразовый `[kv] mirror:` |
| 1.7 | Проверки локально | cargo check metal; parity не затронут (та же math деанва) |

Валидация на yttri-win: short bench ≥35 ток/с; @24K ≥25 ток/с; лог `[kv] mirror: ...`
один раз на слой-слот; VRAM ≤ 96%.

## Фаза P0 — MTP на 35B-A3B

| # | Задача | Команда/файл |
|---|---|---|
| 2.1 | Артефакт | `hf download unsloth/Qwen3.6-35B-A3B-MTP-GGUF --include "*Q8_0*"` → D:\Models\unsloth\ |
| 2.2 | Профиль | component mtp в profile YAML / env-путь (как для 4B — см. BD-024 pipeline) |
| 2.3 | Замер | bench + longctx @8K и @24K, MTP=1 vs 0, draft-n 1..3 |
| 2.4 | Критерий | прирост ≥10% без падения fixed-seed parity; иначе документируем «выключено по умолчанию» |

## Фаза W1 — Unsloth Studio как UI

| # | Задача |
|---|---|
| 3.1 | Установить Unsloth Desktop на yttri-win |
| 3.2 | Connections → Add Provider → Base URL `http://127.0.0.1:18099/v1`, ключ smoke-key |
| 3.3 | Чек-лист совместимости: GET /v1/models поля; SSE-стрим; tool calls (native формат); Think-toggle ↔ наши пресеты (`chat_template_kwargs.enable_thinking`); attachments → /v1/media |
| 3.4 | Глюки → задачи в TASKS.md; наш web/ пометить admin-fallback в README |

## Фаза P2 — Prefill FA2 varlen

| # | Задача |
|---|---|
| 4.1 | Точка: `forward_attn_with_rope` ветка seq>1 CUDA (сейчас materialized scores → softmax_last_dim, строки ~3226/3339) |
| 4.2 | Заменить на `candle_flash_attn::flash_attn(q,k,v,scale,true)` (causal), F16-каст Q/K/V |
| 4.3 | Parity: fixed-seed продолжение совпадает (допустим дрейф <1e-3 logits — зафиксировать порог) |
| 4.4 | DeltaNet-блоки: убедиться что `[pf] chunk` путь не деградировал; узкое место после — отдельная задача FR-P3 |

## Фаза P4 — CUDA graphs (после P1/P2 замеров)

Отладить `decode_batch_graphed` + paged pool на 35B-A3B; критерий включения —
≥10% decode, бит-идентичный вывод.

## Фаза P5 — Expert offload (отдельная спека после P0–P2)

Эскиз: tensor-name router при загрузке (`blk.*.ffn_*exp*` → CPU mmap),
CPU IQQ/QKK gemm (есть ядра k_quants), prefetch-очередь активных экспертов
(CUDA stream + events), vram_plan: профиль offload.

## Прогресс

- [x] Спека + план (этот файл)
- [x] P1.1–1.7 — реализовано; на 12 GB результат отрицательный, opt-in (см. выше)
- [ ] P0.2.1–2.4 — артефакт скачивается, замер следующим шагом
- [ ] W1.3.1–3.4
- [ ] P2.4.1–4.4
- [x] Замеры зафиксированы: short 36 ток/с стабильно; @24K разброс 5–12 ток/с
      из-за фоновой нагрузки десктопа + положение на «кромке» обрыва
