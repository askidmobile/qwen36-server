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

**Статус (2026-08-23): функционально работает, производительность отрицательная →
дефолт OFF, исследование причин — задача P0.5.**

Замер Qwen3.5-4B Q4_K_M, ctx 8192, SLOTS=2 (batched), промпт 6K:

| Конфиг | decode | TTFT |
|---|---|---|
| без MTP | **53 ток/с** | 2.8 с |
| MTP=1 (thin artifact) | **14.2 ток/с** | 3.6 с |

Функционал подтверждён: артефакт грузится, шедулер уходит в спекулятивные раунды,
вывод lossless. Замечание по маршрутам: SLOTS=1 → CandleEngine (без MTP),
SLOTS≥2 → BatchedEngine — замерять только на batched.

| # | Задача | Статус |
|---|---|---|
| 2.1 | Артефакт 35B-A3B: unsloth-репо содержит ПОЛНЫЕ модели со встроенным MTP — тонкого артефакта нет. Нужен конвертер официального MTP-checkpoint'а Qwen → наш thin-GGUF (пайплайн qwen35_artifacts.ps1) | open |
| 2.2 | Профиль | заменён env `QWEN36_MTP_PATH` (main.rs) — постоянное улучшение |
| 2.3 | Замер 4B | выполнен, см. таблицу |
| 2.4 | Критерий ≥10% прироста | НЕ выполнен — регрессия ×3.7 |
| **P0.5** | Профилирование раунда | **done, данные ниже** |

### P0.5 — Данные профилирования MTP-раунда (2026-08-23)

Замер Qwen3.5-4B @ctx~6K, `QWEN36_MTP_TIMING=1` (печать работала — прошлый grep
искал неверный тег; реальный формат `[mtp] slot K m begin/draft/verify/sample/accept/commit`):

```
[mtp] K=3 m=3 begin=1.4 draft=9..15 verify=22..24 sample=2 accept=0..16 commit=0.1 ms
```

- **Acceptance отличный**: почти все раунды m=3 из K=3 (полный accept); редкий
  откат m=2 стоит 15.6 мс (D2D restore KV).
- **draft 9–15 мс** — подозрительно дорого для одного блока: подозрение — D2H-sync
  после argmax каждого чернового токена (`to_vec1::<u32>()` ×3 за раунд).
- **verify 22–24 мс** — легитимная цена multi-token прохода через все блоки.
  Важно: **FA2-varlen в prefill/verify уже включён по умолчанию**
  (`candle_flash_attn::flash_attn(..., causal)`, откат `QWEN36_DISABLE_FLASH_PREFILL`)
  — фаза P2 в исходной постановке фактически выполнена ранее.
- **Главная аномалия**: сумма фаз ~45 мс/раунд, раундов ~43 на 128 токенов ≈ 2 с,
  а wall-time декода ≈ 9 с. **~75% времени — между раундами** (scheduler loop,
  sampler restore/checkpoint, push_verified, SSE-стриминг).
- Следующий шаг P0.5b: инструментировать межраундовый интервал; убрать D2H-sync
  в draft (batched argmax / async copy).

| # | Задача | Статус |
|---|---|---|
| 2.1 | Артефакт 35B-A3B: unsloth-репо содержит ПОЛНЫЕ модели со встроенным MTP — тонкого артефакта нет. Нужен конвертер официального MTP-checkpoint'а Qwen → наш thin-GGUF (пайплайн qwen35_artifacts.ps1) | open |
| 2.2 | Профиль | заменён env `QWEN36_MTP_PATH` (main.rs) — постоянное улучшение |
| 2.3 | Замер 4B | выполнен: 53→14.2 ток/с |
| 2.4 | Критерий ≥10% прироста | НЕ выполнен — регрессия ×3.7 |
| P0.5a | Данные раунда собраны | done |
| P0.5b | Инструментация dispatch-лупа ([mtp-agg]) + GPU-resident draft (один H2D/D2H на раунд): 38→44.4 ток/с (+17%). Остаток: verify 21.5мс, rollback 13.5мс при m<K, draft 8.8мс (=vocab-head) | done, follow-up open |
| P0.5c | Идеи след.: адаптивный width после частичного accept; кэш embedding последнего драфт-токена; батчинг verify двух слотов | open |

## Фаза W1 — Unsloth Studio как UI

| # | Задача |
|---|---|
| 3.1 | Установить Unsloth Desktop на yttri-win |
| 3.2 | Connections → Add Provider → Base URL `http://127.0.0.1:18099/v1`, ключ smoke-key |
| 3.3 | Чек-лист совместимости: GET /v1/models поля; SSE-стрим; tool calls (native формат); Think-toggle ↔ наши пресеты (`chat_template_kwargs.enable_thinking`); attachments → /v1/media |
| 3.4 | Глюки → задачи в TASKS.md; наш web/ пометить admin-fallback в README |

## Фаза P2 — Prefill: FA2 уже включён, искать вне attention

**Обновлено 2026-08-23:** flash-attn v2 causal в prefill-ветке CUDA уже активен
по умолчанию (P2 в исходной постановке выполнен). Замер `[pf] chunk blocks:
delta=160ms attn=137ms` на чанк 512 → при 47 чанках ≈14 с из 150 s общего
префилла @24K. **~135 с — вне блочных счётчиков**: подозреваемые — DeltaNet
prefill внутри чанка (token-by-token хвост), seed/copy KV в batched буферы,
boundary-logits + lm_head, аллокации пула. Следующий шаг — суммарный тайминг
чанка по фазам (расширить `[pf]`-строки) и точечный фикс доминанты.

| # | Задача |
|---|---|
| 4.1 | Расширить [pf]-тайминг: delta-inner / attn / ffn / seed-copy / boundary |
| 4.2 | Зафиксировать доминанту и завести точечную задачу |
| 4.3 | Критерий: префилл @24K ≤ 60 с |

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
- [x] P0 — функционал подтверждён на 4B; производительность отрицательная
      (53→14.2 ток/с), дефолт OFF; P0.5 профилирование раунда — open.
      Для 35B-A3B нужен конвертер тонкого MTP-артефакта
- [ ] W1.3.1–3.4
- [ ] P2.4.1–4.4
- [x] Замеры зафиксированы: short 36 ток/с стабильно; @24K разброс 5–12 ток/с
      из-за фоновой нагрузки десктопа + положение на «кромке» обрыва
