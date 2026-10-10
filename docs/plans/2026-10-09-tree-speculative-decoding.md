# Tree speculative decoding по мотивам Uzu

Дата: 2026-10-09.
Контекст: [research](../research/2026-10-09-qwen35-mtp-linear-vs-baseline.md).
Референс: `trymirai/uzu`, PR #908 (`Use grouped attention for tree verify`, commit `fba8c21`).

Статус: фазы 1-5 пройдены. Benchmark перезапущен 2026-10-11 с парной MTP-головой из того же Q4_K_M GGUF. Результат — отрицательный: tree не быстрее baseline и linear MTP. Отчёт: [2026-10-10-tree-speculative-benchmark.md](../research/2026-10-10-tree-speculative-benchmark.md).

## Цель

Получить на Qwen3.5-9B/Qwen3.5-9B-like модель выигрыш по decode t/s против baseline не за счёт «большей ширины линейного MTP», а за счёт однопроходной проверки ветвящегося дерева.

## Фазы

1. **Тrie-структуры и метрики**
   - [x] `TrieNode` с токеном, родителем, диапазоном поддерева, logprob.
   - [x] `FlatTrie` с линейным порядком и диапазонами поддеревьев.
   - [x] `TrieAccept` по фактическим target-токенам.
   - [x] Юнит-тесты на маленьком дереве: обход, отбор по бюджету, откат по принятому пути.

2. **Ветвящийся драфт**
   - [x] Алгоритм `FlatTrie::build_tree` с бюджетом, глубиной и ветвлением.
   - [x] Общий префикс учитывается один раз.
   - [x] `TreeStaging` для CUDA: токены, родители, глубины, диапазоны поддеревьев.
   - [x] Интеграция построения дерева в `Qwen35Mtp::draft_tree` (eager, корректный контур).
   - [x] Подключение стейджинга к реальному CUDA-пути attention (`speculative_draft_tree` загружает `TreeStaging`).

3. **Однопроходная проверка**
   - [x] API flash-attention принимает `trie_ranges` ([start,end] на узел).
   - [x] CUDA-маска разрешает общий committed-префикс и поддерево текущего узла.
   - [x] `PagedAttn` валидирует форму/dtype `trie_ranges` и передаёт указатель в ядро.
   - [x] Интеграция tree-verify в `ModelWeights::forward_verify_tree_graphed`.
   - [x] Интеграция tree-verify в `Qwen35BatchAdapter::paged_graph_run`.
   - [x] Safe-откат DeltaNet в tree-режиме: checkpoint + causal prefill-перестройка принятого пути в paged pool (commit `6ab9a2d2`).
   - [x] `VERIFY_ONEPASS=1` с реальной trie-маской проходит без зависания (замеры 2026-10-10).
   - [ ] Специализированное tree-ядро DeltaNet (параллельные пути).
   - [ ] CUDA-тест trie-маски против построчного эталона.
   - [x] Host-эталон `trie_mask_allows` с unit-тестом.

4. **Приём и откат**
   - [x] Идти по ветке, а не по линейному zip: `accepted/drafted` считается по фактическому пути (commit `dd7f7c0e`).
   - [x] Откатить KV и DeltaNet-state к границе принятого префикса и перестроить принятый путь (commit `6ab9a2d2`).
   - [x] Обновить `MtpUsage` с `drafted`, `accepted`, `used`, `fallback_category`.

5. **Benchmark и интеграция**
   - [x] A/B на Qwen3.5-9B против baseline.
   - [x] Сравнить с текущим линейным MTP (`MTP_WIDTH=1..3`).
   - [x] Проверить VRAM и TTFT.
   - [ ] Если выигрыш устойчив — включить по умолчанию. Выигрыша нет: tree ~7.2-7.5 t/s, linear ~41-46 t/s, baseline ~53-54 t/s.

## Технические зацепки

- `engine/qwen35-batch/src/real/mtp.rs` — сейчас линейный драфт; сюда добавляется построение дерева.
- `engine/qwen35-batch/src/real/adapter.rs` — `speculative_verify` и `speculative_accept`; сюда встраивается однопроходная проверка дерева.
- `engine/qwen35-batch/src/real/paged_attn.rs` — единственный вызов attention-ядра; сюда нужно добавить trie-маску.
- `engine/qwen35-batch/src/real/paged_kv_cuda.rs` — стейджинг длин для varlen; нужен отдельный буфер trie-узлов.
- `engine/candle-kernels/` — если текущий flash-attention не принимает произвольную маску, сюда добавляется специализированное ядро tree-attention.

## Критерии успеха

- Decode t/s выше baseline на Qwen3.5-9B. **Не выполнен:** tree 7.54-7.65 t/s, linear MTP 44.12-44.28 t/s против baseline 53.73-53.75 t/s.
- Нет деградации VRAM и TTFT. **Частично:** prefill почти не изменился, VRAM вырос с 7698 до 8340 MiB.
- `used=true` и `accepted/drafted ≥ 0.5`. **Не выполнен:** tree acceptance 5.6-11.3%, linear W1 73.0-87.3%, но linear всё равно медленнее baseline.
- Отсутствие выпадения токенов и порчи выдачи. **Стабильность достигнута:** после `6ab9a2d2` рассинхрон paged pool и ошибки `позиция против длины пула` исчезли.

## Исправление pairing 2026-10-11

Первые замеры использовали `lmstudio-community/Qwen3.5-9B-Q4_K_M` и отдельную голову `mtp-Ornith-1.5-9B-head-Q8_0.gguf` — это непарная конфигурация. Корректный вариант: `unsloth/Qwen3.5-9B-MTP-GGUF/Qwen3.5-9B-Q4_K_M.gguf`, где MTP-слой (`qwen35.nextn_predict_layers=1`, `blk.32.nextn.*`) встроен в тот же Q4_K_M GGUF. `MODEL` и `MTP_PATH` указывают на один файл.

| Режим | Decode, t/s | Acceptance |
| --- | ---: | ---: |
| `MTP=0`, baseline | 53.73 / 53.74 / 53.75 | — |
| `MTP=1`, W1, полный словарь | 38.00 / 39.77 / 40.00 | 85.7-87.3% |
| `MTP=1`, W1, `MTP_VOCAB_TOP=32768` | 44.12 / 44.23 / 44.28 | 73.0-76.2% |
| `MTP=1`, tree K=8, полный словарь, ONEPASS=1 | 4.53 / 4.59 / 4.60 | 6.8-11.3% |
| `MTP=1`, tree K=8, `MTP_VOCAB_TOP=32768`, ONEPASS=1 | 7.54 / 7.61 / 7.65 | 5.6-8.7% |

Prefill на 6412 токенах: baseline 4.526 s, linear 4.644 s, tree 4.416 s. VRAM: baseline 7698 MiB, linear 7858 MiB, tree 8340 MiB.

## Итог 2026-10-11

Парная MTP-голова загружается и даёт высокий linear acceptance (73-87%), но baseline всё равно быстрее: лучший linear-режим отстаёт примерно на 18%. Tree остаётся в 7 раз медленнее baseline и даёт acceptance 5.6-11.3%. Причина не в pairing и не в trie-маске, а в том, что MTP-драфт и целевой DeltaNet не tree-aware: узлы веток используют общее последовательное состояние. Для продолжения нужны per-node MTP state и per-node DeltaNet state. Подробности — в [отчёте](../research/2026-10-10-tree-speculative-benchmark.md).
