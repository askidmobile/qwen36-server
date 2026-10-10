# Tree speculative decoding по мотивам Uzu

Дата: 2026-10-09.
Контекст: [research](../research/2026-10-09-qwen35-mtp-linear-vs-baseline.md).
Референс: `trymirai/uzu`, PR #908 (`Use grouped attention for tree verify`, commit `fba8c21`).

Статус: фазы 1-5 пройдены, benchmark выполнен 2026-10-10. Результат — отрицательный: tree не быстрее baseline и linear MTP. Отчёт: [2026-10-10-tree-speculative-benchmark.md](../research/2026-10-10-tree-speculative-benchmark.md).

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

- Decode t/s выше baseline на Qwen3.5-9B. **Не выполнен:** tree 7.2-7.5 t/s против baseline 53-54 t/s.
- Нет деградации VRAM и TTFT. **Частично:** prefill не изменился, VRAM вырос с 7648 до 8614 MiB.
- `used=true` и `accepted/drafted ≥ 0.5`. **Не выполнен:** tree acceptance 2-8%, linear W1 70-87%, но linear всё равно медленнее baseline.
- Отсутствие выпадения токенов и порчи выдачи. **Стабильность достигнута:** после `6ab9a2d2` рассинхрон paged pool и ошибки `позиция против длины пула` исчезли.

## Итог 2026-10-10

Корректность tree-контура доведена до рабочего состояния: DFS-порядок, RoPE по глубине, обход ветки, перестройка принятого пути, ONEPASS с trie-маской. Ускорения нет, потому что MTP-драфт и целевой DeltaNet не tree-aware: узлы веток используют общее последовательное состояние. Для продолжения нужны per-node MTP state и per-node DeltaNet state; без них tree не может конкурировать с baseline. Подробности и все метрики — в [отчёте](../research/2026-10-10-tree-speculative-benchmark.md).
