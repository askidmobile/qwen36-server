# Tree speculative decoding по мотивам Uzu

Дата: 2026-10-09.
Контекст: [research](../research/2026-10-09-qwen35-mtp-linear-vs-baseline.md).
Референс: `trymirai/uzu`, PR #908 (`Use grouped attention for tree verify`, commit `fba8c21`).

Статус: фаза 1 начата — базовые `TrieNode`/`FlatTrie` и приём пути добавлены в `yttri-forge` (коммиты `45b4deb1`, `3d8e58ee`, `e05e9a7e`).

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
   - [ ] Интеграция построения дерева в `Qwen35Mtp::draft`.
   - [ ] Подключение стейджинга к реальному CUDA-пути attention.

3. **Однопроходная проверка**
   - Attention получает trie-маску по диапазонам поддеревьев.
   - Строки дерева верифицируются одним вызовом, а не K последовательными forward.
   - Для DeltaNet — теневые снимки по путям, откат к принятому префиксу.
   - Для MoE/attention — переиспользование существующих CUDA-графов и paged-KV.

4. **Приём и откат**
   - Выбрать самый длинный принятый путь.
   - Откатить KV и DeltaNet-state к границе принятого префикса.
   - Обновить `MtpUsage` с `drafted`, `accepted`, `used`, `fallback_category`.

5. **Benchmark и интеграция**
   - A/B на Qwen3.5-9B против baseline.
   - Сравнить с текущим линейным MTP (`MTP_WIDTH=1..3`).
   - Проверить VRAM и TTFT.
   - Если выигрыш устойчив — включить по умолчанию.

## Технические зацепки

- `engine/qwen35-batch/src/real/mtp.rs` — сейчас линейный драфт; сюда добавляется построение дерева.
- `engine/qwen35-batch/src/real/adapter.rs` — `speculative_verify` и `speculative_accept`; сюда встраивается однопроходная проверка дерева.
- `engine/qwen35-batch/src/real/paged_attn.rs` — единственный вызов attention-ядра; сюда нужно добавить trie-маску.
- `engine/qwen35-batch/src/real/paged_kv_cuda.rs` — стейджинг длин для varlen; нужен отдельный буфер trie-узлов.
- `engine/candle-kernels/` — если текущий flash-attention не принимает произвольную маску, сюда добавляется специализированное ядро tree-attention.

## Критерии успеха

- Decode t/s выше baseline на Qwen3.5-9B.
- Нет деградации VRAM и TTFT.
- `used=true` и `accepted/drafted ≥ 0.5`.
- Отсутствие выпадения токенов и порчи выдачи.
