# Qwen3.5-9B MTP и tree speculative decoding: исправленный A/B с парной головой

Дата: 2026-10-11.
Стенд: `yttri-win` (RTX 3060 12 GB, Windows).
Модель и MTP-голова: **один и тот же файл** `D:\Models\unsloth\Qwen3.5-9B-MTP-GGUF\Qwen3.5-9B-Q4_K_M.gguf`.
Сервер: `qwen36-server` из worktree `qwen36-tree-server`, собран с `cargo build --release --features cuda`, `CUDA_COMPUTE_CAP=86`.
Движок: `yttri-tree-forge`, коммиты `073f314a` (трассировка), `dd7f7c0e` (DFS/RoPE/обход ветки), `6ab9a2d2` (перестройка принятого пути). Локальные эквиваленты: `a272a9c5`, `1d962643`, `f9a03235`.

## Исправление pairing

В первой версии этого отчёта (2026-10-10) использовалась модель `Qwen3.5-9B-Q4_K_M` из `lmstudio-community/Qwen3.5-9B-GGUF` и отдельная MTP-голова `mtp-Ornith-1.5-9B-head-Q8_0.gguf`. Это **непарная конфигурация**: голова от другой модели и в другом кванте. Такие результаты невалидны для сравнения model+head и оставлены только в истории git.

Корректный источник — `unsloth/Qwen3.5-9B-MTP-GGUF/Qwen3.5-9B-Q4_K_M.gguf`. Проверка `gguf-dump` подтверждает:

- `general.name = Qwen3.5-9B`;
- `general.base_model.0.repo_url = https://huggingface.co/Qwen/Qwen3.5-9B`;
- `qwen35.block_count = 33`, `qwen35.nextn_predict_layers = 1`;
- `general.file_type = 15` (Q4_K_M);
- присутствуют тензоры `blk.32.attn_*`, `blk.32.ffn_*`, `blk.32.nextn.eh_proj.weight`, `blk.32.nextn.enorm.weight`, `blk.32.nextn.hnorm.weight`, `blk.32.nextn.shared_head_norm.weight`.

То есть MTP-слой встроен в тот же Q4_K_M GGUF, что и модель. В `.env` `MODEL` и `MTP_PATH` указывают на этот файл, `MTP=1` при тестах MTP и `MTP=0` для baseline.

## Условия теста

- `n_ctx=128000`, `n_slots=1`, `n_batch=2048`, `kv_cache_type=q8_f16`, paged int8 KV, `flash_attn=1`, CUDA graphs включены.
- `PREFIX_CACHE_MIB=0`, `TRACE=0`, `MTP_TIMING=0`.
- Decode: промпт 4 слова (30 prompt-токенов), `max_tokens=128`, `temperature=0`, 3 прогона после загрузки модели.
- Prefill: промпт 1500 слов (6412 prompt-токенов), `max_tokens=1`; приведено полное время запроса как прокси prefill (включает один decode-шаг).
- `MTP_VOCAB_TOP=0` — полный словарь модели (248320 токенов).
- `MTP_VOCAB_TOP=32768` — шортлист первых 32768 id, оптимизация draft/verify.
- `VERIFY_ONEPASS=1` — однопроходная FA2-ветка с реальной trie-маской.

## Decode

| Режим | Decode, t/s (min/median/max) | Drafted / Accepted | Acceptance |
| --- | ---: | ---: | ---: |
| `MTP=0`, baseline | 53.73 / 53.74 / 53.75 | — | — |
| `MTP=1`, `MTP_TREE=0`, W1, полный словарь | 38.00 / 39.77 / 40.00 | 126 / 108-110 | 85.7-87.3% |
| `MTP=1`, `MTP_TREE=0`, W1, `MTP_VOCAB_TOP=32768` | 44.12 / 44.23 / 44.28 | 126 / 92-96 | 73.0-76.2% |
| `MTP=1`, `MTP_TREE=1`, budget=9, depth=4, branching=2, полный словарь | 4.53 / 4.59 / 4.60 | 592-704 / 16-67 | 6.8-11.3% |
| `MTP=1`, `MTP_TREE=1`, budget=9, depth=4, branching=2, `MTP_VOCAB_TOP=32768` | 7.54 / 7.61 / 7.65 | 656-728 / 41-57 | 5.6-8.7% |

`drafted` в tree-режиме — суммарное число строк дерева, поданных на verify за все раунды; `accepted` — число принятых draft-токенов. Acceptance считается как `accepted/drafted`.

## Prefill

| Режим | Время на 6412 prompt-токенов | Оценка prefill |
| --- | ---: | ---: |
| `MTP=0`, baseline | 4.526 s | ~1417 t/s |
| `MTP=1`, `MTP_TREE=0`, W1, полный словарь | 4.644 s | ~1381 t/s |
| `MTP=1`, `MTP_TREE=1`, полный словарь | 4.416 s | ~1452 t/s |

Для linear/tree с `MTP_VOCAB_TOP=32768` отдельный prefill-замер не снимался: шортлист меняет draft-голову, а prefill-путь остаётся тем же.

## VRAM

Замеры `nvidia-smi` после загрузки модели, до активного запроса. До запуска сервера на GPU занято ~481 MiB (рабочий стол/ОС).

| Режим | Всего занято GPU |
| --- | ---: |
| `MTP=0`, baseline | 7698 MiB |
| `MTP=1`, `MTP_TREE=0`, W1 | 7858 MiB |
| `MTP=1`, `MTP_TREE=1`, K=8 | 8340 MiB |

## Вывод

Парная MTP-голова из того же Q4_K_M GGUF загружается и работает: linear W1 даёт acceptance 73-87% и decode 38-44 t/s. Но **baseline всё равно быстрее**: 53.7 t/s. Даже лучший линейный режим (`MTP_VOCAB_TOP=32768`) отстаёт от baseline примерно на 18%, потому что стоимость draft+verify больше, чем экономит один-два принятых токена.

Tree-режим с парной головой остаётся неконкурентоспособным: 4.5-7.7 t/s и acceptance 5.6-11.3%. Причины те же, что и в первой версии отчёта:

- `draft_tree` считает MTP-голову последовательно через общий KV/state; узлы соседних веток загрязняют state друг друга. Linear W1 с той же головой даёт 73-87%, tree — 6-11%.
- Целевой DeltaNet обрабатывает строки дерева последовательно через общий recurrent state. Перестройка принятого пути лечит состояние после раунда, но не делает корректными logits всех веток внутри раунда.
- Раунд tree K=8 стоит ~135-147 мс и принимает 1-2 токена. При baseline ~18.6 мс/токен это кратный проигрыш.

## Что нужно для продолжения

1. **Per-node MTP state/KV для драфта.** Каждый узел дерева должен считать MTP-голову от состояния своего родителя, а не от предыдущего линейного узла.
2. **Per-node DeltaNet state для target-verify** либо отказ от ветвящейся проверки recurrent-слоёв в пользу другой схемы (отдельная draft-модель, self-speculation и т.п.).
3. **Батчевый MTP-драфт на уровень дерева** вместо последовательных `forward_node` с D2H-синхронизациями.
4. **CUDA-тест trie-маски против построчного эталона**, чтобы отделить корректность маски от влияния state.

## Ограничения

- Замеры сделаны на одной RTX 3060 12 GB и одной паре model+head: `unsloth/Qwen3.5-9B-MTP-GGUF` Q4_K_M.
- `MTP_VOCAB_TOP=32768` — эвристический шортлист по первым id; он быстрее, но acceptance ниже, чем при полном словаре.
- Prefill-оценка включает один decode-шаг и не отделяет MTP catch-up от prefill.
- `PREFIX_CACHE_MIB=0`: с включённым prefix cache повторные одинаковые запросы деградировали, это отдельный эффект стенда.
