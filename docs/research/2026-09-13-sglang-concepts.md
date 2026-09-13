# SGLang как референс serving-механик для qwen36-server / yttri-forge

Дата: 2026-09-13
Статус: research, не обязательное решение.
Validation report: [2026-09-13-sglang-validation-report.md](2026-09-13-sglang-validation-report.md).
Scope: изучить концепции SGLang, которые могут дать преимущество над llama.cpp
на нашем runtime, и отделить переносимые механики от чужого стека.

## Главный вывод

SGLang не заменяет llama.cpp как numerical/tokenizer/GGUF-oracle: он не даёт
паритета по 2-bit GGUF-ядрам, chat template и tokenizer behavior. Но он
закрывает другой слой, где мы сейчас упираемся: serving scheduler, hybrid cache,
overlap CPU/GPU, graph capture и speculative decode на linear-attention моделях.

Практический вывод: llama.cpp оставляем эталоном для parity и низкоуровневых
ядер, SGLang используем как эталон serving-архитектуры. Runtime остаётся
собственным Candle/yttri-forge, без переноса Python/PyTorch стека.

Наибольший потенциальный выигрыш сейчас не в ещё одной оптимизации GEMM, а в
четырёх механиках:

1. **Hybrid radix cache** для full attention + GDN/DeltaNet state.
2. **Overlap scheduler + persistent chunked prefill** вместо «prefill-чанк или
   decode-шаг».
3. **Breakable/piecewise CUDA graph** для MTP и prefill.
4. **GDN state checkpointing** (включая int8) и ReplaySSM для speculative decode.

## Почему раньше смотрели на llama.cpp

Исходная рамка проекта была numeric-first: собственный Candle-runtime,
GGUF-кванты, DeltaNet, tokenizer parity, OpenAI/Anthropic API. В этой рамке
llama.cpp естественный oracle: он уже умеет Qwen3.x, GGUF и MTP, и на нём
можно проверить, совпадает ли распределение, tokenizer и скорость.

SGLang в ту рамку не попадал: это не GGUF-runtime и не oracle для Candle-ядер,
а serving-engine на PyTorch/FlashInfer. Его преимущества лежат выше уровня
ядер: планирование, переиспользование KV, графы и спекуляция.

Сейчас рамка изменилась. После упора в prefill-потолок, конфликта MTP с
CUDA graphs и появления hybrid prefix cache нам нужны именно serving-механики.
Поэтому SGLang стоит изучать не вместо llama.cpp, а **рядом с ним**:

- llama.cpp — численный эталон и baseline;
- SGLang — эталон того, как эти же ограничения решаются на уровне сервера.

## Что именно в SGLang отличается

### 1. RadixAttention и Radix Cache

В SGLang KV-кэш организован как radix tree: узел хранит span токенов и KV
этого span, path от root до leaf — префикс запроса. Общий префикс нескольких
запросов переиспользует одни узлы. При неполном совпадении узел split-ится.
Это даёт longest-prefix match без линейного сканирования всех записей.

Первоисточник: [SGLang paper, arXiv:2312.07104](https://arxiv.org/abs/2312.07104).
Исходник: `python/sglang/srt/mem_cache/radix_cache.py`, класс `RadixKey`
с `match()` и `child_key()`.

У нас в `src/prefix_cache.rs` сейчас плоские `buckets: HashMap<u64, Vec<u64>>`
по хешам блочных границ, `by_id` и один `VecDeque` LRU. Поиск идёт от старших
блоков к младшим и сверяет токены. Это корректно, но:

- нет tree-структуры и split-узлов;
- один snapshot покрывает весь prompt, а не span узла;
- eviction — один LRU по байтам, без разделения full-KV и recurrent state;
- нет session-level soft protection;
- нет branch-point checkpoint.

### 2. Mamba/GDN Radix Cache для hybrid-моделей

Для Qwen3.6 SGLang явно поддерживает `--mamba-radix-cache-strategy`:
`no_buffer` (v1) и `extra_buffer` (v2). Cookbook говорит: v1 экономит память,
v2 включает overlap scheduling и branching point caching, требует page size 64
и стоит дороже по mamba state.

В исходнике `mamba_radix_cache.py`:

- у узла есть отдельные `value` (full KV) и `mamba_value` (recurrent state);
- отдельно живут full LRU и mamba LRU: `self.full_lru_list` и
  `self.mamba_lru_list`;
- `mamba_value` нельзя split-ить: при split узла новый промежуточный узел
  становится mamba tombstone, а ребёнок сохраняет state;
- на match считается `mamba_branching_seqlen` — последняя выровненная позиция
  без mamba value, то есть точка, откуда можно replay-ить suffix вместо полного
  prompt;
- `extra_buffer` использует ping-pong track buffer: активный state и
  checkpoint могут сосуществовать, что нужно для overlap и speculative
  rollback;
- есть `MambaCheckpointPool` с int8-хранением GDN/KDA/Mamba2 state:
  `~2x` cached states при фиксированной памяти, conv window остаётся в native
  dtype; state квантуется один раз при сохранении и один раз де-квантизуется
  на hit, внутрь recurrence quantized state не возвращается.

Исходники:
`python/sglang/srt/mem_cache/mamba_radix_cache.py`,
`python/sglang/srt/mem_cache/mamba_checkpoint_pool.py`,
`python/sglang/srt/mem_cache/unified_radix_cache.py`.

Это прямое попадание в нашу задачу: наш prefix cache уже хранит
`StateSnapshot`, но хранит его целиком на prompt, в host memory, без деления
на full-KV и DeltaNet state и без branch-point логики.

### 3. Unified components и cascade eviction

В `UnifiedRadixCache` есть компоненты `FULL`, `SWA`, `MAMBA` и общий tree
core. Cache actions описывают отдельные операции для компонентов:
`FreeDeviceKV`, `FreeDeviceKVFullOnly`, `FreeComponentDeviceSlot`.
Session-aware cache добавляет soft references: KV, привязанный к active
session, evict-ится позже ordinary KV, но остаётся evictable при нехватке
памяти.

Это концепция, которой у нас нет: у нас один snapshot и один LRU. Если
full-KV и DeltaNet state имеют разную стоимость, разный lifetime и разную
возможность split/replay, то их нельзя evict-ить одним списком.

### 4. Chunked prefill и mixed chunk

В SGLang scheduler есть persistent `chunked_req`: незаконченный prefill
переносится в следующий iteration, а не начинается заново. При
`enable_mixed_chunk` decode-запросы могут идти в том же batch, что и
chunked prefill. `PrefillAdder` учитывает `num_mixed_decode_tokens` в бюджете
prefill.

У нас `BatchScheduler::step()` делает **или** один prefill chunk, **или**
один batched decode. `PREFILL_CHUNK=512`; следующий prefill-чанк не может
идти одновременно с decode внутри одного scheduler step. Для агентской
нагрузки с длинными prompt это означает лишние serialization points и
TTFT, зависящий от текущего decode-батча.

Это не значит, что нужно немедленно повторить mixed batch: для этого нужен
совмещённый forward path. Но persistent chunked request и overlap результата
предыдущего шага можно реализовать отдельно и измеримо.

### 5. Overlap scheduler

`event_loop_overlap()` в SGLang запускает текущий batch на forward stream и
обрабатывает результат предыдущего batch в очереди `result_queue`, пока GPU
считает текущий. Комментарий в исходнике: “overlaps the CPU processing and GPU
computation”. Для двух подряд идущих prefill batch overlap иногда отключается
ради TTFT первого batch.

У нас dispatch loop синхронно вызывает `sched.step()` и только после него
возвращается к ingest. При host launch overhead ~4 мс/launch и большом числе
шагов это прямой резерв. Первый переносимый шаг — не менять GPU forward, а
разделить schedule/forward/result-processing на три стадии и перекрыть
result-processing предыдущего batch с forward следующего.

### 6. Breakable CUDA Graph

SGLang `Breakable CUDA Graph` делит forward на несколько captured segments и
вставляет eager-вызовы в точках, которые нельзя захватить: dynamic control
flow, host-device sync, JIT, ops, меняющие behavior между итерациями. Вне
capture такой op ведёт себя normalmente. Включение:
`SGLANG_USE_BREAKABLE_CUDA_GRAPH=1`, decorator `@eager_on_graph`, `break_graph()`.

Это контрастирует с нашим PD-204: при MTP графовый decode отключается целиком,
а eager path использует второй экземпляр KV. Breakable graph предлагает не
выбирать «граф или eager», а сохранить graph segments вокруг MTP/draft/verify
и вынести eager только туда, где действительно нужна динамика.

### 7. Piecewise CUDA Graph

`Piecewise CUDA Graph` делит forward на pieces (примерно по слоям) на split
points вроде MoE dispatch. Для prefill/extend SGLang захватывает pieces для
набора token lengths, а runtime padding-ит вход до ближайшего captured size.
Это позволяет убрать kernel launch overhead при переменной длине prefill.

У нас prefill не захвачен в графы, а именно он упёрся в launch overhead:
в `docs/plans/2026-08-23-prefill-optimization.md` зафиксировано ~4.2 мс × 240
launch и вывод: паритет с llama.cpp недостижим без CUDA graphs для prefill.
Наш chunk size фиксирован 512, что делает piecewise capture даже проще
абстрактного SGLang-случая, если удастся изолировать attention с динамическим
KV length как eager piece.

### 8. Speculative decoding: adaptive, ReplaySSM, token maps

В SGLang есть отдельные механики:

- adaptive speculative decoding — ширина драфта следует за acceptance rate;
- ReplaySSM (`--enable-linear-replayssm-spec`) для linear/hybrid моделей;
- speculative token map / FR-Spec — уменьшает overhead `lm_head` на большом
  vocab;
- NGRAM и другие способы спекуляции без draft model;
- overlap scheduler v2 — overlap draft/verify/sample.

У нас уже есть adaptive width, MTP transaction с rollback/commit и
фазовый timing. Следующий переносимый концепт — ReplaySSM/checkpoint-based
rollback: после speculative verification восстанавливать/переигрывать
принятый GDN state от branch point, а не держать полный второй KV и не
отключать graphs.

### 9. Batch-invariant deterministic inference

SGLang объясняет non-determinism именно тем, что мы зафиксировали в BD-029:
разный batch size даёт разный порядок reduction в GPU-ядрах, из-за
floating-point non-associativity логиты расходятся. Решение — batch-invariant
operators (по мотивам Thinking Machines), совместимые с chunked prefill,
CUDA graph, radix cache и non-greedy sampling.

Для нас это горизонт 3: реализация batch-invariant reductions в Candle-ядрах
дорогая. Но концепт важен как альтернатива текущему решению «сохраняем
распределение, а не token-for-token»: если exact reproducibility снова станет
продуктовым требованием, это не обязательно делать через отказ от батчинга.

### 10. HiCache: L1 GPU / L2 host / L3 storage

HiCache формализует tiering: GPU memory как L1, host memory как L2,
distributed storage как L3. Для нас релевантны не distributed backends, а:

- L1/L2 split: часть hot KV/state держать в VRAM, холодное — в host;
- prefetch во время prefill;
- write-back policy: write-through / selective / write-back;
- page-first layout для host↔device transfer;
- overlap копирования слоя N+1 с вычислением слоя N;
- session-aware eviction.

Наш prefix cache уже L2-only: snapshot живёт в host и поднимается на device
на hit. Следующий шаг — не L3, а корректный L1/L2 lifecycle и transfer
overlap.

## Сопоставление с текущим состоянием

| Concept | SGLang | qwen36-server / yttri-forge сейчас | Что даёт перенос |
|---|---|---|---|
| Prefix index | radix tree, split nodes | flat hash buckets + token verify, `src/prefix_cache.rs` | longest-prefix без скан-бакетов, branch points |
| Recurrent state | `mamba_value` per node, separate LRU | весь `StateSnapshot` на prompt | точечный reuse и eviction GDN state |
| Branch cache | `mamba_branching_seqlen`, v2 `extra_buffer` | нет, только exact prompt prefix | suffix replay вместо полного prefill |
| State storage | bf16 + optional int8 checkpoint | host snapshot, без сжатия | ~2x cached prefixes при fixed host RAM |
| Session policy | soft session refs | нет | multi-turn agent cache не вытесняется one-off запросами |
| Prefill scheduling | persistent chunked req, mixed chunk | prefill chunk OR decode step | ниже TTFT, меньше serialization |
| CPU/GPU overlap | overlap result queue | синхронный `step()` | скрыть result-processing/host work |
| Prefill graph | piecewise capture | eager prefill, launch-bound | закрыть разрыв с llama.cpp |
| MTP graph | breakable graph | MTP отключает graphs целиком, PD-204 | MTP без eager KV-копии |
| Speculative rollback | ReplaySSM / checkpoint | transaction rollback/commit | дешевле verify и rollback GDN |
| Determinism | batch-invariant ops | BD-029: distribution-level equivalence | exact reproducibility при необходимости |
| KV/state tiering | L1/L2/L3, prefetch | host snapshot, to_device on hit | горячий state в VRAM, overlap prefetch |
| Eviction policy | LRU/LFU/SLRU/priority + session refs | LRU по bytes | hit-rate под agent-нагрузкой |
| Metrics | cache hit, eviction, queue, overlap | в основном логи и ad-hoc timing | gate для каждой оптимизации |

## Приоритеты

### P0. Hybrid Radix Cache v2

Что переносить:

- radix tree в Rust с page granularity 64;
- узел с `full_value` и `mamba_value`;
- split full-KV узла без split mamba state: новый узел — mamba tombstone;
- отдельные LRU для full-KV и GDN state;
- cascade eviction: удаление full-KV узла каскадом удаляет зависимые SWA/Mamba
  данные, а не оставляет state без соответствующего full-KV;
- branch-point cache на выровненных позициях;
- session soft refs;
- позже — int8 checkpoint store.

Что не переносить: Python allocator, torch tensors, PagedTokenToKVPool.

Эксперимент: на существующем формате `StateSnapshot` заменить плоский cache на
tree metadata и сохранить тот же протокол `put/find`. Метрики: prefix hit
length, TTFT p50/p95, prefill tokens saved, host RAM, eviction count. Gate:
распределение выхода не меняется за пределами согласованного parity-порога,
WDDM paging не появляется, 4-slot stability сохраняется.

### P1. Persistent chunked prefill + overlap result processing

Что переносить:

- `chunked_req`, который живёт между шагами;
- очередь результата предыдущего batch;
- processing результата предыдущего шага во время forward текущего;
- затем, отдельно, mixed prefill+decode, только после проверки, что adapter
  умеет совмещённый forward.

Эксперимент: агентский trace из 4 клиентов, prompt 32K-128K, medians по TTFT
и tok/s. Gate: TTFT улучшается без роста p95 decode и без WDDM paging.

### P2. Piecewise prefill graph и breakable MTP graph

Что переносить:

- prefill capture для chunk sizes 128/256/512/1024 с padding до captured size;
- attention с динамическим KV length оставить eager piece;
- decode graph с break only вокруг MTP draft/verify;
- graph replay counters в лог.

Эксперимент: prefill 1K/10K/24K, MTP на 32K/128K. Gate: prefill tok/s растёт,
MTP не хуже baseline with graphs, VRAM не превышает agreed budget.

### P3. GDN checkpoint compression и ReplaySSM

Что переносить:

- int8 temporal state, conv window native dtype;
- per-(head, k-channel) symmetric quantization;
- dequantize в активный slot на cache hit;
- replay accepted suffix from checkpoint после speculative verify.

Эксперимент: perplexity/generation parity на 2K-8K токенов, cache capacity,
MTP acceptance/cost. Gate: quality drift ≤ agreed threshold, rollback cheaper
than full state copy.

### P4. Batch-invariant reductions

Не начинать без явного требования exact token-for-token across batch shapes.
Это большой kernel-level проект, но концепт полезен как карта: расходятся не
sampling semantics, а reduction order в ядрах.

## Что не копировать

- Полный SGLang runtime, Python scheduler, PyTorch/FlashInfer/TRT-LLM.
- Tensor/expert/data parallel, DP attention, PD disaggregation: не имеют смысла
  на одной RTX 3060 12 GB; T-002 MoE offload решает другую задачу.
- FP8/NVFP4 model weights: несовместимы с нашим GGUF 2-bit runtime.
- CUDA graph через `torch.compile`: берём идею capture split, не зависимость.
- L3 HiCache/Mooncake/3FS/NIXL: нам нужен только L1/L2 lifecycle.
- Смену numerical oracle: llama.cpp остаётся parity baseline.

## Предлагаемый порядок изучения и проверки

1. Замерить текущий agent workload: prefill/decode split, TTFT, prefix hit
   length, host cache bytes, eviction count, MTP overhead.
2. Реализовать radix metadata поверх существующего `StateSnapshot`, без смены
   формата state.
3. Добавить branch-point checkpoint и session refs; проверить на multi-turn
   trace.
4. Разнести schedule/forward/result-processing и замерить overlap.
5. Отдельно исследовать prefill graph capture и breakable MTP graph.
6. Только после этого решать про int8 checkpoint и mixed prefill+decode.

## Источники

SGLang commit для source-проверки:
`d6fabb74b45d4fb92796cfb6740810b4811b018e` (2026-09-12).

Официальные docs:

- [SGLang Qwen3.6 cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.6.md)
- [SGLang Qwen3.8-27B cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B.md)
- [Session-Aware Radix Cache](https://docs.sglang.io/docs/advanced_features/session_radix_cache.md)
- [Radix Cache Eviction Policies](https://docs.sglang.io/docs/advanced_features/radix_eviction_policy.md)
- [Speculative Decoding](https://docs.sglang.io/docs/advanced_features/speculative_decoding.md)
- [Adaptive Speculative Decoding](https://docs.sglang.io/docs/advanced_features/adaptive_speculative_decoding.md)
- [Breakable CUDA Graph](https://docs.sglang.io/docs/advanced_features/breakable_cuda_graph.md)
- [Piecewise CUDA Graph](https://docs.sglang.io/docs/advanced_features/piecewise_cuda_graph.md)
- [Deterministic Inference](https://docs.sglang.io/docs/advanced_features/deterministic_inference.md)
- [Quantized KV Cache](https://docs.sglang.io/docs/advanced_features/quantized_kv_cache.md)
- [HiCache System Design](https://docs.sglang.io/docs/advanced_features/hicache_design.md)
- [SGLang paper, arXiv:2312.07104](https://arxiv.org/abs/2312.07104)

Локальные опорные точки:

- `src/prefix_cache.rs` — текущий flat hash cache и host snapshots.
- `docs/batch-integration.md` — текущая интеграция BatchScheduler.
- `docs/specs/2026-08-27-mtp-cuda-graphs.md` — PD-204, eager KV и MTP+graphs.
- `docs/specs/2026-08-23-prefill-optimization.md` — prefill launch bottleneck.
- `docs/plans/2026-08-23-prefill-optimization.md` — замер launch overhead.
- `wiki/topics/prefix-cache.md` — текущее описание prefix cache, stale on
  2026-09-01; source остаётся истиной.
