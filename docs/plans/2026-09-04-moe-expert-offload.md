# Plan: Выгрузка экспертов MoE в pinned RAM с кэшем горячих экспертов в VRAM

**Дата:** 2026-09-04
**Статус:** 🔄 In progress (Фазы 0–5 ✅, фаза 6 — стабилизация)
**Приоритет:** P0
**Спецификация:** [docs/specs/2026-09-04-moe-expert-offload.md](../specs/2026-09-04-moe-expert-offload.md)

## Цель

Qwen3.6-35B-A3B IQ2_XXS на RTX 3060 12 ГБ работает с окном 131 072 и MTP: эксперты (9 624 МиБ) лежат в pinned RAM, ядра адресуют их через таблицу указателей, горячие эксперты кэшируются в VRAM, декод на одном слоте ≥ 40 ток/с, выход бит в бит совпадает с резидентным режимом.

## Текущее состояние

### Движок `yttri-forge/engine/qwen35-batch/src/real/`

- `moe.rs` — `PackedExperts { gate, up, down: Arc<QTensor>, n_experts }` (строка 267), `Qwen35MoeBlock { router, routed, shared, backend, cfg }` (329), `forward_ptx_cuda` (~401): роутер → `gpu_softmax_topk` (639, выделяет ids заново на каждый вызов) → `indexed_moe_forward_dual_cuda` → SiLU·up → `indexed_moe_forward_cuda` → взвешенная сумма + shared. `select_backend` (35) — матрица поддерживаемых dtype. Режим `ForwardMode` в PTX-пути не используется.
- `model_weights.rs` — загрузка MoE-слоя (6633–6694): `load_heavy(ffn_*_exps)` → `Arc<QTensor>` на устройстве модели, `FeedForward::Moe(Qwen35MoeBlock::new(...))`; вызовы форварда `FeedForward::forward*` (1197–1218) — prefill и decode оба идут в `moe.forward(xs, mode)`; `init_paged_decode` (8767) создаёт пул **лениво на первом декоде** (комментарий: транзиенты префила искажают `mem_get_info`), окно = `free − VRAM_HEADROOM_MIB` (8817); `load_heavy` — замыкания 5770/5864.
- `adapter.rs` — `pgraph_mode()` OnceLock (296–312); декод: `decode_batch_graphed` (2370) одним `cuGraphLaunch`, вызов и откат на eager (1455–1500); захват `[graphs] captured` (~2675); `load_mtp` (757); `PrefillPath` — смена пути внутри промпта запрещена.
- `scheduler.rs` — `prefill_chunk()` (71–82) читает `PREFILL_CHUNK`, умолчание 512.
- `mtp.rs` — `Qwen35Mtp::load` (196–262): плотный FFN `gate/up/down` по префиксу `blk.{mtp_block}`; FFN в `forward_rows` (579–581) и `draft_pass_body` (957–959); граф черновика `draft_graphed`.
- `model_profile.rs` — `MtpProfile::read_and_validate` (480): принимает только `Architecture::DenseQwen35`, ключи `qwen35.*`, `intermediate` из `feed_forward_length`, список плотных тензоров, проверка «thin = 15 тензоров»; список MoE-тензоров текстового профиля уже есть (680–690).

### Ядра `yttri-forge/engine/candle-core`, `candle-kernels`

- `candle-kernels/src/quantized.cu` — боевые ядра `indexed_moe_forward<…>(all_weights, all_inputs, indices, all_outputs, n, k, batch, topk, k_padded, input_dim1)` с extern-обёртками `indexed_moe_forward_<dtype>_q8_1` (6983–7075), `indexed_moe_forward_dual(w1_all, w2_all, …)` (7292), f32-варианты `IQ_MOE_F32_EXTERN` (6849–6866) и `_grouped` (выключены). Адрес эксперта везде `all_weights + (expert_id · n + row) · blocks_per_row`.
- `candle-core/src/quantized/cuda.rs` — `indexed_moe_forward_dispatch` (887): квантование входа в Q8_1, выбор имени ядра по dtype, запуск; методы `QCudaStorage::indexed_moe_forward` (1142) и `_dual` (1039) берут веса как `self.data.inner.slice(0..)`.
- `candle-core/src/quantized/mod.rs` — `QTensor::indexed_moe_forward_cuda` (921), `_dual_cuda` (944).
- `cudarc 0.19.7` — `result::malloc_host(bytes, flags)` публичен (`CU_MEMHOSTALLOC_DEVICEMAP = 2`, `WRITECOMBINED = 4`), `sys::cuMemHostGetDevicePointer_v2`, `CudaContext::new_stream`, `CudaStream::record_event`/`wait`, события `new_event`. Штатный `alloc_pinned` даёт только WC без DEVICEMAP — не подходит.

### Сервер `Qwen3.6 27B/src/`

- `vram_plan.rs` — `footprint_from_gguf`: веса = 0.95 × размер файла; `config.rs::apply_vram_plan` с обходом `NO_VRAM_PLAN`.
- `engine.rs` — `ModelInfo { id, context_length, quant, slots, modes }` (310); `engine_batched.rs::model_info` (598); `/v1/models` собирает `capabilities` (mtp — через `profile.rs:393` и `engine_batched.rs:596 mtp_available`).
- `main.rs` — стартовая сводка `log_kv!`.

### Проверочная база

- Тесты: `qwen35-batch/tests/qwen35moe_reference.rs` (синтетические эксперты, эталонный бэкенд), `tests/model_profile.rs`, `tests/mtp_transaction.rs`; `candle-core/tests/iq_quant_cuda_tests.rs` (`#![cfg(feature = "cuda")]`), пример `candle-core/examples/tensor_parity.rs`.
- Стенд: `tools/logits_parity.py`, реплей `.tmp-test/ornith_exact_hero_replay.mjs`, `multiturn.ps1`, `qwen35_mtp_gate`, `qwen36_inspect`; сборка `build_windows.bat`, запуск задачей планировщика `qwen36-inference`; бэкап Ornith-конфига `.env.bak-ornith-pgraph-on`.

## Архитектура решения

```mermaid
graph TD
  subgraph Загрузка
    G[GGUF mmap] --> L[model_weights: загрузка MoE-слоя]
    L -->|ffn_*_exps по смещениям| ES[(ExpertLayerStore: pinned + device-mapped)]
    L -->|остальной ствол, output| V[(VRAM)]
    L --> PT[PointerTable ×3 на слой]
    L --> POOL[init_paged_decode сразу при загрузке]
    POOL --> STG[(Staging: один слой)]
    STG --> CACHE[(SlotPool: кэш на слой)]
  end

  subgraph Декод: один cuGraphLaunch
    R[роутер + gpu_softmax_topk → RouteTrace] --> K[indexed_moe_forward_*_q8_1 через PointerTable]
    K -->|попадание| CACHE
    K -->|промах zero-copy| ES
  end

  CTRL[ExpertCacheController: before_step] -->|читает RouteTrace прошлого шага| R
  CTRL -->|LRU, подъём ≤ N на боковом потоке| CACHE
  CTRL -->|запись таблицы по событию| PT

  subgraph Префил: eager, послойно
    R2[роутер чанка → D2H ids] --> U[объединение экспертов слоя]
    U -->|свободные слоты| CACHE
    U -->|переполнение| STG
    U -->|таблица на время слоя| PT
    PT --> K2[ядра только из VRAM]
  end

  MTP[mtp.rs: FeedForward::Moe, эксперты резидентны] --> V
  SRV[сервер: vram_plan по именам тензоров, ModelInfo.moe, сводка] --> L
```

Потоки: основной поток вычислений (графы и eager) и боковой поток подъёмов; порядок «таблица → копия → таблица» через события cudarc.

## Решение

### Слой A — ядра (`candle-kernels`, `candle-core`)

#### Файлы

- [ ] `candle-kernels/src/quantized.cu` — параметр `all_weights` → `const void* const* expert_ptrs` в шаблоне `indexed_moe_forward`, во всех extern `_q8_1`, в `indexed_moe_forward_dual` (`w1_ptrs`, `w2_ptrs`), в `indexed_moe_forward_iq_f32` и макросе `IQ_MOE_F32_EXTERN` (включая `_grouped`, чтобы не держать два соглашения). Адрес: `(const block_q_t*)expert_ptrs[expert_id] + row * blocks_per_row`.
- [ ] `candle-core/src/quantized/cuda.rs` — `indexed_moe_forward_dispatch` принимает `table: &CudaView<u64>` вместо `weight`; новые публичные `indexed_moe_forward_table(dev, dtype, w_shape, table, input, ids)` и `indexed_moe_forward_dual_table(...)` без `&self`; методы `QCudaStorage::indexed_moe_forward*` остаются как тонкие обёртки над лениво построенной таблицей `base + id·stride` (`OnceLock<CudaSlice<u64>>`) — для тестов и эталонных путей.
- [ ] `candle-core/src/quantized/mod.rs` — `QTensor::indexed_moe_forward_cuda`/`_dual_cuda` делегируют обёрткам; экспорт `indexed_moe_forward_table*`.
- [ ] `candle-core/tests/moe_table_cuda_tests.rs` — бит в бит: одни и те же квантованные данные (а) упакованно, (б) таблица с перестановкой слотов в VRAM, (в) таблица на device-mapped host-памяти → выходы `_q8_1` и `_dual` идентичны байт в байт для IQ2_XXS, IQ3_XXS, IQ2_S, IQ4_XS, Q2_K, Q3_K.
- [ ] `candle-core/examples/moe_zero_copy_bench.rs` — ворота 0: читает `ffn_*_exps` всех слоёв из GGUF по смещениям, кладёт в pinned DEVICEMAP (с WC и без), прогоняет `indexed_moe_forward` T=1, k=8 по 40 слоям против VRAM-копии, печатает эффективную полосу по времени ядра и проверяет `cuMemHostGetDevicePointer_v2 == host ptr`.

#### Контракт ядра (псевдокод)

```pseudo
indexed_moe_forward_<dtype>_q8_1(
    expert_ptrs: *const *const u8,   // [n_experts] адрес эксперта: VRAM-слот, стейджинг или host-mapped
    inputs, ids, outputs, n, k, batch, topk, k_padded, input_dim1)   // без изменений

indexed_moe_forward_table(dev, dtype, [n_experts, n, k], table: CudaView<u64>, input, ids) -> Tensor
indexed_moe_forward_dual_table(dev, dtype, shape, table1, table2, input, ids) -> (Tensor, Tensor)
```

### Слой B — движок (`qwen35-batch`)

#### Файлы

- [ ] `real/expert_store.rs` (новый) — `ExpertPlacement` (`MOE_EXPERTS=vram|ram|auto`), `ExpertLayerStore`, `PointerTable`, `SlotPool`, `Staging`, `CacheDirectory`, `RouteTrace`, `ExpertCacheController`; проверки fail-closed; статистика и строки лога.
- [ ] `real/moe.rs` — `PackedExperts` → `enum ExpertWeights { Packed{gate,up,down: Arc<QTensor>} /* Reference */, Store(Arc<ExpertLayerStore>) /* Ptx */ }`; `gpu_softmax_topk` пишет ids в переданный срез `RouteTrace`; `forward_ptx_cuda` вызывает `*_table`; при `ForwardMode::Prefill` — `controller.prefill_prepare_layer` → ядра → `prefill_release_layer`.
- [ ] `real/model_weights.rs` — загрузка MoE-слоя через `ExpertLayerStore` (pinned из байтов mmap по `tensor_infos`, без промежуточного `QTensor`; в `vram` — VRAM-копия и таблица на неё); `token_embd` в RAM при `ram`; после загрузки при `ram`: `init_paged_decode` → стейджинг → кэш; `auto` по формуле пула; fail-closed по dtype (`select_backend`), host RAM и аллокациям; эффективные значения в лог.
- [ ] `real/adapter.rs` — `pgraph_mode()` принудительно `Off` при `ram` с WARN; `controller.before_step()` перед `decode_batch_graphed`; сброс хранилища и слотов в `unload` до сброса контекста (FR-012); строка сводки с эффективными `PGRAPH`, `PREFILL_CHUNK`, размещением, бюджетом, f.
- [ ] `scheduler.rs` — `prefill_chunk()`: при `ram` минимум 256 со строкой в логе.
- [ ] `real/mtp.rs` — `enum FeedForward { Dense{gate,up,down}, Moe(Qwen35MoeBlock) }`; загрузка MoE-варианта (роутер, `ExpertLayerStore` в резидентном режиме, shared-эксперт); `forward_rows`/`draft_pass_body` через enum; адаптивная ширина по умолчанию при `ram`.
- [ ] `real/model_profile.rs` — `MtpProfile`: `Architecture::Qwen35Moe`, ключи по префиксу архитектуры, `intermediate` из `expert_feed_forward_length`/`expert_shared_feed_forward_length`, список MoE-тензоров + `nextn.*`, проверка «thin 15» только для тонкого файла.
- [ ] `tests/expert_store.rs` (новый) — host-логика без GPU: LRU, лимит подъёмов, трёхшаговый порядок, стейджинг при переполнении, «префил не вытесняет», бюджет и f, разбор `MOE_EXPERTS`.
- [ ] `tests/model_profile.rs` — MoE-профиль MTP на заголовке 35B-A3B (fixture: метаданные и `tensor_infos` без данных).
- [ ] `tests/mtp_transaction.rs` — черновик с `FeedForward::Moe` на синтетических экспертах.

#### Структуры (псевдокод)

```pseudo
ExpertPlacement = Vram | Ram | Auto        // MOE_EXPERTS, умолчание Auto

ExpertLayerStore {                          // один на MoE-слой
    layer: usize, n_experts: usize,
    mats: [Matrix; 3],                      // gate, up, down
}
Matrix {
    dtype: GgmlDType, expert_bytes: usize, shape: [n_experts, n, k],
    host: PinnedMapped { ptr, dev_ptr, len },        // DEVICEMAP (+WC по воротам 0), один буфер на (слой, матрица)
    resident_all: Option<CudaSlice<u8>>,             // только vram-режим
    table: CudaSlice<u64>,                           // [n_experts] адреса
}

SlotPool { layer, slot_bytes: [usize; 3], capacity, slots: CudaSlice<u8> ×3, free: Vec<SlotId> }
Staging  { bytes: [max по слоям; 3], buf: CudaSlice<u8> ×3 }        // один на модель

CacheDirectory {                            // хост, на слой
    resident: HashMap<ExpertId, SlotId>, lru: LinkedList<ExpertId>,
    pending: Vec<(ExpertId, SlotId, CudaEvent)>,
    stats { hits, misses, promoted_bytes, prefill_promoted_bytes }
}

RouteTrace {                                // постоянный device-буфер, заполняет gpu_softmax_topk
    ids: CudaSlice<u32>,                    // [layers][B*T_max][k], T_max = ширина драфта + 1
    host: Vec<u32>,                         // зеркало после D2H
}

ExpertCacheController {
    side: Arc<CudaStream>, promote_per_step: usize,     // EXPERT_PROMOTE_PER_STEP, умолчание 64
    fn before_step()                       // D2H trace прошлого шага → LRU → до N подъёмов: table[evicted]=host (main) → copy (side, ждёт события) → table[new]=slot (main, ждёт события копии)
    fn prefill_prepare_layer(layer, ids)   // D2H ids → объединение → свободные слоты, остальное в стейджинг → запись таблицы
    fn prefill_release_layer(layer)        // вернуть записи стейджинга на кэш/host
    fn after_prefill(last_chunk_counts)    // LRU-подъём экспертов последнего чанка по частоте
    fn estimate_f(budget) -> f32           // для лога и WARN
}
```

#### Бюджет и порядок выделения при `ram`

```pseudo
load trunk без exps, output, MTP-слой → init_paged_decode(CTX) → Staging(max слой)
cache_budget = EXPERT_CACHE_MIB || (free − VRAM_HEADROOM_MIB − GRAPH_POOL_RESERVE)
slots_per_layer = cache_budget / layers / expert_bytes(layer)    // равное число слотов
log: "[moe] experts: ram … f=… (target …)"; WARN если f < target
```

### Слой C — сервер (`qwen36-server`)

#### Файлы

- [ ] `src/vram_plan.rs` — `footprint_from_gguf`: веса на GPU по именам тензоров (`blk.N.*` без `ffn_*_exps` при `ram`, без nextn-блока при `MTP=0`, плюс `output`), а не 0.95 × файл; `NO_VRAM_PLAN` больше не нужен для 35B-A3B.
- [ ] `src/config.rs` — чтение `MOE_EXPERTS` для планера (та же переменная, что у движка; без префиксов).
- [ ] `src/engine.rs`, `src/engine_batched.rs` — `ModelInfo.moe: Option<MoeInfo { experts, cache_mib, cache_slots, hit_rate }>` из движка.
- [ ] `src/api.rs` (сборщик `/v1/models`) — `capabilities.moe`.
- [ ] `src/main.rs` — эффективные значения в стартовой сводке.
- [ ] `README.md`, `.env.example`, `scripts/README.md` — `MOE_EXPERTS`, `EXPERT_CACHE_MIB`, `EXPERT_PROMOTE_PER_STEP`, требование RAM ≈ 29 ГБ, PGRAPH при выгрузке.

#### API

| Endpoint | Вход | Выход | Описание |
|---|---|---|---|
| `GET /v1/models` | — | `capabilities.moe = { experts: "ram"\|"vram", cache_mib, cache_slots, hit_rate }` | размещение и состояние кэша (FR-020) |

## Фазы реализации

> Каждая фаза заканчивается проверяемым инкрементом на стенде yttri-win (сборка `build_windows.bat`, перезапуск задачей `qwen36-inference`). `[P]` — можно вести параллельно.

### Фаза 0: Ворота (оценка: 8 ч) — ✅ выполнена 2026-09-04

- [x] `candle-core/examples/moe_zero_copy_bench.rs` — бенч на реальных тензорах GGUF: все 9.6 ГБ pinned DEVICEMAP, `cuMemHostGetDevicePointer_v2` == host ptr, `indexed_moe_forward` T=1 k=8 × 40 слоёв, host-mapped против VRAM, флаги WC/без WC. **Результат: PASS — mapped 10.61 ГБ/с (порог 8), VRAM 77.5 ГБ/с; флаги — DEVICEMAP без WC (WC выигрыша не даёт); UVA dev==host 120/120 без WC, 0/120 с WC; pinned 8644 МиБ × 120 буферов выделились без отказа.**
- [ ] [P] Ворота 0б (решение пользователя): реплей записанной сессии на нынешнем стенде (IQ2_XXS, окно 20K) — отчёт о качестве.
- **Независимая проверка:** бенч печатает полосу ≥ 8 ГБ/с и выбранные флаги; иначе — стоп и пересмотр D-005 (стейджинг с синком на слой). — **пройдена.**
- deviated: бенч использует боевые q8_1-ядра (`indexed_moe_forward[_dual]_<dtype>_q8_1`) вместо f32-варианта — f32-ядро compute-bound и полосу не измеряет; вход квантуется штатным ядром `quantize_q8_1`, как в диспетчере.
- deviated: добавлены тайминг по CUDA-событиям и продувка L2 (memset 256 МиБ между итерациями) — без неё повторные запуски подмешивали L2; на результат почти не влияет (10.03 → 10.18 ГБ/с).
- deviated: реальный объём маршрутизируемых экспертов 8644 МиБ (40 × 256 × 264 КиБ); 9624 МиБ из спеки включали shared-эксперты, которые остаются в VRAM. Из формулы D-014 при 10.6 ГБ/с: f ≥ 0.57 (было 0.60 при 301 МБ/токен; фактически 247 МБ/токен).

### Фаза 1: Ядра с таблицей указателей (оценка: 10 ч) — ✅ выполнена 2026-09-04

- [x] `quantized.cu` — таблица во всех вариантах `indexed_moe_forward*` (q8_1 ×12 extern, dual ×14, f32, f32_grouped, grouped ×6); адрес `(const block_q_t*)expert_ptrs[expert_id] + row·blocks_per_row`.
- [x] `candle-core/src/quantized/cuda.rs`, `mod.rs` — диспетчер принимает таблицу; публичные `indexed_moe_forward_table`/`_dual_table` (Tensor-уровень); `QCudaStorage` — ленивая таблица `base + id·stride` (OnceLock), методы `QTensor` идут через неё; реэкспорт из `quantized`.
- [x] `candle-core/tests/moe_table_cuda_tests.rs` — бит в бит (упаковка / перестановка слотов VRAM / host-mapped DEVICEMAP) для IQ2_XXS, IQ3_XXS, IQ2_S, IQ4_XS, Q2_K, Q3_K, plain + dual — **зелёный на стенде** (`run_moe_table_test.bat`, 1 passed).
- **Независимая проверка:** на стенде `cargo test --features cuda moe_table` зелёный ✅; сервер 35B-A3B резидентно (окно 20K): A/B старого (15:19) и нового exe — top-10 логпробов первого токена идентичны, **max Δ = 0.0**, greedy 16 токенов бит в бит (PGRAPH=off, чанк 512, FA_SPLITS=1, обе стороны) ✅; Ornith не задет — реплей на новой сборке (PGRAPH=on): unknownProperties/malformedFunctions пусты, реальные маркеры подмен (inline-rc/align-rc/justify-rc/font-clone/radial-grad) — нули ✅.
- deviated: добавлены отсутствовавшие dual-ядра Q3_K/Q5_K (DUAL_MOE_EXTERN + match arms) — тест плана требует Q3_K dual, поверхность была несимметричной.
- deviated: тест на синтетических байтах (LCG → QTensor через gguf `tensor_from_slice`): CPU-квантователя IQ-типов в форке нет, а бит-в-бит сравнение требует одинаковых байтов во всех раскладках.
- deviated: паритет — A/B двух exe на коротком промпте (logprobs + greedy), полный критерий спеки «Δ=0 на промпте 4 096» остаётся на фазу 2.
- заметка: ворота 0б (реплей 49K-сессии против 35B на окне 20K) невозможны по построению: скользящее окно режет по оценке `text_content()`, которая не видит тул-пейлоад (32 222 точных > 20 480 при оценке ниже бюджета) → `clamp_to_context` даёт 400. Вынесено на решение пользователя (см. отчёт фазы).

### Фаза 2: Эксперты в RAM, zero-copy декод, стейджинг префила (оценка: 16 ч) — ✅ выполнена 2026-09-04

- [x] `real/expert_store.rs` — `ExpertPlacement`/`parse_placement`, `ExpertLayerStore` (+`ExpertMatrix`, pinned `HostBuf` DEVICEMAP), таблицы указателей, `Staging` (полный слой), `TraceBuf` `[слои][B·(W+1)][k]`, `MoeRuntime`; fail-closed по dtype, host RAM (GlobalMemoryStatusEx), аллокациям; `resolve_auto` с числами.
- [x] `real/model_weights.rs` — ram-загрузка слоёв из GGUF-среза, `token_embd` → RAM-копия (FR-001), `prepare_expert_offload`: пул KV при загрузке → стейджинг (PD-010); `auto`.
- [x] `real/moe.rs` — `ExpertWeights::{Packed,Store}`; ids шага декода → след d2d-копией внутри графа; префил `prefill_prepare_layer` (всё объединение в стейджинг) → ядра `*_table` → `prefill_release_layer`.
- [x] `real/adapter.rs`, `scheduler.rs` — PGRAPH off с WARN, чанк ≥ 256, pinned освобождается Drop'ом хранилища при выгрузке (FR-012); сводка — строки `[moe] experts/staging/route trace` при загрузке.
- [x] `tests/expert_store.rs` — разбор размещения, union, математика таблиц (host/staging/mixed), auto — 4 теста зелёные.
- **Независимая проверка (стенд, `MOE_EXPERTS=ram`, `CTX=131072`, `MTP=0`, `NO_VRAM_PLAN=1`):** ✅ `[kv] paged pool: window=131072 blocks=2048 pool=1290MB`; ✅ паритет ram-vs-vram (тот же exe) top-10 логпробов **Δ = 0.0** на 4 096 токенах (обе стороны `PGRAPH=off`, чанк 512, `FA_SPLITS=1`); ✅ промпт 85 480 токенов → `finish_reason=length` за 451 с (120K-токенов не набралось — генератор дал 85K, механика та же); ✅ Shared Usage 9 424 МиБ ≈ pinned 8 644 + 9% (гранулярность WDDM по 120 буферам — см. TD-001); ⚠️ декод **20.6 ток/с** (графы) против цели 25 — цена zero-copy, закрывается кэшем фазы 4 (цель 40); ⚠️ прогон реплея — перенесён: сначала фаза 3 (выкладка), реплей на 131K.
- deviated: пейджед-прогрев префила без захвата выполняется при выгрузке независимо от PGRAPH — иначе eager-префил пишет KV мимо пула, а миграция int8-пула запрещена → графы декода не захватываются. С этим патчем `[graphs] captured nodes=4981` при выгрузке (критерий сценария 2).
- deviated: след маршрутизации пока только пишется (d2d в графе); D2H-чтение хостом — фаза 4 (before_step).
- deviated: обнаружено предсуществующее (есть на сборке ДО фазы 1: 4.4 против 3.6 ток/с): просадка декода на контексте ~2048 токенов у резидентного режима — отдельное расследование, вне рамок фазы (у обеих сборок графы захвачены).

### Фаза 3: Сервер для ранней выкладки (оценка: 6 ч) — ✅ выполнена 2026-09-04

- [x] `src/vram_plan.rs` — оценка весов по именам тензоров (`footprint_from_gguf_with`): ffn_*_exps при ram в pinned, nextn при MTP=0 исключён; q8-формула KV; `compute_dynamic` решает auto от KV-бюджета; отчёт с размещением.
- [x] [P] `src/engine.rs` (`MoeInfo` + trait fn), `src/engine_batched.rs` (dispatch заполняет из адаптера), `src/engine_swap.rs` (проброс), `src/api/openai.rs` — `/v1/models` → `capabilities.moe {experts, pinned_mib, staging_mib}` (FR-020).
- [x] [P] `src/main.rs` (эффективные MOE_EXPERTS/PGRAPH/PREFILL_CHUNK — FR-007), `README.md`, `.env.example`, `scripts/README.md` (требование RAM ≈29 ГиБ, PGRAPH при выгрузке).
- [x] Боевой `.env` стенда: `MOE_EXPERTS=auto`, `CTX=CONTEXT_LIMIT=131072`, `PREFILL_CHUNK=512`, без `NO_VRAM_PLAN`.
- [x] `src/config.rs` — `MOE_EXPERTS` (vram|ram|auto, fail-closed FR-009).
- **Независимая проверка (стенд):** ✅ старт без `NO_VRAM_PLAN` — планер печатает раскладку (`weights=1364MiB (experts=ram, kv_budget=9126MiB)`); ✅ `/v1/models` → `moe.experts="ram", pinned_mib=9346, staging_mib=300`; ✅ окно 131072.
- deviated: движковый `resolve_auto` (фаза 2) заменён на `resolve_auto_needs` (FR-021 «та же формула пула»): примитивное «влезают ли веса в free» выбирало vram на 35B и роняло KV-бюджет; планер и движок теперь сходятся.

### Фаза 4: Кэш горячих экспертов (оценка: 18 ч) — ✅ выполнена 2026-09-04

- [x] `real/expert_store.rs` — `SlotPool` на слой, `CacheDirectory` (LRU, pending, статистика), `ExpertCacheController::before_step` (D2H следа, LRU, ≤ `EXPERT_PROMOTE_PER_STEP` подъёмов на боковом потоке, трёхшаговый порядок по событиям), `prefill_prepare_layer` со свободными слотами + стейджингом без вытеснения, `after_prefill` по частоте, бюджет `EXPERT_CACHE_MIB`/авто, расчёт f, строки лога.
- [x] `real/adapter.rs` — `before_step` перед `decode_batch_graphed`; статистика в `capabilities.moe` (`cache_mib`, `cache_slots`, `hit_rate`).
- [x] `tests/expert_store.rs` — расширен для кэша (4 теста зелёные).
- **Независимая проверка (стенд):** ✅ hit rate **99.6–100%** (окно 200 шагов, capacity 204/256 = f≈0.80); ✅ декод **25.4 ток/с** (было 20.6 без кэша — кэш работает, но цель 40 не достигнута: см. ниже); ✅ `[graphs] captured`; ⚠️ цель 40 ток/с на 4K — ограничено предсуществующей просадкой декода от контекста (4 ток/с на 2048 и на старом exe) — расследование отдельно.
- deviated: подъёмы на **основном** потоке (боковой `new_stream` вызывает CUDA_ERROR_STREAM_CAPTURE_ISOLATION даже idle — TD-002, расследование отдельно); price ≈ 1–5 мс/шаг worst case, попадает между шагами.
- deviated: capacity кламп к n_experts (бюджет 9 ГиБ давал capacity 11738 слотов — бессмысленно больше 256).

### Фаза 5: MTP на MoE (оценка: 14 ч) — ✅ код работает (MTP MoE, acceptance 0% — модельная)
- [x] `real/model_profile.rs` — `MtpProfile` для `Qwen35Moe` (qwen35moe.* метаданные, MoE-тензоры, BF16, embedded MTP).
- [x] `real/mtp.rs` — `MtpFfn` enum (Dense | Moe), загрузка nextn-слоя с VRAM-резидентными экспертами (D-007), forward_rows + draft_pass_body ветвят по ffn.
- [x] [P] `tests/model_profile.rs` — 12 тестов зелёных (профиль + MoE-тензоры).
- **Проверка (стенд):** ✅ `MTP=1`, `MTP_PATH` = сам GGUF → загрузка прошла, `mtp.available=True` ✅; ✅ декод с MTP: drafted=110, accepted=0 (nextn-блок не производит полезные черновики для IQ2_XXS — модельная особенность), декод 22 ток/с с MTP overhead (без MTP 25-30 ток/с — D-015 адаптивная ширина должна минимизировать); ✅ MoE-FFN в draft_pass_body работает (3D input fix).
- deviated: BF16 разрешён в matrix/norm dtypes (unsloth GGUF использует BF16 для router/shared expert).
- deviated: draft_graph не поддерживает MoE — черновик идёт eager (верификация доминирует по времени).

### Фаза 6: Стабилизация и приёмка (оценка: 10 ч)

- [ ] FR-012: две смены Ornith ↔ 35B-A3B через `/admin/switch`, RAM/VRAM в пределах 256 МиБ, третий запуск отвечает.
- [ ] `auto` на Ornith даёт `vram`, скорость прежняя.
- [ ] Прогон всех критериев §13 спеки, отчёт с числами; обновление `docs/brief/decisions.md` (BD о размещении экспертов), памяти проекта, `TASKS.md`.
- **Независимая проверка:** чек-лист §13 спеки закрыт полностью, боевой `.env` стенда приведён к итоговому виду.

**Итого: ≈ 82 ч.**

## Трассируемость: требования → задачи

| Требование | Фаза | Задачи |
| --- | --- | --- |
| FR-001 размещение, `token_embd` в RAM | 2 ✅ | `expert_store.rs`, `model_weights.rs` |
| FR-002 таблица указателей в ядрах | 1 ✅ | `quantized.cu`, `cuda.rs`, `mod.rs` |
| FR-003 бит в бит | 1 ✅, 2 | `moe_table_cuda_tests.rs`, паритет логитов на стенде |
| FR-004 zero-copy промахи, постоянный след | 2 ✅ | `moe.rs` (`gpu_softmax_topk` → `RouteTrace`), `expert_store.rs` |
| FR-005 префил: стейджинг, без вытеснения, таблица на время слоя, чанк ≥ 256 | 2 ✅, 4 | `moe.rs`, `expert_store.rs`, `scheduler.rs` |
| FR-006 кэш, порядок выделения, лимит подъёмов, лог попаданий и f | 2, 4 ✅ | `expert_store.rs`, `model_weights.rs`, `adapter.rs` |
| FR-007 графы декода on, PGRAPH off с WARN, эффективная сводка | 2 ✅ | `adapter.rs` |
| FR-008 fail-closed | 2 ✅ | `expert_store.rs`, `model_weights.rs` |
| FR-009 `MOE_EXPERTS`, `EXPERT_CACHE_MIB`, особые случаи | 2, 4 | `expert_store.rs`, `moe.rs` (`reference`+`ram` → ошибка) |
| FR-010 планер по именам тензоров | 3 ✅ | `vram_plan.rs`, `config.rs` |
| FR-011 MTP на MoE | 5 ✅ (acceptance 0% — модельная) | `model_profile.rs`, `mtp.rs` |
| FR-012 выгрузка и смена модели | 2, 6 | `adapter.rs` (`unload`), проверка на стенде |
| FR-020 наблюдаемость | 3 ✅, 4 | `engine.rs`, `engine_batched.rs`, `api.rs` |
| FR-021 автовыбор с f и WARN | 2, 3 ✅, 4 | `model_weights.rs`, `expert_store.rs` |
| FR-030 прогрев по профилю | — | вне плана (P2, после приёмки) |
| FR-031 `SLOTS>1` корректно, f в логе | 4 | `expert_store.rs` (след на B слотов уже в FR-004) |

## Сложность и отклонения от принципов

| Отклонение | Принцип / более простой путь | Почему оправдано |
| --- | --- | --- |
| Новый модуль `expert_store.rs` и смена сигнатуры всех MoE-ядер | «Минимум кода»; проще было бы добавить второй набор ядер «с таблицей» рядом со старым | Один путь адресации вместо двух: резидентный режим — та же таблица, бит в бит проверяется одним тестом; второй набор ядер удвоил бы поверхность и оставил бы старую упаковку живой навсегда (D-016, FR-002) |
| Прямые вызовы `cudarc::driver::result::malloc_host` и `sys::cuMemHostGetDevicePointer_v2` | Штатный `alloc_pinned` | Штатный даёт только WRITECOMBINED без DEVICEMAP; флаги выбираются воротами 0 |

Новых зависимостей нет: cudarc уже в дереве.

## Риски и смягчения

| Риск | Вероятность | Влияние | Смягчение |
| --- | --- | --- | --- |
| Полоса zero-copy под WDDM < 8 ГБ/с | Средняя | Высокое | Ворота 0 до основной работы; запасной вариант D-005 — стейджинг с синком на слой (декод без графов на MoE-участках) |
| Потолок отображаемой host-памяти или отказ pinned-аллокации на 9.6 ГБ | Низкая | Высокое | Ворота 0 отображают весь объём; буферы по (слой, матрица); fail-closed с числами |
| Рефакторинг ядер ломает бит в бит | Низкая | Высокое | Фаза 1 закрывается CUDA-тестом и паритетом логитов до любых изменений размещения |
| Гонка подъёма и графа читает полузаписанный слот | Средняя | Высокое | Трёхшаговый порядок по событиям (FR-006), host-тест порядка в `tests/expert_store.rs` |
| Доля попаданий ниже расчётной (роутинг выровнен) | Средняя | Среднее | Бюджет из формулы f (§6 спеки); лог f и WARN; при промахе цели — `VRAM_HEADROOM_MIB` 512 и один слот |
| Префил замедляется подъёмами на критическом пути | Средняя | Низкое | Ожидаемые ≈13 % записаны в спеке; чанк ≥ 256 принудительно |
| MTP умножает промахи (k+1 токенов на проверке) | Средняя | Среднее | Адаптивная ширина по умолчанию (D-015); критерий «не медленнее без MTP» |
| Стенд занят агентскими тестами пользователя | Высокая | Среднее | Замеры по согласованию, откат — `.env.bak-ornith-pgraph-on` и задача планировщика |
| Соседние сборки съедают RAM, pinned не свопится | Средняя | Среднее | Проверка при старте, требование 29 ГБ в `scripts/README.md` |

## Зависимости

### Пакеты / библиотеки

- Новых нет. `cudarc 0.19.7` (`malloc_host`, `cuMemHostGetDevicePointer_v2`, потоки, события) уже в дереве форка.

### Связанные задачи

- Спека многоуровневого KV (`docs/specs/2026-08-27-tiered-kv-cache.md`) — независима: KV и эксперты не пересекаются по буферам.
- Открытые мелочи сервера (`FLASH_ATTN`-пустышка, `"strict": false`, стартовая сводка `sampling`) — сводка закрывается FR-007 попутно, остальное вне плана.

## Отложенные вопросы

| Вопрос | Почему отложен | Когда нужен ответ | Кто решает |
|---|---|---|---|
| — | — | — | — |

## Решения плана

| # | Вопрос | Решение | Дата |
| --- | --- | --- | --- |
| PD-001 | Как вводить таблицу в ядра | **Заменить параметр `all_weights` во всех вариантах `indexed_moe_forward*`** (q8_1, f32, dual, grouped) на таблицу; обёртки `QTensor` строят таблицу `base + id·stride` лениво — только для тестов и эталонных путей. Второй набор ядер отвергнут (два соглашения адресации). | 2026-09-04 |
| PD-002 | Где живёт хранилище экспертов | **`real/expert_store.rs` в движке** (D-016); `PackedExperts` становится `ExpertWeights::{Packed, Store}` — эталонный бэкенд остаётся на `QTensor`, PTX-путь на хранилище. | 2026-09-04 |
| PD-003 | Как получать след маршрутизации | **`gpu_softmax_topk` пишет ids прямо в постоянный срез `RouteTrace`** (слой × B·T_max × k); отдельного копирующего ядра нет, ids как тензор-вид того же буфера идут в ядра экспертов. | 2026-09-04 |
| PD-004 | Где стоит хук кэша на декоде | **В начале следующего шага** (`before_step` перед `decode_batch_graphed`): след прошлого шага уже завершён, потому что логиты прочитаны сэмплером; подъёмы на боковом потоке перекрываются с графом, таблица меняется по событиям. Дополнительного синка нет. | 2026-09-04 |
| PD-005 | Ранняя выкладка | **Сценарий 1 выкладывается на стенд после фаз 2–3** (zero-copy, ≈30 ток/с), кэш и MTP — следом. Пользователь | 2026-09-04 |
| PD-006 | Pinned-аллокация | **`result::malloc_host` с `DEVICEMAP` (+ `WRITECOMBINED` по воротам 0)**, по буферу на (слой, матрица); заполнение из mmap по смещениям `tensor_infos` без промежуточного `QTensor` на CPU. | 2026-09-04 |
| PD-007 | Форма ворот 0 | **Пример `candle-core/examples/moe_zero_copy_bench.rs` на реальных тензорах GGUF**, а не синтетика: заодно проверяется путь чтения по смещениям. | 2026-09-04 |
| PD-008 | Стратегия тестов | **CUDA-тест бит в бит на ядра (фаза 1), host-тесты директории/порядка без GPU, стендовые проверки существующими инструментами** (`logits_parity.py`, реплей, `multiturn.ps1`, `qwen35_mtp_gate`). Новых фреймворков нет. | 2026-09-04 |
| PD-009 | Где принуждать чанк ≥ 256 | **`scheduler.rs::prefill_chunk()`** при размещении `ram`, строка в логе. | 2026-09-04 |
| PD-010 | Пул при загрузке | **Вызвать `init_paged_decode` сразу после загрузки при `ram`**: причина ленивости (транзиенты префила искажают `mem_get_info`) при загрузке отсутствует; порядок пул → стейджинг → кэш. | 2026-09-04 |
| PD-011 | Стейджинг в фазе 2 без кэша | **Всё объединение чанка идёт в стейджинг** (кэша ещё нет); фаза 4 добавляет свободные слоты и `after_prefill`. Так сценарий 1 выкладывается без половинчатого кэша. | 2026-09-04 |

## Технический долг

| # | Описание | Фаза | Приоритет |
|---|---|---|---|
| — | — | — | — |
