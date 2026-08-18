---
topic: Engine Layer
slug: engine-layer
last_compiled: 2026-08-08
sources: 5
status: active
---

# Engine Layer

## Purpose [coverage: high — 5 sources]

Engine-слой — абстракция над инференсом. Контракт: `trait Engine` с методом `generate()` → `mpsc::Receiver<StreamEvent>`. HTTP-слой зависит только от trait, не знает реализацию (CandleEngine single-slot или BatchedEngine 4-slot).

Контрактные типы определены в `src/engine.rs`, реэкспортированы через `src/engine_types.rs` (единый источник). Описание контракта — `docs/engine-api.md`.

## Architecture [coverage: high — 5 sources]

`trait Engine: Send + Sync` (src/engine.rs:96-105):
```rust path=/Volumes/Askid Dev/Projects/Qwen3.6 27B/src/engine.rs start=96
#[async_trait::async_trait]
pub trait Engine: Send + Sync {
    async fn generate(&self, messages: Vec<ChatMessage>, params: GenParams)
        -> Result<mpsc::Receiver<StreamEvent>>;
    fn model_info(&self) -> ModelInfo;
}
```

Типы:
- `GenParams` — temperature, top_p, top_k, min_p, presence_penalty, repetition_penalty, max_tokens, stop, seed
- `ChatMessage` — role (system|user|assistant|tool), content
- `StreamEvent` — Delta(String), Done{finish_reason, prompt_tokens, completion_tokens, truncated}, Error(String)
- `ModelInfo` — id, context_length, quant, slots, modes

Две реализации:
1. **CandleEngine** (src/engine.rs:119) — single-slot, `Arc<Mutex<ModelState>>`. Загрузка GGUF через `ModelWeights::from_gguf` (zero-copy на macOS+Metal). Генерация в `spawn_blocking`. Sliding window + сэмплинг через `crate::sampler`.
2. **BatchedEngine** (src/engine_batched.rs:108) — 4 слота поверх `BatchScheduler<Qwen35BatchAdapter>`. Dispatch loop в отдельном `std::thread` (модель не Send). Каналы `mpsc` для общения с HTTP-хендлерами.

Выбор engine в `main.rs:27`: `slots > 1` → BatchedEngine, иначе CandleEngine.

## Talks To [coverage: high — 5 sources]

- `qwen35_batch::real::ModelWeights` — веса модели, `from_gguf`/`from_gguf_zero_copy`, `forward`, `clear_state`
- `qwen35_batch::real::tokenizer` — `load_from_gguf_path`, `build_chatml_text`, `encode_no_think`, `decode_text`, `strip_thinking`, `ChatMsg`
- `qwen35_batch::scheduler::{BatchScheduler, StepOutcome}` — batched decode
- `qwen35_batch::real::Qwen35BatchAdapter` — `BatchModel` over `ModelWeights`
- `qwen35_batch::{slot::SlotStatus, model::{BatchModel, Sampler}}` — slot FSM, sampler trait
- `crate::sampler` — свой сэмплинг (temperature/top_k/top_p/min_p/penalties)
- `crate::config::Config` — env-конфиг
- `candle_core::{Device, Tensor, quantized::gguf_file}` — тензоры, устройство, GGUF parser

## API Surface [coverage: high — 5 sources]

- `Engine::generate(messages, params) -> Receiver<StreamEvent>` — стрим генерации
- `Engine::model_info() -> ModelInfo` — метаданные модели
- `CandleEngine::load(&Config) -> Result<Self>` — загрузка single-slot
- `BatchedEngine::load(BatchConfig) -> Result<Arc<Self>>` — загрузка batched
- `cancel_guard(&BatchedEngine, req_id) -> CancelGuard` — Drop guard для отмены
- `select_device() -> Result<Device>` — cuda > metal > CPU (BD-011)
- `quant_from_filename(path) -> String` — парсинг кванта из имени файла
- `trim_messages(msgs, budget, count) -> (Vec, bool)` — sliding window (BD-017)
- `floor_char_boundary(s, i) -> usize` — UTF-8-корректный индекс

## Data [coverage: medium — 4 sources]

**CandleEngine:**
- `ModelState { model: ModelWeights, tokenizer: Tokenizer, device: Device }` под `Arc<Mutex>`
- EOS из GGUF metadata `tokenizer.ggml.eos_token_id` (default 151645)
- KV state / recurrent state внутри `ModelWeights` (single-stream)

**BatchedEngine:**
- `BatchScheduler<Qwen35BatchAdapter>` — владеет слотами, внутренней очередью
- `bindings: [Option<SlotBinding>; slots]` — слот → активный запрос
- `pending: VecDeque<AdmitReq>` — backpressure (>4 активных)
- `slot_samplers: HashMap<usize, (GenParams, Rng)>` — per-slot сэмплер
- `IndexedSampler` — shim для форк-trait `Sampler` с per-slot params (TODO-F5)
- KV cache / DeltaNet state в batched GPU-буферах (seed_slot_batched)

## Key Decisions [coverage: high — 5 sources]

- **BD-001**: candle-форк, не llama.cpp
- **BD-017**: sliding window — system сохраняется, режутся старые user/assistant пары, `truncated: true`
- **BD-016**: сэмплинг-пресеты (thinking t=1.0, thinking-coding t=0.6, instruct t=0.7/pp=0.80/penalty=1.5)
- **BD-011**: cuda (Windows) > metal (macOS) > CPU
- Модель не Send → `std::thread` для dispatch loop, не `tokio::spawn`
- Per-request сэмплинг требует indexed sampler патч форка (TODO-F5); до патча — greedy fallback

## Gotchas [coverage: medium — 4 sources]

- CandleEngine: оценка токенов per-message (BPE-границы ~1-2 токена погрешность) — покрыта `TRIM_MARGIN=16`
- CandleEngine: полный decode всей generated последовательности (инкрементальный буфер — ponytail)
- BatchedEngine: prefill неделим — блокирует decode других слотов
- BatchedEngine: `sched.slots_mut()` нужен для сбора Finished (TODO-F6 — патч форка)
- BatchedEngine: cancel mid-prefill невозможен (отложен до конца prefill)
- BatchedEngine: `out.try_send` Full → Cancel (slow client не блокирует батч)
- Ошибка `step()` → Error всем активным + reset всех + `clear_state_batched`, loop жив (BD-008)
- Watchdog: слот без прогресса `QWEN36_REQ_TIMEOUT` (600s) → Cancel + Error

## Sources

- [src/engine.rs](../../src/engine.rs)
- [src/engine_batched.rs](../../src/engine_batched.rs)
- [src/engine_types.rs](../../src/engine_types.rs)
- [docs/engine-api.md](../../docs/engine-api.md)
- [docs/batch-integration.md](../../docs/batch-integration.md)
