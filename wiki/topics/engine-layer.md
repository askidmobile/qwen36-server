---
topic: Engine Layer
slug: engine-layer
last_compiled: 2026-08-30
sources: 8
status: active
---

# Engine Layer

## Purpose [coverage: high — 8 sources]

Engine-слой изолирует API от конкретного runtime. `Engine` принимает `InferenceRequest` и возвращает поток `StreamEvent`; контракт включает structured messages/tools, media, reasoning effort, cancellation, logprobs и usage MTP/media.

## Architecture [coverage: high — 8 sources]

- `BatchedEngine` — основной qwen35/qwen35moe путь даже при `SLOTS=1`: scheduler, chunked prefill, paged KV, CUDA graphs, MTP и prefix reuse.
- `CandleEngine` — single-stream fallback для других архитектур и `QWEN36_FORCE_CANDLE_ENGINE=1`.
- `SwappableEngine` — `RwLock<Option<Arc<dyn Engine>>>`, hot unload/install и стабильный API объект.
- Dispatch loop остаётся выделенным `std::thread`: model/CUDA state принадлежит одному потоку, HTTP общается каналами.

## Talks To [coverage: high — 8 sources]

- `qwen35_batch::{BatchScheduler,Qwen35BatchAdapter,ModelWeights}` из `yttri-forge`.
- tokenizer/chat template для текстовых и tool сообщений.
- `PrefixCache` для `StateSnapshot` на границе prefill chunk.
- media service/component artifacts для multimodal admission.
- `vram_plan` и env-параметры KV/graphs/MTP.

## API Surface [coverage: high — 6 sources]

- `Engine::generate(InferenceRequest)`; `shutdown`, `ready`, `load_error`.
- Capability probes: `supports_vision`, `supports_video`, `supports_mtp`.
- `ModelInfo`: id/context/quant/slots/modes.
- `CancelFlag` + `CancelOnDrop`: обрыв SSE переводится в отмену; keepalive раз в секунду позволяет заметить мёртвый сокет во время prefill.
- `ContextOverflow`: типизированная ошибка для HTTP 400.

## Data [coverage: high — 6 sources]

- Per-slot prompt, decoder text state, sampler RNG, cancellation, usage and media lease.
- Paged KV/static slot regions и DeltaNet recurrent state внутри адаптера.
- MTP head и graph state загружаются при старте только при включённом MTP.
- Prefix snapshots хранятся в host memory, возвращаются на GPU только на hit.

## Key Decisions [coverage: high — 5 sources]

- Один слот тоже использует batched engine, чтобы не терять graphs/paged KV.
- Отмена проверяется между chunk-ами prefill и шагами decode.
- По умолчанию переполнение не обрезает историю молча (BD-031).
- MTP допускает численные расхождения при сохранении корректного распределения (BD-029).

## Gotchas [coverage: high — 7 sources]

- Prefix snapshot нельзя откатить назад: DeltaNet хранит рекуррентное состояние, а не историю по позициям.
- `supports_*` обязан пробрасываться через `SwappableEngine`, иначе `/v1/models` врёт о runtime capabilities.
- VRAM planner и paged-pool allocator пока считают бюджет независимо.
- `CTX` должен быть проброшен в окружение движка; сервер принимает и чистые, и legacy `QWEN36_*` имена.
- MTP может быть корректным вероятностно и при этом менять конкретный greedy-токен около численной ничьей; это отдельная зона диагностики.

## Sources

- [src/engine.rs](../../src/engine.rs)
- [src/engine_batched.rs](../../src/engine_batched.rs)
- [src/engine_types.rs](../../src/engine_types.rs)
- [src/engine_swap.rs](../../src/engine_swap.rs)
- [src/sampler.rs](../../src/sampler.rs)
- [src/vram_plan.rs](../../src/vram_plan.rs)
- [src/main.rs](../../src/main.rs)
- [docs/engine-api.md](../../docs/engine-api.md)
