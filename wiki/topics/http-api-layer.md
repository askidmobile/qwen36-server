---
topic: HTTP API Layer
slug: http-api-layer
last_compiled: 2026-09-01
sources: 11
status: active
---

# HTTP API Layer

## Purpose [coverage: high — 11 sources]

Axum-слой предоставляет три inference API, metadata/admin/HF/media endpoints и SSE. Он нормализует сообщения, tool calls, reasoning, sampling и типизированные ошибки до передачи в `Engine`.

## Architecture [coverage: high — 11 sources]

- Bearer auth сопоставляет ключ с обезличенным owner digest для временных media.
- `prepare_inference_request` собирает `InferenceRequest`; `generate_collect` обслуживает non-stream.
- SSE-forwarder одновременно читает engine events и проверяет закрытие клиента; keepalive = 1 s.
- Tool output поддерживает Qwen/Hermes/Gemma channel форматы. Разбор выполняется только для непустого списка `tools` и завершённой генерации; `finish_reason=length` сохраняет сырой хвост как content.
- Tool-call parser сверяет имя и аргументы с JSON Schema запроса: обязательные поля, object/array/scalar types, properties/items, `additionalProperties=false`, `anyOf`/`oneOf`. Hermes-поля типа `string` остаются строками, даже если их текст похож на JSON; JSON-вызовы с object/null вместо string больше не преобразуются автоматически. Поддерживаются OpenAI `function.parameters` и Anthropic `input_schema`.
- При tools streaming буферизует разметку, чтобы служебные `<tool...>` не утекали как обычный текст.

## Talks To [coverage: high — 11 sources]

- `Engine`, `MediaService`, `SwappableEngine`, sampling presets/policy.
- Hugging Face REST API для поиска/probe/download.
- Unsloth Studio backend через `reqwest` proxy.

## API Surface [coverage: high — 11 sources]

| Группа | Эндпоинты |
|---|---|
| Inference | `POST /v1/chat/completions`, `/v1/responses`, `/v1/messages` |
| Discovery | `GET /v1/models`, `/v1/available_models`, `/v1/model_native_ctx`, `/v1/ctx_matrix` |
| Lifecycle | model switch/unload и sampling admin endpoints |
| HF | search/files/probe/download/downloads |
| Media | upload и использование ссылок/одноразовых объектов |

Переполнение возвращается как `400` с `code=context_length_exceeded`. Loading/unavailable component ошибки имеют отдельные HTTP-типы.

## Data [coverage: high — 8 sources]

- `AppState`: engine/switcher, keys, media, model root, optional profile, CUDA device, downloads, sampling/presets, env file.
- История tool calls хранится структурно в `ChatMessage`, а не только строкой.
- `/v1/models` отдаёт фактические capabilities движка и effective sampling metadata.

## Key Decisions [coverage: high — 7 sources]

- Именованные ключи без ролей (BD-021).
- Media разрешена только runtime, реально умеющим vision/video (BD-022/026).
- Sampling policy применяется одинаково к OpenAI и Anthropic; сервер может игнорировать клиентские overrides при lock.
- Агент должен сам сжимать историю после `context_length_exceeded` (BD-031).
- На выделенном agent endpoint sampling по умолчанию env-wins; reasoning остаётся отдельной осью (BD-032).
- Поле `model` не переключает runtime. Принимаются только фактический `model_info.id` и display alias `id-quant` той же загруженной модели; любое другое имя получает `model_not_loaded`.

## Gotchas [coverage: high — 8 sources]

- Законный, но повторяющийся tool call не считается unknown: server log выявляет только имя вне присланного списка, а cycle breaker остаётся обязанностью агента.
- Нельзя определять тип Hermes-параметра только по его тексту: JSON-тело команды `bash.command` или содержимое `write.content` является строкой по schema. Эвристический JSON-разбор без schema превращал его в object и вызывал клиентское `must be string`.
- Tolerant JSON repair не является разрешением на исполнение: после `length`, при неизвестном tool name или нарушении schema вызов не попадает в `tool_calls`/`tool_use`, а остаётся диагностируемым content. Иначе сервер мог сам достроить оборванный JSON без обязательного `path` и ложно сообщить успех.
- Hermes framing удаляет ровно один служебный перевод строки с каждой границы параметра; `trim_matches('\n')` стирал намеренные пустые строки в `write.content`.
- Anthropic/OpenAI имеют разные формы thinking/tool blocks; изменения должны проверяться в обоих путях.
- Reverse proxy не использует inference auth; `/v1/*` остаётся в основном router.
- `reasoning_effort=high|xhigh` не должен менять temperature/penalties: это template control, не псевдоним coding preset.

## Sources

- [src/api.rs](../../src/api.rs)
- [src/api/openai.rs](../../src/api/openai.rs)
- [src/api/anthropic.rs](../../src/api/anthropic.rs)
- [src/api/responses.rs](../../src/api/responses.rs)
- [src/api/admin.rs](../../src/api/admin.rs)
- [src/api/hf.rs](../../src/api/hf.rs)
- [src/api/media.rs](../../src/api/media.rs)
- [src/api/proxy.rs](../../src/api/proxy.rs)
- [src/api/tools.rs](../../src/api/tools.rs)
- [docs/engine-api.md](../../docs/engine-api.md)
- [Sampling policy recommendation](../../docs/research/2026-08-30-sampling-policy-recommendation.md)
