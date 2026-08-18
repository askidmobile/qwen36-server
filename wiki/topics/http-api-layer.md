---
topic: HTTP API Layer
slug: http-api-layer
last_compiled: 2026-08-08
sources: 6
status: active
---

# HTTP API Layer

## Purpose [coverage: high — 6 sources]

HTTP-слой на axum 0.8 — три API (OpenAI Chat Completions, OpenAI Responses, Anthropic Messages) + `/v1/models` + веб-чат статика. Зависит только от `trait Engine`, не знает реализацию. Auth — мастер-ключ (BD-005). Vision → 400 (BD-004).

## Architecture [coverage: high — 6 sources]

Router (src/api.rs:79-88):
```rust path=/Volumes/Askid Dev/Projects/Qwen3.6 27B/src/api.rs start=79
pub fn build_router(state: AppState) -> Router {
    let v1 = Router::new()
        .route("/chat/completions", axum::routing::post(openai::chat_completions))
        .route("/responses", axum::routing::post(responses::responses))
        .route("/messages", axum::routing::post(anthropic::messages))
        .route("/models", axum::routing::get(openai::list_models))
        .route_layer(axum::middleware::from_fn_with_state(state.clone(), auth));
    Router::new().nest("/v1", v1).with_state(state)
}
```

- **auth middleware** (api.rs:61) — проверка `Authorization: Bearer $QWEN36_API_KEY`; без/неверный → 401 `authentication_error`
- **sse_response** (api.rs:91) — обёртка: `mpsc::Receiver<StreamEvent>` → SSE с `map()` функцией (StreamEvent → 0+ Event)
- **generate_collect** (api.rs:132) — non-stream: собирает весь текст + usage в `GenOutcome`
- **parse_tool_calls** (api.rs:167) — парсинг `⟨tool_call⟩{...}⟨/tool_call⟩` из вывода модели → (text, [(name, args)])

Модули:
- `src/api/openai.rs` — Chat Completions (stream/non-stream, tools, usage) + list_models
- `src/api/anthropic.rs` — Messages (max_tokens обязателен, system, content blocks, tool_use, thinking blocks)
- `src/api/responses.rs` — Responses (input→output, SSE events response.created/output_text.delta/completed)

## Talks To [coverage: high — 6 sources]

- `crate::engine_types::Engine` — единственная зависимость от инференса
- `axum 0.8` — Router, State, middleware, SSE, IntoResponse, Json
- `tokio::sync::mpsc` — каналы StreamEvent
- `serde_json` — JSON bodies
- `uuid` — генерация id (chatcmpl-, msg_, resp_, call_, toolu_)

## API Surface [coverage: high — 6 sources]

| Эндпоинт | Метод | Описание |
|---|---|---|
| `/v1/chat/completions` | POST | OpenAI: stream SSE + tools + usage (stream_options.include_usage) |
| `/v1/responses` | POST | OpenAI Responses: input→output, SSE, store=false (BD-012) |
| `/v1/messages` | POST | Anthropic: max_tokens обязателен, stream, tools, anthropic-version |
| `/v1/models` | GET | id/object/created/owned_by + context_length/quant/slots/modes (BD-015) |
| `/` | GET | статика веб-чата (include_str! web/index.html) |

Auth: `Authorization: Bearer $QWEN36_API_KEY` на всех `/v1/*`.

SSE-форматы:
- OpenAI: `data: {json}\n\n`, финал `data: [DONE]`
- Anthropic: event + data (`message_start`, `content_block_start/delta/stop`, `message_delta`, `message_stop`)
- Responses: `response.created` → `response.output_text.delta` → `response.completed`

## Data [coverage: medium — 4 sources]

- `AppState { engine: Arc<dyn Engine>, api_key: String }` — shared state
- `GenOutcome { text, finish_reason, prompt_tokens, completion_tokens, truncated }` — результат non-stream
- Tool calls: парсинг из накопленного текста, эмит в формате платформы (OpenAI `tool_calls` / Anthropic `tool_use` blocks)
- `truncated: true` в usage при sliding window (BD-017)

## Key Decisions [coverage: high — 6 sources]

- **BD-005**: один мастер-ключ из env, `Authorization: Bearer`
- **BD-004**: vision → 400 (image_url, image, video, document content blocks)
- **BD-012**: Responses API — базовый (input→output, store=false); полный Responses — OQ-6
- **BD-013**: function calling — tools + tool_choice в Chat Completions и Messages; парсинг из вывода
- **BD-015**: /v1/models — OpenAI-минимум + расширения
- **BD-017**: sliding window, `truncated: true` в ответе
- Streaming tool_calls: эмитятся в конце потока дельтами (OpenAI) / целиком (Anthropic) — ponytail: буферизовать весь поток при заданных tools

## Gotchas [coverage: medium — 4 sources]

- Anthropic: `max_tokens` обязателен (по контракту), без него → 400
- Anthropic streaming: tool_use блоки эмитятся целиком после text block stop (допустимо по заданию)
- OpenAI streaming: tool_call-разметка может утечь в поток текста (ограничение; ponytail: буферизовать)
- Vision content (image_url, image, video, document) → 400 на этапе парсинга messages
- `parse_tool_calls` ищет `⟨tool_call⟩...⟨/tool_call⟩` — незакрытый тег остаётся как текст
- Thinking blocks в Anthropic: парсятся из `<think>`/`<thinking>` маркеров в тексте

## Sources

- [src/api.rs](../../src/api.rs)
- [src/api/openai.rs](../../src/api/openai.rs)
- [src/api/anthropic.rs](../../src/api/anthropic.rs)
- [src/api/responses.rs](../../src/api/responses.rs)
- [src/main.rs](../../src/main.rs)
- [docs/engine-api.md](../../docs/engine-api.md)
