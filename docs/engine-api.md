# Engine API contract (v0)

Внутренний контракт крейта `qwen36-server` между engine-слоем и HTTP-слоем. Источник истины по решениям — `docs/brief/decisions.md`.

## Config (env/CLI)

| Параметр | env | default |
|---|---|---|
| Путь к GGUF | `QWEN36_MODEL` | `models/qwen36-27b-q2_k_xl.gguf` |
| Host | `QWEN36_HOST` | `0.0.0.0` |
| Port | `QWEN36_PORT` | `8080` |
| Master API key | `QWEN36_API_KEY` | обязателен; без него сервер не стартует |
| Контекст | `QWEN36_CTX` | `81920` |
| Слоты | `QWEN36_SLOTS` | `4` |

## Engine trait (HTTP-слой зависит только от него)

```rust
pub struct GenParams {
    pub temperature: f32,          // default 1.0
    pub top_p: f32,                // default 0.95
    pub top_k: usize,              // default 20
    pub min_p: f32,                // default 0.0
    pub presence_penalty: f32,     // default 0.0
    pub repetition_penalty: f32,   // default 1.0
    pub max_tokens: usize,         // default 4096
    pub stop: Vec<String>,
    pub seed: Option<u64>,
}

pub struct ChatMessage { pub role: String, pub content: String } // role: system|user|assistant|tool

pub enum StreamEvent {
    Delta(String),          // текстовый кусок (thinking включён в поток, парсит HTTP-слой)
    Done { finish_reason: String, prompt_tokens: usize, completion_tokens: usize, truncated: bool },
    Error(String),
}

#[async_trait::async_trait]
pub trait Engine: Send + Sync {
    /// Стриминговая генерация по chat-сообщениям. sliding window применяется внутри (BD-017).
    async fn generate(&self, messages: Vec<ChatMessage>, params: GenParams)
        -> Result<tokio::sync::mpsc::Receiver<StreamEvent>>;
    fn model_info(&self) -> ModelInfo;
}

pub struct ModelInfo {
    pub id: String,            // "qwen3.6-27b"
    pub context_length: usize, // 81920
    pub quant: String,         // "Q2_K_XL"
    pub slots: usize,          // 4
    pub modes: Vec<String>,    // ["thinking", "instruct"]
}
```

Engine реализуется поверх `qwen35-batch` (`ModelWeights::from_gguf` / `Qwen35BatchAdapter::load`, `forward`, токенизатор `build_chatml_text`/`encode_no_think`/`decode_text`/`strip_thinking` — см. `candle-fork-qwen35-batch/qwen35-batch/src/real/`).

## HTTP endpoints

- `POST /v1/chat/completions` — OpenAI: stream SSE (`data: {json}\n\n`, финал `data: [DONE]`), tools/tool_choice, `usage` в финальном chunk (`stream_options.include_usage`).
- `POST /v1/responses` — базовый: `input` (строка или messages), output text, SSE-события `response.created` → `response.output_text.delta` → `response.completed`; `store=false`.
- `POST /v1/messages` — Anthropic: тело `{model, max_tokens (обязателен), messages, system?, tools?, tool_choice?, stream}`; заголовок `anthropic-version: 2023-06-01`; SSE: `message_start`, `content_block_start/delta/stop` (text | tool_use), `message_delta` (stop_reason + usage), `message_stop`.
- `GET /v1/models` — `{object:"list", data:[{id, object:"model", created, owned_by:"local", context_length, quant, slots, modes}]}`.
- `GET /` — статика веб-чата.
- Auth: `Authorization: Bearer $QWEN36_API_KEY` на всех `/v1/*`; без/неверный → 401 `{error:{type:"authentication_error"}}`.
- Image/video content в messages → 400 `{error:{type:"invalid_request_error", message:"vision not supported"}}` (BD-004).

## Tools

Модель Qwen3.6 эмитит tool calls как `<tool_call>{"name":..,"arguments":{..}}</tool_call>`. HTTP-слой: парсинг из накопленного текста, эмит в формате платформы (OpenAI `tool_calls` / Anthropic `tool_use`).

## Сэмплинг-пресеты (BD-016)

- `thinking`: t=1.0, top_p=0.95, top_k=20, min_p=0, presence_penalty=0, repetition_penalty=1.0
- `thinking-coding`: t=0.6, top_p=0.95, top_k=20
- `instruct`: t=0.7, top_p=0.80, top_k=20, presence_penalty=1.5
