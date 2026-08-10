use axum::{
    extract::State,
    response::{sse::Event, IntoResponse, Response},
    Json,
};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::api::{bad_request, generate_collect, internal_error, parse_tool_calls, sse_response, AppState};
use crate::engine_types::{ChatMessage, GenParams, StreamEvent};

#[derive(Deserialize)]
pub struct ChatCompletionRequest {
    #[allow(dead_code)]
    model: Option<String>,
    messages: Vec<OaiMessage>,
    #[serde(default)]
    stream: bool,
    temperature: Option<f32>,
    top_p: Option<f32>,
    max_tokens: Option<usize>,
    stop: Option<Value>, // string | [string]
    #[allow(dead_code)]
    tools: Option<Value>,
    #[allow(dead_code)]
    tool_choice: Option<Value>,
    stream_options: Option<Value>,
    /// Qwen-конвенция: chat_template_kwargs.enable_thinking (bool).
    chat_template_kwargs: Option<Value>,
    /// Прямой флаг thinking (альтернатива chat_template_kwargs).
    thinking: Option<bool>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum OaiContent {
    Text(String),
    Parts(Vec<Value>),
}

#[derive(Deserialize)]
struct OaiMessage {
    role: String,
    content: Option<OaiContent>,
}

fn flatten_content(m: &OaiMessage) -> Result<String, Response> {
    match &m.content {
        None => Ok(String::new()),
        Some(OaiContent::Text(t)) => Ok(t.clone()),
        Some(OaiContent::Parts(parts)) => {
            let mut out = String::new();
            for p in parts {
                match p.get("type").and_then(|t| t.as_str()) {
                    Some("text") => {
                        if let Some(t) = p.get("text").and_then(|t| t.as_str()) {
                            out.push_str(t);
                        }
                    }
                    Some("image_url") | Some("image") | Some("video") => {
                        return Err(bad_request("vision not supported")); // BD-004
                    }
                    _ => {}
                }
            }
            Ok(out)
        }
    }
}

fn to_gen_params(req: &ChatCompletionRequest) -> GenParams {
    let mut p = GenParams::default();
    if let Some(t) = req.temperature {
        p.temperature = t;
    }
    if let Some(t) = req.top_p {
        p.top_p = t;
    }
    if let Some(m) = req.max_tokens {
        p.max_tokens = m;
    }
    if let Some(stop) = &req.stop {
        match stop {
            Value::String(s) => p.stop.push(s.clone()),
            Value::Array(arr) => {
                p.stop.extend(arr.iter().filter_map(|v| v.as_str().map(String::from)))
            }
            _ => {}
        }
    }
    // thinking: прямой флаг приоритетнее chat_template_kwargs; default true.
    let ctk = req
        .chat_template_kwargs
        .as_ref()
        .and_then(|k| k.get("enable_thinking"))
        .and_then(|v| v.as_bool());
    p.thinking = req.thinking.or(ctk).unwrap_or(true);
    p
}

fn build_messages(req: &ChatCompletionRequest) -> Result<Vec<ChatMessage>, Response> {
    req.messages
        .iter()
        .map(|m| {
            Ok(ChatMessage {
                role: m.role.clone(),
                content: flatten_content(m)?,
            })
        })
        .collect()
}

fn oai_tool_calls(calls: &[(String, String)]) -> Value {
    Value::Array(
        calls
            .iter()
            .enumerate()
            .map(|(i, (name, args))| {
                json!({
                    "id": format!("call_{}", uuid::Uuid::new_v4().simple()),
                    "type": "function",
                    "index": i,
                    "function": {"name": name, "arguments": args},
                })
            })
            .collect(),
    )
}

pub async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    let messages = match build_messages(&req) {
        Ok(m) => m,
        Err(r) => return r,
    };
    let params = to_gen_params(&req);
    let include_usage = req
        .stream_options
        .as_ref()
        .and_then(|o| o.get("include_usage"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    if req.stream {
        return stream_chat(state, messages, params, include_usage).await;
    }

    let out = match generate_collect(state.engine.as_ref(), messages, params).await {
        Ok(o) => o,
        Err(r) => return r,
    };
    let (text, calls) = parse_tool_calls(&out.text);
    let finish = if !calls.is_empty() {
        "tool_calls"
    } else {
        out.finish_reason.as_str()
    };
    let mut message = json!({"role": "assistant", "content": if text.is_empty() { Value::Null } else { json!(text) }});
    if !calls.is_empty() {
        message["tool_calls"] = oai_tool_calls(&calls);
    }
    Json(json!({
        "id": format!("chatcmpl-{}", uuid::Uuid::new_v4().simple()),
        "object": "chat.completion",
        "created": now_unix(),
        "model": state.engine.model_info().id,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish,
        }],
        "usage": {
            "prompt_tokens": out.prompt_tokens,
            "completion_tokens": out.completion_tokens,
            "total_tokens": out.prompt_tokens + out.completion_tokens,
            "truncated": out.truncated,
        },
    }))
    .into_response()
}

pub(crate) fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn chunk(id: &str, model: &str, delta: Value, finish: Option<&str>) -> String {
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": now_unix(),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    })
    .to_string()
}

async fn stream_chat(
    state: AppState,
    messages: Vec<ChatMessage>,
    params: GenParams,
    include_usage: bool,
) -> Response {
    let id = format!("chatcmpl-{}", uuid::Uuid::new_v4().simple());
    let model = state.engine.model_info().id;
    let rx = match state.engine.generate(messages, params).await {
        Ok(r) => r,
        Err(e) => return internal_error(e.to_string()),
    };

    let mut first = true;
    let mut acc = String::new();
    sse_response(rx, move |ev, out| match ev {
        StreamEvent::Delta(d) => {
            if first {
                first = false;
                out.push(Event::default().data(chunk(&id, &model, json!({"role": "assistant"}), None)));
            }
            acc.push_str(&d);
            out.push(Event::default().data(chunk(&id, &model, json!({"content": d}), None)));
            true
        }
        StreamEvent::Done {
            finish_reason,
            prompt_tokens,
            completion_tokens,
            truncated,
        } => {
            if first {
                first = false;
                out.push(Event::default().data(chunk(&id, &model, json!({"role": "assistant"}), None)));
            }
            let (_text, calls) = parse_tool_calls(&acc);
            // tool_calls эмитим дельтами в конце; текст уже ушёл как deltas
            // (ограничение: tool_call-разметка могла утечь в поток —
            // ponytail: буферизовать весь поток при заданных tools)
            for (i, (name, args)) in calls.iter().enumerate() {
                out.push(Event::default().data(chunk(
                    &id,
                    &model,
                    json!({"tool_calls": [{
                        "index": i,
                        "id": format!("call_{}", uuid::Uuid::new_v4().simple()),
                        "type": "function",
                        "function": {"name": name, "arguments": args},
                    }]}),
                    None,
                )));
            }
            let finish = if calls.is_empty() { finish_reason.as_str() } else { "tool_calls" };
            out.push(Event::default().data(chunk(&id, &model, json!({}), Some(finish))));
            if include_usage {
                out.push(Event::default().data(
                    json!({
                        "id": id, "object": "chat.completion.chunk",
                        "created": now_unix(), "model": model,
                        "choices": [],
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens,
                            "truncated": truncated,
                        },
                    })
                    .to_string(),
                ));
            }
            out.push(Event::default().data("[DONE]"));
            false
        }
        StreamEvent::Error(e) => {
            out.push(Event::default()
                .data(json!({"error": {"type": "internal_error", "message": e}}).to_string()));
            false
        }
    })
}

pub async fn list_models(State(state): State<AppState>) -> Response {
    let info = state.engine.model_info();
    // OpenAI-минимум + расширения (BD-015) для внешних систем: нативный
    // контекст, веса, возможности, дефолты и пресеты сэмплинга (BD-016).
    let (path, _, _) = state
        .switcher
        .current
        .read()
        .map(|c| c.clone())
        .unwrap_or_default();
    let (native_ctx, size_mib) = tokio::task::spawn_blocking({
        let p = path.clone();
        move || {
            crate::vram_plan::footprint_from_gguf(&p)
                .map(|fp| (fp.native_ctx, fp.weights_mib))
                .unwrap_or((0, 0))
        }
    })
    .await
    .unwrap_or((0, 0));
    let d = crate::engine_types::GenParams::default();
    Json(json!({
        "object": "list",
        "data": [{
            "id": info.id,
            "object": "model",
            "created": now_unix(),
            "owned_by": "local",
            // активные параметры
            "context_length": info.context_length,
            "quant": info.quant,
            "slots": info.slots,
            "modes": info.modes,
            // паспорт модели
            "native_context_length": native_ctx,
            "file_size_mib": size_mib,
            "path": path.to_string_lossy(),
            // возможности для агентов
            "capabilities": {
                "streaming": true,
                "tools": true,
                "vision": false,
                "thinking": true,
                "apis": ["chat_completions", "responses", "messages"],
            },
            // дефолты сэмплинга (если клиент не задаёт)
            "sampling_defaults": {
                "temperature": d.temperature,
                "top_p": d.top_p,
                "top_k": d.top_k,
                "min_p": d.min_p,
                "presence_penalty": d.presence_penalty,
                "repetition_penalty": d.repetition_penalty,
                "max_tokens": d.max_tokens,
            },
            // пресеты из model card (BD-016)
            "sampling_presets": {
                "thinking":         {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
                "thinking-coding":  {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
                "instruct":         {"temperature": 0.7, "top_p": 0.80, "top_k": 20, "presence_penalty": 1.5},
            },
        }],
    }))
    .into_response()
}
