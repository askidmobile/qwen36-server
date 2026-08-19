use axum::{
    extract::{Extension, State},
    response::{sse::Event, IntoResponse, Response},
    Json,
};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::api::{
    bad_request, engine_error, generate_collect, parse_tool_calls, prepare_inference_request,
    sse_response, ApiKeyIdentity, AppState,
};
use crate::engine_types::{ChatMessage, ContentBlock, GenParams, StreamEvent};
use crate::media::MediaKind;

#[derive(Deserialize)]
pub struct ChatCompletionRequest {
    #[allow(dead_code)]
    model: Option<String>,
    messages: Vec<OaiMessage>,
    #[serde(default)]
    stream: bool,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    min_p: Option<f32>,
    presence_penalty: Option<f32>,
    repetition_penalty: Option<f32>,
    seed: Option<u64>,
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
    /// OpenAI-история: прошлые вызовы инструментов — рендерим в текст,
    /// чтобы модель видела собственные <tool_call> в контексте.
    tool_calls: Option<Value>,
}

fn parse_content(m: &OaiMessage) -> Result<Vec<ContentBlock>, Response> {
    match &m.content {
        None => Ok(vec![]),
        Some(OaiContent::Text(text)) => Ok(vec![super::content::text(text)]),
        Some(OaiContent::Parts(parts)) => parts
            .iter()
            .map(|part| match part.get("type").and_then(Value::as_str) {
                Some("text") => part
                    .get("text")
                    .and_then(Value::as_str)
                    .map(super::content::text)
                    .ok_or_else(|| bad_request("text block requires text")),
                Some("image_url") | Some("image") => {
                    let nested = part.get("image_url").unwrap_or(part);
                    super::content::media(
                        nested,
                        MediaKind::Image,
                        &["url"],
                        part.get("media_id").and_then(Value::as_str),
                    )
                }
                Some("video_url") | Some("video") => {
                    let nested = part.get("video_url").unwrap_or(part);
                    super::content::media(
                        nested,
                        MediaKind::Video,
                        &["url"],
                        part.get("media_id").and_then(Value::as_str),
                    )
                }
                _ => Err(bad_request("unsupported content block type")),
            })
            .collect(),
    }
}

fn to_gen_params(req: &ChatCompletionRequest, d: &crate::config::SamplingDefaults) -> GenParams {
    // База — дефолты из .env (QWEN36_*), не хардкод; поля запроса перекрывают.
    let mut p = GenParams {
        temperature: d.temperature,
        top_p: d.top_p,
        top_k: d.top_k,
        min_p: d.min_p,
        presence_penalty: d.presence_penalty,
        repetition_penalty: d.repetition_penalty,
        max_tokens: d.max_tokens,
        thinking: d.thinking,
        ..GenParams::default()
    };
    if let Some(t) = req.temperature {
        p.temperature = t;
    }
    if let Some(t) = req.top_p {
        p.top_p = t;
    }
    if let Some(k) = req.top_k {
        p.top_k = k;
    }
    if let Some(m) = req.min_p {
        p.min_p = m;
    }
    if let Some(penalty) = req.presence_penalty {
        p.presence_penalty = penalty;
    }
    if let Some(penalty) = req.repetition_penalty {
        p.repetition_penalty = penalty;
    }
    p.seed = req.seed;
    if let Some(m) = req.max_tokens {
        p.max_tokens = m;
    }
    if let Some(stop) = &req.stop {
        match stop {
            Value::String(s) => p.stop.push(s.clone()),
            Value::Array(arr) => p
                .stop
                .extend(arr.iter().filter_map(|v| v.as_str().map(String::from))),
            _ => {}
        }
    }
    // thinking: прямой флаг приоритетнее chat_template_kwargs; default из .env.
    let ctk = req
        .chat_template_kwargs
        .as_ref()
        .and_then(|k| k.get("enable_thinking"))
        .and_then(|v| v.as_bool());
    p.thinking = req.thinking.or(ctk).unwrap_or(d.thinking);
    p
}

fn build_messages(req: &ChatCompletionRequest) -> Result<Vec<ChatMessage>, Response> {
    req.messages
        .iter()
        .map(|m| {
            let content = parse_content(m)?;
            // tool_calls из истории — структурно; chat template сам рендерит
            // их в формате модели (Hermes для Qwen3.8, JSON для 3.5/3.6).
            let tool_calls = match &m.tool_calls {
                Some(Value::Array(calls)) => calls.clone(),
                _ => Vec::new(),
            };
            Ok(ChatMessage {
                role: m.role.clone(),
                content,
                tool_calls,
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
    Extension(owner): Extension<ApiKeyIdentity>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    let messages = match build_messages(&req) {
        Ok(m) => m,
        Err(r) => return r,
    };
    if messages
        .iter()
        .any(|message| message.role == "system" && message.has_media())
    {
        return bad_request("system messages cannot contain media");
    }
    let params = to_gen_params(&req, &state.sampling.read().expect("sampling lock"));
    if params.max_tokens == 0 {
        return bad_request("max_tokens must be greater than 0");
    }
    if params.temperature < 0.0 {
        return bad_request("temperature must be non-negative");
    }
    if !(0.0 < params.top_p && params.top_p <= 1.0) {
        return bad_request("top_p must be greater than 0 and at most 1");
    }
    if !(0.0..=1.0).contains(&params.min_p) {
        return bad_request("min_p must be between 0 and 1");
    }
    if !(-2.0..=2.0).contains(&params.presence_penalty) {
        return bad_request("presence_penalty must be between -2 and 2");
    }
    if params.repetition_penalty <= 0.0 {
        return bad_request("repetition_penalty must be greater than 0");
    }
    let include_usage = req
        .stream_options
        .as_ref()
        .and_then(|o| o.get("include_usage"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let request = match prepare_inference_request(&state, messages, params, &owner, req.tools.clone()).await {
        Ok(request) => request,
        Err(response) => return response,
    };
    if req.stream {
        // tools в запросе → буферизуем текст (иначе <tool_call> разметка
        // утекает в SSE-поток раньше парсинга — аудит 2026-08-10).
        let has_tools = req.tools.as_ref().map(|t| !t.is_null()).unwrap_or(false);
        return stream_chat(state, request, include_usage, has_tools).await;
    }

    let out = match generate_collect(state.engine.as_ref(), request).await {
        Ok(o) => o,
        Err(r) => return r,
    };
    // Отрезать мышление из content: thinking → message.reasoning_content.
    let (reasoning, text_body) = match out.text.find("</think>") {
        Some(pos) => {
            let r = out.text[..pos].trim().to_string();
            let t = out.text[pos + "</think>".len()..].trim_start().to_string();
            (if r.is_empty() { None } else { Some(r) }, t)
        }
        None => (None, out.text.clone()),
    };
    let (text, calls) = parse_tool_calls(&text_body);
    let finish = if !calls.is_empty() {
        "tool_calls"
    } else {
        out.finish_reason.as_str()
    };
    let mut message = json!({"role": "assistant", "content": if text.is_empty() { Value::Null } else { json!(text) }});
    if let Some(r) = reasoning {
        message["reasoning_content"] = json!(r);
    }
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
            "prompt_tokens": out.usage.prompt_tokens,
            "completion_tokens": out.usage.completion_tokens,
            "total_tokens": out.usage.prompt_tokens + out.usage.completion_tokens,
            "truncated": out.usage.truncated,
            "media": out.usage.media,
            "mtp": out.usage.mtp,
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
    request: crate::engine_types::InferenceRequest,
    include_usage: bool,
    has_tools: bool,
) -> Response {
    let id = format!("chatcmpl-{}", uuid::Uuid::new_v4().simple());
    let model = state.engine.model_info().id;
    let cancel = request.cancel.clone();
    // OpenAI-совместимый контракт мышления (vLLM/DeepSeek): thinking идёт в
    // delta.reasoning_content, ответ — в delta.content. Модель генерит
    // мышление + </think> + ответ в одном потоке — делим здесь.
    let thinking = request.params.thinking;
    let rx = match state.engine.generate(request).await {
        Ok(r) => r,
        Err(e) => return engine_error(e),
    };

    let mut first = true;
    let mut acc = String::new();
    // Состояние сплиттера: вся мысль копится в think_acc до закрывающего тега;
    // после него всё идёт в content. Хвост в 8 байт держим под частичный тег.
    let mut think_acc = String::new();
    let mut reasoning_done = !thinking;
    const THINK_CLOSE: &str = "</think>";
    sse_response(rx, cancel, move |ev, out| match ev {
        StreamEvent::Delta(d) => {
            if first {
                first = false;
                out.push(Event::default().data(chunk(
                    &id,
                    &model,
                    json!({"role": "assistant"}),
                    None,
                )));
            }
            // Сплит: мышление → reasoning_content, ответ → content/acc.
            let mut content_owned = String::new();
            if !reasoning_done {
                think_acc.push_str(&d);
                if let Some(pos) = think_acc.find(THINK_CLOSE) {
                    let reasoning = think_acc[..pos].trim().to_string();
                    content_owned = think_acc[pos + THINK_CLOSE.len()..].trim_start().to_string();
                    think_acc.clear();
                    if !reasoning.is_empty() {
                        out.push(Event::default().data(chunk(
                            &id,
                            &model,
                            json!({"reasoning_content": reasoning}),
                            None,
                        )));
                    }
                    reasoning_done = true;
                } else {
                    // Нет тега: эмитим всё, кроме хвоста под частичный `</think>`.
                    let safe = think_acc.len().saturating_sub(THINK_CLOSE.len());
                    let boundary = think_acc.floor_char_boundary(safe);
                    if boundary > 0 {
                        let reasoning = think_acc[..boundary].to_string();
                        think_acc.drain(..boundary);
                        out.push(Event::default().data(chunk(
                            &id,
                            &model,
                            json!({"reasoning_content": reasoning}),
                            None,
                        )));
                    }
                }
            } else {
                content_owned = d;
            }
            let content_part = content_owned.as_str();
            if !content_part.is_empty() {
                acc.push_str(content_part);
                if !has_tools {
                    out.push(Event::default().data(chunk(
                        &id,
                        &model,
                        json!({"content": content_part}),
                        None,
                    )));
                }
            }
            true
        }
        StreamEvent::Done {
            finish_reason,
            usage,
        } => {
            if first {
                first = false;
                out.push(Event::default().data(chunk(
                    &id,
                    &model,
                    json!({"role": "assistant"}),
                    None,
                )));
            }
            let (text, calls) = parse_tool_calls(&acc);
            // tools в запросе: текст буферизован — эмитим его (без tool_call
            // разметки) одной дельтой до tool_calls-чанков.
            if has_tools && !text.is_empty() {
                out.push(Event::default().data(chunk(&id, &model, json!({"content": text}), None)));
            }
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
            let finish = if calls.is_empty() {
                finish_reason.as_str()
            } else {
                "tool_calls"
            };
            out.push(Event::default().data(chunk(&id, &model, json!({}), Some(finish))));
            if include_usage {
                out.push(
                    Event::default().data(
                        json!({
                            "id": id, "object": "chat.completion.chunk",
                            "created": now_unix(), "model": model,
                            "choices": [],
                            "usage": {
                                "prompt_tokens": usage.prompt_tokens,
                                "completion_tokens": usage.completion_tokens,
                                "total_tokens": usage.prompt_tokens + usage.completion_tokens,
                                "truncated": usage.truncated,
                                "media": usage.media,
                                "mtp": usage.mtp,
                            },
                        })
                        .to_string(),
                    ),
                );
            }
            out.push(Event::default().data("[DONE]"));
            false
        }
        StreamEvent::Error(e) => {
            out.push(
                Event::default()
                    .data(json!({"error": {"type": "internal_error", "message": e}}).to_string()),
            );
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
    let d = state.sampling.read().expect("sampling lock").clone();
    let profile = state.profile.as_ref();
    let capabilities = profile.map(|profile| profile.capabilities()).unwrap_or(
        crate::profile::EffectiveCapabilities {
            text: true,
            vision: false,
            video: false,
            mtp: false,
            native_context: native_ctx,
        },
    );
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
            "native_context_length": capabilities.native_context,
            "file_size_mib": size_mib,
            "path": path.to_string_lossy(),
            // возможности для агентов
            "capabilities": {
                "streaming": true,
                "tools": true,
                "text": capabilities.text,
                "vision": capabilities.vision,
                "video": capabilities.video,
                "mtp": {"available": capabilities.mtp, "default_enabled": false},
                "thinking": true,
                "apis": ["chat_completions", "responses", "messages"],
            },
            // дефолты сэмплинга (если клиент не задаёт)
            "profile": profile.map(|profile| json!({
                "id": profile.manifest.profile_id,
                "release_version": profile.manifest.release_version,
                "source_revision": profile.manifest.source.revision,
                "text_quant": profile.manifest.artifacts.text.quant,
                "vision_quant": profile.manifest.artifacts.vision.as_ref().map(|artifact| &artifact.quant),
                "mtp_quant": profile.manifest.artifacts.mtp.as_ref().map(|artifact| &artifact.quant),
                "components": {"vision": profile.vision, "mtp": profile.mtp},
                "media_limits": profile.manifest.limits,
            })),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_sampling_extensions_are_applied() {
        let req: ChatCompletionRequest = serde_json::from_value(json!({
            "messages": [],
            "top_k": 7,
            "min_p": 0.1,
            "presence_penalty": 1.5,
            "repetition_penalty": 1.2,
            "seed": 42
        }))
        .unwrap();
        let params = to_gen_params(&req, &state.sampling.read().expect("sampling lock"));
        assert_eq!(params.top_k, 7);
        assert_eq!(params.min_p, 0.1);
        assert_eq!(params.presence_penalty, 1.5);
        assert_eq!(params.repetition_penalty, 1.2);
        assert_eq!(params.seed, Some(42));
    }
}
