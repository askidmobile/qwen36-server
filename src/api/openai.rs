use axum::{
    extract::{Extension, State},
    http::StatusCode,
    response::{sse::Event, IntoResponse, Response},
    Json,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::atomic::Ordering;

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
    /// Уровень рассуждений: none (без), low (с рассуждениями), high (кодинг).
    /// Мапится на thinking-флаг + пресет сэмплинга + шаблонный reasoning_effort.
    reasoning_effort: Option<String>,
    /// Вернуть logprobs выбранных токенов.
    #[serde(default)]
    logprobs: bool,
    /// Сколько кандидатов вернуть, 0..20. Только вместе с logprobs: true.
    /// Знаковый нарочно: при usize отрицательное значение отсёк бы разбор JSON
    /// своим кодом и невнятным текстом, а по контракту нужен 400 с пояснением.
    top_logprobs: Option<i64>,
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
    /// reasoning_content из истории (preserve_thinking, Qwen3.8).
    reasoning_content: Option<String>,
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

fn to_gen_params(
    req: &ChatCompletionRequest,
    d: &crate::config::SamplingDefaults,
    presets: &crate::config::SamplingPresets,
) -> GenParams {
    // 1. Определяем флаг thinking и уровень рассуждений (reasoning_effort)
    let ctk = req
        .chat_template_kwargs
        .as_ref()
        .and_then(|k| k.get("enable_thinking"))
        .and_then(|v| v.as_bool());
    let effort = req.reasoning_effort.as_deref().unwrap_or("");
    let effort_thinking = match effort {
        "none" => Some(false),
        "low" | "medium" | "high" | "xhigh" => Some(true),
        _ => None,
    };
    let thinking = req
        .thinking
        .or(ctk)
        .or(effort_thinking)
        .unwrap_or(d.thinking);

    // 2. Выбираем базовый пресет сэмплинга в зависимости от режима:
    let preset_name = if thinking {
        if matches!(effort, "high" | "xhigh") {
            "thinking-coding"
        } else {
            "thinking"
        }
    } else {
        "instruct"
    };

    // 3. Базовые параметры: пресет модели или дефолты из .env
    let (mut temp, mut top_p, mut top_k, mut min_p, mut presence_p, mut rep_p) =
        if let Some(pv) = presets.get(preset_name) {
            (
                pv.temperature,
                pv.top_p,
                pv.top_k,
                pv.min_p,
                pv.presence_penalty,
                pv.repetition_penalty,
            )
        } else {
            (
                d.temperature,
                d.top_p,
                d.top_k,
                d.min_p,
                d.presence_penalty,
                d.repetition_penalty,
            )
        };

    // 4. НАИВЫСШИЙ ПРИОРИТЕТ: явные параметры из запроса клиента (pi, WebUI, curl)
    if let Some(t) = req.temperature {
        temp = t;
    }
    if let Some(t) = req.top_p {
        top_p = t;
    }
    if let Some(k) = req.top_k {
        top_k = k;
    }
    if let Some(m) = req.min_p {
        min_p = m;
    }
    if let Some(penalty) = req.presence_penalty {
        presence_p = penalty;
    }
    if let Some(penalty) = req.repetition_penalty {
        rep_p = penalty;
    }

    let mut stop_list = Vec::new();
    if let Some(stop) = &req.stop {
        match stop {
            Value::String(s) => stop_list.push(s.clone()),
            Value::Array(arr) => {
                stop_list.extend(arr.iter().filter_map(|v| v.as_str().map(String::from)))
            }
            _ => {}
        }
    }

    GenParams {
        temperature: temp,
        top_p,
        top_k,
        min_p,
        presence_penalty: presence_p,
        repetition_penalty: rep_p,
        max_tokens: req.max_tokens.unwrap_or(d.max_tokens),
        stop: stop_list,
        seed: req.seed,
        thinking,
        // Проверку валидности делает validate_logprobs до вызова: сюда
        // приходит уже согласованная пара.
        logprobs: req
            .logprobs
            .then(|| req.top_logprobs.unwrap_or(0).max(0) as usize),
    }
}

/// Проверка пары logprobs/top_logprobs по контракту OpenAI.
///
/// Отдельной функцией, а не внутри to_gen_params: та не умеет возвращать
/// ошибку, а клиенту нужен 400 с внятным текстом, как при переполнении
/// контекста, а не молчаливое игнорирование поля.
fn validate_logprobs(req: &ChatCompletionRequest) -> Result<(), Response> {
    match (req.logprobs, req.top_logprobs) {
        (false, Some(n)) => Err(crate::api::bad_request(format!(
            "top_logprobs={n} без logprobs: true — поле игнорировать нельзя, \
             укажите logprobs: true или уберите top_logprobs"
        ))),
        (true, Some(n)) if !(0..=20).contains(&n) => Err(crate::api::bad_request(format!(
            "top_logprobs={n} вне диапазона 0..20 — предел контракта OpenAI"
        ))),
        _ => Ok(()),
    }
}

fn build_messages(req: &ChatCompletionRequest) -> Result<Vec<ChatMessage>, Response> {
    req.messages
        .iter()
        .map(|m| {
            let content = parse_content(m)?;
            // tool_calls из истории — структурно; chat template сам рендерит
            // их в формате модели (Hermes для Qwen3.8, JSON для 3.5/3.6).
            // Аргументы приводим к объекту: по контракту OpenAI клиент шлёт
            // function.arguments СТРОКОЙ JSON, а шаблон Qwen3.8 ждёт объект и
            // иначе падает с «arguments were passed as a JSON string».
            let tool_calls = match &m.tool_calls {
                Some(Value::Array(calls)) => calls.iter().map(normalize_tool_call).collect(),
                _ => Vec::new(),
            };
            Ok(ChatMessage {
                role: m.role.clone(),
                content,
                tool_calls,
                reasoning_content: m.reasoning_content.clone(),
            })
        })
        .collect()
}

/// Привести один tool_call к виду, который понимает chat template.
///
/// По контракту OpenAI `function.arguments` — это СТРОКА с JSON, и все клиенты
/// шлют именно так. Шаблон Qwen3.8 (Hermes) обращается к аргументам как к
/// объекту и на строке падает: «Tool call arguments for function ... were
/// passed as a JSON string». Разбираем строку в объект; если разобрать нельзя
/// (клиент прислал мусор), оставляем как есть — пусть шаблон решает сам,
/// молча терять данные хуже.
fn normalize_tool_call(call: &Value) -> Value {
    let mut call = call.clone();
    let Some(args) = call.pointer("/function/arguments") else {
        return call;
    };
    let Some(text) = args.as_str() else {
        return call; // уже объект — ничего не делаем
    };
    let parsed = if text.trim().is_empty() {
        Some(Value::Object(serde_json::Map::new()))
    } else {
        serde_json::from_str::<Value>(text).ok()
    };
    if let Some(v) = parsed {
        if let Some(slot) = call.pointer_mut("/function/arguments") {
            *slot = v;
        }
    }
    call
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
    // Авто-переключение модели: если req.model не совпадает с текущей —
    // ищем GGUF по имени в available_models и запускаем switch_model.
    let current_id = state.engine.model_info().id;
    let requested = req.model.as_deref().unwrap_or("").trim().to_lowercase();
    if !requested.is_empty() && requested != current_id.to_lowercase() {
        // Ищем matching GGUF в models_dir.
        let mut found: Vec<Value> = Vec::new();
        crate::api::admin::scan_gguf_cached(&state.models_dir, 0, &mut found, true);
        let match_path = found.iter().find_map(|m| {
            let path = m["path"].as_str()?;
            // Сравниваем по model_id_from_filename (нормализованное имя).
            let file_id = crate::engine::model_id_from_filename(std::path::Path::new(path));
            if file_id == requested {
                Some(path.to_string())
            } else {
                None
            }
        });
        if let Some(path) = match_path {
            // Проверяем, не идёт ли уже переключение — если да, НЕ запускаем
            // новое (предотвращает гонки: pi шлёт несколько запросов → OOM).
            if state.switcher.loading.swap(true, Ordering::Relaxed) {
                // Уже идёт загрузка — просто возвращаем 503.
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    Json(json!({
                        "error": {
                            "type": "model_switching",
                            "message": format!("Model switch in progress, please retry in a few seconds"),
                        }
                    })),
                ).into_response();
            }
            // Текущие ctx/slots из state.switcher.current.
            let (cur_ctx, cur_slots) = state
                .switcher
                .current
                .read()
                .map(|current| (current.1, current.2))
                .unwrap_or((8192, 4));
            eprintln!(
                "[auto-switch] req.model='{}' current='{}' → switching to {}",
                requested, current_id, path
            );
            // Запускаем switch в фоне.
            let state2 = state.clone();
            tokio::spawn(async move {
                let result = crate::api::admin::do_switch(
                    &state2,
                    std::path::PathBuf::from(&path),
                    cur_ctx,
                    cur_slots,
                )
                .await;
                if let Err(e) = result {
                    eprintln!("[auto-switch] FAILED: {e:#}");
                }
                state2.switcher.loading.store(false, Ordering::Relaxed);
            });
            // Возвращаем 503 с понятным сообщением — клиент повторит.
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({
                    "error": {
                        "type": "model_switching",
                        "message": format!("Switching to model '{}', please retry in a few seconds", requested),
                    }
                })),
            ).into_response();
        }
        // Не нашли matching GGUF — продолжаем на текущей модели.
    }
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
    if let Err(response) = validate_logprobs(&req) {
        return response;
    }
    let params = to_gen_params(
        &req,
        &state.sampling.read().expect("sampling lock"),
        &state.presets.read().expect("presets lock"),
    );
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

    let request = match prepare_inference_request(
        &state,
        messages,
        params,
        &owner,
        req.tools.clone(),
        req.reasoning_effort.clone(),
    )
    .await
    {
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
    // Отрезать Qwen </think> и Gemma 4 thought-channel из content.
    let (reasoning, text_body) = split_reasoning(&out.text);
    let (text, calls) = parse_tool_calls(&text_body);
    // Gemma fallback: если content пустой, но reasoning есть —
    // переносим reasoning в content, чтобы пользователь не получил пустой ответ
    // (модель часто пишет весь ответ внутри reasoning без перехода к <|channel>final).
    let is_gemma = state.engine.model_info().id.starts_with("gemma-4");
    let (final_content, final_reasoning) = if text.is_empty() && reasoning.is_some() && is_gemma {
        (reasoning.clone(), None)
    } else {
        ((!text.is_empty()).then_some(text), reasoning)
    };
    let finish = if !calls.is_empty() {
        "tool_calls"
    } else {
        out.finish_reason.as_str()
    };
    let mut message = json!({"role": "assistant", "content": final_content.map(Value::String).unwrap_or(Value::Null)});
    if let Some(r) = final_reasoning {
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
            // По контракту поле присутствует всегда: null, если не просили.
            // refusal внутри — тоже часть формата, у нас отказов нет.
            "logprobs": out.logprobs.as_ref().map(|entries| json!({
                "content": entries.iter().map(logprob_json).collect::<Vec<_>>(),
                "refusal": Value::Null,
            })),
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

/// Одна запись logprobs в вид OpenAI.
///
/// `bytes` отдаём массивом чисел, как в контракте: для байтовых и неполных
/// UTF-8 токенов строка содержит U+FFFD, а байты остаются настоящими, и
/// клиент по ним восстановит границы.
fn logprob_json(entry: &crate::engine_types::TokenLogprob) -> Value {
    json!({
        "token": entry.token,
        "logprob": entry.logprob,
        "bytes": entry.bytes,
        "top_logprobs": entry.top.iter().map(|t| json!({
            "token": t.token,
            "logprob": t.logprob,
            "bytes": t.bytes,
        })).collect::<Vec<_>>(),
    })
}

pub fn extract_gemma_fallback_content(reasoning: &str) -> Option<String> {
    let text = reasoning.trim();
    if text.is_empty() {
        return None;
    }
    // Маркеры финального ответа в CoT Gemma 4
    let markers = [
        "**Final Output Generation:**",
        "**Final Output Generation.**",
        "**Final Output:**",
        "**Output Generation:**",
        "Final Output Generation:",
        "Final Output Generation.",
        "Final Output:",
        "Final Answer Generation:",
        "Final Answer Construction",
        "Final Answer:",
        "Select the best option:",
        "Select the final response:",
        "Output:",
    ];
    for m in markers {
        if let Some(pos) = text.rfind(m) {
            let tail = text[pos + m.len()..].trim();
            // Отрезаем скобки типа (In Russian): или (Privet):
            let candidate = tail.lines().next().unwrap_or(tail).trim();
            let candidate = candidate.trim_matches(['*', '_', ' ']);
            // Если ответ в кавычках типа "Привет! Как дела?", извлекаем из кавычек
            if let Some(start) = candidate.find('"') {
                if let Some(end) = candidate[start + 1..].find('"') {
                    let inside = &candidate[start + 1..start + 1 + end];
                    if !inside.trim().is_empty() {
                        return Some(inside.trim().to_string());
                    }
                }
            }
            if !candidate.is_empty() && !candidate.starts_with('(') {
                return Some(candidate.to_string());
            }
        }
    }
    // Fallback: последняя строка в кавычках в тексте мыслей
    if let Some(last_quote_start) = text.rfind('"') {
        if let Some(prev_quote) = text[..last_quote_start].rfind('"') {
            let inside = &text[prev_quote + 1..last_quote_start];
            // Проверяем, что внутри есть буквы и это осмысленный ответ
            if inside.chars().any(char::is_alphabetic) && inside.len() < 500 && !inside.contains('\n') {
                return Some(inside.trim().to_string());
            }
        }
    }
    None
}

fn split_reasoning(text: &str) -> (Option<String>, String) {
    if let Some(close) = text.find("</think>") {
        let reasoning = text[..close].trim().to_string();
        let content = text[close + "</think>".len()..].trim_start().to_string();
        return ((!reasoning.is_empty()).then_some(reasoning), content);
    }
    const GEMMA_OPEN: &str = "<|channel>thought";
    const GEMMA_CLOSE: &str = "<channel|>";
    if let Some(open) = text.rfind(GEMMA_OPEN) {
        if let Some(relative_close) = text[open + GEMMA_OPEN.len()..].find(GEMMA_CLOSE) {
            let close = open + GEMMA_OPEN.len() + relative_close;
            let reasoning = text[open + GEMMA_OPEN.len()..close]
                .trim_start_matches(['\r', '\n'])
                .trim()
                .to_string();
            let tail = text[close + GEMMA_CLOSE.len()..].trim_start();
            let content = tail
                .strip_prefix("<|channel>final")
                .unwrap_or(tail)
                .trim_start_matches(['\r', '\n'])
                .to_string();
            return ((!reasoning.is_empty()).then_some(reasoning), content);
        }
        let reasoning = text[open + GEMMA_OPEN.len()..]
            .trim_start_matches(['\r', '\n'])
            .trim()
            .to_string();
        let fallback_content = extract_gemma_fallback_content(&reasoning)
            .unwrap_or_else(|| reasoning.clone());
        if fallback_content == reasoning {
            return (None, reasoning);
        } else {
            return ((!reasoning.is_empty()).then_some(reasoning), fallback_content);
        }
    }
    (None, text.to_string())
}

#[cfg(test)]
mod reasoning_tests {
    use super::split_reasoning;

    #[test]
    fn splits_gemma_thought_and_final_channels() {
        assert_eq!(
            split_reasoning("<|channel>thought\nwork<channel|>\n<|channel>final\nanswer"),
            (Some("work".into()), "answer".into())
        );
        assert_eq!(
            // Незавершённый thought без final-канала: fallback-extractor
            // (09d0ad0) не может уверенно отделить ответ — текст идёт в
            // content целиком, reasoning-канал не открывается.
            split_reasoning("<|channel>thought\nunfinished"),
            (None, "unfinished".to_string())
        );
    }
}

pub(crate) fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn chunk(id: &str, model: &str, delta: Value, finish: Option<&str>) -> String {
    chunk_with(id, model, delta, finish, None)
}

/// Чанк с необязательным полем logprobs.
///
/// Отдельной функцией, а не пятым параметром `chunk`: вызовов у неё
/// восемнадцать, и добавлять всем `None` ради одного места — шум в диффе.
fn chunk_with(
    id: &str,
    model: &str,
    delta: Value,
    finish: Option<&str>,
    logprobs: Option<Value>,
) -> String {
    let mut choice = json!({"index": 0, "delta": delta, "finish_reason": finish});
    if let Some(lp) = logprobs {
        choice["logprobs"] = lp;
    }
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": now_unix(),
        "model": model,
        "choices": [choice],
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
    let gemma_channel = model.starts_with("gemma-4");
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
    // Что-то из размышления уже ушло клиенту: тогда начало последнего куска
    // обрезать нельзя — оно продолжает уже отправленный текст. Раньше .trim()
    // на последнем куске срезал ведущий пробел удержанного хвоста (≤8 байт под
    // частичный тег): «Let's» + « craft.» уходило как «Let'scraft.».
    let mut reasoning_emitted = false;
    // Контент после </think>: ведущие пробелы/переводы строки режем до первого
    // непустого куска — как split_reasoning в non-stream (trim_start), иначе
    // поток отдаёт лишний «\n\n» на стыке.
    let mut content_started = !thinking;
    const THINK_CLOSE: &str = "</think>";
    // Gemma streaming splitter: теги <|channel>thought ... <channel|> ... <|channel>final ...
    // Эмитим reasoning_content инкрементально, content — после <|channel>final.
    const GEMMA_THOUGHT_OPEN: &str = "<|channel>thought";
    const GEMMA_CLOSE: &str = "<channel|>";
    const GEMMA_FINAL_OPEN: &str = "<|channel>final";
    let mut gemma_buf = String::new();
    // 0=ищем thought-open, 1=в thought (до close), 2=ищем final-open, 3=в final
    let mut gemma_phase: u8 = 0;
    sse_response(rx, cancel, move |ev, out| match ev {
        StreamEvent::Delta { text: d, logprobs: lp } => {
            if gemma_channel {
                if first {
                    first = false;
                    out.push(Event::default().data(chunk(
                        &id,
                        &model,
                        json!({"role": "assistant"}),
                        None,
                    )));
                }
                acc.push_str(&d);
                gemma_buf.push_str(&d);
                loop {
                    match gemma_phase {
                        0 => {
                            // Ищем <|channel>thought. Если нет — возможно prefix мусор;
                            // держим хвост под частичный тег.
                            if let Some(p) = gemma_buf.find(GEMMA_THOUGHT_OPEN) {
                                gemma_buf.drain(..p + GEMMA_THOUGHT_OPEN.len());
                                gemma_phase = 1;
                            } else {
                                let safe = gemma_buf.len().saturating_sub(GEMMA_THOUGHT_OPEN.len());
                                let boundary = gemma_buf.floor_char_boundary(safe);
                                // prefix без thought-open: эмитим как content
                                if boundary > 0 {
                                    let pre = gemma_buf[..boundary].to_string();
                                    gemma_buf.drain(..boundary);
                                    if !pre.trim().is_empty() {
                                        out.push(Event::default().data(chunk(
                                            &id,
                                            &model,
                                            json!({"content": pre}),
                                            None,
                                        )));
                                    }
                                }
                                break;
                            }
                        }
                        1 => {
                            // В thought: ищем <channel|>. Эмитим reasoning_content,
                            // держим хвост под частичный тег.
                            if let Some(p) = gemma_buf.find(GEMMA_CLOSE) {
                                let reasoning = gemma_buf[..p].to_string();
                                gemma_buf.drain(..p + GEMMA_CLOSE.len());
                                if !reasoning.trim().is_empty() {
                                    out.push(Event::default().data(chunk(
                                        &id,
                                        &model,
                                        json!({"reasoning_content": reasoning}),
                                        None,
                                    )));
                                }
                                gemma_phase = 2;
                            } else {
                                let safe = gemma_buf.len().saturating_sub(GEMMA_CLOSE.len());
                                let boundary = gemma_buf.floor_char_boundary(safe);
                                if boundary > 0 {
                                    let reasoning = gemma_buf[..boundary].to_string();
                                    gemma_buf.drain(..boundary);
                                    if !reasoning.is_empty() {
                                        out.push(Event::default().data(chunk(
                                            &id,
                                            &model,
                                            json!({"reasoning_content": reasoning}),
                                            None,
                                        )));
                                    }
                                }
                                break;
                            }
                        }
                        2 => {
                            // Ищем <|channel>final или переход к прямому тексту после <channel|>
                            if let Some(p) = gemma_buf.find(GEMMA_FINAL_OPEN) {
                                gemma_buf.drain(..p + GEMMA_FINAL_OPEN.len());
                                while let Some(c) = gemma_buf.chars().next() {
                                    if c == '\n' || c == '\r' || c == ' ' { gemma_buf.drain(..c.len_utf8()); } else { break; }
                                }
                                gemma_phase = 3;
                            } else if gemma_buf.starts_with("<|channel>") || GEMMA_FINAL_OPEN.starts_with(&gemma_buf) {
                                // Частичный тег <|channel>final — ждем продолжения
                                break;
                            } else if !gemma_buf.trim().is_empty() && !gemma_buf.contains('<') {
                                // Модель не выделила тег final, а сразу начала отвечать текстом
                                while let Some(c) = gemma_buf.chars().next() {
                                    if c == '\n' || c == '\r' || c == ' ' { gemma_buf.drain(..c.len_utf8()); } else { break; }
                                }
                                gemma_phase = 3;
                            } else {
                                break;
                            }
                        }
                        3 => {
                            // В final: эмитим всё как content.
                            if !gemma_buf.is_empty() {
                                let c = std::mem::take(&mut gemma_buf);
                                if !c.is_empty() {
                                    out.push(Event::default().data(chunk(
                                        &id,
                                        &model,
                                        json!({"content": c}),
                                        None,
                                    )));
                                }
                            }
                            break;
                        }
                        _ => break,
                    }
                }
                return true;
            }
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
                    let last = &think_acc[..pos];
                    let reasoning = if reasoning_emitted {
                        last.trim_end().to_string()
                    } else {
                        last.trim().to_string()
                    };
                    content_owned = think_acc[pos + THINK_CLOSE.len()..].to_string();
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
                        let reasoning = if reasoning_emitted {
                            think_acc[..boundary].to_string()
                        } else {
                            think_acc[..boundary].trim_start().to_string()
                        };
                        think_acc.drain(..boundary);
                        if !reasoning.is_empty() {
                            reasoning_emitted = true;
                            out.push(Event::default().data(chunk(
                                &id,
                                &model,
                                json!({"reasoning_content": reasoning}),
                                None,
                            )));
                        }
                    }
                }
            } else {
                content_owned = d;
            }
            if !content_started {
                let trimmed = content_owned.trim_start();
                if trimmed.is_empty() {
                    content_owned.clear();
                } else {
                    content_owned = trimmed.to_string();
                    content_started = true;
                }
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
            // Записи по токенам этого шага — отдельным чанком с пустым
            // содержимым. Пустая строка, а не null: некоторые клиенты падают
            // на null там, где ждут строку.
            //
            // Отдельный чанк, а не поле у текстового: наш сплиттер режет поток
            // на размышление и ответ ПО ТЕКСТУ и придерживает недособранные
            // UTF-8 хвосты, поэтому у одной дельты бывает и ноль текстовых
            // чанков, и несколько. Привязать записи к одному из них без
            // натяжки нельзя, а фаза при этом уже определена точно — по
            // идентификатору токена </think> в движке.
            if let Some(entries) = lp {
                if !entries.is_empty() {
                    out.push(Event::default().data(chunk_with(
                        &id,
                        &model,
                        json!({"content": ""}),
                        None,
                        Some(json!({
                            "content": entries.iter().map(logprob_json).collect::<Vec<_>>(),
                            "refusal": Value::Null,
                        })),
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
            let finish = if gemma_channel {
                // Добиваем хвост gemma_buf по текущей фазе.
                match gemma_phase {
                    0 => {
                        let c = gemma_buf.trim().to_string();
                        if !c.is_empty() {
                            out.push(Event::default().data(chunk(
                                &id,
                                &model,
                                json!({"content": c}),
                                None,
                            )));
                        }
                    }
                    1 => {
                        let r = gemma_buf.trim().to_string();
                        if !r.is_empty() {
                            out.push(Event::default().data(chunk(
                                &id,
                                &model,
                                json!({"reasoning_content": r}),
                                None,
                            )));
                        }
                        // Если за весь стрим не было отправлено ни одного байта контента:
                        // извлекаем ответ или отдаем накопленный текст как content!
                        let fallback = extract_gemma_fallback_content(&acc)
                            .unwrap_or_else(|| acc.clone());
                        let fallback = fallback.trim().to_string();
                        if !fallback.is_empty() {
                            out.push(Event::default().data(chunk(
                                &id,
                                &model,
                                json!({"content": fallback}),
                                None,
                            )));
                        }
                    }
                    2 | 3 => {
                        let c = gemma_buf.trim().to_string();
                        if !c.is_empty() {
                            out.push(Event::default().data(chunk(
                                &id,
                                &model,
                                json!({"content": c}),
                                None,
                            )));
                        }
                    }
                    _ => {}
                }
                finish_reason.as_str()
            } else {
                let (reasoning, text_body) = split_reasoning(&acc);
                if let Some(reasoning) = reasoning {
                    out.push(Event::default().data(chunk(
                        &id,
                        &model,
                        json!({"reasoning_content": reasoning}),
                        None,
                    )));
                }
                let (text, calls) = parse_tool_calls(&text_body);
                // tools в запросе: текст буферизован — эмитим его (без tool_call
                // разметки) одной дельтой до tool_calls-чанков.
                if (has_tools || gemma_channel) && !text.is_empty() {
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
                if calls.is_empty() {
                    finish_reason.as_str()
                } else {
                    "tool_calls"
                }
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
            crate::vram_plan::footprint_from_gguf_cached(&p)
                .map(|fp| (fp.native_ctx, fp.weights_mib))
                .unwrap_or((0, 0))
        }
    })
    .await
    .unwrap_or((0, 0));
    let d = state.sampling.read().expect("sampling lock").clone();
    let presets = state.presets.read().expect("presets lock").clone();
    let default_mode = if d.thinking { "thinking" } else { "instruct" };
    let effective_default = presets
        .get(default_mode)
        .cloned()
        .unwrap_or(crate::config::SamplingPresetValues {
            temperature: d.temperature,
            top_p: d.top_p,
            top_k: d.top_k,
            min_p: d.min_p,
            presence_penalty: d.presence_penalty,
            repetition_penalty: d.repetition_penalty,
        });
    let model_name = path
        .file_name()
        .map(|name| name.to_string_lossy())
        .unwrap_or_default();
    let sampling_family = crate::config::sampling_family_for_model(&model_name);
    let profile = state.profile.as_ref();
    let mut capabilities = profile.map(|profile| profile.capabilities()).unwrap_or(
        crate::profile::EffectiveCapabilities {
            text: true,
            vision: false,
            video: false,
            mtp: false,
            native_context: native_ctx,
        },
    );
    capabilities.vision &= state.engine.supports_vision();
    capabilities.video &= state.engine.supports_video();
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
            // Effective defaults: ровно тот пресет, который использует API,
            // если клиент не передал параметры явно.
            "sampling_family": sampling_family,
            "sampling_default_mode": default_mode,
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
                "temperature": effective_default.temperature,
                "top_p": effective_default.top_p,
                "top_k": effective_default.top_k,
                "min_p": effective_default.min_p,
                "presence_penalty": effective_default.presence_penalty,
                "repetition_penalty": effective_default.repetition_penalty,
                "max_tokens": d.max_tokens,
            },
            // пресеты из model card (BD-016)
            "sampling_presets": presets,
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
        let defaults = crate::config::SamplingDefaults::default();
        let presets = crate::config::default_presets();
        let params = to_gen_params(
            &req,
            &defaults,
            &presets,
        );
        assert_eq!(params.top_k, 7);
        assert_eq!(params.min_p, 0.1);
        assert_eq!(params.presence_penalty, 1.5);
        assert_eq!(params.repetition_penalty, 1.2);
        assert_eq!(params.seed, Some(42));
    }

    #[test]
    fn gemma_uses_its_model_preset_when_request_has_no_sampling_overrides() {
        let req: ChatCompletionRequest = serde_json::from_value(json!({
            "messages": [],
            "chat_template_kwargs": {"enable_thinking": false}
        }))
        .unwrap();
        let defaults = crate::config::SamplingDefaults::default();
        let presets = crate::config::default_presets_for_model("gemma-4-E4B-it-Q8_0.gguf");
        let params = to_gen_params(&req, &defaults, &presets);

        assert_eq!(params.temperature, 1.0);
        assert_eq!(params.top_p, 0.95);
        assert_eq!(params.top_k, 64);
        assert_eq!(params.presence_penalty, 0.0);
        assert_eq!(params.repetition_penalty, 1.0);
    }
}
