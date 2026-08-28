pub mod admin;
pub mod anthropic;
pub mod hf;
pub mod proxy;
pub mod tools;
pub(crate) mod content;
pub mod media;
pub mod openai;
pub mod responses;

use axum::{
    body::Body,
    extract::{DefaultBodyLimit, Request, State},
    http::StatusCode,
    middleware::Next,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    Json, Router,
};
use serde::Serialize;
use std::convert::Infallible;
use std::sync::{Arc, RwLock};

use crate::engine_types::Engine;

#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<dyn Engine>,
    pub media: Arc<crate::media::MediaService>,
    pub api_keys: Arc<[crate::config::ApiKey]>,
    /// Тот же engine, но конкретный тип — для admin switch.
    pub switcher: Arc<crate::engine_swap::SwappableEngine>,
    /// Корень сканирования GGUF (QWEN36_MODELS_DIR или директория модели).
    pub models_dir: std::path::PathBuf,
    /// Validated profile metadata; None для legacy `QWEN36_MODEL`.
    pub profile: Option<Arc<crate::profile::ResolvedProfile>>,
    /// CUDA device handle (для mempool trim при switch; None на macOS/CPU).
    /// Мутабельно: unload_model обнуляет, чтобы CUDA context разрушился и VRAM
    /// освободилась. Без этого Device живёт весь процесс → VRAM не отпускается.
    pub cuda_device: Arc<RwLock<Option<candle_core::Device>>>,
    /// Прогресс фоновых HF-загрузок (repo/file → state).
    pub hf_downloads: hf::Downloads,
    /// Глобальные fallback/limit defaults из .env; режимные значения берутся
    /// из model-specific `presets`.
    pub sampling: Arc<RwLock<crate::config::SamplingDefaults>>,
    /// Effective presets активного семейства (built-in + MODEL_PRESETS).
    pub presets: Arc<RwLock<crate::config::SamplingPresets>>,
    /// Путь к .env для персистентности дефолтов.
    pub env_file: std::path::PathBuf,
}

#[derive(Debug, Clone)]
pub struct ApiKeyIdentity(pub crate::media::OwnerDigest);

#[derive(Serialize)]
pub struct ErrorBody {
    pub error: ErrorInner,
}

#[derive(Serialize)]
pub struct ErrorInner {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub message: String,
    /// Машинный код ошибки. Агенты опознают переполнение контекста именно по
    /// нему (`context_length_exceeded`), а не по тексту сообщения.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<&'static str>,
}

pub fn api_error(status: StatusCode, kind: &'static str, message: impl Into<String>) -> Response {
    (
        status,
        Json(ErrorBody {
            error: ErrorInner {
                kind,
                message: message.into(),
                code: None,
            },
        }),
    )
        .into_response()
}

pub fn bad_request(message: impl Into<String>) -> Response {
    api_error(StatusCode::BAD_REQUEST, "invalid_request_error", message)
}

pub fn internal_error(message: impl Into<String>) -> Response {
    api_error(StatusCode::INTERNAL_SERVER_ERROR, "internal_error", message)
}

pub fn media_error(error: crate::media::MediaError) -> Response {
    api_error(error.status(), error.api_kind(), error.safe_message())
}

fn api_key_matches(candidate: &str, configured: &str) -> bool {
    let candidate = candidate.as_bytes();
    let configured = configured.as_bytes();
    let mut diff = candidate.len() ^ configured.len();
    for idx in 0..candidate.len().max(configured.len()) {
        diff |= (candidate.get(idx).copied().unwrap_or(0)
            ^ configured.get(idx).copied().unwrap_or(0)) as usize;
    }
    diff == 0
}

async fn auth(State(state): State<AppState>, mut req: Request<Body>, next: Next) -> Response {
    let candidate = req
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));
    let identity = candidate.and_then(|key| {
        let mut matched = false;
        for entry in state.api_keys.iter() {
            matched |= api_key_matches(key, &entry.key);
        }
        matched.then(|| ApiKeyIdentity(crate::media::OwnerDigest::from_key(key)))
    });
    let Some(identity) = identity else {
        return api_error(
            StatusCode::UNAUTHORIZED,
            "authentication_error",
            "invalid or missing API key",
        );
    };
    req.extensions_mut().insert(identity);
    next.run(req).await
}

pub fn build_router(state: AppState) -> Router {
    // One video data URL plus bounded JSON overhead. Per-media encoded limits
    // still apply during typed source preparation.
    const MAX_V1_BODY: usize = 4 * 200 * 1024 * 1024 / 3 + 1024 * 1024;
    let v1 = Router::new()
        .route(
            "/chat/completions",
            axum::routing::post(openai::chat_completions),
        )
        .route("/responses", axum::routing::post(responses::responses))
        .route("/messages", axum::routing::post(anthropic::messages))
        .route("/media", axum::routing::post(media::upload))
        .route("/models", axum::routing::get(openai::list_models))
        .route(
            "/available_models",
            axum::routing::get(admin::available_models),
        )
        .route(
            "/model_native_ctx",
            axum::routing::get(admin::model_native_ctx),
        )
        .route("/ctx_matrix", axum::routing::get(admin::ctx_matrix))
        .route("/switch_model", axum::routing::post(admin::switch_model))
        .route("/unload_model", axum::routing::post(admin::unload_model))
        .route(
            "/sampling_defaults",
            axum::routing::post(admin::sampling_defaults),
        )
        .route(
            "/sampling_preset",
            axum::routing::post(admin::sampling_preset),
        )
        .route("/hf/search", axum::routing::get(hf::hf_search))
        .route("/hf/files", axum::routing::get(hf::hf_files))
        .route("/hf/probe", axum::routing::get(hf::hf_probe))
        .route("/hf/download", axum::routing::post(hf::hf_download))
        .route("/hf/downloads", axum::routing::get(hf::hf_downloads))
        .route_layer(axum::middleware::from_fn_with_state(state.clone(), auth))
        .layer(DefaultBodyLimit::max(MAX_V1_BODY));

    Router::new().nest("/v1", v1).with_state(state)
}

/// SSE-ответ из потока engine-событий. `map` превращает StreamEvent в 0+ SSE-событий.
pub fn sse_response<F>(
    rx: tokio::sync::mpsc::Receiver<crate::engine_types::StreamEvent>,
    cancel: crate::engine_types::CancelFlag,
    mut map: F,
) -> Response
where
    F: FnMut(crate::engine_types::StreamEvent, &mut Vec<Event>) -> bool + Send + 'static,
{
    let (tx, out_rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(32);
    tokio::spawn(async move {
        let guard = cancel.guard();
        let mut rx = rx;
        let mut buf: Vec<Event> = Vec::new();
        loop {
            // Ждём либо событие движка, либо обрыв клиента.
            //
            // Без второй ветки отмена не работала во время префила: задача
            // висела на recv(), событий в префиле нет, tx.send не вызывался —
            // и обрыв обнаруживался только когда пойдёт первый токен. На 18K
            // промпта это 14 секунд неостановимой работы видеокарты после
            // нажатия «стоп», на 128K — около минуты.
            //
            // Возврат без disarm роняет CancelOnDrop, тот взводит флаг,
            // и планировщик бросает префил на границе ближайшего чанка.
            let ev = tokio::select! {
                e = rx.recv() => match e {
                    Some(e) => e,
                    None => break,
                },
                _ = tx.closed() => return,
            };
            let cont = map(ev, &mut buf);
            for e in buf.drain(..) {
                if tx.send(Ok(e)).await.is_err() {
                    return;
                }
            }
            if !cont {
                break;
            }
        }
        guard.disarm();
    });
    let mut resp = Sse::new(tokio_stream::wrappers::ReceiverStream::new(out_rx))
        // Секунда вместо стандартных пятнадцати: keep-alive — единственная
        // запись в сокет, пока идёт префил, и только по её провалу hyper
        // узнаёт, что клиент отвалился. При 15 с обрыв во время префила
        // короче четверти минуты оставался незамеченным.
        .keep_alive(KeepAlive::new().interval(std::time::Duration::from_secs(1)))
        .into_response();
    resp.headers_mut().insert(
        axum::http::header::CONTENT_TYPE,
        axum::http::HeaderValue::from_static("text/event-stream; charset=utf-8"),
    );
    resp
}

/// Генерация с накоплением всего текста и финальных метрик.
pub struct GenOutcome {
    pub text: String,
    pub finish_reason: String,
    pub usage: crate::engine_types::GenerationUsage,
}

pub async fn prepare_inference_request(
    _state: &AppState,
    messages: Vec<crate::engine_types::ChatMessage>,
    params: crate::engine_types::GenParams,
    owner: &ApiKeyIdentity,
    tools: Option<serde_json::Value>,
    reasoning_effort: Option<String>,
) -> Result<crate::engine_types::InferenceRequest, Response> {
    Ok(inference_request(messages, params, owner, tools, reasoning_effort))
}

/// Запись запроса целиком (QWEN36_REQ_DEBUG=путь к файлу).
///
/// Нужна, когда дефект воспроизводится только в живой сессии агента и не
/// воспроизводится синтетическим запросом. Без самого запроса — истории,
/// параметров сэмплирования, набора инструментов — сбой остаётся
/// неповторимым, и любая версия причины непроверяема.
fn dump_request(
    messages: &[crate::engine_types::ChatMessage],
    params: &crate::engine_types::GenParams,
    tools: &Option<serde_json::Value>,
) {
    static PATH: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    let Some(path) = PATH.get_or_init(|| std::env::var("QWEN36_REQ_DEBUG").ok()) else {
        return;
    };
    let msgs: Vec<serde_json::Value> = messages
        .iter()
        .map(|m| {
            let text: String = m
                .content
                .iter()
                .filter_map(|b| match b {
                    crate::engine_types::ContentBlock::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(" ");
            // Хвост сообщения: по нему видно, чем закончился прошлый ответ
            // модели — именно там проявляется обрезание.
            let n = text.chars().count();
            let tail: String = text.chars().skip(n.saturating_sub(200)).collect();
            serde_json::json!({
                "role": m.role,
                "chars": n,
                "tool_calls": m.tool_calls.len(),
                "tail": tail,
            })
        })
        .collect();
    let dump = serde_json::json!({
        "temperature": params.temperature,
        "top_p": params.top_p,
        "top_k": params.top_k,
        "presence_penalty": params.presence_penalty,
        "repetition_penalty": params.repetition_penalty,
        "max_tokens": params.max_tokens,
        "thinking": params.thinking,
        "tools_count": tools.as_ref().and_then(|t| t.as_array()).map(|a| a.len()),
        "messages": msgs,
    });
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
        use std::io::Write;
        let _ = writeln!(f, "{dump}");
    }
}

pub fn inference_request(
    messages: Vec<crate::engine_types::ChatMessage>,
    params: crate::engine_types::GenParams,
    owner: &ApiKeyIdentity,
    tools: Option<serde_json::Value>,
    reasoning_effort: Option<String>,
) -> crate::engine_types::InferenceRequest {
    dump_request(&messages, &params, &tools);
    crate::engine_types::InferenceRequest {
        messages,
        params,
        owner: owner.0,
        cancel: Default::default(),
        tools,
        reasoning_effort,
    }
}

pub fn engine_error(error: anyhow::Error) -> Response {
    let error = match error.downcast::<crate::media::MediaError>() {
        Ok(error) => return media_error(error),
        Err(error) => error,
    };
    // Переполнение контекста — вина запроса, а не сервера: 400, не 500.
    match error.downcast::<crate::engine::ContextOverflow>() {
        Ok(overflow) => (
            StatusCode::BAD_REQUEST,
            Json(ErrorBody {
                error: ErrorInner {
                    kind: "invalid_request_error",
                    message: overflow.to_string(),
                    code: Some("context_length_exceeded"),
                },
            }),
        )
            .into_response(),
        Err(error) => internal_error(error.to_string()),
    }
}

pub async fn generate_collect(
    engine: &dyn Engine,
    request: crate::engine_types::InferenceRequest,
) -> Result<GenOutcome, Response> {
    let guard = request.cancel.guard();
    let mut rx = engine.generate(request).await.map_err(engine_error)?;
    let mut text = String::new();
    while let Some(ev) = rx.recv().await {
        match ev {
            crate::engine_types::StreamEvent::Delta { text: d, .. } => text.push_str(&d),
            crate::engine_types::StreamEvent::Done {
                finish_reason,
                usage,
            } => {
                guard.disarm();
                return Ok(GenOutcome {
                    text,
                    finish_reason,
                    usage,
                });
            }
            crate::engine_types::StreamEvent::Error(e) => {
                guard.disarm();
                return Err(internal_error(e));
            }
        }
    }
    Err(internal_error("stream ended without Done"))
}

/// Парсинг tool calls из накопленного текста. Форматы: JSON (Qwen3.5/3.6) и
/// Hermes <function=...> (Qwen3.8), с лечением битого JSON (порт Yttri-опыта).
/// Возвращает (текст без tool_call-блоков, список (name, arguments-json-string)).
pub fn parse_tool_calls(text: &str) -> (String, Vec<(String, String)>) {
    tools::parse_tool_calls(text)
}
