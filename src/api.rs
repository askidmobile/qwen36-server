pub mod admin;
pub mod anthropic;
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
use std::sync::Arc;

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
    pub cuda_device: Option<candle_core::Device>,
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
}

pub fn api_error(status: StatusCode, kind: &'static str, message: impl Into<String>) -> Response {
    (
        status,
        Json(ErrorBody {
            error: ErrorInner {
                kind,
                message: message.into(),
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
            let ev = match rx.recv().await {
                Some(e) => e,
                None => break,
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
    Sse::new(tokio_stream::wrappers::ReceiverStream::new(out_rx))
        .keep_alive(KeepAlive::default())
        .into_response()
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
) -> Result<crate::engine_types::InferenceRequest, Response> {
    Ok(inference_request(messages, params, owner))
}

pub fn inference_request(
    messages: Vec<crate::engine_types::ChatMessage>,
    params: crate::engine_types::GenParams,
    owner: &ApiKeyIdentity,
) -> crate::engine_types::InferenceRequest {
    crate::engine_types::InferenceRequest {
        messages,
        params,
        owner: owner.0,
        cancel: Default::default(),
    }
}

pub fn engine_error(error: anyhow::Error) -> Response {
    match error.downcast::<crate::media::MediaError>() {
        Ok(error) => media_error(error),
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
            crate::engine_types::StreamEvent::Delta(d) => text.push_str(&d),
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

/// Парсинг <tool_call>{...}</tool_call> из накопленного текста.
/// Возвращает (текст без tool_call-блоков, список (name, arguments-json-string)).
pub fn parse_tool_calls(text: &str) -> (String, Vec<(String, String)>) {
    let mut calls = Vec::new();
    let mut rest = String::new();
    let mut s = text;
    loop {
        let Some(start) = s.find("<tool_call>") else {
            rest.push_str(s);
            break;
        };
        rest.push_str(&s[..start]);
        let after = &s[start + "<tool_call>".len()..];
        let Some(end) = after.find("</tool_call>") else {
            // незакрытый тег — оставляем как текст
            rest.push_str(&s[start..]);
            break;
        };
        let body = after[..end].trim();
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(body) {
            let name = v
                .get("name")
                .and_then(|n| n.as_str())
                .unwrap_or("")
                .to_string();
            let args = v
                .get("arguments")
                .map(|a| {
                    if a.is_string() {
                        a.as_str().unwrap().to_string()
                    } else {
                        a.to_string()
                    }
                })
                .unwrap_or_else(|| "{}".to_string());
            if !name.is_empty() {
                calls.push((name, args));
            }
        }
        s = &after[end + "</tool_call>".len()..];
    }
    (rest.trim().to_string(), calls)
}
