pub mod admin;
pub mod anthropic;
pub mod openai;
pub mod responses;

use axum::{
    body::Body,
    extract::{Request, State},
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
    pub api_key: String,
    /// Тот же engine, но конкретный тип — для admin switch.
    pub switcher: Arc<crate::engine_swap::SwappableEngine>,
    /// Корень сканирования GGUF (QWEN36_MODELS_DIR или директория модели).
    pub models_dir: std::path::PathBuf,
    /// CUDA device handle (для mempool trim при switch; None на macOS/CPU).
    pub cuda_device: Option<candle_core::Device>,
}

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

async fn auth(State(state): State<AppState>, req: Request<Body>, next: Next) -> Response {
    let ok = req
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(|k| k == state.api_key)
        .unwrap_or(false);
    if !ok {
        return api_error(
            StatusCode::UNAUTHORIZED,
            "authentication_error",
            "invalid or missing API key",
        );
    }
    next.run(req).await
}

pub fn build_router(state: AppState) -> Router {
    let v1 = Router::new()
        .route("/chat/completions", axum::routing::post(openai::chat_completions))
        .route("/responses", axum::routing::post(responses::responses))
        .route("/messages", axum::routing::post(anthropic::messages))
        .route("/models", axum::routing::get(openai::list_models))
        .route("/available_models", axum::routing::get(admin::available_models))
        .route("/model_native_ctx", axum::routing::get(admin::model_native_ctx))
        .route("/ctx_matrix", axum::routing::get(admin::ctx_matrix))
        .route("/switch_model", axum::routing::post(admin::switch_model))
        .route_layer(axum::middleware::from_fn_with_state(state.clone(), auth));

    Router::new().nest("/v1", v1).with_state(state)
}

/// SSE-ответ из потока engine-событий. `map` превращает StreamEvent в 0+ SSE-событий.
pub fn sse_response<F>(
    rx: tokio::sync::mpsc::Receiver<crate::engine_types::StreamEvent>,
    mut map: F,
) -> Response
where
    F: FnMut(crate::engine_types::StreamEvent, &mut Vec<Event>) -> bool + Send + 'static,
{
    let (tx, out_rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(32);
    tokio::spawn(async move {
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
    });
    Sse::new(tokio_stream::wrappers::ReceiverStream::new(out_rx))
        .keep_alive(KeepAlive::default())
        .into_response()
}

/// Генерация с накоплением всего текста и финальных метрик.
pub struct GenOutcome {
    pub text: String,
    pub finish_reason: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub truncated: bool,
}

pub async fn generate_collect(
    engine: &dyn Engine,
    messages: Vec<crate::engine_types::ChatMessage>,
    params: crate::engine_types::GenParams,
) -> Result<GenOutcome, Response> {
    let mut rx = engine
        .generate(messages, params)
        .await
        .map_err(|e| internal_error(e.to_string()))?;
    let mut text = String::new();
    while let Some(ev) = rx.recv().await {
        match ev {
            crate::engine_types::StreamEvent::Delta(d) => text.push_str(&d),
            crate::engine_types::StreamEvent::Done {
                finish_reason,
                prompt_tokens,
                completion_tokens,
                truncated,
            } => {
                return Ok(GenOutcome {
                    text,
                    finish_reason,
                    prompt_tokens,
                    completion_tokens,
                    truncated,
                })
            }
            crate::engine_types::StreamEvent::Error(e) => return Err(internal_error(e)),
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
