//! Reverse-proxy на Unsloth Studio backend.
//!
//! `/v1/*` остаётся на нашем движке; всё остальное проксируется на
//! `STUDIO_URL` (по умолчанию `http://127.0.0.1:8888`).
//! Если Studio недоступен — fallback на встроенный `web/index.html` для `GET /`.

use axum::{
    body::Body,
    extract::{Request, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    Router,
};
use std::sync::Arc;

/// Fallback HTML (встроенный веб-чат) когда Studio недоступен.
pub fn fallback_html() -> &'static str {
    include_str!("../../web/index.html")
}

/// Состояние proxy: reqwest-клиент + URL Studio backend.
#[derive(Clone)]
pub struct ProxyState {
    pub client: reqwest::Client,
    pub studio_url: String,
}

/// Catch-all handler: проксирует запрос на Studio backend.
/// При ошибке соединения и `GET /` — отдаёт fallback HTML.
pub async fn proxy_handler(
    State(state): State<Arc<ProxyState>>,
    req: Request,
) -> Response {
    let (parts, body_bytes) = match axum::body::to_bytes(req.into_body(), usize::MAX).await {
        Ok(b) => b,
        Err(_) => return (StatusCode::BAD_REQUEST, "body read error").into_response(),
    };
    let path = parts.uri.path().trim_start_matches('/');
    let query = parts.uri.query().map(|q| format!("?{q}")).unwrap_or_default();
    let url = format!("{}/{path}{query}", state.studio_url);

    let method = reqwest::Method::from_bytes(parts.method.as_str().as_bytes())
        .unwrap_or(reqwest::Method::GET);

    let mut fwd = state
        .client
        .request(method, &url)
        .body(body_bytes.to_vec());

    // Копируем заголовки кроме host — Studio backend его отвергает.
    for (name, value) in parts.headers.iter() {
        if name == "host" {
            continue;
        }
        fwd = fwd.header(name, value);
    }

    match fwd.send().await {
        Ok(resp) => {
            let status = StatusCode::from_u16(resp.status().as_u16())
                .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
            let headers = resp.headers().clone();
            let bytes = resp.bytes().await.unwrap_or_default();
            let mut builder = Response::builder().status(status);
            for (name, value) in headers.iter() {
                // axum выставит свои transfer-encoding/content-length
                if name == "transfer-encoding" || name == "content-length" {
                    continue;
                }
                builder = builder.header(name, value);
            }
            builder
                .body(Body::from(bytes))
                .unwrap_or_else(|_| {
                    (StatusCode::INTERNAL_SERVER_ERROR, "proxy build error").into_response()
                })
        }
        Err(_) => {
            // Studio недоступен — fallback на встроенный чат для GET /.
            if parts.method == "GET" && parts.uri.path() == "/" {
                (
                    StatusCode::OK,
                    [("content-type", "text/html; charset=utf-8")],
                    fallback_html(),
                )
                    .into_response()
            } else {
                (StatusCode::BAD_GATEWAY, "Unsloth Studio backend unavailable").into_response()
            }
        }
    }
}

/// Собирает proxy-роутер с catch-all fallback.
pub fn proxy_router(studio_url: String) -> Router {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(600))
        .build()
        .expect("proxy client build");
    let state = Arc::new(ProxyState { client, studio_url });
    Router::new()
        .fallback(proxy_handler)
        .with_state(state)
}

// ponytail: proxy не ходит в engine — ProxyState без api_keys.
// Если понадобится auth для proxy-эндпоинтов, добавить api_keys в ProxyState.