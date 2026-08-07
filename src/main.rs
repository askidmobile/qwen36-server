//! Бинарь qwen36-server: env QWEN36_* → CandleEngine → HTTP (три API + веб-чат).

use anyhow::Result;
use axum::{response::Html, routing::get, Router};
use qwen36_server::{
    api::{build_router, AppState},
    config::Config,
    engine::{CandleEngine, Engine},
};
use std::sync::Arc;

const CHAT_HTML: &str = include_str!("../web/index.html");

#[tokio::main]
async fn main() -> Result<()> {
    let cfg = Config::from_env()?;
    eprintln!(
        "[qwen36] model={:?} ctx={} slots={} listen={}:{}",
        cfg.model, cfg.ctx, cfg.slots, cfg.host, cfg.port
    );

    let engine: Arc<dyn Engine> = Arc::new(CandleEngine::load(&cfg)?);
    let info = engine.model_info();
    eprintln!(
        "[qwen36] loaded: id={} quant={} ctx={} slots={}",
        info.id, info.quant, info.context_length, info.slots
    );

    let state = AppState {
        engine,
        api_key: cfg.api_key.clone(),
    };
    let app = build_router(state).merge(Router::new().route(
        "/",
        get(|| async { Html(CHAT_HTML) }),
    ));
    let listener = tokio::net::TcpListener::bind(format!("{}:{}", cfg.host, cfg.port)).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
