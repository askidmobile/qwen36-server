//! Бинарь qwen36-server: env QWEN36_* → engine → HTTP (три API + веб-чат).
//!
//! Engine выбирается по QWEN36_SLOTS:
//! - slots > 1 → BatchedEngine (BD-007, 4 конкурентных слота через BatchScheduler).
//! - slots == 1 → CandleEngine (single-slot, Mutex — для smoke-тестов/дебага).

use anyhow::Result;
use axum::{response::Html, routing::get, Router};
use qwen36_server::{
    api::{build_router, AppState},
    config::Config,
    engine::{CandleEngine, Engine},
    engine_batched::{BatchConfig, BatchedEngine},
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

    let engine: Arc<dyn Engine> = if cfg.slots > 1 {
        let bcfg = BatchConfig {
            model_path: cfg.model.to_string_lossy().into_owned(),
            slots: cfg.slots,
            max_queue: 64,
            req_timeout: std::time::Duration::from_secs(600),
            context_length: cfg.ctx,
        };
        BatchedEngine::load(bcfg).await?
    } else {
        Arc::new(CandleEngine::load(&cfg)?)
    };
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
