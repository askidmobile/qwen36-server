//! Бинарь qwen36-server: env QWEN36_* → CandleEngine → HTTP.
//! Роутер придёт от агента API-слоя (crate::api::router); пока — placeholder.

use anyhow::Result;
use qwen36_server::{config::Config, engine::{CandleEngine, Engine}};

#[tokio::main]
async fn main() -> Result<()> {
    let cfg = Config::from_env()?;
    eprintln!(
        "[qwen36] model={:?} ctx={} slots={} listen={}:{}",
        cfg.model, cfg.ctx, cfg.slots, cfg.host, cfg.port
    );

    let engine = CandleEngine::load(&cfg)?;
    let info = engine.model_info();
    eprintln!(
        "[qwen36] loaded: id={} quant={} ctx={} slots={}",
        info.id, info.quant, info.context_length, info.slots
    );

    // ponytail: заменить на crate::api::router(engine), когда появится src/api.rs.
    let _ = engine;
    let app = axum::Router::new().route("/health", axum::routing::get(|| async { "ok" }));
    let listener = tokio::net::TcpListener::bind(format!("{}:{}", cfg.host, cfg.port)).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
