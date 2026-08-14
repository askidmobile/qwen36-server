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
    engine_swap::SwappableEngine,
};
use std::sync::Arc;

const CHAT_HTML: &str = include_str!("../web/index.html");

#[tokio::main]
async fn main() -> Result<()> {
    let cfg = Config::load()?;
    let profile = cfg.resolved_profile.clone();
    let media = Arc::new(qwen36_server::media::MediaService::new(
        qwen36_server::media::MediaConfig {
            temp_root: cfg.media_temp.clone(),
            ..Default::default()
        },
    )?);
    qwen36_server::media::MediaStore::spawn_reaper(
        media.store.clone(),
        std::time::Duration::from_secs(60),
    );
    eprintln!(
        "[qwen36] api keys: {} ({})",
        cfg.api_keys.len(),
        cfg.api_keys
            .iter()
            .map(|entry| entry.name.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );
    eprintln!(
        "[qwen36] model={:?} profile={:?} ctx={} slots={} listen={}:{}",
        cfg.model, cfg.profile, cfg.ctx, cfg.slots, cfg.host, cfg.port
    );

    let engine: Arc<dyn Engine> = if cfg.slots > 1 {
        let bcfg = BatchConfig {
            model_path: cfg.model.to_string_lossy().into_owned(),
            slots: cfg.slots,
            max_queue: 64,
            req_timeout: std::time::Duration::from_secs(600),
            context_length: cfg.ctx,
            kv_budget_mib: cfg.kv_budget_mib,
            kv_per_tok_mib: cfg.kv_per_tok_mib,
            prefix_cache_mib: cfg.prefix_cache_mib,
        };
        BatchedEngine::load(bcfg).await?
    } else {
        Arc::new(CandleEngine::load(&cfg)?)
    };
    #[cfg(feature = "cuda")]
    let cuda_device = candle_core::Device::new_cuda(0).ok();
    #[cfg(not(feature = "cuda"))]
    let cuda_device = None;
    let info = engine.model_info();
    eprintln!(
        "[qwen36] loaded: id={} quant={} ctx={} slots={}",
        info.id, info.quant, info.context_length, info.slots
    );

    // Корень сканирования моделей: QWEN36_MODELS_DIR или родитель директории
    // модели (D:\Models\org\repo\model.gguf → D:\Models).
    let models_dir = std::env::var("QWEN36_MODELS_DIR")
        .map(std::path::PathBuf::from)
        .ok()
        .or_else(|| {
            cfg.model
                .parent()
                .and_then(|p| p.parent())
                .map(|p| p.to_path_buf())
        })
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    let switcher = Arc::new(SwappableEngine::new(
        engine,
        cfg.model.clone(),
        cfg.ctx,
        cfg.slots,
    ));
    let state = AppState {
        engine: switcher.clone(),
        media,
        switcher,
        api_keys: cfg.api_keys.clone().into(),
        models_dir,
        profile,
        cuda_device,
    };
    let app =
        build_router(state).merge(Router::new().route("/", get(|| async { Html(CHAT_HTML) })));
    let listener = tokio::net::TcpListener::bind(format!("{}:{}", cfg.host, cfg.port)).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
