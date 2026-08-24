//! Бинарь qwen36-server: env QWEN36_* → engine → HTTP (три API + веб-чат).
//!
//! Engine выбирается по QWEN36_SLOTS:
//! - slots > 1 → BatchedEngine (BD-007, 4 конкурентных слота через BatchScheduler).
//! - slots == 1 → CandleEngine (single-slot, Mutex — для smoke-тестов/дебага).

use anyhow::Result;
use axum::Router;
use qwen36_server::{
    api::{build_router, AppState},
    config::Config,
    engine::{CandleEngine, Engine},
    engine_batched::{BatchConfig, BatchedEngine},
    engine_swap::SwappableEngine,
};
use std::sync::Arc;

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

    let architecture = qwen36_server::engine::gguf_architecture(&cfg.model)?;
    let qwen35 = matches!(architecture.as_str(), "qwen35" | "qwen35moe");
    let engine: Arc<dyn Engine> = if qwen35 && cfg.slots > 1 {
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
        let vision_path = profile.as_ref().and_then(|profile| match &profile.vision {
            qwen36_server::profile::ComponentArtifact::Available { path } => Some(path.clone()),
            _ => None,
        });
        let mtp_enabled = match std::env::var("MTP")
            .or_else(|_| std::env::var("QWEN36_MTP"))
            .as_deref()
        {
            Ok("1") => true,
            Ok("0") | Err(_) => false,
            Ok(value) => anyhow::bail!("MTP must be 0 or 1, got {value:?}"),
        };
        let mtp_path = mtp_enabled
            .then(|| {
                profile.as_ref().and_then(|profile| match &profile.mtp {
                    qwen36_server::profile::ComponentArtifact::Available { path } => {
                        Some(path.clone())
                    }
                    _ => None,
                })
            })
            .flatten()
            // Env-фолбэк: тонкий MTP-артефакт без полного профиля
            // (эксперименты и модели, для которых манифест ещё не собран).
            .or_else(|| {
                std::env::var("QWEN36_MTP_PATH")
                    .ok()
                    .filter(|p| !p.is_empty())
                    .map(std::path::PathBuf::from)
            });
        BatchedEngine::load(bcfg, media.clone(), vision_path, mtp_path).await?
    } else {
        Arc::new(CandleEngine::load(&cfg)?)
    };
    #[cfg(feature = "cuda")]
    let cuda_device = candle_core::Device::new_cuda(0).ok();
    #[cfg(not(feature = "cuda"))]
    let cuda_device = None;
    // Mutably wrapped: unload_model clears it to free CUDA context + VRAM.
    let cuda_device = std::sync::Arc::new(std::sync::RwLock::new(cuda_device));
    let info = engine.model_info();
    eprintln!(
        "[qwen36] loaded: id={} quant={} ctx={} slots={}",
        info.id, info.quant, info.context_length, info.slots
    );

    // Корень сканирования моделей: MODELS_DIR (или QWEN36_MODELS_DIR) или родитель директории
    // модели (D:\Models\org\repo\model.gguf → D:\Models).
    let models_dir = std::env::var("MODELS_DIR")
        .or_else(|_| std::env::var("QWEN36_MODELS_DIR"))
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
        hf_downloads: Default::default(),
        sampling: std::sync::Arc::new(std::sync::RwLock::new(cfg.sampling.clone())),
        presets: std::sync::Arc::new(std::sync::RwLock::new(cfg.presets.clone())),
        env_file: cfg.env_file.clone(),
    };
    let app = build_router(state)
        .merge(qwen36_server::api::proxy::proxy_router(cfg.studio_url.clone()));
    eprintln!("[qwen36] studio proxy → {}", cfg.studio_url);
    let listener = tokio::net::TcpListener::bind(format!("{}:{}", cfg.host, cfg.port)).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
