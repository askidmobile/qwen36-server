//! Бинарь `yforge`: CLI/env/env-файл → engine → HTTP (три API + веб-чат).
//!
//! Настройки берутся из трёх источников по убыванию приоритета: флаги
//! командной строки, окружение процесса, env-файл (`--env`). Разбор — в
//! [`qwen36_server::cli`].
//!
//! Engine выбирается по `SLOTS`:
//! - qwen35/qwen35moe → BatchedEngine при любом числе слотов (BD-007).
//!   Одиночный слот тоже идёт сюда: paged-пул пропорционален числу слотов,
//!   поэтому SLOTS=1 вдвое дешевле по VRAM, а CandleEngine отдавал бы CUDA-графы
//!   вместе со скоростью декода. Откат — FORCE_CANDLE_ENGINE=1.
//! - остальные архитектуры → CandleEngine (single-slot, Mutex).

use anyhow::Result;
use clap::Parser;
use qwen36_server::{
    api::{build_router, AppState},
    cli::Cli,
    config::Config,
    engine::{CandleEngine, Engine},
    engine_batched::{BatchConfig, BatchedEngine},
    engine_swap::SwappableEngine,
};
use std::sync::Arc;

/// Строка лога в стиле llama.cpp: `yforge: ключ = значение`.
macro_rules! log_kv {
    ($key:expr, $($arg:tt)*) => {
        println!("yforge: {:<22} = {}", $key, format_args!($($arg)*))
    };
}

/// Заголовок секции: пустая строка + подпись.
fn log_section(title: &str) {
    println!("\nyforge: === {title} ===");
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    // Флаги — в окружение ДО чтения env-файла: файл пишет только незанятые
    // имена, поэтому приоритет «флаг > окружение > файл» получается сам.
    cli.apply_to_env();

    #[cfg(feature = "cuda")]
    qwen36_server::engine::cuda_prefer_blocking_sync();
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
    log_section("конфигурация");
    log_kv!(
        "build",
        "{} ({})",
        env!("CARGO_PKG_VERSION"),
        backend_name()
    );
    log_kv!("env file", "{}", cfg.env_file.display());
    log_kv!("model", "{}", cfg.model.display());
    if let Some(profile) = &cfg.profile {
        log_kv!("profile", "{}", profile.display());
    }
    log_kv!("listen", "{}:{}", cfg.host, cfg.port);
    log_kv!(
        "api keys",
        "{} ({})",
        cfg.api_keys.len(),
        cfg.api_keys
            .iter()
            .map(|entry| entry.name.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );
    log_kv!("n_ctx", "{}", cfg.ctx);
    log_kv!("n_slots", "{}", cfg.slots);
    log_kv!("n_batch", "{}", cfg.batch_size);
    log_kv!("max_tokens", "{}", cfg.sampling.max_tokens);
    log_kv!("gpu_layers", "{}", cfg.gpu_layers);
    log_kv!("kv_cache_type", "{}", cfg.kv_cache_type);
    log_kv!(
        "kv_pool",
        "{}",
        match std::env::var("KV_POOL_Q8").as_deref() {
            Ok("1") => "q8 (int8, вдвое меньше VRAM)",
            _ => "f16",
        }
    );
    log_kv!(
        "cuda_graphs",
        "{}",
        match std::env::var("CUDA_GRAPHS").as_deref() {
            Ok("1") => "включены",
            _ => "выключены",
        }
    );
    log_kv!(
        "prefix_cache",
        "{}",
        if cfg.prefix_cache_mib == 0 {
            "выключен".to_string()
        } else {
            format!("{} MiB", cfg.prefix_cache_mib)
        }
    );
    log_kv!("flash_attn", "{}", if cfg.flash_attn { "1" } else { "0" });
    log_kv!("req_timeout", "{} s", cfg.req_timeout);
    log_kv!(
        "sampling",
        "temp={} top_p={} top_k={} thinking={}",
        cfg.sampling.temperature,
        cfg.sampling.top_p,
        cfg.sampling.top_k,
        cfg.sampling.thinking
    );

    if cli.dry_run {
        println!("\nyforge: --dry-run: модель не загружается, выход");
        return Ok(());
    }

    let architecture = qwen36_server::engine::gguf_architecture(&cfg.model)?;
    let qwen35 = matches!(architecture.as_str(), "qwen35" | "qwen35moe");
    // BatchedEngine и на одном слоте: paged-пул пропорционален числу слотов
    // (capacity_b x max_blocks), поэтому SLOTS=1 вдвое дешевле по VRAM и на
    // 12 ГБ это прямо удваивает достижимый контекст. Раньше одиночный слот
    // уходил на CandleEngine и терял CUDA-графы вместе со скоростью декода.
    // FORCE_CANDLE_ENGINE=1 возвращает прежний путь для отладки.
    let force_candle = std::env::var("FORCE_CANDLE_ENGINE").as_deref() == Ok("1");
    let engine: Arc<dyn Engine> = if qwen35 && !force_candle {
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
        let mtp_enabled = match std::env::var("MTP").as_deref() {
            Ok("1") => true,
            Ok("0") | Err(_) => false,
            Ok(value) => anyhow::bail!("MTP must be 0 or 1, got {value:?}"),
        };
        // Env-фолбэк должен быть ВНУТРИ проверки mtp_enabled, а не после неё.
        // Раньше он висел на .or_else(), который срабатывает ровно тогда, когда
        // предыдущее звено дало None — то есть при MTP=0. Выключатель включал:
        // при заданном MTP_PATH спекуляция работала и с MTP=0
        // (замер: drafted=316, accepted=130 при MTP=0).
        let mtp_path = if mtp_enabled {
            profile
                .as_ref()
                .and_then(|profile| match &profile.mtp {
                    qwen36_server::profile::ComponentArtifact::Available { path } => {
                        Some(path.clone())
                    }
                    _ => None,
                })
                // Тонкий MTP-артефакт без полного профиля: эксперименты и
                // модели, для которых манифест ещё не собран.
                .or_else(|| {
                    std::env::var("MTP_PATH")
                        .ok()
                        .filter(|p| !p.is_empty())
                        .map(std::path::PathBuf::from)
                })
        } else {
            None
        };
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
    log_section("модель загружена");
    log_kv!("id", "{}", info.id);
    log_kv!("quant", "{}", info.quant);
    log_kv!("n_ctx (факт)", "{}", info.context_length);
    log_kv!("n_slots (факт)", "{}", info.slots);

    // Корень сканирования моделей: MODELS_DIR (или MODELS_DIR) или родитель директории
    // модели (D:\Models\org\repo\model.gguf → D:\Models).
    let models_dir = std::env::var("MODELS_DIR")
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
        sampling_policy: std::sync::Arc::new(std::sync::RwLock::new(cfg.sampling_policy.clone())),
        presets: std::sync::Arc::new(std::sync::RwLock::new(cfg.presets.clone())),
        env_file: cfg.env_file.clone(),
    };
    let app = build_router(state).merge(qwen36_server::api::proxy::proxy_router(
        cfg.studio_url.clone(),
    ));
    let listener = tokio::net::TcpListener::bind(format!("{}:{}", cfg.host, cfg.port)).await?;
    log_section("сервер запущен");
    log_kv!("studio proxy", "{}", cfg.studio_url);
    log_kv!("OpenAI API", "http://{}:{}/v1", cfg.host, cfg.port);
    log_kv!(
        "Anthropic API",
        "http://{}:{}/v1/messages",
        cfg.host,
        cfg.port
    );
    log_kv!("WebUI", "http://{}:{}/", cfg.host, cfg.port);
    println!("\nyforge: готов принимать запросы");
    axum::serve(listener, app).await?;
    Ok(())
}

/// Имя бекенда для строки версии — как `llama.cpp` печатает свой билд.
fn backend_name() -> &'static str {
    if cfg!(feature = "cuda") {
        "CUDA"
    } else if cfg!(feature = "metal") {
        "Metal"
    } else {
        "CPU"
    }
}
