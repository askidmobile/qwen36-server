//! Admin-эндпоинты: список локальных GGUF + горячая замена модели.
//!
//! - GET  /v1/available_models — *.gguf в QWEN36_MODELS_DIR (recursive ≤3 ур.)
//! - POST /v1/switch_model {path, ctx?, slots?} — выгрузить текущий движок,
//!   дождаться освобождения VRAM, загрузить новый, swap. Во время загрузки
//!   generate → 503. Ответ сразу (202), прогресс — через GET /v1/models.

use axum::{extract::State, http::StatusCode, response::{IntoResponse, Response}, Json};
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::Arc;

use super::{api_error, AppState};
use crate::engine_batched::{BatchConfig, BatchedEngine};
use crate::vram_plan;

#[derive(Deserialize)]
pub struct SwitchRequest {
    /// Полный путь к GGUF или имя файла из available_models.
    path: String,
    ctx: Option<usize>,
    slots: Option<usize>,
}

/// split-часть GGUF: `name-00001-of-00002.gguf` (по одному файлу не грузится).
fn is_split_part(name: &str) -> bool {
    let Some(stem) = name.strip_suffix(".gguf") else { return false };
    let Some(pos) = stem.rfind("-of-") else { return false };
    let before = &stem[..pos];
    let after = &stem[pos + 4..];
    let first = before.rsplit('-').next().unwrap_or("");
    first.len() == 5
        && first.bytes().all(|b| b.is_ascii_digit())
        && after.len() == 5
        && after.bytes().all(|b| b.is_ascii_digit())
}

/// Рекурсивный сбор *.gguf (глубина ≤ 3 от корня).
fn scan_gguf(dir: &Path, depth: usize, out: &mut Vec<Value>) {
    if depth > 3 {
        return;
    }
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            scan_gguf(&p, depth + 1, out);
        } else if p.extension().and_then(|s| s.to_str()) == Some("gguf") {
            let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("?");
            // Пропускаем незагружаемые: split-части (-00001-of-00002) и mmproj.
            if is_split_part(name) || name.starts_with("mmproj") {
                continue;
            }
            let size_mib = e.metadata().map(|m| m.len() / 1024 / 1024).unwrap_or(0);
            out.push(json!({
                "name": name,
                "path": p.to_string_lossy(),
                "size_mib": size_mib,
            }));
        }
    }
}

/// Нативный контекст конкретной модели (по пути) — лёгкий metadata-read
/// одного файла для UI-слайдера. НЕ пакетно: Content::read всех файлов подряд
/// блокировал tokio-воркер на 20+ секунд («зависание» 2026-08-10).
pub async fn model_native_ctx(
    State(state): State<AppState>,
    axum::extract::Query(q): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Json<Value> {
    let path = q.get("path").cloned().unwrap_or_default();
    let p = PathBuf::from(&path);
    if !p.starts_with(&state.models_dir) {
        return Json(json!({"native_ctx": 0, "error": "outside models_dir"}));
    }
    let nc = tokio::task::spawn_blocking(move || {
        crate::vram_plan::footprint_from_gguf(&p).map(|fp| fp.native_ctx).unwrap_or(0)
    })
    .await
    .unwrap_or(0);
    Json(json!({"native_ctx": nc}))
}

/// Матрица «слоты → макс. ctx» для модели: честные варианты для UI-селекта.
/// worst-case (KV заполнен полностью). GET /v1/ctx_matrix?path=...
pub async fn ctx_matrix(
    State(state): State<AppState>,
    axum::extract::Query(q): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Json<Value> {
    let path = q.get("path").cloned().unwrap_or_default();
    let p = PathBuf::from(&path);
    if !p.starts_with(&state.models_dir) {
        return Json(json!({"error": "outside models_dir"}));
    }
    let total = vram_plan::total_vram_mib();
    let r = tokio::task::spawn_blocking(move || {
        let fp = vram_plan::footprint_from_gguf(&p)?;
        let (kv_budget_mib, kv_per_tok_mib) = match total {
            Some(t) => {
                let plan = vram_plan::compute_dynamic(t, &fp, 262144, 4)?;
                (plan.kv_budget_mib, plan.kv_per_tok_mib)
            }
            None => (0.0, 0.0),
        };
        Ok::<_, anyhow::Error>(json!({
            "native_ctx": fp.native_ctx,
            "weights_mib": fp.weights_mib,
            "kv_budget_mib": kv_budget_mib,
            "kv_per_tok_mib": kv_per_tok_mib,
        }))
    })
    .await;
    match r {
        Ok(Ok(v)) => Json(v),
        Ok(Err(e)) => Json(json!({"error": format!("{e:#}")})),
        Err(e) => Json(json!({"error": format!("{e:#}")})),
    }
}

pub async fn available_models(State(state): State<AppState>) -> Json<Value> {
    let mut out = Vec::new();
    scan_gguf(&state.models_dir, 0, &mut out);
    out.sort_by(|a, b| a["name"].as_str().cmp(&b["name"].as_str()));
    let (cur, ctx, slots) = state
        .switcher
        .current
        .read()
        .map(|c| c.clone())
        .unwrap_or_else(|_| (PathBuf::new(), 0, 0));
    Json(json!({
        "models_dir": state.models_dir.to_string_lossy(),
        "current": { "path": cur.to_string_lossy(), "ctx": ctx, "slots": slots },
        "loading": state.switcher.loading.load(Ordering::Relaxed),
        "last_error": state.switcher.last_error.read().map(|s| s.clone()).unwrap_or_default(),
        "models": out,
    }))
}

pub async fn switch_model(State(state): State<AppState>, Json(req): Json<SwitchRequest>) -> Response {
    if state.switcher.loading.swap(true, Ordering::Relaxed) {
        return api_error(StatusCode::CONFLICT, "invalid_request_error", "model switch already in progress");
    }

    // Разрешить имя файла в полный путь.
    let path = if Path::new(&req.path).is_absolute() || req.path.contains(['/', '\\']) {
        PathBuf::from(&req.path)
    } else {
        // поиск по имени в models_dir
        let mut found = Vec::new();
        scan_gguf(&state.models_dir, 0, &mut found);
        found
            .iter()
            .find(|m| m["name"].as_str() == Some(req.path.as_str()))
            .map(|m| PathBuf::from(m["path"].as_str().unwrap()))
            .unwrap_or_else(|| PathBuf::from(&req.path))
    };
    // Sandbox: только внутри models_dir, канонизация против `..` (аудит 2026-08-10).
    let canon = path.canonicalize().unwrap_or_else(|_| path.clone());
    let root = state.models_dir.canonicalize().unwrap_or_else(|_| state.models_dir.clone());
    if !canon.starts_with(&root) {
        state.switcher.loading.store(false, Ordering::Relaxed);
        return api_error(
            StatusCode::FORBIDDEN,
            "invalid_request_error",
            format!("path outside models_dir: {}", canon.display()),
        );
    }
    let path = canon;
    if !path.exists() {
        state.switcher.loading.store(false, Ordering::Relaxed);
        return api_error(
            StatusCode::NOT_FOUND,
            "invalid_request_error",
            format!("model not found: {}", path.display()),
        );
    }

    let (_cur_path, cur_ctx, cur_slots) = state
        .switcher
        .current
        .read()
        .map(|c| c.clone())
        .unwrap_or_else(|_| (PathBuf::new(), 8192, 4));
    let req_ctx = req.ctx.unwrap_or(cur_ctx);
    let req_slots = req.slots.unwrap_or(cur_slots).clamp(1, 4);

    tokio::spawn(async move {
        let result = do_switch(&state, path, req_ctx, req_slots).await;
        if let Err(e) = result {
            eprintln!("[switch] FAILED: {e:#}");
            if let Ok(mut le) = state.switcher.last_error.write() {
                *le = format!("{e:#}");
            }
        }
        state.switcher.loading.store(false, Ordering::Relaxed);
    });

    (
        StatusCode::ACCEPTED,
        Json(json!({"status": "loading", "path": req.path, "ctx": req_ctx, "slots": req_slots})),
    )
        .into_response()
}

async fn do_switch(
    state: &AppState,
    path: PathBuf,
    req_ctx: usize,
    req_slots: usize,
) -> anyhow::Result<()> {
    // 1. VRAM-план для новой модели (dynamic: ctx до native, бюджет в движок).
    let fp = vram_plan::footprint_from_gguf(&path)?;
    let (ctx, slots, kv_budget_mib, kv_per_tok_mib) = match vram_plan::total_vram_mib() {
        Some(total) => {
            let plan = vram_plan::compute_dynamic(total, &fp, req_ctx, req_slots)?;
            eprintln!("[switch] {}", plan.report);
            (plan.ctx, plan.slots, plan.kv_budget_mib, plan.kv_per_tok_mib)
        }
        None => (req_ctx, req_slots, 0.0, 0.0),
    };

    // 2. Выгрузить старый движок и дождаться освобождения VRAM.
    let old = state.switcher.take();
    drop(old);
    if vram_plan::total_vram_mib().is_some() {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
        loop {
            let free = vram_plan::free_vram_mib().unwrap_or(0);
            // Порог = веса новой модели (без +512: driver pool/nvidia-smi
            // занижают free, сверхзапрос таймаутит — уже ловили 180s «зависание»).
            if free as usize >= fp.weights_mib {
                break;
            }
            if std::time::Instant::now() > deadline {
                eprintln!("[switch] timeout waiting VRAM free (free={free}MiB), продолжаю");
                break;
            }
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        }
    }

    // 3. Загрузить новый движок (всегда batched при slots>1, иначе single).
    eprintln!(
        "[switch] loading {} ctx={ctx} slots={slots}",
        path.display()
    );
    let engine: Arc<dyn crate::engine::Engine> = if slots > 1 {
        BatchedEngine::load(BatchConfig {
            model_path: path.to_string_lossy().into_owned(),
            slots,
            max_queue: 64,
            req_timeout: std::time::Duration::from_secs(3600),
            context_length: ctx,
            kv_budget_mib,
            kv_per_tok_mib,
        })
        .await?
    } else {
        // single-slot через Config-like структуру нет — используем CandleEngine
        // с минимальным Config.
        let cfg = crate::config::Config {
            model: path.clone(),
            host: String::new(),
            port: 0,
            api_key: String::new(),
            ctx,
            slots,
            kv_budget_mib,
            kv_per_tok_mib,
        };
        Arc::new(crate::engine::CandleEngine::load(&cfg)?)
    };
    let info = engine.model_info();
    eprintln!("[switch] loaded: id={} ctx={} slots={}", info.id, info.context_length, info.slots);
    state.switcher.install(engine, path, ctx, slots);
    Ok(())
}
