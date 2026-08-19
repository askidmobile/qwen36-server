//! Admin-эндпоинты: список локальных GGUF + горячая замена модели.
//!
//! - GET  /v1/available_models — *.gguf в QWEN36_MODELS_DIR (recursive ≤3 ур.)
//! - POST /v1/switch_model {path, ctx?, slots?} — выгрузить текущий движок,
//!   дождаться освобождения VRAM, загрузить новый, swap. Во время загрузки
//!   generate → 503. Ответ сразу (202), прогресс — через GET /v1/models.
//! - POST /v1/unload_model — выгрузить движок, освободить VRAM без загрузки новой модели.

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::Arc;

use super::{api_error, AppState};
use crate::engine::Engine;
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
    let Some(stem) = name.strip_suffix(".gguf") else {
        return false;
    };
    let Some(pos) = stem.rfind("-of-") else {
        return false;
    };
    let before = &stem[..pos];
    let after = &stem[pos + 4..];
    let first = before.rsplit('-').next().unwrap_or("");
    first.len() == 5
        && first.bytes().all(|b| b.is_ascii_digit())
        && after.len() == 5
        && after.bytes().all(|b| b.is_ascii_digit())
}

/// Рекурсивный сбор *.gguf (глубина ≤ 3 от корня).
fn scan_profiles(dir: &Path, depth: usize, out: &mut Vec<Value>) {
    if depth > 3 {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            scan_profiles(&path, depth + 1, out);
        } else if path.file_name().and_then(|name| name.to_str()) == Some("profile.json") {
            match crate::profile::ResolvedProfile::load(&path) {
                Ok(profile) => out.push(json!({
                    "kind": "profile",
                    "id": profile.manifest.profile_id,
                    "release_version": profile.manifest.release_version,
                    "path": path.to_string_lossy(),
                    "capabilities": profile.capabilities(),
                    "components": {"vision": profile.vision, "mtp": profile.mtp},
                })),
                Err(error) => out.push(json!({
                    "kind": "profile",
                    "path": path.to_string_lossy(),
                    "valid": false,
                    "error": format!("{error:#}"),
                })),
            }
        }
    }
}

fn scan_gguf(dir: &Path, depth: usize, out: &mut Vec<Value>) {
    if depth > 3 {
        return;
    }
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
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
            // Пропускаем файлы, которые прямо сейчас докачиваются (mtime < 30s):
            // частичный GGUF валиден по заголовку, но тензоры = мусор
            // (поймано: чат выдавал token ids вместо текста).
            if let Ok(meta) = e.metadata() {
                if let Ok(mtime) = meta.modified() {
                    if mtime.elapsed().map(|d| d.as_secs() < 30).unwrap_or(false) {
                        continue;
                    }
                }
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
        crate::vram_plan::footprint_from_gguf(&p)
            .map(|fp| fp.native_ctx)
            .unwrap_or(0)
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
    let mut profiles = Vec::new();
    scan_profiles(&state.models_dir, 0, &mut profiles);
    profiles.sort_by(|a, b| a["path"].as_str().cmp(&b["path"].as_str()));
    let (cur, ctx, slots) = state
        .switcher
        .current
        .read()
        .map(|c| c.clone())
        .unwrap_or_else(|_| (PathBuf::new(), 0, 0));
    // Выгруженная модель: inner=None → current=null (UI показывает «выгружена»).
    let loaded = state.switcher.is_loaded();
    let current_json = if loaded {
        json!({ "path": cur.to_string_lossy(), "ctx": ctx, "slots": slots })
    } else {
        Value::Null
    };
    let (ready, load_err) = (state.switcher.ready(), state.switcher.load_error());
    Json(json!({
        "models_dir": state.models_dir.to_string_lossy(),
        "current": current_json,
        "loading": state.switcher.loading.load(Ordering::Relaxed),
        "ready": loaded && ready,
        "load_error": load_err,
        "last_error": state.switcher.last_error.read().map(|s| s.clone()).unwrap_or_default(),
        "models": out,
        "profiles": profiles,
        "active_profile": state.profile.as_ref().map(|profile| json!({
            "id": profile.manifest.profile_id,
            "release_version": profile.manifest.release_version,
            "manifest": profile.manifest_path,
            "capabilities": profile.capabilities(),
            "components": {"vision": profile.vision, "mtp": profile.mtp},
        })),
    }))
}

/// Выгрузить текущую модель без загрузки новой: освободить VRAM.
/// Порядок: shutdown dispatch thread → take/drop engine → дождаться выхода
/// потока → trim CUDA mempool. Dispatch обязан выйти (break 'outer, poll 50мс) —
/// тогда adapter drop вызывает cudaFree и VRAM реально возвращается ОС.
/// Сохранить дефолты сэмплинга: обновить runtime + записать в .env.
/// WebUI «Сохранить по умолчанию». .env переписывается по ключам QWEN36_*
/// сэмплинга — остальное (секреты, модель) не трогается.
pub async fn sampling_defaults(
    State(state): State<AppState>,
    Json(req): Json<crate::config::SamplingDefaults>,
) -> Response {
    if !(req.temperature >= 0.0 && req.top_p > 0.0 && req.top_p <= 1.0
        && req.min_p >= 0.0 && req.min_p <= 1.0 && req.presence_penalty >= 0.0
        && req.repetition_penalty > 0.0 && req.max_tokens > 0)
    {
        return api_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "invalid sampling values",
        );
    }
    *state.sampling.write().expect("sampling lock") = req.clone();
    if let Err(e) = persist_sampling(&state.env_file, &req) {
        return api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "api_error",
            format!(".env write failed: {e}"),
        );
    }
    Json(json!({"status": "saved"})).into_response()
}

fn persist_sampling(path: &std::path::Path, d: &crate::config::SamplingDefaults) -> anyhow::Result<()> {
    let keys: Vec<(String, String)> = vec![
        ("QWEN36_TEMPERATURE".into(), d.temperature.to_string()),
        ("QWEN36_TOP_P".into(), d.top_p.to_string()),
        ("QWEN36_TOP_K".into(), d.top_k.to_string()),
        ("QWEN36_MIN_P".into(), d.min_p.to_string()),
        ("QWEN36_PRESENCE_PENALTY".into(), d.presence_penalty.to_string()),
        ("QWEN36_REPETITION_PENALTY".into(), d.repetition_penalty.to_string()),
        ("QWEN36_MAX_TOKENS".into(), d.max_tokens.to_string()),
        ("QWEN36_THINKING".into(), d.thinking.to_string()),
    ];
    let existing = std::fs::read_to_string(path).unwrap_or_default();
    let mut seen = std::collections::HashSet::new();
    let mut out: Vec<String> = existing
        .lines()
        .map(|line| {
            let name = line.split('=').next().unwrap_or("").trim();
            if let Some((_, v)) = keys.iter().find(|(k, _)| k == name) {
                seen.insert(name.to_string());
                format!("{name}={v}")
            } else {
                line.to_string()
            }
        })
        .collect();
    for (k, v) in &keys {
        if !seen.contains(k) {
            out.push(format!("{k}={v}"));
        }
    }
    std::fs::write(path, out.join("\n") + "\n")?;
    Ok(())
}

/// Сохранить пресет режима (instruct/thinking/thinking-coding) в .env
/// (QWEN36_PRESETS JSON) + runtime. WebUI «Сохранить как пресет».
pub async fn sampling_preset(
    State(state): State<AppState>,
    Json(req): Json<PresetRequest>,
) -> Response {
    if req.name.trim().is_empty() || req.name.len() > 32
        || !(req.values.temperature >= 0.0 && req.values.top_p > 0.0 && req.values.top_p <= 1.0)
    {
        return api_error(StatusCode::BAD_REQUEST, "invalid_request_error", "invalid preset");
    }
    {
        let mut presets = state.presets.write().expect("presets lock");
        presets.insert(req.name.clone(), req.values.clone());
    }
    let all = state.presets.read().expect("presets lock").clone();
    if let Err(e) = persist_presets(&state.env_file, &all) {
        return api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "api_error",
            format!(".env write failed: {e}"),
        );
    }
    Json(json!({"status": "saved", "preset": req.name})).into_response()
}

#[derive(serde::Deserialize)]
pub struct PresetRequest {
    name: String,
    values: crate::config::SamplingPresetValues,
}

fn persist_presets(
    path: &std::path::Path,
    presets: &crate::config::SamplingPresets,
) -> anyhow::Result<()> {
    let json_line = format!(
        "QWEN36_PRESETS={}",
        serde_json::to_string(presets)?
    );
    let existing = std::fs::read_to_string(path).unwrap_or_default();
    let mut replaced = false;
    let mut out: Vec<String> = existing
        .lines()
        .map(|line| {
            if line.trim_start().starts_with("QWEN36_PRESETS=") {
                replaced = true;
                json_line.clone()
            } else {
                line.to_string()
            }
        })
        .collect();
    if !replaced {
        out.push(json_line);
    }
    std::fs::write(path, out.join("\n") + "\n")?;
    Ok(())
}

pub async fn unload_model(State(state): State<AppState>) -> Response {
    if state.switcher.loading.swap(true, Ordering::Relaxed) {
        return api_error(
            StatusCode::CONFLICT,
            "invalid_request_error",
            "model switch already in progress",
        );
    }
    let result = tokio::task::spawn_blocking({
        let switcher = state.switcher.clone();
        let cuda_device = state.cuda_device.clone();
        move || {
            // shutdown → dispatch выходит (poll 50мс) → take() возвращает последний
            // Arc → drop вызывает Drop: join dispatch thread, adapter (GPU-память)
            // освобождён синхронно до выхода из этого блока.
            switcher.shutdown();
            let old = switcher.take();
            drop(old);
            #[cfg(feature = "cuda")]
            if let Some(dev) = cuda_device.read().expect("cuda_device lock").as_ref() {
                if let candle_core::Device::Cuda(c) = dev {
                    use candle_core::backend::BackendDevice;
                    let _ = c.synchronize();
                    let _ = candle_core::cuda_backend::mem_pool::trim_default_mempool(&c);
                    let _ = c.synchronize();
                }
            }
            switcher.last_error.write().expect("last_error lock").clear();
        }
    })
    .await;
    state.switcher.loading.store(false, Ordering::Relaxed);
    if let Err(e) = result {
        return api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "api_error",
            format!("unload failed: {e}"),
        );
    }
    (
        StatusCode::OK,
        Json(json!({"status": "unloaded"})),
    )
        .into_response()
}

pub async fn switch_model(
    State(state): State<AppState>,
    Json(req): Json<SwitchRequest>,
) -> Response {
    if state.switcher.loading.swap(true, Ordering::Relaxed) {
        return api_error(
            StatusCode::CONFLICT,
            "invalid_request_error",
            "model switch already in progress",
        );
    }
    // Сброс прошлой ошибки: UI-поллер читает её как сигнал фейла.
    if let Ok(mut le) = state.switcher.last_error.write() {
        le.clear();
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
    let root = state
        .models_dir
        .canonicalize()
        .unwrap_or_else(|_| state.models_dir.clone());
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
    // Отказ для файла, который докачивается прямо сейчас (mtime < 30s).
    if let Ok(meta) = std::fs::metadata(&path) {
        if let Ok(mtime) = meta.modified() {
            if mtime.elapsed().map(|d| d.as_secs() < 30).unwrap_or(false) {
                state.switcher.loading.store(false, Ordering::Relaxed);
                return api_error(
                    StatusCode::CONFLICT,
                    "invalid_request_error",
                    "model file is still downloading (mtime < 30s)",
                );
            }
        }
    }

    let (_cur_path, cur_ctx, cur_slots) = state
        .switcher
        .current
        .read()
        .map(|c| c.clone())
        .unwrap_or_else(|_| (PathBuf::new(), 8192, 4));
    let req_ctx = req.ctx.unwrap_or(cur_ctx);
    let req_slots_raw = req.slots.unwrap_or(cur_slots);
    if req_slots_raw == 0 || req_slots_raw > crate::engine_batched::MAX_SLOTS {
        state.switcher.loading.store(false, Ordering::Relaxed);
        return api_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            format!(
                "slots {} exceeds max {}",
                req_slots_raw,
                crate::engine_batched::MAX_SLOTS
            ),
        );
    }
    let req_slots = req_slots_raw;

    tokio::spawn(async move {
        // Страховка: do_switch паника в spawn → loading застрял бы навсегда
        // (все switch/unload → 409). timeout гарантирует сброс при зависании.
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(600),
            do_switch(&state, path, req_ctx, req_slots),
        )
        .await;
        match result {
            Err(_) => eprintln!("[switch] TIMEOUT 600s — flag reset"),
            Ok(Err(e)) => {
                eprintln!("[switch] FAILED: {e:#}");
                if let Ok(mut le) = state.switcher.last_error.write() {
                    *le = format!("{e:#}");
                }
            }
            Ok(Ok(())) => {}
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
    // Pending/claimed media belongs to old profile and must never survive switch.
    state.media.store.purge_all();
    let fp = vram_plan::footprint_from_gguf(&path)?;
    let (ctx, slots, kv_budget_mib, kv_per_tok_mib) = match vram_plan::total_vram_mib() {
        Some(total) => {
            let plan = vram_plan::compute_dynamic(total, &fp, req_ctx, req_slots)?;
            eprintln!("[switch] {}", plan.report);
            (
                plan.ctx,
                plan.slots,
                plan.kv_budget_mib,
                plan.kv_per_tok_mib,
            )
        }
        None => (req_ctx, req_slots, 0.0, 0.0),
    };

    // 2. Выгрузить старый движок и дождаться освобождения VRAM.
    // shutdown() ЯВНО до take: Arc<dyn Engine> может быть запинен открытым
    // SSE-стримом браузера (generate держит clone на весь запрос) — тогда
    // Drop не вызовется, dispatch-thread не получит флаг и VRAM не освободится
    // до таймаута 60с (наблюдаемая «заминка» при переключении).
    state.switcher.shutdown();
    let old = state.switcher.take();
    drop(old);
    if vram_plan::total_vram_mib().is_some() {
        // Адаптивное ожидание: старый dispatch-thread может ещё грузить адаптер
        // (35B ~60s внутри Qwen35BatchAdapter::load — shutdown там не виден).
        // Ждём роста free; если 3 полла подряд без изменений — стагнация,
        // продолжаем (KV-бюджет и так посчитан от текущего free).
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
        let mut last_free = 0usize;
        let mut stagnant = 0u8;
        loop {
            let free = vram_plan::free_vram_mib().unwrap_or(0) as usize;
            if free >= fp.weights_mib {
                break;
            }
            stagnant = if free == last_free { stagnant + 1 } else { 0 };
            last_free = free;
            if stagnant >= 3 || std::time::Instant::now() > deadline {
                eprintln!("[switch] VRAM wait done (free={free}MiB, stagnant={stagnant}), продолжаю");
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
        BatchedEngine::load(
            BatchConfig {
                model_path: path.to_string_lossy().into_owned(),
                slots,
                max_queue: 64,
                req_timeout: std::time::Duration::from_secs(3600),
                context_length: ctx,
                kv_budget_mib,
                kv_per_tok_mib,
                prefix_cache_mib: 0,
            },
            state.media.clone(),
            None,
            None,
        )
        .await?
    } else {
        // single-slot через Config-like структуру нет — используем CandleEngine
        // с минимальным Config.
        let cfg = crate::config::Config {
            profile: None,
            resolved_profile: None,
            model: path.clone(),
            host: String::new(),
            port: 0,
            api_keys: Vec::new(),
            ctx,
            slots,
            kv_budget_mib,
            kv_per_tok_mib,
            prefix_cache_mib: 0,
            media_temp: std::env::temp_dir().join("qwen36-media"),
            sampling: crate::config::SamplingDefaults::default(),
            presets: crate::config::default_presets(),
            env_file: std::path::PathBuf::from(".env"),
        };
        Arc::new(crate::engine::CandleEngine::load(&cfg)?)
    };
    let info = engine.model_info();
    eprintln!(
        "[switch] loaded: id={} ctx={} slots={}",
        info.id, info.context_length, info.slots
    );
    state.switcher.install(engine, path, ctx, slots);
    Ok(())
}
