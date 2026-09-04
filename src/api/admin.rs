//! Admin-эндпоинты: список локальных GGUF + горячая замена модели.
//!
//! - GET  /v1/available_models — *.gguf в MODELS_DIR (recursive ≤3 ур.)
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

/// Кэш сканирования моделей: (unixtime сек сохранения, список моделей).
/// Скан D:\\Models с 20+ GGUF занимает 30+ секунд (антивирус, глубокие деревья).
/// TTL 60 сек: WebUI поллит каждые 2с — повторные запросы мгновенные.
static SCAN_CACHE: std::sync::OnceLock<std::sync::Mutex<Option<(u64, Vec<Value>)>>> =
    std::sync::OnceLock::new();
const SCAN_CACHE_TTL: u64 = 60;

/// Кэш architecture по (path, mtime, size): чтение GGUF-заголовка у антивируса
/// занимает ~1.6s на файл. Файлы не меняются годами — читаем один раз.
static ARCH_CACHE: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashMap<std::path::PathBuf, (u64, u64, String)>>,
> = std::sync::OnceLock::new();

fn cached_architecture(p: &std::path::Path, mtime: u64, size: u64) -> String {
    let cache = ARCH_CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    if let Ok(guard) = cache.lock() {
        if let Some((m, sz, arch)) = guard.get(p) {
            if *m == mtime && *sz == size {
                return arch.clone();
            }
        }
    }
    let arch = crate::engine::gguf_architecture(p).unwrap_or_else(|_| "unknown".into());
    if let Ok(mut guard) = cache.lock() {
        guard.insert(p.to_path_buf(), (mtime, size, arch.clone()));
    }
    arch
}
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

pub fn scan_gguf(dir: &Path, depth: usize, out: &mut Vec<Value>) {
    scan_gguf_cached(dir, depth, out, false)
}

pub fn scan_gguf_cached(dir: &Path, depth: usize, out: &mut Vec<Value>, use_cache: bool) {
    if depth > 3 {
        return;
    }
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            scan_gguf_cached(&p, depth + 1, out, use_cache);
        } else if matches!(
            p.extension().and_then(|s| s.to_str()),
            Some("gguf") | Some("ytf")
        ) {
            let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("?");
            // Пропускаем незагружаемые: split-части (-00001-of-00002) и mmproj.
            if is_split_part(name) || name.starts_with("mmproj") {
                continue;
            }
            // Пропускаем файлы, которые прямо сейчас докачиваются (mtime < 30s)
            if let Ok(meta) = e.metadata() {
                if let Ok(mtime) = meta.modified() {
                    if mtime.elapsed().map(|d| d.as_secs() < 30).unwrap_or(false) {
                        continue;
                    }
                }
            }
            // Capability registry: только архитектуры с реальным runtime route
            // получают supported=true. Прочие честно помечаются unsupported с reason.
            let size_mib = e.metadata().map(|m| m.len() / 1024 / 1024).unwrap_or(0);
            let mtime = e
                .metadata()
                .and_then(|m| m.modified())
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs())
                .unwrap_or(0);
            let file_size = e.metadata().map(|m| m.len()).unwrap_or(0);
            let arch = cached_architecture(&p, mtime, file_size);
            let (supported, backend, reason) = crate::engine::architecture_capability(&arch);
            out.push(json!({
                "name": name,
                "path": p.to_string_lossy(),
                "size_mib": size_mib,
                "supported": supported,
                "architecture": backend,
                "reason": reason,
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
        crate::vram_plan::footprint_from_gguf_cached(&p)
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
        let fp = vram_plan::footprint_from_gguf_cached(&p)?;
        let (kv_budget_mib, kv_per_tok_mib) = match total {
            Some(t) => {
                let plan = vram_plan::compute_dynamic(t, &fp, 262144, 4, &crate::config::moe_placement_from_env())?;
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
    let t0 = std::time::Instant::now();
    let mut out = Vec::new();
    let t_scan = std::time::Instant::now();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let cached = SCAN_CACHE
        .get()
        .and_then(|c| c.lock().ok())
        .and_then(|guard| {
            guard.as_ref().and_then(|(saved_at, models)| {
                if now.saturating_sub(*saved_at) < SCAN_CACHE_TTL {
                    Some(models.clone())
                } else {
                    None
                }
            })
        });
    let out = match cached {
        Some(models) => models,
        None => {
            scan_gguf(&state.models_dir, 0, &mut out);
            out.sort_by(|a, b| a["name"].as_str().cmp(&b["name"].as_str()));
            let _ = SCAN_CACHE
                .get_or_init(|| std::sync::Mutex::new(None))
                .lock()
                .map(|mut guard| *guard = Some((now, out.clone())));
            out
        }
    };
    let t_prof = std::time::Instant::now();
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
    eprintln!(
        "[avail] total={:.1}s scan={:.1}s profiles={:.1}s models={}",
        t0.elapsed().as_secs_f64(),
        t_scan.elapsed().as_secs_f64(),
        t_prof.elapsed().as_secs_f64(),
        out.len()
    );
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
/// WebUI «Сохранить по умолчанию». .env переписывается по чистым ключам
/// сэмплинга — остальное (секреты, модель) не трогается.
pub async fn sampling_defaults(
    State(state): State<AppState>,
    Json(req): Json<crate::config::SamplingDefaults>,
) -> Response {
    if !(req.temperature >= 0.0
        && req.top_p > 0.0
        && req.top_p <= 1.0
        && req.min_p >= 0.0
        && req.min_p <= 1.0
        && req.presence_penalty >= 0.0
        && req.repetition_penalty > 0.0
        && req.max_tokens > 0)
    {
        return api_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "invalid sampling values",
        );
    }
    if let Err(e) = persist_sampling(&state.env_file, &req) {
        return api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "api_error",
            format!(".env write failed: {e}"),
        );
    }
    *state.sampling.write().expect("sampling lock") = req.clone();
    let lock = state
        .sampling_policy
        .read()
        .expect("sampling policy lock")
        .lock;
    *state.sampling_policy.write().expect("sampling policy lock") =
        crate::config::SamplingPolicy::from_defaults(&req, lock);
    Json(json!({"status": "saved"})).into_response()
}

fn persist_sampling(
    path: &std::path::Path,
    d: &crate::config::SamplingDefaults,
) -> anyhow::Result<()> {
    let keys: Vec<(String, String)> = vec![
        ("TEMPERATURE".into(), d.temperature.to_string()),
        ("TOP_P".into(), d.top_p.to_string()),
        ("TOP_K".into(), d.top_k.to_string()),
        ("MIN_P".into(), d.min_p.to_string()),
        ("PRESENCE_PENALTY".into(), d.presence_penalty.to_string()),
        (
            "REPETITION_PENALTY".into(),
            d.repetition_penalty.to_string(),
        ),
        ("MAX_TOKENS".into(), d.max_tokens.to_string()),
        ("THINKING".into(), d.thinking.to_string()),
    ];
    let existing = std::fs::read_to_string(path).unwrap_or_default();
    let mut seen = std::collections::HashSet::new();
    let mut out: Vec<String> = existing
        .lines()
        .map(|line| {
            let name = line.split('=').next().unwrap_or("").trim();
            // Старый префикс в .env срезаем: строка перезапишется чистым
            // именем, и файл постепенно мигрирует сам.
            let pure_name = name
                .strip_prefix("QWEN36_")
                .or_else(|| name.strip_prefix("YTTRI_"))
                .unwrap_or(name);
            if let Some((k, v)) = keys.iter().find(|(k, _)| k == pure_name) {
                seen.insert(k.to_string());
                format!("{k}={v}")
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

/// Сохранить пресет режима для семейства активной модели в .env
/// (MODEL_PRESETS JSON) + runtime. WebUI «Сохранить как пресет».
pub async fn sampling_preset(
    State(state): State<AppState>,
    Json(req): Json<PresetRequest>,
) -> Response {
    if req.name.trim().is_empty()
        || req.name.len() > 32
        || !(req.values.temperature >= 0.0
            && req.values.top_p > 0.0
            && req.values.top_p <= 1.0
            && req.values.min_p >= 0.0
            && req.values.min_p <= 1.0
            && req.values.presence_penalty >= 0.0
            && req.values.repetition_penalty > 0.0)
    {
        return api_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "invalid preset",
        );
    }
    let (model_path, _, _) = state
        .switcher
        .current
        .read()
        .map(|current| current.clone())
        .unwrap_or_default();
    let model_name = model_path
        .file_name()
        .map(|name| name.to_string_lossy())
        .unwrap_or_default();
    let family = crate::config::sampling_family_for_model(&model_name);
    if let Err(e) = persist_model_preset(&state.env_file, family, &req.name, &req.values) {
        return api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "api_error",
            format!(".env write failed: {e}"),
        );
    }
    state
        .presets
        .write()
        .expect("presets lock")
        .insert(req.name.clone(), req.values.clone());
    Json(json!({
        "status": "saved",
        "preset": req.name,
        "sampling_family": family,
    }))
    .into_response()
}

#[derive(serde::Deserialize)]
pub struct PresetRequest {
    name: String,
    values: crate::config::SamplingPresetValues,
}

fn persist_model_preset(
    path: &std::path::Path,
    family: &str,
    name: &str,
    values: &crate::config::SamplingPresetValues,
) -> anyhow::Result<()> {
    let existing = std::fs::read_to_string(path).unwrap_or_default();
    let existing_json = existing.lines().find_map(|line| {
        let trimmed = line.trim_start();
        ["MODEL_PRESETS=", "MODEL_PRESETS=", "YTTRI_MODEL_PRESETS="]
            .iter()
            .find_map(|prefix| trimmed.strip_prefix(prefix))
            .map(|raw| raw.trim().trim_matches(['\'', '"']))
    });
    let inherited_json = std::env::var("MODEL_PRESETS")
        .or_else(|_| std::env::var("YTTRI_MODEL_PRESETS"))
        .ok();
    let raw = existing_json.or(inherited_json.as_deref());
    let model_presets = updated_model_presets(raw, family, name, values)?;
    let json = serde_json::to_string(&model_presets)?;
    let json_line = format!("MODEL_PRESETS={json}");
    let mut replaced = false;
    let mut out: Vec<String> = existing
        .lines()
        .map(|line| {
            let trimmed = line.trim_start();
            if trimmed.starts_with("MODEL_PRESETS=")
                || trimmed.starts_with("MODEL_PRESETS=")
                || trimmed.starts_with("YTTRI_MODEL_PRESETS=")
            {
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
    // do_switch читает env в том же процессе; обновляем его после успешной записи.
    std::env::set_var("MODEL_PRESETS", json);
    Ok(())
}

fn updated_model_presets(
    raw: Option<&str>,
    family: &str,
    name: &str,
    values: &crate::config::SamplingPresetValues,
) -> anyhow::Result<crate::config::ModelSamplingPresets> {
    let mut model_presets: crate::config::ModelSamplingPresets = raw
        .filter(|raw| !raw.is_empty())
        .map(serde_json::from_str)
        .transpose()?
        .unwrap_or_default();
    model_presets
        .entry(family.to_string())
        .or_default()
        .insert(name.to_string(), values.clone());
    Ok(model_presets)
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
        #[cfg(feature = "cuda")]
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
            switcher
                .last_error
                .write()
                .expect("last_error lock")
                .clear();
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
    (StatusCode::OK, Json(json!({"status": "unloaded"}))).into_response()
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
        scan_gguf_cached(&state.models_dir, 0, &mut found, true);
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

pub async fn do_switch(
    state: &AppState,
    path: PathBuf,
    req_ctx: usize,
    req_slots: usize,
) -> anyhow::Result<()> {
    let model_name = path
        .file_name()
        .map(|name| name.to_string_lossy())
        .unwrap_or_default();
    let sampling_family = crate::config::sampling_family_for_model(&model_name);
    let model_presets = crate::config::presets_from_env(&path)?;

    // 1. VRAM-план для новой модели (dynamic: ctx до native, бюджет в движок).
    // Pending/claimed media belongs to old profile and must never survive switch.
    state.media.store.purge_all();
    let fp = vram_plan::footprint_from_gguf_cached(&path)?;
    let (ctx, slots, kv_budget_mib, kv_per_tok_mib) = match vram_plan::total_vram_mib() {
        Some(total) => {
            let plan = vram_plan::compute_dynamic(total, &fp, req_ctx, req_slots, &crate::config::moe_placement_from_env())?;
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
    #[cfg(feature = "cuda")]
    if let Some(candle_core::Device::Cuda(cuda)) =
        state.cuda_device.read().expect("cuda_device lock").as_ref()
    {
        use candle_core::backend::BackendDevice;
        let _ = cuda.synchronize();
        let _ = candle_core::cuda_backend::mem_pool::trim_default_mempool(cuda);
        let _ = cuda.synchronize();
    }
    if vram_plan::total_vram_mib().is_some() {
        // Адаптивное ожидание: старый dispatch-thread может ещё грузить адаптер
        // (35B ~60s внутри Qwen35BatchAdapter::load — shutdown там не виден).
        // Ждём роста free; если 3 полла подряд без изменений — стагнация,
        // продолжаем (KV-бюджет и так посчитан от текущего free).
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
        let mut last_free = 0usize;
        let mut stagnant = 0u8;
        loop {
            let free = vram_plan::free_vram_mib().unwrap_or(0);
            if free >= fp.weights_mib {
                break;
            }
            stagnant = if free == last_free { stagnant + 1 } else { 0 };
            last_free = free;
            if stagnant >= 3 || std::time::Instant::now() > deadline {
                eprintln!(
                    "[switch] VRAM wait done (free={free}MiB, stagnant={stagnant}), продолжаю"
                );
                break;
            }
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        }
    }

    // 3. Qwen/Ornith используют continuous batching. Остальные архитектуры
    // идут через соответствующий candle-transformers runtime под mutex.
    let architecture = crate::engine::gguf_architecture(&path)?;
    let (supported, _backend, reason) = crate::engine::architecture_capability(&architecture);
    if !supported {
        return Err(anyhow::anyhow!(
            "architecture '{architecture}' is not runtime-supported: {reason}"
        ));
    }
    let qwen35 = matches!(architecture.as_str(), "qwen35" | "qwen35moe");
    eprintln!(
        "[switch] loading {} architecture={architecture} ctx={ctx} slots={slots}",
        path.display()
    );
    let engine: Arc<dyn crate::engine::Engine> = if qwen35 && slots > 1 {
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
            studio_url: String::new(),
            api_keys: Vec::new(),
            ctx,
            slots,
            kv_budget_mib,
            kv_per_tok_mib,
            moe_experts: "auto".to_string(),
            prefix_cache_mib: 0,
            media_temp: std::env::temp_dir().join("yttri-media"),
            sampling: crate::config::SamplingDefaults::default(),
            sampling_policy: crate::config::SamplingPolicy::default(),
            presets: model_presets.clone(),
            env_file: std::path::PathBuf::from(".env"),
            batch_size: 2048,
            threads: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(8),
            gpu_layers: 999,
            kv_cache_type: "q8_f16".into(),
            mmap: true,
            rope_scale: 1.0,
            rope_scale_type: "none".into(),
            seed: 0,
            ctx_overflow: "error".into(),
            frequency_penalty: 0.0,
            max_queue: 64,
            req_timeout: 600,
            flash_attn: true,
        };
        Arc::new(crate::engine::CandleEngine::load(&cfg)?)
    };
    if qwen35 && slots > 1 {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(600);
        loop {
            if engine.ready() {
                break;
            }
            if let Some(error) = engine.load_error() {
                anyhow::bail!("model load failed: {error}");
            }
            if std::time::Instant::now() >= deadline {
                anyhow::bail!("model load timed out after 600s");
            }
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
    }
    let info = engine.model_info();
    eprintln!(
        "[switch] loaded: id={} ctx={} slots={}",
        info.id, info.context_length, info.slots
    );
    eprintln!("[sampling] family={sampling_family} presets=model-specific");
    if let Ok(mut p) = state.presets.write() {
        *p = model_presets;
    }
    state.switcher.install(engine, path, ctx, slots);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn values(temperature: f32, top_k: usize) -> crate::config::SamplingPresetValues {
        crate::config::SamplingPresetValues {
            temperature,
            top_p: 0.95,
            top_k,
            min_p: 0.0,
            presence_penalty: 0.0,
            repetition_penalty: 1.0,
        }
    }

    #[test]
    fn custom_sampling_presets_are_isolated_by_model_family() {
        let qwen = values(0.6, 20);
        let initial = serde_json::json!({
            "qwen-3.8": {"thinking-coding": qwen}
        })
        .to_string();
        let gemma = values(1.0, 64);
        let updated = updated_model_presets(Some(&initial), "gemma-4", "thinking", &gemma).unwrap();

        assert_eq!(updated["gemma-4"]["thinking"], gemma);
        assert_eq!(updated["qwen-3.8"]["thinking-coding"], qwen);
        assert!(!updated["qwen-3.8"].contains_key("thinking"));
    }
}
