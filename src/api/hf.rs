//! HuggingFace GGUF: поиск, список файлов, probe заголовка (Range-запрос),
//! фоновая загрузка в models_dir/HF. Всё за auth-мидлварой /v1.
//!
//! - GET  /v1/hf/search?q=…          — репозитории с GGUF
//! - GET  /v1/hf/files?repo=org/name — .gguf файлы с размерами
//! - GET  /v1/hf/probe?repo=…&file=… — GGUF-заголовок без полного скачивания
//! - POST /v1/hf/download {repo,file} — фоновая загрузка
//! - GET  /v1/hf/downloads            — прогресс активных/недавних загрузок

use std::collections::HashMap;
use std::io::Cursor;
use std::sync::atomic::{AtomicU64, Ordering as AOrd};
use std::sync::{Arc, Mutex};

use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::Deserialize;
use serde_json::{json, Value};

use super::{api_error, AppState};
use crate::vram_plan;

const HF: &str = "https://huggingface.co";
/// Tokenizer-метаданные в GGUF-заголовке (до ~150K токенов) занимают мегабайты —
/// 4MB не хватает («string length exceeds remaining»). 16MB покрывает 248K vocab.
const PROBE_BYTES: u64 = 16 * 1024 * 1024;
/// Макс. размер скачиваемого файла (защита от случайных 100+GB).
const MAX_DOWNLOAD_BYTES: u64 = 64 * 1024 * 1024 * 1024;

pub struct DownloadState {
    pub total: AtomicU64,
    pub done: AtomicU64,
    pub status: Mutex<String>,
    pub dest: String,
}
pub type Downloads = Arc<Mutex<HashMap<String, Arc<DownloadState>>>>;

fn client() -> reqwest::Client {
    reqwest::Client::builder()
        .user_agent("qwen36-server")
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .unwrap_or_default()
}

fn valid_repo(repo: &str) -> bool {
    let mut parts = repo.split('/');
    matches!(parts.next(), Some(a) if !a.is_empty())
        && matches!(parts.next(), Some(b) if !b.is_empty())
        && parts.next().is_none()
        && repo
            .chars()
            .all(|c| c.is_alphanumeric() || matches!(c, '/' | '-' | '_' | '.'))
}

fn valid_file(file: &str) -> bool {
    file.ends_with(".gguf")
        && !file.contains("..")
        && !file.contains(['/', '\\'])
        && file
            .chars()
            .all(|c| c.is_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

#[derive(Deserialize)]
pub struct SearchQuery {
    q: String,
}

pub async fn hf_search(Query(q): Query<SearchQuery>) -> Response {
    let q = q.q.trim();
    if q.len() < 2 {
        return api_error(StatusCode::BAD_REQUEST, "invalid_request_error", "q too short");
    }
    let url = format!("{HF}/api/models?search={q}&filter=gguf&full=true&limit=15");
    let resp = match client().get(&url).send().await {
        Ok(r) => r,
        Err(e) => {
            return api_error(
                StatusCode::BAD_GATEWAY,
                "api_error",
                format!("huggingface unreachable: {e}"),
            )
        }
    };
    let arr: Vec<Value> = match resp.json().await {
        Ok(v) => v,
        Err(e) => {
            return api_error(
                StatusCode::BAD_GATEWAY,
                "api_error",
                format!("huggingface bad json: {e}"),
            )
        }
    };
    let out: Vec<Value> = arr
        .iter()
        .map(|m| {
            let files: Vec<String> = m["siblings"]
                .as_array()
                .map(|s| {
                    s.iter()
                        .filter_map(|f| f["rfilename"].as_str())
                        .filter(|n| n.ends_with(".gguf") && !n.contains('/'))
                        .map(|s| s.to_string())
                        .collect()
                })
                .unwrap_or_default();
            json!({
                "repo": m["id"].as_str().unwrap_or(""),
                "downloads": m["downloads"].as_u64().unwrap_or(0),
                "likes": m["likes"].as_u64().unwrap_or(0),
                "gguf_files": files,
            })
        })
        .filter(|m| !m["gguf_files"].as_array().map(|f| f.is_empty()).unwrap_or(true))
        .collect();
    Json(json!({ "results": out })).into_response()
}

#[derive(Deserialize)]
pub struct FilesQuery {
    repo: String,
}

pub async fn hf_files(Query(q): Query<FilesQuery>) -> Response {
    if !valid_repo(&q.repo) {
        return api_error(StatusCode::BAD_REQUEST, "invalid_request_error", "bad repo");
    }
    let url = format!("{HF}/api/models/{}?blobs=true", q.repo);
    let resp = match client().get(&url).send().await {
        Ok(r) => r,
        Err(e) => {
            return api_error(
                StatusCode::BAD_GATEWAY,
                "api_error",
                format!("huggingface unreachable: {e}"),
            )
        }
    };
    let m: Value = match resp.json().await {
        Ok(v) => v,
        Err(e) => {
            return api_error(
                StatusCode::BAD_GATEWAY,
                "api_error",
                format!("huggingface bad json: {e}"),
            )
        }
    };
    let files: Vec<Value> = m["siblings"]
        .as_array()
        .map(|s| {
            s.iter()
                .filter(|f| {
                    f["rfilename"]
                        .as_str()
                        .map(|n| n.ends_with(".gguf") && !n.contains('/'))
                        .unwrap_or(false)
                })
                .map(|f| {
                    json!({
                        "file": f["rfilename"].as_str().unwrap_or(""),
                        "size_bytes": f["size"].as_u64().unwrap_or(0),
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    Json(json!({ "files": files })).into_response()
}

#[derive(Deserialize)]
pub struct ProbeQuery {
    repo: String,
    file: String,
}

/// GGUF-заголовок по HTTP Range: arch, native ctx, dtype-count + вердикт
/// совместимости с текущей VRAM (влезет ли с весами + минимальный KV).
pub async fn hf_probe(Query(q): Query<ProbeQuery>) -> Response {
    if !valid_repo(&q.repo) || !valid_file(&q.file) {
        return api_error(StatusCode::BAD_REQUEST, "invalid_request_error", "bad repo/file");
    }
    let url = format!("{HF}/{}/resolve/main/{}", q.repo, q.file);
    let resp = match client()
        .get(&url)
        .header("Range", format!("bytes=0-{}", PROBE_BYTES - 1))
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            return api_error(
                StatusCode::BAD_GATEWAY,
                "api_error",
                format!("huggingface unreachable: {e}"),
            )
        }
    };
    if !resp.status().is_success() && resp.status() != StatusCode::PARTIAL_CONTENT {
        return api_error(
            StatusCode::BAD_GATEWAY,
            "api_error",
            format!("huggingface probe: HTTP {}", resp.status()),
        );
    }
    let total_size = resp
        .headers()
        .get("content-range")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.rsplit('/').next())
        .and_then(|v| v.parse::<u64>().ok())
        .or_else(|| {
            resp.headers()
                .get("content-length")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.parse::<u64>().ok())
        })
        .unwrap_or(0);
    let bytes = match resp.bytes().await {
        Ok(b) => b,
        Err(e) => {
            return api_error(
                StatusCode::BAD_GATEWAY,
                "api_error",
                format!("probe read failed: {e}"),
            )
        }
    };
    let mut cursor = Cursor::new(bytes.to_vec());
    let content = match candle_core::quantized::gguf_file::Content::read(&mut cursor) {
        Ok(c) => c,
        Err(e) => {
            return api_error(
                StatusCode::UNPROCESSABLE_ENTITY,
                "invalid_request_error",
                format!("not a valid GGUF header: {e}"),
            )
        }
    };
    let arch = match content.metadata.get("general.architecture") {
        Some(candle_core::quantized::gguf_file::Value::String(s)) => s.clone(),
        _ => String::new(),
    };
    let native_ctx = content
        .metadata
        .get(&format!("{arch}.context_length"))
        .and_then(|v| match v {
            candle_core::quantized::gguf_file::Value::U32(n) => Some(*n as usize),
            candle_core::quantized::gguf_file::Value::U64(n) => Some(*n as usize),
            candle_core::quantized::gguf_file::Value::I32(n) => Some(*n as usize),
            _ => None,
        })
        .unwrap_or(0);
    let size_mib = total_size / (1024 * 1024);
    // Вердикт: веса + минимальный KV (4×8K ~ 1 GiB) + 512 MiB запаса.
    let (fits, total_vram) = match vram_plan::total_vram_mib() {
        Some(total) => {
            let total = total as u64;
            let need = size_mib + 1024 + 512;
            (need < total, total)
        }
        None => (false, 0),
    };
    Json(json!({
        "repo": q.repo,
        "file": q.file,
        "arch": arch,
        "native_ctx": native_ctx,
        "tensors": content.tensor_infos.len(),
        "size_mib": size_mib,
        "vram_total_mib": total_vram,
        "fits": if total_vram > 0 { Some(fits) } else { None },
        "recommended": { "ctx": if native_ctx > 0 { native_ctx.min(32768) } else { 8192 }, "slots": 4 },
    }))
    .into_response()
}

#[derive(Deserialize)]
pub struct DownloadRequest {
    repo: String,
    file: String,
}

pub async fn hf_download(
    State(state): State<AppState>,
    Json(req): Json<DownloadRequest>,
) -> Response {
    if !valid_repo(&req.repo) || !valid_file(&req.file) {
        return api_error(StatusCode::BAD_REQUEST, "invalid_request_error", "bad repo/file");
    }
    let key = format!("{}/{}", req.repo, req.file);
    {
        let map = state.hf_downloads.lock().expect("hf lock");
        if let Some(d) = map.get(&key) {
            if *d.status.lock().expect("status lock") == "downloading" {
                return api_error(
                    StatusCode::CONFLICT,
                    "invalid_request_error",
                    "download already in progress",
                );
            }
        }
    }
    let dir = state
        .models_dir
        .join("HF")
        .join(req.repo.replace('/', "--"));
    if let Err(e) = std::fs::create_dir_all(&dir) {
        return api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "api_error",
            format!("mkdir failed: {e}"),
        );
    }
    let dest = dir.join(&req.file);
    let part = dir.join(format!("{}.part", req.file));
    let ds = Arc::new(DownloadState {
        total: AtomicU64::new(0),
        done: AtomicU64::new(0),
        status: Mutex::new("downloading".into()),
        dest: dest.to_string_lossy().into_owned(),
    });
    state
        .hf_downloads
        .lock()
        .expect("hf lock")
        .insert(key.clone(), Arc::clone(&ds));

    tokio::spawn(async move {
        let result = download_file(&req.repo, &req.file, &part, &dest, &ds).await;
        let mut st = ds.status.lock().expect("status lock");
        *st = match result {
            Ok(()) => "done".into(),
            Err(e) => format!("error: {e}"),
        };
    });

    (StatusCode::ACCEPTED, Json(json!({"status": "downloading", "key": key}))).into_response()
}

async fn download_file(
    repo: &str,
    file: &str,
    part: &std::path::Path,
    dest: &std::path::Path,
    ds: &DownloadState,
) -> anyhow::Result<()> {
    use tokio::io::AsyncWriteExt;
    let url = format!("{HF}/{repo}/resolve/main/{file}");
    let mut resp = client().get(&url).send().await?;
    if !resp.status().is_success() {
        anyhow::bail!("HTTP {}", resp.status());
    }
    if let Some(total) = resp.content_length() {
        if total > MAX_DOWNLOAD_BYTES {
            anyhow::bail!("file too large: {} MiB", total / (1024 * 1024));
        }
        ds.total.store(total, AOrd::Relaxed);
    }
    let mut f = tokio::fs::File::create(part).await?;
    while let Some(chunk) = resp.chunk().await? {
        f.write_all(&chunk).await?;
        ds.done.fetch_add(chunk.len() as u64, AOrd::Relaxed);
    }
    f.flush().await?;
    drop(f);
    let done = ds.done.load(AOrd::Relaxed);
    let total = ds.total.load(AOrd::Relaxed);
    if total > 0 && done != total {
        std::fs::remove_file(part).ok();
        anyhow::bail!("incomplete: {done}/{total} bytes");
    }
    std::fs::rename(part, dest)?;
    Ok(())
}

pub async fn hf_downloads(State(state): State<AppState>) -> Response {
    let map = state.hf_downloads.lock().expect("hf lock");
    let out: Vec<Value> = map
        .iter()
        .map(|(k, d)| {
            json!({
                "key": k,
                "total_bytes": d.total.load(AOrd::Relaxed),
                "done_bytes": d.done.load(AOrd::Relaxed),
                "status": d.status.lock().expect("status lock").clone(),
                "dest": d.dest,
            })
        })
        .collect();
    Json(json!({ "downloads": out })).into_response()
}
