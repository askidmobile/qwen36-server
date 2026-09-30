//! Режим сайдкара Yttri: `yforge --sidecar` обслуживает встроенную модель
//! Yttri на Windows/Linux по stdio вместо HTTP.
//!
//! Протокол — подмножество v11 сайдкара MLX ([`crate::sidecar_protocol`]).
//! Промпт клиент собирает сам по шаблону модели и сам разбирает вызовы
//! инструментов; здесь только движок — слоты, очередь, префикс-кеш, сэмплер.
//! stdout — канал протокола, логи движка идут в stderr. Конфигурация — прод-
//! профиль yforge, вшитый ниже: клиент не передаёт ни одной переменной
//! окружения, env-файл и флаги сервера в этом режиме не читаются.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use futures::future::BoxFuture;
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, BufReader};
use tokio::sync::{mpsc, Notify};

use crate::engine::{CancelFlag, ChatMessage, ContentBlock, GenParams, InferenceRequest, MediaSource, StreamEvent};
use crate::engine_batched::{BatchConfig, BatchedEngine, RawRequest};
use crate::sidecar_protocol::{Event, GenerateReq, Request, PROTOCOL_VERSION};

/// Как часто запрос подаёт признак жизни. Клиент считает процесс мёртвым после
/// 60 с тишины по своему запросу, спека требует не реже раза в 5 с.
const HEARTBEAT: Duration = Duration::from_secs(2);

/// Окно штрафов в токенах — как у mlx-lm в сайдкаре MLX (`repetition_context_size`).
const PENALTY_WINDOW: usize = 20;

/// Окно контекста, если клиент его не прислал (`QWEN35_4B_CONTEXT_WINDOW` Yttri).
const DEFAULT_CTX: usize = 81_920;

/// Слотов батча: столько же у MLX-сайдкара, больше движок не умеет (MAX_SLOTS).
const DEFAULT_SLOTS: usize = 4;

/// Прод-профиль yforge (docs/DEPLOY.md, 12 ГБ, 2026-09-18). Движок читает эти
/// значения из окружения при загрузке, поэтому сайдкар выставляет их себе сам,
/// перезаписывая унаследованные: конфигурация у сайдкара одна, проверенная в
/// проде. Серверные ключи (адрес, ключи API, MODEL) здесь не нужны; окно пула
/// считает движок из `CTX` (ставится из `load`), а `GRAPH_WINDOW` прода задан
/// под его 128K и снимается. Остальные ручки движка остаются по умолчанию.
const PROFILE: &[(&str, &str)] = &[
    ("GPU_ONLY", "1"),
    ("GPU_LAYERS", "999"),
    ("VRAM_HEADROOM_MIB", "1536"),
    ("KV_CACHE_DTYPE", "q8"),
    ("CUDA_GRAPHS", "1"),
    ("KV_POOL_Q8", "1"),
    ("PREFIX_CACHE_MIB", "2048"),
    ("PREFIX_CACHE_CHECKPOINTS", "0"),
    ("PREFIX_CACHE_CHECKPOINT_CAP", "8"),
    ("PGRAPH", "on"),
    ("PGRAPH_MIN_T", "1000000"),
    ("PREFILL_CHUNK", "8192"),
    ("QK_INT8", "1"),
    ("QK_INT8_PREFILL", "1"),
    ("PREFIX_CACHE_TAIL_SPLIT", "1"),
    ("PREFIX_CACHE_TAIL_ALIGN", "64"),
    ("PREFIX_CACHE_FULL_HIT", "0"),
    ("PREFIX_CACHE_POOL_BACKED", "1"),
    ("YTTRI_ATTN_PREP_FUSED", "1"),
    ("MTP", "0"),
];

/// Что сайдкару нужно от движка: `BatchedEngine` в работе, мок в тестах.
#[async_trait::async_trait]
pub trait RawEngine: Send + Sync {
    async fn generate_raw(&self, req: RawRequest) -> Result<mpsc::Receiver<StreamEvent>>;
    /// Чат-путь движка (шаблон модели, медиа) — для `describe_image`.
    async fn generate_chat(&self, req: InferenceRequest) -> Result<mpsc::Receiver<StreamEvent>>;
    /// Рабочие шаги планировщика: растут — движок жив.
    fn steps(&self) -> u64;
    /// Запросы в слотах и в очереди.
    fn in_flight(&self) -> usize;
}

#[async_trait::async_trait]
impl RawEngine for BatchedEngine {
    async fn generate_raw(&self, req: RawRequest) -> Result<mpsc::Receiver<StreamEvent>> {
        BatchedEngine::generate_raw(self, req).await
    }
    async fn generate_chat(&self, req: InferenceRequest) -> Result<mpsc::Receiver<StreamEvent>> {
        crate::engine::Engine::generate(self, req).await
    }
    fn steps(&self) -> u64 {
        BatchedEngine::steps(self)
    }
    fn in_flight(&self) -> usize {
        BatchedEngine::in_flight(self)
    }
}

pub struct LoadSpec {
    pub model_path: String,
    pub slots: usize,
    pub ctx: usize,
    /// `mmproj` для режима `vlm`; `None` — только текст.
    pub vision: Option<std::path::PathBuf>,
}

pub struct Loaded {
    pub engine: Arc<dyn RawEngine>,
    pub slots: usize,
    pub ctx: usize,
    pub vram: Vram,
    pub vision: bool,
}

pub type Loader = Arc<dyn Fn(LoadSpec) -> BoxFuture<'static, Result<Loaded>> + Send + Sync>;

/// Видеопамять, занятая с загрузки модели: свободная до загрузки минус
/// свободная сейчас. Оценка — чужие процессы на той же карте её сдвигают.
#[derive(Default)]
pub struct Vram {
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    dev: Option<candle_core::Device>,
    free_before_mib: u64,
}

impl Vram {
    fn free_mib(&self) -> Option<u64> {
        #[cfg(feature = "cuda")]
        {
            let dev = self.dev.as_ref()?.as_cuda_device().ok()?;
            candle_core::cuda_backend::mem_pool::free_mib(dev).ok()
        }
        #[cfg(not(feature = "cuda"))]
        {
            None
        }
    }

    fn capture(dev: Option<candle_core::Device>) -> Self {
        let mut vram = Self {
            dev,
            free_before_mib: 0,
        };
        vram.free_before_mib = vram.free_mib().unwrap_or(0);
        vram
    }

    fn used_mib(&self) -> u32 {
        self.free_mib()
            .map_or(0, |free| self.free_before_mib.saturating_sub(free) as u32)
    }

    /// Вернуть драйверу свободные блоки пула памяти (незанятые буферы префила).
    fn trim(&self) {
        #[cfg(feature = "cuda")]
        if let Some(Ok(dev)) = self.dev.as_ref().map(|d| d.as_cuda_device()) {
            let _ = candle_core::cuda_backend::mem_pool::trim_default_mempool(dev);
        }
    }
}

/// Точка входа `yforge --sidecar`.
pub async fn run() -> Result<()> {
    let loader: Loader =
        Arc::new(|spec: LoadSpec| -> BoxFuture<'static, Result<Loaded>> { Box::pin(load_batched(spec)) });
    serve(tokio::io::stdin(), tokio::io::stdout(), loader).await
}

async fn load_batched(spec: LoadSpec) -> Result<Loaded> {
    for (key, value) in PROFILE {
        std::env::set_var(key, value);
    }
    std::env::remove_var("GRAPH_WINDOW");
    std::env::set_var("CTX", spec.ctx.to_string());
    #[cfg(feature = "cuda")]
    crate::engine::cuda_prefer_blocking_sync();
    #[cfg(feature = "cuda")]
    let dev = candle_core::Device::new_cuda(0).ok();
    #[cfg(not(feature = "cuda"))]
    let dev = None;
    let vram = Vram::capture(dev);

    // План памяти — как у сервера (Config::apply_vram_plan): окно не режется,
    // общий бюджет KV движок держит сам допуском и очередью.
    let (mut ctx, mut slots, mut kv_budget_mib, mut kv_per_tok_mib) = (spec.ctx, spec.slots, 0.0, 0.0);
    if let Some(total) = crate::vram_plan::total_vram_mib() {
        let path = std::path::Path::new(&spec.model_path);
        let fp = crate::vram_plan::footprint_from_gguf_with(path, "auto")?;
        let plan = crate::vram_plan::compute_dynamic(total, &fp, ctx, slots, "auto")?;
        eprintln!("{}", plan.report);
        (ctx, slots, kv_budget_mib, kv_per_tok_mib) =
            (plan.ctx, plan.slots, plan.kv_budget_mib, plan.kv_per_tok_mib);
    }
    let cfg = BatchConfig {
        model_path: spec.model_path,
        slots,
        max_queue: 64,
        req_timeout: Duration::from_secs(600),
        context_length: ctx,
        kv_budget_mib,
        kv_per_tok_mib,
        prefix_cache_mib: 2048,
        // У Yttri на диске обычный mmproj F16 — он и есть эталон Q8-артефакта.
        vision_reference: true,
    };
    let media = Arc::new(crate::media::MediaService::new(Default::default())?);
    let vision = spec.vision.is_some();
    let engine = BatchedEngine::load(cfg, media, spec.vision, None).await?;
    Ok(Loaded {
        engine,
        slots,
        ctx,
        vram,
        vision,
    })
}

/// Vision-артефакт рядом с моделью: Yttri кладёт `mmproj-*.gguf` в её каталог.
fn find_mmproj(model_path: &str) -> Option<std::path::PathBuf> {
    let dir = std::path::Path::new(model_path).parent()?;
    std::fs::read_dir(dir).ok()?.flatten().map(|e| e.path()).find(|p| {
        p.file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.starts_with("mmproj") && n.ends_with(".gguf"))
    })
}

/// Картинка с диска + вопрос → чат-запрос. Жадно и без штрафов, как у MLX.
fn image_request(
    path: &str,
    prompt: String,
    max_tokens: usize,
    cancel: CancelFlag,
) -> Result<InferenceRequest> {
    use base64::Engine as _;
    let bytes = std::fs::read(path).map_err(|e| anyhow::anyhow!("read image {path}: {e}"))?;
    Ok(InferenceRequest {
        messages: vec![ChatMessage {
            role: "user".into(),
            content: vec![
                ContentBlock::Media {
                    kind: crate::media::MediaKind::Image,
                    // Тип картинки движок определяет сам по сигнатуре.
                    source: MediaSource::DataUrl {
                        declared_mime: "image/*".into(),
                        base64: base64::engine::general_purpose::STANDARD.encode(bytes),
                    },
                },
                ContentBlock::Text { text: prompt },
            ],
            ..Default::default()
        }],
        params: GenParams {
            temperature: 0.0,
            presence_penalty: 0.0,
            repetition_penalty: 1.0,
            max_tokens,
            thinking: false,
            ..Default::default()
        },
        owner: crate::media::OwnerDigest::from_key("yttri-sidecar"),
        cancel,
        tools: None,
        reasoning_effort: None,
    })
}

/// Запрос в работе: флаг отмены движка и будильник пересылки.
struct Running {
    cancel: CancelFlag,
    wake: Arc<Notify>,
}

pub async fn serve<R, W>(input: R, output: W, loader: Loader) -> Result<()>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin + Send + 'static,
{
    let (tx, mut events) = mpsc::unbounded_channel::<Event>();
    let writer = tokio::spawn(async move {
        let mut output = output;
        while let Some(event) = events.recv().await {
            let mut line = serde_json::to_vec(&event).expect("event serializes");
            line.push(b'\n');
            if output.write_all(&line).await.is_err() || output.flush().await.is_err() {
                break;
            }
        }
    });

    let mut input = BufReader::new(input);
    let mut loaded: Option<Arc<Loaded>> = None;
    let running: Arc<std::sync::Mutex<HashMap<u64, Running>>> = Default::default();
    let mut line = Vec::new();
    loop {
        line.clear();
        // EOF — клиента больше нет: выходим и отдаём видеопамять вместе с процессом.
        if input.read_until(b'\n', &mut line).await? == 0 {
            break;
        }
        let json: serde_json::Value = match serde_json::from_slice(&line) {
            Ok(json) => json,
            Err(e) => {
                let _ = tx.send(Event::Error {
                    message: format!("bad frame: {e}"),
                    req_id: 0,
                });
                continue;
            }
        };
        // Блоб кадра (у ASR и малых моделей MLX) здесь не нужен, но его байты
        // надо дочитать, иначе они уедут в следующий заголовок.
        if let Some(len) = json.get("blob_bytes").and_then(|v| v.as_u64()) {
            tokio::io::copy(&mut (&mut input).take(len), &mut tokio::io::sink()).await?;
        }
        let op = json.get("op").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let req_id = json.get("req_id").and_then(|v| v.as_u64()).unwrap_or(0);
        let request: Request = match serde_json::from_value(json) {
            Ok(request) => request,
            Err(e) => {
                let _ = tx.send(Event::Error {
                    message: format!("unsupported op {op:?}: {e}"),
                    req_id,
                });
                continue;
            }
        };
        match request {
            Request::Load {
                model_path,
                mode,
                batch_slots,
                max_seq,
            } => {
                if loaded.is_none() {
                    if model_path.is_empty() {
                        let _ = tx.send(Event::Error {
                            message: "load: model_path is empty".into(),
                            req_id: 0,
                        });
                        continue;
                    }
                    let vision = match (mode.as_str(), find_mmproj(&model_path)) {
                        ("vlm", None) => {
                            let _ = tx.send(Event::Error {
                                message: "load: mode vlm, but no mmproj-*.gguf next to the model".into(),
                                req_id: 0,
                            });
                            continue;
                        }
                        ("vlm", found) => found,
                        _ => None,
                    };
                    let spec = LoadSpec {
                        vision,
                        model_path,
                        slots: match batch_slots as usize {
                            0 => DEFAULT_SLOTS,
                            n => n.min(DEFAULT_SLOTS),
                        },
                        ctx: match max_seq as usize {
                            0 => DEFAULT_CTX,
                            n => n,
                        },
                    };
                    match loader(spec).await {
                        Ok(l) => loaded = Some(Arc::new(l)),
                        Err(e) => {
                            eprintln!("[sidecar] load failed: {e:#}");
                            let _ = tx.send(Event::Error {
                                message: format!("load: {e:#}"),
                                req_id: 0,
                            });
                            continue;
                        }
                    }
                }
                let l = loaded.as_ref().expect("loaded above");
                let _ = tx.send(Event::Loaded {
                    ok: true,
                    batch_slots: l.slots as u32,
                    supports_progress: true,
                    kv_bits: 8,
                    max_seq: l.ctx as u32,
                });
            }
            Request::Health => {
                let (slots, ctx, in_flight, vram) = loaded.as_ref().map_or((0, 0, 0, 0), |l| {
                    (l.slots, l.ctx, l.engine.in_flight(), l.vram.used_mib())
                });
                let active = in_flight.min(slots);
                let _ = tx.send(Event::Health {
                    ok: true,
                    protocol_version: PROTOCOL_VERSION,
                    model_loaded: loaded.is_some(),
                    batch_slots: slots as u32,
                    max_seq: ctx as u32,
                    batch_queue: (in_flight - active) as u32,
                    batch_active: active as u32,
                    active_memory_mb: vram,
                    supports_progress: true,
                    vision: loaded.as_ref().is_some_and(|l| l.vision),
                });
            }
            Request::Generate(req) => {
                let Some(l) = loaded.clone() else {
                    let _ = tx.send(Event::Error {
                        message: "model not loaded".into(),
                        req_id: req.req_id,
                    });
                    continue;
                };
                let run = Running {
                    cancel: CancelFlag::default(),
                    wake: Arc::new(Notify::new()),
                };
                let (cancel, wake) = (run.cancel.clone(), run.wake.clone());
                running
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .insert(req.req_id, run);
                let (tx, running) = (tx.clone(), running.clone());
                tokio::spawn(async move {
                    let id = req.req_id;
                    forward(l, req, cancel, wake, &tx).await;
                    running.lock().unwrap_or_else(|e| e.into_inner()).remove(&id);
                });
            }
            Request::DescribeImage {
                req_id,
                image_path,
                prompt,
                max_tokens,
            } => {
                let Some(l) = loaded.clone().filter(|l| l.vision) else {
                    let _ = tx.send(Event::Error {
                        message: "vision is not loaded (load with mode=vlm)".into(),
                        req_id,
                    });
                    continue;
                };
                let run = Running {
                    cancel: CancelFlag::default(),
                    wake: Arc::new(Notify::new()),
                };
                let (cancel, wake) = (run.cancel.clone(), run.wake.clone());
                running
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .insert(req_id, run);
                let (tx, running) = (tx.clone(), running.clone());
                tokio::spawn(async move {
                    let started = Instant::now();
                    let rx = match image_request(&image_path, prompt, max_tokens, cancel.clone()) {
                        Ok(req) => l.engine.generate_chat(req).await,
                        Err(e) => Err(e),
                    };
                    pump(&l, req_id, started, rx, cancel, wake, &tx).await;
                    running.lock().unwrap_or_else(|e| e.into_inner()).remove(&req_id);
                });
            }
            Request::Cancel { req_id } => {
                // Чужой или уже закончившийся id — игнор, как у MLX.
                if let Some(run) = running
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .get(&req_id)
                {
                    run.cancel.cancel();
                    run.wake.notify_one();
                }
            }
            Request::Trim => {
                let (freed, used) = loaded.as_ref().map_or((0, 0), |l| {
                    let before = l.vram.used_mib();
                    l.vram.trim();
                    let after = l.vram.used_mib();
                    (before.saturating_sub(after), after)
                });
                let _ = tx.send(Event::Trimmed {
                    freed_mb: freed,
                    active_memory_mb: used,
                });
            }
            Request::Unload => break,
        }
    }
    // Незавершённые запросы гасим: процесс уходит, движок с ним.
    for run in running.lock().unwrap_or_else(|e| e.into_inner()).values() {
        run.cancel.cancel();
    }
    drop(tx);
    let _ = writer.await;
    Ok(())
}

/// Запрос `generate`: сырой промпт в движок, дальше — [`pump`].
async fn forward(
    l: Arc<Loaded>,
    req: GenerateReq,
    cancel: CancelFlag,
    wake: Arc<Notify>,
    tx: &mpsc::UnboundedSender<Event>,
) {
    let id = req.req_id;
    let started = Instant::now();
    let raw = RawRequest {
        prompt: req.prompt,
        prompt_ids: (!req.prompt_tokens.is_empty()).then_some(req.prompt_tokens),
        prefix_tokens: (req.prefix_token_count > 0).then_some(req.prefix_token_count),
        prefix_text: req.prefix,
        params: GenParams {
            temperature: req.temperature,
            top_p: req.top_p,
            top_k: req.top_k,
            min_p: req.min_p,
            presence_penalty: req.presence_penalty,
            repetition_penalty: req.repetition_penalty,
            max_tokens: req.max_tokens,
            stop: req.stop_sequences,
            seed: None,
            // Фазу рассуждений generate_raw берёт из хвоста промпта.
            thinking: false,
            logprobs: None,
            suppress_tokens: req.suppress_tokens,
            max_thinking_tokens: req.max_thinking_tokens,
            penalty_window: PENALTY_WINDOW,
        },
        max_prompt_tokens: req.max_prompt_tokens,
        cancel: cancel.clone(),
    };
    let rx = l.engine.generate_raw(raw).await;
    pump(&l, id, started, rx, cancel, wake, tx).await;
}

/// Поток движка → `chunk`*/`progress`* → `done` или `error`.
async fn pump(
    l: &Loaded,
    id: u64,
    started: Instant,
    rx: Result<mpsc::Receiver<StreamEvent>>,
    cancel: CancelFlag,
    wake: Arc<Notify>,
    tx: &mpsc::UnboundedSender<Event>,
) {
    let mut rx = match rx {
        Ok(rx) => rx,
        Err(e) => {
            let _ = tx.send(Event::Error {
                message: format!("{e:#}"),
                req_id: id,
            });
            return;
        }
    };
    let mut text = String::new();
    // Дельт движка: одна на шаг декода слота, то есть почти токены.
    let mut deltas = 0usize;
    let mut first_at: Option<Instant> = None;
    let mut steps = l.engine.steps();
    let mut heartbeat = tokio::time::interval_at(tokio::time::Instant::now() + HEARTBEAT, HEARTBEAT);
    let cancelled = |text: String, deltas: usize, first_at: Option<Instant>| Event::Done {
        req_id: id,
        text,
        tokens: deltas,
        time_ms: started.elapsed().as_millis() as u64,
        prefill_ms: first_at.map_or(0, |t| (t - started).as_millis() as u64),
        prompt_tokens: 0,
        cache_hit: false,
        cached_tokens: 0,
        prefill_tokens: 0,
        stop_reason: "cancelled",
        active_memory_mb: l.vram.used_mib(),
    };
    loop {
        tokio::select! {
            event = rx.recv() => match event {
                Some(StreamEvent::Delta { text: piece, .. }) => {
                    if piece.is_empty() {
                        continue;
                    }
                    deltas += 1;
                    first_at.get_or_insert_with(Instant::now);
                    text.push_str(&piece);
                    let _ = tx.send(Event::Chunk { req_id: id, text: piece });
                }
                Some(StreamEvent::Done { finish_reason, usage, .. }) => {
                    let time_ms = started.elapsed().as_millis() as u64;
                    let _ = tx.send(Event::Done {
                        req_id: id,
                        text,
                        tokens: usage.completion_tokens,
                        time_ms,
                        prefill_ms: first_at.map_or(time_ms, |t| (t - started).as_millis() as u64),
                        prompt_tokens: usage.prompt_tokens,
                        cache_hit: usage.cached_tokens > 0,
                        cached_tokens: usage.cached_tokens,
                        prefill_tokens: usage.prompt_tokens.saturating_sub(usage.cached_tokens),
                        // Движок: "stop" — EOS или стоп-строка, "length" — лимит
                        // ответа/рассуждений или сторож зацикливания.
                        stop_reason: if finish_reason == "length" { "length" } else { "stop" },
                        active_memory_mb: l.vram.used_mib(),
                    });
                    return;
                }
                Some(StreamEvent::Error(message)) => {
                    let _ = tx.send(Event::Error { message, req_id: id });
                    return;
                }
                // Отменённый слот движок закрывает без Done.
                None => {
                    let _ = tx.send(if cancel.is_cancelled() {
                        cancelled(text, deltas, first_at)
                    } else {
                        Event::Error { message: "engine stopped".into(), req_id: id }
                    });
                    return;
                }
            },
            // Отмена отвечает сразу: слот освободится на ближайшей границе шага,
            // а ждать его, стоя в очереди за длинной генерацией, клиенту незачем.
            _ = wake.notified() => {
                let _ = tx.send(cancelled(text, deltas, first_at));
                return;
            }
            // Признак жизни — только пока движок делает шаги: зависший движок
            // должен замолчать, чтобы клиент перезапустил процесс.
            _ = heartbeat.tick() => {
                let now = l.engine.steps();
                if now != steps {
                    steps = now;
                    let rate = first_at.map_or(0.0, |t| deltas as f32 / t.elapsed().as_secs_f32().max(1e-3));
                    let _ = tx.send(Event::Progress {
                        req_id: id,
                        phase: if first_at.is_some() { "decode" } else { "prefill" },
                        processed: deltas,
                        total: 0,
                        tokens_per_sec: rate,
                    });
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::GenerationUsage;
    use serde_json::{json, Value};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Mutex;
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, DuplexStream};

    /// Мок: `echo:<текст>` — отдаёт текст по словам и Done; `hang` — молчит
    /// до отмены; `overflow` — ошибка переполнения. Пока `moving`, каждый
    /// опрос счётчика шагов видит новый шаг.
    #[derive(Default)]
    struct Mock {
        steps: AtomicU64,
        moving: std::sync::atomic::AtomicBool,
        seen: Mutex<Vec<RawRequest>>,
    }

    #[async_trait::async_trait]
    impl RawEngine for Mock {
        async fn generate_raw(&self, req: RawRequest) -> Result<mpsc::Receiver<StreamEvent>> {
            let (tx, rx) = mpsc::channel(16);
            let prompt = req.prompt.clone();
            let cancel = req.cancel.clone();
            self.seen.lock().unwrap().push(req);
            if prompt == "overflow" {
                return Err(crate::engine::ContextOverflow {
                    prompt_tokens: 20000,
                    limit: 16384,
                }
                .into());
            }
            tokio::spawn(async move {
                if let Some(text) = prompt.strip_prefix("echo:") {
                    for word in text.split_inclusive(' ') {
                        let _ = tx
                            .send(StreamEvent::Delta {
                                text: word.into(),
                                logprobs: None,
                            })
                            .await;
                    }
                    let _ = tx
                        .send(StreamEvent::Done {
                            finish_reason: "stop".into(),
                            usage: GenerationUsage {
                                prompt_tokens: 7,
                                completion_tokens: 3,
                                cached_tokens: 4,
                                ..Default::default()
                            },
                            ended_in_thinking: false,
                        })
                        .await;
                } else {
                    while !cancel.is_cancelled() {
                        tokio::time::sleep(Duration::from_millis(10)).await;
                    }
                }
            });
            Ok(rx)
        }
        async fn generate_chat(&self, req: InferenceRequest) -> Result<mpsc::Receiver<StreamEvent>> {
            let (tx, rx) = mpsc::channel(4);
            let has_image = req.messages[0]
                .content
                .iter()
                .any(|b| matches!(b, ContentBlock::Media { .. }));
            let _ = tx.try_send(StreamEvent::Delta {
                text: if has_image { "a cat".into() } else { "no image".into() },
                logprobs: None,
            });
            let _ = tx.try_send(StreamEvent::Done {
                finish_reason: "stop".into(),
                usage: GenerationUsage {
                    prompt_tokens: 300,
                    completion_tokens: 2,
                    ..Default::default()
                },
                ended_in_thinking: false,
            });
            Ok(rx)
        }
        fn steps(&self) -> u64 {
            if self.moving.load(Ordering::Relaxed) {
                self.steps.fetch_add(1, Ordering::Relaxed) + 1
            } else {
                self.steps.load(Ordering::Relaxed)
            }
        }
        fn in_flight(&self) -> usize {
            0
        }
    }

    struct Harness {
        to: DuplexStream,
        from: tokio::io::Lines<BufReader<DuplexStream>>,
        mock: Arc<Mock>,
    }

    impl Harness {
        fn start() -> Self {
            let (to, input) = tokio::io::duplex(1 << 16);
            let (output, from) = tokio::io::duplex(1 << 16);
            let mock = Arc::new(Mock::default());
            let engine = mock.clone();
            let loader: Loader = Arc::new(move |spec: LoadSpec| -> BoxFuture<'static, Result<Loaded>> {
                let engine: Arc<dyn RawEngine> = engine.clone();
                Box::pin(async move {
                    Ok(Loaded {
                        engine,
                        slots: spec.slots,
                        ctx: spec.ctx,
                        vram: Vram::default(),
                        vision: spec.vision.is_some(),
                    })
                })
            });
            tokio::spawn(serve(input, output, loader));
            Self {
                to,
                from: BufReader::new(from).lines(),
                mock,
            }
        }

        async fn send(&mut self, frame: Value) {
            let mut line = serde_json::to_vec(&frame).unwrap();
            line.push(b'\n');
            self.to.write_all(&line).await.unwrap();
        }

        async fn next(&mut self) -> Value {
            let line = tokio::time::timeout(Duration::from_secs(30), self.from.next_line())
                .await
                .expect("событие не пришло")
                .unwrap()
                .expect("поток закрыт");
            serde_json::from_str(&line).unwrap()
        }

        async fn load(&mut self) {
            self.send(json!({"op": "load", "model_dir": "/m", "model_path": "/m/q.gguf", "mode": "text", "batch_slots": 4, "max_seq": 32768, "cache_slots": 4})).await;
            let loaded = self.next().await;
            assert_eq!(loaded["event"], "loaded");
            assert_eq!(loaded["ok"], true);
            assert_eq!(loaded["batch_slots"], 4);
            assert_eq!(loaded["kv_bits"], 8);
            assert_eq!(loaded["max_seq"], 32768);
        }
    }

    #[tokio::test]
    async fn handshake_generate_and_params() {
        let mut h = Harness::start();
        h.send(json!({"op": "generate", "req_id": 1, "prompt": "echo:x", "max_tokens": 5, "temperature": 0.0, "top_p": 1.0})).await;
        let early = h.next().await;
        assert_eq!(early["event"], "error");
        assert_eq!(early["req_id"], 1);
        assert_eq!(early["message"], "model not loaded");

        h.load().await;
        h.send(json!({"op": "health"})).await;
        let health = h.next().await;
        assert_eq!(health["event"], "health");
        assert_eq!(health["protocol_version"], PROTOCOL_VERSION);
        assert_eq!(health["model_loaded"], true);
        assert_eq!(health["supports_progress"], true);

        h.send(json!({
            "op": "generate", "req_id": 2, "prompt": "echo:Нашёл три заметки",
            "prompt_tokens": [1, 2, 3], "prefix_token_count": 2, "max_tokens": 64,
            "temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5,
            "suppress_tokens": [248068], "stop_sequences": ["</tool_call>"],
            "max_thinking_tokens": 512, "max_prompt_tokens": 16384
        }))
        .await;
        let mut text = String::new();
        let done = loop {
            let event = h.next().await;
            match event["event"].as_str().unwrap() {
                "chunk" => text.push_str(event["text"].as_str().unwrap()),
                "done" => break event,
                other => panic!("неожиданное событие {other}: {event}"),
            }
        };
        assert_eq!(text, "Нашёл три заметки");
        assert_eq!(done["req_id"], 2);
        assert_eq!(done["text"], "Нашёл три заметки");
        assert_eq!(done["stop_reason"], "stop");
        assert_eq!(done["tokens"], 3);
        assert_eq!(done["prompt_tokens"], 7);
        assert_eq!(done["cache_hit"], true);
        assert_eq!(done["prefill_tokens"], 3);

        let seen = h.mock.seen.lock().unwrap();
        let raw = seen.last().unwrap();
        assert_eq!(raw.prompt_ids.as_deref(), Some(&[1, 2, 3][..]));
        assert_eq!(raw.prefix_tokens, Some(2));
        assert_eq!(raw.max_prompt_tokens, 16384);
        let p = &raw.params;
        assert_eq!((p.max_tokens, p.max_thinking_tokens, p.top_k), (64, 512, 20));
        assert_eq!(p.suppress_tokens, vec![248068]);
        assert_eq!(p.stop, vec!["</tool_call>".to_string()]);
        assert_eq!(p.penalty_window, PENALTY_WINDOW);
        assert_eq!(p.repetition_penalty, 1.0, "нет поля — штраф выключен");
    }

    #[tokio::test]
    async fn unknown_op_blob_and_overflow_keep_process_alive() {
        let mut h = Harness::start();
        h.load().await;
        h.send(json!({"op": "asr_load", "req_id": 9, "kind": "gigaam", "path": "/x"})).await;
        let err = h.next().await;
        assert_eq!(err["event"], "error");
        assert_eq!(err["req_id"], 9);
        assert!(err["message"].as_str().unwrap().contains("asr_load"), "{err}");

        // Кадр с блобом: байты после заголовка не должны сломать следующий.
        h.to.write_all(b"{\"op\":\"asr_transcribe\",\"req_id\":10,\"blob_bytes\":5}\n\x00\x01\n{}")
            .await
            .unwrap();
        assert_eq!(h.next().await["req_id"], 10);

        h.send(json!({"op": "generate", "req_id": 11, "prompt": "overflow", "max_tokens": 8, "temperature": 0.0, "top_p": 1.0})).await;
        let err = h.next().await;
        assert_eq!(err["req_id"], 11);
        assert!(
            err["message"].as_str().unwrap().starts_with("промпт не помещается в контекст:"),
            "{err}"
        );

        h.send(json!({"op": "trim"})).await;
        assert_eq!(h.next().await["event"], "trimmed");
        h.send(json!({"op": "health"})).await;
        assert_eq!(h.next().await["event"], "health");
    }

    #[tokio::test]
    async fn cancel_ends_with_done_cancelled() {
        let mut h = Harness::start();
        h.load().await;
        h.send(json!({"op": "generate", "req_id": 5, "prompt": "hang", "max_tokens": 8, "temperature": 0.0, "top_p": 1.0})).await;
        // Чужой id игнорируется, свой — отвечает done{cancelled}.
        h.send(json!({"op": "cancel", "req_id": 404})).await;
        h.send(json!({"op": "cancel", "req_id": 5})).await;
        let done = h.next().await;
        assert_eq!(done["event"], "done");
        assert_eq!(done["req_id"], 5);
        assert_eq!(done["stop_reason"], "cancelled");
        let seen = h.mock.seen.lock().unwrap();
        assert!(seen[0].cancel.is_cancelled(), "движок получил отмену");
    }

    /// Vision: `mode=vlm` находит `mmproj-*.gguf` рядом с моделью; подпись идёт
    /// чат-путём с картинкой. В текстовом режиме `describe_image` — ошибка.
    #[tokio::test]
    async fn describe_image_needs_vlm_mode_and_mmproj() {
        let dir = std::env::temp_dir().join(format!("yforge-sidecar-vlm-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let model = dir.join("model.gguf");
        let image = dir.join("cat.png");
        std::fs::write(&model, b"gguf").unwrap();
        std::fs::write(&image, b"\x89PNG\r\n\x1a\n").unwrap();
        let model = model.to_string_lossy().into_owned();
        let image = image.to_string_lossy().into_owned();

        // Без mmproj режим vlm не поднимается, процесс жив.
        let mut h = Harness::start();
        h.send(json!({"op": "load", "model_path": model, "mode": "vlm"})).await;
        let err = h.next().await;
        assert_eq!(err["event"], "error");
        assert!(err["message"].as_str().unwrap().contains("mmproj"), "{err}");
        // Текстовый режим: подписи нет.
        h.send(json!({"op": "load", "model_path": model, "mode": "text"})).await;
        assert_eq!(h.next().await["event"], "loaded");
        h.send(json!({"op": "describe_image", "req_id": 3, "image_path": image, "prompt": "Опиши", "max_tokens": 64})).await;
        let err = h.next().await;
        assert_eq!((err["event"].as_str(), err["req_id"].as_u64()), (Some("error"), Some(3)));

        std::fs::write(dir.join("mmproj-Qwen3.5-4B-F16.gguf"), b"gguf").unwrap();
        let mut h = Harness::start();
        h.send(json!({"op": "load", "model_path": model, "mode": "vlm"})).await;
        assert_eq!(h.next().await["event"], "loaded");
        h.send(json!({"op": "health"})).await;
        assert_eq!(h.next().await["vision"], true);
        h.send(json!({"op": "describe_image", "req_id": 4, "image_path": image, "prompt": "Опиши", "max_tokens": 64})).await;
        let chunk = h.next().await;
        assert_eq!(chunk["event"], "chunk");
        let done = h.next().await;
        assert_eq!(done["event"], "done");
        assert_eq!(done["text"], "a cat");
        assert_eq!(done["prompt_tokens"], 300);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test(start_paused = true)]
    async fn progress_only_while_engine_steps() {
        let mut h = Harness::start();
        h.load().await;
        // Движок шагает — приходит progress по запросу.
        h.mock.moving.store(true, Ordering::Relaxed);
        h.send(json!({"op": "generate", "req_id": 7, "prompt": "hang", "max_tokens": 8, "temperature": 0.0, "top_p": 1.0})).await;
        let progress = h.next().await;
        assert_eq!(progress["event"], "progress");
        assert_eq!(progress["req_id"], 7);
        assert_eq!(progress["phase"], "prefill");
        // Движок встал — тишина, хотя сайдкар жив и отвечает на health.
        h.mock.moving.store(false, Ordering::Relaxed);
        tokio::time::advance(HEARTBEAT * 3).await;
        h.send(json!({"op": "health"})).await;
        assert_eq!(h.next().await["event"], "health");
        h.send(json!({"op": "unload"})).await;
    }
}
