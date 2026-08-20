//! `BatchedEngine` — 4-слотовый engine поверх `BatchScheduler` из
//! `qwen35-batch` (BD-007). Дизайн: docs/batch-integration.md.
//!
//! Реальный `BatchScheduler<Qwen35BatchAdapter>`:
//! - TODO-F1: `Qwen35BatchAdapter::load(gguf, device, slots)` — загружает GGUF.
//! - TODO-F2: real tokenize (`build_chatml_text` + `encode_no_think`) + sliding window (BD-017).
//! - TODO-F3: incremental `decode_text` + stop-holdback в `drain_slot`.
//! - TODO-F4: `sched.submit`/`step_with`/`slots_mut`.
//! - TODO-F5 (форк): `Sampler::sample_indexed(slot, generated, logits)` — per-request params.
//! - TODO-F6 (форк): `BatchScheduler::slots_mut()` — сбор Finished.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail};
use qwen35_batch::model::{MultimodalPrefill, Sampler as ForkSampler, SamplerCheckpoint};
use qwen35_batch::real::multimodal::{build_position_plan, MediaKind as PackedMediaKind};
use qwen35_batch::real::tokenizer::{self, ChatContent, ChatMsg, MultimodalChatMsg};
use qwen35_batch::real::Qwen35BatchAdapter;
use qwen35_batch::scheduler::{BatchScheduler, StepOutcome};
use qwen35_batch::slot::SlotStatus;
use tokio::sync::mpsc;

use crate::engine::{
    floor_char_boundary, model_id_from_filename, quant_from_filename, select_device, trim_messages,
    FindAny,
};
use crate::engine_types::{
    ChatMessage, Engine, GenParams, GenerationUsage, InferenceRequest, MediaUsage, ModelInfo,
    StreamEvent,
};
use crate::media::prepare::{PreparedContentBlock, PreparedLease};
use crate::sampler::{self, Rng};

/// Макс. слотов = DECODE_BATCH_CAPACITY форка.
pub const MAX_SLOTS: usize = 4;
/// Capacity per-slot канала StreamEvent.
const SLOT_CHAN_CAP: usize = 64;
/// Запас токенов под погрешность per-message оценки sliding window.
const TRIM_MARGIN: usize = 16;

#[derive(Debug, Clone)]
pub struct BatchConfig {
    pub model_path: String,
    pub slots: usize,
    pub max_queue: usize,
    pub req_timeout: Duration,
    pub context_length: usize,
    /// Общий KV-бюджет всех слотов (MiB). 0 = без лимита.
    pub kv_budget_mib: f64,
    /// MiB KV на токен на слот.
    pub kv_per_tok_mib: f64,
    /// Бюджет prefix cache (MiB, snapshot'ы state). 0 = выключен.
    pub prefix_cache_mib: usize,
}

impl BatchConfig {
    pub fn from_env() -> Self {
        let get = |k: &str| {
            std::env::var(k)
                .or_else(|_| std::env::var(format!("QWEN36_{k}")))
                .ok()
        };
        let num = |k: &str, d: usize| get(k).and_then(|v| v.parse().ok()).unwrap_or(d);
        let mut slots = num("SLOTS", 4);
        if slots > MAX_SLOTS {
            eprintln!("[batch] SLOTS={slots} > {MAX_SLOTS} (DECODE_BATCH_CAPACITY), clamp");
            slots = MAX_SLOTS;
        }
        Self {
            model_path: get("MODEL").unwrap_or_else(|| "models/qwen36-27b-q2_k_xl.gguf".into()),
            slots,
            max_queue: num("MAX_QUEUE", 64),
            req_timeout: Duration::from_secs(num("REQ_TIMEOUT", 600) as u64),
            context_length: num("CTX", 131072),
            kv_budget_mib: 0.0,
            kv_per_tok_mib: 0.0,
            prefix_cache_mib: num("PREFIX_CACHE_MIB", 0),
        }
    }
}

struct AdmitReq {
    prompt: Vec<u32>,
    params: GenParams,
    truncated: bool,
    prompt_tokens: usize,
    out: mpsc::Sender<StreamEvent>,
    cancel: crate::engine_types::CancelFlag,
    media: Option<MultimodalAdmission>,
}

struct MultimodalAdmission {
    payload: MultimodalPrefill,
    usage: MediaUsage,
    lease: PreparedLease,
}

enum IngestMsg {
    Admit(AdmitReq),
}

type SlotSamplers = Arc<Mutex<HashMap<usize, (GenParams, Rng)>>>;

struct SlotBinding {
    out: mpsc::Sender<StreamEvent>,
    params: GenParams,
    prompt_tokens: usize,
    completion_tokens: usize,
    /// ретроактивно менять ранние байты (многотокенные UTF-8), поэтому
    /// индекс небезопасен — сравниваем префиксы.
    emitted_text: String,
    stop_hit: bool,
    cancelled: bool,
    last_progress: Instant,
    usage: MediaUsage,
    _media_lease: Option<PreparedLease>,
    cancel: crate::engine_types::CancelFlag,
}

pub struct BatchedEngine {
    tx_ingest: mpsc::Sender<IngestMsg>,
    info: ModelInfo,
    max_queue: usize,
    in_flight: Arc<AtomicUsize>,
    tokenizer: Arc<Mutex<tokenizers::Tokenizer>>,
    media: Arc<crate::media::MediaService>,
    vision_path: Option<std::path::PathBuf>,
    /// Shutdown-флаг dispatch thread: выставляется при drop engine (unload).
    shutdown: Arc<AtomicBool>,
    /// JoinHandle dispatch thread — join в Drop гарантирует, что adapter
    /// (GPU-память) освобождён до возврата из drop (иначе trim mempool идёт
    /// по живому adapter и VRAM не возвращается ОС).
    dispatch_handle: Mutex<Option<std::thread::JoinHandle<()>>>,
    /// Готовность: адаптер загружен в dispatch-потоке. До этого generate
    /// стоит в очереди канала (не ошибка), а UI должен показывать «загрузка».
    ready: Arc<AtomicBool>,
    /// Ошибка загрузки адаптера в dispatch-потоке (видна в available_models).
    load_error: Arc<RwLock<String>>,
    /// Официальный chat template из GGUF (minijinja). None → встроенный ChatML.
    chat_tpl: Option<crate::chat_template::ChatTemplate>,
}

impl Drop for BatchedEngine {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        if let Some(h) = self.dispatch_handle.lock().expect("dispatch lock").take() {
            let _ = h.join();
            eprintln!("[batched] dispatch thread joined, adapter dropped");
        }
    }
}

impl BatchedEngine {
    pub async fn load(
        cfg: BatchConfig,
        media: Arc<crate::media::MediaService>,
        vision_path: Option<std::path::PathBuf>,
        mtp_path: Option<std::path::PathBuf>,
    ) -> anyhow::Result<Arc<Self>> {
        // Форк имеет DECODE_BATCH_CAPACITY=4: больше слотов → panic при seed_slot_batched.
        if cfg.slots > MAX_SLOTS {
            anyhow::bail!(
                "slots {} exceeds MAX_SLOTS {} (DECODE_BATCH_CAPACITY in fork)",
                cfg.slots,
                MAX_SLOTS
            );
        }
        // Prefix cache: временно отключён (submit_primed контракт не готов).
        if cfg.prefix_cache_mib != 0 {
            anyhow::bail!("prefix cache temporarily disabled (QWEN36_PREFIX_CACHE_MIB>0)");
        }
        let cfg = Arc::new(cfg);
        let (tx_ingest, rx_ingest) = mpsc::channel(cfg.max_queue);

        // TODO-F1: загрузка адаптера (блокирующе) + токенизатора.
        let device = select_device()?;
        let adapter_device = device.clone();

        let info = ModelInfo {
            id: model_id_from_filename(std::path::Path::new(&cfg.model_path)),
            context_length: cfg.context_length,
            quant: quant_from_filename(std::path::Path::new(&cfg.model_path)),
            slots: cfg.slots,
            modes: vec!["thinking".into(), "instruct".into()],
        };

        let engine = Arc::new(Self {
            tx_ingest,
            info,
            max_queue: cfg.max_queue,
            in_flight: Arc::new(AtomicUsize::new(0)),
            tokenizer: Arc::new(Mutex::new(tokenizer::load_from_gguf_path(
                std::path::Path::new(&cfg.model_path),
            )?)),
            media,
            vision_path,
            shutdown: Arc::new(AtomicBool::new(false)),
            dispatch_handle: Mutex::new(None),
            ready: Arc::new(AtomicBool::new(false)),
            load_error: Arc::new(RwLock::new(String::new())),
            chat_tpl: crate::chat_template::ChatTemplate::from_gguf(std::path::Path::new(
                &cfg.model_path,
            )),
        });

        let tokenizer = engine.tokenizer.clone();
        let in_flight = Arc::clone(&engine.in_flight);
        let cfg2 = Arc::clone(&cfg);
        let shutdown2 = Arc::clone(&engine.shutdown);
        let vision_path2 = engine.vision_path.clone();
        let mtp_path2 = mtp_path.clone();
        // ponytail: Qwen35BatchAdapter владеет raw CUDA graph handles → не Send.
        // Создаём adapter+scheduler прямо в dispatch std::thread (не tokio, не
        // spawn_blocking): closure не содержит non-Send значений.
        let ready = Arc::clone(&engine.ready);
        let load_error = Arc::clone(&engine.load_error);
        let handle = std::thread::spawn(move || {
            let mut adapter = match Qwen35BatchAdapter::load(
                std::path::Path::new(&cfg2.model_path),
                adapter_device,
                cfg2.slots,
            ) {
                Ok(a) => a,
                Err(e) => {
                    eprintln!("[dispatch] adapter load failed: {e:#}");
                    *load_error.write().expect("load_error lock") = format!("{e:#}");
                    return;
                }
            };
            if let Some(path) = vision_path2 {
                if let Err(e) = adapter.load_vision(&path) {
                    eprintln!("[dispatch] vision load failed: {e:#}");
                }
            }
            if let Some(path) = mtp_path2 {
                if let Err(e) = adapter.load_mtp(&path) {
                    eprintln!("[dispatch] mtp load failed: {e:#}");
                }
            }
            #[cfg(feature = "cuda")]
            crate::engine::maybe_retain_mempool(&device);
            let eos = adapter.eos();
            let vocab = adapter.vocab_size();
            let scheduler = BatchScheduler::new(adapter, cfg2.slots, eos, vocab);
            ready.store(true, Ordering::Relaxed);
            dispatch_loop(scheduler, rx_ingest, cfg2, in_flight, tokenizer, shutdown2);
        });
        *engine.dispatch_handle.lock().expect("dispatch lock") = Some(handle);

        Ok(engine)
    }
}

#[async_trait::async_trait]
impl Engine for BatchedEngine {
    fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
    }
    fn ready(&self) -> bool {
        self.ready.load(Ordering::Relaxed)
    }
    fn load_error(&self) -> Option<String> {
        let e = self.load_error.read().expect("load_error lock");
        if e.is_empty() {
            None
        } else {
            Some(e.clone())
        }
    }
    async fn generate(
        &self,
        request: InferenceRequest,
    ) -> anyhow::Result<mpsc::Receiver<StreamEvent>> {
        let InferenceRequest {
            messages,
            mut params,
            owner,
            cancel,
            tools,
            reasoning_effort,
        } = request;
        let has_media = messages.iter().any(ChatMessage::has_media);
        // Адаптер грузится в dispatch-потоке: до ready — отказ (не молчаливый
        // 503-queue). UI уже не пускает (send disabled), но API-клиенты могут.
        if !self.ready.load(Ordering::Relaxed) {
            if let Some(err) = self.load_error() {
                bail!("model load failed: {err}");
            }
            bail!("model still loading");
        }
        if has_media && self.vision_path.is_none() {
            return Err(crate::media::MediaError::new(
                crate::media::MediaErrorKind::ComponentUnavailable,
                "Vision component is unavailable",
            )
            .into());
        }
        if self.in_flight.load(Ordering::Relaxed) >= self.max_queue + MAX_SLOTS {
            bail!("queue full (QWEN36_MAX_QUEUE)");
        }

        let prepared = if has_media {
            Some(
                crate::media::prepare::prepare(
                    &messages,
                    owner,
                    &cancel,
                    &self.media,
                    self.info.context_length,
                )
                .await?,
            )
        } else {
            None
        };

        let (prompt, prompt_tokens, truncated, media) = if let Some(prepared) = prepared {
            let (prepared_messages, usage, lease) = prepared.into_parts();
            let tok = self
                .tokenizer
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            let mut image_grids = Vec::new();
            let mut video_grids = Vec::new();
            let mut media_grids = Vec::new();
            let mut patch_values = Vec::new();
            let mut content_storage = Vec::with_capacity(prepared_messages.len());
            for message in &prepared_messages {
                let mut content = Vec::with_capacity(message.content.len());
                for block in &message.content {
                    match block {
                        PreparedContentBlock::Text(text) => content.push(ChatContent::Text(text)),
                        PreparedContentBlock::Media(media) => {
                            let grid = media.grid;
                            media_grids.push([grid.t, grid.h, grid.w]);
                            patch_values.extend_from_slice(&media.patches);
                            match media.kind {
                                PackedMediaKind::Image => {
                                    image_grids.push(grid);
                                    content.push(ChatContent::Image {
                                        visual_tokens: media.visual_tokens()?,
                                    });
                                }
                                PackedMediaKind::Video => {
                                    video_grids.push(grid);
                                    content.push(ChatContent::Video {
                                        frame_tokens: media.frame_tokens()?,
                                        timestamps: &media.timestamps,
                                    });
                                }
                            }
                        }
                    }
                }
                content_storage.push(content);
            }
            let mixed: Vec<MultimodalChatMsg<'_>> = prepared_messages
                .iter()
                .zip(&content_storage)
                .map(|(message, content)| MultimodalChatMsg {
                    role: &message.role,
                    content,
                })
                .collect();
            let text = tokenizer::build_chatml_multimodal(&mixed)?;
            let encoded = tokenizer::encode_multimodal_no_think(&tok, &text)?;
            let position =
                build_position_plan(&encoded.mm_token_types, &image_grids, &video_grids)?;
            params.clamp_to_context(encoded.ids.len(), self.info.context_length)?;
            let patch_width = 3 * 2 * 16 * 16;
            let patch_rows = patch_values.len() / patch_width;
            let prompt_tokens = encoded.ids.len();
            (
                encoded.ids.clone(),
                prompt_tokens,
                false,
                Some(MultimodalAdmission {
                    payload: MultimodalPrefill {
                        token_ids: encoded.ids,
                        media_grids,
                        patch_values,
                        patch_rows,
                        patch_width,
                        mm_token_types: encoded.mm_token_types,
                        rope_positions: position.rope_positions,
                        decode_rope_delta: position.decode_rope_delta,
                    },
                    usage,
                    lease,
                }),
            )
        } else {
            let tok = self.tokenizer.lock().unwrap_or_else(|e| e.into_inner());
            let budget = self
                .info
                .context_length
                .saturating_sub(params.max_tokens)
                .saturating_sub(TRIM_MARGIN);
            let count = |m: &ChatMessage| -> usize {
                let chunk = format!("<|im_start|>{}\n{}\n<|im_end|>\n", m.role, m.text_content());
                tok.encode(chunk, false)
                    .map(|e| e.get_ids().len())
                    .unwrap_or(0)
            };
            let (kept, was_trimmed) = trim_messages(&messages, budget, count);
            let msgs_text: Vec<(&str, String)> = kept
                .iter()
                .map(|m| (m.role.as_str(), m.text_content_with_reasoning()))
                .collect();
            let msgs: Vec<ChatMsg> = msgs_text
                .iter()
                .map(|(role, content)| ChatMsg { role, content })
                .collect();
            // Официальный Jinja-шаблон из GGUF (think-блок и tool calls в
            // формате обучения модели); fallback — встроенный ChatML-билдер.
            let rendered = self
                .chat_tpl
                .as_ref()
                .map(|tpl| {
                    tpl.render(
                        &kept,
                        tools.as_ref(),
                        params.thinking,
                        reasoning_effort.as_deref(),
                    )
                })
                .transpose()?;
            let ids = match rendered {
                Some(text) => tok
                    .encode(text.as_str(), false)
                    .map(|e| e.get_ids().to_vec())
                    .map_err(|e| anyhow!("encode prompt: {e}"))?,
                None => {
                    let text = tokenizer::build_chatml_text_with_tools(&msgs, tools.as_ref());
                    if params.thinking {
                        let mut ids = tok
                            .encode(text, false)
                            .map(|e| e.get_ids().to_vec())
                            .map_err(|e| anyhow!("encode prompt: {e}"))?;
                        ids.push(qwen35_batch::real::tokenizer::THINK_OPEN_TOKEN_ID);
                        let nl = tok
                            .encode("\n", false)
                            .map(|e| e.get_ids().to_vec())
                            .map_err(|e| anyhow!("encode nl: {e}"))?;
                        ids.extend_from_slice(&nl);
                        ids
                    } else {
                        tokenizer::encode_no_think(&tok, &text)?
                    }
                }
            };
            let n = ids.len();
            params.clamp_to_context(n, self.info.context_length)?;
            (ids, n, was_trimmed, None)
        };

        let (out_tx, out_rx) = mpsc::channel(SLOT_CHAN_CAP);
        self.in_flight.fetch_add(1, Ordering::Relaxed);
        let req = AdmitReq {
            prompt,
            params: GenParams { ..params },
            truncated,
            prompt_tokens,
            out: out_tx,
            cancel,
            media,
        };
        self.tx_ingest
            .send(IngestMsg::Admit(req))
            .await
            .map_err(|_| anyhow!("engine dispatch loop dead"))?;
        Ok(out_rx)
    }

    fn model_info(&self) -> ModelInfo {
        self.info.clone()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Dispatch loop — единственный владелец scheduler'а/модели.
// ────────────────────────────────────────────────────────────────────────────

fn dispatch_loop(
    mut sched: BatchScheduler<Qwen35BatchAdapter>,
    mut rx: mpsc::Receiver<IngestMsg>,
    cfg: Arc<BatchConfig>,
    in_flight: Arc<AtomicUsize>,
    tokenizer: Arc<Mutex<tokenizers::Tokenizer>>,
    shutdown: Arc<AtomicBool>,
) {
    let mut bindings: Vec<Option<SlotBinding>> = (0..cfg.slots).map(|_| None).collect();
    let mut pending: VecDeque<AdmitReq> = VecDeque::new();
    // Per-slot sampling state persists across decode steps. Replacing sampler
    // from a cloned map each step resets every RNG to the same quantile.
    let slot_samplers: SlotSamplers = Arc::new(Mutex::new(HashMap::new()));
    sched.set_sampler(Box::new(IndexedSampler {
        params: Arc::clone(&slot_samplers),
    }));
    // Per-slot last emitted token count (для incremental decode).
    let mut slot_emitted_toks: HashMap<usize, usize> = HashMap::new();
    // Per-slot truncated flag (для Done).
    let mut slot_truncated: HashMap<usize, bool> = HashMap::new();
    // Диагностика: heartbeat раз в 5s пока есть активные слоты.
    let mut last_hb = Instant::now();

    'outer: loop {
        if shutdown.load(Ordering::Relaxed) {
            eprintln!("[dispatch] shutdown flag seen, exiting");
            break;
        }
        // 1. Слить накопленные ingest-сообщения.
        while let Ok(msg) = rx.try_recv() {
            match msg {
                IngestMsg::Admit(req) if req.out.is_closed() => {
                    in_flight.fetch_sub(1, Ordering::Relaxed);
                }
                IngestMsg::Admit(req) => admit(
                    req,
                    &mut sched,
                    &mut bindings,
                    &mut pending,
                    &slot_samplers,
                    &mut slot_truncated,
                    &cfg,
                ),
            }
        }

        // 2. Watchdog: молчащие слоты → Error + free.
        let now = Instant::now();
        for idx in 0..bindings.len() {
            let timed_out = bindings[idx]
                .as_mut()
                .map(|b| {
                    if now.duration_since(b.last_progress) > cfg.req_timeout {
                        let _ = b.out.try_send(StreamEvent::Error("timeout".into()));
                        true
                    } else {
                        false
                    }
                })
                .unwrap_or(false);
            if timed_out {
                free_slot(
                    idx,
                    &mut bindings,
                    &mut sched,
                    &in_flight,
                    &slot_samplers,
                    &mut slot_emitted_toks,
                    &mut slot_truncated,
                );
            }
        }

        // Клиент мог отключиться во время очереди/prefill, до первой Delta.
        for b in bindings.iter_mut().flatten() {
            if b.out.is_closed() || b.cancel.is_cancelled() {
                b.cancelled = true;
            }
        }

        // 3. Persistent sampler advances shared per-slot RNG state in place.
        let trace = qwen35_batch::scheduler::trace_on();
        // GPU-шаг БЕЗ мьютекса токенизатора (аудит 2026-08-10): иначе входящие
        // HTTP-запросы блокируются на lock() в generate() на весь шаг.
        // Токенизатор нужен только drain'у после шага.
        let outcome = sched.step_with(&mut |sidx, _generated| {
            bindings
                .get(sidx)
                .and_then(|b| b.as_ref())
                .map(|b| b.cancelled)
                .unwrap_or(false)
        });
        let did_work = match outcome {
            Ok(StepOutcome::DidPrefill {
                first_token_emitted,
            }) => {
                if first_token_emitted {
                    let tok_guard = tokenizer.lock().unwrap_or_else(|e| e.into_inner());
                    drain_after_step(
                        &mut sched,
                        &mut bindings,
                        &mut slot_emitted_toks,
                        &tok_guard,
                    );
                    drop(tok_guard);
                }
                true
            }
            Ok(StepOutcome::DidDecode(_)) => {
                let tok_guard = tokenizer.lock().unwrap_or_else(|e| e.into_inner());
                drain_after_step(
                    &mut sched,
                    &mut bindings,
                    &mut slot_emitted_toks,
                    &tok_guard,
                );
                true
            }
            Ok(StepOutcome::Idle) => false,
            Err(e) => {
                eprintln!("[batch] scheduler step error: {e:#}");
                for binding in &mut bindings {
                    if let Some(b) = binding.as_mut() {
                        let _ = b.out.try_send(StreamEvent::Error(format!("{e:#}")));
                        b.cancelled = true;
                    }
                }
                true
            }
        };

        // 4. Сбор Finished (TODO-F6: slots_mut) → Done + admit pending.
        let finished_idxs: Vec<usize> = (0..cfg.slots)
            .filter(|&idx| {
                bindings[idx]
                    .as_ref()
                    .map(|b| {
                        b.stop_hit
                            || b.cancelled
                            || sched.slots_mut()[idx].status == SlotStatus::Finished
                    })
                    .unwrap_or(false)
            })
            .collect();
        if trace && !finished_idxs.is_empty() {
            eprintln!("[dl] finishing slots {finished_idxs:?}");
        }
        for idx in finished_idxs {
            finish_slot(
                idx,
                &mut bindings,
                &mut sched,
                &in_flight,
                &slot_samplers,
                &mut slot_emitted_toks,
                &mut slot_truncated,
                &tokenizer,
            );
            if trace {
                eprintln!("[dl] finished slot {idx}");
            }
            admit_from_pending(
                &mut pending,
                &mut sched,
                &mut bindings,
                &slot_samplers,
                &mut slot_truncated,
                &cfg,
                &in_flight,
            );
        }

        // Heartbeat: статусы слотов каждые 5s при активности.
        if last_hb.elapsed() > Duration::from_secs(5) && bindings.iter().any(|b| b.is_some()) {
            let st: Vec<String> = (0..cfg.slots)
                .map(|i| {
                    let s = &sched.slots_mut()[i];
                    format!("{}:{:?}:gen={}", i, s.status, s.generated_tokens().len())
                })
                .collect();
            eprintln!("[hb] pending={} slots={}", pending.len(), st.join(" "));
            last_hb = Instant::now();
        }

        // 5. Нет работы — ждём с polling shutdown (таймаут 50мс, иначе shutdown
        // не виден пока не придёт сообщение; blocking_recv блокирует навечно).
        if !did_work && pending.is_empty() && bindings.iter().all(|b| b.is_none()) {
            std::thread::sleep(std::time::Duration::from_millis(50));
            match rx.try_recv() {
                Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => break 'outer,
                Ok(_) if shutdown.load(Ordering::Relaxed) => break 'outer,
                Ok(IngestMsg::Admit(req)) if req.out.is_closed() => {
                    in_flight.fetch_sub(1, Ordering::Relaxed);
                }
                Ok(IngestMsg::Admit(req)) => admit(
                    req,
                    &mut sched,
                    &mut bindings,
                    &mut pending,
                    &slot_samplers,
                    &mut slot_truncated,
                    &cfg,
                ),
                Err(tokio::sync::mpsc::error::TryRecvError::Empty) => {}
            }
        } else {
            std::thread::yield_now();
        }
    }
}

/// TODO-F3: новые токены слотов → decode_text → Delta (инкрементально).
/// ponytail: полный decode всей generated последовательности (как CandleEngine);
/// инкрементальный буфер хвоста — если профилирование покажет overhead.
fn drain_after_step(
    sched: &mut BatchScheduler<Qwen35BatchAdapter>,
    bindings: &mut [Option<SlotBinding>],
    slot_emitted_toks: &mut HashMap<usize, usize>,
    tokenizer: &tokenizers::Tokenizer,
) {
    let slots = sched.slots_mut();
    for idx in 0..slots.len() {
        let generated = slots[idx].generated_tokens();
        let emitted = *slot_emitted_toks.get(&idx).unwrap_or(&0);
        if generated.len() <= emitted {
            continue;
        }
        let Some(b) = bindings[idx].as_mut() else {
            continue;
        };
        b.completion_tokens = generated.len();
        b.last_progress = Instant::now();

        // Реальный decode: decode_text по всей generated (корректно на UTF-8
        // границах; ponytail: инкрементальный буфер хвоста — если профиль покажет).
        let new_text = tokenizer::decode_text(tokenizer, generated).unwrap_or_default();

        // Stop-строки: ищем в хвосте длиной max_stop_len + последний кусок.
        let max_stop_len = b.params.stop.iter().map(|s| s.len()).max().unwrap_or(0);
        let mut cut_at: Option<usize> = None;
        if !b.params.stop.is_empty() {
            let scan_from = new_text.len().saturating_sub(max_stop_len + 64);
            let scan_from = floor_char_boundary(&new_text, scan_from)
                .max(b.emitted_text.len().min(new_text.len()));
            if let Some(rel) = new_text[scan_from..].find_any(&b.params.stop) {
                cut_at = Some(scan_from + rel);
            }
        }

        let mut end = cut_at.unwrap_or(new_text.len());
        // Holdback: не эмитим хвост, заканчивающийся на U+FFFD — это может быть
        // недо-собранная UTF-8 последовательность (emoji/CJK разрезаны на
        // несколько токенов); следующий токен достроит. Иначе устаревший '�'
        // уходит клиенту и префикс расходится навсегда. Флаш — в finish_slot.
        while end > b.emitted_text.len() && new_text[..end].ends_with('\u{FFFD}') {
            end -= '\u{FFFD}'.len_utf8();
        }
        // Префиксное сравнение: если decode ретроактивно изменил ранние байты —
        // ресинхронизируемся (не эмитим на этом шаге), индексной арифметики нет.
        if new_text.starts_with(&b.emitted_text) && end >= b.emitted_text.len() {
            let delta = &new_text[b.emitted_text.len()..end];
            if !delta.is_empty() {
                match b.out.try_send(StreamEvent::Delta(delta.to_string())) {
                    Ok(()) => {}
                    Err(mpsc::error::TrySendError::Full(_))
                    | Err(mpsc::error::TrySendError::Closed(_)) => {
                        b.cancelled = true;
                    }
                }
            }
            b.emitted_text = new_text[..end].to_string();
        } else if !new_text.starts_with(&b.emitted_text) {
            b.emitted_text = new_text[..end].to_string();
        }
        if cut_at.is_some() {
            b.stop_hit = true;
        }
        slot_emitted_toks.insert(idx, generated.len());
    }
}

/// Текущее использование KV (MiB) всеми привязанными слотами.
fn kv_used_mib(
    sched: &mut BatchScheduler<Qwen35BatchAdapter>,
    bindings: &[Option<SlotBinding>],
    kv_per_tok_mib: f64,
) -> f64 {
    let toks: usize = (0..bindings.len())
        .filter_map(|i| {
            bindings[i]
                .as_ref()
                .map(|b| b.prompt_tokens + sched.slots_mut()[i].generated_tokens().len())
        })
        .sum();
    toks as f64 * kv_per_tok_mib
}

/// Admission control: влезает ли промпт в остаток KV-бюджета.
fn kv_fits(
    sched: &mut BatchScheduler<Qwen35BatchAdapter>,
    bindings: &[Option<SlotBinding>],
    cfg: &BatchConfig,
    new_prompt_tokens: usize,
) -> bool {
    if cfg.kv_budget_mib <= 0.0 {
        return true;
    }
    let used = kv_used_mib(sched, bindings, cfg.kv_per_tok_mib);
    let est = new_prompt_tokens as f64 * cfg.kv_per_tok_mib + 8.0; // +margin
    used + est <= cfg.kv_budget_mib
}

fn admit(
    req: AdmitReq,
    sched: &mut BatchScheduler<Qwen35BatchAdapter>,
    bindings: &mut [Option<SlotBinding>],
    pending: &mut VecDeque<AdmitReq>,
    slot_samplers: &SlotSamplers,
    slot_truncated: &mut HashMap<usize, bool>,
    cfg: &BatchConfig,
) {
    match bindings.iter().position(|b| b.is_none()) {
        Some(idx) if kv_fits(sched, bindings, cfg, req.prompt_tokens) => {
            seed_slot(idx, req, sched, bindings, slot_samplers, slot_truncated)
        }
        _ => pending.push_back(req),
    }
}

fn admit_from_pending(
    pending: &mut VecDeque<AdmitReq>,
    sched: &mut BatchScheduler<Qwen35BatchAdapter>,
    bindings: &mut [Option<SlotBinding>],
    slot_samplers: &SlotSamplers,
    slot_truncated: &mut HashMap<usize, bool>,
    cfg: &BatchConfig,
    in_flight: &AtomicUsize,
) {
    while let Some(idx) = bindings.iter().position(|b| b.is_none()) {
        while pending.front().is_some_and(|req| req.out.is_closed()) {
            pending.pop_front();
            in_flight.fetch_sub(1, Ordering::Relaxed);
        }
        let fits = pending
            .front()
            .map(|r| kv_fits(sched, bindings, cfg, r.prompt_tokens))
            .unwrap_or(false);
        if !fits {
            break;
        }
        let req = pending.pop_front().unwrap();
        seed_slot(idx, req, sched, bindings, slot_samplers, slot_truncated);
    }
}

/// IDLE→PREFILL: submit в scheduler (TODO-F4).
fn seed_slot(
    idx: usize,
    req: AdmitReq,
    sched: &mut BatchScheduler<Qwen35BatchAdapter>,
    bindings: &mut [Option<SlotBinding>],
    slot_samplers: &SlotSamplers,
    slot_truncated: &mut HashMap<usize, bool>,
) {
    let prompt_tokens = req.prompt_tokens;
    let max_new = req.params.max_tokens;
    let seed = req.params.seed.unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos() as u64 ^ d.as_secs())
            .unwrap_or(42)
    });
    let params = req.params.clone();
    sched.submit(req.prompt, max_new);
    let (usage, media_lease) = match req.media {
        Some(media) => {
            use qwen35_batch::model::BatchModel;
            if let Err(error) = sched.model_mut().install_multimodal(idx, media.payload) {
                let _ = req.out.try_send(StreamEvent::Error(error.to_string()));
                sched.slots_mut()[idx].reset();
                return;
            }
            (media.usage, Some(media.lease))
        }
        None => (MediaUsage::default(), None),
    };
    slot_samplers
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(idx, (params.clone(), Rng::new(seed)));
    slot_truncated.insert(idx, req.truncated);
    bindings[idx] = Some(SlotBinding {
        out: req.out,
        params,
        prompt_tokens,
        completion_tokens: 0,
        emitted_text: String::new(),
        stop_hit: false,
        cancelled: false,
        last_progress: Instant::now(),
        usage,
        _media_lease: media_lease,
        cancel: req.cancel,
    });
}

/// FINISHED: Done с usage → reset (TODO-F4/F6).
#[allow(clippy::too_many_arguments)]
fn finish_slot(
    idx: usize,
    bindings: &mut [Option<SlotBinding>],
    sched: &mut BatchScheduler<Qwen35BatchAdapter>,
    in_flight: &AtomicUsize,
    slot_samplers: &SlotSamplers,
    slot_emitted_toks: &mut HashMap<usize, usize>,
    slot_truncated: &mut HashMap<usize, bool>,
    tokenizer: &Arc<Mutex<tokenizers::Tokenizer>>,
) {
    let Some(b) = bindings[idx].take() else {
        return;
    };
    // Flush holdback-хвоста (U+FFFD) перед Done: полный decode generated.
    if !b.cancelled {
        let generated = sched.slots_mut()[idx].generated_tokens().to_vec();
        let text = {
            let tok = tokenizer.lock().unwrap_or_else(|e| e.into_inner());
            tokenizer::decode_text(&tok, &generated).unwrap_or_default()
        };
        if text.starts_with(&b.emitted_text) && text.len() > b.emitted_text.len() {
            let tail = text[b.emitted_text.len()..].to_string();
            let _ = b.out.try_send(StreamEvent::Delta(tail));
        }
    }
    if !b.cancelled {
        let finish_reason = if b.stop_hit {
            "stop"
        } else if b.completion_tokens >= b.params.max_tokens {
            "length"
        } else {
            "stop" // EOS
        };
        let truncated = slot_truncated.remove(&idx).unwrap_or(false);
        let _ = b.out.try_send(StreamEvent::Done {
            finish_reason: finish_reason.into(),
            usage: GenerationUsage {
                prompt_tokens: b.prompt_tokens,
                completion_tokens: b.completion_tokens,
                truncated,
                media: b.usage.clone(),
                mtp: sched
                    .speculative_metrics(idx)
                    .map(|metrics| crate::engine_types::MtpUsage {
                        enabled: metrics.enabled,
                        used: metrics.used,
                        drafted: metrics.drafted,
                        accepted: metrics.accepted,
                        fallback_category: metrics
                            .fallback
                            .map(|category| category.as_str().to_string()),
                    })
                    .unwrap_or_default(),
            },
        });
    } else {
        slot_truncated.remove(&idx);
    }
    sched.slots_mut()[idx].reset();
    use qwen35_batch::model::BatchModel;
    let _ = sched.model_mut().reset_slot(idx);
    slot_samplers
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .remove(&idx);
    slot_emitted_toks.remove(&idx);
    in_flight.fetch_sub(1, Ordering::Relaxed);
}

fn free_slot(
    idx: usize,
    bindings: &mut [Option<SlotBinding>],
    sched: &mut BatchScheduler<Qwen35BatchAdapter>,
    in_flight: &AtomicUsize,
    slot_samplers: &SlotSamplers,
    slot_emitted_toks: &mut HashMap<usize, usize>,
    slot_truncated: &mut HashMap<usize, bool>,
) {
    if bindings[idx].take().is_some() {
        sched.slots_mut()[idx].reset();
        use qwen35_batch::model::BatchModel;
        let _ = sched.model_mut().reset_slot(idx);
        slot_samplers
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&idx);
        slot_emitted_toks.remove(&idx);
        slot_truncated.remove(&idx);
        in_flight.fetch_sub(1, Ordering::Relaxed);
    }
}

// ────────────────────────────────────────────────────────────────────────────
// IndexedSampler: impl форк-trait Sampler с per-slot GenParams (TODO-F5).
// ────────────────────────────────────────────────────────────────────────────

struct IndexedSampler {
    params: SlotSamplers,
}

impl ForkSampler for IndexedSampler {
    fn sample(&mut self, _logits: &[f32]) -> u32 {
        0 // не используется — scheduler всегда sample_indexed
    }

    fn checkpoint(&self, slot_idx: usize) -> SamplerCheckpoint {
        let checkpoint = self
            .params
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .get(&slot_idx)
            .cloned();
        Box::new(checkpoint)
    }

    fn restore(&mut self, slot_idx: usize, checkpoint: SamplerCheckpoint) -> anyhow::Result<()> {
        let checkpoint = checkpoint
            .downcast::<Option<(GenParams, Rng)>>()
            .map_err(|_| anyhow!("sampler checkpoint type mismatch"))?;
        let mut params = self
            .params
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        match *checkpoint {
            Some(state) => {
                params.insert(slot_idx, state);
            }
            None => {
                params.remove(&slot_idx);
            }
        }
        Ok(())
    }

    fn sample_indexed(&mut self, slot_idx: usize, generated: &[u32], logits: &[f32]) -> u32 {
        let mut params = self.params.lock().unwrap_or_else(|e| e.into_inner());
        match params.get_mut(&slot_idx) {
            Some((p, rng)) => sampler::sample(
                logits,
                p.temperature,
                p.top_k,
                p.top_p,
                p.min_p,
                p.presence_penalty,
                p.repetition_penalty,
                generated,
                rng,
            ),
            None => {
                let mut best = 0u32;
                let mut best_v = f32::NEG_INFINITY;
                for (i, &v) in logits.iter().enumerate() {
                    if v > best_v {
                        best_v = v;
                        best = i as u32;
                    }
                }
                best
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batchconfig_clamps_slots() {
        std::env::set_var("QWEN36_SLOTS", "8");
        let c = BatchConfig::from_env();
        assert_eq!(c.slots, MAX_SLOTS);
        std::env::remove_var("QWEN36_SLOTS");
    }

    fn test_config(slots: usize, prefix_cache_mib: usize) -> BatchConfig {
        BatchConfig {
            model_path: "missing.gguf".into(),
            slots,
            max_queue: 1,
            req_timeout: Duration::from_secs(1),
            context_length: 32,
            kv_budget_mib: 0.0,
            kv_per_tok_mib: 0.0,
            prefix_cache_mib,
        }
    }

    #[tokio::test]
    async fn load_rejects_slots_above_fork_capacity_before_model_load() {
        let media = Arc::new(crate::media::MediaService::new(Default::default()).unwrap());
        let err = match BatchedEngine::load(test_config(MAX_SLOTS + 1, 0), media, None, None).await
        {
            Ok(_) => panic!("slots above capacity must be rejected"),
            Err(err) => err.to_string(),
        };
        assert!(err.contains("exceeds MAX_SLOTS"));
    }

    #[tokio::test]
    async fn load_rejects_prefix_cache_before_model_load() {
        let media = Arc::new(crate::media::MediaService::new(Default::default()).unwrap());
        let err = match BatchedEngine::load(test_config(1, 1), media, None, None).await {
            Ok(_) => panic!("prefix cache must be rejected"),
            Err(err) => err.to_string(),
        };
        assert!(err.contains("prefix cache temporarily disabled"));
    }

    #[test]
    fn indexed_sampler_advances_rng_across_steps() {
        let params = GenParams {
            top_k: 4,
            top_p: 1.0,
            ..Default::default()
        };
        let shared = Arc::new(Mutex::new(HashMap::from([(
            0,
            (params.clone(), Rng::new(123)),
        )])));
        let mut indexed = IndexedSampler { params: shared };
        let logits = vec![0.0; 4];
        let actual: Vec<u32> = (0..8)
            .map(|_| indexed.sample_indexed(0, &[], &logits))
            .collect();

        let mut expected_rng = Rng::new(123);
        let expected: Vec<u32> = (0..8)
            .map(|_| {
                sampler::sample(
                    &logits,
                    params.temperature,
                    params.top_k,
                    params.top_p,
                    params.min_p,
                    params.presence_penalty,
                    params.repetition_penalty,
                    &[],
                    &mut expected_rng,
                )
            })
            .collect();
        assert!(expected.windows(2).any(|pair| pair[0] != pair[1]));
        assert_eq!(actual, expected);
    }

    #[test]
    fn indexed_sampler_fallback_greedy_when_no_params() {
        let mut s = IndexedSampler {
            params: Arc::new(Mutex::new(HashMap::new())),
        };
        let logits = vec![0.1, 0.9, 0.2, 5.0];
        assert_eq!(s.sample_indexed(99, &[], &logits), 3);
    }
}
