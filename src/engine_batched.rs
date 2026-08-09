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

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail};
use qwen35_batch::model::{BatchModel, Sampler as ForkSampler};
use qwen35_batch::real::tokenizer::{self, ChatMsg};
use qwen35_batch::real::Qwen35BatchAdapter;
use qwen35_batch::scheduler::{BatchScheduler, StepOutcome};
use qwen35_batch::slot::SlotStatus;
use tokio::sync::mpsc;

use crate::engine::{model_id_from_filename, quant_from_filename, select_device, trim_messages, FindAny, floor_char_boundary};
use crate::engine_types::{ChatMessage, Engine, GenParams, ModelInfo, StreamEvent};
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
}

impl BatchConfig {
    pub fn from_env() -> Self {
        let get = |k: &str| std::env::var(k).ok();
        let num = |k: &str, d: usize| get(k).and_then(|v| v.parse().ok()).unwrap_or(d);
        let mut slots = num("QWEN36_SLOTS", 4);
        if slots > MAX_SLOTS {
            eprintln!(
                "[batch] QWEN36_SLOTS={slots} > {MAX_SLOTS} (DECODE_BATCH_CAPACITY), clamp"
            );
            slots = MAX_SLOTS;
        }
        Self {
            model_path: get("QWEN36_MODEL")
                .unwrap_or_else(|| "models/qwen36-27b-q2_k_xl.gguf".into()),
            slots,
            max_queue: num("QWEN36_MAX_QUEUE", 64),
            req_timeout: Duration::from_secs(num("QWEN36_REQ_TIMEOUT", 600) as u64),
            context_length: num("QWEN36_CTX", 81920),
        }
    }
}

struct AdmitReq {
    req_id: u64,
    prompt: Vec<u32>,
    params: GenParams,
    truncated: bool,
    prompt_tokens: usize,
    out: mpsc::Sender<StreamEvent>,
}

enum IngestMsg {
    Admit(AdmitReq),
    Cancel { req_id: u64 },
}

struct SlotBinding {
    req_id: u64,
    out: mpsc::Sender<StreamEvent>,
    params: GenParams,
    prompt_tokens: usize,
    completion_tokens: usize,
    /// Уже эмитнутый префикс (строка, не индекс): decode_text может
    /// ретроактивно менять ранние байты (многотокенные UTF-8), поэтому
    /// индекс небезопасен — сравниваем префиксы.
    emitted_text: String,
    stop_hit: bool,
    cancelled: bool,
    last_progress: Instant,
}

/// Guard: клиент дропнул Receiver → Cancel в dispatch loop.
pub struct CancelGuard {
    req_id: u64,
    tx: mpsc::Sender<IngestMsg>,
}

impl Drop for CancelGuard {
    fn drop(&mut self) {
        let _ = self
            .tx
            .try_send(IngestMsg::Cancel { req_id: self.req_id });
    }
}

pub struct BatchedEngine {
    tx_ingest: mpsc::Sender<IngestMsg>,
    info: ModelInfo,
    next_id: Arc<AtomicU64>,
    max_queue: usize,
    in_flight: Arc<AtomicUsize>,
    tokenizer: Arc<Mutex<tokenizers::Tokenizer>>,
}

impl BatchedEngine {
    pub async fn load(cfg: BatchConfig) -> anyhow::Result<Arc<Self>> {
        let cfg = Arc::new(cfg);
        let (tx_ingest, rx_ingest) = mpsc::channel(cfg.max_queue);

        // TODO-F1: загрузка адаптера (блокирующе) + токенизатора.
        let device = select_device()?;
        let adapter = tokio::task::spawn_blocking({
            let p = cfg.model_path.clone();
            let slots = cfg.slots;
            move || Qwen35BatchAdapter::load(std::path::Path::new(&p), device, slots)
        })
        .await??;
        let eos = adapter.eos();
        let vocab = adapter.vocab_size();
        let tokenizer = tokenizer::load_from_gguf_path(std::path::Path::new(&cfg.model_path))?;

        let scheduler = BatchScheduler::new(adapter, cfg.slots, eos, vocab);

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
            next_id: Arc::new(AtomicU64::new(1)),
            max_queue: cfg.max_queue,
            in_flight: Arc::new(AtomicUsize::new(0)),
            tokenizer: Arc::new(Mutex::new(tokenizer)),
        });

        let tokenizer = engine.tokenizer.clone();
        let in_flight = Arc::clone(&engine.in_flight);
        let cfg2 = Arc::clone(&cfg);
        // scheduler/adapter не Send (CUDA context не thread-safe) → свой thread,
        // не tokio::spawn. dispatch_loop синхронная; blocking_recv на idle.
        std::thread::spawn(move || {
            dispatch_loop(scheduler, rx_ingest, cfg2, in_flight, tokenizer);
        });

        Ok(engine)
    }
}

#[async_trait::async_trait]
impl Engine for BatchedEngine {
    async fn generate(
        &self,
        messages: Vec<ChatMessage>,
        params: GenParams,
    ) -> anyhow::Result<mpsc::Receiver<StreamEvent>> {
        if self.in_flight.load(Ordering::Relaxed) >= self.max_queue + MAX_SLOTS {
            bail!("queue full (QWEN36_MAX_QUEUE)");
        }

        // TODO-F2: sliding window (BD-017) + build_chatml_text + encode_no_think.
        let (prompt, prompt_tokens, truncated) = {
            let tok = self.tokenizer.lock().unwrap_or_else(|e| e.into_inner());
            let budget = self
                .info
                .context_length
                .saturating_sub(params.max_tokens)
                .saturating_sub(TRIM_MARGIN);
            let count = |m: &ChatMessage| -> usize {
                let chunk = format!("<|im_start|>{}\n{}\n<|im_end|>\n", m.role, m.content);
                tok.encode(chunk, false)
                    .map(|e| e.get_ids().len())
                    .unwrap_or(0)
            };
            let (kept, was_trimmed) = trim_messages(&messages, budget, count);
            let msgs: Vec<ChatMsg> = kept
                .iter()
                .map(|m| ChatMsg {
                    role: &m.role,
                    content: &m.content,
                })
                .collect();
            let text = tokenizer::build_chatml_text(&msgs);
            let ids = tokenizer::encode_no_think(&tok, &text)?;
            let n = ids.len();
            (ids, n, was_trimmed)
        };

        let (out_tx, out_rx) = mpsc::channel(SLOT_CHAN_CAP);
        let req_id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.in_flight.fetch_add(1, Ordering::Relaxed);
        let req = AdmitReq {
            req_id,
            prompt,
            params: GenParams { ..params },
            truncated,
            prompt_tokens,
            out: out_tx,
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

pub fn cancel_guard(engine: &BatchedEngine, req_id: u64) -> CancelGuard {
    CancelGuard {
        req_id,
        tx: engine.tx_ingest.clone(),
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
) {
    let mut bindings: Vec<Option<SlotBinding>> = (0..cfg.slots).map(|_| None).collect();
    let mut pending: VecDeque<AdmitReq> = VecDeque::new();
    let mut cancelled: HashSet<u64> = HashSet::new();
    /// Per-slot sampling state: params + rng.
    let mut slot_samplers: HashMap<usize, (GenParams, Rng)> = HashMap::new();
    /// Per-slot last emitted token count (для incremental decode).
    let mut slot_emitted_toks: HashMap<usize, usize> = HashMap::new();
    /// Per-slot truncated flag (для Done).
    let mut slot_truncated: HashMap<usize, bool> = HashMap::new();
    // Диагностика: heartbeat раз в 5s пока есть активные слоты.
    let mut last_hb = Instant::now();

    loop {
        // 1. Слить накопленные ingest-сообщения.
        while let Ok(msg) = rx.try_recv() {
            match msg {
                IngestMsg::Admit(req) => {
                    if cancelled.remove(&req.req_id) {
                        in_flight.fetch_sub(1, Ordering::Relaxed);
                        continue;
                    }
                    admit(req, &mut sched, &mut bindings, &mut pending, &mut slot_samplers, &mut slot_truncated);
                }
                IngestMsg::Cancel { req_id } => {
                    cancel(req_id, &mut bindings, &mut pending, &mut cancelled, &in_flight);
                }
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
                free_slot(idx, &mut bindings, &mut sched, &in_flight, &mut slot_samplers, &mut slot_emitted_toks, &mut slot_truncated);
            }
        }

        // 3. Обновить sampler (per-slot params могли поменяться при admit) + шаг.
        let sampler_box: Box<dyn ForkSampler> = Box::new(IndexedSampler {
            params: slot_samplers.clone(),
        });
        sched.set_sampler(sampler_box);

        let trace = std::env::var_os("QWEN36_TRACE").is_some();
        if trace {
            eprintln!("[dl] before tokenizer.lock");
        }
        // tok_guard — std::sync::MutexGuard: step_with и drain синхронные.
        let did_work = {
            let tok_guard = tokenizer.lock().unwrap_or_else(|e| e.into_inner());
            if trace {
                eprintln!("[dl] got tokenizer.lock");
            }
            let outcome = sched.step_with(&mut |sidx, _generated| {
                bindings
                    .get(sidx)
                    .and_then(|b| b.as_ref())
                    .map(|b| b.cancelled)
                    .unwrap_or(false)
            });
            match outcome {
                Ok(StepOutcome::DidPrefill { first_token_emitted }) => {
                    if first_token_emitted {
                        drain_after_step(&mut sched, &mut bindings, &mut slot_emitted_toks, &tok_guard);
                    }
                    true
                }
                Ok(StepOutcome::DidDecode(_)) => {
                    if trace {
                        eprintln!("[dl] before drain");
                    }
                    drain_after_step(&mut sched, &mut bindings, &mut slot_emitted_toks, &tok_guard);
                    if trace {
                        eprintln!("[dl] after drain");
                    }
                    true
                }
                Ok(StepOutcome::Idle) => false,
                Err(e) => {
                    eprintln!("[batch] scheduler step error: {e:#}");
                    for idx in 0..bindings.len() {
                        if let Some(b) = bindings[idx].as_mut() {
                            let _ = b.out.try_send(StreamEvent::Error(format!("{e:#}")));
                            b.cancelled = true;
                        }
                    }
                    true
                }
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
            finish_slot(idx, &mut bindings, &mut sched, &in_flight, &mut slot_samplers, &mut slot_emitted_toks, &mut slot_truncated, &tokenizer);
            if trace {
                eprintln!("[dl] finished slot {idx}");
            }
            admit_from_pending(&mut pending, &mut sched, &mut bindings, &mut slot_samplers, &mut slot_truncated);
        }

        // Heartbeat: статусы слотов каждые 5s при активности.
        if last_hb.elapsed() > Duration::from_secs(5)
            && bindings.iter().any(|b| b.is_some())
        {
            let st: Vec<String> = (0..cfg.slots)
                .map(|i| {
                    let s = &sched.slots_mut()[i];
                    format!(
                        "{}:{:?}:gen={}",
                        i,
                        s.status,
                        s.generated_tokens().len()
                    )
                })
                .collect();
            eprintln!("[hb] pending={} slots={}", pending.len(), st.join(" "));
            last_hb = Instant::now();
        }

        // 5. Нет работы — ждём; есть — yield.
        if !did_work && pending.is_empty() && bindings.iter().all(|b| b.is_none()) {
            match rx.blocking_recv() {
                Some(msg) => match msg {
                    IngestMsg::Admit(req) => {
                        if cancelled.remove(&req.req_id) {
                            in_flight.fetch_sub(1, Ordering::Relaxed);
                        } else {
                            admit(req, &mut sched, &mut bindings, &mut pending, &mut slot_samplers, &mut slot_truncated);
                        }
                    }
                    IngestMsg::Cancel { req_id } => {
                        cancel(req_id, &mut bindings, &mut pending, &mut cancelled, &in_flight);
                    }
                },
                None => break,
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
        let Some(b) = bindings[idx].as_mut() else { continue };
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

fn admit(
    req: AdmitReq,
    sched: &mut BatchScheduler<Qwen35BatchAdapter>,
    bindings: &mut [Option<SlotBinding>],
    pending: &mut VecDeque<AdmitReq>,
    slot_samplers: &mut HashMap<usize, (GenParams, Rng)>,
    slot_truncated: &mut HashMap<usize, bool>,
) {
    match bindings.iter().position(|b| b.is_none()) {
        Some(idx) => seed_slot(idx, req, sched, bindings, slot_samplers, slot_truncated),
        None => pending.push_back(req),
    }
}

fn admit_from_pending(
    pending: &mut VecDeque<AdmitReq>,
    sched: &mut BatchScheduler<Qwen35BatchAdapter>,
    bindings: &mut [Option<SlotBinding>],
    slot_samplers: &mut HashMap<usize, (GenParams, Rng)>,
    slot_truncated: &mut HashMap<usize, bool>,
) {
    while let Some(idx) = bindings.iter().position(|b| b.is_none()) {
        let Some(req) = pending.pop_front() else { break };
        seed_slot(idx, req, sched, bindings, slot_samplers, slot_truncated);
    }
}

/// IDLE→PREFILL: submit в scheduler (TODO-F4).
fn seed_slot(
    idx: usize,
    req: AdmitReq,
    sched: &mut BatchScheduler<Qwen35BatchAdapter>,
    bindings: &mut [Option<SlotBinding>],
    slot_samplers: &mut HashMap<usize, (GenParams, Rng)>,
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
    // TODO-F4: sched.submit(prompt, max_new) — форк переводит Slot в Prefilling.
    sched.submit(req.prompt, max_new);
    slot_samplers.insert(idx, (params.clone(), Rng::new(seed)));
    slot_truncated.insert(idx, req.truncated);
    bindings[idx] = Some(SlotBinding {
        req_id: req.req_id,
        out: req.out,
        params,
        prompt_tokens,
        completion_tokens: 0,
        emitted_text: String::new(),
        stop_hit: false,
        cancelled: false,
        last_progress: Instant::now(),
    });
}

fn cancel(
    req_id: u64,
    bindings: &mut [Option<SlotBinding>],
    pending: &mut VecDeque<AdmitReq>,
    cancelled: &mut HashSet<u64>,
    in_flight: &AtomicUsize,
) {
    let before = pending.len();
    pending.retain(|r| r.req_id != req_id);
    if pending.len() != before {
        in_flight.fetch_sub(1, Ordering::Relaxed);
        return;
    }
    for b in bindings.iter_mut().flatten() {
        if b.req_id == req_id {
            b.cancelled = true;
            return;
        }
    }
    cancelled.insert(req_id);
}

/// FINISHED: Done с usage → reset (TODO-F4/F6).
fn finish_slot(
    idx: usize,
    bindings: &mut [Option<SlotBinding>],
    sched: &mut BatchScheduler<Qwen35BatchAdapter>,
    in_flight: &AtomicUsize,
    slot_samplers: &mut HashMap<usize, (GenParams, Rng)>,
    slot_emitted_toks: &mut HashMap<usize, usize>,
    slot_truncated: &mut HashMap<usize, bool>,
    tokenizer: &Arc<Mutex<tokenizers::Tokenizer>>,
) {
    let Some(mut b) = bindings[idx].take() else { return };
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
            prompt_tokens: b.prompt_tokens,
            completion_tokens: b.completion_tokens,
            truncated,
        });
    } else {
        slot_truncated.remove(&idx);
    }
    sched.slots_mut()[idx].reset();
    use qwen35_batch::model::BatchModel;
    let _ = sched.model_mut().reset_slot(idx);
    slot_samplers.remove(&idx);
    slot_emitted_toks.remove(&idx);
    in_flight.fetch_sub(1, Ordering::Relaxed);
}

fn free_slot(
    idx: usize,
    bindings: &mut [Option<SlotBinding>],
    sched: &mut BatchScheduler<Qwen35BatchAdapter>,
    in_flight: &AtomicUsize,
    slot_samplers: &mut HashMap<usize, (GenParams, Rng)>,
    slot_emitted_toks: &mut HashMap<usize, usize>,
    slot_truncated: &mut HashMap<usize, bool>,
) {
    if bindings[idx].take().is_some() {
        sched.slots_mut()[idx].reset();
        use qwen35_batch::model::BatchModel;
        let _ = sched.model_mut().reset_slot(idx);
        slot_samplers.remove(&idx);
        slot_emitted_toks.remove(&idx);
        slot_truncated.remove(&idx);
        in_flight.fetch_sub(1, Ordering::Relaxed);
    }
}

// ────────────────────────────────────────────────────────────────────────────
// IndexedSampler: impl форк-trait Sampler с per-slot GenParams (TODO-F5).
// ────────────────────────────────────────────────────────────────────────────

struct IndexedSampler {
    params: HashMap<usize, (GenParams, Rng)>,
}

impl ForkSampler for IndexedSampler {
    fn sample(&mut self, _logits: &[f32]) -> u32 {
        0 // не используется — scheduler всегда sample_indexed
    }

    fn sample_indexed(&mut self, slot_idx: usize, generated: &[u32], logits: &[f32]) -> u32 {
        match self.params.get_mut(&slot_idx) {
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

    #[test]
    fn indexed_sampler_uses_per_slot_params() {
        let mut params = HashMap::new();
        params.insert(0, (GenParams { top_k: 1, ..Default::default() }, Rng::new(1)));
        let mut s = IndexedSampler { params };
        let logits = vec![0.1, 0.9, 0.2, 5.0];
        for _ in 0..4 {
            assert_eq!(s.sample_indexed(0, &[], &logits), 3);
        }
    }

    #[test]
    fn indexed_sampler_fallback_greedy_when_no_params() {
        let mut s = IndexedSampler { params: HashMap::new() };
        let logits = vec![0.1, 0.9, 0.2, 5.0];
        assert_eq!(s.sample_indexed(99, &[], &logits), 3);
    }
}
