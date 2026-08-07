//! `BatchedEngine` — 4-слотовый engine поверх `BatchScheduler` из
//! `qwen35-batch` (BD-007). Дизайн: docs/batch-integration.md.
//!
//! Статус: СКЕЛЕТ интеграционной фазы. Очередь, dispatch loop, маршрутизация
//! StreamEvent per slot, отмена и watchdog — реальные. Вызовы модели —
//! заглушки с TODO-anchors:
//!
//! - TODO-F1: `Qwen35BatchAdapter::load(gguf, device, slots)` вместо stub-модели.
//! - TODO-F2: `tokenizer::load_from_gguf_path` + `build_chatml_text` +
//!   `encode_no_think` вместо stub-токенизации; sliding window (BD-017).
//! - TODO-F3: incremental `decode_text` + stop-holdback в `drain_slot`.
//! - TODO-F4: `sched.submit`/`step` реального scheduler'а (сейчас stub-loop).
//! - TODO-F5: indexed sampler в форке (per-request GenParams; до патча — greedy).
//! - TODO-F6: `BatchScheduler::slots_mut()` accessor в форке (сбор Finished).
//!
//! ponytail: stub-модель эмитит фиксированные токены — этого хватает, чтобы
//! гонять очередь/backpressure/cancel без GGUF. Upgrade: заменить StubModel
//! на BatchScheduler<Qwen35BatchAdapter> — сигнатуры шагов совпадают.

use std::collections::{HashSet, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail};
use tokio::sync::mpsc;

use crate::engine_types::{ChatMessage, Engine, GenParams, ModelInfo, StreamEvent};

/// Макс. слотов = DECODE_BATCH_CAPACITY форка (model_weights.rs:1411).
pub const MAX_SLOTS: usize = 4;
/// Capacity per-slot канала StreamEvent. Переполнение = мёртвый клиент → Cancel.
const SLOT_CHAN_CAP: usize = 64;

/// Конфиг engine'а (env — docs/engine-api.md + §2.9 дизайна).
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
        let num = |k: &str, d: usize| {
            get(k).and_then(|v| v.parse().ok()).unwrap_or(d)
        };
        let mut slots = num("QWEN36_SLOTS", 4);
        if slots > MAX_SLOTS {
            eprintln!("[batch] QWEN36_SLOTS={slots} > {MAX_SLOTS} (DECODE_BATCH_CAPACITY), clamp");
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

/// Запрос на admit в dispatch loop.
struct AdmitReq {
    req_id: u64,
    /// Токены prompt'а (ChatML + no-think, после sliding window).
    prompt: Vec<u32>,
    params: GenParams,
    out: mpsc::Sender<StreamEvent>,
}

enum IngestMsg {
    Admit(AdmitReq),
    Cancel { req_id: u64 },
}

/// Привязка слота к активному запросу (живёт в dispatch loop).
struct SlotBinding {
    req_id: u64,
    out: mpsc::Sender<StreamEvent>,
    params: GenParams,
    prompt_tokens: usize,
    completion_tokens: usize,
    text_buf: String,
    emitted_chars: usize,
    stop_hit: bool,
    cancelled: bool,
    last_progress: Instant,
}

/// Guard: клиент дропнул Receiver → Cancel в dispatch loop.
/// ponytail: best-effort try_send; переполненный ingest = loop и так
/// обнаружит Closed-канал слота на следующем emit.
pub struct CancelGuard {
    req_id: u64,
    tx: mpsc::Sender<IngestMsg>,
}

impl Drop for CancelGuard {
    fn drop(&mut self) {
        let _ = self.tx.try_send(IngestMsg::Cancel { req_id: self.req_id });
    }
}

pub struct BatchedEngine {
    tx_ingest: mpsc::Sender<IngestMsg>,
    info: ModelInfo,
    next_id: Arc<AtomicU64>,
    max_queue: usize,
    /// Текущая глубина очереди+активные (для reject при переполнении).
    in_flight: Arc<std::sync::atomic::AtomicUsize>,
}

impl BatchedEngine {
    /// Создать engine: загрузить модель (блокирующе — вызывать из
    /// `spawn_blocking` до старта axum) и запустить dispatch loop.
    pub async fn load(cfg: BatchConfig) -> anyhow::Result<Arc<Self>> {
        let cfg = Arc::new(cfg);
        let (tx_ingest, rx_ingest) = mpsc::channel(cfg.max_queue);

        // TODO-F1: здесь Qwen35BatchAdapter::load(Path::new(&cfg.model_path),
        // device, cfg.slots) — блокирующая загрузка GGUF + CUDA/Metal ctx.
        let model = StubModel::load(&cfg).await?;

        let info = ModelInfo {
            id: "qwen3.6-27b".into(),
            context_length: cfg.context_length,
            quant: "Q2_K_XL".into(),
            slots: cfg.slots,
            modes: vec!["thinking".into(), "instruct".into()],
        };

        let engine = Arc::new(Self {
            tx_ingest,
            info,
            next_id: Arc::new(AtomicU64::new(1)),
            max_queue: cfg.max_queue,
            in_flight: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        });

        let in_flight = Arc::clone(&engine.in_flight);
        let cfg2 = Arc::clone(&cfg);
        tokio::spawn(async move {
            dispatch_loop(model, rx_ingest, cfg2, in_flight).await;
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
        // Backpressure: отвергаем сверх max_queue до отправки в loop.
        if self.in_flight.load(Ordering::Relaxed) >= self.max_queue + MAX_SLOTS {
            bail!("queue full (QWEN36_MAX_QUEUE)");
        }

        // TODO-F2: build_chatml_text → sliding window (BD-017) → encode_no_think.
        let prompt = stub_tokenize(&messages);
        let truncated = false; // sliding window выставит

        let (out_tx, out_rx) = mpsc::channel(SLOT_CHAN_CAP);
        let req_id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.in_flight.fetch_add(1, Ordering::Relaxed);
        let req = AdmitReq {
            req_id,
            prompt,
            params: GenParams { ..params },
            out: out_tx,
        };
        self.tx_ingest
            .send(IngestMsg::Admit(req))
            .await
            .map_err(|_| anyhow!("engine dispatch loop dead"))?;
        let _ = truncated; // войдёт в Done через binding
        Ok(out_rx)
    }

    fn model_info(&self) -> ModelInfo {
        self.info.clone()
    }
}

/// Guard привязывается к Receiver на стороне HTTP-слоя: wrap при создании SSE.
/// ponytail: HTTP-агент должен вызывать `BatchedEngine::cancel_guard(req_id)`;
/// до появления хендлеров guard создаётся внутри generate не может (нужен
/// доступ к lifetime Receiver'а) — решается при интеграции с HTTP-слоем.
pub fn cancel_guard(engine: &BatchedEngine, req_id: u64) -> CancelGuard {
    CancelGuard {
        req_id,
        tx: engine.tx_ingest.clone(),
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Dispatch loop — единственный владелец модели/scheduler'а.
// ────────────────────────────────────────────────────────────────────────────

async fn dispatch_loop(
    mut model: StubModel,
    mut rx: mpsc::Receiver<IngestMsg>,
    cfg: Arc<BatchConfig>,
    in_flight: Arc<std::sync::atomic::AtomicUsize>,
) {
    let mut bindings: Vec<Option<SlotBinding>> = (0..cfg.slots).map(|_| None).collect();
    let mut pending: VecDeque<AdmitReq> = VecDeque::new();
    let mut cancelled: HashSet<u64> = HashSet::new();

    loop {
        // 1. Слить все накопленные ingest-сообщения без блокировки.
        while let Ok(msg) = rx.try_recv() {
            match msg {
                IngestMsg::Admit(req) => {
                    if cancelled.remove(&req.req_id) {
                        in_flight.fetch_sub(1, Ordering::Relaxed);
                        continue;
                    }
                    admit(req, &mut model, &mut bindings, &mut pending);
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
                free_slot(idx, &mut bindings, &mut model, &in_flight);
            }
        }

        // 3. Один шаг модели (prefill одного слота ИЛИ batched decode).
        // TODO-F4: заменить на sched.step() реального BatchScheduler.
        let did_work = model_step(&mut model, &mut bindings, &cfg);

        // 4. Сбор завершённых слотов → Done + admit из pending.
        // TODO-F6: для реального scheduler'а — обход sched.slots_mut(),
        // generated_tokens() до Slot::reset().
        for idx in 0..bindings.len() {
            let finished = bindings[idx]
                .as_ref()
                .map(|b| b.stop_hit || b.cancelled || model.slot_finished(idx))
                .unwrap_or(false);
            if finished {
                finish_slot(idx, &mut bindings, &mut model, &in_flight);
                admit_from_pending(&mut pending, &mut model, &mut bindings);
            }
        }

        // 5. Нет работы — ждём новые запросы; есть — yield между шагами.
        if !did_work && pending.is_empty() && bindings.iter().all(|b| b.is_none()) {
            match rx.recv().await {
                Some(msg) => match msg {
                    IngestMsg::Admit(req) => {
                        if cancelled.remove(&req.req_id) {
                            in_flight.fetch_sub(1, Ordering::Relaxed);
                        } else {
                            admit(req, &mut model, &mut bindings, &mut pending);
                        }
                    }
                    IngestMsg::Cancel { req_id } => {
                        cancel(req_id, &mut bindings, &mut pending, &mut cancelled, &in_flight);
                    }
                },
                None => break, // все Sender'ы дропнуты — shutdown
            }
        } else {
            tokio::task::yield_now().await;
        }
    }
}

/// Admit: idle-слот → prefill, иначе в pending-очередь.
fn admit(
    req: AdmitReq,
    model: &mut StubModel,
    bindings: &mut [Option<SlotBinding>],
    pending: &mut VecDeque<AdmitReq>,
) {
    match bindings.iter().position(|b| b.is_none()) {
        Some(idx) => seed_slot(idx, req, model, bindings),
        None => pending.push_back(req),
    }
}

fn admit_from_pending(
    pending: &mut VecDeque<AdmitReq>,
    model: &mut StubModel,
    bindings: &mut [Option<SlotBinding>],
) {
    while let Some(idx) = bindings.iter().position(|b| b.is_none()) {
        let Some(req) = pending.pop_front() else { break };
        seed_slot(idx, req, model, bindings);
    }
}

/// IDLE→PREFILL: загрузить prompt в слот.
fn seed_slot(
    idx: usize,
    req: AdmitReq,
    model: &mut StubModel,
    bindings: &mut [Option<SlotBinding>],
) {
    let prompt_tokens = req.prompt.len();
    // TODO-F4: sched.submit(req.prompt, req.params.max_new) — форк сам
    // переведёт Slot в Prefilling; первый чанк вызовет adapter.prefill_chunk
    // (reset_first → forward() → snapshot_state → seed_slot_batched).
    model.admit(idx, req.prompt, req.params.max_tokens);
    bindings[idx] = Some(SlotBinding {
        req_id: req.req_id,
        out: req.out,
        params: req.params,
        prompt_tokens,
        completion_tokens: 0,
        text_buf: String::new(),
        emitted_chars: 0,
        stop_hit: false,
        cancelled: false,
        last_progress: Instant::now(),
    });
}

/// Отмена: из pending — молча; из слота — пометить (free на шаге сбора).
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
    // Запрос ещё в пути (между generate и admit) — запомнить.
    cancelled.insert(req_id);
}

/// Один шаг: prefill ИЛИ batched decode по активным слотам.
/// Возвращает true, если модель поработала.
fn model_step(
    model: &mut StubModel,
    bindings: &mut [Option<SlotBinding>],
    _cfg: &BatchConfig,
) -> bool {
    // TODO-F4: sched.step(): DidPrefill → первый токен из prefill-логитов;
    // DidDecode(b) → logits per slot → sample (TODO-F5) → push_token.
    let produced = model.step();
    for (idx, toks) in produced {
        let Some(b) = bindings[idx].as_mut() else { continue };
        b.completion_tokens += toks.len();
        b.last_progress = Instant::now();
        drain_slot(idx, toks, bindings);
    }
    model.did_work()
}

/// Токены слота → текст → Delta (инкрементально), stop-строки, EOS.
fn drain_slot(idx: usize, toks: Vec<u32>, bindings: &mut [Option<SlotBinding>]) {
    let Some(b) = bindings[idx].as_mut() else { return };

    // TODO-F3: incremental decode_text(&tokenizer, &generated[..]) с курсором
    // emitted_chars; holdback хвоста max(stop.len())-1 chars до доказанного
    // отсутствия stop-совпадения. Сейчас — грубый per-token текст.
    for t in toks {
        b.text_buf.push_str(&format!("tok{t} "));
    }

    // Stop-строки по накопленному тексту.
    if !b.params.stop.is_empty() && b.params.stop.iter().any(|s| b.text_buf.contains(s)) {
        b.stop_hit = true;
    }

    if b.text_buf.len() > b.emitted_chars {
        let delta = b.text_buf[b.emitted_chars..].to_string();
        b.emitted_chars = b.text_buf.len();
        match b.out.try_send(StreamEvent::Delta(delta)) {
            Ok(()) => {}
            Err(mpsc::error::TrySendError::Full(_)) | Err(mpsc::error::TrySendError::Closed(_)) => {
                // Мёртвый/медленный клиент — отменить слот (§2.5/§2.6 дизайна).
                b.cancelled = true;
            }
        }
    }
}

/// FINISHED: Done с usage → освободить слот.
fn finish_slot(
    idx: usize,
    bindings: &mut [Option<SlotBinding>],
    model: &mut StubModel,
    in_flight: &AtomicUsize,
) {
    let Some(b) = bindings[idx].take() else { return };
    if !b.cancelled {
        let finish_reason = if b.stop_hit {
            "stop"
        } else if b.completion_tokens >= b.params.max_tokens {
            "length"
        } else {
            "stop" // EOS
        };
        let _ = b.out.try_send(StreamEvent::Done {
            finish_reason: finish_reason.into(),
            prompt_tokens: b.prompt_tokens,
            completion_tokens: b.completion_tokens,
            truncated: false, // TODO-F2: из sliding window
        });
    }
    // TODO-F4/F6: Slot::reset() + BatchModel::reset_slot(idx).
    model.reset_slot(idx);
    in_flight.fetch_sub(1, Ordering::Relaxed);
}

/// Освобождение без Done (watchdog-Error уже отправлен).
fn free_slot(
    idx: usize,
    bindings: &mut [Option<SlotBinding>],
    model: &mut StubModel,
    in_flight: &AtomicUsize,
) {
    if bindings[idx].take().is_some() {
        model.reset_slot(idx);
        in_flight.fetch_sub(1, Ordering::Relaxed);
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Stub-модель: тот же интерфейс шагов, что у BatchScheduler (submit/step/reset).
// TODO-F1/F4: удалить при подключении qwen35-batch.
// ────────────────────────────────────────────────────────────────────────────

struct StubSlot {
    remaining: usize,
    active: bool,
}

struct StubModel {
    slots: Vec<StubSlot>,
    did_work: bool,
}

impl StubModel {
    async fn load(cfg: &BatchConfig) -> anyhow::Result<Self> {
        // ponytail: загрузка GGUF — фаза интеграции; проверяем только env-конфиг.
        if cfg.model_path.is_empty() {
            bail!("QWEN36_MODEL пуст");
        }
        Ok(Self {
            slots: (0..cfg.slots)
                .map(|_| StubSlot { remaining: 0, active: false })
                .collect(),
            did_work: false,
        })
    }

    fn admit(&mut self, idx: usize, _prompt: Vec<u32>, max_new: usize) {
        self.slots[idx].remaining = max_new.min(64); // stub: короткие генерации
        self.slots[idx].active = true;
    }

    /// Один «decode-шаг» по всем активным слотам → токены per slot.
    fn step(&mut self) -> Vec<(usize, Vec<u32>)> {
        let mut out = Vec::new();
        self.did_work = false;
        for (idx, s) in self.slots.iter_mut().enumerate() {
            if s.active && s.remaining > 0 {
                s.remaining -= 1;
                out.push((idx, vec![idx as u32]));
                self.did_work = true;
            }
        }
        out
    }

    fn slot_finished(&self, idx: usize) -> bool {
        self.slots[idx].active && self.slots[idx].remaining == 0
    }

    fn reset_slot(&mut self, idx: usize) {
        self.slots[idx].active = false;
        self.slots[idx].remaining = 0;
    }

    fn did_work(&self) -> bool {
        self.did_work
    }
}

/// TODO-F2: заменить на tokenizer::build_chatml_text + encode_no_think.
fn stub_tokenize(messages: &[ChatMessage]) -> Vec<u32> {
    let mut v = Vec::new();
    for m in messages {
        v.extend(m.content.bytes().map(|b| b as u32));
    }
    v
}

// Self-check механики без GGUF: очередь >4, cancel, Done.
// Запуск: cargo test engine_batched
#[cfg(test)]
mod tests {
    use super::*;
    use futures::future::join_all;

    fn cfg(slots: usize) -> BatchConfig {
        BatchConfig {
            model_path: "stub.gguf".into(),
            slots,
            max_queue: 8,
            req_timeout: Duration::from_secs(60),
            context_length: 81920,
        }
    }

    fn msgs(s: &str) -> Vec<ChatMessage> {
        vec![ChatMessage { role: "user".into(), content: s.into() }]
    }

    async fn collect(mut rx: mpsc::Receiver<StreamEvent>) -> (usize, bool) {
        let mut deltas = 0;
        let mut done = false;
        while let Some(ev) = rx.recv().await {
            match ev {
                StreamEvent::Delta(_) => deltas += 1,
                StreamEvent::Done { .. } => {
                    done = true;
                    break;
                }
                StreamEvent::Error(e) => panic!("stream error: {e}"),
            }
        }
        (deltas, done)
    }

    #[tokio::test]
    async fn eight_requests_four_slots_all_done() {
        let engine = BatchedEngine::load(cfg(4)).await.unwrap();
        let handles: Vec<_> = (0..8)
            .map(|_i| {
                let e = Arc::clone(&engine);
                tokio::spawn(async move {
                    let rx = e.generate(msgs("hello"), GenParams::default()).await.unwrap();
                    collect(rx).await
                })
            })
            .collect();
        let results = join_all(handles).await;
        assert_eq!(results.len(), 8);
        for r in results {
            let (deltas, done) = r.unwrap();
            assert!(deltas > 0, "нет deltas");
            assert!(done, "нет Done");
        }
        assert_eq!(engine.in_flight.load(Ordering::Relaxed), 0, "утечка in_flight");
    }

    #[tokio::test]
    async fn cancel_before_admit_does_not_leak() {
        let engine = BatchedEngine::load(cfg(1)).await.unwrap();
        // Занять единственный слот надолго.
        let _hold = engine
            .generate(msgs("hold"), GenParams { max_tokens: 4096, ..Default::default() })
            .await
            .unwrap();
        // Второй уйдёт в pending; дропаем receiver сразу.
        let rx = engine.generate(msgs("drop me"), GenParams::default()).await.unwrap();
        drop(rx);
        // Отменяем второй (req_id достаём атомарно, не хардкодом).
        let req_id = engine.next_id.load(Ordering::Relaxed) - 1;
        engine.tx_ingest.send(IngestMsg::Cancel { req_id }).await.unwrap();
        // Дождаться обработки Cancel (условие вместо фиксированного sleep).
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            // Дать loop'у шаги; hold-слот может ещё жить (64 stub-токена —
            // тысячи yield-итераций), поэтому проверяем только отсутствие
            // утечки ПОСЛЕ завершения hold ниже.
            tokio::time::sleep(Duration::from_millis(5)).await;
            if Instant::now() > deadline {
                break;
            }
            // Отменённый вычтен, когда Cancel обработан: in_flight ≤ 1 (только hold)
            if engine.in_flight.load(Ordering::Relaxed) <= 1 {
                break;
            }
        }
        assert!(
            engine.in_flight.load(Ordering::Relaxed) <= 1,
            "отменённый запрос не вычтен из in_flight"
        );
    }
}
