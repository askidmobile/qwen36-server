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
    StreamEvent, TokenLogprob,
};
use crate::media::prepare::{PreparedContentBlock, PreparedLease};
use crate::prefix_cache::{HostSnapshotWorker, PrefixCache};
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
        let get = |k: &str| std::env::var(k).ok();
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
    /// Позиция, на которой режется первый prefill-чанк (prefix-cache friendly
    /// split). `None` = целиком по PREFILL_CHUNK.
    first_chunk_cut: Option<usize>,
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

/// Готовый промпт после Jinja-шаблона (PROMPT_DEBUG=путь к файлу).
///
/// `REQ_DEBUG` пишет то, что прислал клиент, а модель видит результат рендера —
/// и расходятся именно они. Сверка 2026-09-04 с llama.cpp на одном и том же
/// наборе сообщений дала 25281 токен у нас против 23869 у эталона; без текста
/// промпта такую разницу локализовать нечем.
///
/// Файл перезаписывается на каждом запросе: нужен последний, а не история.
fn dump_prompt(text: &str) {
    static PATH: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    let Some(path) = PATH.get_or_init(|| std::env::var("PROMPT_DEBUG").ok()) else {
        return;
    };
    let _ = std::fs::write(path, text);
}

/// Идентификаторы токенов того же промпта (PROMPT_DEBUG + суффикс `.ids.json`).
/// Текст промпта совпадал с llama.cpp байт в байт, а число токенов — нет
/// (4452 против 4208 на 18 КБ, 2026-09-04): расходится сам токенизатор, и
/// сравнивать надо разбиения, а не текст.
fn dump_prompt_ids(ids: &[u32]) {
    static PATH: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    let Some(path) = PATH.get_or_init(|| std::env::var("PROMPT_DEBUG").ok()) else {
        return;
    };
    let body: Vec<String> = ids.iter().map(|i| i.to_string()).collect();
    let _ = std::fs::write(format!("{path}.ids.json"), format!("[{}]", body.join(",")));
}

/// Хостовые фазы шага (HOST_TIMING=1): сэмплер и drain отдельно от
/// ожидания GPU. CPU-время процесса здесь бесполезно — при spin-wait драйвера
/// оно тождественно времени на стене.
static HOST_SAMPLE_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static HOST_SAMPLE_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static HOST_DRAIN_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Состояние сэмплера слота: параметры, RNG и инкрементальное множество
/// встречавшихся токенов для penalties. Раньше множество собиралось заново из
/// всей generated на каждом токене — O(n) на токен, O(n²) на запрос.
#[derive(Clone)]
struct SlotSampler {
    params: GenParams,
    rng: Rng,
    seen: HashSet<u32>,
    /// Сколько первых токенов generated уже учтено в `seen`.
    seen_len: usize,
    /// Logprobs токенов, выданных сэмплером с прошлого дренажа, в порядке
    /// выдачи. Дренаж забирает их и приписывает к дельте. Со спекуляцией
    /// сэмплер вызывается построчно по логитам проверки, поэтому очередь
    /// накапливает все принятые токены шага — один к одному.
    pending_logprobs: Vec<sampler::TokenLogprobs>,
}
type SlotSamplers = Arc<Mutex<HashMap<usize, SlotSampler>>>;

struct SlotBinding {
    out: mpsc::Sender<StreamEvent>,
    params: GenParams,
    prompt_tokens: usize,
    completion_tokens: usize,
    /// ретроактивно менять ранние байты (многотокенные UTF-8), поэтому
    /// индекс небезопасен — сравниваем префиксы.
    emitted_text: String,
    /// Инкрементальный декод: токены [..stable_toks] декодированы окончательно
    /// (без U+FFFD на конце) и целиком лежат в emitted_text[..stable_len].
    stable_toks: usize,
    stable_len: usize,
    stop_hit: bool,
    /// Генерация зациклилась: один и тот же кусок повторяется без конца.
    looped: bool,
    cancelled: bool,
    last_progress: Instant,
    usage: MediaUsage,
    _media_lease: Option<PreparedLease>,
    cancel: crate::engine_types::CancelFlag,
    /// Сколько токенов generated уже проверено на think-open/close.
    phase_checked: usize,
    /// Идут ли сейчас токены размышления. По контракту OpenAI в logprobs
    /// попадают только токены ответа. Фаза определяется ИДЕНТИФИКАТОРОМ
    /// токена `</think>`, а не текстом: сплиттер режет текст и придерживает
    /// хвосты, а граница размышления — конкретный токен, и совпадать они не
    /// обязаны. Если размышление выключено или модель его не открыла — всё
    /// считается ответом.
    in_reasoning: bool,
    /// Токены промпта — только для записи в prefix cache после префила
    /// (пуст при выключенном кеше или медиа-запросе).
    prompt: Vec<u32>,
    /// Снимок после префила уже снят в кеш.
    prefix_captured: bool,
}

pub struct BatchedEngine {
    tx_ingest: mpsc::Sender<IngestMsg>,
    info: ModelInfo,
    max_queue: usize,
    in_flight: Arc<AtomicUsize>,
    tokenizer: Arc<Mutex<tokenizers::Tokenizer>>,
    media: Arc<crate::media::MediaService>,
    vision_path: Option<std::path::PathBuf>,
    mtp_path: Option<std::path::PathBuf>,
    /// Shutdown-флаг dispatch thread: выставляется при drop engine (unload).
    shutdown: Arc<AtomicBool>,
    /// FR-020: состояние выгрузки экспертов MoE — заполняет dispatch thread
    /// после загрузки адаптера; /v1/models читает для capabilities.moe.
    moe: Arc<std::sync::OnceLock<crate::engine::MoeInfo>>,
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
        // Prefix cache: включается по PREFIX_CACHE_MIB>0 (default 0 =
        // выключен). Int8-пул KV снимками теперь покрыт: снимок переносит
        // байты вместе с масштабами постранично, без де/реквантования.
        let cfg = Arc::new(cfg);
        // Опциональные упреждающие f16-зеркала KV (P1): выгодны на картах с
        // большим запасом VRAM или при одном активном длинном запросе.
        // По умолчанию ВЫКЛЮЧЕНЫ: на 12 GB полное покрытие 24K x 2 слота
        // (~960 MiB) оставляет <100 MiB на транзиенты префилла → регрессия
        // (урок 2026-08-23). Включать явно через KV_MIRROR_TOKENS.
        #[cfg(feature = "cuda")]
        if let Some(mirror_tokens) = std::env::var("KV_MIRROR_TOKENS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
        {
            std::env::set_var("KV_MIRROR_PREPARE", mirror_tokens.to_string());
        }
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

        let moe = Arc::new(std::sync::OnceLock::new());
        let _moe2 = Arc::clone(&moe);
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
            mtp_path: mtp_path.clone(),
            moe,
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
        let mtp_path2 = engine.mtp_path.clone();
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
                // MTP запрошен явно, значит его провал — ошибка запуска, а не
                // повод молча работать без спекуляции: сервер в таком виде
                // отвечает и выглядит исправным, а конфигурация не та, что
                // просили. На этом сгорел целый контрольный замер.
                if let Err(e) = adapter.load_mtp(&path) {
                    eprintln!("[dispatch] mtp load failed: {e:#}");
                    *load_error.write().expect("load_error lock") =
                        format!("MTP запрошен, но не загрузился: {e:#}");
                    return;
                }
            }
            // FR-020: сводка выгрузки → /v1/models.
            #[cfg(feature = "cuda")]
            if let Some(mi) = adapter.moe_summary() {
                let _ = _moe2.set(crate::engine::MoeInfo {
                    experts: "ram",
                    pinned_mib: mi.pinned_bytes / 1024 / 1024,
                    staging_mib: mi.staging_bytes / 1024 / 1024,
                    cache_mib: mi.cache_mib,
                    cache_slots: mi.cache_slots,
                    hit_rate: mi.hit_rate,
                });
            }
            #[cfg(feature = "cuda")]
            crate::engine::maybe_retain_mempool(&device);
            let eos = adapter.eos();
            let vocab = adapter.vocab_size();
            // Лишний снимок на границе чанка стоит копии KV в VRAM — снимаем
            // только когда кеш префикса действительно включён.
            adapter.set_prefix_capture(cfg2.prefix_cache_mib > 0);
            let scheduler = BatchScheduler::new(adapter, cfg2.slots, eos, vocab);
            ready.store(true, Ordering::Relaxed);
            let cache =
                (cfg2.prefix_cache_mib > 0).then(|| PrefixCache::new(cfg2.prefix_cache_mib));
            // Перенос снимков в host-память — в отдельном потоке: D2H ~0.5 ГиБ
            // на 30k контекста иначе задерживает первый токен ответа.
            let host_worker = cache.as_ref().map(|_| HostSnapshotWorker::new());
            dispatch_loop(
                scheduler,
                rx_ingest,
                cfg2,
                in_flight,
                tokenizer,
                shutdown2,
                cache,
                host_worker,
            );
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
            bail!("queue full (MAX_QUEUE)");
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
            if was_trimmed && crate::engine::ctx_overflow_is_error() {
                let total: usize = messages.iter().map(count).sum();
                return Err(crate::engine::ContextOverflow {
                    prompt_tokens: total,
                    limit: budget,
                }
                .into());
            }
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
                Some(text) => {
                    dump_prompt(&text);
                    let ids = tok
                        .encode(text.as_str(), false)
                        .map(|e| e.get_ids().to_vec())
                        .map_err(|e| anyhow!("encode prompt: {e}"))?;
                    dump_prompt_ids(&ids);
                    ids
                }
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

        // Граница, на которой промпт станет cacheable-префиксом следующего
        // хода: начало генерационного суффикса, то есть токен `<|im_start|>`
        // последнего сообщения. Режем первый чанк ровно здесь — иначе промпт
        // длиннее одного чанка не даёт снимка, пригодного следующему ходу.
        let first_chunk_cut = {
            let tok = self.tokenizer.lock().unwrap_or_else(|e| e.into_inner());
            tok.token_to_id("<|im_start|>")
                .and_then(|id| prompt.iter().rposition(|t| *t == id))
                .filter(|p| *p > 0 && *p < prompt.len())
        };

        let (out_tx, out_rx) = mpsc::channel(SLOT_CHAN_CAP);
        self.in_flight.fetch_add(1, Ordering::Relaxed);
        let req = AdmitReq {
            prompt,
            first_chunk_cut,
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

    fn supports_vision(&self) -> bool {
        self.vision_path.is_some()
    }
    fn supports_video(&self) -> bool {
        self.vision_path.is_some()
    }
    // Веса MTP заданы И загрузились: при провале load_error непуст и движок
    // до готовности не доходит, поэтому одного пути достаточно.
    fn supports_mtp(&self) -> bool {
        self.mtp_path.is_some()
    }
    fn moe_info(&self) -> Option<crate::engine::MoeInfo> {
        self.moe.get().cloned()
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
    mut cache: Option<PrefixCache>,
    host_worker: Option<HostSnapshotWorker>,
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

    // P0.5b: куда уходит wall-time шага (decode-фазы внутри шедулера уже
    // считаются в stats.decode_ns/prefill_ns). gap = step_wall - decode - prefill.
    let mut agg = (0u64, 0u64, 0u64, 0u64, 0u64); // steps, decode, prefill, gap, tokens_delta
    let mut agg_last_tokens = 0u64;
    // Время МЕЖДУ шагами планировщика: слив выхода, токенизатор, отправка в
    // поток. Внутренний gap агрегата давно ноль, а сквозная скорость стабильно
    // на 10-15% ниже той, что обещает арифметика раундов, — остаток искали
    // именно здесь.
    let mut host_ns: u64 = 0;
    let mut t_after_step: Option<std::time::Instant> = None;
    // Копим промежуток ТОЛЬКО после шага, который реально работал: иначе в
    // сумму попадает ожидание запросов, и число раздувается в разы.
    let mut last_step_busy = false;
    let host_timing_on = std::env::var("HOST_TIMING")
        .map(|v| v == "1")
        .unwrap_or(false);
    let (mut host_steps, mut host_step_ns) = (0u64, 0u64);
    let timing_on = std::env::var("MTP_TIMING")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let mut since_print = 0u64;
    // Подряд идущие ошибки шага (см. ветку Err ниже): после третьей
    // освобождаем все слоты с отменёнными запросами.
    let mut step_errors = 0u32;

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
                    &mut cache,
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
        let st_before = sched.stats_snapshot();
        if let Some(prev) = t_after_step.take() {
            if last_step_busy {
                host_ns += prev.elapsed().as_nanos() as u64;
            }
        }
        let t_step = std::time::Instant::now();
        let outcome = sched.step_with(&mut |sidx, _generated| {
            bindings
                .get(sidx)
                .and_then(|b| b.as_ref())
                .map(|b| b.cancelled)
                .unwrap_or(false)
        });
        let step_ns = t_step.elapsed().as_nanos() as u64;
        t_after_step = Some(std::time::Instant::now());
        if timing_on {
            let st_after = sched.stats_snapshot();
            let dec = st_after.decode_ns.saturating_sub(st_before.decode_ns) as u64;
            let pre = st_after.prefill_ns.saturating_sub(st_before.prefill_ns) as u64;
            let toks = st_after.total_decode_tokens as u64;
            agg.0 += 1;
            agg.1 += dec;
            agg.2 += pre;
            agg.4 += toks.saturating_sub(agg_last_tokens);
            agg_last_tokens = toks;
            agg.3 += step_ns.saturating_sub(dec + pre);
            last_step_busy = dec + pre > 0;
            since_print += 1;
            if since_print >= 48 {
                let (st, dec, pre, gap, tk) = agg;
                eprintln!(
                    "[mtp-agg] steps={st} decode={dec_ms:.0}ms prefill={pre_ms:.0}ms gap={gap_ms:.0}ms между_шагами={host_ms:.0}ms tokens={tk}",
                    st = st,
                    dec_ms = dec as f64 / 1e6,
                    pre_ms = pre as f64 / 1e6,
                    gap_ms = gap as f64 / 1e6,
                    host_ms = host_ns as f64 / 1e6,
                    tk = tk,
                );
                since_print = 0;
            }
        }
        if host_timing_on {
            host_steps += 1;
            host_step_ns += step_ns;
            if host_steps % 48 == 0 {
                let s_ns = HOST_SAMPLE_NS.swap(0, Ordering::Relaxed);
                let s_n = HOST_SAMPLE_CALLS.swap(0, Ordering::Relaxed);
                let d_ns = HOST_DRAIN_NS.swap(0, Ordering::Relaxed);
                eprintln!(
                    "[host] steps=48 step={:.2}ms sample={:.3}ms/call x{} drain={:.3}ms/step",
                    host_step_ns as f64 / 48e6,
                    if s_n > 0 {
                        s_ns as f64 / s_n as f64 / 1e6
                    } else {
                        0.0
                    },
                    s_n,
                    d_ns as f64 / 48e6,
                );
                host_step_ns = 0;
            }
        }
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
                        &slot_samplers,
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
                    &slot_samplers,
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
                // Слот, на котором упал шаг, обязан вернуться в Idle. Раньше
                // ветка просто возвращала `true`, следующий шаг снова брал тот
                // же чанк prefill, снова получал ту же ошибку — и так до
                // перезапуска сервиса с GPU на 100 %. Воспроизведено
                // 2026-09-16: CUDA_ERROR_OUT_OF_MEMORY на prefill и на
                // повторном префиле из prefix-кеша (chunk 16384).
                step_errors += 1;
                for idx in 0..bindings.len() {
                    if !bindings[idx]
                        .as_ref()
                        .map(|b| b.cancelled)
                        .unwrap_or(false)
                    {
                        continue;
                    }
                    let st = sched.slots_mut()[idx].status;
                    // Prefilling-слот чиним сразу (это и есть зацикливающийся
                    // случай). Прочие статусы — после третьей подряд ошибки,
                    // чтобы не разбирать чужие запросы из-за одной транзиентной.
                    if st == SlotStatus::Prefilling
                        || (step_errors >= 3 && st != SlotStatus::Idle)
                    {
                        sched.slots_mut()[idx].reset();
                    }
                }
                false
            }
        };
        // Успешный шаг сбрасывает счётчик подряд идущих ошибок (Err-ветка
        // всегда возвращает did_work = false).
        if did_work {
            step_errors = 0;
        }

        // Prefix cache: снять снимок после завершения префила. В этот момент
        // adapter хранит состояние ровно на длине промпта (последний чанк
        // prefill), декод ещё не менял slot_snaps.
        if let Some(pc) = cache.as_mut() {
            // Сначала забираем то, что воркер уже перенёс в host-память.
            if let Some(worker) = host_worker.as_ref() {
                for (tokens, host_snaps) in worker.drain() {
                    pc.put_many(&tokens, host_snaps);
                }
            }
            for (idx, b) in bindings.iter_mut().enumerate() {
                let Some(b) = b else { continue };
                if b.prefix_captured || b.cancelled || b._media_lease.is_some() {
                    continue;
                }
                if sched.slots_mut()[idx].status != SlotStatus::Decoding {
                    continue;
                }
                b.prefix_captured = true;
                // Кешируем не конец промпта, а границу последнего чанка
                // префила: хвост промпта (суффикс генерации) на следующем
                // ходу заменяется ответом ассистента, и запись во всю длину
                // перестаёт быть его префиксом — попадания не было бы никогда.
                let snapshots = sched.model_mut().take_prefix_snapshots(idx);
                if snapshots.is_empty() {
                    eprintln!("[pcache] no boundary snapshot for slot {idx}");
                } else {
                    let prompt = std::mem::take(&mut b.prompt);
                    let total = snapshots.len();
                    match host_worker.as_ref() {
                        Some(worker) => {
                            worker.submit(prompt, snapshots);
                            eprintln!(
                                "[pcache] snapshots queued: {total}, entries {}, bytes {} MiB",
                                pc.len(),
                                pc.total_bytes() / (1024 * 1024),
                            );
                        }
                        None => {
                            let saved = pc.put_many(&prompt, snapshots);
                            eprintln!(
                                "[pcache] snapshots saved: {saved}/{total}, entries {}, bytes {} MiB",
                                pc.len(),
                                pc.total_bytes() / (1024 * 1024),
                            );
                        }
                    }
                }
            }
        }

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
                &mut cache,
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
            let pc = cache
                .as_ref()
                .map(|c| format!("{} ent/{}MiB", c.len(), c.total_bytes() / (1024 * 1024)))
                .unwrap_or_else(|| "off".into());
            eprintln!(
                "[hb] pending={} slots={} pcache={}",
                pending.len(),
                st.join(" "),
                pc
            );
            last_hb = Instant::now();
        }

        // 5. Нет работы — ждём с polling shutdown (таймаут 50мс, иначе shutdown
        // не виден пока не придёт сообщение; blocking_recv блокирует навечно).
        if !did_work && pending.is_empty() && bindings.iter().all(|b| b.is_none()) {
            // Спим коротко: 50 мс стояли прямо в TTFT коротких запросов (замер
            // 2026-09-16: 11-токенный промпт отвечал ~0.24 с при ~25 мс работы).
            // Верхняя граница ожидания нового запроса теперь 5 мс.
            std::thread::sleep(std::time::Duration::from_millis(5));
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
                    &mut cache,
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
/// Один токен из сэмплера в вид, годный для API: текст, честные байты, число.
///
/// Байты берём из decode_bytes, а не из строки: для байтовых и неполных
/// UTF-8 токенов строка обязана содержать U+FFFD, а байты — оставаться
/// настоящими, иначе клиент не восстановит границы.
fn to_api_logprob(
    tokenizer: &tokenizers::Tokenizer,
    id: u32,
    logprob: f32,
    top: &[(u32, f32)],
) -> TokenLogprob {
    let one = |id: u32, lp: f32| -> TokenLogprob {
        let bytes = tokenizer::decode_bytes(tokenizer, &[id]).unwrap_or_default();
        TokenLogprob {
            token: String::from_utf8_lossy(&bytes).into_owned(),
            bytes,
            logprob: lp,
            top: Vec::new(),
        }
    };
    let mut entry = one(id, logprob);
    entry.top = top.iter().map(|&(i, lp)| one(i, lp)).collect();
    entry
}

fn drain_after_step(
    sched: &mut BatchScheduler<Qwen35BatchAdapter>,
    bindings: &mut [Option<SlotBinding>],
    slot_emitted_toks: &mut HashMap<usize, usize>,
    tokenizer: &tokenizers::Tokenizer,
    slot_samplers: &SlotSamplers,
) {
    let t_drain = Instant::now();
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
        // Забираем logprobs токенов, появившихся с прошлого дренажа. Привязка
        // к токену, а не к тексту: дренаж придерживает хвост, заканчивающийся
        // на U+FFFD, поэтому в таком чанке токен уже есть, а текста ещё нет.
        // Число записей равно числу новых токенов, текст догоняет следующим.
        let pending = slot_samplers
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get_mut(&idx)
            .map(|st| std::mem::take(&mut st.pending_logprobs))
            .unwrap_or_default();
        // Фаза по идентификатору токена для ВСЕХ запросов: length-обрыв внутри
        // think-блока обязан докладываться в Done.ended_in_thinking, а фаза
        // раньше обновлялась только когда клиент просил logprobs. Новые токены
        // [phase_checked..generated.len()) сверяем с id think-open/close.
        // ID фиксированы константами форка (у Ornith/Qwen3.8 те же ID 248068/69;
        // token_to_id по строкам из vocab Qwen3.5 ненадёжен — glyph не строковый).
        const THINK_OPEN_ID: u32 = 248068;
        const THINK_CLOSE_ID: u32 = 248069;
        let _ = tokenizer;
        for &t in &generated[b.phase_checked..] {
            if t == THINK_CLOSE_ID {
                b.in_reasoning = false;
            } else if t == THINK_OPEN_ID {
                b.in_reasoning = true;
            }
        }
        b.phase_checked = generated.len();
        // В logprobs по контракту идут только токены ответа. Сами
        // `<think>` и `</think>` туда не попадают. Записи выравниваем по
        // хвосту generated: если очередь разошлась с историей (откат
        // спекуляции), считаем всё ответом, чем врать о фазе.
        let chunk_logprobs: Option<Vec<TokenLogprob>> = b.params.logprobs.map(|_| {
            let think_open = tokenizer.token_to_id("<think>");
            let think_close = tokenizer.token_to_id("</think>");
            let tail_start = generated.len().saturating_sub(pending.len());
            let aligned = pending
                .iter()
                .enumerate()
                .all(|(i, lp)| generated.get(tail_start + i) == Some(&lp.token));
            let mut out = Vec::with_capacity(pending.len());
            for lp in pending.iter() {
                if !aligned {
                    out.push(to_api_logprob(tokenizer, lp.token, lp.logprob, &lp.top));
                    continue;
                }
                if Some(lp.token) == think_close {
                    b.in_reasoning = false;
                    continue;
                }
                if Some(lp.token) == think_open {
                    b.in_reasoning = true;
                    continue;
                }
                if b.in_reasoning {
                    continue;
                }
                let entry = to_api_logprob(tokenizer, lp.token, lp.logprob, &lp.top);
                // Служебные токены записей не получают вовсе: у OpenAI
                // стоп-токен не входит ни в content, ни в logprobs, а у нас он
                // декодируется в пустую строку — для клиента это мусор с
                // непустым числом. Признак служебности — пустые байты:
                // decode_bytes отбрасывает токены вида <|...|>.
                if entry.bytes.is_empty() {
                    continue;
                }
                out.push(entry);
            }
            out
        });
        b.completion_tokens = generated.len();
        b.last_progress = Instant::now();

        // Инкрементальный декод: токены [..stable_toks] декодированы
        // окончательно и целиком лежат в emitted_text[..stable_len]; декодируем
        // только хвост. decode_text собирает байты всех токенов и делает один
        // from_utf8_lossy, поэтому decode(prefix) + decode(tail) == decode(all),
        // пока граница не режет многобайтовый символ — на такой границе
        // (U+FFFD на конце) мы не останавливаемся. Раньше decode шёл по всей
        // generated на каждом шаге и emitted_text копировался целиком:
        // O(n) на токен, O(n²) на запрос.
        let tail_text =
            tokenizer::decode_text(tokenizer, &generated[b.stable_toks..]).unwrap_or_default();
        let emitted_tail_len = b.emitted_text.len() - b.stable_len;

        // Stop-строки: ищем в хвосте длиной max_stop_len + последний кусок.
        let max_stop_len = b.params.stop.iter().map(|s| s.len()).max().unwrap_or(0);
        let mut cut_at: Option<usize> = None;
        if !b.params.stop.is_empty() {
            let scan_from = tail_text.len().saturating_sub(max_stop_len + 64);
            let scan_from = floor_char_boundary(&tail_text, scan_from)
                .max(emitted_tail_len.min(tail_text.len()));
            if let Some(rel) = tail_text[scan_from..].find_any(&b.params.stop) {
                cut_at = Some(scan_from + rel);
            }
        }

        let mut end = cut_at.unwrap_or(tail_text.len());
        // Holdback: не эмитим хвост, заканчивающийся на U+FFFD — это может быть
        // недо-собранная UTF-8 последовательность (emoji/CJK разрезаны на
        // несколько токенов); следующий токен достроит. Иначе устаревший '�'
        // уходит клиенту и префикс расходится навсегда. Флаш — в finish_slot.
        while end > emitted_tail_len && tail_text[..end].ends_with('\u{FFFD}') {
            end -= '\u{FFFD}'.len_utf8();
        }
        // Префиксное сравнение хвоста: если decode ретроактивно изменил байты —
        // ресинхронизируемся (не эмитим на этом шаге), индексной арифметики нет.
        let prefix_ok = tail_text.starts_with(&b.emitted_text[b.stable_len..]);
        if prefix_ok && end >= emitted_tail_len {
            let delta = tail_text[emitted_tail_len..end].to_string();
            let has_logprobs = chunk_logprobs.as_ref().is_some_and(|v| !v.is_empty());
            if !delta.is_empty() || has_logprobs {
                match b.out.try_send(StreamEvent::Delta {
                    text: delta.clone(),
                    logprobs: chunk_logprobs,
                }) {
                    Ok(()) => {}
                    Err(mpsc::error::TrySendError::Full(_))
                    | Err(mpsc::error::TrySendError::Closed(_)) => {
                        b.cancelled = true;
                    }
                }
                b.emitted_text.push_str(&delta);
            }
        } else if !prefix_ok {
            b.emitted_text.truncate(b.stable_len);
            b.emitted_text.push_str(&tail_text[..end]);
        }
        // Стабильная граница: хвост эмитирован целиком и не обрывает символ.
        if cut_at.is_none()
            && end == tail_text.len()
            && !tail_text.ends_with('\u{FFFD}')
            && b.emitted_text.len() == b.stable_len + tail_text.len()
        {
            b.stable_toks = generated.len();
            b.stable_len = b.emitted_text.len();
        }
        if cut_at.is_some() {
            b.stop_hit = true;
        }
        // Зацикливание: модель повторяет один и тот же блок токенов и без
        // потолка max_tokens не остановится. Наблюдалось вживую — 15 000
        // токенов подряд после того, как модель заметила свою же опечатку и
        // ушла в бесконечную самопроверку. repetition_penalty у карточки
        // Ornith равен 1.0, то есть штраф выключен и не мешает повтору.
        //
        // Проверяем хвост: если последние N токенов встречались подряд
        // MIN_REPEATS раз, дальше смысла нет. Порог с запасом: осмысленный
        // текст (списки, таблицы, повторяющаяся разметка) даёт совпадения
        // короткие и не подряд.
        if !b.stop_hit && !b.looped && loop_guard_enabled() {
            if let Some(period) = detect_loop(generated) {
                b.looped = true;
                b.stop_hit = true;
                eprintln!(
                    "[loop] slot {idx}: генерация зациклилась (период {period} токенов,                      сгенерировано {}), обрываем",
                    generated.len()
                );
            }
        }
        slot_emitted_toks.insert(idx, generated.len());
    }
    HOST_DRAIN_NS.fetch_add(t_drain.elapsed().as_nanos() as u64, Ordering::Relaxed);
}

/// Минимум повторов подряд, при котором хвост считается петлёй.
const LOOP_MIN_REPEATS: usize = 4;
/// Максимальная длина повторяющегося блока, которую ищем.
const LOOP_MAX_PERIOD: usize = 96;
/// Раньше этого числа токенов не смотрим: короткий ответ с повторами
/// (список, таблица) — норма, а не петля.
const LOOP_MIN_TOKENS: usize = 256;
/// Сколько токенов обязан покрыть повторяющийся хвост, чтобы считаться петлёй.
/// Без этого порога период 1 срабатывал на четырёх одинаковых токенах подряд —
/// `9999` в `z-index`, `----` в разделителе, `0000` в цвете, — и рвал вёрстку
/// посреди файла. Настоящая петля занимает десятки токенов, а не единицы.
const LOOP_MIN_SPAN: usize = 96;

/// Сторож зацикливания. `LOOP_GUARD=0` выключает.
fn loop_guard_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("LOOP_GUARD").as_deref() != Ok("0"))
}

/// Период повторяющегося хвоста, если генерация зациклилась.
///
/// Хвост длиной `period` должен повториться подряд `LOOP_MIN_REPEATS` раз.
/// Возвращает самый короткий такой период — он точнее описывает петлю.
fn detect_loop(generated: &[u32]) -> Option<usize> {
    let n = generated.len();
    if n < LOOP_MIN_TOKENS {
        return None;
    }
    for period in 1..=LOOP_MAX_PERIOD.min(n / LOOP_MIN_REPEATS) {
        // Одного числа повторов мало. `z-index: 9999` — это четыре одинаковых
        // токена подряд, то есть формально «период 1, четыре повтора», и по
        // старому порогу генерация обрывалась прямо посреди CSS. Поэтому
        // повторы обязаны покрыть ещё и LOOP_MIN_SPAN токенов: для периода 1
        // это 96 одинаковых токенов подряд, для периода 24 — четыре повтора.
        let repeats = LOOP_MIN_REPEATS.max(LOOP_MIN_SPAN.div_ceil(period));
        let need = period * repeats;
        if n < need {
            continue;
        }
        let tail = &generated[n - need..];
        let first = &tail[..period];
        if tail.chunks_exact(period).all(|c| c == first) {
            return Some(period);
        }
    }
    None
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
    cache: &mut Option<PrefixCache>,
) {
    match bindings.iter().position(|b| b.is_none()) {
        Some(idx) if kv_fits(sched, bindings, cfg, req.prompt_tokens) => seed_slot(
            idx,
            req,
            sched,
            bindings,
            slot_samplers,
            slot_truncated,
            cache,
        ),
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
    cache: &mut Option<PrefixCache>,
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
        seed_slot(
            idx,
            req,
            sched,
            bindings,
            slot_samplers,
            slot_truncated,
            cache,
        );
    }
}

/// IDLE→PREFILL: submit в scheduler (TODO-F4). При попадании в prefix cache —
/// primed-admit: снимок восстанавливается в слот, prefill досчитывает только
/// хвост промпта. None от submit_primed (нет свободного слота шедулера) —
/// трактуется как промах, обычный submit.
#[allow(clippy::too_many_arguments)]
fn seed_slot(
    idx: usize,
    req: AdmitReq,
    sched: &mut BatchScheduler<Qwen35BatchAdapter>,
    bindings: &mut [Option<SlotBinding>],
    slot_samplers: &SlotSamplers,
    slot_truncated: &mut HashMap<usize, bool>,
    cache: &mut Option<PrefixCache>,
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

    // Медиа-запросы мимо кеша: снимок не покрывает vision-фичи.
    let cacheable = cache.is_some() && req.media.is_none();
    // Промпт нужен в binding только для записи в кеш после префила.
    let prompt_for_cache = if cacheable {
        req.prompt.clone()
    } else {
        Vec::new()
    };
    let hit = match cache.as_mut() {
        Some(pc) if req.media.is_none() => {
            let device = sched.model_mut().device().clone();
            pc.find(&req.prompt, &device)
        }
        _ => None,
    };
    match hit {
        Some(hit) => match sched.submit_primed(req.prompt.clone(), max_new, hit.prefix_len) {
            Some(sidx) => {
                sched.model_mut().inject_slot_snapshot(sidx, hit.snap);
                eprintln!(
                    "[pcache] slot={sidx} primed: {} из {} токенов из снимка, досчитать {}",
                    hit.prefix_len,
                    req.prompt.len(),
                    req.prompt.len() - hit.prefix_len,
                );
            }
            // Нет свободного слота шедулера → промах, обычный submit.
            None => {
                sched.submit(req.prompt, max_new);
            }
        },
        None => {
            sched.submit_with_first_chunk(req.prompt, max_new, req.first_chunk_cut);
        }
    }
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
        .insert(
            idx,
            SlotSampler {
                params: params.clone(),
                rng: Rng::new(seed),
                seen: HashSet::new(),
                seen_len: 0,
                pending_logprobs: Vec::new(),
            },
        );
    slot_truncated.insert(idx, req.truncated);
    let params_thinking = params.thinking;
    bindings[idx] = Some(SlotBinding {
        out: req.out,
        params,
        prompt_tokens,
        completion_tokens: 0,
        emitted_text: String::new(),
        stable_toks: 0,
        stable_len: 0,
        stop_hit: false,
        looped: false,
        cancelled: false,
        last_progress: Instant::now(),
        usage,
        _media_lease: media_lease,
        phase_checked: 0,
        in_reasoning: params_thinking,
        prompt: prompt_for_cache,
        prefix_captured: false,
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
            let _ = b.out.try_send(StreamEvent::Delta {
                text: tail,
                logprobs: None,
            });
        }
    }
    if !b.cancelled {
        let finish_reason = if b.looped {
            // Не "stop": клиент должен видеть, что ответ оборван сторожем,
            // а не завершён моделью. Иначе агент примет мусорный хвост за
            // законченную мысль.
            "length"
        } else if b.stop_hit {
            "stop"
        } else if b.completion_tokens >= b.params.max_tokens {
            "length"
        } else {
            "stop" // EOS
        };
        let truncated = slot_truncated.remove(&idx).unwrap_or(false);
        // Закончили ВНУТРИ think-блока (length-обрыв до think-close)? Весь
        // накопленный text — thinking, HTTP-слой обязан убрать его из content
        // (иначе клиент видит сырые рассуждения вместо ответа).
        let ended_in_thinking = b.in_reasoning;
        let _ = b.out.try_send(StreamEvent::Done {
            finish_reason: finish_reason.into(),
            ended_in_thinking,
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
            .downcast::<Option<SlotSampler>>()
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
            Some(st) => {
                // Множество для penalties — инкрементально: generated только
                // растёт (после отката спекуляции состояние восстанавливается
                // из checkpoint вместе с seen); укорочение — пересборка.
                if generated.len() < st.seen_len {
                    st.seen.clear();
                    st.seen_len = 0;
                }
                st.seen.extend(generated[st.seen_len..].iter().copied());
                st.seen_len = generated.len();
                let t0 = Instant::now();
                let (tok, lp) = sampler::sample_with_seen_logprobs(
                    logits,
                    st.params.temperature,
                    st.params.top_k,
                    st.params.top_p,
                    st.params.min_p,
                    st.params.presence_penalty,
                    st.params.repetition_penalty,
                    &st.seen,
                    &mut st.rng,
                    st.params.logprobs,
                );
                if let Some(lp) = lp {
                    st.pending_logprobs.push(lp);
                }
                HOST_SAMPLE_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                HOST_SAMPLE_CALLS.fetch_add(1, Ordering::Relaxed);
                tok
            }
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
    fn detect_loop_catches_repeat_and_spares_normal_text() {
        // Петля: блок из 3 токенов, повторённый до конца.
        let mut looped: Vec<u32> = (0..200).collect();
        for _ in 0..40 {
            looped.extend_from_slice(&[7, 8, 9]);
        }
        assert_eq!(detect_loop(&looped), Some(3));

        // Осмысленный текст: повторы есть, но не подряд одним блоком.
        let normal: Vec<u32> = (0..400).map(|i| (i * 37 % 900) as u32).collect();
        assert_eq!(detect_loop(&normal), None);

        // Короткий ответ не трогаем, даже если он весь из повторов.
        let short: Vec<u32> = std::iter::repeat(5).take(LOOP_MIN_TOKENS - 1).collect();
        assert_eq!(detect_loop(&short), None);

        // Ровно на границе включения петля уже ловится.
        let at_edge: Vec<u32> = std::iter::repeat(5).take(LOOP_MIN_TOKENS).collect();
        assert_eq!(detect_loop(&at_edge), Some(1));
    }

    /// Короткие серии одинаковых токенов — обычный код, а не петля.
    /// Регрессия: сторож рвал генерацию на `z-index: 9999` (четыре токена
    /// `9` подряд), обрывая CSS посреди файла с finish_reason "length".
    #[test]
    fn detect_loop_spares_short_runs_of_identical_tokens() {
        let digit = 24u32; // условный токен «9»
        for run in [4usize, 8, 16, LOOP_MIN_SPAN - 1] {
            let mut css: Vec<u32> = (0..600).map(|i| (i * 31 % 800) as u32).collect();
            css.extend(std::iter::repeat(digit).take(run));
            assert_eq!(
                detect_loop(&css),
                None,
                "серия из {run} одинаковых токенов не должна считаться петлёй"
            );
        }

        // А вот залипание на одном токене на всю длину — уже петля.
        let mut stuck: Vec<u32> = (0..600).map(|i| (i * 31 % 800) as u32).collect();
        stuck.extend(std::iter::repeat(digit).take(LOOP_MIN_SPAN));
        assert_eq!(detect_loop(&stuck), Some(1));
    }

    #[test]
    fn batchconfig_clamps_slots() {
        std::env::set_var("SLOTS", "8");
        let c = BatchConfig::from_env();
        assert_eq!(c.slots, MAX_SLOTS);
        std::env::remove_var("SLOTS");
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

    #[test]
    fn indexed_sampler_advances_rng_across_steps() {
        let params = GenParams {
            top_k: 4,
            top_p: 1.0,
            ..Default::default()
        };
        let shared = Arc::new(Mutex::new(HashMap::from([(
            0,
            SlotSampler {
                params: params.clone(),
                rng: Rng::new(123),
                seen: HashSet::new(),
                seen_len: 0,
                pending_logprobs: Vec::new(),
            },
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
