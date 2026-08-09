//! Engine-слой по контракту docs/engine-api.md.
//! Типы GenParams/ChatMessage/StreamEvent/ModelInfo/Engine — точная копия контракта.
//! Реализация CandleEngine — поверх qwen35-batch (ModelWeights::from_gguf, токенизатор real).

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

use qwen35_batch::real::tokenizer::{self, ChatMsg};
use qwen35_batch::real::ModelWeights;

use crate::config::Config;
use crate::sampler::{self, Rng, SamplingPreset};

// ── Контрактные типы (docs/engine-api.md §Engine trait) ─────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenParams {
    pub temperature: f32,        // default 1.0
    pub top_p: f32,              // default 0.95
    pub top_k: usize,            // default 20
    pub min_p: f32,              // default 0.0
    pub presence_penalty: f32,   // default 0.0
    pub repetition_penalty: f32, // default 1.0
    pub max_tokens: usize,       // default 4096
    #[serde(default)]
    pub stop: Vec<String>,
    #[serde(default)]
    pub seed: Option<u64>,
    /// thinking-режим: false → prompt получает пустой <think></think> суффикс
    /// (encode_no_think). default true (BD-016, model card).
    #[serde(default = "default_thinking")]
    pub thinking: bool,
}

fn default_thinking() -> bool {
    true
}

impl Default for GenParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 0.95,
            top_k: 20,
            min_p: 0.0,
            presence_penalty: 0.0,
            repetition_penalty: 1.0,
            max_tokens: 4096,
            stop: vec![],
            seed: None,
            thinking: true,
        }
    }
}

impl GenParams {
    /// Применить пресет BD-016: поля пресета замещают defaults,
    /// явно заданные stop/seed/max_tokens сохраняются.
    pub fn from_preset(p: SamplingPreset) -> Self {
        let mut g = Self::default();
        match p {
            SamplingPreset::Thinking => {} // == defaults
            SamplingPreset::ThinkingCoding => g.temperature = 0.6,
            SamplingPreset::Instruct => {
                g.temperature = 0.7;
                g.top_p = 0.80;
                g.presence_penalty = 1.5;
            }
        }
        g
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,    // system|user|assistant|tool
    pub content: String,
}

#[derive(Debug)]
pub enum StreamEvent {
    /// Текстовый кусок (thinking включён в поток, парсит HTTP-слой).
    Delta(String),
    Done {
        finish_reason: String, // "stop" | "length"
        prompt_tokens: usize,
        completion_tokens: usize,
        truncated: bool,
    },
    Error(String),
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub id: String,            // "qwen3.6-27b"
    pub context_length: usize, // 81920
    pub quant: String,         // "Q2_K_XL"
    pub slots: usize,          // 4
    pub modes: Vec<String>,    // ["thinking", "instruct"]
}

#[async_trait::async_trait]
pub trait Engine: Send + Sync {
    /// Стриминговая генерация по chat-сообщениям. sliding window внутри (BD-017).
    async fn generate(
        &self,
        messages: Vec<ChatMessage>,
        params: GenParams,
    ) -> Result<mpsc::Receiver<StreamEvent>>;
    fn model_info(&self) -> ModelInfo;
}

// ── CandleEngine ────────────────────────────────────────────────────────────

/// Запас токенов под погрешность per-message оценки в sliding window.
const TRIM_MARGIN: usize = 16;

struct ModelState {
    model: ModelWeights,
    tokenizer: tokenizers::Tokenizer,
    /// Устройство модели (для создания входных тензоров без повторного probe).
    device: candle_core::Device,
}

pub struct CandleEngine {
    /// Модель не thread-safe — вся генерация под мьютексом.
    /// Параллелизм 4 слотов (BD-007) даст batched-планировщик форка
    /// (Qwen35BatchAdapter + BatchScheduler) — отдельная фаза (BD-019).
    state: Arc<Mutex<ModelState>>,
    eos: u32,
    ctx: usize,
    info: ModelInfo,
}

impl CandleEngine {
    pub fn load(cfg: &Config) -> Result<Self> {
        let device = select_device()?;
        let (model, eos) = load_model(&cfg.model, &device)?;
        let tokenizer = tokenizer::load_from_gguf_path(&cfg.model)?;
        Ok(Self {
            state: Arc::new(Mutex::new(ModelState {
                model,
                tokenizer,
                device,
            })),
            eos,
            ctx: cfg.ctx,
            info: ModelInfo {
                id: model_id_from_filename(&cfg.model),
                context_length: cfg.ctx,
                quant: quant_from_filename(&cfg.model),
                slots: cfg.slots,
                modes: vec!["thinking".into(), "instruct".into()],
            },
        })
    }
}

#[async_trait::async_trait]
impl Engine for CandleEngine {
    async fn generate(
        &self,
        messages: Vec<ChatMessage>,
        params: GenParams,
    ) -> Result<mpsc::Receiver<StreamEvent>> {
        // Sliding window (BD-017): system сохраняется, режутся старые пары.
        // Оценка: токены каждого сообщения отдельно (BPE-границы дают погрешность
        // ~1-2 токена на сообщение) — покрыта запасом TRIM_MARGIN.
        // ponytail: точный подсчёт = encode всего prompt'а; добавить, если упрёмся в KV.
        let budget = self
            .ctx
            .saturating_sub(params.max_tokens)
            .saturating_sub(TRIM_MARGIN);
        let prompt_ids;
        let truncated;
        {
            let st = self.state.lock().expect("engine mutex");
            let count = |m: &ChatMessage| -> usize {
                let chunk = format!("<|im_start|>{}\n{}<|im_end|>\n", m.role, m.content);
                st.tokenizer
                    .encode(chunk, false)
                    .map(|e| e.get_ids().len())
                    .unwrap_or(0)
            };
            let (kept, was_trimmed) = trim_messages(&messages, budget, count);
            truncated = was_trimmed;
            let msgs: Vec<ChatMsg> = kept
                .iter()
                .map(|m| ChatMsg {
                    role: &m.role,
                    content: &m.content,
                })
                .collect();
            let text = tokenizer::build_chatml_text(&msgs);
            prompt_ids = if params.thinking {
                st.tokenizer
                    .encode(text, false)
                    .map(|e| e.get_ids().to_vec())
                    .map_err(|e| anyhow::anyhow!("encode prompt: {e}"))?
            } else {
                tokenizer::encode_no_think(&st.tokenizer, &text)?
            };
        }
        let prompt_tokens = prompt_ids.len();

        let (tx, rx) = mpsc::channel(64);
        let state = Arc::clone(&self.state);
        let eos = self.eos;
        // Инференс блокирующий и долгий — в blocking-пул, канал стримит наружу.
        tokio::task::spawn_blocking(move || {
            run_generation(state, prompt_ids, prompt_tokens, params, eos, truncated, tx);
        });
        Ok(rx)
    }

    fn model_info(&self) -> ModelInfo {
        ModelInfo {
            id: self.info.id.clone(),
            context_length: self.info.context_length,
            quant: self.info.quant.clone(),
            slots: self.info.slots,
            modes: self.info.modes.clone(),
        }
    }
}

/// Блокирующий цикл генерации: prefill → decode по 1 токену до EOS/max_tokens/stop.
fn run_generation(
    state: Arc<Mutex<ModelState>>,
    prompt_ids: Vec<u32>,
    prompt_tokens: usize,
    params: GenParams,
    eos: u32,
    truncated: bool,
    tx: mpsc::Sender<StreamEvent>,
) {
    let finish = move |reason: &str, completion: usize, tx: &mpsc::Sender<StreamEvent>| {
        let _ = tx.blocking_send(StreamEvent::Done {
            finish_reason: reason.into(),
            prompt_tokens,
            completion_tokens: completion,
            truncated,
        });
    };

    let result = (|| -> Result<()> {
        let mut st = state.lock().expect("engine mutex");
        let ModelState {
            model,
            tokenizer,
            device: dev,
        } = &mut *st;
        model.clear_state();

        let ids = candle_core::Tensor::from_vec(
            prompt_ids.clone(),
            (1usize, prompt_ids.len()),
            &dev,
        )?;
        let logits_t = model.forward(&ids, 0)?;
        let mut logits = last_logits(&logits_t)?;
        let mut pos = prompt_ids.len();

        let seed = params.seed.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos() as u64 ^ d.as_secs())
                .unwrap_or(42)
        });
        let mut rng = Rng::new(seed);
        let mut generated: Vec<u32> = Vec::new();
        let mut full_text = String::new();
        // Эмитнутый префикс (строка, не индекс): decode_text может ретроактивно
        // менять ранние байты (многотокенные UTF-8) — индекс небезопасен.
        let mut emitted_text = String::new();
        let max_stop_len = params.stop.iter().map(|s| s.len()).max().unwrap_or(0);

        // Flush holdback-хвоста перед Done.
        macro_rules! flush_tail {
            () => {
                if full_text.starts_with(&emitted_text) && full_text.len() > emitted_text.len() {
                    let tail = full_text[emitted_text.len()..].to_string();
                    let _ = tx.blocking_send(StreamEvent::Delta(tail));
                }
            };
        }

        for _ in 0..params.max_tokens {
            let tok = sampler::sample(
                &logits,
                params.temperature,
                params.top_k,
                params.top_p,
                params.min_p,
                params.presence_penalty,
                params.repetition_penalty,
                &generated,
                &mut rng,
            );
            if tok == eos {
                flush_tail!();
                finish("stop", generated.len(), &tx);
                return Ok(());
            }
            generated.push(tok);

            // Decode всего вывода (корректно на границах многотокенных UTF-8).
            // ponytail: инкрементальный decode с буфером хвоста — добавить, если профилирование покажет.
            full_text = tokenizer::decode_text(tokenizer, &generated).unwrap_or_default();

            // Stop-строки: ищем в хвосте длиной max_stop_len + последний кусок.
            let mut cut_at: Option<usize> = None;
            if !params.stop.is_empty() {
                let scan_from = full_text.len().saturating_sub(max_stop_len + 64);
                let scan_from = floor_char_boundary(&full_text, scan_from)
                .max(emitted_text.len().min(full_text.len()));
                if let Some(rel) = full_text[scan_from..].find_any(&params.stop) {
                    cut_at = Some(scan_from + rel);
                }
            }

            let mut end = cut_at.unwrap_or(full_text.len());
            // Holdback U+FFFD-хвоста: недостроенная UTF-8 последовательность
            // (emoji/CJK на границе токенов) достроится следующим токеном.
            // Flush — при завершении (finish).
            while end > emitted_text.len() && full_text[..end].ends_with('\u{FFFD}') {
                end -= '\u{FFFD}'.len_utf8();
            }
            if full_text.starts_with(&emitted_text) && end >= emitted_text.len() {
                let chunk = &full_text[emitted_text.len()..end];
                if !chunk.is_empty()
                    && tx.blocking_send(StreamEvent::Delta(chunk.to_string())).is_err()
                {
                    return Ok(()); // клиент отключился
                }
                emitted_text = full_text[..end].to_string();
            }
            if cut_at.is_some() {
                flush_tail!();
                finish("stop", generated.len(), &tx);
                return Ok(());
            }

            // Следующий decode-шаг: forward по 1 токену (single-slot state уже в модели).
            let next = candle_core::Tensor::from_vec(vec![tok], (1usize, 1usize), &dev)?;
            let logits_t = model.forward(&next, pos)?;
            logits = last_logits(&logits_t)?;
            pos += 1;
        }
        flush_tail!();
        finish("length", generated.len(), &tx);
        Ok(())
    })();

    if let Err(e) = result {
        let _ = tx.blocking_send(StreamEvent::Error(format!("{e:#}")));
    }
}

/// Логиты последнего токена: forward возвращает [1, vocab] (seq=1) либо [1, seq, vocab].
fn last_logits(t: &candle_core::Tensor) -> Result<Vec<f32>> {
    use candle_core::DType;
    let t = t.squeeze(0)?;
    let t = match t.rank() {
        1 => t,
        2 => {
            let seq = t.dim(0)?;
            t.get(seq - 1)?
        }
        r => anyhow::bail!("unexpected logits rank {r}"),
    };
    Ok(t.to_dtype(DType::F32)?.to_vec1()?)
}

pub fn floor_char_boundary(s: &str, mut i: usize) -> usize {
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// Загрузка GGUF: веса через ModelWeights (zero-copy на macOS+Metal),
/// EOS из metadata `tokenizer.ggml.eos_token_id` (default 151645, как в адаптере).
fn load_model(path: &Path, device: &candle_core::Device) -> Result<(ModelWeights, u32)> {
    use anyhow::anyhow;
    use candle_core::quantized::gguf_file;
    use std::sync::Arc;

    let file = std::fs::File::open(path).map_err(|e| anyhow!("open GGUF {path:?}: {e}"))?;
    let mmap = unsafe { memmap2::MmapOptions::new().map(&file) }
        .map_err(|e| anyhow!("mmap GGUF: {e}"))?;
    let mmap = Arc::new(mmap);
    let mut c = std::io::Cursor::new(mmap.as_ref());
    let ct = gguf_file::Content::read(&mut c).map_err(|e| anyhow!("read GGUF: {e}"))?;
    let eos = ct
        .metadata
        .get("tokenizer.ggml.eos_token_id")
        .and_then(|v| v.to_u32().ok())
        .unwrap_or(151645);

    #[cfg(all(target_os = "macos", feature = "metal"))]
    let model = if matches!(device, candle_core::Device::Metal(_)) {
        ModelWeights::from_gguf_zero_copy(ct, mmap, device)
            .map_err(|e| anyhow!("load weights zero-copy: {e}"))?
    } else {
        ModelWeights::from_gguf(ct, mmap, device).map_err(|e| anyhow!("load weights: {e}"))?
    };
    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    let model = ModelWeights::from_gguf(ct, mmap, device).map_err(|e| anyhow!("load weights: {e}"))?;
    Ok((model, eos))
}

/// Выбор устройства: cuda (Windows) > metal (macOS) > CPU (BD-011).
pub fn select_device() -> Result<candle_core::Device> {
    #[cfg(feature = "cuda")]
    {
        return Ok(candle_core::Device::new_cuda(0)?);
    }
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        qwen35_batch::real::metal_utils::configure_metal_env();
        let d = candle_core::Device::new_metal(0)?;
        qwen35_batch::real::metal_utils::metal_probe(&d)?;
        return Ok(d);
    }
    #[allow(unreachable_code)]
    Ok(candle_core::Device::Cpu)
}

/// Id модели из имени файла: `Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf` → "qwen3.6-35b-a3b".
/// Срезаем суффикс кванта (`-UD-*` / `-IQ*`/`-Q*` последний сегмент — его
/// отдельно показывает quant).
pub fn model_id_from_filename(path: &Path) -> String {
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");
    // unsloth: <name>-UD-<QUANT>; lmstudio: <name>-<QUANT>.
    let base = stem
        .rsplit_once("-UD-")
        .map(|(b, _)| b)
        .unwrap_or_else(|| {
            stem.rsplit_once('-')
                .map(|(b, _)| b)
                .unwrap_or(stem)
        });
    base.to_lowercase()
}

/// Квант из имени файла: `qwen36-27b-q2_k_xl.gguf` → "Q2_K_XL".
pub fn quant_from_filename(path: &Path) -> String {
    path.file_stem()
        .and_then(|s| s.to_str())
        .and_then(|s| s.rsplit('-').next())
        .map(|q| q.to_uppercase())
        .filter(|q| !q.is_empty())
        .unwrap_or_else(|| "unknown".into())
}

/// Sliding window (BD-017): сохраняем ведущие system-сообщения, режем старые
/// user/assistant пары с головы, пока суммарная оценка токенов не влезет в budget.
/// Последнее сообщение (текущий user) не удаляем никогда.
/// Возвращает (оставшиеся сообщения, был ли trim).
pub fn trim_messages(
    messages: &[ChatMessage],
    budget: usize,
    count: impl Fn(&ChatMessage) -> usize,
) -> (Vec<ChatMessage>, bool) {
    let total: usize = messages.iter().map(&count).sum();
    if total <= budget {
        return (messages.to_vec(), false);
    }
    let n_system = messages
        .iter()
        .take_while(|m| m.role == "system")
        .count();
    let (systems, body) = messages.split_at(n_system);
    let sys_tokens: usize = systems.iter().map(&count).sum();
    let mut body: Vec<ChatMessage> = body.to_vec();
    let mut body_tokens: usize = body.iter().map(&count).sum();

    // Режем парами (user+assistant) с головы; последнее сообщение не трогаем.
    while sys_tokens + body_tokens > budget && body.len() > 1 {
        let take = if body.len() > 2 { 2 } else { 1 };
        for m in body.drain(..take) {
            body_tokens = body_tokens.saturating_sub(count(&m));
        }
    }
    let mut out: Vec<ChatMessage> = systems.to_vec();
    out.extend(body);
    (out, true)
}

/// Найти первую из stop-строк (минимальная позиция).
pub trait FindAny {
    fn find_any(&self, needles: &[String]) -> Option<usize>;
}
impl FindAny for str {
    fn find_any(&self, needles: &[String]) -> Option<usize> {
        needles
            .iter()
            .filter_map(|n| self.find(n.as_str()))
            .min()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: role.into(),
            content: content.into(),
        }
    }

    #[test]
    fn default_params_match_contract() {
        let g = GenParams::default();
        assert_eq!(g.temperature, 1.0);
        assert_eq!(g.top_p, 0.95);
        assert_eq!(g.top_k, 20);
        assert_eq!(g.min_p, 0.0);
        assert_eq!(g.presence_penalty, 0.0);
        assert_eq!(g.repetition_penalty, 1.0);
        assert_eq!(g.max_tokens, 4096);
    }

    #[test]
    fn presets_match_bd016() {
        let t = GenParams::from_preset(SamplingPreset::Thinking);
        assert_eq!((t.temperature, t.top_p, t.top_k), (1.0, 0.95, 20));
        let c = GenParams::from_preset(SamplingPreset::ThinkingCoding);
        assert_eq!((c.temperature, c.top_p, c.top_k), (0.6, 0.95, 20));
        let i = GenParams::from_preset(SamplingPreset::Instruct);
        assert_eq!(i.temperature, 0.7);
        assert_eq!(i.top_p, 0.80);
        assert_eq!(i.presence_penalty, 1.5);
    }

    #[test]
    fn trim_noop_when_fits() {
        let m = vec![msg("system", "s"), msg("user", "hello")];
        let (out, truncated) = trim_messages(&m, 1000, |m| m.content.len());
        assert!(!truncated);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn trim_drops_oldest_pairs_keeps_system() {
        // budget 20: system(6) + 3 пары по 10 символов.
        let m = vec![
            msg("system", "sys123"),
            msg("user", "u1--------"),
            msg("assistant", "a1--------"),
            msg("user", "u2--------"),
            msg("assistant", "a2--------"),
            msg("user", "u3--------"),
        ];
        let (out, truncated) = trim_messages(&m, 20, |m| m.content.len());
        assert!(truncated);
        assert_eq!(out[0].role, "system");
        assert_eq!(out.last().unwrap().content, "u3--------");
        // 6 + 10 = 16 <= 20; пара (u2,a2) тоже должна быть срезана (16+20>20? нет: 6+10+20=36>20 при проверке до реза u2/a2)
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn trim_keeps_last_message_even_over_budget() {
        let m = vec![msg("system", "s"), msg("user", "x".repeat(100).as_str())];
        let (out, truncated) = trim_messages(&m, 10, |m| m.content.len());
        assert!(truncated);
        assert_eq!(out.len(), 2); // system + последний user
    }

    #[test]
    fn model_id_from_filenames() {
        assert_eq!(
            model_id_from_filename(Path::new("D:/models/Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf")),
            "qwen3.6-35b-a3b"
        );
        assert_eq!(
            model_id_from_filename(Path::new("models/qwen36-27b-q2_k_xl.gguf")),
            "qwen36-27b"
        );
    }

    #[test]
    fn quant_parsed_from_filename() {
        assert_eq!(
            quant_from_filename(Path::new("models/qwen36-27b-q2_k_xl.gguf")),
            "Q2_K_XL"
        );
        assert_eq!(quant_from_filename(Path::new("m.gguf")), "M");
    }
}
