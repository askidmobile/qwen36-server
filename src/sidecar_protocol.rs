//! Протокол сайдкара Yttri v11 — подмножество, которое обслуживает `yforge --sidecar`.
//!
//! Эталон форм — `Yttri/frontend/src-tauri/src/modules/ai/local_llm/mlx_protocol.rs`
//! (там же история версий). Кадр: строка JSON, за ней `blob_bytes` сырых байт, если
//! поле есть. Запросы помечены `op`, ответы — `event`. Имена и умолчания полей —
//! как у клиента: он не шлёт нули и пустые списки, поэтому у всех полей здесь
//! `#[serde(default)]`, а незнакомые поля (настройки MLX) молча игнорируются.

use serde::{Deserialize, Serialize};

/// Версия протокола. Клиент сверяет её в `health` и не работает с другой.
pub const PROTOCOL_VERSION: u32 = 11;

fn one() -> f32 {
    1.0
}

#[derive(Debug, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Request {
    /// Загрузка модели. `max_seq` — окно контекста, `batch_slots` — слоты.
    /// Остальные поля MLX (`cache_*`, `kv_bits`, `mode`) здесь не нужны:
    /// KV всегда int8, префикс-кеш и память планирует движок.
    Load {
        #[serde(default)]
        model_path: String,
        #[serde(default)]
        batch_slots: u32,
        #[serde(default)]
        max_seq: u32,
    },
    Health,
    Generate(GenerateReq),
    Cancel {
        req_id: u64,
    },
    Trim,
    Unload,
}

#[derive(Debug, Deserialize)]
pub struct GenerateReq {
    #[serde(default)]
    pub req_id: u64,
    pub prompt: String,
    /// Стабильное начало промпта текстом (клиент без токенайзера модели).
    #[serde(default)]
    pub prefix: String,
    #[serde(default)]
    pub prompt_tokens: Vec<u32>,
    #[serde(default)]
    pub prefix_token_count: usize,
    pub max_tokens: usize,
    #[serde(default)]
    pub temperature: f32,
    #[serde(default = "one")]
    pub top_p: f32,
    #[serde(default)]
    pub top_k: usize,
    #[serde(default)]
    pub min_p: f32,
    #[serde(default)]
    pub presence_penalty: f32,
    #[serde(default = "one")]
    pub repetition_penalty: f32,
    #[serde(default)]
    pub suppress_tokens: Vec<u32>,
    #[serde(default)]
    pub stop_sequences: Vec<String>,
    #[serde(default)]
    pub max_thinking_tokens: usize,
    /// Продуктовый потолок промпта клиента; 0 — только окно контекста.
    #[serde(default)]
    pub max_prompt_tokens: usize,
}

#[derive(Debug, Serialize, PartialEq)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum Event {
    Loaded {
        ok: bool,
        batch_slots: u32,
        supports_progress: bool,
        /// KV-пул int8 (T-670: 8 — int8, 0 — плотный).
        kv_bits: u32,
        max_seq: u32,
    },
    Health {
        ok: bool,
        protocol_version: u32,
        model_loaded: bool,
        batch_slots: u32,
        max_seq: u32,
        batch_queue: u32,
        batch_active: u32,
        /// Видеопамять, занятая процессом с момента загрузки (МБ, оценка по
        /// `cuMemGetInfo`: чужие процессы на той же карте её сдвигают).
        active_memory_mb: u32,
        supports_progress: bool,
        vision: bool,
    },
    Chunk {
        req_id: u64,
        text: String,
    },
    Progress {
        req_id: u64,
        phase: &'static str,
        processed: usize,
        total: usize,
        tokens_per_sec: f32,
    },
    Done {
        req_id: u64,
        text: String,
        tokens: usize,
        time_ms: u64,
        prefill_ms: u64,
        prompt_tokens: usize,
        cache_hit: bool,
        cached_tokens: usize,
        prefill_tokens: usize,
        stop_reason: &'static str,
        active_memory_mb: u32,
    },
    Error {
        message: String,
        #[serde(skip_serializing_if = "is_zero")]
        req_id: u64,
    },
    Trimmed {
        freed_mb: u32,
        active_memory_mb: u32,
    },
}

fn is_zero(v: &u64) -> bool {
    *v == 0
}
