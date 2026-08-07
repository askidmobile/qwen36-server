//! Типы внутреннего контракта engine↔HTTP — копия `docs/engine-api.md` (v0).
//!
//! ponytail: дубль контракта намеренный — engine-агент вливает свою реализацию
//! в main параллельно; при слиянии заменить этот модуль на единый
//! `crate::engine_types` из main (типы должны совпадать 1:1, конфликт = сигнал
//! расхождения контрактов).

use serde::{Deserialize, Serialize};

/// Параметры генерации (сэмплинг). Дефолты — пресет `thinking` (BD-016).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenParams {
    #[serde(default = "d_temperature")]
    pub temperature: f32,
    #[serde(default = "d_top_p")]
    pub top_p: f32,
    #[serde(default = "d_top_k")]
    pub top_k: usize,
    #[serde(default)]
    pub min_p: f32,
    #[serde(default)]
    pub presence_penalty: f32,
    #[serde(default = "d_repetition_penalty")]
    pub repetition_penalty: f32,
    #[serde(default = "d_max_tokens")]
    pub max_tokens: usize,
    #[serde(default)]
    pub stop: Vec<String>,
    #[serde(default)]
    pub seed: Option<u64>,
}

const fn d_temperature() -> f32 {
    1.0
}
const fn d_top_p() -> f32 {
    0.95
}
const fn d_top_k() -> usize {
    20
}
const fn d_repetition_penalty() -> f32 {
    1.0
}
const fn d_max_tokens() -> usize {
    4096
}

impl Default for GenParams {
    fn default() -> Self {
        Self {
            temperature: d_temperature(),
            top_p: d_top_p(),
            top_k: d_top_k(),
            min_p: 0.0,
            presence_penalty: 0.0,
            repetition_penalty: d_repetition_penalty(),
            max_tokens: d_max_tokens(),
            stop: Vec::new(),
            seed: None,
        }
    }
}

/// Одно chat-сообщение. role: system|user|assistant|tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Событие стрима генерации (engine → HTTP-слой).
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// Текстовый кусок (thinking включён в поток, парсит HTTP-слой).
    Delta(String),
    /// Финал генерации.
    Done {
        finish_reason: String,
        prompt_tokens: usize,
        completion_tokens: usize,
        truncated: bool,
    },
    /// Ошибка генерации этого запроса (engine жив).
    Error(String),
}

/// Карточка модели для `GET /v1/models` (BD-015).
#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub context_length: usize,
    pub quant: String,
    pub slots: usize,
    pub modes: Vec<String>,
}

/// Контракт engine-слоя. HTTP зависит только от него.
#[async_trait::async_trait]
pub trait Engine: Send + Sync {
    /// Стриминговая генерация по chat-сообщениям.
    /// sliding window применяется внутри (BD-017).
    async fn generate(
        &self,
        messages: Vec<ChatMessage>,
        params: GenParams,
    ) -> anyhow::Result<tokio::sync::mpsc::Receiver<StreamEvent>>;
    fn model_info(&self) -> ModelInfo;
}
