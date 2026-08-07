// ponytail: копия контракта до мержа с agent/engine — заменить на use crate::engine::*
#![allow(dead_code)]

use tokio::sync::mpsc;

#[derive(Debug, Clone)]
pub struct GenParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub min_p: f32,
    pub presence_penalty: f32,
    pub repetition_penalty: f32,
    pub max_tokens: usize,
    pub stop: Vec<String>,
    pub seed: Option<u64>,
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
            stop: Vec::new(),
            seed: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String, // system|user|assistant|tool
    pub content: String,
}

#[derive(Debug)]
pub enum StreamEvent {
    Delta(String),
    Done {
        finish_reason: String,
        prompt_tokens: usize,
        completion_tokens: usize,
        truncated: bool,
    },
    Error(String),
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub id: String,
    pub context_length: usize,
    pub quant: String,
    pub slots: usize,
    pub modes: Vec<String>,
}

#[async_trait::async_trait]
pub trait Engine: Send + Sync {
    async fn generate(
        &self,
        messages: Vec<ChatMessage>,
        params: GenParams,
    ) -> anyhow::Result<mpsc::Receiver<StreamEvent>>;
    fn model_info(&self) -> ModelInfo;
}
