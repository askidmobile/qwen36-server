//! qwen36-server — сервер инференса Qwen3.6-27B GGUF (contract: docs/engine-api.md).

pub mod config;
pub mod engine;
pub mod sampler;
// pub mod api; — добавит агент HTTP-слоя.
