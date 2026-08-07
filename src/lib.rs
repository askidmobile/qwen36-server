//! qwen36-server — сервер инференса Qwen3.6-27B GGUF (contract: docs/engine-api.md).

pub mod config;
pub mod engine;
pub mod sampler;

// Result<T, Response> намеренно: Response — готовый early-return для axum
#[allow(clippy::result_large_err)]
pub mod api;
pub mod engine_types;
