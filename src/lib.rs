//! qwen36-server — сервер инференса Qwen3.6-27B GGUF (contract: docs/engine-api.md).
//! Батчинг 4 слотов: docs/batch-integration.md.

pub mod config;
pub mod engine;
pub mod sampler;

// Result<T, Response> намеренно: Response — готовый early-return для axum
#[allow(clippy::result_large_err)]
pub mod api;
pub mod engine_batched;
pub mod engine_types;
