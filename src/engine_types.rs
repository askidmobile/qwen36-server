//! Реэкспорт контрактных типов: единственный источник — crate::engine.
//! (Раньше — ponytail-копия для независимой сборки api-агента.)

pub use crate::engine::{
    CancelFlag, ChatMessage, ContentBlock, Engine, GenParams, GenerationUsage, InferenceRequest,
    MediaSource, MediaUsage, ModelInfo, MtpUsage, StreamEvent,
};
