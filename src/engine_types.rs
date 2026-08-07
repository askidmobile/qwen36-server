//! Реэкспорт контрактных типов: единственный источник — crate::engine.
//! (Раньше — ponytail-копия для независимой сборки api-агента.)

pub use crate::engine::{ChatMessage, Engine, GenParams, ModelInfo, StreamEvent};
