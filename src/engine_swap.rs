//! SwappableEngine — обёртка над движком с горячей заменой модели.
//! HTTP-слой видит обычный `dyn Engine`; admin-эндпоинт меняет внутренность.
//! Во время загрузки generate() → Err("model is loading") → 503.

use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, RwLock};

use anyhow::Result;
use tokio::sync::mpsc;

use crate::engine::{Engine, InferenceRequest, ModelInfo, StreamEvent};

pub struct SwappableEngine {
    inner: RwLock<Option<Arc<dyn Engine>>>,
    /// Идёт загрузка новой модели.
    pub loading: AtomicBool,
    /// Последняя ошибка загрузки (для диагностики через API).
    pub last_error: RwLock<String>,
    /// Текущие параметры (model_path, ctx, slots) — для /v1/models и UI.
    pub current: RwLock<(PathBuf, usize, usize)>,
}

impl SwappableEngine {
    pub fn new(engine: Arc<dyn Engine>, model_path: PathBuf, ctx: usize, slots: usize) -> Self {
        Self {
            inner: RwLock::new(Some(engine)),
            loading: AtomicBool::new(false),
            last_error: RwLock::new(String::new()),
            current: RwLock::new((model_path, ctx, slots)),
        }
    }

    /// Вынуть текущий движок (VRAM освобождается после drop последних Arc).
    pub fn take(&self) -> Option<Arc<dyn Engine>> {
        self.inner.write().expect("engine lock").take()
    }

    /// Загружен ли движок (false = выгружен / loading).
    pub fn is_loaded(&self) -> bool {
        self.inner.read().expect("engine lock").is_some()
    }

    pub fn install(&self, engine: Arc<dyn Engine>, model_path: PathBuf, ctx: usize, slots: usize) {
        *self.inner.write().expect("engine lock") = Some(engine);
        *self.current.write().expect("current lock") = (model_path, ctx, slots);
    }
}

#[async_trait::async_trait]
impl Engine for SwappableEngine {
    fn shutdown(&self) {
        // Пробрасываем внутрь (для unload_model: shutdown до take).
        if let Some(e) = self.inner.read().expect("engine lock").as_ref() {
            e.shutdown();
        }
    }
    fn ready(&self) -> bool {
        match self.inner.read().expect("engine lock").as_ref() {
            Some(e) => e.ready(),
            None => false,
        }
    }
    fn load_error(&self) -> Option<String> {
        self.inner
            .read()
            .expect("engine lock")
            .as_ref()
            .and_then(|e| e.load_error())
    }
    async fn generate(&self, request: InferenceRequest) -> Result<mpsc::Receiver<StreamEvent>> {
        let engine = { self.inner.read().expect("engine lock").clone() };
        match engine {
            Some(e) => e.generate(request).await,
            None => Err(anyhow::anyhow!("model is loading")),
        }
    }

    fn supports_vision(&self) -> bool {
        self.inner
            .read()
            .expect("engine lock")
            .as_ref()
            .map(|e| e.supports_vision())
            .unwrap_or(false)
    }
    fn supports_video(&self) -> bool {
        self.inner
            .read()
            .expect("engine lock")
            .as_ref()
            .map(|e| e.supports_video())
            .unwrap_or(false)
    }
    fn supports_mtp(&self) -> bool {
        self.inner
            .read()
            .expect("engine lock")
            .as_ref()
            .map(|e| e.supports_mtp())
            .unwrap_or(false)
    }
    fn model_info(&self) -> ModelInfo {
        let (path, ctx, slots) = self.current.read().expect("current lock").clone();
        let engine = { self.inner.read().expect("engine lock").clone() };
        match engine {
            Some(e) => {
                let mut info = e.model_info();
                info.context_length = ctx;
                info.slots = slots;
                info
            }
            None => ModelInfo {
                id: format!("loading: {}", crate::engine::model_id_from_filename(&path)),
                context_length: ctx,
                quant: crate::engine::quant_from_filename(&path),
                slots,
                modes: vec![],
            },
        }
    }
}
