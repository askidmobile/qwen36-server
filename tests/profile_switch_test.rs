use std::path::PathBuf;
use std::sync::Arc;

use qwen36_server::engine::{Engine, InferenceRequest, ModelInfo, StreamEvent};
use qwen36_server::engine_swap::SwappableEngine;
use tokio::sync::mpsc;

struct MockEngine {
    id: String,
}

#[async_trait::async_trait]
impl Engine for MockEngine {
    async fn generate(
        &self,
        _request: InferenceRequest,
    ) -> anyhow::Result<mpsc::Receiver<StreamEvent>> {
        let (tx, rx) = mpsc::channel(1);
        let _ = tx
            .send(StreamEvent::Done {
                ended_in_thinking: false,
                finish_reason: "stop".into(),
                usage: Default::default(),
            })
            .await;
        Ok(rx)
    }

    fn model_info(&self) -> ModelInfo {
        ModelInfo {
            id: self.id.clone(),
            context_length: 4096,
            quant: "Q4_K_M".into(),
            slots: 4,
            modes: vec!["instruct".into()],
        }
    }
}

#[tokio::test]
async fn swappable_engine_drain_and_switch() {
    let mock1 = Arc::new(MockEngine {
        id: "model-1".into(),
    });
    let switcher = SwappableEngine::new(mock1, PathBuf::from("model1.gguf"), 4096, 4);

    assert_eq!(switcher.model_info().id, "model-1");

    // Take out engine during switch
    let prev = switcher.take();
    assert!(prev.is_some());
    assert!(switcher.model_info().id.starts_with("loading:"));

    let mock2 = Arc::new(MockEngine {
        id: "model-2".into(),
    });
    switcher.install(mock2, PathBuf::from("model2.gguf"), 8192, 2);

    assert_eq!(switcher.model_info().id, "model-2");
    assert_eq!(switcher.model_info().context_length, 8192);
    assert_eq!(switcher.model_info().slots, 2);
}
