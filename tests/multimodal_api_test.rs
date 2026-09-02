use std::sync::{Arc, Mutex};

use axum::body::{to_bytes, Body};
use axum::http::{Request, StatusCode};
use qwen36_server::api::{build_router, AppState};
use qwen36_server::config::ApiKey;
use qwen36_server::engine_types::{
    CancelFlag, ContentBlock, Engine, GenerationUsage, InferenceRequest, MediaSource, ModelInfo,
    StreamEvent,
};
use tower::ServiceExt;

#[derive(Default)]
struct CaptureEngine {
    requests: Mutex<Vec<InferenceRequest>>,
}

#[async_trait::async_trait]
impl Engine for CaptureEngine {
    async fn generate(
        &self,
        request: InferenceRequest,
    ) -> anyhow::Result<tokio::sync::mpsc::Receiver<StreamEvent>> {
        self.requests.lock().unwrap().push(request);
        let (tx, rx) = tokio::sync::mpsc::channel(4);
        tokio::spawn(async move {
            let _ = tx
                .send(StreamEvent::Delta {
                    text: "ответ".into(),
                    logprobs: None,
                })
                .await;
            let _ = tx
                .send(StreamEvent::Done {
                    finish_reason: "stop".into(),
                    ended_in_thinking: false,
                    usage: GenerationUsage {
                        prompt_tokens: 12,
                        completion_tokens: 2,
                        ..Default::default()
                    },
                })
                .await;
        });
        Ok(rx)
    }

    fn model_info(&self) -> ModelInfo {
        ModelInfo {
            id: "mock".into(),
            context_length: 32768,
            quant: "Q4_K_M".into(),
            slots: 4,
            modes: vec!["thinking".into(), "instruct".into()],
        }
    }
}

fn app(engine: Arc<CaptureEngine>) -> axum::Router {
    let switcher = Arc::new(qwen36_server::engine_swap::SwappableEngine::new(
        engine.clone(),
        "mock.gguf".into(),
        32768,
        4,
    ));
    let media = Arc::new(
        qwen36_server::media::MediaService::new(qwen36_server::media::MediaConfig {
            temp_root: std::env::temp_dir().join(format!("qwen36-mm-api-{}", uuid::Uuid::new_v4())),
            ..Default::default()
        })
        .unwrap(),
    );
    build_router(AppState {
        engine: switcher.clone(),
        switcher,
        media,
        api_keys: vec![ApiKey {
            key: "test-key".into(),
            name: "test".into(),
        }]
        .into(),
        models_dir: ".".into(),
        profile: None,
        cuda_device: Arc::new(std::sync::RwLock::new(None)),
        hf_downloads: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
        sampling: Arc::new(std::sync::RwLock::new(
            qwen36_server::config::SamplingDefaults::default(),
        )),
        sampling_policy: Arc::new(std::sync::RwLock::new(
            qwen36_server::config::SamplingPolicy::default(),
        )),
        presets: Arc::new(std::sync::RwLock::new(
            qwen36_server::config::SamplingPresets::new(),
        )),
        env_file: std::path::PathBuf::from(".env"),
    })
}

fn request(uri: &str, body: serde_json::Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(uri)
        .header("authorization", "Bearer test-key")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

async fn body(resp: axum::response::Response) -> serde_json::Value {
    serde_json::from_slice(&to_bytes(resp.into_body(), 1 << 20).await.unwrap()).unwrap()
}

#[tokio::test]
async fn openai_chat_preserves_russian_mixed_order() {
    let engine = Arc::new(CaptureEngine::default());
    let resp = app(engine.clone())
        .oneshot(request(
            "/v1/chat/completions",
            serde_json::json!({
                "messages": [{"role":"user","content":[
                    {"type":"text","text":"до"},
                    {"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgo="}},
                    {"type":"text","text":"после"}
                ]}]
            }),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(body(resp).await["usage"]["media"]["image_count"], 0);
    let requests = engine.requests.lock().unwrap();
    assert!(
        matches!(&requests[0].messages[0].content[0], ContentBlock::Text { text } if text == "до")
    );
    assert!(matches!(
        &requests[0].messages[0].content[1],
        ContentBlock::Media {
            source: MediaSource::DataUrl { .. },
            ..
        }
    ));
    assert!(
        matches!(&requests[0].messages[0].content[2], ContentBlock::Text { text } if text == "после")
    );
}

#[tokio::test]
async fn responses_and_anthropic_map_media_sources() {
    for (uri, payload) in [
        (
            "/v1/responses",
            serde_json::json!({"input":[{"role":"user","content":[
                {"type":"input_text","text":"смотри"},
                {"type":"input_video","media_id":"0123456789abcdef0123456789abcdef"}
            ]}]}),
        ),
        (
            "/v1/messages",
            serde_json::json!({"max_tokens":8,"messages":[{"role":"user","content":[
                {"type":"text","text":"смотри"},
                {"type":"image","source":{"type":"url","url":"https://example.com/a.png"}}
            ]}]}),
        ),
    ] {
        let engine = Arc::new(CaptureEngine::default());
        let resp = app(engine.clone())
            .oneshot(request(uri, payload))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK, "{uri}");
        let captured = engine.requests.lock().unwrap();
        assert_eq!(captured[0].messages[0].content.len(), 2);
        assert!(matches!(
            captured[0].messages[0].content[1],
            ContentBlock::Media { .. }
        ));
    }
}

#[tokio::test]
async fn malformed_sources_and_system_media_fail_before_engine() {
    for body in [
        serde_json::json!({"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"http://private/image.png"}}]}]}),
        serde_json::json!({"messages":[{"role":"system","content":[{"type":"image_url","image_url":{"url":"https://example.com/image.png"}}]}]}),
        serde_json::json!({"messages":[{"role":"user","content":[{"type":"image_url","media_id":"bad/id"}]}]}),
    ] {
        let engine = Arc::new(CaptureEngine::default());
        let resp = app(engine.clone())
            .oneshot(request("/v1/chat/completions", body))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        assert!(engine.requests.lock().unwrap().is_empty());
    }
}

#[tokio::test]
async fn decoded_image_preparation_reports_usage_and_cleans_claim() {
    let root = std::env::temp_dir().join(format!("qwen36-mm-prepare-{}", uuid::Uuid::new_v4()));
    let service = qwen36_server::media::MediaService::new(qwen36_server::media::MediaConfig {
        temp_root: root.clone(),
        ..Default::default()
    })
    .unwrap();
    let owner = qwen36_server::media::OwnerDigest::from_key("test-key");
    let encoded = root.join("fixture.png");
    image::RgbImage::from_pixel(32, 32, image::Rgb([4, 8, 12]))
        .save(&encoded)
        .unwrap();
    let bytes = std::fs::read(&encoded).unwrap();
    let uploaded = service.store.upload(owner, "image/png", &bytes).unwrap();
    let helper = std::env::var_os("CARGO_BIN_EXE_qwen36-media-helper").unwrap();
    std::env::set_var("QWEN36_MEDIA_HELPER", helper);
    let messages = vec![qwen36_server::engine_types::ChatMessage {
        role: "user".into(),
        tool_calls: Vec::new(),
        reasoning_content: None,
        content: vec![
            ContentBlock::Text {
                text: "до".into()
            },
            ContentBlock::Media {
                kind: qwen36_server::media::MediaKind::Image,
                source: MediaSource::UploadId {
                    id: uploaded.id.clone(),
                },
            },
            ContentBlock::Text {
                text: "после".into(),
            },
        ],
    }];
    let prepared = qwen36_server::media::prepare::prepare(
        &messages,
        owner,
        &CancelFlag::default(),
        &service,
        32768,
    )
    .await
    .unwrap();
    std::env::remove_var("QWEN36_MEDIA_HELPER");
    let (messages, usage, lease) = prepared.into_parts();
    assert_eq!(usage.image_count, 1);
    assert_eq!(usage.visual_tokens, 64);
    assert_eq!(messages[0].content.len(), 3);
    assert!(matches!(
        messages[0].content[1],
        qwen36_server::media::prepare::PreparedContentBlock::Media(_)
    ));
    drop(lease);
    assert!(!uploaded.path.exists());
    let _ = std::fs::remove_dir_all(root);
}

#[tokio::test]
async fn upload_claim_is_deleted_when_preparation_is_cancelled() {
    let root = std::env::temp_dir().join(format!("qwen36-mm-cancel-{}", uuid::Uuid::new_v4()));
    let service = qwen36_server::media::MediaService::new(qwen36_server::media::MediaConfig {
        temp_root: root.clone(),
        ..Default::default()
    })
    .unwrap();
    let owner = qwen36_server::media::OwnerDigest::from_key("test-key");
    let png = b"\x89PNG\r\n\x1a\nfixture";
    let uploaded = service.store.upload(owner, "image/png", png).unwrap();
    assert!(uploaded.path.exists());
    let cancel = CancelFlag::default();
    cancel.cancel();
    let messages = vec![qwen36_server::engine_types::ChatMessage {
        role: "user".into(),
        tool_calls: Vec::new(),
        reasoning_content: None,
        content: vec![ContentBlock::Media {
            kind: qwen36_server::media::MediaKind::Image,
            source: MediaSource::UploadId {
                id: uploaded.id.clone(),
            },
        }],
    }];
    let error = qwen36_server::media::prepare::prepare(&messages, owner, &cancel, &service, 32768)
        .await
        .unwrap_err();
    assert!(error.safe_message().contains("cancelled"));
    assert!(!uploaded.path.exists());
    assert!(service.store.claim_batch(owner, &[uploaded.id]).is_err());
    let _ = std::fs::remove_dir_all(root);
}
