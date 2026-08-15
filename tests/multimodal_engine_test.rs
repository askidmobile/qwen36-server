#![cfg(feature = "cuda")]

use qwen36_server::engine_batched::{BatchConfig, BatchedEngine};
use qwen36_server::engine_types::{
    ChatMessage, ContentBlock, Engine, GenParams, InferenceRequest, MediaSource, StreamEvent,
};
use qwen36_server::media::{MediaConfig, MediaKind, MediaService, OwnerDigest};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

#[tokio::test]
async fn real_cuda_image_runs_through_media_scheduler_and_usage() -> anyhow::Result<()> {
    let Some(text) = std::env::var_os("QWEN35_TEXT_GGUF") else {
        return Ok(());
    };
    let Some(vision) = std::env::var_os("QWEN35_VISION_GGUF") else {
        return Ok(());
    };
    let Some(image) = std::env::var_os("QWEN35_MULTIMODAL_IMAGE") else {
        return Ok(());
    };
    let root = std::env::temp_dir().join(format!("qwen36-real-mm-{}", uuid::Uuid::new_v4()));
    let media = Arc::new(MediaService::new(MediaConfig {
        temp_root: root.clone(),
        ..Default::default()
    })?);
    let owner = OwnerDigest::from_key("phase6-gate");
    let bytes = std::fs::read(image)?;
    let uploaded = media.store.upload(owner, "image/png", &bytes)?;
    let uploaded_path = uploaded.path.clone();
    let engine = BatchedEngine::load(
        BatchConfig {
            model_path: PathBuf::from(text).to_string_lossy().into_owned(),
            slots: 1,
            max_queue: 4,
            req_timeout: Duration::from_secs(600),
            context_length: 32768,
            kv_budget_mib: 0.0,
            kv_per_tok_mib: 0.0,
            prefix_cache_mib: 0,
        },
        media,
        Some(PathBuf::from(vision)),
    )
    .await?;
    let mut rx = engine
        .generate(InferenceRequest {
            messages: vec![ChatMessage {
                role: "user".into(),
                content: vec![
                    ContentBlock::Text {
                        text: "Read the image.".into(),
                    },
                    ContentBlock::Media {
                        kind: MediaKind::Image,
                        source: MediaSource::UploadId { id: uploaded.id },
                    },
                ],
            }],
            params: GenParams {
                temperature: 0.0,
                max_tokens: 4,
                thinking: false,
                ..Default::default()
            },
            owner,
            cancel: Default::default(),
        })
        .await?;
    let mut done = None;
    while let Some(event) = rx.recv().await {
        match event {
            StreamEvent::Done { usage, .. } => {
                done = Some(usage);
                break;
            }
            StreamEvent::Error(error) => anyhow::bail!(error),
            StreamEvent::Delta(_) => {}
        }
    }
    let usage = done.ok_or_else(|| anyhow::anyhow!("stream ended without Done"))?;
    assert_eq!(usage.media.image_count, 1);
    assert_eq!(usage.media.video_count, 0);
    assert!(usage.media.visual_tokens > 0);
    assert!(usage.prompt_tokens > usage.media.visual_tokens);
    assert!((1..=4).contains(&usage.completion_tokens));
    for _ in 0..50 {
        if !uploaded_path.exists() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(!uploaded_path.exists(), "claimed upload was not deleted");
    let _ = std::fs::remove_dir_all(root);
    Ok(())
}
