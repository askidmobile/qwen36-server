use base64::Engine as _;
use qwen35_batch::real::multimodal::{
    process_image, process_video, select_video_frame_indices, DecodedRgb, PackedMedia,
};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::helper_protocol::{DecodeLimits, DecodeRequest, DecodeResult, PROTOCOL_VERSION};
use super::{
    ClaimedMedia, DecodedReservation, MediaError, MediaErrorKind, MediaKind, MediaService,
    OwnerDigest, StoredMedia,
};
use crate::engine_types::{CancelFlag, ChatMessage, ContentBlock, MediaSource, MediaUsage};

const MAX_DIMENSION: u32 = 32_768;
const IMAGE_DECODE_BYTES: u64 = 3 * 16_777_216;
const VIDEO_DECODE_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const HELPER_CPU_MS: u64 = 30_000;
const HELPER_WALL_MS: u64 = 30_000;
const HELPER_RAM_BYTES: u64 = 2 * 1024 * 1024 * 1024;

#[derive(Debug)]
pub enum PreparedContentBlock {
    Text(String),
    Media(PackedMedia),
}

#[derive(Debug)]
pub struct PreparedChatMessage {
    pub role: String,
    pub content: Vec<PreparedContentBlock>,
}

#[derive(Debug)]
pub struct PreparedRequest {
    messages: Vec<PreparedChatMessage>,
    usage: MediaUsage,
    claims: Vec<ClaimedMedia>,
    decoded: Vec<DecodedReservation>,
    outputs: Vec<OutputDir>,
}

#[derive(Debug)]
pub struct PreparedLease {
    _claims: Vec<ClaimedMedia>,
    _decoded: Vec<DecodedReservation>,
    _outputs: Vec<OutputDir>,
}

impl PreparedRequest {
    pub fn into_parts(self) -> (Vec<PreparedChatMessage>, MediaUsage, PreparedLease) {
        (
            self.messages,
            self.usage,
            PreparedLease {
                _claims: self.claims,
                _decoded: self.decoded,
                _outputs: self.outputs,
            },
        )
    }
}

pub async fn prepare(
    messages: &[ChatMessage],
    owner: OwnerDigest,
    cancel: &CancelFlag,
    service: &MediaService,
    visual_budget: usize,
) -> Result<PreparedRequest, MediaError> {
    let mut upload_ids = Vec::new();
    let mut image_count = 0usize;
    let mut video_count = 0usize;
    for message in messages {
        if message.role == "system" && message.has_media() {
            return Err(MediaError::invalid("system messages cannot contain media"));
        }
        for block in &message.content {
            if let ContentBlock::Media { kind, source } = block {
                match kind {
                    MediaKind::Image => image_count += 1,
                    MediaKind::Video => video_count += 1,
                }
                if let MediaSource::UploadId { id } = source {
                    upload_ids.push(id.clone());
                }
            }
        }
    }
    if image_count > 8 || video_count > 1 {
        return Err(MediaError::invalid(
            "request supports at most 8 images and one video",
        ));
    }
    if image_count + video_count == 0 {
        return Err(MediaError::invalid("request has no media"));
    }

    let mut claims = Vec::new();
    let uploaded = service.store.claim_batch(owner, &upload_ids)?;
    let uploaded_by_id: HashMap<String, StoredMedia> = uploaded
        .media
        .iter()
        .cloned()
        .map(|media| (media.id.clone(), media))
        .collect();
    claims.push(uploaded);

    let mut outputs = Vec::new();
    let mut decoded = Vec::new();
    let mut usage = MediaUsage {
        image_count,
        video_count,
        ..Default::default()
    };
    let mut prepared_messages = Vec::with_capacity(messages.len());
    let per_video_budget = visual_budget.max(1);

    for message in messages {
        let mut content = Vec::with_capacity(message.content.len());
        for block in &message.content {
            check_cancel(cancel)?;
            match block {
                ContentBlock::Text { text } => {
                    content.push(PreparedContentBlock::Text(text.clone()))
                }
                ContentBlock::Media { kind, source } => {
                    let stored = match source {
                        MediaSource::UploadId { id } => uploaded_by_id
                            .get(id)
                            .cloned()
                            .ok_or_else(|| MediaError::invalid("media ID is unavailable"))?,
                        MediaSource::DataUrl {
                            declared_mime,
                            base64,
                        } => {
                            let bytes = base64::engine::general_purpose::STANDARD
                                .decode(base64)
                                .map_err(|_| MediaError::invalid("invalid base64 media data"))?;
                            upload_and_claim(service, owner, declared_mime, &bytes, &mut claims)?
                        }
                        MediaSource::HttpsUrl { url } => {
                            let limit = match kind {
                                MediaKind::Image => service.config.max_image_bytes,
                                MediaKind::Video => service.config.max_video_bytes,
                            };
                            let (bytes, mime) =
                                super::fetch::fetch_public_https(url, limit).await?;
                            check_cancel(cancel)?;
                            upload_and_claim(service, owner, &mime, &bytes, &mut claims)?
                        }
                    };
                    if stored.kind != *kind {
                        return Err(MediaError::invalid(
                            "media kind does not match content block",
                        ));
                    }
                    let packed = match kind {
                        MediaKind::Image => {
                            decoded.push(service.store.reserve_decoded(IMAGE_DECODE_BYTES)?);
                            let output = OutputDir::create(&service.config.temp_root)?;
                            let result =
                                decode(&stored, &output.path, vec![], IMAGE_DECODE_BYTES).await?;
                            let frame = result
                                .frames
                                .first()
                                .ok_or_else(|| MediaError::invalid("decoded image has no frame"))?;
                            let image = read_rgb(frame)?;
                            outputs.push(output);
                            process_image(&image).map_err(processor_error)?
                        }
                        MediaKind::Video => {
                            decoded.push(service.store.reserve_decoded(VIDEO_DECODE_BYTES)?);
                            let probe_output = OutputDir::create(&service.config.temp_root)?;
                            let probe =
                                decode(&stored, &probe_output.path, vec![], VIDEO_DECODE_BYTES)
                                    .await?;
                            let fps = probe
                                .source_fps
                                .filter(|value| value.is_finite() && *value > 0.0)
                                .ok_or_else(|| MediaError::invalid("video has invalid FPS"))?;
                            let duration_ms = probe
                                .duration_ms
                                .filter(|value| *value > 0 && *value <= 10 * 60 * 1000)
                                .ok_or_else(|| {
                                    MediaError::new(
                                        MediaErrorKind::DecodedTooLarge,
                                        "video duration exceeds limit",
                                    )
                                })?;
                            let total_frames =
                                ((duration_ms as f64 / 1000.0) * fps).floor().max(1.0) as usize;
                            let indices = select_video_frame_indices(
                                total_frames,
                                fps,
                                probe.height as usize,
                                probe.width as usize,
                                per_video_budget,
                            )
                            .map_err(processor_error)?;
                            let output = OutputDir::create(&service.config.temp_root)?;
                            let result =
                                decode(&stored, &output.path, indices.clone(), VIDEO_DECODE_BYTES)
                                    .await?;
                            let frames = result
                                .frames
                                .iter()
                                .map(read_rgb)
                                .collect::<Result<Vec<_>, _>>()?;
                            outputs.push(probe_output);
                            outputs.push(output);
                            usage.sampled_frames =
                                usage.sampled_frames.checked_add(indices.len()).ok_or_else(
                                    || MediaError::invalid("sampled frame count overflow"),
                                )?;
                            usage.effective_fps =
                                Some(indices.len() as f64 / (duration_ms as f64 / 1000.0));
                            process_video(&frames, &indices, fps).map_err(processor_error)?
                        }
                    };
                    usage.visual_tokens = usage
                        .visual_tokens
                        .checked_add(packed.visual_tokens().map_err(processor_error)?)
                        .ok_or_else(|| MediaError::invalid("visual token count overflow"))?;
                    content.push(PreparedContentBlock::Media(packed));
                }
            }
        }
        prepared_messages.push(PreparedChatMessage {
            role: message.role.clone(),
            content,
        });
    }
    usage.audio_processed = false;
    Ok(PreparedRequest {
        messages: prepared_messages,
        usage,
        claims,
        decoded,
        outputs,
    })
}

fn upload_and_claim(
    service: &MediaService,
    owner: OwnerDigest,
    mime: &str,
    bytes: &[u8],
    claims: &mut Vec<ClaimedMedia>,
) -> Result<StoredMedia, MediaError> {
    let uploaded = service.store.upload(owner, mime, bytes)?;
    let claim = service.store.claim_batch(owner, &[uploaded.id.clone()])?;
    let stored = claim
        .media
        .first()
        .cloned()
        .ok_or_else(|| MediaError::invalid("media upload claim failed"))?;
    claims.push(claim);
    Ok(stored)
}

async fn decode(
    media: &StoredMedia,
    output_dir: &Path,
    frame_indices: Vec<usize>,
    max_output_bytes: u64,
) -> Result<DecodeResult, MediaError> {
    let request = DecodeRequest {
        protocol_version: PROTOCOL_VERSION,
        input_path: media.path.clone(),
        output_dir: output_dir.to_path_buf(),
        declared_mime: media.declared_mime.clone(),
        expected_kind: media.kind,
        frame_indices,
        limits: DecodeLimits {
            encoded_bytes: media.encoded_bytes,
            max_width: MAX_DIMENSION,
            max_height: MAX_DIMENSION,
            max_frames: 768,
            max_output_bytes,
            cpu_time_ms: HELPER_CPU_MS,
            wall_time_ms: HELPER_WALL_MS,
            ram_bytes: HELPER_RAM_BYTES,
        },
    };
    super::helper::decode(&request).await
}

fn read_rgb(frame: &super::helper_protocol::DecodedFrame) -> Result<DecodedRgb, MediaError> {
    let bytes = std::fs::read(&frame.path)
        .map_err(|_| MediaError::invalid("cannot read decoded media frame"))?;
    DecodedRgb::new(frame.width, frame.height, bytes).map_err(processor_error)
}

fn processor_error(error: anyhow::Error) -> MediaError {
    MediaError::new(MediaErrorKind::DecodedTooLarge, error.to_string())
}

fn check_cancel(cancel: &CancelFlag) -> Result<(), MediaError> {
    if cancel.is_cancelled() {
        Err(MediaError::invalid("request cancelled"))
    } else {
        Ok(())
    }
}

#[derive(Debug)]
struct OutputDir {
    path: PathBuf,
}

impl OutputDir {
    fn create(root: &Path) -> Result<Self, MediaError> {
        let path = root.join(format!("{}.decoded", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir(&path).map_err(|_| {
            MediaError::new(
                MediaErrorKind::Internal,
                "cannot create media output directory",
            )
        })?;
        Ok(Self { path })
    }
}

impl Drop for OutputDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}
