pub mod fetch;
pub mod helper;
pub mod helper_protocol;
pub mod prepare;
pub mod store;

use axum::http::StatusCode;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

pub use store::{ClaimedMedia, DecodedReservation, MediaStore, StoredMedia};

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct OwnerDigest([u8; 32]);

impl OwnerDigest {
    pub fn from_key(key: &str) -> Self {
        use sha2::{Digest, Sha256};
        Self(Sha256::digest(key.as_bytes()).into())
    }

    pub(crate) fn bytes(self) -> [u8; 32] {
        self.0
    }
}

impl fmt::Debug for OwnerDigest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("OwnerDigest([REDACTED])")
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MediaKind {
    Image,
    Video,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaErrorKind {
    Invalid,
    Timeout,
    Conflict,
    EncodedTooLarge,
    DecodedTooLarge,
    UploadLimit,
    ComponentUnavailable,
    StorageExhausted,
    Internal,
}

#[derive(Debug, Clone)]
pub struct MediaError {
    pub kind: MediaErrorKind,
    message: String,
}

impl MediaError {
    pub fn new(kind: MediaErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    pub fn invalid(message: impl Into<String>) -> Self {
        Self::new(MediaErrorKind::Invalid, message)
    }

    pub fn safe_message(&self) -> &str {
        &self.message
    }

    pub fn status(&self) -> StatusCode {
        match self.kind {
            MediaErrorKind::Invalid => StatusCode::BAD_REQUEST,
            MediaErrorKind::Timeout => StatusCode::REQUEST_TIMEOUT,
            MediaErrorKind::Conflict => StatusCode::CONFLICT,
            MediaErrorKind::EncodedTooLarge => StatusCode::PAYLOAD_TOO_LARGE,
            MediaErrorKind::DecodedTooLarge => StatusCode::UNPROCESSABLE_ENTITY,
            MediaErrorKind::UploadLimit => StatusCode::TOO_MANY_REQUESTS,
            MediaErrorKind::ComponentUnavailable => StatusCode::SERVICE_UNAVAILABLE,
            MediaErrorKind::StorageExhausted => StatusCode::INSUFFICIENT_STORAGE,
            MediaErrorKind::Internal => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    pub fn api_kind(&self) -> &'static str {
        match self.kind {
            MediaErrorKind::Invalid => "invalid_request_error",
            MediaErrorKind::Timeout => "timeout_error",
            MediaErrorKind::Conflict => "conflict_error",
            MediaErrorKind::EncodedTooLarge => "payload_too_large",
            MediaErrorKind::DecodedTooLarge => "media_budget_error",
            MediaErrorKind::UploadLimit => "rate_limit_error",
            MediaErrorKind::ComponentUnavailable => "component_unavailable",
            MediaErrorKind::StorageExhausted => "storage_exhausted",
            MediaErrorKind::Internal => "internal_error",
        }
    }
}

impl fmt::Display for MediaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for MediaError {}

#[derive(Debug, Clone)]
pub struct MediaConfig {
    pub temp_root: PathBuf,
    pub max_image_bytes: u64,
    pub max_video_bytes: u64,
    pub max_pending_uploads: usize,
    pub max_temp_bytes: u64,
    pub max_decoded_bytes: u64,
    pub upload_ttl: Duration,
}

impl Default for MediaConfig {
    fn default() -> Self {
        Self {
            temp_root: std::env::temp_dir().join("qwen36-media"),
            max_image_bytes: 20 * 1024 * 1024,
            max_video_bytes: 200 * 1024 * 1024,
            max_pending_uploads: 16,
            max_temp_bytes: 1024 * 1024 * 1024,
            max_decoded_bytes: 2 * 1024 * 1024 * 1024,
            upload_ttl: Duration::from_secs(15 * 60),
        }
    }
}

#[derive(Clone)]
pub struct MediaService {
    pub store: Arc<MediaStore>,
    pub config: MediaConfig,
}

impl MediaService {
    pub fn new(mut config: MediaConfig) -> Result<Self, MediaError> {
        let store = Arc::new(MediaStore::new(config.clone())?);
        config.temp_root = config.temp_root.canonicalize().map_err(|_| {
            MediaError::new(MediaErrorKind::Internal, "cannot canonicalize media store")
        })?;
        store.cleanup_orphans();
        Ok(Self { store, config })
    }
}

pub fn sniff_kind(bytes: &[u8], declared_mime: &str) -> Result<MediaKind, MediaError> {
    let detected = if bytes.starts_with(b"\x89PNG\r\n\x1a\n")
        || bytes.starts_with(b"\xff\xd8\xff")
        || bytes.starts_with(b"GIF87a")
        || bytes.starts_with(b"GIF89a")
        || bytes.starts_with(b"BM")
        || bytes.starts_with(b"II*\0")
        || bytes.starts_with(b"MM\0*")
        || (bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WEBP")
    {
        MediaKind::Image
    } else if (bytes.len() >= 12 && &bytes[4..8] == b"ftyp")
        || bytes.starts_with(b"\x1a\x45\xdf\xa3")
    {
        MediaKind::Video
    } else {
        return Err(MediaError::invalid("unsupported or malformed media"));
    };
    let declared_kind = if declared_mime.starts_with("image/") {
        MediaKind::Image
    } else if declared_mime.starts_with("video/") {
        MediaKind::Video
    } else {
        return Err(MediaError::invalid("unsupported media Content-Type"));
    };
    if detected != declared_kind {
        return Err(MediaError::invalid("media MIME does not match content"));
    }
    Ok(detected)
}
