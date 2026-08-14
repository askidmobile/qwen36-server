use super::MediaKind;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub const PROTOCOL_VERSION: u32 = 1;

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DecodeLimits {
    pub encoded_bytes: u64,
    pub max_width: u32,
    pub max_height: u32,
    pub max_frames: usize,
    pub max_output_bytes: u64,
    pub cpu_time_ms: u64,
    pub wall_time_ms: u64,
    pub ram_bytes: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DecodeRequest {
    pub protocol_version: u32,
    pub input_path: PathBuf,
    pub output_dir: PathBuf,
    pub declared_mime: String,
    pub expected_kind: MediaKind,
    #[serde(default)]
    pub frame_indices: Vec<usize>,
    pub limits: DecodeLimits,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DecodedFrame {
    pub path: PathBuf,
    pub width: u32,
    pub height: u32,
    pub timestamp_ms: u64,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DecodeResult {
    pub protocol_version: u32,
    pub kind: MediaKind,
    pub detected_format: String,
    pub codec: Option<String>,
    pub width: u32,
    pub height: u32,
    pub duration_ms: Option<u64>,
    pub source_fps: Option<f64>,
    pub static_gif: bool,
    pub frames: Vec<DecodedFrame>,
    pub audio_processed: bool,
}
