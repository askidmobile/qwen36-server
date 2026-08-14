//! Versioned model profile loading and fail-closed bundle validation.

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs::File;
use std::io::Read;
use std::path::{Component, Path, PathBuf};

pub const PROFILE_SCHEMA_VERSION: u32 = 1;
pub const CURRENT_SCHEMA_VERSION: u32 = 1;
pub const SERVER_ABI: &str = "qwen36-profile-v1";
pub const QWEN35_REPOSITORY: &str = "Qwen/Qwen3.5-4B";
pub const QWEN35_REVISION: &str = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a";

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CurrentPointer {
    pub schema_version: u32,
    pub release: PathBuf,
    pub manifest_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProfileManifest {
    pub schema_version: u32,
    pub profile_id: String,
    pub release_version: String,
    pub server_abi: String,
    pub source: SourceProvenance,
    pub artifacts: ArtifactSet,
    pub tokenizer: TokenizerProfile,
    pub processor: ProcessorProfile,
    pub runtime: RuntimeBundle,
    pub capabilities: DeclaredCapabilities,
    pub limits: MediaLimits,
    pub gates: Vec<GateResultRef>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SourceProvenance {
    pub repository: String,
    pub revision: String,
    pub license: String,
    pub files: Vec<SourceFile>,
    pub source_inventory: SourceInventory,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SourceFile {
    pub name: String,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SourceInventory {
    pub text: usize,
    pub vision: usize,
    pub mtp: usize,
    pub total: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactSet {
    pub text: ArtifactRef,
    pub vision: Option<ArtifactRef>,
    pub mtp: Option<ArtifactRef>,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactKind {
    Text,
    Vision,
    Mtp,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LoadPolicy {
    Startup,
    OnDemand,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactRef {
    pub kind: ArtifactKind,
    pub path: PathBuf,
    pub bytes: u64,
    pub sha256: String,
    pub quant: String,
    pub component_abi: String,
    pub load: LoadPolicy,
    pub required_for_release: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HashedFile {
    pub path: PathBuf,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TokenizerProfile {
    pub end_of_text: u32,
    pub chat_eos: u32,
    pub pad: u32,
    pub im_start: u32,
    pub vision_start: u32,
    pub vision_end: u32,
    pub image_pad: u32,
    pub video_pad: u32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProcessorProfile {
    pub transformers_revision: String,
    pub image_config: HashedFile,
    pub video_config: HashedFile,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub merge_size: usize,
    pub mrope_sections: [usize; 3],
    pub video_fps: usize,
    pub min_frames: usize,
    pub max_frames: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeBundle {
    pub server: HashedFile,
    pub media_helper: HashedFile,
    pub ffmpeg: HashedFile,
    pub ffprobe: HashedFile,
    #[serde(default)]
    pub licenses: Vec<HashedFile>,
    pub codecs_network_disabled: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DeclaredCapabilities {
    pub text: bool,
    pub vision: bool,
    pub video: bool,
    pub mtp: bool,
    pub native_context: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MediaLimits {
    pub max_images: usize,
    pub max_image_bytes: u64,
    pub max_video_bytes: u64,
    pub max_video_seconds: usize,
    pub max_pending_uploads: usize,
    pub max_temp_bytes: u64,
    pub max_decoded_bytes: u64,
    pub upload_ttl_seconds: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GateResultRef {
    pub name: String,
    pub passed: bool,
    pub result: HashedFile,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum ComponentArtifact {
    Absent,
    Available { path: PathBuf },
    Error { message: String },
}

impl ComponentArtifact {
    pub fn available(&self) -> bool {
        matches!(self, Self::Available { .. })
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct EffectiveCapabilities {
    pub text: bool,
    pub vision: bool,
    pub video: bool,
    pub mtp: bool,
    pub native_context: usize,
}

#[derive(Debug, Clone)]
pub struct ResolvedProfile {
    pub manifest: ProfileManifest,
    pub manifest_path: PathBuf,
    pub release_dir: PathBuf,
    pub text_path: PathBuf,
    pub vision: ComponentArtifact,
    pub mtp: ComponentArtifact,
    pub media_runtime_error: Option<String>,
}

impl ResolvedProfile {
    pub fn load(path: &Path) -> Result<Self> {
        let document = read_json(path)?;
        if document.get("release").is_some() {
            let pointer: CurrentPointer = serde_json::from_value(document)
                .with_context(|| format!("parse current pointer {}", path.display()))?;
            Self::from_pointer(path, pointer)
        } else {
            let manifest: ProfileManifest = serde_json::from_value(document)
                .with_context(|| format!("parse profile manifest {}", path.display()))?;
            Self::from_manifest(path, manifest)
        }
    }

    fn from_pointer(pointer_path: &Path, pointer: CurrentPointer) -> Result<Self> {
        if pointer.schema_version != CURRENT_SCHEMA_VERSION {
            bail!(
                "unsupported current pointer schema {} (expected {})",
                pointer.schema_version,
                CURRENT_SCHEMA_VERSION
            );
        }
        validate_sha256(&pointer.manifest_sha256, "current.manifest_sha256")?;
        let root = pointer_path
            .parent()
            .ok_or_else(|| anyhow!("current pointer has no parent"))?;
        let release_dir = resolve_relative(root, &pointer.release, "current.release")?;
        let manifest_path = release_dir.join("profile.json");
        validate_file_hash(
            &manifest_path,
            None,
            &pointer.manifest_sha256,
            "current manifest",
        )?;
        let document = read_json(&manifest_path)?;
        let manifest: ProfileManifest = serde_json::from_value(document)
            .with_context(|| format!("parse profile manifest {}", manifest_path.display()))?;
        Self::from_manifest(&manifest_path, manifest)
    }

    fn from_manifest(path: &Path, manifest: ProfileManifest) -> Result<Self> {
        validate_manifest_contract(&manifest)?;
        let release_dir = path
            .parent()
            .ok_or_else(|| anyhow!("profile manifest has no parent"))?
            .canonicalize()
            .with_context(|| format!("canonicalize release directory for {}", path.display()))?;

        let mut bundle_paths = HashSet::new();
        for (label, file) in manifest_hashed_files(&manifest) {
            validate_relative_path(&file.path, label)?;
            if !bundle_paths.insert(file.path.clone()) {
                bail!("duplicate bundle path: {}", file.path.display());
            }
        }
        for (label, artifact) in manifest_artifacts(&manifest) {
            validate_relative_path(&artifact.path, label)?;
            if !bundle_paths.insert(artifact.path.clone()) {
                bail!("duplicate bundle path: {}", artifact.path.display());
            }
        }

        let text_path =
            validate_artifact(&release_dir, &manifest.artifacts.text, ArtifactKind::Text)
                .context("text artifact")?;
        validate_hashed_file(&release_dir, &manifest.runtime.server, "runtime.server")?;
        for gate in &manifest.gates {
            if gate.name.trim().is_empty() {
                bail!("gate name cannot be empty");
            }
            if !gate.passed {
                bail!("mandatory gate did not pass: {}", gate.name);
            }
            validate_hashed_file(&release_dir, &gate.result, "gate result")?;
        }

        let vision = optional_artifact(
            &release_dir,
            manifest.artifacts.vision.as_ref(),
            ArtifactKind::Vision,
        );
        let mtp = optional_artifact(
            &release_dir,
            manifest.artifacts.mtp.as_ref(),
            ArtifactKind::Mtp,
        );
        let media_runtime_error = validate_media_support(&release_dir, &manifest)
            .err()
            .map(|e| format!("{e:#}"));
        let vision = if vision.available() {
            match &media_runtime_error {
                Some(message) => ComponentArtifact::Error {
                    message: message.clone(),
                },
                None => vision,
            }
        } else {
            vision
        };

        Ok(Self {
            manifest,
            manifest_path: path.to_path_buf(),
            release_dir,
            text_path,
            vision,
            mtp,
            media_runtime_error,
        })
    }

    pub fn capabilities(&self) -> EffectiveCapabilities {
        let vision = self.manifest.capabilities.vision && self.vision.available();
        EffectiveCapabilities {
            text: true,
            vision,
            video: self.manifest.capabilities.video && vision && self.media_runtime_error.is_none(),
            mtp: self.manifest.capabilities.mtp && self.mtp.available(),
            native_context: self.manifest.capabilities.native_context,
        }
    }
}

fn read_json(path: &Path) -> Result<serde_json::Value> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parse JSON {}", path.display()))
}

fn validate_manifest_contract(manifest: &ProfileManifest) -> Result<()> {
    if manifest.schema_version != PROFILE_SCHEMA_VERSION {
        bail!(
            "unsupported profile schema {} (expected {})",
            manifest.schema_version,
            PROFILE_SCHEMA_VERSION
        );
    }
    if manifest.server_abi != SERVER_ABI {
        bail!(
            "incompatible server ABI {:?} (expected {:?})",
            manifest.server_abi,
            SERVER_ABI
        );
    }
    for (label, value) in [
        ("profile_id", manifest.profile_id.as_str()),
        ("release_version", manifest.release_version.as_str()),
        ("source.repository", manifest.source.repository.as_str()),
        ("source.revision", manifest.source.revision.as_str()),
        ("source.license", manifest.source.license.as_str()),
    ] {
        if value.trim().is_empty() || value.trim() != value {
            bail!("{label} must be non-empty and trimmed");
        }
    }
    if manifest.profile_id == "qwen3.5-4b" {
        if manifest.source.repository != QWEN35_REPOSITORY
            || manifest.source.revision != QWEN35_REVISION
        {
            bail!("qwen3.5-4b source repository/revision mismatch");
        }
        let expected = SourceInventory {
            text: 426,
            vision: 297,
            mtp: 15,
            total: 738,
        };
        if manifest.source.source_inventory != expected {
            bail!("qwen3.5-4b source inventory mismatch");
        }
        validate_qwen35_tokenizer(&manifest.tokenizer)?;
        if manifest.artifacts.text.quant != "Q4_K_M" {
            bail!("qwen3.5-4b text quant must be Q4_K_M");
        }
        for (name, artifact) in [
            ("vision", manifest.artifacts.vision.as_ref()),
            ("mtp", manifest.artifacts.mtp.as_ref()),
        ] {
            if let Some(artifact) = artifact {
                if artifact.quant != "Q8_0" {
                    bail!("qwen3.5-4b {name} quant must be Q8_0");
                }
            }
        }
    } else if manifest.source.source_inventory.total
        != manifest.source.source_inventory.text
            + manifest.source.source_inventory.vision
            + manifest.source.source_inventory.mtp
    {
        bail!("source inventory total does not match component counts");
    }
    if manifest.gates.is_empty() {
        bail!("profile must contain at least one mandatory gate");
    }
    if !manifest.capabilities.text || manifest.capabilities.native_context == 0 {
        bail!("profile must declare text capability and non-zero native context");
    }
    if manifest.artifacts.text.kind != ArtifactKind::Text
        || manifest.artifacts.text.load != LoadPolicy::Startup
        || !manifest.artifacts.text.required_for_release
    {
        bail!("text artifact must be required startup component");
    }
    if manifest.processor.patch_size == 0
        || manifest.processor.temporal_patch_size == 0
        || manifest.processor.merge_size == 0
        || manifest.processor.min_frames == 0
        || manifest.processor.min_frames > manifest.processor.max_frames
    {
        bail!("invalid processor dimensions/frame bounds");
    }
    if manifest.processor.mrope_sections.iter().sum::<usize>() == 0 {
        bail!("mrope sections cannot all be zero");
    }
    if !manifest.runtime.codecs_network_disabled {
        bail!("runtime codecs must have network disabled");
    }
    let mut source_names = HashSet::new();
    for source in &manifest.source.files {
        if source.name.trim().is_empty() || !source_names.insert(source.name.as_str()) {
            bail!("source file names must be non-empty and unique");
        }
        validate_sha256(&source.sha256, "source file SHA-256")?;
    }
    Ok(())
}

fn validate_qwen35_tokenizer(tokenizer: &TokenizerProfile) -> Result<()> {
    let actual = [
        tokenizer.end_of_text,
        tokenizer.im_start,
        tokenizer.chat_eos,
        tokenizer.vision_start,
        tokenizer.vision_end,
        tokenizer.image_pad,
        tokenizer.video_pad,
        tokenizer.pad,
    ];
    let expected = [
        248044, 248045, 248046, 248053, 248054, 248056, 248057, 248044,
    ];
    if actual != expected {
        bail!("qwen3.5-4b tokenizer ID contract mismatch");
    }
    Ok(())
}

fn manifest_artifacts(manifest: &ProfileManifest) -> Vec<(&'static str, &ArtifactRef)> {
    let mut files = vec![("artifacts.text", &manifest.artifacts.text)];
    if let Some(vision) = manifest.artifacts.vision.as_ref() {
        files.push(("artifacts.vision", vision));
    }
    if let Some(mtp) = manifest.artifacts.mtp.as_ref() {
        files.push(("artifacts.mtp", mtp));
    }
    files
}

fn manifest_hashed_files(manifest: &ProfileManifest) -> Vec<(&'static str, &HashedFile)> {
    let mut files = vec![
        ("processor.image_config", &manifest.processor.image_config),
        ("processor.video_config", &manifest.processor.video_config),
        ("runtime.server", &manifest.runtime.server),
        ("runtime.media_helper", &manifest.runtime.media_helper),
        ("runtime.ffmpeg", &manifest.runtime.ffmpeg),
        ("runtime.ffprobe", &manifest.runtime.ffprobe),
    ];
    for license in &manifest.runtime.licenses {
        files.push(("runtime.license", license));
    }
    for gate in &manifest.gates {
        files.push(("gate.result", &gate.result));
    }
    files
}

fn validate_media_support(root: &Path, manifest: &ProfileManifest) -> Result<()> {
    validate_hashed_file(
        root,
        &manifest.processor.image_config,
        "processor.image_config",
    )?;
    validate_hashed_file(
        root,
        &manifest.processor.video_config,
        "processor.video_config",
    )?;
    validate_hashed_file(root, &manifest.runtime.media_helper, "runtime.media_helper")?;
    validate_hashed_file(root, &manifest.runtime.ffmpeg, "runtime.ffmpeg")?;
    validate_hashed_file(root, &manifest.runtime.ffprobe, "runtime.ffprobe")?;
    for license in &manifest.runtime.licenses {
        validate_hashed_file(root, license, "runtime.license")?;
    }
    Ok(())
}

fn optional_artifact(
    root: &Path,
    artifact: Option<&ArtifactRef>,
    expected_kind: ArtifactKind,
) -> ComponentArtifact {
    let Some(artifact) = artifact else {
        return ComponentArtifact::Absent;
    };
    match validate_artifact(root, artifact, expected_kind) {
        Ok(path) => ComponentArtifact::Available { path },
        Err(error) => ComponentArtifact::Error {
            message: format!("{error:#}"),
        },
    }
}

fn validate_artifact(
    root: &Path,
    artifact: &ArtifactRef,
    expected: ArtifactKind,
) -> Result<PathBuf> {
    if artifact.kind != expected {
        bail!(
            "component kind {:?} does not match {:?}",
            artifact.kind,
            expected
        );
    }
    if artifact.component_abi != SERVER_ABI {
        bail!(
            "component ABI {:?} does not match {:?}",
            artifact.component_abi,
            SERVER_ABI
        );
    }
    validate_hashed_file(
        root,
        &HashedFile {
            path: artifact.path.clone(),
            bytes: artifact.bytes,
            sha256: artifact.sha256.clone(),
        },
        "artifact",
    )
}

fn validate_hashed_file(root: &Path, file: &HashedFile, label: &str) -> Result<PathBuf> {
    let path = resolve_relative(root, &file.path, label)?;
    validate_file_hash(&path, Some(file.bytes), &file.sha256, label)?;
    Ok(path)
}

fn validate_file_hash(path: &Path, bytes: Option<u64>, expected: &str, label: &str) -> Result<()> {
    validate_sha256(expected, label)?;
    let metadata =
        std::fs::metadata(path).with_context(|| format!("stat {label}: {}", path.display()))?;
    if !metadata.is_file() {
        bail!("{label} is not a file: {}", path.display());
    }
    if let Some(bytes) = bytes {
        if metadata.len() != bytes {
            bail!(
                "{label} size mismatch for {}: expected {}, got {}",
                path.display(),
                bytes,
                metadata.len()
            );
        }
    }
    let actual = sha256_file(path)?;
    if !actual.eq_ignore_ascii_case(expected) {
        bail!("{label} SHA-256 mismatch for {}", path.display());
    }
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn validate_sha256(value: &str, label: &str) -> Result<()> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("{label} must be a 64-character hexadecimal SHA-256");
    }
    Ok(())
}

fn validate_relative_path(path: &Path, label: &str) -> Result<()> {
    if path.as_os_str().is_empty() || path.is_absolute() {
        bail!("{label} path must be non-empty and relative");
    }
    if path.components().any(|component| {
        matches!(
            component,
            Component::ParentDir | Component::RootDir | Component::Prefix(_)
        )
    }) {
        bail!("{label} path traversal is forbidden: {}", path.display());
    }
    Ok(())
}

fn resolve_relative(root: &Path, relative: &Path, label: &str) -> Result<PathBuf> {
    validate_relative_path(relative, label)?;
    let canonical_root = root
        .canonicalize()
        .with_context(|| format!("canonicalize root {}", root.display()))?;
    let candidate = canonical_root.join(relative);
    let canonical = candidate
        .canonicalize()
        .with_context(|| format!("canonicalize {label}: {}", candidate.display()))?;
    if !canonical.starts_with(&canonical_root) {
        bail!("{label} escapes release root: {}", relative.display());
    }
    Ok(canonical)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_format_is_strict() {
        assert!(validate_sha256(&"a".repeat(64), "test").is_ok());
        assert!(validate_sha256(&"g".repeat(64), "test").is_err());
        assert!(validate_sha256("abc", "test").is_err());
    }

    #[test]
    fn relative_paths_reject_escape() {
        assert!(validate_relative_path(Path::new("artifacts/model.gguf"), "test").is_ok());
        assert!(validate_relative_path(Path::new("../model.gguf"), "test").is_err());
        assert!(validate_relative_path(Path::new("/tmp/model.gguf"), "test").is_err());
    }
}
