use qwen36_server::profile::{
    ArtifactKind, ArtifactRef, ArtifactSet, DeclaredCapabilities, GateResultRef, HashedFile,
    LoadPolicy, MediaLimits, ProcessorProfile, ProfileManifest, ResolvedProfile, RuntimeBundle,
    SourceInventory, SourceProvenance, TokenizerProfile, QWEN35_REPOSITORY, QWEN35_REVISION,
    SERVER_ABI,
};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT_ID: AtomicUsize = AtomicUsize::new(1);

struct Fixture {
    root: PathBuf,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn file(root: &Path, relative: &str, bytes: &[u8]) -> HashedFile {
    let path = root.join(relative);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, bytes).unwrap();
    HashedFile {
        path: PathBuf::from(relative),
        bytes: bytes.len() as u64,
        sha256: digest(bytes),
    }
}

fn artifact(
    root: &Path,
    relative: &str,
    bytes: &[u8],
    kind: ArtifactKind,
    quant: &str,
) -> ArtifactRef {
    let hashed = file(root, relative, bytes);
    ArtifactRef {
        kind,
        path: hashed.path,
        bytes: hashed.bytes,
        sha256: hashed.sha256,
        quant: quant.into(),
        component_abi: SERVER_ABI.into(),
        load: if kind == ArtifactKind::Text {
            LoadPolicy::Startup
        } else {
            LoadPolicy::OnDemand
        },
        required_for_release: true,
    }
}

fn new_fixture() -> Fixture {
    let root = std::env::temp_dir().join(format!(
        "qwen36-profile-test-{}-{}",
        std::process::id(),
        NEXT_ID.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&root).unwrap();
    Fixture { root }
}

fn make_manifest(root: &Path) -> ProfileManifest {
    let text = artifact(
        root,
        "artifacts/text.gguf",
        b"text",
        ArtifactKind::Text,
        "Q4_K_M",
    );
    let vision = artifact(
        root,
        "artifacts/vision.gguf",
        b"vision",
        ArtifactKind::Vision,
        "Q8_0",
    );
    let mtp = artifact(
        root,
        "artifacts/mtp.gguf",
        b"mtp",
        ArtifactKind::Mtp,
        "Q8_0",
    );
    ProfileManifest {
        schema_version: 1,
        profile_id: "qwen3.5-4b".into(),
        release_version: "test-v1".into(),
        server_abi: SERVER_ABI.into(),
        source: SourceProvenance {
            repository: QWEN35_REPOSITORY.into(),
            revision: QWEN35_REVISION.into(),
            license: "Apache-2.0".into(),
            files: vec![],
            source_inventory: SourceInventory {
                text: 426,
                vision: 297,
                mtp: 15,
                total: 738,
            },
        },
        artifacts: ArtifactSet {
            text,
            vision: Some(vision),
            mtp: Some(mtp),
        },
        tokenizer: TokenizerProfile {
            end_of_text: 248044,
            chat_eos: 248046,
            pad: 248044,
            im_start: 248045,
            vision_start: 248053,
            vision_end: 248054,
            image_pad: 248056,
            video_pad: 248057,
        },
        processor: ProcessorProfile {
            transformers_revision: "00e8e49eb3eda67290f635f6bdf59f236f6adf7e".into(),
            image_config: file(root, "processor/image.json", b"image"),
            video_config: file(root, "processor/video.json", b"video"),
            patch_size: 16,
            temporal_patch_size: 2,
            merge_size: 2,
            mrope_sections: [11, 11, 10],
            video_fps: 2,
            min_frames: 4,
            max_frames: 768,
        },
        runtime: RuntimeBundle {
            server: file(root, "runtime/server.exe", b"server"),
            media_helper: file(root, "runtime/helper.exe", b"helper"),
            ffmpeg: file(root, "runtime/ffmpeg.exe", b"ffmpeg"),
            ffprobe: file(root, "runtime/ffprobe.exe", b"ffprobe"),
            licenses: vec![file(root, "runtime/licenses/LICENSE", b"license")],
            codecs_network_disabled: true,
        },
        capabilities: DeclaredCapabilities {
            text: true,
            vision: true,
            video: true,
            mtp: true,
            native_context: 262144,
        },
        limits: MediaLimits {
            max_images: 8,
            max_image_bytes: 20 * 1024 * 1024,
            max_video_bytes: 200 * 1024 * 1024,
            max_video_seconds: 600,
            max_pending_uploads: 16,
            max_temp_bytes: 1024 * 1024 * 1024,
            max_decoded_bytes: 2 * 1024 * 1024 * 1024,
            upload_ttl_seconds: 900,
        },
        gates: vec![GateResultRef {
            name: "phase-1".into(),
            passed: true,
            result: file(root, "gates/phase-1.json", b"pass"),
        }],
    }
}

fn write_manifest(root: &Path, manifest: &ProfileManifest) -> PathBuf {
    let path = root.join("profile.json");
    std::fs::write(&path, serde_json::to_vec_pretty(manifest).unwrap()).unwrap();
    path
}

#[test]
fn valid_profile_derives_capabilities() {
    let fixture = new_fixture();
    let path = write_manifest(&fixture.root, &make_manifest(&fixture.root));
    let profile = ResolvedProfile::load(&path).unwrap();
    let capabilities = profile.capabilities();
    assert!(capabilities.text && capabilities.vision && capabilities.video && capabilities.mtp);
    assert_eq!(capabilities.native_context, 262144);
}

#[test]
fn missing_optional_components_preserve_text() {
    let fixture = new_fixture();
    let manifest = make_manifest(&fixture.root);
    std::fs::remove_file(fixture.root.join("artifacts/vision.gguf")).unwrap();
    std::fs::remove_file(fixture.root.join("artifacts/mtp.gguf")).unwrap();
    let path = write_manifest(&fixture.root, &manifest);
    let profile = ResolvedProfile::load(&path).unwrap();
    let capabilities = profile.capabilities();
    assert!(capabilities.text);
    assert!(!capabilities.vision && !capabilities.video && !capabilities.mtp);
}

#[test]
fn corrupt_text_fails_profile() {
    let fixture = new_fixture();
    let manifest = make_manifest(&fixture.root);
    std::fs::write(fixture.root.join("artifacts/text.gguf"), b"corrupt").unwrap();
    let error = ResolvedProfile::load(&write_manifest(&fixture.root, &manifest)).unwrap_err();
    assert!(format!("{error:#}").contains("text artifact"));
}

#[test]
fn bad_hash_and_path_traversal_fail() {
    let fixture = new_fixture();
    let mut manifest = make_manifest(&fixture.root);
    manifest.artifacts.text.sha256 = "0".repeat(64);
    assert!(ResolvedProfile::load(&write_manifest(&fixture.root, &manifest)).is_err());

    let fixture = new_fixture();
    let mut manifest = make_manifest(&fixture.root);
    manifest.artifacts.text.path = PathBuf::from("../outside.gguf");
    let error = ResolvedProfile::load(&write_manifest(&fixture.root, &manifest)).unwrap_err();
    assert!(format!("{error:#}").contains("traversal"));
}

#[test]
fn unknown_schema_fields_and_duplicate_paths_fail() {
    let fixture = new_fixture();
    let manifest = make_manifest(&fixture.root);
    let path = write_manifest(&fixture.root, &manifest);
    let mut document: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    document["unknown"] = json!(true);
    std::fs::write(&path, serde_json::to_vec(&document).unwrap()).unwrap();
    assert!(ResolvedProfile::load(&path).is_err());

    let fixture = new_fixture();
    let mut manifest = make_manifest(&fixture.root);
    manifest.runtime.ffprobe = manifest.runtime.ffmpeg.clone();
    let error = ResolvedProfile::load(&write_manifest(&fixture.root, &manifest)).unwrap_err();
    assert!(format!("{error:#}").contains("duplicate bundle path"));
}

#[test]
fn incompatible_abi_and_failed_gate_fail() {
    let fixture = new_fixture();
    let mut manifest = make_manifest(&fixture.root);
    manifest.server_abi = "future-v2".into();
    assert!(ResolvedProfile::load(&write_manifest(&fixture.root, &manifest)).is_err());

    let fixture = new_fixture();
    let mut manifest = make_manifest(&fixture.root);
    manifest.gates[0].passed = false;
    let error = ResolvedProfile::load(&write_manifest(&fixture.root, &manifest)).unwrap_err();
    assert!(format!("{error:#}").contains("mandatory gate"));
}

#[test]
fn current_pointer_validates_manifest_hash() {
    let fixture = new_fixture();
    let release = fixture.root.join("releases/v1");
    std::fs::create_dir_all(&release).unwrap();
    let manifest = make_manifest(&release);
    let manifest_path = write_manifest(&release, &manifest);
    let manifest_bytes = std::fs::read(&manifest_path).unwrap();
    let current = fixture.root.join("current");
    std::fs::write(
        &current,
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "release": "releases/v1",
            "manifest_sha256": digest(&manifest_bytes),
        }))
        .unwrap(),
    )
    .unwrap();
    assert!(ResolvedProfile::load(&current).is_ok());

    std::fs::write(&current, br#"{"schema_version":1,"release":"releases/v1","manifest_sha256":"0000000000000000000000000000000000000000000000000000000000000000"}"#).unwrap();
    assert!(ResolvedProfile::load(&current).is_err());
}
