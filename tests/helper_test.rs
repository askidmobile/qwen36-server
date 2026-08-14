use qwen36_server::media::helper_protocol::{DecodeLimits, DecodeRequest, PROTOCOL_VERSION};
use qwen36_server::media::MediaKind;
use std::io::Write;
use std::process::{Command, Stdio};

fn helper() -> std::path::PathBuf {
    std::env::var_os("CARGO_BIN_EXE_qwen36-media-helper")
        .map(Into::into)
        .expect("helper binary path")
}

fn request(input: &std::path::Path, output: &std::path::Path) -> DecodeRequest {
    DecodeRequest {
        protocol_version: PROTOCOL_VERSION,
        input_path: input.to_path_buf(),
        output_dir: output.to_path_buf(),
        declared_mime: "image/png".into(),
        expected_kind: MediaKind::Image,
        frame_indices: vec![],
        limits: DecodeLimits {
            encoded_bytes: 1024 * 1024,
            max_width: 1024,
            max_height: 1024,
            max_frames: 1,
            max_output_bytes: 4 * 1024 * 1024,
            cpu_time_ms: 5000,
            wall_time_ms: 5000,
            ram_bytes: 256 * 1024 * 1024,
        },
    }
}

fn run(body: &[u8]) -> std::process::Output {
    let mut child = Command::new(helper())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(body).unwrap();
    child.wait_with_output().unwrap()
}

#[test]
fn malformed_and_truncated_input_fail_without_server_process() {
    assert!(!run(b"not json").status.success());
    let root = std::env::temp_dir().join(format!("qwen36-helper-test-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&root).unwrap();
    let input = root.join("bad.png");
    let output = root.join("out");
    std::fs::write(&input, b"\x89PNG\r\n\x1a\ntruncated").unwrap();
    let request = request(&input, &output);
    let result = run(&serde_json::to_vec(&request).unwrap());
    assert!(!result.status.success());
    assert!(result.stdout.is_empty());
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn png_decodes_to_exact_rgb_contract() {
    let root = std::env::temp_dir().join(format!("qwen36-helper-test-{}", uuid::Uuid::new_v4()));
    let output = root.join("out");
    std::fs::create_dir_all(&root).unwrap();
    let input = root.join("image.png");
    let image = image::RgbaImage::from_pixel(2, 1, image::Rgba([255, 0, 0, 128]));
    image.save(&input).unwrap();
    let result = run(&serde_json::to_vec(&request(&input, &output)).unwrap());
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let result: qwen36_server::media::helper_protocol::DecodeResult =
        serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(result.kind, MediaKind::Image);
    assert_eq!((result.width, result.height), (2, 1));
    assert!(!result.audio_processed);
    assert_eq!(result.frames.len(), 1);
    let rgb = std::fs::read(&result.frames[0].path).unwrap();
    assert_eq!(rgb.len(), 6);
    assert_eq!(&rgb[..3], &[255, 127, 127]);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn output_budget_blocks_decode() {
    let root = std::env::temp_dir().join(format!("qwen36-helper-test-{}", uuid::Uuid::new_v4()));
    let output = root.join("out");
    std::fs::create_dir_all(&root).unwrap();
    let input = root.join("image.png");
    image::RgbaImage::new(10, 10).save(&input).unwrap();
    let mut request = request(&input, &output);
    request.limits.max_output_bytes = 10;
    assert!(!run(&serde_json::to_vec(&request).unwrap()).status.success());
    let _ = std::fs::remove_dir_all(root);
}

#[tokio::test]
async fn server_wrapper_validates_helper_result() {
    let root = std::env::temp_dir().join(format!("qwen36-helper-test-{}", uuid::Uuid::new_v4()));
    let output = root.join("out");
    std::fs::create_dir_all(&root).unwrap();
    let input = root.join("image.png");
    image::RgbaImage::from_pixel(2, 2, image::Rgba([0, 255, 0, 255]))
        .save(&input)
        .unwrap();
    std::env::set_var("QWEN36_MEDIA_HELPER", helper());
    let result = qwen36_server::media::helper::decode(&request(&input, &output))
        .await
        .unwrap();
    std::env::remove_var("QWEN36_MEDIA_HELPER");
    assert_eq!(result.frames[0].bytes, 12);
    assert!(result.frames[0].path.starts_with(&output));
    let _ = std::fs::remove_dir_all(root);
}
