use image::{AnimationDecoder, ImageDecoder, ImageFormat};
use moxcms::{ColorProfile, Layout, TransformOptions};
use qwen36_server::media::helper_protocol::{
    DecodeRequest, DecodeResult, DecodedFrame, PROTOCOL_VERSION,
};
use qwen36_server::media::MediaKind;
use sha2::{Digest, Sha256};
use std::io::{BufReader, Read};
use std::path::Path;

fn main() {
    if let Err(error) = run() {
        eprintln!("decode_error");
        let _ = error;
        std::process::exit(1);
    }
}

fn run() -> anyhow::Result<()> {
    let mut body = Vec::new();
    std::io::stdin().take(1024 * 1024).read_to_end(&mut body)?;
    let request: DecodeRequest = serde_json::from_slice(&body)?;
    anyhow::ensure!(request.protocol_version == PROTOCOL_VERSION);
    anyhow::ensure!(request.input_path.is_absolute() && request.output_dir.is_absolute());
    anyhow::ensure!(request.input_path.metadata()?.len() <= request.limits.encoded_bytes);
    std::fs::create_dir_all(&request.output_dir)?;
    let result = match request.expected_kind {
        MediaKind::Image => decode_image(&request)?,
        MediaKind::Video => decode_video(&request)?,
    };
    serde_json::to_writer(std::io::stdout(), &result)?;
    Ok(())
}

fn decode_image(request: &DecodeRequest) -> anyhow::Result<DecodeResult> {
    let format = image::ImageFormat::from_path(&request.input_path)
        .or_else(|_| image::guess_format(&std::fs::read(&request.input_path)?))?;
    if format == ImageFormat::Gif {
        let decoder = image::codecs::gif::GifDecoder::new(BufReader::new(std::fs::File::open(
            &request.input_path,
        )?))?;
        let frames = decoder.into_frames().collect_frames()?;
        anyhow::ensure!(frames.len() == 1, "animated GIF is video");
    }
    let reader = image::ImageReader::open(&request.input_path)?.with_guessed_format()?;
    let mut decoder = reader.into_decoder()?;
    let orientation = decoder
        .orientation()
        .unwrap_or(image::metadata::Orientation::NoTransforms);
    let icc = decoder.icc_profile().ok().flatten();
    let mut image = image::DynamicImage::from_decoder(decoder)?;
    image.apply_orientation(orientation);
    let image = image.into_rgba8();
    let (width, height) = image.dimensions();
    validate_dimensions(request, width, height, 1)?;
    let mut rgb = Vec::with_capacity(width as usize * height as usize * 3);
    for pixel in image.pixels() {
        let alpha = u16::from(pixel[3]);
        for channel in &pixel.0[..3] {
            let composited = (u16::from(*channel) * alpha + 255 * (255 - alpha) + 127) / 255;
            rgb.push(composited as u8);
        }
    }
    if let Some(icc) = icc {
        let source = ColorProfile::new_from_slice(&icc)?;
        let destination = ColorProfile::new_srgb();
        let transform = source.create_transform_8bit(
            Layout::Rgb,
            &destination,
            Layout::Rgb,
            TransformOptions::default(),
        )?;
        let mut converted = vec![0; rgb.len()];
        for (input, output) in rgb
            .chunks_exact(width as usize * 3)
            .zip(converted.chunks_exact_mut(width as usize * 3))
        {
            transform.transform(input, output)?;
        }
        rgb = converted;
    }
    let frame = write_frame(&request.output_dir, 0, width, height, 0, &rgb)?;
    Ok(DecodeResult {
        protocol_version: PROTOCOL_VERSION,
        kind: MediaKind::Image,
        detected_format: format_name(format).into(),
        codec: None,
        width,
        height,
        duration_ms: None,
        source_fps: None,
        static_gif: format == ImageFormat::Gif,
        frames: vec![frame],
        audio_processed: false,
    })
}

fn decode_video(request: &DecodeRequest) -> anyhow::Result<DecodeResult> {
    let runtime = std::env::current_exe()?.parent().unwrap().to_path_buf();
    let ffprobe = runtime.join(if cfg!(windows) {
        "ffprobe.exe"
    } else {
        "ffprobe"
    });
    let ffmpeg = runtime.join(if cfg!(windows) {
        "ffmpeg.exe"
    } else {
        "ffmpeg"
    });
    anyhow::ensure!(
        ffprobe.is_file() && ffmpeg.is_file(),
        "codec sibling missing"
    );
    let probe = std::process::Command::new(&ffprobe)
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,r_frame_rate,duration",
            "-of",
            "json",
        ])
        .arg(&request.input_path)
        .env_clear()
        .output()?;
    anyhow::ensure!(probe.status.success() && probe.stdout.len() <= 1024 * 1024);
    let document: serde_json::Value = serde_json::from_slice(&probe.stdout)?;
    let stream = document["streams"]
        .as_array()
        .and_then(|items| items.first())
        .ok_or_else(|| anyhow::anyhow!("no video stream"))?;
    let width = stream["width"]
        .as_u64()
        .and_then(|v| u32::try_from(v).ok())
        .ok_or_else(|| anyhow::anyhow!("bad width"))?;
    let height = stream["height"]
        .as_u64()
        .and_then(|v| u32::try_from(v).ok())
        .ok_or_else(|| anyhow::anyhow!("bad height"))?;
    let fps = parse_rate(stream["r_frame_rate"].as_str().unwrap_or("0/1"))?;
    let duration = stream["duration"]
        .as_str()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.0);
    anyhow::ensure!(fps.is_finite() && duration.is_finite());
    let indices = if request.frame_indices.is_empty() {
        vec![0]
    } else {
        request.frame_indices.clone()
    };
    validate_dimensions(request, width, height, indices.len())?;
    let filter = indices
        .iter()
        .map(|index| format!("eq(n\\,{index})"))
        .collect::<Vec<_>>()
        .join("+");
    let pattern = request.output_dir.join("frame-%06d.rgb");
    let output = std::process::Command::new(&ffmpeg)
        .args(["-v", "error", "-nostdin", "-i"])
        .arg(&request.input_path)
        .args([
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            &format!("select='{filter}'"),
            "-vsync",
            "0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
        ])
        .arg(&pattern)
        .env_clear()
        .output()?;
    anyhow::ensure!(output.status.success() && output.stderr.len() <= 64 * 1024);
    let frame_bytes = u64::from(width) * u64::from(height) * 3;
    let mut frames = Vec::new();
    for (position, index) in indices.iter().enumerate() {
        let source = request
            .output_dir
            .join(format!("frame-{:06}.rgb", position + 1));
        let bytes = std::fs::read(&source)?;
        anyhow::ensure!(bytes.len() as u64 == frame_bytes);
        frames.push(DecodedFrame {
            path: source,
            width,
            height,
            timestamp_ms: if fps > 0.0 {
                ((*index as f64 / fps) * 1000.0).round() as u64
            } else {
                0
            },
            bytes: frame_bytes,
            sha256: format!("{:x}", Sha256::digest(&bytes)),
        });
    }
    anyhow::ensure!(frames.len() == indices.len());
    Ok(DecodeResult {
        protocol_version: PROTOCOL_VERSION,
        kind: MediaKind::Video,
        detected_format: "video".into(),
        codec: stream["codec_name"].as_str().map(str::to_string),
        width,
        height,
        duration_ms: Some((duration.max(0.0) * 1000.0).round() as u64),
        source_fps: Some(fps),
        static_gif: false,
        frames,
        audio_processed: false,
    })
}

fn validate_dimensions(
    request: &DecodeRequest,
    width: u32,
    height: u32,
    frames: usize,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        width > 0
            && height > 0
            && width <= request.limits.max_width
            && height <= request.limits.max_height
    );
    anyhow::ensure!(frames > 0 && frames <= request.limits.max_frames);
    let bytes = u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|value| value.checked_mul(3))
        .and_then(|value| value.checked_mul(frames as u64))
        .ok_or_else(|| anyhow::anyhow!("output overflow"))?;
    anyhow::ensure!(bytes <= request.limits.max_output_bytes);
    Ok(())
}

fn write_frame(
    output: &Path,
    index: usize,
    width: u32,
    height: u32,
    timestamp_ms: u64,
    bytes: &[u8],
) -> anyhow::Result<DecodedFrame> {
    let path = output.join(format!("frame-{index:06}.rgb"));
    std::fs::write(&path, bytes)?;
    Ok(DecodedFrame {
        path,
        width,
        height,
        timestamp_ms,
        bytes: bytes.len() as u64,
        sha256: format!("{:x}", Sha256::digest(bytes)),
    })
}

fn parse_rate(value: &str) -> anyhow::Result<f64> {
    let (numerator, denominator) = value
        .split_once('/')
        .ok_or_else(|| anyhow::anyhow!("bad rate"))?;
    let denominator: f64 = denominator.parse()?;
    anyhow::ensure!(denominator != 0.0);
    Ok(numerator.parse::<f64>()? / denominator)
}

fn format_name(format: ImageFormat) -> &'static str {
    match format {
        ImageFormat::Jpeg => "jpeg",
        ImageFormat::Png => "png",
        ImageFormat::WebP => "webp",
        ImageFormat::Bmp => "bmp",
        ImageFormat::Tiff => "tiff",
        ImageFormat::Gif => "gif",
        _ => "unsupported",
    }
}
