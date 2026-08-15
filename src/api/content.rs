use axum::response::Response;
use serde_json::Value;

use crate::api::bad_request;
use crate::engine_types::{ContentBlock, MediaSource};
use crate::media::MediaKind;

pub fn text(text: impl Into<String>) -> ContentBlock {
    ContentBlock::Text { text: text.into() }
}

pub fn media(
    value: &Value,
    kind: MediaKind,
    url_fields: &[&str],
    media_id: Option<&str>,
) -> Result<ContentBlock, Response> {
    if let Some(id) = media_id {
        validate_media_id(id)?;
        return Ok(ContentBlock::Media {
            kind,
            source: MediaSource::UploadId { id: id.into() },
        });
    }
    let url = url_fields
        .iter()
        .find_map(|field| value.get(*field).and_then(Value::as_str))
        .ok_or_else(|| bad_request("media block requires URL/data URL or media_id"))?;
    let source = parse_url_source(url)?;
    if let MediaSource::DataUrl { declared_mime, .. } = &source {
        validate_declared_kind(declared_mime, kind)?;
    }
    Ok(ContentBlock::Media { kind, source })
}

pub fn anthropic_media(value: &Value, kind: MediaKind) -> Result<ContentBlock, Response> {
    let source = value
        .get("source")
        .and_then(Value::as_object)
        .ok_or_else(|| bad_request("media block requires source"))?;
    let source = match source.get("type").and_then(Value::as_str) {
        Some("url") => parse_url_source(
            source
                .get("url")
                .and_then(Value::as_str)
                .ok_or_else(|| bad_request("media URL is required"))?,
        )?,
        Some("base64") => {
            let declared_mime = source
                .get("media_type")
                .and_then(Value::as_str)
                .ok_or_else(|| bad_request("media_type is required"))?;
            validate_declared_kind(declared_mime, kind)?;
            let base64 = source
                .get("data")
                .and_then(Value::as_str)
                .ok_or_else(|| bad_request("base64 data is required"))?;
            validate_base64(base64)?;
            MediaSource::DataUrl {
                declared_mime: declared_mime.into(),
                base64: base64.into(),
            }
        }
        Some("media_id") => {
            let id = source
                .get("media_id")
                .and_then(Value::as_str)
                .ok_or_else(|| bad_request("media_id is required"))?;
            validate_media_id(id)?;
            MediaSource::UploadId { id: id.into() }
        }
        _ => return Err(bad_request("unsupported media source type")),
    };
    Ok(ContentBlock::Media { kind, source })
}

fn parse_url_source(url: &str) -> Result<MediaSource, Response> {
    if let Some(rest) = url.strip_prefix("data:") {
        let (declared_mime, data) = rest
            .split_once(";base64,")
            .ok_or_else(|| bad_request("media data URL must use base64"))?;
        if !declared_mime.starts_with("image/") && !declared_mime.starts_with("video/") {
            return Err(bad_request("unsupported media data URL MIME"));
        }
        validate_base64(data)?;
        return Ok(MediaSource::DataUrl {
            declared_mime: declared_mime.into(),
            base64: data.into(),
        });
    }
    let parsed = url::Url::parse(url).map_err(|_| bad_request("invalid media URL"))?;
    if parsed.scheme() != "https"
        || !parsed.username().is_empty()
        || parsed.password().is_some()
        || parsed.fragment().is_some()
    {
        return Err(bad_request(
            "media URL must be credential-free HTTPS or a data URL",
        ));
    }
    Ok(MediaSource::HttpsUrl { url: url.into() })
}

fn validate_base64(data: &str) -> Result<(), Response> {
    if data.len() > 4 * 200 * 1024 * 1024 / 3 + 4 {
        return Err(crate::api::api_error(
            axum::http::StatusCode::PAYLOAD_TOO_LARGE,
            "payload_too_large",
            "encoded media exceeds size limit",
        ));
    }
    let mut decoder = base64::read::DecoderReader::new(
        data.as_bytes(),
        &base64::engine::general_purpose::STANDARD,
    );
    std::io::copy(&mut decoder, &mut std::io::sink())
        .map(|_| ())
        .map_err(|_| bad_request("invalid base64 media data"))
}

fn validate_media_id(id: &str) -> Result<(), Response> {
    if id.len() != 32 || !id.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(bad_request("invalid media_id"));
    }
    Ok(())
}

fn validate_declared_kind(mime: &str, kind: MediaKind) -> Result<(), Response> {
    let matches = match kind {
        MediaKind::Image => mime.starts_with("image/"),
        MediaKind::Video => mime.starts_with("video/"),
    };
    if !matches {
        return Err(bad_request("media_type does not match content block"));
    }
    Ok(())
}
