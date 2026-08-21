use axum::body::{to_bytes, Body};
use axum::extract::{Multipart, Request, State};
use axum::http::header::CONTENT_TYPE;
use axum::response::{IntoResponse, Response};
use axum::{Json, RequestExt};
use serde_json::json;

use super::{media_error, ApiKeyIdentity, AppState};

pub async fn upload(State(state): State<AppState>, request: Request<Body>) -> Response {
    if !state.engine.supports_vision() && !state.engine.supports_video() {
        return crate::api::api_error(
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            "component_unavailable",
            "Active runtime has no vision component. Gemma 4 GGUF is loaded text-only; mmproj runtime is not implemented.",
        );
    }
    let owner = match request.extensions().get::<ApiKeyIdentity>() {
        Some(owner) => owner.clone(),
        None => return crate::api::internal_error("API key identity missing"),
    };
    let content_type = request
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("")
        .to_string();
    let result = if content_type.starts_with("multipart/form-data") {
        upload_multipart(&state, owner, request).await
    } else {
        upload_raw(&state, owner, request, &content_type).await
    };
    match result {
        Ok(media) => Json(json!({
            "id": media.id,
            "object": "media",
            "kind": media.kind,
            "bytes": media.encoded_bytes,
            "expires_in": state.media.config.upload_ttl.as_secs(),
        }))
        .into_response(),
        Err(error) => media_error(error),
    }
}

async fn upload_raw(
    state: &AppState,
    owner: ApiKeyIdentity,
    request: Request<Body>,
    content_type: &str,
) -> Result<crate::media::StoredMedia, crate::media::MediaError> {
    let limit = state.media.config.max_video_bytes as usize;
    let bytes = to_bytes(request.into_body(), limit + 1)
        .await
        .map_err(|_| {
            crate::media::MediaError::new(
                crate::media::MediaErrorKind::EncodedTooLarge,
                "media upload exceeds size limit",
            )
        })?;
    state.media.store.upload(owner.0, content_type, &bytes)
}

async fn upload_multipart(
    state: &AppState,
    owner: ApiKeyIdentity,
    request: Request<Body>,
) -> Result<crate::media::StoredMedia, crate::media::MediaError> {
    let mut multipart = request
        .extract::<Multipart, _>()
        .await
        .map_err(|_| crate::media::MediaError::invalid("invalid multipart upload"))?;
    let mut media = None;
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|_| crate::media::MediaError::invalid("invalid multipart upload"))?
    {
        if field.name() != Some("file") || media.is_some() {
            return Err(crate::media::MediaError::invalid(
                "multipart upload must contain exactly one file field",
            ));
        }
        let mime = field.content_type().map(str::to_string).ok_or_else(|| {
            crate::media::MediaError::invalid("multipart file has no Content-Type")
        })?;
        let bytes = field
            .bytes()
            .await
            .map_err(|_| crate::media::MediaError::invalid("cannot read multipart file"))?;
        media = Some(state.media.store.upload(owner.0, &mime, &bytes)?);
    }
    media.ok_or_else(|| crate::media::MediaError::invalid("multipart file field missing"))
}
