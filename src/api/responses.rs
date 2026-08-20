use axum::{
    extract::{Extension, State},
    response::{sse::Event, IntoResponse, Response},
    Json,
};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::api::{
    bad_request, engine_error, generate_collect, prepare_inference_request, sse_response,
    ApiKeyIdentity, AppState,
};
use crate::engine_types::{ChatMessage, ContentBlock, GenParams, GenerationUsage, StreamEvent};
use crate::media::MediaKind;

use super::openai::now_unix;

#[derive(Deserialize)]
pub struct ResponsesRequest {
    #[allow(dead_code)]
    model: Option<String>,
    input: Value, // string | messages-массив
    #[serde(default)]
    stream: bool,
    max_output_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
}

fn input_to_messages(input: &Value) -> Result<Vec<ChatMessage>, Response> {
    match input {
        Value::String(s) => Ok(vec![ChatMessage::text("user", s)]),
        Value::Array(arr) => arr
            .iter()
            .map(|m| {
                let role = m
                    .get("role")
                    .and_then(|r| r.as_str())
                    .unwrap_or("user")
                    .to_string();
                let content = match m.get("content") {
                    Some(Value::String(s)) => vec![super::content::text(s)],
                    Some(Value::Array(parts)) => parts
                        .iter()
                        .map(|part| match part.get("type").and_then(Value::as_str) {
                            Some("input_text") | Some("text") | Some("output_text") => part
                                .get("text")
                                .and_then(Value::as_str)
                                .map(super::content::text)
                                .ok_or_else(|| bad_request("text block requires text")),
                            Some("input_image") | Some("image_url") => super::content::media(
                                part,
                                MediaKind::Image,
                                &["image_url", "url"],
                                part.get("media_id").and_then(Value::as_str),
                            ),
                            Some("input_video") | Some("video_url") => super::content::media(
                                part,
                                MediaKind::Video,
                                &["video_url", "url"],
                                part.get("media_id").and_then(Value::as_str),
                            ),
                            _ => Err(bad_request("unsupported content block type")),
                        })
                        .collect::<Result<Vec<ContentBlock>, Response>>()?,
                    _ => vec![],
                };
                Ok(ChatMessage { role, content, tool_calls: Vec::new(), reasoning_content: None })
            })
            .collect(),
        _ => Err(bad_request(
            "input must be a string or an array of messages",
        )),
    }
}

fn response_object(
    id: &str,
    model: &str,
    text: &str,
    usage: GenerationUsage,
    status: &str,
) -> Value {
    json!({
        "id": id,
        "object": "response",
        "created_at": now_unix(),
        "status": status,
        "model": model,
        "output": [{
            "type": "message",
            "id": format!("msg_{}", uuid::Uuid::new_v4().simple()),
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text}],
        }],
        "usage": {
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "total_tokens": usage.prompt_tokens + usage.completion_tokens,
            "truncated": usage.truncated,
            "media": usage.media,
            "mtp": usage.mtp,
        },
    })
}

pub async fn responses(
    State(state): State<AppState>,
    Extension(owner): Extension<ApiKeyIdentity>,
    Json(req): Json<ResponsesRequest>,
) -> Response {
    let messages = match input_to_messages(&req.input) {
        Ok(m) => m,
        Err(r) => return r,
    };
    let mut params = GenParams::default();
    if let Some(m) = req.max_output_tokens {
        if m == 0 {
            return bad_request("max_output_tokens must be greater than 0");
        }
        params.max_tokens = m;
    }
    if let Some(t) = req.temperature {
        params.temperature = t;
    }
    if let Some(t) = req.top_p {
        params.top_p = t;
    }

    let id = format!("resp_{}", uuid::Uuid::new_v4().simple());
    let model = state.engine.model_info().id;

    let request = match prepare_inference_request(&state, messages, params, &owner, None, None).await {
        Ok(request) => request,
        Err(response) => return response,
    };
    if !req.stream {
        let out = match generate_collect(state.engine.as_ref(), request).await {
            Ok(o) => o,
            Err(r) => return r,
        };
        return Json(response_object(
            &id,
            &model,
            &out.text,
            out.usage,
            "completed",
        ))
        .into_response();
    }

    let cancel = request.cancel.clone();
    let rx = match state.engine.generate(request).await {
        Ok(r) => r,
        Err(e) => return engine_error(e),
    };

    let mut acc = String::new();
    let mut created_sent = false;
    let created = json!({
        "type": "response.created",
        "response": response_object(&id, &model, "", GenerationUsage::default(), "in_progress"),
    })
    .to_string();

    sse_response(rx, cancel, move |ev, out| match ev {
        StreamEvent::Delta(d) => {
            if !created_sent {
                created_sent = true;
                out.push(
                    Event::default()
                        .event("response.created")
                        .data(created.clone()),
                );
            }
            acc.push_str(&d);
            out.push(
                Event::default()
                    .event("response.output_text.delta")
                    .data(json!({"type": "response.output_text.delta", "delta": d}).to_string()),
            );
            true
        }
        StreamEvent::Done { usage, .. } => {
            if !created_sent {
                created_sent = true;
                out.push(
                    Event::default()
                        .event("response.created")
                        .data(created.clone()),
                );
            }
            let obj = response_object(&id, &model, &acc, usage, "completed");
            out.push(
                Event::default()
                    .event("response.completed")
                    .data(json!({"type": "response.completed", "response": obj}).to_string()),
            );
            false
        }
        StreamEvent::Error(e) => {
            out.push(
                Event::default()
                    .data(json!({"error": {"type": "internal_error", "message": e}}).to_string()),
            );
            false
        }
    })
}
