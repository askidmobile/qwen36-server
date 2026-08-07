use axum::{
    extract::State,
    response::{sse::Event, IntoResponse, Response},
    Json,
};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::api::{bad_request, generate_collect, internal_error, sse_response, AppState};
use crate::engine_types::{ChatMessage, GenParams, StreamEvent};

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
        Value::String(s) => Ok(vec![ChatMessage {
            role: "user".into(),
            content: s.clone(),
        }]),
        Value::Array(arr) => arr
            .iter()
            .map(|m| {
                let role = m
                    .get("role")
                    .and_then(|r| r.as_str())
                    .unwrap_or("user")
                    .to_string();
                let content = match m.get("content") {
                    Some(Value::String(s)) => s.clone(),
                    Some(Value::Array(parts)) => {
                        let mut out = String::new();
                        for p in parts {
                            match p.get("type").and_then(|t| t.as_str()) {
                                Some("input_text") | Some("text") | Some("output_text") => {
                                    if let Some(t) = p.get("text").and_then(|t| t.as_str()) {
                                        out.push_str(t);
                                    }
                                }
                                Some("input_image") | Some("image_url") => {
                                    return Err(bad_request("vision not supported")); // BD-004
                                }
                                _ => {}
                            }
                        }
                        out
                    }
                    _ => String::new(),
                };
                Ok(ChatMessage { role, content })
            })
            .collect(),
        _ => Err(bad_request("input must be a string or an array of messages")),
    }
}

fn response_object(id: &str, model: &str, text: &str, usage: (usize, usize), status: &str) -> Value {
    response_object_ext(id, model, text, usage, status, false)
}

fn response_object_ext(id: &str, model: &str, text: &str, usage: (usize, usize), status: &str, truncated: bool) -> Value {
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
            "input_tokens": usage.0,
            "output_tokens": usage.1,
            "total_tokens": usage.0 + usage.1,
            "truncated": truncated,
        },
    })
}

pub async fn responses(State(state): State<AppState>, Json(req): Json<ResponsesRequest>) -> Response {
    let messages = match input_to_messages(&req.input) {
        Ok(m) => m,
        Err(r) => return r,
    };
    let mut params = GenParams::default();
    if let Some(m) = req.max_output_tokens {
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

    if !req.stream {
        let out = match generate_collect(state.engine.as_ref(), messages, params).await {
            Ok(o) => o,
            Err(r) => return r,
        };
        return Json(response_object_ext(
            &id,
            &model,
            &out.text,
            (out.prompt_tokens, out.completion_tokens),
            "completed",
            out.truncated,
        ))
        .into_response();
    }

    let rx = match state.engine.generate(messages, params).await {
        Ok(r) => r,
        Err(e) => return internal_error(e.to_string()),
    };

    let mut acc = String::new();
    let mut created_sent = false;
    let created = json!({
        "type": "response.created",
        "response": response_object(&id, &model, "", (0, 0), "in_progress"),
    })
    .to_string();

    sse_response(rx, move |ev, out| match ev {
        StreamEvent::Delta(d) => {
            if !created_sent {
                created_sent = true;
                out.push(Event::default().event("response.created").data(created.clone()));
            }
            acc.push_str(&d);
            out.push(
                Event::default()
                    .event("response.output_text.delta")
                    .data(json!({"type": "response.output_text.delta", "delta": d}).to_string()),
            );
            true
        }
        StreamEvent::Done {
            prompt_tokens,
            completion_tokens,
            truncated,
            ..
        } => {
            if !created_sent {
                created_sent = true;
                out.push(Event::default().event("response.created").data(created.clone()));
            }
            let obj = response_object_ext(&id, &model, &acc, (prompt_tokens, completion_tokens), "completed", truncated);
            out.push(
                Event::default()
                    .event("response.completed")
                    .data(json!({"type": "response.completed", "response": obj}).to_string()),
            );
            false
        }
        StreamEvent::Error(e) => {
            out.push(Event::default()
                .data(json!({"error": {"type": "internal_error", "message": e}}).to_string()));
            false
        }
    })
}
