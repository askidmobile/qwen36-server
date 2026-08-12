use axum::{
    extract::State,
    response::{sse::Event, IntoResponse, Response},
    Json,
};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::api::{
    bad_request, generate_collect, internal_error, parse_tool_calls, sse_response, AppState,
};
use crate::engine_types::{ChatMessage, GenParams, StreamEvent};

#[derive(Deserialize)]
pub struct MessagesRequest {
    #[allow(dead_code)]
    model: Option<String>,
    max_tokens: Option<usize>, // обязателен по контракту Anthropic
    system: Option<Value>,     // string | [{type:"text", text}]
    messages: Vec<AntMessage>,
    #[allow(dead_code)]
    tools: Option<Value>,
    #[allow(dead_code)]
    tool_choice: Option<Value>,
    #[serde(default)]
    stream: bool,
    temperature: Option<f32>,
    top_p: Option<f32>,
    stop_sequences: Option<Vec<String>>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum AntContent {
    Text(String),
    Blocks(Vec<Value>),
}

#[derive(Deserialize)]
struct AntMessage {
    role: String,
    content: AntContent,
}

fn flatten_blocks(m: &AntMessage) -> Result<String, Response> {
    match &m.content {
        AntContent::Text(t) => Ok(t.clone()),
        AntContent::Blocks(blocks) => {
            let mut out = String::new();
            for b in blocks {
                match b.get("type").and_then(|t| t.as_str()) {
                    Some("text") => {
                        if let Some(t) = b.get("text").and_then(|t| t.as_str()) {
                            out.push_str(t);
                        }
                    }
                    Some("image") | Some("document") | Some("video") => {
                        return Err(bad_request("vision not supported")); // BD-004
                    }
                    // tool_result и прочее — берём текстовое представление
                    Some("tool_result") => {
                        if let Some(c) = b.get("content") {
                            match c {
                                Value::String(s) => out.push_str(s),
                                Value::Array(parts) => {
                                    for p in parts {
                                        if let Some(t) = p.get("text").and_then(|t| t.as_str()) {
                                            out.push_str(t);
                                        }
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                    _ => {}
                }
            }
            Ok(out)
        }
    }
}

fn system_text(system: &Value) -> String {
    match system {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

fn map_stop_reason(reason: &str, has_tools: bool) -> &'static str {
    if has_tools {
        return "tool_use";
    }
    match reason {
        "length" => "max_tokens",
        "stop" | "end_turn" => "end_turn",
        _ => "end_turn",
    }
}

fn text_and_thinking_blocks(text: &str) -> Vec<Value> {
    // необязательная обёртка thinking-части в content block type:"thinking"
    let mut blocks = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find("<think>") {
        let before = rest[..start].trim();
        if !before.is_empty() {
            blocks.push(json!({"type": "text", "text": before}));
        }
        let after = &rest[start + "<think>".len()..];
        let Some(end) = after.find("</think>") else {
            break;
        };
        blocks.push(json!({"type": "thinking", "thinking": after[..end]}));
        rest = &after[end + "</think>".len()..];
    }
    let tail = rest.trim();
    if !tail.is_empty() {
        blocks.push(json!({"type": "text", "text": tail}));
    }
    blocks
}

fn build_content_blocks(text: &str, calls: &[(String, String)]) -> Vec<Value> {
    let mut blocks = text_and_thinking_blocks(text);
    for (name, args) in calls {
        let input: Value = serde_json::from_str(args).unwrap_or_else(|_| json!({}));
        blocks.push(json!({
            "type": "tool_use",
            "id": format!("toolu_{}", uuid::Uuid::new_v4().simple()),
            "name": name,
            "input": input,
        }));
    }
    if blocks.is_empty() {
        blocks.push(json!({"type": "text", "text": ""}));
    }
    blocks
}

pub async fn messages(State(state): State<AppState>, Json(req): Json<MessagesRequest>) -> Response {
    let Some(max_tokens) = req.max_tokens else {
        return bad_request("max_tokens is required");
    };
    if max_tokens == 0 {
        return bad_request("max_tokens must be greater than 0");
    }

    let mut msgs: Vec<ChatMessage> = Vec::new();
    if let Some(sys) = &req.system {
        let s = system_text(sys);
        if !s.is_empty() {
            msgs.push(ChatMessage {
                role: "system".into(),
                content: s,
            });
        }
    }
    for m in &req.messages {
        let content = match flatten_blocks(m) {
            Ok(c) => c,
            Err(r) => return r,
        };
        msgs.push(ChatMessage {
            role: m.role.clone(),
            content,
        });
    }

    let mut params = GenParams {
        max_tokens,
        ..Default::default()
    };
    if let Some(t) = req.temperature {
        params.temperature = t;
    }
    if let Some(t) = req.top_p {
        params.top_p = t;
    }
    if let Some(stop) = req.stop_sequences {
        params.stop = stop;
    }

    let id = format!("msg_{}", uuid::Uuid::new_v4().simple());
    let model = state.engine.model_info().id;

    if !req.stream {
        let out = match generate_collect(state.engine.as_ref(), msgs, params).await {
            Ok(o) => o,
            Err(r) => return r,
        };
        let (text, calls) = parse_tool_calls(&out.text);
        let content = build_content_blocks(&text, &calls);
        return Json(json!({
            "id": id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": content,
            "stop_reason": map_stop_reason(&out.finish_reason, !calls.is_empty()),
            "stop_sequence": null,
            "usage": {
                "input_tokens": out.prompt_tokens,
                "output_tokens": out.completion_tokens,
                "truncated": out.truncated,
            },
        }))
        .into_response();
    }

    let has_tools = req.tools.as_ref().map(|t| !t.is_null()).unwrap_or(false);
    let rx = match state.engine.generate(msgs, params).await {
        Ok(r) => r,
        Err(e) => return internal_error(e.to_string()),
    };

    let mut acc = String::new();
    let mut started = false;
    let message_start = json!({
        "type": "message_start",
        "message": {
            "id": id, "type": "message", "role": "assistant", "model": model,
            "content": [], "stop_reason": null, "stop_sequence": null,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    })
    .to_string();

    sse_response(rx, move |ev, out| match ev {
        StreamEvent::Delta(d) => {
            if !started {
                started = true;
                out.push(
                    Event::default()
                        .event("message_start")
                        .data(message_start.clone()),
                );
                out.push(
                    Event::default().event("content_block_start").data(
                        json!({"type": "content_block_start", "index": 0,
                           "content_block": {"type": "text", "text": ""}})
                        .to_string(),
                    ),
                );
            }
            acc.push_str(&d);
            if !has_tools {
                out.push(
                    Event::default().event("content_block_delta").data(
                        json!({"type": "content_block_delta", "index": 0,
                           "delta": {"type": "text_delta", "text": d}})
                        .to_string(),
                    ),
                );
            }
            true
        }
        StreamEvent::Done {
            finish_reason,
            prompt_tokens,
            completion_tokens,
            truncated,
        } => {
            if !started {
                started = true;
                out.push(
                    Event::default()
                        .event("message_start")
                        .data(message_start.clone()),
                );
                out.push(
                    Event::default().event("content_block_start").data(
                        json!({"type": "content_block_start", "index": 0,
                           "content_block": {"type": "text", "text": ""}})
                        .to_string(),
                    ),
                );
            }
            let (text, calls) = parse_tool_calls(&acc);
            // tools в запросе: текст был буферизован — эмитим без разметки.
            if has_tools && !text.is_empty() {
                out.push(
                    Event::default().event("content_block_delta").data(
                        json!({"type": "content_block_delta", "index": 0,
                           "delta": {"type": "text_delta", "text": text}})
                        .to_string(),
                    ),
                );
            }
            out.push(
                Event::default()
                    .event("content_block_stop")
                    .data(json!({"type": "content_block_stop", "index": 0}).to_string()),
            );
            // tool_use блоки — буферизованно, целиком (допустимо по заданию)
            for (i, (name, args)) in calls.iter().enumerate() {
                let idx = i + 1;
                let input: Value = serde_json::from_str(args).unwrap_or_else(|_| json!({}));
                out.push(
                    Event::default().event("content_block_start").data(
                        json!({"type": "content_block_start", "index": idx,
                           "content_block": {"type": "tool_use",
                               "id": format!("toolu_{}", uuid::Uuid::new_v4().simple()),
                               "name": name, "input": {}}})
                        .to_string(),
                    ),
                );
                out.push(Event::default().event("content_block_delta").data(
                    json!({"type": "content_block_delta", "index": idx,
                           "delta": {"type": "input_json_delta", "partial_json": input.to_string()}})
                    .to_string(),
                ));
                out.push(
                    Event::default()
                        .event("content_block_stop")
                        .data(json!({"type": "content_block_stop", "index": idx}).to_string()),
                );
            }
            out.push(
                Event::default().event("message_delta").data(
                    json!({"type": "message_delta",
                       "delta": {"stop_reason": map_stop_reason(&finish_reason, !calls.is_empty()),
                                  "stop_sequence": null},
                       "usage": {"input_tokens": prompt_tokens, "output_tokens": completion_tokens,
                                  "truncated": truncated}})
                    .to_string(),
                ),
            );
            out.push(
                Event::default()
                    .event("message_stop")
                    .data(json!({"type": "message_stop"}).to_string()),
            );
            false
        }
        StreamEvent::Error(e) => {
            out.push(
                Event::default().event("error").data(
                    json!({"type": "error", "error": {"type": "internal_error", "message": e}})
                        .to_string(),
                ),
            );
            false
        }
    })
}
