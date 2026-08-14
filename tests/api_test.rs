use std::sync::Arc;

use axum::body::{to_bytes, Body};
use axum::http::{Request, StatusCode};
use qwen36_server::api::{build_router, AppState};
use qwen36_server::config::ApiKey;
use qwen36_server::engine_types::{ChatMessage, Engine, GenParams, ModelInfo, StreamEvent};
use tower::ServiceExt;

struct MockEngine {
    deltas: Vec<String>,
}

#[async_trait::async_trait]
impl Engine for MockEngine {
    async fn generate(
        &self,
        _messages: Vec<ChatMessage>,
        _params: GenParams,
    ) -> anyhow::Result<tokio::sync::mpsc::Receiver<StreamEvent>> {
        let (tx, rx) = tokio::sync::mpsc::channel(8);
        let deltas = self.deltas.clone();
        tokio::spawn(async move {
            for d in deltas {
                let _ = tx.send(StreamEvent::Delta(d)).await;
            }
            let _ = tx
                .send(StreamEvent::Done {
                    finish_reason: "stop".into(),
                    prompt_tokens: 10,
                    completion_tokens: 3,
                    truncated: false,
                })
                .await;
        });
        Ok(rx)
    }

    fn model_info(&self) -> ModelInfo {
        ModelInfo {
            id: "qwen3.6-27b".into(),
            context_length: 81920,
            quant: "Q2_K_XL".into(),
            slots: 4,
            modes: vec!["thinking".into(), "instruct".into()],
        }
    }
}

fn app(deltas: Vec<&str>) -> axum::Router {
    let mock: Arc<dyn Engine> = Arc::new(MockEngine {
        deltas: deltas.into_iter().map(String::from).collect(),
    });
    let switcher = Arc::new(qwen36_server::engine_swap::SwappableEngine::new(
        mock,
        std::path::PathBuf::from("mock.gguf"),
        8192,
        4,
    ));
    let media_root =
        std::env::temp_dir().join(format!("qwen36-api-media-{}", uuid::Uuid::new_v4()));
    let media = Arc::new(
        qwen36_server::media::MediaService::new(qwen36_server::media::MediaConfig {
            temp_root: media_root,
            ..Default::default()
        })
        .unwrap(),
    );
    build_router(AppState {
        engine: switcher.clone(),
        media,
        switcher,
        api_keys: vec![
            ApiKey {
                key: "test-key".into(),
                name: "primary".into(),
            },
            ApiKey {
                key: "second-key".into(),
                name: "secondary".into(),
            },
        ]
        .into(),
        models_dir: std::path::PathBuf::from("."),
        profile: None,
        cuda_device: None,
    })
}

fn authed(req: Request<Body>) -> Request<Body> {
    let (mut parts, body) = req.into_parts();
    parts
        .headers
        .insert("authorization", "Bearer test-key".parse().unwrap());
    Request::from_parts(parts, body)
}

fn raw_req(method: &str, uri: &str, content_type: &str, body: Vec<u8>) -> Request<Body> {
    Request::builder()
        .method(method)
        .uri(uri)
        .header("content-type", content_type)
        .body(Body::from(body))
        .unwrap()
}

fn json_req(method: &str, uri: &str, body: serde_json::Value) -> Request<Body> {
    Request::builder()
        .method(method)
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

async fn body_string(resp: axum::response::Response) -> String {
    String::from_utf8(to_bytes(resp.into_body(), 1 << 20).await.unwrap().to_vec()).unwrap()
}

#[tokio::test]
async fn unauthorized_without_key() {
    let resp = app(vec![])
        .oneshot(json_req(
            "POST",
            "/v1/chat/completions",
            serde_json::json!({"messages": []}),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    let body = body_string(resp).await;
    assert!(body.contains("authentication_error"));
}

#[tokio::test]
async fn media_upload_requires_auth_and_returns_owner_bound_id() {
    let bytes = b"\x89PNG\r\n\x1a\nfixture".to_vec();
    let resp = app(vec![])
        .oneshot(raw_req("POST", "/v1/media", "image/png", bytes.clone()))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

    let resp = app(vec![])
        .oneshot(authed(raw_req("POST", "/v1/media", "image/png", bytes)))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let value: serde_json::Value = serde_json::from_str(&body_string(resp).await).unwrap();
    assert_eq!(value["object"], "media");
    assert_eq!(value["kind"], "image");
    assert_eq!(value["expires_in"], 900);
    assert_eq!(value["id"].as_str().unwrap().len(), 32);
}

#[tokio::test]
async fn media_upload_rejects_mime_spoof() {
    let resp = app(vec![])
        .oneshot(authed(raw_req(
            "POST",
            "/v1/media",
            "video/mp4",
            b"\x89PNG\r\n\x1a\nfixture".to_vec(),
        )))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    assert!(body_string(resp).await.contains("invalid_request_error"));
}

#[tokio::test]
async fn zero_output_tokens_are_bad_requests() {
    for (uri, body) in [
        (
            "/v1/chat/completions",
            serde_json::json!({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 0}),
        ),
        (
            "/v1/responses",
            serde_json::json!({"input": "hi", "max_output_tokens": 0}),
        ),
        (
            "/v1/messages",
            serde_json::json!({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 0}),
        ),
    ] {
        let resp = app(vec![])
            .oneshot(authed(json_req("POST", uri, body)))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST, "{uri}");
    }
}

#[tokio::test]
async fn invalid_chat_sampling_parameters_are_bad_requests() {
    for body in [
        serde_json::json!({"messages": [], "temperature": -0.1}),
        serde_json::json!({"messages": [], "top_p": 0.0}),
        serde_json::json!({"messages": [], "min_p": 1.1}),
        serde_json::json!({"messages": [], "presence_penalty": 2.1}),
        serde_json::json!({"messages": [], "repetition_penalty": 0.0}),
    ] {
        let resp = app(vec![])
            .oneshot(authed(json_req("POST", "/v1/chat/completions", body)))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }
}

#[tokio::test]
async fn every_configured_api_key_is_accepted() {
    let req = Request::builder()
        .uri("/v1/models")
        .header("authorization", "Bearer second-key")
        .body(Body::empty())
        .unwrap();
    let resp = app(vec![]).oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let req = Request::builder()
        .uri("/v1/models")
        .header("authorization", "Bearer second-key-wrong")
        .body(Body::empty())
        .unwrap();
    let resp = app(vec![]).oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn models_list_format() {
    let resp = app(vec![])
        .oneshot(authed(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = serde_json::from_str(&body_string(resp).await).unwrap();
    assert_eq!(v["object"], "list");
    assert_eq!(v["data"][0]["id"], "qwen3.6-27b");
    assert_eq!(v["data"][0]["object"], "model");
    assert_eq!(v["data"][0]["owned_by"], "local");
    assert_eq!(v["data"][0]["context_length"], 81920);
    assert_eq!(v["data"][0]["quant"], "Q2_K_XL");
    assert_eq!(v["data"][0]["slots"], 4);
    assert_eq!(v["data"][0]["modes"][0], "thinking");
    assert_eq!(v["data"][0]["capabilities"]["text"], true);
    assert_eq!(v["data"][0]["capabilities"]["vision"], false);
    assert_eq!(v["data"][0]["capabilities"]["video"], false);
    assert_eq!(v["data"][0]["capabilities"]["mtp"]["available"], false);
    assert!(v["data"][0]["profile"].is_null());
}

#[tokio::test]
async fn chat_completions_non_stream() {
    let resp = app(vec!["Hello", " world"])
        .oneshot(authed(json_req(
            "POST",
            "/v1/chat/completions",
            serde_json::json!({"messages": [{"role": "user", "content": "hi"}]}),
        )))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = serde_json::from_str(&body_string(resp).await).unwrap();
    assert_eq!(v["object"], "chat.completion");
    assert_eq!(v["choices"][0]["message"]["role"], "assistant");
    assert_eq!(v["choices"][0]["message"]["content"], "Hello world");
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
    assert_eq!(v["usage"]["prompt_tokens"], 10);
    assert_eq!(v["usage"]["total_tokens"], 13);
}

#[tokio::test]
async fn chat_completions_stream_has_done() {
    let resp = app(vec!["a", "b"])
        .oneshot(authed(json_req(
            "POST",
            "/v1/chat/completions",
            serde_json::json!({
                "messages": [{"role": "user", "content": "hi"}],
                "stream": true,
                "stream_options": {"include_usage": true},
            }),
        )))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_string(resp).await;
    assert!(body.contains("data: [DONE]"), "body: {body}");
    assert!(body.contains("\"role\":\"assistant\""));
    assert!(body.contains("\"content\":\"a\""));
    assert!(body.contains("\"finish_reason\":\"stop\""));
    assert!(body.contains("\"usage\""), "include_usage chunk: {body}");
}

#[tokio::test]
async fn chat_completions_tool_calls() {
    let resp = app(vec![
        "let me check<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}</tool_call>",
    ])
    .oneshot(authed(json_req(
        "POST",
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": [{"type": "function", "function": {"name": "get_weather"}}],
        }),
    )))
    .await
    .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = serde_json::from_str(&body_string(resp).await).unwrap();
    assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(
        v["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
        "get_weather"
    );
    assert_eq!(v["choices"][0]["message"]["content"], "let me check");
}

#[tokio::test]
async fn image_content_rejected() {
    let resp = app(vec![])
        .oneshot(authed(json_req(
            "POST",
            "/v1/chat/completions",
            serde_json::json!({
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "what is this"},
                    {"type": "image_url", "image_url": {"url": "http://x/y.png"}},
                ]}],
            }),
        )))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    assert!(body_string(resp).await.contains("invalid_request_error"));
}

#[tokio::test]
async fn anthropic_requires_max_tokens() {
    let resp = app(vec![])
        .oneshot(authed(json_req(
            "POST",
            "/v1/messages",
            serde_json::json!({"messages": [{"role": "user", "content": "hi"}]}),
        )))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    assert!(body_string(resp).await.contains("max_tokens"));
}

#[tokio::test]
async fn anthropic_non_stream() {
    let resp = app(vec!["Hi", "!"])
        .oneshot(authed(json_req(
            "POST",
            "/v1/messages",
            serde_json::json!({
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hi"}],
            }),
        )))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = serde_json::from_str(&body_string(resp).await).unwrap();
    assert!(v["id"].as_str().unwrap().starts_with("msg_"));
    assert_eq!(v["type"], "message");
    assert_eq!(v["role"], "assistant");
    assert_eq!(v["content"][0]["type"], "text");
    assert_eq!(v["content"][0]["text"], "Hi!");
    assert_eq!(v["stop_reason"], "end_turn");
    assert_eq!(v["usage"]["input_tokens"], 10);
    assert_eq!(v["usage"]["output_tokens"], 3);
}

#[tokio::test]
async fn anthropic_stream() {
    let resp = app(vec!["x"])
        .oneshot(authed(json_req(
            "POST",
            "/v1/messages",
            serde_json::json!({
                "max_tokens": 100,
                "stream": true,
                "messages": [{"role": "user", "content": "hi"}],
            }),
        )))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_string(resp).await;
    for ev in [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ] {
        assert!(
            body.contains(&format!("event: {ev}")),
            "missing {ev} in: {body}"
        );
    }
    assert!(body.contains("text_delta"));
    assert!(body.contains("\"stop_reason\":\"end_turn\""));
}

#[tokio::test]
async fn switch_model_rejects_too_many_slots_before_unloading_current_engine() {
    let app = app(vec![]);
    let resp = app
        .clone()
        .oneshot(authed(json_req(
            "POST",
            "/v1/switch_model",
            serde_json::json!({"path": "Cargo.toml", "slots": 5}),
        )))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

    let resp = app
        .oneshot(authed(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = serde_json::from_str(&body_string(resp).await).unwrap();
    assert_eq!(v["data"][0]["id"], "qwen3.6-27b");
}

#[tokio::test]
async fn responses_non_stream_and_stream() {
    // non-stream
    let resp = app(vec!["answer"])
        .oneshot(authed(json_req(
            "POST",
            "/v1/responses",
            serde_json::json!({"input": "question"}),
        )))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = serde_json::from_str(&body_string(resp).await).unwrap();
    assert_eq!(v["object"], "response");
    assert_eq!(v["status"], "completed");
    assert_eq!(v["output"][0]["content"][0]["text"], "answer");
    assert_eq!(v["usage"]["input_tokens"], 10);

    // stream
    let resp = app(vec!["a", "b"])
        .oneshot(authed(json_req(
            "POST",
            "/v1/responses",
            serde_json::json!({"input": "q", "stream": true}),
        )))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_string(resp).await;
    assert!(body.contains("event: response.created"), "{body}");
    assert!(body.contains("event: response.output_text.delta"), "{body}");
    assert!(body.contains("event: response.completed"), "{body}");
}
