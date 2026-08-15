# Engine API & Multimodal Specification

## 1. Media Uploads (`POST /v1/media`)
- **Headers:** `Authorization: Bearer <KEY>`, `Content-Type: multipart/form-data` or raw body.
- **Response:** `{"id": "<uuid>", "kind": "image"|"video", "bytes": N}`
- **TTL:** 15 minutes. Single-use, atomic claim bound to the API key.

## 2. Multimodal Formats
- **OpenAI Chat:**
  ```json
  {
    "model": "qwen3.5-4b",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Describe image:"},
          {"type": "image_url", "image_url": {"url": "https://..."}, "media_id": "<optional_id>"}
        ]
      }
    ]
  }
  ```
- **Anthropic Messages:**
  ```json
  {
    "model": "qwen3.5-4b",
    "max_tokens": 1024,
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Describe:"},
          {"type": "image", "source": {"type": "media_id", "media_id": "<id>"}}
        ]
      }
    ]
  }
  ```

## 3. Streaming Usage
Final `data: {"usage": ...}` payload includes:
```json
{
  "prompt_tokens": 512,
  "completion_tokens": 32,
  "truncated": false,
  "media": {
    "image_count": 1,
    "video_count": 0,
    "sampled_frames": 0,
    "visual_tokens": 240,
    "effective_fps": null,
    "audio_processed": false
  },
  "mtp": {
    "enabled": true,
    "used": true,
    "drafted": 12,
    "accepted": 8,
    "fallback_category": null
  }
}
```
