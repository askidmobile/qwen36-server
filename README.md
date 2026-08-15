# Qwen3.6 / Qwen3.5-4B Multimodal & Speculative Inference Server

Inference server for Qwen3.5/3.6 on a custom `candle` fork supporting Text (Q4_K_M), Vision (mixed Q8_0), Video, and Transactional MTP (Q8_0).

## Endpoints

- `POST /v1/chat/completions`: OpenAI Chat API with image, video, and text support.
- `POST /v1/responses`: OpenAI Responses API mapping.
- `POST /v1/messages`: Anthropic Messages API with image and video extensions.
- `POST /v1/media`: Authenticated single-use temporary upload endpoint.
- `GET /v1/models`: Capabilities, limits, and profile info.
- `GET /`: Accessible WebUI chat with keyboard controls and attachment tray.

## Environment Variables

See `.env.example` for details.
- `QWEN36_PROFILE`: Path to profile directory or `current` pointer JSON.
- `QWEN36_PORT`: Port (default `18099`).
- `QWEN36_API_KEYS`: JSON array of `[{"key":"...","name":"..."}]`.
- `QWEN36_SLOTS`: Number of continuous batch slots (1..4, default 4).
- `QWEN36_CTX`: Context window length.
- `QWEN36_MTP`: Enable/disable MTP speculative decoding (`0` or `1`).

## Rollback and Staging

Model releases are published immutably to `releases/<version>/` and activated atomically via `current` pointer write-through updates. To rollback, point `current` to the previous validated release directory.
