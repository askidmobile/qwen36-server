# Qwen3.6 27B — candle inference server

Rust-сервер инференса **Qwen3.6-27B** (GGUF, 2-bit) на собственном candle-форке
[candle-fork-qwen35-batch](../candle-fork-qwen35-batch). Три API
(Chat Completions, Responses, Messages) + веб-чат. 4 одновременных клиента через
batch-планировщик. Целевая площадка: Windows + CUDA (yttri-win, RTX 3060 12 GB).

- Модель: [unsloth/Qwen3.6-27B-GGUF](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF), стартовый квант **UD-Q2_K_XL (11.8 GB)**, фаза 2 — IQ2.
- Контекст: **81 920** токенов на старте (нативный потолок модели 262 144).
- llama.cpp не используется как компонент — только эталон для parity-замеров.

## Статус

Проект на этапе брифа. См. [docs/brief/PROJECT-BRIEF.md](docs/brief/PROJECT-BRIEF.md)
(видение, scope, риски, дорожная карта) и [docs/brief/decisions.md](docs/brief/decisions.md)
(обязательные решения BD-001…BD-020).

## Стек

Rust · candle (форк `candle-fork-qwen35-batch`, path-зависимость) · CUDA/Metal/CPU.

## Пайплайн

```
✅ /create-brief → docs/brief/            ← пройдено
⬜ /create-spec <фича>                    (по каждому пункту дорожной карты)
⬜ /create-spec-plan → /create-spec-implement → /create-spec-review
```
