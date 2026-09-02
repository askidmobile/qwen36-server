---
topic: Project Overview
slug: project-overview
last_compiled: 2026-09-01
sources: 9
status: active
---

# Project Overview

## Purpose [coverage: high — 9 sources]

`qwen36-server` вырос из сервера одной Qwen3.6-27B в **Yttri Self-Inference Server**: локальный Rust-сервер GGUF-инференса для Qwen 3.5/3.6/3.8, Ornith 1.0/1.5 и Gemma 4. Он предоставляет OpenAI Chat Completions, OpenAI Responses, Anthropic Messages, admin/HuggingFace/media API и проксирует Unsloth Studio.

Основной runtime — собственный `yttri-forge`, подключённый path-зависимостями. Целевая площадка остаётся Windows + CUDA, в частности RTX 3060 12 GB; macOS/Metal используется для разработки. Критерий стабильности BD-008 — четыре слота с длинными генерациями без падений и утечек.

## Architecture [coverage: high — 9 sources]

- `main.rs`: env/profile → выбор движка → `SwappableEngine` → axum.
- `engine_batched.rs`: основной путь qwen35/qwen35moe, включая один слот; continuous batching, paged KV, CUDA graphs, MTP и prefix cache.
- `engine.rs`: общий контракт и `CandleEngine` для других архитектур/диагностики.
- `engine_swap.rs`: горячая замена модели без изменения HTTP-контракта.
- `api/`: OpenAI, Anthropic, Responses, admin, HF, media и Studio proxy.
- `profile.rs`: валидированные профили и component artifacts.
- `media/`: временное хранение и подготовка image/video.
- `prefix_cache.rs`, `vram_plan.rs`: повторное использование префила и планирование памяти.
- `scripts/README.md`: единственный эксплуатационный runbook для Windows-задачи `qwen36-inference`, актуального exe, логов и smoke-проверки.

## Talks To [coverage: high — 9 sources]

- `../yttri-forge/engine/{qwen35-batch,candle-core,candle-transformers}` — инференс и CUDA/Metal kernels.
- axum/tokio — HTTP, SSE, очереди и фоновые задачи.
- Hugging Face — поиск/probe/download GGUF через server API.
- Unsloth Studio — reverse proxy на `STUDIO_URL`; встроенный WebUI только fallback.
- API-клиенты OpenAI/Anthropic-совместимого формата.

## API Surface [coverage: high — 7 sources]

- `/v1/chat/completions`, `/v1/responses`, `/v1/messages`, `/v1/models`.
- Admin: список/переключение/выгрузка моделей, контекстная матрица, пресеты сэмплинга.
- HF: search/files/probe/download/downloads.
- Media upload для runtimes с vision/video.
- Всё вне `/v1/*` проксируется в Studio; `GET /` получает встроенный чат при недоступном Studio.

## Data [coverage: high — 6 sources]

- GGUF/YTF и versioned profile artifacts на диске.
- Временные media-файлы удаляются после использования/TTL/cancel/error (BD-026).
- Model state, paged KV и CUDA graphs — VRAM; prefix snapshots — системная RAM.
- Чаты встроенного WebUI — localStorage; сервер не ведёт историю диалогов.

## Key Decisions [coverage: high — 5 sources]

- BD-021 заменил мастер-ключ именованными API-ключами.
- BD-022 разрешил Vision/Video для валидированных profiles.
- BD-029: MTP обязан сохранять распределение, но не побитовую траекторию при смене формы GPU-вычислений.
- BD-030: включённый MTP загружается при старте; выключенный не занимает VRAM.
- BD-031: переполнение по умолчанию — `400 context_length_exceeded`; sliding window только через `CTX_OVERFLOW=sliding_window`.
- BD-032: выделенный agent endpoint держит sampling в server env; reasoning не выбирает sampling profile.

## Gotchas [coverage: high — 7 sources]

- `CLAUDE.md` и BD-028 ещё называют `candle-fork`, но live `Cargo.toml` зависит только от `yttri-forge`; при расхождении доверять сборочному контракту.
- Исторический brief описывает Qwen3.6-27B/Q2/80K; фактический scope зафиксирован в `PROJECT-BRIEF-addendum.md`.
- VRAM-планер даёт справочную оценку; реальное окно paged pool движок вычисляет отдельно по свободной памяти.
- Полный контекст на 12 GB возможен только при тщательно выбранных slots/KV dtype; WDDM shared usage — обязательная метрика paging.
- Текущее имя модели в `CLAUDE.md`/runbook — снимок документации, а не live-гарантия; перед диагностикой нужно сверять `loaded: id=... quant=...` в свежем серверном логе.

## Sources

- [README.md](../../README.md)
- [CLAUDE.md](../../CLAUDE.md)
- [Cargo.toml](../../Cargo.toml)
- [PROJECT-BRIEF.md](../../docs/brief/PROJECT-BRIEF.md)
- [PROJECT-BRIEF-addendum.md](../../docs/brief/PROJECT-BRIEF-addendum.md)
- [decisions.md](../../docs/brief/decisions.md)
- [open-questions.md](../../docs/brief/open-questions.md)
- [src/main.rs](../../src/main.rs)
- [scripts/README.md](../../scripts/README.md)
