---
topic: Project Overview
slug: project-overview
last_compiled: 2026-08-08
sources: 9
status: active
---

# Project Overview

## Purpose [coverage: high — 9 sources]

`qwen36-server` — Rust-сервер инференса **Qwen3.6-27B** (GGUF, 2-bit) на собственном candle-форке [candle-fork](../candle-fork). Три API (OpenAI Chat Completions, OpenAI Responses, Anthropic Messages) + веб-чат. 4 одновременных клиента через batch-планировщик. Целевая площадка: Windows + CUDA (yttri-win, RTX 3060 12 GB).

Ключевые параметры:
- Модель: unsloth/Qwen3.6-27B-GGUF → старт UD-Q2_K_XL (11.8 GB); фаза 2 — IQ2 (BD-003)
- Контекст: 81 920 токенов на старте (нативный потолок 262 144) (BD-009)
- llama.cpp НЕ используется как компонент — только эталон для parity (BD-001)
- Критерий успеха v1: 4 слота × генерации 8-16K токенов без падений и утечек VRAM (BD-008)

Статус: собранный сервер работает. Smoke на реальной GGUF (Qwen3.5-4B) пройден — chat/stream/messages/responses/models. BatchedEngine подключён (фаза 4). IQ2-ядра — фаза 5 (не начата).

## Architecture [coverage: high — 9 sources]

Single Rust-крейт, path-зависимость на `candle-fork`. Структура:

```
src/
  main.rs          — entry point: env → engine → axum HTTP
  lib.rs           — модули: config, engine, sampler, api, engine_batched, engine_types
  config.rs        — Config::from_env (QWEN36_* env vars)
  engine.rs        — Engine trait + CandleEngine (single-slot, Mutex)
  engine_batched.rs — BatchedEngine (4 слота, dispatch loop + BatchScheduler)
  engine_types.rs  — реэкспорт контрактных типов из engine
  sampler.rs       — свой сэмплер (temperature/top_k/top_p/min_p/penalties) + пресеты BD-016
  api/
    openai.rs      — POST /v1/chat/completions + GET /v1/models
    anthropic.rs   — POST /v1/messages
    responses.rs   — POST /v1/responses
  api.rs           — Router, auth middleware, SSE helper, generate_collect, parse_tool_calls
web/
  index.html       — статичный веб-чат (SSE, localStorage, thinking-парсер)
docs/
  brief/           — PROJECT-BRIEF, decisions (BD-001..BD-020), open-questions, research, interview-log
  engine-api.md    — внутренний контракт Engine trait + HTTP endpoints
  batch-integration.md — дизайн BatchedEngine
tests/
  api_test.rs      — 10 интеграционных тестов (MockEngine)
  stability_plan.md — план критерия BD-008
scripts/
  stability_smoke.sh — 4-client smoke для BD-008
  bench.ps1        — PowerShell бенчмарк (TTFT, decode, prefill, concurrent)
```

Engine выбирается по `QWEN36_SLOTS`: slots > 1 → BatchedEngine; slots == 1 → CandleEngine (single-slot, для smoke/дебага).

## Talks To [coverage: high — 9 sources]

- **candle-fork** (path-dep) — `qwen35-batch` крейт: `ModelWeights`, `Qwen35BatchAdapter`, `BatchScheduler`, `tokenizer` (build_chatml_text, encode_no_think, decode_text, strip_thinking). `candle-core` — Device/Tokenizer типы.
- **axum 0.8** — HTTP-сервер, SSE, Router, middleware
- **tokio** — async runtime, mpsc channels, spawn_blocking
- **tokenizers 0.22** — BPE-токенизатор (загружается из GGUF metadata)
- **Клиенты**: OpenAI SDK, Anthropic SDK, браузерный чат, curl — через LAN HTTP

## API Surface [coverage: high — 9 sources]

| Эндпоинт | Спека | Статус |
|---|---|---|
| `POST /v1/chat/completions` | OpenAI | полный: stream SSE, tools, usage |
| `POST /v1/responses` | OpenAI Responses | базовый: input→output, SSE, store=false (BD-012) |
| `POST /v1/messages` | Anthropic | полный: stream, tools, anthropic-version |
| `GET /v1/models` | OpenAI + ext | id/object/created/owned_by + контекст, квант, слоты, режимы (BD-015) |
| `GET /` | — | статика веб-чата |

Auth: `Authorization: Bearer $QWEN36_API_KEY` на всех `/v1/*`. Vision → 400 (BD-004).

## Data [coverage: high — 9 sources]

- GGUF-модель на диске (D: на yttri-win, путь из `QWEN36_MODEL`)
- Логи — stdout (BD-014)
- Чаты — localStorage браузера
- Persistent state в памяти: candle ModelWeights (одна загрузка), BatchScheduler + slots, KV cache / DeltaNet state (batched buffers в форке)
- Ничего на диске кроме модели (BD-014)

## Key Decisions [coverage: high — 9 sources]

Решения BD-001..BD-020 обязательны; отмена — только новой строкой «overrides BD-XXX». Полный лог — [../docs/brief/decisions.md](../docs/brief/decisions.md).

Ключевые:
- BD-001: candle-форк, не llama.cpp
- BD-007: 4 слота через BatchScheduler
- BD-008: критерий v1 — только стабильность
- BD-009: контекст 81 920
- BD-016: сэмплинг-пресеты из model card
- BD-017: sliding window (system сохраняется, `truncated: true`)
- BD-019: порядок работ — Q2_K_XL single-slot → API+чат → 4 слота → IQ2
- BD-020: главный риск — качество 2-bit, план отступления 3-bit/4-bit

## Gotchas [coverage: medium — 5 sources]

- `DECODE_BATCH_CAPACITY = 4` хардкод в форке → `QWEN36_SLOTS ≤ 4` (clamp с warning)
- Prefill неделим (PREFILL_CHUNK=usize::MAX) — длинный prompt одного клиента блокирует decode остальных
- Cancel mid-prefill невозможен (потолок форка)
- Модель не Send-friendly (CUDA/Metal contexts) — все вызовы сериализованы через один std::thread (dispatch loop), не tokio::spawn
- Q2_K_XL (11.8 GB) превышает VRAM 12 GB при 4 слотах — работает через системный RAM swap; IQ2_XXS (9.39 GB) освободит ~2.4 GB (фаза 5)
- Бенчмарк на Q4_K_M показывает низкие скорости из-за swap-bound режима (TTFT 2595 ms, decode 1.1 tok/s)

## Sources

- [README.md](../../README.md)
- [CLAUDE.md](../../CLAUDE.md)
- [AGENTS.md](../../AGENTS.md)
- [docs/brief/PROJECT-BRIEF.md](../../docs/brief/PROJECT-BRIEF.md)
- [docs/brief/decisions.md](../../docs/brief/decisions.md)
- [docs/brief/open-questions.md](../../docs/brief/open-questions.md)
- [docs/brief/research.md](../../docs/brief/research.md)
- [docs/brief/interview-log.md](../../docs/brief/interview-log.md)
- [Cargo.toml](../../Cargo.toml)
