---
topic: Testing and Validation
slug: testing
last_compiled: 2026-08-08
sources: 3
status: active
---

# Testing and Validation

## Purpose [coverage: high — 3 sources]

Тестирование: unit-тесты (lib + sampler + config + engine), интеграционные API-тесты (MockEngine), stability smoke (критерий v1 BD-008), бенчмарк производительности.

## Architecture [coverage: high — 3 sources]

**Unit-тесты** (в модулях, `cargo test`):
- `src/config.rs:tests` — `from_env_defaults_and_required_key` (env парсинг, обязательный API key)
- `src/engine.rs:tests` — default params, presets BD-016, trim_messages (no-op, drops pairs, keeps last), quant_from_filename
- `src/sampler.rs:tests` — greedy, top_k=1, min_p drops tail, presence_penalty, repetition_penalty, seeded RNG
- `src/engine_batched.rs:tests` — BatchConfig clamps slots, IndexedSampler per-slot params, fallback greedy

**Интеграционные API-тесты** (`tests/api_test.rs`):
- `MockEngine` реализует `trait Engine` (фиксированные deltas)
- 10 тестов через axum `ServiceExt::oneshot`: auth, models, chat non-stream/stream/tool_calls, image rejected, anthropic requires max_tokens, anthropic non-stream/stream, responses non-stream/stream

**Stability smoke** (`scripts/stability_smoke.sh`):
- 4 параллельных curl-клиента, stream, max_tokens 8K/12K/16K/16K
- Контроль процесса и VRAM каждые 30s
- Проверки: streams done, process alive, vram stable (delta ≤512 MiB), engine reuse

**Бенчмарк** (`scripts/bench.ps1`, PowerShell на yttri-win):
- TTFT (time to first token, stream)
- Decode tok/s (single-slot)
- Prefill tok/s (long prompt, short output)
- N-concurrent aggregate throughput (runspaces)
- VRAM (nvidia-smi)

## Talks To [coverage: high — 3 sources]

- `cargo test --features metal` (macOS) / `--features cuda` (Windows) — unit + integration
- `axum::body::to_bytes`, `tower::ServiceExt::oneshot` — API integration
- `curl -N` (no-buffer SSE) — stability smoke
- `nvidia-smi --query-gpu=memory.used` — VRAM контроль
- `ssh` (опционально) — удалённый контроль на yttri-win

## API Surface [coverage: medium — 1 sources]

- `MockEngine` (tests/api_test.rs:9) — test double для `trait Engine`
- `app(deltas: Vec<&str>) -> Router` — сборка тестового приложения
- `authed(req)` — добавление Bearer header
- `json_req(method, uri, body)` —构建 JSON request
- `body_string(resp)` — извлечение тела ответа

## Data [coverage: medium — 2 sources]

- Stability smoke артефакты: `out/stability-<ts>/` — `models.json`, `client-N.sse`, `client-N.err`, `vram.log`
- Тесты не пишут на диск (mock engine, in-memory)

## Key Decisions [coverage: high — 3 sources]

- **BD-008**: критерий v1 — только стабильность: 4 слота × 8-16K генераций без падений и утечек VRAM. Скорость и API-совместимость — вехи, не критерии.
- Stability smoke: 3 последовательных PASS-прогона (прогон 1 — после рестарта; прогоны 2-3 — на том же процессе; утечка между прогонами = V0(i+1) vs V1(i))
- VRAM tolerance: 512 MiB (допуск на фрагментацию аллокатора)
- Watchdog молчания слота: `QWEN36_REQ_TIMEOUT` 600s

## Gotchas [coverage: medium — 2 sources]

- Stability smoke: VRAM через nvidia-smi — общая по GPU (не per-process на WDDM); дельта — верхняя граница
- curl SSE: `-N` обязательно, иначе стрим склеится
- Windows-имя процесса: `qwen36-server.exe`; macOS/Linux — `qwen36-server`
- Bench на Q4_K_M (16.8 GB > VRAM 12 GB) — swap-bound, низкие скорости (TTFT 2595 ms, decode 1.1 tok/s); ожидается рост после IQ2-ядер (фаза 5)
- Тесты `from_env_defaults_and_required_key` гоняют env процесса — сериализованы в одном тесте
- `stability_smoke.sh` без `SSH_HOST` и без локального `nvidia-smi` → VRAM проверки SKIP

## Sources

- [tests/api_test.rs](../../tests/api_test.rs)
- [tests/stability_plan.md](../../tests/stability_plan.md)
- [scripts/stability_smoke.sh](../../scripts/stability_smoke.sh)
