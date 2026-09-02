# qwen36-server Knowledge Base

Last compiled: 2026-09-01
Total topics: 8 | Total sources: 56

## Topics

| Topic | Also Known As | Sources | Last Updated | Status |
|---|---|---:|---|---|
| [topics/project-overview](topics/project-overview.md) | Yttri Self-Inference Server, overview | 9 | 2026-09-01 | active |
| [topics/engine-layer](topics/engine-layer.md) | Engine, BatchedEngine, SwappableEngine | 10 | 2026-09-01 | active |
| [topics/http-api-layer](topics/http-api-layer.md) | OpenAI, Anthropic, Responses, admin API | 11 | 2026-09-01 | active |
| [topics/configuration-and-profiles](topics/configuration-and-profiles.md) | env, profiles, presets, sampling lock | 11 | 2026-09-01 | active |
| [topics/prefix-cache](topics/prefix-cache.md) | prefix reuse, StateSnapshot, host cache | 6 | 2026-08-31 | active |
| [topics/multimodal-and-components](topics/multimodal-and-components.md) | vision, video, media, components | 10 | 2026-08-30 | active |
| [topics/web-chat](topics/web-chat.md) | Studio proxy, fallback WebUI | 4 | 2026-08-30 | active |
| [topics/testing](topics/testing.md) | CUDA gates, stability, parity, benchmark | 13 | 2026-09-01 | active |

## Concepts

| Concept | Connects | Last Updated |
|---|---|---|
| [concepts/candle-fork-coupling](concepts/candle-fork-coupling.md) | project-overview, engine-layer, prefix-cache, testing | 2026-09-01 |
| [concepts/not-send-serialization](concepts/not-send-serialization.md) | engine-layer, prefix-cache, testing | 2026-08-30 |

## Recent Changes

- 2026-09-01: Tool-call boundary сделан fail-closed: `length` и schema-invalid вызовы не исполняются; добавлен безопасный `id-quant` alias загруженной модели и регрессии OpenAI/Anthropic stream/non-stream.
- 2026-09-01: Добавлены production-runbook yttri-win и детерминированный long-context canary; уточнены launcher, live-model verification, WDDM и ограничения диагностического oracle.
- 2026-08-31: Задокументировано устранение двойного KV device snapshot, WDDM A/B на 40K/70K и раздельная проверка MTP eager/graph путей.
- 2026-08-30: Инкрементальная компиляция после `f1110e0` подтвердила отсутствие drift между live source и статьями wiki.
- 2026-08-30: Проверены Q8/F16 prefix-cache miss/hit; задокументированы schema-aware tool parsing, отсутствие MTP state в snapshot и очистка stale slot owner после отмены.
- 2026-08-30: Уточнён sampling contract (BD-032): env-wins для agent endpoint, reasoning отделён от sampling; добавлено исследование Ollama/LM Studio/vLLM.
- 2026-08-30: Полная перекомпиляция после перехода к `yttri-forge`, multi-model/multimodal runtime, server-owned sampling и host-backed prefix cache.
- 2026-08-30: Добавлены темы configuration/profiles, prefix cache и multimodal/components; старые утверждения о невозможной отмене prefill и обязательном sliding window удалены.
- 2026-08-08: Initial compile — 5 topics, 2 concepts.
