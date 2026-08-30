# qwen36-server Knowledge Base

Last compiled: 2026-08-30
Total topics: 8 | Total sources: 51

## Topics

| Topic | Also Known As | Sources | Last Updated | Status |
|---|---|---:|---|---|
| [topics/project-overview](topics/project-overview.md) | Yttri Self-Inference Server, overview | 8 | 2026-08-30 | active |
| [topics/engine-layer](topics/engine-layer.md) | Engine, BatchedEngine, SwappableEngine | 8 | 2026-08-30 | active |
| [topics/http-api-layer](topics/http-api-layer.md) | OpenAI, Anthropic, Responses, admin API | 11 | 2026-08-30 | active |
| [topics/configuration-and-profiles](topics/configuration-and-profiles.md) | env, profiles, presets, sampling lock | 10 | 2026-08-30 | active |
| [topics/prefix-cache](topics/prefix-cache.md) | prefix reuse, StateSnapshot, host cache | 4 | 2026-08-30 | active |
| [topics/multimodal-and-components](topics/multimodal-and-components.md) | vision, video, media, components | 10 | 2026-08-30 | active |
| [topics/web-chat](topics/web-chat.md) | Studio proxy, fallback WebUI | 4 | 2026-08-30 | active |
| [topics/testing](topics/testing.md) | CUDA gates, stability, parity, benchmark | 10 | 2026-08-30 | active |

## Concepts

| Concept | Connects | Last Updated |
|---|---|---|
| [concepts/candle-fork-coupling](concepts/candle-fork-coupling.md) | project-overview, engine-layer, prefix-cache, testing | 2026-08-30 |
| [concepts/not-send-serialization](concepts/not-send-serialization.md) | engine-layer, prefix-cache, testing | 2026-08-30 |

## Recent Changes

- 2026-08-30: Уточнён sampling contract (BD-032): env-wins для agent endpoint, reasoning отделён от sampling; добавлено исследование Ollama/LM Studio/vLLM.
- 2026-08-30: Полная перекомпиляция после перехода к `yttri-forge`, multi-model/multimodal runtime, server-owned sampling и host-backed prefix cache.
- 2026-08-30: Добавлены темы configuration/profiles, prefix cache и multimodal/components; старые утверждения о невозможной отмене prefill и обязательном sliding window удалены.
- 2026-08-08: Initial compile — 5 topics, 2 concepts.
