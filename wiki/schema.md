# Wiki Schema — qwen36-server

Last compiled: 2026-08-08

## Topics

| Slug | Name | Description |
|---|---|---|
| project-overview | Project Overview | Общий обзор: цель, стек, статус, дорожная карта, ключевые решения BD-001..BD-020 |
| engine-layer | Engine Layer | Engine trait, CandleEngine (single-slot), BatchedEngine (4 слота), контракт docs/engine-api.md |
| http-api-layer | HTTP API Layer | axum router, три API (OpenAI/Anthropic/Responses), auth, SSE, tool calls |
| web-chat | Web Chat | Статичный HTML-чат: localStorage, SSE-стрим, thinking-парсер, пресеты, прогресс-бар |
| testing | Testing and Validation | Unit-тесты, API integration (MockEngine), stability smoke (BD-008), bench.ps1 |

## Concepts

| Slug | Name | Connects |
|---|---|---|
| candle-fork-coupling | Path-Dependency on candle-fork | project-overview, engine-layer, testing |
| not-send-serialization | Not-Send Model Serialization | engine-layer, project-overview |

## Evolution Log

- 2026-08-08: Initial schema generated from 5 topics, 2 concepts (first compile, codebase mode)
