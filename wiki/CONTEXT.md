# Codebase Wiki — Navigation Guide

This project has a compiled knowledge wiki. Use it instead of scanning raw files.

## How to use this wiki

1. Start at INDEX.md — scan the topic table to find relevant modules
2. Read 1-3 topic articles relevant to your current task
3. Check coverage tags:
   - [coverage: high] — trust this section, skip raw files
   - [coverage: medium] — good overview, check raw sources for implementation details
   - [coverage: low] — read the raw source files listed in Sources
4. Check concepts/ for cross-cutting patterns (path-dependency on candle-fork, not-Send serialization)
5. Only read raw source files when you need code-level detail

## When NOT to use the wiki

- Writing new code (read the actual source files for exact syntax/types)
- Debugging a specific function (go to the file directly)
- The wiki article says [coverage: low] for what you need

## Stats

Compiled: 2026-08-08 | Topics: 5 | Sources: 25 | Auto-updates on session start

## Topics

- **project-overview** — цель, стек, статус, дорожная карта, решения BD-001..BD-020
- **engine-layer** — Engine trait, CandleEngine, BatchedEngine, контракт
- **http-api-layer** — axum, три API, auth, SSE, tool calls
- **web-chat** — статичный HTML-чат, localStorage, thinking-парсер
- **testing** — unit-тесты, API integration, stability smoke, bench

## Concepts

- **candle-fork-coupling** — path-зависимость на candle-fork, TODO-F1..F6
- **not-send-serialization** — модель не Send → один поток + channels
