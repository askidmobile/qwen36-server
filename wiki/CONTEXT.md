# Codebase Wiki — Navigation Guide

This project has a compiled knowledge wiki. Use it for orientation, then read live source for code changes and debugging.

## How to use this wiki

1. Начать с `INDEX.md`.
2. Для runtime-задач читать `engine-layer`, затем `configuration-and-profiles` или `prefix-cache`.
3. Для API/tool-loop/sampling — `http-api-layer` + `configuration-and-profiles`.
4. Для VRAM/CUDA validation — `testing` и concept `candle-fork-coupling` (название файла историческое; runtime теперь `yttri-forge`).
5. При расхождении docs и live code доверять `Cargo.toml`, исходникам и targeted tests; фиксировать documentation drift отдельно.

## When NOT to use the wiki alone

- Реализация или отладка конкретной функции.
- Проверка текущей конфигурации живого Windows-сервера.
- CUDA/MSVC/driver поведение и performance numbers.
- Точные внешние contracts, которые могли измениться.

## Stats

Compiled: 2026-08-30 | Topics: 8 | Sources: 51 | Auto-updates on session start

## Topics

- **project-overview** — фактический scope и архитектура.
- **engine-layer** — engine contract, scheduler, cancellation, MTP.
- **http-api-layer** — inference/admin/HF/media API, SSE и tools.
- **configuration-and-profiles** — env, profiles, presets и sampling lock.
- **prefix-cache** — state snapshots, host RAM, int8 KV и LRU.
- **multimodal-and-components** — media pipeline и component lifecycle.
- **web-chat** — Studio proxy и fallback UI.
- **testing** — CUDA gates, parity, stability и WDDM memory.

## Concepts

- **candle-fork-coupling** — историческое имя статьи о текущей path-зависимости на `yttri-forge`.
- **not-send-serialization** — один владелец GPU state и channel-based API.
