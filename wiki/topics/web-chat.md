---
topic: Web UI and Studio Proxy
slug: web-chat
last_compiled: 2026-08-30
sources: 4
status: active
---

# Web UI and Studio Proxy

## Purpose [coverage: high — 4 sources]

Основной UI теперь Unsloth Studio за reverse proxy. Встроенный `web/index.html` — legacy/admin fallback для `GET /`, когда Studio недоступен.

## Architecture [coverage: high — 4 sources]

- `/v1/*` обслуживает Rust router.
- Остальные маршруты проксируются на `STUDIO_URL`.
- При connection error proxy возвращает встроенный HTML только для `GET /`; прочим маршрутам — 502.
- Fallback WebUI хранит ключ/чаты в localStorage и использует Chat Completions SSE.

## Gotchas [coverage: high — 4 sources]

- WebUI не определяет server sampling policy; effective значения надо читать из `/v1/models`.
- Удаление fallback отменило бы BD-025 и требует отдельного решения.
- Proxy копирует headers кроме Host и сам не применяет inference auth к Studio routes.

## Sources

- [src/api/proxy.rs](../../src/api/proxy.rs)
- [web/index.html](../../web/index.html)
- [README.md](../../README.md)
- [tests/webui_render_test.mjs](../../tests/webui_render_test.mjs)
