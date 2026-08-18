---
topic: Web Chat
slug: web-chat
last_compiled: 2026-08-08
sources: 2
status: active
---

# Web Chat

## Purpose [coverage: high — 2 sources]

Статичная HTML-страница от того же сервера (`GET /`). Веб-чат с Qwen3.6-27B: ввод API-ключа (localStorage), SSE-стрим, скорость генерации tok/s, прогресс-бар контекста, сворачиваемые thinking-блоки, переключатель thinking/instruct, полные настройки сэмплинга, история чатов в браузере.

## Architecture [coverage: high — 2 sources]

`web/index.html` — один файл, include_str! в `main.rs:17`. Ванильный JS, без сборки.

Компоненты:
- **header** — API key (password input), модель badge, режим (select: thinking/thinking-coding/instruct), прогресс-бар контекста, кнопка Стоп
- **sidebar** — список чатов (localStorage `q36_chats`), кнопка +
- **messages** — рендер с thinking-парсером (`<think>`/`<thinking>` маркеры → сворачиваемые `<details>`)
- **sampling panel** — `details` с grid: temperature, top_p, top_k, min_p, presence_penalty, repetition_penalty, max_tokens
- **composer** — textarea + кнопка Отправить; Enter → send, Shift+Enter → новая строка

Store (localStorage, BD-005/BD-014):
- `q36_api_key` — API ключ
- `q36_chats` — `[{id, title, messages: [{role, content, meta}]}]`
- `q36_current` — id активного чата

## Talks To [coverage: high — 2 sources]

- `GET /v1/models` — проверка ключа + метаданные (ctxLen, quant, slots)
- `POST /v1/chat/completions` — streaming SSE (`stream: true`, `stream_options.include_usage`)
- Bearer auth из localStorage

## API Surface [coverage: medium — 1 sources]

Внутренние функции (не экспортируется):
- `api(path, opts)` — fetch с Bearer auth; 401 → showKeyForm
- `send()` — отправка сообщения, SSE-парсинг (`data: ...\n\n`), инкрементальный рендер
- `parseThinking(text)` — деление потока на think/text части
- `applyPreset(name)` — пресеты BD-016 в UI поля
- `updateCtxBar()` — оценка chars/4 как proxy для токенов

## Data [coverage: high — 2 sources]

- localStorage: `q36_api_key`, `q36_chats`, `q36_current`
- Ничего не отправляется на сервер кроме API-запросов
- Чат-история только в браузере (BD-014)
- API ключ в localStorage (BD-005)

## Key Decisions [coverage: high — 2 sources]

- **BD-005**: API key в localStorage, форма ввода при 401
- **BD-014**: чаты в localStorage, ничего на сервере
- **BD-016**: пресеты сэмплинга из model card (thinking/thinking-coding/instruct) с полным набором параметров в UI
- **BD-009**: контекст 81920 (уточняется из /v1/models)
- Instruct режим: system `/no_think` (сервер добавляет no-think суффикс через токенизатор)
- Прогресс-бар: оценка chars/4 (грубая эвристика; точный подсчёт только на сервере)

## Gotchas [coverage: medium — 2 sources]

- Оценка токенов chars/4 — грубая эвристика для смешанного текста; точная только на сервере
- Thinking-парсер: fallback — если маркеров нет, но thinking-режим, ведущий блок >200 символов до `\n\n` считается рассуждением
- Метаданные (tok/s, токены) обновляются в реальном времени через re-render сообщения
- `truncated` флаг проверяется в usage, extra, и choices (несколько вариантов для совместимости)
- ctxLen уточняется из /v1/models (fallback 81920)
- Нет автопрокрутки при ручной прокрутке вверх (всегда scrollTop = scrollHeight)

## Sources

- [web/index.html](../../web/index.html)
- [docs/brief/PROJECT-BRIEF.md](../../docs/brief/PROJECT-BRIEF.md)
