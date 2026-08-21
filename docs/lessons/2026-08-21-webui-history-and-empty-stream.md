# WebUI lost Gemma history and hid empty streams

**Date:** 2026-08-21

## Symptoms

- Follow-up requests behaved as if previous assistant messages did not exist.
- Some completed Gemma requests left a blank assistant bubble and looked stuck even though GPU was idle.

## Root causes

### History

WebUI persisted separate fields:

```text
content
thinkText
```

but built API history using only `{ role, content }`. Older Gemma responses could have `content=""` and the visible answer in `thinkText` due to previous channel-rendering bugs. Such responses were sent back as empty assistant messages.

### Silence

SSE parser ignored top-level `{ "error": ... }` events. It also accepted `finish_reason=stop` with zero content and rendered an empty bubble. Server logs showed `<turn|>` stop and GPU idle, so this was not an active prefill.

## Fix

- Assistant `thinkText` is sent as `reasoning_content` when normal content exists.
- Legacy reasoning-only assistant messages use `thinkText` as content, never empty assistant history.
- Top-level SSE error raises visible WebUI error.
- Zero-text completion renders explicit diagnostic:
  `модель завершила генерацию без текстового ответа: stop`.

## Chrome gates

Legacy history fixture:

```text
stored assistant: content="", thinkText="Кодовое слово: АЛЬФА"
real request: assistant.content="Кодовое слово: АЛЬФА"
Gemma response: "АЛЬФА"
```

Zero-token stream:

```text
prompt_tokens: 7,820
completion_tokens: 0
finish: stop
DOM: [модель завершила генерацию без текстового ответа: stop]
```

## Commit

- `a6d1768` preserve Gemma history; surface empty/error streams
