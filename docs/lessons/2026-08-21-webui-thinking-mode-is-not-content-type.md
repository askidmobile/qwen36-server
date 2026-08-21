# WebUI thinking mode is not content type

**Date:** 2026-08-21
**Source:** Chrome DevTools investigation of Gemma 4 reasoning-only bubbles

## What happened

Gemma 4 responses appeared only inside collapsed «Рассуждения» blocks. Curl/API smoke looked healthy, so server generation was incorrectly suspected.

Chrome DevTools Protocol reproduced the complete browser path and showed:

```text
SSE content = "Привет! Чем могу помочь?"
localStorage message.content = same text
localStorage message.thinkText = ""
localStorage message.thinking = true
DOM = <details class="think">...</details>
```

## Root cause

`web/index.html::fillBubble()` called `parseThinking(content, true)` whenever request used thinking mode and no separate reasoning existed. `parseThinking()` then classified all normal `content` as reasoning.

`message.thinking` describes request mode. It does not describe semantic type of `message.content`. Server already separates semantic channels:

- `reasoning_content` → `thinkText`
- `content` → answer

Gemma fallback correctly moved a thought-only answer into `content`, but renderer immediately moved it back into reasoning visually.

## Fix

Removed `assumeThinking` classification. `parseThinking()` now recognizes only explicit legacy `<think>`/`</think>` markers. Normal `content` always renders as answer.

Added runnable check:

```text
node tests/webui_render_test.mjs
```

## Browser gate

Chrome 151 via DevTools Protocol, live server and Gemma 4 Q4_K_M:

```text
HTTP 200
SSE content: "Привет! Чем могу помочь? 😊"
localStorage content: same text
DOM answerBlocks: 1
DOM reasoningBlocks: 0
console errors: 0
```

Separate thought+final generation rendered as one `<details class="think">` plus one ordinary answer block.

## References

- `web/index.html`
- `tests/webui_render_test.mjs`
- commit `8ab354e`
