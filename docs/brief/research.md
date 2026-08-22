# Research notes — Qwen3.6 27B server

Дата: 2026-08-07. Источники: HF model card, файлы репозиториев пользователя, llama.cpp PR-ы.

## Модель Qwen3.6-27B (HF: unsloth/Qwen3.6-27B-GGUF, base Qwen/Qwen3.6-27B)

- Dense, 27B параметров, **с vision encoder** (Causal LM + Vision). GGUF unsloth — **text-only** (веса LLM; vision в GGUF не входит).
- Hidden 5120, 64 слоя, layout `16 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))`.
- Gated DeltaNet: V heads 48, QK heads 16, head dim 128. Gated Attention: Q heads 24, KV heads 4, head dim 256, RoPE dim 64.
- FFN intermediate 17408. Vocab 248320 (padded). MTP: обучена multi-step.
- Контекст: нативно 262 144, расширяемо до ~1 010 000. Рекомендация пользователя: старт с 81 920 (80K).
- Sampling-дефолты (model card):
  - Thinking general: temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0
  - Thinking precise coding: temperature=0.6, top_p=0.95, top_k=20
  - Instruct (non-thinking): temperature=0.7, top_p=0.80, top_k=20, presence_penalty=1.5

## Кванты в репо unsloth (выборка)

| Файл | Размер | Примечание |
|---|---|---|
| UD-IQ2_XXS | 9.39 GB | влезает в 12 GB VRAM целиком (впритык с KV) |
| UD-IQ2_M | 10.8 GB | лучше качество, VRAM впритык |
| UD-Q2_K_XL | 11.8 GB | веса ~на грани VRAM; Q2_K ядра уже есть в candle |
| UD-IQ3_XXS | 12.0 GB | уже не влезает с KV |
| Q4_K_M | 16.8 GB | референс качества |

Решение: старт с **Q2_K_XL** (быстрая валидация на готовых ядрах), затем **IQ2** (XXS/M) как фаза 2 — требуются новые dequant/matmul ядра (CPU + CUDA).

## Состояние candle-форков

### candle-fork (ветка feat/qwen35-batching)
- `qwen35-batch/` крейт: BatchScheduler (4 слота), real-model path (`src/real/`), time-multiplexing + настоящий batched DeltaNet decode (Metal + CUDA).
- Коммиты: Phase 1-6 batched decode, CPU fallback для forward_decode_batch, CUDA валидация на RTX 3060 (parity + quality + throughput, ×1.44 aggregate на Metal).
- `REAL_MODEL.md`: настоящий batched decode на текущем Candle невозможен без переписки Metal-ядер; реализован обход (общие веса + per-slot state snapshot ~114 MB).
- Q4K fast-path matmul: B=4 с fallback m=4.
- candle-core: `GgmlDType::{IQ2XXS, IQ2XS, IQ2S, Q2K, ...}` парсятся (mod.rs), `BlockQ2K` + `impl GgmlType for BlockQ2K` есть (k_quants.rs:112,748); **IQ2-блоков нет** в k_quants — только enum-парсинг.
- Дефолт QStorage: «CPU dequantization is not implemented» (mod.rs:532) — IQ2-дефолт падает; значит для IQ2 нужны блоки.

### candle-fork (master)
- Модели: quantized_qwen3, quantized_qwen3_moe, granitemoehybrid и др. **Нет** qwen35/DeltaNet/GatedDeltaNet в models/ — вся Qwen3.5-работа живёт в qwen35-batch форке.
- Q2_K и IQ-enum присутствуют в quantized/ аналогично.

### Qwen3.5 4b
- llama.cpp-эталон: скрипты bench (prefill pp512..8192, decode tg128..2048), memory, Claude judge, tool_calling через llama-server OpenAI API, отчёты. Полезен как референс методики тестов, не как код сервера.

## llama.cpp-референс (про Qwen3.6)
- llama.cpp уже поддерживает qwen35/qwen35moe (arch name `qwen35`); MTP draft-heads (`--spec-type draft-mtp`) — speedup 1.73x на 27B dense (RTX PRO 6000, Q8_0). EAGLE3 сторонние драфты тоже есть.
- Известные подводные камни загрузки GGUF (из PR #25334): `rope.dimension_sections` 3 vs 4 записи; `ssm_dt` без `.bias`; bundled vision (`v.*`) и MTP (`mtp.*`) тензоры надо терпеть при text-only загрузке; per-layer KV-head dims.
- Позиция пользователя: наши candle-наработки лучше llama.cpp → сервер строим на своём candle, llama.cpp используем только как эталон качества/скорости для проверки.

## Железо и развёртывание
- Целевая машина: yttri-win (192.168.2.89), Windows, RTX 3060 12 GB, работа на диске D: (C: переполнен).
- macOS (Metal) — машина разработки.
- Контекст 81 920: KV для 16 Gated-Attention слоёв оценка ~2.5-5 GB (f16); recurrent state DeltaNet ~4 GB/слот при 4 слотах (оценка, уточнить по конфигу).

## API-спецификации (по памяткам OpenAI/Anthropic)
- Chat Completions: POST /v1/chat/completions, stream SSE `data:`, tools/tool_calls, usage.
- Responses API: POST /v1/responses, input→output, SSE events (`response.created`, `response.output_text.delta`, ...), v1 — базовый уровень (store=false, без встроенных tools).
- Messages API: POST /v1/messages, заголовок `anthropic-version`, max_tokens обязателен, stream события `message_start`/`content_block_*`, tool_use блоки.
- /v1/models: OpenAI-список + расширения (контекст, квант, слоты, режимы).
