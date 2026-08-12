# CLAUDE.md

## Питч в одну строку

Rust-сервер инференса Qwen3.6-27B (GGUF 2-bit) на собственном candle-форке: три API (OpenAI Chat Completions, OpenAI Responses, Anthropic Messages) + веб-чат, 4 слота через batch-планировщик, цель — Windows/RTX 3060 12 GB (yttri-win).

## Обязательный контекст (читать первым)

- [docs/brief/PROJECT-BRIEF.md](docs/brief/PROJECT-BRIEF.md) — видение, scope v1, критерий успеха, дорожная карта.
- [docs/brief/decisions.md](docs/brief/decisions.md) — **обязывающие** решения BD-001…BD-020. Отмена решения — только новой строкой «overrides BD-XXX».
- [docs/brief/open-questions.md](docs/brief/open-questions.md) — отложенное (IQ2, vision, MTP, VRAM-бюджет).

## Принципы

- Инференс — **только свой candle** (candle-fork-qwen35-batch, path-зависимость). llama.cpp — эталон для parity, не компонент (BD-001).
- Стек: Rust. Платформы: Windows+CUDA первично, macOS+Metal для разработки, CPU fallback (BD-011).
- Минимум кода: stdlib/уже установленные зависимости прежде новых; удаление лучше добавления.
- Критерий успеха v1 — стабильность 4 слотов × 8-16K генераций без падений и утечек VRAM (BD-008). Скорость и совместимость — вехи.
- Ничего лишнего на диске: логи в stdout, чаты в localStorage (BD-014).
- Windows WDDM: process private commit включает CUDA GPU allocations; paging проверять через GPU Process Memory Shared Usage, не PrivateMemorySize64. Details: [docs/lessons/2026-08-12-yttri-win-stability.md](docs/lessons/2026-08-12-yttri-win-stability.md).

## Ключевые параметры

- Модель: unsloth/Qwen3.6-27B-GGUF → старт UD-Q2_K_XL (11.8 GB); фаза 2 — IQ2 (BD-003).
- Контекст: 81 920 токенов (BD-009). Sliding window: system сохраняется, `truncated: true` (BD-017).
- Auth: один мастер-ключ из конфига/env (BD-005). LAN-only, HTTP (BD-006).
- Сэмплинг-пресеты из model card Qwen3.6 (BD-016).

## Пайплайн работы

`/create-spec <фича>` → `/create-spec-plan` → `/create-spec-implement` → `/create-spec-review`; трекинг — `/tasks`.
