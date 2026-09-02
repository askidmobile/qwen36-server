---
topic: Configuration and Profiles
slug: configuration-and-profiles
last_compiled: 2026-09-01
sources: 11
status: active
---

# Configuration and Profiles

## Purpose [coverage: high — 11 sources]

Конфигурация объединяет `.env`, чистые имена переменных и legacy `QWEN36_*`, validated profile manifests и model-family sampling presets. Она определяет модель, память, capabilities и server-owned sampling policy.

## Architecture [coverage: high — 11 sources]

- `Config::load()` сначала читает env-файл, затем process env; process env имеет приоритет.
- `get_env_var(NAME)` ищет `NAME`, затем `QWEN36_NAME`.
- Profile может задать text/vision/MTP artifacts и проверяемые hashes/ABI.
- Без profile `MODEL` остаётся legacy прямым путём к GGUF.
- Sampling: model-card preset → явные env overrides → client overrides только если lock выключен.
- `reasoning_effort` управляет chat template/thinking и не выбирает coding sampling preset.
- На yttri-win production-процесс запускается планировщиком через `scripts/windows/inference-run.bat`; `Start-Process` из SSH не является устойчивым launcher path.

## API Surface [coverage: high — 8 sources]

- Основные: `MODEL`, `MODELS_DIR`, `CTX`, `SLOTS`, `MTP`, `MTP_PATH`, `PREFIX_CACHE_MIB`, `CTX_OVERFLOW`.
- Sampling: `TEMPERATURE`, `TOP_P`, `TOP_K`, `MIN_P`, `PRESENCE_PENALTY`, `REPETITION_PENALTY`, `SAMPLING_LOCK`, `MODEL_PRESETS`.
- `GET /v1/models` публикует `sampling_family`, effective defaults/presets и `sampling_locked`.
- Admin endpoints меняют defaults/preset и persist их в тот же env-файл.

## Key Decisions [coverage: high — 5 sources]

- Presets привязаны к семейству модели, а не ко всему серверу.
- Server lock нужен, чтобы клиентский `temperature=0` не отменял рекомендованный agent profile.
- Lock — сознательная policy выделенного agent endpoint (BD-032), а не обычная OpenAI-compatible семантика; generic Ollama, LM Studio и vLLM используют client-wins request overrides.
- MTP выключатель и путь независимы: наличие `MTP_PATH` не должно включать MTP при `MTP=0`.

## Gotchas [coverage: high — 9 sources]

- Пустая env-переменная означает отсутствие override, а не ноль.
- Чистое имя имеет приоритет над legacy `QWEN36_*`; тесты должны очищать оба.
- `SAMPLING_LOCK=1` молча игнорирует request sampling fields; `/v1/models` обязан публиковать effective values и `sampling_locked`.
- Нельзя выводить sampling mode из `reasoning_effort`: `high` не означает precise coding.
- Profile metadata и live engine capability могут расходиться; `/v1/models` объединяет их с фактом `supports_*`.
- Серверная Windows-копия лежит глубже локальной, поэтому её корректный path-dependency на `yttri-forge` отличается на один `..`; механически переносить локальный `Cargo.toml` нельзя.

## Sources

- [src/config.rs](../../src/config.rs)
- [src/profile.rs](../../src/profile.rs)
- [src/main.rs](../../src/main.rs)
- [src/api/admin.rs](../../src/api/admin.rs)
- [src/api/openai.rs](../../src/api/openai.rs)
- [README.md](../../README.md)
- [.env.example](../../.env.example)
- [docs/brief/decisions.md](../../docs/brief/decisions.md)
- [Sampling servers research](../../docs/research/2026-08-30-sampling-ollama-lmstudio-vllm.md)
- [Sampling policy recommendation](../../docs/research/2026-08-30-sampling-policy-recommendation.md)
- [scripts/README.md](../../scripts/README.md)
