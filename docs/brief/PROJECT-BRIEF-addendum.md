# PROJECT BRIEF — Дополнение (post-v1 эволюция)

Версия 1.0 · 2026-08-21 · Статус: фиксация факта

Дополняет [PROJECT-BRIEF.md](PROJECT-BRIEF.md), не отменяет его. Расхождения с BD-решениями оформлены строками «overrides BD-XXX» в [decisions.md](decisions.md) (BD-022…BD-027); здесь — только свод фактического состояния.

## 1. Что изменилось против брифа v1

Бриф v1 описывал text-only сервер одной модели Qwen3.6-27B (Q2_K_XL, 81 920 ctx, 4 слота). Проект перерос этот scope и стал мультимодельным мультимодальным сервером «Yttri Self-Inference Server».

| Область | Бриф v1 | Факт (2026-08) |
|---|---|---|
| Модели | Одна: Qwen3.6-27B Q2_K_XL | Семейства Qwen 3.5/3.6/3.8 (вкл. MoE), Ornith 1.0/1.5, Gemma 4; кванты Q4_K_M / Q6_K; горячая смена (SwappableEngine, admin API) |
| Модальность | Text-only, vision → 400 (BD-004) | Vision + Video через media-конвейер и внешний helper (overridden BD-022); text-only модели отклоняют media до inference |
| Декодинг | Обычный batch decode | MTP speculative decode B=1..4, token-for-token parity при fixed seed (BD-023) |
| Контекст | 81 920 (BD-009) | Конфигурируется, фактически 131 072; chunked long prefill (Gemma 4 lesson) |
| API | 3 API + /v1/models | + admin API (смена моделей, пресеты), HF-загрузки моделей, media uploads с TTL |
| Пресеты | 3 пресета Qwen3.6 (BD-016) | Пресеты per model family, заменяются при смене модели; пользовательские в MODEL_PRESETS |
| Хранение | Ничего кроме модели (BD-014) | + versioned artifacts и временные media (TTL 15 мин, гарантированное удаление) — BD-026 overrides storage scope |
| Компоненты | Монолит | Vision/MTP — lazy load, warm TTL 60 с, выгрузка при VRAM pressure (BD-027) |

## 2. Фактическая архитектура (src/)

- `engine.rs` — CandleEngine, single-slot (smoke/дебаг).
- `engine_batched.rs` — BatchedEngine, 4 слота через BatchScheduler форка; выбор по SLOTS и архитектуре GGUF (qwen35/qwen35moe).
- `engine_swap.rs` — SwappableEngine, hot-swap моделей без рестарта.
- `api/` — openai (Chat Completions), anthropic (Messages), responses (базовый), admin (модели/пресеты/VRAM), hf (загрузки), tools (function calling).
- `media/` — store (TTL-репер), fetch, prepare, helper-протокол; бинарь qwen36-media-helper.
- `profile.rs` — validated profiles с компонентами Vision/MTP (BD-022, BD-024).
- `component_manager.rs`, `vram_plan.rs`, `prefix_cache.rs` — управление VRAM и компонентами.

## 3. Актуальные риски (дополняют раздел 11 брифа)

1. **Тесная связка с candle-форком** — 3 path-зависимости, standalone-сборка невозможна (wiki: concepts/candle-fork-coupling.md). Смягчение: форк под тем же владельцем.
2. **Файлы-монолиты** — engine_batched.rs (~1150), openai.rs (~1050), engine.rs (~1000), admin.rs (~870). Рефакторить при касании, не превентивно.
3. **WDDM paging на Windows** — VRAM впритык; диагностика через GPU Process Memory Shared Usage (lessons/2026-08-12-yttri-win-stability.md).
4. **Критерий BD-008 не автоматизирован** — stability 4 слотов × 8-16K проверяется вручную (tests/stability_plan.md, scripts/stability_smoke.sh); интеграционного теста нет.
5. **Env-дублирование** — чистые имена + legacy QWEN36_* fallback. Убрать legacy после миграции окружений.

## 4. Актуальная дорожная карта (дополняет раздел 12)

Фазы 1–4 брифа выполнены и превышены. Дальше:

1. Автоматизация stability-критерия BD-008 (интеграционный тест 4 слота).
2. Закрытие OQ-1 (IQ2-ядра) — по-прежнему открыто, актуальность под вопросом после перехода на Q4_K_M+.
3. OQ-6 (полный Responses API) — по запросу клиентов.
4. Миграция env на чистые имена, удаление QWEN36_* fallback.
5. Перенос functionality в приложение Yttri — вне scope этого репо (BD-025).
