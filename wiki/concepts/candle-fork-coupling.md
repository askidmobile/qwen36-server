---
concept: Path-Dependency on candle-fork
last_compiled: 2026-08-08
topics_connected: [project-overview, engine-layer, testing]
status: active
---

# Path-Dependency on candle-fork

## Pattern

`qwen36-server` — не standalone-крейт в смысле зависимостей: он path-dependent на локальный candle-fork-qwen35-batch (`/Volumes/Askid Dev/Projects/candle-fork-qwen35-batch`). Это проявляется в:
- `Cargo.toml` path-deps на `qwen35-batch` и `candle-core` из форка
- Переиспользование `ModelWeights`, `BatchScheduler`, `Qwen35BatchAdapter`, `tokenizer` напрямую из форка
- Протокол интеграции (docs/batch-integration.md) описывает TODO-F1..F6 — точки расширения, требующие мини-патчи в форке

## Instances

- **2026-08-07** in [project-overview](../topics/project-overview): BD-010 — код сервера standalone крейт с path-зависимостью на форк. Выбор против «в воркспейсе форка» и «отдельный репо».
- **2026-08-07** in [engine-layer](../topics/engine-layer): Engine-слой импортирует `qwen35_batch::real::{ModelWeights, Qwen35BatchAdapter, tokenizer}`, `qwen35_batch::scheduler::BatchScheduler`. TODO-F5 (indexed sampler) и TODO-F6 (slots_mut) — патчи в форк.
- **2026-08-07** in [testing](../topics/testing): unit-тесты требуют features `metal`/`cuda` из форка; bench на yttri-win зависит от CUDA-сборки форка.

## What This Means

Сервер и форк — единый организм: невозможно собрать/протестировать сервер без доступного форка на диске. Мини-патчи форка (TODO-F5/F6) — точка координации: пока их нет, BatchedEngine работает на greedy fallback (приемлемо для прогона стабильности, но пресеты BD-016 не применяются). Это сознательный выбор (BD-010): сохранить parity-механику форка, не дублировать scheduler в сервере. Риск: эволюция форка может ломать сервер — контроль через parity-тесты форка (real_qwen35_batch.rs).

## Sources

- [topics/project-overview](../topics/project-overview)
- [topics/engine-layer](../topics/engine-layer)
- [topics/testing](../topics/testing)
