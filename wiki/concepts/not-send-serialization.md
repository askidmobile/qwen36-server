---
concept: Not-Send Model Serialization
last_compiled: 2026-08-08
topics_connected: [engine-layer, project-overview]
status: active
---

# Not-Send Model Serialization

## Pattern

candle `ModelWeights` / `Qwen35BatchAdapter` не Send-friendly: GPU contexts (CUDA/Metal) не thread-safe для произвольного доступа. Паттерн: вся работа с моделью сериализована через один выделенный поток/таск, HTTP-хендлеры общаются с ним только каналами.

## Instances

- **2026-08-07** in [engine-layer](../topics/engine-layer): CandleEngine — `Arc<Mutex<ModelState>>`, генерация в `spawn_blocking`. BatchedEngine — dispatch loop в `std::thread::spawn` (не `tokio::spawn`), channels `mpsc` для IngestMsg/StreamEvent.
- **2026-08-07** in [project-overview](../topics/project-overview): архитектура — HTTP tasks → `tx_ingest` → dispatch loop (один владелец scheduler'а) → per-slot `out` channels.

## What This Means

Это архитектурный потолок: параллелизм 4 слотов — НЕ 4 потока, а batched decode в одном потоке (scheduler делает один шаг = один batched GPU-вызов). Prefill одного слота блокирует decode остальных (PREFILL неделим). Отмена mid-prefill невозможна. Это цена за использование candle напрямую без переписки Metal-ядер (см. REAL_MODEL.md форка). Митигация — batch shrink после раннего EOS, batched state snapshot/restore. Для v1 (стабильность) приемлемо; для latency-sensitive сценариев потребовало бы чанк-чекпойнты GDN state.

## Sources

- [topics/engine-layer](../topics/engine-layer)
- [topics/project-overview](../topics/project-overview)
