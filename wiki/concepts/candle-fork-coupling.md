---
concept: Runtime Path-Dependency on yttri-forge
last_compiled: 2026-08-30
topics_connected: [project-overview, engine-layer, prefix-cache, testing]
status: active
---

# Runtime Path-Dependency on yttri-forge

## Pattern

Сервер не standalone на уровне инференса: `Cargo.toml` подключает `qwen35-batch`, `candle-core` и `candle-transformers` из соседнего `/Volumes/Askid Dev/Projects/yttri-forge/engine`. Файл сохранил историческое имя `candle-fork-coupling.md`, но актуальный runtime — `yttri-forge`; `candle-fork` в комментариях старых документов устарел.

## Instances

- `BatchedEngine` использует scheduler/model/tokenizer и experimental CUDA paths форка.
- Prefix cache требует согласованных `StateSnapshot::to_host/to_device`, включая int8 scales.
- FlashAttention/MSVC и cancellation changes могут затрагивать оба репозитория.

## What This Means

Коммит сервера без соответствующего engine commit может не собраться или работать иначе. Проверка/публикация должна фиксировать SHA обоих репозиториев; CUDA изменения проверяются на Windows/Linux GPU, а не только на macOS.

## Sources

- [project-overview](../topics/project-overview.md)
- [engine-layer](../topics/engine-layer.md)
- [prefix-cache](../topics/prefix-cache.md)
- [testing](../topics/testing.md)
