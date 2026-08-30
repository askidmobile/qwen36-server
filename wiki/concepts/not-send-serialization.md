---
concept: Single-Owner GPU State
last_compiled: 2026-08-30
topics_connected: [engine-layer, prefix-cache, testing]
status: active
---

# Single-Owner GPU State

## Pattern

Scheduler, ModelWeights и CUDA/Metal state принадлежат одному dispatch thread. HTTP handlers не вызывают модель напрямую: они отправляют request/cancel по каналам и получают `StreamEvent`.

## Instances

- Continuous batching означает несколько slots в одном GPU step, а не отдельный поток на клиента.
- Chunked prefill позволяет проверять cancellation и обслуживать runtime boundaries между chunk-ами.
- Prefix snapshot/restore выполняется тем же владельцем scheduler state.
- `SwappableEngine` меняет целый `Arc<dyn Engine>`, не переносит model state между потоками.

## What This Means

Так сохраняются порядок CUDA operations и целостность per-slot state. Latency зависит от размера prefill chunk и batching; async HTTP не превращает GPU model в thread-safe объект.

## Sources

- [engine-layer](../topics/engine-layer.md)
- [prefix-cache](../topics/prefix-cache.md)
- [testing](../topics/testing.md)
