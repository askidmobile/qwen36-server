---
topic: Testing and Validation
slug: testing
last_compiled: 2026-08-30
sources: 10
status: active
---

# Testing and Validation

## Purpose [coverage: high — 10 sources]

Проверка разделена на CPU/API unit tests, platform feature builds, CUDA parity/benchmark на yttri-win и ручной stability gate BD-008. GPU-оптимизации нельзя принимать только по локальному non-CUDA `cargo check`.

## Architecture [coverage: high — 10 sources]

- Unit tests внутри config/engine/sampler/prefix cache.
- API tests через `MockEngine` для auth, streaming, tools, reasoning, errors и metadata.
- Profile/component/media/multimodal suites отдельными integration files.
- Windows scripts собирают CUDA release и снимают TTFT/prefill/decode/concurrency/VRAM.
- Stability smoke запускает четыре длинных клиента и сравнивает VRAM между прогонами.

## Key Gates [coverage: high — 8 sources]

- Prefix snapshot parity: короткий и длинный prefill, DeltaNet + все attention layers.
- Q8 snapshots: CUDA путь с настоящим paged pool, включая quant scales.
- Prefix-cache A/B: при `temperature=0` отдельно сравниваются cache miss/hit на Q8 и F16; первый tool call должен совпасть побайтово внутри каждого режима.
- MTP: correctness/distribution отдельно от speed; greedy divergence диагностируется по verification logits.
- Tool parsing: JSON-похожий `bash.command`/`write.content` проверяется по OpenAI и Anthropic schema в non-stream и stream путях.
- Cancellation: разрыв соединения во время long prefill должен останавливать GPU примерно за keepalive/chunk latency.
- Memory: WDDM Shared Usage, а не process PrivateMemorySize.

## Gotchas [coverage: high — 9 sources]

- macOS mtimes после tar могут заставить Cargo ошибочно считать Windows artifacts свежими; после sync нужны корректные timestamps/clean target selection.
- `cargo check` без CUDA не инстанцирует nvcc/MSVC templates.
- Один зелёный тест без фактической ветки paged pool может быть ложным gate.
- Ответы при temperature > 0 нельзя сравнивать посимвольно как доказательство MTP parity.
- Q8 и F16 могут выбрать разные токены около численной ничьей; это не доказывает порчу Q8. Сравнивать нужно miss/hit внутри одного KV dtype.
- BD-008 всё ещё не полностью автоматизирован.

## Sources

- [tests/api_test.rs](../../tests/api_test.rs)
- [tests/component_manager_test.rs](../../tests/component_manager_test.rs)
- [tests/media_test.rs](../../tests/media_test.rs)
- [tests/multimodal_api_test.rs](../../tests/multimodal_api_test.rs)
- [tests/multimodal_engine_test.rs](../../tests/multimodal_engine_test.rs)
- [tests/profile_test.rs](../../tests/profile_test.rs)
- [tests/profile_switch_test.rs](../../tests/profile_switch_test.rs)
- [scripts/stability_smoke.sh](../../scripts/stability_smoke.sh)
- [scripts/bench.ps1](../../scripts/bench.ps1)
- [scripts/build_windows.bat](../../scripts/build_windows.bat)
