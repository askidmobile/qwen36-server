---
topic: Testing and Validation
slug: testing
last_compiled: 2026-09-01
sources: 13
status: active
---

# Testing and Validation

## Purpose [coverage: high — 12 sources]

Проверка разделена на CPU/API unit tests, platform feature builds, CUDA parity/benchmark на yttri-win и ручной stability gate BD-008. GPU-оптимизации нельзя принимать только по локальному non-CUDA `cargo check`.

## Architecture [coverage: high — 12 sources]

- Unit tests внутри config/engine/sampler/prefix cache.
- API tests через `MockEngine` для auth, streaming, tools, reasoning, errors и metadata.
- Регрессии tool safety покрывают OpenAI и Anthropic в stream/non-stream: обрыв по `length`, отсутствие/`null` обязательного аргумента, несовместимый schema type, literal tool markup без объявленных tools и честный stop reason.
- Profile/component/media/multimodal suites отдельными integration files.
- Windows scripts собирают CUDA release и снимают TTFT/prefill/decode/concurrency/VRAM.
- Stability smoke запускает четыре длинных клиента и сравнивает VRAM между прогонами.
- `tools/kv_corruption_test.py` строит детерминированный многоходовый контекст с маркерами/ключами и отдельно считает полные, потерянные и частично искажённые канарейки.
- `scripts/README.md` фиксирует production launcher, свежий лог, binary path и обязательные post-start признаки модели/MTP.

## Key Gates [coverage: high — 10 sources]

- Prefix snapshot parity: короткий и длинный prefill, DeltaNet + все attention layers.
- Q8 snapshots: CUDA путь с настоящим paged pool, включая quant scales.
- Prefix-cache A/B: при `temperature=0` отдельно сравниваются cache miss/hit на Q8 и F16; первый tool call должен совпасть побайтово внутри каждого режима.
- MTP: correctness/distribution отдельно от speed; greedy divergence диагностируется по verification logits.
- Tool parsing: JSON-похожий `bash.command`/`write.content` проверяется по OpenAI и Anthropic schema в non-stream и stream путях.
- Model identity: display alias `current-id-current-quant` принимается без switch, а действительно другая модель по-прежнему получает `model_not_loaded`.
- Оборванный tool call на лимите токенов обязан остаться text/content и сохранить `length` (`max_tokens` в Anthropic), без исполняемого `tool_calls`/`tool_use`.
- Cancellation: разрыв соединения во время long prefill должен останавливать GPU примерно за keepalive/chunk latency.
- Memory: WDDM Shared Usage, а не process PrivateMemorySize.
- Memory A/B снимается в трёх точках: после warmup, на miss и на повторном prefix hit; для длинного запроса нужен частый sampler, чтобы не пропустить краткий shared spill.
- MTP graph A/B обязан отдельно фиксировать `usage.mtp`, fallback category и KV growth: `MTP_GRAPH=0` не выключает MTP, а переводит draft в eager-CUDA.
- Long-context canary должен запускаться сериями с фиксированными seed/runtime settings; он проверяет сохранность фактов, но сам по себе не локализует parser, sampler, KV, MTP или agent-history слой.

## Gotchas [coverage: high — 11 sources]

- macOS mtimes после tar могут заставить Cargo ошибочно считать Windows artifacts свежими; после sync нужны корректные timestamps/clean target selection.
- `cargo check` без CUDA не инстанцирует nvcc/MSVC templates.
- Full API suite требует согласованного checkout соседнего `yttri-forge`: cfg/backend drift может остановить сборку зависимости до компиляции тестов сервера. Изолированный `rustc --test src/api/tools.rs` проверяет parser, но не заменяет полный API/CUDA gate.
- Один зелёный тест без фактической ветки paged pool может быть ложным gate.
- Ответы при temperature > 0 нельзя сравнивать посимвольно как доказательство MTP parity.
- Q8 и F16 могут выбрать разные токены около численной ничьей; это не доказывает порчу Q8. Сравнивать нужно miss/hit внутри одного KV dtype.
- Совпадение текстового дефекта с WDDM paging — сильная корреляция, но не доказательство битовой порчи при копировании: требуются повтор той же агентской нагрузки без shared spill и, при необходимости, layer/logit parity.
- BD-008 всё ещё не полностью автоматизирован.
- `kv_corruption_test.py` не отправляет поле `model` и при `SAMPLING_LOCK=1` запрошенный `temperature` может не быть effective value; live sampling нужно читать из `/v1/models` и серверного лога.
- Тест записывает JSON в текущую директорию: задавать явный `--out` внутри разрешённого workspace и не смешивать результаты разных runtime-профилей.

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
- [yttri-forge mtp.rs](../../../yttri-forge/engine/qwen35-batch/src/real/mtp.rs)
- [scripts/README.md](../../scripts/README.md)
- [tools/kv_corruption_test.py](../../tools/kv_corruption_test.py)
