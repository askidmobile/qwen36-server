---
topic: Prefix Cache
slug: prefix-cache
last_compiled: 2026-08-30
sources: 4
status: active
---

# Prefix Cache

## Purpose [coverage: high — 4 sources]

Prefix cache сохраняет полное recurrent/KV state префикса и позволяет следующему ходу диалога досчитать только хвост. На измеренной Ornith-сессии около 39K токенов второй ход сократился с десятков секунд до примерно четырёх без изменения ответа.

## Architecture [coverage: high — 4 sources]

- Снимок снимается **на границе последнего завершённого prefill chunk**, а не в конце prompt.
- Поиск идёт по chain hash полных 64-token блоков, затем обязательно сверяет реальные токены.
- Выбирается самый длинный сохранённый префикс; exact full match — miss, потому что primed path должен досчитать хотя бы один токен.
- LRU ограничен MiB-бюджетом; одинаковые token prefixes дедуплицируются до копирования.
- Snapshot переносится в system RAM на `put` и обратно на model device на `find`.

## Talks To [coverage: high — 4 sources]

- `BatchedEngine`: capture после chunk, `submit_primed` при hit, статистика cache.
- `StateSnapshot` в `yttri-forge`: DeltaNet state + attention KV, включая int8 pages и scales.
- Снимок содержит состояние target-модели, но пока не содержит отдельные KV/hidden MTP-головы; после cache hit спекуляция для этого запроса отключается.
- `PREFIX_CACHE_MIB`: ноль выключает cache.

## Data [coverage: high — 3 sources]

Entry хранит token vector, boundary key, host `StateSnapshot`, размер и LRU id. Hash не является доказательством равенства; token comparison защищает от collision.

## Key Decisions [coverage: high — 4 sources]

- Snapshot нельзя обрезать после снятия: DeltaNet state необратим по позиции.
- Host storage выбрана, чтобы cache не выталкивал paged KV из 12 GB VRAM.
- Int8 paged pool обязателен для полного 129K контекста на RTX 3060; snapshot сохраняет quant scales.
- Детерминированный CUDA A/B подтвердил одинаковый первый tool call при cache miss/hit отдельно для Q8 и F16; Q8 restore не является источником наблюдаемого цикла.

## Gotchas [coverage: high — 4 sources]

- Конец первого assistant prompt содержит generation suffix, который на следующем ходе заменяется ответом; snapshot в самом конце никогда не дал бы hit.
- Разный набор tools меняет начало chat template, поэтому cache не переиспользуется между такими сессиями.
- Перенос snapshot между устройствами обязан переносить int8 scales; текстовое merge старой реализации уже теряло их.
- После инъекции target snapshot нельзя продолжать старое MTP-состояние слота: до появления MTP snapshot формат помечает слот невыравненным и запрещает speculation.
- Тестовый harness без paged pool не доказывает Q8 snapshot path; нужен CUDA integration/боевой parity test.

## Sources

- [src/prefix_cache.rs](../../src/prefix_cache.rs)
- [src/engine_batched.rs](../../src/engine_batched.rs)
- [src/config.rs](../../src/config.rs)
- [MMVQ batch reuse spec](../../docs/specs/2026-08-29-mmvq-batch-reuse.md)
