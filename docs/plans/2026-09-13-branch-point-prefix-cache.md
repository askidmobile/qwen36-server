# Plan: Branch-point prefix cache

**Date:** 2026-09-13
**Status:** ✅ Done (verified on yttri-win)
**Priority:** P1
**Source:** [SGLang validation report](../research/2026-09-13-sglang-validation-report.md)

## Goal

Убрать подтверждённый промах divergent branch в prefix cache: сохранять не
только последнюю границу prefill-чанка, но и ранние checkpoint'ы, чтобы ветка
с общим system/tool-префиксом могла попасть в более раннюю согласованную точку.

## Current state

Validation report показал:

- linear extension: hit 4096/5632 токенов, TTFT `4.566 → 0.643 s`;
- divergent branch: common prefix 4003 токена, но cache miss, потому что
  хранился только snapshot на 4096;
- repeat той же ветки: hit и TTFT 0.613 s.

Код до изменения:

- `src/prefix_cache.rs` — один `put(tokens, snap)` на запись;
- `src/engine_batched.rs` — забирал `take_prefix_snapshot` и клал один snapshot;
- `yttri-forge/.../adapter.rs` — `slot_prefix_snaps: Vec<Option<(usize, StateSnapshot)>>`.

## Solution

### PrefixCache

- Добавлен `put_many(tokens, snapshots)`.
- Каждый checkpoint кладётся обычным `put` на срезе prompt до своей позиции.
- `find` уже выбирает самый длинный валидный prefix, поэтому ранний branch
  hit работает без изменения алгоритма поиска.

### Adapter

- `slot_prefix_snaps` заменён на `Vec<Vec<(usize, StateSnapshot)>>`.
- `take_prefix_snapshots()` возвращает все checkpoint'ы слота.
- Дополнительные checkpoint'ы снимаются в степенях двойки до
  `PREFIX_CACHE_CHECKPOINT_MAX` (default `8192`).
- Финальная граница чанка сохраняется как раньше.
- `PREFIX_CACHE_CHECKPOINTS=0` возвращает прежнее поведение.

### Server

- `engine_batched` использует `take_prefix_snapshots` + `put_many`.
- Лог: `[pcache] snapshots saved: N/M, entries ..., bytes ... MiB`.

## Verification

- Unit: `cargo test --lib prefix_cache` — 11 passed, включая
  `multi_boundary_put_hits_earlier_branch_point`.
- Runtime smoke на yttri-win: release build
  SHA-256 `C3BC0BC3D9EB9C4328ED0BDAFBAC0FCD5B7AA4EF4279532336A308B4A55515B8`.
- Divergent branch: `3.176 s → 1.989 s`, `primed: 2048 из 4490`,
  `[pcache] snapshots saved: 4/4`.
- Repeat branch: `0.614 s`, без регрессии против прежних `0.613 s`.

## Design trade-off

Checkpoint'ы стоят host RAM. Степени двойки до 8192 дают ограниченный набор
ранних точек без роста числа snapshots для длинных prompt'ов. Для 6K prompt
будет 5 snapshot'ов вместо 1; для 128K — примерно `log2(8192/512) + 1`, а не
256. Это осознанный компромисс: сначала закрыть branch-point промах на
system/tool-префиксах, а не строить полный radix tree.

## Plan decisions

| # | Question | Decision | Date |
|---|----------|----------|------|
| PD-001 | Хранить checkpoint на каждой границе чанка? | Нет; только степени двойки до `PREFIX_CACHE_CHECKPOINT_MAX` плюс финальная граница | 2026-09-13 |
| PD-002 | Можно ли откатить? | Да, `PREFIX_CACHE_CHECKPOINTS=0` | 2026-09-13 |
| PD-003 | Менять формат `StateSnapshot`? | Нет; multi-snapshot реализован поверх существующего формата | 2026-09-13 |

## Risks

| Risk | Mitigation |
|------|------------|
| Рост host RAM на длинных prompt'ах | Ограничить checkpoint'ы степенями двойки и документировать env |
| Дубликаты позиций | Проверка `already_captured` перед snapshot |
| Регрессия linear hit | Финальная граница по-прежнему сохраняется |
