# yttri-win stability and memory interpretation

**Date:** 2026-08-12  
**Source:** long CUDA stability run on RTX 3060 12 GB

## What happened

Task Manager showed roughly 11 GB private memory while running Qwen3.6-35B-A3B IQ2_XXS. This looked like model weights had paged into system RAM. A four-slot 8K stability run was also needed to verify state isolation and memory stability.

## Root cause

Windows WDDM accounts CUDA GPU allocations in process private commit. `PrivateMemorySize64` therefore is not physical system-RAM residency. Relevant counters are `GPU Process Memory/Dedicated Usage`, `Shared Usage`, and process working set.

`QWEN36_TRACE=0` was also interpreted as enabled because code checked environment-variable presence instead of a truthy value. Per-token trace made logs noisy and added avoidable overhead.

Long generation exposed context-boundary and API-edge issues: requested output was not clamped after exact prompt tokenization, `max_tokens=0` reached engine as HTTP 500, and `truncated` was true even when no history message was removed.

## Fix

- Treat only `1`, `true`, `yes`, and `on` as enabled trace values.
- Clamp output after exact tokenization to `context_length - prompt_tokens`.
- Reject zero output limits with HTTP 400 in all three APIs.
- Set `truncated=true` only after actual history removal.
- Drop disconnected queued requests through `mpsc::Sender::is_closed()` before GPU work.
- Keep prefix cache rejected until snapshot/logits restore has parity tests.

## Validation

Qwen3.6-35B-A3B IQ2_XXS, `ctx=8192`, `slots=4`:

- Four concurrent identical requests completed `8140` tokens each (`52 + 8140 = 8192`).
- Four outputs were bit-exact by content hash.
- Four SSE streams had terminal `finish_reason=length` and `[DONE]`.
- Zero malformed events, duplicate generated lines, or client errors.
- Elapsed: 1525 seconds.
- GPU shared system RAM stayed at 78 MiB.
- Server answered post-run smoke in about 1.4 seconds.
- Disconnected queued request produced zero output and was not admitted after slots freed.

## References

- `src/engine.rs`
- `src/engine_batched.rs`
- `src/api/openai.rs`
- `src/api/responses.rs`
- `src/api/anthropic.rs`
- `scripts/stability_smoke.sh`
- `tests/stability_plan.md`
- candle fork `qwen35-batch/src/scheduler.rs`

## Сжатие KV-кеша против выхода за VRAM (2026-08-29)

Симптом со стороны пользователя: «модель не отвечает вообще ничего». На деле
сервер работал, но префил 65 тысяч токенов занимал две минуты вместо сорока
секунд, и клиент отваливался по таймауту.

Причина — выход за видеопамять. При ctx 129728 и F16-пуле:

```
[kv] paged pool: window=125632 pool=3926MB free=4439MB  WARN: окно меньше контекста
VRAM процесса: 11602 МиБ, подкачка: 846 МиБ
```

Пул брал всю свободную VRAM минус 512 МиБ резерва, но после него выделяются
графы, голова MTP (834 МБ) и активации префила. WDDM не падает — он молча
уводит лишнее в системную память, и префил замедляется втрое.

Лечение — `QWEN36_KV_POOL_Q8=1`: int8-пул, вдвое меньше байт на токен.

| | F16 | int8 |
|---|---|---|
| пул | 3926 МБ | 2042 МБ |
| окно | 125 632, с вытеснением | 129 728, весь контекст |
| подкачка | 846 МиБ | 78 МиБ (постоянные буферы драйвера, не растут) |
| префил | 515 ток/с | 656 ток/с |
| пик под нагрузкой | — | 10 271 МиБ из 12 288 |

Точность int8 на KV — 0.75% round-trip, на ответах не сказалась.

Резерв под пул стал настраиваемым (`QWEN36_VRAM_HEADROOM_MIB`, умолчание 1024
вместо 512) — он важен там, где окно упирается в видеопамять, а не в контекст.

**Чего не делать на живом сервере:** `QWEN36_PREFILL_CHUNK=4096` раздувает
активации префила и уводит процесс за 12 ГБ (наблюдалось 14 ГБ суммарно с
подкачкой) при приросте скорости в 4%. Рабочее значение — 1024; 2048 даёт
680 ток/с против 656 и требует отдельной проверки по памяти.
