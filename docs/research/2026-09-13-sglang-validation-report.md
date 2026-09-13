# SGLang concept validation report — yttri-win

Дата: 2026-09-13
Spec: [docs/specs/2026-09-13-sglang-serving-concepts-validation.md](../specs/2026-09-13-sglang-serving-concepts-validation.md)
Plan: [docs/plans/2026-09-13-sglang-concepts-validation.md](../plans/2026-09-13-sglang-concepts-validation.md)
Research: [docs/research/2026-09-13-sglang-concepts.md](2026-09-13-sglang-concepts.md)

## Итог

Проверены три теории на реальном yttri-win runtime:

1. Prefix cache даёт большой выигрыш на линейном продолжении prompt.
2. Плоский prefix cache **не покрывает раннее ветвление** prompt: divergent
   branch с общим префиксом 4003 токена получил cache miss, хотя repeat того же
   branch получил hit. Это главный подтверждённый аргумент за radix/branch-point
   cache.
3. Overlap scheduler имеет малый headroom на текущем single-slot decode:
   host sample ~1.4–2.2% от step wall, drain ~0.004–0.009 ms/step.
4. Существующий `PGRAPH` на Qwen3.5-9B не дал устойчивого выигрыша по prefill
   на 1K/4K/8K. Piecewise prefill graph как следующий приоритет не подтверждён.
5. Determinism при fixed seed и одинаковой форме батча стабилен; batch-shape
   чувствительность не проверена.

Практический вывод: сначала делать **branch-point/radix cache**, а не overlap
scheduler или новый prefill graph. Overlap и batch-invariant kernels отложить.
MTP/ReplaySSM не проверялись из-за отсутствия MTP-артефакта для тестовой модели.

## Environment

- Host: `yttri-win` (`Askid-PC`), Windows, RTX 3060 12 GB.
- Binary: `D:\Projects\yttri-inference\qwen36-server\target\release\yforge.exe`.
- Binary SHA-256: `D67FC97E3678B0D4071802D21124E967BB140D67226DE63DB5F5369A6D4EC302`.
- Binary mtime: `2026-09-05T19:38:09.3836541+03:00`.
- Model, использованная в тестах:
  `D:\Models\lmstudio-community\Qwen3.5-9B-GGUF\Qwen3.5-9B-Q4_K_M.gguf`.
- Test instance: отдельный scheduled task `yforge-sglang-probe`, порт `18100`,
  `--api-key probe-key`; production task `\qwen36-inference` не менялся.
- Production `.env` path: `D:\Projects\yttri-inference\qwen36-server\.env`.
- Production task после cleanup: `Ready`, процесс `yforge` отсутствует,
  VRAM `466 MiB / 12288 MiB`.

### Production drift

`qwen36-server\.env` указывает на
`D:\Models\empero-ai\Qwen3.8-9B-Distill-GGUF\Qwen3.8-9B-Distill-Q4_K_M.gguf`,
но этого файла на yttri-win нет. Задача `\qwen36-inference` находится в `Ready`,
но при запуске не сможет загрузить заявленную модель. Для validation использован
доступный Qwen3.5-9B Q4_K_M. Это отдельный production-config дефект, не
исправлялся в рамках исследования.

## Methods

Тесты шли через отдельную scheduled task, чтобы не нарушать production
launch path `scripts/README.md`. Тестовый env создавался отдельно и не содержал
production API keys. После каждого сценария инстанс останавливался, временный
task удалялся, VRAM проверялась через `nvidia-smi`.

Raw evidence сохранён локально в `/tmp/sglang-probe-evidence/`,
`/tmp/sglang-branch-probe.log`, `/tmp/sglang-*-probe*.json`, `/tmp/sglang-probe-server.log`
на машине Codex. Remote probe-каталоги удалены.

Все runtime-числа ниже — однократные probe-прогоны, кроме трёх повторов
determinism. Это достаточно для выбора следующего шага, но не заменяет
повторяемый benchmark серию.

### Exact commands

Создать отдельный probe env и launcher:

```powershell
# ad-hoc remote layout used for this validation
D:\Projects\yttri-inference\logs\sglang-probe-<timestamp>\probe.env
D:\Projects\yttri-inference\logs\sglang-probe-<timestamp>\run-probe.bat
```

Запустить изолированный инстанс:

```powershell
Register-ScheduledTask -TaskName yforge-sglang-probe -Action $action -Settings $settings
Start-ScheduledTask -TaskName yforge-sglang-probe
```

Прогнать сценарии:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File prefix_probe3.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File prefill_probe.ps1 -Tag pgraph-on
powershell -NoProfile -ExecutionPolicy Bypass -File prefill_probe.ps1 -Tag pgraph-off
powershell -NoProfile -ExecutionPolicy Bypass -File decode_probe.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File determinism_probe.ps1
```

Снять decode/VRAM bench:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File qwen36-server\scripts\bench.ps1 ^
  -BaseUrl "http://127.0.0.1:18100" -ApiKey "probe-key" -Model "qwen3.5-9b" ^
  -DecodeTokens 64 -Concurrent 1
```

Очистить probe:

```powershell
Stop-ScheduledTask -TaskName yforge-sglang-probe
Get-Process yforge -ErrorAction SilentlyContinue | Stop-Process -Force
Unregister-ScheduledTask -TaskName yforge-sglang-probe -Confirm:$false
Remove-Item -Recurse -Force D:\Projects\yttri-inference\logs\sglang-probe-<timestamp>
```

## Theory 1 — Prefix cache на линейном продолжении

### Setup

- `PREFIX_CACHE_MIB=8192`, `PGRAPH=on`.
- Prompt 1: system + long user (~6021 prompt token).
- Prompt 2: тот же prompt + assistant acknowledgement + user follow-up
  (~6036 prompt token), то есть линейное продолжение.
- `max_tokens=4`, non-stream.

### Result

| Run | TTFT wall | Prompt tokens | Cache event |
|---|---:|---:|---|
| miss | 4.566 s | 6021 | `snap saved: 5632 tok (138 MiB)` |
| hit-extended | 0.643 s | 6036 | `slot=0 primed: 5632 из 6036 токенов, досчитать 404` |
| hit-extended-repeat | 0.645 s | 6036 | `slot=0 primed: 5632 из 6036 токенов, досчитать 404` |

Снижение времени: `4.566 → 0.643 s`, около `86%`.

### Verdict

**Adopt / keep.** Текущий flat prefix cache уже даёт большой выигрыш на
линейном продолжении агентского диалога. Сам по себе этот тест не оправдывает
radix tree как замену: линейный extension уже работает.

## Theory 2 — Radix/branch-point advantage

### Setup

- `PREFIX_CACHE_MIB=8192`, `PGRAPH=on`.
- Shared prefix: ~4003 токена.
- Branch A: shared prefix + suffix A.
- Branch B: shared prefix + suffix B, divergence раньше сохранённого chunk
  boundary.
- Branch A repeat: тот же prompt, что A.

### Result

| Run | TTFT wall | Prompt tokens | Cache event |
|---|---:|---:|---|
| branch-A-first | 3.673 s | 4490 | `snap saved: 4096 tok (114 MiB)` |
| branch-B-divergent | 3.176 s | 4490 | `miss: prompt 4490 tok, longest common prefix 4003` |
| branch-A-repeat | 0.613 s | 4490 | `slot=0 primed: 4096 из 4490 токенов, досчитать 394` |

Branch B имеет общий префикс 4003 токена, но получает miss, потому что
сохранённый snapshot покрывает 4096 токенов и расходится с B до этой границы.
При этом repeat branch A получает hit и идёт за 0.613 s.

### Verdict

**Adopt as next runtime work.** Это подтверждает ценность branch-point/radix
cache: текущая схема хранит только последнюю границу чанка и не даёт
переиспользовать более раннюю общую точку ветвления. Ветвящаяся агентская
нагрузка (несколько agent branches от общего system/tool prefix) будет платить
полный prefill там, где radix tree мог бы вернуть почти-hit.

## Theory 3 — Overlap scheduler headroom

### Setup

- `HOST_TIMING=1`, `TRACE=1`, `PGRAPH=on`, single slot.
- Prompt ~2010 токенов, `max_tokens=128`.
- Wall time: 4.658 s.

### Result

Активные строки `[host]`:

```text
[host] steps=48 step=20.72ms sample=0.521ms/call x27 drain=0.005ms/step
[host] steps=48 step=10.76ms sample=0.537ms/call x21 drain=0.004ms/step
```

В другом прогоне:

```text
[host] steps=48 step=22.64ms sample=0.504ms/call x48 drain=0.009ms/step
[host] steps=48 step=22.76ms sample=0.492ms/call x48 drain=0.009ms/step
```

Если считать sample по числу вызовов на шаг: при 27 вызовах на 48 шагов это
~0.293 ms/step, при 22.64 ms step — около 1.3%. При 48 вызовах на 48 шагов —
~0.5 ms/step, около 2.2%. Drain — меньше 0.01 ms/step.

### Verdict

**Defer.** На текущем single-slot decode host processing не является узким
местом. Overlap scheduler может стать полезен на 4 слотах, с очередью и
богатой host-логикой, но как первый шаг SGLang-направления он не окупается.

## Theory 4 — Prefill graph (PGRAPH) A/B

### Setup

- Одинаковые prompts 1K/4K/8K, `max_tokens=1`, prefix cache off для A/B.
- Вариант A: `PGRAPH=on`; вариант B: `PGRAPH=off`.

### Result — prefill probe

| Prompt | PGRAPH on, tok/s | PGRAPH off, tok/s | Разница |
|---|---:|---:|---:|
| ~910 tok | 976.4 | 828.5 | +17.8% on |
| ~3810 tok | 1441.8 | 1541.9 | -6.5% on |
| ~7610 tok | 1418.1 | 1529.8 | -7.3% on |

### Result — bench.ps1 at ~2010 prompt tokens

| Variant | Bench decode parser | Prefill tok/s | 1-slot aggregate tok/s | VRAM after |
|---|---|---:|---:|---:|
| PGRAPH off | parser 0 (не использован как основной metric) | 1374.6 | 45.1 | 7004 MiB |
| PGRAPH on | parser 0 (не использован как основной metric) | 1317.3 | 46.4 | 7098 MiB |

Logs подтверждают, что PGRAPH path действительно активен:

```text
[pg] captured prefill T=512 slot=0 nodes=3067
[pg] chunk T=512 slot=0 pos=0 hit=1 lru=1 run=311.4ms
[pg] chunk T=512 slot=0 pos=512 hit=1 lru=1 run=312.7ms
```

### Verdict

**Defer as next priority on this workload.** В однократном A/B на Qwen3.5-9B
Q4_K_M при 1K–8K prefill `PGRAPH=on` не дал устойчивого выигрыша и был чуть
медленнее на 4K/8K. Это не отменяет PGRAPH-ценность для decode/VRAM на других моделях, но
не подтверждает необходимость piecewise prefill graph как следующего шага.
Повторить на целевой 27B IQ2_XXS можно отдельно, когда будет валидный MTP/model
profile.

## Theory 5 — Determinism

### Setup

- Fixed prompt, `seed=42`, `temperature=0.6`, `max_tokens=16`, три повтора.
- Одинаковая форма батча, single slot.

### Result

Все три прогона вернули одинаковый текст:

```text
Thinking Process:

1.  **Analyze the Request:**
    *
```

### Verdict

**Defer batch-invariant work.** В одинаковой форме батча output стабилен.
Чувствительность к разной форме батча не проверена, потому что тестовый
профиль был single-slot; для этого нужен multi-slot experiment.

## Theory 6 — MTP + graph / ReplaySSM

Не проверялась: у тестовой Qwen3.5-9B модели нет MTP-артефакта, production
MTP_PATH указывает на отсутствующий файл. Тест с MTP=0 был бы invalid.

### Verdict

**Deferred.** Вернуться после выбора MTP-профиля или 27B-модели с валидным
MTP artifact.

## Decision summary

| Theory | Verdict | Evidence |
|---|---|---|
| Prefix cache на линейном extension | Adopt/keep | 86% TTFT reduction, `[pcache] primed` |
| Radix/branch-point cache | Adopt as next | divergent branch miss при common prefix 4003, repeat hit |
| Overlap scheduler | Defer | host sample ~1.3–2.2% step wall |
| Piecewise prefill graph | Reject/defer as next | PGRAPH on не быстрее off на 4K/8K |
| Determinism/batch-invariant | Defer | same-shape stable, batch-shape не проверен |
| MTP/ReplaySSM | Deferred | нет MTP artifact |

## Follow-up implementation

Branch-point cache prototype реализован поверх существующего `StateSnapshot`:

- `src/prefix_cache.rs`: добавлен `put_many` для нескольких checkpoint'ов
  одного prompt'а.
- `src/engine_batched.rs`: сервер забирает все checkpoint'ы слота и кладёт их
  через `put_many`.
- `yttri-forge/engine/qwen35-batch/src/real/adapter.rs`: adapter хранит набор
  checkpoint'ов, снимая степени двойки до `PREFIX_CACHE_CHECKPOINT_MAX`
  (default `8192`) плюс финальную границу.
- Откат: `PREFIX_CACHE_CHECKPOINTS=0` возвращает прежнее поведение.

Unit verification: `cargo test --lib prefix_cache` — 11 passed, включая
`multi_boundary_put_hits_earlier_branch_point`.

Runtime verification на yttri-win, новая сборка
SHA-256 `C3BC0BC3D9EB9C4328ED0BDAFBAC0FCD5B7AA4EF4279532336A308B4A55515B8`:

| Run | До branch cache | После branch cache | Cache event |
|---|---:|---:|---|
| branch-A-first | 3.673 s | 3.962 s | `snapshots saved: 4/4`, checkpoint на 2048 |
| branch-B-divergent | 3.176 s (miss) | **1.989 s** | `primed: 2048 из 4490`, досчитать 2442 |
| branch-A-repeat | 0.613 s | 0.614 s | `primed: 4096 из 4490`, досчитать 394 |

Divergent branch перестал быть полным miss: ранний checkpoint на 2048 токенов
дал reuse 2048 из 4490 и сократил wall time с 3.176 до 1.989 s (`-37%`).
Первый прогон стал немного дороже из-за снятия четырёх checkpoint'ов; repeat
не изменился.

## Next steps

1. [x] Спроектировать branch-point cache поверх текущего `StateSnapshot`:
   checkpoint на нескольких границах, а не только на последней.
2. Повторить branch scenario на целевой 27B IQ2_XXS, когда model profile
   валиден.
3. Отдельно измерить multi-slot host/GPU headroom перед overlap scheduler.
4. Defer piecewise prefill graph до измерения на 27B с реальными 32K+ prefill.
5. Исправить production drift: `qwen36-server\.env` указывает на отсутствующий
   MODEL path.
