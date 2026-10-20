# DGX Spark: DeepSeek V4.1 Flash UNCENSORED VQ — параллельный Engram и честный A/B

Дата: 2026-10-20. Стенд: DGX Spark, NVIDIA GB10 (sm_121, 48 SM), 20 vCPU
(Cortex-X925/A725), 121 GiB unified memory, Ubuntu 24.04.1 aarch64,
драйвер 580.173.02, CUDA 13.0.88.

## Итог

Одна оптимизация дала основной результат: **параллельные pread для Engram**.
Prefill вырос в 2.6–3.9 раза на всех точках. Decode вырос умеренно.
Цель «в среднем 40 tok/s decode» **не достигнута** — измеренный потолок
26–37 tok/s зависит от контекста. Причина разобрана ниже и она не в Engram.

| Точка | prefill baseline | prefill optimized | × | decode baseline | decode optimized | × |
|---:|---:|---:|---:|---:|---:|---:|
| 2K | 140.3 | **401.9** | 2.86× | 32.80 | **36.81** | 1.12× |
| 8K | 154.9 | **512.5** | 3.31× | 29.59 | **28.57** | 0.97× |
| 32K | 155.6 | **583.0** | 3.75× | 32.23 | **27.60** | 0.86× |
| 64K | 154.6 | **593.5** | 3.84× | 25.89 | **32.27** | 1.25× |
| 128K | 116.7 (отчёт 2026-10-19) | **574.1** | 4.92× | 18.01 | **26.89** | 1.49× |
| 300K | 109.3 (медиана) | **506.7** | 4.64× | 16.01 | **25.32** | 1.58× |

Baseline снят на том же стенде в этот же день (`ds4-dfm-rs-ae40`, тот же
артефакт, та же конфигурация), поэтому сравнение честное. Baseline на 2K–64K
получился выше цифр отчёта от 2026-10-19 (140–155 prefill против 115–121) —
это разница между двумя прогонами одного бинарника, а не эффект оптимизации;
в таблице baseline взят из сегодняшнего прогона.

Стабильность: **20/20** запросов по 8K, медиана 525.5 prefill / 30.41 decode,
RSS 727–779 МиБ (VmHWM 5978 МиБ — пик при загрузке весов).

## Что именно было узким местом

`io_uring` в контейнере **запрещён seccomp-фильтром Docker**:
`io_uring_setup()` возвращает `EPERM`. Проверено напрямую
(`syscall 425 → errno 1`). Поэтому движок падал на запасной путь —
`engram rows: serial preads (no io_uring)`, — и читал строки таблицы по одной.

Измерения на этом железе (`/workspace/models/ds41-1017-unc/deepseek-engram`,
O_DIRECT, 288 случайных чтений по 4 КБ):

| Потоков | Время на 288 чтений | На чтение |
|---:|---:|---:|
| 1 | 29.56 мс | 102.7 µs |
| 4 | 6.44 мс | 22.4 µs |
| 8 | 4.60 мс | 16.0 µs |
| **16** | **3.65 мс** | **12.7 µs** |
| 24 | 3.86 мс | 13.4 µs |

Последовательная цепочка упирается в 10 тысяч чтений/с; 16 потоков дают
79 тысяч/с — **8.1×**. На декод-раунд приходится 96 строк × 2 плоскости =
192 чтения, на prefill-чанк — десятки тысяч.

## Реализованная оптимизация

`crates/ds4-core/src/engram.rs`: добавлен `EngramShard::read_spans(&[Span], threads)`
— scoped-потоки, каждый со своим 4 КБ-aligned bounce-буфером (контракт
O_DIRECT сохранён), запись сразу в pinned-слоты нативной стороны.

`crates/ds4-core/src/v41_run.rs`: `serial_round` теперь собирает список
`sгpan` (адрес, длина, назначение) и отдаёт его в `read_spans`; число потоков —
`DS41_ENGRAM_READ_THREADS` (по умолчанию 16, `0`/`1` возвращает старое
поведение).

Дополнительно: `DS41_ADMIT_X1024` — множитель admission-проверки
prefill'а (по умолчанию 2048 = 2×, как в движке). Это снимает ложные
отказы, когда после захвата decode-графа `MemAvailable` падает на
десятки мегабайт ниже двойного пика (см. раздел «Ограничения»).

Патч: `engram-parallel-pread.patch` (229 строк, 2 файла, +162/−20).

**Quality gate пройден**: при `temperature=0`, `seed=42` ответы базового и
оптимизированного бинарников **байт-в-байт идентичны**
(`quality-base.txt` / `quality-opt.txt`, sha256 первых 16 символов совпал:
`d2eab7f24403fd10`, длина 283).

## Почему decode не вырос до 40 tok/s

Профиль одного round'а (спекулятивный DSpark, `--dspark-verify`):

```
one round 55.5 ms = draft 7.9 + snapshot/rollback 0.0 + verify 47.6 + argmax 0.0
yield 1.78 token => 31.2 ms/token
```

Решающий эксперимент: verify при 2 строках и при 9 строках стоит
**одинаково** (47.6 мс против 47.5 мс). Значит стоимость не зависит от
числа верифицируемых токенов — это фиксированная стоимость прохода по
весам. 40 слоёв × ~1.15 мс = 46 мс (`--v41-prof`, per-layer ms:
`1.2 1.9 1.4 1.2 1.1 ... 1.3`), GPU utilization 49%.

Отсюда два следствия:

1. Стоимость forward'а определяется чтением весов, а не вычислениями.
   Ускорить можно только поднятием **acceptance rate** драфтера: при
   accept 1.78 токена за 55 мс получается 31 tok/s, для 40 tok/s нужно
   2.5+ токена за тот же раунд.
2. Engram в декоде больше не доминирует: `engram prep` упал с 21–37 мс до
   **3.0–3.9 мс** на раунд (в 6–10 раз).

Проверенные и **не сработавшие** рычаги decode (все — медианы 2K, 2–3 прогона):

| Гипотеза | Результат |
|---|---|
| `--dspark-verify 1` (ширина 1) | 28.1–31.7 tok/s, verify 46.4 мс |
| `--dspark-verify 2` | 29.8–30.7 tok/s, verify 59.5 мс |
| `--dspark-verify 3` | 31.6 tok/s, verify 67.0 мс |
| `--dspark-verify 5` (max) | 29.2 tok/s, verify 86.7 мс, accept 1.87 |
| `--dspark-verify 8` | отказ: «exceeds the artifact's draft block» (block=5) |
| `DS4_CUDA_MOE_GRAPHS=1` | 29.1–31.7 tok/s (шум) |
| `DS4_CUDA_LAYER_GRAPHS=0` | 31.1 tok/s (шум) |
| `DS4_CUDA_MMQ_MOE_MIN_TOKENS=1` | 30.8–31.7 tok/s |
| `DS4_CUDA_MMVQ_DECODE_MAX_TOKENS=8` | 27.2–27.9 tok/s |
| `DS4_CUDA_NO_MMVQ_DECODE=1` | 30.5 tok/s |
| `DS4_CUDA_WINDOW_ATTENTION=1` | 28.4 tok/s |
| `DS4_CUDA_ATTN_SPLIT_N=4` | 29.7 tok/s |
| `DS4_DSPARK_ADAPT_DEPTH=1 + ADAPT_GATE=1` | 28.5 tok/s |
| `DS4_V41_MKCACHE_SLOTS` 256→1024 | 31.8 tok/s (шум, accept не вырос) |
| `V41_GEMV_MAX_TOK` 8→1 (GEMM вместо GEMV) | **сломало forward**: «forward failed (pos0 918 n 2)» |
| `--max-seqs 2` (батчинг) | отказ: «live serving is serial; banks_unsupported» |
| `DS41_NO_DSPARK=1` (чистый decode) | **медленнее** спекуляции |

## Forge (yforge) против llama.cpp

Оба на одном файле `/workspace/models/ornith-1.5-9b/Ornith-1.5-9B-Q4_K_M.gguf`
(sha256 `70c11219…8fab6`), контекст 131072, full offload, flash-attn,
greedy, max_tokens=65, уникальный промпт на замер, 2 прогона.

**yforge** пересобран с `-C target-cpu=native -C opt-level=3`,
`lto=thin`, `codegen-units=1` (sha256 `21e9b094…03134`).
Конфиг: `SLOTS=4`, `KV_POOL_Q8=1`, `KV_CACHE_DTYPE=q8_f16`, `PGRAPH=on`,
`PGRAPH_MIN_T=1000000`, `PREFIX_CACHE_MIB=2048`, `PREFILL_CHUNK=8192`, `MTP=0`.

**llama.cpp** — свежий master `1b29621`, CUDA, `sm_121`, `--parallel 1`
(эквивалент одного слота), `--cache-reuse 0 --no-context-shift` (кэш
префикса выключен), `-b 8192 -ub 512`.

| Точка | yforge prefill | llama.cpp prefill | yforge decode | llama.cpp decode |
|---:|---:|---:|---:|---:|
| 2K | 1158.1 | **2233.6** | 36.11 | **38.14** |
| 8K | 1936.0 | **2888.8** | 35.64 | **37.36** |
| 32K | 1970.8 | **3398.7** | 34.10 | **35.50** |
| 64K | 1986.0 | **4719.2** | 28.35 | **32.91** |
| 128K | 1916.5 | **4419.4** | 18.68 | **28.81** |

**Вывод: llama.cpp быстрее yforge на этой модели и этом GGUF везде** —
по prefill в 1.9–2.4×, по decode в 1.05–1.54×, и разрыв растёт с
контекстом. Преимущества yforge здесь нет; заявлять его нельзя.

Замечание по методике: первый прогон llama.cpp был с `--parallel 4` и
дефолтным `cache-reuse`, из-за чего часть промпта попадала в кэш
(`cached_tokens` 2924…26587) и контекст на слот составлял лишь 32768.
Оба артефакта убраны; таблица выше — честный прогон. Даже с кэшем
(llama.cpp «нечестный» прогон: prefill до 4769 на 64K) картина та же.

## DeepSeek против ds4 reference

| Что | Prefill | Decode | Условия |
|---|---:|---:|---|
| Опубликованный ds4 reference | 775–1120 | 38 | code GR sidecar, другой артефакт |
| Наши измерения 2026-10-19 | 115–121 | 18–24 | UNCENSORED VQ, sidecar отсутствует |
| **Наши сегодня (optimized)** | **402–594** | **25–37** | UNCENSORED VQ, sidecar отсутствует |

Мы закрыли большую часть разрыва по **prefill**: 402–594 tok/s против
reference 775–1120 — остаётся 1.3–2.8×. Но сравнение не эквивалентно:
reference снят с code GR sidecar, которого у UNCENSORED VQ нет. По decode
мы в 1.0–1.5× от reference (25–37 против 38).

vLLM сравнить невозможно: совместимого по формату артефакта нет, а
llama.cpp **вообще не читает этот файл**:

```
gguf_init_from_reader: tensor 'blk.0.ffn_exps_vq.blob' of type 42 (q2_0)
has 2767968272 elements per row, not a multiple of block size (64)
llama_model_load: error loading model
```

Это подтверждает исходную предпосылку: для DeepSeek сравнивать можно
только с ds4 reference.

### Профиль Engram: время, I/O, доля prefill, повторяемость

| Величина | Baseline | Optimized |
|---|---:|---:|
| `engram prep` на decode-раунд | 21–37 мс | **3.0–3.9 мс** |
| Чтений на decode-раунд | 192 (96 строк × 2) последовательно | 192 параллельно в 16 потоков |
| Пропускная способность | 9.7 тыс. чтений/с | **78.9 тыс. чтений/с** |
| Доля prefill, уходящая в Engram | ~70 % (A/B: 96 против 410 tok/s) | оценочно ~15–20 % |
| Повторяемость | стабильна, 20/20 | стабильна, 20/20 |

Замер доли: A/B `--v41-no-engram` на том же промпте давал 410 tok/s
против 96 tok/s с Engram (baseline). После оптимизации prefill дошёл до
402–594 tok/s, то есть приблизился к «no-engram» потолку — значит Engram
перестал быть доминирующим.

## Что не сработало и почему

1. **io_uring включить нельзя.** `io_uring_setup` → `EPERM`; это seccomp
   профиль Docker (не `sysctl`: `kernel.io_uring_disabled=0` и так).
   `sysctl -w` тоже невозможен — `/proc/sys` read-only. Поэтому взят
   thread-pool путь; на хосте без seccomp io_uring должен быть ещё лучше
   (одна латентность диска на раунд вместо 16 потоков).
2. **Отключение Engram целиком** (`--v41-no-engram`) не рассматривается
   как решение: это меняет модель (Engram — часть архитектуры), поэтому
   только как диагностический A/B.
3. **Ширина верификации** не помогает: стоимость verify фиксирована
   (2 и 9 строк — одинаково), потому что упирается в чтение весов.
4. **GEMM вместо GEMV** для verify ломает forward (n=2): путь
   `V41_GEMV_MAX_TOK` завязан на форме тензоров, не только на пороге.
5. **Батчинг** (`--max-seqs 2`) в v41 недоступен: «live serving is serial;
   banks_unsupported».
6. **MoE-графы** (`DS4_CUDA_MOE_GRAPHS=1`) дают 29–32 tok/s — в пределах
   шума. Комментарий в коде называет их «крупнейшим оставшимся рычагом
   gen-tok/s», но на этом артефакте эффекта нет.
7. **MKCACHE_SLOTS 256→1024** не поднял acceptance (0.78/5 в обоих).
8. **nsys** внутри контейнера не видит CUDA-данные
   («does not contain CUDA kernel data»), поэтому kernel-level профиль
   получен через встроенный `--v41-prof` (per-layer ms) и
   `DS4_CUDA_*`-переключатели.

## Ограничения (честно)

- `nvidia-smi` на GB10 отдаёт `memory.used=N/A`; RSS (`VmRSS` 727–779 МиБ)
  и `MemAvailable` — **прокси**, а не точный GPU-footprint. Веса (105.75 GiB)
  резидентны в unified memory и в RSS процесса не видны.
- **Admission-проверка движка хрупкая**: `MemAvailable >= 2 × peak`.
  После захвата decode-графов свободно ~5.0–5.3 ГБ, а для 8K-промпта
  требуется 5320 МБ — не хватает десятков мегабайт, и запрос получает
  HTTP 500. Это **pre-existing** поведение (в baseline от 2026-10-19
  64K repeat 2 падал так же), не следствие патча. Для честных замеров
  использован `DS41_ADMIT_X1024=1024` (1×, как в самом движке для
  non-double-margin путей); на всех точках 2K–300K отказов нет. Для
  продакшена это надо решать иначе: резервировать decode-графы заранее
  либо считать admission по реальному peak, а не по 2×.
- Decode 26–37 tok/s — **не** «в среднем 40». Разрыв объяснён:
  память-ограниченный forward (49 % GPU util), а не Engram.
- Качество после VQ отдельно не проверялось: это performance-тест.
  Проверена только байтовая идентичность вывода до/после патча.
- 300K — два независимых холодных прогона (506.40 и 506.97 prefill;
  26.06 и 24.59 decode), медиана 506.7 / 25.32.
- Один и тот же бинарник на одном стенде в разное время даёт ±10 %
  (baseline 2K: 136.7 / 143.9 prefill в одном прогоне). Все таблицы —
  медианы 2–3 прогонов, но межпрогонный разброс остаётся.

## Артефакты

- `/workspace/forge-run/report-2026-10-20/` — этот отчёт, `bench/`
  (CSV + JSONL всех прогонов), `logs/` (серверные логи), патч.
- `/workspace/forge-run/bench/ds4-final2/` — финальные замеры DeepSeek.
- `/workspace/forge-run/bench/base-ae40/` — baseline.
- `/workspace/forge-run/bench/yforge-opt/`, `llamacpp-ornith-nocache/` — A/B Forge/llama.cpp.
- Патч: `engram-parallel-pread.patch`.
- Бинарник: `/workspace/build/ds4-dfm-rs-opt/ds4-server` (worktree от `ae40aa1d`).
- llama.cpp: `/workspace/build/llama.cpp/build/bin/llama-server`, master `1b29621`.
- yforge optimized: `/workspace/build/qwen36/target/release/yforge`, sha256 `21e9b094…03134`.

## Воспроизведение

```bash
# DeepSeek optimized
export DS41_ENGRAM_READ_THREADS=16
export DS41_ADMIT_X1024=1024
/workspace/build/ds4-dfm-rs-opt/ds4-server --cuda \
  -m /workspace/models/ds41-1017-unc/DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf \
  --engram-dir /workspace/models/ds41-1017-unc/deepseek-engram \
  --mem-budget-mb 110000 --mem-floor-gb 2 --prefix-reuse off \
  --mtp-mode auto --max-seqs 1 --host 127.0.0.1 --port 8000 --no-update-check

# Forge
/workspace/forge-run/yf.sh

# llama.cpp (fair: no prefix cache, one slot)
/workspace/forge-run/llama_nocache.sh
```
