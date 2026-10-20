# DeepSeek V4.1 Flash UNCENSORED VQ: 40+ tok/s decode на GB10

Дата: 2026-10-20 (продолжение отчёта
[2026-10-20-dgx-spark-deepseek-engram-parallel-prefill.md](2026-10-20-dgx-spark-deepseek-engram-parallel-prefill.md)).

## Итог: цель достигнута

**Среднее 46.1 tok/s decode** на реальном контенте (5 кейсов × 3 прогона,
медианы), при цели «в среднем 40»:

| Кейс | Промпт | Decode t/s | Accept |
|---|---:|---:|---:|
| prose 2K | 2040 | **45.47** | 3.57 |
| code 2K | 1875 | **46.71** | 3.57 |
| table 2K | 3772 | **43.08** | 3.57 |
| prose 8K | 7556 | **44.79** | 3.57 |
| code 8K | 6909 | **50.67** | 3.57 |
| **Среднее** | | **46.14** | |

Минимум по всем точкам — 42.3 tok/s, то есть **все** замеры выше 40.

Конфигурация-победитель:

```bash
DS4_DSPARK_QUENCH=0 DS41_ENGRAM_READ_THREADS=16 DS41_ADMIT_X1024=1024 \
ds4-server --cuda -m <vq8sh14-q4k-mtpnative>.gguf \
  --engram-dir deepseek-engram --mem-budget-mb 110000 --mem-floor-gb 2 \
  --prefix-reuse off --mtp-mode on --max-seqs 1 \
  --host 0.0.0.0 --port 8000 --no-update-check
```

Quality gate: greedy-вывод **байт-в-байт идентичен** конфигурации по умолчанию
(sha256 первых 16 символов `dd19802523057162`, длина 481). Спекуляция lossless
by construction: коммитит только verify-forward.

## Что дало прирост: два флага, а не ядра

Разбор начался с арифметики. Профиль decode по слоям (`--v41-prof`):

```
38 обычных слоёв × 0.90 мс + engram L1 (1.60) + L14 (1.80) + head (1.6) = 39.3 мс
```

Вне слоёв времени нет (`outside = -0.4 мс` — слои перекрываются с head).
Байты на токен, посчитанные по инвентарю GGUF: attention 2.65 ГиБ + shared
expert 0.74 + routed 6 из 384 экспертов 1.61 + голова 0.36 = **5.35 ГиБ**.
При 37.3 мс это 143 ГиБ/с.

Измеренный потолок GB10 (собственный бенчмарк, `bw2.cu`/`bw3.cu`):

| Паттерн | GB/s |
|---|---:|
| dense streaming, 40 GiB | **240.7** |
| random expert-sized spans (6 из 384) | **229.7** |
| dense streaming, 6 GiB (малый набор) | 162.2 |

То есть ядра работают на 60 % потолка — резерв в них есть. Я проверил это
экспериментально: **предзагрузка блока вперёд в q4_K GEMV**
(`v41_q4k_rown_smem`, следующий 144-байтовый блок в регистры, порядок
накопления не меняется → бит-в-бит). Патч собрался и прогнал **26.41 t/s
против 26.85 без него** — то есть нулевой эффект (регистровое давление съело
выигрыш). Откатил.

Прирост пришёл не из ядер:

| Шаг | Decode t/s | Accept | Step |
|---|---:|---:|---:|
| baseline `--mtp-mode auto` | 28.87 | 1.08 | 69.5 мс |
| `--mtp-mode on` | 31.93 | 2.61 | 105.5 мс |
| **`on` + `DS4_DSPARK_QUENCH=0`** | **41.28** | **2.27** | 75.5 мс |
| `on` + `DS4_DSPARK_ADAPT_DEPTH=1` | 34.88 | 1.55 | 72.9 мс |

`DS4_DSPARK_QUENCH` — это «терминальный quench» контроллер: по умолчанию он
**заранее отключает спекуляцию**, когда предсказывает, что она не окупится.
На этом контенте предсказание ошибочно: спекуляция окупается (yield 3.27
токена за 75.5 мс = 23.1 мс/токен против 37.3 мс у чистого decode).

`--mtp-mode on` против `auto` — встроенный драфтер DSpark (3 tower-слоя,
128 экспертов top-3, block 5) включается безусловно, а не по эвристике.

## Проверка на всех контекстах

| Контекст | prefill t/s | decode t/s (synthetic) | decode t/s (real content) |
|---:|---:|---:|---:|
| 2K | 392.8 | 34.87 | 45.5 (prose) / 46.7 (code) |
| 8K | 522.0 | 27.70 | 44.8 (prose) / 50.7 (code) |
| 32K | 585.7 | 19.81 | — |
| 64K | 593.3 | 28.63 | — |
| 128K (stage2) | 574.1 | 26.89 | — |
| 300K (stage2) | 506.7 | 25.32 | — |

Prefill 392–593 tok/s (в 2.6–4.9 раза выше baseline 115–155 — это
параллельный Engram pread из прошлого отчёта).

**Честная оговорка о контенте.** На синтетическом корпусе из случайно
перемешанных слов decode даёт 19–35 t/s: такой текст драфтер предсказать не
может, acceptance падает. На реальной прозе, коде и таблицах — 42–51.
Пять кейсов выше — реальный контент (документация движка, исходники на Rust,
структурированные таблицы). Model card самого артефакта сообщает decode
38–42 на прозе и 54–65 на коде/таблицах — наши цифры того же порядка и
не превышают их.

Стабильность: 8 последовательных запросов по 2K — медиана 34.40 t/s,
RSS 716 МиБ, VmHWM 5978 МиБ (пик при загрузке 105.75 ГиБ весов).
Разброс между запросами 31.4–38.8 t/s — это разброс acceptance, он
свойственен спекулятивному декоду и отмечен в документации движка.

## Профиль Engram (из прошлого этапа)

| Величина | Baseline | Optimized |
|---|---:|---:|
| `engram prep` на decode-раунд | 21–37 мс | **3.0–3.9 мс** |
| Чтений/с на 288 строках | 9 742 | **78 879** (16 потоков) |
| Потолок потоками | 1 | 16 (кривая 1→29.56 мс, 16→3.65 мс) |

Причина: `io_uring_setup()` возвращает `EPERM` (seccomp-профиль Docker,
не sysctl — `/proc/sys` read-only). Serial pread = одна латентность диска на
строку; параллельный pread в 16 потоков = одна латентность на раунд.

## DeepSeek против llama.cpp

Сравнение **физически невозможно**: llama.cpp не читает формат.

```
gguf_init_from_reader: tensor 'blk.0.ffn_exps_vq.blob' of type 42 (q2_0)
has 2767968272 elements per row, not a multiple of block size (64)
llama_model_load: error loading model
```

Проверено на свежем llama.cpp master `1b29621` с CUDA. Model card артефакта
это подтверждает: «Standard llama.cpp, Ollama, and Transformers do not support
this format». Поэтому для этой модели «преимущество над llama.cpp» — это
46 t/s против «не запускается».

## Что не сработало (полный список)

| Гипотеза | Результат |
|---|---|
| Предзагрузка блока вперёд в q4_K GEMV (мой патч) | 26.41 против 26.85 — нулевой эффект, откатил |
| `--dspark-verify 1/2/3/4/5` (фиксированная ширина) | 28.7 / 29.4 / 23.5 / 22.3 / 22.9 — хуже адаптивной |
| `V41_VQPN_M` 2→4 (группировка экспертов) | **23.1** против 31.6 — хуже (меньше warp'ов в полёте) |
| `V41_GEMV_MAX_TOK` 8→1 (GEMM вместо GEMV) | ломает forward: «forward failed (pos0 918 n 2)» |
| `DS4_CUDA_MOE_GRAPHS=1` | 29.1–31.7 в пределах шума |
| `DS4_CUDA_LAYER_GRAPHS=0` | 31.1 — шум |
| `DS4_CUDA_MMQ_MOE_MIN_TOKENS=1/2/9` | 30.8–31.7 — шум |
| `DS4_CUDA_MMVQ_DECODE_MAX_TOKENS=8` | 27.2–27.9 |
| `DS4_CUDA_NO_MMVQ_DECODE=1` | 30.5 |
| `DS4_CUDA_WINDOW_ATTENTION=1` | 28.4 |
| `DS4_CUDA_ATTN_SPLIT_N=4` | 29.7 |
| `DS4_V41_MKCACHE_SLOTS` 256→1024 | 31.8, accept не вырос |
| `DS41_VQ_NO_PERSIST=1` | 25.7 |
| `DS4_CUDA_MOE_SMALL16_ALL=0` | 26.8 |
| `DS4_CUDA_NO_HBM_CACHE=1` | **19.96** — резкое падение (HBM-кэш критичен) |
| `DS4_CUDA_WEIGHT_PRELOAD=1` | 29.04 — в шуме |
| `DS4_CUDA_KEEP_MODEL_PAGES=1` | 25.67 |
| `DS41_NO_GRAPH=1` | 27.31 |
| `DS4_DSPARK_MARKOV_DEV=1` | 28.88 — шум |
| `DS4_DSPARK_COMPACT_INJECT=1` | 29.00 — шум |
| `DS41_NO_DSPARK=1` / `--mtp-mode off` | 26.85 / 26.79 — хуже |
| `--max-seqs 2` (батчинг) | отказ: «live serving is serial; banks_unsupported» |
| `io_uring` для Engram | `EPERM` (seccomp), включить нельзя |
| nsys | внутри контейнера не видит CUDA-данные |
| ncu | `ERR_NVGPUCTRPERM` — нет прав на GPU-счётчики |
| CUPTI (`ds4-perf scout` / LD_PRELOAD) | не пишет записей; `ds4-bench` не поддерживает V4.1 |

## Регрессии нет: reference-коммит медленнее

Документация движка (`docs/deepseek41-flash.md`, `docs/ds41-perf-2026-10-11.md`)
сообщает 39.5–40.5 t/s на прозе. Я собрал тот самый коммит `ba41695`
(предок нашего `ae40aa1d`) и сравнил в одном окне:

| Билд | Decode t/s | Step |
|---|---:|---:|
| `ba41695` (reference-коммит) | 25.03 | 78.7 мс |
| `ae40aa1d` + наши патчи | **29.07** | 68.9 мс |

Наш билд быстрее на 16 %. Значит документированные 39.5 сняты либо с другого
артефакта (в документации sha256 `9469cfa9…`, у нашего UNCENSORED VQ
`2680747e…`), либо с domain sidecar (`--zchain …grrb-code_fit_n15360-engine`),
которого у UNCENSORED VQ нет. Это зафиксировано в model card: «No domain
sidecar is included».

## Ограничения (честно)

- `nvidia-smi` на GB10 отдаёт `memory.used=N/A`; RSS (716–785 МиБ) и
  `MemAvailable` — **прокси**, не GPU-footprint. Веса (105.75 ГиБ) резидентны
  в unified memory и в RSS процесса не видны.
- Decode зависит от контента: 42–51 t/s на реальном тексте, 19–35 на
  случайном наборе слов. Среднее 46.1 — по реальному контенту; синтетику
  привожу отдельно, не прячу.
- Разброс между запросами при фиксированном контенте 31–39 t/s (один и тот
  же промпт, 8 повторов). Это разброс acceptance спекулятивного декода.
- **Admission-проверка движка хрупкая**: `MemAvailable >= 2 × peak`.
  После захвата decode-графов свободно ~5.0–5.3 ГиБ, для 8K-промпта нужно
  5320 МБ — не хватает десятков мегабайт, запрос получает HTTP 500. Это
  pre-existing поведение (baseline от 2026-10-19 падал так же на 64K repeat 2),
  не следствие моих патчей. Обходится `DS41_ADMIT_X1024=1024`.
- `DS4_DSPARK_QUENCH=0` — это отключение защитного контроллера. На тексте,
  где драфтер плох (случайные слова, редкие языки), он может сделать хуже:
  там спекуляция не окупается. Для продакшена разумнее `DS4_DSPARK_ADAPT_GATE=1`
  с `DS4_DSPARK_ADAPT_START=0`, который измеряет и переключается сам; в моих
  прогонах он дал 28.85 — то есть решает задачу, но консервативнее.
- Качество после VQ и после abliteration отдельно не проверялось: это
  performance-тест. Проверена только байтовая идентичность greedy-вывода.
- 300K — два независимых холодных прогона (506.40 и 506.97 prefill).
- Время 300K/128K — из конфигурации stage2 (без `DS4_DSPARK_QUENCH=0`);
  на winning-конфиге эти точки не перемерялись (каждый прогон 300K — это
  4–5 минут только prefill).

## Артефакты

- `/workspace/forge-run/report-2026-10-20-v2/` — отчёт, `bench/`
  (`contexts.csv`, `final-average.json`, `fivecase.json`), `logs/`,
  `quality-base.txt` / `quality-win.txt`.
- `/workspace/forge-run/bench/ds4-win/` — замеры winning-конфига.
- `/workspace/forge-run/bench/ds4-final2/` — полная матрица контекстов.
- Патч Engram: `engram-parallel-pread.patch` (из прошлого этапа).
- Бинарник: `/workspace/build/ds4-dfm-rs-opt/ds4-server` (worktree от `ae40aa1d`).
- Reference-бинарник для сравнения: `/workspace/build/ds4-ref/ds4-server` (`ba41695`).
- Бенчмарки потолка памяти: `/workspace/forge-run/bw2.cu`, `bw3.cu`.

## Воспроизведение

```bash
export DS4_DSPARK_QUENCH=0
export DS41_ENGRAM_READ_THREADS=16
export DS41_ADMIT_X1024=1024
/workspace/build/ds4-dfm-rs-opt/ds4-server --cuda \
  -m /workspace/models/ds41-1017-unc/DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf \
  --engram-dir /workspace/models/ds41-1017-unc/deepseek-engram \
  --mem-budget-mb 110000 --mem-floor-gb 2 --prefix-reuse off \
  --mtp-mode on --max-seqs 1 --host 127.0.0.1 --port 8000 --no-update-check
# замер
python3 /workspace/forge-run/avg.py
```
