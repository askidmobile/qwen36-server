# DeepSeek V4.1: архитектурно-зависимое квантование под RTX 3060 12 GB и 128K

Дата: 2026-10-28.
Цель: максимизировать полезный объём модели, который помещается в VRAM и локальную
память, не сводя задачу к универсальному scalar-кванту всего GGUF.

## Вывод

DeepSeek-V4.1-Flash-1017 содержит около 763.2 млрд параметров: 40 MoE-слоёв,
по 384 routed expert и 6 активных expert на токен, плюс Engram. Даже основная часть
модели не помещается в 12 GB ни в 2-битном, ни в 1.75-битном представлении.
Практическая задача поэтому состоит не в том, чтобы весь checkpoint стал меньше
12 GB, а в совместной оптимизации трёх вещей:

1. какие плотные веса постоянно живут в VRAM;
2. какая точность нужна каждому expert;
3. какие experts и в каком формате кэшируются в оставшихся 5–6 GiB.

Универсальный GGUF плохо решает пункты 2 и 3. В текущем `llama.cpp` каждый routed
expert role хранится одним тензором `{n_embd, n_ff, n_expert}` с одним `ggml_type`.
`--override-tensor` и `--n-cpu-moe` выбирают целый тензор или слой, а не отдельного
expert. Новый `--moe-cache-mib` уже кэширует отдельных host-expert в VRAM, но
требует одинаковых shape/type/layout внутри группы кэша. Поэтому разные уровни
точности для experts одного слоя стандартным GGUF не выражаются.

## Главный принцип: precision и placement оптимизировать отдельно

Ошибка на реальной карте `dsx-2bit` показывает, что routing frequency нельзя
использовать как замену измерению чувствительности:

- верхние 1 000 из 15 360 experts покрывают 31.4% маршрутов;
- precision tier и routing frequency имеют лишь умеренную Spearman correlation 0.305;
- только 12.8% experts получили `Q3_K`, но на них пришлось 23.8% маршрутов;
- без активационной калибровки YoungAi получил почти равномерное оптимальное
  распределение: относительные ошибки experts различаются только на 1.9%.

Выбор точности должен минимизировать activation-weighted ошибку выхода:

```text
min Σ error_eval(layer, expert, codec)
s.t. Σ bytes(layer, expert, codec) <= budget
```

Ошибка измеряется на held-out токенах после GPTQ/refit. Для expert с малым числом
eval-токенов нужен повышенный precision floor: отсутствие измерения не является
доказательством, что дешёвый tier безопасен.

VRAM-размещение решается отдельно по ожидаемым попаданиям на сериализованный байт:

```text
rank(layer, expert) = routed_picks / (codec_bytes + cache_slot_overhead)
```

На опубликованной hotness-карте при бюджете около 5.1 GiB сортировка только по
частоте дала 23.99% покрытия маршрутов, а сортировка по `hits / bytes` — 25.26%.
Profile должен строиться на routing traces целевой рабочей нагрузки, а не только
на общем code/agent-корпусе.

## Почему обычный Q2/Q1 GGUF не является оптимумом

1. Один scalar type назначается всему tensor/layer. Эксперт с ошибкой на калибровке
   в несколько раз выше соседа получает тот же формат.
2. Per-tensor RCO хорошо выбирает тип между слоями, но не внутри слоя.
3. CPU/GPU placement на уровне целого tensor не позволяет держать три горячих
   expert слоя в VRAM, а остальные 381 читать из RAM.
4. Универсальные K/IQ-ядра оптимизируют общий случай. Меньшая битность может быть
   медленнее: на RTX 3060 Q3 иногда проигрывал Q4, а 4-битный KV — q8.
5. Доменная коррекция не должна менять общий base. Небольшой sidecar с row gains и
   router bias оказался гораздо эффективнее повторного повышения битности.

Activation-calibrated tier allocation уже показывает реальный потенциал: при
одинаковом бюджете около 2 bpw uniform Q2 дал proxy error `1.537`, смесь
`PTQ1/Q2/Q3` — `0.682` (−55.6%). Это proxy ошибка отдельных экспертов, поэтому
её нельзя выдавать за измеренный KLD всей модели, но она достаточна, чтобы
направлять следующий quality gate.

## Формат QAMQ-DS41: индексный expert-pack

Предлагаемый контейнер:

```text
dense pack
  attention, router, shared expert, embedding/output head
  стандартные Q4_K/Q8_0/F16 по чувствительности

expert index
  (layer, role, expert) -> offset, length, codec_id

codec descriptor
  scalar / VQ-8 / T16-trellis
  bit width, codebook, row-gain layout, Hadamard, tile order

expert profile
  workload-specific hotness для VRAM/RAM/LRU

domain sidecar
  row gains + router bias, с hash привязкой к base
```

Для первой версии достаточно output `L e nc` для YoungAi и hotness profile для
одного из двух Strata runtime. `strata-dsv41` принимает raw little-endian массив
`uint32` counts размером `layers × experts`; generic Strata engine принимает
ранжированный `STRP` version 1. Полный per-expert mix требует DQVL v4 или
эквивалентного indexed формата:
текущий v3 имеет одну геометрию/codebook слоя, а shared-codebook path в YoungAi
ожидает одну ширину на слой. Ядра должны обрабатывать experts группами одинакового
codec, чтобы routing не добавлял новую ветку на каждого эксперта.

Кандидаты для VQ-пути: 10–14 index bits. 12 и 13 бит уже имеют GPU-ядра YoungAi;
прочие ширины требуют v4. Для scalar/trellis-пути разумная стартовая сетка:
`PTQ1`, `IQ2_XXS`, `Q2_0`, `Q3_K`. Dense, router, norms и shared expert в
экспериментальный низкобитный tier не опускаются.

## Бюджет RTX 3060 12 GB при 128K

Рабочий бюджет для V4.1:

| Компонент | Измерено/посчитано |
|---|---:|
| dense/attention/shared/router/head | 5.69 GB (5.30 GiB) по RCO |
| dequant/GEMM workspace dense-слоёв | ~573 MiB |
| compressed KV + indexer + window @128K | 225 MiB |
| prefill workspace | 1 024 MiB |
| CUDA context, WDDM | 512 MiB |
| **hot expert cache** | **4 452 MiB** |

Уточнение 2026-10-28: ранняя оценка «dense 4.5 GiB и hot cache 5–6 GiB» была
неверной. Dense посчитан пересчётом всех 1043 тензоров официальной
`rco-allocation.txt`, KV — по `config.json` и форме кэшей в reference
`inference/model.py`. Детали: `2026-10-28-deepseek-v41-12gb-custom-quant.md`.

Активный путь токена выбирает 240 experts: 6 experts × 40 слоёв. При среднем
сериализованном размере около 6.7–9 MB это 1.6–2.2 GiB активных expert weights
на токен. Если в VRAM попадает только 25–30%, остальные 1.1–1.6 GiB должны
декодироваться непосредственно из RAM/SSD внутри MoE GEMM. Промежуточная запись
полного dequantized expert в GPU бесполезна: она увеличивает трафик и отнимает
кэш у других experts.

## Pipeline

1. Собрать calibration из code/agent/tool/RU/long-context и отдельный held-out.
2. Снять routing и входные активации через DeepSeek reference/runtime.
3. Для каждого candidate format посчитать held-out output error после GPTQ/refit.
4. Запустить Lagrangian allocator при целевом байтовом бюджете.
5. Сравнить mixed map с uniform map той же стоимости. Non-uniform принимать только
   при измеримом выигрыше; текущий стартовый gate — не меньше 2%.
6. Собрать expert-pack; проверить каждый offset/codec/shape и round-trip decode.
7. По routing traces построить hotness profile и симулировать LRU/SLRU до измерения
   на GPU.
8. Добавить domain sidecar и независимо проверить KLD/top-1/задачи.
9. Запустить 128K на реальном runtime с sampling VRAM, RAM и дискового трафика.

## Gates

Квант нельзя считать удачным только по размеру файла. Обязательные проверки:

- `mixed proxy error < uniform proxy error` на тех же байтах и held-out строках;
- mixed mode включается только при наличии activation stats либо явного
  experimental flag; routing-only tiers по умолчанию запрещены;
- KLD/top-1 и task evals не хуже выбранного baseline за пределами заданного CI;
- needle и tool-calling на 64K/128K;
- decode и prefill на 4K/32K/128K;
- `GPU Process Memory Shared Usage` не растёт, dedicated VRAM оставляет не меньше
  256 MiB;
- hit coverage и `disk_bytes_per_token` совпадают с профилем;
- повторный процесс и холодный старт дают те же tier/profile hashes.

## Прототип

`scripts/qamq/ds41_vram_alloc.py` строит:

- uniform baseline;
- безопасный mixed allocation только по статистике (`routing-only` — отдельный
  experimental режим);
- `L e nc` таблицу для YoungAi;
- raw `uint32` counts для `strata-dsv41` (формат по умолчанию);
- опциональный `STRP` ranking для generic Strata expert cache;
- JSON с байтами, tier histogram, predicted error и VRAM/RAM/SSD coverage.

Пример запуска для подготовленного calibration/stats набора:

```sh
python scripts/qamq/ds41_vram_alloc.py \
  --routing calibration/expert-rank.json \
  --routing-format auto \
  --stats calibration/stats-*.txt \
  --budget-gb 120 \
  --dense-mib 4608 --kv-mib 256 --workspace-mib 1024 --reserve-mib 512 \
  --ram-mib 32768 --min-improvement 0.02 --min-stats-coverage 1.0 \
  --out-table calibration/ds41-nc.txt \
  --profile-format ds4-counts \
  --out-profile calibration/ds41-expert-profile.bin \
  --out-report calibration/ds41-layout.json
```

Для прямого DS4 engine можно снять `DS4_TRACE` или `DS4_DUMP_R`. Первый компактнее
и содержит ids маршрутов: укажите `--routing-format ds4-trace --ds4-top-k 6`
для Full quality либо `--ds4-top-k 5` для Faster. `DS4_DUMP_R` выбирается через
`--routing-format ds4-dump-r`; из каждого токена учитываются только первые `k` из
`kPicks` кандидатов, а не весь look-ahead bucket. `auto` сначала проверяет обычный
Strata trace, затем `DS4_DUMP_R` и `DS4_TRACE`, поэтому для повреждённого файла
формат лучше задать явно.

Различие профилей проверено по Strata commit `46c152a1310f`: `ds4_model.cu`
читает `(layers + MTP) × experts` как `uint32` counts, тогда как
`src/core/expert_cache.cpp` читает `STRP` для generic engine. Поэтому
`--profile-format strata-ranked` предназначен для второго runtime, а не для
`strata-dsv41`.

Routing-only эксперимент возможен только с явным `--allow-estimated-tiers`; такой
report помечается `safe_for_build=false` и не должен становиться основой релиза.

Он пока не создаёт готовый model artifact. Текущий текстовый parser принимает
YoungAi `sse/energy`; для activation-calibrated ошибок нужен adapter к
`stats/LNN.json` из dsx/Xolotl либо использование их `v41_alloc.py`. Нельзя
подменять эти два вида ошибки без отдельного пересчёта.

Следующий обязательный инженерный шаг — DQVL v4 с per-slot codec descriptor
и fused dispatch по группам experts. Без этого нельзя одновременно использовать
mixed precision и существующие v3-ядра без дорогих отдельных codebook на каждого
expert.

## Сухая проверка аллокатора на опубликованной hotness-карте

Для uniform 12-битного DQVL-варианта и бюджета `4 452 MiB` под experts аллокатор
на карте `dsx-2bit` прогнозирует:

| Параметр | Значение |
|---|---:|
| всего experts | 15 360 |
| VRAM experts | 701 |
| покрытие всех routed picks | 26.1% |
| RAM budget | 31 GiB |
| RAM experts | 5 001 |
| VRAM + RAM coverage | 71.8% |
| experts только на SSD | 9 646 |
| routed expert payload | 95.2 GiB |

Это арифметика выбранных offsets и profile, а не замер скорости нового runtime.
Она показывает, почему индексный формат полезен: 701 expert помещается в 4 452 MiB
только потому, что средний 12-битный VQ expert занимает 6.66 MB.
При scalar Q2 размере около 9–10 MB в тот же бюджет вошло бы примерно на 30% меньше
experts.

## Источники

- [DeepSeek V4.1 Flash 1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017):
  конфигурация 40 слоёв, 384 experts, top-6, 1M context.
- [ISTA-DASLab GSQ-RCO](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF):
  mixed scalar tiers и quality results.
- [lxe dsx](https://github.com/lxe/dsx): activation-calibrated per-expert allocation,
  routing ranking, T16 и CPU/GPU placement.
- [YoungAi technical notes](https://github.com/fodelf/YoungAi/blob/7fb26cc332d740416792204a29c7d13d2986caf5/docs/TECHNICAL.md):
  DQVL VQ, shared codebooks, domain sidecars и отрицательные результаты
  weight-only dynamic allocation.
- [Strata DeepSeek guide](https://github.com/Niko1221/Strata/blob/main/docs/DEEPSEEK41.md):
  tiered CPU/GPU/SSD execution, context budget и 12 GB profile.
- Актуальный [llama.cpp MoE cache](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-moe-cache.cpp):
  кэш host-experts на GPU при ограничении одинакового type/layout внутри группы.
