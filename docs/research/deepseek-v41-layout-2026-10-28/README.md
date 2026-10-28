# DeepSeek V4.1 Flash на RTX 3060 12 GB

Состояние на 2026-10-28. Это рабочий план и машинно-проверяемая раскладка, а не
результат полного inference-прогона.

## Выбранный эксперимент

Запускать `Strata 0.2.7` с `strata-dsv41 0.1.52` и официальным набором
`IST-DAS-2.00bpw`: 138.24 GB backbone, 110.60 GB Engram и 7.98 GB DSpark.
Набор сохраняет все 15 360 routed experts; pruning не используется.

`2.00bpw` выбран вместо `1.75bpw`, потому что опубликованный quality proxy заметно
лучше: perplexity +15.5% против +21%, code +8.6% против +8.9%, agent work +5.3%
против +10.2%. Простая проза у `1.75bpw` проседает на 53%. Если `2.00bpw` окажется
слишком медленным, `1.75bpw` нужен как отдельный speed-first A/B, а не как default.

Первый запуск делать в Full quality (`--experts 6`). Faster (`--experts 5`) допустим
только следующим отдельным прогоном: он экономит примерно один expert из шести, но
меняет качество и профиль маршрутов.

## Что помещается в VRAM

Для 128K закладывается 12 288 MiB, из них:

| Компонент | MiB |
|---|---:|
| dense/attention/router/shared/head | 5 427 |
| dequant/GEMM workspace dense-слоёв | 573 |
| 128K KV, index keys, scratch | 225 |
| workspace и prompt buffers | 1 024 |
| WDDM/CUDA/reserve | 512 |
| routed expert cache | 4 713 |

Dense посчитан пересчётом официальной `rco-allocation.txt` по таблице блоков GGML:
5.69 GB, а не оценка. При 12-битном VQ expert 6.66 MiB это 701 эксперт и 26.1%
routing-weighted покрытия; при 10 битах 5.55 MiB — 841 эксперт и 28.7%. Scalar
IQ2_XXS expert весит 8.60 MB, поэтому его в тот же VRAM влезает 542 (22.7%).

128K KV у DeepSeek V4.1 неожиданно дешёвый. В `config.json` четыре KV source:

- ratios `2,2,2,1`: `(3 * 288 / 2 + 288) * 131072 = 94 371 840` bytes;
- восемь index sources с ratios `2,2,2,1,1,1,1,1`:
  `(3 * 40 + 5 * 80) * 131072 = 68 157 440` bytes;
- итого `162 529 280` bytes = 155 MiB, плюс около 4 MiB score slab,
  ring/window buffers и scratch.

Поэтому бюджет 256 MiB достаточен для 128K и оставляет 54 MiB запаса. Утверждения
про KV 1.37 GiB и тем более 5 GiB относятся к другому расчёту или конфигурации.

## Реальное ограничение этой машины

На `yttri-win`: RTX 3060 12 GB, 64 GB physical RAM, около 29 GiB свободно в момент
проверки, pagefile 7.5 GiB. Strata plan для 2.00bpw требует 133.8 GB RAM, если
держать всех CPU experts в памяти, и 40+ GiB для измеренного `tight`-режима.
Здесь большая часть CPU experts будет идти с диска.

До оптимизации раскладки модель скачивается на `X:` (SATA HDD). Для inference это
худший вариант. После завершения и SHA-256 проверки файлы нужно разнести так:

- backbone (138 GB) на `C:` NVMe: 217.8 GB free;
- shared Engram + DSpark + metadata (около 119 GB) на `F:` SATA SSD: 155.5 GB free;
- `X:` оставить только как источник загрузки;
- на `X:` создать NTFS junctions на эти каталоги, чтобы не менять абсолютные пути
  pack, и повторить `strata-dsv41 --pack-info` и smoke test.

Перед переносом setup завершает pack и короткий smoke test. Переносить файлы во
время активной загрузки нельзя.

## Артефакты

- `reference/expert-rank.json` - опубликованная hotness-карта dsx: 62 914 560
  routed picks, 15 360 пар `(layer, expert)`.
- `expert-profile.bin` - 15 360 little-endian `uint32` counts, ровно формат
  `strata-dsv41 --expert-profile`; не `STRP`.
- `expert-profile.strp` - тот же ranking в `STRP` version 1 для generic Strata
  expert cache; это отдельный формат и другой runtime.
- `summary.json` - компактная сводка обеих раскладок (uniform 12 бит и 13/10/9).
- `recipe.txt`, `recipe-hotness.txt` - готовые команды llama.cpp/Strata.
- `nc-table.txt`, `nc-hotness.txt` - `L e nc` map для YoungAi.
- `expert-profile-hotness.bin` - counts для mixed-раскладки.

Проверенные SHA-256:

```text
4afadccff5134056539193ae92c5a8e79278867ef06bdc52b4db3df278491674  reference/expert-rank.json
6c3682cbaefccf2417cf6891d74e9a4054314ea759bc92fb590618e4163c9114  expert-profile.bin
367cbf053deb4cdd13953f648467a08452a81d74d594404cbc798839a254c66c  expert-profile.strp
ce097d4355d4fb5e4318dbf323f7bb1618d289dd2963ae0b194e74335cf953b3  nc-table.txt
8e0645934cd5b0da8f7a0b3c2236d291bb50888479089f9db382f5602c14fefe  summary.json
```

Профиль воспроизводится командой:

```sh
python3 scripts/qamq/ds41_vram_alloc.py \
  --routing reference/expert-rank.json \
  --candidates 12 --reference-bits 12 --mode uniform \
  --vram-mib 12288 --dense-mib 6000 --kv-mib 225 \
  --workspace-mib 1024 --reserve-mib 512 --ram-mib 31744 \
  --out-recipe recipe.txt \
  --out-profile expert-profile.bin \
  --profile-format ds4-counts \
  --out-report layout.json \
  --out-table nc-table.txt
```

`--profile-format strata-ranked` и отдельный `--out-profile` дают `STRP` для generic
engine. Отчёт проверяется по полям `allocation.safe_for_build`,
`allocation.runtime.build_compatible`, `routing.hot_coverage` и
`routing.vram_ram_coverage`; роль VRAM-покрытия выполняет `hot_coverage`.

Mixed-раскладка воспроизводится так:

```sh
python3 scripts/qamq/ds41_vram_alloc.py \
  --routing reference/expert-rank.json \
  --candidates 9,10,11,12,13,14 --min-bits 9 --max-bits 14 \
  --mode hotness --tier-plan 13,10,9 \
  --vram-mib 12288 --dense-mib 6000 --kv-mib 225 \
  --workspace-mib 1024 --reserve-mib 512 --ram-mib 31744 \
  --out-profile expert-profile-hotness.bin \
  --out-table nc-hotness.txt \
  --out-recipe recipe-hotness.txt \
  --out-report /tmp/layout-hotness.json
```

Она даёт `safe_for_build=false`: 9 и 10 бит требуют DQVL v4, а по измеренной кривой
ошибки ещё и проигрывают uniform 12 битам (см. `docs/research/2026-10-28-deepseek-v41-12gb-custom-quant.md`).

## Почему не per-expert precision «на глаз»

Routing frequency и чувствительность к квантованию коррелируют лишь на уровне
Spearman 0.305. YoungAi на реальном checkpoint получил разброс оптимальной
точности весов около 1.9% и отрицательный результат для dynamic per-layer /
per-expert allocation без activation-calibrated sidecar. Поэтому hotness управляет
только размещением. Точность меняется лишь по измеренной output error.

Текущий DQVL v3 принимает 12/13-битные индексы и одну ширину на слой. Смешанные
ширины отдельных experts требуют DQVL v4 с per-slot codec descriptor и fused dispatch
по группам одинаковых codecs. `safe_for_build=true` в текущем отчёте означает только
layer-uniform 12-битную раскладку, а не доказанное превосходство mixed VQ.

## Сравнение с Qwen3.8 27B

Прямого одинакового eval пока нет. Внешний Qwen benchmark сообщает для одного
12 GB-стенда: `UD-IQ1_S` 1.84 bpw дал 0 из 12 корректных structured outputs,
`UD-IQ1_M` около 2 bpw - 10/21 (41.5% contract), `UD-IQ2_XXS` 2.16 bpw - 19/27
(58.3%). ISTA-DASLab для DeepSeek 2.00bpw заявляет 98.7% task average относительно
исходной модели, но одновременно perplexity +15.5%; независимого повтора этого
заявления на 3060 пока нет. Эти числа нельзя ставить в одну таблицу без одинаковых
задач, reasoning budget и KV quant.

Ожидание: DeepSeek на 2 bpw может выиграть по качеству у Qwen Q1/Q2, но проиграет
по latency на этой машине из-за 64 GB RAM, HDD и потоковой подкачки experts.

## Обязательная приёмка после загрузки

1. `strata-dsv41 --pack-info`: все файлы pack присутствуют.
2. Smoke test отвечает `Paris`, процесс не завершился с OOM.
3. `POST /v1/chat/completions` с `reasoning_effort: none`, `temperature: 0`.
4. На 4K, 32K и 128K записать TTFT, prefill tok/s, decode tok/s, `/metrics`,
   dedicated/shared GPU memory, free RAM и disk bytes read.
5. На 64K и 128K проверить needle на 10/50/90% глубины и tool-call.
6. Повторить cold и warm; сравнить `disk_bytes_per_token` с ожидаемым профилем.
7. Только после этого сравнивать на одних task evals с Qwen3.8 27B IQ2/Q2 и решать,
   выиграл ли эксперимент.
