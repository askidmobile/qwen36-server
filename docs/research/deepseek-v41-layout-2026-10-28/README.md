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
| dense/attention/router/shared/head/hc | 5 680 |
| dequant/GEMM workspace dense-слоёв | 600 |
| 128K KV, index keys, scratch | 300 |
| workspace и prompt buffers | 1 024 |
| WDDM/CUDA/reserve | 512 |
| routed expert cache | 4 172 |

Dense посчитан пересчётом официальной `rco-allocation.txt` по таблице блоков GGML:
5.96 GB, и модель сходится с реальными шардами до 210 MB. При 12-битном VQ expert
6.66 MB это 657 экспертов и 25.2% routing-weighted покрытия; при 10 битах 5.55 MB —
788 экспертов и 27.7%. Scalar IQ2_XXS expert весит 8.60 MB, поэтому в тот же VRAM
влезает 542 эксперта (22.7%).

128K KV у DeepSeek V4.1 дешёвый, и это подтверждено реализацией, а не только
расчётом. В `src/llama-kv-cache-dsv4.cpp` у V4.1 `n_embd_comp = n_embd_head_k +
indexer_head_size = 640`, index-K хранится в той же строке, что сжатый KV:

- CSA (`ratio 2`): `pad(131072/2, 256) = 65 536` строк;
- HCA (`ratio 1`): `pad(131072, 256) = 131 072` строки;
- всего `196 608 × 640 = 125 829 120` элементов плюс raw window
  `40 × 128 × 512 = 2 621 440`.

| Тип K | compressed | raw | итого 128K |
|---|---:|---:|---:|
| `q8_0` | 127.5 MiB | 2.7 MiB | **130.2 MiB** |
| `f16` | 240.0 MiB | 5.0 MiB | **245.0 MiB** |

Бюджет 300 MiB покрывает оба варианта с запасом. Утверждения про KV 1.37 GiB и
тем более 5 GiB относятся к другой архитектуре или к неверной формуле.

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
- `summary.json` - сводка uniform, hotness и measured-curve раскладок.
- `layout-calibrated.json`, `recipe-calibrated.txt`, `nc-calibrated.txt`,
  `expert-profile-calibrated.bin` - раскладка по измеренной VQ-кривой.
- `vq-curve-v41-9experts.json` - измерения ошибки на реальных expert V4.1.
- `recipe.txt`, `recipe-hotness.txt` - команды для uniform и tier-plan.

Проверенные SHA-256:

```text
4afadccff5134056539193ae92c5a8e79278867ef06bdc52b4db3df278491674  reference/expert-rank.json
6c3682cbaefccf2417cf6891d74e9a4054314ea759bc92fb590618e4163c9114  expert-profile.bin
367cbf053deb4cdd13953f648467a08452a81d74d594404cbc798839a254c66c  expert-profile.strp
ce097d4355d4fb5e4318dbf323f7bb1618d289dd2963ae0b194e74335cf953b3  nc-table.txt
53f19b7bc2c9eb0c61baddac2b2a275ae6b46ad8c29b348e00c79dd62113915c  nc-calibrated.txt
2e44a6c78ea345e0126f2b33ac0ba77a41e2b36da424820e0733c01909cbc6d1  summary.json
2303ca491049fd048a843509f90956c93718da1e8a157a92724d7923af2cafe2  recipe-calibrated.txt
ddce3900edd2d09ccbe3b000746d7be51371ebff5bddb4c7552b2894f4b5aa09  vq-curve-v41-9experts.json
```

Профиль воспроизводится командой:

```sh
python3 scripts/qamq/ds41_vram_alloc.py \
  --routing reference/expert-rank.json \
  --candidates 12 --reference-bits 12 --mode uniform \
  --vram-mib 12288 --dense-mib 6280 --kv-mib 300 \
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
  --vram-mib 12288 --dense-mib 6280 --kv-mib 300 \
  --workspace-mib 1024 --reserve-mib 512 --ram-mib 31744 \
  --out-profile expert-profile-hotness.bin \
  --out-table nc-hotness.txt \
  --out-recipe recipe-hotness.txt \
  --out-report /tmp/layout-hotness.json
```

Она даёт `safe_for_build=false`: tier-plan не использует измеренную кривую.
Рабочая mixed-раскладка строится через `--mode calibrated --curve
vq-curve-v41-9experts.json`; она снижает routing-weighted ошибку на 26.2% при том же
размере, но требует DQVL v4 и увеличивает SSD traffic на 10%.

## Почему не per-expert precision «на глаз»

Routing frequency и чувствительность к квантованию коррелируют лишь на уровне
Spearman 0.305. YoungAi на реальном checkpoint получил разброс оптимальной
точности весов около 1.9% и отрицательный результат для dynamic per-layer /
per-expert allocation без activation-calibrated sidecar. Поэтому hotness управляет
только размещением. Точность меняется лишь по измеренной output error.

Текущий DQVL v3 принимает 12/13-битные индексы и одну ширину на слой. Смешанные
ширины отдельных experts требуют DQVP/DQVL v4 с per-slot codec descriptor и fused
dispatch по группам одинаковых codecs. Поэтому `safe_for_build=true` относится к
layer-uniform 12-битной раскладке; mixed map пока reference-only.

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

## Загрузка и конвертация

Проверено 2026-10-28 23:25 на `yttri-win`:

- backbone: 4 шарда, 138.24 GB, все `.done`;
- Engram: 85.6 GB из 110.6 GB, ещё качается.

`scripts/qamq/preflight_shards.py` подтверждает, что скачанный backbone — **старый
layout до 27 октября**: 6 тензоров `engram_q`/`engram_k`/`engram_wkv`, нет
`per_layer_token_embd`, нет `engram_gate`/`engram_kv`. llama.cpp b11870+ и текущий
Strata такой набор не откроют.

Нужна конвертация:

```sh
# once, in the model folder
hf download ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF \
  convert_ds41_upstream.py chat_template.jinja exp_probs_b_vl.f32 --local-dir .

# Engram must be complete first: the converter rewrites its header in place
python convert_ds41_upstream.py X:\Strata\data\models\deepseek-v4.1-flash-IST-DAS-2.00bpw
```

Требования и ограничения:

- нужно **138 GB свободного места** под новые шарды рядом со старыми; на `C:` 178 GB,
  на `X:` около 930 GB;
- без `--delete-old` старые шарды остаются, то есть операция обратима;
- конвертер переписывает только заголовок Engram-файла, 110 GB таблиц остаются на месте,
  но файл должен быть **докачан и проверен**;
- перед запуском повторить `preflight_shards.py` — он должен сказать
  `ready: current layout`.

**Диск важнее конвертации.** Backbone сейчас на `X:` (SATA HDD). При 456 MB трафика
на токен это 0.3 tok/s. Перенос на `C:` NVMe — обязательное условие, а не оптимизация.

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
