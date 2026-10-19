# DGX Spark: DeepSeek V4.1 Flash 1017 UNCENSORED VQ

Дата теста: 2026-10-19. Стенд: DGX Spark, NVIDIA GB10, 48 SM,
20 vCPU, 121 GiB unified memory, Ubuntu 24.04.1 aarch64, driver 580.173.02,
CUDA 13.0.88.

## Артефакт

Использован
[`Baekpica/DeepSeek-V4.1-Flash-1017-UNCENSORED-VQ-GGUF`](https://huggingface.co/Baekpica/DeepSeek-V4.1-Flash-1017-UNCENSORED-VQ-GGUF),
revision `610a2d71400313f63228589bf4cea4182ad360e9`. Это VQ-квантование
[`dealignai/DeepSeek-V4.1-Flash-1017-UNCENSORED-FP8`](https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-1017-UNCENSORED-FP8),
а не base-модель.

| Файл | Размер | SHA-256 |
|---|---:|---|
| `DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf` | 113,556,639,424 B | `2680747eaec26d62c3f40c1a8fce5b931d210a709f71868bde5d68766bccf0a1` |
| `DeepSeek-V4.1-Flash.vision.gguf` | 971,066,016 B | `3a8b194a011e630e8e49a51b5f4e9821e48a9e5674e678d52c0302d6b164698c` |
| Engram shard 47 | 101,535,150,936 B | `76a05106ece259be83d1bfea0dcb54f23377864ee60fd321690f35e6698a2af5` |
| Engram shard 48 | 101,537,926,640 B | `9dc51433e195bc78b5ad846b68643cc0351d3b174e5e4309222d3b7402c83afd` |

Все четыре позиции проверены `sha256sum -c`; лог:
[`ds41-unc-verify-all.log`](2026-10-19-dgx-spark-deepseek-artifacts/ds41-unc-verify-all.log).

## Runtime

`ds4-dfm-rs`:

- основной тест — commit `ae40aa1d`, тот же известный DGX Spark build path;
- текущий `main` (`e99161b`) также собран, но замеры ниже сняты на `ae40aa1d`;
- `make cuda-spark`, `compute_121a`;
- `ds4-server` 66 MiB;
- startup до `GET /v1/models = 200`: **131.5 s**;
- загружено **87/87** promote units, 105.75 GiB;
- `--ctx` не передавался: контекст берётся из metadata и равен **1,048,576**;
- `--prefix-reuse off`, `--max-seqs 1`, `--mtp-mode auto`;
- `--mem-budget-mb 108500`, `--mem-floor-gb 2`.

`--check-config` вернул `issues=[]`, plan total 116.759 GB при доступных
122.308 GB; лог:
[`stage5-check-config-full.log`](2026-10-19-dgx-spark-deepseek-artifacts/stage5-check-config-full.log).

## Методика

Единый harness `scripts/bench/dgx_spark/http_bench.py`:

- `temperature=0`;
- `max_tokens=65`;
- prompt-файл `speed-bench/promessi_sposi.txt` (естественный текст, без
  искусственного повторения одного слова);
- уникальный nonce и сдвиг окна на каждый прогон;
- prompt tokens из `usage.prompt_tokens`, completion tokens из
  `usage.completion_tokens`;
- TTFT включает prefill;
- два независимых прогона на точку, медиана;
- 300K: два независимых прогона после холодного старта, по одному на процесс;
- stability: 20 последовательных запросов по ~8K с `--prefix-reuse off`.

`nvidia-smi` на GB10 отдаёт `memory.used=N/A`, поэтому в таблице RSS процесса
и `MemAvailable` хоста, а не точный GPU-only resident footprint.

## Результаты

| Target | Факт. prompt | TTFT, s | Prefill tok/s | Decode tok/s | RSS max, MiB | MemAvailable min, MiB |
|---:|---:|---:|---:|---:|---:|---:|
| 2K | 2,082 | 18.087 | 115.12 | 21.35 | 684.9 | 3,729.3 |
| 8K | 8,368 | 68.811 | 121.61 | 19.41 | 684.9 | 3,589.1 |
| 32K | 33,381 | 278.692 | 119.81 | 23.82 | 685.6 | 3,064.4 |
| 64K | 66,040 | 558.782 | 118.18 | 17.71 | 754.7 | 3,331.8 |
| 128K | 132,725 | 1,137.001 | 116.73 | 18.01 | 748.6 | 3,128.8 |
| 300K | 308,262 | 2,794.562 | 110.31 | 15.67 | 697.1 | 3,182.7 |

Для 300K второй независимый прогон дал 108.22 tok/s prefill / 16.36 decode,
медиана двух: **109.27 tok/s prefill / 16.01 tok/s decode**.

Сырые CSV:
[`ds4-unc-ae40-109.csv`](2026-10-19-dgx-spark-deepseek-artifacts/ds4-unc-ae40-109.csv),
[`64k.csv`](2026-10-19-dgx-spark-deepseek-artifacts/64k.csv),
[`128k-a.csv`](2026-10-19-dgx-spark-deepseek-artifacts/128k-a.csv),
[`128k-b.csv`](2026-10-19-dgx-spark-deepseek-artifacts/128k-b.csv),
[`300k-a.csv`](2026-10-19-dgx-spark-deepseek-artifacts/300k-a.csv),
[`300k-b.csv`](2026-10-19-dgx-spark-deepseek-artifacts/300k-b.csv).

## Engram

Отдельный A/B на 556 prompt tokens:

| Режим | Prefill tok/s |
|---|---:|
| полный Engram | ~96–97 |
| `--v41-no-engram` | ~410 |

То есть на этом стенде основная часть prefill-времени уходит в Engram
lookup/чтение таблиц, а не в VQ GEMM. Это также объясняет, почему наши
~110–120 tok/s нельзя напрямую сравнивать с опубликованными 775–1,120 tok/s:
reference-график DS4 снят с **code GR sidecar**, а у UNCENSORED VQ артефакта
sidecar отсутствует. Сам artifact card также предупреждает, что качество и
поведение после VQ не измерены.

## Stability

20 последовательных запросов по 8K: **20/20 успешно, 0 ошибок**.

- медиана prefill: **119.51 tok/s**;
- медиана decode: **18.49 tok/s**;
- RSS: 689.8 MiB (первые 5) → 708.9 MiB (последние 5), рост около 19 MiB;
- `MemAvailable` оставался около 5.5 GiB, без устойчивого падения;
- `cached_tokens=0` на всех запросах, потому что prefix reuse выключен.

Сырой CSV:
[`ds4-unc-ae40-1085-stability.csv`](2026-10-19-dgx-spark-deepseek-artifacts/ds4-unc-ae40-1085-stability.csv).

## Ограничения

- VTuber/uncensored behavior после VQ не проверялся: это performance-тест,
  а не quality gate.
- Нет сравнимого llama.cpp/vLLM прогона того же VQ-артефакта.
- `nvidia-smi` не даёт VRAM на GB10; RSS/MemAvailable — прокси.
- Back-to-back повтор 300K без перезапуска сервера один раз был отклонён
  memory admission; после холодного рестарта 300K снова прошёл. Это
  ограничение памяти/workspace, а не ошибка модели.
