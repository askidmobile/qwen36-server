# DGX Spark: первая сборка Forge и baseline Ornith 1.5 9B

Дата: 2026-10-19. Хост: `dgx-forge.gpuwerk.com`, NVIDIA GB10,
`compute_cap=12.1`, 48 SM, 20 vCPU, 121 GiB unified memory, Ubuntu 24.04.1
aarch64. Driver 580.173.02, CUDA toolkit 13.0.88, Rust 1.99.1.

## Что собрано

- `yforge`: `/workspace/build/qwen36/target/release/yforge`, 286 MiB.
- `ds4-dfm-rs`: `make cuda-spark`, commit `e99161bafe4682973a73143a833f82ee581ebd8d`;
  `ds4-server` 45 MiB. Runtime запускается, но запрошенный FP8 checkpoint на
  этот стенд не помещается (см. отчёт по DeepSeek ниже).
- Флаги flash-attn собраны под `sm_121f`.
- Для ссылки на CUDA runtime итоговый `yforge` собран через
  `cargo rustc ... -C link-arg=-lcudart`; статическая `cudart_static`
  на aarch64 не попала в финальную линковку.
- Найден и исправлен баг `candle-kernels/build.rs`: он понижал PTX header до
  `.version 8.6`, оставляя `.target sm_121f`. Драйвер отвечал
  `CUDA_ERROR_INVALID_PTX`. Для `sm_121` downgrade теперь не выполняется,
  PTX остаётся `.version 9.0`.

## Модель

`ornith-ai/Ornith-1.5-9B-GGUF`, файл `Ornith-1.5-9B-Q4_K_M.gguf`:

- 5,780,090,816 B;
- SHA-256 `70c112196e0b7023803c9762752e46d29e612a92c83f995bc3ba1ceb07e8fab6`;
- architecture `qwen35`, native context 262144;
- served context на стенде: 131072, 4 slot.

## Конфигурация сервера

`CTX=131072`, `CONTEXT_LIMIT=131072`, `SLOTS=4`, `GPU_ONLY=1`,
`GPU_LAYERS=999`, `KV_CACHE_TYPE=q8_f16`, `KV_POOL_Q8=1`, `PGRAPH=on`,
`PGRAPH_MIN_T=1000000`, `CUDA_GRAPHS=1`, `FLASH_ATTN=1`,
`PREFIX_CACHE_MIB=2048`, `PREFILL_CHUNK=8192`, `MTP=0`.

Время от запуска процесса до `GET /v1/models = 200`: **15.14 s**.
`/v1/models`: 200, WebUI `/`: 200 (70,530 bytes).

## Методика

Один и тот же HTTP-харнесс `scripts/bench/dgx_spark/http_bench.py`:
temperature 0, 65 completion tokens, уникальный prompt на каждый прогон,
два warmup-запроса, три измерения на точку и медиана. Prompt tokens берутся
из `usage.prompt_tokens`; завершённые токены — из `usage.completion_tokens`.
TTFT включает префил.

На GB10 `nvidia-smi` возвращает `memory.used=N/A`, поэтому вместо VRAM
записаны пик `VmRSS` процесса сервера и минимум `MemAvailable` хоста.

## Результаты

| Target | Prompt tokens | TTFT, s | Prefill tok/s | Decode tok/s | RSS max, MiB | MemAvailable min, MiB |
|---:|---:|---:|---:|---:|---:|---:|
| 2K | 1,958 | 1.071 | 1,827.42 | 37.60 | 6,587 | 99,790 |
| 8K | 7,645 | 3.871 | 1,975.16 | 36.90 | 6,738 | 97,908 |
| 32K | 30,396 | 15.071 | 2,016.80 | 34.36 | 6,902 | 97,448 |
| 64K | 60,730 | 31.486 | 1,928.78 | 31.45 | 7,112 | 97,158 |
| 128K | 121,397 | 70.548 | 1,720.78 | 27.11 | 7,330 | 96,739 |

Пик prefill — около 2K tok/s на 32K. На 128K скорость падает до 1.72K tok/s,
decode — с ~37.6 до 27.1 tok/s.

## Устойчивость

20 последовательных запросов по ~7.6K prompt tokens: **20/20 успешно**.
Медианы: prefill 1,953.75 tok/s, decode 36.83 tok/s. Ошибок и падений нет.
RSS вырос с ~7.35 GiB до ~8.1 GiB по мере заполнения prefix cache
(`[pcache] ... 2013MiB`), после чего пять дополнительных запросов держали
RSS на уровне ~8.11 GiB — продолжающейся утечки не видно.

## DeepSeek V4.1 Flash 1017 UNCENSORED FP8

Запрошенный `dealignai/DeepSeek-V4.1-Flash-1017-UNCENSORED-FP8` имеет
510.3 GB на диске (475.3 GiB), 763B параметров и официальный runtime с
tensor parallel на 4×H200. Base weights без Engram — 286.1 GiB, больше всей
доступной unified memory GB10 (121 GiB). Поэтому нативный FP8 на одном Spark
не запускается. Подробный разбор и варианты:
[2026-10-19-dgx-spark-deepseek-v41-uncensored-fp8.md](2026-10-19-dgx-spark-deepseek-v41-uncensored-fp8.md).

## Ограничения

- `nvidia-smi` не даёт VRAM на GB10; RSS и MemAvailable являются прокси, а не
  точным GPU-only resident footprint.
- Опубликованный VQ-GGUF для 1017 — base, не UNCENSORED-вариант.
- Сравнение с llama.cpp/vLLM на этом стенде не снималось; цель этапа — проверить
  саму сборку и получить baseline Forge.

## Четыре параллельных слота

Четыре одновременных запроса по 7,629 prompt tokens завершились 4/4:
per-request prefill около 506 tok/s, decode около 30.7 tok/s, общий prefill
примерно 2,024 tok/s, aggregate decode 122.8 tok/s, wall time 17.16 s.
