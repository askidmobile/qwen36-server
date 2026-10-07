# Ornith-1.5-35B-A3B: харнессы стенда forge-gpu

Скрипты сняты со стенда `forge-gpu` 2026-10-07, чтобы сравнение yforge ↔
llama.cpp ↔ vLLM было воспроизводимым после выключения сервера. До этого жили
только в `/root`.

Результаты и разбор: `docs/research/2026-10-07-ornith-decode-win-over-llamacpp.md`.

## Стенд

| компонент | значение |
|---|---|
| GPU | NVIDIA GeForce RTX 4090, 49 140 МиБ |
| Драйвер / CUDA драйвера | 580.65.06 / 13.0 |
| Тулчейн сборки | CUDA 12.8, `CUDA_COMPUTE_CAP=89` |
| CPU | AMD EPYC 9554, выделено 12 vCPU; RAM 62 ГиБ |
| Модель | `/root/models/Ornith-1.5-35B-Q8_0.gguf` (36 ГиБ, окно 262 144) |
| vLLM | 0.28.0, модель FP8 `/root/models/ornith-fp8` |

Все пути захардкожены под layout стенда (`/root/...`). При переносе поправить
константы в шапке скрипта.

## Скрипты

| файл | что делает |
|---|---|
| `ornith_ab.py` | yforge или llama.cpp: TTFT, префил, декод, dVRAM; сэмплинг greedy |
| `ornith_ab_fair.py` | то же, но одинаковые поля сэмплинга у обоих (temp=0.6, top_k=20, top_p=0.95, seed=7) |
| `ornith_ab_topk.py` | развёртка по `top_k` при фиксированном числе потоков отбора |
| `vllm_bench.py` | метрики для уже поднятого vLLM |
| `three_way2.sh` | vLLM → yforge → llama.cpp подряд на одном окне |
| `interleave.sh` | чередование yforge/llama.cpp (контроль дрейфа) |
| `start_ttft.py` | «старт процесса → первый токен» для обоих движков |
| `determinism20.py` | 20 одинаковых запросов в одном процессе |
| `load_loop.py` | нагрузочный цикл для профилей nsys |
| `pdec_tile.sh` | профиль nsys только по окну декода |
| `build_yf.sh` | сборка yforge на сервере (CUDA 12.8, sm_89) |

## Запуск

```bash
# трёхсторонний замер на 32K
nohup ./three_way2.sh 32768 16384 64 3 > /root/logs/3way-32k.log 2>&1 &

# yforge против llama.cpp с одинаковым сэмплингом
CTX=32768 PROMPT_TOKENS=16384 MAXTOK=96 REPS=3 ENGINE=yforge VARIANT=yf \
  EXTRA='{"PREFILL_CHUNK":"4096","SAMPLING_LOCK":"0"}' PORT=18820 \
  python3 ornith_ab_fair.py
CTX=32768 PROMPT_TOKENS=16384 MAXTOK=96 REPS=3 ENGINE=llamacpp VARIANT=ll \
  EXTRA='{}' PORT=18820 python3 ornith_ab_fair.py

# старт -> первый токен
python3 start_ttft.py
```

## Грабли

- `PREFILL_CHUNK=4096` обязателен: на дефолтных 512 префил падает с ~7600 до
  ~5000 t/s.
- `CUDA_GRAPHS=1` обязателен: без графов декод падает с ~153 до ~64 t/s.
- Уникальный промпт на каждый повтор: иначе prompt cache llama.cpp даёт
  фальшивый TTFT (наблюдалось 24k t/s против реальных 4.9k).
- `SAMPLING_LOCK=1` подставляет model-card (temp=0.6, top_k=20) независимо от
  запроса — для честного сравнения либо снять лок, либо брать
  `ornith_ab_fair.py`.
- Движки не должны работать одновременно: перед запуском дренировать GPU до
  ≤400 МиБ.
