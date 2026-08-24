# Qwen3.5-4B Benchmark: candle-fork vs llama.cpp (2026-08-24)

RTX 3060 12GB, CTX=16384, SLOTS=2 (batched engine), чистый декод (ток/с),
prefill = wall max_tokens=1. Промпт "word "*N + инструкция, 64 токенов генерации.
llama.cpp b10472 (Unsloth), -ngl 999, CUDA graphs on (default).

## Матрица

| Квант      | Движок     | ~0 ctx | 2K    | 8K    | Prefill 8K |
|------------|-----------|--------|-------|-------|------------|
| Q4_K_M     | Ours      | 62.6   | 59.3  | 56.4  | 2.19s      |
|            | llama.cpp | ~60*   | ~102  | 58.8  | 1.57s      |
| Q6_K       | Ours      | 50.0   | 50.8  | 46.0  | 2.32s      |
|            | llama.cpp | 70.9   | ~148  | 67.2  | 1.73s      |
| Q8_0       | Ours      | 46.6   | 48.1  | 44.5  | 2.17s      |
|            | llama.cpp | 60.3   | ~103  | 58.8  | 1.57s      |
| UD-Q4+MTP  | Ours +MTP | 57.5   | 56.8  | 47.3  | 2.37s      |
|            | Ours -MTP | 58.7   | 56.9  | 51.1  | -          |
|            | llcpp+d-mtp | 144  | ~200+ | 125.6 | -          |

* дифф-метод; точных timings у llama.cpp для 0ctx нет.

## Выводы

1. Наш декод слабо зависит от длины контекста (62→44..56); у llama.cpp
   падает заметнее на Q4_K_M (90→59).
2. llama.cpp быстрее по декоду x1.1-3 (CUDA graphs default + зрелые ядра).
3. MTP: наш acceptance 79% (width-based) не даёт выигрыша на 4B
   (47 vs 51 без MTP @8K). llama.cpp draft-mtp: acceptance 94%, mean len
   3.76 -> x2.1 ускорение (125.6 vs 59).
4. Prefill у нас x1.3 медленнее (cudarc host launch overhead ~4ms/launch).
5. SLOTS=1 использует старый CandleEngine (18 tok/s @8K!) — всегда SLOTS>=2.

## Конфиги

Ours:
  MODEL=<gguf> CTX=16384 SLOTS=2 API_KEYS='[{"key":"x"}]'
  [MTP=1 QWEN36_MTP_PATH=<тот же gguf с nextn>]

llama.cpp:
  llama-server.exe -m <gguf> -c 16384 --port 18098 --host 127.0.0.1 -ngl 999 --api-key x
  [+ --spec-type draft-mtp]

Модели: D:\Models\yttri\qwen3.5-4b\Q4_K_M, D:\Models\lmstudio-community\Qwen3.5-4B-GGUF\{Q6_K,Q8_0},
D:\Models\unsloth\Qwen3.5-4B-MTP-GGUF\UD-Q4_K_XL (содержит nextn blk.24).

Скрипты: D:\Projects\yttri-inference\scripts\bench_4b.ps1 <ours|llamacpp> <model> [mtp]


## Дополнение 2026-08-24: MTP deep-dive (фазовые тайминги + adaptive width)

Фазовый разбор MTP-раунда (QWEN36_MTP_TIMING=1, UD-Q4 @8K):
  begin ~1ms | draft 22ms (3 последовательных MTP-forward по ~7.3ms) |
  verify 29ms (батчевый target-forward K=4 — эффективен) |
  accept 0–13ms (restore DeltaNet-checkpoint + re-run при m<K)

Неудачный раунд (m=1) стоит ~65мс за 1 токен против 19.6мс baseline.

Реализован P0.5c adaptive width (форк 3bf49d56, откат QWEN36_MTP_ADAPTIVE=0):
K следует за принимаемостью — m>=K наращивает, m<=1 срезает и пропускает раунд
(probe каждый 4-й шаг). Результат: MTP больше не деградирует относительно
baseline (фикс K=8 давал 17.8 ток/с vs 44.9 baseline; adaptive не ниже baseline).

Итоговая матрица ours @8K (SLOTS=2): без MTP 44.9–56.4, +MTP adaptive 43.6–47.3.
MTP на 4B в текущей реализации нейтрален: draft-цикл host-bound (cudarc
launch overhead ~7мс/forward). Путь к llama.cpp-уровню (125 ток/с): CUDA-graphed
draft path. На больших моделях (27B/35B) draft дороже относительно verify —
адaptive там даёт больший выигрыш.


## Дополнение: Qwen3.8-27B (2026-08-24)

⚠️ УРОК: сервер имеет auto-switch по req.model — бенч с model='qwen3.5-4b'
на загруженном qwen3.8 молча переключал движок на 4B! Первые «27B» замеры
(52-57 ток/с) были на самом деле 4B. Честные числа ниже (model='qwen3.8-27b').

Qwen3.8-27B UD-IQ2_XXS (8.39 GB), архитектура совместима с форком (qwen35),
RTX 3060, SLOTS=2:

| Метрика          | Ours   | llama.cpp b10472 |
|------------------|--------|------------------|
| Decode ~0 ctx    | 18.4   | 22.4             |
| Decode 8K        | 16.7   | 21.1             |
| Prefill 8K       | 9.65s  | 8.7s             |
| +MTP @8K         | 15.1 (acc 70%) | —        |

Паритет с llama.cpp на 27B (декод x0.8, префилл x1.1). Качество генерации
проверено (Paris/France тест) — оба корректны.

Кванты для 12GB: IQ2_XXS единственный разумный (weights 8.2 + KV 2.1 + state
0.3 = 10.6 GiB). Q4_K_M 16.8 / Q6_K 22.4 / Q8_0 29 GB — только для 48GB.

Для RTX 6000 Ada 48GB: Q4_K_M влезает с большим контекстом; bandwidth ~2.6x
выше -> ожидаемый декод ~40-45 tok/s у нас. Наш сервер готов к архитектуре.
MTP-артефакт qwen3.8: unsloth/Qwen3.8-27B-GGUF/MTP/mtp-Qwen3.8-27B-Q4_0.gguf
(1.37 GB), включение: MTP=1 QWEN36_MTP_PATH=<mtp.gguf>.


## Дополнение: CUDA Graphs включаются только на BatchedEngine (2026-08-24)

Ключевая находка дня: QWEN36_CUDA_GRAPHS=1 работает ТОЛЬКО при SLOTS>=2
(BatchedEngine). Ранние тесты graphs на SLOTS=1 были невалидны.

Qwen3.5-4B Q4_K_M @RTX3060, SLOTS=2 + QWEN36_CUDA_GRAPHS=1:
  ~0ctx: 70.2 | 2K: 70.8 | 8K: 60.6 ток/с
vs без graphs: 62.6 / 59.3 / 56.4 (+13% и плоская кривая)
vs llama.cpp pure: ~88 / ? / 81.9

Итог по декоду: отставание x1.2-1.35 на длинном контексте сохраняется.
Паритета нет; причина системная — cudarc launch overhead вне графированного
участка (prefill, MTP draft) + paged-FA2 медленнее их FA2.

РЕКОМЕНДАЦИИ ПРОДАКШЕНА (12GB):
- Модели <=10GB весов: SLOTS=2 + QWEN36_CUDA_GRAPHS=1 (обязательно)
- 27B IQ2_XXS: SLOTS=2 БЕЗ graphs (paged F16 pool + q8 KV = VRAM double-dip,
  @8K коллапс 16.7 -> 2.3 ток/с)

ДЛЯ RTX 6000 ADA 48GB:
- graphs ON для всех квантов (Q4_K_M 16.8GB + pool влезает свободно)
- bandwidth x2.6 сокращает GPU-часть; graphed path убирает host overhead ->
  есть шанс реального паритета/преимущества (наш fused DeltaNet + батчевый
  MTP verify архитектурно сильнее их последовательного MTP).
