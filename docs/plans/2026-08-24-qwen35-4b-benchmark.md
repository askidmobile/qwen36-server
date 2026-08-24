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
