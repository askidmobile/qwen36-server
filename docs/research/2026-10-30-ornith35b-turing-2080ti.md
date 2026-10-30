# Ornith-1.5-35B-A3B Q4_K_M на RTX 2080 Ti (Turing, sm_75) — 2026-10-30

Стенд: сервер Михаила `tinycoder.dev:24` (anton), RTX 2080 Ti 11 ГиБ, CUDA 12.4,
2× Xeon E5-2686 v4 (36 ядер / 72 потока), 125 ГиБ RAM. Модель
`/data/models/ornith/Ornith-1.5-35B-Q4_K_M.gguf` (21 ГиБ, 40 trunk-блоков,
256 экспертов × 8 активных, head_dim 256, kv_heads 2).

## Результат: модель впервые запущена на Turing

Два блокирующих дефекта найдены и исправлены (yttri-forge main):

1. **memlock 8 МиБ** (`ulimit -l` на арендованном стенде, sudo недоступен) —
   pinned `malloc_host(DEVICEMAP)` падал на первом же выделении.
   Исправление: `HostBuf::alloc` теперь fallback на
   `cuMemAllocManaged(CU_MEM_ATTACH_HOST)` — страницы остаются на хосте,
   zero-copy семантика сохраняется (ядро decode читает через PCIe,
   `memcpy_htod` prefill работает) — коммит `90642856`.
   Цена: эффективная полоса host-трафика ~2,4–4 ГБ/с против 10,6 ГБ/с у pinned.
2. **`fused_prep`-префил звал FlashAttention без turing-гейта** (head_dim==256
   у 35B включает эту ветку; у 9B head_dim 128 — ветка не активировалась,
   поэтому на 9B дефект не проявлялся). В cuda-turing сборке FA - это
   turing-stub, который паникует. Исправление: на Turing ветка идёт через
   пер-строчный `dispatch_flash_decode_paged` (строка i видит префикс
   kv0+i+1), без Turing — прежний FA — коммит `667399a3`.
   Ограничение: `MAX_VERIFY_ROWS=512` → на Turing `PREFILL_CHUNK=512`.

Профиль запуска `/tmp/mike35.env`: `MOE_EXPERTS=auto` (эксперты 18,6 ГиБ в
RAM), `CTX=131072`, `KV_POOL_Q8=0`, `PREFILL_CHUNK=512`, `CUDA_GRAPHS=1`,
`MTP=0`. VRAM: ствол+выход 1,3 ГиБ + KV-пул 1,6 ГиБ + кэш экспертов ~2,7 ГиБ
+ стейджинг 0,5 ГиБ.

## Бенч (505-токенный промпт, 128 токенов, temp=0)

| Движок | CTX | Prefill, т/с | Generation, т/с |
|---|---|---|---|
| llama.cpp (experts→CPU, 36 потоков) | 8192 | 222,7 | **26,5** |
| yforge | 8192 | **244,7** | 15,52 |
| yforge | 131072 | 208,3 | 10,64 |

Prefill yforge выигрывает у llama.cpp (+9,9 % на 8k). Generation проигрывает:
llama.cpp считает экспертов на CPU, у нас хост-чтение промахов через managed.

## Эксперименты promotion (CTX=8192)

| EXPERT_PROMOTE_PER_STEP | Hit | Gen, т/с |
|---|---|---|
| 0 (только zero-copy) | 42 % | 10,35 |
| 64 (дефолт) | 90,5 % | **15,52** |
| 128 | 90,5 % | 10,66 |

## Prefill на длинном контексте (CTX=32768)

| Промпт | Prefill, т/с |
|---|---|
| 4 095 | 298 |
| 8 191 | 253 |
| 16 383 | 193 |

Квадратика пер-строчного decode мягкая: ~3,4 + 0,148·T мкс/токен.
Экстраполяция: 128k ≈ 45–55 мин (идёт фоновая валидация), 256k ≈ 3 ч.

## Что даёт основной выигрыш generation

1. **Поднять memlock** (Mikhail: `ulimit -l unlimited` для anton или запуск от
   root/systemd) → pinned 10,6 ГБ/с вместо managed ~3 ГБ/с. Оценка decode:
   25–40 т/с на CTX=8k — обгон llama.cpp. Флаг никуда не девается: pinned
   остаётся основным путём, managed — только fallback.
2. **Асинхронный promote** (отдельный CUDA-стрим, перекрытие копий с compute) —
   минус ~15–20 мс/шаг.
3. **CPU-путь экспертов** (как у llama.cpp) — дни работы, план Б для
   CPU-богатых хостов.
4. **Полноценный prefill-kernel sm75** (тайловый FA2-подобный, mma.m16n8k8 +
   cp.async Turing) — снимет квадратику на 128k/256k prefill; сейчас пер-строчный
   путь — единственный корректный без FA.
