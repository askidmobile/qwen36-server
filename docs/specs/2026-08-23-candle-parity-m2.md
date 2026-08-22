# Specification: candle vs llama.cpp — путь до паритета (M2)

**Date:** 2026-08-23
**Priority:** P1
**Type:** Performance / roadmap execution
**Baseline:** fix f6cac6b1 (bounded mempool threshold + shared f16 scratch):
36.8 ток/с @8K bench, 11.2 ток/с @24K, TTFT-prefill @24K ≈ 150–275 s.

## 1. Problem

После фикса WDDM-коллапса candle остаётся позади llama.cpp на трёх осях:

| Ось | candle сейчас | llama.cpp (эталон, та же карта/модель) |
|---|---|---|
| Декод @24K | 11.2 ток/с | не измерен чисто (оценочно 15–25) |
| Prefill | ~160–256 ток/с (TTFT @24K ≈ 150–275 s) | ~850 ток/с (TTFT @24K ≈ 30 s) |
| Память MoE | только full-VRAM кванты | `--n-cpu-moe`: Q4_K_XL качество на 12 GB |

Решение BD-001 подтверждено владельцем: остаёмся на candle, llama.cpp — только
эталон для замеров. WebUI переезжает на Unsloth Studio (см. §6), агенты ходят
напрямую в наш сервер.

## 2. Goal

Декод и префилл на гибриде DeltaNet ≥ llama.cpp на целевых сценариях
(1–2 слота, ctx 8–32K, RTX 3060 12 GB), сохранение всех наших преимуществ
(3 API, транзакционный MTP, VRAM-планер, media lifecycle).

## 3. Фазы

### P0 — Спекулятивный декодинг MTP на 35B-A3B (есть, включить и измерить)
Реализован (BD-023): транзакционный draft/verify/commit, B=1..4,
fixed-seed token-for-token паритет. Не включён на 35B-A3B: нет артефакта
`unsloth/Qwen3.6-35B-A3B-MTP-GGUF`.
- FR-P0.1: скачать артефакт, подключить через профиль (`mtp` component).
- FR-P0.2: замерить прирост на 8K/24K протоколе (ожидание ×1.15–1.25 для MoE
  по данным Unsloth; dense даёт ×1.4–2).
- FR-P0.3: проверить VRAM-fit c MTP (+~1 GB) при BUDGET_FRAC 0.89.

### P1 — Инкрементальное f16-зеркало KV (Stage-2)
Убирает per-step деанвализационный fill (остаточные 11 против 36 ток/с).
- FR-P1.1: per-(layer,slot) плоское f16-зеркало, ленивая аллокация под глобальный
  бюджет `QWEN36_KV_MIRROR_MIB` (default 640 MiB).
- FR-P1.2: инкрементальный top-up — на каждом шаге деквантируются ТОЛЬКО новые
  строки (обычно 1 токен), декод читает zero-copy view.
- FR-P1.3: инвалидация на bulk-перезаписях: `shift_and_append` (eviction),
  `seed_slot_batched`; рост кэша → расширение зеркала или graceful fallback.
- FR-P1.4: каскад fallback: зеркало → shared scratch (Stage-1) → `f16_prefix`.
- Acceptance: декод @24K (1 активный слот) ≥ 25 ток/с; @8K bench без регресса
  (<5%); бит-в-бит то же продолжение при fixed-seed (значения K/V идентичны —
  меняется только способ их получения).

### P2 — Prefill через flash-attention varlen
- FR-P2.1: prefill-путь attention (seq>1) перевести с materialized
  scores+softmax на `candle_flash_attn` (в декоде уже используется).
- FR-P2.2: убрать промежуточные F32-расширения; F16 scores где возможно
  (T-288 identity уже описан в коде).
- Acceptance: prefill ≥ 500 ток/с на 24K промпте (TTFT ≤ 48 s).

### P3 — Fused/chunked DeltaNet prefill
- FR-P3.1: DeltaNet prefill чанками без token-by-token syncs (FR-021 из спеки
  2026-08-10; частично сделано — `[pf] chunk blocks` в логах).
- Acceptance: вклад DeltaNet-блоков в префилл ≤ внимания.

### P4 — CUDA graphs decode
- FR-P4.1: отладить существующий graphed path (`QWEN36_CUDA_GRAPHS=1`,
  `paged_kv_cuda`) на 35B-A3B; включать по умолчанию при выигрыше ≥10% decode.
- Acceptance: decode-step wall ≤ 0.9× eager при том же выводе.

### P5 — Expert offload для MoE (аналог `--n-cpu-moe`)
- FR-P5.1: раскладка тензоров по устройствам при загрузке (эксперты FFN → RAM
  mmap, внимание/DeltaNet/shared → GPU).
- FR-P5.2: CPU-матмул квантованных экспертов + async prefetch активных
  экспертов на GPU (pipeline поверх CUDA pool).
- FR-P5.3: VRAM-планер учитывает offload-профиль.
- Acceptance: Qwen3.6-35B-A3B UD-Q4_K_XL работает на 12 GB карте с декодом
  ≥ 15 ток/с — качество выше IQ2_XXS при сопоставимой скорости.
- Оценка: крупнейший блок работ (недели); делать после P0–P2.

## 4. Non-goals

- Поддержка новых бэкендов (Vulkan/SYCL).
- Классический two-model speculative decoding (чужой draft) — MTP закрывает.
- Паритет экосистемы llama.cpp (tools/server фичи) — наш сервер уже шире в своих
  доменах (3 API, media, профили).

## 5. Риски

| Риск | Митигация |
|---|---|
| Зеркало съедает VRAM при 2 длинных слотах | глобальный бюджет + graceful fallback на Stage-1 |
| FA2 prefill численный дрейф | parity-тест fixed-seed на коротких промптах |
| MTP + зеркало взаимодействие (verify читает seq>1) | verify идёт через отдельный multi-token путь — покрыть тестом |
| Регрессия prompt-cache/migrate путей | инвалидация зеркала во всех bulk-writers, список в плане |

## 6. WebUI → Unsloth Studio (параллельный трек, вне этой спеки)

W1: подключить Studio к нашему `/v1` как внешний provider; W2: закрыть
несовместимости API; W3: `web/` — admin-fallback, Studio — основной UI.
Деталь — отдельная спека после W1-обследования.
