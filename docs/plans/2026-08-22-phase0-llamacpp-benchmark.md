# Phase 0: замер llama.cpp против candle-движка (re-eval BD-001)

**Дата:** 2026-08-22
**Статус:** ожидает прогона на yttri-win
**Предпосылки:** см. обсуждение 2026-08-22 — Qwen3.6-35B-A3B на candle упирается в VRAM,
RAM не используется, скорость 3–4 ток/с; владелец сомневается в BD-001.

## 1. Зачем

- Текущий движок кладёт **все веса в VRAM** (спека 2026-08-10, D-001 отклонил host-offload;
  `BUDGET_FRAC=0.93` в `src/vram_plan.rs` — «работает впритык» при 98% карты → WDDM paging → коллапс).
  Для dense 27B это осмысленно, для MoE 35B-A3B — нет.
- llama.cpp master уже поддерживает `qwen35`/`qwen35moe` (DeltaNet), MTP (`--spec-type draft-mtp`),
  vision (mmproj) и **все три наших API** (`/v1/chat/completions`, `/v1/responses`, `/v1/messages`),
  плюс `-ncmoe/--n-cpu-moe` и `-ot exps=CPU` — готовый механизм «эксперты в RAM, остальное на GPU».

## 2. Гипотеза

`llama-server -ngl 99 -ncmoe N` на RTX 3060 12 GB + системная RAM даёт на
Qwen3.6-35B-A3B UD-Q2_K_XL **≥5x** к текущим 3–4 ток/с (ожидание: 15–25 ток/с B=1),
без деградации от paging. Если подтвердится — фиксируем override BD-001 и выбираем
сценарий A (полный переход + Unsloth Studio UI) или B (гибрид с тонким Rust-гейтом).

## 3. Матрица прогонов

Бенч один и тот же — `scripts/bench.ps1` (OpenAI API). Каждая строка ~2–3 минуты.

| # | Конфиг | Команда |
|---|--------|---------|
| 0 | candle baseline (наш сервер, :18099) | `bench.ps1 -BaseUrl http://localhost:18099 -ApiKey smoke-key -Concurrent 2` и `-Concurrent 4` |
| 1 | llama.cpp: все эксперты в RAM | `phase0_llamacpp.ps1 -ModelPath <gguf> -DownloadLlama -RunBench -Ncmoe 999` |
| 2 | llama.cpp: часть экспертов на GPU | `phase0_llamacpp.ps1 ... -Ncmoe 28` (из 40 блоков) |
| 3 | llama.cpp: больше на GPU | `phase0_llamacpp.ps1 ... -Ncmoe 20` |
| 4 | (опц.) MTP GGUF | скачать `unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-Q2_K_XL`, запустить с `-ExtraArgs --spec-type,draft-mtp,--spec-draft-n-max,2` |

Сравнивать строки между собой только при **одинаковом `-Concurrent`** (число слотов).

## 4. Фиксируем в результаты (таблица в этот файл)

| Конфиг | decode B=1 ток/с | aggregate B=N ток/с | TTFT ms | VRAM MiB | RAM GiB |
|--------|------------------|---------------------|---------|----------|---------|
| candle (baseline) | | | | | |
| llama -ncmoe 999 | | | | | |
| llama -ncmoe 28 | | | | | |
| llama -ncmoe 20 | | | | | |

## 5. Правило решения

- **Гипотеза подтвердилась (≥5x, без paging):** пишем в decisions.md строку
  «BD-028 overrides BD-001/BD-002: инференс — llama-server как отдельный компонент;
  candle-fork-qwen35-batch архивируется». Далее выбор A/B для API-гейта и UI.
- **Подтвердилось частично (2–5x):** тот же override, но тестируем кванты выше
  (UD-Q4_K_XL в RAM+VRAM суммарно) и MTP перед выбором конфигурации по умолчанию.
- **Не подтвердилось (<2x):** возвращаемся к сценарию C (offload в candle) с данными
  о реальном узком месте.

## 6. Примечания

- RAM на yttri-win должна быть ≥ 32 GB для комфортного `-ncmoe 999` (Unsloth указывает
  ~17–23 GB суммарно RAM+VRAM для 2–4 бит). Скрипт печатает объём RAM при старте.
- Если llama-server ругается на `-fa on` (гибридная arch) — убрать флаг из `$sargs`.
- Vulkan/HIP-бэкенды пока глючат на qwen35 — используем только CUDA (yttri-win).
- Unsloth Studio (UI) ставится параллельно: умеет подключать наш/llama-server как
  внешний провайдер (Settings → Connections → Add Provider → Base URL) — проверяем
  отдельно от бенча.
