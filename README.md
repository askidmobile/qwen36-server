# Qwen3.6 27B — candle inference server

Rust-сервер инференса **Qwen3.6-27B** (GGUF, 2-bit) на собственном candle-форке
[candle-fork-qwen35-batch](../candle-fork-qwen35-batch). Три API
(Chat Completions, Responses, Messages) + веб-чат. 4 одновременных клиента через
batch-планировщик. Целевая площадка: Windows + CUDA (yttri-win, RTX 3060 12 GB).

- Модель: [unsloth/Qwen3.6-27B-GGUF](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF), стартовый квант **UD-Q2_K_XL (11.8 GB)**, фаза 2 — IQ2.
- Контекст: **81 920** токенов на старте (нативный потолок модели 262 144).
- llama.cpp не используется как компонент — только эталон для parity-замеров.

## Статус

Собранный сервер `qwen36-server` работает: три API + веб-чат, smoke на
реальной GGUF (Qwen3.5-4B) пройден — chat/stream/messages/responses/models.
См. [docs/brief/PROJECT-BRIEF.md](docs/brief/PROJECT-BRIEF.md)
(видение, scope, риски, дорожная карта) и [docs/brief/decisions.md](docs/brief/decisions.md)
(обязательные решения BD-001…BD-020).

**2026-08-09 — BatchedEngine на yttri-win:** 4 слота × Qwen3.6-27B
**IQ2_XXS** — 4 конкурентных запроса завершились корректно (70 tok каждый,
finish=stop, VRAM 9.9 GB стабильно, decode step B=4 = 2.08s).
Важно: на **Q2_K_XL** (11.8 GB, VRAM 12022/12288 MiB) decode деградирует до
13.4s/step из-за WDDM paging — 4-слотовый режим требует VRAM-запаса,
на 12 GB карте это IQ2_XXS. Включённый раньше срок — фаза 5 по BD-019
(IQ2-загрузка уже работает через dequant+cuBLAS fallback).

## Запуск

```bash
# macOS (разработка, Metal)
cargo build --release --features metal
# Windows (yttri-win, CUDA): cargo build --release --features cuda
# Linux (CUDA, напр. RTX 4090): scripts/build_linux.sh

export QWEN36_API_KEY=<ключ>
export QWEN36_MODEL=<путь к GGUF>   # напр. D:\models\Qwen3.6-27B-UD-Q2_K_XL.gguf
./target/release/qwen36-server     # слушает 0.0.0.0:8080
```

## Linux + CUDA (арендованный GPU-сервер)

`scripts/build_linux.sh` — полный цикл: клонирует форк соседней директорией
(если нет), проверяет nvcc/cargo, собирает с `--features cuda`.
Path-зависимости относительные (`../candle-fork-qwen35-batch`), раскладка:

```
<workdir>/candle-fork-qwen35-batch   # ветка feat/qwen35-batching
<workdir>/Qwen3.6 27B                # этот репо
```

`CUDA_COMPUTE_CAP`: 89 = RTX 4090 (default в скрипте), 86 = 3060, 90 = H100.
На Linux нет WDDM paging — Q2_K_XL 27B (11.8 GB) и Q4_K_M (15.4 GB) спокойно
живут на 24 GB, деградации как на Windows не будет.

Опционально: `QWEN36_HOST`, `QWEN36_PORT` (8080), `QWEN36_CTX` (81920), `QWEN36_SLOTS` (4).
Веб-чат: `http://<host>:8080/`.

## Тесты

```bash
cargo test --features metal        # 15 lib + 10 API integration
scripts/stability_smoke.sh         # критерий BD-008 (на yttri-win)
```

## Производительность

Бенчмарк на yttri-win (RTX 3060 12 GB, CUDA 12.4, driver 591.86).
Модель: Qwen3.6-27B Q4_K_M (15.4 GB GGUF), ctx 8192, 4 слота.
Скрипт: `scripts/bench.ps1` (HttpWebRequest для real-streaming TTFT, runspaces для
конкурентности).

Примечание: модель превышает VRAM 12 GB — работает через системный RAM swap
(working set ~20 GB), поэтому скорости низкие. Ожидаемый режим после фазы 5
(IQ2-ядра) — Q2_K_XL/IQ2_XXS полностью в VRAM, скорости вырастут на порядок.

- **TTFT** (time to first token, single-slot stream): 2595 ms
- **Decode** (single-slot, 128 tok): 1.1 tok/s
- **Prefill** (2013 tok prompt): 8.2 tok/s
- **4-slot concurrent** (64 tok each): aggregate 1.5 tok/s (per-slot 0.4 tok/s)
- **VRAM**: 12032 / 12288 MiB (99% — swap-bound), GPU util 100% под нагрузкой

Прогон:`scripts/bench.ps1 -BaseUrl http://localhost:18099 -ApiKey <key>`

## Стек

Rust · candle (форк `candle-fork-qwen35-batch`, path-зависимость) · CUDA/Metal/CPU.

## Пайплайн

```
✅ /create-brief → docs/brief/            ← пройдено
⬜ /create-spec <фича>                    (по каждому пункту дорожной карты)
⬜ /create-spec-plan → /create-spec-implement → /create-spec-review
```

## Осталось (дорожная карта брифа)

- Фаза 4: реальный BatchScheduler в `BatchedEngine` (скелет + дизайн готовы;
  нужны 2 мини-патча в форк — TODO-F5 indexed sampler, TODO-F6 slots_mut()).
- Фаза 5: IQ2-ядра (CPU dequant + CUDA/Metal matmul), переход Q2_K_XL → IQ2_XXS/IQ2_M.
- Прогон на yttri-win: Q2_K_XL Qwen3.6-27B + stability smoke (критерий v1).
