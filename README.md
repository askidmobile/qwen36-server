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

## Запуск

```bash
# macOS (разработка, Metal)
cargo build --release --features metal
# Windows (yttri-win, CUDA): cargo build --release --features cuda

export QWEN36_API_KEY=<ключ>
export QWEN36_MODEL=<путь к GGUF>   # напр. D:\models\Qwen3.6-27B-UD-Q2_K_XL.gguf
./target/release/qwen36-server     # слушает 0.0.0.0:8080
```

Опционально: `QWEN36_HOST`, `QWEN36_PORT` (8080), `QWEN36_CTX` (81920), `QWEN36_SLOTS` (4).
Веб-чат: `http://<host>:8080/`.

## Тесты

```bash
cargo test --features metal        # 15 lib + 10 API integration
scripts/stability_smoke.sh         # критерий BD-008 (на yttri-win)
```

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
