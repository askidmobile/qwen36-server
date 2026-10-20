# Open WebUI → DeepSeek V4.1 Flash на dgx-forge

Open WebUI работает **на сервере** (порт 8080), модель — локальный
`ds4-server` (порт 8000). На Mac ничего ставить не нужно.

## Запуск

```bash
# 1. модель
ssh -i ~/.ssh/id_ed25519_gpuwerk root@dgx-forge.gpuwerk.com \
  'tmux new-session -d -s ds4 /workspace/forge-run/serve_win.sh'

# 2. Open WebUI
ssh -i ~/.ssh/id_ed25519_gpuwerk root@dgx-forge.gpuwerk.com \
  'tmux new-session -d -s owuirun "bash /workspace/run_owui.sh"'

# 3. туннель
ssh -N -L 8080:127.0.0.1:8080 -i ~/.ssh/id_ed25519_gpuwerk \
    root@dgx-forge.gpuwerk.com
```

Открыть <http://127.0.0.1:8080> — авторизации нет.

## Две модели в списке

| Модель | Что делает |
|---|---|
| **чат (без рассуждений)** | `reasoning_effort: none`, отвечает сразу. Для обычных вопросов. |
| **reasoner (с рассуждениями)** | `reasoning_effort: high`, сначала думает (блок «Думаю…» раскрывается). Для сложных задач. |

Обе созданы в таблице `model` Open WebUI с preset-параметрами
(`scripts/mkmodels.py`), потому что ds4-server по умолчанию включает
thinking и `effort=high` — без этого модель уходила в многостраничные
рассуждения даже на «2+2» и интерфейс выглядел зависшим.

## Скорость генерации

Клик по времени сообщения (например «20 окт., 14:29») раскрывает панель с
метриками от ds4-server:

```
prompt_tokens: 6327      completion_tokens: 678
prefill_tok_s: 574.9     decode_tok_s: 40.4      tok_per_step: 3
ttft_ms: 19720
```

Этого не было: ds4-server отдавал `timings` только в non-streaming ответах,
а Open WebUI всегда стримит. Патч в `crates/ds4-server/src/stream.rs`
(`sse_usage_chunk`) добавляет тот же `timings` в финальный usage-чанк.

## RAG включён

`RAG_EMBEDDING_ENGINE=sentence-transformers` с локальной моделью
`all-MiniLM-L6-v2` (скачана один раз в `/root/.cache/huggingface`).
Работает поиск по загруженным файлам. Веб-поиск выключен
(`ENABLE_RAG_WEB_SEARCH=false`) — включается в Settings → Web Search.

## Файлы на сервере

| Путь | Что |
|---|---|
| `/workspace/owui-venv` | venv с Open WebUI 0.12.3 |
| `/workspace/owui-data` | база, настройки, загруженные файлы |
| `/workspace/run_owui.sh` | запуск Open WebUI |
| `/workspace/forge-run/serve_win.sh` | запуск ds4-server |
| `/workspace/mkmodels.py` | создание двух моделей с preset-параметрами |

## Ограничения

- Один пользователь, авторизация выключена. Слушает `127.0.0.1`, снаружи
  доступен только через SSH-туннель.
- Docker на сервере нет, поэтому venv. На Mac Docker удалён.
