# Open WebUI → DeepSeek V4.1 Flash на dgx-forge

Open WebUI работает **на сервере** (порт 8080), модель — локальный
`ds4-server` (порт 8000). На Mac ничего ставить не нужно.

## Запуск

```bash
# 1. модель (если не запущена)
ssh -i ~/.ssh/id_ed25519_gpuwerk root@dgx-forge.gpuwerk.com \
  'tmux new-session -d -s ds4 /workspace/forge-run/serve_win.sh'

# 2. Open WebUI
ssh -i ~/.ssh/id_ed25519_gpuwerk root@dgx-forge.gpuwerk.com \
  'tmux new-session -d -s owuirun "bash /workspace/run_owui.sh"'

# 3. туннель на порт 8080
ssh -N -L 8080:127.0.0.1:8080 -i ~/.ssh/id_ed25519_gpuwerk \
    root@dgx-forge.gpuwerk.com
```

Открыть <http://127.0.0.1:8080> — авторизации нет.

## Что настроено

`/workspace/run_owui.sh`:

| Переменная | Значение |
|---|---|
| `OPENAI_API_BASE_URL` | `http://127.0.0.1:8000/v1` |
| `DEFAULT_MODELS` | `deepseek-v4.1-flash-1017-uncensored-vq` |
| `WEBUI_AUTH` | `false` (без логина) |
| `ENABLE_OLLAMA_API` | `false` |
| `OFFLINE_MODE` | `true` (без обращений в интернет) |
| `DATA_DIR` | `/workspace/owui-data` |

Установлен в `/workspace/owui-venv` (Python venv, без Docker — на сервере
Docker нет). Версия 0.12.3.

Ошибка `SentenceTransformer` в логе при старте безвредна: это RAG-эмбеддинги,
они выключены (`OFFLINE_MODE=1`), чат от них не зависит.

## Ограничения

- RAG/веб-поиск/embeddings не работают (offline). Для чата с моделью не нужны.
- Один пользователь, авторизация выключена. Слушает `127.0.0.1`, снаружи
  доступен только через SSH-туннель.
- Прокси `serve.py` и `index.html` удалены — они больше не нужны.
