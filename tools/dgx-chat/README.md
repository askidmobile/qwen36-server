# Чат с DeepSeek V4.1 Flash UNCENSORED VQ на dgx-forge

Встроенного WebUI у `ds4-server` нет — он отдаёт только OpenAI-совместимый API
(`/v1/models`, `/v1/chat/completions`, `/metrics`, `/v1/stats`) и не ставит
CORS-заголовки. Поэтому здесь лёгкий UI (один `index.html`) плюс локальный
прокси на stdlib, который отдаёт UI с того же origin и перекладывает `/v1/*`
в SSH-туннель.

## Запуск

1. Сервер на dgx-forge (уже поднят; если нет — перезапустить):

```bash
ssh -i ~/.ssh/id_ed25519_gpuwerk root@dgx-forge.gpuwerk.com \
  'tmux new-session -d -s ds4 /workspace/forge-run/serve_win.sh'
```

2. Туннель (порт 8000 на сервере → 8000 локально):

```bash
ssh -N -L 8000:127.0.0.1:8000 -i ~/.ssh/id_ed25519_gpuwerk root@dgx-forge.gpuwerk.com
```

3. Прокси + UI:

```bash
cd tools/dgx-chat
python3 serve.py            # http://127.0.0.1:8787
```

Открыть <http://127.0.0.1:8787>.

## Что в конфигурации

Сервер запускается с `DS4_DSPARK_QUENCH=0` и `--mtp-mode on` — это
конфигурация из отчёта `docs/research/2026-10-20-dgx-spark-deepseek-40tok-achieved.md`
(среднее 46 tok/s decode). `DS41_ADMIT_X1024=1024` снимает ложные отказы
admission на длинных промптах.

## Замечания

- `reasoning_effort: none` выставляется в запросе: модель «тяжёлый
  рассуждатель», без этого она тратит токены на размышления до ответа.
- Температура 1.0 / top_p 1.0 — как рекомендует model card артефакта.
- `max_tokens` по умолчанию 512; для длинных ответов поднимите поле в UI.
- Прокси однопользовательский и слушает только 127.0.0.1.
