# Запуск сервера

Бинарь называется **`yforge`** (`yforge.exe` на Windows). Логи идут в
stdout/stderr — перенаправляйте средствами оболочки, как у `llama.cpp`.

Настройки берутся из трёх источников, приоритет по убыванию:
флаги командной строки → окружение процесса → env-файл (`--env`).
Имена переменных без префикса (`CTX`, `SLOTS`) — их читают и сервер, и
движок. Формы `QWEN36_*` / `YTTRI_*` больше не поддерживаются. Полный список флагов — `yforge --help`
и [README.md](../README.md).

```bat
:: то же самое тремя способами
yforge.exe --model D:\Models\m.gguf --api-key KEY --ctx 131072 --slots 1
set CTX=131072 && yforge.exe --model D:\Models\m.gguf --api-key KEY
yforge.exe --env D:\configs\prod.env

:: проверить итоговую конфигурацию, не загружая модель
yforge.exe --env D:\configs\prod.env --dry-run
```

## Раскладка

| ОС | Скрипт | Что делает |
| --- | --- | --- |
| Windows (yttri-win, продакшен) | `windows/inference-run.bat` | экзешник + `.env` + лог в `logs\server.log` |
| Windows (профильная схема) | `windows/run_windows.bat` → `qwen35_run_current.ps1` | запуск релиза через `current`-указатель с проверкой SHA-256 профиля |
| macOS (разработка) | `macos/run-metal.sh` | `cargo run --release --features metal` |
| Linux (CUDA) | `linux/run-cuda.sh` | сборка при необходимости + запуск бинарника |
| Windows (yttri-win) | `windows/openwebui-install.ps1` | Open WebUI из исходников: clone, сборка, `.env`, задача, firewall, старт |
| Windows (yttri-win) | `windows/openwebui-run.bat` + `openwebui-run.ps1` + `openwebui-setup.ps1` | запуск Open WebUI как задачи планировщика |

Сборка отдельно: Windows — `../build_windows.bat` (VS 2022 + CUDA 13.2), Linux — `../build_linux.sh`.

## yttri-win — продакшен-хост (Windows, RTX 3060 12 GB)

Доступ: `ssh yttri-win` (конфиг в `~/.ssh/config`).

Рабочая директория: `D:\Projects\yttri-inference\qwen36-server` (копия этого репо).
Секреты: `D:\Projects\yttri-inference\qwen36-server\.env` (в git не ходит, бэкап `.env.bak-ornith`).
Экзешник: `D:\Projects\yttri-inference\qwen36-server\target\release\yforge.exe` — всегда свежее корневого `D:\Projects\yttri-inference\target\release`, сверяйте даты.
Модели: `D:\Models\<org>\<repo>\<file>.gguf`.

### Как сервер живёт

Вне ssh-сессий — **задача планировщика** `\qwen36-inference` (cmd → `inference-run.bat`,
cwd = `qwen36-server`, stdout+stderr → `logs\server.log`). `Start-Process` из ssh-сессии
умирает вместе с сессией — так не запускать.

### Запуск / перезапуск / остановка

```bat
:: запустить
schtasks /run /tn qwen36-inference

:: остановить (мягко — задача; жёстко — по PID)
schtasks /end /tn qwen36-inference
netstat -ano | findstr :18099
taskkill /PID <pid> /F

:: статус
schtasks /query /tn qwen36-inference
netstat -ano | findstr :18099
```

После старта читать `logs\server.log`: строка `[vram] total=... plan(dynamic)` —
раскладка VRAM, `loaded: id=... quant=...` — модель в памяти.

### Смена модели

1. Положить GGUF в `D:\Models\<org>\<repo>\`.
2. Правка `.env` (см. ключи ниже), затем перезапуск задачи.
   Горячая смена без рестарта — admin API `POST /admin/switch` (скан `MODELS_DIR`).

### Модель без готового GGUF: загрузка и квантование

Когда на HF лежат только `safetensors` (например `empero-ai/Qwen3.8-9B-Distill`):

```bat
:: 1. Мелкие файлы (конфиг, токенизатор) — hf.exe из venv llama.cpp
hf download <org>/<repo> --local-dir D:\Models\source\<repo> --exclude "*.safetensors"

:: 2. Веса — параллельно, N диапазонов (hf.exe на 20 ГБ встаёт, один curl ~1.5 МБ/с)
powershell -File scripts\windows\parallel-dl.ps1 ^
  -Url https://huggingface.co/<org>/<repo>/resolve/main/model.safetensors ^
  -OutFile D:\Models\source\<repo>\model.safetensors

:: 3. F16 + квант (F16 остаётся рядом — следующий квант без повторной конвертации)
scripts\windows\convert-hf-gguf.bat D:\Models\source\<repo> ^
  D:\Models\<org>\<repo>-GGUF <basename> Q4_K_M
```

Проверить перед запуском сервера: `qwen36_inspect.exe <gguf>` — `general.architecture`
(`qwen35` для семейства Qwen3.5/3.8), `block_count` (включает nextn-слой),
`tokenizer.ggml.pre`. HF не отдаёт sha256 для xet-файлов, поэтому целостность
проверяется заголовком safetensors: `8 + header_len + data_end == размер файла`.
Замер 2026-09-05 на 9B: загрузка 19 ГБ ≈ 20 мин, F16 ≈ 2 мин, квант ≈ 2 мин.

### Ключи .env, которые трогаем при смене модели

```env
MODEL=D:\Models\unsloth\Qwen3.8-27B-GGUF\Qwen3.8-27B-UD-IQ2_XXS.gguf
MTP=1
MTP_PATH=D:\Models\unsloth\Qwen3.8-27B-GGUF\mtp-Qwen3.8-27B-Q4_0.gguf
CTX=131072
SLOTS=4
GPU_ONLY=1
KV_POOL_Q8=1
CUDA_GRAPHS=1
```

Те же значения флагами, без правки файла:

```bat
yforge.exe --env .env --model D:\Models\...\Qwen3.8-27B-UD-IQ2_XXS.gguf --mtp 1
```

`MTP_PATH` — draft-модель для спекуляции; работает и без профиля-манифеста.
`KV_POOL_Q8=1` (`--kv-pool q8`) вдвое сокращает постоянный пул KV — он
выделяется на всё окно сразу при старте и является главной статьёй VRAM
после весов.
VRAM-планер сам ужмёт ctx/slots под 12 GB — следить за `[vram] plan` в логе.

### Smoke-проверка после перезапуска

```powershell
curl.exe -s -m 180 http://127.0.0.1:18099/v1/chat/completions `
  -H 'Content-Type: application/json' `
  -H 'Authorization: Bearer <ключ из .env>' `
  -d '@D:\Projects\yttri-inference\logs\smoke.json'
```

В ответе `usage.mtp.enabled=true` — спекуляция работает. Пейлоад `smoke.json` —
любой валидный chat-completions JSON.

### VRAM-замер A/B (`scripts\windows\vram_ab.ps1`)

Меряет по фазам (restart / idle / request / after) для процесса `yforge`:
dedicated и shared GPU-память (WDDM, `Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory`),
`nvidia-smi` used/free, power, SM clock и свободную RAM; плюс decode/TTFT из
`bench-live.ps1`. Профили по умолчанию: `q8-128k`, `f16-64k`, `q8-64k`
(плюс `q8-128k-c16k`, `f16-64k-c16k` для чанка 16384); `-Only <name>`,
`-Prompt <file>`, `-MaxTokens N`. В конце возвращает прод-профиль из `.env`.
Результат 2026-09-18: q8-128k и f16-64k по VRAM равны (2064 против 2048 МиБ
пула), вытеснения в shared нет, цена int8 — распаковка в декоде, не память.

### Синхронизация кода с yttri-win (аудит 2026-08-31)

### Выгрузка экспертов MoE (2026-09-04, план moe-expert-offload)

- **Требование к RAM при `MOE_EXPERTS=ram`**: свободной физической памяти ≥
  эксперты (~8.6 ГиБ на 35B IQ2_XXS) + PREFIX_CACHE_MIB + 2 ГиБ (на стенде
  ≈29 ГиБ из 64). Pinned-память не свопится — параллельная сборка другого
  агента обязана это учитывать (упрётесь в 64 ГиБ — pinned-аллокация падает
  fail-closed при старте).
- **PGRAPH при выгрузке**: графовый префил выключается автоматически (WARN
  `[pg] disabled: experts in ram`), префил идёт пейджед-прогревом (KV сразу
  в пул), графы декода захватываются — `[graphs] captured` в логе.
- Переменные: `MOE_EXPERTS=vram|ram|auto` (default auto), `EXPERT_CACHE_MIB`
  и `EXPERT_PROMOTE_PER_STEP` — фаза 4 (кэш горячих экспертов).
- Откат — `.env.bak-ornith-pgraph-on` + задача `\qwen36-inference`.

### Runtime-заметки (2026-08-31, Qwen3.8)

- **cudart PATH**: exe линкуется с cudart 12.4; в `inference-run.bat` PATH должен включать `...CUDA\v12.4\bin` ПЕРЕД v13.2 — иначе процесс умирает молча (exit 0, ноль вывода, Event Log: APPCRASH nvcuda64.dll 0xc0000005 при холодных запусках).
- **IQ1S**: UD-кванты unsloth содержат до ~20 тензоров IQ1S; в форке (44385cfd→76c8c773) — нативные CUDA-ядра: mmvq `mul_mat_vec_iq1_s_q8_1_b{1..8}`, dequantize f32/f16 (block_dim **32**, как в llama.cpp — при 256 тредах раскладка il/ib ломается и weights читаются мусором), MMQ-инстансы отключены (MMA-тайл IQ1_S даёт NaN, prefill через tiled dequant+cuBLAS). Если «unsupported IQ1S» — форк/сборка не синхронны.
- **VRAM-остаток после force-kill**: после нескольких `Stop-Process -Force` WDDM не успевает вычистить backing — used VRAM остаётся высоким при пустых процессах, а следующий инстанс получает выселенную память (генерация через paging, на слот может попадать мусорный контекст и петли). Лечение: убедиться, что процессов нет, пауза 5–10 с, затем чистый старт; после старта сверить `nvidia-smi` used ≈ weights+MTP+KV (~11 ГБ на Q2_K_XL).
- Петли «-2K» на 2-битных квантах: причина комбинация входных повторов + t=1.0 + слабые штрафы; `REPETITION_PENALTY=1.05`, `PRESENCE_PENALTY=0.3` + чистая VRAM дают нулевые петли. При возврате петель — поднять repetition до 1.1 или снизить TEMPERATURE до 0.7.

### Синхронизация кода с yttri-win (аудит 2026-08-31)

Проверено: продакшен-код на yttri-win соответствует репозиториям, потерь нет.

- Серверная копия репо: `D:\Projects\yttri-inference\qwen36-server` — `git master` старее локального `main` на ~288 коммитов, уникальных коммитов нет; рабочие файлы = `main`.
- Форк: `D:\Projects\yttri-forge` — история разошлась (2 коммита поверх общего базы), но содержимое деревьев совпадает с `main`; незакоммиченные правки рабочего дерева = уже закоммиченные в `main`.
- **Path-dependency различается между копиями:** локально `Cargo.toml` → `../yttri-forge`, на серверной копии → `../../yttri-forge` (копия лежит на уровень глубже). Не «лечить» заменой — следить при переносе.
- **git на yttri-win подменён обёрткой** (`C:\Program Files\Git\bin\git.exe`, от 20.08): `git status` выдаёт человеко-формат (`~ Modified: N files` / `clean — nothing to commit`), posh-git в ssh-профиле добавляет шум. `rev-parse`/`log`/`diff`/`bundle` работают норм. Для точного статуса: `git diff --name-only`, `git status --porcelain` не доверять.

Правки в код делать в локальном репо → коммит → push → перенос на yttri-win (`git pull` там, либо scp конкретных файлов). Обратно — только через bundle/diff + сверка содержимого, как выше.

## Open WebUI (сборка из исходников, yttri-win)

Браузерный чат вместо встроенного `web/index.html` и Unsloth Studio: собран из
исходников [github.com/open-webui/open-webui](https://github.com/open-webui/open-webui)
(тег v0.9.6), живёт на yttri-win по адресу **http://192.168.2.89:8080/** и закрыт
входом по email + пароль (`WEBUI_AUTH=true`, самостоятельная регистрация выключена).
Порт 18099 остаётся API (OpenAI/Responses/Anthropic) и fallback-чатом.

### Комплект поставки

| Файл (репо → yttri-win) | Роль |
| --- | --- |
| `scripts/windows/openwebui-install.ps1` → `D:\Projects\yttri-inference\openwebui-install.ps1` | установка/обновление: clone тега → `npm ci` + `npm run build` → `uv venv` + `pip install -r backend\requirements.txt` → `.env` → задача + firewall → старт и health-проба |
| `scripts/windows/openwebui-run.bat` → `…\openwebui-run.bat` | точка входа задачи планировщика, лог в `logs\open-webui.log` |
| `scripts/windows/openwebui-run.ps1` → `…\openwebui-run.ps1` | читает ключ yforge, опрашивает `/v1/models`, пишет `.env`, запускает uvicorn |
| `scripts/windows/openwebui-setup.ps1` → `…\openwebui-setup.ps1` | задача `\open-webui` (BootTrigger, SYSTEM, RestartOnFailure 3×1 мин) + правило фаервола `Open WebUI 8080` |

Развёртывание с нуля (yttri-win, от администратора; нужны `git`, `node`, `uv`):

```powershell
# скопировать 4 скрипта в D:\Projects\yttri-inference, затем:
powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File D:\Projects\yttri-inference\openwebui-install.ps1
```

Повторный запуск идемпотентен: готовые стадии (сборка фронта, venv) пропускаются,
`.env`, задача и правило фаервола перепроверяются, сервис перезапускается.
`-Rebuild` форсирует `npm ci`/`npm run build` и переустановку зависимостей,
`-AdminPassword '<pw>'` задаёт пароль админа (без него — существующий из `.env`
или новый сгенерированный), `-SkipTask` — не трогать задачу и firewall,
`-SkipStart` — только подготовка без запуска. Скрипт вызывает `npm.cmd`
(а не `npm.ps1`) — PowerShell-шим npm ломает аргументы (`Unknown command: "pm"`).

### Доступ и учётные данные

- Логин по умолчанию `admin@localhost`; пароль лежит в `open-webui\.env`
  (`WEBUI_ADMIN_PASSWORD`) и печатается при первой установке.
- `WEBUI_ADMIN_*` — bootstrap-значения: Open WebUI создаёт админа из них **только
  когда в базе ещё нет пользователей**. У работающего сервиса пароль меняется в
  интерфейсе: **Settings → Account → Password**.
- Восстановление доступа: остановить задачу `\open-webui`, удалить
  `open-webui\backend\data\webui.db*` (история чатов пропадёт), запустить задачу —
  админ пересоздастся из `.env`.
- `ENABLE_SIGNUP=false` — новых пользователей заводит админ (Admin → Users).

### Модель и подключение

- yforge подключён как OpenAI-совместимый провайдер:
  `OPENAI_API_BASE_URLS=http://127.0.0.1:18099/v1`, ключ — запись `web-chat` из
  `qwen36-server\.env`.
- `DEFAULT_MODELS` синхронизируется из `/v1/models` при каждом старте, поэтому в
  интерфейсе уже выбрана реально загруженная модель. Сменили модель — перезапустите
  задачу `\open-webui`.
- RAG-эмбеддер скачивается при первом старте в `open-webui\.cache\huggingface`
  (`HF_HOME` задаёт лончер, кэш не дублируется в профиле SYSTEM).

### Управление и проверка

```bat
schtasks /run  /tn open-webui
schtasks /query /tn open-webui
netstat -ano | findstr :8080
taskkill /PID <pid> /F
```

```powershell
curl.exe -s http://127.0.0.1:8080/api/config     # "auth":true
# токен: POST /api/v1/auths/signin {"email":"admin@localhost","password":"<pw>"}
curl.exe -s http://127.0.0.1:8080/api/models -H "Authorization: Bearer <token>"
```

## Профильная схема (run_windows.bat)

`YFORGE_CURRENT=D:\Models\yttri\qwen3.5-4b\current` — JSON-указатель на релиз
с манифестом (`profile.json`, SHA-256). Сервер и артефакты (vision/mtp) берутся
из манифеста. Используется для проверок тестовой модели 3.5-4B, не для продакшена 27B.
