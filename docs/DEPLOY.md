# Полное развёртывание на новом сервере

Инструкция «с нуля»: чистый Windows-сервер → рабочий стек Yttri Forge:

- **`yforge`** — сервер инференса GGUF (OpenAI Chat Completions, Responses, Anthropic Messages) на порту **18099**;
- **Open WebUI** — веб-чат с входом по логину и паролю на порту **8080**, собран из исходников.

Оба сервиса живут как задачи планировщика Windows и поднимаются при загрузке.
Разработка (macOS/Linux/CPU) описана в [README.md](../README.md); здесь — продакшен-стенд.

Проверено 2026-09-18 на `yttri-win` (RTX 3060 12 GB, драйвер 591.86, CUDA 13.2 + 12.4,
VS 2022 BuildTools, Rust 1.95.0, Node 22.18, Python 3.12.5, uv 0.8.10, 64 ГБ RAM):
`--dry-run` прод-конфига, живой API, вход в WebUI, установка WebUI в чистый каталог,
повторный (идемпотентный) прогон установщика.

## 1. Железо и ОС

| Параметр | Минимум | Проверенный стенд |
| --- | --- | --- |
| ОС | Windows 10/11 x64, Server 2019+ | Windows (yttri-win) |
| GPU | NVIDIA ≥ 8 ГБ VRAM; CPU-режим возможен, но медленный | RTX 3060 12 GB |
| RAM | 16 ГБ (профили с выгрузкой экспертов MoE — 32…64 ГБ) | 64 ГБ |
| Диск | ~20 ГБ на сборку и кэши + место под GGUF | `D:` 196 ГБ свободно |
| Сеть | LAN; исходящий доступ нужен на этапе установки/скачивания моделей | — |

Порт 18099 — API, 8080 — WebUI. Оба рассчитаны на LAN; наружу не публикуйте без HTTPS-прокси.

## 2. Раскладка каталогов (продакшен)

```
D:\Projects\yttri-inference\                 корень развёртывания: лаунчеры, логи, open-webui\
D:\Projects\yttri-inference\qwen36-server\   код сервера + .env + target\release\yforge.exe
D:\Projects\yttri-forge\                     движок (Cargo path-зависимость)
D:\Models\<org>\<repo>\<file>.gguf           модели
```

Скрипты комплекта (`scripts\windows\inference-run.bat`, `openwebui-*.ps1`) захардкожены
на `D:\Projects\yttri-inference` — либо соблюдайте раскладку, либо правьте `ROOT` в скриптах.

**Важно про движок.** `Cargo.toml` подключает `yttri-forge` относительным путём:
в репозитории это `../yttri-forge` (клоны-соседи), а в раскладке выше — `../../yttri-forge`.
После клонирования поправьте четыре строки зависимостей (см. шаг 4).

## 3. Инструменты

```powershell
winget install --id Git.Git -e
winget install --id OpenJS.NodeJS.LTS -e           # нужен для сборки фронта Open WebUI
winget install --id Python.Python.3.12 -e
winget install --id Rustlang.Rustup -e
winget install --id astral-sh.uv -e                # venv/пакеты Open WebUI
winget install --id Microsoft.VisualStudio.2022.BuildTools -e `
  --override "--wait --quiet --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

CUDA Toolkit **13.2** — с [архива NVIDIA](https://developer.nvidia.com/cuda-toolkit-archive)
(winget `Nvidia.CUDA` отдаёт более новый релиз; тогда задайте `YTTRI_CUDA` в `build_windows.bat`).
На стенде рядом лежит **12.4** — она нужна рантайму (см. шаг 9).

Проверка после установки:

```powershell
cargo --version; node -v; uv --version; python --version; nvcc --version
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
```

## 4. Код

```powershell
New-Item -ItemType Directory -Force D:\Projects\yttri-inference | Out-Null
git clone https://github.com/askidmobile/yttri-forge.git D:\Projects\yttri-forge
git clone https://github.com/askidmobile/qwen36-server.git D:\Projects\yttri-inference\qwen36-server

# пути к движку под раскладку выше (4 строки зависимостей: qwen35-batch,
# candle-transformers, candle-core, candle-flash-attn)
$cargo = 'D:\Projects\yttri-inference\qwen36-server\Cargo.toml'
$text  = Get-Content $cargo -Raw
# якорь на кавычку: уже исправленные "../../yttri-forge" не трогаются,
# скрипт можно запускать повторно
$text  = $text -replace '"\../yttri-forge', '"../../yttri-forge'
Set-Content $cargo -Value $text -NoNewline -Encoding utf8
Select-String -Path $cargo -Pattern 'yttri-forge'   # ожидаем ../../yttri-forge
```

## 5. Сборка сервера

```cmd
D:\Projects\yttri-inference\qwen36-server\scripts\build_windows.bat
```

Скрипт поднимает окружение VS 2022 (`VsDevCmd`), ставит `CUDA_PATH` и зовёт
`cargo build --release --features cuda`. Результат — `target\release\yforge.exe`.
Переопределения: `set YTTRI_CUDA=...\v12.4` (другая CUDA), `set CUDA_COMPUTE_CAP=89`
(RTX 40xx) / `120` (RTX 50xx); по умолчанию 86 — RTX 30xx.

Проверка:

```cmd
cd /d D:\Projects\yttri-inference\qwen36-server
target\release\yforge.exe --help
```

## 6. Модель

Продакшен-модель стенда — **Ornith-1.5-9B-Q4_K_M.gguf** (5512 МиБ):

```powershell
New-Item -ItemType Directory -Force D:\Models\deepreinforce-ai\Ornith-1.5-9B-GGUF | Out-Null
# вариант A: hf.exe из venv llama.cpp (если уже есть на машине)
# hf download <org>/<repo> Ornith-1.5-9B-Q4_K_M.gguf --local-dir D:\Models\deepreinforce-ai\Ornith-1.5-9B-GGUF

# вариант B: параллельная докачка (крупные файлы, монотонный канал)
powershell -File D:\Projects\yttri-inference\qwen36-server\scripts\windows\parallel-dl.ps1 `
  -Url https://huggingface.co/<org>/<repo>/resolve/main/Ornith-1.5-9B-Q4_K_M.gguf `
  -OutFile D:\Models\deepreinforce-ai\Ornith-1.5-9B-GGUF\Ornith-1.5-9B-Q4_K_M.gguf
```

Только safetensors (GGUF не выложен) — конвейер HF → F16 → квант:

```cmd
D:\Projects\yttri-inference\qwen36-server\scripts\windows\convert-hf-gguf.bat ^
  D:\Models\source\<repo> D:\Models\<org>\<repo>-GGUF <basename> Q4_K_M
```

Опционально MTP-драфт для спекулятивного декода: `D:\Models\ornith-mtp\mtp-Ornith-1.5-9B-head-Q8_0.gguf`
(+ `MTP=1`, `MTP_PATH=...` в `.env`; продакшен работает с `MTP=0`).

## 7. Конфигурация сервера (`.env`)

Файл — `D:\Projects\yttri-inference\qwen36-server\.env` (в git не ходит). Прод-профиль
для 12 ГБ VRAM на 2026-09-18 (ключи заменить на свои):

```env
API_KEYS='[{"key":"yttri-api-...","name":"codex"},{"key":"yttri-api-...","name":"web-chat"}]'
MODEL=D:\Models\deepreinforce-ai\Ornith-1.5-9B-GGUF\Ornith-1.5-9B-Q4_K_M.gguf
MODELS_DIR=D:\Models
HOST=0.0.0.0
PORT=18099
CTX=128000
CONTEXT_LIMIT=128000
SLOTS=1
TRACE=0
MOE_BACKEND=ptx
GPU_ONLY=1
GPU_LAYERS=999
VRAM_HEADROOM_MIB=1536
KV_CACHE_DTYPE=q8
CUDA_GRAPHS=1
KV_POOL_Q8=1
PREFIX_CACHE_MIB=2048
PREFIX_CACHE_CHECKPOINTS=0
PREFIX_CACHE_CHECKPOINT_CAP=8
PGRAPH=on
PGRAPH_MIN_T=1000000
PREFILL_CHUNK=8192
GRAPH_WINDOW=131072
QK_INT8=1
QK_INT8_PREFILL=1
PREFIX_CACHE_TAIL_SPLIT=1
PREFIX_CACHE_TAIL_ALIGN=64
PREFIX_CACHE_FULL_HIT=0
PREFIX_CACHE_POOL_BACKED=1
YTTRI_ATTN_PREP_FUSED=1
QWEN36_FA_PREFILL_TILE=7
MOE_EXPERTS=auto
MTP=0
MTP_PATH=D:\Models\ornith-mtp\mtp-Ornith-1.5-9B-head-Q8_0.gguf
MTP_WIDTH=3
MTP_VOCAB_TOP=32768
MTP_TIMING=0
VERIFY_ONEPASS=1
REQ_DEBUG=D:\Projects\yttri-inference\logs\requests-debug.jsonl
```

Это полный набор рабочего прод-файла (39 ключей; служебную строку `NONE=1` от замеров
движок игнорирует). `MTP_*` указаны для полноты — при `MTP=0` они не используются.

Почему так (полные замеры — в [docs/research](research/2026-09-16-head-to-head-llamacpp-and-phase-profile.md)):

- `CTX=128000` + `KV_POOL_Q8=1` + `GRAPH_WINDOW=131072` — 128k-окно в 12 ГБ (пул 2 ГБ вместо 4 ГБ);
- `PGRAPH=on` обязателен для 128k, но `PGRAPH_MIN_T=1000000` **отключает захват графов префила**
  (их replay на WDDM медленнее eager в 4…17 раз); графы декода при этом работают;
- `PREFILL_CHUNK=8192` — 16384 на ~127k-промпте уходит в OOM;
- `QK_INT8(+_PREFILL)=1` — int8-QK в декоде и префиле (быстрее int8-f16-пула);
- `SLOTS=1` — на 12 ГБ четыре слота не влезают вместе с 128k-пулом (см. «Планирование VRAM» в README);
- `REQ_DEBUG` — опционально, пишет тела запросов (для отладки; на нагруженном стенде выключить).

`API_KEYS` — массив `[{"key","name"}]`; именованные ключи удобно раздавать клиентам
(`web-chat` использует Open WebUI). Права у всех одинаковые.

## 8. Проверка конфигурации без загрузки модели

```powershell
cd D:\Projects\yttri-inference\qwen36-server
.\target\release\yforge.exe --env .env --dry-run
```

Ожидаемые строки: `model = ...`, `listen = 0.0.0.0:18099`, `n_ctx = 128000`, `n_slots = 1`,
`kv_pool = q8`, `[vram] plan(dynamic): ctx=128000 slots=1 — KV ~2016MiB/слот …`.
Если `--dry-run` ругается на модель/ключи — правим `.env` до запуска задачи.

## 9. Лаунчер и задача планировщика

Скопируйте `scripts\windows\inference-run.bat` из репозитория в `D:\Projects\yttri-inference\`
(на стенде этот файл — источник истины; держите копию в репозитории синхронной):

```bat
@echo off
setlocal EnableExtensions
set "ROOT=D:\Projects\yttri-inference"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
set "CUDA_INCLUDE_DIR=%CUDA_PATH%\include"
set "CUDA_COMPUTE_CAP=86"
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin;%CUDA_PATH%\bin;%PATH%"
rem Bench hook: KEY=VALUE lines from bench-overrides.env are applied to the
rem process environment BEFORE the server loads .env. The .env loader skips
rem names already present in the environment, so these overrides win.
if exist "%ROOT%\bench-overrides.env" for /f "usebackq tokens=*" %%L in ("%ROOT%\bench-overrides.env") do set "%%L"
set "QWEN36_ENV_FILE=%ROOT%\qwen36-server\.env"
cd /d "%ROOT%\qwen36-server" || exit /b 1
"%ROOT%\qwen36-server\target\release\yforge.exe" --env "%QWEN36_ENV_FILE%" 1>>"%ROOT%\logs\server.log" 2>&1
exit /b %ERRORLEVEL%
```

Что здесь важно: `CUDA\v12.4\bin` идёт **первой** в `PATH` (exe линкуется с cudart 12.4;
иначе процесс падает молча), `bench-overrides.env` применяется поверх `.env` только для
замеров, а сервер получает явный `--env %QWEN36_ENV_FILE%`.

`Start-Process` из ssh-сессии умирает вместе с сессией — запускать только задачей.
Регистрация (PowerShell от администратора):

```powershell
$action    = New-ScheduledTaskAction -Execute 'cmd.exe' `
  -Argument '/c D:\Projects\yttri-inference\inference-run.bat' -WorkingDirectory 'D:\Projects\yttri-inference'
$trigger   = New-ScheduledTaskTrigger -AtStartup
$principal = New-ScheduledTaskPrincipal -UserId 'SYSTEM' -LogonType ServiceAccount -RunLevel Highest
$settings  = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
  -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew `
  -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
Register-ScheduledTask -TaskName 'qwen36-inference' -Action $action -Trigger $trigger `
  -Principal $principal -Settings $settings `
  -Description 'Yttri Forge inference server' -Force
```

Правило фаервола:

```powershell
New-NetFirewallRule -DisplayName 'qwen36-server 18099' -Direction Inbound -Action Allow `
  -Protocol TCP -LocalPort 18099 -Profile Any -Description 'yforge API'
```

Запуск и наблюдение:

```bat
schtasks /run /tn qwen36-inference
schtasks /query /tn qwen36-inference
type D:\Projects\yttri-inference\logs\server.log
```

В логе ждём `[vram] total=… plan(dynamic)`, `[kv] paged pool …`, `loaded: id=…`, `сервер запущен`.
Модель грузится десятки секунд (mmap с диска); до готовности `/v1/models` не отвечает.

## 10. Open WebUI (веб-интерфейс с логином и паролем)

1. Скопируйте из репозитория в `D:\Projects\yttri-inference\` четыре файла:
   `openwebui-install.ps1`, `openwebui-run.ps1`, `openwebui-run.bat`, `openwebui-setup.ps1`.
2. Запустите установщик (PowerShell от администратора):

```powershell
powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File D:\Projects\yttri-inference\openwebui-install.ps1
```

Установщик идемпотентен и делает всё сам: клонирует open-webui (тег `v0.9.6`) →
`npm ci` + `npm run build` → `uv venv` + `backend\requirements.txt` → пишет `.env`
(auth ON, signup OFF, подключение к yforge, admin-bootstrap) → регистрирует задачу
`\open-webui` (BootTrigger, SYSTEM) и правило фаервола 8080 → запускает и ждёт health.
Пароль админа генерируется и печатается в конце; свой — `-AdminPassword '<pw>'`,
пересборка — `-Rebuild`, только подготовка — `-SkipTask -SkipStart`.

Данные и учётные записи: логин `admin@localhost`, пароль — `WEBUI_ADMIN_PASSWORD`
в `open-webui\.env`; смена пароля — в UI (Settings → Account). Сброс — удалить
`open-webui\backend\data\webui.db*` и перезапустить задачу. Подробности и операции —
в [scripts/README.md](../scripts/README.md).

## 11. Приёмка (чек-лист)

| Шаг | Команда | Ожидание |
| --- | --- | --- |
| 1. Конфиг | `yforge.exe --env .env --dry-run` | `model=…`, `n_ctx`, `[vram] plan(dynamic)` |
| 2. Процесс | `netstat -ano \| findstr :18099` | LISTENING на 0.0.0.0:18099 |
| 3. Список моделей | `curl.exe -s -H "Authorization: Bearer <key>" http://127.0.0.1:18099/v1/models` | JSON с `id` модели |
| 4. Генерация | `curl.exe -s -X POST http://127.0.0.1:18099/v1/chat/completions -H "Authorization: Bearer <key>" -H "Content-Type: application/json" -d '{"model":"<id>","messages":[{"role":"user","content":"привет"}],"stream":false}'` | `choices[0].message.content` |
| 5. VRAM | `nvidia-smi --query-gpu=memory.used --format=csv` | ≈ веса + пул (для прод-профиля ~7–8 ГБ) |
| 6. WebUI | `curl.exe -s http://127.0.0.1:8080/api/config` | `"auth":true` |
| 7. Анонимный API | `curl.exe -s -o NUL -w "%{http_code}" http://127.0.0.1:8080/api/models` | `401` |
| 8. Вход | `POST /api/v1/auths/signin` с `{"email","password"}` | `token` в ответе |
| 9. Чат | `POST /api/chat/completions` с токеном | ответ модели (тот же `id`) |
| 10. Снаружи | с другого хоста: `curl http://<host>:8080/` | HTML страницы входа |

Живой пример ответа `/api/config`: `{"status":true,"version":"0.9.6","features":{"auth":true,"enable_signup":false,…}}`.

## 12. Эксплуатация

- **Перезапуск API**: `schtasks /end /tn qwen36-inference` → `schtasks /run /tn qwen36-inference`
  (мягко; жёстко — `taskkill /PID <pid> /F`). После force-kill выждите 5–10 с: WDDM
  не сразу освобождает VRAM.
- **Смена модели**: правка `MODEL=` в `.env` → перезапуск задачи. Альтернатива без рестарта —
  `POST /admin/switch` (скан `MODELS_DIR`).
- **Обновление кода**: `git pull` в `D:\Projects\yttri-inference\qwen36-server` →
  `scripts\build_windows.bat` → перезапуск задачи. Бинарь —
  `qwen36-server\target\release\yforge.exe`, лаунчер — `D:\Projects\yttri-inference\inference-run.bat`.
- **Синхронизация WebUI с моделью**: `DEFAULT_MODELS` обновляется при каждом старте задачи
  `\open-webui` из `/v1/models`; после смены модели перезапустите и её.
- **Бэкапы**: `.env` (ключи, профиль), `open-webui\backend\data\webui.db` (пользователи/чаты),
  `bench-overrides.env`. Модели — просто файлы GGUF.
- **Логи**: `logs\server.log` (API), `logs\open-webui.log` (WebUI), `logs\requests-debug.jsonl`
  (если включён `REQ_DEBUG`).

## 13. Типичные проблемы

| Симптом | Причина / лечение |
| --- | --- |
| Процесс падает молча (exit 0, в логе пусто, Event Log: `nvcuda64.dll 0xc0000005`) | В `PATH` нет `...CUDA\v12.4\bin` первой — exe линкуется с cudart 12.4 (см. лаунчер) |
| `unsupported dtype for quantized matmul IQ1S` | Сборка/форк не синхронны: пересоберите `yttri-forge` и сервер |
| Деградация/мигание после `Stop-Process -Force` | WDDM не вернул память: дождаться освобождения, сверить `nvidia-smi`, только потом старт |
| Пустая выдача, длинный префил на 128k | Проверьте, что `PGRAPH=on` и `PGRAPH_MIN_T=1000000` (иначе графы префила уходят в shared-память) |
| `context_length_exceeded` (HTTP 400) | Запрос длиннее `CONTEXT_LIMIT`; клиент должен сжать историю либо поднять `CTX` |
| Порт занят | `netstat -ano \| findstr :18099`, затем `taskkill /PID <pid> /F` |
| WebUI: `Unknown command: "pm"` при сборке | PowerShell-шим `npm.ps1` ломает аргументы — установщик уже вызывает `npm.cmd` |
| WebUI недоступен с другого хоста | Нет правила фаервола 8080 (`openwebui-setup.ps1`) или процесс слушает 127.0.0.1 |
| Задача не стартует после ssh-сессии | Запуск был через `Start-Process` — используйте `schtasks /run` |

## 14. Где что лежит

- [README.md](../README.md) — CLI/env-параметры, планирование VRAM, режимы сэмплинга, dev-запуск по ОС.
- [scripts/README.md](../scripts/README.md) — операционка на yttri-win: задачи, смена модели, WebUI-комплект.
- [docs/lessons/](../docs/lessons/) — разборы инцидентов (WDDM, cudart, IQ1S, порча состояния).
- [docs/research/](../docs/research/) — замеры (паритет с llama.cpp, 128k-профиль, int8-QK).
- [docs/brief/decisions.md](brief/decisions.md) — обязывающие решения (BD-001…BD-032): LAN-only и HTTP без TLS (BD-006), именованные API-ключи (BD-021).
