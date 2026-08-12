# qwen36-server

Локальный Rust-сервер для Qwen3.6 GGUF на собственном candle-форке. Сервер поддерживает Qwen3.6-27B и Qwen3.6-35B-A3B, четыре одновременных слота, CUDA на Windows/Linux и Metal на macOS.

HTTP-поверхность:

- OpenAI Chat Completions: `POST /v1/chat/completions`;
- OpenAI Responses: `POST /v1/responses`;
- Anthropic Messages: `POST /v1/messages`;
- список и параметры активной модели: `GET /v1/models`;
- список локальных GGUF и горячее переключение модели;
- встроенный веб-чат: `GET /`.

Vision не поддерживается. llama.cpp в сервер не входит.

## Проверенная конфигурация

На Windows-машине с RTX 3060 12 GB проверена модель `Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf`:

- CUDA 12.4;
- `QWEN36_CTX=8192`;
- `QWEN36_SLOTS=4`;
- четыре одновременных ответа по 8140 токенов;
- четыре результата совпали побитно по тексту;
- все SSE-потоки завершились штатно;
- shared GPU memory оставалась 78 MiB.

Подробности: [docs/lessons/2026-08-12-yttri-win-stability.md](docs/lessons/2026-08-12-yttri-win-stability.md).

## Структура директорий

`Cargo.toml` использует локальные path-зависимости. Репозиторий сервера и candle-форк должны лежать рядом:

```text
workdir/
├── candle-fork-qwen35-batch/   # https://github.com/askidmobile/candle.git
└── qwen36-server/              # https://github.com/askidmobile/qwen36-server.git
```

Клонирование:

```bash
mkdir qwen36-workdir
cd qwen36-workdir
git clone --branch feat/qwen35-batching https://github.com/askidmobile/candle.git candle-fork-qwen35-batch
git clone https://github.com/askidmobile/qwen36-server.git qwen36-server
cd qwen36-server
```

## Конфигурация и API-ключи

Сервер при запуске читает `.env` из текущей директории. Другой файл задаётся через `QWEN36_ENV_FILE`. Переменные окружения процесса имеют приоритет над значениями из файла.

Создайте локальный файл:

```bash
cp .env.example .env
```

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Windows `cmd.exe`:

```bat
copy .env.example .env
```

Главный параметр аутентификации — `QWEN36_API_KEYS`. Это JSON-массив объектов `key`/`name`, записанный в одну строку:

```dotenv
QWEN36_API_KEYS='[{"key":"yttri-api-change-me","name":"test"},{"key":"yttri-api-change-me-too","name":"backup"}]'
```

- `key` передаётся клиентом как Bearer token;
- `name` — уникальная операторская метка;
- права всех ключей одинаковы;
- пустые массивы, пустые поля, повторяющиеся имена и ключи отклоняются;
- старый `QWEN36_API_KEY` больше не поддерживается.

`.env` исключён из git. Не копируйте реальные ключи в `.env.example`, README, issue, логи или unit-файлы. После утечки удалите ключ из массива и сгенерируйте новый.

Генерация нового ключа без дополнительных зависимостей:

```bash
python3 -c 'import secrets; print("yttri-api-" + secrets.token_urlsafe(32))'
```

PowerShell:

```powershell
$bytes = New-Object byte[] 32
$rng = [Security.Cryptography.RandomNumberGenerator]::Create()
$rng.GetBytes($bytes)
$rng.Dispose()
$token = [Convert]::ToBase64String($bytes).TrimEnd('=').Replace('+','-').Replace('/','_')
"yttri-api-$token"
```

Минимальный `.env`:

```dotenv
QWEN36_API_KEYS='[{"key":"yttri-api-change-me","name":"test"}]'
QWEN36_MODEL=/absolute/path/to/model.gguf
QWEN36_HOST=0.0.0.0
QWEN36_PORT=8080
QWEN36_CTX=8192
QWEN36_SLOTS=4
QWEN36_PREFIX_CACHE_MIB=0
```

На Windows допустим обычный путь с обратными слешами:

```dotenv
QWEN36_MODEL=D:\Models\unsloth\Qwen3.6-35B-A3B-GGUF\Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf
QWEN36_MODELS_DIR=D:\Models
```

Парсер `.env` намеренно простой: одна переменная на строку, формат `NAME=value`, без shell-подстановок. JSON держите на одной строке.

### Все параметры

| Переменная | Default | Назначение |
|---|---:|---|
| `QWEN36_API_KEYS` | нет | Обязательный JSON-массив API-ключей |
| `QWEN36_MODEL` | `models/qwen36-27b-q2_k_xl.gguf` | Путь к активному GGUF |
| `QWEN36_MODELS_DIR` | вычисляется из пути модели | Корень для поиска и переключения GGUF |
| `QWEN36_HOST` | `0.0.0.0` | Адрес HTTP listener |
| `QWEN36_PORT` | `8080` | TCP-порт |
| `QWEN36_CTX` | `81920` | Максимальный контекст модели; фактический output clamp учитывает prompt |
| `QWEN36_SLOTS` | `4` | Число слотов, допустимо `1..=4` |
| `QWEN36_PREFIX_CACHE_MIB` | `0` | Должен оставаться `0`: prefix cache временно отключён |
| `QWEN36_ENV_FILE` | `.env` | Альтернативный env-файл |
| `QWEN36_REQ_TIMEOUT` | `600` | Timeout запроса batched engine, секунды |
| `QWEN36_MAX_QUEUE` | `64` | Максимальная очередь |
| `QWEN36_TRACE` | off | Trace: `1`, `true`, `yes` или `on` |
| `QWEN36_MOE_BACKEND` | `reference` | Для CUDA MoE можно выбрать `ptx` |
| `QWEN36_PREFILL_CHUNK` | внутренний default | Размер prefill chunk |
| `QWEN36_NO_VRAM_PLAN` | unset | Отключить автоматический VRAM-план |

## Windows + NVIDIA CUDA

### Требования

- Windows 10/11 x64;
- NVIDIA driver, проверка: `nvidia-smi`;
- CUDA Toolkit 12.4, включая `nvcc`;
- Visual Studio 2022 Build Tools с workload **Desktop development with C++**;
- Rust stable MSVC: `rustup default stable-x86_64-pc-windows-msvc`;
- Git.

Проверьте инструменты:

```powershell
nvidia-smi
& 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\nvcc.exe' --version
cargo --version
```

### Сборка

Из корня `qwen36-server`:

```bat
scripts\build_windows.bat
```

Скрипт:

1. находит Visual Studio через `vswhere`;
2. вызывает `VsDevCmd.bat` для `cl.exe`;
3. добавляет CUDA `bin` в `PATH`;
4. добавляет CUDA `lib\x64` в `LIB` — без этого linker выдаёт `LNK1181: cudart.lib`;
5. запускает `cargo build --release --features cuda`.

Для другой версии CUDA или GPU:

```bat
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
set "CUDA_COMPUTE_CAP=86"
scripts\build_windows.bat
```

Compute capability:

- RTX 3060: `86`;
- RTX 4090: `89`;
- H100: `90`.

### Запуск

Отредактируйте `.env`, затем:

```bat
scripts\run_windows.bat
```

Launcher добавляет CUDA `bin` в `PATH` и передаёт абсолютный путь к `.env`. Прямой запуск из корня репозитория тоже работает:

```bat
target\release\qwen36-server.exe
```

Если запуск идёт не из корня репозитория:

```bat
set "QWEN36_ENV_FILE=D:\Projects\qwen36-server\.env"
D:\Projects\qwen36-server\target\release\qwen36-server.exe
```

CUDA DLL должны быть доступны через `PATH`:

```bat
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin;%PATH%"
target\release\qwen36-server.exe
```

На RTX 3060 12 GB для 35B-A3B используйте IQ2_XXS, `ctx=8192`, `slots=4`. Более крупные кванты могут уйти в WDDM paging. В Task Manager process private commit включает CUDA allocations; реальный paging смотрите в Performance Counters `GPU Process Memory / Shared Usage`.

## Linux + NVIDIA CUDA

### Требования

Пример для Ubuntu/Debian:

```bash
sudo apt update
sudo apt install -y build-essential pkg-config git curl clang cmake
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
```

Установите NVIDIA driver и CUDA Toolkit с [официальной страницы NVIDIA](https://developer.nvidia.com/cuda-downloads). Нужен именно toolkit с `nvcc`, не только driver.

```bash
nvidia-smi
nvcc --version
cargo --version
```

### Сборка

Для RTX 4090:

```bash
cd qwen36-server
CUDA_COMPUTE_CAP=89 ./scripts/build_linux.sh
```

Для RTX 3060:

```bash
CUDA_COMPUTE_CAP=86 ./scripts/build_linux.sh
```

Скрипт сам клонирует отсутствующий sibling-форк и выполняет:

```bash
cargo build --release --features cuda
```

Бинарник: `target/release/qwen36-server`.

### Запуск

```bash
cp .env.example .env
$EDITOR .env
./target/release/qwen36-server
```

Для фонового запуска используйте systemd/supervisor. Передавайте `QWEN36_ENV_FILE` абсолютным путём, чтобы запуск не зависел от `WorkingDirectory`:

```ini
[Service]
WorkingDirectory=/opt/qwen36/qwen36-server
Environment=QWEN36_ENV_FILE=/opt/qwen36/qwen36-server/.env
ExecStart=/opt/qwen36/qwen36-server/target/release/qwen36-server
Restart=on-failure
```

## macOS + Metal

### Требования

- Apple Silicon Mac;
- macOS с Metal;
- Xcode Command Line Tools;
- Rust stable;
- Git.

```bash
xcode-select --install
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
```

### Сборка

```bash
cargo build --release --features metal
```

### Запуск

```bash
cp .env.example .env
$EDITOR .env
./target/release/qwen36-server
```

На macOS память unified. Выбирайте квант и контекст с учётом общей RAM системы. macOS используется как Metal development path; основной production/stability path проверен на Windows CUDA.

## Проверка сервера

Пусть один ключ из `.env` сохранён в shell:

```bash
export KEY='yttri-api-...'
export BASE_URL='http://127.0.0.1:8080'
```

PowerShell:

```powershell
$Key = 'yttri-api-...'
$BaseUrl = 'http://127.0.0.1:8080'
```

Список моделей:

```bash
curl "$BASE_URL/v1/models" -H "Authorization: Bearer $KEY"
```

Chat Completions:

```bash
curl "$BASE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $KEY" \
  -d '{"model":"qwen3.6","messages":[{"role":"user","content":"Привет"}],"max_tokens":128,"thinking":false}'
```

Streaming:

```bash
curl -N "$BASE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $KEY" \
  -d '{"model":"qwen3.6","stream":true,"messages":[{"role":"user","content":"Считай от 1 до 20"}],"max_tokens":128,"thinking":false}'
```

Веб-чат: `http://127.0.0.1:8080/`. Вставьте в UI значение поля `key`, не весь JSON-массив.

## Тесты

macOS/Metal:

```bash
cargo test --features metal --all-targets
cargo clippy --features metal --all-targets --no-deps -- -D warnings
```

Windows/Linux CUDA:

```bash
cargo test --features cuda --all-targets
```

Stability smoke против уже запущенного сервера:

```bash
KEY='yttri-api-...' \
HOST='http://192.168.2.89:8080' \
SSH_HOST='yttri-win' \
MAX_TOKENS=8192 \
./scripts/stability_smoke.sh
```

Windows benchmark:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\bench.ps1 `
  -BaseUrl 'http://localhost:8080' `
  -ApiKey 'yttri-api-...'
```

## Частые ошибки

### `LNK1181: cannot open input file cudart.lib`

CUDA library directory отсутствует в `LIB`:

```bat
set "LIB=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\lib\x64;%LIB%"
```

Либо используйте `scripts\build_windows.bat`.

### `nvcc fatal: Cannot find compiler 'cl.exe' in PATH`

Сначала активируйте Visual Studio environment:

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64
```

### `invalid or missing API key`

Проверьте:

- сервер перезапущен после изменения `.env`;
- клиент передаёт только значение `key`;
- заголовок имеет вид `Authorization: Bearer <key>`;
- JSON в `QWEN36_API_KEYS` валиден и находится на одной строке.

### `prefix cache temporarily disabled`

Оставьте:

```dotenv
QWEN36_PREFIX_CACHE_MIB=0
```

Restore path prefix cache намеренно закрыт до inference parity-теста.

### Сервер стартовал без CUDA

Соберите с feature:

```bash
cargo build --release --features cuda
```

На Windows используйте MSVC toolchain и `scripts\build_windows.bat`.

### Модель уходит в системную RAM на Windows

Проверьте `GPU Process Memory / Shared Usage`, а не только private commit процесса. Если shared usage растёт, уменьшите квант, контекст или число слотов.

## Безопасность

Сервер по умолчанию слушает `0.0.0.0` и работает без TLS. Это режим для доверенной LAN. Не публикуйте порт напрямую в интернет. Для внешнего доступа ставьте reverse proxy с TLS, firewall и отдельными ключами. Ротация ключа требует изменения `QWEN36_API_KEYS` и перезапуска процесса.

## Дополнительная документация

- [Engine/API contract](docs/engine-api.md)
- [Batch integration](docs/batch-integration.md)
- [Project brief](docs/brief/PROJECT-BRIEF.md)
- [Decision log](docs/brief/decisions.md)
- [Stability plan](tests/stability_plan.md)
