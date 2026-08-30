# Yttri Self-Inference Server

> Высокопроизводительный сервер локального инференса для моделей Qwen 3.5/3.6/3.8 и Ornith 1.0/1.5 (GGUF) на базе оптимизированного форка `candle` с поддержкой FlashAttention-2, Tensor-Core MMQ, аппаратно-ускоренного MTP (Next-N speculative decoding), непрерывного батчинга, multimodal (Vision/Video) и автоматической горячей смены моделей.

> High-performance local inference server for Qwen 3.5/3.6/3.8 and Ornith 1.0/1.5 (GGUF) based on an optimized `candle` fork supporting FlashAttention-2, Tensor-Core MMQ, hardware-accelerated MTP speculative decoding, continuous batching, multimodal vision/video, and automatic hot-swapping.

---

## 🇷🇺 Конфигурация и параметры запуска (.env)

Сервер поддерживает чистые имена переменных (без устаревшего префикса `QWEN36_`, но сохраняет полную обратную совместимость с ним).

```env
# ==============================================================================
# 1. СЕТЬ И АВТОРИЗАЦИЯ
# ==============================================================================
# JSON-массив авторизованных API-ключей и их клиентов
API_KEYS='[{"key":"your-secret-api-key-here","name":"primary"}]'
# IP-адрес для прослушивания (0.0.0.0 — доступен в локальной сети)
HOST=0.0.0.0
# Порт сервера
PORT=18099

# ==============================================================================
# 2. ПУТИ К МОДЕЛЯМ
# ==============================================================================
# Путь к стартовому файлу GGUF-модели
MODEL=D:\Models\deepreinforce-ai\Ornith-1.5-9B-GGUF\Ornith-1.5-9B-Q6_K.gguf
# Корневая папка с локальными моделями для сканирования и авто-переключения
MODELS_DIR=D:\Models

# ==============================================================================
# 3. ПАМЯТЬ И ПРОИЗВОДИТЕЛЬНОСТЬ
# ==============================================================================
# Размер контекстного окна в токенах (131072 = 128K)
CTX=131072
# Количество параллельных непрерывных слотов инференса (1..4)
SLOTS=4
# Максимальное число генерируемых токенов по умолчанию
MAX_TOKENS=32768
# Включение спекулятивного декодинга MTP (0 = выкл, 1 = вкл)
MTP=0
# Таймаут запроса в секундах
REQ_TIMEOUT=600

# ==============================================================================
# 4. СЕРВЕРНАЯ ПОЛИТИКА СЭМПЛИНГА
# По умолчанию клиентские temperature/top_p/top_k/min_p/penalties игнорируются.
# Источник истины: карточка семейства модели и явные значения в этом env-файле.
# ==============================================================================
# 1 = держать серверную политику; 0 = разрешить клиенту переопределять её.
SAMPLING_LOCK=1
# Следующие ключи НЕ задавайте, если нужны рекомендации карточки модели.
# Заданное значение переопределяет карточку для всех режимов.
# TEMPERATURE=1.0
# TOP_P=0.95
# TOP_K=20
# MIN_P=0.0
# PRESENCE_PENALTY=1.5
# REPETITION_PENALTY=1.0
# Режим рассуждений по умолчанию (true = <think>...</think>, false = прямой ответ)
THINKING=true
```

`reasoning_effort` управляет глубиной рассуждений в chat template, но не
переключает сэмплинг на «precise coding». Клиентские sampling-поля начинают
действовать только при явном `SAMPLING_LOCK=0`.

---

## 🇷🇺 Режимы сэмплинга моделей (Official Model Cards)

Сервер определяет семейство по имени активного GGUF и при старте/переключении
полностью заменяет набор пресетов. WebUI получает этот же набор через
`GET /v1/models`; значения предыдущей модели не переносятся.

| Семейство | Режим | temperature | top_p | top_k | presence_penalty |
|---|---|---:|---:|---:|---:|
| Gemma 4 | любой | 1.0 | 0.95 | 64 | 0.0 |
| Ornith 1.5 | general | 1.0 | 0.95 | 20 | 1.5 |
| Ornith 1.5 | precise coding | 0.6 | 0.95 | 20 | 0.0 |
| Qwen 3.6 / 3.8 | thinking | 1.0 | 0.95 | 20 | 0.0 |
| Qwen 3.6 / 3.8 | instruct | 0.7 | 0.80 | 20 | 1.5 |
| Qwen 3.6 MoE (35B-A3B) | thinking | 1.0 | 0.95 | 20 | 1.5 |
| Qwen 3.5 | thinking | 1.0 | 0.95 | 20 | 1.5 |

Во всех встроенных пресетах `min_p=0.0` и `repetition_penalty=1.0`.
Пользовательские изменения кнопкой «Сохранить как пресет режима» хранятся
раздельно по семействам в `MODEL_PRESETS`, поэтому настройка Gemma не меняет
Qwen или Ornith.

---

## 🇷🇺 Быстрый старт и сборка

### Windows (CUDA 13.2):
```cmd
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64
cargo build --release --features cuda
target\release\qwen36-server.exe
```

### Linux (CUDA):
```bash
cargo build --release --features cuda
./target/release/qwen36-server
```

### macOS (Apple Silicon Metal):
```bash
cargo build --release --features metal
./target/release/qwen36-server
```

---

## Unsloth Studio (WebUI)

Сервер отдаёт Unsloth Studio как основной UI на том же порту `:18099`.
Все запросы кроме `/v1/*` проксируются на Python-бэкенд Studio.

### Запуск

1. **Установить Unsloth Studio** на машине с сервером (yttri-win):
   ```powershell
   irm https://unsloth.ai/install.ps1 | iex
   ```

2. **Запустить Studio backend**:
   ```powershell
   unsloth studio -H 127.0.0.1 -p 8888
   ```

3. **Rust-сервер** (`:18099`) — проксирует `/*` на `STUDIO_URL` (по умолчанию `http://127.0.0.1:8888`).
   При недоступности Studio — fallback на встроенный `web/index.html` (admin-чат).

4. **В Studio UI**: Connections → Add Provider → Base URL `http://127.0.0.1:18099/v1`, ключ `smoke-key`.

```env
STUDIO_URL=http://127.0.0.1:8888
```

---

## 🇬🇧 English Configuration (.env)

```env
# Authentication & Network
API_KEYS='[{"key":"your-api-key","name":"primary"}]'
HOST=0.0.0.0
PORT=18099

# Model paths
MODEL=D:\Models\deepreinforce-ai\Ornith-1.5-9B-GGUF\Ornith-1.5-9B-Q6_K.gguf
MODELS_DIR=D:\Models

# Compute & Context
CTX=131072
SLOTS=4
MAX_TOKENS=32768
MTP=0
REQ_TIMEOUT=600

# Server-owned sampling policy. Omit values to use the model-card preset.
SAMPLING_LOCK=1
# TEMPERATURE=1.0
# TOP_P=0.95
# TOP_K=20
# MIN_P=0.0
# PRESENCE_PENALTY=1.5
# REPETITION_PENALTY=1.0
THINKING=true
```
