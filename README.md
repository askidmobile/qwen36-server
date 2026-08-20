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
API_KEYS='[{"key":"yttri-api-Grt6l4bjIm-97jkoPxFFq4FGpiCmkn3ZUULIjYPc8Tg","name":"primary"}]'
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
# 4. ДЕФОЛТНЫЕ ПАРАМЕТРЫ СЭМПЛИНГА (OpenAI-стандарт)
# Примечание: Любой параметр, переданный клиентом в запросе, имеет наивысший приоритет!
# ==============================================================================
# Температура сэмплинга (0.6 для точного кодинга, 1.0 для общих задач)
TEMPERATURE=0.6
# Nucleus sampling порог
TOP_P=0.95
# Top-K фильтрация (топ наиболее вероятных токенов)
TOP_K=20
# Min-P фильтрация относительно вероятности лучшего токена
MIN_P=0.0
# Штраф за присутствие (0.0 в режиме рассуждений для предотвращения зацикливаний!)
PRESENCE_PENALTY=0.0
# Штраф за повторение
REPETITION_PENALTY=1.0
# Режим рассуждений по умолчанию (true = <think>...</think>, false = прямой ответ)
THINKING=true
```

---

## 🇷🇺 Режимы сэмплинга моделей (Official Model Cards)

### 🐦 Ornith 1.5 (9B / 35B)
1. **Thinking mode for precise coding tasks (WebDev / Agentic)**:
   - `temperature = 0.6`, `top_p = 0.95`, `top_k = 20`, `min_p = 0.0`, `presence_penalty = 0.0`, `repetition_penalty = 1.0`
2. **Thinking mode for general tasks**:
   - `temperature = 1.0`, `top_p = 0.95`, `top_k = 20`, `min_p = 0.0`, `presence_penalty = 1.5`, `repetition_penalty = 1.0`
3. **Instruct (non-thinking) mode**:
   - `temperature = 0.7`, `top_p = 0.80`, `top_k = 20`, `min_p = 0.0`, `presence_penalty = 1.5`, `repetition_penalty = 1.0`

---

## 🇷🇺 Быстрый старт и сборка

### Windows (CUDA 12.4+):
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

# Sampling Defaults
TEMPERATURE=0.6
TOP_P=0.95
TOP_K=20
MIN_P=0.0
PRESENCE_PENALTY=0.0
REPETITION_PENALTY=1.0
THINKING=true
```
