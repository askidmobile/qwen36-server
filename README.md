# Yttri Self-Inference Server

> Высокопроизводительный сервер локального инференса для моделей Qwen 3.5/3.6/3.8 и Ornith 1.0/1.5 (GGUF) на базе оптимизированного форка `candle` с поддержкой FlashAttention-2, Tensor-Core MMQ, аппаратно-ускоренного MTP (Next-N speculative decoding), непрерывного батчинга, multimodal (Vision/Video) и автоматической горячей смены моделей.

> High-performance local inference server for Qwen 3.5/3.6/3.8 and Ornith 1.0/1.5 (GGUF) based on an optimized `candle` fork supporting FlashAttention-2, Tensor-Core MMQ, hardware-accelerated MTP speculative decoding, continuous batching, multimodal vision/video, and automatic hot-swapping.

---

## 🇷🇺 Инструкция на русском языке

### Возможности
- **Поддержка архитектур**: Qwen 3.5 (4B/9B), Qwen 3.6 (35B-A3B MoE), Qwen 3.8 (27B), Ornith 1.0/1.5 (9B/35B).
- **Совместимость с OpenAI & Anthropic API**: Эндпоинты `/v1/chat/completions`, `/v1/responses`, `/v1/messages`.
- **Автоматическое переключение моделей (Hot-Switching)**: Сервер автоматически переключает модель при поступлении запроса с другим `model` id.
- **Официальный Jinja Chat Template**: Рендеринг встроенного шаблона из GGUF с поддержкой `reasoning_effort` (`low`/`medium`/`xhigh`), `preserve_thinking` и вызова инструментов (tool calling).
- **Встроенный WebUI**: Современный интерфейс со сплиттером мышления, live-счетчиком tok/s, управлением сервером и интеграцией с Hugging Face.
- **Поддержка квантов**: Q4_K_M, Q5_K_M, Q6_K, Q8_0, IQ2_XXS, IQ3_M, IQ4_XS.

---

### Требования к системе
- **ОС**: Windows 10/11 x64, Linux (Ubuntu 22.04+), macOS (Apple Silicon).
- **GPU**: NVIDIA GPU с CUDA Compute Capability ≥ 8.0 (RTX 3060/3070/3080/4090 и др., CUDA 12.4+). Для macOS — Apple Silicon (Metal).
- **Инструменты**: 
  - Rust stable (1.80+)
  - Python 3.10+
  - CMake & MSVC C++ Build Tools (на Windows) / gcc & g++ (на Linux)

---

### Быстрый старт

#### 1. Клонирование репозитория
```bash
git clone https://github.com/askidmobile/qwen36-server.git
cd qwen36-server
```

#### 2. Настройка конфигурации (.env)
Создайте файл `.env` в корневой директории проекта (см. `.env.example`):

```env
# Именованные API-ключи для авторизации
QWEN36_API_KEYS='[{"key":"yttri-secret-key-12345","name":"admin"}]'

# Путь к стартовой GGUF-модели
QWEN36_MODEL=D:\Models\deepreinforce-ai\Ornith-1.5-9B-GGUF\Ornith-1.5-9B-Q6_K.gguf

# Директория с локальными моделями для сканирования и авто-переключения
QWEN36_MODELS_DIR=D:\Models

# Сетевые параметры
QWEN36_HOST=0.0.0.0
QWEN36_PORT=18099

# Контекстное окно и количество параллельных слотов
QWEN36_CTX=131072
QWEN36_SLOTS=4
QWEN36_MAX_TOKENS=32768

# Дефолтные параметры сэмплинга
QWEN36_TEMPERATURE=0.6
QWEN36_TOP_P=0.95
QWEN36_TOP_K=20
QWEN36_MIN_P=0.0
QWEN36_PRESENCE_PENALTY=0.0
QWEN36_REPETITION_PENALTY=1.0
QWEN36_THINKING=true
```

#### 3. Сборка и запуск

##### Windows (CUDA):
```cmd
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64
cargo build --release --features cuda
target\release\qwen36-server.exe
```

##### Linux (CUDA):
```bash
cargo build --release --features cuda
./target/release/qwen36-server
```

##### macOS (Apple Silicon Metal):
```bash
cargo build --release --features metal
./target/release/qwen36-server
```

---

### Подключение клиентов (pi-coding-agent / Cline / OpenCode)

Добавьте в `~/.pi/agent/models.json`:

```json
{
  "providers": {
    "yttri-local": {
      "baseUrl": "http://127.0.0.1:18099/v1",
      "api": "openai-completions",
      "apiKey": "yttri-secret-key-12345",
      "compat": {
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": false,
        "thinkingFormat": "qwen-chat-template"
      },
      "models": [
        {
          "id": "ornith-1.5-9b",
          "name": "Ornith 1.5 9B Q6_K",
          "reasoning": true,
          "input": ["text", "image"],
          "contextWindow": 131072,
          "maxTokens": 32768
        },
        {
          "id": "qwen3.8-27b",
          "name": "Qwen3.8 27B",
          "reasoning": true,
          "input": ["text"],
          "contextWindow": 131072,
          "maxTokens": 32768
        }
      ]
    }
  }
}
```

---
---

## 🇬🇧 English Documentation

### Features
- **Supported Architectures**: Qwen 3.5 (4B/9B), Qwen 3.6 (35B-A3B MoE), Qwen 3.8 (27B), Ornith 1.0/1.5 (9B/35B).
- **OpenAI & Anthropic API Compatible**: `/v1/chat/completions`, `/v1/responses`, `/v1/messages`.
- **Automatic Model Hot-Switching**: The server automatically switches models on incoming requests when the requested `model` changes.
- **Native GGUF Jinja Chat Template**: Uses embedded Jinja chat templates with `reasoning_effort` (`low`/`medium`/`xhigh`), `preserve_thinking`, and tool calling support.
- **Built-in WebUI**: Web interface with thought-folding, live tok/s counters, server management panel, and Hugging Face GGUF downloader.
- **Supported Quant Types**: Q4_K_M, Q5_K_M, Q6_K, Q8_0, IQ2_XXS, IQ3_M, IQ4_XS.

---

### Prerequisites
- **OS**: Windows 10/11 x64, Linux (Ubuntu 22.04+), macOS (Apple Silicon).
- **GPU**: NVIDIA GPU with CUDA Compute Capability ≥ 8.0 (RTX 3060/3070/3080/4090, CUDA 12.4+). macOS requires Apple Silicon (Metal).
- **Tools**:
  - Rust stable (1.80+)
  - Python 3.10+
  - CMake & MSVC C++ Build Tools (Windows) / gcc & g++ (Linux)

---

### Quick Start

#### 1. Clone the repository
```bash
git clone https://github.com/askidmobile/qwen36-server.git
cd qwen36-server
```

#### 2. Configure environment (.env)
Create a `.env` file in the root directory:

```env
# Named API keys for client authorization
QWEN36_API_KEYS='[{"key":"yttri-secret-key-12345","name":"admin"}]'

# Path to the initial GGUF model
QWEN36_MODEL=D:\Models\deepreinforce-ai\Ornith-1.5-9B-GGUF\Ornith-1.5-9B-Q6_K.gguf

# Directory containing local models for scanning and auto-switching
QWEN36_MODELS_DIR=D:\Models

# Network options
QWEN36_HOST=0.0.0.0
QWEN36_PORT=18099

# Context length and continuous batch slots
QWEN36_CTX=131072
QWEN36_SLOTS=4
QWEN36_MAX_TOKENS=32768

# Default sampling parameters
QWEN36_TEMPERATURE=0.6
QWEN36_TOP_P=0.95
QWEN36_TOP_K=20
QWEN36_MIN_P=0.0
QWEN36_PRESENCE_PENALTY=0.0
QWEN36_REPETITION_PENALTY=1.0
QWEN36_THINKING=true
```

#### 3. Build and Run

##### Windows (CUDA):
```cmd
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64
cargo build --release --features cuda
target\release\qwen36-server.exe
```

##### Linux (CUDA):
```bash
cargo build --release --features cuda
./target/release/qwen36-server
```

##### macOS (Apple Silicon Metal):
```bash
cargo build --release --features metal
./target/release/qwen36-server
```

---

### Connecting Agents & Coding Assistants

Add to your `~/.pi/agent/models.json` (or equivalent client configuration):

```json
{
  "providers": {
    "yttri-local": {
      "baseUrl": "http://127.0.0.1:18099/v1",
      "api": "openai-completions",
      "apiKey": "yttri-secret-key-12345",
      "compat": {
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": false,
        "thinkingFormat": "qwen-chat-template"
      },
      "models": [
        {
          "id": "ornith-1.5-9b",
          "name": "Ornith 1.5 9B Q6_K",
          "reasoning": true,
          "input": ["text", "image"],
          "contextWindow": 131072,
          "maxTokens": 32768
        },
        {
          "id": "qwen3.8-27b",
          "name": "Qwen3.8 27B",
          "reasoning": true,
          "input": ["text"],
          "contextWindow": 131072,
          "maxTokens": 32768
        }
      ]
    }
  }
}
```

---

### License
Dual-licensed under MIT and Apache 2.0.
