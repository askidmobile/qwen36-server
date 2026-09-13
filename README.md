# Yttri Forge (`yforge`)

> Высокопроизводительный сервер локального инференса для моделей Qwen 3.5/3.6/3.8 и Ornith 1.0/1.5 (GGUF) на базе оптимизированного форка `candle` с поддержкой FlashAttention-2, Tensor-Core MMQ, аппаратно-ускоренного MTP (Next-N speculative decoding), непрерывного батчинга, multimodal (Vision/Video) и автоматической горячей смены моделей.

> High-performance local inference server for Qwen 3.5/3.6/3.8 and Ornith 1.0/1.5 (GGUF) based on an optimized `candle` fork supporting FlashAttention-2, Tensor-Core MMQ, hardware-accelerated MTP speculative decoding, continuous batching, multimodal vision/video, and automatic hot-swapping.

Совместимые API: **OpenAI Chat Completions**, **OpenAI Responses**, **Anthropic Messages**, плюс встроенный веб-чат.

---

## Быстрый старт

```bash
# минимум: модель и ключ
yforge --model /models/Ornith-1.5-9B-Q4_K_M.gguf --api-key my-secret-key

# рабочая конфигурация на 12 ГБ VRAM, без MTP
yforge --model /models/Ornith-1.5-9B-Q4_K_M.gguf \
       --api-key my-secret-key \
       --ctx 131072 --slots 1 \
       --kv-pool q8 --cuda-graphs 1 --prefix-cache 8192 --mtp 0
# PGRAPH=on: paged/graph-prefill включён (после починки токенизатора чист);
# при off на int8-пуле раньше ломался prefix cache — тоже починено.

# всё то же самое из файла
yforge --env prod.env
```

Логи идут в **stdout/stderr**, как у `llama.cpp` — перенаправляйте средствами оболочки:

```bash
yforge --env prod.env > server.log 2>&1        # Linux/macOS
yforge.exe --env prod.env > server.log 2>&1    # Windows
```

Пример вывода при старте:

```
yforge: === конфигурация ===
yforge: build                  = 0.1.0 (CUDA)
yforge: env file               = prod.env
yforge: model                  = D:\Models\Ornith-1.5-9B-Q4_K_M.gguf
yforge: listen                 = 0.0.0.0:18099
yforge: n_ctx                  = 131072
yforge: n_slots                = 1
yforge: kv_pool                = q8 (int8, вдвое меньше VRAM)
yforge: cuda_graphs            = включены
...
yforge: === модель загружена ===
yforge: id                     = ornith-1.5-9b
yforge: quant                  = Q4_K_M
yforge: === сервер запущен ===
yforge: OpenAI API             = http://0.0.0.0:18099/v1
yforge: готов принимать запросы
```

---

## Три источника настроек

Приоритет по убыванию, как у `llama.cpp`:

| # | Источник | Пример |
| --- | --- | --- |
| 1 | Флаг командной строки | `yforge --ctx 131072` |
| 2 | Переменная окружения | `CTX=131072 yforge` |
| 3 | Env-файл | `yforge --env prod.env` |

Флаг перекрывает окружение, окружение перекрывает файл:

```bash
CTX=65536 yforge --env prod.env --ctx 32768   # победит 32768 (флаг)
CTX=65536 yforge --env prod.env               # победит 65536 (окружение)
yforge --env prod.env                          # победит значение из файла
```

Проверить итоговую конфигурацию, не загружая модель:

```bash
yforge --env prod.env --dry-run
```

### Имена переменных

Без префикса: `CTX`, `SLOTS`, `MODEL`. Одно имя читают обе стороны — сервер и
движок. Прежние формы `QWEN36_*` и `YTTRI_*` больше не поддерживаются: если в
вашем `.env` они остались, уберите префикс.

---

## Параметры командной строки

### Модель

| Флаг | Переменная | Умолчание | Описание |
| --- | --- | --- | --- |
| `-m, --model PATH` | `MODEL` | — | Путь к GGUF-файлу модели |
| `--models-dir DIR` | `MODELS_DIR` | родитель модели | Каталог для сканирования и горячей смены |
| `--profile PATH` | `PROFILE` | — | Профиль модели (manifest.json), заменяет `--model` |

### Сеть и доступ

| Флаг | Переменная | Умолчание | Описание |
| --- | --- | --- | --- |
| `--host IP` | `HOST` | `0.0.0.0` | Адрес прослушивания |
| `-p, --port PORT` | `PORT` | `18099` | Порт |
| `--api-key KEY` | — | — | Один ключ (короткая форма) |
| `--api-keys JSON` | `API_KEYS` | — | `[{"key":"...","name":"..."}]` |

### Контекст и слоты

| Флаг | Переменная | Умолчание | Описание |
| --- | --- | --- | --- |
| `-c, --ctx N` | `CTX` | `131072` | Размер контекстного окна, токенов |
| `-s, --slots N` | `SLOTS` | `4` | Параллельных слотов генерации |
| `-n, --max-tokens N` | `MAX_TOKENS` | `32768` | Потолок генерации по умолчанию |
| `--req-timeout SEC` | `REQ_TIMEOUT` | `600` | Таймаут запроса |
| `--max-queue N` | `MAX_QUEUE` | `64` | Длина очереди |

### Память и вычисления

| Флаг | Переменная | Умолчание | Описание |
| --- | --- | --- | --- |
| `--gpu-layers N`, `--ngl` | `GPU_LAYERS` | `999` | Слоёв на GPU (999 = все) |
| `--kv-pool q8\|f16` | `KV_POOL_Q8` | `f16` | Тип **постоянного** страничного пула KV |
| `--kv-cache-type q8\|q8_f16` | `KV_CACHE_DTYPE` | `q8_f16` | Тип **временного** batched-KV |
| `--vram-headroom MIB` | `VRAM_HEADROOM_MIB` | `1024` | Резерв VRAM под транзиенты |
| `--prefix-cache MIB` | `PREFIX_CACHE_MIB` | `0` | Кеш префикса промпта (системная память) |
| — | `PREFIX_CACHE_CHECKPOINTS` | `1` | Дополнительные branch-point checkpoints (степени двойки); `0` = только последняя граница чанка |
| — | `PREFIX_CACHE_CHECKPOINT_MAX` | `8192` | Верхняя позиция дополнительного checkpoint'а, токенов |
| `--cuda-graphs 0\|1` | `CUDA_GRAPHS` | `0` | CUDA-графы |
| `--flash-attn 0\|1`, `--fa` | `FLASH_ATTN` | `1` | Flash Attention |
| `-t, --threads N` | `THREADS` | ядра CPU | Потоков CPU |
| `-b, --batch-size N` | `BATCH_SIZE` | `2048` | Батч префилла |
| `--gpu-only 0\|1` | `GPU_ONLY` | — | Держать все веса на GPU |

| `--moe-experts vram\|ram\|auto` | `MOE_EXPERTS` | `auto` | Размещение маршрутизируемых экспертов MoE: `ram` — pinned host-память (zero-copy, 35B-A3B на 12 ГБ → окно 131072); требует RAM ≥ эксперты + prefix cache + 2 ГиБ |
| `--expert-cache MIB` | `EXPERT_CACHE_MIB` | auto | VRAM-кэш горячих экспертов (фаза 4 плана moe-expert-offload) |

> **`--kv-pool` — главный рычаг VRAM.** Постоянный пул выделяется на **всё
> окно контекста сразу при старте**, а не растёт по мере заполнения. На 128K
> это ≈2.6 ГБ в `f16` против ≈1.3 ГБ в `q8`. Если пул не помещается в
> видеопамять, Windows (WDDM) молча уводит его в RAM, и генерация замедляется
> в разы. Точность int8 на KV — 0.75 % round-trip, на качестве ответов не
> сказывается. См. раздел «Планирование VRAM».

### Спекулятивное декодирование (MTP)

| Флаг | Переменная | Умолчание | Описание |
| --- | --- | --- | --- |
| `--mtp 0\|1` | `MTP` | `0` | Включить MTP |
| `--mtp-path PATH` | `MTP_PATH` | — | GGUF головы MTP |
| `--mtp-shortlist FILE` | `MTP_VOCAB_SHORTLIST` | — | Шортлист словаря черновика (файл с id токенов) |
| `--verify-onepass 0\|1` | `VERIFY_ONEPASS` | `0` | Однопроходная проверка спекуляции |

### Сэмплинг

| Флаг | Переменная | Умолчание | Описание |
| --- | --- | --- | --- |
| `--temp F` | `TEMPERATURE` | из карточки модели | Температура |
| `--top-p F` | `TOP_P` | из карточки | Top-p (nucleus) |
| `--top-k N` | `TOP_K` | из карточки | Top-k |
| `--min-p F` | `MIN_P` | из карточки | Min-p |
| `--seed N` | `SEED` | `0` (случайный) | Seed |
| `--thinking true\|false` | `THINKING` | `true` | Режим рассуждений по умолчанию |
| `--sampling-lock 0\|1` | `SAMPLING_LOCK` | `1` | Держать серверную политику сэмплинга |

### Прочее

| Флаг | Переменная | Умолчание | Описание |
| --- | --- | --- | --- |
| `--env FILE` | `ENV_FILE` | `.env` | Env-файл с параметрами |
| `--studio-url URL` | `STUDIO_URL` | `http://127.0.0.1:8888` | Бэкенд Unsloth Studio для WebUI |
| `--dry-run` | — | — | Показать конфигурацию и выйти |

Полный список: `yforge --help`.

---

## Env-файл

Формат `NAME=value`, строки с `#` — комментарии, `export` в начале
допускается. Файл читается **последним**, поэтому не перетирает флаги и
окружение.

```env
# ── Сеть и авторизация ────────────────────────────────────────────────
API_KEYS='[{"key":"your-secret-api-key","name":"primary"}]'
HOST=0.0.0.0
PORT=18099

# ── Модель ────────────────────────────────────────────────────────────
MODEL=D:\Models\deepreinforce-ai\Ornith-1.5-9B-GGUF\Ornith-1.5-9B-Q4_K_M.gguf
MODELS_DIR=D:\Models

# ── Контекст и слоты ──────────────────────────────────────────────────
CTX=131072
SLOTS=1
MAX_TOKENS=32768
REQ_TIMEOUT=600

# ── Память (см. «Планирование VRAM») ──────────────────────────────────
GPU_ONLY=1
GPU_LAYERS=999
KV_POOL_Q8=1              # int8-пул: вдвое меньше VRAM
KV_CACHE_DTYPE=q8         # временный batched-KV без F16-дубля
CUDA_GRAPHS=1             # ускоряет decode
PGRAPH=on                 # paged-prefill: KV сразу в пул, декод ~3x; «порча состояния» была дефектом токенизатора
VRAM_HEADROOM_MIB=1536    # резерв под транзиенты
PREFIX_CACHE_MIB=8192     # кеш префикса в системной памяти
# PREFIX_CACHE_CHECKPOINTS=0        # отключить branch-point checkpoints
# PREFIX_CACHE_CHECKPOINT_MAX=8192  # верхняя позиция раннего checkpoint'а

# ── MTP (production Ornith Q4_K_M работает без него) ──────────────────
MTP=0

# ── Политика сэмплинга ────────────────────────────────────────────────
# По умолчанию клиентские temperature/top_p/top_k/min_p/penalties
# игнорируются: источник истины — карточка модели и явные значения ниже.
SAMPLING_LOCK=1
THINKING=true
# TEMPERATURE=1.0
# TOP_P=0.95
# TOP_K=20
```

`MAX_TOKENS` — только умолчание: клиентские `max_tokens` и
`max_completion_tokens` в запросе имеют приоритет, `SAMPLING_LOCK` их не
затрагивает. Если присланы оба поля, их значения должны совпадать.

---

## Установка и запуск по ОС

Единственный артефакт — исполняемый файл `yforge` (`yforge.exe` на Windows).
Внешних зависимостей во время работы нет; модели — обычные GGUF-файлы.

### Windows + CUDA

**Требования:** NVIDIA-драйвер с поддержкой CUDA 12.4+, CUDA Toolkit 13.2,
Visual Studio 2022 Build Tools, Rust (MSVC toolchain).

**Сборка:**

```cmd
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2
set CUDA_COMPUTE_CAP=86
cargo build --release --features cuda --bin yforge
```

Готовый файл: `target\release\yforge.exe`.

**Запуск:**

```cmd
target\release\yforge.exe --env D:\configs\prod.env
```

**Как служба (переживает выход из сессии).** `Start-Process` из ssh-сессии
умирает вместе с ней — используйте планировщик заданий. Bat-обёртка:

```bat
@echo off
setlocal EnableExtensions
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin;%PATH%"
cd /d "D:\Projects\yttri-inference\qwen36-server"
target\release\yforge.exe --env .env 1>>"D:\Projects\logs\server.log" 2>&1
```

Регистрация и управление:

```cmd
schtasks /create /tn yforge /tr "cmd /c D:\path\run.bat" /sc onstart /ru SYSTEM
schtasks /run   /tn yforge
schtasks /end   /tn yforge
schtasks /query /tn yforge
```

`CUDA_COMPUTE_CAP` под вашу карту: 86 — RTX 30xx, 89 — RTX 40xx, 120 — RTX 50xx.

### Linux + CUDA

**Требования:** NVIDIA-драйвер, CUDA Toolkit 12.4+, `build-essential`, Rust.

```bash
export CUDA_COMPUTE_CAP=86
cargo build --release --features cuda --bin yforge
./target/release/yforge --env prod.env
```

**Как systemd-служба** — `/etc/systemd/system/yforge.service`:

```ini
[Unit]
Description=Yttri Forge inference server
After=network.target

[Service]
Type=simple
User=yforge
WorkingDirectory=/opt/yforge
ExecStart=/opt/yforge/yforge --env /opt/yforge/prod.env
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now yforge
journalctl -u yforge -f          # логи
```

### macOS + Metal (Apple Silicon)

**Требования:** macOS 13+, Xcode Command Line Tools, Rust.

```bash
cargo build --release --features metal --bin yforge
./target/release/yforge --env dev.env
```

Metal-сборка предназначена для разработки. Флаги `--cuda-graphs`,
`--kv-pool` и MTP относятся к CUDA-пути и на Metal не действуют.

**Как launchd-агент** — `~/Library/LaunchAgents/ai.yttri.yforge.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<plist version="1.0"><dict>
  <key>Label</key><string>ai.yttri.yforge</string>
  <key>ProgramArguments</key>
  <array>
    <string>/usr/local/bin/yforge</string>
    <string>--env</string><string>/usr/local/etc/yforge.env</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>StandardOutPath</key><string>/tmp/yforge.log</string>
  <key>StandardErrorPath</key><string>/tmp/yforge.err.log</string>
</dict></plist>
```

```bash
launchctl load ~/Library/LaunchAgents/ai.yttri.yforge.plist
```

### CPU (без GPU)

```bash
cargo build --release --bin yforge
./target/release/yforge --model model.gguf --api-key key --ctx 8192 --slots 1
```

Годится для маленьких моделей и проверки API; скорость генерации низкая.

---

## Планирование VRAM

Считать нужно **весь** бюджет, а не только KV:

| Статья | Как оценить |
| --- | --- |
| Веса модели | размер GGUF-файла |
| Голова MTP | размер её GGUF (если `--mtp 1`) |
| **Страничный пул KV** | `окно × слои_внимания × 2 × n_kv × (hd+2)` для q8, `×2·hd` для f16 |
| Транзиенты префилла | `--vram-headroom`, по умолчанию 1 ГиБ |
| Рабочий стол ОС | 0.3–1 ГБ на Windows |

Пул выделяется на **полное окно сразу при старте**, поэтому занятость
видеопамяти не зависит от того, насколько контекст заполнен.

Исторический замер на RTX 3060 12 ГБ, Ornith-1.5-9B Q6_K + MTP Q8_0, 128K, 1 слот (не текущая production-модель):

| Конфигурация | Idle VRAM | Пик | Выгрузка в RAM | Декод |
| --- | --- | --- | --- | --- |
| f16-пул, без графов | 10.4 ГБ | 10.8 ГБ | есть | 5.6 ток/с |
| **q8-пул + графы** | **6.9 ГБ** | **8.9 ГБ** | нет | **30.2 ток/с** |

Если в логе строка `[kv] paged pool` содержит `WARN: окно меньше контекста`,
пул не вместил запрошенный `--ctx` и будет вытеснять старые блоки. Лечение:
`--kv-pool q8`, меньший `--ctx` или меньше слотов.

---

## Режимы сэмплинга (карточки моделей)

Сервер определяет семейство по имени активного GGUF и при старте/переключении
полностью заменяет набор пресетов. WebUI получает этот же набор через
`GET /v1/models`; значения предыдущей модели не переносятся.

| Семейство | Режим | temperature | top_p | top_k | presence_penalty |
| --- | --- | ---: | ---: | ---: | ---: |
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

`reasoning_effort` управляет глубиной рассуждений в chat template, но не
переключает сэмплинг на «precise coding». Клиентские sampling-поля начинают
действовать только при явном `--sampling-lock 0`.

---

## Unsloth Studio (WebUI)

Сервер отдаёт Unsloth Studio как основной UI на том же порту `:18099`.
Все запросы кроме `/v1/*` проксируются на Python-бэкенд Studio.

1. Установить Studio на машине с сервером:

   ```powershell
   irm https://unsloth.ai/install.ps1 | iex
   ```

2. Запустить бэкенд:

   ```bash
   unsloth studio -H 127.0.0.1 -p 8888
   ```

3. Сервер проксирует `/*` на `--studio-url`. При недоступности Studio —
   fallback на встроенный `web/index.html` (admin-чат).
4. В Studio UI: Connections → Add Provider → Base URL
   `http://127.0.0.1:18099/v1`, ключ из `--api-key`.

---

## Диагностика

| Симптом | Причина | Что делать |
| --- | --- | --- |
| `CUDA_ERROR_OUT_OF_MEMORY` при префилле | пул KV + веса не влезли | `--kv-pool q8`, поднять `--vram-headroom`, снизить `--ctx`/`--slots` |
| Генерация замедлилась в разы | пул выгружен в RAM через WDDM | то же; проверить `GPU Process Memory → Shared Usage` |
| `WARN: окно меньше контекста` | пул не вместил `--ctx` | `--kv-pool q8` или меньший `--ctx` |
| `API_KEYS обязателен` | ключ не задан | `--api-key KEY` или `API_KEYS` в env-файле |
| Ответ обрывается, в тексте сырые рассуждения | клиентский `max_tokens` меньше длины thinking | увеличить `max_tokens` на клиенте |

Проверить, что именно применилось: `yforge --env prod.env --dry-run`.

---

## English quick reference

```bash
yforge --model model.gguf --api-key KEY --ctx 131072 --slots 1 \
       --kv-pool q8 --cuda-graphs 1
yforge --env prod.env                    # all settings from a file
yforge --env prod.env --dry-run          # print resolved config and exit
yforge --help                            # full flag list
```

Precedence: **CLI flags > process environment > env file**. Variable names
carry no prefix (`CTX`, `SLOTS`, `MODEL`); the legacy `QWEN36_*` and `YTTRI_*`
spellings are no longer accepted. Logs go to stdout/stderr in `llama.cpp`
style.

Build: `--features cuda` (Windows/Linux), `--features metal` (macOS), no
feature flag for CPU-only.
