# Запуск сервера

Бинарь называется **`yforge`** (`yforge.exe` на Windows). Логи идут в
stdout/stderr — перенаправляйте средствами оболочки, как у `llama.cpp`.

Настройки берутся из трёх источников, приоритет по убыванию:
флаги командной строки → окружение процесса → env-файл (`--env`).
Имена переменных без префикса (`CTX`, `SLOTS`), устаревшие `QWEN36_*`
по-прежнему принимаются. Полный список флагов — `yforge --help`
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
|---|---|---|
| Windows (yttri-win, продакшен) | `windows/inference-run.bat` | экзешник + `.env` + лог в `logs\server.log` |
| Windows (профильная схема) | `windows/run_windows.bat` → `qwen35_run_current.ps1` | запуск релиза через `current`-указатель с проверкой SHA-256 профиля |
| macOS (разработка) | `macos/run-metal.sh` | `cargo run --release --features metal` |
| Linux (CUDA) | `linux/run-cuda.sh` | сборка при необходимости + запуск бинарника |

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

### Синхронизация кода с yttri-win (аудит 2026-08-31)

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

## Профильная схема (run_windows.bat)

`QWEN36_CURRENT=D:\Models\yttri\qwen3.5-4b\current` — JSON-указатель на релиз
с манифестом (`profile.json`, SHA-256). Сервер и артефакты (vision/mtp) берутся
из манифеста. Используется для проверок тестовой модели 3.5-4B, не для продакшена 27B.
