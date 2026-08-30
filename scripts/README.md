# Запуск сервера

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
Экешник: `D:\Projects\yttri-inference\qwen36-server\target\release\qwen36-server.exe` — всегда свежее корневого `D:\Projects\yttri-inference\target\release`, сверяйте даты.
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
QWEN36_MODEL=D:\Models\unsloth\Qwen3.8-27B-GGUF\Qwen3.8-27B-UD-IQ2_XXS.gguf
QWEN36_MTP=1
QWEN36_MTP_PATH=D:\Models\unsloth\Qwen3.8-27B-GGUF\mtp-Qwen3.8-27B-Q4_0.gguf
QWEN36_CTX=131072
QWEN36_SLOTS=4
GPU_ONLY=1
```

`QWEN36_MTP_PATH` — draft-модель для спекуляции; работает и без профиля-манифеста.
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
