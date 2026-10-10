# QAMQ: рабочий план Qwen-aware mixed quantization

Дата: 2026-10-11.
Контекст: [research](../research/2026-10-11-quantization-theory-and-custom-schemes.md).
Задача в TASKS.md: **T-010**.
Цель: снизить bytes/token и/или увеличить tokens/pass для Qwen3.5-9B на `yttri-win` (RTX 3060 12 GB), сохранив качество на RU/EN/code/agent/long-context задачах.

## Статус

| Фаза | Статус | Отчёт |
|---|---|---|
| Phase 0 — measurement/sensitivity | 🔄 In progress | будет `docs/research/2026-10-11-qamq-phase0.md` |
| Phase 1 — QAMQ-A mixed K-quant | ⏳ Ожидает Phase 0 | будет `docs/research/2026-10-11-qamq-a.md` |
| Phase 2 — QAMQ-B rotation/AWQ/low-rank W4 | 📝 Запланировано | — |
| Phase 3 — QAMQ-C DeltaNet/SSM/KV/MTP | 📝 Запланировано | — |
| Phase 4 — QAMQ-D VQ/TCQ 2–3 бита | 📝 Запланировано | — |

## Артефакты и пути

### Локально (macOS)
- Рабочий план: `docs/plans/2026-10-11-qwen-aware-quantization.md`.
- Research: `docs/research/2026-10-11-quantization-theory-and-custom-schemes.md`.
- Отчёты фаз: `docs/research/2026-10-11-qamq-phase0.md`, `docs/research/2026-10-11-qamq-a.md`.
- Скрипты подготовки калибровки: `scripts/qamq/` (создать при необходимости).

### yttri-win
- Корень работ: `D:\Projects\yttri-inference\qamq\`.
- Калибровка: `D:\Projects\yttri-inference\qamq\calibration\`.
- Imatrix: `D:\Projects\yttri-inference\qamq\imatrix\`.
- Модели: `D:\Models\unsloth\Qwen3.5-9B-MTP-GGUF\`.
- Инструменты llama.cpp: `D:\Tools\llama.cpp\` (prebuilt release или сборка).
- Логи: `D:\Projects\yttri-inference\logs\qamq-*.log`.
- Результаты A/B: `D:\Projects\yttri-inference\qamq\results\`.

## Методология A/B

- Один и тот же stand: `yttri-win`, RTX 3060 12 GB.
- Один и тот же MTP-файл для baseline: `unsloth/Qwen3.5-9B-MTP-GGUF/Qwen3.5-9B-Q4_K_M.gguf`.
- QAMQ-A собирается из **BF16/F16** исходника (не из Q4_K_M) с imatrix, собранным на нашем калибровочном наборе.
- `PREFIX_CACHE_MIB=0`, `TRACE=0`, `MTP_TIMING=0` на всех замерах.
- Decode: 30 prompt-токенов, `max_tokens=128`, `temperature=0`, 3 прогона после загрузки.
- Prefill: 6412 prompt-токенов, `max_tokens=1`, полное время запроса.
- VRAM: `nvidia-smi` после загрузки модели до запроса.
- Качество: PPL/KLD на held-out наборе, MMLU/GSM8K/HumanEval/MBPP, RU-задачи, tool-calling, needle/RULER.
- Все прогоны логируются; raw-логи сохраняются в `D:\Projects\yttri-inference\logs\qamq-*`.

## Phase 0 — measurement and sensitivity map

### 0.1 Инструменты llama.cpp
- [ ] Проверить наличие `llama-imatrix.exe`, `llama-quantize.exe`, `llama-perplexity.exe`, `llama-cli.exe`, `llama-server.exe`.
- [ ] Если нет — скачать prebuilt Windows CUDA release с поддержкой Qwen3.5/MTP и `--nextn`; иначе собрать из master.
- [ ] Проверить версию: `llama-cli --version`, `llama-quantize --help` (наличие `--tensor-type`, `--tensor-type-file`, `--imatrix`), `llama-imatrix --help` (наличие `--nextn`).
- [ ] Зафиксировать версию/commit в отчёте Phase 0.

### 0.2 Калибровочный набор
- [ ] Собрать `calibration\qamq-calib.txt`: RU/EN code, chat, tool-calling, long context, технические тексты.
- [ ] Целевой объём: 2–5 МБ; проверить, что нет бинарных/секретных данных.
- [ ] Разделить на `calib` и `heldout` (held-out не используется в imatrix).
- [ ] Зафиксировать sha256 файлов.

### 0.3 Baseline измерения
- [ ] Загрузить `Qwen3.5-9B-Q4_K_M.gguf` (unsloth MTP).
- [ ] `MTP=0`: decode/prefill/VRAM.
- [ ] `MTP=1`, `MTP_TREE=0`, W1: decode/prefill/VRAM, acceptance.
- [ ] `llama-perplexity` на held-out: PPL.
- [ ] Если доступно — KLD к BF16/F16 reference.
- [ ] Сохранить raw-логи и сводку.

### 0.4 Imatrix
- [ ] `llama-imatrix -m <model> -f calib.txt --nextn -o imatrix.gguf -ngl 99` (или partial offload).
- [ ] Если BF16 не влезает — использовать Q4_K_M для статистики; зафиксировать это как ограничение.
- [ ] Проверить, что imatrix содержит MTP/NextN тензоры.
- [ ] Сохранить imatrix и лог.

### 0.5 Sensitivity map
- [ ] Извлечь per-tensor статистику (mean/std/skew/kurtosis/H-norm, если доступно).
- [ ] Собрать карту: tensor → current type → candidate type → PPL/KLD/score.
- [ ] Приоритет: SSM (`ssm_a/alpha/beta/conv1d/dt/norm/out`), attention (`q/k/v/o/gate/qkv`), FFN (`gate/up/down`), embeddings/output, MTP `nextn.*`.
- [ ] Выход: `qamq-phase0-sensitivity.md` + JSON/CSV.

### 0.6 Отчёт Phase 0
- [ ] `docs/research/2026-10-11-qamq-phase0.md`: что сделано, что не получилось, какие ограничения, какие выводы для QAMQ-A.

## Phase 1 — QAMQ-A mixed K-quant

### 1.1 Рецепт
- [ ] На основе sensitivity map задать `--tensor-type` overrides.
- [ ] Стартовая политика:
  - F32/F16: нормы, `ssm_a`, `ssm_dt.bias`, `ssm_norm`;
  - Q8/Q6: `ssm_alpha`, `ssm_beta`, `ssm_conv1d`, `ssm_out`, embeddings/output, MTP `nextn.*`;
  - Q5/Q6: attention q/k/v/o/gate/qkv;
  - Q4/Q5: DeltaNet in_proj/out_proj, ffn_down;
  - Q3/Q4: ffn_gate/ffn_up.
- [ ] Проверить, что все выбранные типы поддерживаются нашим Candle-загрузчиком.

### 1.2 Сборка кванта
- [ ] `llama-quantize --imatrix imatrix.gguf --tensor-type ... <BF16/F16 source> qamq-a.gguf <base type>`.
- [ ] Зафиксировать effective bpw и распределение типов.
- [ ] Проверить размер и sha256.

### 1.3 Проверка загрузки
- [ ] Загрузить `qamq-a.gguf` в `yforge` с `MTP=0` и `MTP=1`.
- [ ] Проверить отсутствие ошибок загрузки/падений, корректность MTP head.

### 1.4 A/B
- [ ] Baseline `Q4_K_M` vs `QAMQ-A`: decode/prefill/VRAM.
- [ ] PPL/KLD на held-out.
- [ ] Task evals: MMLU/GSM8K/HumanEval/MBPP, RU, tool-calling, needle/RULER.
- [ ] MTP acceptance для `MTP=1`, W1.

### 1.5 Gate и отчёт
- Gate: качество не хуже Q4_K_M на выбранном наборе; decode ≥ 58 t/s.
- [ ] `docs/research/2026-10-11-qamq-a.md`: методика, сырые метрики, что сработало/не сработало, вывод и решение по Phase 2/3.

## Рабочий журнал

| Дата | Действие | Результат | Артефакт/лог |
|---|---|---|---|
| 2026-10-11 | Research теорий и методов | Отчёт с 40+ источниками | `docs/research/2026-10-11-quantization-theory-and-custom-schemes.md` |
| 2026-10-11 | Создан рабочий план и T-010 | План зафиксирован | этот файл, `TASKS.md` |
| 2026-10-11 | Начата Phase 0: подготовка инструментов | — | — |

## Риски и решения

- **BF16 не влезает в 12 GB для imatrix** → использовать Q4_K_M/Q8_0 для статистики, BF16 для quantize; либо partial offload.
- **Наш Candle-загрузчик не поддерживает какой-то тип** → исключить тип из рецепта или добавить поддержку.
- **llama.cpp prebuilt не содержит MTP/nextn** → собрать из master или использовать release после PR #22673.
- **Quality gate не проходит** → вернуться к sensitivity map, поднять битность sensitive tensors, либо перейти к QAMQ-B/C.
- **Server занят/выключен** → зафиксировать паузу и продолжить локальную подготовку артефактов.

## Definition of Done для T-010

- Phase 0 отчёт с sensitivity map.
- QAMQ-A GGUF собран и загружается в `yforge`.
- Полный A/B на `yttri-win`: decode/prefill/VRAM/PPL/KLD/task evals.
- Phase 1 отчёт с выводом: прошёл/не прошёл gate, что делать в Phase 2/3.
- Все артефакты и логи сохранены; коммит в репозитории; TASKS.md обновлён.
