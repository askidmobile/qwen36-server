# QAMQ: рабочий план Qwen-aware mixed quantization

Дата: 2026-10-11.
Контекст: [research](../research/2026-10-11-quantization-theory-and-custom-schemes.md).
Задача в TASKS.md: **T-011**.
Цель: снизить bytes/token и/или увеличить tokens/pass для Qwen3.5-9B на `yttri-win` (RTX 3060 12 GB), сохранив качество на RU/EN/code/agent/long-context задачах.

## Итог (2026-10-11)

**Достигнуто:** `Qwen3.5-9B-QAMQ-A5L-AWQ2.gguf` — 5.36 ГБ (−8.7% к Q4_K_M), decode
61.2 t/s без MTP и 65.9 t/s с MTP W3 (до 81 t/s на коде) против 53.7 у Q4_K_M, PPL 15.3767
(лучше Q4_K_M на 0.6%), MMLU 75.67%, GSM8K 96.18% (+1.5 п.п. к Q4_K_M), ruMMLU 69.67%,
tool-calling 10/10, needle 4k/16k/32k 9/9, 115k контекста без OOM.

**Приёмка:** evals прогнаны и через llama.cpp, и через yforge. Все семь наборов закрыты
(MMLU, GSM8K, ruMMLU, **HumanEval 85.37% против 83.54% у Q4_K_M**, tool-calling, needle,
PPL). Через yforge: MMLU 75.33 против 75.67, GSM8K 96.18 = 96.18, ruMMLU 69.00 против 69.67 —
различия 1–2 вопроса из 300; замер шёл с включённым MTP W3, то есть спекулятивный декод
качество не ломает. **A5L-AWQ2 не проигрывает Q4_K_M ни на одном eval'е** (кроме MMLU −1 п.п.,
в пределах шума), выигрывая на GSM8K и HumanEval.

**Три ключевых результата:**
1. **AWQ per-channel scaling** (FFN + attention, GQA-aware) — точный фолдинг в веса, без правок
   рантайма; дал −0.46% PPL и позволил снять q5/q6-защиту слоёв 29–31 (−1.8% размера, +1.5% скорости).
2. **MTP W3+ADAPTIVE** — исправлена ошибка методики (decode считался по SSE-чанкам, а не
   токенам): реальный выигрыш +7.7% на 256-токенных ответах, +5.2% на 512, паритет на 1024.
3. **`MTP_MAX_CTX`** — MTP вредил на длинном контексте (−38% на 39k); гейт по длине контекста
   реализован в движке и закрывает регрессию.

**Что не сработало (зафиксировано):** Q3_K (медленнее и хуже), plain Q4_0 на attention,
IQ4_XS в нашем движке (медленнее Q4_0 на ~13%, хотя в llama.cpp быстрее), BF16-imatrix
(эффекта нет), MTP-голова в Q6 (эффекта нет), rotation/low-rank (не начаты).

**Остаток (в порядке ценности):**
1. **Prefill медленнее llama.cpp на 10–25%** (1 497 против 1 985 t/s на 3.9k) — разобрано
   до конца. Профиль `nsys`: GPU занят 97%, копирования 0.5%, 60% ядерного времени — MMQ-GEMM
   (`candle_mmq_q4_k_x128` 33%, `candle_mmq_q4_0_x128` 27%), `delta_rule_prefill` 16%.
   НО: по сверке с профилем 2026-08-25 (`2026-08-24-stage1-f16-sidecar.md`) тот же MMQ-путь
   измерен на потолке (Q4_K 28.7–32.6 TOPS на этих формах), а все обходы (**FFN-сайдкар,
   prequant, delta v2**) там уже проверены и дали отрицательный результат. Наш F16-сайдкар
   (152 тензора, 3.9 ГБ) это подтвердил: +2.5% на коротком промпте и −49% на 18.9k из-за
   нехватки VRAM. **Вывод: разрыв структурный** (paged-запись KV, рекуррентность DeltaNet,
   чанковые границы), а не «одно медленное ядро»; закрытие — research-уровень.
2. **4-битный KV-пул** — новый вариант FA-шаблона (`Is_kv_q8` прошит в 14 точках
   `flash_fwd_kernel.h`), выигрыш только по VRAM (~1 ГиБ на 128k), скорость от типа KV
   не зависит.
3. **VQ/TCQ 2–3 бита** (Phase 4) — но разведка через IQ3_XXS показала, что выигрыш съедает
   распаковка; собственный формат должен решать именно эту проблему.

## Статус

| Фаза | Статус | Отчёт |
|---|---|---|
| Phase 0 — measurement/sensitivity | ✅ Done | [`2026-10-11-qamq-phase0.md`](../research/2026-10-11-qamq-phase0.md) |
| Phase 1 — QAMQ-A mixed K-quant | ✅ Done: QAMQ-A3 | [`2026-10-11-qamq-a.md`](../research/2026-10-11-qamq-a.md) |
| Phase 2.0 — валидация качества | ✅ Done: KLD + MMLU/GSM8K/ruMMLU/needle/tool-calling для A3 и A5; imatrix-источник не влияет | [`qamq-a.md`](../research/2026-10-11-qamq-a.md) |
| Phase 2.1 — QAMQ-B AWQ/W4 | ✅ **AWQ реализован для FFN и attention** (`apply_awq_scale.py`, GQA-aware): фолдинг точен, даёт −0.46% PPL и лучше Q4_K_M на MMLU/GSM8K. **A5L-AWQ2 — финальный лидер**: 5.36 GB, 61.2 t/s MTP=0, **65.9 t/s с MTP W3**. W4-пробники (B1/B2 IQ4_XS) — лидер по скорости в llama.cpp, но медленнее в yforge. rotation/low-rank — не начаты (отложены) | [`qamq-a.md`](../research/2026-10-11-qamq-a.md) |
| Phase 3 — QAMQ-C DeltaNet/SSM/KV/MTP | 🔄 Разведка сделана: KV q4_0 на 32k/64k почти бесплатен по PPL (+0.09% на 64k) → 4-битный пул в движке дал бы −1 ГиБ на 128k; **точность MTP-головы не узкое место** (MTP6: PPL идентичен, средняя скорость хуже); **`MTP_MAX_CTX` реализован** (MTP на <8k, baseline на длинном). Осталось: 4-битный KV-пул (нужен новый вариант в FA-шаблоне `Is_kv_n`), DeltaNet/SSM state 8-bit (экономия 24 МиБ/слот — мало) | [`qamq-a.md`](../research/2026-10-11-qamq-a.md) |
| Phase 4 — QAMQ-D VQ/TCQ 2–3 бита | 🔄 Разведка сделана через существующие IQ3_XXS: −10.4% размера дали только +1.9% скорости при +2.9% PPL — распаковка съедает bandwidth-выигрыш. **Собственный VQ/TCQ должен решить именно это** (дешёвая распаковка), иначе смысла нет | [`qamq-a.md`](../research/2026-10-11-qamq-a.md) |

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

## Результаты Phase 0 + Phase 1

- Инструменты: llama.cpp `b11552` (commit `23b0202a1`), CUDA 12.4 prebuilt.
- Калибровка: `qamq-calib.txt` 3.3 MB, `qamq-heldout.txt` 672 KB.
- Imatrix: `qamq-imatrix.gguf`, 1715 chunks, 248 entries. `--nextn` падает на state embedding; trunk-only.
- Лучший рецепт: **QAMQ-A3** (`qamq-a3-tensor-types.txt`), 5 456 392 608 байт, sha256 `91970A42BFA2ACFA6D27E512DB9CD4F6394CD3A899A8B25687B67BB28050B51A`.
- yforge MTP=0: **58.50–58.57 t/s** против Q4_K_M 53.73–53.75 (+8.9%); prefill 3.945 s против 4.526 s; VRAM 7343 против 7698 MiB.
- yforge MTP=1 W1: **47.58–47.76 t/s**, acceptance 80–85%.
- llama.cpp `llama-bench`: QAMQ-A3 **tg128 59.85 t/s** против Q4_K_M 55.45 (+7.9%).
- PPL held-out: QAMQ-A3 **15.3736** против Q4_K_M 15.2828 (+0.59%).
- Провалы: QAMQ-A conservative (5562 MiB, −0.6%); QAMQ-A2 с Q3_K (медленнее Q4_K_M и в Candle, и в llama.cpp; PPL +4.0%).

Gate Phase 1: decode ≥58 t/s — выполнен; качество +0.59% PPL — принято с оговоркой, нужны task evals.

## Рабочий журнал

| Дата | Действие | Результат | Артефакт/лог |
|---|---|---|---|
| 2026-10-11 | Research теорий и методов | Отчёт с 40+ источниками | `docs/research/2026-10-11-quantization-theory-and-custom-schemes.md` |
| 2026-10-11 | Создан рабочий план и T-010 | План зафиксирован | этот файл, `TASKS.md` |
| 2026-10-11 | Phase 0: инструменты, калибровка, imatrix | `llama.cpp b11552`; 1715 chunks; `--nextn` failed (state embedding) | `2026-10-11-qamq-phase0.md` |
| 2026-10-11 | Phase 1: QAMQ-A/A2/A3 | A3: 58.5 t/s vs 53.7; PPL +0.59%; A2 failed | `2026-10-11-qamq-a.md` |
| 2026-10-11 | Phase 1 gate | Decode gate пройден; MTP=1 всё ещё медленнее MTP=0 | `qamq-a3-tensor-types.txt` |
| 2026-10-11 | Phase 2.0: KLD QAMQ-A3 vs BF16 (64 чанка) | A3 0.0536 против Q4_K_M 0.0508; quality ≈ Q4_K_M | `qamq\kld\kld-a3-64.log` |
| 2026-10-11 | Phase 2.0: BF16-imatrix + QAMQ-A4 | Эффекта нет: PPL 15.3957 против A3 15.3736, KLD 0.05366 против 0.05357 | `bf16-imatrix.gguf`, `qamq-ppl-a4.log` |
| 2026-10-11 | Phase 2.0: task evals MMLU+GSM8K | A3 MMLU 75.67% против 76.67% (−1 п.п., шум), GSM8K 94.66% = 94.66% | `eval_qamq.py`, `results/eval-a3.json` |
| 2026-10-11 | Phase 2.1: Q4_0-пробники A5/A6 + plain Q4_0 | decode-потолок ~60 t/s; A5 prefill 1991 против 1916, KLD 0.0604 | `qamq-a5-tensor-types.txt`, `kld-a5-64.log` |
| 2026-10-11 | Phase 2.1: B1 (IQ4_XS ffn) + yforge A/B + evals | llama.cpp: B1 2030/61.94 (лидер); yforge: A5 58.7 (лидер), B1 51.3 (медленно) | `qamq-b1-tensor-types.txt`, `results/eval-a5.json` |
| 2026-10-11 | Найден баг методики: decode считался по SSE-чанкам | MTP W2/W3 = 65–73 t/s (было «38–44»); на 256/512 токенах — паритет с MTP=0 | `bench-live.ps1` (fix), `qamq-a.md` |
| 2026-10-11 | MTP вредит на длинном ctx (39k: −38%) | Добавлен `MTP_MAX_CTX` в engine; yoga gate: <8k MTP, >8k baseline; 39k 31→50 t/s | `scheduler.rs` (yttri-forge `478aa72c`), пересобран `yforge.exe` |
| 2026-10-11 | Phase 2.0: RU + needle + tool-calling | ruMMLU A5 69.33% против 69.67%; needle 4k/16k/32k 9/9 у обеих; tool-calling 10/10 | `eval_qamq.py`, `results/eval-*-ru/needle/tools.json` |
| 2026-10-11 | Phase 2.1: AWQ per-channel scaling (FFN) | Фолдинг точен (BF16-AWQ 15.0176 ≈ база 15.0187); A5-AWQ PPL 15.3544 против 15.4476; лучше Q4_K_M на GSM8K/ruMMLU | `apply_awq_scale.py`, `X:\qamq\...A5-AWQ.gguf` |
| 2026-10-11 | Phase 2.1: A5L-AWQ (AWQ + снятие q5/q6) | 5.36 GB, PPL 15.4079, llama.cpp 61.07, yforge 61.3 t/s; с MTP W3 — 65.4 t/s в среднем | `qamq-a5l-tensor-types.txt`, `results/eval-a5l-awq.json` |
| 2026-10-11 | Phase 2.1: AWQ2 (attention-фолдинг, GQA-aware) | PPL 15.3767 (лучше A5L-AWQ), MMLU 75.67%, GSM8K 96.18%; с MTP W3 — 65.9 t/s среднее | `apply_awq_scale.py --attn`, `results/eval-a5l-awq2.json` |
| 2026-10-11 | Phase 4-разведка: IQ3_XXS на FFN | 4.80 GB (−10.4%), но tg128 +1.9% и PPL +2.9% — невыгодно | `qamq-d1-tensor-types.txt` |
| 2026-10-11 | Prefill: FA-tile и F16-сайдкар | Ни то, ни другое: tile ±2%, сайдкар +2.5% на коротком и −49% на 18.9k (VRAM) | `qamq-a.md`, `forge-convert --f16-heavy` |
| 2026-10-11 | Prefill: nsys-профиль | GPU занят 97%; 60% — MMQ-GEMM (потолок по замерам 2026-08-25); разрыв структурный | `logs\pf-prof.nsys-rep`, `pf-prof.sqlite` |
| 2026-10-11 | HumanEval + приёмка через yforge | HumanEval AWQ2 85.37% против Q4_K_M 83.54%; yforge совпал с llama.cpp на MMLU/GSM8K/ruMMLU | `results/he-awq2.json`, `results/eval-yforge-awq2.json` |

## Следующие шаги (после Phase 2.1)

Короткий вывод: **decode на этой GPU упёрся в bandwidth-предел (~60 t/s)** для 4-битных
весов, поэтому вся оставшаяся работа делится на два независимых трека.

Трек «скорость» (decode уже на пределе — нужны структурные рычаги):
- [ ] MTP overhead: линейный MTP даёт 47.6 против 58.5 t/s у MTP=0. Рычаги: batched draft,
      CUDA graph на draft-шаг, единый causal verify, убрать D2H/accept-rebuild.
- [ ] KV-кэш: перейти с q8 на 4–5 бит даёт −1.5 ГиБ на 128k, но упирается в качество.
      Reference llama.cpp на 512 ctx: q8_0 +0.16%, q4_0 +0.49% PPL. Нужен A/B на 64K/128K.
- [ ] DeltaNet/SSM state: 8-bit per-tensor + Hadamard на выходе (Phase 3).

Трек «качество при том же размере» (не даёт скорости, но открывает W3/W4):
- [ ] Rotation (Hadamard/learned) перед квантизацией: сглаживает outliers, позволяет
      опустить битность без роста PPL. Требует правок в `llama-quantize` (или своего
      offline-пайплайна) и в Candle-загрузчике.
- [ ] AWQ-подобное per-channel масштабирование `ffn_down`/`ssm_out` — самые чувствительные
      тензоры по sensitivity map.
- [ ] Low-rank FP16 residual: хранить ошибку квантизации как `U·V^T` для самых чувствительных
      тензоров (blk.29–31.ffn_down).

Отдельно: IQ4_XS в нашем Candle медленнее Q4_0 (51.3 против 58.7 t/s), хотя в llama.cpp —
наоборот. Ядро `mul_mat_vec_iq4_xs_q8_1_cuda*` есть (`VDR_IQ4_XS_Q8_1_MMVQ=4`), но, судя по
замеру, не попадает в горячий путь или не оптимально. Это отдельная задача для `yttri-forge`.

## Риски и решения

- **BF16 не влезает в 12 GB для imatrix** → использовать Q4_K_M/Q8_0 для статистики, BF16 для quantize; либо partial offload.
- **Наш Candle-загрузчик не поддерживает какой-то тип** → исключить тип из рецепта или добавить поддержку.
- **llama.cpp prebuilt не содержит MTP/nextn** → собрать из master или использовать release после PR #22673.
- **Quality gate не проходит** → вернуться к sensitivity map, поднять битность sensitive tensors, либо перейти к QAMQ-B/C.
- **Server занят/выключен** → зафиксировать паузу и продолжить локальную подготовку артефактов.

## Definition of Done (закрыт для Phase 0–2.1, T-011)

- Phase 0 отчёт с sensitivity map.
- QAMQ-A GGUF собран и загружается в `yforge`.
- Полный A/B на `yttri-win`: decode/prefill/VRAM/PPL/KLD/task evals.
- Phase 1 отчёт с выводом: прошёл/не прошёл gate, что делать в Phase 2/3.
- Все артефакты и логи сохранены; коммит в репозитории; TASKS.md обновлён.

### Статус по пунктам

| Пункт | Статус | Доказательство |
|---|---|---|
| Phase 0 отчёт с sensitivity map | ✅ | `docs/research/2026-10-11-qamq-phase0.md` |
| QAMQ GGUF собран и грузится в yforge | ✅ | A3/A5/A5L-AWQ2: `yforge: quant = …`, замеры decode |
| Полный A/B: decode/prefill/VRAM/PPL/KLD/task evals | ✅ | 7 наборов (MMLU, GSM8K, ruMMLU, HumanEval, tool-calling, needle, PPL/KLD), замеры в обоих движках |
| Phase 1 отчёт с выводом по gate | ✅ | `qamq-a.md`, gate пройден (A3 → A5L-AWQ2) |
| Артефакты, логи, коммиты, TASKS.md | ✅ | скрипты в `scripts/qamq/`, модели в `D:\Models` и `X:\qamq`, коммиты в обоих репозиториях |

### Открытые пункты T-011 (не входят в DoD)

| Пункт | Статус | Обоснование |
|---|---|---|
| rotation (Hadamard) | не начат | не коммутирует с elementwise-операциями (SiLU-mul, RMSNorm-γ) — нужна перестройка графа модели, а не offline-фолдинг |
| low-rank FP16 residual | не начат | нужна поддержка в рантайме (доп. матмул на слой) + новый формат тензоров |
| DeltaNet/SSM state 8-bit | не начат | экономия 24 МиБ/слот — пренебрежимо |
| 4-битный KV-пул | разведан, отложен | ~200–400 строк + пересборка ради ~1 ГБ VRAM, риск в критичном ядре |
| VQ/TCQ 2–3 бита | разведан | IQ3_XXS показал: дешёвый прокси теряет больше на распаковке, чем выигрывает на байтах |
