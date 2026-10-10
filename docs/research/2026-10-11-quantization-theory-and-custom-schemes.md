# Собственная квантизация для Qwen3.5-9B: теория, дефекты GGUF-форматов и варианты дизайна
Дата: 2026-10-11.
Контекст: Qwen3.5-9B, RTX 3060 12 GB, `yttri-forge`/Candle, собственные CUDA-ядра, задача — снизить байты на токен и/или повысить число токенов за проход весов.
Тип документа: deep research + инженерное предложение. Ниже явно разделены **sourced fact**, **inference** и **recommendation**.

## Executive summary

1. **llama.cpp Q1–Q8 — это не «единый оптимальный формат», а семейство универсальных блочных скалярных квантизаторов.** K-quants используют affine-представление `w = q*S + M` с суперблоком 256 весов и двухуровневыми метаданными; IQ-quants добавляют importance matrix и низкоразмерные кодовые книги (включая E8-lattice). Это сильные инженерные форматы, но они не учитывают архитектуру Qwen3.5 гибридно (Gated DeltaNet + full attention + MTP) и не используют rotation/low-rank/state-aware квантизацию ([llama.cpp Tensor Encoding Schemes](https://github.com/ggml-org/llama.cpp/wiki/Tensor-Encoding-Schemes), [Bonn thesis](https://bonndoc.ulb.uni-bonn.de/xmlui/bitstream/handle/20.500.11811/13751/2025_LLM_Hardware.pdf)).
2. **Главный предел сейчас — память, а не вычисления.** У RTX 3060 12 GB пропускная способность ~360 GB/s ([Gigabyte RTX 3060 12G](https://www.gigabyte.com/us/Graphics-Card/GV-N3060WF2-12GD-rev-20/sp)). При весе Q4_K_M 5.87 GB теоретический потолок batch=1 decode ≈ 61.3 ток/с. Наш baseline 53.7 ток/с — это ~88% потолка. Значит, +5–15% можно получить ядрами, а кратный рост — только через **меньше байт на токен** или **несколько токенов за один проход весов**.
3. **Основные недостающие математические компоненты для «своей» квантизации:**
   - rotation/incoherence (Hadamard/learned/affine) — выравнивает распределение и убирает outliers (QuaRot, SpinQuant, FlatQuant, DuQuant);
   - low-rank/outlier-компенсация (SVDQuant, SpQR/OWQ-подходы) — выносит «тяжёлые» компоненты в FP16-ветку;
   - mixed-precision allocation по чувствительности слоёв/групп (MixQuant, MXSens, SliM-LLM, Q-Strata);
   - SSM/DeltaNet-aware квантизация состояния и input-dependent параметров (Quamba/Quamba2; отдельно — накопление ошибки рекуррентного состояния);
   - vector/trellis-квантизация для 2–3 бит (AQLM, QuIP#, QTIP, VPTQ, EXL3).
4. **Практический вывод:** начинать не с экзотики, а с трёх шагов:
   - **QAMQ-A: mixed K-quant** — per-tensor/ per-group типы из уже существующих GGUF-форматов, без новых ядер. Ожидание: 10–20% decode при сохранении качества.
   - **QAMQ-B: rotation + AWQ-масштаб + low-rank residual + W4** — fused-ядра, качество ближе к BF16,目标是 15–30% decode и лучше long-context.
   - **QAMQ-C: DeltaNet/SSM и KV-state квантизация** — уникальная для Qwen3.5 часть; без неё любой «общий» формат будет терять качество на длинном контексте.
   - **QAMQ-D: 2–3 бит VQ/TCQ** — долгий research-путь, потенциально 80–95 ток/с, но требует кодовых книг/trellis-ядер и серьёзного quality gate.

## 0. Границы доказательств

- **Sourced fact:** результаты и утверждения из перечисленных статей, документации llama.cpp, Unsloth, NVIDIA и измерений на нашем стенде.
- **Inference:** перенос методов на Qwen3.5-9B и нашу Candle/CUDA-архитектуру.
- **Recommendation:** предлагаемые схемы QAMQ, порядок внедрения и критерии приёмки.

Мы не нашли публичных работ, которые одновременно квантизуют **Gated DeltaNet + full attention + MTP head** Qwen3.5-9B и публикуют полный набор decode/prefill/KLD/агентных метрик. Ближайшие релевантные работы — Quamba/Quamba2 (SSM), KVQuant/KIVI/TurboQuant (KV), QuaRot/SpinQuant/FlatQuant/DuQuant (rotation), AQLM/QuIP#/QTIP/VPTQ (extreme low-bit). Это означает: готовая «идеальная» схема для нашей модели в литературе отсутствует; её нужно строить и валидировать самостоятельно.

## 1. Математические основы

### 1.1 Блочная скалярная квантизация

Классическая affine-схема:
```
q_i = round((x_i - z) / s),   ŷ_i = s q_i + z
```
где `s` — scale, `z` — zero-point/min, `q_i` — целое. Ошибка раскладывается на:
```
e = x - ŷ = e_round + e_clip + e_scale_quant
```
- `e_round` — ошибка округления (при высоком rate её дисперсия ≈ Δ²/12);
- `e_clip` — потери на насыщении (outliers);
- `e_scale_quant` — ошибка от квантизации самих scale/min (в K-quants метаданные тоже квантуются: 6 бит для Q4_K, 8 бит для Q6_K).

Блочная схема делит тензор на блоки/субблоки и хранит свои `s_j, z_j`. Компромисс:
- маленький блок → меньше ошибка формы, но больше метаданных;
- большой блок → меньше overhead, но хуже адаптация к локальной гистограмме.

Для высокого rate оптимальное размещение бит между компонентами с дисперсиями `σ_i²`:
```
b_i ≈ 0.5 * log2(σ_i² / D)   (water-filling)
```
где `D` — целевая дисторсия. Это прямо обосновывает mixed precision: одинаковый бит на все тензоры не оптимален, если у них разная чувствительность/дисперсия.

### 1.2 Векторная квантизация и rate–distortion

Скалярная квантизация не использует корреляции между весами. Векторная квантизация (VQ) кодирует блок из `d` весов одним индексом кодовой книги:
```
W ≈ C[i],   i = argmin_j ||W_block - C[j]||²
```
При том же rate VQ даёт **space-filling gain** и лучше использует форму распределения. Плата — размер кодовой книги растёт экспоненциально от размерности: `|C| = 2^{R d}`. Именно поэтому AQLM использует несколько маленьких кодовых книг (additive quantization), QuIP# — E8-lattice с высокой симметрией, а QTIP — trellis-coded quantization, где состояние декодера отделяет размер кодовой книги от эффективной размерности ([AQLM](https://arxiv.org/abs/2401.06118), [QuIP#](https://arxiv.org/abs/2402.04396), [QTIP](https://arxiv.org/abs/2406.11235)).

### 1.3 Чувствительность и mixed-precision allocation

Пусть `D_i(b_i)` — дисторсия/потеря качества i-го тензора при `b_i` битах, `n_i` — число параметров, `B` — бюджет. Задача:
```
min Σ D_i(b_i)   s.t.   Σ n_i b_i ≤ B
```
Лагранжиан даёт условие `∂D_i/∂b_i = λ n_i`. На практике `D_i` заменяют на:
- Hessian/Fisher/Hutchinson-оценку кривизны;
- KL-дивергенцию выхода слоя/модели к FP16;
- реконструкционную ошибку `||WX - W_qX||²`;
- эмпирическую деградацию на калибровочном наборе.

Важный результат 2025–2026: **чувствительность слоя зависит от битности вышестоящих слоёв**; поэтому статическая таблица «этот слой всегда Q4» неоптимальна ([MixQuant](https://arxiv.org/abs/2607.23047), [MXSens](https://arxiv.org/abs/2607.17733)). MXSens отдельно выделяет 8-битные **column-wise outliers** и распределяет 6-битные/4-битные мантиссы; SliM-LLM делает salience-driven allocation на уровне групп; Q-Strata решает allocation для MoE на уровне экспертов ([SliM-LLM](https://proceedings.mlr.press/v267/huang25aa.html), [Q-Strata](https://arxiv.org/abs/2608.30564)).

### 1.4 Outliers, incoherence и rotation

LLM-активации и веса имеют heavy-tailed распределение и «massive outliers». Uniform-сетка тратит динамический диапазон на редкие большие значения. Идея rotation:
```
Y = X W = (X R) (Rᵀ W)
```
где `R` — ортогональная матрица. Полное преобразование не меняет выход в FP32, но перемешивает координаты, снижая kurtosis и максимальные значения. Варианты:
- **QuaRot** — случайные Hadamard-ротации, W4A4/KV4 ([QuaRot](https://arxiv.org/abs/2404.00456));
- **SpinQuant** — обучаемые rotation-матрицы (Cayley SGD) на Stiefel manifold; random rotation может отличаться на 13 пунктов zero-shot, а обучение занимает от 30 минут до 3.5 часов ([SpinQuant](https://arxiv.org/abs/2405.16406));
- **FlatQuant** — обучаемое affine-преобразование на слой, Kronecker-декомпозиция, fused-ядро; W4A4 с <1% падения на LLaMA-3-70B ([FlatQuant](https://arxiv.org/abs/2410.09426));
- **DuQuant** — блочные rotation + zigzag permutation для massive outliers ([DuQuant](https://arxiv.org/abs/2406.01721)).

Практический вывод: rotation — самый сильный известный способ сделать 4-битные **активации и KV** жизнеспособными. Для weight-only W4 rotation можно «вшить» в веса офлайн; для активаций/KV нужен fused online-Hadamard, иначе overhead съест выигрыш.

### 1.5 Low-rank и outlier-компенсация

SVDQuant предлагает:
```
W = L_r + R,   L_r = U_r S_r V_rᵀ
```
где `L_r` — low-rank FP16-ветка, `R` — остаток, который квантизуется в 4 бита. Outliers «вытягиваются» в low-rank компонент. Отдельная ветка стоит дорого по памяти, поэтому авторы fused-ят её в low-bit kernel (Nunchaku): 12B FLUX на 16 GB laptop 4090 — 3.5× меньше памяти и 3× быстрее W4A16 baseline ([SVDQuant](https://arxiv.org/abs/2411.05007)).

AWQ использует эквивалентное масштабирование:
```
W' = W diag(s),   X' = diag(s)⁻¹ X
```
и защищает ~1% salient каналов, выбранных по активациям, без backprop и смешанной точности ([AWQ](https://arxiv.org/abs/2306.00978)). SmoothQuant переносит сложность активаций в веса:
```
Y = (X diag(s)⁻¹)(diag(s) W)
```
и включает W8A8 ([SmoothQuant](https://arxiv.org/abs/2211.10438)). GPTQ решает layer-wise задачу
```
min_{Ŵ} ||W X - Ŵ X||²
```
с приближённым Hessian `H = 2XXᵀ`, последовательно квантизуя столбцы и компенсируя ошибку ([GPTQ](https://arxiv.org/abs/2210.17323)). PV-Tuning показывает, что для 1–2 бит одной PTQ мало: нужна осторожная quantization-aware fine-tuning, и она даёт Pareto-optimal 2 бита на Llama-2 ([PV-Tuning](https://arxiv.org/abs/2405.14852)).

### 1.6 Рекуррентные SSM/DeltaNet: ошибка накапливается

Для SSM/linear attention состояние обновляется рекуррентно:
```
S_t = A_t S_{t-1} + B_t x_t
y_t = C_t S_t
```
Ошибка квантизации состояния и параметров `A/B/C` не исчезает: `δS_t ≈ A δS_{t-1} + ...`, то есть **ошибка накапливается вдоль последовательности**. NVIDIA Nemotron technical report прямо отмечает: «Because Mamba decoding is recurrent, quantization error from previous steps propagates into future steps and accumulates over time» ([NVIDIA Nemotron 3 Super](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf)). Quamba/Quamba2 добавляют:
- sensitive feature maps внутри selective scan;
- massive outliers в output-активациях SSM;
- static 8-bit per-tensor для SSM + Hadamard на output;
- per-state-group quantization для input-dependent `B/C`;
- сортировку/кластеризацию входов и перестановку весов офлайн;
- W8A8/W4A8/W4A16 варианты ([Quamba](https://arxiv.org/abs/2410.13229), [Quamba2](https://arxiv.org/abs/2503.22879)).

Для Qwen3.5 это критично: 24 из 32 слоёв — Gated DeltaNet. Любая схема, которая квантизует эти слои как обычный FFN, будет терять качество на длинном контексте.

## 2. Что именно делает llama.cpp и где это неоптимально

### 2.1 Формулы K-quants

K-quants используют суперблок 256 весов и двухуровневые метаданные ([wiki](https://github.com/ggml-org/llama.cpp/wiki/Tensor-Encoding-Schemes), [Bonn thesis](https://bonndoc.ulb.uni-bonn.de/xmlui/bitstream/handle/20.500.11811/13751/2025_LLM_Hardware.pdf)):

| Формат | Вес | Субблоки | Scale/min | bpw |
|---|---:|---:|---:|---:|
| Q2_K | 2 бита | 16×16 | 4/4 бита | 2.5625 |
| Q3_K | 3 бита | 16×16 | 6/— | 3.4375 |
| Q4_K | 4 бита | 8×32 | 6/6 бит | 4.50 |
| Q5_K | 5 бит | 8×32 | 6/6 бит | 5.50 |
| Q6_K | 6 бит | 16×16 | 8/— | 6.5625 |

Q4_K_M: `bits = 256*4 + 8*(6+6) + 2*16 = 1152` на 256 весов = 4.50 bpw. Q6_K: `256*6 + 16*8 + 16 = 1680` = 6.5625 bpw. Для Q4_K де-квантизация:
```
ŵ_i = q_i * S_j + M_j
```
где `S_j, M_j` восстанавливаются из суперблоковых `Δ_sb, m_sb` и 6-битных `q_Δ,j, q_m,j`. Q6_K — симметричный вариант: `M_j = 0`.

**Достоинства:** дешёвые метаданные, хорошее качество на 4–6 битах, широкая поддержка ядер (MMQ/MMVQ), imatrix.

**Дефекты для нашей задачи:**
- **Фиксированный суперблок 256.** Не адаптируется к локальной геометрии весов и к архитектуре слоя.
- **Аффинная монотонная сетка.** Не оптимальна для heavy-tailed/outlier-распределений; Q2/Q3 страдают особенно.
- **Нет rotation/incoherence.** Hadamard/learned rotations не используются, поэтому outliers приходится «оплачивать» битами.
- **imatrix — диагональное приближение.** Он использует `<a_i²>` (диагональ `E[aaᵀ]`), а не полный Hessian/GPTQ-объект; это дешёво, но не оптимально ([imatrix README](https://github.com/ggml-org/llama.cpp/blob/master/tools/imatrix/README.md), [discussion #5006](https://github.com/ggml-org/llama.cpp/discussions/5006)).
- **Шаблонность K-quant-рецептов.** Q4_K_M/Q5_K_M и т.п. — это ручные правила по ролям тензоров и индексам слоёв, а не результат per-model sensitivity optimization.
- **IQ-quants** лучше на 1.5–3.5 bpw за счёт importance signal и кодовых книг (E8-lattice и др.), но используют низкоразмерные книги, backend-dependent lookup и всё ещё не rotation/low-rank/SSM-aware ([wiki](https://github.com/ggml-org/llama.cpp/wiki/Tensor-Encoding-Schemes), [Bonn thesis](https://bonndoc.ulb.uni-bonn.de/xmlui/bitstream/handle/20.500.11811/13751/2025_LLM_Hardware.pdf)).
- **Архитектурная слепота:** для Qwen3.5 нет отдельной политики для `ssm_a/alpha/beta/conv1d/dt/norm/out`, `attn_gate`, `attn_qkv`, full-attention слоёв, MTP `nextn.*` и разделяемой головы. Универсальный рецепт не может знать, что `ssm_out` в hybrid-модели «взрывает KLD», если его квантизовать как обычный FFN.

### 2.2 Что уже делает Unsloth Dynamic 2.0/3.0

Unsloth Dynamic 2.0 — это не новый block-format, а **per-layer selection** поверх существующих GGUF-типов: important layers upcast to 8/16-bit, imatrix на калибровочном наборе (chat/code/tool-calling/long context), улучшенный алгоритм размещения ([Unsloth Dynamic 2.0](https://docs.unsloth.ai/basics/unsloth-dynamic-2.0-ggufs), [Qwen3.5 guide](https://unsloth.ai/docs/models)). Для Qwen3.5 их материалы отдельно отмечают, что некоторые тензоры очень чувствительны к квантизации, а QLoRA (4-битное обучение) не рекомендуется из-за «higher than normal quantization differences».

Наш собственный dump `unsloth/Qwen3.5-9B-MTP-GGUF/Qwen3.5-9B-Q4_K_M.gguf` подтверждает, что даже «Q4_K_M» — это mixed-квант:
- `ssm_a`, `ssm_alpha`, `ssm_beta`, `ssm_conv1d`, `ssm_dt.bias`, `ssm_norm` — F32;
- `ssm_out.weight` — Q5_K;
- в attention-слоях `attn_v`/`ffn_down` местами Q6_K;
- `ffn_gate`/`ffn_up` — в основном Q4_K;
- `blk.32.nextn.eh_proj.weight` — Q8_0;
- нормы — F32.

**Sourced fact + inference:** индустрия уже пришла к mixed precision для Qwen3.5, но делает это вручную/полуавтоматически фиксированным шаблоном. Следующий шаг — автоматическая per-tensor/per-group оптимизация под нашу конкретную модель, калибровку и задачи.

### 2.3 Scaling laws: чем больше обучена модель, тем она чувствительнее

Kumar et al. показали, что деградация от post-training quantization растёт с отношением данных к параметрам `D/N`; сильно over-trained модели могут стать чувствительнее к квантизации, и дополнительные данные иногда делают PTQ-качество хуже ([Scaling Laws for Precision](https://arxiv.org/abs/2411.04330)). Qwen3.5-9B — большая, вероятно over-trained модель, поэтому универсальные рецепты, откалиброванные на меньших/других моделях, особенно рискованны.

## 3. Обзор современных методов: что брать и что это даёт

| Метод | Биты | Что оптимизирует | Цена/риск | Применимость к нам |
|---|---:|---|---|---|
| GPTQ | 3–4 | layer-wise Hessian-компенсация | калибровка, kernel | базовый W4-метод, совместим с rotation |
| AWQ | 4 | activation-aware scaling, 1% salient | калибровка, kernel | простой и сильный для instruction/code моделей |
| SmoothQuant | W8A8 | перенос outliers activation→weight | нужен W8A8 kernel | больше для prefill/тензорных ядер |
| QuaRot | W4A4/KV4 | случайные Hadamard | fused rotation | лучший «бесплатный» rotation baseline |
| SpinQuant | W4A4/KV4 | обучаемые rotation | 30 мин–3.5 ч калибровки | лучше QuaRot, но нужна оптимизация |
| FlatQuant | W4A4 | affine + Kronecker, fused | обучение преобразований | SOTA quality/overhead, сложнее внедрить |
| DuQuant | W4A4 | rotation + permutation для massive outliers | сложные rotation-схемы | полезно для Qwen/LLaMA-подобных |
| SVDQuant | W4A4 | low-rank residual + 4-bit | fused low-rank kernel | идея для sensitive tensors |
| AQLM | 2–3 | additive vector quantization | codebook training, kernel | 2–3 бита, Pareto-optimal |
| QuIP# | 2–4 | Hadamard incoherence + E8 lattice | kernel, fine-tune | очень сильное 2-битное качество |
| QTIP | 2–4 | trellis-coded quantization + incoherence | custom trellis kernel | лучше QuIP# по quality/speed; EXL3 использует этот класс |
| VPTQ | <2–4 | second-order VQ, channel-independent | research-only, English-only tests | самый практичный VQ-кандидат |
| GPTVQ | 2–4 | быстрый VQ | kernel | промежуточный вариант |
| PV-Tuning | 1–2 | QAT/PV поверх VQ | нужны данные/обучение | если PTQ не проходит quality gate |
| MixQuant/MXSens/SliM-LLM/Q-Strata | mixed | per-layer/group bit allocation | калибровка | автоматизация выбора бит |
| KIVI | KV 2 | per-channel K, per-token V | kernel | обязательно для long-context |
| KVQuant | KV 3 | pre-RoPE K, nuqX, dense-and-sparse | calibration + kernel | near-lossless 3-bit KV |
| TurboQuant | KV 3.5 | PolarQuant + QJL, training-free | kernel | 3.5 бита quality-neutral, 2.5 бита marginal |
| Quamba/Quamba2 | SSM 8/4 | per-tensor SSM + per-state-group B/C | SSM-specific kernel | прямо релевантно DeltaNet |

**Ключевая мысль:** generic scalar K-quants дают хорошее качество на 4–6 битах, но на 3 битах и ниже выигрывают vector/trellis-схемы (AQLM/QuIP#/QTIP/VPTQ), а на 4 битах с активациями/KV — rotation-схемы (QuaRot/SpinQuant/FlatQuant/DuQuant). Наш собственный дизайн должен комбинировать обе идеи, а не выбирать одну.

## 4. Аппаратные и kernel-ограничения RTX 3060

- **RTX 3060 12 GB:** 360 GB/s, 192-bit, Ampere GA106 ([Gigabyte](https://www.gigabyte.com/us/Graphics-Card/GV-N3060WF2-12GD-rev-20/sp), [TechPowerUp](https://www.techpowerup.com/gpu-specs/geforce-rtx-3060-12-gb.c3682)).
- **Roofline:** `tokens/s ≤ BW / bytes_per_token`; для 5.87 GB Q4_K_M: `360/5.87 ≈ 61.3` ток/с. Baseline 53.7 = ~88% потолка.
- **Ampere tensor cores** поддерживают INT8/INT4 ([NVIDIA Ampere whitepaper](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf)). Но при batch=1 decode основная операция — не tensor-core GEMM, а memory-bound GEMV/dequant: нужно читать квантованные веса и умножать на активации. Поэтому **kernel де-квантования + GEMV** важнее, чем «сырые» TOPS.
- **llama.cpp MMQ/MMVQ:** MMQ ускоряет quantized matmul для Q2_K..Q6_K, IQ4_XS, FP4/NVFP4 и т.п.; при малом batch используется vector kernel MMVQ, при большом — MMQ/tensor-core crossover ([mmq.cuh](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmq.cuh), [DeepWiki CUDA backend](https://deepwiki.com/ggml-org/llama.cpp/5.1-cuda-backend-%28nvidia%29)).
- **Практические примеры fused kernels:** Marlin W4A16, AWQ/TinyChat W4A16, EXL3 fused Hadamard+dp4a GEMV, SVDQuant/Nunchaku low-rank+4-bit ([Marlin](https://github.com/IST-DASLab/marlin), [EXL3](https://github.com/turboderp-org/exllamav3)).
- **Следствие:** любой новый формат должен иметь fused dequant+GEMV kernel. Если декодировать в FP16 и потом умножать, bandwidth-выигрыш исчезает. Это главный инженерный фильтр для всех идей ниже.

## 5. Предлагаемая схема: QAMQ (Qwen-Aware Mixed Quantization)

### 5.1 Принципы

1. **Не один формат, а политика.** QAMQ — это не «Q4», а набор правил: каким тензорам сколько бит, какой алгоритм калибровки, какие transforms и какие kernels.
2. **Сначала измерение, потом формат.** Строим per-tensor sensitivity map (KL/PPL/reconstruction/Hessian/Fisher) на нашем сценарии: код, русский/английский чат, tool-calling, long context.
3. **Сохраняем outliers явно.** Top-p% каналов/весов — FP16/FP8, остальное — low-bit. Это дешевле, чем поднимать битность всего тензора.
4. **Rotation + low-rank — для активаций и sensitive tensors, а не для всего.** Офлайн rotation вшивается в веса; online Hadamard — только там, где он реально нужен (KV/активации).
5. **DeltaNet/SSM — отдельный класс.** State и input-dependent параметры квантизуются по правилам Quamba2, а не как FFN.
6. **Формат должен быть либо GGUF-совместимым, либо с собственным версионированным sidecar + CUDA-ядром.**

### 5.2 Политика по тензорам Qwen3.5-9B (стартовое предложение)

| Группа тензоров | Стартовый тип | Почему |
|---|---|---|
| Нормы (`*_norm`, `post_attention_norm`, `nextn.norm`) | F32/F16 | дёшево, критично для стабильности |
| `ssm_a`, `ssm_dt.bias`, `ssm_norm` | F32 | рекуррентная динамика и gate; ошибка накапливается |
| `ssm_alpha`, `ssm_beta`, `ssm_conv1d`, `ssm_out` | Q8/Q6 + outlier FP16 | Quamba2: B/C и выход SSM чувствительны; `ssm_out` уже upcast в Q5_K |
| Full-attention `attn_q/k/v/o`, `attn_gate`, `attn_qkv` | Q5_K/Q6_K | attention и q/k norm чувствительны; V особенно |
| DeltaNet `in_proj_qkv`/`out_proj`, FFN | Q4_K/Q5_K | основная масса параметров; можно агрессивнее |
| FFN `ffn_gate/up` | Q3_K/Q4_K | наиболее терпимые к квантизации |
| FFN `ffn_down` | Q4_K/Q5_K | down_proj традиционно более чувствителен |
| Embeddings / output / shared head | Q6_K/Q8_0 | вход/выход словаря, нельзя экономить слишком агрессивно |
| MTP `blk.32.nextn.*`, `eh_proj` | Q6_K/Q8_0; norms F32 | MTP head маленький, но критичен для acceptance |
| KV cache K | Q4/Q5, per-channel, pre-RoPE | KIVI/KVQuant: K имеет persistent channel outliers |
| KV cache V | Q4/Q5, per-token | V не имеет channel-outlier структуры |
| DeltaNet state | Q8/F16, per-state-group | ошибка рекуррентного состояния накапливается |

**Ожидаемый размер:** 4.6–5.2 GB (в зависимости от того, сколько sensitive тензоров оставить в Q5/Q6/F32) вместо 5.87 GB. **Теоретический decode:** 69–78 ток/с; **реалистичный:** 58–66 ток/с с учётом overhead.

### 5.3 Четыре варианта QAMQ

**QAMQ-A: mixed K-quant на существующих ядрах (ближайший шаг, 1–2 недели).**
- Никаких новых форматов: используем `llama-quantize --imatrix --tensor-type` и существующие Q2_K..Q6_K, IQ4_XS, Q8_0 ([quantize README](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md)).
- Калибровка: `llama-imatrix --nextn` на нашем датасете (код, RU/EN, tool-calling, long context).
- Per-tensor overrides по таблице 5.2.
- Плюс: быстро, безопасно, совместимо с llama.cpp/нашим загрузчиком.
- Минус: нет rotation/low-rank/VQ; потолок ~10–20%.

**QAMQ-B: rotation + AWQ + low-rank residual + W4 (средний срок, 4–8 недель).**
- Offline: Hadamard/learned rotation + AWQ-масштаб, вшитые в веса.
- Quantized residual: 4-битная блочная квантизация с per-group scale; для sensitive tensors — SVDQuant-подобная low-rank FP16-ветка.
- Online: fused rotation для KV/активаций; fused low-rank+W4 GEMV.
- Цель: W4A16/W4A8 с качеством ближе к BF16, 15–30% decode, лучше long-context.
- Риск: нужны CUDA-ядра; без них overhead может быть больше выигрыша.

**QAMQ-C: DeltaNet/SSM + KV-state (2–4 недели, уникально для Qwen3.5).**
- State: 8-бит per-tensor, Hadamard на output, per-state-group для B/C; сортировка/кластеризация входов; сохранить FP32 для `ssm_a/dt`.
- MTP head: Q6/Q8 + shortlist/two-stage projection; при необходимости low-rank для full-vocab 248320.
- KV: Q4/Q5 per-channel K / per-token V, pre-RoPE K; при 3–4 битах — non-uniform/TurboQuant-подход.
- Цель: удержать качество на 32K–256K и снизить memory/bandwidth KV/state.

**QAMQ-D: 2–3 бита VQ/TCQ (2–4 месяца, research).**
- AQLM/QuIP#/QTIP/VPTQ-класс: multi-codebook/trellis, Hadamard incoherence, second-order optimization.
- Цель: 3.5–4.0 GB, 80–95 ток/с теоретически.
- Требуется: codebook/trellis decoder kernel, calibration, PV-Tuning при необходимости, полный quality gate (RU/EN/code/long-context/агентные задачи).
- VPTQ — наиболее практичный кандидат по скорости/сложности; QTIP/EXL3 — лучший quality/latency класс; QuIP# — проверенный E8-lattice baseline.

### 5.4 Как это встраивается в наш движок

- **Сейчас:** загрузчик GGUF поддерживает F32/F16/BF16/Q8_0/Q2_K..Q6_K; CUDA MMQ/де-квантование уже есть для этих типов. Значит, QAMQ-A можно проверить без изменений ядра.
- **Далее:** добавить собственный тип (или sidecar) для:
  - block size 32/64 + FP8/FP16 scale/zero;
  - optional per-channel AWQ scale;
  - optional rotation metadata (Hadamard seed/learned matrix);
  - low-rank residual pair (rank, dtype, tensors);
  - group-wise bit-width map.
- **Ядра:** fused dequant+GEMV; fused Hadamard+dequant+GEMV; fused low-rank+W4; trellis/codebook decoder; SSM state quant/dequant; KV per-channel/per-token attention.
- **Формат:** версия + capability bits; backward-compatible reader; fallback to existing types if kernel unsupported.

## 6. План внедрения и quality gate

### Phase 0 — измерение (1–2 недели)
- Собрать **sensitivity map**: per-tensor KL/PPL/reconstruction, kurtosis/outlier stats, SSM state stats.
- Калибровочный набор: RU/EN code, chat, tool-calling, long context; `imatrix --nextn`.
- Baseline: BF16 reference, BF16 PPL/KLD, task evals (MMLU, GSM8K, HumanEval/MBPP, RU-задачи, tool-calling, needle/RULER).
- Выход: таблица «тензор → текущий тип → рекомендуемый тип → ожидаемая ошибка».

### Phase 1 — QAMQ-A (2–3 недели)
- Собрать mixed K-quant на существующих ядрах.
- Сравнить: size, decode/prefill t/s, VRAM, PPL/KLD, task evals, long-context, MTP acceptance.
- Gate: quality не хуже Q4_K_M на выбранном наборе; decode ≥ 58 ток/с.

### Phase 2 — QAMQ-B (4–8 недель)
- Реализовать offline rotation + AWQ-масштаб, 4-битный residual, low-rank для sensitive tensors.
- Fused kernels: rotation+dequant+GEMV, low-rank+W4.
- Gate: quality в пределах 1–2% от BF16 на задачах; decode ≥ 65 ток/с; VRAM не выше baseline.

### Phase 3 — QAMQ-C (параллельно)
- SSM state/KV/MTP-head квантизация.
- Gate: long-context (64K/128K/256K) без деградации >2%; MTP acceptance не ниже текущего.

### Phase 4 — QAMQ-D (research)
- AQLM/QuIP#/QTIP/VPTQ/EXL3-подход; PV-Tuning если нужно.
- Gate: 2–3 бита с приемлемым качеством на RU/EN/code/agent; decode 75–90 ток/с.

### Метрики, которые обязательно фиксировать
- decode t/s, prefill t/s, TTFT, VRAM, размер файла/effective bpw;
- PPL (WikiText-2/C4), KLD к BF16, MMLU/GSM8K/HumanEval/MBPP;
- русский язык, tool-calling/agent, long-context needle/RULER;
- MTP acceptance и скорость;
- корректность DeltaNet state на длинных последовательностях.

## 7. Прямые ответы

**«Это предел?»** Нет. Это предел текущего Q4_K_M на 3060 при batch=1: мы уже у ~88% bandwidth roofline. Предел можно поднять только уменьшением bytes/token (квантизация) или увеличением tokens/pass (спекуляция/MTP). Собственная квантизация — самый прямой рычаг.

**«llama.cpp Q1–Q8 не идеальна?»** Да, для Qwen3.5-9B. Это универсальные блочные scalar-форматы с фиксированным суперблоком, без rotation, low-rank, vector quantization и без архитектурной политики для Gated DeltaNet/MTP. Они хороши как baseline и как готовая инфраструктура, но не как оптимум для конкретной модели.

**«Что делать в первую очередь?»**
1. QAMQ-A: mixed K-quant по нашей таблице + imatrix + `--tensor-type` — быстро и без новых ядер.
2. Параллельно Phase 0: sensitivity map и quality gate.
3. QAMQ-B: rotation + AWQ + low-rank W4 — главный кандидат на 15–30% decode.
4. QAMQ-C: DeltaNet/SSM/KV/MTP-head — обязательная часть для long-context и качества.
5. QAMQ-D: 2–3 бита VQ/TCQ — долгий, но самый большой потенциальный выигрыш.

## Источники

- [llama.cpp Tensor Encoding Schemes](https://github.com/ggml-org/llama.cpp/wiki/Tensor-Encoding-Schemes)
- [llama.cpp imatrix README](https://github.com/ggml-org/llama.cpp/blob/master/tools/imatrix/README.md)
- [llama.cpp quantize README](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md)
- [llama.cpp MMQ kernel](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmq.cuh)
- [Small and Fast LLMs on Commodity Hardware: PTQ in llama.cpp (Bonn thesis)](https://bonndoc.ulb.uni-bonn.de/xmlui/bitstream/handle/20.500.11811/13751/2025_LLM_Hardware.pdf)
- [GPTQ](https://arxiv.org/abs/2210.17323)
- [AWQ](https://arxiv.org/abs/2306.00978)
- [SmoothQuant](https://arxiv.org/abs/2211.10438)
- [LLM.int8](https://arxiv.org/abs/2208.07339)
- [QuaRot](https://arxiv.org/abs/2404.00456)
- [SpinQuant](https://arxiv.org/abs/2405.16406)
- [FlatQuant](https://arxiv.org/abs/2410.09426)
- [DuQuant](https://arxiv.org/abs/2406.01721)
- [SVDQuant](https://arxiv.org/abs/2411.05007)
- [AQLM](https://arxiv.org/abs/2401.06118)
- [QuIP](https://arxiv.org/abs/2307.13304)
- [QuIP#](https://arxiv.org/abs/2402.04396)
- [QTIP](https://arxiv.org/abs/2406.11235)
- [VPTQ](https://arxiv.org/abs/2409.17066)
- [GPTVQ](https://arxiv.org/abs/2402.15319)
- [PV-Tuning](https://arxiv.org/abs/2405.14852)
- [Scaling Laws for Precision](https://arxiv.org/abs/2411.04330)
- [KVQuant](https://www.stat.berkeley.edu/~mmahoney/pubs/neurips-2024-kvquant.pdf)
- [KIVI](https://arxiv.org/abs/2402.02750)
- [TurboQuant](https://arxiv.org/abs/2504.19874)
- [Quamba](https://arxiv.org/abs/2410.13229)
- [Quamba2](https://arxiv.org/abs/2503.22879)
- [MixQuant](https://arxiv.org/abs/2607.23047)
- [MXSens](https://arxiv.org/abs/2607.17733)
- [Q-Strata](https://arxiv.org/abs/2608.30564)
- [SliM-LLM](https://proceedings.mlr.press/v267/huang25aa.html)
- [Marlin](https://github.com/IST-DASLab/marlin)
- [EXL3](https://github.com/turboderp-org/exllamav3)
- [Unsloth Dynamic 2.0](https://docs.unsloth.ai/basics/unsloth-dynamic-2.0-ggufs)
- [Unsloth Qwen3.5 guide](https://unsloth.ai/docs/models)
- [Qwen3.5 blog](https://qwen.ai/blog?id=qwen3.5)
- [Qwen3.5-9B on Hugging Face](https://huggingface.co/Qwen/Qwen3.5-9B)
- [NVIDIA Ampere GA102 whitepaper](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf)
- [Gigabyte RTX 3060 12G specs](https://www.gigabyte.com/us/Graphics-Card/GV-N3060WF2-12GD-rev-20/sp)
- [NVIDIA Nemotron 3 Super technical report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf)
- [TreeWY: Speculative Verification for Gated DeltaNet Hybrids](https://arxiv.org/abs/2608.20961)
