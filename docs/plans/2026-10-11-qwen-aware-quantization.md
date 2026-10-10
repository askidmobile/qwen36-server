# Qwen-aware mixed quantization (QAMQ)

Дата: 2026-10-11.
Контекст: [research](../research/2026-10-11-quantization-theory-and-custom-schemes.md).
Цель: снизить bytes/token и/или увеличить tokens/pass для Qwen3.5-9B на RTX 3060 12 GB, сохранив качество на RU/EN/code/agent/long-context задачах.

Статус: research завершён, реализация не начата.

## Фазы

### Phase 0 — measurement and sensitivity map
- [ ] Калибровочный набор: RU/EN code, chat, tool-calling, long context.
- [ ] `llama-imatrix --nextn` + анализ per-tensor statistics.
- [ ] Per-tensor KL/PPL/reconstruction sensitivity.
- [ ] SSM state/outlier statistics, DeltaNet error propagation на 32K/128K/256K.
- [ ] Baseline: BF16 reference, Q4_K_M, decode/prefill/VRAM/PPL/KLD/task evals.

### Phase 1 — QAMQ-A: mixed K-quant на существующих ядрах
- [ ] Политика по тензорам: F32/Q8/Q6/Q5/Q4/Q3 (см. research, таблица 5.2).
- [ ] `llama-quantize --imatrix --tensor-type` рецепт.
- [ ] Проверка загрузки в `yttri-forge`/Candle.
- [ ] Метрики: size, decode/prefill t/s, VRAM, PPL/KLD, MMLU/GSM8K/HumanEval/MBPP, RU, tool-calling, needle/RULER, MTP acceptance.
- Gate: качество не хуже Q4_K_M на выбранном наборе; decode ≥ 58 t/s.

### Phase 2 — QAMQ-B: rotation + AWQ + low-rank residual + W4
- [ ] Offline Hadamard/learned rotation + AWQ-масштаб, вшитые в веса.
- [ ] 4-битный residual с per-group scale; low-rank FP16 для sensitive tensors.
- [ ] Fused kernels: rotation+dequant+GEMV, low-rank+W4 GEMV.
- [ ] KV/активации: online rotation только там, где он окупается.
- Gate: качество в пределах 1–2% от BF16; decode ≥ 65 t/s; VRAM не выше baseline.

### Phase 3 — QAMQ-C: DeltaNet/SSM, KV и MTP head
- [ ] SSM state: 8-bit per-tensor + Hadamard output + per-state-group B/C.
- [ ] KV: per-channel K / per-token V, pre-RoPE K, 4–5 бит.
- [ ] MTP head: Q6/Q8 + shortlist/two-stage/low-rank.
- Gate: 64K/128K/256K без деградации >2%; MTP acceptance не ниже текущего.

### Phase 4 — QAMQ-D: 2–3 бита VQ/TCQ
- [ ] Выбрать базу: VPTQ vs QTIP/EXL3 vs AQLM/QuIP#.
- [ ] Codebook/trellis decoder kernel.
- [ ] PV-Tuning/QAT при необходимости.
- Gate: 2–3 бита с приемлемым качеством на RU/EN/code/agent; decode 75–90 t/s.

## Обязательные метрики

- size/effective bpw, decode t/s, prefill t/s, TTFT, VRAM;
- PPL (WikiText-2/C4), KLD к BF16, MMLU/GSM8K/HumanEval/MBPP;
- русский язык, tool-calling/agent, needle/RULER;
- MTP acceptance и скорость;
- корректность DeltaNet state на длинных последовательностях.

## Риски

- Rotation/low-rank могут не окупиться без fused kernels.
- 2–3 бита могут дать плохое качество на коде/русском/агентных задачах.
- Qwen3.5 уже over-trained и чувствителен к квантизации; scaling laws говорят, что PTQ-деградация растёт с D/N.
- Без SSM/KV-учёта long-context quality может падать раньше, чем short-context PPL.
