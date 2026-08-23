# FR-000: Bandwidth-потолок — аналитический расчёт

## Теоретический bandwidth-потолок IQ2 на RTX 3060

RTX 3060 memory bandwidth: 360 GB/s
IQ2_XXS: 2.0625 бита/вес
Active params (A3B): ~3B

### Per-chunk bandwidth (512 токенов):
Веса читаются ОДИН РАЗ на чанк (batched MoE, FA2, DeltaNet fused):
- Compressed weights: 3B × 2.0625/8 = 0.774 GB
- At 360 GB/s: 2.15 мс/чанк (bandwidth cost)

### Per-token bandwidth (если бы per-token):
- 0.774 GB / 512 = 1.51 МБ/токен
- Bandwidth-limited: 360 GB/s / 1.51 MB = 238K ток/с

**Вывод: bandwidth НЕ bottleneck.** Prefill @300 ток/с упирается в compute/launch/occupancy, не в memory.

### Цель 850 ток/с — достижима?
- Чанк 512 при 850 ток/с = 0.6 с
- Bandwidth cost: 2.15 мс (0.4%) — пренебрежимо
- Compute cost: должен уложиться в 600 мс
- Текущий: 1860 мс (×3.1 ускорения нужно)

**Потолок — не bandwidth, а kernel efficiency + launch overhead.** Цель 850 реалистична при:
1. MoE grouped kernel (увеличение occupancy)
2. DeltaNet tile optimization
3. Launch overhead reduction

## Альтернативная проверка через llama.cpp
llama.cpp на том же железе: ~850 ток/с. Использует MMQ (register-level dequant, tensor core GEMM).
Наши MMQ-ядра для shared-весов уже работают (0.3 мс). MoE через indexed_moe_forward — НЕ MMQ.
Если MoE получит MMQ-level efficiency → 850 достижимо.