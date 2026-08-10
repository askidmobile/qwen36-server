//! VRAM-планер (FR-002): до загрузки модели считает бюджет и снижает
//! ctx/slots, чтобы сервер не ушёл в WDDM paging (12 GB) или CUDA OOM.
//!
//! Источник размеров — GGUF metadata (читается быстро, tensor data не
//! загружается). Формулы KV/state — из параметров архитектуры.

use anyhow::{anyhow, Result};
use std::path::Path;

/// Результат планирования.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan {
    pub ctx: usize,
    pub slots: usize,
    /// Текст раскладки для stdout (spec §8).
    pub report: String,
}

/// Параметры модели, нужные для расчёта (из GGUF metadata).
#[derive(Debug, Clone)]
pub struct ModelFootprint {
    /// Байты весов, грузящихся в VRAM (оценка = file_size × 0.95).
    pub weights_mib: usize,
    /// Блоков всего / полно-Attention блоков / DeltaNet блоков.
    pub attn_blocks: usize,
    pub delta_blocks: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// Нативный контекст модели (metadata context_length).
    pub native_ctx: usize,
    /// ssm state: n_v_heads × state_size² × 4B на DeltaNet блок на слот.
    pub ssm_state_mib_per_block: f64,
}

/// Чтение footprint из GGUF (только metadata + tensor_infos, без данных).
pub fn footprint_from_gguf(path: &Path) -> Result<ModelFootprint> {
    use candle_core::quantized::gguf_file::Content;
    let file_size = std::fs::metadata(path)?.len();
    let mut file = std::fs::File::open(path)?;
    let ct = Content::read(&mut file)?;
    let md = &ct.metadata;

    let arch = match md.get("general.architecture") {
        Some(candle_core::quantized::gguf_file::Value::String(s)) => s.clone(),
        _ => return Err(anyhow!("no general.architecture")),
    };
    let prefix = match arch.as_str() {
        "qwen35" => "qwen35",
        "qwen35moe" => "qwen35moe",
        other => return Err(anyhow!("unsupported architecture for vram plan: {other}")),
    };
    let g = |k: &str| -> Option<usize> {
        md.get(&format!("{prefix}.{k}"))
            .and_then(|v| v.to_u32().ok())
            .map(|v| v as usize)
    };
    let block_count = g("block_count").ok_or_else(|| anyhow!("no block_count"))?;
    let interval = g("full_attention_interval").unwrap_or(4).max(1);
    let attn_blocks = (0..block_count).filter(|i| (i + 1) % interval == 0).count();
    let delta_blocks = block_count - attn_blocks;
    let kv_heads = g("attention.head_count_kv").unwrap_or(2);
    let head_dim = g("attention.key_length").unwrap_or(256);
    let n_v_heads = g("ssm.time_step_rank").unwrap_or(32);
    let state_size = g("ssm.state_size").unwrap_or(128);
    let ssm_state_mib_per_block =
        (n_v_heads * state_size * state_size * 4) as f64 / 1024.0 / 1024.0;

    let native_ctx = g("context_length").unwrap_or(0);

    Ok(ModelFootprint {
        weights_mib: (file_size as f64 * 0.95 / 1024.0 / 1024.0) as usize,
        attn_blocks,
        delta_blocks,
        kv_heads,
        head_dim,
        native_ctx,
        ssm_state_mib_per_block,
    })
}

/// Total VRAM (MiB) через nvidia-smi (Windows/Linux). None если недоступно
/// (macOS/CPU — планер не применяется, unified memory).
pub fn total_vram_mib() -> Option<usize> {
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8_lossy(&out.stdout)
        .split_whitespace()
        .next()?
        .parse()
        .ok()
}

/// Свободная VRAM (MiB) через nvidia-smi. Для ожидания освобождения при switch.
pub fn free_vram_mib() -> Option<usize> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.free", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8_lossy(&out.stdout)
        .split_whitespace()
        .next()?
        .parse()
        .ok()
}

/// Рабочий запас: cuBLAS workspace, dequant scratch, logits, фрагментация.
const WORKSPACE_MIB: usize = 384;
/// Доля карты, выше которой начинается paging/риск OOM.
/// 93%: на 12 GB карте это ~500 MiB запаса — ниже начинается WDDM paging
/// (измерено: 98% → коллапс скорости). 90% было бы безопаснее, но тогда
/// 35B-A3B IQ2_XXS не влезает даже минимально — а работает (впритык).
const BUDGET_FRAC: f64 = 0.93;

fn kv_mib_per_slot(fp: &ModelFootprint, ctx: usize) -> f64 {
    // 2 = K+V; 2B = F16.
    fp.attn_blocks as f64 * 2.0 * fp.kv_heads as f64 * fp.head_dim as f64 * ctx as f64 * 2.0
        / 1024.0
        / 1024.0
}

fn state_mib_per_slot(fp: &ModelFootprint) -> f64 {
    fp.delta_blocks as f64 * fp.ssm_state_mib_per_block
}

/// Подбор (ctx, slots): ctx вычисляется аналитически из остатка бюджета
/// (линейно от KV/токен), slots режутся только если даже ctx=2048 не влезает.
pub fn compute(total_mib: usize, fp: &ModelFootprint, req_ctx: usize, req_slots: usize) -> Result<Plan> {
    let budget = total_mib as f64 * BUDGET_FRAC;
    let state = state_mib_per_slot(fp);
    // KV на 1 токен контекста на слот (MiB).
    let kv_per_tok = kv_mib_per_slot(fp, 1);

    for slots in (1..=req_slots).rev() {
        let base = fp.weights_mib as f64 + WORKSPACE_MIB as f64 + slots as f64 * state;
        let room = budget - base;
        if room <= 0.0 {
            continue;
        }
        let max_ctx = (room / (slots as f64 * kv_per_tok)) as usize;
        // Snap вниз до 1024; минимально допустимый — 2048.
        let max_ctx = max_ctx / 1024 * 1024;
        if max_ctx < 2048 {
            continue;
        }
        let ctx = req_ctx.min(max_ctx);
        if ctx < 2048 {
            continue;
        }
        let need = fp.weights_mib as f64
            + slots as f64 * (kv_mib_per_slot(fp, ctx) + state)
            + WORKSPACE_MIB as f64;
        let changed = ctx != req_ctx || slots != req_slots;
        let report = format!(
            "[vram] total={total_mib}MiB weights={}MiB kv={:.0}MiB/slot state={:.0}MiB/slot workspace={WORKSPACE_MIB}MiB\n\
             [vram] plan: ctx={ctx} slots={slots}{}",
            fp.weights_mib,
            kv_mib_per_slot(fp, ctx),
            state,
            if changed {
                format!(
                    " (запрошено {req_ctx}/{req_slots} — снижено под {:.0}% карты, ~{need:.0}MiB)",
                    BUDGET_FRAC * 100.0
                )
            } else {
                " (как запрошено)".to_string()
            }
        );
        return Ok(Plan { ctx, slots, report });
    }
    let min_need = fp.weights_mib as f64
        + kv_mib_per_slot(fp, 2048)
        + state
        + WORKSPACE_MIB as f64;
    Err(anyhow!(
        "VRAM не хватает даже для 1 слота × ctx 2048: нужно ~{min_need:.0}MiB, бюджет {budget:.0}MiB ({total_mib}MiB total)"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fp_35b() -> ModelFootprint {
        // Qwen3.6-35B-A3B: 40 блоков, interval 4 → 10 attn / 30 delta.
        ModelFootprint {
            weights_mib: 10500,
            attn_blocks: 10,
            delta_blocks: 30,
            kv_heads: 2,
            head_dim: 256,
            native_ctx: 262144,
            ssm_state_mib_per_block: 2.0,
        }
    }

    #[test]
    fn fits_as_requested_on_24gb() {
        let p = compute(24576, &fp_35b(), 81920, 4).unwrap();
        // На 24 GB должно влезть что-то разумное без жёсткого урезания.
        assert!(p.slots >= 2);
        assert!(p.ctx >= 8192);
    }

    #[test]
    fn clamps_on_12gb() {
        let p = compute(12288, &fp_35b(), 81920, 4).unwrap();
        assert!(p.ctx < 81920 || p.slots < 4, "должно снизиться: {p:?}");
        assert!(p.report.contains("снижено"));
    }

    #[test]
    fn fails_when_nothing_fits() {
        let mut fp = fp_35b();
        fp.weights_mib = 12000; // веса больше карты
        assert!(compute(12288, &fp, 8192, 4).is_err());
    }

    #[test]
    fn kv_formula_scales_with_ctx() {
        let fp = fp_35b();
        let a = kv_mib_per_slot(&fp, 4096);
        let b = kv_mib_per_slot(&fp, 8192);
        assert!((b - 2.0 * a).abs() < 1e-9);
    }
}
