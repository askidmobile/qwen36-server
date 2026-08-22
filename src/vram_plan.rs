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

/// Кэш footprint по (path, mtime, size): антивирус тормозит открытие GGUF на 1-3с.
static FP_CACHE: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashMap<std::path::PathBuf, (u64, u64, ModelFootprint)>>,
> = std::sync::OnceLock::new();

pub fn footprint_from_gguf_cached(path: &Path) -> Result<ModelFootprint> {
    let meta = std::fs::metadata(path)?;
    let mtime = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let size = meta.len();
    let cache = FP_CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    if let Ok(guard) = cache.lock() {
        if let Some((m, sz, fp)) = guard.get(path) {
            if *m == mtime && *sz == size {
                return Ok(fp.clone());
            }
        }
    }
    let fp = footprint_from_gguf(path)?;
    if let Ok(mut guard) = cache.lock() {
        guard.insert(path.to_path_buf(), (mtime, size, fp.clone()));
    }
    Ok(fp)
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
        // Стандартные трансформеры (Gemma, Llama, Mistral, Qwen2, Phi и др.):
        // KV = 2 × n_layers × n_kv_heads × head_dim × ctx × dtype_bytes.
        // Нет DeltaNet (delta_blocks=0, ssm_state=0). Префикс в GGUF = arch.
        other => other,
    };
    let g = |k: &str| -> Option<usize> {
        md.get(&format!("{prefix}.{k}"))
            .and_then(|v| v.to_u32().ok())
            .map(|v| v as usize)
    };
    let block_count = g("block_count").ok_or_else(|| anyhow!("no block_count"))?;
    let (attn_blocks, delta_blocks) = if matches!(arch.as_str(), "qwen35" | "qwen35moe") {
        let interval = g("full_attention_interval").unwrap_or(4).max(1);
        let attn = (0..block_count).filter(|i| (i + 1) % interval == 0).count();
        (attn, block_count - attn)
    } else {
        (block_count, 0)
    };
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
/// Кэш total VRAM: nvidia-smi subprocess на Windows стоит 2-3s на вызов.
static TOTAL_VRAM_CACHE: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();

pub fn total_vram_mib() -> Option<usize> {
    *TOTAL_VRAM_CACHE.get_or_init(|| {
        let out = std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
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
    })
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
/// Оценка VRAM под on-demand компоненты (Vision ~600 MiB, MTP ~150 MiB).
pub const VISION_COMPONENT_ESTIMATE_MIB: usize = 650;
pub const MTP_COMPONENT_ESTIMATE_MIB: usize = 160;
/// Доля карты, выше которой начинается paging/риск OOM.
/// 0.89: оставляем постоянные ~1.35 GB dedicated-запаса на 12 GB карте под
/// страницы CUDA-пула и транзиенты декода. При 0.93 карта уходила в 97-98%
/// занятости → транзиентные буферы decode селились в WDDM shared memory →
/// коллапс 2.4 tok/s при ctx 24K (измерено 2026-08-22, см. BD-лог и урок).
const BUDGET_FRAC: f64 = 0.89;

fn kv_mib_per_slot(fp: &ModelFootprint, ctx: usize) -> f64 {
    // 2 = K+V; 2B = F16.
    fp.attn_blocks as f64 * 2.0 * fp.kv_heads as f64 * fp.head_dim as f64 * ctx as f64 * 2.0
        / 1024.0
        / 1024.0
}

fn state_mib_per_slot(fp: &ModelFootprint) -> f64 {
    fp.delta_blocks as f64 * fp.ssm_state_mib_per_block
}

/// Динамический план (vLLM-style): ctx НЕ режется под worst-case 4×full.
/// Слоты растут лениво; общий KV-бюджет enforce'ится рантаймом (admission +
/// очередь + force-finish самого длинного). Возвращает бюджет для движка.
pub struct DynPlan {
    pub ctx: usize,
    pub slots: usize,
    /// Общий бюджет KV+state для всех слотов (MiB).
    pub kv_budget_mib: f64,
    /// MiB на токен KV на слот.
    pub kv_per_tok_mib: f64,
    pub report: String,
}

pub fn compute_dynamic(
    total_mib: usize,
    fp: &ModelFootprint,
    req_ctx: usize,
    req_slots: usize,
) -> Result<DynPlan> {
    let budget = total_mib as f64 * BUDGET_FRAC;
    let state = state_mib_per_slot(fp);
    let kv_per_tok = kv_mib_per_slot(fp, 1);
    let kv_budget =
        budget - fp.weights_mib as f64 - WORKSPACE_MIB as f64 - req_slots as f64 * state;
    if kv_budget < kv_per_tok * 2048.0 {
        return Err(anyhow!(
            "VRAM не хватает: после весов и workspace на KV остаётся {kv_budget:.0}MiB (<2K токенов на слот)"
        ));
    }
    let ctx = req_ctx.min(if fp.native_ctx > 0 {
        fp.native_ctx
    } else {
        req_ctx
    });
    // Сколько слотов могут быть одновременно заполнены ctx полностью.
    let full_concurrent = (kv_budget / (kv_per_tok * ctx as f64)).floor() as usize;
    let report = format!(
        "[vram] total={total_mib}MiB weights={}MiB kv_budget={kv_budget:.0}MiB state={state:.0}MiB/slot\n\
         [vram] plan(dynamic): ctx={ctx} slots={req_slots} — ~{full_concurrent} слот(а) полного ctx одновременно, очередь FIFO",
        fp.weights_mib,
    );
    Ok(DynPlan {
        ctx,
        slots: req_slots,
        kv_budget_mib: kv_budget,
        kv_per_tok_mib: kv_per_tok,
        report,
    })
}

/// Подбор (ctx, slots): ctx вычисляется аналитически из остатка бюджета
/// (линейно от KV/токен), slots режутся только если даже ctx=2048 не влезает.
pub fn compute(
    total_mib: usize,
    fp: &ModelFootprint,
    req_ctx: usize,
    req_slots: usize,
) -> Result<Plan> {
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
    let min_need = fp.weights_mib as f64 + kv_mib_per_slot(fp, 2048) + state + WORKSPACE_MIB as f64;
    Err(anyhow!(
        "VRAM не хватает даже для 1 слота × ctx 2048: нужно ~{min_need:.0}MiB, бюджет {budget:.0}MiB ({total_mib}MiB total)"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fp_35b() -> ModelFootprint {
        // Qwen3.6-35B-A3B: 40 блоков, interval 4 → 10 attn / 30 delta.
        // weights ≈ реальный UD-IQ2_XXS (9745 MiB по GGUF metadata).
        ModelFootprint {
            weights_mib: 9745,
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
