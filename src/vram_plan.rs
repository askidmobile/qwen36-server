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
    /// Веса ствола+output на GPU (без экспертов и nextn, FR-010).
    pub weights_mib: usize,
    /// Веса при резидентных экспертах (ствол+эксперты, без nextn при MTP=0).
    pub weights_all_mib: usize,
    /// Маршрутизируемые эксперты (ffn_*_exps) — VRAM при vram, pinned RAM при ram.
    pub experts_mib: usize,
    /// Размещение по умолчанию из MOE_EXPERTS ("vram"|"ram" — решение auto
    /// принимает compute_dynamic от KV-бюджета).
    pub moe_placement: &'static str,
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
/// FR-010: веса на GPU считаются по именам тензоров — `ffn_*_exps` при
/// размещении в RAM исключаются (pinned host), nextn-блок исключается при
/// MTP=0; `output` и ствол считаются как есть.
pub fn footprint_from_gguf(path: &Path) -> Result<ModelFootprint> {
    footprint_from_gguf_with(path, &crate::config::moe_placement_from_env())
}

pub fn footprint_from_gguf_with(path: &Path, moe_experts: &str) -> Result<ModelFootprint> {
    let (ct, _mmap) = qwen35_batch::real::ytf16::content_any_path(path)?;
    let md = &ct.metadata;

    let arch = match md.get("general.architecture") {
        Some(candle_core::quantized::gguf_file::Value::String(s)) => s.clone(),
        _ => return Err(anyhow!("no general.architecture")),
    };
    let prefix = arch.as_str();
    let g = |k: &str| -> Option<usize> {
        md.get(&format!("{prefix}.{k}"))
            .and_then(|v| v.to_u32().ok())
            .map(|v| v as usize)
    };
    // Qwen3.8+: block_count включает nextn/MTP-слои (blk.<last>), а движок их
    // при MTP=0 не грузит (model_weights.rs: block_count_raw - nextn_layers).
    // Без этой поправки план считал в весах лишний слой: на 27B это 430 МиБ
    // фантомного расхода (замер 2026-09-18), из-за чего VRAM-бюджет и
    // решение auto о размещении экспертов были смещены.
    let block_count_raw = g("block_count").ok_or_else(|| anyhow!("no block_count"))?;
    let nextn_layers = g("nextn_predict_layers").unwrap_or(0);
    let block_count = block_count_raw.saturating_sub(nextn_layers);
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

    // FR-010: раскладка весов по именам тензоров. Прежняя оценка
    // «file_size × 0.95» для 35B-A3B завышала VRAM на ~8.6 ГиБ экспертов
    // и отказывала планеру до загрузки (приходился NO_VRAM_PLAN=1).
    let mtp_on = std::env::var("MTP").as_deref() == Ok("1");
    let mut gpu_bytes: u64 = 0;
    let mut experts_bytes: u64 = 0;
    for (name, info) in ct.tensor_infos.iter() {
        let size = (info.shape.elem_count() / info.ggml_dtype.block_size()
            * info.ggml_dtype.type_size()) as u64;
        let layer = name
            .strip_prefix("blk.")
            .and_then(|r| r.split('.').next())
            .and_then(|i| i.parse::<usize>().ok());
        let is_exps = name.contains(".ffn_gate_exps.")
            || name.contains(".ffn_up_exps.")
            || name.contains(".ffn_down_exps.");
        let is_nextn = layer.map(|l| l >= block_count).unwrap_or(false);
        if is_nextn && !mtp_on {
            continue; // nextn не грузится при MTP=0
        }
        if is_exps {
            experts_bytes += size;
            if moe_experts != "ram" {
                gpu_bytes += size; // резидентные эксперты — VRAM
            }
        } else if !name.starts_with("token_embd") {
            gpu_bytes += size; // ствол/output — VRAM (token_embd живёт в RAM)
        }
    }
    let mib = |b: u64| (b as f64 / 1024.0 / 1024.0) as usize;
    let (moe_placement, placement_note) = if experts_bytes == 0 {
        ("vram", "модель без маршрутизируемых экспертов")
    } else {
        match moe_experts {
            "ram" => ("ram", "запрошено ram"),
            "vram" => ("vram", "запрошено vram"),
            _ => {
                // auto: та же логика, что у движка (expert_store::resolve_auto)
                let need_all = gpu_bytes;
                let trunk = gpu_bytes - experts_bytes;
                let free = free_vram_mib()
                    .map(|f| (f * 1024 * 1024) as u64)
                    .unwrap_or(0);
                if free >= need_all && need_all > 0 {
                    ("vram", "auto: ствол+эксперты помещаются в свободную VRAM")
                } else if free >= trunk {
                    (
                        "ram",
                        "auto: ствол+эксперты не помещаются, ствол помещается — эксперты в RAM",
                    )
                } else {
                    ("ram", "auto: тесно и без экспертов — эксперты в RAM")
                }
            }
        }
    };
    eprintln!(
        "[vram] moe: requested={} → эксперты {:.0} МиБ ({}) — решение в плане от KV-бюджета; {}",
        moe_experts,
        experts_bytes as f64 / 1024.0 / 1024.0,
        if moe_placement == "ram" {
            "pinned RAM"
        } else {
            "VRAM"
        },
        placement_note,
    );

    Ok(ModelFootprint {
        weights_mib: mib(gpu_bytes - experts_bytes),
        weights_all_mib: mib(gpu_bytes),
        experts_mib: mib(experts_bytes),
        moe_placement,
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
    // 2 = K+V. Производственный пул — int8 (KV_POOL_Q8=1): hd+2 байта на
    // строку (байты + масштаб); иначе F16 = 2 байта.
    let kv_bytes_per_head = if std::env::var("KV_POOL_Q8").as_deref() == Ok("1") {
        (fp.head_dim + 2) as f64
    } else {
        fp.head_dim as f64 * 2.0
    };
    fp.attn_blocks as f64 * 2.0 * fp.kv_heads as f64 * kv_bytes_per_head * ctx as f64
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
    moe_experts: &str,
) -> Result<DynPlan> {
    let budget = total_mib as f64 * BUDGET_FRAC;
    let state = state_mib_per_slot(fp);
    let kv_per_tok = kv_mib_per_slot(fp, 1);
    // FR-021: решение auto принимает KV-бюджет, а не «влезают ли веса».
    // При vram-весах на KV остаётся меньше: если там KV на 2048 ток/слот
    // не влезает, а при ram-весах влезает — эксперты уходят в pinned RAM.
    let (weights_mib, placement) = if fp.experts_mib == 0 || moe_experts == "vram" {
        (fp.weights_all_mib, "vram")
    } else if moe_experts == "ram" {
        (fp.weights_mib, "ram")
    } else {
        // auto: пробуем резидент, при нехватке KV — выгрузку.
        let kv_resident =
            budget - fp.weights_all_mib as f64 - WORKSPACE_MIB as f64 - req_slots as f64 * state;
        let kv_ram =
            budget - fp.weights_mib as f64 - WORKSPACE_MIB as f64 - req_slots as f64 * state;
        if kv_resident >= kv_per_tok * req_ctx as f64 {
            (fp.weights_all_mib, "vram")
        } else if kv_ram >= kv_per_tok * req_ctx as f64 {
            (fp.weights_mib, "ram")
        } else if kv_resident >= kv_per_tok * 2048.0 {
            (fp.weights_all_mib, "vram")
        } else if kv_ram >= kv_per_tok * 2048.0 {
            (fp.weights_mib, "ram")
        } else {
            return Err(anyhow!(
                "VRAM не хватает при любом размещении: KV после весов {:.0}MiB (vram) / {:.0}MiB (ram) < 2K ток/слот",
                kv_resident, kv_ram
            ));
        }
    };
    let kv_budget = budget - weights_mib as f64 - WORKSPACE_MIB as f64 - req_slots as f64 * state;
    if kv_budget < kv_per_tok * 2048.0 {
        return Err(anyhow!(
            "VRAM не хватает: после весов ({placement}) и workspace на KV остаётся {kv_budget:.0}MiB (<2K токенов на слот)"
        ));
    }
    let ctx = req_ctx.min(if fp.native_ctx > 0 {
        fp.native_ctx
    } else {
        req_ctx
    });
    // Сколько слотов могут быть одновременно заполнены ctx полностью.
    let full_concurrent = (kv_budget / (kv_per_tok * ctx as f64)).floor() as usize;
    // Прямая цена одного полного ctx. Без неё строка читается навыворот:
    // q8 на 128k и f16 на 64k стоят одинаково (~2 ГиБ), а «~5 слотов» у
    // q8-64k означает вдвое более дешёвый слот, а не вдвое больший расход.
    let kv_per_slot = kv_per_tok * ctx as f64;
    // Оценка справочная, и это важно проговорить: движок считает окно пула
    // заново, от фактически свободной VRAM в момент создания, и бюджет отсюда
    // не смотрит. Числа расходятся — на 3060 планировщик дал «~0 слотов
    // полного ctx» при 131072, а движок выделил окно 129728 и всё работало.
    // Без этой оговорки строка читается как «не влезет», хотя это не так.
    let report = format!(
        "[vram] total={total_mib}MiB weights={}MiB (experts={}, kv_budget={kv_budget:.0}MiB state={state:.0}MiB/slot)\n\
         [vram] plan(dynamic): ctx={ctx} slots={req_slots} — KV {kv_per_slot:.0}MiB/слот, в бюджет {kv_budget:.0}MiB влезает ~{full_concurrent} полных ctx, очередь FIFO\n\
         [vram] оценка справочная: окно пула движок считает сам от свободной VRAM — фактическое смотри в строке [kv] paged pool",
        weights_mib,
        placement,
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
            weights_all_mib: 9745,
            experts_mib: 0,
            moe_placement: "vram",
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
