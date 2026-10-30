//! fa_probe — изолированный замер префильного flash-attention (FA2 hdim=256).
//!
//! Зачем: `[pfp-attn]` в движке печатает время вокруг вызова FA2 вместе с
//! `cudaStreamSynchronize`, то есть в это число попадает и вся ранее
//! поставленная в поток работа. Чтобы понять реальную цену ядра, его надо
//! мерить отдельно — здесь, без модели и без движка.
//!
//! Запуск: fa_probe [seqlen] [iters]
//! FA_PROBE_COMPARE=1 — сверка с F32-эталоном, seqlen <= 2048.

use candle_core::{DType, Device, Tensor};
use std::time::Instant;

fn main() -> candle_core::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let t: usize = args.first().and_then(|v| v.parse().ok()).unwrap_or(16384);
    let iters: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(3);
    let compare = std::env::var("FA_PROBE_COMPARE").as_deref() == Ok("1");
    if t == 0 || iters == 0 || (compare && t > 2048) {
        candle_core::bail!(
            "seqlen/iters must be positive; FA_PROBE_COMPARE requires seqlen <= 2048"
        );
    }
    let (h, h_k, d) = (16usize, 4usize, 256usize); // Ornith 1.5 9B
    let dev = Device::new_cuda(0)?;
    let cuda = dev.as_cuda_device()?.clone();
    let scale = (1.0 / (d as f64).sqrt()) as f32;

    // ВАЖНО: candle-flash-attn ждёт раскладку [b, seq, h, d] (не [b, h, seq, d]).
    // Движок так и делает: `q.transpose(1,2)?` перед вызовом
    // (model_weights.rs, ветка Prefill). Первая версия пробника подавала
    // [1,h,T,d] — ядро читало это как (b=1, seq=h, heads=T) и считало
    // крошечную задачу (4.9 мс «на 450 TFLOPS»).
    let q = Tensor::randn(0f32, 1f32, (1, t, h, d), &dev)?.to_dtype(DType::F16)?;
    let k = Tensor::randn(0f32, 1f32, (1, t, h_k, d), &dev)?.to_dtype(DType::F16)?;
    let v = Tensor::randn(0f32, 1f32, (1, t, h_k, d), &dev)?.to_dtype(DType::F16)?;

    // FLOPs причинного внимания: (T(T+1)/2) пар × d × 4 (QK+PV) × h голов.
    let pairs = (t as f64) * ((t + 1) as f64) / 2.0;
    let flops = pairs * (d as f64) * 4.0 * (h as f64);

    let mut best = f64::MAX;
    for it in 0..iters + 1 {
        let _ = cuda.cuda_stream().synchronize();
        let t0 = Instant::now();
        let out = candle_flash_attn::flash_attn(&q, &k, &v, scale, true)?;
        let _ = cuda.cuda_stream().synchronize();
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        std::hint::black_box(&out);
        if it > 0 {
            best = best.min(ms);
        }
    }
    if compare {
        // Численная сверка с F32-эталоном на той же форме: FA2 считает в F16 с
        // F32-аккумулятором, поэтому ждём ~1e-3, а не побитового совпадения.
        let out = candle_flash_attn::flash_attn(&q, &k, &v, scale, true)?;
        let qf = q.to_dtype(DType::F32)?.transpose(1, 2)?.contiguous()?; // [1,h,T,d]
        let kf = k.to_dtype(DType::F32)?.transpose(1, 2)?.contiguous()?;
        let vf = v.to_dtype(DType::F32)?.transpose(1, 2)?.contiguous()?;
        let g = h / h_k;
        let kf = kf
            .unsqueeze(2)?
            .broadcast_as((1, h_k, g, t, d))?
            .contiguous()?
            .reshape((1, h, t, d))?;
        let vf = vf
            .unsqueeze(2)?
            .broadcast_as((1, h_k, g, t, d))?
            .contiguous()?
            .reshape((1, h, t, d))?;
        let scores = (qf.matmul(&kf.transpose(2, 3)?.contiguous()?)? * scale as f64)?;
        let mask: Vec<f32> = (0..t)
            .flat_map(|i| (0..t).map(move |j| if j <= i { 0.0 } else { f32::NEG_INFINITY }))
            .collect();
        let mask = Tensor::from_vec(mask, (1, 1, t, t), &dev)?;
        let scores = scores.broadcast_add(&mask)?;
        let m = scores.max_keepdim(candle_core::D::Minus1)?;
        let e = scores.broadcast_sub(&m)?.exp()?;
        let z = e.sum_keepdim(candle_core::D::Minus1)?;
        let probs = e.broadcast_div(&z)?;
        let reference = probs.matmul(&vf)?.transpose(1, 2)?.contiguous()?;
        let error = (&out.to_dtype(DType::F32)? - &reference)?;
        let diff = error.abs()?.max_all()?.to_scalar::<f32>()?;
        // Где именно расходится: по головам и по строкам. Если ломается только
        // первая голова — это GQA-индекс; если хвост строк — тайл или маска.
        let (per_head, per_row) = error_profile(&error)?;
        println!("per-head max diff: {per_head:?}");
        let bad: Vec<(usize, f32)> = per_row
            .iter()
            .enumerate()
            .filter(|(_, v)| **v > 1e-2)
            .map(|(i, v)| (i, *v))
            .take(8)
            .collect();
        println!("first bad rows (idx,maxdiff): {bad:?}");
        println!("max |FA2 - F32 ref| = {diff:.3e}");
    }
    println!(
        "FA2 dense causal: T={t} h={h} h_k={h_k} d={d} -> {best:.1} ms = {:.2} TFLOPS",
        flops / (best * 1e-3) / 1e12
    );
    Ok(())
}

// Ошибка имеет раскладку FA2 [1, seq, heads, dim].
fn error_profile(error: &Tensor) -> candle_core::Result<(Vec<f32>, Vec<f32>)> {
    let abs = error.abs()?;
    let per_head = abs.max(3)?.max(1)?.squeeze(0)?.to_vec1::<f32>()?;
    let per_row = abs.max(3)?.max(2)?.squeeze(0)?.to_vec1::<f32>()?;
    Ok((per_head, per_row))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_profile_separates_heads_from_sequence_rows() -> candle_core::Result<()> {
        let error = Tensor::from_vec(
            vec![-1f32, 2., 3., 4., 5., -6., 7., 8., 9., 10., -11., 12.],
            (1, 3, 2, 2),
            &Device::Cpu,
        )?;
        let (heads, rows) = error_profile(&error)?;
        assert_eq!(heads, vec![10., 12.]);
        assert_eq!(rows, vec![4., 8., 12.]);
        Ok(())
    }
}
