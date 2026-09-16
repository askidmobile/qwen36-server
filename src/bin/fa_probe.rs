//! fa_probe — изолированный замер префильного flash-attention (FA2 hdim=256).
//!
//! Зачем: `[pfp-attn]` в движке печатает время вокруг вызова FA2 вместе с
//! `cudaStreamSynchronize`, то есть в это число попадает и вся ранее
//! поставленная в поток работа. Чтобы понять реальную цену ядра, его надо
//! мерить отдельно — здесь, без модели и без движка.
//!
//! Запуск: fa_probe [seqlen] [iters]

use candle_core::{DType, Device, Tensor};
use std::time::Instant;

fn main() -> candle_core::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let t: usize = args.first().and_then(|v| v.parse().ok()).unwrap_or(16384);
    let iters: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(3);
    let (h, h_k, d) = (16usize, 4usize, 256usize);
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
    println!(
        "FA2 dense causal: T={t} h={h} h_k={h_k} d={d} -> {best:.1} ms = {:.2} TFLOPS",
        flops / (best * 1e-3) / 1e12
    );
    Ok(())
}
