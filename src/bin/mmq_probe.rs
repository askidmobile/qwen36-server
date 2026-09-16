//! mmq_probe — изолированный замер MMQ на РЕАЛЬНЫХ формах Ornith-1.5-9B.
//!
//! Зачем: nsys к нашему серверу не цепляется, а `TRACE_MMQ` меряет с
//! синхронизацией на каждый вызов — числа складываются в сумму больше самого
//! префила. Поэтому формы меряем отдельным процессом, без модели.
//!
//! Запуск: mmq_probe [dtype: q4k|q6k] [m] [n] [k] [iters]

use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{Device, Module, Tensor};

fn main() -> candle_core::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dtype = match args.first().map(|s| s.as_str()).unwrap_or("q4k") {
        "q6k" => GgmlDType::Q6K,
        _ => GgmlDType::Q4K,
    };
    let m: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(16384);
    let n: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(4096);
    let k: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(4096);
    let iters: usize = args.get(4).and_then(|v| v.parse().ok()).unwrap_or(3);

    let dev = Device::new_cuda(0)?;
    let w = Tensor::randn(0f32, 0.02f32, (n, k), &Device::Cpu)?;
    let qm = QMatMul::from_qtensor(QTensor::quantize_onto(&w, dtype, &dev)?)?;
    let x = Tensor::randn(0f32, 1f32, (m, k), &dev)?;

    let mut best = f64::MAX;
    for it in 0..iters + 2 {
        dev.synchronize()?;
        let t0 = std::time::Instant::now();
        let y = qm.forward(&x)?;
        dev.synchronize()?;
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        std::hint::black_box(&y);
        if it >= 2 {
            best = best.min(ms);
        }
    }
    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    println!(
        "{dtype:?} [{n},{k}] M={m}: {best:7.2} ms = {:5.1} TFLOPS",
        flops / (best * 1e-3) / 1e12
    );
    Ok(())
}
