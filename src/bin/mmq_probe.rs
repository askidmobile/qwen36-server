//! mmq_probe — изолированный замер MMQ на РЕАЛЬНЫХ формах Ornith-1.5-9B.
//!
//! Зачем: nsys к нашему серверу не цепляется, а `TRACE_MMQ` меряет с
//! синхронизацией на каждый вызов — числа складываются в сумму больше самого
//! префила. Поэтому формы меряем отдельным процессом, без модели.
//!
//! Запуск: mmq_probe [dtype: q4k|q6k|q4_0|q4_1|q5_0|q5_1|q8_0] [m] [n] [k] [iters]

use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{Device, Module, Tensor};

fn main() -> candle_core::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dtype = match args.first().map(|s| s.as_str()).unwrap_or("q4k") {
        "q6k" => GgmlDType::Q6K,
        "q4_0" => GgmlDType::Q4_0,
        "q4_1" => GgmlDType::Q4_1,
        "q5_0" => GgmlDType::Q5_0,
        "q5_1" => GgmlDType::Q5_1,
        "q8_0" => GgmlDType::Q8_0,
        _ => GgmlDType::Q4K,
    };
    let m: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(16384);
    let n: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(4096);
    let k: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(4096);
    let iters: usize = args.get(4).and_then(|v| v.parse().ok()).unwrap_or(3);
    let f16_path = std::env::var("MMQ_PROBE_F16").as_deref() == Ok("1");
    // MMQ_PROBE_F16_CACHED=1 — заранее деквантизованный F16-вес, как в
    // prefill-кэше. Отличие от MMQ_PROBE_F16: там dequantize_f16() зовётся
    // внутри каждого forward и попадает в замер.
    let f16_cached = std::env::var("MMQ_PROBE_F16_CACHED").as_deref() == Ok("1");

    let dev = Device::new_cuda(0)?;
    let w = Tensor::randn(0f32, 0.02f32, (n, k), &Device::Cpu)?;
    let qm = QMatMul::from_qtensor(QTensor::quantize_onto(&w, dtype, &dev)?)?;
    let x = Tensor::randn(0f32, 1f32, (m, k), &dev)?;
    let f16w = if f16_cached {
        Some(QMatMul::TensorF16(qm.dequantize_f16()?))
    } else {
        None
    };

    let mut best = f64::MAX;
    for it in 0..iters + 2 {
        dev.synchronize()?;
        let t0 = std::time::Instant::now();
        let y = if let Some(w) = &f16w {
            w.forward(&x.to_dtype(candle_core::DType::F16)?)?
                .to_dtype(candle_core::DType::F32)?
        } else if f16_path {
            qm.forward_via_f16(&x)?
        } else {
            qm.forward(&x)?
        };
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

    // MMQ_PROBE_COMPARE=1 — сверить stream-k и обычную сетку на тех же
    // данных. Оба варианта считают одну и ту же матрицу, разница должна
    // быть на уровне порядка суммирования.
    if std::env::var("MMQ_PROBE_COMPARE").as_deref() == Ok("1") {
        std::env::remove_var("MMQ_STREAMK");
        dev.synchronize()?;
        let reference = qm.forward(&x)?;
        std::env::set_var("MMQ_STREAMK", "1");
        dev.synchronize()?;
        let sk = qm.forward(&x)?;
        let diff = (&reference - &sk)?.abs()?;
        let max = diff.max_all()?.to_scalar::<f32>()?;
        let mean = diff.mean_all()?.to_scalar::<f32>()?;
        let scale = reference.abs()?.mean_all()?.to_scalar::<f32>()?.max(1e-9);
        let mask = diff.gt(1e-2)?.to_dtype(candle_core::DType::F32)?;
        let mismatch = mask.sum_all()?.to_scalar::<f32>()?;
        println!(
            "compare: max={max:.3e} mean={mean:.3e} rel={:.3e} mismatch(>1e-2)={}",
            max / scale,
            mismatch
        );
    }
    Ok(())
}
