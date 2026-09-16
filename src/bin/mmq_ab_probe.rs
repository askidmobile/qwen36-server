//! mmq_ab_probe — сверка dp4a- и mma-ветки MMQ на одной и той же задаче.
//!
//! Зачем: с 2026-09-16 малые батчи (m <= 64) идут на dp4a-ядро
//! (candle_mmq_dp4a.cu), потому что mma-тайл всегда считает 128 столбцов.
//! Сверка с де-квантованным эталоном на CPU отвечает на вопрос «числа те же»,
//! не поднимая модель и не перезапуская сервер.
//!
//! Запуск: mmq_ab_probe [dtype: q4k|q6k|q5k|q8_0] [m] [n] [k]

use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{Device, Module, Tensor};

fn max_abs_diff(a: &Tensor, b: &Tensor) -> candle_core::Result<(f64, f64)> {
    let d = (a - b)?.abs()?;
    let max = d.max_all()?.to_scalar::<f32>()? as f64;
    let mean = d.mean_all()?.to_scalar::<f32>()? as f64;
    Ok((max, mean))
}

fn run(qm: &QMatMul, x: &Tensor, dp4a: bool) -> candle_core::Result<Tensor> {
    if dp4a {
        std::env::set_var("MMQ_DP4A", "1");
    } else {
        std::env::set_var("MMQ_DP4A", "0");
    }
    qm.forward(x)
}

fn main() -> candle_core::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dtype = match args.first().map(|s| s.as_str()).unwrap_or("q4k") {
        "q6k" => GgmlDType::Q6K,
        "q5k" => GgmlDType::Q5K,
        "q8_0" => GgmlDType::Q8_0,
        "q4_0" => GgmlDType::Q4_0,
        _ => GgmlDType::Q4K,
    };
    let m: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(32);
    let n: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(1024);
    let k: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(4096);

    let dev = Device::new_cuda(0)?;
    let w = Tensor::randn(0f32, 0.02f32, (n, k), &Device::Cpu)?;
    let x_cpu = Tensor::randn(0f32, 1f32, (m, k), &Device::Cpu)?;
    let qt = QTensor::quantize_onto(&w, dtype, &dev)?;
    let qm = QMatMul::from_qtensor(qt)?;
    let x = x_cpu.to_device(&dev)?;

    // Эталон: де-квантованные веса × вход, всё в f32 на CPU.
    let w_ref = qm
        .dequantize_f16()?
        .to_dtype(candle_core::DType::F32)?
        .to_device(&Device::Cpu)?;
    let y_ref = x_cpu.matmul(&w_ref.t()?.contiguous()?)?;

    let y_mma = run(&qm, &x, false)?.to_device(&Device::Cpu)?.to_dtype(candle_core::DType::F32)?;
    let y_dp4a = run(&qm, &x, true)?.to_device(&Device::Cpu)?.to_dtype(candle_core::DType::F32)?;

    let (mma_max, mma_mean) = max_abs_diff(&y_mma, &y_ref)?;
    let (dp4a_max, dp4a_mean) = max_abs_diff(&y_dp4a, &y_ref)?;
    let (ab_max, ab_mean) = max_abs_diff(&y_dp4a, &y_mma)?;
    println!(
        "{dtype:?} [n={n},k={k}] m={m}: mma|max={mma_max:.3e} mean={mma_mean:.3e} | dp4a|max={dp4a_max:.3e} mean={dp4a_mean:.3e} | mma-dp4a|max={ab_max:.3e} mean={ab_mean:.3e}"
    );
    Ok(())
}
