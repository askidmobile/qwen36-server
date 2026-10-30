//! Диагностика: Q4_K B=1 matvec против CPU-эталона после деквантования.

#[cfg(feature = "cuda")]
fn main() -> candle_core::Result<()> {
    use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
    use candle_core::{Device, Module, Tensor};
    use std::sync::Arc;

    let dev = Device::new_cuda(0)?;
    let (n, k) = (4096usize, 4096usize);
    let w = Tensor::randn(0f32, 0.02f32, (n, k), &Device::Cpu)?;
    let qt = Arc::new(QTensor::quantize(&w, GgmlDType::Q4K)?);
    let qm = QMatMul::from_arc(qt.clone())?;
    let x = Tensor::randn(0f32, 1f32, (1usize, 1usize, k), &dev)?;
    let got = qm.forward(&x)?.to_device(&Device::Cpu)?.reshape((1, n))?;
    let want = x
        .to_device(&Device::Cpu)?
        .reshape((1, k))?
        .matmul(&w.t()?.contiguous()?)?;
    let diff = (&got - &want)?.abs()?;
    let max = diff.max_all()?.to_scalar::<f32>()?;
    let finite = got
        .flatten_all()?
        .to_vec1::<f32>()?
        .iter()
        .filter(|v| v.is_finite())
        .count();
    println!("Q4K B=1 n={n} k={k} max_abs_diff={max:.6} finite={finite}/{n}");
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("requires --features cuda or cuda-turing");
}
