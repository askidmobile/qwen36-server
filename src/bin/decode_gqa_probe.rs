//! Parity и микробенч paged GQA decode-attention.
//! Запуск: decode_gqa_probe [kv_len] [iters]

use candle_core::{DType, Device, Result, Tensor};

fn to_f32_vec(t: &Tensor) -> Result<Vec<f32>> {
    Ok(t.to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1()?)
}

fn shuffled_pages(count: usize) -> Vec<u32> {
    let mut pages: Vec<u32> = (0..count as u32).collect();
    let mut state = 0x9e37_79b9u32;
    for i in (1..pages.len()).rev() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        pages.swap(i, state as usize % (i + 1));
    }
    pages
}

fn parity(dev: &Device) -> Result<()> {
    use qwen35_batch::real::flash_decode_cuda::dispatch_flash_decode_paged;

    let lengths = [1usize, 15, 16, 17, 63, 64, 65, 127, 128, 129, 257, 1025];
    let max_blocks = lengths.iter().max().copied().unwrap().div_ceil(64);
    let physical_pages = max_blocks + 7;
    let k = Tensor::randn(0f32, 1f32, (physical_pages, 64, 2, 256), dev)?.to_dtype(DType::F16)?;
    let v = Tensor::randn(0f32, 1f32, (physical_pages, 64, 2, 256), dev)?.to_dtype(DType::F16)?;
    let q = Tensor::randn(0f32, 1f32, (1, 16, 256), dev)?.to_dtype(DType::F16)?;
    let pages = shuffled_pages(physical_pages);
    let cuda = dev.as_cuda_device()?.clone();
    std::env::set_var("DECODE_GQA_FOLD", "1");

    for len in lengths {
        let blocks = len.div_ceil(64);
        let table = Tensor::from_vec(pages[..blocks].to_vec(), (1, blocks), dev)?;
        let lens = Tensor::from_vec(vec![0u32, len as u32], 2, dev)?;

        std::env::set_var("DECODE_GQA_MMA", "1");
        let mma = dispatch_flash_decode_paged(&cuda, &q, &k, &v, &lens, &table, 0.0625)?;
        std::env::set_var("DECODE_GQA_MMA", "0");
        let scalar = dispatch_flash_decode_paged(&cuda, &q, &k, &v, &lens, &table, 0.0625)?;
        cuda.cuda_stream()
            .synchronize()
            .map_err(candle_core::Error::wrap)?;
        let mma = to_f32_vec(&mma)?;
        let scalar = to_f32_vec(&scalar)?;
        let max_abs = mma
            .iter()
            .zip(&scalar)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        if !max_abs.is_finite() || max_abs > 2e-3 {
            candle_core::bail!("GQA parity kv={len}: max|delta|={max_abs}");
        }
        eprintln!("[parity] kv={len} max|delta|={max_abs:.7}");
    }
    Ok(())
}

fn bench(dev: &Device, kv_len: usize, iters: usize) -> Result<()> {
    use qwen35_batch::real::flash_decode_cuda::dispatch_flash_decode_paged;

    let blocks = kv_len.div_ceil(64);
    let k = Tensor::randn(0f32, 1f32, (blocks, 64, 2, 256), dev)?.to_dtype(DType::F16)?;
    let v = Tensor::randn(0f32, 1f32, (blocks, 64, 2, 256), dev)?.to_dtype(DType::F16)?;
    let q = Tensor::randn(0f32, 1f32, (1, 16, 256), dev)?.to_dtype(DType::F16)?;
    let table = Tensor::from_vec((0..blocks as u32).collect::<Vec<_>>(), (1, blocks), dev)?;
    let lens = Tensor::from_vec(vec![0u32, kv_len as u32], 2, dev)?;
    let cuda = dev.as_cuda_device()?.clone();
    std::env::set_var("DECODE_GQA_MMA", "1");
    dispatch_flash_decode_paged(&cuda, &q, &k, &v, &lens, &table, 0.0625)?;
    cuda.cuda_stream()
        .synchronize()
        .map_err(candle_core::Error::wrap)?;
    let t = std::time::Instant::now();
    for _ in 0..iters {
        dispatch_flash_decode_paged(&cuda, &q, &k, &v, &lens, &table, 0.0625)?;
    }
    cuda.cuda_stream()
        .synchronize()
        .map_err(candle_core::Error::wrap)?;
    eprintln!(
        "kv={kv_len} iters={iters} splits={} ms={:.3}",
        std::env::var("DECODE_GQA_SPLITS").unwrap_or_else(|_| "auto".into()),
        t.elapsed().as_secs_f64() * 1e3 / iters as f64
    );
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dev = Device::new_cuda(0)?;
    parity(&dev)?;
    if let Some(kv_len) = args.first().and_then(|v| v.parse().ok()) {
        let iters = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(20);
        bench(&dev, kv_len, iters)?;
    }
    Ok(())
}
