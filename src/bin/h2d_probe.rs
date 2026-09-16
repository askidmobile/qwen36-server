//! h2d_probe — цена чтения логитов (248 320 × f32 = 993 КБ) с устройства:
//! обычная pageable-копия (как `Tensor::to_vec1`) против pinned-буфера.
//!
//! Запуск: h2d_probe [iters]

use candle_core::{DType, Device, Tensor};

fn main() -> candle_core::Result<()> {
    let iters: usize = std::env::args()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(50);
    let dev = Device::new_cuda(0)?;
    let n = 248_320usize;
    let x = Tensor::randn(0f32, 1f32, (n,), &dev)?;
    let cuda = dev.as_cuda_device()?.clone();

    // 1) Штатный путь candle: pageable D2H + Vec<f32>.
    let mut page_ms = f64::MAX;
    for it in 0..iters + 5 {
        let t0 = std::time::Instant::now();
        let v = x.to_vec1::<f32>()?;
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        std::hint::black_box(&v);
        if it >= 5 {
            page_ms = page_ms.min(ms);
        }
    }

    // 2) Pinned-буфер через драйверный API.
    use candle_core::cuda_backend::cudarc;
    let bytes = n * std::mem::size_of::<f32>();
    let mut host_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let res = unsafe {
        cudarc::driver::sys::cuMemHostAlloc(&mut host_ptr, bytes, 0)
    };
    if res != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
        println!("pageable: {page_ms:.3} ms; pinned недоступен (cuMemHostAlloc={res:?})");
        return Ok(());
    }
    // Указатель на данные устройства.
    let (storage, layout) = x.storage_and_layout();
    let src = match &*storage {
        candle_core::Storage::Cuda(c) => {
            let slice = c.as_cuda_slice::<f32>()?;
            let stream = cuda.cuda_stream();
            let (ptr, _g) = cudarc::driver::DevicePtr::device_ptr(slice, &stream);
            ptr
        }
        _ => unreachable!(),
    };
    let _ = layout;
    let mut pin_ms = f64::MAX;
    for it in 0..iters + 5 {
        let t0 = std::time::Instant::now();
        unsafe {
            cudarc::driver::sys::cuMemcpyDtoH_v2(host_ptr, src, bytes);
        }
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        if it >= 5 {
            pin_ms = pin_ms.min(ms);
        }
    }
    unsafe { cudarc::driver::sys::cuMemFreeHost(host_ptr) };
    println!("pageable: {page_ms:.3} ms   pinned: {pin_ms:.3} ms");
    Ok(())
}
