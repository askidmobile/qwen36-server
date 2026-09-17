@echo off
setlocal EnableExtensions
set "ROOT=D:\Projects\yttri-inference"
set "RUSTUP_HOME=C:\Users\Askid\.rustup"
set "CARGO_HOME=C:\Users\Askid\.cargo"
set "RUSTUP_TOOLCHAIN=stable-x86_64-pc-windows-msvc"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
set "CUDA_INCLUDE_DIR=%CUDA_PATH%\include"
set "CUDA_COMPUTE_CAP=86"
set "CUDAFORGE_THREADS=4"
set "CANDLE_FLASH_ATTN_BUILD_DIR=%ROOT%\fa-build-minimal"
set "CANDLE_FLASH_ATTN_MINIMAL=1"
set "LIB=%CUDA_PATH%\lib\x64;%LIB%"
set "PATH=C:\Users\Askid\.cargo\bin;%CUDA_PATH%\bin;%PATH%"
set "CARGO_TARGET_DIR=%ROOT%\target"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 || exit /b 1
cd /d D:\Projects\yttri-forge\engine\qwen35-batch || exit /b 1
cargo build --release --features cuda,real-model --bin fa_decode_probe || exit /b 2
exit /b 0
