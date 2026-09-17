@echo off
setlocal
set "RUSTUP_HOME=C:\Users\Askid\.rustup"
set "CARGO_HOME=C:\Users\Askid\.cargo"
set "RUSTUP_TOOLCHAIN=stable-x86_64-pc-windows-msvc"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
set "CUDA_INCLUDE_DIR=%CUDA_PATH%\include"
set "CUDA_COMPUTE_CAP=86"
set "CANDLE_FLASH_ATTN_BUILD_DIR=D:\Projects\yttri-inference\fa-build-cache"
set "LIB=%CUDA_PATH%\lib\x64;%LIB%"
set "PATH=C:\Users\Askid\.cargo\bin;%CUDA_PATH%\bin;%PATH%"
set "CARGO_TARGET_DIR=D:\Projects\yttri-inference\target"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul
cd /d D:\Projects\yttri-forge\engine\qwen35-batch || exit /b 1
cargo test --lib --features cuda,real-model q_int8 -- --nocapture || exit /b 2
exit /b 0
