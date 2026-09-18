@echo off
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
set "PATH=%CUDA_PATH%\bin;%PATH%"
set "CUTLASS=C:\Users\Askid\.cudaforge\git\checkouts\cutlass-7d49e6c7e2f8896c\include"
"%CUDA_PATH%\bin\nvcc.exe" -O3 -std=c++17 -arch=sm_86 -I "%CUTLASS%" -I D:\Projects\yttri-forge\engine\candle-flash-attn\kernels -o D:\Projects\yttri-inference\tools\qk_scope_ncontig.exe D:\Projects\yttri-inference\tools\qk_scope_ncontig.cu || exit /b 1
D:\Projects\yttri-inference\tools\qk_scope_ncontig.exe
exit /b %ERRORLEVEL%
