@echo off
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
set "PATH=%CUDA_PATH%\bin;%PATH%"
"%CUDA_PATH%\bin\nvcc.exe" -O3 -arch=sm_86 -o D:\Projects\yttri-inference\tools\dequant_bench.exe D:\Projects\yttri-inference\tools\dequant_bench.cu || exit /b 1
D:\Projects\yttri-inference\tools\dequant_bench.exe
exit /b %ERRORLEVEL%
