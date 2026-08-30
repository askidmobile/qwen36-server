@echo off
setlocal EnableExtensions
set "ROOT=D:\Projects\yttri-inference"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
set "CUDA_INCLUDE_DIR=%CUDA_PATH%\include"
set "CUDA_COMPUTE_CAP=86"
set "PATH=%CUDA_PATH%\bin;%PATH%"
set "QWEN36_ENV_FILE=%ROOT%\qwen36-server\.env"
cd /d "%ROOT%\qwen36-server" || exit /b 1
"%ROOT%\qwen36-server\target\release\qwen36-server.exe" 1>>"%ROOT%\logs\server.log" 2>&1
exit /b %ERRORLEVEL%

