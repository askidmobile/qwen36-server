@echo off
setlocal EnableExtensions

if not defined CUDA_PATH set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
if not defined QWEN36_ENV_FILE set "QWEN36_ENV_FILE=%~dp0..\.env"
set "PATH=%CUDA_PATH%\bin;%PATH%"

"%~dp0..\target\release\qwen36-server.exe"
set "EXIT_CODE=%ERRORLEVEL%"
endlocal & exit /b %EXIT_CODE%
