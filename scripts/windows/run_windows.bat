@echo off
setlocal EnableExtensions

if not defined CUDA_PATH set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
if not defined ENV_FILE set "ENV_FILE=%~dp0..\.env"
if not defined YFORGE_CURRENT set "YFORGE_CURRENT=D:\Models\yttri\qwen3.5-4b\current"
set "PATH=%CUDA_PATH%\bin;%PATH%"

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0qwen35_run_current.ps1" -Current "%YFORGE_CURRENT%" -EnvFile "%ENV_FILE%"
set "EXIT_CODE=%ERRORLEVEL%"
endlocal & exit /b %EXIT_CODE%
