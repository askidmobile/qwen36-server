@echo off
setlocal EnableExtensions
set "ROOT=D:\Projects\yttri-inference"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
set "CUDA_INCLUDE_DIR=%CUDA_PATH%\include"
set "CUDA_COMPUTE_CAP=86"
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin;%CUDA_PATH%\bin;%PATH%"
rem Bench hook: KEY=VALUE lines from bench-overrides.env are applied to the
rem process environment BEFORE the server loads .env. The .env loader skips
rem names already present in the environment, so these overrides win.
if exist "%ROOT%\bench-overrides.env" for /f "usebackq tokens=*" %%L in ("%ROOT%\bench-overrides.env") do set "%%L"
set "QWEN36_ENV_FILE=%ROOT%\qwen36-server\.env"
cd /d "%ROOT%\qwen36-server" || exit /b 1
"%ROOT%\qwen36-server\target\release\yforge.exe" --env "%QWEN36_ENV_FILE%" 1>>"%ROOT%\logs\server.log" 2>&1
exit /b %ERRORLEVEL%
