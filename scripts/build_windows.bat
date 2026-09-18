@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem qwen36-server Windows CUDA build. Production: VS 2022 + CUDA 13.2.
rem CUDA 12.4 rollback: set YTTRI_CUDA=...\v12.4 перед запуском.
rem CUDA_PATH устанавливается ПОСЛЕ VsDevCmd, иначе VS env её перетирает.
if not defined CUDA_COMPUTE_CAP set "CUDA_COMPUTE_CAP=86"
set "YTTRI_CUDA=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"

if not exist "%CUDA_PATH%\bin\nvcc.exe" (
  echo ERROR: nvcc not found: "%CUDA_PATH%\bin\nvcc.exe"
  echo Set CUDA_PATH to installed CUDA Toolkit directory.
  exit /b 1
)
where cargo >nul 2>nul || (
  echo ERROR: cargo not found. Install Rust from https://rustup.rs/
  exit /b 1
)

where cl >nul 2>nul
if errorlevel 1 (
  set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
  if not exist "!VSWHERE!" (
    echo ERROR: vswhere not found. Install Visual Studio 2022 Build Tools with Desktop development with C++.
    exit /b 1
  )
  for /f "usebackq tokens=*" %%I in (`"!VSWHERE!" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2^>nul`) do set "VSINSTALL=%%I"
  if not defined VSINSTALL (
    echo ERROR: Visual Studio C++ build tools not found.
    exit /b 1
  )
  call "!VSINSTALL!\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 || exit /b 1
)

set "CUDA_PATH=%YTTRI_CUDA%"
set "PATH=%CUDA_PATH%\bin;%PATH%"
set "LIB=%CUDA_PATH%\lib\x64;%LIB%"

echo [build] CUDA_PATH=%CUDA_PATH%
echo [build] CUDA_COMPUTE_CAP=%CUDA_COMPUTE_CAP%
cargo build --release --features cuda || exit /b 1

echo [ok] target\release\yforge.exe
endlocal
