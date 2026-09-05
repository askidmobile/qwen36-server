@echo off
:: HF safetensors -> GGUF F16 -> квант. Конвейер стенда yttri-win.
:: Использование: convert-hf-gguf.bat <SRC_DIR> <OUT_DIR> <BASENAME> [QUANT]
::   convert-hf-gguf.bat D:\Models\source\Qwen3.8-9B-Distill ^
::                       D:\Models\empero-ai\Qwen3.8-9B-Distill-GGUF ^
::                       Qwen3.8-9B-Distill Q4_K_M
:: F16 остаётся рядом: следующий квант не требует повторной конвертации.
:: MTP/nextn-слой попадает в GGUF автоматически, vision-башня отбрасывается.
setlocal EnableExtensions
if "%~3"=="" (
  echo usage: %~nx0 ^<SRC_DIR^> ^<OUT_DIR^> ^<BASENAME^> [QUANT]
  exit /b 64
)
set "SRC=%~1"
set "OUT_DIR=%~2"
set "NAME=%~3"
set "QUANT=%~4"
if "%QUANT%"=="" set "QUANT=Q4_K_M"

set "PY=D:\Projects\yttri-inference\tools\llama.cpp-8e7f22b67ef4667b4ddd50230771287f328cfb3f\.venv\Scripts\python.exe"
set "LLAMA_DIR=D:\Projects\yttri-inference\tools\llama.cpp-8e7f22b67ef4667b4ddd50230771287f328cfb3f"
set "QUANT_EXE=D:\Projects\yttri-inference\llama-b10375\llama-quantize.exe"
set "F16=%OUT_DIR%\%NAME%-F16.gguf"
set "OUT=%OUT_DIR%\%NAME%-%QUANT%.gguf"

if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"
if exist "%F16%" goto quantize

echo === 1. HF -^> GGUF F16 ===
"%PY%" "%LLAMA_DIR%\convert_hf_to_gguf.py" "%SRC%" --outfile "%F16%" --outtype f16
if errorlevel 1 exit /b 1

:quantize
echo === 2. Quantize %QUANT% ===
"%QUANT_EXE%" "%F16%" "%OUT%" %QUANT%
if errorlevel 1 exit /b 2
echo === DONE: %OUT% ===
