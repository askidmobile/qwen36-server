@echo off
rem Launch Open WebUI (source build) for yforge - logic lives in openwebui-run.ps1
setlocal EnableExtensions
set "ROOT=D:\Projects\yttri-inference"
if not exist "%ROOT%\logs" mkdir "%ROOT%\logs"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0openwebui-run.ps1" -Root "%ROOT%" >>"%ROOT%\logs\open-webui.log" 2>&1
exit /b %ERRORLEVEL%
