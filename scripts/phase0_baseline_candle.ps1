#Requires -Version 5
<#
phase0_baseline_candle.ps1 - строка 0 матрицы Phase 0: наш qwen36-server
на Qwen3.6-35B-A3B UD-IQ2_XXS, тот же протокол bench.ps1.

Usage (on yttri-win):
    powershell -ExecutionPolicy Bypass -File scripts\phase0_baseline_candle.ps1
#>
[CmdletBinding()]
param(
    [string]$ModelPath = "D:\Models\unsloth\Qwen3.6-35B-A3B-GGUF\Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf",
    [string]$ServerExe = "D:\Projects\yttri-inference\target\release\qwen36-server.exe",
    [int]$Ctx          = 8192,
    [int]$Slots        = 2,
    [int]$Port         = 18099,
    [string]$ApiKey    = "smoke-key",
    # CUDA-рантайм для запуска (в SSH-сессии его нет в PATH)
    [string]$CudaBin   = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"
)

$ErrorActionPreference = 'Stop'
if (Test-Path $CudaBin) { $env:PATH = "$CudaBin;$env:PATH" }
if (-not (Test-Path $ServerExe)) { Write-Host "FATAL: нет $ServerExe" -ForegroundColor Red; exit 1 }
if (-not (Test-Path $ModelPath)) { Write-Host "FATAL: нет $ModelPath" -ForegroundColor Red; exit 1 }

$env:MODEL     = $ModelPath
$env:CTX       = "$Ctx"
$env:SLOTS     = "$Slots"
$env:PORT      = "$Port"
$env:API_KEYS  = '[{"key":"' + $ApiKey + '","name":"bench"}]'

$log = Join-Path $env:TEMP "phase0-candle-$Port.log"
Write-Host "start: qwen36-server (ctx=$Ctx slots=$Slots) лог: $log"
$p = Start-Process -FilePath $ServerExe -WorkingDirectory (Split-Path $ServerExe) `
    -RedirectStandardOutput $log -RedirectStandardError "$log.err" `
    -WindowStyle Hidden -PassThru

try {
    Write-Host "waiting for /v1/models..." -NoNewline
    $deadline = (Get-Date).AddMinutes(12)
    while ((Get-Date) -lt $deadline) {
        if ($p.HasExited) {
            Write-Host " FATAL: процесс упал:" -ForegroundColor Red
            Get-Content "$log.err" -Tail 30; exit 1
        }
        try {
            Invoke-RestMethod -Uri "http://localhost:$Port/v1/models" `
                -Headers @{ Authorization = "Bearer $ApiKey" } -TimeoutSec 2 | Out-Null
            break
        } catch {}
        Write-Host "." -NoNewline; Start-Sleep 5
    }
    if ((Get-Date) -ge $deadline) { Write-Host " TIMEOUT" -ForegroundColor Red; exit 1 }
    Write-Host " ok"

    # Реальный id загруженной модели + ожидание готовности весов
    $mid = $null
    $deadline2 = (Get-Date).AddMinutes(10)
    while ((Get-Date) -lt $deadline2) {
        try {
            $mid = (Invoke-RestMethod -Uri "http://localhost:$Port/v1/models" `
                -Headers @{ Authorization = "Bearer $ApiKey" } -TimeoutSec 5).data[0].id
            $probe = @{ model = $mid; stream = $false; max_tokens = 1;
                        messages = @(@{ role = "user"; content = "hi" }) } | ConvertTo-Json -Depth 5 -Compress
            Invoke-RestMethod -Uri "http://localhost:$Port/v1/chat/completions" `
                -Headers @{ Authorization = "Bearer $ApiKey" } -Method Post `
                -ContentType "application/json" -Body $probe -TimeoutSec 60 | Out-Null
            break
        } catch {
            $msg = $_.ErrorDetails.Message
            if ($msg -notmatch "loading|switching") { throw }
            Write-Host "." -NoNewline; Start-Sleep 5
        }
    }
    if (-not $mid) { Write-Host " TIMEOUT (model not ready)" -ForegroundColor Red; exit 1 }
    Write-Host "model id: $mid"

    & (Join-Path $PSScriptRoot "bench.ps1") `
        -BaseUrl "http://localhost:$Port" -ApiKey $ApiKey -Model $mid -Concurrent $Slots

    $gs = Get-Process -Id $p.Id -ErrorAction SilentlyContinue
    if ($gs) { Write-Host ("server RAM working set: {0:N1} GiB" -f ($gs.WorkingSet64 / 1GB)) }
} finally {
    if ($p -and -not $p.HasExited) {
        Stop-Process -Id $p.Id -Force
        Write-Host "`nqwen36-server остановлен."
    }
}