#Requires -Version 5
<#
phase0_llamacpp.ps1 - Phase 0: замер llama.cpp против candle (BD-001 re-eval).

Поднимает llama-server на Qwen3.6-35B-A3B с выгрузкой экспертов в RAM
(-ncmoe) и опционально прогоняет bench.ps1 для сравнения с нашим сервером.

Usage (on yttri-win):
    powershell -ExecutionPolicy Bypass -File scripts\phase0_llamacpp.ps1 `
        -ModelPath "D:\Models\yttri\Qwen3.6-35B-A3B\Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf" `
        -DownloadLlama -RunBench

Скачивание модели (если ещё нет):
    hf download unsloth/Qwen3.6-35B-A3B-GGUF --local-dir D:\Models\yttri\Qwen3.6-35B-A3B `
        --include "*UD-Q2_K_XL*"
#>
[CmdletBinding()]
param(
    [string]$ModelPath  = "D:\Models\unsloth\Qwen3.6-35B-A3B-GGUF\Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf",
    [string]$LlamaDir   = "D:\Projects\yttri-inference\llama.cpp-bin",
    # CUDA-рантайм для сборки cuda124 (в SSH-сессии его нет в PATH)
    [string]$CudaBin    = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin",
    # 999 = все MoE-эксперты в RAM; меньше = часть экспертов остаётся на GPU.
    # Сладспот обычно между: попробуй 999, затем ~28, ~20 из 40 блоков.
    [int]   $Ncmoe      = 999,
    [int]   $Ctx        = 8192,
    [int]   $Parallel   = 2,
    [int]   $Port       = 8081,
    [string]$ApiKey     = "smoke-key",
    # Доп. флаги llama-server, напр. для MTP-GGUF:
    #   --spec-type draft-mtp --spec-draft-n-max 2
    [string[]]$ExtraArgs = @(),
    [switch]$DownloadLlama,
    [switch]$RunBench,
    [switch]$KeepServer
)

$ErrorActionPreference = 'Stop'
if (Test-Path $CudaBin) { $env:PATH = "$CudaBin;$env:PATH" }

# ── 0. Система ────────────────────────────────────────────────────────────────
$ramGiB = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)
Write-Host "=== Phase 0: llama.cpp vs candle ==="
Write-Host ("RAM total: {0} GiB (для -ncmoe желательно >= 32)" -f $ramGiB)
$vramLines = @(& nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits 2>$null)
if ($LASTEXITCODE -eq 0 -and $vramLines.Count -gt 0) {
    $p = ($vramLines[0] -split ',')
    if ($p.Count -ge 2) {
        Write-Host ("VRAM: {0} / {1} MiB used at idle" -f $p[1].Trim(), $p[0].Trim())
    }
}

# ── 1. Бинарники llama.cpp ────────────────────────────────────────────────────
$serverExe = Join-Path $LlamaDir "llama-server.exe"
if (-not (Test-Path $serverExe)) {
    if (-not $DownloadLlama) {
        Write-Host "FATAL: $serverExe не найден. Запусти с -DownloadLlama." -ForegroundColor Red
        exit 1
    }
    Write-Host "llama-server не найден, качаю последний CUDA-release..."
    $hdr = @{ "User-Agent" = "phase0-bench" }
    $rel = Invoke-RestMethod -Uri "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest" -Headers $hdr
    $asset = $rel.assets | Where-Object { $_.name -match "win-cuda.*x64\.zip$" } | Select-Object -First 1
    if (-not $asset) { throw "Не нашёл win-cuda asset в release $($rel.tag_name)" }
    $zip = Join-Path $env:TEMP $asset.name
    Write-Host ("  {0} ({1:N0} MB)..." -f $asset.name, ($asset.size / 1MB))
    Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $zip -Headers $hdr
    New-Item -ItemType Directory -Force -Path $LlamaDir | Out-Null
    Expand-Archive -Path $zip -DestinationPath $LlamaDir -Force
    Remove-Item $zip
    if (-not (Test-Path $serverExe)) {
        # Некоторые релизы кладут exe в подпапку — поднимаем на верхний уровень.
        $nested = Get-ChildItem $LlamaDir -Recurse -Filter "llama-server.exe" | Select-Object -First 1
        if ($nested) { Copy-Item $nested.FullName $LlamaDir }
    }
}
if (-not (Test-Path $serverExe)) { throw "llama-server.exe так и не найден в $LlamaDir" }
Write-Host ("llama-server: {0}" -f $serverExe)

# ── 2. Модель ─────────────────────────────────────────────────────────────────
if (-not (Test-Path $ModelPath)) {
    Write-Host "FATAL: модель не найдена: $ModelPath" -ForegroundColor Red
    Write-Host "Скачай: hf download unsloth/Qwen3.6-35B-A3B-GGUF --local-dir <dir> --include `"*UD-Q2_K_XL*`""
    exit 1
}
$alias = [IO.Path]::GetFileNameWithoutExtension($ModelPath).ToLower()
Write-Host ("model: {0} ({1:N1} GB)" -f $ModelPath, ((Get-Item $ModelPath).Length / 1GB))

# ── 3. Запуск llama-server ────────────────────────────────────────────────────
$log = Join-Path $env:TEMP "phase0-llama-$Port.log"
$sargs = @(
    "-m", $ModelPath,
    "--alias", $alias,
    "-ngl", "99",            # всё кроме экспертов — на GPU
    "-ncmoe", "$Ncmoe",      # MoE FFN первых N слоёв — в RAM (mmap)
    "-c", "$Ctx",
    "-np", "$Parallel",
    "-fa", "on",
    "--jinja",
    "--api-key", $ApiKey,
    "--host", "127.0.0.1",
    "--port", "$Port"
) + $ExtraArgs
Write-Host ""
Write-Host ("start: llama-server {0}" -f ($sargs -join ' '))
$proc = Start-Process -FilePath $serverExe -ArgumentList $sargs `
    -RedirectStandardOutput $log -RedirectStandardError "$log.err" -PassThru

try {
    # ── 4. Ожидание загрузки (10.8 GB c диска может занять несколько минут) ────
    Write-Host "waiting for /health..." -NoNewline
    $deadline = (Get-Date).AddMinutes(15)
    while ((Get-Date) -lt $deadline) {
        if ($proc.HasExited) {
            Write-Host " FATAL: процесс упал, лог:" -ForegroundColor Red
            Get-Content "$log.err" -Tail 30; exit 1
        }
        try {
            $h = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 2
            if ($h.status -in @("ok")) { break }
        } catch {}
        Write-Host "." -NoNewline; Start-Sleep 5
    }
    if ((Get-Date) -ge $deadline) { Write-Host " TIMEOUT" -ForegroundColor Red; exit 1 }
    Write-Host " ok"

    $gs = Get-Process -Id $proc.Id
    Write-Host ("llama-server RAM working set: {0:N1} GiB" -f ($gs.WorkingSet64 / 1GB))

    # ── 5. Бенч (тот же протокол, что и для нашего сервера) ────────────────────
    if ($RunBench) {
        & (Join-Path $PSScriptRoot "bench.ps1") `
            -BaseUrl "http://localhost:$Port" -ApiKey $ApiKey -Model $alias `
            -Concurrent $Parallel
    } else {
        Write-Host ""
        Write-Host "Теперь прогони бенч:"
        Write-Host ("  powershell -ExecutionPolicy Bypass -File scripts\bench.ps1 -BaseUrl `"http://localhost:{0}`" -ApiKey `"{1}`" -Model `"{2}`" -Concurrent {3}" -f $Port, $ApiKey, $alias, $Parallel)
        Write-Host "Или перезапусти этот скрипт с -RunBench."
    }

    # RAM после нагрузки
    $gs = Get-Process -Id $proc.Id -ErrorAction SilentlyContinue
    if ($gs) { Write-Host ("llama-server RAM after bench: {0:N1} GiB" -f ($gs.WorkingSet64 / 1GB)) }
} finally {
    if (-not $KeepServer -and $proc -and -not $proc.HasExited) {
        Stop-Process -Id $proc.Id -Force
        Write-Host "`nllama-server остановлен (лог: $log)"
    } elseif ($KeepServer) {
        Write-Host "`nllama-server оставлен работать на порту $Port (PID $($proc.Id))."
    }
}