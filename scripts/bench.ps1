#Requires -Version 5
<#
bench.ps1 - qwen36-server performance benchmark.
Measures: TTFT, decode tok/s (single), prefill tok/s, N-concurrent throughput, VRAM.

Usage (on yttri-win):
    powershell -ExecutionPolicy Bypass -File bench.ps1 -BaseUrl "http://localhost:18099" -ApiKey "smoke-key"
#>
[CmdletBinding()]
param(
    [string]$BaseUrl = "http://localhost:18099",
    [string]$ApiKey  = "smoke-key",
    [string]$Model   = "qwen3.6-27b",
    [int]   $DecodeTokens = 128,
    [int]   $Concurrent   = 4,
    [switch]$NoWarmup
)

$Headers = @{
    Authorization = "Bearer $ApiKey"
    "Content-Type" = "application/json"
}

function Get-VRAM {
    $raw = & nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>$null
    if ($LASTEXITCODE -eq 0 -and $raw) {
        $parts = $raw.Trim() -split ','
        return @{
            used_mib   = [int]$parts[0].Trim()
            total_mib  = [int]$parts[1].Trim()
            gpu_util   = [int]$parts[2].Trim()
        }
    }
    return $null
}

function Measure-Decode {
    $body = @{
        model       = $Model
        stream      = $true
        max_tokens  = $DecodeTokens
        temperature = 0.0
        messages    = @(@{ role = "user"; content = "Write a story about a robot learning to paint." })
    } | ConvertTo-Json -Depth 5 -Compress

    $url = "$BaseUrl/v1/chat/completions"
    $req = [System.Net.HttpWebRequest]::Create($url)
    $req.Method = "POST"
    $req.ContentType = "application/json"
    $req.Headers.Add("Authorization", "Bearer $ApiKey")
    $req.Timeout = 600000
    $req.ReadWriteTimeout = 600000

    $bytes = [System.Text.Encoding]::UTF8.GetBytes($body)
    $req.ContentLength = $bytes.Length
    $stream = $req.GetRequestStream()
    $stream.Write($bytes, 0, $bytes.Length)
    $stream.Close()

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $resp = $req.GetResponse()
    $reader = New-Object System.IO.StreamReader($resp.GetResponseStream())

    $ttft = $null
    $tokCount = 0

    while (-not $reader.EndOfStream) {
        $line = $reader.ReadLine()
        if ($line -match '^data: ') {
            $json = $line.Substring(6).Trim()
            if ($json -eq '[DONE]') { continue }
            try {
                $obj = $json | ConvertFrom-Json
                if ($obj.choices -and $obj.choices[0].delta -and $obj.choices[0].delta.content) {
                    if (-not $ttft) { $ttft = $sw.Elapsed.TotalSeconds }
                    $tokCount++
                }
            } catch {}
        }
    }
    $sw.Stop()
    $reader.Close()
    $resp.Close()

    $totalSec = $sw.Elapsed.TotalSeconds
    $decodeTps = 0
    if ($tokCount -gt 0 -and $ttft -and ($totalSec - $ttft) -gt 0) {
        $decodeTps = [math]::Round($tokCount / ($totalSec - $ttft), 1)
    }
    $ttftMs = 0
    if ($ttft) { $ttftMs = [math]::Round($ttft * 1000) }

    return @{
        ttft_ms    = $ttftMs
        tokens     = $tokCount
        total_s    = [math]::Round($totalSec, 2)
        decode_tps = $decodeTps
    }
}

function Measure-Prefill {
    $longText = "The quick brown fox jumps over the lazy dog. " * 200
    $body = @{
        model       = $Model
        stream      = $false
        max_tokens  = 8
        temperature = 0.0
        messages    = @(@{ role = "user"; content = $longText })
    } | ConvertTo-Json -Depth 5 -Compress

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $resp = Invoke-RestMethod -Uri "$BaseUrl/v1/chat/completions" -Method Post `
        -Headers $Headers -Body $body -TimeoutSec 300
    $sw.Stop()

    $promptTokens = $resp.usage.prompt_tokens
    $totalSec = $sw.Elapsed.TotalSeconds
    $prefillTps = if ($totalSec -gt 0 -and $promptTokens -gt 0) { [math]::Round($promptTokens / $totalSec, 1) } else { 0 }

    return @{
        prompt_tokens = $promptTokens
        total_s       = [math]::Round($totalSec, 2)
        prefill_tps   = $prefillTps
    }
}

function Measure-Concurrent {
    param([int]$N = $Concurrent)
    $runspaces = @()
    $results = @()

    for ($i = 0; $i -lt $N; $i++) {
        $body = @{
            model       = $Model
            stream      = $false
            max_tokens  = 64
            temperature = 0.3
            messages    = @(@{ role = "user"; content = "Count from 1 to $(50 + $i). Numbers only." })
        } | ConvertTo-Json -Depth 5 -Compress

        $ps = [powershell]::Create().AddScript({
            param($body, $BaseUrl, $ApiKey)
            $hdrs = @{ Authorization = "Bearer $ApiKey"; "Content-Type" = "application/json" }
            $sw = [System.Diagnostics.Stopwatch]::StartNew()
            $resp = Invoke-RestMethod -Uri "$BaseUrl/v1/chat/completions" -Method Post `
                -Headers $hdrs -Body $body -TimeoutSec 300
            $sw.Stop()
            return @{
                tokens  = $resp.usage.completion_tokens
                wall_s  = [math]::Round($sw.Elapsed.TotalSeconds, 2)
            }
        }).AddArgument($body).AddArgument($BaseUrl).AddArgument($ApiKey)

        $runspaces += [PSCustomObject]@{
            PS      = $ps
            Handle  = $ps.BeginInvoke()
        }
    }

    foreach ($rs in $runspaces) {
        $results += $rs.PS.EndInvoke($rs.Handle)
        $rs.PS.Dispose()
    }

    $totalTokens = 0
    foreach ($r in $results) { $totalTokens += $r.tokens }
    $maxWall = 0
    foreach ($r in $results) { if ($r.wall_s -gt $maxWall) { $maxWall = $r.wall_s } }
    $aggregateTps = if ($maxWall -gt 0) { [math]::Round($totalTokens / $maxWall, 1) } else { 0 }

    return @{
        clients       = $N
        total_tokens  = $totalTokens
        max_wall_s    = [math]::Round($maxWall, 2)
        aggregate_tps = $aggregateTps
        per_slot_tps  = [math]::Round($aggregateTps / $N, 1)
    }
}

# ── Main ──────────────────────────────────────────────────────────────────────

Write-Host "=== qwen36-server benchmark ==="
Write-Host "host: $BaseUrl  model: $Model"
Write-Host ""

# Preflight
try {
    $models = Invoke-RestMethod -Uri "$BaseUrl/v1/models" -Headers $Headers -TimeoutSec 10
    Write-Host ("model: {0}  slots: {1}  ctx: {2}  quant: {3}" -f `
        $models.data[0].id, $models.data[0].slots, $models.data[0].context_length, $models.data[0].quant)
} catch {
    Write-Host "FATAL: server not reachable at $BaseUrl" -ForegroundColor Red
    exit 1
}

$vramBase = Get-VRAM
if ($vramBase) {
    Write-Host ("VRAM idle: {0} / {1} MiB  GPU util: {2}%" -f `
        $vramBase.used_mib, $vramBase.total_mib, $vramBase.gpu_util)
}
Write-Host ""

if (-not $NoWarmup) {
    Write-Host "warmup..." -NoNewline
    $warmBody = @{
        model      = $Model
        stream     = $false
        max_tokens = 16
        messages   = @(@{ role = "user"; content = "hello" })
    } | ConvertTo-Json -Depth 5
    Invoke-RestMethod -Uri "$BaseUrl/v1/chat/completions" -Method Post `
        -Headers $Headers -Body $warmBody -TimeoutSec 60 | Out-Null
    Write-Host "done"
    Write-Host ""
}

# 1. Single-slot decode
Write-Host "-- 1. Single-slot decode (max_tokens=$DecodeTokens, stream) --"
$d = Measure-Decode
Write-Host ("    TTFT:       {0} ms" -f $d.ttft_ms)
Write-Host ("    tokens:     {0}" -f $d.tokens)
Write-Host ("    total:      {0} s" -f $d.total_s)
Write-Host ("    decode:     {0} tok/s" -f $d.decode_tps)
Write-Host ""

Write-Host "-- 2. Prefill (long prompt, short output) --"
$p = Measure-Prefill
Write-Host ("    prompt:     {0} tokens" -f $p.prompt_tokens)
Write-Host ("    total:      {0} s" -f $p.total_s)
Write-Host ("    prefill:    {0} tok/s" -f $p.prefill_tps)
Write-Host ""

Write-Host "-- 3. Concurrent ($Concurrent slots, max_tokens=64 each) --"
$c = Measure-Concurrent
Write-Host ("    clients:    {0}" -f $c.clients)
Write-Host ("    total toks: {0}" -f $c.total_tokens)
Write-Host ("    max wall:   {0} s" -f $c.max_wall_s)
Write-Host ("    aggregate:  {0} tok/s" -f $c.aggregate_tps)
Write-Host ("    per slot:   {0} tok/s" -f $c.per_slot_tps)
Write-Host ""

$vramPeak = Get-VRAM
if ($vramPeak) {
    Write-Host "-- 4. VRAM after benchmark --"
    Write-Host ("    used:       {0} / {1} MiB" -f $vramPeak.used_mib, $vramPeak.total_mib)
    Write-Host ("    GPU util:   {0}%" -f $vramPeak.gpu_util)
    if ($vramBase) {
        $delta = $vramPeak.used_mib - $vramBase.used_mib
        Write-Host ("    delta:      {0} MiB" -f $delta)
    }
}
Write-Host ""

Write-Host "=== Summary ==="
$vramStr = if ($vramPeak) { "$($vramPeak.used_mib)MiB" } else { "n/a" }
Write-Host ("TTFT={0}ms  decode={1}tok/s  prefill={2}tok/s  {3}slot={4}tok/s (per={5})  VRAM={6}" -f `
    $d.ttft_ms, $d.decode_tps, $p.prefill_tps, $Concurrent, $c.aggregate_tps, $c.per_slot_tps, $vramStr)
