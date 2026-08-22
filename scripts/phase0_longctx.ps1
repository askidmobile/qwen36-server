#Requires -Version 5
<#
phase0_longctx.ps1 - декод на ДЛИННОМ контексте (~6K токенов промпт),
где реальная работа деградирует. Запускается против уже поднятого сервера
(candle или llama-server) с готовой моделью.

Usage:
    powershell -File phase0_longctx.ps1 -BaseUrl http://localhost:18099 -ApiKey smoke-key -Model qwen3.6-35b-a3b
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$BaseUrl,
    [Parameter(Mandatory=$true)][string]$ApiKey,
    [Parameter(Mandatory=$true)][string]$Model,
    [int]$PromptTokensTarget = 6000,
    [int]$DecodeTokens = 128
)

$Headers = @{ Authorization = "Bearer $ApiKey" }

# ~4.5 байта на токен для этого текста
$unit = "The quick brown fox jumps over the lazy dog near the river bank every morning. "
$repeat = [math]::Ceiling($PromptTokensTarget * 4.5 / $unit.Length)
$longText = $unit * $repeat

function Measure-Stream([string]$prompt, [int]$maxTok) {
    $body = @{ model = $Model; stream = $true; max_tokens = $maxTok; temperature = 0.0;
               messages = @(@{ role = "user"; content = $prompt }) } | ConvertTo-Json -Depth 5 -Compress
    $req = [System.Net.HttpWebRequest]::Create("$BaseUrl/v1/chat/completions")
    $req.Method = "POST"; $req.ContentType = "application/json"
    $req.Headers.Add("Authorization", "Bearer $ApiKey")
    $req.Timeout = 900000; $req.ReadWriteTimeout = 900000
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($body)
    $req.ContentLength = $bytes.Length
    $s = $req.GetRequestStream(); $s.Write($bytes, 0, $bytes.Length); $s.Close()
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $resp = $req.GetResponse()
    $rd = New-Object IO.StreamReader($resp.GetResponseStream())
    $ttft = $null; $toks = 0
    while (-not $rd.EndOfStream) {
        $line = $rd.ReadLine()
        if ($line -match '^data: ') {
            $j = $line.Substring(6).Trim()
            if ($j -eq '[DONE]') { continue }
            try {
                $o = $j | ConvertFrom-Json
                if ($o.choices -and $o.choices[0].delta) {
                    $d = $o.choices[0].delta
                    if ($d.content -or $d.reasoning_content) {
                        if (-not $ttft) { $ttft = $sw.Elapsed.TotalSeconds }
                        $toks++
                    }
                }
            } catch {}
        }
    }
    $sw.Stop(); $rd.Close(); $resp.Close()
    $genSec = $sw.Elapsed.TotalSeconds - $ttft
    return @{ ttft_s = [math]::Round($ttft, 2); tokens = $toks;
              total_s = [math]::Round($sw.Elapsed.TotalSeconds, 2);
              decode_tps = if ($genSec -gt 0) { [math]::Round($toks / $genSec, 1) } else { 0 } }
}

Write-Host "=== long-context probe: ~$PromptTokensTarget tok prompt -> decode $DecodeTokens ==="

# 1. Prefill длинного промпта (кэшируем нечего, каждый запуск с нуля)
$r1 = Measure-Stream $longText 1
Write-Host ("prefill {0} tok за {1}s" -f $r1.tokens, $r1.total_s)

# 2. Тот же промпт ещё раз + длинная генерация: декод при полном KV
$r2 = Measure-Stream $longText $DecodeTokens
Write-Host ("decode @ctx~{0}K: TTFT={1}s tokens={2} total={3}s decode={4} tok/s" -f `
    $PromptTokensTarget, $r2.ttft_s, $r2.tokens, $r2.total_s, $r2.decode_tps)

$vram = @(& nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>$null)
if ($LASTEXITCODE -eq 0 -and $vram.Count -gt 0) {
    Write-Host ("VRAM during: {0}" -f $vram[0])
}
Write-Host ("SUMMARY-LONGCTX: prefill_ttft={0}s decode={1}tok/s VRAM={2}" -f $r2.ttft_s, $r2.decode_tps, $vram[0])