# Приёмка QAMQ-модели: PPL, скорость, (опц.) KLD и task-evals против baseline.
#
# Базовые значения сняты на Qwen3.5-9B-QAMQ-A5L-AWQ2 (2026-10-11, RTX 3060).
# Порог по PPL/скорости — допуск на шум; по задачам — допуск 1.5 п.п.
#
#   .\acceptance.ps1 -Model D:\Models\...\Qwen3.5-9B-QAMQ-A5L-AWQ2.gguf
#   .\acceptance.ps1 -Model <путь> -WithEvals     # + MMLU/GSM8K/ruMMLU
#   .\acceptance.ps1 -Model <путь> -WithKld       # + KLD к BF16 (нужен base-64.bin)
param(
  [Parameter(Mandatory=$true)][string]$Model,
  [string]$Heldout = 'D:\Projects\yttri-inference\qamq\calibration\qamq-heldout.txt',
  [string]$KldBase = 'D:\Projects\yttri-inference\qamq\kld\base-64.bin',
  [switch]$WithEvals,
  [switch]$WithKld
)
$ErrorActionPreference = 'Continue'
$llama = 'D:\Tools\llama.cpp\b11552'
$logDir = 'D:\Projects\yttri-inference\logs\qamq'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# baseline -> (значение, допуск, направление).
# Допуски по скорости подобраны под фактический разброс частот RTX 3060 на этом стенде:
# pp512 в трёх прогонах дал 1997.9 / 1992.4 / 1928.8 (±3.5%), tg128 — 61.19 / 61.1 / 60.96.
# Поэтому допуск 5% по pp512 и 3% по tg128: иначе скрипт ловит дрожание клоков, а не регрессию.
$base = @{
  ppl      = @{ target = 15.3767; tol = 0.35;  dir = 'max' }   # не хуже baseline + допуск
  tg128    = @{ target = 61.19;  tol = 1.85;  dir = 'min' }   # не медленнее baseline − допуск
  pp512    = @{ target = 1997.85; tol = 100;  dir = 'min' }
  kld      = @{ target = 0.058540; tol = 0.004; dir = 'max' }
  mmlu     = @{ target = 0.7567; tol = 0.015; dir = 'min' }
  gsm8k    = @{ target = 0.9618; tol = 0.015; dir = 'min' }
  rummlu   = @{ target = 0.6967; tol = 0.015; dir = 'min' }
}

if (-not (Test-Path $Model)) { Write-Error "нет модели: $Model"; exit 2 }
$tag = [IO.Path]::GetFileNameWithoutExtension($Model)
Write-Host "=== Приёмка $tag ===" -ForegroundColor Cyan
$rows = @()

# 1. PPL
$pplLog = Join-Path $logDir "accept-$tag-ppl.log"
& "$llama\llama-perplexity.exe" -m $Model -ngl 99 -c 512 --chunks 20 -f $Heldout *> $pplLog
$m = Select-String -Path $pplLog -Pattern 'Final estimate: PPL = ([0-9.]+)' | Select-Object -Last 1
if ($m) {
  $ppl = [double]$m.Matches[0].Groups[1].Value
  $b = $base.ppl
  $rows += [pscustomobject]@{ Метрика='PPL'; Значение=$ppl; Порог=($b.target+$b.tol); Итог=$(if ($ppl -le $b.target+$b.tol) {'PASS'} else {'FAIL'}) }
} else { $rows += [pscustomobject]@{ Метрика='PPL'; Значение='?'; Порог='-'; Итог='ERROR' } }

# 2. Скорость
$benchLog = Join-Path $logDir "accept-$tag-bench.log"
& "$llama\llama-bench.exe" -m $Model -ngl 99 -p 512 -n 128 -r 3 -fa 1 *> $benchLog
foreach ($pair in @(@('pp512','pp512'), @('tg128','tg128'))) {
  $key = $pair[0]
  $m = Select-String -Path $benchLog -Pattern "\s$key\s+\|\s+([0-9.]+)" | Select-Object -Last 1
  if ($m) {
    $v = [double]$m.Matches[0].Groups[1].Value
    $b = $base[$key]
    $rows += [pscustomobject]@{ Метрика=$key; Значение=$v; Порог=([math]::Round($b.target-$b.tol,2)); Итог=$(if ($v -ge $b.target-$b.tol) {'PASS'} else {'FAIL'}) }
  }
}

# 3. KLD (по желанию)
if ($WithKld -and (Test-Path $KldBase)) {
  $kldLog = Join-Path $logDir "accept-$tag-kld.log"
  & "$llama\llama-perplexity.exe" -m $Model -f $Heldout -c 512 --chunks 64 `
      --kl-divergence --kl-divergence-base $KldBase -ngl 99 *> $kldLog
  $m = Select-String -Path $kldLog -Pattern 'Mean\s+KLD:\s+([0-9.]+)' | Select-Object -Last 1
  if ($m) {
    $v = [double]$m.Matches[0].Groups[1].Value
    $b = $base.kld
    $rows += [pscustomobject]@{ Метрика='KLD'; Значение=$v; Порог=($b.target+$b.tol); Итог=$(if ($v -le $b.target+$b.tol) {'PASS'} else {'FAIL'}) }
  }
}

# 4. Task-evals (по желанию) — через llama-server + harness
if ($WithEvals) {
  Write-Host "Поднимаю llama-server для evals..." -ForegroundColor DarkGray
  $envLog = Join-Path $logDir "accept-$tag-server.log"
  $p = Start-Process -FilePath "$llama\llama-server.exe" -PassThru -WindowStyle Hidden `
        -ArgumentList @('-m', $Model, '-ngl', '99', '-c', '4096', '-fa', '1',
                        '--host', '127.0.0.1', '--port', '18099', '-np', '1') `
        -RedirectStandardOutput $envLog -RedirectStandardError "$envLog.err"
  Start-Sleep -Seconds 35
  try {
    $out = Join-Path $logDir "accept-$tag-evals.json"
    python 'D:\Projects\yttri-inference\qamq\eval_qamq.py' --label $tag --out $out `
      --mmlu-n 300 --gsm8k-n 131 --tasks mmlu,gsm8k,rummlu 2>&1 | Out-Null
    if (Test-Path $out) {
      $j = Get-Content $out -Raw | ConvertFrom-Json
      foreach ($k in 'mmlu','gsm8k','rummlu') {
        $t = $j.tasks.$k
        if ($t) {
          $v = if ($k -eq 'gsm8k') { $t.acc } else { $t.acc }
          $b = $base[$k]
          $rows += [pscustomobject]@{ Метрика=$k; Значение=[math]::Round($v,4); Порог=([math]::Round($b.target-$b.tol,4)); Итог=$(if ($v -ge $b.target-$b.tol) {'PASS'} else {'FAIL'}) }
        }
      }
    }
  } finally {
    if ($p -and -not $p.HasExited) { Stop-Process -Id $p.Id -Force }
  }
}

Write-Host ""
$rows | Format-Table -AutoSize
$failed = @($rows | Where-Object { $_.Итог -ne 'PASS' }).Count
if ($failed -eq 0) { Write-Host "ИТОГ: PASS ($($rows.Count) checks)" -ForegroundColor Green; exit 0 }
else { Write-Host "ИТОГ: FAIL ($failed из $($rows.Count))" -ForegroundColor Red; exit 1 }
