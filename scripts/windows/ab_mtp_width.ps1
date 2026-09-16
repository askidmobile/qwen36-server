$ErrorActionPreference = 'Continue'
$root = 'D:\Projects\yttri-inference'
$envFile = "$root\qwen36-server\.env"
$line = (Select-String -Path $envFile -Pattern 'API_KEYS=' | Select-Object -First 1).Line
$v = $line.Substring($line.IndexOf('API_KEYS=') + 9)
$k = [regex]::Match($v, '"key"\s*:\s*"([^"]+)"').Groups[1].Value
$url = 'http://127.0.0.1:18099/v1/chat/completions'
$prose = 'Write a long detailed story about a robot learning to paint. Steps in order, no lists.'
$code = "def fib(n):`n    if n < 2:`n        return n`n    return fib(n-1) + fib(n-2)`n`nContinue this file with a memoized fib and a test runner. Output only code."
function Run([string]$tag, [string]$prompt, [int]$n) {
  $b = @{model='ornith-1.5-9b';stream=$false;max_tokens=$n;messages=@(@{role='user';content=$prompt});temperature=0.6;top_p=0.95;top_k=20} | ConvertTo-Json -Depth 6 -Compress
  [System.IO.File]::WriteAllText("$env:TEMP\mw.json", $b, [System.Text.UTF8Encoding]::new($false))
  $t = [double](curl.exe -s -X POST $url -H "Content-Type: application/json" -H "Authorization: Bearer $k" -d "@$env:TEMP\mw.json" -o "$env:TEMP\mw.out" -w "%{time_total}")
  $o = $null; try { $o = Get-Content "$env:TEMP\mw.out" -Raw | ConvertFrom-Json } catch {}
  if (-not $o -or -not $o.usage) { Write-Output "$tag FAILED"; return }
  $m = $o.usage.mtp; $ct = $o.usage.completion_tokens
  $acc = 0.0; if ($m.drafted -gt 0) { $acc = [math]::Round(100.0*$m.accepted/$m.drafted,1) }
  Write-Output ("$tag tokens=$ct t=$([math]::Round($t,2))s tps=$([math]::Round($ct/$t,2)) drafted=$($m.drafted) accepted=$($m.accepted) acc=$acc%")
}
foreach ($cfg in @(@('mtp0',@()), @('w1',@('MTP=1','MTP_WIDTH=1')), @('w2',@('MTP=1','MTP_WIDTH=2')), @('w3',@('MTP=1','MTP_WIDTH=3')))) {
  if ($cfg[1].Count -gt 0) { Set-Content -Path "$root\bench-overrides.env" -Value $cfg[1] -Encoding ascii } else { Remove-Item "$root\bench-overrides.env" -EA SilentlyContinue }
  Stop-ScheduledTask -TaskName qwen36-inference -EA SilentlyContinue | Out-Null
  Start-Sleep 2
  Get-Process yforge -EA SilentlyContinue | Stop-Process -Force -EA SilentlyContinue
  Start-Sleep 3
  Start-ScheduledTask -TaskName qwen36-inference
  $ok=$false
  for ($i=1; $i -le 60; $i++) {
    try { $b = @{model='ornith-1.5-9b';stream=$false;max_tokens=1;messages=@(@{role='user';content='hi'})} | ConvertTo-Json -Depth 5 -Compress
      Invoke-RestMethod $url -Headers @{Authorization="Bearer $k"} -Method Post -ContentType 'application/json' -Body $b -TimeoutSec 60 | Out-Null; $ok=$true; break } catch { Start-Sleep -Seconds 3 }
  }
  if (-not $ok) { Write-Output "$($cfg[0]) NOT READY"; continue }
  Start-Sleep 2
  Run "MTP[$($cfg[0])-prose]" $prose 200
  Run "MTP[$($cfg[0])-code]" $code 200
}
Remove-Item "$root\bench-overrides.env" -EA SilentlyContinue
Stop-ScheduledTask -TaskName qwen36-inference -EA SilentlyContinue | Out-Null
Start-Sleep 1
Start-ScheduledTask -TaskName qwen36-inference
Write-Output 'done'
