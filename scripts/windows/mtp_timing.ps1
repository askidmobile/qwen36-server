param([string]$Kind = 'code')
$ErrorActionPreference = 'Continue'
$root = 'D:\Projects\yttri-inference'
$envFile = "$root\qwen36-server\.env"
$line = (Select-String -Path $envFile -Pattern 'API_KEYS=' | Select-Object -First 1).Line
$v = $line.Substring($line.IndexOf('API_KEYS=') + 9)
$k = [regex]::Match($v, '"key"\s*:\s*"([^"]+)"').Groups[1].Value
$url = 'http://127.0.0.1:18099/v1/chat/completions'
Set-Content -Path "$root\bench-overrides.env" -Value @('MTP=1','MTP_WIDTH=2','MTP_TIMING=1') -Encoding ascii
Stop-ScheduledTask -TaskName qwen36-inference -EA SilentlyContinue | Out-Null
Start-Sleep 2
Get-Process yforge -EA SilentlyContinue | Stop-Process -Force -EA SilentlyContinue
Start-Sleep 3
$mark = (Get-Content "$root\logs\server.log" | Measure-Object -Line).Lines
Start-ScheduledTask -TaskName qwen36-inference
for ($i=1; $i -le 60; $i++) {
  try { $b = @{model='ornith-1.5-9b';stream=$false;max_tokens=1;messages=@(@{role='user';content='hi'})} | ConvertTo-Json -Depth 5 -Compress
    Invoke-RestMethod $url -Headers @{Authorization="Bearer $k"} -Method Post -ContentType 'application/json' -Body $b -TimeoutSec 60 | Out-Null; break } catch { Start-Sleep -Seconds 3 }
}
if ($Kind -eq 'code') {
  $p = "def fib(n):`n    if n < 2:`n        return n`n    return fib(n-1) + fib(n-2)`n`nContinue this file with a memoized fib and a test runner. Output only code."
} else {
  $p = 'Write a long detailed story about a robot learning to paint.'
}
$b2 = @{model='ornith-1.5-9b';stream=$false;max_tokens=120;messages=@(@{role='user';content=$p});temperature=0.6;top_p=0.95;top_k=20} | ConvertTo-Json -Depth 6 -Compress
[System.IO.File]::WriteAllText("$env:TEMP\mt.json", $b2, [System.Text.UTF8Encoding]::new($false))
$t = [double](curl.exe -s -X POST $url -H "Content-Type: application/json" -H "Authorization: Bearer $k" -d "@$env:TEMP\mt.json" -o "$env:TEMP\mt.out" -w "%{time_total}")
$o = $null; try { $o = Get-Content "$env:TEMP\mt.out" -Raw | ConvertFrom-Json } catch {}
Write-Output "MTP[$Kind] tokens=$($o.usage.completion_tokens) t=$([math]::Round($t,2))s tps=$([math]::Round($o.usage.completion_tokens/$t,2)) drafted=$($o.usage.mtp.drafted) accepted=$($o.usage.mtp.accepted)"
Write-Output '--- timing lines ---'
Get-Content "$root\logs\server.log" | Select-Object -Skip $mark | Where-Object { $_ -match '^\[mtp|^\[spec|round' } | Select-Object -First 12
Remove-Item "$root\bench-overrides.env" -EA SilentlyContinue
Stop-ScheduledTask -TaskName qwen36-inference -EA SilentlyContinue | Out-Null
Start-Sleep 1
Start-ScheduledTask -TaskName qwen36-inference
