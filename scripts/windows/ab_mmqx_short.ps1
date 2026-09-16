$ErrorActionPreference = 'Continue'
$root = 'D:\Projects\yttri-inference'
$envFile = "$root\qwen36-server\.env"
$line = (Select-String -Path $envFile -Pattern 'API_KEYS=' | Select-Object -First 1).Line
$v = $line.Substring($line.IndexOf('API_KEYS=') + 9)
$k = [regex]::Match($v, '"key"\s*:\s*"([^"]+)"').Groups[1].Value
$url = 'http://127.0.0.1:18099/v1/chat/completions'
$sb = New-Object System.Text.StringBuilder
for ($i=0; $i -lt 52; $i++) { [void]$sb.Append("word$i ") }
$prompt = $sb.ToString() + 'Count from 1 to 5.'
$b = @{model='ornith-1.5-9b';stream=$false;max_tokens=1;messages=@(@{role='user';content=$prompt});temperature=0} | ConvertTo-Json -Depth 6 -Compress
[System.IO.File]::WriteAllText("$env:TEMP\x.json", $b, [System.Text.UTF8Encoding]::new($false))
foreach ($x in @('128','64','32')) {
  Set-Content -Path "$root\bench-overrides.env" -Value @("MMQ_X=$x") -Encoding ascii
  Stop-ScheduledTask -TaskName qwen36-inference -EA SilentlyContinue | Out-Null
  Start-Sleep 2
  Get-Process yforge -EA SilentlyContinue | Stop-Process -Force -EA SilentlyContinue
  Start-Sleep 3
  Start-ScheduledTask -TaskName qwen36-inference
  for ($i=1; $i -le 60; $i++) {
    try { $bb = @{model='ornith-1.5-9b';stream=$false;max_tokens=1;messages=@(@{role='user';content='hi'})} | ConvertTo-Json -Depth 5 -Compress
      Invoke-RestMethod $url -Headers @{Authorization="Bearer $k"} -Method Post -ContentType 'application/json' -Body $bb -TimeoutSec 60 | Out-Null; break } catch { Start-Sleep -Seconds 3 }
  }
  Start-Sleep 2
  $times = @()
  foreach ($r in 1..4) {
    $t = [double](curl.exe -s -X POST $url -H "Content-Type: application/json" -H "Authorization: Bearer $k" -d "@$env:TEMP\x.json" -o "$env:TEMP\x.out" -w "%{time_total}")
    $times += [math]::Round($t,3)
  }
  $avg = ($times | Measure-Object -Average).Average
  Write-Output ("MMQX[$x] 164tok prefill runs=[$($times -join ',')] avg=$([math]::Round($avg,4))s = $([math]::Round(164/$avg,0)) t/s")
}
Remove-Item "$root\bench-overrides.env" -EA SilentlyContinue
Stop-ScheduledTask -TaskName qwen36-inference -EA SilentlyContinue | Out-Null
Start-Sleep 1
Start-ScheduledTask -TaskName qwen36-inference
Write-Output 'done'
