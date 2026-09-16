$ErrorActionPreference = 'Continue'
$root = 'D:\Projects\yttri-inference'
$envFile = "$root\qwen36-server\.env"
$line = (Select-String -Path $envFile -Pattern 'API_KEYS=' | Select-Object -First 1).Line
$v = $line.Substring($line.IndexOf('API_KEYS=') + 9)
$k = [regex]::Match($v, '"key"\s*:\s*"([^"]+)"').Groups[1].Value
$url = 'http://127.0.0.1:18099/v1/chat/completions'
$sb = New-Object System.Text.StringBuilder
for ($i = 0; $i -lt 6264; $i++) { [void]$sb.Append("word$i ") }
$prompt = $sb.ToString() + 'Count from 1 to 5.'
$b = @{model='ornith-1.5-9b';stream=$false;max_tokens=1;messages=@(@{role='user';content=$prompt});temperature=0} | ConvertTo-Json -Depth 6 -Compress
[System.IO.File]::WriteAllText("$env:TEMP\t3.json", $b, [System.Text.UTF8Encoding]::new($false))
foreach ($tile in @(0,1)) {
  Set-Content -Path "$root\bench-overrides.env" -Value @("QWEN36_FA_PREFILL_TILE=$tile") -Encoding ascii
  Stop-ScheduledTask -TaskName qwen36-inference -EA SilentlyContinue | Out-Null
  Start-Sleep 2
  Get-Process yforge -EA SilentlyContinue | Stop-Process -Force -EA SilentlyContinue
  Start-Sleep 3
  Start-ScheduledTask -TaskName qwen36-inference
  for ($i = 1; $i -le 60; $i++) {
    try { $bb = @{model='ornith-1.5-9b';stream=$false;max_tokens=1;messages=@(@{role='user';content='hi'})} | ConvertTo-Json -Depth 5 -Compress
      Invoke-RestMethod $url -Headers @{Authorization="Bearer $k"} -Method Post -ContentType 'application/json' -Body $bb -TimeoutSec 60 | Out-Null; break } catch { Start-Sleep -Seconds 3 }
  }
  Start-Sleep 2
  $times = @()
  foreach ($r in 1..3) {
    $best = 999.0
    foreach ($rr in 1..1) {
      $t = [double](curl.exe -s -X POST $url -H "Content-Type: application/json" -H "Authorization: Bearer $k" -d "@$env:TEMP\t3.json" -o "$env:TEMP\t3.out" -w "%{time_total}")
      if ($t -lt $best) { $best = $t }
    }
    $times += [math]::Round($best,2)
  }
  $avg = ($times | Measure-Object -Average).Average
  Write-Output ("TILE3[$tile] 30k-prefill runs=[$($times -join ',')] avg=$([math]::Round($avg,2))s = $([math]::Round(30228/$avg,0)) t/s")
}
Remove-Item "$root\bench-overrides.env" -EA SilentlyContinue
Write-Output 'done'
