$ErrorActionPreference = 'Continue'
$root = 'D:\Projects\yttri-inference'
$envFile = "$root\qwen36-server\.env"
$line = (Select-String -Path $envFile -Pattern 'API_KEYS=' | Select-Object -First 1).Line
$v = $line.Substring($line.IndexOf('API_KEYS=') + 9)
$k = [regex]::Match($v, '"key"\s*:\s*"([^"]+)"').Groups[1].Value
$url = 'http://127.0.0.1:18099/v1/chat/completions'
function Prompt([int]$w) { $sb = New-Object System.Text.StringBuilder; for ($i=0; $i -lt $w; $i++) { [void]$sb.Append("word$i ") }; return $sb.ToString() + 'Count from 1 to 5.' }
foreach ($cfg in @(@('pcache8k',@()), @('pcache0',@('PREFIX_CACHE_MIB=0')))) {
  if ($cfg[1].Count -gt 0) { Set-Content -Path "$root\bench-overrides.env" -Value $cfg[1] -Encoding ascii } else { Remove-Item "$root\bench-overrides.env" -EA SilentlyContinue }
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
  foreach ($w in @(52, 131)) {
    $b = @{model='ornith-1.5-9b';stream=$false;max_tokens=1;messages=@(@{role='user';content=(Prompt $w)});temperature=0} | ConvertTo-Json -Depth 6 -Compress
    [System.IO.File]::WriteAllText("$env:TEMP\pc.json", $b, [System.Text.UTF8Encoding]::new($false))
    $times = @()
    foreach ($r in 1..4) { $times += [double](curl.exe -s -X POST $url -H "Content-Type: application/json" -H "Authorization: Bearer $k" -d "@$env:TEMP\pc.json" -o "$env:TEMP\pc.out" -w "%{time_total}") }
    $avg = ($times | Measure-Object -Average).Average
    $pt = 0; try { $o = Get-Content "$env:TEMP\pc.out" -Raw | ConvertFrom-Json; $pt = $o.usage.prompt_tokens } catch {}
    Write-Output ("PC[$($cfg[0])] n=$pt avg=$([math]::Round($avg,4))s = $([math]::Round($pt/$avg,0)) t/s  runs=[$([string]::Join(',', ($times | ForEach-Object { [math]::Round($_,3) })))]")
  }
}
Remove-Item "$root\bench-overrides.env" -EA SilentlyContinue
Stop-ScheduledTask -TaskName qwen36-inference -EA SilentlyContinue | Out-Null
Start-Sleep 1
Start-ScheduledTask -TaskName qwen36-inference
Write-Output 'done'
