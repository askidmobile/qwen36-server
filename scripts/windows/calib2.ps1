$ErrorActionPreference = 'Continue'
$root = 'D:\Projects\yttri-inference'
$envFile = "$root\qwen36-server\.env"
$line = (Select-String -Path $envFile -Pattern 'API_KEYS=' | Select-Object -First 1).Line
$v = $line.Substring($line.IndexOf('API_KEYS=') + 9)
$k = [regex]::Match($v, '"key"\s*:\s*"([^"]+)"').Groups[1].Value
$url = 'http://127.0.0.1:18099/v1/chat/completions'
Remove-Item "$root\bench-overrides.env" -EA SilentlyContinue
Start-ScheduledTask -TaskName qwen36-inference
$ready = $false
for ($i = 1; $i -le 60; $i++) {
  try { $b = @{model='ornith-1.5-9b';stream=$false;max_tokens=1;messages=@(@{role='user';content='hi'})} | ConvertTo-Json -Depth 5 -Compress
    Invoke-RestMethod $url -Headers @{Authorization="Bearer $k"} -Method Post -ContentType 'application/json' -Body $b -TimeoutSec 60 | Out-Null; $ready=$true; break } catch { Start-Sleep -Seconds 3 }
}
Write-Output "ready=$ready"
foreach ($words in @(131, 524, 1047)) {
  $sb = New-Object System.Text.StringBuilder
  for ($i = 0; $i -lt $words; $i++) { [void]$sb.Append("word$i ") }
  $p = $sb.ToString() + 'Count from 1 to 5.'
  $b2 = @{model='ornith-1.5-9b';stream=$false;max_tokens=1;messages=@(@{role='user';content=$p});temperature=0} | ConvertTo-Json -Depth 6 -Compress
  [System.IO.File]::WriteAllText("$env:TEMP\cb.json", $b2, [System.Text.UTF8Encoding]::new($false))
  $t = [double](curl.exe -s -X POST $url -H "Content-Type: application/json" -H "Authorization: Bearer $k" -d "@$env:TEMP\cb.json" -o "$env:TEMP\cb.out" -w "%{time_total}")
  $o = $null; try { $o = Get-Content "$env:TEMP\cb.out" -Raw | ConvertFrom-Json } catch {}
  if ($o -and $o.usage) { Write-Output ("pp$($words): prompt=$($o.usage.prompt_tokens) time=$([math]::Round($t,3))s = $([math]::Round($o.usage.prompt_tokens/$t,0)) t/s") } else { Write-Output "pp$($words): FAILED" }
}
