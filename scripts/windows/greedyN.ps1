param([string]$Tag = 'x', [int]$N = 3, [string[]]$Overrides = @())
$ErrorActionPreference = 'Continue'
$root = 'D:\Projects\yttri-inference'
$envFile = "$root\qwen36-server\.env"
$line = (Select-String -Path $envFile -Pattern 'API_KEYS=' | Select-Object -First 1).Line
$v = $line.Substring($line.IndexOf('API_KEYS=') + 9)
$k = [regex]::Match($v, '"key"\s*:\s*"([^"]+)"').Groups[1].Value
$url = 'http://127.0.0.1:18099/v1/chat/completions'
if ($Overrides.Count -gt 0) { Set-Content -Path "$root\bench-overrides.env" -Value $Overrides -Encoding ascii } else { Remove-Item "$root\bench-overrides.env" -EA SilentlyContinue }
Stop-ScheduledTask -TaskName qwen36-inference -EA SilentlyContinue | Out-Null
Start-Sleep 2
Get-Process yforge -EA SilentlyContinue | Stop-Process -Force -EA SilentlyContinue
Start-Sleep 3
Start-ScheduledTask -TaskName qwen36-inference
for ($i = 1; $i -le 60; $i++) {
  try { $b = @{model='ornith-1.5-9b';stream=$false;max_tokens=1;messages=@(@{role='user';content='hi'})} | ConvertTo-Json -Depth 5 -Compress
    Invoke-RestMethod $url -Headers @{Authorization="Bearer $k"} -Method Post -ContentType 'application/json' -Body $b -TimeoutSec 60 | Out-Null; break } catch { Start-Sleep -Seconds 3 }
}
$prompt = 'Explain in three short sentences why the sky appears blue during the day and red at sunset. Then list two everyday consequences.'
$b2 = @{model='ornith-1.5-9b';stream=$false;max_tokens=220;temperature=0.0;top_k=1;top_p=1.0;min_p=0.0;presence_penalty=0.0;repetition_penalty=1.0;messages=@(@{role='user';content=$prompt})} | ConvertTo-Json -Depth 6 -Compress
[System.IO.File]::WriteAllText("$env:TEMP\g.json", $b2, [System.Text.UTF8Encoding]::new($false))
for ($r = 1; $r -le $N; $r++) {
  $t = [double](curl.exe -s -X POST $url -H "Content-Type: application/json" -H "Authorization: Bearer $k" -d "@$env:TEMP\g.json" -o "$env:TEMP\g.out" -w "%{time_total}")
  $o = $null; try { $o = Get-Content "$env:TEMP\g.out" -Raw | ConvertFrom-Json } catch {}
  if (-not $o -or -not $o.usage) { Write-Output "G[$Tag] run=$r FAILED"; continue }
  $txt = $o.choices[0].message.content
  [System.IO.File]::WriteAllText("$root\logs\greedy-$Tag-$r.txt", $txt, [System.Text.UTF8Encoding]::new($false))
  Write-Output ("G[$Tag] run=$r tokens=$($o.usage.completion_tokens) tps=$([math]::Round($o.usage.completion_tokens/$t,2)) sha=$(([System.BitConverter]::ToString([System.Security.Cryptography.SHA256]::Create().ComputeHash([System.Text.Encoding]::UTF8.GetBytes($txt)))).Replace('-','').Substring(0,12))")
}
