param([string]$Tag = 'base', [string[]]$Overrides = @())
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
function Req([string]$prompt) {
  $b = @{model='ornith-1.5-9b';stream=$false;max_tokens=1;messages=@(@{role='user';content=$prompt});temperature=0} | ConvertTo-Json -Depth 6 -Compress
  [System.IO.File]::WriteAllText("$env:TEMP\bs.json", $b, [System.Text.UTF8Encoding]::new($false))
  return [double](curl.exe -s -X POST $url -H "Content-Type: application/json" -H "Authorization: Bearer $k" -d "@$env:TEMP\bs.json" -o "$env:TEMP\bs.out" -w "%{time_total}")
}
$t1 = Req 'Explain briefly what the sky is.'
$t2 = Req 'Explain briefly what the sky is.'
$sb = New-Object System.Text.StringBuilder
for ($i=0; $i -lt 131; $i++) { [void]$sb.Append("word$i ") }
$t3 = Req ($sb.ToString() + 'Count from 1 to 5.')
$t4 = Req ($sb.ToString() + 'Count from 1 to 5.')
Write-Output ("BISECT[$Tag] tiny=$([math]::Round($t1,3))/$([math]::Round($t2,3))s 432tok=$([math]::Round($t3,3))/$([math]::Round($t4,3))s  overrides=[$($Overrides -join ',')]")
