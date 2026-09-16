param([string]$Override = '', [int]$Split = -1)
$ErrorActionPreference = 'Continue'
$root = 'D:\Projects\yttri-inference'
if ($Split -ge 0) { $Override = "QWEN36_FA_SPLITS=$Split" }
if ([string]::IsNullOrEmpty($Override)) { Remove-Item "$root\bench-overrides.env" -ErrorAction SilentlyContinue }
else { Set-Content -Path "$root\bench-overrides.env" -Value @($Override) -Encoding ascii }
Stop-ScheduledTask -TaskName qwen36-inference -ErrorAction SilentlyContinue | Out-Null
Start-Sleep 2
Get-Process yforge -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep 3
Start-ScheduledTask -TaskName qwen36-inference
$envFile = "$root\qwen36-server\.env"
$line = (Select-String -Path $envFile -Pattern 'API_KEYS=' | Select-Object -First 1).Line
$v = $line.Substring($line.IndexOf('API_KEYS=') + 9)
$k = [regex]::Match($v, '"key"\s*:\s*"([^"]+)"').Groups[1].Value
$url = 'http://127.0.0.1:18099/v1/chat/completions'
for ($i = 1; $i -le 80; $i++) {
  try {
    $b = @{model='ornith-1.5-9b';stream=$false;max_tokens=1;messages=@(@{role='user';content='hi'})} | ConvertTo-Json -Depth 5 -Compress
    Invoke-RestMethod $url -Headers @{Authorization="Bearer $k"} -Method Post -ContentType 'application/json' -Body $b -TimeoutSec 90 | Out-Null
    Write-Output "READY override='$Override'"
    exit 0
  } catch { Start-Sleep -Seconds 3 }
}
Write-Output "NOT READY override='$Override'"
