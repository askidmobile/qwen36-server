$ErrorActionPreference = 'Continue'
$root = 'D:\Projects\yttri-inference'
$ovr = "$root\bench-overrides.env"
Set-Content -Path $ovr -Value @('QK_INT8=1','QK_INT8_PREFILL=1') -Encoding ascii
& powershell -NoProfile -ExecutionPolicy Bypass -File "$root\scripts\restart_only.ps1" | Select-Object -Last 1 | Out-Null
$envFile = "$root\qwen36-server\.env"
$line = (Select-String -Path $envFile -Pattern 'API_KEYS=' | Select-Object -First 1).Line
$v = $line.Substring($line.IndexOf('API_KEYS=') + 9)
$k = [regex]::Match($v, '"key"\s*:\s*"([^"]+)"').Groups[1].Value
$url = 'http://127.0.0.1:18099/v1/chat/completions'
foreach ($q in @('What is 17*23? Answer with one number.', 'Capital of Australia? Answer with one word.')) {
  $b = @{model='ornith-1.5-9b';stream=$false;max_tokens=32;temperature=0;messages=@(@{role='user';content=$q})} | ConvertTo-Json -Depth 6 -Compress
  $r = Invoke-RestMethod $url -Headers @{Authorization="Bearer $k"} -Method Post -ContentType 'application/json' -Body $b -TimeoutSec 300
  Write-Output ("Q: " + $q + "  A: " + ($r.choices[0].message.content -replace "`r?`n", " "))
}
Remove-Item $ovr -ErrorAction SilentlyContinue
& powershell -NoProfile -ExecutionPolicy Bypass -File "$root\scripts\restart_only.ps1" | Select-Object -Last 1
