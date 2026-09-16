param([int]$Words = 0, [int]$MaxTokens = 60)
$ErrorActionPreference = 'Continue'
$root = 'D:\Projects\yttri-inference'
Set-Content -Path "$root\bench-overrides.env" -Value @('GPROF=1','QWEN36_FA_DEBUG=1') -Encoding ascii
Stop-ScheduledTask -TaskName qwen36-inference -ErrorAction SilentlyContinue | Out-Null
Start-Sleep 2
Get-Process yforge -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep 3
$mark = (Get-Content "$root\logs\server.log" | Measure-Object -Line).Lines
Start-ScheduledTask -TaskName qwen36-inference
$envFile = "$root\qwen36-server\.env"
$line = (Select-String -Path $envFile -Pattern 'API_KEYS=' | Select-Object -First 1).Line
$v = $line.Substring($line.IndexOf('API_KEYS=') + 9)
$k = [regex]::Match($v, '"key"\s*:\s*"([^"]+)"').Groups[1].Value
$url = 'http://127.0.0.1:18099/v1/chat/completions'
$ready = $false
for ($i = 1; $i -le 60; $i++) {
  try {
    $b = @{model='ornith-1.5-9b';stream=$false;max_tokens=1;messages=@(@{role='user';content='hi'})} | ConvertTo-Json -Depth 5 -Compress
    Invoke-RestMethod $url -Headers @{Authorization="Bearer $k"} -Method Post -ContentType 'application/json' -Body $b -TimeoutSec 60 | Out-Null
    $ready = $true; break
  } catch { Start-Sleep -Seconds 3 }
}
if (-not $ready) { Write-Output 'NOT READY'; exit 1 }
$sb = New-Object System.Text.StringBuilder
for ($i = 0; $i -lt $Words; $i++) { [void]$sb.Append("word$i ") }
$prompt = $sb.ToString() + 'Write a long detailed story about a robot learning to paint.'
$b2 = @{model='ornith-1.5-9b';stream=$false;max_tokens=$MaxTokens;messages=@(@{role='user';content=$prompt});temperature=0} | ConvertTo-Json -Depth 6 -Compress
[System.IO.File]::WriteAllText("$env:TEMP\p1.json", $b2, [System.Text.UTF8Encoding]::new($false))
$t = [double](curl.exe -s -X POST $url -H "Content-Type: application/json" -H "Authorization: Bearer $k" -d "@$env:TEMP\p1.json" -o "$env:TEMP\p1.out" -w "%{time_total}")
$o = Get-Content "$env:TEMP\p1.out" -Raw | ConvertFrom-Json
Write-Output "words=$Words prompt=$($o.usage.prompt_tokens) generated=$($o.usage.completion_tokens) total=$([math]::Round($t,2))s"
Start-Sleep 2
Get-Content "$root\logs\server.log" | Select-Object -Skip $mark | Where-Object { $_ -match 'gGPU|\[fa\]' }
