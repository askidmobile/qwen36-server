param([int]$Words = 6264)
$ErrorActionPreference = 'Continue'
$root = 'D:\Projects\yttri-inference'
$envFile = "$root\qwen36-server\.env"
$line = (Select-String -Path $envFile -Pattern 'API_KEYS=' | Select-Object -First 1).Line
$v = $line.Substring($line.IndexOf('API_KEYS=') + 9)
$k = [regex]::Match($v, '"key"\s*:\s*"([^"]+)"').Groups[1].Value
$url = 'http://127.0.0.1:18099/v1/chat/completions'
$sb = New-Object System.Text.StringBuilder
for ($i = 0; $i -lt $Words; $i++) { [void]$sb.Append("word$i ") }
$big = $sb.ToString() + 'Write a short story about a robot learning to paint.'
$b = @{model='ornith-1.5-9b';stream=$false;max_tokens=8;messages=@(@{role='user';content=$big});temperature=0} | ConvertTo-Json -Depth 6 -Compress
[System.IO.File]::WriteAllText("$env:TEMP\oom.json", $b, [System.Text.UTF8Encoding]::new($false))
# 1) Большой запрос: ожидаем ошибку (OOM), но НЕ зависание сервиса.
$sw = [System.Diagnostics.Stopwatch]::StartNew()
$out = & curl.exe -s -m 180 -X POST $url -H "Content-Type: application/json" -H "Authorization: Bearer $k" -d "@$env:TEMP\oom.json" -w "|HTTP=%{http_code}|t=%{time_total}"
$sw.Stop()
Write-Output ("BIG: " + ($out | Out-String).Trim())
# 2) Малый запрос после ошибки: сервис должен отвечать.
$b2 = @{model='ornith-1.5-9b';stream=$false;max_tokens=8;messages=@(@{role='user';content='Say hi in one word.'});temperature=0} | ConvertTo-Json -Depth 6 -Compress
[System.IO.File]::WriteAllText("$env:TEMP\small.json", $b2, [System.Text.UTF8Encoding]::new($false))
$out2 = & curl.exe -s -m 120 -X POST $url -H "Content-Type: application/json" -H "Authorization: Bearer $k" -d "@$env:TEMP\small.json" -w "|HTTP=%{http_code}|t=%{time_total}"
Write-Output ("SMALL: " + ($out2 | Out-String).Trim())
