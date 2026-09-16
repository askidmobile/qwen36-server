param([int]$Words = 14000)
$ErrorActionPreference = 'Continue'
$root = 'D:\Projects\yttri-inference'
$envFile = "$root\qwen36-server\.env"
$line = (Select-String -Path $envFile -Pattern 'API_KEYS=' | Select-Object -First 1).Line
$v = $line.Substring($line.IndexOf('API_KEYS=') + 9)
$k = [regex]::Match($v, '"key"\s*:\s*"([^"]+)"').Groups[1].Value
$url = 'http://127.0.0.1:18099/v1/chat/completions'
$sb = New-Object System.Text.StringBuilder
for ($i = 0; $i -lt $Words; $i++) { [void]$sb.Append("word$i ") }
$prompt = $sb.ToString() + 'Write a short story about a robot learning to paint.'
Write-Output ("LT override: " + ((Get-Content "$root\bench-overrides.env" -EA SilentlyContinue) -join ' | '))
$b = @{model='ornith-1.5-9b';stream=$false;max_tokens=4;messages=@(@{role='user';content=$prompt});temperature=0} | ConvertTo-Json -Depth 6 -Compress
[System.IO.File]::WriteAllText("$env:TEMP\lt.json", $b, [System.Text.UTF8Encoding]::new($false))
$t = [double](curl.exe -s -X POST $url -H "Content-Type: application/json" -H "Authorization: Bearer $k" -d "@$env:TEMP\lt.json" -o "$env:TEMP\lt.out" -w "%{time_total}")
$o = $null; try { $o = Get-Content "$env:TEMP\lt.out" -Raw | ConvertFrom-Json } catch {}
if ($o -and $o.usage) { Write-Output ("LT ok: prompt=$($o.usage.prompt_tokens) gen=$($o.usage.completion_tokens) time=$([math]::Round($t,1))s = $([math]::Round($o.usage.prompt_tokens/$t,0)) t/s") }
else { $raw = (Get-Content "$env:TEMP\lt.out" -Raw); Write-Output ("LT FAILED after $([math]::Round($t,1))s : " + $raw.Substring(0,[Math]::Min(160,$raw.Length))) }
nvidia-smi --query-gpu=memory.used --format=csv,noheader
Get-Content "$root\logs\server.log" -Tail 8 | Where-Object { $_ -match 'paged pool|OOM|OUT_OF_MEMORY|step error' }
