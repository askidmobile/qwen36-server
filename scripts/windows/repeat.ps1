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
$prompt = $sb.ToString() + 'Write a short story about a robot learning to paint.'
$b = @{model='ornith-1.5-9b';stream=$false;max_tokens=2;messages=@(@{role='user';content=$prompt});temperature=0} | ConvertTo-Json -Depth 6 -Compress
[System.IO.File]::WriteAllText("$env:TEMP\rp.json", $b, [System.Text.UTF8Encoding]::new($false))
for ($i = 1; $i -le 3; $i++) {
  $t = [double](curl.exe -s -X POST $url -H "Content-Type: application/json" -H "Authorization: Bearer $k" -d "@$env:TEMP\rp.json" -o "$env:TEMP\rp.out" -w "%{time_total}")
  $o = $null; try { $o = Get-Content "$env:TEMP\rp.out" -Raw | ConvertFrom-Json } catch {}
  if ($o -and $o.usage) { Write-Output ("REPEAT run=$i prompt=$($o.usage.prompt_tokens) total=$([math]::Round($t,2))s = $([math]::Round($o.usage.prompt_tokens/$t,0)) t/s") } else { Write-Output "REPEAT run=$i FAILED $([math]::Round($t,2))s" }
}
Get-Content "$root\logs\server.log" -Tail 25 | Where-Object { $_ -match 'pcache' }
