$root = 'D:\Projects\yttri-inference'
$envFile = "$root\qwen36-server\.env"
$line = (Select-String -Path $envFile -Pattern 'API_KEYS=' | Select-Object -First 1).Line
$v = $line.Substring($line.IndexOf('API_KEYS=') + 9)
$k = [regex]::Match($v, '"key"\s*:\s*"([^"]+)"').Groups[1].Value
$url = 'http://127.0.0.1:18099/v1/chat/completions'
foreach ($p in @('hi','Say hello in one word.','Explain briefly what the sky is.')) {
  $b = @{model='ornith-1.5-9b';stream=$false;max_tokens=1;messages=@(@{role='user';content=$p});temperature=0} | ConvertTo-Json -Depth 6 -Compress
  [System.IO.File]::WriteAllText("$env:TEMP\ty.json", $b, [System.Text.UTF8Encoding]::new($false))
  $t = [double](curl.exe -s -X POST $url -H "Content-Type: application/json" -H "Authorization: Bearer $k" -d "@$env:TEMP\ty.json" -o "$env:TEMP\ty.out" -w "%{time_total}")
  $o = $null; try { $o = Get-Content "$env:TEMP\ty.out" -Raw | ConvertFrom-Json } catch {}
  if ($o -and $o.usage) { Write-Output ("'$p' prompt=$($o.usage.prompt_tokens) t=$($t)s") }
}
