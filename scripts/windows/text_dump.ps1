param([string]$PromptFile, [string]$Tag, [int]$MaxTokens = 64)
$ErrorActionPreference = 'Continue'
$root = 'D:\Projects\yttri-inference'
$line = (Select-String -Path "$root\qwen36-server\.env" -Pattern 'API_KEYS=' | Select-Object -First 1).Line
$v = $line.Substring($line.IndexOf('API_KEYS=') + 9)
$k = [regex]::Match($v, '"key"\s*:\s*"([^"]+)"').Groups[1].Value
$url = 'http://127.0.0.1:18099/v1/chat/completions'
$prompt = [System.IO.File]::ReadAllText($PromptFile)
$b = @{model='ornith-1.5-9b';stream=$false;max_tokens=$MaxTokens;messages=@(@{role='user';content=$prompt});temperature=0} | ConvertTo-Json -Depth 6 -Compress
[System.IO.File]::WriteAllText("$env:TEMP\td.json", $b, [System.Text.UTF8Encoding]::new($false))
$null = & curl.exe -s -m 900 -X POST $url -H "Content-Type: application/json" -H "Authorization: Bearer $k" -d "@$env:TEMP\td.json" -o "$env:TEMP\td.out"
$o = $null; try { $o = Get-Content "$env:TEMP\td.out" -Raw | ConvertFrom-Json } catch {}
if ($null -eq $o -or $null -eq $o.choices) { Write-Output "DUMP[$Tag] ERROR"; exit 1 }
$txt = $o.choices[0].message.content
[System.IO.File]::WriteAllText("$root\logs\text-$Tag.txt", $txt, [System.Text.UTF8Encoding]::new($false))
Write-Output ("DUMP[$Tag] gen=$($o.usage.completion_tokens) len=" + $txt.Length)
