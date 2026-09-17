param([string]$PromptFile = 'D:\Projects\yttri-inference\prompt_omega.txt', [int]$MaxTokens = 32, [string]$Tag = '')
$ErrorActionPreference = 'Continue'
$root = 'D:\Projects\yttri-inference'
$line = (Select-String -Path "$root\qwen36-server\.env" -Pattern 'API_KEYS=' | Select-Object -First 1).Line
$v = $line.Substring($line.IndexOf('API_KEYS=') + 9)
$k = [regex]::Match($v, '"key"\s*:\s*"([^"]+)"').Groups[1].Value
$url = 'http://127.0.0.1:18099/v1/chat/completions'
$prompt = [System.IO.File]::ReadAllText($PromptFile)
$b = @{model='ornith-1.5-9b';stream=$false;max_tokens=$MaxTokens;messages=@(@{role='user';content=$prompt});temperature=0} | ConvertTo-Json -Depth 6 -Compress
[System.IO.File]::WriteAllText("$env:TEMP\ts.json", $b, [System.Text.UTF8Encoding]::new($false))
$sw = [System.Diagnostics.Stopwatch]::StartNew()
$null = & curl.exe -s -m 900 -X POST $url -H "Content-Type: application/json" -H "Authorization: Bearer $k" -d "@$env:TEMP\ts.json" -o "$env:TEMP\ts.out"
$sw.Stop()
$o = $null; try { $o = Get-Content "$env:TEMP\ts.out" -Raw | ConvertFrom-Json } catch {}
if ($null -eq $o -or $null -eq $o.choices) {
  $raw = (Get-Content "$env:TEMP\ts.out" -Raw)
  Write-Output ("TEXTSHA[$Tag] ERROR wall=$([math]::Round($sw.Elapsed.TotalSeconds,2))s body=" + $raw.Substring(0, [Math]::Min(200, $raw.Length)).Replace("`n", ' '))
  exit 1
}
$txt = $o.choices[0].message.content
$sha = [System.BitConverter]::ToString([System.Security.Cryptography.SHA256]::Create().ComputeHash([System.Text.Encoding]::UTF8.GetBytes($txt))).Replace('-','').Substring(0,16)
Write-Output ("TEXTSHA[$Tag] gen=$($o.usage.completion_tokens) prompt=$($o.usage.prompt_tokens) sha=$sha wall=$([math]::Round($sw.Elapsed.TotalSeconds,2))s len=$($txt.Length)")
