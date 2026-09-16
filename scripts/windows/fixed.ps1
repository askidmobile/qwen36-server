$ErrorActionPreference = 'Continue'
$root = 'D:\Projects\yttri-inference'
$envFile = "$root\qwen36-server\.env"
$line = (Select-String -Path $envFile -Pattern 'API_KEYS=' | Select-Object -First 1).Line
$v = $line.Substring($line.IndexOf('API_KEYS=') + 9)
$k = [regex]::Match($v, '"key"\s*:\s*"([^"]+)"').Groups[1].Value
$url = 'http://127.0.0.1:18099/v1/chat/completions'
function Req([string]$prompt, [string]$label) {
  $b = @{model='ornith-1.5-9b';stream=$false;max_tokens=1;messages=@(@{role='user';content=$prompt});temperature=0} | ConvertTo-Json -Depth 6 -Compress
  [System.IO.File]::WriteAllText("$env:TEMP\fx.json", $b, [System.Text.UTF8Encoding]::new($false))
  $t = [double](curl.exe -s -X POST $url -H "Content-Type: application/json" -H "Authorization: Bearer $k" -d "@$env:TEMP\fx.json" -o "$env:TEMP\fx.out" -w "%{time_total}")
  $o = $null; try { $o = Get-Content "$env:TEMP\fx.out" -Raw | ConvertFrom-Json } catch {}
  if ($o -and $o.usage) { Write-Output ("$label prompt=$($o.usage.prompt_tokens) t=$([math]::Round($t,3))s = $([math]::Round($o.usage.prompt_tokens/$t,0)) t/s") } else { Write-Output "$label FAILED t=$([math]::Round($t,2))" }
}
$sb = New-Object System.Text.StringBuilder
for ($i=0; $i -lt 131; $i++) { [void]$sb.Append("word$i ") }
$pA = $sb.ToString() + 'Count from 1 to 5.'
$sb2 = New-Object System.Text.StringBuilder
for ($i=0; $i -lt 131; $i++) { [void]$sb2.Append("tok$i ") }
$pB = $sb2.ToString() + 'Say hello.'
foreach ($r in 1..3) { Req $pA "A#$r" }
foreach ($r in 1..3) { Req $pB "B#$r" }
