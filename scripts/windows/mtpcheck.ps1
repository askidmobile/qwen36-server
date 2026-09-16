param([int]$MaxTokens = 200)
$ErrorActionPreference = 'Continue'
$root = 'D:\Projects\yttri-inference'
Set-Content -Path "$root\bench-overrides.env" -Value @('MTP=1','MTP_TIMING=0') -Encoding ascii
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
$ready = $false
for ($i = 1; $i -le 60; $i++) {
  try {
    $b = @{model='ornith-1.5-9b';stream=$false;max_tokens=1;messages=@(@{role='user';content='hi'})} | ConvertTo-Json -Depth 5 -Compress
    Invoke-RestMethod $url -Headers @{Authorization="Bearer $k"} -Method Post -ContentType 'application/json' -Body $b -TimeoutSec 60 | Out-Null
    $ready = $true; break
  } catch { Start-Sleep -Seconds 3 }
}
if (-not $ready) { Write-Output 'NOT READY'; exit 1 }
$code = @'
def fib(n):
    if n < 2:
        return n
    return fib(n-1) + fib(n-2)

def main():
    for i in range(20):
        print(i, fib(i))
'@
foreach ($case in @(@('prose','Write a long detailed story about a robot learning to paint. Steps in order, no lists.'), @('code',($code + "`nContinue this file with a memoized fib and a test runner. Output only code.")))) {
  $b2 = @{model='ornith-1.5-9b';stream=$false;max_tokens=$MaxTokens;messages=@(@{role='user';content=$case[1]});temperature=0} | ConvertTo-Json -Depth 6 -Compress
  [System.IO.File]::WriteAllText("$env:TEMP\mc.json", $b2, [System.Text.UTF8Encoding]::new($false))
  $t = [double](curl.exe -s -X POST $url -H "Content-Type: application/json" -H "Authorization: Bearer $k" -d "@$env:TEMP\mc.json" -o "$env:TEMP\mc.out" -w "%{time_total}")
  $o = Get-Content "$env:TEMP\mc.out" -Raw | ConvertFrom-Json
  $m = $o.usage.mtp
  $ct = $o.usage.completion_tokens
  $d = $m.drafted; $a = $m.accepted
  $acc = 0.0; if ($d -gt 0) { $acc = [math]::Round(100.0*$a/$d,1) }
  Write-Output ("MTP[$($case[0])] tokens=$ct time=$([math]::Round($t,2))s tps=$([math]::Round($ct/$t,2)) enabled=$($m.enabled) used=$($m.used) drafted=$d accepted=$a accept=$acc% fallback=$($m.fallback_category)")
}
