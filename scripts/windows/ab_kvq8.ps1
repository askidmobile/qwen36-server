$ErrorActionPreference = 'Continue'
$root = 'D:\Projects\yttri-inference'
$envFile = "$root\qwen36-server\.env"
$line = (Select-String -Path $envFile -Pattern 'API_KEYS=' | Select-Object -First 1).Line
$v = $line.Substring($line.IndexOf('API_KEYS=') + 9)
$k = [regex]::Match($v, '"key"\s*:\s*"([^"]+)"').Groups[1].Value
$url = 'http://127.0.0.1:18099/v1/chat/completions'
Add-Type -AssemblyName System.Net.Http | Out-Null
function Stream-Bench([string]$prompt, [int]$maxTokens) {
  $body = @{ model='ornith-1.5-9b'; stream=$true; max_tokens=$maxTokens; messages=@(@{role='user'; content=$prompt}); temperature=0.0 } | ConvertTo-Json -Depth 6 -Compress
  $client = New-Object System.Net.Http.HttpClient
  $client.Timeout = [TimeSpan]::FromMinutes(30)
  $req = New-Object System.Net.Http.HttpRequestMessage('Post', $url)
  $req.Headers.TryAddWithoutValidation('Authorization', "Bearer $k") | Out-Null
  $req.Content = New-Object System.Net.Http.StringContent($body, [System.Text.Encoding]::UTF8, 'application/json')
  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  $resp = $client.SendAsync($req, [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead).Result
  $sr = $resp.Content.ReadAsStreamAsync().Result
  $reader = New-Object System.IO.StreamReader($sr)
  $first=$null; $last=$null; $n=0
  while (($ln = $reader.ReadLine()) -ne $null) {
    if (-not $ln.StartsWith('data: ')) { continue }
    $payload = $ln.Substring(6); if ($payload -eq '[DONE]') { continue }
    try { $o = $payload | ConvertFrom-Json } catch { continue }
    $d = $null; if ($o.choices -and $o.choices.Count -gt 0) { $d = $o.choices[0].delta }
    if ($d -and (($d.content -ne $null -and $d.content -ne '') -or ($d.reasoning_content -ne $null -and $d.reasoning_content -ne ''))) { $n++; if ($null -eq $first) { $first = $sw.Elapsed.TotalSeconds }; $last = $sw.Elapsed.TotalSeconds }
  }
  $sw.Stop(); $reader.Dispose(); $resp.Dispose(); $client.Dispose()
  $dec = 0.0; if ($n -gt 1 -and $last -gt $first) { $dec = ($n-1)/($last-$first) }
  return [pscustomobject]@{ n=$n; ttft=$first; dec=$dec }
}
$sb = New-Object System.Text.StringBuilder
for ($i=0; $i -lt 6264; $i++) { [void]$sb.Append("word$i ") }
$long = $sb.ToString() + 'Write a short story about a robot.'
$short = 'Write a short story about a robot.'
foreach ($cfg in @(@('q8',@('KV_POOL_Q8=1')), @('f16',@()))) {
  if ($cfg[1].Count -gt 0) { Set-Content -Path "$root\bench-overrides.env" -Value $cfg[1] -Encoding ascii } else { Remove-Item "$root\bench-overrides.env" -EA SilentlyContinue }
  Stop-ScheduledTask -TaskName qwen36-inference -EA SilentlyContinue | Out-Null
  Start-Sleep 2
  Get-Process yforge -EA SilentlyContinue | Stop-Process -Force -EA SilentlyContinue
  Start-Sleep 3
  Start-ScheduledTask -TaskName qwen36-inference
  $ok=$false
  for ($i=1; $i -le 60; $i++) {
    try { $b = @{model='ornith-1.5-9b';stream=$false;max_tokens=1;messages=@(@{role='user';content='hi'})} | ConvertTo-Json -Depth 5 -Compress
      Invoke-RestMethod $url -Headers @{Authorization="Bearer $k"} -Method Post -ContentType 'application/json' -Body $b -TimeoutSec 60 | Out-Null; $ok=$true; break } catch { Start-Sleep -Seconds 3 }
  }
  if (-not $ok) { Write-Output "KV[$($cfg[0])] NOT READY"; continue }
  Start-Sleep 2
  $d1 = Stream-Bench $long 120
  $d2 = Stream-Bench $short 120
  $pool = (Get-Content "$root\logs\server.log" -Tail 60 | Where-Object { $_ -match 'paged pool' } | Select-Object -Last 1)
  Write-Output ("KV[$($cfg[0])] prefill30k=$([math]::Round($d1.ttft,1))s decode30k=$([math]::Round($d1.dec,2)) t/s | short=$([math]::Round($d2.dec,2)) t/s")
  Write-Output ("KV[$($cfg[0])] " + $pool)
}
Remove-Item "$root\bench-overrides.env" -EA SilentlyContinue
Stop-ScheduledTask -TaskName qwen36-inference -EA SilentlyContinue | Out-Null
Start-Sleep 1
Start-ScheduledTask -TaskName qwen36-inference
Write-Output 'done'
