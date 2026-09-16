$ErrorActionPreference = 'Continue'
$root = 'D:\Projects\yttri-inference'
$envFile = "$root\qwen36-server\.env"
$line = (Select-String -Path $envFile -Pattern 'API_KEYS=' | Select-Object -First 1).Line
$v = $line.Substring($line.IndexOf('API_KEYS=') + 9)
$k = [regex]::Match($v, '"key"\s*:\s*"([^"]+)"').Groups[1].Value
$url = 'http://127.0.0.1:18099/v1/chat/completions'
$code = "def fib(n):`n    if n < 2:`n        return n`n    return fib(n-1) + fib(n-2)`n`nContinue this file with a memoized fib and a test runner. Output only code."
function Run([string]$tag, [string]$prompt, [int]$n) {
  $b = @{model='ornith-1.5-9b';stream=$true;max_tokens=$n;messages=@(@{role='user';content=$prompt});temperature=0.6;top_p=0.95;top_k=20} | ConvertTo-Json -Depth 6 -Compress
  [System.IO.File]::WriteAllText("$env:TEMP\mf.json", $b, [System.Text.UTF8Encoding]::new($false))
  Add-Type -AssemblyName System.Net.Http | Out-Null
  $client = New-Object System.Net.Http.HttpClient
  $client.Timeout = [TimeSpan]::FromMinutes(20)
  $req = New-Object System.Net.Http.HttpRequestMessage('Post', $url)
  $req.Headers.TryAddWithoutValidation('Authorization', "Bearer $k") | Out-Null
  $req.Content = New-Object System.Net.Http.StringContent($b, [System.Text.Encoding]::UTF8, 'application/json')
  $sw = [Diagnostics.Stopwatch]::StartNew()
  $resp = $client.SendAsync($req, [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead).Result
  $sr = $resp.Content.ReadAsStreamAsync().Result
  $reader = New-Object System.IO.StreamReader($sr)
  $first=$null; $last=$null; $n2=0
  while (($ln = $reader.ReadLine()) -ne $null) {
    if (-not $ln.StartsWith('data: ')) { continue }
    $pl = $ln.Substring(6); if ($pl -eq '[DONE]') { continue }
    try { $o = $pl | ConvertFrom-Json } catch { continue }
    $d = $null; if ($o.choices -and $o.choices.Count -gt 0) { $d = $o.choices[0].delta }
    if ($d -and (($d.content -ne $null -and $d.content -ne '') -or ($d.reasoning_content -ne $null -and $d.reasoning_content -ne ''))) { $n2++; if ($null -eq $first) { $first = $sw.Elapsed.TotalSeconds }; $last = $sw.Elapsed.TotalSeconds }
  }
  $sw.Stop(); $reader.Dispose(); $resp.Dispose(); $client.Dispose()
  $dec = 0.0; if ($n2 -gt 1 -and $last -gt $first) { $dec = ($n2-1)/($last-$first) }
  Write-Output ("MTPF[$tag] chunks=$n2 decode=$([math]::Round($dec,2)) t/s")
}
foreach ($cfg in @(@('off',@()), @('w2',@('MTP=1','MTP_WIDTH=2')), @('w2noadapt',@('MTP=1','MTP_WIDTH=2','MTP_ADAPTIVE=0')), @('w3noadapt',@('MTP=1','MTP_WIDTH=3','MTP_ADAPTIVE=0')))) {
  if ($cfg[1].Count -gt 0) { Set-Content -Path "$root\bench-overrides.env" -Value $cfg[1] -Encoding ascii } else { Remove-Item "$root\bench-overrides.env" -EA SilentlyContinue }
  Stop-ScheduledTask -TaskName qwen36-inference -EA SilentlyContinue | Out-Null
  Start-Sleep 2
  Get-Process yforge -EA SilentlyContinue | Stop-Process -Force -EA SilentlyContinue
  Start-Sleep 3
  Start-ScheduledTask -TaskName qwen36-inference
  for ($i=1; $i -le 60; $i++) {
    try { $bb = @{model='ornith-1.5-9b';stream=$false;max_tokens=1;messages=@(@{role='user';content='hi'})} | ConvertTo-Json -Depth 5 -Compress
      Invoke-RestMethod $url -Headers @{Authorization="Bearer $k"} -Method Post -ContentType 'application/json' -Body $bb -TimeoutSec 60 | Out-Null; break } catch { Start-Sleep -Seconds 3 }
  }
  Start-Sleep 2
  Run $cfg[0] $code 300
}
Remove-Item "$root\bench-overrides.env" -EA SilentlyContinue
Stop-ScheduledTask -TaskName qwen36-inference -EA SilentlyContinue | Out-Null
Start-Sleep 1
Start-ScheduledTask -TaskName qwen36-inference
Write-Output 'done'
