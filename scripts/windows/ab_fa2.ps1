param([int]$Words = 6264, [int]$MaxTokens = 200, [int]$Reps = 3)
$ErrorActionPreference = 'Continue'
Add-Type -AssemblyName System.Net.Http | Out-Null
$root = 'D:\Projects\yttri-inference'
$envFile = "$root\qwen36-server\.env"
$line = (Select-String -Path $envFile -Pattern 'API_KEYS=' | Select-Object -First 1).Line
$v = $line.Substring($line.IndexOf('API_KEYS=') + 9)
$k = [regex]::Match($v, '"key"\s*:\s*"([^"]+)"').Groups[1].Value
$url = 'http://127.0.0.1:18099/v1/chat/completions'

$sb = New-Object System.Text.StringBuilder
for ($i = 0; $i -lt $Words; $i++) { [void]$sb.Append("word$i ") }
$longPrompt = $sb.ToString() + 'Write a short story about a robot learning to paint.'
$shortPrompt = 'Write a short story about a robot learning to paint.'
[System.IO.File]::WriteAllText("$env:TEMP\pf.json", (@{model='ornith-1.5-9b';stream=$false;max_tokens=1;messages=@(@{role='user';content=$longPrompt});temperature=0} | ConvertTo-Json -Depth 6 -Compress), [System.Text.UTF8Encoding]::new($false))

function Stop-Server {
  Stop-ScheduledTask -TaskName qwen36-inference -ErrorAction SilentlyContinue | Out-Null
  Start-Sleep -Seconds 2
  Get-Process yforge -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
  Start-Sleep -Seconds 3
}
function Start-Server { Start-ScheduledTask -TaskName qwen36-inference | Out-Null }
function Wait-Ready {
  for ($i = 1; $i -le 80; $i++) {
    try {
      $b = @{model='ornith-1.5-9b';stream=$false;max_tokens=1;messages=@(@{role='user';content='hi'})} | ConvertTo-Json -Depth 5 -Compress
      Invoke-RestMethod $url -Headers @{Authorization="Bearer $k"} -Method Post -ContentType 'application/json' -Body $b -TimeoutSec 120 | Out-Null
      return $true
    } catch { Start-Sleep -Seconds 3 }
  }
  return $false
}
function Stream-Bench([string]$prompt, [int]$maxTokens) {
  $body = @{ model='ornith-1.5-9b'; stream=$true; max_tokens=$maxTokens;
             messages=@(@{role='user'; content=$prompt}); temperature=0.0 } | ConvertTo-Json -Depth 6 -Compress
  $client = New-Object System.Net.Http.HttpClient
  $client.Timeout = [TimeSpan]::FromMinutes(60)
  $req = New-Object System.Net.Http.HttpRequestMessage('Post', $url)
  $req.Headers.TryAddWithoutValidation('Authorization', "Bearer $k") | Out-Null
  $req.Content = New-Object System.Net.Http.StringContent($body, [System.Text.Encoding]::UTF8, 'application/json')
  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  $resp = $client.SendAsync($req, [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead).Result
  $sr = $resp.Content.ReadAsStreamAsync().Result
  $reader = New-Object System.IO.StreamReader($sr)
  $first = $null; $last = $null; $n = 0
  while (($ln = $reader.ReadLine()) -ne $null) {
    if (-not $ln.StartsWith('data: ')) { continue }
    $payload = $ln.Substring(6); if ($payload -eq '[DONE]') { continue }
    try { $o = $payload | ConvertFrom-Json } catch { continue }
    $d = $null; if ($o.choices -and $o.choices.Count -gt 0) { $d = $o.choices[0].delta }
    if ($d -and (($d.content -ne $null -and $d.content -ne '') -or ($d.reasoning_content -ne $null -and $d.reasoning_content -ne ''))) {
      $n++; if ($null -eq $first) { $first = $sw.Elapsed.TotalSeconds }; $last = $sw.Elapsed.TotalSeconds
    }
  }
  $sw.Stop(); $reader.Dispose(); $resp.Dispose(); $client.Dispose()
  $dec = 0.0; if ($n -gt 1 -and $last -gt $first) { $dec = ($n - 1) / ($last - $first) }
  return [pscustomobject]@{ n=$n; ttft=$first; dec=$dec }
}
function Bench-Config([string]$name, [string[]]$lines) {
  if ($lines.Count -gt 0) { Set-Content -Path "$root\bench-overrides.env" -Value $lines -Encoding ascii }
  else { Remove-Item "$root\bench-overrides.env" -ErrorAction SilentlyContinue }
  Stop-Server; Start-Server
  if (-not (Wait-Ready)) { Write-Output "AB[$name] NOT READY"; return }
  $pf = [double](curl.exe -s -X POST $url -H "Content-Type: application/json" -H "Authorization: Bearer $k" -d "@$env:TEMP\pf.json" -o "$env:TEMP\pf.out" -w "%{time_total}")
  $longRates = @(); $shortRates = @()
  for ($r = 1; $r -le $Reps; $r++) {
    $d1 = Stream-Bench $longPrompt $MaxTokens
    $d2 = Stream-Bench $shortPrompt $MaxTokens
    $longRates += [math]::Round($d1.dec, 2); $shortRates += [math]::Round($d2.dec, 2)
  }
  $lm = ($longRates | Measure-Object -Average).Average
  $sm = ($shortRates | Measure-Object -Average).Average
  Write-Output ("AB[$name] prefill=$([math]::Round($pf,2))s long=[$($longRates -join ',')] avg=$([math]::Round($lm,2)) short=[$($shortRates -join ',')] avg=$([math]::Round($sm,2))")
}
Bench-Config 'f16_s63'  @('QWEN36_FA_DEBUG=1')
Bench-Config 'f16_s32'  @('QWEN36_FA_DEBUG=1','QWEN36_FA_SPLITS=32')
Bench-Config 'f16_s16'  @('QWEN36_FA_DEBUG=1','QWEN36_FA_SPLITS=16')
Bench-Config 'q8_s63'   @('QWEN36_FA_DEBUG=1','KV_POOL_Q8=1')
Bench-Config 'q8_s32'   @('QWEN36_FA_DEBUG=1','KV_POOL_Q8=1','QWEN36_FA_SPLITS=32')
Remove-Item "$root\bench-overrides.env" -ErrorAction SilentlyContinue
Stop-Server; Start-Server
Write-Output 'AB2 done (production restored)'
