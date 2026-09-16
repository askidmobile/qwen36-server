param([int]$Words = 6264, [int]$MaxTokens = 200, [int]$Reps = 3, [string]$KvType = 'q8_0')
$ErrorActionPreference = 'Continue'
Add-Type -AssemblyName System.Net.Http | Out-Null
$llama = 'D:\Projects\yttri-inference\llama.cpp-bin\llama-server.exe'
$gguf  = 'D:\Models\deepreinforce-ai\Ornith-1.5-9B-GGUF\Ornith-1.5-9B-Q4_K_M.gguf'
$port  = 18098
$url   = "http://127.0.0.1:$port/v1/chat/completions"
$key   = 'benchkey'
$env:PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin;' + $env:PATH

$sb = New-Object System.Text.StringBuilder
for ($i = 0; $i -lt $Words; $i++) { [void]$sb.Append("word$i ") }
$longPrompt = $sb.ToString() + 'Write a short story about a robot learning to paint.'
$shortPrompt = 'Write a short story about a robot learning to paint.'

Stop-ScheduledTask -TaskName qwen36-inference -ErrorAction SilentlyContinue | Out-Null
Start-Sleep 2
Get-Process yforge -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep 3

$sargs = "-m `"$gguf`" --host 127.0.0.1 --port $port -ngl 999 -c 131072 -fa on -np 1 -ctk $KvType -ctv $KvType -a ornith-1.5-9b --api-key $key --no-webui"
$p = Start-Process -FilePath $llama -ArgumentList $sargs -WorkingDirectory 'D:\Projects\yttri-inference\llama.cpp-bin' `
     -RedirectStandardOutput 'D:\Projects\yttri-inference\logs\hc-llama.out' `
     -RedirectStandardError  'D:\Projects\yttri-inference\logs\hc-llama.err' -WindowStyle Hidden -PassThru

$ready = $false
for ($i = 1; $i -le 120; $i++) {
  try {
    $b = @{model='ornith-1.5-9b';stream=$false;max_tokens=1;messages=@(@{role='user';content='hi'})} | ConvertTo-Json -Depth 5 -Compress
    Invoke-RestMethod $url -Headers @{Authorization="Bearer $key"} -Method Post -ContentType 'application/json' -Body $b -TimeoutSec 120 | Out-Null
    $ready = $true; break
  } catch { Start-Sleep -Seconds 3 }
}
if (-not $ready) { Write-Output "LLAMA[$KvType] NOT READY"; Get-Content 'D:\Projects\yttri-inference\logs\hc-llama.err' -Tail 10; exit 1 }
Write-Output "llama[$KvType] ready (pid $($p.Id))"

function Stream-Bench([string]$prompt, [int]$maxTokens) {
  $body = @{ model='ornith-1.5-9b'; stream=$true; max_tokens=$maxTokens;
             messages=@(@{role='user'; content=$prompt}); temperature=0.0 } | ConvertTo-Json -Depth 6 -Compress
  $client = New-Object System.Net.Http.HttpClient
  $client.Timeout = [TimeSpan]::FromMinutes(60)
  $req = New-Object System.Net.Http.HttpRequestMessage('Post', $url)
  $req.Headers.TryAddWithoutValidation('Authorization', "Bearer $key") | Out-Null
  $req.Content = New-Object System.Net.Http.StringContent($body, [System.Text.Encoding]::UTF8, 'application/json')
  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  $resp = $client.SendAsync($req, [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead).Result
  $sr = $resp.Content.ReadAsStreamAsync().Result
  $reader = New-Object System.IO.StreamReader($sr)
  $first = $null; $last = $null; $n = 0; $tim = $null
  while (($ln = $reader.ReadLine()) -ne $null) {
    if (-not $ln.StartsWith('data: ')) { continue }
    $payload = $ln.Substring(6); if ($payload -eq '[DONE]') { continue }
    try { $o = $payload | ConvertFrom-Json } catch { continue }
    if ($o.timings) { $tim = $o.timings }
    $d = $null; if ($o.choices -and $o.choices.Count -gt 0) { $d = $o.choices[0].delta }
    if ($d -and (($d.content -ne $null -and $d.content -ne '') -or ($d.reasoning_content -ne $null -and $d.reasoning_content -ne ''))) {
      $n++; if ($null -eq $first) { $first = $sw.Elapsed.TotalSeconds }; $last = $sw.Elapsed.TotalSeconds
    }
  }
  $sw.Stop(); $reader.Dispose(); $resp.Dispose(); $client.Dispose()
  $dec = 0.0; if ($n -gt 1 -and $last -gt $first) { $dec = ($n - 1) / ($last - $first) }
  return [pscustomobject]@{ n=$n; ttft=$first; dec=$dec; tim=$tim }
}

$lp = Stream-Bench $longPrompt 1
$pfSec = $lp.tim.prompt_ms / 1000.0
$pt = $lp.tim.prompt_n
$longRates = @(); $shortRates = @()
foreach ($r in 1..$Reps) {
  $d1 = Stream-Bench $longPrompt $MaxTokens
  $d2 = Stream-Bench $shortPrompt $MaxTokens
  $longRates += [math]::Round($d1.dec, 2); $shortRates += [math]::Round($d2.dec, 2)
}
$lm = ($longRates | Measure-Object -Average).Average
$sm = ($shortRates | Measure-Object -Average).Average
Write-Output ("LLAMA[$KvType]|prompt=$pt|prefill=$([math]::Round($pfSec,2))s=$([math]::Round($pt/$pfSec,0))tps|long=[$($longRates -join ',')]avg=$([math]::Round($lm,2))|short=[$($shortRates -join ',')]avg=$([math]::Round($sm,2))|vram=$((nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -join '')")
Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
Start-Sleep 3
Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Write-Output "LLAMA[$KvType] done"
