param(
  [int]$Words = 0,
  [int]$MaxTokens = 200,
  [int]$Runs = 1,
  [switch]$NoStream,
  [string]$PromptFile = '',
  [string]$Tag = ''
)
$ErrorActionPreference = 'Stop'
[System.Net.ServicePointManager]::Expect100Continue = $false
Add-Type -AssemblyName System.Net.Http | Out-Null

$envFile = 'D:\Projects\yttri-inference\qwen36-server\.env'
$line = (Select-String -Path $envFile -Pattern 'API_KEYS=' | Select-Object -First 1).Line
$v = $line.Substring($line.IndexOf('API_KEYS=') + 9)
$k = [regex]::Match($v, '"key"\s*:\s*"([^"]+)"').Groups[1].Value
$url = 'http://127.0.0.1:18099/v1/chat/completions'

if ($PromptFile -ne '') {
  $prompt = [System.IO.File]::ReadAllText($PromptFile)
} else {
  $sb = New-Object System.Text.StringBuilder
  for ($i = 0; $i -lt $Words; $i++) { [void]$sb.Append("word$i ") }
  if ($Words -gt 0) { [void]$sb.Append('. ') }
  $prompt = $sb.ToString() + 'Write a short story about a robot learning to paint.'
}

function Invoke-Streaming([string]$prompt, [int]$maxTokens) {
  $body = @{ model='ornith-1.5-9b'; stream=$true; max_tokens=$maxTokens;
             messages=@(@{role='user'; content=$prompt}); temperature=0.0 } | ConvertTo-Json -Depth 6 -Compress
  $client = New-Object System.Net.Http.HttpClient
  $client.Timeout = [TimeSpan]::FromMinutes(60)
  $req = New-Object System.Net.Http.HttpRequestMessage('Post', $url)
  $req.Headers.TryAddWithoutValidation('Authorization', "Bearer $k") | Out-Null
  $req.Content = New-Object System.Net.Http.StringContent($body, [System.Text.Encoding]::UTF8, 'application/json')
  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  $resp = $client.SendAsync($req, [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead).Result
  $stream = $resp.Content.ReadAsStreamAsync().Result
  $reader = New-Object System.IO.StreamReader($stream)
  $first = $null; $last = $null; $n = 0; $usage = $null; $tps = $null
  while (($line = $reader.ReadLine()) -ne $null) {
    if ($line.Length -lt 6 -or -not $line.StartsWith('data: ')) { continue }
    $payload = $line.Substring(6)
    if ($payload -eq '[DONE]') { continue }
    try { $o = $payload | ConvertFrom-Json } catch { continue }
    if ($o.usage) { $usage = $o.usage }
    if ($o.timings) { $tps = $o.timings }
    $d = $null
    if ($o.choices -and $o.choices.Count -gt 0) { $d = $o.choices[0].delta }
    if ($d -and (($d.content -ne $null -and $d.content -ne '') -or ($d.reasoning_content -ne $null -and $d.reasoning_content -ne ''))) {
      $n++
      if ($null -eq $first) { $first = $sw.Elapsed.TotalSeconds }
      $last = $sw.Elapsed.TotalSeconds
    }
  }
  $sw.Stop()
  $reader.Dispose(); $resp.Dispose(); $client.Dispose()
  $dec = 0.0
  if ($n -gt 1 -and $last -gt $first) { $dec = [math]::Round(($n - 1) / ($last - $first), 2) }
  return [pscustomobject]@{ chunks=$n; ttft=[math]::Round($first,3); total=[math]::Round($sw.Elapsed.TotalSeconds,3); decode_tps=$dec; usage=$usage; timings=$tps }
}

for ($r = 1; $r -le $Runs; $r++) {
  $res = Invoke-Streaming $prompt $MaxTokens
  $u = $res.usage
  $mtp = ''
  if ($u -and $u.mtp) { $mtp = "mtp(enabled=$($u.mtp.enabled),used=$($u.mtp.used),drafted=$($u.mtp.drafted),accepted=$($u.mtp.accepted))" }
  $pt = if ($u) { $u.prompt_tokens } else { '?' }
  $ct = if ($u) { $u.completion_tokens } else { '?' }
  Write-Output ("BENCH[$Tag] run=$r words=$Words prompt_tokens=$pt completion_tokens=$ct chunks=$($res.chunks) TTFT=$($res.ttft)s total=$($res.total)s decode=$($res.decode_tps) t/s $mtp")
}
