param([string]$Tag = 'run', [string[]]$Overrides = @())
$ErrorActionPreference = 'Continue'
$root = 'D:\Projects\yttri-inference'
$envFile = "$root\qwen36-server\.env"
$line = (Select-String -Path $envFile -Pattern 'API_KEYS=' | Select-Object -First 1).Line
$v = $line.Substring($line.IndexOf('API_KEYS=') + 9)
$k = [regex]::Match($v, '"key"\s*:\s*"([^"]+)"').Groups[1].Value
$url = 'http://127.0.0.1:18099/v1/chat/completions'

function Start-Engine {
  if ($Overrides.Count -gt 0) { Set-Content -Path "$root\bench-overrides.env" -Value $Overrides -Encoding ascii }
  else { Remove-Item "$root\bench-overrides.env" -EA SilentlyContinue }
  Stop-ScheduledTask -TaskName qwen36-inference -EA SilentlyContinue | Out-Null
  Start-Sleep 2
  Get-Process yforge -EA SilentlyContinue | Stop-Process -Force -EA SilentlyContinue
  Start-Sleep 3
  Start-ScheduledTask -TaskName qwen36-inference
  for ($i = 1; $i -le 60; $i++) {
    try { $b = @{model='ornith-1.5-9b';stream=$false;max_tokens=1;messages=@(@{role='user';content='hi'})} | ConvertTo-Json -Depth 5 -Compress
      Invoke-RestMethod $url -Headers @{Authorization="Bearer $k"} -Method Post -ContentType 'application/json' -Body $b -TimeoutSec 60 | Out-Null
      return $true } catch { Start-Sleep -Seconds 3 }
  }
  return $false
}

function Greedy([string]$tag) {
  $prompt = 'Explain in three short sentences why the sky appears blue during the day and red at sunset. Then list two everyday consequences.'
  $b = @{model='ornith-1.5-9b';stream=$false;max_tokens=220;temperature=0;top_k=1;messages=@(@{role='user';content=$prompt})} | ConvertTo-Json -Depth 6 -Compress
  [System.IO.File]::WriteAllText("$env:TEMP\gr.json", $b, [System.Text.UTF8Encoding]::new($false))
  $t = [double](curl.exe -s -X POST $url -H "Content-Type: application/json" -H "Authorization: Bearer $k" -d "@$env:TEMP\gr.json" -o "$env:TEMP\gr.out" -w "%{time_total}")
  $o = $null; try { $o = Get-Content "$env:TEMP\gr.out" -Raw | ConvertFrom-Json } catch {}
  if (-not $o -or -not $o.usage) { Write-Output "GREEDY[$tag] FAILED in $([math]::Round($t,1))s"; return }
  $txt = $o.choices[0].message.content
  [System.IO.File]::WriteAllText("$root\logs\greedy-$tag.txt", $txt, [System.Text.UTF8Encoding]::new($false))
  $ct = $o.usage.completion_tokens
  Write-Output ("GREEDY[$tag] tokens=$ct time=$([math]::Round($t,2))s tps=$([math]::Round($ct/$t,2)) chars=$($txt.Length)")
  Write-Output ("GREEDY[$tag] head: " + ($txt.Substring(0, [Math]::Min(120, $txt.Length)) -replace "`n", ' | '))
}

if (-not (Start-Engine)) { Write-Output "ABFUSED[$Tag] NOT READY"; exit 1 }
Write-Output ("ABFUSED[$Tag] overrides: " + (($Overrides -join ', ')))
Greedy $Tag
& 'D:\Projects\yttri-inference\scripts\bench-live.ps1' -Words 0 -MaxTokens 200 -Runs 3 -Tag $Tag
& 'D:\Projects\yttri-inference\scripts\bench-live.ps1' -Words 6264 -MaxTokens 150 -Runs 2 -Tag "$Tag-30k"
