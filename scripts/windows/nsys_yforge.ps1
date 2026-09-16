param([int]$Words = 6264)
$ErrorActionPreference = 'Continue'
$root = 'D:\Projects\yttri-inference'
$nsys = 'C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.6.3\target-windows-x64\nsys.exe'
$exe = "$root\qwen36-server\target\release\yforge.exe"
$out = "$root\yforge_pf"
Stop-ScheduledTask -TaskName qwen36-inference -EA SilentlyContinue | Out-Null
Start-Sleep 2
Get-Process yforge -EA SilentlyContinue | Stop-Process -Force -EA SilentlyContinue
Start-Sleep 3
Remove-Item "$out.nsys-rep" -EA SilentlyContinue
# Без графов и без кэша промптов: чистый eager-префил, как у llama под nsys.
$env:CUDA_GRAPHS = '0'
$env:PGRAPH = 'off'
$env:PREFIX_CACHE_MIB = '0'
$p = Start-Process -FilePath $nsys -ArgumentList @('profile','--force-overwrite=true','--trace=cuda','--sample=none','--capture-range=none','-o',$out,"$exe",'--env',"$root\qwen36-server\.env") -WorkingDirectory "$root\qwen36-server" -WindowStyle Hidden -PassThru
$envFile = "$root\qwen36-server\.env"
$line = (Select-String -Path $envFile -Pattern 'API_KEYS=' | Select-Object -First 1).Line
$v = $line.Substring($line.IndexOf('API_KEYS=') + 9)
$k = [regex]::Match($v, '"key"\s*:\s*"([^"]+)"').Groups[1].Value
$url = 'http://127.0.0.1:18099/v1/chat/completions'
$ready = $false
for ($i = 1; $i -le 120; $i++) {
  try { $b = @{model='ornith-1.5-9b';stream=$false;max_tokens=1;messages=@(@{role='user';content='hi'})} | ConvertTo-Json -Depth 5 -Compress
    Invoke-RestMethod $url -Headers @{Authorization="Bearer $k"} -Method Post -ContentType 'application/json' -Body $b -TimeoutSec 120 | Out-Null
    $ready = $true; break } catch { Start-Sleep -Seconds 3 }
}
Write-Output "yforge under nsys ready=$ready (waited $($i*3)s)"
if ($ready) {
  $sb = New-Object System.Text.StringBuilder
  for ($i = 0; $i -lt $Words; $i++) { [void]$sb.Append("word$i ") }
  $b2 = @{model='ornith-1.5-9b';stream=$false;max_tokens=1;messages=@(@{role='user';content=($sb.ToString() + 'Count from 1 to 5.')});temperature=0} | ConvertTo-Json -Depth 6 -Compress
  [System.IO.File]::WriteAllText("$env:TEMP\yp.json", $b2, [System.Text.UTF8Encoding]::new($false))
  $t = [double](curl.exe -s -X POST $url -H "Content-Type: application/json" -H "Authorization: Bearer $k" -d "@$env:TEMP\yp.json" -o "$env:TEMP\yp.out" -w "%{time_total}")
  $o = $null; try { $o = Get-Content "$env:TEMP\yp.out" -Raw | ConvertFrom-Json } catch {}
  if ($o -and $o.usage) { Write-Output ("yforge prefill $($o.usage.prompt_tokens) tok in $([math]::Round($t,1))s = $([math]::Round($o.usage.prompt_tokens/$t,0)) t/s") } else { Write-Output "prefill failed wall=$([math]::Round($t,1))s" }
}
Start-Sleep 5
Get-Process yforge -EA SilentlyContinue | Stop-Process -Force -EA SilentlyContinue
Wait-Process -Id $p.Id -Timeout 180 -ErrorAction SilentlyContinue
Start-Sleep 5
Get-Process nsys -EA SilentlyContinue | Stop-Process -Force -EA SilentlyContinue
Start-Sleep 5
if (Test-Path "$out.nsys-rep") { Write-Output 'profile saved' } else { Write-Output 'NO PROFILE' }
Remove-Item Env:\CUDA_GRAPHS -EA SilentlyContinue
Remove-Item Env:\PGRAPH -EA SilentlyContinue
Remove-Item Env:\PREFIX_CACHE_MIB -EA SilentlyContinue
