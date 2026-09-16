param([int]$Words = 6264)
$ErrorActionPreference = 'Continue'
$nsys = 'C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.6.3\target-windows-x64\nsys.exe'
$llama = 'D:\Projects\yttri-inference\llama.cpp-bin\llama-server.exe'
$gguf  = 'D:\Models\deepreinforce-ai\Ornith-1.5-9B-GGUF\Ornith-1.5-9B-Q4_K_M.gguf'
$port = 18098
$out = 'D:\Projects\yttri-inference\llama_pf2'
$env:PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin;' + $env:PATH
Get-Process llama-server,nsys -EA SilentlyContinue | Stop-Process -Force -EA SilentlyContinue
Start-Sleep 2
Remove-Item "$out.nsys-rep" -EA SilentlyContinue
$args = @('profile','--force-overwrite=true','--trace=cuda','--sample=none','-o',$out,$llama,
          '-m',$gguf,'--host','127.0.0.1','--port',"$port",'-ngl','999','-c','32768','-fa','on','-np','1','-ctk','f16','-ctv','f16','-a','ornith-1.5-9b','--api-key','benchkey','--no-webui')
$p = Start-Process -FilePath $nsys -ArgumentList $args -WorkingDirectory 'D:\Projects\yttri-inference\llama.cpp-bin' -WindowStyle Hidden -PassThru
$url = "http://127.0.0.1:$port/v1/chat/completions"
$ready = $false
for ($i = 1; $i -le 100; $i++) {
  try { $b = @{model='ornith-1.5-9b';stream=$false;max_tokens=1;messages=@(@{role='user';content='hi'})} | ConvertTo-Json -Depth 5 -Compress
    Invoke-RestMethod $url -Headers @{Authorization="Bearer benchkey"} -Method Post -ContentType 'application/json' -Body $b -TimeoutSec 90 | Out-Null
    $ready = $true; break } catch { Start-Sleep -Seconds 3 }
}
Write-Output "llama under nsys ready=$ready (waited $($i*3)s)"
if ($ready) {
  $sb = New-Object System.Text.StringBuilder
  for ($i = 0; $i -lt $Words; $i++) { [void]$sb.Append("word$i ") }
  $b2 = @{model='ornith-1.5-9b';stream=$false;max_tokens=2;messages=@(@{role='user';content=($sb.ToString() + 'Count from 1 to 5.')});temperature=0} | ConvertTo-Json -Depth 6 -Compress
  [System.IO.File]::WriteAllText("$env:TEMP\nl.json", $b2, [System.Text.UTF8Encoding]::new($false))
  $t = [double](curl.exe -s -X POST $url -H "Content-Type: application/json" -H "Authorization: Bearer benchkey" -d "@$env:TEMP\nl.json" -o "$env:TEMP\nl.out" -w "%{time_total}")
  $o = $null; try { $o = Get-Content "$env:TEMP\nl.out" -Raw | ConvertFrom-Json } catch {}
  if ($o -and $o.timings) { Write-Output ("llama prefill $($o.timings.prompt_n) tok = $([math]::Round($o.timings.prompt_ms/1000,1))s ($([math]::Round($o.timings.prompt_per_second,0)) t/s)") } else { Write-Output "wall $([math]::Round($t,1))s" }
}
Start-Sleep 5
# Убиваем ЦЕЛЬ, а не профайлер: nsys должен сам завершиться и записать отчёт.
Get-Process llama-server -EA SilentlyContinue | Stop-Process -Force -EA SilentlyContinue
Wait-Process -Id $p.Id -Timeout 180 -ErrorAction SilentlyContinue
Start-Sleep 5
Get-Process nsys -EA SilentlyContinue | Stop-Process -Force -EA SilentlyContinue
Start-Sleep 5
if (Test-Path "$out.nsys-rep") {
  & $nsys stats --report cuda_gpu_kern_sum --format csv --force-export=true -o $out "$out.nsys-rep" | Out-Null
  Write-Output "profile saved"
} else { Write-Output 'NO PROFILE' }
