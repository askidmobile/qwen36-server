param([string]$PromptFile='D:\Projects\yttri-inference\prompt_omega.txt', [int]$MaxTokens=200, [int]$Reps=1, [string]$Ctk='q8_0', [string]$Ctv='q8_0', [int]$Ctx=131072)
$ErrorActionPreference='Continue'
$llama='D:\Projects\yttri-inference\llama.cpp-bin\llama-server.exe'
$gguf='D:\Models\deepreinforce-ai\Ornith-1.5-9B-GGUF\Ornith-1.5-9B-Q4_K_M.gguf'
$port=18098; $url="http://127.0.0.1:$port/v1/chat/completions"; $key='benchkey'
$env:PATH='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin;'+$env:PATH
Stop-ScheduledTask -TaskName qwen36-inference -ErrorAction SilentlyContinue | Out-Null
Start-Sleep 2
Get-Process yforge -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep 3
$sargs="-m `"$gguf`" --host 127.0.0.1 --port $port -ngl 999 -c $Ctx -fa on -np 1 -ctk $Ctk -ctv $Ctv -a ornith-1.5-9b --api-key $key --no-webui"
$p=Start-Process -FilePath $llama -ArgumentList $sargs -WorkingDirectory 'D:\Projects\yttri-inference\llama.cpp-bin' -RedirectStandardOutput 'D:\Projects\yttri-inference\logs\hc-llama.out' -RedirectStandardError 'D:\Projects\yttri-inference\logs\hc-llama.err' -WindowStyle Hidden -PassThru
$ready=$false
for($i=1;$i -le 120;$i++){ try{ Invoke-RestMethod $url -Headers @{Authorization="Bearer $key"} -Method Post -ContentType 'application/json' -Body (@{model='ornith-1.5-9b';stream=$false;max_tokens=1;messages=@(@{role='user';content='hi'})}|ConvertTo-Json -Depth 5 -Compress) -TimeoutSec 180 | Out-Null; $ready=$true; break } catch { Start-Sleep 3 } }
Write-Output "llama($Ctk/$Ctv ctx=$Ctx) ready=$ready"
if($ready){
  $prompt=[System.IO.File]::ReadAllText($PromptFile)
  $enc=New-Object System.Text.UTF8Encoding($false)
  $body=@{model='ornith-1.5-9b';stream=$false;max_tokens=$MaxTokens;temperature=0.0;messages=@(@{role='user';content=$prompt})}|ConvertTo-Json -Depth 6 -Compress
  [System.IO.File]::WriteAllText("$env:TEMP\llfile.json",$body,$enc)
  for($r=1;$r -le $Reps;$r++){
    $sw=[System.Diagnostics.Stopwatch]::StartNew()
    curl.exe -s -X POST $url -H 'Content-Type: application/json' -H "Authorization: Bearer $key" --data-binary "@$env:TEMP\llfile.json" -o "$env:TEMP\llfile.out" | Out-Null
    $sw.Stop()
    $o=$null; try{$o=Get-Content "$env:TEMP\llfile.out" -Raw|ConvertFrom-Json}catch{}
    if($o -and $o.timings){ Write-Output ("LLAMA[$Ctk] run=$r prompt_n=$($o.timings.prompt_n) prefill=$([math]::Round($o.timings.prompt_ms/1000,2))s ($([math]::Round($o.timings.prompt_per_second,0)) t/s) decode=$([math]::Round($o.timings.predicted_per_second,2)) t/s wall=$([math]::Round($sw.Elapsed.TotalSeconds,2))s") } else { Write-Output "run=$r no timings" }
  }
  $vram = (nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  Write-Output "llama vram_used=$vram MiB"
}
Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
Start-Sleep 3
Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Write-Output "llama done"
