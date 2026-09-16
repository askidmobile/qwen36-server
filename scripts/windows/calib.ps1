$ErrorActionPreference = 'Continue'
$root = 'D:\Projects\yttri-inference'
$env:PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin;' + $env:PATH
$gguf = 'D:\Models\deepreinforce-ai\Ornith-1.5-9B-GGUF\Ornith-1.5-9B-Q4_K_M.gguf'

# --- llama.cpp: чистый prefill без внимания на длинном контексте ---
Stop-ScheduledTask -TaskName qwen36-inference -EA SilentlyContinue | Out-Null
Start-Sleep 2
Get-Process yforge,llama-server -EA SilentlyContinue | Stop-Process -Force -EA SilentlyContinue
Start-Sleep 3
$lb = 'D:\Projects\yttri-inference\llama.cpp-bin\llama-bench.exe'
& $lb -m $gguf -p 512,2048,4096 -n 0 -ngl 999 -fa 1 -ctk f16 -ctv f16 -r 2 2>&1 | Select-String -Pattern 'pp512|pp2048|pp4096|model|backend' | ForEach-Object { $_.Line }
