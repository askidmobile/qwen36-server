param([int]$Window = 65536)
$ErrorActionPreference = 'Continue'
$root = 'D:\Projects\yttri-inference'
Set-Content -Path "$root\bench-overrides.env" -Value @("PGRAPH=on", "GRAPH_WINDOW=$Window") -Encoding ascii
$log = "$root\logs\server.log"
$before = 0
if (Test-Path $log) { $before = (Get-Content $log).Count }
& powershell -ExecutionPolicy Bypass -File "$root\scripts\restart_only.ps1" | Select-Object -Last 1 | Out-Null
& powershell -ExecutionPolicy Bypass -File "$root\scripts\bench-live.ps1" -PromptFile "$root\prompt_omega.txt" -MaxTokens 200 -Runs 1 -Tag "win$Window" 2>&1 | Where-Object { $_ -like 'BENCH*' }
if (Test-Path $log) { Get-Content $log | Select-Object -Skip $before | Where-Object { $_ -like '*chunk T=*' } }
