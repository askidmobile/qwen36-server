param([string]$Prompt = 'D:\Projects\yttri-inference\prompt_50k.txt', [string]$Tag = 'ovr', [string]$Ovr = 'PGRAPH=on', [int]$MaxTokens = 200)
$root = 'D:\Projects\yttri-inference'
Set-Content -Path "$root\bench-overrides.env" -Value ($Ovr -split ';') -Encoding ascii
& powershell -ExecutionPolicy Bypass -File "$root\scripts\restart_only.ps1" | Select-Object -Last 1 | Out-Null
$log = "$root\logs\server.log"
$before = 0
if (Test-Path $log) { $before = (Get-Content $log).Count }
& powershell -ExecutionPolicy Bypass -File "$root\scripts\bench-live.ps1" -PromptFile $Prompt -MaxTokens $MaxTokens -Runs 1 -Tag $Tag 2>&1 | Where-Object { $_ -like 'BENCH*' }
Get-Content $log | Select-Object -Skip $before | Where-Object { $_ -like '*chunk T=*' -or $_ -like '*paged pool*' }
