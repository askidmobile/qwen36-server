param([string]$Prompt = 'D:\Projects\yttri-inference\prompt_omega.txt', [string]$Tag = 'run', [int]$MaxTokens = 200)
$root = 'D:\Projects\yttri-inference'
$log = "$root\logs\server.log"
$before = 0
if (Test-Path $log) { $before = (Get-Content $log).Count }
& powershell -ExecutionPolicy Bypass -File "$root\scripts\bench-live.ps1" -PromptFile $Prompt -MaxTokens $MaxTokens -Runs 1 -Tag $Tag 2>&1 | Where-Object { $_ -like 'BENCH*' }
if (Test-Path $log) { Get-Content $log | Select-Object -Skip $before | Where-Object { $_ -like '*chunk T=*' -or $_ -like '*scheduler step error*' -or $_ -like '*paged pool*' } }
