param([string]$Prompt, [string]$Tag, [int]$MaxTokens = 64, [string]$Ovr = '')
$root = 'D:\Projects\yttri-inference'
if ($Ovr -ne '') { Set-Content -Path "$root\bench-overrides.env" -Value ($Ovr -split ';') -Encoding ascii }
else { Remove-Item "$root\bench-overrides.env" -ErrorAction SilentlyContinue }
& powershell -ExecutionPolicy Bypass -File "$root\scripts\restart_only.ps1" | Select-Object -Last 1 | Out-Null
& powershell -ExecutionPolicy Bypass -File "$root\scripts\bench-live.ps1" -PromptFile $Prompt -MaxTokens $MaxTokens -Runs 1 -Tag $Tag 2>&1 | Where-Object { $_ -like 'BENCH*' }
