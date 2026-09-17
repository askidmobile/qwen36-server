$ErrorActionPreference = 'Continue'
$root = 'D:\Projects\yttri-inference'
Remove-Item "$root\bench-overrides.env" -ErrorAction SilentlyContinue
for ($i = 1; $i -le 3; $i++) {
  & powershell -ExecutionPolicy Bypass -File "$root\scripts\restart_only.ps1" | Select-Object -Last 1 | Out-Null
  $b = & powershell -ExecutionPolicy Bypass -File "$root\scripts\bench-live.ps1" -PromptFile "$root\prompt_omega.txt" -MaxTokens 200 -Runs 1 -Tag ("ours" + $i) 2>&1 | Where-Object { $_ -like 'BENCH*' }
  Write-Output ("[ours$i] " + $b)
  $l = & powershell -ExecutionPolicy Bypass -File "$root\scripts\bench_llama_q8.ps1" -PromptFile "$root\prompt_omega.txt" -MaxTokens 200 -Reps 1 2>&1 | Where-Object { $_ -like 'LLAMA*' }
  Write-Output ("[llama$i] " + $l)
}
