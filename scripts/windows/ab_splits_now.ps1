$ErrorActionPreference = 'Continue'
$root = 'D:\Projects\yttri-inference'
$pf = "$root\prompt_omega.txt"
foreach ($v in @(0, 8, 12, 16, 24, 32)) {
  $ovr = @('KV_POOL_Q8=1','CTX=128000','CONTEXT_LIMIT=128000','GRAPH_WINDOW=131072','PGRAPH=on','PGRAPH_MIN_T=1000000','PREFILL_CHUNK=8192')
  if ($v -gt 0) { $ovr += "QWEN36_FA_SPLITS=$v" }
  Set-Content -Path "$root\bench-overrides.env" -Value $ovr -Encoding ascii
  & powershell -ExecutionPolicy Bypass -File "$root\scripts\restart_only.ps1" | Select-Object -Last 1 | Out-Null
  $b = & powershell -ExecutionPolicy Bypass -File "$root\scripts\bench-live.ps1" -PromptFile $pf -MaxTokens 200 -Runs 1 -Tag ("q8-s$v") 2>&1 | Where-Object { $_ -like 'BENCH*' }
  Write-Output ("[splits=$v] " + $b)
}
Remove-Item "$root\bench-overrides.env" -ErrorAction SilentlyContinue
