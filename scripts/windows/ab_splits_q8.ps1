$ErrorActionPreference = 'Continue'
$root = 'D:\Projects\yttri-inference'
$pf = "$root\prompt_omega.txt"
$vals = @(0, 2, 4, 7, 14, 28)
foreach ($v in $vals) {
  $ovr = @('KV_POOL_Q8=1','CTX=65536','GRAPH_WINDOW=65536','PGRAPH=off','PREFILL_CHUNK=16384','DECODE_GQA_FOLD=0')
  if ($v -gt 0) { $ovr += "QWEN36_FA_SPLITS=$v" }
  Set-Content -Path "$root\bench-overrides.env" -Value $ovr -Encoding ascii
  & powershell -ExecutionPolicy Bypass -File "$root\scripts\restart_only.ps1" | Select-Object -Last 1 | Out-Null
  $b = & powershell -ExecutionPolicy Bypass -File "$root\scripts\bench-live.ps1" -PromptFile $pf -MaxTokens 200 -Runs 1 -Tag ("q8-s$v") 2>&1 | Where-Object { $_ -like 'BENCH*' }
  Write-Output ("[q8 splits=$v] " + $b)
}
Remove-Item "$root\bench-overrides.env" -ErrorAction SilentlyContinue
