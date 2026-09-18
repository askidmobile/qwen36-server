param([int]$Pairs = 2)
$ErrorActionPreference = 'Continue'
$root = 'D:\Projects\yttri-inference'
$ovr = "$root\bench-overrides.env"
$f16 = @('KV_POOL_Q8=0','CTX=128000','CONTEXT_LIMIT=128000','GRAPH_WINDOW=131072','PGRAPH=on','PGRAPH_MIN_T=1000000','PREFILL_CHUNK=8192')
$q8  = @('KV_POOL_Q8=1','CTX=128000','CONTEXT_LIMIT=128000','GRAPH_WINDOW=131072','PGRAPH=on','PGRAPH_MIN_T=1000000','PREFILL_CHUNK=8192','QK_INT8=1')
foreach ($pair in 1..$Pairs) {
  foreach ($cfg in @(@{n='f16-128k'; v=$f16}, @{n='q8-128k'; v=$q8})) {
    Set-Content -Path $ovr -Value $cfg.v -Encoding ascii
    $r = & powershell -NoProfile -ExecutionPolicy Bypass -File "$root\scripts\restart_only.ps1" | Select-Object -Last 1
    Write-Output ("[$($cfg.n) pair$pair] " + $r)
    & powershell -NoProfile -ExecutionPolicy Bypass -File "$root\scripts\bench-live.ps1" -PromptFile "$root\prompt_128k.txt" -MaxTokens 100 -Runs 1 -Tag "$($cfg.n)-127k-p$pair" 2>&1 | Where-Object { $_ -like 'BENCH*' } | ForEach-Object { Write-Output $_ }
  }
}
Remove-Item $ovr -ErrorAction SilentlyContinue
& powershell -NoProfile -ExecutionPolicy Bypass -File "$root\scripts\restart_only.ps1" | Select-Object -Last 1
