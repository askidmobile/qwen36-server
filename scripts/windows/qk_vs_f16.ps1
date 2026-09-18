param([int]$Runs = 2, [int]$Pairs = 2)
$ErrorActionPreference = 'Continue'
$root = 'D:\Projects\yttri-inference'
$ovr = "$root\bench-overrides.env"
$f16 = @('KV_POOL_Q8=0','CTX=65536','CONTEXT_LIMIT=65536','GRAPH_WINDOW=65536','PGRAPH=on','PGRAPH_MIN_T=1000000','PREFILL_CHUNK=8192')
$q8  = @('KV_POOL_Q8=1','CTX=128000','CONTEXT_LIMIT=128000','GRAPH_WINDOW=131072','PGRAPH=on','PGRAPH_MIN_T=1000000','PREFILL_CHUNK=8192','QK_INT8=1')
foreach ($pair in 1..$Pairs) {
  foreach ($cfg in @(@{n='f16-64k'; v=$f16}, @{n='q8-128k-int8qk'; v=$q8})) {
    Set-Content -Path $ovr -Value $cfg.v -Encoding ascii
    $r = & powershell -NoProfile -ExecutionPolicy Bypass -File "$root\scripts\restart_only.ps1" | Select-Object -Last 1
    Write-Output ("[$($cfg.n) pair$pair] " + $r)
    for ($i = 1; $i -le $Runs; $i++) {
      & powershell -NoProfile -ExecutionPolicy Bypass -File "$root\scripts\bench-live.ps1" -PromptFile "$root\prompt_omega.txt" -MaxTokens 200 -Runs 1 -Tag "$($cfg.n)-p$pair-$i" 2>&1 | Where-Object { $_ -like 'BENCH*' } | ForEach-Object { Write-Output $_ }
    }
  }
}
Remove-Item $ovr -ErrorAction SilentlyContinue
& powershell -NoProfile -ExecutionPolicy Bypass -File "$root\scripts\restart_only.ps1" | Select-Object -Last 1
