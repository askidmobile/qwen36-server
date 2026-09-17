$ErrorActionPreference = 'Continue'
$root = 'D:\Projects\yttri-inference'
$pf = "$root\prompt_omega.txt"
$cfgs = @(
  @{ name = 'f16-64k-prod'; ovr = @('KV_POOL_Q8=0','CTX=65536','CONTEXT_LIMIT=65536','GRAPH_WINDOW=65536','PGRAPH=off','PREFILL_CHUNK=16384','PGRAPH_MIN_T=8') },
  @{ name = 'q8-64k-pgoff'; ovr = @('KV_POOL_Q8=1','CTX=65536','CONTEXT_LIMIT=65536','GRAPH_WINDOW=65536','PGRAPH=off','PREFILL_CHUNK=16384') },
  @{ name = 'q8-128k-pgon'; ovr = @('KV_POOL_Q8=1','CTX=128000','CONTEXT_LIMIT=128000','GRAPH_WINDOW=131072','PGRAPH=on','PGRAPH_MIN_T=1000000','PREFILL_CHUNK=8192') }
)
foreach ($c in $cfgs) {
  Set-Content -Path "$root\bench-overrides.env" -Value $c.ovr -Encoding ascii
  & powershell -ExecutionPolicy Bypass -File "$root\scripts\restart_only.ps1" | Select-Object -Last 1 | Out-Null
  for ($r = 1; $r -le 2; $r++) {
    $b = & powershell -ExecutionPolicy Bypass -File "$root\scripts\bench-live.ps1" -PromptFile $pf -MaxTokens 200 -Runs 1 -Tag ("$($c.name)-r$r") 2>&1 | Where-Object { $_ -like 'BENCH*' }
    Write-Output ("[" + $c.name + " r" + $r + "] " + $b)
  }
}
Remove-Item "$root\bench-overrides.env" -ErrorAction SilentlyContinue
