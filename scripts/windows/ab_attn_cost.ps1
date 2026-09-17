$ErrorActionPreference = 'Continue'
$root = 'D:\Projects\yttri-inference'
$pf = "$root\prompt_omega.txt"
$cfgs = @(
  @{ name = 'f16-attn-on';  ovr = @('KV_POOL_Q8=0','CTX=65536','GRAPH_WINDOW=65536','PGRAPH=off','PREFILL_CHUNK=16384','GRAPH_SKIP_ATTN=0') },
  @{ name = 'f16-attn-off'; ovr = @('KV_POOL_Q8=0','CTX=65536','GRAPH_WINDOW=65536','PGRAPH=off','PREFILL_CHUNK=16384','GRAPH_SKIP_ATTN=1') },
  @{ name = 'q8-attn-on';   ovr = @('KV_POOL_Q8=1','CTX=65536','GRAPH_WINDOW=65536','PGRAPH=off','PREFILL_CHUNK=16384','GRAPH_SKIP_ATTN=0') },
  @{ name = 'q8-attn-off';  ovr = @('KV_POOL_Q8=1','CTX=65536','GRAPH_WINDOW=65536','PGRAPH=off','PREFILL_CHUNK=16384','GRAPH_SKIP_ATTN=1') }
)
foreach ($c in $cfgs) {
  Set-Content -Path "$root\bench-overrides.env" -Value $c.ovr -Encoding ascii
  & powershell -ExecutionPolicy Bypass -File "$root\scripts\restart_only.ps1" | Select-Object -Last 1 | Out-Null
  $b = & powershell -ExecutionPolicy Bypass -File "$root\scripts\bench-live.ps1" -PromptFile $pf -MaxTokens 200 -Runs 1 -Tag $c.name 2>&1 | Where-Object { $_ -like 'BENCH*' }
  Write-Output ("[" + $c.name + "] " + $b)
}
Remove-Item "$root\bench-overrides.env" -ErrorAction SilentlyContinue
