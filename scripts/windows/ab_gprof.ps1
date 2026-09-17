$ErrorActionPreference = 'Continue'
$root = 'D:\Projects\yttri-inference'
$pf = "$root\prompt_omega.txt"
$log = "$root\logs\server.log"
$cfgs = @(
  @{ name = 'f16'; ovr = @('KV_POOL_Q8=0','CTX=65536','GRAPH_WINDOW=65536','PGRAPH=off','PREFILL_CHUNK=16384','GPROF=1') },
  @{ name = 'q8';  ovr = @('KV_POOL_Q8=1','CTX=65536','GRAPH_WINDOW=65536','PGRAPH=off','PREFILL_CHUNK=16384','GPROF=1') }
)
foreach ($c in $cfgs) {
  Set-Content -Path "$root\bench-overrides.env" -Value $c.ovr -Encoding ascii
  $before = (Get-Content $log).Count
  & powershell -ExecutionPolicy Bypass -File "$root\scripts\restart_only.ps1" | Select-Object -Last 1 | Out-Null
  $b = & powershell -ExecutionPolicy Bypass -File "$root\scripts\bench-live.ps1" -PromptFile $pf -MaxTokens 200 -Runs 1 -Tag $c.name 2>&1 | Where-Object { $_ -like 'BENCH*' }
  Write-Output ("[" + $c.name + "] " + $b)
  Get-Content $log | Select-Object -Skip $before | Where-Object { $_ -like '*gGPU*' } | Select-Object -First 12
}
Remove-Item "$root\bench-overrides.env" -ErrorAction SilentlyContinue
