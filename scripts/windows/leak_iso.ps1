param([string]$Extra = '', [string]$Tag = 'iso', [int]$Repeats = 3)
$ErrorActionPreference = 'Continue'
$root = 'D:\Projects\yttri-inference'
$ovr = "$root\bench-overrides.env"
$base = @('KV_POOL_Q8=1','CTX=128000','CONTEXT_LIMIT=128000','GRAPH_WINDOW=131072','PGRAPH=on','PGRAPH_MIN_T=1000000','PREFILL_CHUNK=8192','QK_INT8=1','QK_INT8_PREFILL=1')
$lines = if ($Extra -eq '') { $base } else { $base + ($Extra -split ';') }
Set-Content -Path $ovr -Value $lines -Encoding ascii
& powershell -NoProfile -ExecutionPolicy Bypass -File "$root\scripts\restart_only.ps1" | Select-Object -Last 1 | Out-Null
$v0 = (nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -join ''
Write-Output ("[$Tag] after-load VRAM=$v0 MiB  cfg=" + ($lines -join ';'))
for ($i = 1; $i -le $Repeats; $i++) {
  & powershell -NoProfile -ExecutionPolicy Bypass -File "$root\scripts\bench-live.ps1" -PromptFile "$root\prompt_omega.txt" -MaxTokens 8 -Runs 1 -Tag "$Tag-r$i" 2>&1 | Where-Object { $_ -like 'BENCH*' } | ForEach-Object { Write-Output $_ }
  $v = (nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -join ''
  Write-Output ("[$Tag] after-r$i VRAM=$v MiB")
}
Remove-Item $ovr -ErrorAction SilentlyContinue
& powershell -NoProfile -ExecutionPolicy Bypass -File "$root\scripts\restart_only.ps1" | Select-Object -Last 1
