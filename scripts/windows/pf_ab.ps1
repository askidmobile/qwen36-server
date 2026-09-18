$ErrorActionPreference = 'Continue'
$root = 'D:\Projects\yttri-inference'
$ovr = "$root\bench-overrides.env"
function Run-One([string]$name, [string[]]$lines, [string]$prompt) {
  Set-Content -Path $ovr -Value $lines -Encoding ascii
  & powershell -NoProfile -ExecutionPolicy Bypass -File "$root\scripts\restart_only.ps1" | Select-Object -Last 1 | Out-Null
  & powershell -NoProfile -ExecutionPolicy Bypass -File "$root\scripts\bench-live.ps1" -PromptFile "$root\$prompt" -MaxTokens 100 -Runs 1 -Tag $name 2>&1 | Where-Object { $_ -like 'BENCH*' } | ForEach-Object { Write-Output $_ }
}
Run-One 'pf-off-30k'  @('QK_INT8=1') 'prompt_omega.txt'
Run-One 'pf-on-30k'   @('QK_INT8=1','QK_INT8_PREFILL=1') 'prompt_omega.txt'
Run-One 'pf-off-127k' @('QK_INT8=1') 'prompt_128k.txt'
Run-One 'pf-on-127k'  @('QK_INT8=1','QK_INT8_PREFILL=1') 'prompt_128k.txt'
Remove-Item $ovr -ErrorAction SilentlyContinue
& powershell -NoProfile -ExecutionPolicy Bypass -File "$root\scripts\restart_only.ps1" | Select-Object -Last 1
