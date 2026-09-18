param([int]$Runs = 2, [int]$Pairs = 2)
$ErrorActionPreference = 'Continue'
$root = 'D:\Projects\yttri-inference'
$ovr = "$root\bench-overrides.env"
foreach ($pair in 1..$Pairs) {
  foreach ($cfg in @(@{n='off'; v='QK_INT8=0'}, @{n='on'; v='QK_INT8=1'})) {
    Set-Content -Path $ovr -Value @($cfg.v) -Encoding ascii
    $r = & powershell -NoProfile -ExecutionPolicy Bypass -File "$root\scripts\restart_only.ps1" | Select-Object -Last 1
    Write-Output ("[qk $($cfg.n) pair$pair] " + $r)
    for ($i = 1; $i -le $Runs; $i++) {
      & powershell -NoProfile -ExecutionPolicy Bypass -File "$root\scripts\bench-live.ps1" -PromptFile "$root\prompt_omega.txt" -MaxTokens 200 -Runs 1 -Tag "qk$($cfg.n)-p$pair-$i" 2>&1 | Where-Object { $_ -like 'BENCH*' } | ForEach-Object { Write-Output $_ }
    }
  }
}
Remove-Item $ovr -ErrorAction SilentlyContinue
& powershell -NoProfile -ExecutionPolicy Bypass -File "$root\scripts\restart_only.ps1" | Select-Object -Last 1
