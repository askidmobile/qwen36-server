param([int]$T = 16384, [int]$Iters = 5, [string]$Variants = '0,1,2,3,4,5', [int]$H = 16, [int]$Hk = 4)
$ErrorActionPreference = 'Continue'
$exe = 'D:\Projects\yttri-inference\target\release\fa_prefill_probe.exe'
foreach ($v in $Variants.Split(',')) {
  $env:QWEN36_FA_PREFILL_TILE = $v
  $out = & $exe $T $Iters 2>&1 | Select-String -Pattern 'dense|TFLOPS|ms' | ForEach-Object { $_.Line }
  Write-Output ("TILE=$v :: " + ($out -join ' | '))
}
Remove-Item Env:QWEN36_FA_PREFILL_TILE -ErrorAction SilentlyContinue
