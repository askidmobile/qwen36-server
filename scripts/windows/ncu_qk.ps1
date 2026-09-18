$ErrorActionPreference = 'Continue'
$root = 'D:\Projects\yttri-inference'
$ncu = 'C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.1.1\ncu.bat'
$exe = "$root\target\release\fa_decode_probe.exe"
$kv = if ($args.Count -ge 1) { $args[0] } else { '30232' }
foreach ($cfg in @(@{n='q8-qkoff'; env='0'; only=1}, @{n='q8-qkon'; env='1'; only=1}, @{n='f16'; env='0'; only=0})) {
  $out = "$root\logs\ncu-qk-$($cfg.n)-$kv.txt"
  Remove-Item $out -ErrorAction SilentlyContinue
  $env:QK_INT8 = $cfg.env
  & cmd.exe /c "`"$ncu`" --target-processes all --kernel-name regex:flash_fwd_splitkv_kernel --launch-skip 3 --launch-count 1 --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section Occupancy `"$exe`" $kv 8 0 1 $($cfg.only) > `"$out`" 2>&1"
  Write-Output "=== $($cfg.n) kv=$kv ==="
  Select-String -Path $out -Pattern 'Duration|Compute \(SM\)|Memory Throughput|Achieved Occupancy|Achieved Active Warps|Issued Warp Per Scheduler|No Eligible|Warp Cycles Per Issued|Stall Barrier|Stall Long Scoreboard|Stall Wait|Stall Short Scoreboard|Stall Not Selected|Stall MIO Throttle|Stall Dispatch|Stall Math|Stall LG Throttle|Stall Drain' | ForEach-Object { $_.Line.Trim() } | Select-Object -First 26
}
