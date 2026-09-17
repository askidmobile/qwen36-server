$ErrorActionPreference = 'Continue'
$root = 'D:\Projects\yttri-inference'
$ncu = 'C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.1.1\ncu.bat'
$exe = "$root\target\release\fa_decode_probe.exe"
foreach ($cfg in @(@{n='f16'; only=0}, @{n='q8'; only=1})) {
  $out = "$root\logs\ncu-probe2-$($cfg.n).txt"
  Remove-Item $out -ErrorAction SilentlyContinue
  & cmd.exe /c "`"$ncu`" --target-processes all --kernel-name regex:flash_fwd_splitkv_kernel --launch-skip 3 --launch-count 1 --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section Occupancy $exe 30232 6 0 1 $($cfg.only) > `"$out`" 2>&1"
  Write-Output "=== $($cfg.n) ==="
  Select-String -Path $out -Pattern 'Duration|Compute \(SM\)|Memory Throughput|Achieved Occupancy|Achieved Active Warps Per SM|Block Limit Shared Mem|Block Limit Registers|Block Limit SM|Issued Warp Per Scheduler|No Eligible|Warp Cycles Per Issued Instruction|Stall Barrier|Stall Long Scoreboard|Stall Wait|Stall Short Scoreboard|Stall Not Selected|Stall MIO Throttle' | ForEach-Object { $_.Line.Trim() } | Select-Object -First 20
}
