param([string]$Kv = '126917', [string]$Qk = '1', [int]$Only = 1)
$root = 'D:\Projects\yttri-inference'
$ncu = 'C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.1.1\ncu.bat'
$exe = "$root\target\release\fa_decode_probe.exe"
$out = "$root\logs\ncu-stall-qk$Qk.txt"
Remove-Item $out -ErrorAction SilentlyContinue
$env:QK_INT8 = $Qk
$metrics = 'smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,smsp__warp_issue_stalled_wait_per_warp_active.pct,smsp__warp_issue_stalled_not_selected_per_warp_active.pct,smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct,smsp__warp_issue_stalled_drain_per_warp_active.pct,smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct,smsp__warp_issue_stalled_selected_per_warp_active.pct,smsp__warp_issue_stalled_branch_resolving_per_warp_active.pct'
& cmd.exe /c "`"$ncu`" --target-processes all --kernel-name regex:flash_fwd_splitkv_kernel --launch-skip 3 --launch-count 1 --metrics $metrics `"$exe`" $Kv 8 0 1 $Only > `"$out`" 2>&1"
Get-Content $out | Select-String -Pattern 'stalled|Metric Name' | ForEach-Object { $_.Line.Trim() }
