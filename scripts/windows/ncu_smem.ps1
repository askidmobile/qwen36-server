$ErrorActionPreference = 'Continue'
$root = 'D:\Projects\yttri-inference'
$ncu = 'C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.1.1\ncu.bat'
$exe = "$root\target\release\fa_decode_probe.exe"
$m = 'launch__shared_mem_per_block_dynamic,launch__shared_mem_per_block_static,launch__registers_per_thread,launch__occupancy_limit_shared_mem,launch__occupancy_limit_registers,launch__waves_per_multiprocessor,gpu__time_duration.sum'
foreach ($cfg in @(@{n='f16'; only=0}, @{n='q8'; only=1})) {
  $out = "$root\logs\ncu-smem-$($cfg.n).txt"
  Remove-Item $out -ErrorAction SilentlyContinue
  & cmd.exe /c "`"$ncu`" --target-processes all --kernel-name regex:flash_fwd_splitkv_kernel --launch-skip 3 --launch-count 1 --metrics $m $exe 30232 6 0 1 $($cfg.only) > `"$out`" 2>&1"
  Write-Output "=== $($cfg.n) ==="
  Select-String -Path $out -Pattern 'shared_mem|registers_per_thread|occupancy_limit|waves_per|time_duration' | ForEach-Object { $_.Line.Trim() } | Select-Object -First 10
}
