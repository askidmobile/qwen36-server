$ErrorActionPreference = 'Continue'
$root = 'D:\Projects\yttri-inference'
$ncu = 'C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.1.1\ncu.bat'
$exe = "$root\target\release\fa_decode_probe.exe"
$m = 'sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active,sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active,sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active,sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active,sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active,smsp__inst_executed.sum,smsp__sass_inst_executed_op_shared_ld.sum,smsp__sass_inst_executed_op_shared_st.sum,smsp__sass_inst_executed_op_global_ld.sum'
foreach ($cfg in @(@{n='f16'; only=0}, @{n='q8'; only=1})) {
  $out = "$root\logs\ncu-pipes-$($cfg.n).txt"
  Remove-Item $out -ErrorAction SilentlyContinue
  & cmd.exe /c "`"$ncu`" --target-processes all --kernel-name regex:flash_fwd_splitkv_kernel --launch-skip 3 --launch-count 1 --metrics $m $exe 30232 6 0 1 $($cfg.only) > `"$out`" 2>&1"
  Write-Output "=== $($cfg.n) ==="
  Select-String -Path $out -Pattern 'inst_executed_pipe|inst_executed\.sum|op_shared_ld|op_shared_st|op_global_ld' | ForEach-Object { $_.Line.Trim() } | Select-Object -First 12
}
