param([string]$Kv = '126917', [string]$Qk = '1', [int]$Only = 1)
$root = 'D:\Projects\yttri-inference'
$ncu = 'C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.1.1\ncu.bat'
$exe = "$root\target\release\fa_decode_probe.exe"
$out = "$root\logs\ncu-inst-qk$Qk.txt"
Remove-Item $out -ErrorAction SilentlyContinue
$env:QK_INT8 = $Qk
$metrics = 'smsp__inst_executed_op_shared_ld.sum,smsp__inst_executed_op_shared_st.sum,smsp__inst_executed_op_global_ld.sum,dram__bytes_read.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum,smsp__inst_executed.sum,sm__inst_executed_pipe_tensor_op_hmma.sum,sm__inst_executed_pipe_tensor_op_imma.sum'
& cmd.exe /c "`"$ncu`" --target-processes all --kernel-name regex:flash_fwd_splitkv_kernel --launch-skip 3 --launch-count 1 --metrics $metrics `"$exe`" $Kv 8 0 1 $Only > `"$out`" 2>&1"
Get-Content $out | Select-String -Pattern 'Metric Name|shared_ld|shared_st|global_ld|dram__bytes_read|wavefronts|inst_executed' | ForEach-Object { $_.Line.Trim() }
