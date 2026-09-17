$cuda = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4'
$cuobjdump = Join-Path $cuda 'bin\cuobjdump.exe'
$obj = 'D:\Projects\yttri-inference\fa-build-cache\flash_fwd_splitkv_hdim256_fp16_sm80-7cbd6062ff964c54.o'
$out = 'D:\Projects\yttri-inference\logs\sass-splitkv.txt'
& $cuobjdump -sass $obj > $out 2>&1
$lines = Get-Content $out
Write-Output ("lines=" + $lines.Count)
$ops = @{}
foreach ($l in $lines) {
  if ($l -match '/\*[0-9a-f]{4,}\*/\s+([A-Z][A-Z0-9._]*)') {
    $op = $Matches[1]
    if ($ops.ContainsKey($op)) { $ops[$op]++ } else { $ops[$op] = 1 }
  }
}
$ops.GetEnumerator() | Sort-Object Value -Descending | Select-Object -First 25 | ForEach-Object { "{0,8} {1}" -f $_.Value, $_.Key }
