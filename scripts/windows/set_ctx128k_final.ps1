$ErrorActionPreference = 'Stop'
$root = 'D:\Projects\yttri-inference'
$p = "$root\qwen36-server\.env"
Copy-Item $p "$p.bak-20260917-final-128k" -Force
$map = @{
  'CTX' = '128000'
  'CONTEXT_LIMIT' = '128000'
  'KV_POOL_Q8' = '1'
  'GRAPH_WINDOW' = '131072'
  'PGRAPH' = 'on'
  'PGRAPH_MIN_T' = '1000000'
  'PREFILL_CHUNK' = '8192'
  'PREFIX_CACHE_MIB' = '2048'
  'PREFIX_CACHE_POOL_BACKED' = '1'
}
$lines = [System.IO.File]::ReadAllLines($p)
$out = New-Object System.Collections.Generic.List[string]
$seen = @{}
foreach ($line in $lines) {
  $eq = $line.IndexOf('=')
  if ($eq -gt 0) {
    $k = $line.Substring(0, $eq).Trim()
    if ($map.ContainsKey($k)) { $out.Add("$k=$($map[$k])"); $seen[$k] = $true; continue }
  }
  $out.Add($line)
}
foreach ($k in $map.Keys) { if (-not $seen.ContainsKey($k)) { $out.Add("$k=$($map[$k])") } }
[System.IO.File]::WriteAllLines($p, [string[]]$out)
Remove-Item "$root\bench-overrides.env" -ErrorAction SilentlyContinue
$out | Where-Object { $_ -match '^(CTX|CONTEXT_LIMIT|KV_POOL_Q8|GRAPH_WINDOW|PGRAPH|PGRAPH_MIN_T|PREFILL_CHUNK|PREFIX_CACHE_MIB|PREFIX_CACHE_POOL_BACKED|KV_CACHE_DTYPE)=' }
