# vram_ab.ps1 - VRAM accounting A/B for qwen36-server (yforge) on yttri-win.
# Per-phase metrics: yforge dedicated/shared GPU memory (WDDM), nvidia-smi total
# used/free, power, SM clock, host RAM available. Restores prod .env at the end.
param(
  [string]$Prompt = 'D:\Projects\yttri-inference\prompt_omega.txt',
  [int]$MaxTokens = 200,
  [string]$Only = ''
)
$ErrorActionPreference = 'Continue'
$root    = 'D:\Projects\yttri-inference'
$outDir  = "$root\logs\vram-ab"
$logPath = "$root\logs\server.log"
$ovrPath = "$root\bench-overrides.env"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$configs = @(
  [pscustomobject]@{ name='q8-128k'; ovr=@('KV_POOL_Q8=1','CTX=128000','CONTEXT_LIMIT=128000','GRAPH_WINDOW=131072','PGRAPH=on','PGRAPH_MIN_T=1000000','PREFILL_CHUNK=8192') },
  [pscustomobject]@{ name='f16-64k'; ovr=@('KV_POOL_Q8=0','CTX=65536','CONTEXT_LIMIT=65536','GRAPH_WINDOW=65536','PGRAPH=on','PGRAPH_MIN_T=1000000','PREFILL_CHUNK=8192') },
  [pscustomobject]@{ name='q8-64k';  ovr=@('KV_POOL_Q8=1','CTX=65536','CONTEXT_LIMIT=65536','GRAPH_WINDOW=65536','PGRAPH=on','PGRAPH_MIN_T=1000000','PREFILL_CHUNK=8192') }
  [pscustomobject]@{ name='q8-128k-c16k'; ovr=@('KV_POOL_Q8=1','CTX=128000','CONTEXT_LIMIT=128000','GRAPH_WINDOW=131072','PGRAPH=on','PGRAPH_MIN_T=1000000','PREFILL_CHUNK=16384') }
  [pscustomobject]@{ name='f16-64k-c16k'; ovr=@('KV_POOL_Q8=0','CTX=65536','CONTEXT_LIMIT=65536','GRAPH_WINDOW=65536','PGRAPH=on','PGRAPH_MIN_T=1000000','PREFILL_CHUNK=16384') }
)
if ($Only -ne '') { $configs = @($configs | Where-Object { $_.name -eq $Only }) }

function ToNum([string]$s) {
  $v = ($s -replace '[^0-9\.\-]','')
  if ($v -eq '' -or $v -eq '-' -or $v -eq '.') { return 0.0 }
  return [double]$v
}

function Start-Sampler([string]$csv, [string]$phaseFile, [int]$maxSec) {
  return Start-Job -ArgumentList $csv,$phaseFile,$maxSec -ScriptBlock {
    param($csv,$phaseFile,$maxSec)
    $rows = New-Object System.Collections.Generic.List[string]
    $t0 = Get-Date
    $i = 0
    while (((Get-Date) - $t0).TotalSeconds -lt $maxSec) {
      $phase = 'init'
      if (Test-Path $phaseFile) { $phase = (Get-Content $phaseFile -Raw -ErrorAction SilentlyContinue).Trim() }
      if ($phase -eq 'stop') { break }
      $ded = 0; $shr = 0
      $p = Get-Process yforge -ErrorAction SilentlyContinue | Sort-Object StartTime -Descending | Select-Object -First 1
      if ($p) {
        $inst = Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -like "pid_$($p.Id)_*" }
        if ($inst) {
          $ded = [int64](($inst | Measure-Object -Property DedicatedUsage -Maximum).Maximum)
          $shr = [int64](($inst | Measure-Object -Property SharedUsage -Maximum).Maximum)
        }
      }
      $ns = (nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu,power.draw,clocks.sm --format=csv,noheader,nounits) -join ''
      $parts = ($ns -replace ' ','') -split ','
      $avail = ''
      if (($i % 4) -eq 0) {
        $mem = Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory -ErrorAction SilentlyContinue
        if ($mem) { $avail = "$($mem.AvailableMBytes)" }
      }
      $t = [math]::Round(((Get-Date) - $t0).TotalSeconds,2)
      $rows.Add(("{0};{1};{2};{3};{4};{5};{6};{7};{8};{9}" -f $t,$phase,$ded,$shr,$parts[0],$parts[1],$parts[2],$parts[3],$parts[4],$avail))
      $i++
      Start-Sleep -Milliseconds 400
    }
    $rows | Set-Content -Path $csv -Encoding ascii
  }
}

function Phase-Stats($rows, $ph) {
  $sel = @($rows | Where-Object { $_.phase -eq $ph })
  if ($sel.Count -eq 0) { return $null }
  return [pscustomobject]@{
    n        = $sel.Count
    dedMax   = [math]::Round((($sel | Measure-Object -Property ded -Maximum).Maximum)/1MB,0)
    shrMax   = [math]::Round((($sel | Measure-Object -Property shr -Maximum).Maximum)/1MB,0)
    usedMax  = ($sel | Measure-Object -Property used -Maximum).Maximum
    freeMin  = ($sel | Measure-Object -Property free -Minimum).Minimum
    powerMax = [math]::Round((($sel | Measure-Object -Property power -Maximum).Maximum),1)
    clockMax = ($sel | Measure-Object -Property clock -Maximum).Maximum
    availMin = ($sel | Where-Object { $_.avail -gt 0 } | Measure-Object -Property avail -Minimum).Minimum
  }
}

foreach ($c in $configs) {
  Write-Output ("=== " + $c.name + " ===")
  Set-Content -Path $ovrPath -Value $c.ovr -Encoding ascii
  $csv = "$outDir\$($c.name).csv"
  $phaseFile = "$outDir\$($c.name).phase"
  Remove-Item $csv,$phaseFile -ErrorAction SilentlyContinue
  Set-Content -Path $phaseFile -Value 'restart' -Encoding ascii
  $job = Start-Sampler $csv $phaseFile 1500
  $t0 = Get-Date
  $restartLine = (& powershell -NoProfile -ExecutionPolicy Bypass -File "$root\scripts\restart_only.ps1" 2>&1 | Select-Object -Last 1)
  $tReady = [math]::Round(((Get-Date) - $t0).TotalSeconds,1)
  Write-Output ("restart " + $restartLine + " (" + $tReady + "s)")
  foreach ($pat in @('\[kv\] paged pool','plan\(dynamic\)','\[vram\] total')) {
    $ln = (Select-String -Path $logPath -Pattern $pat | Select-Object -Last 1)
    if ($ln) { Write-Output ("log " + $ln.Line.Trim()) }
  }
  Set-Content -Path $phaseFile -Value 'idle' -Encoding ascii
  Start-Sleep -Seconds 12
  Set-Content -Path $phaseFile -Value 'req' -Encoding ascii
  $req0 = Get-Date
  $bench = (& powershell -NoProfile -ExecutionPolicy Bypass -File "$root\scripts\bench-live.ps1" -PromptFile $Prompt -MaxTokens $MaxTokens -Runs 1 -Tag $c.name 2>&1 | Where-Object { $_ -like 'BENCH*' }) -join ' '
  $reqSec = [math]::Round(((Get-Date) - $req0).TotalSeconds,1)
  Write-Output ("bench (" + $reqSec + "s) " + $bench)
  Set-Content -Path $phaseFile -Value 'after' -Encoding ascii
  Start-Sleep -Seconds 8
  Set-Content -Path $phaseFile -Value 'stop' -Encoding ascii
  Wait-Job $job -Timeout 40 | Out-Null
  Remove-Job $job -Force -ErrorAction SilentlyContinue

  $rows = Get-Content $csv | Where-Object { $_ -match ';' } | ForEach-Object {
    $f = $_ -split ';'
    [pscustomobject]@{
      t=[double]($f[0] -replace ',','.'); phase=$f[1]; ded=[int64]$f[2]; shr=[int64]$f[3];
      used=[int](ToNum $f[4]); free=[int](ToNum $f[5]);
      power=(ToNum $f[7]); clock=[int](ToNum $f[8]); avail=[int](ToNum $f[9])
    }
  }
  foreach ($ph in @('restart','idle','req','after')) {
    $s = Phase-Stats $rows $ph
    if ($s) {
      Write-Output ("  {0,-7} n={1,3} ded_max={2,5} MiB shr_max={3,4} MiB gpu_used_max={4,5} MiB gpu_free_min={5,5} MiB power_max={6,6} W clock_max={7,4} MHz ram_avail_min={8,6} MB" -f $ph,$s.n,$s.dedMax,$s.shrMax,$s.usedMax,$s.freeMin,$s.powerMax,$s.clockMax,$s.availMin)
    }
  }
  Write-Output ("  csv " + $csv)
}
Remove-Item $ovrPath -ErrorAction SilentlyContinue
$prod = (& powershell -NoProfile -ExecutionPolicy Bypass -File "$root\scripts\restart_only.ps1" 2>&1 | Select-Object -Last 1)
Write-Output ("prod restored: " + $prod)
