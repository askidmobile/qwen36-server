param(
  [string]$Ovr = '',
  [string]$Prompt = 'D:\Projects\yttri-inference\prompt_omega.txt',
  [int]$MaxTokens = 200,
  [string]$Tag = 'mem'
)
$ErrorActionPreference = 'Continue'
$root = 'D:\Projects\yttri-inference'
$csv = "$root\logs\gpu-mem-$Tag.csv"
$stopFile = "$csv.stop"
Remove-Item $csv, $stopFile -ErrorAction SilentlyContinue
$t0 = Get-Date
$job = Start-Job -ArgumentList $csv, $stopFile, $t0 -ScriptBlock {
  param($csv, $stopFile, $t0)
  $lines = New-Object System.Collections.Generic.List[string]
  while (-not (Test-Path $stopFile)) {
    $p = Get-Process yforge -ErrorAction SilentlyContinue
    $ded = 0; $shr = 0
    if ($p) {
      $inst = Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory -ErrorAction SilentlyContinue |
              Where-Object { $_.Name -like "pid_$($p.Id)_*" }
      if ($inst) {
        $ded = [int64](($inst | Measure-Object -Property DedicatedUsage -Maximum).Maximum)
        $shr = [int64](($inst | Measure-Object -Property SharedUsage -Maximum).Maximum)
      }
    }
    $ns = (nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits) -join ''
    $parts = ($ns -replace ' ', '') -split ','
    $t = ((Get-Date) - $t0).TotalSeconds
    $lines.Add(("{0:N1};{1};{2};{3};{4}" -f $t, $ded, $shr, $parts[0], $parts[1]))
    Start-Sleep -Milliseconds 400
  }
  $lines | Set-Content -Path $csv -Encoding ascii
}
# phase marker: idle before restart
Start-Sleep -Seconds 3
$tRestart = ((Get-Date) - $t0).TotalSeconds
if ($Ovr -ne '') { Set-Content -Path "$root\bench-overrides.env" -Value ($Ovr -split ';') -Encoding ascii }
else { Remove-Item "$root\bench-overrides.env" -ErrorAction SilentlyContinue }
& powershell -ExecutionPolicy Bypass -File "$root\scripts\restart_only.ps1" | Select-Object -Last 1 | Out-Null
Start-Sleep -Seconds 4
$tIdle = ((Get-Date) - $t0).TotalSeconds
$bench = & powershell -ExecutionPolicy Bypass -File "$root\scripts\bench-live.ps1" -PromptFile $Prompt -MaxTokens $MaxTokens -Runs 1 -Tag $Tag 2>&1 | Where-Object { $_ -like 'BENCH*' }
$tEnd = ((Get-Date) - $t0).TotalSeconds
Start-Sleep -Seconds 3
New-Item -ItemType File -Path $stopFile -Force | Out-Null
Wait-Job $job -Timeout 30 | Out-Null
Remove-Job $job -Force -ErrorAction SilentlyContinue
Remove-Item $stopFile -ErrorAction SilentlyContinue
$rows = Get-Content $csv | Where-Object { $_ -match ';' } | ForEach-Object {
  $f = $_ -split ';'
  [pscustomobject]@{ t = [double]($f[0] -replace ',', '.'); ded = [int64]$f[1]; shr = [int64]$f[2]; used = [int64]$f[3]; free = [int64]$f[4] }
}
function Stat($sel, $label) {
  if (-not $sel -or $sel.Count -eq 0) { Write-Output ("{0,-14} no samples" -f $label); return }
  $d = ($sel | Measure-Object -Property ded -Maximum).Maximum / 1MB
  $s = ($sel | Measure-Object -Property shr -Maximum).Maximum / 1MB
  $u = ($sel | Measure-Object -Property used -Maximum).Maximum
  $f = ($sel | Measure-Object -Property free -Minimum).Minimum
  Write-Output ("{0,-14} yforge_dedicated_max={1,7:N0} MiB  yforge_shared_max={2,6:N0} MiB  nvsmi_used_max={3,5} MiB  nvsmi_free_min={4,5} MiB" -f $label, $d, $s, $u, $f)
}
Write-Output ("TAG $Tag  prompt=$([IO.Path]::GetFileName($Prompt))  ovr='$Ovr'")
Stat ($rows | Where-Object { $_.t -le $tRestart }) 'before-restart'
Stat ($rows | Where-Object { $_.t -gt $tRestart -and $_.t -le $tIdle }) 'load+pool'
Stat ($rows | Where-Object { $_.t -gt $tIdle -and $_.t -le $tEnd }) 'request'
Stat ($rows | Where-Object { $_.t -gt $tEnd }) 'after'
Write-Output ("csv: $csv")
$bench
