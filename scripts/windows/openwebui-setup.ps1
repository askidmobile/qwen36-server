# openwebui-setup.ps1 -- host setup for the Open WebUI source deployment (re-runnable).
#
#   * registers the \open-webui scheduled task (boot trigger, SYSTEM, restart on failure)
#     that runs D:\Projects\yttri-inference\openwebui-run.bat
#   * adds the inbound firewall rule for the Open WebUI port
#
# Usage:  powershell -ExecutionPolicy Bypass -File openwebui-setup.ps1

param(
    [string]$Root = 'D:\Projects\yttri-inference',
    [int]$Port = 8080,
    [string]$TaskName = 'open-webui',
    [string]$RuleName = 'Open WebUI 8080'
)

$ErrorActionPreference = 'Stop'

$action = New-ScheduledTaskAction -Execute 'cmd.exe' `
    -Argument "/c $Root\openwebui-run.bat" -WorkingDirectory $Root
$trigger = New-ScheduledTaskTrigger -AtStartup
$principal = New-ScheduledTaskPrincipal -UserId 'SYSTEM' -LogonType ServiceAccount -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew `
    -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Principal $principal `
    -Settings $settings -Description 'Open WebUI (source build): local chat UI for yforge, no auth' -Force | Out-Null
Write-Output "task registered: $TaskName"

if (-not (Get-NetFirewallRule -DisplayName $RuleName -ErrorAction SilentlyContinue)) {
    New-NetFirewallRule -DisplayName $RuleName -Direction Inbound -Action Allow -Protocol TCP `
        -LocalPort $Port -Profile Any -Description 'Open WebUI (source build)' | Out-Null
    Write-Output "firewall rule added: $RuleName"
} else {
    Write-Output "firewall rule already exists: $RuleName"
}
