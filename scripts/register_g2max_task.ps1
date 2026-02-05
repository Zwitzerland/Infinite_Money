param(
  [string]$Time = "09:35"
)

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Resolve-Path (Join-Path $root "..")
$taskName = "G2MAXPaperDaily"
$scriptPath = Join-Path $projectRoot "scripts\run_daily_paper_g2max.ps1"

$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-ExecutionPolicy Bypass -File `"$scriptPath`""
$trigger = New-ScheduledTaskTrigger -Daily -At $Time

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Force
Write-Output "Registered task $taskName at $Time"
