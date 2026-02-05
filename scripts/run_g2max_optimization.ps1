$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Resolve-Path (Join-Path $root "..")
$venv = Join-Path $projectRoot ".venv"
$python = Join-Path $venv "Scripts\python.exe"

if (-not (Test-Path $python)) {
  python -m venv $venv
  & "$python" -m pip install -e "$projectRoot"
}

& "$python" -m optimizer.g2max_optuna --trials 120 --seeds "7,13,21" --max-drawdown 0.18

$optDir = Get-ChildItem "$projectRoot\artifacts" -Directory -Filter "g2max_opt_*" |
  Sort-Object Name -Descending | Select-Object -First 1

if ($optDir) {
  $chartPath = Join-Path $optDir.FullName "equity_comparison.png"
  Start-Process -FilePath $chartPath
  Write-Output "Chart: $chartPath"
  Write-Output "Summary: $(Join-Path $optDir.FullName 'summary.json')"
}
