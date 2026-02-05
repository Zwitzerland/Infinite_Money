param(
  [int]$MaxParallel = 5,
  [string]$Prd = "PRD.md",
  [switch]$Sandbox = $true,
  [switch]$Fast = $false
)

$env:OPENCODE_CLIENT = "cli"
$env:OPENCODE_SERVER_USERNAME = ""
$env:OPENCODE_SERVER_PASSWORD = ""

$argsList = @(
  "--opencode",
  "--parallel",
  "--max-parallel", $MaxParallel,
  "--prd", $Prd
)

if ($Sandbox) {
  $argsList += "--sandbox"
}

if ($Fast) {
  $argsList += "--fast"
}

ralphy @argsList
