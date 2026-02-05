param(
    [string]$OutputPath = "artifacts/aws_inventory.json"
)

if ($args -contains "-h" -or $args -contains "--help") {
    Write-Host "Usage: aws_collect.ps1 [-OutputPath <path>]"
    exit 0
}

$ErrorActionPreference = "Stop"

function Run-AwsJson {
    param([string]$Command)
    $output = & aws $Command --output json 2>$null
    if (-not $output) {
        return $null
    }
    return $output | ConvertFrom-Json
}

$inventory = [ordered]@{}

try {
    $identity = Run-AwsJson "sts get-caller-identity"
    $inventory.identity = $identity

    $inventory.roles = Run-AwsJson "iam list-roles"
    $inventory.ecr = Run-AwsJson "ecr describe-repositories"
    $inventory.vpcs = Run-AwsJson "ec2 describe-vpcs"
    $inventory.subnets = Run-AwsJson "ec2 describe-subnets"
    $inventory.securityGroups = Run-AwsJson "ec2 describe-security-groups"
    $inventory.batchCompute = Run-AwsJson "batch describe-compute-environments"
    $inventory.batchQueues = Run-AwsJson "batch describe-job-queues"
    $inventory.stepFunctions = Run-AwsJson "stepfunctions list-state-machines"
    $inventory.secrets = Run-AwsJson "secretsmanager list-secrets"
    $inventory.braket = Run-AwsJson "braket list-devices"
} catch {
    Write-Host "AWS CLI call failed: $($_.Exception.Message)"
    exit 1
}

$outputDir = Split-Path $OutputPath -Parent
if ($outputDir -and -not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir | Out-Null
}

$inventory | ConvertTo-Json -Depth 6 | Out-File -FilePath $OutputPath -Encoding utf8
Write-Host "AWS inventory saved to $OutputPath"
