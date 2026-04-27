param(
    [int]$NumSteps = 100,
    [int]$NumProcesses = 0,
    [string]$ModelPath = "random_forest_model_amp.pkl"
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Join-Path $scriptDir "..\.venv\Scripts\python.exe"
$mainPy = Join-Path $scriptDir "main.py"

if (!(Test-Path $pythonExe)) {
    Write-Host "ERROR: Python not found at $pythonExe" -ForegroundColor Red
    Write-Host "Create the venv first: python -m venv ..\.venv" -ForegroundColor Yellow
    exit 1
}

if (!(Test-Path $mainPy)) {
    Write-Host "ERROR: main.py not found at $mainPy" -ForegroundColor Red
    exit 1
}

Write-Host "Using Python: $pythonExe" -ForegroundColor Cyan
Write-Host "Running tanimoto scoring..." -ForegroundColor Cyan

Push-Location $scriptDir
try {
    & $pythonExe $mainPy `
        --scoring-function tanimoto `
        --num-steps $NumSteps `
        --num-processes $NumProcesses `
        --scoring-function-kwargs clf_path $ModelPath

    $exitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}

if ($exitCode -ne 0) {
    Write-Host "Run failed with exit code $exitCode" -ForegroundColor Red
    exit $exitCode
}

Write-Host "Run completed." -ForegroundColor Green
