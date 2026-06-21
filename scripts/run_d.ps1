# Runs only Variant D (strict_v3), once.

$python      = ".venv\Scripts\python.exe"
$stepsPerRun = 250

$totalStart = Get-Date

Write-Host ""
Write-Host "================================================================"
Write-Host "Variant D: strict_v2 + reject 3+ consecutive identical AAs"
Write-Host "================================================================"
& $python main.py --scoring-function tanimoto --num-steps $stepsPerRun --sigma 80 --prior-weight 0.8 --filter-mode strict_v3
if ($LASTEXITCODE -ne 0) {
    Write-Host "Run failed with exit code $LASTEXITCODE -- aborting." -ForegroundColor Red
    exit $LASTEXITCODE
}

$elapsed = (Get-Date) - $totalStart
$mins = [math]::Round($elapsed.TotalMinutes, 1)
Write-Host ""
Write-Host "Variant D finished in $mins min."
