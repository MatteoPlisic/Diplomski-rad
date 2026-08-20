# Runs Variant F (strict_v5) — like E, but a stronger 0.5x penalty for noncanonical AAs.

$python      = ".venv\Scripts\python.exe"
$stepsPerRun = 250

$totalStart = Get-Date

Write-Host ""
Write-Host "================================================================"
Write-Host "Variant F: strict_v3 + 0.5x penalty for noncanonical AAs"
Write-Host "================================================================"
& $python main.py --scoring-function tanimoto --num-steps $stepsPerRun --sigma 80 --prior-weight 0.8 --filter-mode strict_v5
if ($LASTEXITCODE -ne 0) {
    Write-Host "Run failed with exit code $LASTEXITCODE -- aborting." -ForegroundColor Red
    exit $LASTEXITCODE
}

$elapsed = (Get-Date) - $totalStart
$mins = [math]::Round($elapsed.TotalMinutes, 1)
Write-Host ""
Write-Host "Variant F finished in $mins min."
