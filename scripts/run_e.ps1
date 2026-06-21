# Runs Variant E (strict_v4) — like D, plus 0.85x soft penalty for noncanonical AAs.

$python      = ".venv\Scripts\python.exe"
$stepsPerRun = 250

$totalStart = Get-Date

Write-Host ""
Write-Host "================================================================"
Write-Host "Variant E: strict_v3 + 0.85x penalty for noncanonical AAs"
Write-Host "================================================================"
& $python main.py --scoring-function tanimoto --num-steps $stepsPerRun --sigma 80 --prior-weight 0.8 --filter-mode strict_v4
if ($LASTEXITCODE -ne 0) {
    Write-Host "Run failed with exit code $LASTEXITCODE -- aborting." -ForegroundColor Red
    exit $LASTEXITCODE
}

$elapsed = (Get-Date) - $totalStart
$mins = [math]::Round($elapsed.TotalMinutes, 1)
Write-Host ""
Write-Host "Variant E finished in $mins min."
