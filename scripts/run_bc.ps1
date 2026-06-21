# Runs only variants B (strict) and C (strict_v2), once each.

$python      = ".venv\Scripts\python.exe"
$stepsPerRun = 250

$variants = @(
    @{ mode = "strict";    desc = "Variant B: + reject 3+ consecutive identical AAs" },
    @{ mode = "strict_v2"; desc = "Variant C: max 25% noncanonical + min 4 AAs" }
)

$totalStart = Get-Date

foreach ($v in $variants) {
    Write-Host ""
    Write-Host "================================================================"
    Write-Host $v.desc
    Write-Host "================================================================"
    & $python main.py --scoring-function tanimoto --num-steps $stepsPerRun --sigma 80 --prior-weight 0.8 --filter-mode $v.mode
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Run failed with exit code $LASTEXITCODE -- aborting." -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

$elapsed = (Get-Date) - $totalStart
$mins = [math]::Round($elapsed.TotalMinutes, 1)
Write-Host ""
Write-Host "All runs finished in $mins min."
