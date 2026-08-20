# Runs 3 training variants, 3 times each.
# Each run uses --num-steps from $stepsPerRun (set to 3 for smoke-test, bump to 250 later).

$python      = ".venv\Scripts\python.exe"
$stepsPerRun = 250

$variants = @(
    @{ mode = "basic";     runs = 3; desc = "Variant A: no triple-repeat penalty (original)" },
    @{ mode = "strict";    runs = 1; desc = "Variant B: + reject 3+ consecutive identical AAs" },
    @{ mode = "strict_v2"; runs = 1; desc = "Variant C: strict + max 25% noncanonical + min 4 AAs" }
)

$totalStart = Get-Date

foreach ($v in $variants) {
    Write-Host ""
    Write-Host "================================================================"
    Write-Host $v.desc
    Write-Host "================================================================"
    for ($i = 1; $i -le $v.runs; $i++) {
        Write-Host ""
        Write-Host "--- $($v.mode) run $i / $($v.runs) ---"
        & $python main.py --scoring-function tanimoto --num-steps $stepsPerRun --sigma 80 --prior-weight 0.8 --filter-mode $v.mode
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Run failed with exit code $LASTEXITCODE -- aborting." -ForegroundColor Red
            exit $LASTEXITCODE
        }
    }
}

$elapsed = (Get-Date) - $totalStart
$mins = [math]::Round($elapsed.TotalMinutes, 1)
Write-Host ""
Write-Host "All runs finished in $mins min."
