# Runs Variant D (strict_v3) but with prior + initial agent from Variant C's run.

$python      = ".venv\Scripts\python.exe"
$stepsPerRun = 250
$cCheckpoint = "data\results\run_2026-05-21-21_52_37_C_strict_v2\Agent.ckpt"

if (-not (Test-Path $cCheckpoint)) {
    Write-Host "Cannot find Variant C checkpoint at: $cCheckpoint" -ForegroundColor Red
    exit 1
}

$totalStart = Get-Date

Write-Host ""
Write-Host "================================================================"
Write-Host "Variant D-from-C: strict_v3 filter, warm-started from Variant C"
Write-Host "  prior + agent = $cCheckpoint"
Write-Host "================================================================"
& $python main.py --scoring-function tanimoto --num-steps $stepsPerRun --sigma 80 --prior-weight 0.8 --filter-mode strict_v3 --prior $cCheckpoint --agent $cCheckpoint
if ($LASTEXITCODE -ne 0) {
    Write-Host "Run failed with exit code $LASTEXITCODE -- aborting." -ForegroundColor Red
    exit $LASTEXITCODE
}

$elapsed = (Get-Date) - $totalStart
$mins = [math]::Round($elapsed.TotalMinutes, 1)
Write-Host ""
Write-Host "Run finished in $mins min."
