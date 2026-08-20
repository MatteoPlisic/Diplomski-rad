# Par 1: Varijanta A (basic) -> B (strict), 300 koraka svaka.
# Pokretati IZ KORIJENA projekta:  .\scripts\run_pair_AB.ps1

$python      = ".venv\Scripts\python.exe"
$stepsPerRun = 300

function Run-Variant($mode, $desc) {
    Write-Host ""
    Write-Host "================================================================"
    Write-Host $desc
    Write-Host "================================================================"
    & $python main.py --scoring-function tanimoto --num-steps $stepsPerRun --sigma 80 --prior-weight 0.8 --filter-mode $mode
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Run ($mode) pao s exit kodom $LASTEXITCODE -- prekidam." -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

$start = Get-Date
Run-Variant "basic"  "Varijanta A (basic)"
Run-Variant "strict" "Varijanta B (strict)"
$mins = [math]::Round(((Get-Date) - $start).TotalMinutes, 1)
Write-Host ""
Write-Host "Par A+B gotov u $mins min."
