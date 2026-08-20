# Par 3: Varijanta E (strict_v4) -> F (strict_v5), 300 koraka svaka.
# Pokretati IZ KORIJENA projekta:  .\scripts\run_pair_EF.ps1

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
Run-Variant "strict_v4" "Varijanta E (strict_v4)"
Run-Variant "strict_v5" "Varijanta F (strict_v5)"
$mins = [math]::Round(((Get-Date) - $start).TotalMinutes, 1)
Write-Host ""
Write-Host "Par E+F gotov u $mins min."
