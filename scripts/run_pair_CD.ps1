# Par 2: Varijanta C (strict_v2) -> D (strict_v3), 300 koraka svaka.
# Nakon C-a zapisuje putanju njegovog run foldera u _latest_C.txt,
# kako bi G (warm-start) mogao krenuti iz NOVOG C checkpointa.
# Pokretati IZ KORIJENA projekta:  .\scripts\run_pair_CD.ps1

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

Run-Variant "strict_v2" "Varijanta C (strict_v2)"

# Zapamti C-jev run folder (najnoviji) za kasniji warm-start (G)
$cFolder = Get-ChildItem "data\results" -Directory |
           Where-Object { $_.Name -like "run_*" } |
           Sort-Object LastWriteTime | Select-Object -Last 1
Set-Content -Path "data\results\_latest_C.txt" -Value $cFolder.FullName -Encoding utf8
Write-Host ""
Write-Host "C run folder zapamcen za G: $($cFolder.Name)"

Run-Variant "strict_v3" "Varijanta D (strict_v3)"

$mins = [math]::Round(((Get-Date) - $start).TotalMinutes, 1)
Write-Host ""
Write-Host "Par C+D gotov u $mins min."
