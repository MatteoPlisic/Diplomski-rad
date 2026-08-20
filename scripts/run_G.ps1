# Varijanta G: strict_v3 (kao D), ali topli start iz NOVOG 300-koracnog C-a.
# Cita _latest_C.txt koji je zapisao run_pair_CD.ps1.
# POKRENUTI TEK NAKON run_pair_CD.ps1!
# Pokretati IZ KORIJENA projekta:  .\scripts\run_G.ps1

$python      = ".venv\Scripts\python.exe"
$stepsPerRun = 300
$marker      = "data\results\_latest_C.txt"

if (-not (Test-Path $marker)) {
    Write-Host "Ne mogu naci $marker -- prvo pokreni .\scripts\run_pair_CD.ps1" -ForegroundColor Red
    exit 1
}

$cFolder = (Get-Content $marker -Raw).Trim()
$ckpt = Join-Path $cFolder "Agent.ckpt"
if (-not (Test-Path $ckpt)) {
    Write-Host "Ne mogu naci C checkpoint: $ckpt" -ForegroundColor Red
    exit 1
}

$start = Get-Date
Write-Host ""
Write-Host "================================================================"
Write-Host "Varijanta G: strict_v3 + topli start iz C"
Write-Host "  C checkpoint: $ckpt"
Write-Host "================================================================"
& $python main.py --scoring-function tanimoto --num-steps $stepsPerRun --sigma 80 --prior-weight 0.8 --filter-mode strict_v3 --prior $ckpt --agent $ckpt
if ($LASTEXITCODE -ne 0) {
    Write-Host "Run (G) pao s exit kodom $LASTEXITCODE -- prekidam." -ForegroundColor Red
    exit $LASTEXITCODE
}

$mins = [math]::Round(((Get-Date) - $start).TotalMinutes, 1)
Write-Host ""
Write-Host "Varijanta G gotova u $mins min."
