# Guardiano del run E6: attende la fine dell'app Modal, poi scarica il parquet.
# Lanciato in background: la sua uscita e' la notifica che il run e' finito.
$ErrorActionPreference = "Continue"
$env:PYTHONUTF8 = "1"
$APP = "ap-pYpRC7OsVoUDKQBJiIOdYH"
$GPU = "C:\Users\Utente\MODELLIPERSONALIZZATI\DC2026\AIMO\gpu"
$OUT = Join-Path $GPU "out"
if (-not (Test-Path $OUT)) { New-Item -ItemType Directory $OUT | Out-Null }
Set-Location $GPU

$giri = 0
while ($true) {
    $giri++
    $stato = "?"
    try {
        $lista = python -m modal app list --json 2>$null | ConvertFrom-Json
        $riga = $lista | Where-Object { $_.'App ID' -eq $APP }
        if ($riga) { $stato = $riga.State } else { $stato = "assente" }
    } catch { $stato = "errore-query" }

    Write-Output ("[{0}] giro {1}: stato={2}" -f (Get-Date -Format "HH:mm:ss"), $giri, $stato)
    if ($stato -notlike "*ephemeral*") { break }
    if ($giri -gt 200) { Write-Output "LIMITE GIRI RAGGIUNTO"; break }
    Start-Sleep -Seconds 300
}

Write-Output "=== RUN NON PIU' ATTIVO (stato finale: $stato) ==="

# scarica il parquet dal volume Modal
try {
    python -m modal volume get aimo-e6-out feature_greedy.parquet (Join-Path $OUT "feature_greedy.parquet") --force 2>&1 | Out-String | Write-Output
} catch { Write-Output "download completo fallito: $_" }
try {
    python -m modal volume get aimo-e6-out feature_parziale.parquet (Join-Path $OUT "feature_parziale.parquet") --force 2>&1 | Out-String | Write-Output
} catch { Write-Output "download parziale fallito: $_" }

Get-ChildItem $OUT -File | ForEach-Object { Write-Output ("scaricato: {0} ({1:N0} byte)" -f $_.Name, $_.Length) }
Write-Output "=== GUARDIANO TERMINATO ==="
