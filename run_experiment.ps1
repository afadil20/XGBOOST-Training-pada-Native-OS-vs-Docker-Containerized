Write-Host "========================================================"
Write-Host "Memulai Eksperimen XGBoost (Multi-Trial: Native vs Docker)"
Write-Host "========================================================"

Write-Host "`nMenjalankan 5x trial untuk Native dan Docker, serta membandingkannya..." -ForegroundColor Cyan
python multi_trial.py --trials 5
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Eksekusi multi-trial gagal. Pastikan Docker Desktop sudah berjalan." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "`n========================================================"
Write-Host "Eksperimen Selesai!" -ForegroundColor Green
Write-Host "Hasil komparasi (dari rata-rata 5 trial) dapat dilihat di folder 'hasil_monitoring'"
Write-Host "========================================================"
