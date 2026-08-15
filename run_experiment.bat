@echo off
echo ========================================================
echo Memulai Eksperimen XGBoost (Multi-Trial: Native vs Docker)
echo ========================================================

echo.
echo Menjalankan 5x trial untuk Native dan Docker, serta membandingkannya...
python multi_trial.py --trials 5
if %errorlevel% neq 0 (
    echo [ERROR] Eksekusi multi-trial gagal. Pastikan Docker Desktop sudah berjalan.
    exit /b %errorlevel%
)

echo.
echo ========================================================
echo Eksperimen Selesai!
echo Hasil komparasi (dari rata-rata 5 trial) dapat dilihat di folder 'hasil_monitoring'
echo ========================================================
pause
