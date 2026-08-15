# 🎓 PANDUAN SIDANG — XGBoost Native vs Docker
## Demo & Presentasi

---

## 📁 FILE YANG DIJELASKAN (5 file)

| File | Fungsi | Kata Kunci |
|------|--------|------------|
| **`multi_trial.py`** | Orkestrator utama. 1 perintah jalanin semua | "File utama eksperimen" |
| **`training.py`** | Training XGBoost + monitor CPU/RAM | "Proses inti ML" |
| **`comparasi.py`** | Baca CSV → bikin grafik perbandingan | "Pembuat grafik akhir" |
| **`Dockerfile`** | Resep container Python | "Lingkungan Docker" |
| **`docker-compose.yml`** | Batasi resource Docker (2 CPU, 4GB RAM) | "Agar perbandingan fair" |

> File lain (requirements.txt, data/*.csv, run_experiment.bat/ps1) — cukup bilang "file pendukung"

---

## 🖥️ URUTAN DEMO (5-7 menit)

### LANGKAH 1 — Tampilkan Dataset
```powershell
python -c "import pandas as pd; df=pd.read_csv('data/Monday-WorkingHours.pcap_ISCX.csv'); print(f'Baris: {len(df):,}'); print(f'Kolom: {list(df.columns[:5])}...'); print(df.head(3))"
```
🗣️ *"Dataset ISCX 2012, 8 file CSV traffic jaringan."*

---

### LANGKAH 2 — Training Native 1x
```powershell
python training.py --env native --dataset ./data/
```
🗣️ *"Training XGBoost + catat CPU/RAM tiap 0.5 detik."*
⏳ Tunggu ~1-2 menit selesai.

---

### LANGKAH 3 — Lihat Hasil CSV
```powershell
cat ./hasil_monitoring/summary_native.csv
```
🗣️ *"Semua metrik disimpan di CSV: akurasi, durasi, CPU, RAM, CM."*

---

### LANGKAH 4 — Bikin Grafik Perbandingan
```powershell
python comparasi.py
```
🗣️ *"comparasi.py baca 8 CSV → jadi 1 grafik besar."*

---

### LANGKAH 5 — Buka Grafik Final
```powershell
Invoke-Item ./hasil_monitoring/fwa_comparison_report.png
```
🗣️ *"Output final. Satu gambar = tabel + akurasi + CPU/RAM + loss + CM + feature importance."*

---

### (OPSIONAL) LANGKAH 6 — Multi-Trial Cepat
```powershell
python multi_trial.py --trials 3 --skip-docker
```
🗣️ *"multi_trial.py otomatis training 3x + rata-rata + grafik."*

---

## 💬 JAWABAN SIDANG (Q&A)

**Q: Kenapa akurasi tinggi sekali?**
A: Dataset ISCX sudah dalam bentuk fitur numerik informatif. XGBoost emang unggul buat tabular classification.

**Q: Kenapa hasil Native vs Docker beda?**
A: Docker dibatasi 2 CPU & 4GB RAM via `docker-compose.yml`. Beda resource → wajar beda performa.

**Q: Manualnya gimana?**
A: Cukup 1 command `python multi_trial.py --trials 5`. Semua otomatis: training, monitoring, grafik.

**Q: Berapa lama eksperimen penuh?**
A: 5 trial native + 5 trial docker + grafik ≈ 20-30 menit (tergantung spesifikasi laptop).

---

## ⚠️ PENGINGAT SEBELUM SIDANG

- [ ] Docker Desktop sudah running (kalau demo docker)
- [ ] Terminal sudah di folder `d:\jurnal-xgboost\`
- [ ] `hasil_monitoring/` sudah ada isinya (cadangan kalau demo gagal)
- [ ] Sudah install requirements: `pip install -r requirements.txt`
