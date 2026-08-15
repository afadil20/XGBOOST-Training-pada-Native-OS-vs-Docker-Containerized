# XGBoost Native vs Docker Benchmark (CIC IDS 2017)

Benchmark training XGBoost: perbandingan performa **Native OS** vs **Docker Containerized** menggunakan dataset **ISCX-UNB 2012 IDS** (Intrusion Detection System).

---

## 📁 Struktur Project

```
.
├── training.py              # Training XGBoost + monitoring CPU/RAM (Native & Docker)
├── multi_trial.py           # Orkestrator multi-trial otomatis (Native + Docker)
├── comparasi.py             # Visualisasi perbandingan hasil
├── docker-compose.yml       # Config Docker + resource limit (2 CPU, 4GB RAM)
├── Dockerfile               # Image definition
├── requirements.txt         # Dependencies Python
├── run_experiment.bat       # Script Windows (batch)
├── run_experiment.ps1       # Script Windows (PowerShell)
├── demo_data.py             # Info dataset untuk demo sidang
├── tabel_hasil.py           # Tabel visualisasi 20 trial
├── _cek_benign.py           # Cek distribusi label BENIGN
├── .dockerignore            # Docker build ignore
├── .gitignore               # Git ignore
├── data/                    # Dataset ISCX-UNB 2012 (8 file CSV) — NOT IN REPO
├── hasil_monitoring/        # Output CSV & grafik — NOT IN REPO
├── hasil_monitoring_docker/ # Output Docker — NOT IN REPO
└── _archive/_SIDANG_CATATAN.md  # Catatan presentasi sidang
```

---

## 🔧 Persiapan Environment

### 1. Clone Repository
```bash
git clone https://github.com/afadil20/XGBOOST-Training-pada-Native-OS-vs-Docker-Containerized.git
cd XGBOOST-Training-pada-Native-OS-vs-Docker-Containerized
```

### 2. Install Dependencies (Native)
```bash
pip install -r requirements.txt
```

### 3. Dataset
Dataset **ISCX-UNB 2012** (8 file CSV) **tidak termasuk di repo** (ukuran besar).  
Download manual dari: https://www.unb.ca/cic/datasets/ids-2012.html  
Atau gunakan file yang sudah ada di folder `data/` lokal.

Struktur folder `data/`:
```
data/
├── Monday-WorkingHours.pcap_ISCX.csv
├── Tuesday-WorkingHours.pcap_ISCX.csv
├── Wednesday-workingHours.pcap_ISCX.csv
├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
├── Friday-WorkingHours-Morning.pcap_ISCX.csv
├── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
└── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
```

---

## 🚀 Cara Menjalankan Eksperimen

### Opsi A: Script Otomatis (Recommended)

**Windows Batch:**
```cmd
run_experiment.bat
```

**Windows PowerShell:**
```powershell
.\run_experiment.ps1
```

Script akan:
1. Build Docker image
2. Jalankan multi-trial Native (default 5x)
3. Jalankan multi-trial Docker (default 5x)
4. Generate grafik perbandingan (`comparasi.py`)
5. Output tersimpan di `hasil_monitoring/`

---

### Opsi B: Manual Step-by-Step

#### 1. Cek Info Dataset (Demo Sidang)
```bash
python demo_data.py
```
Output: jumlah file, baris per file, distribusi label, jumlah fitur.

#### 2. Training Single Trial (Native)
```bash
python training.py --env native --dataset ./data/
```
Output: `hasil_monitoring/summary_native.csv`, `log_resource_native.csv`, `logloss_native.csv`, `feature_importance_native.csv`

#### 3. Training Single Trial (Docker)
```bash
docker-compose up --build
```
Atau via script:
```bash
python training.py --env docker --dataset ./data/
```
Output: `hasil_monitoring/summary_docker.csv`, `log_resource_docker.csv`, `logloss_docker.csv`, `feature_importance_docker.csv`

#### 4. Multi-Trial Otomatis (Native + Docker)
```bash
python multi_trial.py --trials 5
```
Parameter:
- `--trials N` : jumlah percobaan per environment (default: 5)
- `--ndata N`  : jumlah sampel per trial (default: 2,827,876 = full dataset)
- `--skip-docker` : lewati Docker
- `--skip-native` : lewati Native

Output:
- `hasil_monitoring/multi_trial_native.csv` — raw data per trial Native
- `hasil_monitoring/multi_trial_docker.csv` — raw data per trial Docker
- `hasil_monitoring/summary_native.csv` — rata-rata Native
- `hasil_monitoring/summary_docker.csv` — rata-rata Docker

#### 5. Visualisasi Perbandingan
```bash
python comparasi.py
```
Output: `hasil_monitoring/fwa_comparison_report.png` (grafik lengkap: akurasi, CPU, RAM, loss, confusion matrix, feature importance)

#### 6. Tabel Hasil 20 Trial (Opsional)
```bash
python tabel_hasil.py
```
Output: `hasil_monitoring/tabel_20_trial.png`

---

## 🐳 Detail Docker

### Resource Limit (Fair Comparison)
`docker-compose.yml` membatasi container:
```yaml
# mem_limit: 4g
# cpus: 2.0
```
Uncomment baris di atas untuk aktifkan limit. Default: **unlimited** (sesuai host).

### Build Image Manual
```bash
docker build -t fwa-xgboost:latest .
```

### Jalankan Container Manual
```bash
docker run --rm -v ${PWD}/data:/app/data -v ${PWD}/hasil_monitoring:/app/hasil_monitoring fwa-xgboost:latest python training.py --env docker --ndata 2827876
```

---

## 📊 Output & Metrik

Setiap trial menghasilkan:

| File | Isi |
|------|-----|
| `summary_<env>.csv` | Akurasi, Precision, Recall, F1, Durasi, CPU avg/peak, RAM avg/peak |
| `log_resource_<env>.csv` | Log CPU & RAM tiap 0.5 detik (time, cpu_pct, ram_mb) |
| `logloss_<env>.csv` | Train/Validation loss per estimator (boosting round) |
| `feature_importance_<env>.csv` | Score importance per fitur (gain/weight/cover) |

Grafik final (`fwa_comparison_report.png`) berisi:
1. Tabel ringkasan metrik
2. Bar chart akurasi / F1 / Precision / Recall
3. Line chart CPU & RAM over time
4. Log-loss curves
5. Confusion matrix (Native vs Docker)
6. Top-10 Feature Importance

---

## 🖥️ Demo Sidang (5-7 Menit)

Lihat `_archive/_SIDANG_CATATAN.md` untuk script lengkap.

Ringkas:
```powershell
# 1. Tampilkan dataset
python demo_data.py

# 2. Training Native 1x
python training.py --env native --dataset ./data/

# 3. Lihat hasil CSV
cat ./hasil_monitoring/summary_native.csv

# 4. Generate grafik perbandingan (butuh hasil Docker juga)
python comparasi.py

# 5. Buka grafik final
Invoke-Item ./hasil_monitoring/fwa_comparison_report.png
```

---

## ⚙️ Konfigurasi Penting

### training.py — Parameter Utama
```python
DATASET_PATH = './data/FWA.csv'   # Ganti path dataset
TOTAL_DATA   = 100_000            # Jumlah sampel (sample acak)
```

### multi_trial.py — Parameter CLI
```bash
python multi_trial.py --trials 10 --ndata 500000
```

### docker-compose.yml — Resource Limit
```yaml
services:
  xgboost-experiment:
    # mem_limit: 4g    # Uncomment untuk batasi RAM
    # cpus: 2.0        # Uncomment untuk batasi CPU
```

---

## 🛠️ Troubleshooting

| Masalah | Solusi |
|---------|--------|
| `ModuleNotFoundError: xgboost` | `pip install -r requirements.txt` |
| Docker: `no space left on device` | `docker system prune -a` |
| Dataset tidak ditemukan | Pastikan folder `data/` berisi 8 file CSV |
| Memory error saat training | Kurangi `TOTAL_DATA` di `training.py` atau `--ndata` di `multi_trial.py` |
| Grafik tidak muncul | Pastikan `matplotlib` backend `Agg` (sudah di-set di code) |
| Push GitHub gagal (file besar) | Pastikan `data/` dan `hasil_monitoring/` ada di `.gitignore` |

---

## 📝 Catatan Teknis

- **Dataset**: ISCX-UNB 2012 (8 hari traffic jaringan, label: BENIGN + 4 attack types)
- **Model**: XGBoost Classifier (default params, bisa diubah di `training.py`)
- **Monitoring**: `psutil` sampling CPU/RAM tiap 0.5 detik di background thread
- **Preprocessing**: LabelEncoder untuk kolom kategorikal, drop NaN/inf
- **Split**: 80% train / 20% test, stratified
- **Evaluasi**: Accuracy, Precision, Recall, F1-score (macro), Confusion Matrix

---

## 📄 Lisensi

Project ini untuk keperluan akademik (jurnal/sidang). Dataset ISCX-UNB 2012 milik UNB/CIC.

---

## 👤 Author

**afadil20** — https://github.com/afadil20
