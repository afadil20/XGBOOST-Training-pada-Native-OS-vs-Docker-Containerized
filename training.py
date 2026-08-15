"""
fwa_train.py — Training + Resource Monitoring
=============================================
Jalankan di NATIVE OS:
    python fwa_train.py --env native --dataset ./FWA.csv

Jalankan di DOCKER:
    python fwa_train.py --env docker --dataset ./FWA.csv

Output tersimpan di folder ./hasil_monitoring/:
    - log_resource_<env>.csv   → log CPU & RAM tiap 0.5 detik
    - summary_<env>.csv        → akurasi, durasi, statistik resource
    - logloss_<env>.csv        → train/val loss per estimator
    - feature_importance_<env>.csv → skor tiap fitur
"""

import os
import time
import threading
import argparse
import psutil
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix,
                             classification_report)
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 1. ARGUMEN
# ==========================================
parser = argparse.ArgumentParser(description="FWA XGBoost Training Script")
parser.add_argument('--env',     type=str, default='native',
                    choices=['native', 'docker'],
                    help="Lingkungan eksekusi: native atau docker")
parser.add_argument('--dataset', type=str, default='./data/',
                    help="Path ke file atau folder dataset CSV")
parser.add_argument('--ndata',   type=int, default=2_827_876,
                    help="Jumlah sample yang digunakan (default: 2827876 — seluruh dataset)")
args = parser.parse_args()

ENV          = args.env
DATASET_PATH = args.dataset
TOTAL_DATA   = args.ndata
OUTPUT_DIR   = './hasil_monitoring'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\n{'='*55}")
print(f"  FWA XGBoost Training — Lingkungan: {ENV.upper()}")
print(f"{'='*55}")

# ==========================================
# 2. MONITORING RESOURCE (BACKGROUND THREAD)
# ==========================================
cpu_logs   = []
ram_logs   = []
time_logs  = []
stop_flag  = False
t_global   = time.time()

def monitor_resource():
    process = psutil.Process(os.getpid())
    while not stop_flag:
        elapsed  = time.time() - t_global
        cpu      = psutil.cpu_percent(interval=None)
        ram      = process.memory_info().rss / (1024 * 1024)  # MB
        time_logs.append(elapsed)
        cpu_logs.append(cpu)
        ram_logs.append(ram)
        time.sleep(0.5)

# ==========================================
# 3. LOAD & PREPROCESSING
# ==========================================
print(f"[{ENV}] Membaca dataset: {DATASET_PATH}")
try:
    if os.path.isdir(DATASET_PATH):
        csv_files = [os.path.join(DATASET_PATH, f) for f in os.listdir(DATASET_PATH) if f.endswith('.csv')]
        if not csv_files:
            print(f"[ERROR] Tidak ada file .csv di folder: '{DATASET_PATH}'")
            exit(1)
        print(f"[{ENV}] Menggabungkan {len(csv_files)} file CSV menjadi satu...")
        df = pd.concat((pd.read_csv(f, engine='python', on_bad_lines='skip') for f in csv_files), ignore_index=True)
    else:
        df = pd.read_csv(DATASET_PATH, engine='python', on_bad_lines='skip')
except Exception as e:
    print(f"[ERROR] Gagal membaca dataset: {e}")
    exit(1)

df.columns = df.columns.str.strip()

# Pastikan kolom target ada
if 'Label' not in df.columns:
    print(f"[ERROR] Kolom 'Label' tidak ditemukan. Kolom yang ada: {list(df.columns)}")
    exit(1)

# Sample acak
if len(df) > TOTAL_DATA:
    df = df.sample(n=TOTAL_DATA, random_state=42).reset_index(drop=True)
print(f"[{ENV}] {len(df):,} baris dimuat.")

# Taktik B: Binarisasi Label (Semua jenis serangan menjadi 'ATTACK')
df['Label'] = df['Label'].apply(lambda x: 'BENIGN' if str(x).strip().upper() == 'BENIGN' else 'ATTACK')

# Label encoding untuk kolom teks
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

# Bersihkan inf dan NaN (umum di dataset traffic jaringan)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
n_before = len(df)
df.dropna(inplace=True)
n_dropped = n_before - len(df)
if n_dropped > 0:
    print(f"[{ENV}] {n_dropped} baris dihapus karena NaN/Inf.")
print(f"[{ENV}] Shape final: {df.shape}")

# Distribusi label
label_series = pd.Series(le.inverse_transform(df['Label'].astype(int)))
print(f"[{ENV}] Distribusi label:\n{label_series.value_counts().to_string()}")

# Split fitur & target
X = df.drop(columns=['Label'])
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"[{ENV}] Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ==========================================
# 4. TRAINING + MONITORING
# ==========================================
print(f"[{ENV}] Memulai training XGBoost...")

model = xgb.XGBClassifier(
    n_estimators  = 150,
    max_depth     = 8,
    learning_rate = 0.1,
    tree_method   = 'hist',
    n_jobs        = -1
)

# Jalankan monitoring di background
monitor_thread = threading.Thread(target=monitor_resource, daemon=True)
monitor_thread.start()

train_start = time.time()
model.fit(
    X_train, y_train,
    eval_set    = [(X_train, y_train), (X_test, y_test)],
    verbose     = False
)
train_end = time.time()

# Hentikan monitoring
stop_flag = True
monitor_thread.join()

# ==========================================
# 5. EVALUASI MODEL
# ==========================================
y_pred   = model.predict(X_test)
accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall    = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1        = f1_score(y_test, y_pred, average='weighted', zero_division=0)
duration  = train_end - train_start

avg_cpu  = float(np.mean(cpu_logs))
peak_cpu = float(np.max(cpu_logs))
avg_ram  = float(np.mean(ram_logs))
peak_ram = float(np.max(ram_logs))
cm       = confusion_matrix(y_test, y_pred)

print(f"\n{'='*55}")
print(f"  HASIL — {ENV.upper()}")
print(f"{'='*55}")
print(f"  Akurasi        : {accuracy*100:.4f} %")
print(f"  Precision      : {precision*100:.4f} %")
print(f"  Recall         : {recall*100:.4f} %")
print(f"  F1-Score       : {f1*100:.4f} %")
print(f"  Waktu Training : {duration:.4f} detik")
print(f"  Avg CPU        : {avg_cpu:.2f} %  |  Peak: {peak_cpu:.2f} %")
print(f"  Avg RAM        : {avg_ram:.2f} MB |  Peak: {peak_ram:.2f} MB")
print(f"{'='*55}")
print(classification_report(y_test, y_pred, target_names=le.classes_, digits=4))

# ==========================================
# 6. SIMPAN SEMUA HASIL KE CSV
# ==========================================

# 6a. Log resource (CPU & RAM per waktu)
log_resource_path = f"{OUTPUT_DIR}/log_resource_{ENV}.csv"
pd.DataFrame({
    'Waktu_Detik': time_logs,
    'CPU_Percent': cpu_logs,
    'RAM_MB'     : ram_logs
}).to_csv(log_resource_path, index=False)

# 6b. Summary metrik (1 baris per eksperimen)
summary_path = f"{OUTPUT_DIR}/summary_{ENV}.csv"
pd.DataFrame([{
    'env'           : ENV,
    'accuracy'      : round(accuracy, 6),
    'precision'     : round(precision, 6),
    'recall'        : round(recall, 6),
    'f1_score'      : round(f1, 6),
    'duration_sec'  : round(duration, 4),
    'avg_cpu_pct'   : round(avg_cpu, 2),
    'peak_cpu_pct'  : round(peak_cpu, 2),
    'avg_ram_mb'    : round(avg_ram, 2),
    'peak_ram_mb'   : round(peak_ram, 2),
    'n_train'       : len(X_train),
    'n_test'        : len(X_test),
    'n_features'    : X.shape[1],
    'cm_tn'         : int(cm[0, 0]),
    'cm_fp'         : int(cm[0, 1]),
    'cm_fn'         : int(cm[1, 0]),
    'cm_tp'         : int(cm[1, 1]),
}]).to_csv(summary_path, index=False)

# 6c. Loss curve per estimator
results      = model.evals_result()
logloss_path = f"{OUTPUT_DIR}/logloss_{ENV}.csv"
pd.DataFrame({
    'estimator'  : list(range(1, len(results['validation_0']['logloss']) + 1)),
    'train_loss' : results['validation_0']['logloss'],
    'val_loss'   : results['validation_1']['logloss'],
}).to_csv(logloss_path, index=False)

# 6d. Feature importance
fi_path = f"{OUTPUT_DIR}/feature_importance_{ENV}.csv"
pd.Series(model.feature_importances_, index=X.columns,
          name='importance').sort_values(ascending=False
          ).to_csv(fi_path, header=True)

print(f"\n[{ENV}] Semua hasil tersimpan di '{OUTPUT_DIR}/':")
print(f"  - log_resource_{ENV}.csv")
print(f"  - summary_{ENV}.csv")
print(f"  - logloss_{ENV}.csv")
print(f"  - feature_importance_{ENV}.csv")
print(f"\n[{ENV}] Setelah menjalankan kedua lingkungan,")
print(f"         jalankan: python fwa_compare.py")