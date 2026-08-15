"""
multi_trial.py — Pengujian Multi-Trial Otomatis (Native & Docker)
=================================================================
Script ini menjalankan training.py (Native) dan docker-compose up (Docker)
sebanyak N kali percobaan, mengumpulkan semua hasilnya, menghitung rata-rata,
lalu memanggil comparasi.py secara otomatis.

Cara menjalankan:
    python multi_trial.py --trials 5

Output:
    - ./hasil_monitoring/multi_trial_native.csv  → semua hasil raw Native per trial
    - ./hasil_monitoring/multi_trial_docker.csv  → semua hasil raw Docker per trial
    - ./hasil_monitoring/summary_native.csv      → rata-rata Native (untuk comparasi.py)
    - ./hasil_monitoring/summary_docker.csv      → rata-rata Docker (untuk comparasi.py)
"""

import os
import sys
import time
import argparse
import subprocess
import pandas as pd

# ==========================================
# 1. ARGUMEN
# ==========================================
parser = argparse.ArgumentParser(description="Multi-Trial XGBoost Experiment")
parser.add_argument('--trials', type=int, default=5,
                    help="Jumlah percobaan untuk setiap lingkungan (default: 5)")
parser.add_argument('--ndata', type=int, default=2_827_876,
                    help="Jumlah sampel per trial (default: 2827876 — full dataset)")
parser.add_argument('--skip-docker', action='store_true',
                    help="Lewati pengujian Docker (hanya Native)")
parser.add_argument('--skip-native', action='store_true',
                    help="Lewati pengujian Native (hanya Docker)")
args = parser.parse_args()

N_TRIALS   = args.trials
os.environ['NDATA'] = str(args.ndata)
OUTPUT_DIR = './hasil_monitoring'
os.makedirs(OUTPUT_DIR, exist_ok=True)

SUMMARY_COLS = [
    'env', 'trial', 'accuracy', 'precision', 'recall', 'f1_score',
    'duration_sec', 'avg_cpu_pct', 'peak_cpu_pct', 'avg_ram_mb', 'peak_ram_mb',
    'n_train', 'n_test', 'n_features',
    'cm_tn', 'cm_fp', 'cm_fn', 'cm_tp',
    'e2e_sec'  # Waktu End-to-End dari sisi sistem (termasuk startup)
]

def print_banner(text, char='=', width=60):
    print(f"\n{char*width}")
    print(f"  {text}")
    print(f"{char*width}")

def run_native_trial(trial_num):
    """Menjalankan 1 percobaan pada lingkungan Native."""
    print(f"\n  [Native] ── Trial {trial_num}/{N_TRIALS} ──")
    t_start = time.time()
    result = subprocess.run(
        [sys.executable, 'training.py', '--env', 'native', '--ndata', str(args.ndata)],
        capture_output=False,
        text=True
    )
    e2e = round(time.time() - t_start, 4)

    if result.returncode != 0:
        print(f"  [Native] ✗ Trial {trial_num} GAGAL!")
        return None

    # Baca summary yang baru saja ditulis oleh training.py
    try:
        row = pd.read_csv(f'{OUTPUT_DIR}/summary_native.csv').iloc[0].to_dict()
        row['trial'] = trial_num
        row['e2e_sec'] = e2e
        print(f"  [Native] ✓ Trial {trial_num} selesai | "
              f"E2E: {e2e:.2f}s | "
              f"Training: {row['duration_sec']:.2f}s | "
              f"Avg RAM: {row['avg_ram_mb']:.1f} MB | "
              f"Avg CPU: {row['avg_cpu_pct']:.1f}%")
        return row
    except Exception as e:
        print(f"  [Native] ✗ Gagal membaca summary: {e}")
        return None

def run_docker_trial(trial_num):
    """Menjalankan 1 percobaan pada lingkungan Docker."""
    print(f"\n  [Docker] ── Trial {trial_num}/{N_TRIALS} ──")
    t_start = time.time()
    result = subprocess.run(
        ['docker-compose', 'up', '--abort-on-container-exit'],
        capture_output=False,
        text=True
    )
    e2e = round(time.time() - t_start, 4)

    if result.returncode != 0:
        print(f"  [Docker] ✗ Trial {trial_num} GAGAL! (exit code: {result.returncode})")
        return None

    # Baca summary yang baru saja di-mount ke folder lokal oleh Docker
    try:
        row = pd.read_csv(f'{OUTPUT_DIR}/summary_docker.csv').iloc[0].to_dict()
        row['trial'] = trial_num
        row['e2e_sec'] = e2e
        print(f"  [Docker] ✓ Trial {trial_num} selesai | "
              f"E2E: {e2e:.2f}s | "
              f"Training: {row['duration_sec']:.2f}s | "
              f"Avg RAM: {row['avg_ram_mb']:.1f} MB | "
              f"Avg CPU: {row['avg_cpu_pct']:.1f}%")
        return row
    except Exception as e:
        print(f"  [Docker] ✗ Gagal membaca summary: {e}")
        return None

def compute_and_save_average(all_rows, env_name):
    """Menghitung rata-rata semua trial dan menyimpannya ke summary_<env>.csv"""
    df_trials = pd.DataFrame(all_rows)

    # Simpan semua hasil raw per trial
    raw_path = f'{OUTPUT_DIR}/multi_trial_{env_name}.csv'
    df_trials.to_csv(raw_path, index=False)
    print(f"\n  [{env_name.upper()}] Data semua trial tersimpan: {raw_path}")

    # Hitung rata-rata kolom numerik
    numeric_cols = [
        'accuracy', 'precision', 'recall', 'f1_score',
        'duration_sec', 'avg_cpu_pct', 'peak_cpu_pct',
        'avg_ram_mb', 'peak_ram_mb', 'e2e_sec',
        'n_train', 'n_test', 'n_features',
        'cm_tn', 'cm_fp', 'cm_fn', 'cm_tp'
    ]
    avg_row = {'env': env_name}
    for col in numeric_cols:
        if col in df_trials.columns:
            avg_row[col] = round(df_trials[col].mean(), 6)

    # Timpa summary_<env>.csv dengan nilai rata-rata (untuk dibaca comparasi.py)
    summary_path = f'{OUTPUT_DIR}/summary_{env_name}.csv'
    pd.DataFrame([avg_row]).to_csv(summary_path, index=False)

    # Tampilkan ringkasan di terminal
    print(f"\n  ┌─ RATA-RATA {env_name.upper()} ({len(all_rows)} Trial) ─────────────────────┐")
    print(f"  │  Accuracy       : {avg_row['accuracy']*100:.4f} %")
    print(f"  │  F1-Score       : {avg_row['f1_score']*100:.4f} %")
    print(f"  │  Waktu Training : {avg_row['duration_sec']:.4f} detik")
    print(f"  │  Waktu E2E      : {avg_row['e2e_sec']:.4f} detik")
    print(f"  │  Avg CPU        : {avg_row['avg_cpu_pct']:.2f} %")
    print(f"  │  Peak CPU       : {avg_row['peak_cpu_pct']:.2f} %")
    print(f"  │  Avg RAM        : {avg_row['avg_ram_mb']:.2f} MB")
    print(f"  │  Peak RAM       : {avg_row['peak_ram_mb']:.2f} MB")
    print(f"  └───────────────────────────────────────────────────────┘")

    return avg_row

# ==========================================
# 2. EKSEKUSI UTAMA
# ==========================================
print_banner(f"MULTI-TRIAL EXPERIMENT — {N_TRIALS} Percobaan per Lingkungan")

native_results = []
docker_results = []

# ── Native Trials ──────────────────────────────────────────────
if not args.skip_native:
    print_banner("FASE 1: Pengujian Native OS", char='─')
    for i in range(1, N_TRIALS + 1):
        row = run_native_trial(i)
        if row:
            native_results.append(row)
        time.sleep(2)  # Jeda 2 detik antar trial agar sistem stabil

    if native_results:
        native_avg = compute_and_save_average(native_results, 'native')
    else:
        print("\n[ERROR] Tidak ada hasil Native yang berhasil dikumpulkan!")
        sys.exit(1)
else:
    print("\n[INFO] Pengujian Native dilewati (--skip-native aktif)")

# ── Docker Trials ──────────────────────────────────────────────
if not args.skip_docker:
    print_banner("FASE 2: Pengujian Docker", char='─')
    # Build image sekali saja sebelum semua trial Docker
    print("\n  [Docker] Membangun image Docker... (hanya sekali)")
    build_result = subprocess.run(['docker-compose', 'build'], capture_output=False)
    if build_result.returncode != 0:
        print("  [Docker] ✗ Docker build GAGAL! Periksa Dockerfile Anda.")
        sys.exit(1)
    print("  [Docker] ✓ Image berhasil di-build.\n")

    for i in range(1, N_TRIALS + 1):
        row = run_docker_trial(i)
        if row:
            docker_results.append(row)
        time.sleep(3)  # Jeda 3 detik antar trial Docker agar container bersih

    if docker_results:
        docker_avg = compute_and_save_average(docker_results, 'docker')
    else:
        print("\n[ERROR] Tidak ada hasil Docker yang berhasil dikumpulkan!")
        sys.exit(1)
else:
    print("\n[INFO] Pengujian Docker dilewati (--skip-docker aktif)")

# ── Generate Laporan Perbandingan Otomatis ─────────────────────
print_banner("FASE 3: Generate Laporan Perbandingan Akhir")
print("\n  Menjalankan comparasi.py...")
compare_result = subprocess.run([sys.executable, 'comparasi.py'], capture_output=False)
if compare_result.returncode == 0:
    print("\n  ✓ Grafik laporan perbandingan berhasil dibuat!")
    print(f"  ✓ Lihat: {OUTPUT_DIR}/fwa_comparison_report.png")
else:
    print("\n  ✗ comparasi.py gagal dijalankan. Jalankan manual: python comparasi.py")

print_banner(f"SELESAI! Semua {N_TRIALS} trial selesai dijalankan.", char='=')
print(f"\n  File yang dihasilkan:")
print(f"  ├── {OUTPUT_DIR}/multi_trial_native.csv   ← Data raw semua trial Native")
print(f"  ├── {OUTPUT_DIR}/multi_trial_docker.csv   ← Data raw semua trial Docker")
print(f"  ├── {OUTPUT_DIR}/summary_native.csv       ← Rata-rata Native (input comparasi)")
print(f"  ├── {OUTPUT_DIR}/summary_docker.csv       ← Rata-rata Docker (input comparasi)")
print(f"  └── {OUTPUT_DIR}/fwa_comparison_report.png ← Grafik final untuk Jurnal\n")
