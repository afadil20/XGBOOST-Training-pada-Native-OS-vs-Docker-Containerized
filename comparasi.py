"""
fwa_compare.py — Visualisasi Perbandingan Native vs Docker
==========================================================
Jalankan SETELAH kedua eksperimen selesai:
    python fwa_train.py --env native --dataset ./FWA.csv
    python fwa_train.py --env docker --dataset ./FWA.csv

Kemudian:
    python fwa_compare.py

Output: ./hasil_monitoring/fwa_comparison_report.png
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Gunakan backend non-GUI agar bisa berjalan tanpa layar/Tcl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# ==========================================
# 1. KONFIGURASI WARNA
# ==========================================
C_NATIVE = '#2980b9'   # Biru  → Native
C_DOCKER = '#e67e22'   # Oranye → Docker
C_NATIVE_LIGHT = '#aed6f1'
C_DOCKER_LIGHT = '#fad7a0'
OUTPUT_DIR = './hasil_monitoring'

# ==========================================
# 2. LOAD & VALIDASI DATA
# ==========================================
required_files = {
    'native': [
        f'{OUTPUT_DIR}/summary_native.csv',
        f'{OUTPUT_DIR}/log_resource_native.csv',
        f'{OUTPUT_DIR}/logloss_native.csv',
        f'{OUTPUT_DIR}/feature_importance_native.csv',
    ],
    'docker': [
        f'{OUTPUT_DIR}/summary_docker.csv',
        f'{OUTPUT_DIR}/log_resource_docker.csv',
        f'{OUTPUT_DIR}/logloss_docker.csv',
        f'{OUTPUT_DIR}/feature_importance_docker.csv',
    ]
}

missing = []
for env, files in required_files.items():
    for f in files:
        if not os.path.exists(f):
            missing.append(f)

if missing:
    print("\n[ERROR] File hasil berikut belum ada:")
    for f in missing:
        print(f"  ✗ {f}")
    print("\nPastikan Anda sudah menjalankan:")
    print("  python fwa_train.py --env native --dataset ./FWA.csv")
    print("  python fwa_train.py --env docker --dataset ./FWA.csv")
    sys.exit(1)

print("[compare] Membaca semua file hasil...")

# Summary
sn = pd.read_csv(f'{OUTPUT_DIR}/summary_native.csv').iloc[0]
sd = pd.read_csv(f'{OUTPUT_DIR}/summary_docker.csv').iloc[0]

# Log resource
rn = pd.read_csv(f'{OUTPUT_DIR}/log_resource_native.csv')
rd = pd.read_csv(f'{OUTPUT_DIR}/log_resource_docker.csv')

# Loss curve
ln = pd.read_csv(f'{OUTPUT_DIR}/logloss_native.csv')
ld = pd.read_csv(f'{OUTPUT_DIR}/logloss_docker.csv')

# Feature importance
fn = pd.read_csv(f'{OUTPUT_DIR}/feature_importance_native.csv', index_col=0)
fd = pd.read_csv(f'{OUTPUT_DIR}/feature_importance_docker.csv', index_col=0)
fn.columns = ['native']
fd.columns = ['docker']

print("[compare] Data berhasil dimuat. Membuat grafik perbandingan...")

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def bar_pair(ax, labels, val_native, val_docker, title, ylabel,
             fmt='.2f', pct=False):
    """Buat bar chart berpasangan Native vs Docker."""
    x = np.arange(len(labels))
    w = 0.35
    b1 = ax.bar(x - w/2, val_native, w, label='Native', color=C_NATIVE,
                edgecolor='white', linewidth=0.8)
    b2 = ax.bar(x + w/2, val_docker,  w, label='Docker', color=C_DOCKER,
                edgecolor='white', linewidth=0.8)
    suffix = '%' if pct else ''
    for bar, v in zip(b1, val_native):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(val_native + val_docker) * 0.01,
                f'{v:{fmt}}{suffix}', ha='center', va='bottom',
                fontsize=8, fontweight='bold', color=C_NATIVE)
    for bar, v in zip(b2, val_docker):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(val_native + val_docker) * 0.01,
                f'{v:{fmt}}{suffix}', ha='center', va='bottom',
                fontsize=8, fontweight='bold', color=C_DOCKER)
    ax.set_title(title, fontweight='bold', pad=8)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)

def speedup_label(ax, val_n, val_d, unit=''):
    """Tambahkan label selisih/speedup di pojok kanan atas."""
    diff = val_d - val_n
    sign = '+' if diff > 0 else ''
    txt  = f"Δ {sign}{diff:.2f}{unit}"
    color = '#e74c3c' if diff > 0 else '#27ae60'
    ax.annotate(txt, xy=(0.97, 0.94), xycoords='axes fraction',
                ha='right', va='top', fontsize=9, fontweight='bold',
                color=color,
                bbox=dict(boxstyle='round,pad=0.3', fc='white',
                          ec=color, alpha=0.8))

# ==========================================
# 4. LAYOUT UTAMA (5 BARIS × 3 KOLOM)
# ==========================================
fig = plt.figure(figsize=(22, 26))
fig.patch.set_facecolor('#f8f9fa')
fig.suptitle(
    "Perbandingan Native OS vs Docker — XGBoost pada Dataset FWA\n"
    "Efisiensi Resource & Performa Model",
    fontsize=17, fontweight='bold', y=0.99, color='#2c3e50'
)
gs = gridspec.GridSpec(5, 3, figure=fig, hspace=0.50, wspace=0.35,
                       top=0.95, bottom=0.04, left=0.07, right=0.97)

# ── Baris 0: Tabel Ringkasan (full width) ───────────────────────
ax_tbl = fig.add_subplot(gs[0, :])
ax_tbl.axis('off')

metrics = ['Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)',
           'Duration (s)', 'Avg CPU (%)', 'Peak CPU (%)',
           'Avg RAM (MB)', 'Peak RAM (MB)']
keys_n  = ['accuracy', 'precision', 'recall', 'f1_score',
           'duration_sec', 'avg_cpu_pct', 'peak_cpu_pct',
           'avg_ram_mb', 'peak_ram_mb']
scale   = [100, 100, 100, 100, 1, 1, 1, 1, 1]  # ×100 untuk persen

rows = []
for m, k, s in zip(metrics, keys_n, scale):
    vn = sn[k] * s
    vd = sd[k] * s
    diff = vd - vn
    sign = '+' if diff > 0 else ''
    # Tentukan siapa yang lebih baik (untuk akurasi/f1 lebih tinggi = baik;
    # untuk waktu/cpu/ram lebih rendah = baik)
    better_higher = k in ('accuracy', 'precision', 'recall', 'f1_score')
    if abs(diff) < 1e-6:
        winner = 'Sama'
    elif better_higher:
        winner = 'Native ✓' if vn > vd else 'Docker ✓'
    else:
        winner = 'Native ✓' if vn < vd else 'Docker ✓'
    rows.append([m, f'{vn:.4f}', f'{vd:.4f}',
                 f'{sign}{diff:.4f}', winner])

col_labels = ['Metrik', 'Native', 'Docker', 'Δ (Docker−Native)', 'Lebih Baik']
tbl = ax_tbl.table(cellText=rows, colLabels=col_labels,
                   loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(9.5)
tbl.scale(1, 1.6)

# Warna header
for c in range(len(col_labels)):
    tbl[0, c].set_facecolor('#2c3e50')
    tbl[0, c].set_text_props(color='white', fontweight='bold')

# Warna baris data
for r in range(1, len(rows) + 1):
    winner_val = rows[r-1][4]
    for c in range(len(col_labels)):
        if c == 4:
            if 'Native' in winner_val:
                tbl[r, c].set_facecolor('#d5e8d4')
                tbl[r, c].set_text_props(color='#1a5c1a', fontweight='bold')
            elif 'Docker' in winner_val:
                tbl[r, c].set_facecolor('#ffe6cc')
                tbl[r, c].set_text_props(color='#7b3c00', fontweight='bold')
            else:
                tbl[r, c].set_facecolor('#f5f5f5')
        elif r % 2 == 0:
            tbl[r, c].set_facecolor('#f0f4f8')
        # Warna kolom Native & Docker
        if c == 1:
            tbl[r, c].set_text_props(color=C_NATIVE, fontweight='bold')
        elif c == 2:
            tbl[r, c].set_text_props(color=C_DOCKER, fontweight='bold')

ax_tbl.set_title('Tabel Perbandingan Lengkap: Native vs Docker',
                 fontweight='bold', fontsize=12, pad=10, color='#2c3e50')

# ── Baris 1: Metrik Akurasi Model ───────────────────────────────
ax_acc = fig.add_subplot(gs[1, 0])
bar_pair(ax_acc,
         ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
         [sn['accuracy']*100, sn['precision']*100,
          sn['recall']*100,   sn['f1_score']*100],
         [sd['accuracy']*100, sd['precision']*100,
          sd['recall']*100,   sd['f1_score']*100],
         'Perbandingan Akurasi Model', 'Nilai (%)', fmt='.3f', pct=False)
ax_acc.set_ylim(
    min(sn['accuracy'], sd['accuracy'])*100 - 0.5,
    max(sn['accuracy'], sd['accuracy'])*100 + 0.5
)

# ── Baris 1: Waktu Training ─────────────────────────────────────
ax_dur = fig.add_subplot(gs[1, 1])
envs = ['Native', 'Docker']
durs = [sn['duration_sec'], sd['duration_sec']]
bars = ax_dur.bar(envs, durs,
                  color=[C_NATIVE, C_DOCKER], edgecolor='white',
                  linewidth=0.8, width=0.45)
for bar, v in zip(bars, durs):
    ax_dur.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(durs)*0.02,
                f'{v:.2f} det', ha='center', fontweight='bold', fontsize=10)
ax_dur.set_title('Waktu Training', fontweight='bold')
ax_dur.set_ylabel('Detik')
ax_dur.grid(True, alpha=0.3, axis='y'); ax_dur.set_axisbelow(True)
speedup_label(ax_dur, sn['duration_sec'], sd['duration_sec'], ' det')

# ── Baris 1: CPU & RAM Rata-rata ────────────────────────────────
ax_res = fig.add_subplot(gs[1, 2])
bar_pair(ax_res,
         ['Avg CPU', 'Peak CPU', 'Avg RAM\n(/10)', 'Peak RAM\n(/10)'],
         [sn['avg_cpu_pct'], sn['peak_cpu_pct'],
          sn['avg_ram_mb']/10, sn['peak_ram_mb']/10],
         [sd['avg_cpu_pct'], sd['peak_cpu_pct'],
          sd['avg_ram_mb']/10, sd['peak_ram_mb']/10],
         'CPU & RAM (RAM dibagi 10 untuk skala)', 'Nilai',
         fmt='.1f', pct=False)

# ── Baris 2: CPU Usage Timeline ─────────────────────────────────
ax_cpu = fig.add_subplot(gs[2, :2])
ax_cpu.plot(rn['Waktu_Detik'], rn['CPU_Percent'],
            color=C_NATIVE, linewidth=1.3, label='Native', alpha=0.85)
ax_cpu.plot(rd['Waktu_Detik'], rd['CPU_Percent'],
            color=C_DOCKER, linewidth=1.3, label='Docker', alpha=0.85)
ax_cpu.axhline(sn['avg_cpu_pct'], color=C_NATIVE, linestyle='--',
               linewidth=0.9, alpha=0.6,
               label=f"Avg Native {sn['avg_cpu_pct']:.1f}%")
ax_cpu.axhline(sd['avg_cpu_pct'], color=C_DOCKER, linestyle='--',
               linewidth=0.9, alpha=0.6,
               label=f"Avg Docker {sd['avg_cpu_pct']:.1f}%")
ax_cpu.set_title('CPU Usage Timeline: Native vs Docker', fontweight='bold')
ax_cpu.set_xlabel('Waktu (detik)'); ax_cpu.set_ylabel('CPU (%)')
ax_cpu.legend(fontsize=8); ax_cpu.grid(True, alpha=0.3)

# ── Baris 2: RAM Usage Timeline ─────────────────────────────────
ax_ram = fig.add_subplot(gs[2, 2])
ax_ram.fill_between(rn['Waktu_Detik'], rn['RAM_MB'],
                    alpha=0.25, color=C_NATIVE)
ax_ram.fill_between(rd['Waktu_Detik'], rd['RAM_MB'],
                    alpha=0.25, color=C_DOCKER)
ax_ram.plot(rn['Waktu_Detik'], rn['RAM_MB'],
            color=C_NATIVE, linewidth=1.3, label='Native')
ax_ram.plot(rd['Waktu_Detik'], rd['RAM_MB'],
            color=C_DOCKER, linewidth=1.3, label='Docker')
ax_ram.set_title('RAM Usage Timeline: Native vs Docker', fontweight='bold')
ax_ram.set_xlabel('Waktu (detik)'); ax_ram.set_ylabel('RAM (MB)')
ax_ram.legend(fontsize=8); ax_ram.grid(True, alpha=0.3)
speedup_label(ax_ram, sn['peak_ram_mb'], sd['peak_ram_mb'], ' MB')

# ── Baris 3: Loss Curve ─────────────────────────────────────────
ax_loss = fig.add_subplot(gs[3, :2])
ax_loss.plot(ln['estimator'], ln['train_loss'],
             color=C_NATIVE, linewidth=1.4, linestyle='-',
             label='Native — Train Loss')
ax_loss.plot(ln['estimator'], ln['val_loss'],
             color=C_NATIVE, linewidth=1.4, linestyle='--',
             label='Native — Val Loss')
ax_loss.plot(ld['estimator'], ld['train_loss'],
             color=C_DOCKER, linewidth=1.4, linestyle='-',
             label='Docker — Train Loss')
ax_loss.plot(ld['estimator'], ld['val_loss'],
             color=C_DOCKER, linewidth=1.4, linestyle='--',
             label='Docker — Val Loss')
ax_loss.set_title('Kurva Loss Training vs Validasi: Native vs Docker',
                  fontweight='bold')
ax_loss.set_xlabel('Estimator (Pohon)')
ax_loss.set_ylabel('Log Loss')
ax_loss.legend(fontsize=8); ax_loss.grid(True, alpha=0.3)

# ── Baris 3: Confusion Matrix Diff ──────────────────────────────
ax_cm = fig.add_subplot(gs[3, 2])
cm_n = np.array([[sn['cm_tn'], sn['cm_fp']],
                  [sn['cm_fn'], sn['cm_tp']]])
cm_d = np.array([[sd['cm_tn'], sd['cm_fp']],
                  [sd['cm_fn'], sd['cm_tp']]])
cm_diff = cm_d.astype(int) - cm_n.astype(int)

cmap_div = LinearSegmentedColormap.from_list(
    'div', ['#2980b9', 'white', '#e67e22'])
im = ax_cm.imshow(cm_diff, cmap=cmap_div,
                  vmin=-abs(cm_diff).max(), vmax=abs(cm_diff).max())
plt.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
for i in range(2):
    for j in range(2):
        v = cm_diff[i, j]
        ax_cm.text(j, i, f'{v:+d}', ha='center', va='center',
                   fontsize=14, fontweight='bold',
                   color='white' if abs(v) > cm_diff.max()*0.5 else '#2c3e50')
ax_cm.set_title('Confusion Matrix Δ (Docker − Native)', fontweight='bold')
ax_cm.set_xticks([0, 1]); ax_cm.set_yticks([0, 1])
ax_cm.set_xticklabels(['Pred NEG', 'Pred POS'])
ax_cm.set_yticklabels(['Actual NEG', 'Actual POS'])
ax_cm.set_xlabel('Biru = Native lebih baik  |  Oranye = Docker lebih baik',
                 fontsize=7)

# ── Baris 4: Feature Importance Perbandingan ────────────────────
ax_fi = fig.add_subplot(gs[4, :])
merged = pd.concat([fn, fd], axis=1).fillna(0)
top15  = merged['native'].add(merged['docker']).nlargest(15).index
top_fi = merged.loc[top15].sort_values('native')

y_pos = np.arange(len(top_fi))
w = 0.38
ax_fi.barh(y_pos - w/2, top_fi['native'], w,
           color=C_NATIVE, label='Native', edgecolor='white', linewidth=0.5)
ax_fi.barh(y_pos + w/2, top_fi['docker'], w,
           color=C_DOCKER, label='Docker', edgecolor='white', linewidth=0.5)
ax_fi.set_yticks(y_pos); ax_fi.set_yticklabels(top_fi.index, fontsize=8)
ax_fi.set_title('Top-15 Feature Importance: Native vs Docker',
                fontweight='bold')
ax_fi.set_xlabel('Importance Score')
ax_fi.legend(fontsize=9); ax_fi.grid(True, alpha=0.3, axis='x')
ax_fi.set_axisbelow(True)

# ==========================================
# 5. SIMPAN GRAFIK
# ==========================================
os.makedirs(OUTPUT_DIR, exist_ok=True)
out_path = f'{OUTPUT_DIR}/fwa_comparison_report.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"\n[compare] Grafik perbandingan tersimpan: {out_path}")
plt.show()