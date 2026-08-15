"""
tabel_hasil.py — Generate Tabel Hasil 20 Trial Native vs Docker
Output: ./hasil_monitoring/tabel_20_trial.png
"""
import os, sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.table import Table

OUTPUT_DIR = './hasil_monitoring'
C_NATIVE = '#2980b9'
C_DOCKER = '#e67e22'

# ── Load Data ────────────────────────────────────────────────
dn = pd.read_csv(f'{OUTPUT_DIR}/multi_trial_native.csv')
dd = pd.read_csv(f'{OUTPUT_DIR}/multi_trial_docker.csv')

# ── Helper: hitung mean & std ────────────────────────────────
METRIK = ['accuracy', 'precision', 'recall', 'f1_score', 'duration_sec',
          'avg_cpu_pct', 'peak_cpu_pct', 'avg_ram_mb', 'peak_ram_mb', 'e2e_sec']

def mean_std_str(col, df):
    m = df[col].mean()
    s = df[col].std()
    if col in ['accuracy', 'precision', 'recall', 'f1_score']:
        m *= 100; s *= 100
        return f"{m:.4f} ± {s:.4f}"
    elif col in ['avg_ram_mb', 'peak_ram_mb']:
        return f"{m:.2f} ± {s:.2f}"
    elif col in ['avg_cpu_pct', 'peak_cpu_pct']:
        return f"{m:.2f} ± {s:.2f}"
    else:
        return f"{m:.4f} ± {s:.4f}"

def val_str(col, val):
    if col in ['accuracy', 'precision', 'recall', 'f1_score']:
        return f"{val*100:.4f}"
    elif col in ['avg_ram_mb', 'peak_ram_mb']:
        return f"{val:.2f}"
    elif col in ['duration_sec', 'e2e_sec']:
        return f"{val:.4f}"
    elif col in ['avg_cpu_pct', 'peak_cpu_pct']:
        return f"{val:.2f}"
    return f"{val}"

# ── Build Figure ──────────────────────────────────────────────
fig = plt.figure(figsize=(28, 32))
fig.patch.set_facecolor('#f8f9fa')
fig.suptitle(
    "Hasil 20 Trial XGBoost — Native OS vs Docker (Pembatasan 2 CPU / 4 GB)\n"
    "Dataset ISCX 2012 — 100.000 Sampel",
    fontsize=18, fontweight='bold', y=0.98, color='#2c3e50'
)

# ── Tabel 1: Native 20 Trial ─────────────────────────────────
ax1 = fig.add_axes([0.03, 0.56, 0.44, 0.38])
ax1.axis('off')
ax1.set_title('Tabel 1: Native OS — 20 Kali Percobaan', fontweight='bold',
              fontsize=13, pad=8, color=C_NATIVE)

cols_show = ['trial', 'accuracy', 'precision', 'recall', 'f1_score',
             'duration_sec', 'avg_cpu_pct', 'peak_cpu_pct', 'avg_ram_mb', 'peak_ram_mb', 'e2e_sec']
headers  = ['Trial', 'Acc (%)', 'Prec (%)', 'Rec (%)', 'F1 (%)',
            'Dur (s)', 'CPU Avg (%)', 'CPU Punc (%)', 'RAM Avg (MB)', 'RAM Punc (MB)', 'E2E (s)']

tbl_data_n = []
for _, r in dn.iterrows():
    tbl_data_n.append([str(int(r['trial']))] + [val_str(c, r[c]) for c in cols_show[1:]])

# Row rata-rata
avg_row_n = ['Rata-rata'] + [mean_std_str(c, dn) for c in cols_show[1:]]
tbl_data_n.append(avg_row_n)

table1 = ax1.table(cellText=tbl_data_n, colLabels=headers,
                   loc='center', cellLoc='center', colWidths=[0.08]*len(headers))
table1.auto_set_font_size(False)
table1.set_fontsize(7)
table1.scale(1, 1.3)

# Style header
for c in range(len(headers)):
    table1[0, c].set_facecolor(C_NATIVE)
    table1[0, c].set_text_props(color='white', fontweight='bold')

# Style baris data
for r in range(1, len(tbl_data_n)+1):
    is_avg = (r == len(tbl_data_n))
    for c in range(len(headers)):
        if is_avg:
            table1[r, c].set_facecolor('#d5e8d4')
            table1[r, c].set_text_props(fontweight='bold')
        elif r % 2 == 0:
            table1[r, c].set_facecolor('#f0f4f8')

# ── Tabel 2: Docker 20 Trial ────────────────────────────────
ax2 = fig.add_axes([0.53, 0.56, 0.44, 0.38])
ax2.axis('off')
ax2.set_title('Tabel 2: Docker (2 CPU / 4 GB) — 20 Kali Percobaan', fontweight='bold',
              fontsize=13, pad=8, color=C_DOCKER)

tbl_data_d = []
for _, r in dd.iterrows():
    tbl_data_d.append([str(int(r['trial']))] + [val_str(c, r[c]) for c in cols_show[1:]])

avg_row_d = ['Rata-rata'] + [mean_std_str(c, dd) for c in cols_show[1:]]
tbl_data_d.append(avg_row_d)

table2 = ax2.table(cellText=tbl_data_d, colLabels=headers,
                   loc='center', cellLoc='center', colWidths=[0.08]*len(headers))
table2.auto_set_font_size(False)
table2.set_fontsize(7)
table2.scale(1, 1.3)

for c in range(len(headers)):
    table2[0, c].set_facecolor(C_DOCKER)
    table2[0, c].set_text_props(color='white', fontweight='bold')

for r in range(1, len(tbl_data_d)+1):
    is_avg = (r == len(tbl_data_d))
    for c in range(len(headers)):
        if is_avg:
            table2[r, c].set_facecolor('#ffe6cc')
            table2[r, c].set_text_props(fontweight='bold')
        elif r % 2 == 0:
            table2[r, c].set_facecolor('#fef9f0')

# ── Tabel 3: Perbandingan ───────────────────────────────────
ax3 = fig.add_axes([0.1, 0.05, 0.8, 0.42])
ax3.axis('off')
ax3.set_title('Tabel 3: Perbandingan Metrik Performa & Resource — Native vs Docker',
              fontweight='bold', fontsize=14, pad=10, color='#2c3e50')

compare_headers = ['Metrik', 'Native (Rata-rata)', 'Docker (Rata-rata)', 'Δ (Docker - Native)', 'Selisih (%)', 'Lebih Baik']

def delta_str(col, df_n, df_d):
    m_n = df_n[col].mean(); m_d = df_d[col].mean()
    diff = m_d - m_n
    sign = '+' if diff > 0 else ''
    if col in ['accuracy', 'precision', 'recall', 'f1_score']:
        return f"{sign}{diff*100:.4f}"
    return f"{sign}{diff:.4f}"

def pct_str(col, df_n, df_d):
    m_n = df_n[col].mean(); m_d = df_d[col].mean()
    if m_n == 0: return 'N/A'
    pct = ((m_d - m_n) / abs(m_n)) * 100
    return f"{'+' if pct > 0 else ''}{pct:.2f}%"

def winner_str(col, df_n, df_d):
    m_n = df_n[col].mean(); m_d = df_d[col].mean()
    better_higher = col in ('accuracy', 'precision', 'recall', 'f1_score')
    if abs(m_d - m_n) < 1e-10:
        return 'Sama'
    if better_higher:
        return 'Native ✓' if m_n > m_d else 'Docker ✓'
    else:
        return 'Native ✓' if m_n < m_d else 'Docker ✓'

METRIK_LABEL = {
    'accuracy': 'Accuracy (%)', 'precision': 'Precision (%)',
    'recall': 'Recall (%)', 'f1_score': 'F1-Score (%)',
    'duration_sec': 'Durasi Training (s)', 'e2e_sec': 'Waktu End-to-End (s)',
    'avg_cpu_pct': 'CPU Rata-rata (%)', 'peak_cpu_pct': 'CPU Puncak (%)',
    'avg_ram_mb': 'RAM Rata-rata (MB)', 'peak_ram_mb': 'RAM Puncak (MB)'
}

compare_rows = []
for col in METRIK:
    m_n = dn[col].mean()
    m_d = dd[col].mean()
    compare_rows.append([
        METRIK_LABEL[col],
        val_str(col, m_n),
        val_str(col, m_d),
        delta_str(col, dn, dd),
        pct_str(col, dn, dd),
        winner_str(col, dn, dd)
    ])

# Additional metrics: n_train, n_test (should be same)
n_train_n = dn['n_train'].mean(); n_train_d = dd['n_train'].mean()
n_test_n  = dn['n_test'].mean();  n_test_d  = dd['n_test'].mean()
compare_rows.append(['n_train (Jumlah Train)', f"{n_train_n:.0f}", f"{n_train_d:.0f}",
                     f"{n_train_d - n_train_n:.0f}", '0.00%', 'Sama'])
compare_rows.append(['n_test (Jumlah Test)', f"{n_test_n:.0f}", f"{n_test_d:.0f}",
                     f"{n_test_d - n_test_n:.0f}", '0.00%', 'Sama'])

table3 = ax3.table(cellText=compare_rows, colLabels=compare_headers,
                   loc='center', cellLoc='center')
table3.auto_set_font_size(False)
table3.set_fontsize(10)
table3.scale(1, 1.8)

# Style header
for c in range(len(compare_headers)):
    table3[0, c].set_facecolor('#2c3e50')
    table3[0, c].set_text_props(color='white', fontweight='bold')

# Style baris
for r in range(1, len(compare_rows)+1):
    for c in range(len(compare_headers)):
        if c == 5:  # kolom "Lebih Baik"
            val = compare_rows[r-1][5]
            if 'Native' in val:
                table3[r, c].set_facecolor('#d5e8d4')
                table3[r, c].set_text_props(color='#1a5c1a', fontweight='bold')
            elif 'Docker' in val:
                table3[r, c].set_facecolor('#ffe6cc')
                table3[r, c].set_text_props(color='#7b3c00', fontweight='bold')
            else:
                table3[r, c].set_facecolor('#f5f5f5')
        elif c == 1:  # Native
            table3[r, c].set_text_props(color=C_NATIVE, fontweight='bold')
        elif c == 2:  # Docker
            table3[r, c].set_text_props(color=C_DOCKER, fontweight='bold')
        elif r % 2 == 0:
            table3[r, c].set_facecolor('#f0f4f8')
        # Warna selisih
        if c == 3 and r <= len(METRIK):
            val = compare_rows[r-1][3]
            if val.startswith('+'):
                is_bad = compare_rows[r-1][0] in ['Durasi Training (s)', 'Waktu End-to-End (s)',
                                                   'CPU Rata-rata (%)', 'CPU Puncak (%)',
                                                   'RAM Rata-rata (MB)', 'RAM Puncak (MB)']
                if is_bad:
                    table3[r, c].set_facecolor('#ffe6cc')
                    table3[r, c].set_text_props(color='#c0392b', fontweight='bold')
                else:
                    table3[r, c].set_facecolor('#d5e8d4')
                    table3[r, c].set_text_props(color='#27ae60', fontweight='bold')
            elif val.startswith('-'):
                is_good = compare_rows[r-1][0] in ['Durasi Training (s)', 'Waktu End-to-End (s)',
                                                    'CPU Rata-rata (%)', 'CPU Puncak (%)',
                                                    'RAM Rata-rata (MB)', 'RAM Puncak (MB)']
                if is_good:
                    table3[r, c].set_facecolor('#d5e8d4')
                    table3[r, c].set_text_props(color='#27ae60', fontweight='bold')
                else:
                    table3[r, c].set_facecolor('#ffe6cc')
                    table3[r, c].set_text_props(color='#c0392b', fontweight='bold')

# ── Simpan ──────────────────────────────────────────────────
output_path = f'{OUTPUT_DIR}/tabel_20_trial.png'
fig.savefig(output_path, dpi=200, bbox_inches='tight')
print(f"\n✅ Tabel tersimpan: {output_path}")
print(f"\n{'='*70}")
print(f"  RINGKASAN PERBANDINGAN — Rata-rata 20 Trial")
print(f"{'='*70}")
print(f"  {'Metrik':<25} {'Native':>15} {'Docker':>15} {'Δ':>15}")
print(f"  {'-'*70}")
for col in METRIK:
    m_n = dn[col].mean(); m_d = dd[col].mean()
    diff = m_d - m_n
    s_n = dn[col].std(); s_d = dd[col].std()
    if col in ['accuracy', 'precision', 'recall', 'f1_score']:
        print(f"  {METRIK_LABEL[col]:<25} {m_n*100:>8.4f}±{s_n*100:.4f} {m_d*100:>8.4f}±{s_d*100:.4f} {diff*100:>+10.4f}")
    elif col in ['avg_ram_mb', 'peak_ram_mb']:
        print(f"  {METRIK_LABEL[col]:<25} {m_n:>8.2f}±{s_n:.2f} {m_d:>8.2f}±{s_d:.2f} {diff:>+10.2f}")
    elif col in ['duration_sec', 'e2e_sec']:
        print(f"  {METRIK_LABEL[col]:<25} {m_n:>8.4f}±{s_n:.4f} {m_d:>8.4f}±{s_d:.4f} {diff:>+10.4f}")
    else:
        print(f"  {METRIK_LABEL[col]:<25} {m_n:>8.2f}±{s_n:.2f} {m_d:>8.2f}±{s_d:.2f} {diff:>+10.2f}")
print(f"{'='*70}")
