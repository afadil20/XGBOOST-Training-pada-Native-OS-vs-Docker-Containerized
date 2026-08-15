"""
demo_data.py — Tampilkan info dataset untuk demo sidang
"""
import pandas as pd, glob, os

files = sorted(glob.glob('data/*.csv'))
print(f"📁 Jumlah file: {len(files)}\n")

rows_total = 0
for f in files:
    name = os.path.basename(f)
    df = pd.read_csv(f)
    rows_total += len(df)
    print(f"  {name:55s} → {len(df):>8,} baris")

print(f"\n{'─'*70}")
print(f"  {'TOTAL':55s} → {rows_total:>8,} baris")

# Gabung untuk lihat label
df_all = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
label_counts = df_all.iloc[:, -1].value_counts()
print(f"\n🏷️  Distribusi Label ({len(label_counts)} kelas):")
for i, (label, count) in enumerate(label_counts.items(), 1):
    pct = count / len(df_all) * 100
    print(f"  {i:2d}. {str(label):25s} → {count:>8,} baris ({pct:.4f}%)")

print(f"\n📊 Jumlah fitur: {df_all.shape[1] - 1} kolom numerik + 1 kolom label")
print(f"\n✅ Dataset siap digunakan.")
