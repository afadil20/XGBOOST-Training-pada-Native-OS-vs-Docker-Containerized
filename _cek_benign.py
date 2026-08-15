"""Cek distribusi BENIGN per file"""
import pandas as pd, glob

total_benign = 0
total_all = 0

for f in sorted(glob.glob('data/*.csv')):
    df = pd.read_csv(f)
    label_col = df.columns[-1]
    b = (df[label_col] == 'BENIGN').sum()
    total_benign += b
    total_all += len(df)
    print(f"{f.split('/')[-1].split(chr(92))[-1]:55s} BENIGN: {b:>8,}  total: {len(df):>8,}")

print(f"\nTotal BENIGN: {total_benign:,} dari {total_all:,} baris")
