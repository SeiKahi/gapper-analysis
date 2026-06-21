import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import glob

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

meta = pd.read_parquet(str(PROJECT_ROOT / 'data' / 'metadata' / 'metadata_master.parquet'))
val_meta = meta[(meta['date'] > '2022-12-31') & (meta['date'] <= '2023-12-31')]
sample = val_meta.head(5)

for _, r in sample.iterrows():
    t = r['ticker']
    d = r['date']
    path = os.path.join(str(PROJECT_ROOT / 'data' / 'vwap'), t, f'{t}_{d}.parquet')
    print(f'{t} {d} -> {path} exists={os.path.exists(path)}')

# Also check what the actual file structure looks like
vwap_dir = str(PROJECT_ROOT / 'data' / 'vwap')
subdirs = os.listdir(vwap_dir)[:5]
for sd in subdirs:
    files = os.listdir(os.path.join(vwap_dir, sd))[:3]
    print(f"  {sd}/: {files}")
