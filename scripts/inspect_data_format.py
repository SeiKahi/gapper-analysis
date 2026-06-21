"""Quick inspection of data file formats for RL pipeline."""
import pandas as pd
import os

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

os.chdir(str(PROJECT_ROOT))

# raw_1min
t = os.listdir('data/raw_1min')[0]
d = os.listdir(f'data/raw_1min/{t}')[0]
df = pd.read_parquet(f'data/raw_1min/{t}/{d}')
print(f'=== RAW_1MIN {t}/{d} ===')
print(f'Shape: {df.shape}')
print(f'Columns: {list(df.columns)}')
if 'session' in df.columns:
    print(f'Sessions: {df["session"].unique()}')
print(df.head(2).to_string())
print()

# baseline
t2 = os.listdir('data/baseline_1min')[0]
d2 = os.listdir(f'data/baseline_1min/{t2}')[0]
df2 = pd.read_parquet(f'data/baseline_1min/{t2}/{d2}')
print(f'=== BASELINE {t2}/{d2} ===')
print(f'Shape: {df2.shape}, Cols: {list(df2.columns)}')
print()

# vwap
t3 = os.listdir('data/vwap')[0]
d3 = os.listdir(f'data/vwap/{t3}')[0]
df3 = pd.read_parquet(f'data/vwap/{t3}/{d3}')
print(f'=== VWAP {t3}/{d3} ===')
print(f'Shape: {df3.shape}, Cols: {list(df3.columns)}')
print()

# daily
files = os.listdir('data/daily')[:2]
print(f'Daily: {len(os.listdir("data/daily"))} files')
df4 = pd.read_parquet(f'data/daily/{files[0]}')
print(f'Shape: {df4.shape}, Cols: {list(df4.columns)}')
print(df4.head(2).to_string())
print()

# Check POLYGON_API_KEY
key = os.environ.get('POLYGON_API_KEY', '')
print(f'POLYGON_API_KEY set: {bool(key)}, len: {len(key)}')
