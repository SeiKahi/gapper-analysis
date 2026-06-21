import pandas as pd
import os

# 1. Metadata columns
meta = pd.read_parquet('data/metadata/metadata_master.parquet')
print('=== METADATA COLUMNS ===')
for c in sorted(meta.columns):
    sample_val = meta[c].dropna().iloc[0] if meta[c].notna().any() else "EMPTY"
    print(f'  {c}: {meta[c].dtype}, non-null={meta[c].notna().sum()}, sample={sample_val}')

# 2. Half 1 filter
meta['date'] = pd.to_datetime(meta['date'])
h1 = meta[(meta['date'] >= '2021-02-21') & (meta['date'] <= '2023-12-31')]
print(f'\nHalf 1: {len(h1)} rows')
print(f'GapUp: {(h1["gap_direction"]=="up").sum()}, GapDown: {(h1["gap_direction"]=="down").sum()}')

# 3. Check key columns
for col in ['rvol_30','adr_10','gap_size_in_adr','spy_return_day','prev_close','open_price',
            'prior_10d_return','rsi_14_prev','opening_drive_direction','close_price',
            'intraday_high','intraday_low','close_location']:
    if col in h1.columns:
        if h1[col].dtype in ['float64','int64']:
            print(f'{col}: non-null={h1[col].notna().sum()}, mean={h1[col].mean():.4f}')
        else:
            print(f'{col}: non-null={h1[col].notna().sum()}, dtype={h1[col].dtype}')
    else:
        print(f'{col}: MISSING')

# 4. Sample raw_1min file
sample_ticker = h1.iloc[0]['ticker']
sample_date = h1.iloc[0]['date'].strftime('%Y-%m-%d')
raw_path = f'data/raw_1min/{sample_ticker}/{sample_date}.parquet'
if os.path.exists(raw_path):
    raw = pd.read_parquet(raw_path)
    print(f'\n=== RAW_1MIN SAMPLE ({raw_path}) ===')
    print(f'Columns: {list(raw.columns)}')
    print(f'Sessions: {raw["session"].unique() if "session" in raw.columns else "NO SESSION COL"}')
    print(f'Shape: {raw.shape}')
    print(raw.head(3).to_string())
else:
    print(f'File not found: {raw_path}')

# 5. Sample VWAP file
vwap_path = f'data/vwap/{sample_ticker}/{sample_date}.parquet'
if os.path.exists(vwap_path):
    vwap = pd.read_parquet(vwap_path)
    print(f'\n=== VWAP SAMPLE ({vwap_path}) ===')
    print(f'Columns: {list(vwap.columns)}')
    print(vwap.head(3).to_string())
else:
    print(f'VWAP file not found: {vwap_path}')

# 6. Sample baseline_1min
base_path = f'data/baseline_1min/{sample_ticker}/{sample_date}.parquet'
if os.path.exists(base_path):
    base = pd.read_parquet(base_path)
    print(f'\n=== BASELINE_1MIN SAMPLE ({base_path}) ===')
    print(f'Columns: {list(base.columns)}')
    print(f'Shape: {base.shape}')
    print(base.head(3).to_string())
else:
    print(f'Baseline file not found: {base_path}')
