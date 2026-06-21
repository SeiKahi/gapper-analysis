import pandas as pd, glob, os, numpy as np

# 1. raw_1min
print("=" * 60)
print("1. RAW_1MIN")
files = glob.glob('data/raw_1min/**/*.parquet', recursive=True)
print(f"  Total files: {len(files)}")
if files:
    df = pd.read_parquet(files[0])
    print(f"  Sample file: {files[0]}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Dtypes:\n{df.dtypes}")
    print(f"  Head:\n{df.head(3).to_string()}")
    if 'time_et' in df.columns:
        print(f"  Time range: {df['time_et'].min()} to {df['time_et'].max()}")
        print(f"  Pre-market bars (before 09:30): {(df['time_et'] < '09:30').sum()}")
    # Check all time-like columns
    for c in df.columns:
        if 'time' in c.lower() or 'date' in c.lower() or 'timestamp' in c.lower():
            print(f"  Column '{c}': min={df[c].min()}, max={df[c].max()}")
else:
    print("  No files found!")

# 2. daily
print("\n" + "=" * 60)
print("2. DAILY")
daily_files = glob.glob('data/daily/**/*.parquet', recursive=True)
print(f"  Total files: {len(daily_files)}")
if daily_files:
    df2 = pd.read_parquet(daily_files[0])
    print(f"  Sample: {daily_files[0]}")
    print(f"  Shape: {df2.shape}, Columns: {list(df2.columns)}")
    print(f"  Head:\n{df2.head(3).to_string()}")
else:
    print("  No files found!")

# 3. volume_profile
print("\n" + "=" * 60)
print("3. VOLUME_PROFILE")
vp_files = glob.glob('data/volume_profile/**/*.parquet', recursive=True)
print(f"  Total files: {len(vp_files)}")
if vp_files:
    df3 = pd.read_parquet(vp_files[0])
    print(f"  Sample: {vp_files[0]}")
    print(f"  Shape: {df3.shape}, Columns: {list(df3.columns)}")
    print(f"  Head:\n{df3.head(3).to_string()}")
else:
    print("  No files found!")

# 4. normalized
print("\n" + "=" * 60)
print("4. NORMALIZED")
norm_files = glob.glob('data/normalized/**/*.parquet', recursive=True)
print(f"  Total files: {len(norm_files)}")
if norm_files:
    df4 = pd.read_parquet(norm_files[0])
    print(f"  Sample: {norm_files[0]}")
    print(f"  Shape: {df4.shape}, Columns: {list(df4.columns)}")
    print(f"  Head:\n{df4.head(3).to_string()}")
else:
    print("  No files found!")

# 5. metadata - ALL columns
print("\n" + "=" * 60)
print("5. METADATA (all columns)")
meta = pd.read_parquet('data/metadata/metadata_master.parquet')
print(f"  Shape: {meta.shape}")
print(f"  Columns and dtypes:")
for col in meta.columns:
    non_null = meta[col].notna().sum()
    print(f"    {col:<35} {str(meta[col].dtype):<12} non-null: {non_null}/{len(meta)}")

# 6. Check a few metadata rows
print(f"\n  Sample rows (first 2):")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
print(meta.head(2).T.to_string())

# 7. Unique values for categorical columns
for col in ['gap_direction', 'sector', 'catalyst_type']:
    if col in meta.columns:
        print(f"\n  {col} unique values: {meta[col].nunique()}")
        print(f"    {meta[col].value_counts().head(10).to_string()}")

# 8. Also check data/vwap for comparison
print("\n" + "=" * 60)
print("6. VWAP (for reference)")
vwap_files = glob.glob('data/vwap/**/*.parquet', recursive=True)
print(f"  Total files: {len(vwap_files)}")
if vwap_files:
    df5 = pd.read_parquet(vwap_files[0])
    print(f"  Sample: {vwap_files[0]}")
    print(f"  Shape: {df5.shape}, Columns: {list(df5.columns)}")
    print(f"  Head:\n{df5.head(3).to_string()}")

# 9. List top-level data directories
print("\n" + "=" * 60)
print("7. ALL DATA DIRECTORIES")
for item in os.listdir('data'):
    full = os.path.join('data', item)
    if os.path.isdir(full):
        count = len(glob.glob(os.path.join(full, '**/*.parquet'), recursive=True))
        print(f"  data/{item}/  -> {count} parquet files")
    else:
        print(f"  data/{item}  (file)")
