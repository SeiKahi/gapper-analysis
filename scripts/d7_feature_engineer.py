"""
Durchlauf 7.0 — Aufgabe 0: Feature Engineering
Berechnet RVOL_5, PM/RTH5, rest_drift, close_935, od_strength, first_candle etc.
Speichert erweiterte Metadata als metadata_v7.parquet

Ausfuehrung: .\gapper_env\Scripts\python.exe scripts/d7_feature_engineer.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import warnings
import time
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
BASE = Path('.')
RAW_DIR = BASE / 'data' / 'raw_1min'
BASELINE_DIR = BASE / 'data' / 'baseline_1min'
META_PATH = BASE / 'data' / 'metadata' / 'metadata_master.parquet'
OUT_PATH = BASE / 'data' / 'metadata' / 'metadata_v7.parquet'
REPORT_PATH = BASE / 'results' / 'd7_feature_engineer.txt'

TEST_MODE = '--test' in sys.argv  # Run on 100 gappers for testing

# ══════════════════════════════════════════════════════════════════════════════
# LOAD METADATA
# ══════════════════════════════════════════════════════════════════════════════
print("Loading metadata...", file=sys.stderr)
meta = pd.read_parquet(META_PATH)
meta['date'] = pd.to_datetime(meta['date'])
print(f"Total gappers: {len(meta)}", file=sys.stderr)

if TEST_MODE:
    meta = meta.head(100)
    print(f"TEST MODE: Processing {len(meta)} gappers", file=sys.stderr)

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

# Pre-allocate new columns
new_cols = {
    'rvol_5': np.nan, 'pm_rth5': np.nan,
    'close_935': np.nan, 'close_1000': np.nan, 'close_1100': np.nan,
    'rest_drift': np.nan, 'rest_drift_1000': np.nan, 'rest_drift_1100': np.nan,
    'od_strength': np.nan, 'od_direction': '', 'od_long': np.nan,
    'first_candle_dir': '', 'first_candle_size': np.nan,
    'full_drift': np.nan, 'cl': np.nan,
    'pm_vol': np.nan, 'rth5_vol': np.nan, 'rth30_vol': np.nan,
    'pm_rth30_computed': np.nan,
}

for col, default in new_cols.items():
    if col not in meta.columns:
        meta[col] = default

records = []
errors = 0
skipped = 0
no_baseline = 0

t0 = time.time()

for idx, row in tqdm(meta.iterrows(), total=len(meta), desc="Feature Engineering", file=sys.stderr):
    ticker = row['ticker']
    date_str = row['date'].strftime('%Y-%m-%d')
    adr = row.get('adr_10', np.nan)
    gap_dir = row['gap_direction']
    rth_open_price = row.get('rth_open', np.nan)
    rth_close_price = row.get('rth_close', np.nan)
    rth_high_price = row.get('rth_high', np.nan)
    rth_low_price = row.get('rth_low', np.nan)
    prev_close = row.get('prev_close', np.nan)

    rec = {'idx': idx}

    if pd.isna(adr) or adr <= 0:
        skipped += 1
        records.append(rec)
        continue

    raw_path = RAW_DIR / ticker / f'{date_str}.parquet'
    if not raw_path.exists():
        skipped += 1
        records.append(rec)
        continue

    try:
        raw = pd.read_parquet(raw_path)
    except Exception:
        errors += 1
        records.append(rec)
        continue

    # ── Split sessions ──
    pm = raw[raw['session'] == 'premarket'] if 'session' in raw.columns else pd.DataFrame()
    rth = raw[raw['session'] == 'rth'] if 'session' in raw.columns else raw.copy()

    if len(rth) < 10:
        skipped += 1
        records.append(rec)
        continue

    rth = rth.sort_values('time_et').reset_index(drop=True)

    # ══════════════════════════════════════════════════════════════════════
    # 0a: RVOL_5 — from baseline_1min
    # ══════════════════════════════════════════════════════════════════════
    # First 5 RTH bars (09:30-09:34)
    rth5 = rth.head(5)
    vol_5min_today = rth5['volume'].sum()
    rec['rth5_vol'] = vol_5min_today

    # Use existing rvol_at_time_5min if available (already computed from baseline)
    existing_rvol5 = row.get('rvol_at_time_5min', np.nan)
    if pd.notna(existing_rvol5):
        rec['rvol_5'] = existing_rvol5
    else:
        # Fallback: compute from baseline_1min
        baseline_dir = BASELINE_DIR / ticker
        if baseline_dir.exists():
            baseline_files = sorted(baseline_dir.glob('*.parquet'))
            # Take last 10 files before this date
            baseline_files = [f for f in baseline_files if f.stem < date_str][-10:]
            if len(baseline_files) >= 1:
                baseline_vols = []
                for bf in baseline_files:
                    try:
                        bdf = pd.read_parquet(bf)
                        bdf = bdf.sort_values('time_et').reset_index(drop=True)
                        bvol = bdf.head(5)['volume'].sum()
                        if bvol > 0:
                            baseline_vols.append(bvol)
                    except Exception:
                        pass
                if len(baseline_vols) >= 1:
                    baseline_avg = np.mean(baseline_vols)
                    if baseline_avg > 0:
                        rec['rvol_5'] = vol_5min_today / baseline_avg
        if 'rvol_5' not in rec or pd.isna(rec.get('rvol_5')):
            no_baseline += 1

    # ══════════════════════════════════════════════════════════════════════
    # 0b: PM/RTH5
    # ══════════════════════════════════════════════════════════════════════
    pm_vol = pm['volume'].sum() if len(pm) > 0 else 0
    rec['pm_vol'] = pm_vol
    rec['pm_rth5'] = pm_vol / vol_5min_today if vol_5min_today > 0 else np.nan

    # Also compute PM/RTH30 for comparison
    rth30 = rth.head(30)
    rth30_vol = rth30['volume'].sum()
    rec['rth30_vol'] = rth30_vol
    rec['pm_rth30_computed'] = pm_vol / rth30_vol if rth30_vol > 0 else np.nan

    # ══════════════════════════════════════════════════════════════════════
    # 0c: Price at key times (close_935, close_1000, close_1100)
    # ══════════════════════════════════════════════════════════════════════
    # Bar at 09:35 = index 5 (09:30=0, 09:31=1, ..., 09:35=5)
    # Actually: 09:35 bar is the 6th bar (index 5), its close is the price at 9:35
    # But the OD is "first 5 bars" = 09:30-09:34, so close_935 is the close of bar 09:34 (index 4)
    # Wait - the task says: "close_935 = Close-Preis der Bar um 09:35 (letzte Bar des OD)"
    # The "bar at 09:35" in 1-min data would be the bar that opens at 09:35.
    # But the OD is first 5 bars: 09:30, 09:31, 09:32, 09:33, 09:34.
    # The close at 09:34 is the OD close (available at 09:35).
    # Let's use index 4 (09:34 bar close) = close_935
    if len(rth) > 4:
        rec['close_935'] = rth.iloc[4]['close']

    # close_1000 = 30th bar (09:30 + 30 = 10:00, index 29 = bar 09:59 close)
    if len(rth) > 29:
        rec['close_1000'] = rth.iloc[29]['close']

    # close_1100 = 90th bar (09:30 + 90 = 11:00, index 89 = bar 10:59 close)
    if len(rth) > 89:
        rec['close_1100'] = rth.iloc[89]['close']

    # ══════════════════════════════════════════════════════════════════════
    # 0c: Rest-Drift (from 9:35 to various times)
    # ══════════════════════════════════════════════════════════════════════
    close_935 = rec.get('close_935', np.nan)

    if pd.notna(close_935) and pd.notna(rth_close_price) and adr > 0:
        if gap_dir == 'up':
            rec['rest_drift'] = (rth_close_price - close_935) / adr
        else:
            rec['rest_drift'] = (close_935 - rth_close_price) / adr

    close_1000 = rec.get('close_1000', np.nan)
    if pd.notna(close_935) and pd.notna(close_1000) and adr > 0:
        if gap_dir == 'up':
            rec['rest_drift_1000'] = (close_1000 - close_935) / adr
        else:
            rec['rest_drift_1000'] = (close_935 - close_1000) / adr

    close_1100 = rec.get('close_1100', np.nan)
    if pd.notna(close_935) and pd.notna(close_1100) and adr > 0:
        if gap_dir == 'up':
            rec['rest_drift_1100'] = (close_1100 - close_935) / adr
        else:
            rec['rest_drift_1100'] = (close_935 - close_1100) / adr

    # ══════════════════════════════════════════════════════════════════════
    # 0d: Opening Drive features
    # ══════════════════════════════════════════════════════════════════════
    od_bars = rth.head(5)  # 09:30-09:34
    if len(od_bars) >= 5:
        od_open = od_bars.iloc[0]['open']
        od_close = od_bars.iloc[-1]['close']
        od_move = abs(od_close - od_open) / adr
        od_long = od_close > od_open

        rec['od_strength'] = od_move
        rec['od_long'] = od_long

        if gap_dir == 'up':
            rec['od_direction'] = 'with_gap' if od_long else 'against_gap'
        else:
            rec['od_direction'] = 'with_gap' if not od_long else 'against_gap'

    # ══════════════════════════════════════════════════════════════════════
    # 0d: First candle features (09:30 bar)
    # ══════════════════════════════════════════════════════════════════════
    first_bar = rth.iloc[0]
    candle_body = first_bar['close'] - first_bar['open']
    candle_size = abs(candle_body) / adr
    candle_up = candle_body > 0

    rec['first_candle_size'] = candle_size
    if gap_dir == 'up':
        rec['first_candle_dir'] = 'with_gap' if candle_up else 'against_gap'
    else:
        rec['first_candle_dir'] = 'with_gap' if not candle_up else 'against_gap'

    # ══════════════════════════════════════════════════════════════════════
    # 0d: Full drift and Close Location
    # ══════════════════════════════════════════════════════════════════════
    if pd.notna(rth_open_price) and pd.notna(rth_close_price) and adr > 0:
        if gap_dir == 'up':
            rec['full_drift'] = (rth_close_price - rth_open_price) / adr
        else:
            rec['full_drift'] = (rth_open_price - rth_close_price) / adr

    if pd.notna(rth_high_price) and pd.notna(rth_low_price) and (rth_high_price - rth_low_price) > 0:
        rec['cl'] = (rth_close_price - rth_low_price) / (rth_high_price - rth_low_price)

    records.append(rec)

elapsed = time.time() - t0
print(f"\nDone in {elapsed:.1f}s. Errors={errors}, Skipped={skipped}, No baseline={no_baseline}", file=sys.stderr)

# ══════════════════════════════════════════════════════════════════════════════
# MERGE RESULTS INTO METADATA
# ══════════════════════════════════════════════════════════════════════════════
print("Merging results...", file=sys.stderr)

# Convert records to dataframe
recs_df = pd.DataFrame(records)
recs_df = recs_df.set_index('idx')

# Update metadata with computed features
feature_cols = ['rvol_5', 'pm_rth5', 'close_935', 'close_1000', 'close_1100',
                'rest_drift', 'rest_drift_1000', 'rest_drift_1100',
                'od_strength', 'od_direction', 'od_long',
                'first_candle_dir', 'first_candle_size',
                'full_drift', 'cl',
                'pm_vol', 'rth5_vol', 'rth30_vol', 'pm_rth30_computed']

for col in feature_cols:
    if col in recs_df.columns:
        meta.loc[recs_df.index, col] = recs_df[col]

# ══════════════════════════════════════════════════════════════════════════════
# COVERAGE REPORT
# ══════════════════════════════════════════════════════════════════════════════
report = []
def w(s=""): report.append(s)

w("=" * 80)
w("DURCHLAUF 7.0 — AUFGABE 0: FEATURE ENGINEERING REPORT")
w("=" * 80)
w()
w(f"Total gappers: {len(meta)}")
w(f"Errors: {errors}")
w(f"Skipped (no adr/no file/too short): {skipped}")
w(f"No baseline for RVOL_5 fallback: {no_baseline}")
w(f"Processing time: {elapsed:.1f}s")
w()

w("--- COVERAGE ---")
for col in feature_cols:
    if col in meta.columns:
        valid = meta[col].notna().sum() if meta[col].dtype != 'object' else (meta[col] != '').sum()
        pct = valid / len(meta) * 100
        w(f"  {col:25s}: {valid:6d} / {len(meta)} ({pct:.1f}%)")
w()

# Numeric summaries
w("--- NUMERIC SUMMARIES ---")
num_cols = ['rvol_5', 'pm_rth5', 'close_935', 'rest_drift', 'rest_drift_1000', 'rest_drift_1100',
            'od_strength', 'first_candle_size', 'full_drift', 'cl', 'pm_rth30_computed']
for col in num_cols:
    if col in meta.columns and meta[col].dtype in ['float64', 'float32', 'int64']:
        s = meta[col].dropna()
        if len(s) > 0:
            w(f"  {col:25s}: mean={s.mean():.4f}, median={s.median():.4f}, "
              f"std={s.std():.4f}, min={s.min():.4f}, max={s.max():.4f}")
w()

# Categorical summaries
w("--- CATEGORICAL SUMMARIES ---")
for col in ['od_direction', 'first_candle_dir']:
    if col in meta.columns:
        vc = meta[col].value_counts()
        w(f"  {col}:")
        for val, cnt in vc.items():
            if val:
                w(f"    {val}: {cnt} ({cnt/len(meta)*100:.1f}%)")
w()

# RVOL_5 vs existing rvol_at_time_5min comparison
w("--- RVOL_5 VERIFICATION ---")
if 'rvol_5' in meta.columns and 'rvol_at_time_5min' in meta.columns:
    both = meta[meta['rvol_5'].notna() & meta['rvol_at_time_5min'].notna()]
    if len(both) > 0:
        corr = both['rvol_5'].corr(both['rvol_at_time_5min'])
        diff = (both['rvol_5'] - both['rvol_at_time_5min']).abs()
        w(f"  N with both: {len(both)}")
        w(f"  Correlation: {corr:.6f}")
        w(f"  Mean abs diff: {diff.mean():.6f}")
        w(f"  Max abs diff: {diff.max():.6f}")
        w(f"  => Using rvol_at_time_5min as rvol_5 where available")
w()

# PM/RTH5 vs PM/RTH30 comparison
w("--- PM/RTH5 vs PM/RTH30 PRELIMINARY ---")
if 'pm_rth5' in meta.columns and 'pm_rth30_computed' in meta.columns:
    both = meta[meta['pm_rth5'].notna() & meta['pm_rth30_computed'].notna()]
    if len(both) > 0:
        corr = both['pm_rth5'].corr(both['pm_rth30_computed'])
        w(f"  N with both: {len(both)}")
        w(f"  Correlation PM/RTH5 vs PM/RTH30: {corr:.4f}")
        w(f"  PM/RTH5  mean={both['pm_rth5'].mean():.4f}, median={both['pm_rth5'].median():.4f}")
        w(f"  PM/RTH30 mean={both['pm_rth30_computed'].mean():.4f}, median={both['pm_rth30_computed'].median():.4f}")
w()

# Split stats
w("--- SPLIT STATS ---")
h1 = meta[(meta['date'] >= '2021-02-21') & (meta['date'] <= '2023-12-31')]
h2 = meta[(meta['date'] >= '2024-01-01') & (meta['date'] <= '2026-02-06')]
w(f"  H1 (IS):  {len(h1)} gappers")
w(f"  H2 (OOS): {len(h2)} gappers")
w(f"  H1 rvol_5 valid: {h1['rvol_5'].notna().sum()}")
w(f"  H2 rvol_5 valid: {h2['rvol_5'].notna().sum()}")
w(f"  H1 pm_rth5 valid: {h1['pm_rth5'].notna().sum()}")
w(f"  H2 pm_rth5 valid: {h2['pm_rth5'].notna().sum()}")
w(f"  H1 rest_drift valid: {h1['rest_drift'].notna().sum()}")
w(f"  H2 rest_drift valid: {h2['rest_drift'].notna().sum()}")

# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════
# Ensure date is string for parquet compatibility
meta['date'] = meta['date'].dt.strftime('%Y-%m-%d')

meta.to_parquet(OUT_PATH, index=False)
print(f"\nSaved metadata_v7.parquet: {len(meta)} rows, {len(meta.columns)} columns", file=sys.stderr)

# Save report
with open(REPORT_PATH, 'w') as f:
    f.write('\n'.join(report))
print(f"Saved report: {REPORT_PATH}", file=sys.stderr)

# Print report to stdout too
print('\n'.join(report))
