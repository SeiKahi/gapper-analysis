"""
Durchlauf 4.5 — Aufgabe 1: SPY Context with Parameter Filters

For each SPY category (stark rot <-1%, leicht rot, leicht gruen, stark gruen >+1%):
  a) Gap Size: <1 ADR, 1-2 ADR, 2-3 ADR, >3 ADR
  b) RVOL (rvol_30): <1x, 1-2x, 2-3x, 3-5x, >5x
  c) ADR%: <5%, 5-10%, >10%

Metrics: N, Drift (Close-Open)/ADR, Fill-Rate, Close-Location, MFE, MAE
Trading window: Open to 11:00 for new trades.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import warnings
warnings.filterwarnings('ignore')

# === LOAD METADATA ===
meta = pd.read_parquet('data/metadata/metadata_master.parquet')
meta['date'] = pd.to_datetime(meta['date'])
h1 = meta[(meta['date'] >= '2021-02-21') & (meta['date'] <= '2023-12-31')].copy()
print(f"Half 1: {len(h1)} gapper days", file=sys.stderr)

# === COMPUTE MFE/MAE WITHIN TRADING WINDOW (to 11:00) ===
results = []

for _, row in tqdm(h1.iterrows(), total=len(h1), desc="Processing", file=sys.stderr):
    ticker = row['ticker']
    date_str = row['date'].strftime('%Y-%m-%d')
    adr = row['adr_10']
    gap_dir = row['gap_direction']
    if pd.isna(adr) or adr <= 0:
        continue

    raw_path = Path(f'data/raw_1min/{ticker}/{date_str}.parquet')
    if not raw_path.exists():
        continue

    try:
        raw = pd.read_parquet(raw_path)
    except Exception:
        continue

    rth = raw[raw['session'] == 'rth'].copy()
    if len(rth) < 10:
        continue

    rth = rth.sort_values('time_et').reset_index(drop=True)
    rth['min_idx'] = range(len(rth))

    open_price = rth.iloc[0]['open']
    prev_close = row.get('prev_close', np.nan)

    # Trading window: until 11:00 (90 min)
    window = rth[rth['min_idx'] <= 90]
    if len(window) < 5:
        continue

    # MFE/MAE in gap direction within trading window
    if gap_dir == 'up':
        mfe = (window['high'].max() - open_price) / adr
        mae = (open_price - window['low'].min()) / adr
    else:
        mfe = (open_price - window['low'].min()) / adr
        mae = (window['high'].max() - open_price) / adr

    # Close at 11:00
    close_1100 = window.iloc[-1]['close']
    drift_1100 = (close_1100 - open_price) / adr

    # Fill within window
    if pd.notna(prev_close):
        if gap_dir == 'up':
            filled = window['low'].min() <= prev_close
        else:
            filled = window['high'].max() >= prev_close
    else:
        filled = np.nan

    # Full day metrics
    rth_close = row.get('rth_close', np.nan)
    rth_high_val = row.get('rth_high', np.nan)
    rth_low_val = row.get('rth_low', np.nan)
    cl = (rth_close - rth_low_val) / (rth_high_val - rth_low_val) if pd.notna(rth_high_val) and pd.notna(rth_low_val) and (rth_high_val - rth_low_val) > 0 else np.nan
    drift_day = (rth_close - open_price) / adr if pd.notna(rth_close) else np.nan

    # === CLASSIFY ===
    spy_ret = row.get('spy_return_day', np.nan)
    gap_size = row.get('gap_size_in_adr', np.nan)
    rvol = row.get('rvol_open_30min', np.nan)
    if pd.isna(rvol):
        rvol = row.get('rvol_at_time_30min', np.nan)
    today_open = row.get('today_open', np.nan)
    adr_pct = adr / today_open * 100 if pd.notna(today_open) and today_open > 0 else np.nan

    # SPY bucket
    if pd.notna(spy_ret):
        if spy_ret < -0.01:
            spy_bucket = 'stark_rot (<-1%)'
        elif spy_ret < 0:
            spy_bucket = 'leicht_rot (-1% to 0)'
        elif spy_ret < 0.01:
            spy_bucket = 'leicht_gruen (0 to +1%)'
        else:
            spy_bucket = 'stark_gruen (>+1%)'
    else:
        spy_bucket = 'unknown'

    # Gap size bucket (finer)
    if pd.notna(gap_size):
        if gap_size < 1:
            gap_bucket = '<1 ADR'
        elif gap_size < 2:
            gap_bucket = '1-2 ADR'
        elif gap_size < 3:
            gap_bucket = '2-3 ADR'
        else:
            gap_bucket = '>3 ADR'
    else:
        gap_bucket = 'unknown'

    # RVOL bucket (finer)
    if pd.notna(rvol):
        if rvol < 1:
            rvol_bucket = '<1x'
        elif rvol < 2:
            rvol_bucket = '1-2x'
        elif rvol < 3:
            rvol_bucket = '2-3x'
        elif rvol < 5:
            rvol_bucket = '3-5x'
        else:
            rvol_bucket = '>5x'
    else:
        rvol_bucket = 'unknown'

    # ADR% bucket
    if pd.notna(adr_pct):
        if adr_pct < 5:
            adr_pct_bucket = '<5%'
        elif adr_pct < 10:
            adr_pct_bucket = '5-10%'
        else:
            adr_pct_bucket = '>10%'
    else:
        adr_pct_bucket = 'unknown'

    results.append({
        'ticker': ticker,
        'date': date_str,
        'gap_dir': gap_dir,
        'spy_bucket': spy_bucket,
        'gap_bucket': gap_bucket,
        'rvol_bucket': rvol_bucket,
        'adr_pct_bucket': adr_pct_bucket,
        'spy_ret': spy_ret,
        'mfe': mfe,
        'mae': mae,
        'drift_1100': drift_1100,
        'drift_day': drift_day,
        'filled': filled,
        'cl': cl,
    })

df = pd.DataFrame(results)
df.to_parquet('results/cheat_v2_spy_filtered_raw.parquet', index=False)
print(f"\nProcessed {len(df)} gapper days", file=sys.stderr)

# === ANALYSIS ===
def bootstrap_ci(data, func=np.mean, n_boot=1000, ci=0.95):
    if len(data) < 3:
        return np.nan, np.nan
    boots = [func(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_boot)]
    lo = np.percentile(boots, (1 - ci) / 2 * 100)
    hi = np.percentile(boots, (1 + ci) / 2 * 100)
    return lo, hi

def write_group(sub, label, f):
    n = len(sub)
    if n == 0:
        return
    low = " [LOW N]" if n < 50 else ""
    drift = sub['drift_1100'].mean() if sub['drift_1100'].notna().any() else np.nan
    drift_ci = bootstrap_ci(sub['drift_1100'].dropna().values) if n >= 3 else (np.nan, np.nan)
    fill = sub['filled'].mean() * 100 if sub['filled'].notna().any() else np.nan
    cl = sub['cl'].mean() if sub['cl'].notna().any() else np.nan
    mfe_m = sub['mfe'].mean()
    mae_m = sub['mae'].mean()
    drift_str = f"{drift:+.3f}" if pd.notna(drift) else "---"
    fill_str = f"{fill:.0f}%" if pd.notna(fill) else "---"
    cl_str = f"{cl:.2f}" if pd.notna(cl) else "---"
    ci_str = f"({drift_ci[0]:+.3f} to {drift_ci[1]:+.3f})" if pd.notna(drift_ci[0]) else ""
    f.write(f"    {label}{low}: N={n:>4}, Drift={drift_str:>7} {ci_str}, Fill={fill_str:>4}, CL={cl_str}, MFE={mfe_m:.3f}, MAE={mae_m:.3f}\n")


with open('results/cheat_sheet_v2_spy_filtered.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("CHEAT SHEET v2 — AUFGABE 1: SPY-KONTEXT MIT PARAMETER-FILTERN\n")
    f.write("=" * 80 + "\n")
    f.write(f"Datenbasis: {len(df)} Gapper-Tage (Halfte 1)\n")
    f.write(f"Drift/MFE/MAE gemessen bis 11:00 ET (Trading-Fenster)\n\n")

    spy_buckets = ['stark_rot (<-1%)', 'leicht_rot (-1% to 0)', 'leicht_gruen (0 to +1%)', 'stark_gruen (>+1%)']
    gap_buckets_fine = ['<1 ADR', '1-2 ADR', '2-3 ADR', '>3 ADR']
    rvol_buckets_fine = ['<1x', '1-2x', '2-3x', '3-5x', '>5x']
    adr_pct_buckets = ['<5%', '5-10%', '>10%']

    for gd in ['up', 'down']:
        f.write(f"\n{'='*80}\n")
        f.write(f"GAP {gd.upper()}\n")
        f.write(f"{'='*80}\n")

        gd_df = df[df['gap_dir'] == gd]

        # SPY baseline
        f.write(f"\n  --- SPY-Baseline (nur Gap-Richtung) ---\n")
        for sb in spy_buckets:
            sub = gd_df[gd_df['spy_bucket'] == sb]
            write_group(sub, sb, f)

        # a) SPY x Gap Size
        f.write(f"\n  --- SPY x Gap-Groesse ---\n")
        for sb in spy_buckets:
            f.write(f"\n  SPY: {sb}\n")
            for gb in gap_buckets_fine:
                sub = gd_df[(gd_df['spy_bucket'] == sb) & (gd_df['gap_bucket'] == gb)]
                write_group(sub, gb, f)

        # b) SPY x RVOL
        f.write(f"\n  --- SPY x RVOL ---\n")
        for sb in spy_buckets:
            f.write(f"\n  SPY: {sb}\n")
            for rb in rvol_buckets_fine:
                sub = gd_df[(gd_df['spy_bucket'] == sb) & (gd_df['rvol_bucket'] == rb)]
                write_group(sub, rb, f)

        # c) SPY x ADR%
        f.write(f"\n  --- SPY x ADR% ---\n")
        for sb in spy_buckets:
            f.write(f"\n  SPY: {sb}\n")
            for ab in adr_pct_buckets:
                sub = gd_df[(gd_df['spy_bucket'] == sb) & (gd_df['adr_pct_bucket'] == ab)]
                write_group(sub, ab, f)

    # === KEY TAKEAWAYS ===
    f.write("\n\n" + "=" * 80 + "\n")
    f.write("KERN-ERKENNTNISSE\n")
    f.write("=" * 80 + "\n")

    # Find strongest SPY x parameter combinations
    all_combos = []
    for gd in ['up', 'down']:
        gd_df = df[df['gap_dir'] == gd]
        for sb in spy_buckets:
            for gb in gap_buckets_fine:
                sub = gd_df[(gd_df['spy_bucket'] == sb) & (gd_df['gap_bucket'] == gb)]
                if len(sub) >= 20:
                    d = sub['drift_1100'].mean()
                    all_combos.append((gd, 'GapSize', sb, gb, len(sub), d))

            for rb in rvol_buckets_fine:
                sub = gd_df[(gd_df['spy_bucket'] == sb) & (gd_df['rvol_bucket'] == rb)]
                if len(sub) >= 20:
                    d = sub['drift_1100'].mean()
                    all_combos.append((gd, 'RVOL', sb, rb, len(sub), d))

    if all_combos:
        all_combos.sort(key=lambda x: x[5])
        f.write("\n  Top 10 STAERKSTE Drift-Kombis (SPY x Parameter):\n")
        for gd, ptype, sb, bucket, n, d in all_combos[-10:][::-1]:
            f.write(f"    Gap{gd.title()} + {sb} + {ptype}={bucket}: N={n}, Drift={d:+.3f}\n")

        f.write("\n  Top 10 SCHWÄCHSTE Drift-Kombis:\n")
        for gd, ptype, sb, bucket, n, d in all_combos[:10]:
            f.write(f"    Gap{gd.title()} + {sb} + {ptype}={bucket}: N={n}, Drift={d:+.3f}\n")

    # SPY effect strength by parameter
    f.write("\n\n  SPY-EFFEKT-STAERKE nach Parametern:\n")
    f.write("  (Spread = stark_gruen Drift - stark_rot Drift)\n\n")

    for gd in ['up', 'down']:
        gd_df = df[df['gap_dir'] == gd]
        f.write(f"  Gap{gd.title()}:\n")

        for gb in gap_buckets_fine:
            gruen = gd_df[(gd_df['spy_bucket'] == 'stark_gruen (>+1%)') & (gd_df['gap_bucket'] == gb)]
            rot = gd_df[(gd_df['spy_bucket'] == 'stark_rot (<-1%)') & (gd_df['gap_bucket'] == gb)]
            if len(gruen) >= 10 and len(rot) >= 10:
                spread = gruen['drift_1100'].mean() - rot['drift_1100'].mean()
                f.write(f"    GapSize {gb}: Spread={spread:+.3f} (gruen={gruen['drift_1100'].mean():+.3f}, rot={rot['drift_1100'].mean():+.3f})\n")

        for rb in rvol_buckets_fine:
            gruen = gd_df[(gd_df['spy_bucket'] == 'stark_gruen (>+1%)') & (gd_df['rvol_bucket'] == rb)]
            rot = gd_df[(gd_df['spy_bucket'] == 'stark_rot (<-1%)') & (gd_df['rvol_bucket'] == rb)]
            if len(gruen) >= 10 and len(rot) >= 10:
                spread = gruen['drift_1100'].mean() - rot['drift_1100'].mean()
                f.write(f"    RVOL {rb}: Spread={spread:+.3f} (gruen={gruen['drift_1100'].mean():+.3f}, rot={rot['drift_1100'].mean():+.3f})\n")

print("Done! Results in results/cheat_sheet_v2_spy_filtered.txt", file=sys.stderr)
