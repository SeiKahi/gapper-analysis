"""
Durchlauf 4.5 — Aufgabe 5: Premarket Volume as Signal

raw_1min has premarket data (session='premarket', from ~04:00 ET).
baseline_1min has NO premarket (RTH only).

Compute:
1. PM Volume (absolute sum)
2. Percentile rank among all gappers
3. PM Volume / RTH-30min-Volume (own baseline per day)
4. PM Range (High-Low in ADR) as additional filter

Buckets: Bottom 25%, 25-50%, 50-75%, Top 25% PM Volume
Metrics: Fill-Rate, Drift, Close-Location, DayRange, MFE60, MAE60
Breakdown by gap direction and gap size.
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

# === COLLECT PREMARKET + RTH DATA ===
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

    # Premarket data
    pm = raw[raw['session'] == 'premarket'].copy()
    rth = raw[raw['session'] == 'rth'].copy()

    if len(rth) < 10:
        continue

    rth = rth.sort_values('time_et').reset_index(drop=True)
    rth['min_idx'] = range(len(rth))

    # PM Volume
    pm_vol = pm['volume'].sum() if len(pm) > 0 else 0
    pm_high = pm['high'].max() if len(pm) > 0 else np.nan
    pm_low = pm['low'].min() if len(pm) > 0 else np.nan
    pm_range_adr = (pm_high - pm_low) / adr if len(pm) > 0 and pd.notna(pm_high) and adr > 0 else np.nan

    # RTH first 30 min volume
    rth_30 = rth[rth['min_idx'] < 30]
    rth_30_vol = rth_30['volume'].sum() if len(rth_30) > 0 else 0

    # PM/RTH30 ratio
    pm_rth_ratio = pm_vol / rth_30_vol if rth_30_vol > 0 else np.nan

    open_price = rth.iloc[0]['open']
    prev_close = row.get('prev_close', np.nan)

    # Trading window metrics
    window = rth[rth['min_idx'] <= 90]
    if len(window) < 5:
        continue

    if gap_dir == 'up':
        mfe60 = (window['high'].max() - open_price) / adr
        mae60 = (open_price - window['low'].min()) / adr
    else:
        mfe60 = (open_price - window['low'].min()) / adr
        mae60 = (window['high'].max() - open_price) / adr

    close_1100 = window.iloc[-1]['close']
    drift_1100 = (close_1100 - open_price) / adr

    day_range = (rth['high'].max() - rth['low'].min()) / adr

    # Fill
    if pd.notna(prev_close):
        if gap_dir == 'up':
            filled = window['low'].min() <= prev_close
        else:
            filled = window['high'].max() >= prev_close
    else:
        filled = np.nan

    rth_close = row.get('rth_close', np.nan)
    rth_high_val = row.get('rth_high', np.nan)
    rth_low_val = row.get('rth_low', np.nan)
    cl = (rth_close - rth_low_val) / (rth_high_val - rth_low_val) if pd.notna(rth_high_val) and pd.notna(rth_low_val) and (rth_high_val - rth_low_val) > 0 else np.nan
    drift_day = (rth_close - open_price) / adr if pd.notna(rth_close) else np.nan

    gap_size = row.get('gap_size_in_adr', np.nan)
    if pd.notna(gap_size):
        if gap_size < 1:
            gap_bucket = '<1 ADR'
        elif gap_size < 2:
            gap_bucket = '1-2 ADR'
        else:
            gap_bucket = '>2 ADR'
    else:
        gap_bucket = 'unknown'

    results.append({
        'ticker': ticker,
        'date': date_str,
        'gap_dir': gap_dir,
        'gap_bucket': gap_bucket,
        'pm_vol': pm_vol,
        'pm_range_adr': pm_range_adr,
        'pm_rth_ratio': pm_rth_ratio,
        'pm_bars': len(pm),
        'mfe60': mfe60,
        'mae60': mae60,
        'drift_1100': drift_1100,
        'drift_day': drift_day,
        'day_range': day_range,
        'filled': filled,
        'cl': cl,
    })

df = pd.DataFrame(results)

# === COMPUTE PERCENTILES ===
df['pm_vol_pct'] = df['pm_vol'].rank(pct=True)
df['pm_range_pct'] = df['pm_range_adr'].rank(pct=True)

# PM Volume quartile buckets
df['pm_vol_bucket'] = pd.cut(df['pm_vol_pct'], bins=[0, 0.25, 0.50, 0.75, 1.0],
                              labels=['Bottom 25%', '25-50%', '50-75%', 'Top 25%'],
                              include_lowest=True)

# PM Range buckets
df['pm_range_bucket'] = 'unknown'
df.loc[df['pm_range_adr'].notna() & (df['pm_range_adr'] < 0.3), 'pm_range_bucket'] = 'eng (<0.3 ADR)'
df.loc[df['pm_range_adr'].notna() & (df['pm_range_adr'] >= 0.3) & (df['pm_range_adr'] < 0.7), 'pm_range_bucket'] = 'mittel (0.3-0.7 ADR)'
df.loc[df['pm_range_adr'].notna() & (df['pm_range_adr'] >= 0.7), 'pm_range_bucket'] = 'weit (>0.7 ADR)'

# PM/RTH ratio buckets
df['pm_rth_bucket'] = 'unknown'
df.loc[df['pm_rth_ratio'].notna() & (df['pm_rth_ratio'] < 0.1), 'pm_rth_bucket'] = '<10%'
df.loc[df['pm_rth_ratio'].notna() & (df['pm_rth_ratio'] >= 0.1) & (df['pm_rth_ratio'] < 0.3), 'pm_rth_bucket'] = '10-30%'
df.loc[df['pm_rth_ratio'].notna() & (df['pm_rth_ratio'] >= 0.3), 'pm_rth_bucket'] = '>30%'

df.to_parquet('results/cheat_v2_premarket_raw.parquet', index=False)
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
    if n < 10:
        f.write(f"    {label}: N={n} (TOO FEW)\n")
        return
    low = " [LOW N]" if n < 50 else ""
    drift = sub['drift_1100'].mean()
    drift_ci = bootstrap_ci(sub['drift_1100'].dropna().values)
    fill = sub['filled'].mean() * 100 if sub['filled'].notna().any() else np.nan
    cl = sub['cl'].mean() if sub['cl'].notna().any() else np.nan
    mfe = sub['mfe60'].mean()
    mae = sub['mae60'].mean()
    dr = sub['day_range'].mean()
    ci_str = f"(CI: {drift_ci[0]:+.3f} to {drift_ci[1]:+.3f})" if pd.notna(drift_ci[0]) else ""
    fill_str = f"{fill:.0f}%" if pd.notna(fill) else "---"
    cl_str = f"{cl:.2f}" if pd.notna(cl) else "---"
    f.write(f"    {label}{low}: N={n:>4}, Drift={drift:+.3f} {ci_str}, Fill={fill_str}, CL={cl_str}, MFE={mfe:.3f}, MAE={mae:.3f}, DayRange={dr:.2f}\n")


with open('results/cheat_sheet_v2_premarket.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("CHEAT SHEET v2 — AUFGABE 5: PREMARKET-VOLUMEN ALS SIGNAL\n")
    f.write("=" * 80 + "\n")
    f.write(f"Datenbasis: {len(df)} Gapper-Tage (Halfte 1)\n")
    f.write(f"PM Volume als Percentile unter allen Gappern\n")
    f.write(f"PM/RTH30 Ratio = Premarket Vol / erste 30min RTH Vol\n\n")

    # Stats overview
    f.write("-" * 80 + "\n")
    f.write("PM-VOLUMEN UEBERSICHT\n")
    f.write("-" * 80 + "\n")
    f.write(f"  PM Volume: Mean={df['pm_vol'].mean():.0f}, Median={df['pm_vol'].median():.0f}\n")
    f.write(f"  PM Bars: Mean={df['pm_bars'].mean():.0f}, Median={df['pm_bars'].median():.0f}\n")
    has_pm = df[df['pm_vol'] > 0]
    f.write(f"  Tage mit PM-Daten: {len(has_pm)} ({len(has_pm)/len(df)*100:.1f}%)\n")
    f.write(f"  PM Range (ADR): Mean={df['pm_range_adr'].mean():.3f}, Median={df['pm_range_adr'].median():.3f}\n")
    f.write(f"  PM/RTH30 Ratio: Mean={df['pm_rth_ratio'].mean():.3f}, Median={df['pm_rth_ratio'].median():.3f}\n")

    # === BY PM VOLUME QUARTILE ===
    f.write("\n" + "-" * 80 + "\n")
    f.write("NACH PM-VOLUMEN QUARTILE\n")
    f.write("-" * 80 + "\n")
    for gd in ['up', 'down']:
        f.write(f"\n  Gap{gd.title()}:\n")
        gd_df = df[df['gap_dir'] == gd]
        for bucket in ['Bottom 25%', '25-50%', '50-75%', 'Top 25%']:
            sub = gd_df[gd_df['pm_vol_bucket'] == bucket]
            write_group(sub, bucket, f)

    # === BY PM/RTH RATIO ===
    f.write("\n" + "-" * 80 + "\n")
    f.write("NACH PM/RTH30 RATIO\n")
    f.write("-" * 80 + "\n")
    for gd in ['up', 'down']:
        f.write(f"\n  Gap{gd.title()}:\n")
        gd_df = df[df['gap_dir'] == gd]
        for bucket in ['<10%', '10-30%', '>30%']:
            sub = gd_df[gd_df['pm_rth_bucket'] == bucket]
            write_group(sub, f"PM/RTH {bucket}", f)

    # === BY PM RANGE ===
    f.write("\n" + "-" * 80 + "\n")
    f.write("NACH PM-RANGE (Bonus)\n")
    f.write("-" * 80 + "\n")
    for gd in ['up', 'down']:
        f.write(f"\n  Gap{gd.title()}:\n")
        gd_df = df[df['gap_dir'] == gd]
        for bucket in ['eng (<0.3 ADR)', 'mittel (0.3-0.7 ADR)', 'weit (>0.7 ADR)']:
            sub = gd_df[gd_df['pm_range_bucket'] == bucket]
            write_group(sub, bucket, f)

    # === PM VOLUME x GAP SIZE ===
    f.write("\n" + "-" * 80 + "\n")
    f.write("PM-VOLUMEN x GAP-GROESSE\n")
    f.write("-" * 80 + "\n")
    for gd in ['up', 'down']:
        f.write(f"\n  Gap{gd.title()}:\n")
        gd_df = df[df['gap_dir'] == gd]
        for gb in ['<1 ADR', '1-2 ADR', '>2 ADR']:
            f.write(f"    --- {gb} ---\n")
            for vb in ['Bottom 25%', '25-50%', '50-75%', 'Top 25%']:
                sub = gd_df[(gd_df['gap_bucket'] == gb) & (gd_df['pm_vol_bucket'] == vb)]
                write_group(sub, vb, f)

    # === KERN-ERKENNTNISSE ===
    f.write("\n\n" + "=" * 80 + "\n")
    f.write("KERN-ERKENNTNISSE\n")
    f.write("=" * 80 + "\n")

    for gd in ['up', 'down']:
        gd_df = df[df['gap_dir'] == gd]
        top = gd_df[gd_df['pm_vol_bucket'] == 'Top 25%']
        bottom = gd_df[gd_df['pm_vol_bucket'] == 'Bottom 25%']
        if len(top) >= 20 and len(bottom) >= 20:
            d_top = top['drift_1100'].mean()
            d_bot = bottom['drift_1100'].mean()
            f.write(f"\n  Gap{gd.title()}: Top25% PM Drift={d_top:+.3f} vs Bottom25% Drift={d_bot:+.3f} (Spread={d_top-d_bot:+.3f})\n")

print("Done! Results in results/cheat_sheet_v2_premarket.txt", file=sys.stderr)
