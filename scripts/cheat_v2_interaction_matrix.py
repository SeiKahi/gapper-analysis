"""
Durchlauf 4.5 — Aufgabe 2: Gap-ADR x RVOL x Prior Performance Interaction Matrix

Dimensions:
  Gap Size: <1 ADR, 1-2 ADR, >2 ADR
  RVOL (rvol_30): <2x, 2-5x, >5x
  Prior 10d Return: <-10%, -10% to +10%, >+10%
  Gap Direction: GapUp, GapDown

= 3 x 3 x 3 x 2 = 54 cells

Metrics: N, Fill-Rate, Drift (ADR), Close-Location, MFE60 (ADR), MAE60 (ADR)
Also test Prior 30d Return for interesting subgroups.

Trading window: 9:30-11:00 for MFE/MAE calculations.
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

# === COMPUTE MFE60 AND MAE60 FROM RAW DATA ===
# MFE60 = max favorable excursion in first 60min (9:30-10:30) in gap direction
# MAE60 = max adverse excursion in first 60min against gap direction

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

    # First 60 min (bars 0-59) for MFE60/MAE60
    first60 = rth[rth['min_idx'] < 60]
    # First 90 min for MFE/MAE within trading window
    first90 = rth[rth['min_idx'] < 91]

    if len(first60) < 5:
        continue

    # MFE/MAE relative to open, in gap direction
    if gap_dir == 'up':
        # Favorable = price goes up (with gap)
        mfe60 = (first60['high'].max() - open_price) / adr
        mae60 = (open_price - first60['low'].min()) / adr
        mfe90 = (first90['high'].max() - open_price) / adr if len(first90) > 0 else np.nan
        mae90 = (open_price - first90['low'].min()) / adr if len(first90) > 0 else np.nan
    else:
        # Favorable = price goes down (with gap)
        mfe60 = (open_price - first60['low'].min()) / adr
        mae60 = (first60['high'].max() - open_price) / adr
        mfe90 = (open_price - first90['low'].min()) / adr if len(first90) > 0 else np.nan
        mae90 = (first90['high'].max() - open_price) / adr if len(first90) > 0 else np.nan

    # Close at 11:00 or last available bar before 11:00
    bars_1100 = rth[rth['min_idx'] <= 90]
    if len(bars_1100) > 0:
        close_1100 = bars_1100.iloc[-1]['close']
        drift_1100 = (close_1100 - open_price) / adr
    else:
        drift_1100 = np.nan

    # Fill check
    if pd.notna(prev_close):
        if gap_dir == 'up':
            filled = first90['low'].min() <= prev_close if len(first90) > 0 else False
        else:
            filled = first90['high'].max() >= prev_close if len(first90) > 0 else False
    else:
        filled = np.nan

    # Day-end metrics from metadata
    rth_close = row.get('rth_close', np.nan)
    rth_high_val = row.get('rth_high', np.nan)
    rth_low_val = row.get('rth_low', np.nan)
    cl = (rth_close - rth_low_val) / (rth_high_val - rth_low_val) if pd.notna(rth_high_val) and pd.notna(rth_low_val) and (rth_high_val - rth_low_val) > 0 else np.nan
    drift_day = (rth_close - open_price) / adr if pd.notna(rth_close) else np.nan

    # === CLASSIFY ===
    gap_size = row['gap_size_in_adr'] if pd.notna(row.get('gap_size_in_adr')) else np.nan
    rvol = row.get('rvol_open_30min', np.nan)
    if pd.isna(rvol):
        rvol = row.get('rvol_at_time_30min', np.nan)
    prior_10d = row.get('prior_return_10d', np.nan)
    prior_30d = row.get('prior_return_30d', np.nan)

    # Buckets
    if pd.notna(gap_size):
        if gap_size < 1:
            gap_bucket = '<1 ADR'
        elif gap_size < 2:
            gap_bucket = '1-2 ADR'
        else:
            gap_bucket = '>2 ADR'
    else:
        gap_bucket = 'unknown'

    if pd.notna(rvol):
        if rvol < 2:
            rvol_bucket = '<2x'
        elif rvol < 5:
            rvol_bucket = '2-5x'
        else:
            rvol_bucket = '>5x'
    else:
        rvol_bucket = 'unknown'

    if pd.notna(prior_10d):
        if prior_10d < -0.10:
            prior_bucket = '<-10%'
        elif prior_10d < 0.10:
            prior_bucket = '-10% to +10%'
        else:
            prior_bucket = '>+10%'
    else:
        prior_bucket = 'unknown'

    if pd.notna(prior_30d):
        if prior_30d < -0.20:
            prior30_bucket = '<-20%'
        elif prior_30d < 0.20:
            prior30_bucket = '-20% to +20%'
        else:
            prior30_bucket = '>+20%'
    else:
        prior30_bucket = 'unknown'

    results.append({
        'ticker': ticker,
        'date': date_str,
        'gap_dir': gap_dir,
        'gap_bucket': gap_bucket,
        'rvol_bucket': rvol_bucket,
        'prior_bucket': prior_bucket,
        'prior30_bucket': prior30_bucket,
        'gap_size': gap_size,
        'rvol': rvol,
        'prior_10d': prior_10d,
        'prior_30d': prior_30d,
        'mfe60': mfe60,
        'mae60': mae60,
        'mfe90': mfe90,
        'mae90': mae90,
        'drift_1100': drift_1100,
        'drift_day': drift_day,
        'filled': filled,
        'cl': cl,
    })

df = pd.DataFrame(results)
df.to_parquet('results/cheat_v2_interaction_raw.parquet', index=False)
print(f"\nProcessed {len(df)} gapper days with MFE/MAE data", file=sys.stderr)

# === ANALYSIS ===
def bootstrap_ci(data, func=np.mean, n_boot=1000, ci=0.95):
    if len(data) < 3:
        return np.nan, np.nan
    boots = [func(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_boot)]
    lo = np.percentile(boots, (1 - ci) / 2 * 100)
    hi = np.percentile(boots, (1 + ci) / 2 * 100)
    return lo, hi

with open('results/cheat_sheet_v2_interaction_matrix.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("CHEAT SHEET v2 — AUFGABE 2: INTERAKTIONS-MATRIX\n")
    f.write("Gap-Groesse x RVOL x Prior 10d Return x Gap-Richtung\n")
    f.write("=" * 80 + "\n")
    f.write(f"Datenbasis: {len(df)} Gapper-Tage (Halfte 1)\n")
    f.write(f"MFE/MAE gemessen bis 60min (MFE60/MAE60) und bis 11:00 (MFE90/MAE90)\n")
    f.write(f"Fill-Rate gemessen bis 11:00\n\n")

    # === MAIN MATRIX: 54 cells ===
    f.write("=" * 80 + "\n")
    f.write("HAUPT-MATRIX (54 Zellen)\n")
    f.write("=" * 80 + "\n\n")

    gap_buckets = ['<1 ADR', '1-2 ADR', '>2 ADR']
    rvol_buckets = ['<2x', '2-5x', '>5x']
    prior_buckets = ['<-10%', '-10% to +10%', '>+10%']

    # Track interesting cells for 30d analysis
    interesting_cells = []

    for gd in ['up', 'down']:
        f.write(f"\n{'='*80}\n")
        f.write(f"GAP {gd.upper()}\n")
        f.write(f"{'='*80}\n")

        for gb in gap_buckets:
            f.write(f"\n  --- Gap Size: {gb} ---\n")
            f.write(f"  {'RVOL':<10} {'Prior10d':<18} {'N':>5} {'Fill%':>6} {'Drift':>8} {'CL':>6} {'MFE60':>7} {'MAE60':>7} {'Note'}\n")
            f.write(f"  {'-'*75}\n")

            for rb in rvol_buckets:
                for pb in prior_buckets:
                    sub = df[(df['gap_dir'] == gd) &
                             (df['gap_bucket'] == gb) &
                             (df['rvol_bucket'] == rb) &
                             (df['prior_bucket'] == pb)]
                    n = len(sub)
                    if n == 0:
                        f.write(f"  {rb:<10} {pb:<18} {0:>5} {'---':>6} {'---':>8} {'---':>6} {'---':>7} {'---':>7}\n")
                        continue

                    fill = sub['filled'].mean() * 100 if sub['filled'].notna().any() else np.nan
                    drift = sub['drift_1100'].mean() if sub['drift_1100'].notna().any() else np.nan
                    cl = sub['cl'].mean() if sub['cl'].notna().any() else np.nan
                    mfe60 = sub['mfe60'].mean()
                    mae60 = sub['mae60'].mean()

                    note = ""
                    if n < 50:
                        note = "[LOW N]"

                    # Flag interesting cells
                    if n >= 30 and pd.notna(drift) and abs(drift) > 0.2:
                        note += " ***STRONG***"
                        interesting_cells.append((gd, gb, rb, pb, n, drift))

                    fill_str = f"{fill:.0f}%" if pd.notna(fill) else "---"
                    drift_str = f"{drift:+.3f}" if pd.notna(drift) else "---"
                    cl_str = f"{cl:.2f}" if pd.notna(cl) else "---"

                    f.write(f"  {rb:<10} {pb:<18} {n:>5} {fill_str:>6} {drift_str:>8} {cl_str:>6} {mfe60:>7.3f} {mae60:>7.3f} {note}\n")

    # === TRADER'S HYPOTHESIS ===
    f.write("\n\n" + "=" * 80 + "\n")
    f.write("TRADER-HYPOTHESEN-CHECK\n")
    f.write("=" * 80 + "\n")

    # Hypothesis 1: Small gap + high RVOL = new positioning -> Momentum
    f.write("\n  H1: Kleiner Gap + hohes RVOL = neue Positionierung -> Momentum?\n")
    for gd in ['up', 'down']:
        sub = df[(df['gap_dir'] == gd) & (df['gap_bucket'] == '<1 ADR') & (df['rvol_bucket'].isin(['>5x', '2-5x']))]
        if len(sub) >= 10:
            drift = sub['drift_1100'].mean()
            fill = sub['filled'].mean() * 100 if sub['filled'].notna().any() else np.nan
            f.write(f"    Gap{gd.title()} <1ADR + RVOL>2x: N={len(sub)}, Drift={drift:+.3f}, Fill={fill:.0f}%\n")

    # Hypothesis 2: Large gap + high RVOL after long rally = profit taking -> Fade
    f.write("\n  H2: Grosser Gap + hohes RVOL nach Rally = Gewinnmitnahmen -> Fade?\n")
    sub = df[(df['gap_dir'] == 'up') & (df['gap_bucket'] == '>2 ADR') &
             (df['rvol_bucket'].isin(['>5x', '2-5x'])) & (df['prior_bucket'] == '>+10%')]
    if len(sub) >= 10:
        drift = sub['drift_1100'].mean()
        fill = sub['filled'].mean() * 100 if sub['filled'].notna().any() else np.nan
        f.write(f"    GapUp >2ADR + RVOL>2x + Prior>+10%: N={len(sub)}, Drift={drift:+.3f}, Fill={fill:.0f}%\n")
    else:
        f.write(f"    GapUp >2ADR + RVOL>2x + Prior>+10%: N={len(sub)} (zu wenig)\n")

    # Hypothesis 3: Large gap + high RVOL after long downtrend = capitulation -> Bounce
    f.write("\n  H3: Grosser Gap + hohes RVOL nach Downtrend = Kapitulation -> Bounce?\n")
    sub = df[(df['gap_dir'] == 'down') & (df['gap_bucket'] == '>2 ADR') &
             (df['rvol_bucket'].isin(['>5x', '2-5x'])) & (df['prior_bucket'] == '<-10%')]
    if len(sub) >= 10:
        drift = sub['drift_1100'].mean()
        fill = sub['filled'].mean() * 100 if sub['filled'].notna().any() else np.nan
        f.write(f"    GapDn >2ADR + RVOL>2x + Prior<-10%: N={len(sub)}, Drift={drift:+.3f}, Fill={fill:.0f}%\n")
    else:
        f.write(f"    GapDn >2ADR + RVOL>2x + Prior<-10%: N={len(sub)} (zu wenig)\n")

    # === INTERESTING CELLS: Test Prior 30d ===
    f.write("\n\n" + "=" * 80 + "\n")
    f.write("PRIOR 30d RETURN FÜR INTERESSANTE SUBGRUPPEN\n")
    f.write("=" * 80 + "\n")

    if len(interesting_cells) > 0:
        for gd, gb, rb, pb, n, drift in interesting_cells:
            f.write(f"\n  Original: Gap{gd.title()} + {gb} + RVOL {rb} + Prior10d {pb}: N={n}, Drift={drift:+.3f}\n")
            base_sub = df[(df['gap_dir'] == gd) & (df['gap_bucket'] == gb) & (df['rvol_bucket'] == rb)]
            for p30b in ['<-20%', '-20% to +20%', '>+20%']:
                sub30 = base_sub[base_sub['prior30_bucket'] == p30b]
                if len(sub30) >= 10:
                    d30 = sub30['drift_1100'].mean()
                    f.write(f"    Prior30d {p30b}: N={len(sub30)}, Drift={d30:+.3f}\n")
    else:
        f.write("  Keine Zellen mit |Drift| > 0.2 und N >= 30 gefunden.\n")

    # === MARGINAL EFFECTS ===
    f.write("\n\n" + "=" * 80 + "\n")
    f.write("MARGINALE EFFEKTE (ueber alle anderen Dimensionen)\n")
    f.write("=" * 80 + "\n")

    for gd in ['up', 'down']:
        f.write(f"\n  Gap{gd.title()}:\n")
        gd_df = df[df['gap_dir'] == gd]

        f.write(f"    Gap-Groesse:\n")
        for gb in gap_buckets:
            sub = gd_df[gd_df['gap_bucket'] == gb]
            if len(sub) >= 10:
                d = sub['drift_1100'].mean()
                ci = bootstrap_ci(sub['drift_1100'].dropna().values)
                f.write(f"      {gb}: N={len(sub)}, Drift={d:+.3f} (CI: {ci[0]:+.3f} to {ci[1]:+.3f})\n")

        f.write(f"    RVOL:\n")
        for rb in rvol_buckets:
            sub = gd_df[gd_df['rvol_bucket'] == rb]
            if len(sub) >= 10:
                d = sub['drift_1100'].mean()
                ci = bootstrap_ci(sub['drift_1100'].dropna().values)
                f.write(f"      {rb}: N={len(sub)}, Drift={d:+.3f} (CI: {ci[0]:+.3f} to {ci[1]:+.3f})\n")

        f.write(f"    Prior 10d:\n")
        for pb in prior_buckets:
            sub = gd_df[gd_df['prior_bucket'] == pb]
            if len(sub) >= 10:
                d = sub['drift_1100'].mean()
                ci = bootstrap_ci(sub['drift_1100'].dropna().values)
                f.write(f"      {pb}: N={len(sub)}, Drift={d:+.3f} (CI: {ci[0]:+.3f} to {ci[1]:+.3f})\n")

    # === KEY TAKEAWAYS ===
    f.write("\n\n" + "=" * 80 + "\n")
    f.write("KERN-ERKENNTNISSE\n")
    f.write("=" * 80 + "\n")

    # Find strongest cells
    all_cells = []
    for gd in ['up', 'down']:
        for gb in gap_buckets:
            for rb in rvol_buckets:
                for pb in prior_buckets:
                    sub = df[(df['gap_dir'] == gd) & (df['gap_bucket'] == gb) &
                             (df['rvol_bucket'] == rb) & (df['prior_bucket'] == pb)]
                    if len(sub) >= 20:
                        d = sub['drift_1100'].mean()
                        all_cells.append((gd, gb, rb, pb, len(sub), d))

    if all_cells:
        all_cells.sort(key=lambda x: x[5])
        f.write("\n  Top 5 STAERKSTE Drift-Zellen (Drift bis 11:00):\n")
        for gd, gb, rb, pb, n, d in all_cells[-5:][::-1]:
            f.write(f"    Gap{gd.title()} + {gb} + RVOL {rb} + Prior {pb}: N={n}, Drift={d:+.3f}\n")

        f.write("\n  Top 5 SCHWÄCHSTE Drift-Zellen:\n")
        for gd, gb, rb, pb, n, d in all_cells[:5]:
            f.write(f"    Gap{gd.title()} + {gb} + RVOL {rb} + Prior {pb}: N={n}, Drift={d:+.3f}\n")

print("Done! Results in results/cheat_sheet_v2_interaction_matrix.txt", file=sys.stderr)
