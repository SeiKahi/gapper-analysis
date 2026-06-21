"""
Durchlauf 4.5 — Aufgabe 4: VWAP Rejection at Later Timepoints
After 9:50 when bands are wider and levels have more meaning.

Definition:
  DownGap Rejection: Price rises to VWAP, touches it (High >= VWAP), but Close < VWAP
  UpGap Rejection: Price falls to VWAP, touches it (Low <= VWAP), but Close > VWAP

Only rejections after 9:50 (bands more stable).
Only FIRST rejection per day.

Measures (all in ADR, relative to bands AT TIME of rejection):
  a) Does price reach next sigma level in gap direction before rejection price broken?
  b) Median time to sigma levels
  c) MAE (how far against before sigma reached)
  d) How often is rejection price broken before sigma reached?

Breakdowns: gap direction, time of rejection, gap size, RVOL, bandwidth, prior VWAP tests
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

# === PROCESS EACH GAPPER DAY ===
results = []

for _, row in tqdm(h1.iterrows(), total=len(h1), desc="Processing", file=sys.stderr):
    ticker = row['ticker']
    date_str = row['date'].strftime('%Y-%m-%d')
    adr = row['adr_10']
    gap_dir = row['gap_direction']
    if pd.isna(adr) or adr <= 0:
        continue

    # Load raw_1min for OHLC
    raw_path = Path(f'data/raw_1min/{ticker}/{date_str}.parquet')
    if not raw_path.exists():
        continue

    # Load VWAP file for VWAP + StdDev bands
    vwap_path = Path(f'data/vwap/{ticker}/{date_str}.parquet')
    if not vwap_path.exists():
        continue

    try:
        raw = pd.read_parquet(raw_path)
        vwap_df = pd.read_parquet(vwap_path)
    except Exception:
        continue

    # Filter RTH
    rth = raw[raw['session'] == 'rth'].copy()
    if len(rth) < 20:
        continue

    # Parse time for both
    rth['time_et'] = pd.to_datetime(rth['time_et'], format='%H:%M:%S', errors='coerce')
    vwap_df['time_et'] = pd.to_datetime(vwap_df['time_et'], format='%H:%M:%S', errors='coerce')

    rth = rth.sort_values('time_et').reset_index(drop=True)
    vwap_df = vwap_df.sort_values('time_et').reset_index(drop=True)

    # Merge OHLC with VWAP data on time
    rth['time_str'] = rth['time_et'].dt.strftime('%H:%M')
    vwap_df['time_str'] = vwap_df['time_et'].dt.strftime('%H:%M')

    # Rename VWAP file's vwap to running_vwap to avoid collision with raw's bar-level vwap
    vwap_cols = vwap_df[['time_str', 'vwap', 'std_dev',
                          'upper_1std', 'lower_1std',
                          'upper_2std', 'lower_2std',
                          'upper_3std', 'lower_3std']].rename(columns={'vwap': 'running_vwap'})

    merged = pd.merge(rth, vwap_cols, on='time_str', how='left')

    if merged['running_vwap'].isna().all():
        continue

    merged = merged.sort_values('time_et').reset_index(drop=True)
    merged['min_idx'] = range(len(merged))

    # Filter: only bars after 9:50 (minute 20) and before 11:00 (minute 90)
    # But we also need pre-9:50 data to count VWAP tests
    # Count VWAP tests before our window
    pre_window = merged[merged['min_idx'] < 20]
    prior_vwap_tests = 0
    vwap_broken_before = False

    for _, bar in pre_window.iterrows():
        if pd.isna(bar['running_vwap']):
            continue
        if gap_dir == 'down':
            # VWAP is resistance for DownGap
            if bar['high'] >= bar['running_vwap']:
                prior_vwap_tests += 1
            # Check if VWAP was broken (close above)
            if bar['close'] > bar['running_vwap']:
                vwap_broken_before = True
        else:
            # VWAP is support for UpGap
            if bar['low'] <= bar['running_vwap']:
                prior_vwap_tests += 1
            if bar['close'] < bar['running_vwap']:
                vwap_broken_before = True

    # Search for FIRST rejection after 9:50
    window = merged[(merged['min_idx'] >= 20) & (merged['min_idx'] <= 90)]
    rejection_found = False

    for idx, bar in window.iterrows():
        if pd.isna(bar['running_vwap']) or pd.isna(bar['std_dev']):
            continue

        vwap_val = bar['running_vwap']
        std_val = bar['std_dev']

        if gap_dir == 'down':
            # DownGap: price rises to VWAP, high >= VWAP but close < VWAP
            if bar['high'] >= vwap_val and bar['close'] < vwap_val:
                rejection_found = True
                rejection_bar_idx = bar['min_idx']
                rejection_high = bar['high']
                rejection_time = bar['time_str']

                # Sigma levels at time of rejection (in gap direction = down)
                sigma1 = bar.get('lower_1std', vwap_val - std_val)
                sigma2 = bar.get('lower_2std', vwap_val - 2 * std_val)
                sigma3 = bar.get('lower_3std', vwap_val - 3 * std_val)

                # Bandwidth at rejection
                bandwidth = 2 * std_val / adr if adr > 0 else np.nan

                break
        else:
            # UpGap: price falls to VWAP, low <= VWAP but close > VWAP
            if bar['low'] <= vwap_val and bar['close'] > vwap_val:
                rejection_found = True
                rejection_bar_idx = bar['min_idx']
                rejection_low = bar['low']
                rejection_time = bar['time_str']

                # Sigma levels at time of rejection (in gap direction = up)
                sigma1 = bar.get('upper_1std', vwap_val + std_val)
                sigma2 = bar.get('upper_2std', vwap_val + 2 * std_val)
                sigma3 = bar.get('upper_3std', vwap_val + 3 * std_val)

                bandwidth = 2 * std_val / adr if adr > 0 else np.nan

                break

    if not rejection_found:
        continue

    # Also count this as a VWAP test
    prior_vwap_tests += 1

    # === TRACK POST-REJECTION BEHAVIOR ===
    # From rejection bar+1 onwards until 11:00
    post_rej = merged[(merged['min_idx'] > rejection_bar_idx) & (merged['min_idx'] <= 90)]

    if len(post_rej) == 0:
        continue

    # Track: does price reach sigma levels before rejection price is broken?
    sigma1_reached = False
    sigma2_reached = False
    sigma3_reached = False
    sigma1_time = None
    sigma2_time = None
    sigma3_time = None

    rejection_broken = False
    rejection_broken_time = None

    mae_before_sigma1 = 0.0  # Max adverse (beyond rejection) before sigma1

    for _, pbar in post_rej.iterrows():
        bar_min = pbar['min_idx'] - rejection_bar_idx  # minutes after rejection

        if gap_dir == 'down':
            # Trade direction is DOWN (with gap)
            # Sigma levels are BELOW current price
            # Rejection price is rejection_high (above)

            # Check sigma levels (price goes down)
            if not sigma1_reached and pbar['low'] <= sigma1:
                sigma1_reached = True
                sigma1_time = bar_min

            if not sigma2_reached and pbar['low'] <= sigma2:
                sigma2_reached = True
                sigma2_time = bar_min

            if not sigma3_reached and pbar['low'] <= sigma3:
                sigma3_reached = True
                sigma3_time = bar_min

            # Check rejection broken (price goes above rejection_high)
            if not rejection_broken and pbar['high'] > rejection_high:
                rejection_broken = True
                rejection_broken_time = bar_min

            # MAE (adverse = price goes UP above rejection_high)
            adverse = (pbar['high'] - rejection_high) / adr if pbar['high'] > rejection_high else 0
            if adverse > mae_before_sigma1 and not sigma1_reached:
                mae_before_sigma1 = adverse

        else:
            # UpGap: Trade direction is UP (with gap)
            # Sigma levels are ABOVE
            # Rejection price is rejection_low (below)

            if not sigma1_reached and pbar['high'] >= sigma1:
                sigma1_reached = True
                sigma1_time = bar_min

            if not sigma2_reached and pbar['high'] >= sigma2:
                sigma2_reached = True
                sigma2_time = bar_min

            if not sigma3_reached and pbar['high'] >= sigma3:
                sigma3_reached = True
                sigma3_time = bar_min

            if not rejection_broken and pbar['low'] < rejection_low:
                rejection_broken = True
                rejection_broken_time = bar_min

            adverse = (rejection_low - pbar['low']) / adr if pbar['low'] < rejection_low else 0
            if adverse > mae_before_sigma1 and not sigma1_reached:
                mae_before_sigma1 = adverse

    # Was sigma reached BEFORE rejection broken?
    sigma1_before_broken = sigma1_reached and (not rejection_broken or
                                                (sigma1_time is not None and rejection_broken_time is not None
                                                 and sigma1_time <= rejection_broken_time))
    sigma2_before_broken = sigma2_reached and (not rejection_broken or
                                                (sigma2_time is not None and rejection_broken_time is not None
                                                 and sigma2_time <= rejection_broken_time))
    sigma3_before_broken = sigma3_reached and (not rejection_broken or
                                                (sigma3_time is not None and rejection_broken_time is not None
                                                 and sigma3_time <= rejection_broken_time))

    # Rejection broken before sigma1?
    broken_before_sigma1 = rejection_broken and (not sigma1_reached or
                                                  (rejection_broken_time is not None and sigma1_time is not None
                                                   and rejection_broken_time < sigma1_time))

    # Time bucket for rejection
    rej_min = rejection_bar_idx  # minutes since 9:30
    if rej_min < 30:
        rej_time_bucket = '9:50-10:00'
    elif rej_min < 45:
        rej_time_bucket = '10:00-10:15'
    elif rej_min < 60:
        rej_time_bucket = '10:15-10:30'
    else:
        rej_time_bucket = '10:30-11:00'

    # Gap size bucket
    gap_size = row['gap_size_in_adr'] if pd.notna(row.get('gap_size_in_adr')) else np.nan
    if pd.notna(gap_size):
        if gap_size < 1:
            gap_bucket = '<1 ADR'
        elif gap_size < 2:
            gap_bucket = '1-2 ADR'
        else:
            gap_bucket = '>2 ADR'
    else:
        gap_bucket = 'unknown'

    # RVOL bucket
    rvol = row.get('rvol_open_30min', np.nan)
    if pd.isna(rvol):
        rvol = row.get('rvol_at_time_30min', np.nan)
    if pd.notna(rvol):
        if rvol < 2:
            rvol_bucket = '<2x'
        elif rvol < 5:
            rvol_bucket = '2-5x'
        else:
            rvol_bucket = '>5x'
    else:
        rvol_bucket = 'unknown'

    # Bandwidth bucket
    if pd.notna(bandwidth):
        if bandwidth < 0.2:
            bw_bucket = 'eng (<0.2)'
        elif bandwidth < 0.4:
            bw_bucket = 'mittel (0.2-0.4)'
        else:
            bw_bucket = 'weit (>0.4)'
    else:
        bw_bucket = 'unknown'

    results.append({
        'ticker': ticker,
        'date': date_str,
        'gap_dir': gap_dir,
        'rejection_time': rejection_time,
        'rej_time_bucket': rej_time_bucket,
        'gap_bucket': gap_bucket,
        'rvol_bucket': rvol_bucket,
        'bw_bucket': bw_bucket,
        'bandwidth': bandwidth,
        'prior_vwap_tests': prior_vwap_tests,
        'vwap_broken_before': vwap_broken_before,
        'sigma1_reached': sigma1_reached,
        'sigma2_reached': sigma2_reached,
        'sigma3_reached': sigma3_reached,
        'sigma1_before_broken': sigma1_before_broken,
        'sigma2_before_broken': sigma2_before_broken,
        'sigma3_before_broken': sigma3_before_broken,
        'sigma1_time': sigma1_time,
        'sigma2_time': sigma2_time,
        'sigma3_time': sigma3_time,
        'rejection_broken': rejection_broken,
        'rejection_broken_time': rejection_broken_time,
        'broken_before_sigma1': broken_before_sigma1,
        'mae_before_sigma1': mae_before_sigma1,
    })

df = pd.DataFrame(results)
df.to_parquet('results/cheat_v2_vwap_rejection_raw.parquet', index=False)
print(f"\nProcessed {len(df)} rejection events", file=sys.stderr)

# === ANALYSIS ===
def bootstrap_ci(data, func=np.mean, n_boot=1000, ci=0.95):
    if len(data) < 3:
        return np.nan, np.nan
    boots = [func(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_boot)]
    lo = np.percentile(boots, (1 - ci) / 2 * 100)
    hi = np.percentile(boots, (1 + ci) / 2 * 100)
    return lo, hi

def analyze_rejection_group(group_df, label, f):
    n = len(group_df)
    if n < 10:
        f.write(f"\n  {label}: N={n} (TOO FEW)\n")
        return

    low_n = " [LOW N]" if n < 50 else ""

    # Sigma reach rates (regardless of rejection broken)
    s1_rate = group_df['sigma1_reached'].mean() * 100
    s2_rate = group_df['sigma2_reached'].mean() * 100
    s3_rate = group_df['sigma3_reached'].mean() * 100

    # Sigma reached BEFORE rejection broken
    s1_before = group_df['sigma1_before_broken'].mean() * 100
    s2_before = group_df['sigma2_before_broken'].mean() * 100
    s3_before = group_df['sigma3_before_broken'].mean() * 100

    s1_ci = bootstrap_ci(group_df['sigma1_before_broken'].values.astype(float))

    # Rejection broken before sigma1
    broken_before = group_df['broken_before_sigma1'].mean() * 100
    broken_ci = bootstrap_ci(group_df['broken_before_sigma1'].values.astype(float))

    # Median time to sigma1 (when reached)
    s1_times = group_df[group_df['sigma1_reached']]['sigma1_time'].dropna()
    med_s1_time = s1_times.median() if len(s1_times) > 0 else np.nan

    s2_times = group_df[group_df['sigma2_reached']]['sigma2_time'].dropna()
    med_s2_time = s2_times.median() if len(s2_times) > 0 else np.nan

    # MAE before sigma1
    mae_mean = group_df['mae_before_sigma1'].mean()
    mae_med = group_df['mae_before_sigma1'].median()
    mae_p75 = group_df['mae_before_sigma1'].quantile(0.75)

    # Bandwidth
    bw_mean = group_df['bandwidth'].mean() if 'bandwidth' in group_df.columns else np.nan

    f.write(f"\n  {label}{low_n}:\n")
    f.write(f"    N={n}, Bandwidth={bw_mean:.3f} ADR\n")
    f.write(f"    Sigma-1 erreicht: {s1_rate:.1f}% (gesamt), {s1_before:.1f}% BEVOR Rejection gebrochen (CI: {s1_ci[0]*100:.1f}-{s1_ci[1]*100:.1f}%)\n")
    f.write(f"    Sigma-2 erreicht: {s2_rate:.1f}% (gesamt), {s2_before:.1f}% BEVOR Rejection gebrochen\n")
    f.write(f"    Sigma-3 erreicht: {s3_rate:.1f}% (gesamt), {s3_before:.1f}% BEVOR Rejection gebrochen\n")
    f.write(f"    Rejection gebrochen BEVOR Sigma-1: {broken_before:.1f}% (CI: {broken_ci[0]*100:.1f}-{broken_ci[1]*100:.1f}%)\n")
    f.write(f"    Median Zeit bis Sigma-1: {med_s1_time:.0f} min | Sigma-2: {med_s2_time:.0f} min\n")
    f.write(f"    MAE (ueber Rejection hinaus) bevor Sigma-1: Mean={mae_mean:.3f}, Median={mae_med:.3f}, P75={mae_p75:.3f} ADR\n")


# === WRITE RESULTS ===
with open('results/cheat_sheet_v2_vwap_rejection.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("CHEAT SHEET v2 — AUFGABE 4: VWAP-ABLEHNUNG NACH 9:50\n")
    f.write("=" * 80 + "\n")
    f.write(f"Datenbasis: {len(df)} Ablehnungs-Events (Halfte 1)\n")
    f.write(f"Definition: Erste VWAP-Ablehnung nach 9:50 pro Tag\n")
    f.write(f"  DownGap: High >= VWAP aber Close < VWAP\n")
    f.write(f"  UpGap: Low <= VWAP aber Close > VWAP\n")
    f.write(f"Sigma-Levels fixiert zum Zeitpunkt der Ablehnung\n\n")

    # === OVERALL ===
    f.write("-" * 80 + "\n")
    f.write("GESAMT-UEBERSICHT\n")
    f.write("-" * 80 + "\n")
    analyze_rejection_group(df, "ALL", f)

    # === BY GAP DIRECTION ===
    f.write("\n" + "-" * 80 + "\n")
    f.write("NACH GAP-RICHTUNG\n")
    f.write("-" * 80 + "\n")
    for gd in ['up', 'down']:
        analyze_rejection_group(df[df['gap_dir'] == gd], f"Gap{gd.title()}", f)

    # === BY TIME OF REJECTION ===
    f.write("\n" + "-" * 80 + "\n")
    f.write("NACH ZEITPUNKT DER ABLEHNUNG\n")
    f.write("-" * 80 + "\n")
    for gd in ['up', 'down']:
        for tb in ['9:50-10:00', '10:00-10:15', '10:15-10:30', '10:30-11:00']:
            sub = df[(df['gap_dir'] == gd) & (df['rej_time_bucket'] == tb)]
            analyze_rejection_group(sub, f"Gap{gd.title()} + {tb}", f)

    # === BY GAP SIZE ===
    f.write("\n" + "-" * 80 + "\n")
    f.write("NACH GAP-GROESSE\n")
    f.write("-" * 80 + "\n")
    for gd in ['up', 'down']:
        for gb in ['<1 ADR', '1-2 ADR', '>2 ADR']:
            sub = df[(df['gap_dir'] == gd) & (df['gap_bucket'] == gb)]
            analyze_rejection_group(sub, f"Gap{gd.title()} + {gb}", f)

    # === BY RVOL ===
    f.write("\n" + "-" * 80 + "\n")
    f.write("NACH RVOL\n")
    f.write("-" * 80 + "\n")
    for gd in ['up', 'down']:
        for rb in ['<2x', '2-5x', '>5x']:
            sub = df[(df['gap_dir'] == gd) & (df['rvol_bucket'] == rb)]
            analyze_rejection_group(sub, f"Gap{gd.title()} + RVOL {rb}", f)

    # === BY BANDWIDTH ===
    f.write("\n" + "-" * 80 + "\n")
    f.write("NACH BANDBREITE (2*StdDev/ADR zum Zeitpunkt der Ablehnung)\n")
    f.write("-" * 80 + "\n")
    for gd in ['up', 'down']:
        for bw in ['eng (<0.2)', 'mittel (0.2-0.4)', 'weit (>0.4)']:
            sub = df[(df['gap_dir'] == gd) & (df['bw_bucket'] == bw)]
            analyze_rejection_group(sub, f"Gap{gd.title()} + {bw}", f)

    # === BY PRIOR VWAP TESTS ===
    f.write("\n" + "-" * 80 + "\n")
    f.write("NACH ANZAHL BISHERIGER VWAP-TESTS\n")
    f.write("-" * 80 + "\n")
    for gd in ['up', 'down']:
        for n_tests in ['1', '2-3', '4+']:
            if n_tests == '1':
                sub = df[(df['gap_dir'] == gd) & (df['prior_vwap_tests'] <= 1)]
            elif n_tests == '2-3':
                sub = df[(df['gap_dir'] == gd) & (df['prior_vwap_tests'].between(2, 3))]
            else:
                sub = df[(df['gap_dir'] == gd) & (df['prior_vwap_tests'] >= 4)]
            analyze_rejection_group(sub, f"Gap{gd.title()} + Tests={n_tests}", f)

    # === BY WHETHER VWAP WAS BROKEN BEFORE ===
    f.write("\n" + "-" * 80 + "\n")
    f.write("VWAP VORHER SCHON GEBROCHEN vs ERSTE ABLEHNUNG\n")
    f.write("-" * 80 + "\n")
    for gd in ['up', 'down']:
        sub_first = df[(df['gap_dir'] == gd) & (~df['vwap_broken_before'])]
        sub_nth = df[(df['gap_dir'] == gd) & (df['vwap_broken_before'])]
        analyze_rejection_group(sub_first, f"Gap{gd.title()} + VWAP noch nie gebrochen", f)
        analyze_rejection_group(sub_nth, f"Gap{gd.title()} + VWAP vorher gebrochen", f)

    # === KEY TAKEAWAYS ===
    f.write("\n" + "=" * 80 + "\n")
    f.write("KERN-ERKENNTNISSE\n")
    f.write("=" * 80 + "\n")

    s1_overall = df['sigma1_before_broken'].mean() * 100
    broken_overall = df['broken_before_sigma1'].mean() * 100

    f.write(f"\n  1. Sigma-1 erreicht BEVOR Rejection gebrochen: {s1_overall:.1f}%\n")
    f.write(f"  2. Rejection gebrochen BEVOR Sigma-1: {broken_overall:.1f}%\n")
    f.write(f"  3. Gesamt-Events: {len(df)} (von {len(h1)} Gapper-Tagen)\n")
    f.write(f"  4. Median Bandwidth bei Rejection: {df['bandwidth'].median():.3f} ADR\n")

print("Done! Results in results/cheat_sheet_v2_vwap_rejection.txt", file=sys.stderr)
