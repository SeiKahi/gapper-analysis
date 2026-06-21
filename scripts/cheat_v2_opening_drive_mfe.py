"""
Durchlauf 4.5 — Aufgabe 3: Opening Drive MFE before SL is hit
Trading window: 9:30-11:00 ET (no new trades after 11:00)

For each gapper day:
1. Determine Opening Drive direction from first 5 min (9:30-9:35)
2. Set SL at High/Low of first 5 min
3. Track MFE in OD direction before SL hit (using OHLC!)
4. Track win rates at various targets

Breakdowns: Gap direction, OD with/against gap, gap size, RVOL, ADR%, OD strength
Also: Alternative OD filter (first 3x1min bars all same direction)
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
    if pd.isna(adr) or adr <= 0:
        continue

    raw_path = Path(f'data/raw_1min/{ticker}/{date_str}.parquet')
    if not raw_path.exists():
        continue

    try:
        raw = pd.read_parquet(raw_path)
    except Exception:
        continue

    # Filter RTH only
    rth = raw[raw['session'] == 'rth'].copy()
    if len(rth) < 10:
        continue

    # Parse time
    rth['time_et'] = pd.to_datetime(rth['time_et'], format='%H:%M:%S', errors='coerce')
    if rth['time_et'].isna().all():
        rth['time_et'] = pd.to_datetime(rth['time_et'].astype(str))

    # Create minutes since 9:30
    rth = rth.sort_values('time_et').reset_index(drop=True)
    t930 = rth['time_et'].iloc[0]
    rth['min_idx'] = range(len(rth))

    # Limit to trading window: 9:30-11:00 = first 90 minutes (bars 0-89)
    # But we need data until 11:00 for tracking
    rth_window = rth[rth['min_idx'] < 91].copy()  # 0 to 90 inclusive

    if len(rth_window) < 6:  # Need at least 5min + 1 for OD
        continue

    # === OPENING DRIVE (first 5 min: bars 0-4) ===
    od_bars = rth_window[rth_window['min_idx'] < 5]
    if len(od_bars) < 5:
        continue

    open_930 = od_bars.iloc[0]['open']
    close_935 = od_bars.iloc[-1]['close']
    od_high = od_bars['high'].max()
    od_low = od_bars['low'].min()

    od_range_adr = (od_high - od_low) / adr if adr > 0 else 0

    if close_935 == open_930:
        continue  # No direction

    od_long = close_935 > open_930

    # SL levels
    sl_long = od_low    # SL for long = low of first 5min
    sl_short = od_high   # SL for short = high of first 5min

    sl_price = sl_long if od_long else sl_short
    sl_size_adr = abs(close_935 - sl_price) / adr if adr > 0 else 0

    # === ALTERNATIVE OD FILTER: First 3 bars all same direction ===
    first3 = rth_window[rth_window['min_idx'] < 3]
    if len(first3) >= 3:
        all_up = all(first3.iloc[i]['close'] > first3.iloc[i]['open'] for i in range(3))
        all_down = all(first3.iloc[i]['close'] < first3.iloc[i]['open'] for i in range(3))
        alt_od = 'up' if all_up else ('down' if all_down else 'mixed')
    else:
        alt_od = 'mixed'

    # === TRACK MFE/MAE from bar 5 onwards (after 9:35) until 11:00 ===
    post_od = rth_window[rth_window['min_idx'] >= 5].copy()
    if len(post_od) < 1:
        continue

    entry_price = close_935

    mfe = 0.0  # Max favorable excursion (in OD direction)
    mae = 0.0  # Max adverse excursion (against OD direction)
    mfe_before_sl = 0.0
    sl_hit = False
    sl_hit_bar = None
    sl_hit_time_min = None

    # Track targets
    targets_adr = [0.25, 0.50, 0.75, 1.00]
    target_hit = {t: False for t in targets_adr}
    target_hit_before_sl = {t: False for t in targets_adr}
    target_hit_time = {t: None for t in targets_adr}

    for _, bar in post_od.iterrows():
        bar_idx = bar['min_idx']

        if od_long:
            # Favorable = price goes up
            bar_mfe = (bar['high'] - entry_price) / adr
            bar_mae = (entry_price - bar['low']) / adr

            # Check SL hit (low < sl_long)
            bar_sl_hit = bar['low'] <= sl_long

            # Check targets (high >= target)
            for t in targets_adr:
                if not target_hit[t]:
                    if bar['high'] >= entry_price + t * adr:
                        target_hit[t] = True
                        target_hit_time[t] = bar_idx - 5  # minutes after entry
                        if not sl_hit:
                            target_hit_before_sl[t] = True
        else:
            # Favorable = price goes down
            bar_mfe = (entry_price - bar['low']) / adr
            bar_mae = (bar['high'] - entry_price) / adr

            # Check SL hit (high > sl_short)
            bar_sl_hit = bar['high'] >= sl_short

            # Check targets (low <= target)
            for t in targets_adr:
                if not target_hit[t]:
                    if bar['low'] <= entry_price - t * adr:
                        target_hit[t] = True
                        target_hit_time[t] = bar_idx - 5
                        if not sl_hit:
                            target_hit_before_sl[t] = True

        # Update MFE/MAE
        if bar_mfe > mfe:
            mfe = bar_mfe
        if bar_mae > mae:
            mae = bar_mae

        # Track MFE before SL
        if not sl_hit:
            if bar_mfe > mfe_before_sl:
                mfe_before_sl = bar_mfe

        # Check SL
        if not sl_hit and bar_sl_hit:
            sl_hit = True
            sl_hit_bar = bar_idx
            sl_hit_time_min = bar_idx - 5  # minutes after entry

    # === MFE at specific times (15min, 30min, 60min after entry) ===
    def get_mfe_at_time(post_bars, entry_px, is_long, adr_val, target_min):
        bars_in_window = post_bars[post_bars['min_idx'] < 5 + target_min]
        if len(bars_in_window) == 0:
            return np.nan
        if is_long:
            return (bars_in_window['high'].max() - entry_px) / adr_val
        else:
            return (entry_px - bars_in_window['low'].min()) / adr_val

    mfe_15 = get_mfe_at_time(post_od, entry_price, od_long, adr, 15)
    mfe_30 = get_mfe_at_time(post_od, entry_price, od_long, adr, 30)
    mfe_60 = get_mfe_at_time(post_od, entry_price, od_long, adr, 60)

    # === CLASSIFY ===
    gap_dir = row['gap_direction']
    od_dir = 'with_gap' if (od_long and gap_dir == 'up') or (not od_long and gap_dir == 'down') else 'against_gap'

    gap_size = row['gap_size_in_adr'] if pd.notna(row.get('gap_size_in_adr')) else np.nan
    rvol = row.get('rvol_open_30min', np.nan)
    if pd.isna(rvol):
        rvol = row.get('rvol_at_time_30min', np.nan)

    adr_pct = adr / row['today_open'] * 100 if pd.notna(row.get('today_open')) and row['today_open'] > 0 else np.nan

    # Gap size bucket
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
    if pd.notna(rvol):
        if rvol < 2:
            rvol_bucket = '<2x'
        elif rvol < 5:
            rvol_bucket = '2-5x'
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

    # OD strength bucket
    if od_range_adr < 0.2:
        od_str_bucket = '<0.2'
    elif od_range_adr < 0.5:
        od_str_bucket = '0.2-0.5'
    else:
        od_str_bucket = '>0.5'

    # Close vs Open drift
    rth_close = row.get('rth_close', np.nan)
    today_open = row.get('today_open', np.nan)
    if pd.notna(rth_close) and pd.notna(today_open):
        drift_adr = (rth_close - today_open) / adr
    else:
        drift_adr = np.nan

    results.append({
        'ticker': ticker,
        'date': date_str,
        'gap_dir': gap_dir,
        'od_long': od_long,
        'od_dir': od_dir,  # with_gap or against_gap
        'gap_bucket': gap_bucket,
        'rvol_bucket': rvol_bucket,
        'adr_pct_bucket': adr_pct_bucket,
        'od_str_bucket': od_str_bucket,
        'alt_od': alt_od,
        'sl_size_adr': sl_size_adr,
        'od_range_adr': od_range_adr,
        'mfe': mfe,
        'mae': mae,
        'mfe_before_sl': mfe_before_sl,
        'sl_hit': sl_hit,
        'sl_hit_time_min': sl_hit_time_min,
        'mfe_15': mfe_15,
        'mfe_30': mfe_30,
        'mfe_60': mfe_60,
        'drift_adr': drift_adr,
        **{f'tgt_{t}_hit': target_hit[t] for t in targets_adr},
        **{f'tgt_{t}_before_sl': target_hit_before_sl[t] for t in targets_adr},
        **{f'tgt_{t}_time': target_hit_time[t] for t in targets_adr},
    })

df = pd.DataFrame(results)
df.to_parquet('results/cheat_v2_od_mfe_raw.parquet', index=False)
print(f"\nProcessed {len(df)} gapper days", file=sys.stderr)

# === ANALYSIS ===
def bootstrap_ci(data, func=np.mean, n_boot=1000, ci=0.95):
    """Bootstrap confidence interval."""
    if len(data) < 3:
        return np.nan, np.nan
    boots = [func(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_boot)]
    lo = np.percentile(boots, (1 - ci) / 2 * 100)
    hi = np.percentile(boots, (1 + ci) / 2 * 100)
    return lo, hi

def analyze_group(group_df, label, f):
    n = len(group_df)
    if n < 10:
        f.write(f"\n  {label}: N={n} (TOO FEW)\n")
        return

    low_n = " [LOW N]" if n < 50 else ""

    sl_rate = group_df['sl_hit'].mean() * 100
    sl_ci = bootstrap_ci(group_df['sl_hit'].values.astype(float))

    # MFE before SL
    mfe_bs_mean = group_df['mfe_before_sl'].mean()
    mfe_bs_med = group_df['mfe_before_sl'].median()
    mfe_bs_ci = bootstrap_ci(group_df['mfe_before_sl'].values)

    # MFE when SL NOT hit
    no_sl = group_df[~group_df['sl_hit']]
    if len(no_sl) > 0:
        mfe_nosl_mean = no_sl['mfe'].mean()
        mfe_nosl_med = no_sl['mfe'].median()
    else:
        mfe_nosl_mean = mfe_nosl_med = np.nan

    # MFE before SL when SL IS hit
    sl_only = group_df[group_df['sl_hit']]
    if len(sl_only) > 0:
        mfe_sl_mean = sl_only['mfe_before_sl'].mean()
        mfe_sl_med = sl_only['mfe_before_sl'].median()
    else:
        mfe_sl_mean = mfe_sl_med = np.nan

    # Median time to SL
    sl_times = sl_only['sl_hit_time_min'].dropna()
    med_sl_time = sl_times.median() if len(sl_times) > 0 else np.nan

    # Win rates at targets (target hit BEFORE SL)
    wr = {}
    for t in [0.25, 0.50, 0.75, 1.00]:
        wr[t] = group_df[f'tgt_{t}_before_sl'].mean() * 100

    # SL size
    sl_size_mean = group_df['sl_size_adr'].mean()

    # Risk-Reward: SL size vs MFE
    rr_mfe = mfe_bs_mean / sl_size_mean if sl_size_mean > 0 else np.nan

    f.write(f"\n  {label}{low_n}:\n")
    f.write(f"    N={n}, SL-Hit={sl_rate:.1f}% (CI: {sl_ci[0]*100:.1f}-{sl_ci[1]*100:.1f}%)\n")
    f.write(f"    SL-Size: {sl_size_mean:.3f} ADR\n")
    f.write(f"    MFE before SL: Mean={mfe_bs_mean:.3f} ADR (CI: {mfe_bs_ci[0]:.3f}-{mfe_bs_ci[1]:.3f}), Median={mfe_bs_med:.3f}\n")
    f.write(f"    MFE when SL hit (before SL): Mean={mfe_sl_mean:.3f}, Median={mfe_sl_med:.3f}\n")
    f.write(f"    MFE when SL NOT hit (full): Mean={mfe_nosl_mean:.3f}, Median={mfe_nosl_med:.3f}\n")
    f.write(f"    Median time to SL hit: {med_sl_time:.0f} min\n")
    f.write(f"    Win-Rates (target before SL): 0.25={wr[0.25]:.1f}%, 0.50={wr[0.50]:.1f}%, 0.75={wr[0.75]:.1f}%, 1.00={wr[1.00]:.1f}%\n")
    f.write(f"    Risk-Reward (MFE/SL): {rr_mfe:.2f}\n")
    f.write(f"    MFE@15min: {group_df['mfe_15'].mean():.3f}, MFE@30min: {group_df['mfe_30'].mean():.3f}, MFE@60min: {group_df['mfe_60'].mean():.3f}\n")


# === WRITE RESULTS ===
with open('results/cheat_sheet_v2_opening_drive_mfe.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("CHEAT SHEET v2 — AUFGABE 3: OPENING DRIVE MFE VOR SL-HIT\n")
    f.write("=" * 80 + "\n")
    f.write(f"Datenbasis: {len(df)} Gapper-Tage (Halfte 1: 2021-02-21 bis 2023-12-31)\n")
    f.write(f"Trading-Fenster: 9:35 (nach OD) bis 11:00 ET\n")
    f.write(f"SL: High/Low der ersten 5 Minuten (9:30-9:35)\n")
    f.write(f"OD-Richtung: Close 9:35 vs Open 9:30\n\n")

    # === OVERALL ===
    f.write("-" * 80 + "\n")
    f.write("GESAMT-UEBERSICHT\n")
    f.write("-" * 80 + "\n")
    analyze_group(df, "ALL", f)

    # === BY GAP DIRECTION ===
    f.write("\n" + "-" * 80 + "\n")
    f.write("NACH GAP-RICHTUNG\n")
    f.write("-" * 80 + "\n")
    for gd in ['up', 'down']:
        analyze_group(df[df['gap_dir'] == gd], f"Gap{gd.title()}", f)

    # === BY OD DIRECTION (with/against gap) ===
    f.write("\n" + "-" * 80 + "\n")
    f.write("NACH OD-RICHTUNG (mit/gegen Gap)\n")
    f.write("-" * 80 + "\n")
    for gd in ['up', 'down']:
        for od in ['with_gap', 'against_gap']:
            sub = df[(df['gap_dir'] == gd) & (df['od_dir'] == od)]
            analyze_group(sub, f"Gap{gd.title()} + OD {od}", f)

    # === BY GAP SIZE ===
    f.write("\n" + "-" * 80 + "\n")
    f.write("NACH GAP-GROESSE\n")
    f.write("-" * 80 + "\n")
    for gd in ['up', 'down']:
        for gb in ['<1 ADR', '1-2 ADR', '>2 ADR']:
            sub = df[(df['gap_dir'] == gd) & (df['gap_bucket'] == gb)]
            analyze_group(sub, f"Gap{gd.title()} + {gb}", f)

    # === BY RVOL ===
    f.write("\n" + "-" * 80 + "\n")
    f.write("NACH RVOL\n")
    f.write("-" * 80 + "\n")
    for gd in ['up', 'down']:
        for rb in ['<2x', '2-5x', '>5x']:
            sub = df[(df['gap_dir'] == gd) & (df['rvol_bucket'] == rb)]
            analyze_group(sub, f"Gap{gd.title()} + RVOL {rb}", f)

    # === BY ADR% ===
    f.write("\n" + "-" * 80 + "\n")
    f.write("NACH ADR%\n")
    f.write("-" * 80 + "\n")
    for gd in ['up', 'down']:
        for ab in ['<5%', '5-10%', '>10%']:
            sub = df[(df['gap_dir'] == gd) & (df['adr_pct_bucket'] == ab)]
            analyze_group(sub, f"Gap{gd.title()} + ADR% {ab}", f)

    # === BY OD STRENGTH ===
    f.write("\n" + "-" * 80 + "\n")
    f.write("NACH OD-STAERKE (5min Range / ADR)\n")
    f.write("-" * 80 + "\n")
    for gd in ['up', 'down']:
        for od in ['with_gap', 'against_gap']:
            for os_b in ['<0.2', '0.2-0.5', '>0.5']:
                sub = df[(df['gap_dir'] == gd) & (df['od_dir'] == od) & (df['od_str_bucket'] == os_b)]
                analyze_group(sub, f"Gap{gd.title()} + OD {od} + Strength {os_b}", f)

    # === SL HIT RATE BY TIME ===
    f.write("\n" + "-" * 80 + "\n")
    f.write("SL-HIT NACH ZEITPUNKT (Minuten nach 9:35)\n")
    f.write("-" * 80 + "\n")
    sl_df = df[df['sl_hit']].copy()
    time_bins = [(0, 5), (5, 10), (10, 15), (15, 30), (30, 60), (60, 90)]
    for lo, hi in time_bins:
        in_bin = sl_df[(sl_df['sl_hit_time_min'] >= lo) & (sl_df['sl_hit_time_min'] < hi)]
        pct = len(in_bin) / len(df) * 100 if len(df) > 0 else 0
        cum_pct = len(sl_df[sl_df['sl_hit_time_min'] < hi]) / len(df) * 100 if len(df) > 0 else 0
        f.write(f"  {lo}-{hi}min: {len(in_bin)} SL hits ({pct:.1f}%), Cumulative: {cum_pct:.1f}%\n")

    # === ALTERNATIVE OD FILTER (3 bars same direction) ===
    f.write("\n" + "-" * 80 + "\n")
    f.write("ALTERNATIVER OD-FILTER: Erste 3 Bars alle gleiche Richtung\n")
    f.write("-" * 80 + "\n")
    f.write(f"  3 Bars Up: N={len(df[df['alt_od']=='up'])}\n")
    f.write(f"  3 Bars Down: N={len(df[df['alt_od']=='down'])}\n")
    f.write(f"  Mixed: N={len(df[df['alt_od']=='mixed'])}\n\n")

    # Compare: standard OD vs alternative for GapUp
    for gd in ['up', 'down']:
        f.write(f"  Gap{gd.title()} — Standard 5min OD:\n")
        analyze_group(df[(df['gap_dir'] == gd) & (df['od_dir'] == 'with_gap')],
                     f"Standard OD with_gap", f)

        # Alt: 3 bars aligned with gap
        if gd == 'up':
            alt_with = df[(df['gap_dir'] == gd) & (df['alt_od'] == 'up')]
            alt_against = df[(df['gap_dir'] == gd) & (df['alt_od'] == 'down')]
        else:
            alt_with = df[(df['gap_dir'] == gd) & (df['alt_od'] == 'down')]
            alt_against = df[(df['gap_dir'] == gd) & (df['alt_od'] == 'up')]

        analyze_group(alt_with, f"Alt 3-Bar with_gap", f)
        analyze_group(alt_against, f"Alt 3-Bar against_gap", f)

    # === RISK-REWARD SUMMARY ===
    f.write("\n" + "-" * 80 + "\n")
    f.write("RISK-REWARD ZUSAMMENFASSUNG\n")
    f.write("-" * 80 + "\n")

    f.write("\n  SL-Groesse Verteilung (5min Range in ADR):\n")
    f.write(f"    Mean: {df['sl_size_adr'].mean():.3f}\n")
    f.write(f"    Median: {df['sl_size_adr'].median():.3f}\n")
    f.write(f"    P25: {df['sl_size_adr'].quantile(0.25):.3f}\n")
    f.write(f"    P75: {df['sl_size_adr'].quantile(0.75):.3f}\n")
    f.write(f"    P90: {df['sl_size_adr'].quantile(0.90):.3f}\n")

    f.write("\n  Gesamt-MFE Verteilung (Max Favorable in OD direction, alle Bars bis 11:00):\n")
    f.write(f"    Mean: {df['mfe'].mean():.3f}\n")
    f.write(f"    Median: {df['mfe'].median():.3f}\n")
    f.write(f"    P75: {df['mfe'].quantile(0.75):.3f}\n")
    f.write(f"    P90: {df['mfe'].quantile(0.90):.3f}\n")

    # === KEY TAKEAWAYS ===
    f.write("\n" + "=" * 80 + "\n")
    f.write("KERN-ERKENNTNISSE\n")
    f.write("=" * 80 + "\n")

    sl_rate_all = df['sl_hit'].mean() * 100

    # Best groups
    best_wr_50 = 0
    best_label = ""
    for gd in ['up', 'down']:
        for od in ['with_gap', 'against_gap']:
            sub = df[(df['gap_dir'] == gd) & (df['od_dir'] == od)]
            if len(sub) >= 50:
                wr = sub['tgt_0.5_before_sl'].mean() * 100
                if wr > best_wr_50:
                    best_wr_50 = wr
                    best_label = f"Gap{gd.title()} + OD {od}"

    f.write(f"\n  1. SL-Hit-Rate gesamt: {sl_rate_all:.1f}%\n")
    f.write(f"  2. Beste WR bei 0.50 ADR Target: {best_label} = {best_wr_50:.1f}%\n")
    f.write(f"  3. Median SL-Groesse: {df['sl_size_adr'].median():.3f} ADR\n")
    f.write(f"  4. Median MFE before SL: {df['mfe_before_sl'].median():.3f} ADR\n")

print("Done! Results in results/cheat_sheet_v2_opening_drive_mfe.txt", file=sys.stderr)
