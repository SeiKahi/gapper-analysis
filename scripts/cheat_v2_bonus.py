"""
Durchlauf 4.5 — Aufgabe 6: Bonus Trader Questions

6.1 If VWAP not broken in first 30min: How often does VWAP hold as resistance/support until 11:00?
6.2 First candle (1min): Large (>0.3 ADR) vs small (<0.1 ADR) -> different day behavior?
6.3 Gap + Previous day close location: Yesterday near HOD (CL>0.8) + GapUp = momentum or exhaustion?

Trading window: 9:30-11:00 ET
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

# === PROCESS ===
results = []

for _, row in tqdm(h1.iterrows(), total=len(h1), desc="Processing", file=sys.stderr):
    ticker = row['ticker']
    date_str = row['date'].strftime('%Y-%m-%d')
    adr = row['adr_10']
    gap_dir = row['gap_direction']
    if pd.isna(adr) or adr <= 0:
        continue

    raw_path = Path(f'data/raw_1min/{ticker}/{date_str}.parquet')
    vwap_path = Path(f'data/vwap/{ticker}/{date_str}.parquet')
    if not raw_path.exists() or not vwap_path.exists():
        continue

    try:
        raw = pd.read_parquet(raw_path)
        vwap_df = pd.read_parquet(vwap_path)
    except Exception:
        continue

    rth = raw[raw['session'] == 'rth'].copy()
    if len(rth) < 30:
        continue

    rth = rth.sort_values('time_et').reset_index(drop=True)
    rth['min_idx'] = range(len(rth))

    vwap_df['time_et'] = pd.to_datetime(vwap_df['time_et'], format='%H:%M:%S', errors='coerce')
    vwap_df = vwap_df.sort_values('time_et').reset_index(drop=True)
    rth['time_str'] = rth['time_et'].astype(str).str[-8:-3] if rth['time_et'].dtype == 'object' else pd.to_datetime(rth['time_et'], format='%H:%M:%S', errors='coerce').dt.strftime('%H:%M')
    vwap_df['time_str'] = vwap_df['time_et'].dt.strftime('%H:%M')

    merged = pd.merge(rth, vwap_df[['time_str', 'vwap']], on='time_str', how='left', suffixes=('', '_v'))
    merged = merged.sort_values('min_idx').reset_index(drop=True)

    open_price = rth.iloc[0]['open']
    prev_close = row.get('prev_close', np.nan)

    # === 6.1: VWAP not broken in first 30min ===
    first30 = merged[merged['min_idx'] < 30]
    vwap_broken_30 = False

    # "Broken" = 3 consecutive closes beyond VWAP
    consecutive = 0
    for _, bar in first30.iterrows():
        if pd.isna(bar.get('vwap', np.nan)):
            consecutive = 0
            continue
        if gap_dir == 'down':
            # VWAP is resistance; broken if close > VWAP for 3 bars
            if bar['close'] > bar['vwap']:
                consecutive += 1
            else:
                consecutive = 0
        else:
            # VWAP is support; broken if close < VWAP for 3 bars
            if bar['close'] < bar['vwap']:
                consecutive += 1
            else:
                consecutive = 0
        if consecutive >= 3:
            vwap_broken_30 = True
            break

    # If VWAP NOT broken in 30min, check if it holds until 11:00
    vwap_holds_1100 = np.nan
    if not vwap_broken_30:
        bars_30_90 = merged[(merged['min_idx'] >= 30) & (merged['min_idx'] <= 90)]
        consecutive = 0
        vwap_holds_1100 = True
        for _, bar in bars_30_90.iterrows():
            if pd.isna(bar.get('vwap', np.nan)):
                consecutive = 0
                continue
            if gap_dir == 'down':
                if bar['close'] > bar['vwap']:
                    consecutive += 1
                else:
                    consecutive = 0
            else:
                if bar['close'] < bar['vwap']:
                    consecutive += 1
                else:
                    consecutive = 0
            if consecutive >= 3:
                vwap_holds_1100 = False
                break

    # === 6.2: First candle analysis ===
    first_candle = rth.iloc[0]
    first_range_adr = (first_candle['high'] - first_candle['low']) / adr if adr > 0 else 0

    if first_range_adr > 0.3:
        first_candle_bucket = 'grosse Kerze (>0.3 ADR)'
    elif first_range_adr < 0.1:
        first_candle_bucket = 'kleine Kerze (<0.1 ADR)'
    else:
        first_candle_bucket = 'mittlere Kerze (0.1-0.3 ADR)'

    first_candle_dir = 'up' if first_candle['close'] > first_candle['open'] else 'down'

    # === 6.3: Previous day close location ===
    # We need previous day's close location. Check if metadata has it.
    # We can compute it from the gap: if prev_close is near prev_high => CL near 1
    # But we need prev_high/prev_low. Check metadata for these.
    prev_cl = row.get('prev_close_location', np.nan)

    # If not available, skip
    if pd.notna(prev_cl):
        if prev_cl > 0.8:
            prev_cl_bucket = 'near HOD (>0.8)'
        elif prev_cl < 0.2:
            prev_cl_bucket = 'near LOD (<0.2)'
        else:
            prev_cl_bucket = 'middle (0.2-0.8)'
    else:
        prev_cl_bucket = 'unknown'

    # Day metrics
    rth_close = row.get('rth_close', np.nan)
    rth_high_val = row.get('rth_high', np.nan)
    rth_low_val = row.get('rth_low', np.nan)
    cl = (rth_close - rth_low_val) / (rth_high_val - rth_low_val) if pd.notna(rth_high_val) and pd.notna(rth_low_val) and (rth_high_val - rth_low_val) > 0 else np.nan
    drift_day = (rth_close - open_price) / adr if pd.notna(rth_close) else np.nan

    window = rth[rth['min_idx'] <= 90]
    if len(window) >= 5:
        close_1100 = window.iloc[-1]['close']
        drift_1100 = (close_1100 - open_price) / adr

        if gap_dir == 'up':
            mfe = (window['high'].max() - open_price) / adr
            mae = (open_price - window['low'].min()) / adr
        else:
            mfe = (open_price - window['low'].min()) / adr
            mae = (window['high'].max() - open_price) / adr
    else:
        drift_1100 = mfe = mae = np.nan

    day_range = (rth['high'].max() - rth['low'].min()) / adr

    # Fill
    if pd.notna(prev_close):
        if gap_dir == 'up':
            filled = window['low'].min() <= prev_close if len(window) >= 5 else np.nan
        else:
            filled = window['high'].max() >= prev_close if len(window) >= 5 else np.nan
    else:
        filled = np.nan

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
        'vwap_broken_30': vwap_broken_30,
        'vwap_holds_1100': vwap_holds_1100,
        'first_range_adr': first_range_adr,
        'first_candle_bucket': first_candle_bucket,
        'first_candle_dir': first_candle_dir,
        'prev_cl_bucket': prev_cl_bucket,
        'prev_cl': prev_cl,
        'drift_1100': drift_1100,
        'drift_day': drift_day,
        'mfe': mfe,
        'mae': mae,
        'day_range': day_range,
        'filled': filled,
        'cl': cl,
    })

df = pd.DataFrame(results)
df.to_parquet('results/cheat_v2_bonus_raw.parquet', index=False)
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
    mfe_m = sub['mfe'].mean()
    mae_m = sub['mae'].mean()
    dr = sub['day_range'].mean()
    ci_str = f"(CI: {drift_ci[0]:+.3f} to {drift_ci[1]:+.3f})" if pd.notna(drift_ci[0]) else ""
    fill_str = f"{fill:.0f}%" if pd.notna(fill) else "---"
    cl_str = f"{cl:.2f}" if pd.notna(cl) else "---"
    f.write(f"    {label}{low}: N={n:>4}, Drift={drift:+.3f} {ci_str}, Fill={fill_str}, CL={cl_str}, MFE={mfe_m:.3f}, MAE={mae_m:.3f}, DayRange={dr:.2f}\n")


with open('results/cheat_sheet_v2_bonus.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("CHEAT SHEET v2 — AUFGABE 6: BONUS TRADER-FRAGEN\n")
    f.write("=" * 80 + "\n")
    f.write(f"Datenbasis: {len(df)} Gapper-Tage (Halfte 1)\n\n")

    # === 6.1: VWAP HOLD WITHIN TRADING WINDOW ===
    f.write("=" * 80 + "\n")
    f.write("6.1: VWAP NICHT GEBROCHEN IN ERSTEN 30min -> HAELT BIS 11:00?\n")
    f.write("=" * 80 + "\n")
    f.write("Definition: 'Gebrochen' = 3+ aufeinanderfolgende 1-min Closes jenseits VWAP\n\n")

    for gd in ['up', 'down']:
        gd_df = df[df['gap_dir'] == gd]
        not_broken = gd_df[~gd_df['vwap_broken_30']]
        n_total = len(gd_df)
        n_not_broken = len(not_broken)

        f.write(f"  Gap{gd.title()}: {n_not_broken} von {n_total} ({n_not_broken/n_total*100:.1f}%) NICHT gebrochen in 30min\n")

        if n_not_broken > 0:
            holds = not_broken['vwap_holds_1100']
            hold_rate = holds.mean() * 100
            hold_ci = bootstrap_ci(holds.dropna().values.astype(float))
            f.write(f"    Davon haelt VWAP bis 11:00: {hold_rate:.1f}% (CI: {hold_ci[0]*100:.1f}-{hold_ci[1]*100:.1f}%)\n")

            # Metrics for those where VWAP holds
            vwap_hold = not_broken[not_broken['vwap_holds_1100'] == True]
            vwap_break = not_broken[not_broken['vwap_holds_1100'] == False]

            if len(vwap_hold) >= 10:
                write_group(vwap_hold, "VWAP haelt bis 11:00", f)
            if len(vwap_break) >= 10:
                write_group(vwap_break, "VWAP bricht 30-90min", f)

        f.write("\n")

    # === 6.2: FIRST CANDLE SIZE ===
    f.write("=" * 80 + "\n")
    f.write("6.2: ERSTE KERZE (1min) — GROSS vs KLEIN\n")
    f.write("=" * 80 + "\n")
    f.write("Grosse Kerze = Range > 0.3 ADR | Kleine Kerze = Range < 0.1 ADR\n\n")

    for gd in ['up', 'down']:
        f.write(f"  Gap{gd.title()}:\n")
        gd_df = df[df['gap_dir'] == gd]

        for bucket in ['grosse Kerze (>0.3 ADR)', 'mittlere Kerze (0.1-0.3 ADR)', 'kleine Kerze (<0.1 ADR)']:
            sub = gd_df[gd_df['first_candle_bucket'] == bucket]
            write_group(sub, bucket, f)

        # First candle direction
        f.write(f"\n  Gap{gd.title()} — nach Kerzenrichtung:\n")
        for cd in ['up', 'down']:
            sub = gd_df[gd_df['first_candle_dir'] == cd]
            write_group(sub, f"Erste Kerze {cd}", f)

        # Interaction: size x direction
        f.write(f"\n  Gap{gd.title()} — Grosse Kerze x Richtung:\n")
        for cd in ['up', 'down']:
            sub = gd_df[(gd_df['first_candle_bucket'] == 'grosse Kerze (>0.3 ADR)') & (gd_df['first_candle_dir'] == cd)]
            write_group(sub, f"Grosse Kerze + {cd}", f)

        f.write("\n")

    # === 6.3: PREVIOUS DAY CLOSE LOCATION ===
    f.write("=" * 80 + "\n")
    f.write("6.3: VORHERIGER TAG CLOSE-LOCATION + GAP\n")
    f.write("=" * 80 + "\n")
    f.write("Near HOD (>0.8) = Gestern stark geschlossen | Near LOD (<0.2) = Gestern schwach\n\n")

    known_cl = df[df['prev_cl_bucket'] != 'unknown']
    f.write(f"  Tage mit bekannter Prev-CL: {len(known_cl)} ({len(known_cl)/len(df)*100:.1f}%)\n\n")

    for gd in ['up', 'down']:
        f.write(f"  Gap{gd.title()}:\n")
        gd_df = known_cl[known_cl['gap_dir'] == gd]

        for bucket in ['near HOD (>0.8)', 'middle (0.2-0.8)', 'near LOD (<0.2)']:
            sub = gd_df[gd_df['prev_cl_bucket'] == bucket]
            write_group(sub, bucket, f)

        f.write("\n")

    # Specific questions
    f.write("  --- Spezifische Fragen ---\n\n")

    # GapUp + Yesterday near HOD = Momentum or Exhaustion?
    gu_hod = known_cl[(known_cl['gap_dir'] == 'up') & (known_cl['prev_cl_bucket'] == 'near HOD (>0.8)')]
    gu_other = known_cl[(known_cl['gap_dir'] == 'up') & (known_cl['prev_cl_bucket'] != 'near HOD (>0.8)')]
    f.write(f"  GapUp + Gestern near HOD:\n")
    if len(gu_hod) >= 10:
        d = gu_hod['drift_1100'].mean()
        fill = gu_hod['filled'].mean() * 100 if gu_hod['filled'].notna().any() else np.nan
        f.write(f"    N={len(gu_hod)}, Drift={d:+.3f}, Fill={fill:.0f}%\n")
        d_other = gu_other['drift_1100'].mean()
        f.write(f"    vs Rest: N={len(gu_other)}, Drift={d_other:+.3f}\n")
        f.write(f"    -> {'MOMENTUM (Continuation)' if d > 0 else 'EXHAUSTION (Fade)'}\n")

    # GapDown + Yesterday near LOD = Continuation or Bounce?
    gd_lod = known_cl[(known_cl['gap_dir'] == 'down') & (known_cl['prev_cl_bucket'] == 'near LOD (<0.2)')]
    gd_other = known_cl[(known_cl['gap_dir'] == 'down') & (known_cl['prev_cl_bucket'] != 'near LOD (<0.2)')]
    f.write(f"\n  GapDown + Gestern near LOD:\n")
    if len(gd_lod) >= 10:
        d = gd_lod['drift_1100'].mean()
        fill = gd_lod['filled'].mean() * 100 if gd_lod['filled'].notna().any() else np.nan
        f.write(f"    N={len(gd_lod)}, Drift={d:+.3f}, Fill={fill:.0f}%\n")
        d_other = gd_other['drift_1100'].mean()
        f.write(f"    vs Rest: N={len(gd_other)}, Drift={d_other:+.3f}\n")
        f.write(f"    -> {'BOUNCE' if d > 0 else 'CONTINUATION (weiter runter)'}\n")

    # === KEY TAKEAWAYS ===
    f.write("\n\n" + "=" * 80 + "\n")
    f.write("KERN-ERKENNTNISSE\n")
    f.write("=" * 80 + "\n")

    f.write("\n  6.1: VWAP-Hold bis 11:00:\n")
    for gd in ['up', 'down']:
        not_broken = df[(df['gap_dir'] == gd) & (~df['vwap_broken_30'])]
        if len(not_broken) > 0:
            hold = not_broken['vwap_holds_1100'].mean() * 100
            f.write(f"    Gap{gd.title()}: {hold:.1f}% der Tage wo VWAP in 30min nicht bricht, haelt er auch bis 11:00\n")

    f.write("\n  6.2: Erste Kerze:\n")
    for gd in ['up', 'down']:
        big = df[(df['gap_dir'] == gd) & (df['first_candle_bucket'] == 'grosse Kerze (>0.3 ADR)')]
        small = df[(df['gap_dir'] == gd) & (df['first_candle_bucket'] == 'kleine Kerze (<0.1 ADR)')]
        if len(big) >= 10 and len(small) >= 10:
            f.write(f"    Gap{gd.title()}: Grosse Kerze Drift={big['drift_1100'].mean():+.3f} vs Kleine Kerze Drift={small['drift_1100'].mean():+.3f}\n")

print("Done! Results in results/cheat_sheet_v2_bonus.txt", file=sys.stderr)
