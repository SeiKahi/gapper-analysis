"""
Cheat Sheet Category 4: Zeitverhalten und Tagesstruktur
Questions 4.1-4.5: HOD/LOD timing, opening drive, lunch zone, power hour, early patterns
"""
import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm

# ── helpers ──────────────────────────────────────────────────────────
def bootstrap_ci(data, n_boot=1000, ci=0.95):
    if len(data) < 5:
        return np.nan, np.nan
    rng = np.random.default_rng(42)
    stats = np.array([np.mean(rng.choice(data, size=len(data), replace=True)) for _ in range(n_boot)])
    return np.percentile(stats, (1-ci)/2*100), np.percentile(stats, (1+ci)/2*100)

def fmt_pct(val, ci_lo, ci_hi):
    return f"{val*100:.1f}% (CI: {ci_lo*100:.0f}-{ci_hi*100:.0f}%)"

# ── load metadata ────────────────────────────────────────────────────
print("Loading metadata...", file=sys.stderr)
meta = pd.read_parquet("data/metadata/metadata_master.parquet")
meta = meta[(meta['date'] <= '2023-12-31') & (meta['adr_10'].notna())].copy()
meta['adr_pct'] = meta['adr_10'] / meta['today_open'] * 100
meta['gap_size_in_adr'] = meta['gap_size_in_adr'].fillna(
    (meta['today_open'] - meta['prev_close']).abs() / meta['adr_10']
)
meta['day_range_adr'] = (meta['rth_high'] - meta['rth_low']) / meta['adr_10']
meta['close_loc'] = (meta['rth_close'] - meta['rth_low']) / (meta['rth_high'] - meta['rth_low'])

def gap_size_bucket(v):
    if v < 1: return '<1x'
    if v < 2: return '1-2x'
    if v < 3: return '2-3x'
    return '>3x'
meta['gap_bucket'] = meta['gap_size_in_adr'].apply(gap_size_bucket)

print(f"Halfte 1: {len(meta)} gapper days", file=sys.stderr)

# ── Process each day for time-based analysis ─────────────────────────
day_results = []

for idx, row in tqdm(meta.iterrows(), total=len(meta), desc="Processing", file=sys.stderr):
    ticker = row['ticker']
    date = row['date']
    gap_dir = row['gap_direction']
    adr = row['adr_10']
    od_dir = row['opening_drive_direction']

    raw_path = f"data/raw_1min/{ticker}/{date}.parquet"
    if not os.path.exists(raw_path):
        continue

    try:
        raw_df = pd.read_parquet(raw_path)
    except:
        continue

    if 'session' in raw_df.columns:
        raw_df = raw_df[raw_df['session'] == 'rth'].copy()

    if len(raw_df) < 30 or adr <= 0:
        continue

    raw_df = raw_df.sort_values('time_et').reset_index(drop=True)

    def time_to_min(t):
        if isinstance(t, str):
            parts = t.split(':')
            return (int(parts[0]) - 9) * 60 + (int(parts[1]) - 30)
        return np.nan

    raw_df['min_from_open'] = raw_df['time_et'].apply(time_to_min)
    raw_df = raw_df[raw_df['min_from_open'] >= 0].reset_index(drop=True)

    if len(raw_df) < 20:
        continue

    day_result = {
        'ticker': ticker, 'date': date, 'gap_dir': gap_dir,
        'adr': adr, 'gap_bucket': row['gap_bucket'],
        'od_dir': od_dir,
    }

    # ── Q4.1: HOD/LOD Timing ──
    cummax = raw_df['high'].cummax()
    cummin = raw_df['low'].cummin()

    # HOD time = last time a new high was set
    hod_bar = raw_df.loc[raw_df['high'] == raw_df['high'].max()].iloc[-1]
    lod_bar = raw_df.loc[raw_df['low'] == raw_df['low'].min()].iloc[-1]

    day_result['hod_min'] = hod_bar['min_from_open']
    day_result['lod_min'] = lod_bar['min_from_open']

    # HOD/LOD in first 30 min?
    first_30 = raw_df[raw_df['min_from_open'] <= 30]
    day_result['hod_in_first30'] = (hod_bar['min_from_open'] <= 30)
    day_result['lod_in_first30'] = (lod_bar['min_from_open'] <= 30)

    # ── Q4.2: Opening Drive Character ──
    first_15 = raw_df[raw_df['min_from_open'] <= 15]
    if len(first_15) >= 5:
        od_range = (first_15['high'].max() - first_15['low'].min()) / adr
        od_direction_actual = 1 if first_15['close'].iloc[-1] > first_15['open'].iloc[0] else -1
        day_result['od_range_adr'] = od_range
        day_result['od_strong'] = od_range > 0.5
        day_result['od_weak'] = od_range < 0.2

        # OD continuation vs gap
        if gap_dir == 'up':
            day_result['od_with_gap'] = od_direction_actual > 0  # bullish OD with gap up
        else:
            day_result['od_with_gap'] = od_direction_actual < 0  # bearish OD with gap down

    # Day close direction
    if gap_dir == 'up':
        day_result['day_bullish'] = raw_df['close'].iloc[-1] > raw_df['open'].iloc[0]
    else:
        day_result['day_bullish'] = raw_df['close'].iloc[-1] < raw_df['open'].iloc[0]  # gap down, close lower = "with gap"

    # ── Q4.3: Lunch Zone (120-240 min from open = 11:30-13:30) ──
    morning = raw_df[raw_df['min_from_open'] <= 120]
    lunch = raw_df[(raw_df['min_from_open'] > 120) & (raw_df['min_from_open'] <= 240)]

    if len(morning) >= 20 and len(lunch) >= 20:
        morning_range = morning['high'].max() - morning['low'].min()
        lunch_range = lunch['high'].max() - lunch['low'].min()
        day_result['lunch_range_vs_morning'] = lunch_range / max(morning_range, 0.01)
        day_result['lunch_narrower'] = lunch_range < morning_range

        # Morning high/low broken in lunch?
        m_high = morning['high'].max()
        m_low = morning['low'].min()
        day_result['morning_high_broken_lunch'] = (lunch['high'] > m_high).any()
        day_result['morning_low_broken_lunch'] = (lunch['low'] < m_low).any()

        # If lunch breaks morning range -> follow through?
        if (lunch['high'] > m_high).any():
            # Check afternoon (after lunch)
            afternoon = raw_df[raw_df['min_from_open'] > 240]
            if len(afternoon) > 10:
                day_result['lunch_break_high_follow'] = afternoon['high'].max() > lunch['high'].max()

        if (lunch['low'] < m_low).any():
            afternoon = raw_df[raw_df['min_from_open'] > 240]
            if len(afternoon) > 10:
                day_result['lunch_break_low_follow'] = afternoon['low'].min() < lunch['low'].min()

    # ── Q4.4: Power Hour (330-390 min from open = 15:00-16:00) ──
    power_hour = raw_df[raw_df['min_from_open'] >= 330]
    before_ph = raw_df[raw_df['min_from_open'] < 330]

    if len(power_hour) >= 10 and len(before_ph) >= 30:
        pre_ph_high = before_ph['high'].max()
        pre_ph_low = before_ph['low'].min()
        day_result['ph_new_hod'] = (power_hour['high'] > pre_ph_high).any()
        day_result['ph_new_lod'] = (power_hour['low'] < pre_ph_low).any()

    # Close location (already in metadata but recalculate)
    day_high = raw_df['high'].max()
    day_low = raw_df['low'].min()
    if day_high > day_low:
        day_result['close_loc'] = (raw_df['close'].iloc[-1] - day_low) / (day_high - day_low)

    # ── Q4.5: Early Patterns (first 5 min, first 3 bars) ──
    first_5 = raw_df[raw_df['min_from_open'] <= 5]
    if len(first_5) >= 3:
        # First 5-min candle direction
        if len(first_5) >= 5:
            first_5min_close = first_5.iloc[4]['close']
            first_5min_open = first_5.iloc[0]['open']
        else:
            first_5min_close = first_5.iloc[-1]['close']
            first_5min_open = first_5.iloc[0]['open']

        day_result['first_5m_bullish'] = first_5min_close > first_5min_open
        day_result['first_5m_bearish'] = first_5min_close < first_5min_open
        day_result['first_5m_doji'] = abs(first_5min_close - first_5min_open) / adr < 0.02

        # First 3 bars all same direction
        first3 = raw_df.iloc[:3] if len(raw_df) >= 3 else raw_df
        if len(first3) == 3:
            all_up = all(first3['close'] > first3['open'])
            all_down = all(first3['close'] < first3['open'])
            day_result['first_3_all_up'] = all_up
            day_result['first_3_all_down'] = all_down

    day_results.append(day_result)

# ── Convert to DataFrame ──────────────────────────────────────────
print(f"\nProcessed {len(day_results)} days", file=sys.stderr)
df = pd.DataFrame(day_results)
df.to_parquet("results/cheat_cat4_raw.parquet", index=False)

# ── Generate Report ─────────────────────────────────────────────────
out = []
out.append("=" * 80)
out.append("CHEAT SHEET — KATEGORIE 4: ZEITVERHALTEN UND TAGESSTRUKTUR")
out.append("=" * 80)
out.append(f"Datenbasis: Halfte 1, N={len(df)}")
out.append("")

# ── Q4.1: HOD/LOD Timing ──
out.append("=" * 70)
out.append("FRAGE 4.1: Wann wird HOD/LOD gesetzt?")
out.append("=" * 70)

for gd in ['up', 'down']:
    label = 'GapUp' if gd == 'up' else 'GapDown'
    gsub = df[df['gap_dir'] == gd]
    out.append(f"\n--- {label} (N={len(gsub)}) ---")

    hod = gsub['hod_min'].dropna()
    lod = gsub['lod_min'].dropna()

    out.append(f"  HOD Timing: Median={hod.median():.0f}min, Mean={hod.mean():.0f}min")
    # Distribution
    for cutoff, label_t in [(30, '0-30min'), (60, '30-60min'), (120, '60-120min'), (240, '120-240min'), (390, '240-390min')]:
        prev = 0 if cutoff == 30 else [30, 60, 120, 240][[30, 60, 120, 240, 390].index(cutoff) - 1] if cutoff != 30 else 0
        if cutoff == 30:
            pct = (hod <= 30).mean()
        else:
            prev = [0, 30, 60, 120, 240][[30, 60, 120, 240, 390].index(cutoff)]
            pct = ((hod > prev) & (hod <= cutoff)).mean()
        out.append(f"    HOD in {label_t}: {pct*100:.0f}%")

    out.append(f"\n  LOD Timing: Median={lod.median():.0f}min, Mean={lod.mean():.0f}min")
    for cutoff, label_t in [(30, '0-30min'), (60, '30-60min'), (120, '60-120min'), (240, '120-240min'), (390, '240-390min')]:
        if cutoff == 30:
            pct = (lod <= 30).mean()
        else:
            prev = [0, 30, 60, 120, 240][[30, 60, 120, 240, 390].index(cutoff)]
            pct = ((lod > prev) & (lod <= cutoff)).mean()
        out.append(f"    LOD in {label_t}: {pct*100:.0f}%")

    # If HOD in first 30 min -> holds rest of day?
    hod30 = gsub['hod_in_first30'].dropna()
    lod30 = gsub['lod_in_first30'].dropna()
    out.append(f"\n  HOD in ersten 30min: {hod30.mean()*100:.0f}% der Tage")
    out.append(f"  LOD in ersten 30min: {lod30.mean()*100:.0f}% der Tage")

    # Breakdown by gap size
    out.append(f"\n  Breakdown nach Gap-Groesse:")
    for bucket in ['<1x', '1-2x', '2-3x', '>3x']:
        bsub = gsub[gsub['gap_bucket'] == bucket]
        if len(bsub) < 30:
            out.append(f"    {bucket}: N={len(bsub)} (LOW N)")
            continue
        out.append(f"    {bucket}: HOD Median={bsub['hod_min'].median():.0f}m, LOD Median={bsub['lod_min'].median():.0f}m, "
                   f"HOD<30m={bsub['hod_in_first30'].mean()*100:.0f}%, LOD<30m={bsub['lod_in_first30'].mean()*100:.0f}%, N={len(bsub)}")

# ── Q4.2: Opening Drive Character ──
out.append("\n" + "=" * 70)
out.append("FRAGE 4.2: Opening Drive Charakter")
out.append("=" * 70)

od_sub = df[df['od_range_adr'].notna()]

for gd in ['up', 'down']:
    label = 'GapUp' if gd == 'up' else 'GapDown'
    gsub = od_sub[od_sub['gap_dir'] == gd]
    out.append(f"\n--- {label} ---")

    # Strong OD
    strong = gsub[gsub['od_strong'] == True]
    weak = gsub[gsub['od_weak'] == True]
    medium = gsub[(gsub['od_strong'] == False) & (gsub['od_weak'] == False)]

    for slabel, ssub in [('Starker OD (>0.5 ADR in 15min)', strong),
                          ('Mittlerer OD (0.2-0.5 ADR)', medium),
                          ('Schwacher OD (<0.2 ADR)', weak)]:
        if len(ssub) < 20:
            out.append(f"  {slabel}: N={len(ssub)} (LOW N)")
            continue

        day_bull = ssub['day_bullish'].dropna().astype(float)
        cl = ssub['close_loc'].dropna()
        out.append(f"  {slabel}: N={len(ssub)}")
        if len(day_bull) > 0:
            out.append(f"    Tagesrichtung = Gap-Richtung: {day_bull.mean()*100:.0f}%")
        if len(cl) > 0:
            out.append(f"    Close Location Median: {cl.median():.2f}")

    # OD with vs against gap
    with_gap = gsub[gsub['od_with_gap'] == True]
    against_gap = gsub[gsub['od_with_gap'] == False]

    for slabel, ssub in [('OD MIT Gap (Continuation)', with_gap),
                          ('OD GEGEN Gap (Reversal)', against_gap)]:
        if len(ssub) < 30:
            out.append(f"  {slabel}: N={len(ssub)} (LOW N)")
            continue
        day_bull = ssub['day_bullish'].dropna().astype(float)
        cl = ssub['close_loc'].dropna()
        out.append(f"  {slabel}: N={len(ssub)}")
        if len(day_bull) > 0:
            mean_db = day_bull.mean()
            lo, hi = bootstrap_ci(day_bull.values)
            out.append(f"    Tagesrichtung = Gap-Richtung: {fmt_pct(mean_db, lo, hi)}")
        if len(cl) > 0:
            out.append(f"    Close Location Median: {cl.median():.2f}")

# ── Q4.3: Lunch Zone ──
out.append("\n" + "=" * 70)
out.append("FRAGE 4.3: Mittags-Lunch-Zone (11:30-13:30)")
out.append("=" * 70)

lunch_sub = df[df['lunch_narrower'].notna()]
out.append(f"\nLunch-Range < Morning-Range: {lunch_sub['lunch_narrower'].mean()*100:.0f}% (N={len(lunch_sub)})")
out.append(f"Median Lunch/Morning Range Ratio: {lunch_sub['lunch_range_vs_morning'].median():.2f}")

for gd in ['up', 'down']:
    label = 'GapUp' if gd == 'up' else 'GapDown'
    gsub = lunch_sub[lunch_sub['gap_dir'] == gd]
    out.append(f"\n  {label}:")

    # Morning high broken in lunch?
    mhb = gsub['morning_high_broken_lunch'].dropna().astype(float)
    mlb = gsub['morning_low_broken_lunch'].dropna().astype(float)
    out.append(f"    Morning-High in Lunch gebrochen: {mhb.mean()*100:.0f}% (N={len(mhb)})")
    out.append(f"    Morning-Low in Lunch gebrochen: {mlb.mean()*100:.0f}% (N={len(mlb)})")

    # Follow-through after lunch break
    lbhf = gsub['lunch_break_high_follow'].dropna().astype(float)
    lblf = gsub['lunch_break_low_follow'].dropna().astype(float)
    if len(lbhf) > 10:
        out.append(f"    Lunch-Break-High -> Follow-Through PM: {lbhf.mean()*100:.0f}% (N={len(lbhf)})")
    if len(lblf) > 10:
        out.append(f"    Lunch-Break-Low -> Follow-Through PM: {lblf.mean()*100:.0f}% (N={len(lblf)})")

# ── Q4.4: Power Hour ──
out.append("\n" + "=" * 70)
out.append("FRAGE 4.4: Power Hour (15:00-16:00)")
out.append("=" * 70)

ph_sub = df[df['ph_new_hod'].notna()]
for gd in ['up', 'down']:
    label = 'GapUp' if gd == 'up' else 'GapDown'
    gsub = ph_sub[ph_sub['gap_dir'] == gd]
    out.append(f"\n  {label} (N={len(gsub)}):")

    new_hod = gsub['ph_new_hod'].astype(float).values
    new_lod = gsub['ph_new_lod'].astype(float).values
    out.append(f"    Power Hour neues HOD: {np.mean(new_hod)*100:.0f}%")
    out.append(f"    Power Hour neues LOD: {np.mean(new_lod)*100:.0f}%")

    cl = gsub['close_loc'].dropna()
    out.append(f"    Close Location (0=Low, 1=High): Median={cl.median():.2f}, Mean={cl.mean():.2f}")

    # Close location distribution
    out.append(f"    Close Location Verteilung:")
    out.append(f"      < 0.25 (near LOD): {(cl < 0.25).mean()*100:.0f}%")
    out.append(f"      0.25-0.50: {((cl >= 0.25) & (cl < 0.50)).mean()*100:.0f}%")
    out.append(f"      0.50-0.75: {((cl >= 0.50) & (cl < 0.75)).mean()*100:.0f}%")
    out.append(f"      > 0.75 (near HOD): {(cl >= 0.75).mean()*100:.0f}%")

# ── Q4.5: Early Patterns ──
out.append("\n" + "=" * 70)
out.append("FRAGE 4.5: Fruehe Muster (vor 10:00)")
out.append("=" * 70)

for gd in ['up', 'down']:
    label = 'GapUp' if gd == 'up' else 'GapDown'
    gsub = df[df['gap_dir'] == gd]
    out.append(f"\n--- {label} ---")

    # First 5-min candle
    for pattern, col in [('Bullish', 'first_5m_bullish'), ('Bearish', 'first_5m_bearish'), ('Doji', 'first_5m_doji')]:
        psub = gsub[gsub[col] == True]
        if len(psub) < 20:
            out.append(f"  Erste 5min {pattern}: N={len(psub)} (LOW N)")
            continue
        day_bull = psub['day_bullish'].dropna().astype(float)
        cl = psub['close_loc'].dropna()
        if len(day_bull) > 0:
            mean_db = day_bull.mean()
            lo, hi = bootstrap_ci(day_bull.values)
            out.append(f"  Erste 5min {pattern}: Tagesrichtung=Gap -> {fmt_pct(mean_db, lo, hi)}, "
                       f"Close-Loc={cl.median():.2f}, N={len(psub)}")

    # First 3 bars all same direction
    f3u = gsub[gsub.get('first_3_all_up', pd.Series(dtype=bool)) == True] if 'first_3_all_up' in gsub.columns else pd.DataFrame()
    f3d = gsub[gsub.get('first_3_all_down', pd.Series(dtype=bool)) == True] if 'first_3_all_down' in gsub.columns else pd.DataFrame()

    if len(f3u) >= 20:
        day_bull = f3u['day_bullish'].dropna().astype(float)
        out.append(f"  Erste 3 Bars ALLE UP: Tagesrichtung=Gap -> {day_bull.mean()*100:.0f}%, N={len(f3u)}")
    if len(f3d) >= 20:
        day_bull = f3d['day_bullish'].dropna().astype(float)
        out.append(f"  Erste 3 Bars ALLE DOWN: Tagesrichtung=Gap -> {day_bull.mean()*100:.0f}%, N={len(f3d)}")

# ── Write output ────────────────────────────────────────────────────
report = "\n".join(out)
with open("results/cheat_cat4_time_structure.txt", "w", encoding="utf-8") as f:
    f.write(report)

print(report)
print(f"\nResults saved to results/cheat_cat4_time_structure.txt", file=sys.stderr)
