"""
Cheat Sheet Category 2: Level-Respekt und Preis-Gedaechtnis (OPTIMIZED)
Vectorized approach: detect level touches per-day using pandas operations
"""
import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm

def bootstrap_ci(data, n_boot=1000, ci=0.95):
    if len(data) < 5:
        return np.nan, np.nan
    rng = np.random.default_rng(42)
    stats = np.array([np.mean(rng.choice(data, size=len(data), replace=True)) for _ in range(n_boot)])
    return np.percentile(stats, (1-ci)/2*100), np.percentile(stats, (1+ci)/2*100)

def fmt_pct(val, ci_lo, ci_hi):
    return f"{val*100:.1f}% (CI: {ci_lo*100:.0f}-{ci_hi*100:.0f}%)"

# Load metadata
print("Loading metadata...", file=sys.stderr)
meta = pd.read_parquet("data/metadata/metadata_master.parquet")
meta = meta[(meta['date'] <= '2023-12-31') & (meta['adr_10'].notna())].copy()
print(f"Halfte 1: {len(meta)}", file=sys.stderr)

LEVELS = ['vwap', 'upper_1std', 'lower_1std', 'upper_2std', 'lower_2std', 'upper_3std', 'lower_3std']
LEVEL_NAMES = {'vwap': 'VWAP', 'upper_1std': '+1s', 'lower_1std': '-1s',
               'upper_2std': '+2s', 'lower_2std': '-2s', 'upper_3std': '+3s', 'lower_3std': '-3s'}

# Simplified approach: For each day, find the FIRST touch of each level
# and track what happens next (hold/break, time to next level)
touch_events = []
multi_touch_data = []

for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Processing", file=sys.stderr):
    ticker = row['ticker']
    date = row['date']
    gap_dir = row['gap_direction']
    adr = row['adr_10']

    vwap_path = f"data/vwap/{ticker}/{date}.parquet"
    raw_path = f"data/raw_1min/{ticker}/{date}.parquet"
    if not os.path.exists(vwap_path) or not os.path.exists(raw_path):
        continue

    try:
        vwap_df = pd.read_parquet(vwap_path)
        raw_df = pd.read_parquet(raw_path)
    except:
        continue

    if 'session' in raw_df.columns:
        raw_df = raw_df[raw_df['session'] == 'rth'].copy()

    if len(vwap_df) < 30 or len(raw_df) < 30 or adr <= 0:
        continue

    # Merge
    m = pd.merge(
        raw_df[['time_et', 'open', 'high', 'low', 'close', 'volume', 'transactions']],
        vwap_df[['time_et', 'vwap', 'upper_1std', 'lower_1std', 'upper_2std', 'lower_2std',
                  'upper_3std', 'lower_3std']],
        on='time_et', how='inner'
    ).sort_values('time_et').reset_index(drop=True)

    if len(m) < 30:
        continue

    # Time in minutes
    def t2m(t):
        if isinstance(t, str):
            p = t.split(':')
            return (int(p[0]) - 9) * 60 + (int(p[1]) - 30)
        return -1
    m['mfo'] = m['time_et'].apply(t2m)
    m = m[m['mfo'] >= 0].reset_index(drop=True)
    m = m.dropna(subset=['vwap']).reset_index(drop=True)

    if len(m) < 20:
        continue

    day_avg_vol = m['volume'].mean()

    # Vectorized: for each level, find where price touches from above/below
    for level_col in ['vwap', 'upper_1std', 'lower_1std', 'upper_2std', 'lower_2std']:
        if level_col not in m.columns or m[level_col].isna().all():
            continue

        lname = LEVEL_NAMES[level_col]
        lvals = m[level_col].values
        closes = m['close'].values
        highs = m['high'].values
        lows = m['low'].values
        vols = m['volume'].values
        mfo = m['mfo'].values
        n = len(m)

        # Track touches per direction
        touch_count_support = 0
        touch_count_resist = 0
        last_touch_bar = -10

        for i in range(2, n - 3):
            if i - last_touch_bar < 5:  # cooldown
                continue

            lv = lvals[i]
            if np.isnan(lv):
                continue

            # Touch from above (support test): low dips to level, close stays above
            is_support_touch = (lows[i] <= lv * 1.002 and closes[i] > lv and closes[i-1] > lv)
            # Touch from below (resistance test): high reaches level, close stays below
            is_resist_touch = (highs[i] >= lv * 0.998 and closes[i] < lv and closes[i-1] < lv)

            if not is_support_touch and not is_resist_touch:
                continue

            last_touch_bar = i
            touch_dir = 'support' if is_support_touch else 'resistance'

            if touch_dir == 'support':
                touch_count_support += 1
                touch_num = touch_count_support
                rejection_price = lows[i]
            else:
                touch_count_resist += 1
                touch_num = touch_count_resist
                rejection_price = highs[i]

            vol_ratio = vols[i] / max(day_avg_vol, 1)
            bar_range = (highs[i] - lows[i]) / adr

            # Check 15/30/60 bar windows
            for window, wname in [(15, '15min'), (30, '30min'), (60, '60min')]:
                end = min(i + window, n)
                future_close = closes[i+1:end]
                future_high = highs[i+1:end]
                future_low = lows[i+1:end]

                if len(future_close) < 3:
                    continue

                if touch_dir == 'support':
                    held = not np.any(future_close < rejection_price)
                else:
                    held = not np.any(future_close > rejection_price)

                touch_events.append({
                    'gap_dir': gap_dir, 'level': lname, 'touch_dir': touch_dir,
                    'window': wname, 'held': held,
                    'vol_ratio': vol_ratio, 'bar_range_adr': bar_range,
                    'touch_num': touch_num, 'min_from_open': mfo[i],
                })

            # Multi-touch: track break rate by touch number
            # Look ahead 10 bars to see if level breaks
            future_10 = closes[i+1:min(i+11, n)]
            if len(future_10) >= 3:
                if touch_dir == 'support':
                    broke = np.any(future_10 < lv)
                else:
                    broke = np.any(future_10 > lv)

                multi_touch_data.append({
                    'gap_dir': gap_dir, 'level': lname, 'touch_dir': touch_dir,
                    'touch_num': touch_num, 'broke': broke,
                })

print(f"\nTouch events: {len(touch_events)}", file=sys.stderr)
print(f"Multi-touch events: {len(multi_touch_data)}", file=sys.stderr)

touch_df = pd.DataFrame(touch_events)
multi_df = pd.DataFrame(multi_touch_data)

# Save
touch_df.to_parquet("results/cheat_cat2_touches.parquet", index=False)

# Generate Report
out = []
out.append("=" * 80)
out.append("CHEAT SHEET -- KATEGORIE 2: LEVEL-RESPEKT UND PREIS-GEDAECHTNIS")
out.append("=" * 80)
out.append(f"Datenbasis: {len(touch_df)} Level-Touch Events")
out.append("")

# Q2.1: How often does the fixed rejection price hold?
out.append("=" * 70)
out.append("FRAGE 2.1: Fixierter Ablehnungspreis -- Haelt er beim naechsten Anlauf?")
out.append("=" * 70)

for window in ['15min', '30min', '60min']:
    out.append(f"\n--- Zeitfenster: {window} ---")
    wsub = touch_df[touch_df['window'] == window]

    for level in ['VWAP', '-1s', '+1s', '-2s', '+2s']:
        for tdir in ['support', 'resistance']:
            lsub = wsub[(wsub['level'] == level) & (wsub['touch_dir'] == tdir)]
            if len(lsub) < 30:
                continue
            held = lsub['held'].astype(float).values
            mean_h = np.mean(held)
            lo, hi = bootstrap_ci(held)
            out.append(f"  {level} als {tdir}: Haelt={fmt_pct(mean_h, lo, hi)}, N={len(lsub)}")

    # Q2.1 by gap direction (only for 30min window to keep output manageable)
    if window == '30min':
        out.append(f"\n  Breakdown nach Gap-Richtung ({window}):")
        for gd in ['up', 'down']:
            gdl = 'GapUp' if gd == 'up' else 'GapDown'
            gsub = wsub[wsub['gap_dir'] == gd]
            for level in ['VWAP', '-1s', '+1s']:
                for tdir in ['support', 'resistance']:
                    lsub = gsub[(gsub['level'] == level) & (gsub['touch_dir'] == tdir)]
                    if len(lsub) < 20:
                        continue
                    held = lsub['held'].astype(float).values
                    mean_h = np.mean(held)
                    out.append(f"    {gdl} {level} {tdir}: Haelt={mean_h*100:.0f}%, N={len(lsub)}")

# Q2.3: Multi-touch -- break rate by touch number
out.append("\n\n" + "=" * 70)
out.append("FRAGE 2.3: Multi-Touch -- Bricht Level eher beim 2., 3., N-ten Anlauf?")
out.append("=" * 70)

if len(multi_df) > 0:
    for level in ['VWAP', '-1s', '+1s', '-2s', '+2s']:
        lsub = multi_df[multi_df['level'] == level]
        if len(lsub) < 20:
            continue
        out.append(f"\n  Level: {level}")
        for touch_n in sorted(lsub['touch_num'].unique()):
            if touch_n > 5:  # cap at 5
                break
            nsub = lsub[lsub['touch_num'] == touch_n]
            if len(nsub) < 10:
                n_str = " (LOW N)" if len(nsub) < 50 else ""
                out.append(f"    Touch #{touch_n}: Break={nsub['broke'].mean()*100:.0f}%, N={len(nsub)}{n_str}")
                continue
            break_rate = nsub['broke'].astype(float).values
            mean_br = np.mean(break_rate)
            lo, hi = bootstrap_ci(break_rate)
            out.append(f"    Touch #{touch_n}: Break-Rate={fmt_pct(mean_br, lo, hi)}, N={len(nsub)}")

# Q2.4: Volume at level touch
out.append("\n\n" + "=" * 70)
out.append("FRAGE 2.4: Volumen bei Level-Touch -- Absorption vs Breakout")
out.append("=" * 70)

wsub = touch_df[touch_df['window'] == '30min']

for level in ['VWAP', '-1s', '+1s', '-2s', '+2s']:
    lsub = wsub[wsub['level'] == level]
    if len(lsub) < 50:
        continue

    out.append(f"\n  Level: {level}")

    # High vol + small bar (absorption)
    absorption = lsub[(lsub['vol_ratio'] > 2.0) & (lsub['bar_range_adr'] < 0.05)]
    if len(absorption) >= 10:
        held = absorption['held'].astype(float).values
        mean_h = np.mean(held)
        lo, hi = bootstrap_ci(held)
        out.append(f"    Absorption (HiVol+kleine Bar): Haelt={fmt_pct(mean_h, lo, hi)}, N={len(absorption)}")
    else:
        out.append(f"    Absorption: N={len(absorption)} (LOW N)")

    # High vol + large bar
    breakout = lsub[(lsub['vol_ratio'] > 2.0) & (lsub['bar_range_adr'] > 0.1)]
    if len(breakout) >= 10:
        held = breakout['held'].astype(float).values
        mean_h = np.mean(held)
        lo, hi = bootstrap_ci(held)
        out.append(f"    Breakout-Versuch (HiVol+grosse Bar): Haelt={fmt_pct(mean_h, lo, hi)}, N={len(breakout)}")
    else:
        out.append(f"    Breakout-Versuch: N={len(breakout)} (LOW N)")

    # Low vol
    lowvol = lsub[lsub['vol_ratio'] < 0.5]
    if len(lowvol) >= 10:
        held = lowvol['held'].astype(float).values
        mean_h = np.mean(held)
        lo, hi = bootstrap_ci(held)
        out.append(f"    Low-Vol Touch: Haelt={fmt_pct(mean_h, lo, hi)}, N={len(lowvol)}")

    # Normal vol for comparison
    normal = lsub[(lsub['vol_ratio'] >= 0.5) & (lsub['vol_ratio'] <= 2.0)]
    if len(normal) >= 10:
        held = normal['held'].astype(float).values
        out.append(f"    Normal-Vol Touch: Haelt={np.mean(held)*100:.0f}%, N={len(normal)}")

# Q2.2: Time of day for level tests
out.append("\n\n" + "=" * 70)
out.append("BONUS 2.5: Level-Respekt nach Tageszeit")
out.append("=" * 70)

wsub = touch_df[touch_df['window'] == '30min']
for period, lo_m, hi_m in [('Morning (0-60min)', 0, 60), ('Midday (60-180min)', 60, 180), ('Afternoon (180-390min)', 180, 390)]:
    psub = wsub[(wsub['min_from_open'] >= lo_m) & (wsub['min_from_open'] < hi_m)]
    if len(psub) < 50:
        continue
    held = psub['held'].astype(float).values
    mean_h = np.mean(held)
    lo, hi = bootstrap_ci(held)
    out.append(f"  {period}: Haelt={fmt_pct(mean_h, lo, hi)}, N={len(psub)}")

report = "\n".join(out)
with open("results/cheat_cat2_level_respect.txt", "w", encoding="utf-8") as f:
    f.write(report)

print(report)
print(f"\nResults saved to results/cheat_cat2_level_respect.txt", file=sys.stderr)
