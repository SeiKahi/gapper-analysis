"""
Cheat Sheet Category 1: VWAP als Tages-Regime
Questions 1.1-1.4: VWAP resistance/support persistence, breaks, retests, slope
"""
import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm

# ── helpers ──────────────────────────────────────────────────────────
def bootstrap_ci(data, n_boot=1000, ci=0.95):
    """Bootstrap CI for a proportion or mean."""
    if len(data) < 5:
        return np.nan, np.nan
    rng = np.random.default_rng(42)
    stats = np.array([np.mean(rng.choice(data, size=len(data), replace=True)) for _ in range(n_boot)])
    lo = np.percentile(stats, (1-ci)/2*100)
    hi = np.percentile(stats, (1+ci)/2*100)
    return lo, hi

def fmt_pct(val, ci_lo, ci_hi):
    return f"{val*100:.1f}% (CI: {ci_lo*100:.0f}-{ci_hi*100:.0f}%)"

def fmt_row(label, arr, n_label=True):
    """Format a result row: label, pct, CI, N."""
    if len(arr) == 0:
        return f"  {label}: N=0"
    mean = np.mean(arr)
    lo, hi = bootstrap_ci(arr)
    n = len(arr)
    return f"  {label}: {fmt_pct(mean, lo, hi)}, N={n}"

# ── load metadata (Halfte 1 only) ──────────────────────────────────
print("Loading metadata...", file=sys.stderr)
meta = pd.read_parquet("data/metadata/metadata_master.parquet")
meta = meta[meta['date'] <= '2023-12-31'].copy()
meta = meta[meta['adr_10'].notna()].copy()
meta['gap_size_in_adr'] = meta['gap_size_in_adr'].fillna(
    (meta['today_open'] - meta['prev_close']).abs() / meta['adr_10']
)
meta['adr_pct'] = meta['adr_10'] / meta['today_open'] * 100
print(f"Halfte 1: {len(meta)} gapper days", file=sys.stderr)

# ── buckets ─────────────────────────────────────────────────────────
def gap_size_bucket(v):
    if v < 1: return '<1x'
    if v < 2: return '1-2x'
    if v < 3: return '2-3x'
    return '>3x'

def adr_pct_bucket(v):
    if v < 3: return '<3%'
    if v < 5: return '3-5%'
    if v < 10: return '5-10%'
    return '>10%'

meta['gap_bucket'] = meta['gap_size_in_adr'].apply(gap_size_bucket)
meta['adr_pct_bucket'] = meta['adr_pct'].apply(adr_pct_bucket)

# ── process each gapper day ─────────────────────────────────────────
results_raw = []  # collect per-day features for all questions

for idx, row in tqdm(meta.iterrows(), total=len(meta), desc="Processing", file=sys.stderr):
    ticker = row['ticker']
    date = row['date']
    gap_dir = row['gap_direction']
    adr = row['adr_10']

    # Load vwap data
    vwap_path = f"data/vwap/{ticker}/{date}.parquet"
    raw_path = f"data/raw_1min/{ticker}/{date}.parquet"
    if not os.path.exists(vwap_path) or not os.path.exists(raw_path):
        continue

    try:
        vwap_df = pd.read_parquet(vwap_path)
        raw_df = pd.read_parquet(raw_path)
    except:
        continue

    # Filter RTH for raw
    if 'session' in raw_df.columns:
        raw_df = raw_df[raw_df['session'] == 'rth'].copy()

    if len(vwap_df) < 30 or len(raw_df) < 30 or adr <= 0:
        continue

    # Merge raw OHLC with vwap bands using time_et
    vwap_df = vwap_df.sort_values('time_et').reset_index(drop=True)
    raw_df = raw_df.sort_values('time_et').reset_index(drop=True)

    # Merge on time_et
    merged = pd.merge(raw_df[['time_et','open','high','low','close','volume','transactions']],
                       vwap_df[['time_et','vwap','upper_1std','lower_1std','upper_2std','lower_2std',
                                'upper_3std','lower_3std','z_score','std_dev']],
                       on='time_et', how='inner')

    if len(merged) < 30:
        continue

    merged = merged.sort_values('time_et').reset_index(drop=True)

    # Convert time_et to minutes from 09:30
    def time_to_min(t):
        if isinstance(t, str):
            parts = t.split(':')
            h, m = int(parts[0]), int(parts[1])
            return (h - 9) * 60 + (m - 30)
        return np.nan

    merged['min_from_open'] = merged['time_et'].apply(time_to_min)
    merged = merged[merged['min_from_open'] >= 0].copy()  # only RTH

    if len(merged) < 20:
        continue

    # VWAP data might have NaN in first few rows
    merged = merged.dropna(subset=['vwap']).reset_index(drop=True)
    if len(merged) < 20:
        continue

    # ── Helper: check if VWAP sustainably broken by time T ──
    def vwap_broken_by_time(direction, cutoff_min, n_bars=3):
        """
        For DownGap: check if price broke ABOVE vwap (close > vwap for n consecutive bars) by cutoff_min.
        For UpGap: check if price broke BELOW vwap.
        Returns: (broken: bool, break_min: int or None)
        """
        subset = merged[merged['min_from_open'] <= cutoff_min]
        if len(subset) < n_bars:
            return False, None

        if direction == 'down':
            above = (subset['close'] > subset['vwap']).astype(int).values
        else:  # up
            above = (subset['close'] < subset['vwap']).astype(int).values

        # Find n consecutive bars
        for i in range(len(above) - n_bars + 1):
            if all(above[i:i+n_bars]):
                return True, int(subset.iloc[i]['min_from_open'])
        return False, None

    def vwap_broken_strict(direction, cutoff_min, n_bars=3, min_dist_adr=0.1):
        """Strict: n bars + close at least min_dist_adr beyond VWAP."""
        subset = merged[merged['min_from_open'] <= cutoff_min]
        if len(subset) < n_bars:
            return False, None

        if direction == 'down':
            beyond = ((subset['close'] > subset['vwap']) &
                      ((subset['close'] - subset['vwap']) / adr >= min_dist_adr)).astype(int).values
        else:
            beyond = ((subset['close'] < subset['vwap']) &
                      ((subset['vwap'] - subset['close']) / adr >= min_dist_adr)).astype(int).values

        for i in range(len(beyond) - n_bars + 1):
            if all(beyond[i:i+n_bars]):
                return True, int(subset.iloc[i]['min_from_open'])
        return False, None

    # ── Check rest-of-day behavior ──
    def rest_of_day_levels(from_min):
        """After a cutoff time, check if various sigma levels are reached."""
        rest = merged[merged['min_from_open'] > from_min]
        if len(rest) < 5:
            return {}

        result = {}

        if gap_dir == 'down':
            # If VWAP is resistance, check if lower levels reached
            result['vwap_stays_resistance'] = all(rest['close'] <= rest['vwap'])
            # More lenient: allow max 2 bars above vwap
            bars_above = (rest['close'] > rest['vwap']).sum()
            result['vwap_mostly_resistance'] = bars_above <= 2

            # Check if sigma levels reached (using lows for targets)
            for sigma_name, col in [('-1s', 'lower_1std'), ('-2s', 'lower_2std'), ('-3s', 'lower_3std')]:
                if col in rest.columns and rest[col].notna().any():
                    reached = (rest['low'] <= rest[col]).any()
                    result[f'reached_{sigma_name}'] = reached
                    if reached:
                        idx_reached = rest[rest['low'] <= rest[col]].index[0]
                        pos = rest.index.get_loc(idx_reached)
                        result[f'time_to_{sigma_name}'] = int(rest.iloc[pos]['min_from_open'] - from_min)
        else:
            # UpGap: VWAP as support, check upper levels
            result['vwap_stays_support'] = all(rest['close'] >= rest['vwap'])
            bars_below = (rest['close'] < rest['vwap']).sum()
            result['vwap_mostly_support'] = bars_below <= 2

            for sigma_name, col in [('+1s', 'upper_1std'), ('+2s', 'upper_2std'), ('+3s', 'upper_3std')]:
                if col in rest.columns and rest[col].notna().any():
                    reached = (rest['high'] >= rest[col]).any()
                    result[f'reached_{sigma_name}'] = reached
                    if reached:
                        idx_reached = rest[rest['high'] >= rest[col]].index[0]
                        pos = rest.index.get_loc(idx_reached)
                        result[f'time_to_{sigma_name}'] = int(rest.iloc[pos]['min_from_open'] - from_min)

        return result

    # ── Q1.1/1.2: VWAP not broken by time T -> stays resistance/support? ──
    day_result = {
        'ticker': ticker, 'date': date, 'gap_dir': gap_dir,
        'adr': adr, 'gap_bucket': row['gap_bucket'],
        'adr_pct_bucket': row['adr_pct_bucket'],
        'gap_size_in_adr': row['gap_size_in_adr'],
    }

    for cutoff in [15, 30, 45, 60, 90]:  # minutes from open (9:45, 10:00, 10:15, 10:30, 11:00)
        for n_bars in [3, 5]:
            broken, break_min = vwap_broken_by_time(gap_dir, cutoff, n_bars)
            day_result[f'vwap_broken_{cutoff}m_{n_bars}bar'] = broken

            if not broken:
                rod = rest_of_day_levels(cutoff)
                for k, v in rod.items():
                    day_result[f'rod_{cutoff}m_{n_bars}bar_{k}'] = v

        # Strict definition
        broken_strict, _ = vwap_broken_strict(gap_dir, cutoff, 3, 0.1)
        day_result[f'vwap_broken_strict_{cutoff}m'] = broken_strict

    # ── Q1.3: If VWAP broken, does it retest and hold or follow through? ──
    # Find first sustainable break (3 bars), then look for retest
    broken_ever, break_min_ever = vwap_broken_by_time(gap_dir, 390, 3)  # full day
    day_result['vwap_ever_broken'] = broken_ever
    day_result['vwap_break_min'] = break_min_ever

    if broken_ever and break_min_ever is not None:
        after_break = merged[merged['min_from_open'] > break_min_ever + 5]  # 5min cooldown
        if len(after_break) > 5:
            if gap_dir == 'down':
                # Broke above VWAP - does it come back below?
                retest = (after_break['close'] <= after_break['vwap']).any()
                if retest:
                    retest_idx = after_break[after_break['close'] <= after_break['vwap']].index[0]
                    pos = after_break.index.get_loc(retest_idx)
                    # After retest, does it go back above (reject) or stay below?
                    post_retest = after_break.iloc[pos+1:pos+11]  # next 10 bars
                    if len(post_retest) > 3:
                        reject = (post_retest['close'] > post_retest['vwap']).sum() >= 3
                        day_result['vwap_retest_reject'] = reject
                        day_result['vwap_retest_follow'] = not reject
                    day_result['vwap_retest'] = True
                else:
                    day_result['vwap_retest'] = False
                    day_result['vwap_follow_through'] = True
            else:
                retest = (after_break['close'] >= after_break['vwap']).any()
                if retest:
                    retest_idx = after_break[after_break['close'] >= after_break['vwap']].index[0]
                    pos = after_break.index.get_loc(retest_idx)
                    post_retest = after_break.iloc[pos+1:pos+11]
                    if len(post_retest) > 3:
                        reject = (post_retest['close'] < post_retest['vwap']).sum() >= 3
                        day_result['vwap_retest_reject'] = reject
                        day_result['vwap_retest_follow'] = not reject
                    day_result['vwap_retest'] = True
                else:
                    day_result['vwap_retest'] = False
                    day_result['vwap_follow_through'] = True

    # ── Q1.4: VWAP Slope ──
    # Compute VWAP slope: change in VWAP over first 60min normalized by ADR
    first_60 = merged[merged['min_from_open'] <= 60]
    if len(first_60) >= 10:
        vwap_vals = first_60['vwap'].dropna()
        if len(vwap_vals) >= 10:
            slope = (vwap_vals.iloc[-1] - vwap_vals.iloc[0]) / adr
            day_result['vwap_slope_60m'] = slope
            if abs(slope) < 0.05:
                day_result['vwap_slope_cat'] = 'flat'
            elif slope > 0:
                day_result['vwap_slope_cat'] = 'rising'
            else:
                day_result['vwap_slope_cat'] = 'falling'

    # ── BONUS: Close location (for later use) ──
    day_range = merged['high'].max() - merged['low'].min()
    if day_range > 0:
        day_result['close_location'] = (merged['close'].iloc[-1] - merged['low'].min()) / day_range

    results_raw.append(day_result)

# ── Convert to DataFrame ──────────────────────────────────────────
print(f"\nProcessed {len(results_raw)} days", file=sys.stderr)
df = pd.DataFrame(results_raw)
df.to_parquet("results/cheat_cat1_raw.parquet", index=False)

# ── Generate Report ─────────────────────────────────────────────────
out = []
out.append("=" * 80)
out.append("CHEAT SHEET — KATEGORIE 1: VWAP ALS TAGES-REGIME")
out.append("=" * 80)
out.append(f"Datenbasis: Halfte 1 (bis 2023-12-31), N={len(df)} Gapper-Tage")
out.append(f"  GapUp: {(df['gap_dir']=='up').sum()}, GapDown: {(df['gap_dir']=='down').sum()}")
out.append("")

# ── Q1.1: DownGap — VWAP as resistance ──
out.append("=" * 70)
out.append("FRAGE 1.1: VWAP als Widerstand bei DownGap")
out.append("=" * 70)

for cutoff, time_label in [(15, '9:45'), (30, '10:00'), (45, '10:15'), (60, '10:30'), (90, '11:00')]:
    for n_bars in [3, 5]:
        col_broken = f'vwap_broken_{cutoff}m_{n_bars}bar'
        sub = df[(df['gap_dir'] == 'down') & (df[col_broken].notna())]
        not_broken = sub[sub[col_broken] == False]

        out.append(f"\n--- DownGap, VWAP nicht gebrochen bis {time_label} ({n_bars}-Bar-Definition) ---")
        out.append(f"  Gesamt DownGaps: {len(sub)}, davon VWAP nicht gebrochen: {len(not_broken)} ({len(not_broken)/max(len(sub),1)*100:.0f}%)")

        if len(not_broken) < 10:
            out.append("  LOW N — uebersprungen")
            continue

        # VWAP stays resistance (strict: ALL bars below)
        col_strict = f'rod_{cutoff}m_{n_bars}bar_vwap_stays_resistance'
        col_mostly = f'rod_{cutoff}m_{n_bars}bar_vwap_mostly_resistance'

        if col_strict in not_broken.columns:
            vals_strict = not_broken[col_strict].dropna()
            if len(vals_strict) > 0:
                out.append(fmt_row("VWAP bleibt STRIKT Widerstand (0 Bars drueber)", vals_strict.astype(float)))

        if col_mostly in not_broken.columns:
            vals_mostly = not_broken[col_mostly].dropna()
            if len(vals_mostly) > 0:
                out.append(fmt_row("VWAP bleibt FAST Widerstand (max 2 Bars drueber)", vals_mostly.astype(float)))

        # Sigma levels reached
        for sigma in ['-1s', '-2s', '-3s']:
            col_reached = f'rod_{cutoff}m_{n_bars}bar_reached_{sigma}'
            col_time = f'rod_{cutoff}m_{n_bars}bar_time_to_{sigma}'
            if col_reached in not_broken.columns:
                vals = not_broken[col_reached].dropna()
                if len(vals) > 0:
                    pct = np.mean(vals)
                    lo, hi = bootstrap_ci(vals.astype(float).values)
                    time_vals = not_broken[col_time].dropna() if col_time in not_broken.columns else pd.Series(dtype=float)
                    med_time = f", Median {time_vals.median():.0f}min" if len(time_vals) > 0 else ""
                    out.append(f"  Erreicht {sigma}: {fmt_pct(pct, lo, hi)}{med_time}")

        # Breakdown by gap size (only for 3-bar, 10:00)
        if n_bars == 3 and cutoff == 30:
            out.append(f"\n  Breakdown nach Gap-Groesse (3-Bar, 10:00):")
            for bucket in ['<1x', '1-2x', '2-3x', '>3x']:
                bsub = not_broken[not_broken['gap_bucket'] == bucket]
                if len(bsub) < 10:
                    out.append(f"    {bucket}: N={len(bsub)} (LOW N)")
                    continue
                if col_mostly in bsub.columns:
                    vals = bsub[col_mostly].dropna()
                    if len(vals) > 0:
                        out.append(f"    {bucket}: VWAP Widerstand={np.mean(vals)*100:.0f}%, N={len(vals)}")

# ── Q1.2: UpGap — VWAP as support ──
out.append("\n" + "=" * 70)
out.append("FRAGE 1.2: VWAP als Support bei UpGap")
out.append("=" * 70)

for cutoff, time_label in [(15, '9:45'), (30, '10:00'), (45, '10:15'), (60, '10:30'), (90, '11:00')]:
    for n_bars in [3, 5]:
        col_broken = f'vwap_broken_{cutoff}m_{n_bars}bar'
        sub = df[(df['gap_dir'] == 'up') & (df[col_broken].notna())]
        not_broken = sub[sub[col_broken] == False]

        out.append(f"\n--- UpGap, VWAP nicht gebrochen bis {time_label} ({n_bars}-Bar-Definition) ---")
        out.append(f"  Gesamt UpGaps: {len(sub)}, davon VWAP nicht gebrochen: {len(not_broken)} ({len(not_broken)/max(len(sub),1)*100:.0f}%)")

        if len(not_broken) < 10:
            out.append("  LOW N — uebersprungen")
            continue

        col_strict = f'rod_{cutoff}m_{n_bars}bar_vwap_stays_support'
        col_mostly = f'rod_{cutoff}m_{n_bars}bar_vwap_mostly_support'

        if col_strict in not_broken.columns:
            vals_strict = not_broken[col_strict].dropna()
            if len(vals_strict) > 0:
                out.append(fmt_row("VWAP bleibt STRIKT Support (0 Bars drunter)", vals_strict.astype(float)))

        if col_mostly in not_broken.columns:
            vals_mostly = not_broken[col_mostly].dropna()
            if len(vals_mostly) > 0:
                out.append(fmt_row("VWAP bleibt FAST Support (max 2 Bars drunter)", vals_mostly.astype(float)))

        for sigma in ['+1s', '+2s', '+3s']:
            col_reached = f'rod_{cutoff}m_{n_bars}bar_reached_{sigma}'
            col_time = f'rod_{cutoff}m_{n_bars}bar_time_to_{sigma}'
            if col_reached in not_broken.columns:
                vals = not_broken[col_reached].dropna()
                if len(vals) > 0:
                    pct = np.mean(vals)
                    lo, hi = bootstrap_ci(vals.astype(float).values)
                    time_vals = not_broken[col_time].dropna() if col_time in not_broken.columns else pd.Series(dtype=float)
                    med_time = f", Median {time_vals.median():.0f}min" if len(time_vals) > 0 else ""
                    out.append(f"  Erreicht {sigma}: {fmt_pct(pct, lo, hi)}{med_time}")

        if n_bars == 3 and cutoff == 30:
            out.append(f"\n  Breakdown nach Gap-Groesse (3-Bar, 10:00):")
            for bucket in ['<1x', '1-2x', '2-3x', '>3x']:
                bsub = not_broken[not_broken['gap_bucket'] == bucket]
                if len(bsub) < 10:
                    out.append(f"    {bucket}: N={len(bsub)} (LOW N)")
                    continue
                if col_mostly in bsub.columns:
                    vals = bsub[col_mostly].dropna()
                    if len(vals) > 0:
                        out.append(f"    {bucket}: VWAP Support={np.mean(vals)*100:.0f}%, N={len(vals)}")

# ── Q1.3: VWAP Break Behavior ──
out.append("\n" + "=" * 70)
out.append("FRAGE 1.3: Verhalten nach VWAP-Break (3-Bar Definition)")
out.append("=" * 70)

for gd in ['down', 'up']:
    label = 'DownGap' if gd == 'down' else 'UpGap'
    broken_sub = df[(df['gap_dir'] == gd) & (df['vwap_ever_broken'] == True)]
    out.append(f"\n--- {label}: VWAP irgendwann nachhaltig gebrochen ---")
    out.append(f"  N={len(broken_sub)} (von {(df['gap_dir']==gd).sum()} total)")

    if len(broken_sub) < 20:
        out.append("  LOW N")
        continue

    # Median break time
    break_times = broken_sub['vwap_break_min'].dropna()
    if len(break_times) > 0:
        out.append(f"  Median Break-Zeitpunkt: {break_times.median():.0f}min nach Open")

    # Retest
    retest_vals = broken_sub['vwap_retest'].dropna()
    if len(retest_vals) > 0:
        out.append(fmt_row("Preis kommt zurueck zum VWAP (Retest)", retest_vals.astype(float)))

    # Retest -> Reject vs Follow-Through
    retest_sub = broken_sub[broken_sub['vwap_retest'] == True]
    if len(retest_sub) > 10:
        reject_vals = retest_sub['vwap_retest_reject'].dropna()
        if len(reject_vals) > 0:
            out.append(fmt_row("  Retest -> VWAP haelt (Reject, weiter in Break-Richtung)", reject_vals.astype(float)))
        follow_vals = retest_sub['vwap_retest_follow'].dropna()
        if len(follow_vals) > 0:
            out.append(fmt_row("  Retest -> VWAP bricht erneut (Follow-Through gescheitert)", follow_vals.astype(float)))

# ── Q1.4: VWAP Slope ──
out.append("\n" + "=" * 70)
out.append("FRAGE 1.4: VWAP-Slope Einfluss auf Regime")
out.append("=" * 70)

slope_df = df[df['vwap_slope_cat'].notna()]
for gd in ['down', 'up']:
    label = 'DownGap' if gd == 'down' else 'UpGap'
    out.append(f"\n--- {label} ---")
    gsub = slope_df[slope_df['gap_dir'] == gd]

    for slope_cat in ['flat', 'rising', 'falling']:
        ssub = gsub[gsub['vwap_slope_cat'] == slope_cat]
        out.append(f"  VWAP {slope_cat}: N={len(ssub)}")

        if len(ssub) < 20:
            out.append("    LOW N")
            continue

        # For DownGap: VWAP not broken at 10:00 -> how often stays resistance?
        col_broken = 'vwap_broken_30m_3bar'
        col_resist = 'rod_30m_3bar_vwap_mostly_resistance' if gd == 'down' else 'rod_30m_3bar_vwap_mostly_support'

        not_br = ssub[ssub[col_broken] == False]
        if len(not_br) > 10 and col_resist in not_br.columns:
            vals = not_br[col_resist].dropna()
            if len(vals) > 0:
                label_str = "Widerstand" if gd == 'down' else "Support"
                out.append(f"    VWAP nicht gebrochen bis 10:00 -> bleibt {label_str}: {np.mean(vals)*100:.0f}%, N={len(vals)}")

        # Close location
        cl = ssub['close_location'].dropna()
        if len(cl) > 10:
            out.append(f"    Median Close Location (0=Low, 1=High): {cl.median():.2f}")

# ── Write output ────────────────────────────────────────────────────
report = "\n".join(out)
with open("results/cheat_cat1_vwap_regime.txt", "w", encoding="utf-8") as f:
    f.write(report)

print(report)
print(f"\nResults saved to results/cheat_cat1_vwap_regime.txt", file=sys.stderr)
print(f"Raw data saved to results/cheat_cat1_raw.parquet", file=sys.stderr)
