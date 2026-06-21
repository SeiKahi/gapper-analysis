###############################################################################
# BEHAVIORAL ANALYSIS v1 — Durchlauf 2.0
#
# PARADIGMENWECHSEL: Keine SL/Target-Simulation.
# Stattdessen: "Was passiert typischerweise wenn Bedingungen X,Y,Z?"
#
# Metriken: MFE, MAE, Time-to-Move, Conditional Cascading,
#           Reversal Rate, Gap Fill %, Directional Drift
#
# Regeln:
#   - Keine Signale vor 9:45 ET (erste 15min = Noise)
#   - Nur Haelfte 1 (2021-02-21 bis 2023-12-31)
#   - Slippage-Awareness: 0.005-0.01 ADR
#
# Run:
#   .\gapper_env\Scripts\python.exe scripts\behavioral_analysis_v1.py
###############################################################################

import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import glob
import warnings
from tqdm import tqdm
from collections import defaultdict

warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE_DIR = str(PROJECT_ROOT)
VWAP_DIR = os.path.join(BASE_DIR, 'data', 'vwap')
METADATA_PATH = os.path.join(BASE_DIR, 'data', 'metadata', 'metadata_master.parquet')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

DATA_START = '2021-02-21'
DATA_END = '2023-12-31'
TRAIN_END = '2022-12-31'

# No signals before 9:45 ET = bar index 15 (0-indexed from 09:30)
MIN_BAR_INDEX = 15  # 9:45 ET

# Time windows for MFE/MAE analysis (in bars = minutes)
TIME_WINDOWS = [15, 30, 60, 120]

# Move thresholds (in ADR multiples) for time-to-move
MOVE_THRESHOLDS = [0.10, 0.20, 0.50]

# Level cross scenarios
SCENARIOS = [
    # (name, gap_dir, level_col, trade_dir, next_level_col)
    ('GapUp_2s_Fade', 'up', 'upper_2std', 'short', 'upper_1std'),
    ('GapUp_1s_Fade', 'up', 'upper_1std', 'short', 'vwap'),
    ('GapUp_VWAP_Brk', 'up', 'vwap', 'short', 'lower_1std'),
    ('GapDn_2s_Fade', 'down', 'lower_2std', 'long', 'lower_1std'),
    ('GapDn_1s_Fade', 'down', 'lower_1std', 'long', 'vwap'),
    ('GapDn_VWAP_Brk', 'down', 'vwap', 'long', 'upper_1std'),
]


# ============================================================
# LOAD METADATA
# ============================================================
def load_metadata():
    meta = pd.read_parquet(METADATA_PATH)
    meta['date'] = pd.to_datetime(meta['date']).dt.strftime('%Y-%m-%d')
    meta = meta[(meta['date'] >= DATA_START) & (meta['date'] <= DATA_END)]
    meta = meta.dropna(subset=['adr_10'])
    meta = meta[meta['adr_10'] > 0]
    return meta


# ============================================================
# LOAD VWAP FILE
# ============================================================
def load_vwap(ticker, date_str):
    """Load a single VWAP parquet file."""
    path = os.path.join(VWAP_DIR, ticker, f"{date_str}.parquet")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_parquet(path)
        # Filter to regular trading hours only
        if 'time_et' in df.columns:
            df = df[(df['time_et'] >= '09:30') & (df['time_et'] <= '15:59')].reset_index(drop=True)
        # Drop rows where VWAP is NaN (first bar typically)
        df = df.dropna(subset=['vwap']).reset_index(drop=True)
        return df
    except:
        return None


# ============================================================
# DETECT LEVEL CROSSES (after 9:45 ET)
# ============================================================
def detect_crosses(vwap_df, level_col, trade_dir):
    """
    Find all bars where close crosses a VWAP level.
    Returns list of bar indices where the cross occurs.
    Only returns crosses after MIN_BAR_INDEX (9:45 ET).
    """
    crosses = []
    closes = vwap_df['close'].values
    levels = vwap_df[level_col].values

    for i in range(max(1, MIN_BAR_INDEX), len(closes)):
        if np.isnan(closes[i]) or np.isnan(closes[i-1]) or np.isnan(levels[i]):
            continue

        if trade_dir == 'short':
            # Fade down: close crosses from above to below the level
            if closes[i-1] >= levels[i-1] and closes[i] < levels[i]:
                crosses.append(i)
        else:  # long
            # Fade up: close crosses from below to above the level
            if closes[i-1] <= levels[i-1] and closes[i] > levels[i]:
                crosses.append(i)

    return crosses


# ============================================================
# COMPUTE BEHAVIORAL METRICS FOR A SINGLE CROSS EVENT
# ============================================================
def compute_cross_metrics(vwap_df, cross_bar, trade_dir, adr, next_level_col):
    """
    For a single cross event, compute:
    - MFE (max favorable excursion) at 15/30/60/120 min
    - MAE (max adverse excursion) at same windows
    - Time to move 0.1/0.2/0.5 ADR
    - Whether next level is reached
    - Reversal (returns to cross level within 10 bars)
    - Directional drift at each window
    """
    closes = vwap_df['close'].values
    entry_price = closes[cross_bar]
    n_bars = len(closes)

    if adr <= 0 or entry_price <= 0:
        return None

    is_long = (trade_dir == 'long')
    sign = 1.0 if is_long else -1.0

    result = {
        'cross_bar': cross_bar,
        'entry_price': entry_price,
    }

    # MFE/MAE for each time window
    for w in TIME_WINDOWS:
        end_bar = min(cross_bar + w, n_bars)
        if end_bar <= cross_bar:
            result[f'mfe_{w}'] = np.nan
            result[f'mae_{w}'] = np.nan
            result[f'drift_{w}'] = np.nan
            continue

        future_closes = closes[cross_bar+1:end_bar]
        if len(future_closes) == 0:
            result[f'mfe_{w}'] = np.nan
            result[f'mae_{w}'] = np.nan
            result[f'drift_{w}'] = np.nan
            continue

        moves = sign * (future_closes - entry_price) / adr
        result[f'mfe_{w}'] = np.max(moves)
        result[f'mae_{w}'] = np.min(moves)  # most negative = worst adverse
        result[f'drift_{w}'] = moves[-1]  # final position

    # Time to move thresholds
    for thr in MOVE_THRESHOLDS:
        end_bar = min(cross_bar + 240, n_bars)  # max 4h
        found = False
        for j in range(cross_bar + 1, end_bar):
            move = sign * (closes[j] - entry_price) / adr
            if move >= thr:
                result[f'ttm_{thr}'] = j - cross_bar
                found = True
                break
        if not found:
            result[f'ttm_{thr}'] = np.nan  # didn't reach threshold

    # Next level reached?
    next_levels = vwap_df[next_level_col].values
    end_bar = min(cross_bar + 120, n_bars)
    result['next_level_reached'] = False
    result['next_level_time'] = np.nan
    for j in range(cross_bar + 1, end_bar):
        if is_long:
            if closes[j] >= next_levels[j]:
                result['next_level_reached'] = True
                result['next_level_time'] = j - cross_bar
                break
        else:
            if closes[j] <= next_levels[j]:
                result['next_level_reached'] = True
                result['next_level_time'] = j - cross_bar
                break

    # Reversal: does price return to cross level within 10 bars?
    level_at_cross = vwap_df.iloc[cross_bar]
    result['reversal_10'] = False
    for j in range(cross_bar + 1, min(cross_bar + 11, n_bars)):
        if is_long:
            # Long trade: reversal = price goes back below the level
            move = (closes[j] - entry_price) / adr
            if move < -0.005:  # at least 0.5% ADR adverse = meaningful reversal
                result['reversal_10'] = True
                break
        else:
            move = (entry_price - closes[j]) / adr
            if move < -0.005:
                result['reversal_10'] = True
                break

    return result


# ============================================================
# MAIN ANALYSIS LOOP
# ============================================================
def main():
    print("Loading metadata...", file=sys.stderr)
    meta = load_metadata()
    print(f"  {len(meta)} gapper-days loaded (Half 1: {DATA_START} to {DATA_END})", file=sys.stderr)

    # Split into train/val
    meta['split'] = np.where(meta['date'] <= TRAIN_END, 'train', 'val')
    print(f"  Train: {(meta['split']=='train').sum()}, Val: {(meta['split']=='val').sum()}", file=sys.stderr)

    # Storage for all cross events
    all_events = []

    # Process each gapper day
    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Processing gappers", file=sys.stderr):
        ticker = row['ticker']
        date_str = row['date']
        adr = row['adr_10']
        gap_dir = row['gap_direction']
        gap_size_adr = row.get('gap_size_in_adr', np.nan)

        # Metadata features
        rvol = row.get('rvol_at_time_30min', np.nan)
        sector = row.get('sector', 'Unknown')
        vix = row.get('vix_level', np.nan)
        spy_ret = row.get('spy_return_day', np.nan)
        open_price = row.get('open_price', np.nan)

        vwap_df = load_vwap(ticker, date_str)
        if vwap_df is None or len(vwap_df) < 30:
            continue

        # Compute early range (first 15 bars = 09:30-09:45)
        early_closes = vwap_df['close'].values[:MIN_BAR_INDEX]
        if len(early_closes) >= 2:
            early_range_15 = (np.max(early_closes) - np.min(early_closes)) / adr
        else:
            early_range_15 = np.nan

        # Compute VWAP slope at 9:45 (slope over first 15 bars)
        vwaps = vwap_df['vwap'].values
        if len(vwaps) > MIN_BAR_INDEX and vwaps[0] > 0:
            vwap_slope_early = (vwaps[MIN_BAR_INDEX] - vwaps[0]) / vwaps[0]
        else:
            vwap_slope_early = np.nan

        # pct_adr_used at 9:45
        if len(early_closes) >= 2:
            pct_adr_used_945 = (np.max(early_closes) - np.min(early_closes)) / adr
        else:
            pct_adr_used_945 = np.nan

        # Check each scenario that matches this gapper's direction
        for scen_name, scen_gap_dir, level_col, trade_dir, next_level_col in SCENARIOS:
            if gap_dir != scen_gap_dir:
                continue
            if level_col not in vwap_df.columns or next_level_col not in vwap_df.columns:
                continue

            crosses = detect_crosses(vwap_df, level_col, trade_dir)

            # Only take FIRST cross per scenario per day (no multiple entries)
            if len(crosses) == 0:
                continue

            cross_bar = crosses[0]

            # Compute volume ratio at cross
            volumes = vwap_df['volume'].values
            if cross_bar >= 10:
                avg_vol = np.mean(volumes[cross_bar-10:cross_bar])
                vol_ratio = volumes[cross_bar] / avg_vol if avg_vol > 0 else 1.0
            else:
                vol_ratio = np.nan

            # Compute pct_adr_used at cross time
            up_to_cross = vwap_df['close'].values[:cross_bar+1]
            pct_adr_at_cross = (np.max(up_to_cross) - np.min(up_to_cross)) / adr if len(up_to_cross) >= 2 else np.nan

            metrics = compute_cross_metrics(vwap_df, cross_bar, trade_dir, adr, next_level_col)
            if metrics is None:
                continue

            event = {
                'ticker': ticker,
                'date': date_str,
                'split': row['split'],
                'gap_direction': gap_dir,
                'scenario': scen_name,
                'trade_dir': trade_dir,
                'adr': adr,
                'gap_size_adr': gap_size_adr,
                'rvol': rvol,
                'sector': sector,
                'vix_level': vix,
                'spy_return': spy_ret,
                'open_price': open_price,
                'early_range_15': early_range_15,
                'vwap_slope_early': vwap_slope_early,
                'pct_adr_used_945': pct_adr_used_945,
                'cross_bar': metrics['cross_bar'],
                'cross_time_min': metrics['cross_bar'],  # minutes since 9:30
                'entry_price': metrics['entry_price'],
                'volume_ratio_at_cross': vol_ratio,
                'pct_adr_at_cross': pct_adr_at_cross,
            }

            # Add MFE/MAE/drift for each window
            for w in TIME_WINDOWS:
                event[f'mfe_{w}'] = metrics.get(f'mfe_{w}', np.nan)
                event[f'mae_{w}'] = metrics.get(f'mae_{w}', np.nan)
                event[f'drift_{w}'] = metrics.get(f'drift_{w}', np.nan)

            # Time to move
            for thr in MOVE_THRESHOLDS:
                event[f'ttm_{thr}'] = metrics.get(f'ttm_{thr}', np.nan)

            event['next_level_reached'] = metrics['next_level_reached']
            event['next_level_time'] = metrics['next_level_time']
            event['reversal_10'] = metrics['reversal_10']

            all_events.append(event)

    # ============================================================
    # CONVERT TO DATAFRAME
    # ============================================================
    df = pd.DataFrame(all_events)
    print(f"\nTotal cross events: {len(df)}", file=sys.stderr)
    if len(df) == 0:
        print("ERROR: No cross events found! Check file paths and cross detection.", file=sys.stderr)
        return
    print(f"  Train: {(df['split']=='train').sum()}, Val: {(df['split']=='val').sum()}", file=sys.stderr)

    # Save raw events
    df.to_parquet(os.path.join(RESULTS_DIR, 'behavioral_events_v1.parquet'), index=False)
    print("Saved behavioral_events_v1.parquet", file=sys.stderr)

    # ============================================================
    # ANALYSIS: Write results to file
    # ============================================================
    output_path = os.path.join(RESULTS_DIR, 'behavioral_analysis_v1.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("BEHAVIORAL ANALYSIS v1 — Durchlauf 2.0\n")
        f.write("Keine SL/Target. Nur beobachtetes Verhalten nach Level-Cross.\n")
        f.write("Keine Signale vor 9:45 ET. Nur Haelfte 1 (2021-2023).\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Total Events: {len(df)}\n")
        f.write(f"Train: {(df['split']=='train').sum()}, Val: {(df['split']=='val').sum()}\n")
        f.write(f"Date Range: {df['date'].min()} to {df['date'].max()}\n\n")

        # ----- 1. SCENARIO OVERVIEW -----
        f.write("=" * 80 + "\n")
        f.write("1. SZENARIO-UEBERBLICK\n")
        f.write("=" * 80 + "\n\n")

        for split_name in ['train', 'val']:
            f.write(f"--- {split_name.upper()} ---\n")
            split_df = df[df['split'] == split_name]

            f.write(f"{'Scenario':<25} {'N':>6} {'MFE30_mean':>10} {'MFE30_med':>10} {'MAE30_mean':>10} "
                    f"{'Drift30_mean':>12} {'NextLvl%':>8} {'Rev10%':>7}\n")
            f.write("-" * 90 + "\n")

            for scen in sorted(df['scenario'].unique()):
                s = split_df[split_df['scenario'] == scen]
                if len(s) == 0:
                    continue
                n = len(s)
                mfe30 = s['mfe_30'].mean()
                mfe30_med = s['mfe_30'].median()
                mae30 = s['mae_30'].mean()
                drift30 = s['drift_30'].mean()
                next_pct = s['next_level_reached'].mean() * 100
                rev_pct = s['reversal_10'].mean() * 100
                f.write(f"{scen:<25} {n:>6} {mfe30:>10.4f} {mfe30_med:>10.4f} {mae30:>10.4f} "
                        f"{drift30:>12.4f} {next_pct:>7.1f}% {rev_pct:>6.1f}%\n")
            f.write("\n")

        # ----- 2. MFE/MAE DISTRIBUTIONS BY TIME WINDOW -----
        f.write("=" * 80 + "\n")
        f.write("2. MFE/MAE VERTEILUNGEN (in ADR) — Validation Set\n")
        f.write("=" * 80 + "\n\n")

        val = df[df['split'] == 'val']

        for scen in sorted(val['scenario'].unique()):
            s = val[val['scenario'] == scen]
            f.write(f"\n--- {scen} (N={len(s)}) ---\n")
            f.write(f"{'Window':>8} | {'MFE_mean':>9} {'MFE_p25':>8} {'MFE_p50':>8} {'MFE_p75':>8} | "
                    f"{'MAE_mean':>9} {'MAE_p25':>8} {'MAE_p50':>8} {'MAE_p75':>8} | {'Drift_mean':>10}\n")
            f.write("-" * 105 + "\n")

            for w in TIME_WINDOWS:
                mfe = s[f'mfe_{w}'].dropna()
                mae = s[f'mae_{w}'].dropna()
                drift = s[f'drift_{w}'].dropna()
                if len(mfe) == 0:
                    continue
                f.write(f"{w:>6}m | {mfe.mean():>9.4f} {mfe.quantile(0.25):>8.4f} {mfe.median():>8.4f} "
                        f"{mfe.quantile(0.75):>8.4f} | {mae.mean():>9.4f} {mae.quantile(0.25):>8.4f} "
                        f"{mae.median():>8.4f} {mae.quantile(0.75):>8.4f} | {drift.mean():>10.4f}\n")
            f.write("\n")

        # ----- 3. CONDITIONAL CASCADING -----
        f.write("=" * 80 + "\n")
        f.write("3. CONDITIONAL CASCADING — P(Next Level | Cross) — Val\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"{'Scenario':<25} {'N':>6} {'NextLvl%':>8} {'MedTime':>8} {'P25Time':>8} {'P75Time':>8}\n")
        f.write("-" * 65 + "\n")

        for scen in sorted(val['scenario'].unique()):
            s = val[val['scenario'] == scen]
            n = len(s)
            hit_pct = s['next_level_reached'].mean() * 100
            times = s.loc[s['next_level_reached'], 'next_level_time']
            med_t = times.median() if len(times) > 0 else np.nan
            p25_t = times.quantile(0.25) if len(times) > 0 else np.nan
            p75_t = times.quantile(0.75) if len(times) > 0 else np.nan
            f.write(f"{scen:<25} {n:>6} {hit_pct:>7.1f}% {med_t:>7.0f}m {p25_t:>7.0f}m {p75_t:>7.0f}m\n")
        f.write("\n")

        # ----- 4. TIME-TO-MOVE ANALYSIS -----
        f.write("=" * 80 + "\n")
        f.write("4. TIME-TO-MOVE (Minuten bis X ADR Bewegung) — Val\n")
        f.write("=" * 80 + "\n\n")

        for scen in sorted(val['scenario'].unique()):
            s = val[val['scenario'] == scen]
            f.write(f"\n--- {scen} (N={len(s)}) ---\n")
            f.write(f"{'Threshold':>10} {'%Reached':>9} {'MedTime':>8} {'P25':>6} {'P75':>6}\n")
            f.write("-" * 45 + "\n")

            for thr in MOVE_THRESHOLDS:
                col = f'ttm_{thr}'
                reached = s[col].notna()
                pct = reached.mean() * 100
                times = s.loc[reached, col]
                med = times.median() if len(times) > 0 else np.nan
                p25 = times.quantile(0.25) if len(times) > 0 else np.nan
                p75 = times.quantile(0.75) if len(times) > 0 else np.nan
                f.write(f"{thr:>9.2f}x {'':>1} {pct:>7.1f}% {med:>7.0f}m {p25:>5.0f}m {p75:>5.0f}m\n")
            f.write("\n")

        # ----- 5. TRADER HYPOTHESEN -----
        f.write("=" * 80 + "\n")
        f.write("5. TRADER-HYPOTHESEN TESTS — Val\n")
        f.write("=" * 80 + "\n\n")

        # H1: Morning Flush & Recovery (GapDn crosses after 9:45)
        f.write("--- H1: Morning vs Late Cross (GapDn) ---\n")
        gapdn = val[val['gap_direction'] == 'down']
        if len(gapdn) > 0:
            morning = gapdn[gapdn['cross_bar'] <= 30]  # 9:45-10:00
            midday = gapdn[(gapdn['cross_bar'] > 30) & (gapdn['cross_bar'] <= 90)]
            afternoon = gapdn[gapdn['cross_bar'] > 90]

            for label, subset in [('Morning 9:45-10:00', morning), ('Midday 10:00-11:00', midday), ('Afternoon >11:00', afternoon)]:
                if len(subset) == 0:
                    continue
                f.write(f"  {label}: N={len(subset)}, MFE30={subset['mfe_30'].mean():.4f}, "
                        f"MAE30={subset['mae_30'].mean():.4f}, Drift30={subset['drift_30'].mean():.4f}, "
                        f"NextLvl={subset['next_level_reached'].mean()*100:.1f}%\n")
        f.write("\n")

        # H3: Volume Confirms the Cross
        f.write("--- H3: Volume bei Cross (High Vol vs Low Vol) ---\n")
        vol_med = val['volume_ratio_at_cross'].median()
        if not np.isnan(vol_med):
            hi_vol = val[val['volume_ratio_at_cross'] >= 2.0]
            lo_vol = val[val['volume_ratio_at_cross'] < 1.0]
            mid_vol = val[(val['volume_ratio_at_cross'] >= 1.0) & (val['volume_ratio_at_cross'] < 2.0)]

            for label, subset in [('Low Vol (<1x)', lo_vol), ('Mid Vol (1-2x)', mid_vol), ('High Vol (>2x)', hi_vol)]:
                if len(subset) == 0:
                    continue
                f.write(f"  {label}: N={len(subset)}, MFE30={subset['mfe_30'].mean():.4f}, "
                        f"Drift30={subset['drift_30'].mean():.4f}, NextLvl={subset['next_level_reached'].mean()*100:.1f}%\n")
        f.write("\n")

        # H4: Gap Size Sweet Spot
        f.write("--- H4: Gap Size Buckets ---\n")
        gap_bins = [(0, 1.0, 'Small <1.0 ADR'), (1.0, 2.0, 'Medium 1-2 ADR'),
                    (2.0, 3.0, 'Large 2-3 ADR'), (3.0, 99, 'Extreme >3 ADR')]
        for lo, hi, label in gap_bins:
            subset = val[(val['gap_size_adr'] >= lo) & (val['gap_size_adr'] < hi)]
            if len(subset) < 20:
                continue
            f.write(f"  {label}: N={len(subset)}, MFE30={subset['mfe_30'].mean():.4f}, "
                    f"MAE30={subset['mae_30'].mean():.4f}, Drift30={subset['drift_30'].mean():.4f}, "
                    f"NextLvl={subset['next_level_reached'].mean()*100:.1f}%\n")
        f.write("\n")

        # H5: VWAP Slope
        f.write("--- H5: VWAP Slope (flach vs trending) ---\n")
        slope_med = val['vwap_slope_early'].abs().median()
        if not np.isnan(slope_med):
            flat = val[val['vwap_slope_early'].abs() <= slope_med]
            trending = val[val['vwap_slope_early'].abs() > slope_med]

            for label, subset in [('Flat VWAP', flat), ('Trending VWAP', trending)]:
                if len(subset) == 0:
                    continue
                f.write(f"  {label}: N={len(subset)}, MFE30={subset['mfe_30'].mean():.4f}, "
                        f"Drift30={subset['drift_30'].mean():.4f}, NextLvl={subset['next_level_reached'].mean()*100:.1f}%\n")
        f.write("\n")

        # H6: Sector Effects
        f.write("--- H6: Sector Effects (Top 5 by N) ---\n")
        sector_counts = val['sector'].value_counts()
        top_sectors = sector_counts.head(8).index
        for sec in top_sectors:
            subset = val[val['sector'] == sec]
            if len(subset) < 20:
                continue
            f.write(f"  {sec}: N={len(subset)}, MFE30={subset['mfe_30'].mean():.4f}, "
                    f"Drift30={subset['drift_30'].mean():.4f}, NextLvl={subset['next_level_reached'].mean()*100:.1f}%\n")
        f.write("\n")

        # H7: RVOL Threshold
        f.write("--- H7: RVOL Buckets ---\n")
        rvol_bins = [(0, 1.5, 'Low RVOL <1.5'), (1.5, 3.0, 'Mid RVOL 1.5-3'),
                     (3.0, 5.0, 'High RVOL 3-5'), (5.0, 999, 'Extreme RVOL >5')]
        for lo, hi, label in rvol_bins:
            subset = val[(val['rvol'] >= lo) & (val['rvol'] < hi)]
            if len(subset) < 20:
                continue
            f.write(f"  {label}: N={len(subset)}, MFE30={subset['mfe_30'].mean():.4f}, "
                    f"MAE30={subset['mae_30'].mean():.4f}, Drift30={subset['drift_30'].mean():.4f}, "
                    f"NextLvl={subset['next_level_reached'].mean()*100:.1f}%\n")
        f.write("\n")

        # ----- 6. %ADR USED AT CROSS -----
        f.write("=" * 80 + "\n")
        f.write("6. %ADR USED AT CROSS — Val\n")
        f.write("=" * 80 + "\n\n")

        adr_bins = [(0, 0.3, '<30%'), (0.3, 0.5, '30-50%'), (0.5, 0.75, '50-75%'), (0.75, 99, '>75%')]
        for lo, hi, label in adr_bins:
            subset = val[(val['pct_adr_at_cross'] >= lo) & (val['pct_adr_at_cross'] < hi)]
            if len(subset) < 20:
                continue
            f.write(f"  {label}: N={len(subset)}, MFE30={subset['mfe_30'].mean():.4f}, "
                    f"MAE30={subset['mae_30'].mean():.4f}, Drift30={subset['drift_30'].mean():.4f}, "
                    f"NextLvl={subset['next_level_reached'].mean()*100:.1f}%\n")
        f.write("\n")

        # ----- 7. TRAIN vs VAL COMPARISON -----
        f.write("=" * 80 + "\n")
        f.write("7. TRAIN vs VAL STABILITAET\n")
        f.write("=" * 80 + "\n\n")

        train = df[df['split'] == 'train']

        f.write(f"{'Scenario':<25} {'Train_N':>7} {'Val_N':>7} {'Train_MFE30':>11} {'Val_MFE30':>11} "
                f"{'Train_Drift30':>13} {'Val_Drift30':>13}\n")
        f.write("-" * 90 + "\n")

        for scen in sorted(df['scenario'].unique()):
            t = train[train['scenario'] == scen]
            v = val[val['scenario'] == scen]
            if len(t) == 0 or len(v) == 0:
                continue
            f.write(f"{scen:<25} {len(t):>7} {len(v):>7} {t['mfe_30'].mean():>11.4f} {v['mfe_30'].mean():>11.4f} "
                    f"{t['drift_30'].mean():>13.4f} {v['drift_30'].mean():>13.4f}\n")
        f.write("\n")

        # ----- 8. REVERSAL ANALYSIS -----
        f.write("=" * 80 + "\n")
        f.write("8. REVERSAL-ANALYSE (False Breakout Rate) — Val\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"{'Scenario':<25} {'N':>6} {'Rev10%':>7} {'MFE15_if_rev':>12} {'MFE15_if_ok':>12}\n")
        f.write("-" * 65 + "\n")

        for scen in sorted(val['scenario'].unique()):
            s = val[val['scenario'] == scen]
            rev = s[s['reversal_10'] == True]
            ok = s[s['reversal_10'] == False]
            rev_pct = len(rev) / len(s) * 100 if len(s) > 0 else 0
            mfe_rev = rev['mfe_15'].mean() if len(rev) > 0 else np.nan
            mfe_ok = ok['mfe_15'].mean() if len(ok) > 0 else np.nan
            f.write(f"{scen:<25} {len(s):>6} {rev_pct:>6.1f}% {mfe_rev:>12.4f} {mfe_ok:>12.4f}\n")
        f.write("\n")

        # ----- 9. SLIPPAGE IMPACT ON BEHAVIORAL METRICS -----
        f.write("=" * 80 + "\n")
        f.write("9. SLIPPAGE-IMPACT (0.005 / 0.010 ADR) — Val\n")
        f.write("=" * 80 + "\n\n")

        for slip in [0.005, 0.010]:
            f.write(f"--- Slippage = {slip:.3f} ADR ---\n")
            f.write(f"{'Scenario':<25} {'N':>6} {'MFE30_adj':>10} {'MFE30_raw':>10} {'Drift30_adj':>11} {'Drift30_raw':>11}\n")
            f.write("-" * 75 + "\n")

            for scen in sorted(val['scenario'].unique()):
                s = val[val['scenario'] == scen]
                if len(s) == 0:
                    continue
                # Slippage reduces MFE and drift by the slip amount
                mfe_adj = s['mfe_30'].mean() - slip
                mfe_raw = s['mfe_30'].mean()
                drift_adj = s['drift_30'].mean() - slip
                drift_raw = s['drift_30'].mean()
                f.write(f"{scen:<25} {len(s):>6} {mfe_adj:>10.4f} {mfe_raw:>10.4f} "
                        f"{drift_adj:>11.4f} {drift_raw:>11.4f}\n")
            f.write("\n")

        # ----- 10. BOOTSTRAP CI FOR KEY METRICS -----
        f.write("=" * 80 + "\n")
        f.write("10. BOOTSTRAP 95% CI — MFE30 und Drift30 — Val\n")
        f.write("=" * 80 + "\n\n")

        np.random.seed(42)
        n_boot = 5000

        for scen in sorted(val['scenario'].unique()):
            s = val[val['scenario'] == scen]
            if len(s) < 30:
                continue

            # Bootstrap MFE30
            mfe_vals = s['mfe_30'].dropna().values
            drift_vals = s['drift_30'].dropna().values

            mfe_boots = [np.mean(np.random.choice(mfe_vals, size=len(mfe_vals), replace=True)) for _ in range(n_boot)]
            drift_boots = [np.mean(np.random.choice(drift_vals, size=len(drift_vals), replace=True)) for _ in range(n_boot)]

            mfe_ci = np.percentile(mfe_boots, [2.5, 97.5])
            drift_ci = np.percentile(drift_boots, [2.5, 97.5])
            drift_p = np.mean([b <= 0 for b in drift_boots])

            f.write(f"{scen} (N={len(s)}):\n")
            f.write(f"  MFE30:  mean={np.mean(mfe_vals):.4f}, 95% CI [{mfe_ci[0]:.4f}, {mfe_ci[1]:.4f}]\n")
            f.write(f"  Drift30: mean={np.mean(drift_vals):.4f}, 95% CI [{drift_ci[0]:.4f}, {drift_ci[1]:.4f}], P(<=0)={drift_p:.4f}\n\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("ENDE BEHAVIORAL ANALYSIS v1\n")
        f.write("=" * 80 + "\n")

    print(f"\nErgebnisse geschrieben: {output_path}", file=sys.stderr)
    print("Done.", file=sys.stderr)


if __name__ == '__main__':
    main()
