"""
D13 Late-Day Entry Strategy for Gap Stocks
==========================================
Entry: Last 90 minutes of gap day (RTH close)
Hold: Overnight / Multi-day Swing
PnL: Based on next-day open, next-day close, multi-day closes

Phases:
  1. Basic PnL measurement (IS)
  2. VWAP-based entry conditions (IS)
  3. Multi-parameter combinations (IS)
  4. Optimal holding period analysis
  5. OOS validation with bootstrap
  6. Swing-trade with trailing stop
"""

import pandas as pd
import numpy as np
import os
import glob
import sys
import warnings
from collections import defaultdict
import time

warnings.filterwarnings('ignore')

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASE = str(PROJECT_ROOT)
META_PATH = os.path.join(BASE, 'data/metadata/metadata_v9.parquet')
RAW_1MIN_DIR = os.path.join(BASE, 'data/raw_1min')
VWAP_DIR = os.path.join(BASE, 'data/vwap')
OUTPUT_PATH = os.path.join(BASE, 'd13_team_analysis/results/d13_lateday_results.txt')

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_next_trading_day_bars(ticker, current_date_str):
    """Find next trading day after current_date and load its bars."""
    ticker_dir = os.path.join(RAW_1MIN_DIR, ticker)
    if not os.path.isdir(ticker_dir):
        return None, None
    files = sorted(glob.glob(os.path.join(ticker_dir, '*.parquet')))
    dates = [os.path.basename(f).replace('.parquet', '') for f in files]
    try:
        idx = dates.index(current_date_str)
        if idx + 1 < len(dates):
            next_date = dates[idx + 1]
            bars = pd.read_parquet(files[idx + 1])
            rth = bars[bars['session'] == 'rth'] if 'session' in bars.columns else bars
            return next_date, rth
    except ValueError:
        pass
    return None, None


def get_multi_day_bars(ticker, current_date_str, num_days=5):
    """Load bars for the next num_days trading days."""
    ticker_dir = os.path.join(RAW_1MIN_DIR, ticker)
    if not os.path.isdir(ticker_dir):
        return []
    files = sorted(glob.glob(os.path.join(ticker_dir, '*.parquet')))
    dates = [os.path.basename(f).replace('.parquet', '') for f in files]
    try:
        idx = dates.index(current_date_str)
    except ValueError:
        return []
    result = []
    for i in range(1, num_days + 1):
        if idx + i < len(files):
            bars = pd.read_parquet(files[idx + i])
            rth = bars[bars['session'] == 'rth'] if 'session' in bars.columns else bars
            result.append((dates[idx + i], rth))
    return result


def get_gap_day_bars(ticker, date_str):
    """Load gap day bars."""
    fpath = os.path.join(RAW_1MIN_DIR, ticker, f'{date_str}.parquet')
    if not os.path.isfile(fpath):
        return None
    bars = pd.read_parquet(fpath)
    rth = bars[bars['session'] == 'rth'] if 'session' in bars.columns else bars
    return rth


def get_vwap_data(ticker, date_str):
    """Load VWAP data for gap day."""
    fpath = os.path.join(VWAP_DIR, ticker, f'{date_str}.parquet')
    if not os.path.isfile(fpath):
        return None
    return pd.read_parquet(fpath)


def get_bar_at_time(bars, target_time):
    """Get the bar closest to target_time (e.g. '15:00')."""
    if bars is None or len(bars) == 0:
        return None
    if 'time_et' in bars.columns:
        match = bars[bars['time_et'] == target_time]
        if len(match) > 0:
            return match.iloc[-1]
        # Try closest
        match = bars[bars['time_et'] <= target_time]
        if len(match) > 0:
            return match.iloc[-1]
    return None


def get_rth_close_price(bars):
    """Get RTH close price (last RTH bar)."""
    if bars is None or len(bars) == 0:
        return None
    return bars.iloc[-1]['close']


def get_rth_open_price(bars):
    """Get RTH open price (first RTH bar)."""
    if bars is None or len(bars) == 0:
        return None
    return bars.iloc[0]['open']


def get_rth_high(bars):
    if bars is None or len(bars) == 0:
        return None
    return bars['high'].max()


def get_rth_low(bars):
    if bars is None or len(bars) == 0:
        return None
    return bars['low'].min()


def compute_stats(pnl_array, label=""):
    """Compute standard statistics for a PnL array."""
    arr = np.array(pnl_array)
    if len(arr) == 0:
        return {}
    wins = arr[arr > 0]
    losses = arr[arr <= 0]
    return {
        'label': label,
        'N': len(arr),
        'mean': np.mean(arr),
        'median': np.median(arr),
        'std': np.std(arr),
        'win_rate': len(wins) / len(arr) if len(arr) > 0 else 0,
        'avg_win': np.mean(wins) if len(wins) > 0 else 0,
        'avg_loss': np.mean(losses) if len(losses) > 0 else 0,
        'max': np.max(arr),
        'min': np.min(arr),
        'sharpe': np.mean(arr) / np.std(arr) if np.std(arr) > 0 else 0,
        'profit_factor': abs(np.sum(wins) / np.sum(losses)) if len(losses) > 0 and np.sum(losses) != 0 else float('inf'),
    }


def bootstrap_ci(pnl_array, n_boot=10000, ci=0.95):
    """Bootstrap confidence interval for mean."""
    arr = np.array(pnl_array)
    if len(arr) < 5:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(42)
    means = np.array([np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    lo = np.percentile(means, alpha * 100)
    hi = np.percentile(means, (1 - alpha) * 100)
    return np.mean(arr), lo, hi


def format_stats(stats):
    """Format stats dict to string."""
    if not stats or stats.get('N', 0) == 0:
        return "  No data\n"
    lines = []
    lines.append(f"  N={stats['N']}, Mean={stats['mean']:.4f}, Median={stats['median']:.4f}, Std={stats['std']:.4f}")
    lines.append(f"  WinRate={stats['win_rate']:.1%}, AvgWin={stats['avg_win']:.4f}, AvgLoss={stats['avg_loss']:.4f}")
    lines.append(f"  Sharpe={stats['sharpe']:.3f}, PF={stats['profit_factor']:.2f}, Max={stats['max']:.4f}, Min={stats['min']:.4f}")
    return '\n'.join(lines) + '\n'


# ============================================================
# DATA LOADING
# ============================================================

def load_and_prepare():
    """Load metadata and split IS/OOS."""
    meta = pd.read_parquet(META_PATH)
    is_mask = (meta['date'] >= '2021-02-21') & (meta['date'] <= '2023-12-31')
    oos_mask = (meta['date'] >= '2024-01-01') & (meta['date'] <= '2026-02-13')
    is_data = meta[is_mask].copy().reset_index(drop=True)
    oos_data = meta[oos_mask].copy().reset_index(drop=True)
    return meta, is_data, oos_data


# ============================================================
# PHASE 1: Basic PnL Measurement
# ============================================================

def phase1_basic_pnl(events, label="IS"):
    """Compute overnight and multi-day PnL for all events."""
    print(f"\n{'='*70}")
    print(f"PHASE 1: Basic PnL Measurement ({label}, N={len(events)})")
    print(f"{'='*70}")

    results = []
    skipped = 0
    t0 = time.time()

    for i, row in events.iterrows():
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  Progress: {i+1}/{len(events)} ({elapsed:.1f}s)")

        ticker = row['ticker']
        date_str = row['date']
        gap_dir = row['gap_direction']
        sign = 1.0 if gap_dir == 'up' else -1.0

        # ADR for normalization
        adr = row.get('adr_20', None)
        if adr is None or pd.isna(adr) or adr <= 0:
            adr = row.get('adr_10', None)
        if adr is None or pd.isna(adr) or adr <= 0:
            skipped += 1
            continue

        # Entry = RTH close from metadata
        entry_price = row.get('rth_close', None)
        if entry_price is None or pd.isna(entry_price) or entry_price <= 0:
            skipped += 1
            continue

        # Get multi-day bars (up to 5 days)
        multi_days = get_multi_day_bars(ticker, date_str, num_days=5)
        if len(multi_days) == 0:
            skipped += 1
            continue

        rec = {
            'ticker': ticker,
            'date': date_str,
            'gap_direction': gap_dir,
            'sign': sign,
            'entry_price': entry_price,
            'adr': adr,
            'gap_size_in_adr': row.get('gap_size_in_adr', np.nan),
            'od_strength': row.get('od_strength', np.nan),
            'close_location': row.get('close_location', None),
            'gap_filled': row.get('gap_filled', None),
            'rsi_14_prev': row.get('rsi_14_prev', np.nan),
            'vwap_held': row.get('vwap_held', None),
        }

        # Overnight PnL (next day open)
        nd_date, nd_bars = multi_days[0]
        nd_open = get_rth_open_price(nd_bars)
        nd_close = get_rth_close_price(nd_bars)
        nd_high = get_rth_high(nd_bars)
        nd_low = get_rth_low(nd_bars)

        if nd_open is not None:
            rec['overnight_pnl'] = (nd_open - entry_price) / adr * sign
        else:
            rec['overnight_pnl'] = np.nan

        if nd_close is not None:
            rec['day1_close_pnl'] = (nd_close - entry_price) / adr * sign
        else:
            rec['day1_close_pnl'] = np.nan

        # Max drawdown during next day (for long: low - entry, for short: entry - high)
        if nd_low is not None and nd_high is not None:
            if sign > 0:  # long
                rec['day1_max_dd'] = (nd_low - entry_price) / adr
            else:  # short
                rec['day1_max_dd'] = (entry_price - nd_high) / adr

        # Multi-day PnL
        for d in range(len(multi_days)):
            day_label = f'day{d+1}'
            _, day_bars = multi_days[d]
            day_close = get_rth_close_price(day_bars)
            if day_close is not None:
                rec[f'{day_label}_pnl'] = (day_close - entry_price) / adr * sign
            else:
                rec[f'{day_label}_pnl'] = np.nan

            # Track max drawdown up to this day
            day_low = get_rth_low(day_bars)
            day_high = get_rth_high(day_bars)
            if day_low is not None and day_high is not None:
                if sign > 0:
                    rec[f'{day_label}_worst'] = (day_low - entry_price) / adr
                else:
                    rec[f'{day_label}_worst'] = (entry_price - day_high) / adr

        results.append(rec)

    print(f"  Completed: {len(results)} valid, {skipped} skipped ({time.time()-t0:.1f}s)")
    df = pd.DataFrame(results)
    return df


def analyze_phase1(df, out):
    """Analyze and write Phase 1 results."""
    out.write("=" * 70 + "\n")
    out.write("PHASE 1: BASIC PnL MEASUREMENT\n")
    out.write("=" * 70 + "\n")
    out.write(f"Total valid events: {len(df)}\n\n")

    # Overall stats
    for col_label, col_name in [
        ("Overnight (next open)", "overnight_pnl"),
        ("Day+1 Close", "day1_close_pnl"),
        ("Day+2 Close", "day2_pnl"),
        ("Day+3 Close", "day3_pnl"),
        ("Day+5 Close", "day5_pnl"),
    ]:
        vals = df[col_name].dropna().values if col_name in df.columns else []
        stats = compute_stats(vals, col_label)
        out.write(f"--- {col_label} ---\n")
        out.write(format_stats(stats))
        out.write("\n")

    # Split by gap_direction
    for gap_dir in ['up', 'down']:
        sub = df[df['gap_direction'] == gap_dir]
        out.write(f"\n--- Gap Direction: {gap_dir.upper()} (N={len(sub)}) ---\n")
        for col_label, col_name in [
            ("Overnight", "overnight_pnl"),
            ("Day+1 Close", "day1_close_pnl"),
            ("Day+3 Close", "day3_pnl"),
            ("Day+5 Close", "day5_pnl"),
        ]:
            vals = sub[col_name].dropna().values if col_name in sub.columns else []
            stats = compute_stats(vals, f"{gap_dir} {col_label}")
            out.write(f"  {col_label}: ")
            if stats.get('N', 0) > 0:
                out.write(f"N={stats['N']}, Mean={stats['mean']:.4f}, WR={stats['win_rate']:.1%}, Sharpe={stats['sharpe']:.3f}\n")
            else:
                out.write("No data\n")

    # Split by gap_size_in_adr bins
    out.write("\n--- By Gap Size (ADR) ---\n")
    bins = [0, 1.0, 1.5, 2.0, 3.0, 5.0, 100]
    labels = ['0-1', '1-1.5', '1.5-2', '2-3', '3-5', '5+']
    df['gap_bin'] = pd.cut(df['gap_size_in_adr'], bins=bins, labels=labels, right=False)
    for b in labels:
        sub = df[df['gap_bin'] == b]
        if len(sub) < 10:
            continue
        for col_name, col_label in [("overnight_pnl", "ON"), ("day1_close_pnl", "D1")]:
            vals = sub[col_name].dropna().values
            stats = compute_stats(vals)
            out.write(f"  Gap {b} ADR - {col_label}: N={stats.get('N',0)}, Mean={stats.get('mean',0):.4f}, WR={stats.get('win_rate',0):.1%}\n")

    # Split by od_strength bins
    out.write("\n--- By OD Strength ---\n")
    od_bins = [0, 0.1, 0.3, 0.5, 0.7, 1.0, 10]
    od_labels = ['0-0.1', '0.1-0.3', '0.3-0.5', '0.5-0.7', '0.7-1.0', '1.0+']
    df['od_bin'] = pd.cut(df['od_strength'], bins=od_bins, labels=od_labels, right=False)
    for b in od_labels:
        sub = df[df['od_bin'] == b]
        if len(sub) < 10:
            continue
        for col_name, col_label in [("overnight_pnl", "ON"), ("day1_close_pnl", "D1")]:
            vals = sub[col_name].dropna().values
            stats = compute_stats(vals)
            out.write(f"  OD {b} - {col_label}: N={stats.get('N',0)}, Mean={stats.get('mean',0):.4f}, WR={stats.get('win_rate',0):.1%}\n")

    # Max Drawdown analysis
    out.write("\n--- Max Drawdown During Hold ---\n")
    for col in ['day1_max_dd']:
        if col in df.columns:
            vals = df[col].dropna().values
            out.write(f"  {col}: Mean={np.mean(vals):.4f}, Median={np.median(vals):.4f}, P5={np.percentile(vals,5):.4f}, P10={np.percentile(vals,10):.4f}\n")

    return df


# ============================================================
# PHASE 2: VWAP-Based Entry Conditions
# ============================================================

def phase2_vwap_conditions(events, phase1_df, out):
    """Load VWAP data and test entry conditions."""
    print(f"\n{'='*70}")
    print(f"PHASE 2: VWAP-Based Entry Conditions")
    print(f"{'='*70}")

    out.write("\n\n" + "=" * 70 + "\n")
    out.write("PHASE 2: VWAP-BASED ENTRY CONDITIONS\n")
    out.write("=" * 70 + "\n")

    # Load VWAP z_score at 15:00 and 15:30 for each event
    vwap_data = []
    t0 = time.time()
    for i, row in phase1_df.iterrows():
        if (i + 1) % 500 == 0:
            print(f"  VWAP loading: {i+1}/{len(phase1_df)} ({time.time()-t0:.1f}s)")

        ticker = row['ticker']
        date_str = row['date']
        vdf = get_vwap_data(ticker, date_str)
        if vdf is None:
            vwap_data.append({'z_1500': np.nan, 'z_1530': np.nan, 'z_1550': np.nan})
            continue

        z1500 = None
        z1530 = None
        z1550 = None

        if 'time_et' in vdf.columns:
            for tgt, key in [('15:00', 'z_1500'), ('15:30', 'z_1530'), ('15:50', 'z_1550')]:
                match = vdf[vdf['time_et'] == tgt]
                if len(match) > 0:
                    val = match.iloc[-1].get('z_score', np.nan)
                    if key == 'z_1500':
                        z1500 = val
                    elif key == 'z_1530':
                        z1530 = val
                    else:
                        z1550 = val
                else:
                    # Try closest before
                    before = vdf[vdf['time_et'] <= tgt]
                    if len(before) > 0:
                        val = before.iloc[-1].get('z_score', np.nan)
                        if key == 'z_1500':
                            z1500 = val
                        elif key == 'z_1530':
                            z1530 = val
                        else:
                            z1550 = val

        vwap_data.append({'z_1500': z1500, 'z_1530': z1530, 'z_1550': z1550})

    vwap_df = pd.DataFrame(vwap_data)
    phase1_df = phase1_df.copy()
    phase1_df['z_1500'] = vwap_df['z_1500'].values
    phase1_df['z_1530'] = vwap_df['z_1530'].values
    phase1_df['z_1550'] = vwap_df['z_1550'].values

    print(f"  VWAP data loaded ({time.time()-t0:.1f}s)")
    print(f"  z_1500 available: {phase1_df['z_1500'].notna().sum()}/{len(phase1_df)}")

    # Adjusted z-score: for gap-up, z>0 is favorable; for gap-down, z<0 is favorable
    # So we create "aligned_z" = z * sign
    phase1_df['aligned_z_1500'] = phase1_df['z_1500'] * phase1_df['sign']
    phase1_df['aligned_z_1530'] = phase1_df['z_1530'] * phase1_df['sign']

    # Test VWAP conditions
    conditions = [
        ("z_1500 aligned > 0 (price in gap direction)", phase1_df['aligned_z_1500'] > 0),
        ("z_1500 aligned > 0.5", phase1_df['aligned_z_1500'] > 0.5),
        ("z_1500 aligned > 1.0 (strong)", phase1_df['aligned_z_1500'] > 1.0),
        ("z_1500 aligned > 2.0 (extended)", phase1_df['aligned_z_1500'] > 2.0),
        ("z_1500 aligned 0 to 1.0 (pullback zone)", (phase1_df['aligned_z_1500'] > 0) & (phase1_df['aligned_z_1500'] <= 1.0)),
        ("z_1500 aligned < 0 (against gap)", phase1_df['aligned_z_1500'] < 0),
        ("z_1530 aligned > 0", phase1_df['aligned_z_1530'] > 0),
        ("z_1530 aligned > 1.0", phase1_df['aligned_z_1530'] > 1.0),
    ]

    out.write("\nVWAP Entry Condition Tests (z_score aligned with gap direction):\n")
    out.write("Positive aligned_z = price is in gap direction relative to VWAP\n\n")

    for desc, mask in conditions:
        mask = mask & phase1_df['z_1500'].notna()
        sub = phase1_df[mask]
        if len(sub) < 20:
            out.write(f"  {desc}: N={len(sub)} (too few)\n")
            continue

        out.write(f"\n  {desc} (N={len(sub)}):\n")
        for col_label, col_name in [
            ("Overnight", "overnight_pnl"),
            ("Day+1", "day1_close_pnl"),
            ("Day+3", "day3_pnl"),
            ("Day+5", "day5_pnl"),
        ]:
            vals = sub[col_name].dropna().values
            if len(vals) > 0:
                stats = compute_stats(vals)
                out.write(f"    {col_label}: N={stats['N']}, Mean={stats['mean']:.4f}, WR={stats['win_rate']:.1%}, Sharpe={stats['sharpe']:.3f}\n")

    # By gap direction + VWAP
    out.write("\n\n--- VWAP Conditions Split by Gap Direction ---\n")
    for gap_dir in ['up', 'down']:
        sign_val = 1.0 if gap_dir == 'up' else -1.0
        sub_dir = phase1_df[phase1_df['gap_direction'] == gap_dir]
        for z_desc, z_mask in [
            ("z>0 (above VWAP)", sub_dir['z_1500'] > 0),
            ("z>1.0", sub_dir['z_1500'] > 1.0),
            ("z<0 (below VWAP)", sub_dir['z_1500'] < 0),
            ("z<-1.0", sub_dir['z_1500'] < -1.0),
        ]:
            sel = sub_dir[z_mask & sub_dir['z_1500'].notna()]
            if len(sel) < 15:
                continue
            on_vals = sel['overnight_pnl'].dropna().values
            d1_vals = sel['day1_close_pnl'].dropna().values
            on_stats = compute_stats(on_vals)
            d1_stats = compute_stats(d1_vals)
            out.write(f"  Gap {gap_dir.upper()} + {z_desc}: N={len(sel)}, ON={on_stats.get('mean',0):.4f}, D1={d1_stats.get('mean',0):.4f}, WR_D1={d1_stats.get('win_rate',0):.1%}\n")

    return phase1_df


# ============================================================
# PHASE 3: Multi-Parameter Combinations
# ============================================================

def phase3_combinations(df, out):
    """Test multi-parameter combinations."""
    print(f"\n{'='*70}")
    print(f"PHASE 3: Multi-Parameter Combinations")
    print(f"{'='*70}")

    out.write("\n\n" + "=" * 70 + "\n")
    out.write("PHASE 3: MULTI-PARAMETER COMBINATIONS\n")
    out.write("=" * 70 + "\n\n")

    # Define parameter ranges
    gap_size_thresholds = [1.0, 1.5, 2.0, 3.0]
    od_strength_thresholds = [0.0, 0.3, 0.5, 0.7]
    gap_directions = ['up', 'down', 'both']
    vwap_conditions = [
        ('any', lambda d: pd.Series(True, index=d.index)),
        ('z>0', lambda d: d['aligned_z_1500'] > 0),
        ('z>0.5', lambda d: d['aligned_z_1500'] > 0.5),
        ('z>1.0', lambda d: d['aligned_z_1500'] > 1.0),
    ]
    gap_filled_opts = [None, True, False]
    close_loc_opts = [None, 'above_vwap', 'below_vwap']

    best_combos = []
    combo_count = 0

    for gap_dir in gap_directions:
        for gap_min in gap_size_thresholds:
            for od_min in od_strength_thresholds:
                for vwap_label, vwap_fn in vwap_conditions:
                    for gf in gap_filled_opts:
                        for cl_opt in close_loc_opts:
                            combo_count += 1

                            # Apply filters
                            mask = pd.Series(True, index=df.index)

                            if gap_dir != 'both':
                                mask &= df['gap_direction'] == gap_dir

                            mask &= df['gap_size_in_adr'] >= gap_min
                            mask &= df['od_strength'] >= od_min

                            if vwap_label != 'any':
                                vmask = vwap_fn(df)
                                mask &= vmask & df['aligned_z_1500'].notna()

                            if gf is not None:
                                mask &= df['gap_filled'] == gf

                            if cl_opt is not None:
                                mask &= df['close_location'] == cl_opt

                            sub = df[mask]
                            if len(sub) < 30:
                                continue

                            # Compute PnL for multiple hold periods
                            for hold_col, hold_label in [
                                ('overnight_pnl', 'ON'),
                                ('day1_close_pnl', 'D1'),
                                ('day3_pnl', 'D3'),
                                ('day5_pnl', 'D5'),
                            ]:
                                vals = sub[hold_col].dropna().values
                                if len(vals) < 30:
                                    continue
                                mean_pnl = np.mean(vals)
                                if mean_pnl > 0.05:  # Only positive EV
                                    stats = compute_stats(vals)
                                    gf_str = str(gf) if gf is not None else 'any'
                                    cl_str = cl_opt if cl_opt is not None else 'any'
                                    combo_desc = f"dir={gap_dir},gap>={gap_min},od>={od_min},vwap={vwap_label},filled={gf_str},cl={cl_str},hold={hold_label}"
                                    best_combos.append({
                                        'desc': combo_desc,
                                        'hold': hold_label,
                                        'N': stats['N'],
                                        'mean': stats['mean'],
                                        'median': stats['median'],
                                        'win_rate': stats['win_rate'],
                                        'sharpe': stats['sharpe'],
                                        'pf': stats['profit_factor'],
                                        'gap_dir': gap_dir,
                                        'gap_min': gap_min,
                                        'od_min': od_min,
                                        'vwap': vwap_label,
                                        'gap_filled': gf_str,
                                        'close_loc': cl_str,
                                    })

    print(f"  Tested {combo_count} combinations, found {len(best_combos)} with positive EV")

    # Sort by mean PnL
    best_combos.sort(key=lambda x: x['mean'], reverse=True)

    out.write(f"Tested {combo_count} combinations, {len(best_combos)} with positive EV (>0.05 ADR)\n\n")

    # Top 30
    out.write("TOP 30 COMBINATIONS (sorted by Mean PnL in ADR):\n")
    out.write(f"{'Rank':<5} {'N':<6} {'Mean':<8} {'Median':<8} {'WR':<8} {'Sharpe':<8} {'PF':<8} Description\n")
    out.write("-" * 130 + "\n")
    for i, c in enumerate(best_combos[:30]):
        out.write(f"{i+1:<5} {c['N']:<6} {c['mean']:<8.4f} {c['median']:<8.4f} {c['win_rate']:<8.1%} {c['sharpe']:<8.3f} {c['pf']:<8.2f} {c['desc']}\n")

    # Filter for N>=50 and mean>=0.10
    robust = [c for c in best_combos if c['N'] >= 50 and c['mean'] >= 0.10]
    robust.sort(key=lambda x: x['sharpe'], reverse=True)

    out.write(f"\n\nROBUST COMBINATIONS (N>=50, Mean>=0.10, sorted by Sharpe):\n")
    out.write(f"{'Rank':<5} {'N':<6} {'Mean':<8} {'Median':<8} {'WR':<8} {'Sharpe':<8} {'PF':<8} Description\n")
    out.write("-" * 130 + "\n")
    for i, c in enumerate(robust[:20]):
        out.write(f"{i+1:<5} {c['N']:<6} {c['mean']:<8.4f} {c['median']:<8.4f} {c['win_rate']:<8.1%} {c['sharpe']:<8.3f} {c['pf']:<8.2f} {c['desc']}\n")

    return best_combos


# ============================================================
# PHASE 4: Optimal Holding Period
# ============================================================

def phase4_holding_period(df, best_combos, out):
    """Analyze optimal holding period for top strategies."""
    print(f"\n{'='*70}")
    print(f"PHASE 4: Optimal Holding Period")
    print(f"{'='*70}")

    out.write("\n\n" + "=" * 70 + "\n")
    out.write("PHASE 4: OPTIMAL HOLDING PERIOD\n")
    out.write("=" * 70 + "\n\n")

    # Pick top unique filter combos (ignoring hold period)
    seen_filters = set()
    top_filters = []
    for c in best_combos:
        # Extract filter key without hold period
        filter_key = f"dir={c['gap_dir']},gap>={c['gap_min']},od>={c['od_min']},vwap={c['vwap']},filled={c['gap_filled']},cl={c['close_loc']}"
        if filter_key not in seen_filters and c['N'] >= 30:
            seen_filters.add(filter_key)
            top_filters.append(c)
            if len(top_filters) >= 10:
                break

    for filt in top_filters:
        # Reconstruct mask
        mask = pd.Series(True, index=df.index)
        if filt['gap_dir'] != 'both':
            mask &= df['gap_direction'] == filt['gap_dir']
        mask &= df['gap_size_in_adr'] >= filt['gap_min']
        mask &= df['od_strength'] >= filt['od_min']

        if filt['vwap'] == 'z>0':
            mask &= (df['aligned_z_1500'] > 0) & df['aligned_z_1500'].notna()
        elif filt['vwap'] == 'z>0.5':
            mask &= (df['aligned_z_1500'] > 0.5) & df['aligned_z_1500'].notna()
        elif filt['vwap'] == 'z>1.0':
            mask &= (df['aligned_z_1500'] > 1.0) & df['aligned_z_1500'].notna()

        if filt['gap_filled'] == 'True':
            mask &= df['gap_filled'] == True
        elif filt['gap_filled'] == 'False':
            mask &= df['gap_filled'] == False

        if filt['close_loc'] == 'above_vwap':
            mask &= df['close_location'] == 'above_vwap'
        elif filt['close_loc'] == 'below_vwap':
            mask &= df['close_location'] == 'below_vwap'

        sub = df[mask]
        filter_desc = f"dir={filt['gap_dir']},gap>={filt['gap_min']},od>={filt['od_min']},vwap={filt['vwap']},filled={filt['gap_filled']},cl={filt['close_loc']}"

        out.write(f"\nFilter: {filter_desc} (N={len(sub)})\n")
        out.write(f"  {'Hold':<12} {'N':<6} {'Mean':<8} {'Median':<8} {'WR':<8} {'Sharpe':<8} {'AvgDD':<8}\n")

        for hold_col, hold_label in [
            ('overnight_pnl', 'Overnight'),
            ('day1_close_pnl', 'Day+1'),
            ('day2_pnl', 'Day+2'),
            ('day3_pnl', 'Day+3'),
            ('day4_pnl', 'Day+4'),
            ('day5_pnl', 'Day+5'),
        ]:
            if hold_col not in sub.columns:
                continue
            vals = sub[hold_col].dropna().values
            if len(vals) < 10:
                continue
            stats = compute_stats(vals)

            # Average worst drawdown
            worst_col = hold_col.replace('_pnl', '_worst').replace('overnight_pnl', 'day1_worst')
            if worst_col == 'day1_close_worst':
                worst_col = 'day1_worst'
            avg_dd = np.nan
            if worst_col in sub.columns:
                dd_vals = sub[worst_col].dropna().values
                if len(dd_vals) > 0:
                    avg_dd = np.mean(dd_vals)

            out.write(f"  {hold_label:<12} {stats['N']:<6} {stats['mean']:<8.4f} {stats['median']:<8.4f} {stats['win_rate']:<8.1%} {stats['sharpe']:<8.3f} {avg_dd:<8.4f}\n")


# ============================================================
# PHASE 5: OOS Validation
# ============================================================

def phase5_oos_validation(is_df, oos_events, best_combos, out):
    """Validate top IS strategies on OOS data."""
    print(f"\n{'='*70}")
    print(f"PHASE 5: OOS Validation")
    print(f"{'='*70}")

    out.write("\n\n" + "=" * 70 + "\n")
    out.write("PHASE 5: OOS VALIDATION\n")
    out.write("=" * 70 + "\n\n")

    # First, compute OOS PnL data
    print("  Computing OOS PnL data...")
    oos_pnl = phase1_basic_pnl(oos_events, label="OOS")

    # Load VWAP for OOS
    print("  Loading OOS VWAP data...")
    vwap_data = []
    t0 = time.time()
    for i, row in oos_pnl.iterrows():
        if (i + 1) % 500 == 0:
            print(f"  OOS VWAP loading: {i+1}/{len(oos_pnl)} ({time.time()-t0:.1f}s)")

        ticker = row['ticker']
        date_str = row['date']
        vdf = get_vwap_data(ticker, date_str)
        z1500, z1530 = np.nan, np.nan
        if vdf is not None and 'time_et' in vdf.columns:
            for tgt, key in [('15:00', 'z1500'), ('15:30', 'z1530')]:
                match = vdf[vdf['time_et'] == tgt]
                if len(match) > 0:
                    val = match.iloc[-1].get('z_score', np.nan)
                else:
                    before = vdf[vdf['time_et'] <= tgt]
                    val = before.iloc[-1].get('z_score', np.nan) if len(before) > 0 else np.nan
                if key == 'z1500':
                    z1500 = val
                else:
                    z1530 = val
        vwap_data.append({'z_1500': z1500, 'z_1530': z1530})

    vwap_df = pd.DataFrame(vwap_data)
    oos_pnl['z_1500'] = vwap_df['z_1500'].values
    oos_pnl['z_1530'] = vwap_df['z_1530'].values
    oos_pnl['aligned_z_1500'] = oos_pnl['z_1500'] * oos_pnl['sign']
    oos_pnl['aligned_z_1530'] = oos_pnl['z_1530'] * oos_pnl['sign']

    # Select top 3 unique IS strategies
    seen = set()
    top3 = []
    # Prioritize robust combos (N>=50, Sharpe > 0)
    robust = [c for c in best_combos if c['N'] >= 50 and c['mean'] >= 0.10 and c['sharpe'] > 0]
    robust.sort(key=lambda x: x['sharpe'], reverse=True)

    for c in robust:
        filter_key = f"{c['gap_dir']}|{c['gap_min']}|{c['od_min']}|{c['vwap']}|{c['gap_filled']}|{c['close_loc']}|{c['hold']}"
        if filter_key not in seen:
            seen.add(filter_key)
            top3.append(c)
            if len(top3) >= 3:
                break

    # If not enough, add from all best_combos
    if len(top3) < 3:
        for c in best_combos:
            if c['N'] >= 30:
                filter_key = f"{c['gap_dir']}|{c['gap_min']}|{c['od_min']}|{c['vwap']}|{c['gap_filled']}|{c['close_loc']}|{c['hold']}"
                if filter_key not in seen:
                    seen.add(filter_key)
                    top3.append(c)
                    if len(top3) >= 3:
                        break

    out.write(f"OOS data computed: {len(oos_pnl)} events\n\n")

    for rank, strat in enumerate(top3):
        out.write(f"\n--- Strategy {rank+1}: {strat['desc']} ---\n")
        out.write(f"IS: N={strat['N']}, Mean={strat['mean']:.4f}, WR={strat['win_rate']:.1%}, Sharpe={strat['sharpe']:.3f}\n")

        # Apply same filters on OOS
        hold_map = {'ON': 'overnight_pnl', 'D1': 'day1_close_pnl', 'D3': 'day3_pnl', 'D5': 'day5_pnl',
                     'D2': 'day2_pnl', 'D4': 'day4_pnl'}
        hold_col = hold_map.get(strat['hold'], 'day1_close_pnl')

        mask = pd.Series(True, index=oos_pnl.index)
        if strat['gap_dir'] != 'both':
            mask &= oos_pnl['gap_direction'] == strat['gap_dir']
        mask &= oos_pnl['gap_size_in_adr'] >= strat['gap_min']
        mask &= oos_pnl['od_strength'] >= strat['od_min']

        if strat['vwap'] == 'z>0':
            mask &= (oos_pnl['aligned_z_1500'] > 0) & oos_pnl['aligned_z_1500'].notna()
        elif strat['vwap'] == 'z>0.5':
            mask &= (oos_pnl['aligned_z_1500'] > 0.5) & oos_pnl['aligned_z_1500'].notna()
        elif strat['vwap'] == 'z>1.0':
            mask &= (oos_pnl['aligned_z_1500'] > 1.0) & oos_pnl['aligned_z_1500'].notna()

        if strat['gap_filled'] == 'True':
            mask &= oos_pnl['gap_filled'] == True
        elif strat['gap_filled'] == 'False':
            mask &= oos_pnl['gap_filled'] == False

        if strat['close_loc'] == 'above_vwap':
            mask &= oos_pnl['close_location'] == 'above_vwap'
        elif strat['close_loc'] == 'below_vwap':
            mask &= oos_pnl['close_location'] == 'below_vwap'

        oos_sub = oos_pnl[mask]
        vals = oos_sub[hold_col].dropna().values

        if len(vals) < 5:
            out.write(f"OOS: N={len(vals)} (insufficient)\n")
            continue

        oos_stats = compute_stats(vals)
        out.write(f"OOS: N={oos_stats['N']}, Mean={oos_stats['mean']:.4f}, Median={oos_stats['median']:.4f}, ")
        out.write(f"WR={oos_stats['win_rate']:.1%}, Sharpe={oos_stats['sharpe']:.3f}, PF={oos_stats['profit_factor']:.2f}\n")

        # Bootstrap CI
        mean_val, ci_lo, ci_hi = bootstrap_ci(vals, n_boot=10000, ci=0.95)
        out.write(f"Bootstrap 95% CI: Mean={mean_val:.4f}, CI=[{ci_lo:.4f}, {ci_hi:.4f}]\n")
        if ci_lo > 0:
            out.write(f"  >>> STATISTICALLY SIGNIFICANT (CI lower bound > 0) <<<\n")
        else:
            out.write(f"  CI includes zero - not statistically significant\n")

        # Show all hold periods for this OOS filter
        out.write(f"  OOS Hold Period Comparison:\n")
        for hc, hl in [('overnight_pnl','ON'), ('day1_close_pnl','D1'), ('day2_pnl','D2'),
                        ('day3_pnl','D3'), ('day5_pnl','D5')]:
            if hc in oos_sub.columns:
                hvals = oos_sub[hc].dropna().values
                if len(hvals) >= 5:
                    hstats = compute_stats(hvals)
                    out.write(f"    {hl}: N={hstats['N']}, Mean={hstats['mean']:.4f}, WR={hstats['win_rate']:.1%}, Sharpe={hstats['sharpe']:.3f}\n")

    return oos_pnl


# ============================================================
# PHASE 6: Swing-Trade with Trailing Stop
# ============================================================

def phase6_swing_trailing(events, label, out):
    """Implement multi-day trailing stop strategy."""
    print(f"\n{'='*70}")
    print(f"PHASE 6: Swing-Trade with Trailing ({label})")
    print(f"{'='*70}")

    out.write("\n\n" + "=" * 70 + "\n")
    out.write(f"PHASE 6: SWING-TRADE WITH TRAILING STOP ({label})\n")
    out.write("=" * 70 + "\n\n")
    out.write("Logic:\n")
    out.write("  Entry: RTH close of gap day\n")
    out.write("  Initial SL: 0.5 ADR from entry (against gap direction)\n")
    out.write("  1R = 0.5 ADR\n")
    out.write("  If +1R reached intraday: move SL to breakeven\n")
    out.write("  Trail on daily close: if close worse than entry (for trade direction), exit\n")
    out.write("  Max hold: 5 trading days\n\n")

    results = []
    skipped = 0
    t0 = time.time()

    for i, row in events.iterrows():
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  Progress: {i+1}/{len(events)} ({elapsed:.1f}s)")

        ticker = row['ticker']
        date_str = row['date']
        gap_dir = row['gap_direction']
        sign = 1.0 if gap_dir == 'up' else -1.0

        adr = row.get('adr_20', None)
        if adr is None or pd.isna(adr) or adr <= 0:
            adr = row.get('adr_10', None)
        if adr is None or pd.isna(adr) or adr <= 0:
            skipped += 1
            continue

        entry_price = row.get('rth_close', None)
        if entry_price is None or pd.isna(entry_price) or entry_price <= 0:
            skipped += 1
            continue

        sl_dist = 0.5 * adr  # 1R = 0.5 ADR
        one_r = sl_dist
        initial_sl = entry_price - sign * sl_dist  # SL against gap direction

        multi_days = get_multi_day_bars(ticker, date_str, num_days=5)
        if len(multi_days) == 0:
            skipped += 1
            continue

        # Simulate trailing
        current_sl = initial_sl
        be_activated = False
        exit_price = None
        exit_day = None
        exit_reason = None
        max_favorable = 0

        for d_idx, (d_date, d_bars) in enumerate(multi_days):
            if d_bars is None or len(d_bars) == 0:
                continue

            day_high = d_bars['high'].max()
            day_low = d_bars['low'].min()
            day_close = d_bars.iloc[-1]['close']

            # Check if SL hit during the day
            if sign > 0:  # Long
                if day_low <= current_sl:
                    exit_price = current_sl
                    exit_day = d_idx + 1
                    exit_reason = 'SL_hit'
                    break
                # Track max favorable
                fav = (day_high - entry_price) / one_r
                max_favorable = max(max_favorable, fav)
                # If +1R reached, move to BE
                if day_high >= entry_price + one_r and not be_activated:
                    current_sl = entry_price
                    be_activated = True
                # Trail on close: if close < entry (and BE active)
                if be_activated and day_close < entry_price:
                    exit_price = day_close
                    exit_day = d_idx + 1
                    exit_reason = 'trail_close'
                    break
            else:  # Short
                if day_high >= current_sl:
                    exit_price = current_sl
                    exit_day = d_idx + 1
                    exit_reason = 'SL_hit'
                    break
                fav = (entry_price - day_low) / one_r
                max_favorable = max(max_favorable, fav)
                if day_low <= entry_price - one_r and not be_activated:
                    current_sl = entry_price
                    be_activated = True
                if be_activated and day_close > entry_price:
                    exit_price = day_close
                    exit_day = d_idx + 1
                    exit_reason = 'trail_close'
                    break

        # If not exited, exit at last available day close
        if exit_price is None:
            last_day_close = None
            for d_idx in range(len(multi_days) - 1, -1, -1):
                _, d_bars = multi_days[d_idx]
                if d_bars is not None and len(d_bars) > 0:
                    last_day_close = d_bars.iloc[-1]['close']
                    exit_day = d_idx + 1
                    exit_reason = 'max_hold'
                    break
            exit_price = last_day_close

        if exit_price is None:
            skipped += 1
            continue

        pnl_dollar = (exit_price - entry_price) * sign
        pnl_r = pnl_dollar / one_r
        pnl_adr = pnl_dollar / adr

        results.append({
            'ticker': ticker,
            'date': date_str,
            'gap_direction': gap_dir,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'exit_day': exit_day,
            'exit_reason': exit_reason,
            'pnl_r': pnl_r,
            'pnl_adr': pnl_adr,
            'max_favorable_r': max_favorable,
            'be_activated': be_activated,
            'adr': adr,
            'gap_size_in_adr': row.get('gap_size_in_adr', np.nan),
            'od_strength': row.get('od_strength', np.nan),
            'close_location': row.get('close_location', None),
            'gap_filled': row.get('gap_filled', None),
            'sign': sign,
        })

    elapsed = time.time() - t0
    print(f"  Completed: {len(results)} trades, {skipped} skipped ({elapsed:.1f}s)")

    swing_df = pd.DataFrame(results)

    # Overall stats
    pnl_r = swing_df['pnl_r'].values
    pnl_adr = swing_df['pnl_adr'].values
    stats_r = compute_stats(pnl_r, 'Swing PnL (R)')
    stats_adr = compute_stats(pnl_adr, 'Swing PnL (ADR)')

    out.write(f"Overall Results ({len(swing_df)} trades):\n")
    out.write(f"  PnL in R-multiples:\n")
    out.write(format_stats(stats_r))
    out.write(f"  PnL in ADR:\n")
    out.write(format_stats(stats_adr))

    # Exit reason distribution
    out.write(f"\nExit Reason Distribution:\n")
    for reason, count in swing_df['exit_reason'].value_counts().items():
        sub = swing_df[swing_df['exit_reason'] == reason]
        avg_r = sub['pnl_r'].mean()
        out.write(f"  {reason}: N={count} ({count/len(swing_df):.1%}), Avg PnL={avg_r:.3f}R\n")

    # Exit day distribution
    out.write(f"\nExit Day Distribution:\n")
    for day in sorted(swing_df['exit_day'].unique()):
        sub = swing_df[swing_df['exit_day'] == day]
        out.write(f"  Day {day}: N={len(sub)} ({len(sub)/len(swing_df):.1%}), Avg PnL={sub['pnl_r'].mean():.3f}R\n")

    # By gap direction
    out.write(f"\nBy Gap Direction:\n")
    for gd in ['up', 'down']:
        sub = swing_df[swing_df['gap_direction'] == gd]
        if len(sub) > 0:
            s = compute_stats(sub['pnl_r'].values)
            out.write(f"  {gd.upper()}: N={s['N']}, Mean={s['mean']:.4f}R, WR={s['win_rate']:.1%}, Sharpe={s['sharpe']:.3f}\n")

    # Filtered: gap_size >= 1.5, od >= 0.3
    out.write(f"\nFiltered (gap>=1.5 ADR, od>=0.3):\n")
    filt = swing_df[(swing_df['gap_size_in_adr'] >= 1.5) & (swing_df['od_strength'] >= 0.3)]
    if len(filt) >= 10:
        s = compute_stats(filt['pnl_r'].values)
        out.write(f"  N={s['N']}, Mean={s['mean']:.4f}R, WR={s['win_rate']:.1%}, Sharpe={s['sharpe']:.3f}, PF={s['profit_factor']:.2f}\n")

    # Check if mean >= 0.25R
    out.write(f"\n--- MINIMUM EV CHECK (0.25R) ---\n")
    if stats_r['mean'] >= 0.25:
        out.write(f"  PASS: Overall mean = {stats_r['mean']:.4f}R >= 0.25R\n")
    else:
        out.write(f"  FAIL: Overall mean = {stats_r['mean']:.4f}R < 0.25R\n")
        # Find filtered versions that pass
        out.write(f"  Searching for filtered combinations that pass...\n")
        for gap_min in [1.0, 1.5, 2.0]:
            for od_min in [0.0, 0.3, 0.5]:
                for gd in ['up', 'down', 'both']:
                    for cl in [None, 'above_vwap', 'below_vwap']:
                        for gf in [None, True, False]:
                            mask = pd.Series(True, index=swing_df.index)
                            mask &= swing_df['gap_size_in_adr'] >= gap_min
                            mask &= swing_df['od_strength'] >= od_min
                            if gd != 'both':
                                mask &= swing_df['gap_direction'] == gd
                            if cl is not None:
                                mask &= swing_df['close_location'] == cl
                            if gf is not None:
                                mask &= swing_df['gap_filled'] == gf
                            sub = swing_df[mask]
                            if len(sub) >= 30:
                                avg_r = sub['pnl_r'].mean()
                                if avg_r >= 0.25:
                                    wr = (sub['pnl_r'] > 0).mean()
                                    cl_str = cl if cl else 'any'
                                    gf_str = str(gf) if gf is not None else 'any'
                                    out.write(f"    PASS: gap>={gap_min},od>={od_min},dir={gd},cl={cl_str},filled={gf_str} -> N={len(sub)}, Mean={avg_r:.3f}R, WR={wr:.1%}\n")

    return swing_df


# ============================================================
# MAIN
# ============================================================

def main():
    t_start = time.time()
    print("D13 Late-Day Entry Strategy Analysis")
    print("=" * 70)

    meta, is_data, oos_data = load_and_prepare()
    print(f"IS events: {len(is_data)}, OOS events: {len(oos_data)}")

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as out:
        out.write("D13 LATE-DAY ENTRY STRATEGY RESULTS\n")
        out.write("=" * 70 + "\n")
        out.write(f"Generated: {pd.Timestamp.now()}\n")
        out.write(f"IS period: 2021-02-21 to 2023-12-31 ({len(is_data)} events)\n")
        out.write(f"OOS period: 2024-01-01 to 2026-02-13 ({len(oos_data)} events)\n")
        out.write(f"Entry: RTH close of gap day\n")
        out.write(f"Trade direction: gap_direction (long for gap-up, short for gap-down)\n")
        out.write(f"PnL normalized by ADR-20\n\n")

        # Phase 1: Basic PnL
        is_pnl = phase1_basic_pnl(is_data, label="IS")
        is_pnl = analyze_phase1(is_pnl, out)

        # Phase 2: VWAP conditions
        is_pnl = phase2_vwap_conditions(is_data, is_pnl, out)

        # Phase 3: Combinations
        best_combos = phase3_combinations(is_pnl, out)

        # Phase 4: Optimal holding period
        phase4_holding_period(is_pnl, best_combos, out)

        # Phase 5: OOS validation
        oos_pnl = phase5_oos_validation(is_pnl, oos_data, best_combos, out)

        # Phase 6: Swing trailing (IS)
        out.write("\n\n" + "=" * 70 + "\n")
        out.write("PHASE 6A: SWING-TRADE IS\n")
        out.write("=" * 70 + "\n")
        is_swing = phase6_swing_trailing(is_data, "IS", out)

        # Phase 6: Swing trailing (OOS)
        out.write("\n\n" + "=" * 70 + "\n")
        out.write("PHASE 6B: SWING-TRADE OOS\n")
        out.write("=" * 70 + "\n")
        oos_swing = phase6_swing_trailing(oos_data, "OOS", out)

        # Final Summary
        elapsed = time.time() - t_start
        out.write("\n\n" + "=" * 70 + "\n")
        out.write("FINAL SUMMARY\n")
        out.write("=" * 70 + "\n\n")

        # Best IS strategies
        robust = [c for c in best_combos if c['N'] >= 50 and c['mean'] >= 0.10]
        robust.sort(key=lambda x: x['sharpe'], reverse=True)

        out.write("Top IS Strategies (N>=50, Mean>=0.10, by Sharpe):\n")
        for i, c in enumerate(robust[:5]):
            out.write(f"  {i+1}. {c['desc']}\n")
            out.write(f"     N={c['N']}, Mean={c['mean']:.4f}, WR={c['win_rate']:.1%}, Sharpe={c['sharpe']:.3f}, PF={c['pf']:.2f}\n")

        # Swing trade summary
        out.write(f"\nSwing Trade Summary:\n")
        is_r = compute_stats(is_swing['pnl_r'].values)
        oos_r = compute_stats(oos_swing['pnl_r'].values)
        out.write(f"  IS:  N={is_r['N']}, Mean={is_r['mean']:.4f}R, WR={is_r['win_rate']:.1%}, Sharpe={is_r['sharpe']:.3f}\n")
        out.write(f"  OOS: N={oos_r['N']}, Mean={oos_r['mean']:.4f}R, WR={oos_r['win_rate']:.1%}, Sharpe={oos_r['sharpe']:.3f}\n")

        out.write(f"\nTotal runtime: {elapsed:.1f}s\n")

    print(f"\nResults saved to: {OUTPUT_PATH}")
    print(f"Total runtime: {time.time()-t_start:.1f}s")


if __name__ == '__main__':
    main()
