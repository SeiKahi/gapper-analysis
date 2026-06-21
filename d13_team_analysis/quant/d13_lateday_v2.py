"""
D13 Late-Day Swing Strategy v2 -- CORRECTED follow-up data
============================================================
Uses the follow-up day mapping (Polygon API data) instead of
file-directory listing, which previously introduced a fatal
multi-week skip bug.

Phases:
  1. Basic PnL measurement (IS)
  2. Filter combinations (IS)
  3. Top-5 IS -> OOS bootstrap validation
  4. Swing-Trail variant (IS + OOS)

Optimized: pre-loads all needed bar summaries into memory first,
then runs all analyses on the cached data.
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
import time
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings('ignore')

# ============================================================
# PATHS
# ============================================================

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASE = str(PROJECT_ROOT)
META_PATH = os.path.join(BASE, 'data/metadata/metadata_v9.parquet')
MAPPING_PATH = os.path.join(BASE, 'data/metadata/followup_day_mapping.parquet')
RAW_1MIN_DIR = os.path.join(BASE, 'data/raw_1min')
OUTPUT_PATH = os.path.join(BASE, 'd13_team_analysis/results/d13_lateday_v2_results.txt')

IS_START = '2021-02-21'
IS_END   = '2023-12-31'
OOS_START = '2024-01-01'
OOS_END   = '2026-02-13'

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def compute_stats(arr):
    """Compute standard statistics for a PnL array."""
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return {'N': 0}
    wins = arr[arr > 0]
    losses = arr[arr <= 0]
    std = np.std(arr, ddof=1) if len(arr) > 1 else 0.0
    return {
        'N': len(arr),
        'mean': np.mean(arr),
        'median': np.median(arr),
        'std': std,
        'win_rate': len(wins) / len(arr),
        'avg_win': np.mean(wins) if len(wins) > 0 else 0.0,
        'avg_loss': np.mean(losses) if len(losses) > 0 else 0.0,
        'sharpe': np.mean(arr) / std if std > 0 else 0.0,
        'profit_factor': abs(np.sum(wins) / np.sum(losses)) if len(losses) > 0 and np.sum(losses) != 0 else float('inf'),
        'max_val': np.max(arr),
        'min_val': np.min(arr),
    }


def format_stats(s, indent=2):
    """Format stats dict to string."""
    if s.get('N', 0) == 0:
        return " " * indent + "No data\n"
    pad = " " * indent
    lines = [
        f"{pad}N={s['N']}, Mean={s['mean']:.4f}, Median={s['median']:.4f}, Std={s['std']:.4f}",
        f"{pad}WR={s['win_rate']:.1%}, AvgWin={s['avg_win']:.4f}, AvgLoss={s['avg_loss']:.4f}",
        f"{pad}Sharpe={s['sharpe']:.3f}, PF={s['profit_factor']:.2f}, Max={s['max_val']:.4f}, Min={s['min_val']:.4f}",
    ]
    return '\n'.join(lines) + '\n'


def bootstrap_ci(arr, n_boot=10000, ci=0.95):
    """Bootstrap confidence interval for mean PnL."""
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 5:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(42)
    means = np.array([
        np.mean(rng.choice(arr, size=len(arr), replace=True))
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    lo = np.percentile(means, alpha * 100)
    hi = np.percentile(means, (1 - alpha) * 100)
    return np.mean(arr), lo, hi


# ============================================================
# FAST BATCH DATA LOADING
# ============================================================

def extract_bar_summary(ticker, date_str):
    """
    Read a parquet file and return a summary dict:
      rth_open, rth_close, rth_high, rth_low, price_1430
    Returns None if file not found or invalid.
    """
    fpath = os.path.join(RAW_1MIN_DIR, ticker, f'{date_str}.parquet')
    if not os.path.isfile(fpath):
        return None
    try:
        df = pd.read_parquet(fpath)
        if 'session' in df.columns:
            rth = df[df['session'] == 'rth']
        else:
            rth = df
        if len(rth) == 0:
            return None

        result = {
            'rth_open': rth.iloc[0]['open'],
            'rth_close': rth.iloc[-1]['close'],
            'rth_high': rth['high'].max(),
            'rth_low': rth['low'].min(),
        }

        # 14:30 price
        if 'time_et' in rth.columns:
            match = rth[rth['time_et'] <= '14:30']
            result['price_1430'] = match.iloc[-1]['close'] if len(match) > 0 else np.nan
        else:
            result['price_1430'] = np.nan

        return result
    except Exception:
        return None


def load_bar_summary_for_swing(ticker, date_str):
    """
    For swing-trail: need intraday high/low/close from RTH bars.
    Returns (rth_high, rth_low, rth_close) or None.
    """
    fpath = os.path.join(RAW_1MIN_DIR, ticker, f'{date_str}.parquet')
    if not os.path.isfile(fpath):
        return None
    try:
        df = pd.read_parquet(fpath)
        if 'session' in df.columns:
            rth = df[df['session'] == 'rth']
        else:
            rth = df
        if len(rth) == 0:
            return None
        return (rth['high'].max(), rth['low'].min(), rth.iloc[-1]['close'])
    except Exception:
        return None


def batch_preload_summaries(events, followup_lookup, need_gap_day=True):
    """
    Pre-load bar summaries for all files we'll need.
    Returns cache: dict of (ticker, date) -> summary_dict
    """
    # Collect all (ticker, date) pairs we need
    needed = set()
    for idx in range(len(events)):
        row = events.iloc[idx]
        ticker = row['ticker']
        gap_date = row['date']

        if need_gap_day:
            needed.add((ticker, gap_date))

        key = (ticker, gap_date)
        fu_map = followup_lookup.get(key, {})
        for offset in range(1, 11):
            fu_date = fu_map.get(offset, None)
            if fu_date is not None:
                needed.add((ticker, fu_date))

    print(f"  Need to load {len(needed)} unique bar files...")

    cache = {}
    loaded = 0
    missing = 0

    # Use thread pool for I/O-bound loading
    def _load_one(item):
        ticker, date_str = item
        return item, extract_bar_summary(ticker, date_str)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_load_one, item): item for item in needed}
        for future in tqdm(as_completed(futures), total=len(futures),
                          desc="  Loading bar files", ncols=100):
            item, result = future.result()
            if result is not None:
                cache[item] = result
                loaded += 1
            else:
                missing += 1

    print(f"  Loaded: {loaded}, Missing: {missing}")
    return cache


# ============================================================
# DATA LOADING & SPLITTING
# ============================================================

def load_and_split():
    """Load metadata, split IS/OOS."""
    meta = pd.read_parquet(META_PATH)
    is_mask = (meta['date'] >= IS_START) & (meta['date'] <= IS_END)
    oos_mask = (meta['date'] >= OOS_START) & (meta['date'] <= OOS_END)
    return meta, meta[is_mask].copy().reset_index(drop=True), meta[oos_mask].copy().reset_index(drop=True)


def build_followup_lookup(mapping_df):
    """Returns dict: (ticker, gap_date) -> {offset: actual_date, ...}"""
    lookup = {}
    for _, row in mapping_df.iterrows():
        key = (row['ticker'], row['gap_date'])
        if key not in lookup:
            lookup[key] = {}
        lookup[key][row['offset']] = row['actual_date']
    return lookup


# ============================================================
# PnL EXTRACTION (uses pre-loaded cache)
# ============================================================

def extract_pnl_all(events, followup_lookup, cache, label="IS"):
    """
    For each event compute PnL for overnight, day+1..+5, day+10 closes.
    Also supports 14:30 entry variant.
    Uses pre-loaded cache for speed.
    """
    records = []
    skipped = 0

    for idx in tqdm(range(len(events)), desc=f"  {label} PnL calc", ncols=100):
        row = events.iloc[idx]
        ticker = row['ticker']
        gap_date = row['date']
        gap_dir = row['gap_direction']
        sign = 1.0 if gap_dir == 'up' else -1.0

        # ADR for normalization
        adr = row.get('adr_20', np.nan)
        if pd.isna(adr) or adr <= 0:
            adr = row.get('adr_5', np.nan)
        if pd.isna(adr) or adr <= 0:
            skipped += 1
            continue

        # Entry prices from metadata
        entry_close = row.get('rth_close', np.nan)
        if pd.isna(entry_close) or entry_close <= 0:
            skipped += 1
            continue

        # 14:30 entry from cached gap day bars
        gap_summary = cache.get((ticker, gap_date))
        entry_1430 = gap_summary['price_1430'] if gap_summary is not None else np.nan

        # Follow-up days from mapping
        key = (ticker, gap_date)
        fu_map = followup_lookup.get(key, {})

        rec = {
            'ticker': ticker,
            'date': gap_date,
            'gap_direction': gap_dir,
            'sign': sign,
            'entry_close': entry_close,
            'entry_1430': entry_1430,
            'adr': adr,
            'gap_size_in_adr': row.get('gap_size_in_adr', np.nan),
            'od_strength': row.get('od_strength', np.nan),
            'gap_filled': row.get('gap_filled', None),
        }

        # Process each follow-up day (offsets 1..10)
        for offset in range(1, 11):
            fu_date = fu_map.get(offset, None)
            if fu_date is None:
                rec[f'day{offset}_open'] = np.nan
                rec[f'day{offset}_close'] = np.nan
                rec[f'day{offset}_high'] = np.nan
                rec[f'day{offset}_low'] = np.nan
                continue

            fu_summary = cache.get((ticker, fu_date))
            if fu_summary is None:
                rec[f'day{offset}_open'] = np.nan
                rec[f'day{offset}_close'] = np.nan
                rec[f'day{offset}_high'] = np.nan
                rec[f'day{offset}_low'] = np.nan
            else:
                rec[f'day{offset}_open'] = fu_summary['rth_open']
                rec[f'day{offset}_close'] = fu_summary['rth_close']
                rec[f'day{offset}_high'] = fu_summary['rth_high']
                rec[f'day{offset}_low'] = fu_summary['rth_low']
            rec[f'day{offset}_date'] = fu_date

        # ---- Compute PnL columns ----
        d1_open = rec.get('day1_open', np.nan)
        if not np.isnan(d1_open):
            rec['pnl_overnight'] = (d1_open - entry_close) / adr * sign
        else:
            rec['pnl_overnight'] = np.nan

        for d in [1, 2, 3, 5, 10]:
            d_close = rec.get(f'day{d}_close', np.nan)
            if not np.isnan(d_close):
                rec[f'pnl_day{d}_close'] = (d_close - entry_close) / adr * sign
            else:
                rec[f'pnl_day{d}_close'] = np.nan

        # 14:30 entry variants
        if not np.isnan(entry_1430):
            rec['pnl_overnight_1430'] = (d1_open - entry_1430) / adr * sign if not np.isnan(d1_open) else np.nan
            for d in [1, 2, 3, 5, 10]:
                d_close = rec.get(f'day{d}_close', np.nan)
                if not np.isnan(d_close):
                    rec[f'pnl_day{d}_close_1430'] = (d_close - entry_1430) / adr * sign
                else:
                    rec[f'pnl_day{d}_close_1430'] = np.nan
        else:
            rec['pnl_overnight_1430'] = np.nan
            for d in [1, 2, 3, 5, 10]:
                rec[f'pnl_day{d}_close_1430'] = np.nan

        records.append(rec)

    print(f"  {label}: {len(records)} valid events, {skipped} skipped")
    return pd.DataFrame(records)


# ============================================================
# PHASE 1: Basic PnL Measurement
# ============================================================

def phase1_basic_pnl(pnl_df, out):
    out.write("=" * 70 + "\n")
    out.write("PHASE 1: BASIC PnL MEASUREMENT (IS)\n")
    out.write("=" * 70 + "\n")
    out.write(f"Total valid events: {len(pnl_df)}\n\n")

    hold_periods = [
        ("Overnight (entry=close, exit=D+1 open)", "pnl_overnight"),
        ("Day+1 Close", "pnl_day1_close"),
        ("Day+2 Close", "pnl_day2_close"),
        ("Day+3 Close", "pnl_day3_close"),
        ("Day+5 Close", "pnl_day5_close"),
        ("Day+10 Close", "pnl_day10_close"),
    ]

    out.write("--- Overall (all events, entry = RTH close) ---\n")
    for label, col in hold_periods:
        vals = pnl_df[col].dropna().values if col in pnl_df.columns else np.array([])
        s = compute_stats(vals)
        out.write(f"\n  {label}:\n")
        out.write(format_stats(s, indent=4))

    hold_periods_1430 = [
        ("Overnight (entry=14:30, exit=D+1 open)", "pnl_overnight_1430"),
        ("Day+1 Close (14:30 entry)", "pnl_day1_close_1430"),
        ("Day+5 Close (14:30 entry)", "pnl_day5_close_1430"),
    ]
    out.write("\n--- 14:30 Entry Variant ---\n")
    for label, col in hold_periods_1430:
        vals = pnl_df[col].dropna().values if col in pnl_df.columns else np.array([])
        s = compute_stats(vals)
        out.write(f"\n  {label}:\n")
        out.write(format_stats(s, indent=4))

    out.write("\n--- By Gap Direction ---\n")
    for gap_dir in ['up', 'down']:
        sub = pnl_df[pnl_df['gap_direction'] == gap_dir]
        out.write(f"\n  Gap {gap_dir.upper()} (N={len(sub)}):\n")
        for label, col in [
            ("Overnight", "pnl_overnight"),
            ("Day+1 Close", "pnl_day1_close"),
            ("Day+3 Close", "pnl_day3_close"),
            ("Day+5 Close", "pnl_day5_close"),
            ("Day+10 Close", "pnl_day10_close"),
        ]:
            vals = sub[col].dropna().values
            s = compute_stats(vals)
            if s['N'] > 0:
                out.write(f"    {label}: N={s['N']}, Mean={s['mean']:.4f}, WR={s['win_rate']:.1%}, Sharpe={s['sharpe']:.3f}\n")
            else:
                out.write(f"    {label}: No data\n")

    out.write("\n--- By Gap Filled ---\n")
    for gf in [True, False]:
        sub = pnl_df[pnl_df['gap_filled'] == gf]
        out.write(f"\n  gap_filled={gf} (N={len(sub)}):\n")
        for label, col in [
            ("Overnight", "pnl_overnight"),
            ("Day+1 Close", "pnl_day1_close"),
            ("Day+5 Close", "pnl_day5_close"),
        ]:
            vals = sub[col].dropna().values
            s = compute_stats(vals)
            if s['N'] > 0:
                out.write(f"    {label}: N={s['N']}, Mean={s['mean']:.4f}, WR={s['win_rate']:.1%}, Sharpe={s['sharpe']:.3f}\n")

    out.write("\n--- By OD Strength ---\n")
    od_bins = [(0, 0.3, '0-0.3'), (0.3, 0.5, '0.3-0.5'), (0.5, 0.7, '0.5-0.7'), (0.7, 1.0, '0.7-1.0'), (1.0, 100, '1.0+')]
    for lo, hi, lbl in od_bins:
        sub = pnl_df[(pnl_df['od_strength'] >= lo) & (pnl_df['od_strength'] < hi)]
        if len(sub) < 10:
            continue
        out.write(f"\n  OD {lbl} (N={len(sub)}):\n")
        for clabel, col in [("ON", "pnl_overnight"), ("D1", "pnl_day1_close"), ("D5", "pnl_day5_close")]:
            vals = sub[col].dropna().values
            s = compute_stats(vals)
            if s['N'] > 0:
                out.write(f"    {clabel}: N={s['N']}, Mean={s['mean']:.4f}, WR={s['win_rate']:.1%}, Sharpe={s['sharpe']:.3f}\n")

    out.write("\n--- By Gap Size (ADR) ---\n")
    gs_bins = [(0, 1.0, '0-1'), (1.0, 1.5, '1-1.5'), (1.5, 2.0, '1.5-2'), (2.0, 3.0, '2-3'), (3.0, 5.0, '3-5'), (5.0, 999, '5+')]
    for lo, hi, lbl in gs_bins:
        sub = pnl_df[(pnl_df['gap_size_in_adr'] >= lo) & (pnl_df['gap_size_in_adr'] < hi)]
        if len(sub) < 10:
            continue
        out.write(f"\n  Gap {lbl} ADR (N={len(sub)}):\n")
        for clabel, col in [("ON", "pnl_overnight"), ("D1", "pnl_day1_close"), ("D5", "pnl_day5_close")]:
            vals = sub[col].dropna().values
            s = compute_stats(vals)
            if s['N'] > 0:
                out.write(f"    {clabel}: N={s['N']}, Mean={s['mean']:.4f}, WR={s['win_rate']:.1%}, Sharpe={s['sharpe']:.3f}\n")


# ============================================================
# PHASE 2: Filter Combinations (IS only)
# ============================================================

def phase2_filter_combos(pnl_df, out):
    out.write("\n\n" + "=" * 70 + "\n")
    out.write("PHASE 2: FILTER COMBINATIONS (IS)\n")
    out.write("=" * 70 + "\n\n")

    gap_dirs = ['up', 'down', 'all']
    gap_sizes = [1.0, 1.5, 2.0, 3.0]
    od_thresholds = [0.3, 0.5, 0.7]
    gap_filled_opts = [True, False, 'any']
    hold_periods = [1, 2, 3, 5, 10]
    entry_times = ['close', '1430']

    results = []
    combo_count = 0

    for gap_dir in gap_dirs:
        for gs_min in gap_sizes:
            for od_min in od_thresholds:
                for gf in gap_filled_opts:
                    for hold in hold_periods:
                        for entry in entry_times:
                            combo_count += 1

                            mask = pd.Series(True, index=pnl_df.index)
                            if gap_dir != 'all':
                                mask &= pnl_df['gap_direction'] == gap_dir
                            mask &= pnl_df['gap_size_in_adr'] >= gs_min
                            mask &= pnl_df['od_strength'] >= od_min
                            if gf != 'any':
                                mask &= pnl_df['gap_filled'] == gf

                            if entry == 'close':
                                pnl_col = f'pnl_day{hold}_close'
                            else:
                                pnl_col = f'pnl_day{hold}_close_1430'

                            if pnl_col not in pnl_df.columns:
                                continue

                            sub = pnl_df[mask]
                            vals = sub[pnl_col].dropna().values

                            if len(vals) < 30:
                                continue

                            s = compute_stats(vals)
                            if s['mean'] <= 0:
                                continue

                            gf_str = str(gf) if gf != 'any' else 'any'
                            desc = f"dir={gap_dir},gap>={gs_min},od>={od_min},filled={gf_str},hold=D{hold},entry={entry}"

                            results.append({
                                'desc': desc,
                                'gap_dir': gap_dir,
                                'gs_min': gs_min,
                                'od_min': od_min,
                                'gf': gf_str,
                                'hold': hold,
                                'entry': entry,
                                'pnl_col': pnl_col,
                                **s,
                            })

    print(f"  Tested {combo_count} combos, {len(results)} with positive mean & N>=30")

    results.sort(key=lambda x: x['sharpe'], reverse=True)

    out.write(f"Tested {combo_count} combinations\n")
    out.write(f"Positive mean & N>=30: {len(results)}\n\n")

    out.write("TOP 40 COMBINATIONS (sorted by Sharpe):\n")
    out.write(f"{'#':<4} {'N':<6} {'Mean':<8} {'Med':<8} {'WR':<7} {'Sharpe':<8} {'PF':<7} Description\n")
    out.write("-" * 130 + "\n")
    for i, c in enumerate(results[:40]):
        out.write(f"{i+1:<4} {c['N']:<6} {c['mean']:<8.4f} {c['median']:<8.4f} {c['win_rate']:<7.1%} {c['sharpe']:<8.3f} {c['profit_factor']:<7.2f} {c['desc']}\n")

    return results


# ============================================================
# PHASE 3: Top-5 IS -> OOS bootstrap validation
# ============================================================

def phase3_oos_validation(is_combos, oos_pnl, out):
    out.write("\n\n" + "=" * 70 + "\n")
    out.write("PHASE 3: TOP-5 IS -> OOS VALIDATION\n")
    out.write("=" * 70 + "\n\n")

    candidates = [c for c in is_combos if c['N'] >= 30 and c['mean'] > 0]
    candidates.sort(key=lambda x: x['sharpe'], reverse=True)

    seen = set()
    top5 = []
    for c in candidates:
        sig = f"{c['gap_dir']}|{c['gs_min']}|{c['od_min']}|{c['gf']}|{c['hold']}|{c['entry']}"
        if sig not in seen:
            seen.add(sig)
            top5.append(c)
            if len(top5) >= 5:
                break

    for rank, strat in enumerate(top5):
        out.write(f"\n{'='*60}\n")
        out.write(f"STRATEGY {rank+1}: {strat['desc']}\n")
        out.write(f"{'='*60}\n")
        out.write(f"IS Results: N={strat['N']}, Mean={strat['mean']:.4f}, WR={strat['win_rate']:.1%}, Sharpe={strat['sharpe']:.3f}, PF={strat['profit_factor']:.2f}\n\n")

        mask = pd.Series(True, index=oos_pnl.index)
        if strat['gap_dir'] != 'all':
            mask &= oos_pnl['gap_direction'] == strat['gap_dir']
        mask &= oos_pnl['gap_size_in_adr'] >= strat['gs_min']
        mask &= oos_pnl['od_strength'] >= strat['od_min']
        if strat['gf'] == 'True':
            mask &= oos_pnl['gap_filled'] == True
        elif strat['gf'] == 'False':
            mask &= oos_pnl['gap_filled'] == False

        pnl_col = strat['pnl_col']
        if pnl_col not in oos_pnl.columns:
            out.write("  OOS: PnL column not available\n")
            continue

        oos_sub = oos_pnl[mask]
        vals = oos_sub[pnl_col].dropna().values

        if len(vals) < 5:
            out.write(f"  OOS: N={len(vals)} (insufficient data)\n")
            continue

        oos_s = compute_stats(vals)
        out.write(f"OOS Results: N={oos_s['N']}, Mean={oos_s['mean']:.4f}, Median={oos_s['median']:.4f}, WR={oos_s['win_rate']:.1%}, Sharpe={oos_s['sharpe']:.3f}, PF={oos_s['profit_factor']:.2f}\n")

        mean_val, ci_lo, ci_hi = bootstrap_ci(vals, n_boot=10000, ci=0.95)
        out.write(f"Bootstrap 95% CI: Mean={mean_val:.4f}, CI=[{ci_lo:.4f}, {ci_hi:.4f}]\n")
        if ci_lo > 0:
            out.write("  >>> STATISTICALLY SIGNIFICANT (CI lower bound > 0) <<<\n")
        else:
            out.write("  CI includes zero -- NOT statistically significant\n")

        out.write(f"\n  OOS: All hold periods for this filter (entry={strat['entry']}):\n")
        entry_sfx = '' if strat['entry'] == 'close' else '_1430'
        for d in [1, 2, 3, 5, 10]:
            col_check = f'pnl_day{d}_close{entry_sfx}'
            if col_check in oos_sub.columns:
                hvals = oos_sub[col_check].dropna().values
                if len(hvals) >= 5:
                    hs = compute_stats(hvals)
                    out.write(f"    D{d}: N={hs['N']}, Mean={hs['mean']:.4f}, WR={hs['win_rate']:.1%}, Sharpe={hs['sharpe']:.3f}\n")

        on_col = f'pnl_overnight{entry_sfx}'
        if on_col in oos_sub.columns:
            on_vals = oos_sub[on_col].dropna().values
            if len(on_vals) >= 5:
                on_s = compute_stats(on_vals)
                out.write(f"    ON:  N={on_s['N']}, Mean={on_s['mean']:.4f}, WR={on_s['win_rate']:.1%}, Sharpe={on_s['sharpe']:.3f}\n")


# ============================================================
# PHASE 4: Swing-Trail Variant (uses pre-loaded cache)
# ============================================================

def phase4_swing_trail(events, followup_lookup, cache, label, out):
    """
    Swing with trailing stop using cached bar summaries.
    SL=0.5 ADR, +1R->BE, close-based trail, max 5 days.
    """
    out.write("\n\n" + "=" * 70 + "\n")
    out.write(f"PHASE 4: SWING-TRAIL ({label})\n")
    out.write("=" * 70 + "\n\n")
    out.write("Logic:\n")
    out.write("  Entry: RTH close of gap day\n")
    out.write("  Initial SL: 0.5 ADR against trade direction\n")
    out.write("  1R = 0.5 ADR\n")
    out.write("  +1R reached intraday -> SL moves to breakeven\n")
    out.write("  Close-based trail: if close reverts past entry, exit at close\n")
    out.write("  Max hold: 5 trading days\n\n")

    results = []
    skipped = 0

    for idx in tqdm(range(len(events)), desc=f"  {label} Swing-Trail", ncols=100):
        row = events.iloc[idx]
        ticker = row['ticker']
        gap_date = row['date']
        gap_dir = row['gap_direction']
        sign = 1.0 if gap_dir == 'up' else -1.0

        adr = row.get('adr_20', np.nan)
        if pd.isna(adr) or adr <= 0:
            adr = row.get('adr_5', np.nan)
        if pd.isna(adr) or adr <= 0:
            skipped += 1
            continue

        entry_price = row.get('rth_close', np.nan)
        if pd.isna(entry_price) or entry_price <= 0:
            skipped += 1
            continue

        sl_dist = 0.5 * adr
        one_r = sl_dist
        initial_sl = entry_price - sign * sl_dist

        key = (ticker, gap_date)
        fu_map = followup_lookup.get(key, {})

        current_sl = initial_sl
        be_activated = False
        exit_price = None
        exit_day = None
        exit_reason = None
        max_favorable_r = 0.0
        last_valid_close = None
        last_valid_day = None

        for offset in range(1, 6):  # max 5 days
            fu_date = fu_map.get(offset, None)
            if fu_date is None:
                continue

            fu_summary = cache.get((ticker, fu_date))
            if fu_summary is None:
                continue

            day_high = fu_summary['rth_high']
            day_low = fu_summary['rth_low']
            day_close = fu_summary['rth_close']

            if np.isnan(day_high) or np.isnan(day_low) or np.isnan(day_close):
                continue

            last_valid_close = day_close
            last_valid_day = offset

            if sign > 0:  # Long
                if day_low <= current_sl:
                    exit_price = current_sl
                    exit_day = offset
                    exit_reason = 'SL_hit'
                    break

                fav = (day_high - entry_price) / one_r
                max_favorable_r = max(max_favorable_r, fav)

                if day_high >= entry_price + one_r and not be_activated:
                    current_sl = entry_price
                    be_activated = True

                if be_activated and day_close < entry_price:
                    exit_price = day_close
                    exit_day = offset
                    exit_reason = 'trail_close'
                    break

            else:  # Short
                if day_high >= current_sl:
                    exit_price = current_sl
                    exit_day = offset
                    exit_reason = 'SL_hit'
                    break

                fav = (entry_price - day_low) / one_r
                max_favorable_r = max(max_favorable_r, fav)

                if day_low <= entry_price - one_r and not be_activated:
                    current_sl = entry_price
                    be_activated = True

                if be_activated and day_close > entry_price:
                    exit_price = day_close
                    exit_day = offset
                    exit_reason = 'trail_close'
                    break

        if exit_price is None:
            if last_valid_close is not None:
                exit_price = last_valid_close
                exit_day = last_valid_day
                exit_reason = 'max_hold'
            else:
                skipped += 1
                continue

        pnl_dollar = (exit_price - entry_price) * sign
        pnl_r = pnl_dollar / one_r
        pnl_adr = pnl_dollar / adr

        results.append({
            'ticker': ticker,
            'date': gap_date,
            'gap_direction': gap_dir,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'exit_day': exit_day,
            'exit_reason': exit_reason,
            'pnl_r': pnl_r,
            'pnl_adr': pnl_adr,
            'max_favorable_r': max_favorable_r,
            'be_activated': be_activated,
            'adr': adr,
            'gap_size_in_adr': row.get('gap_size_in_adr', np.nan),
            'od_strength': row.get('od_strength', np.nan),
            'gap_filled': row.get('gap_filled', None),
        })

    print(f"  {label} Swing: {len(results)} trades, {skipped} skipped")

    if len(results) == 0:
        out.write("  No valid trades.\n")
        return pd.DataFrame()

    swing_df = pd.DataFrame(results)

    s_r = compute_stats(swing_df['pnl_r'].values)
    s_adr = compute_stats(swing_df['pnl_adr'].values)

    out.write(f"Overall ({len(swing_df)} trades):\n")
    out.write(f"  PnL in R-multiples:\n")
    out.write(format_stats(s_r, indent=4))
    out.write(f"  PnL in ADR:\n")
    out.write(format_stats(s_adr, indent=4))

    out.write(f"\nExit Reason Distribution:\n")
    for reason in swing_df['exit_reason'].value_counts().index:
        sub = swing_df[swing_df['exit_reason'] == reason]
        avg_r = sub['pnl_r'].mean()
        out.write(f"  {reason}: N={len(sub)} ({len(sub)/len(swing_df):.1%}), Avg={avg_r:.3f}R\n")

    out.write(f"\nExit Day Distribution:\n")
    for day in sorted(swing_df['exit_day'].dropna().unique()):
        sub = swing_df[swing_df['exit_day'] == day]
        out.write(f"  Day {int(day)}: N={len(sub)} ({len(sub)/len(swing_df):.1%}), Avg={sub['pnl_r'].mean():.3f}R\n")

    out.write(f"\nBy Gap Direction:\n")
    for gd in ['up', 'down']:
        sub = swing_df[swing_df['gap_direction'] == gd]
        if len(sub) > 0:
            s = compute_stats(sub['pnl_r'].values)
            out.write(f"  {gd.upper()}: N={s['N']}, Mean={s['mean']:.4f}R, WR={s['win_rate']:.1%}, Sharpe={s['sharpe']:.3f}\n")

    out.write(f"\nFiltered Subsets:\n")
    filter_combos = [
        ("gap>=1.0, od>=0.3", lambda d: (d['gap_size_in_adr'] >= 1.0) & (d['od_strength'] >= 0.3)),
        ("gap>=1.5, od>=0.3", lambda d: (d['gap_size_in_adr'] >= 1.5) & (d['od_strength'] >= 0.3)),
        ("gap>=1.5, od>=0.5", lambda d: (d['gap_size_in_adr'] >= 1.5) & (d['od_strength'] >= 0.5)),
        ("gap>=2.0, od>=0.5", lambda d: (d['gap_size_in_adr'] >= 2.0) & (d['od_strength'] >= 0.5)),
        ("gap>=2.0, od>=0.7", lambda d: (d['gap_size_in_adr'] >= 2.0) & (d['od_strength'] >= 0.7)),
        ("gap>=1.0, od>=0.3, filled=False", lambda d: (d['gap_size_in_adr'] >= 1.0) & (d['od_strength'] >= 0.3) & (d['gap_filled'] == False)),
        ("gap>=1.5, od>=0.5, filled=False", lambda d: (d['gap_size_in_adr'] >= 1.5) & (d['od_strength'] >= 0.5) & (d['gap_filled'] == False)),
        ("gap>=1.0, od>=0.3, dir=up", lambda d: (d['gap_size_in_adr'] >= 1.0) & (d['od_strength'] >= 0.3) & (d['gap_direction'] == 'up')),
        ("gap>=1.0, od>=0.3, dir=down", lambda d: (d['gap_size_in_adr'] >= 1.0) & (d['od_strength'] >= 0.3) & (d['gap_direction'] == 'down')),
        ("gap>=1.5, od>=0.5, dir=up", lambda d: (d['gap_size_in_adr'] >= 1.5) & (d['od_strength'] >= 0.5) & (d['gap_direction'] == 'up')),
        ("gap>=1.5, od>=0.5, dir=down", lambda d: (d['gap_size_in_adr'] >= 1.5) & (d['od_strength'] >= 0.5) & (d['gap_direction'] == 'down')),
    ]
    for desc, fn in filter_combos:
        mask = fn(swing_df)
        sub = swing_df[mask]
        if len(sub) < 10:
            out.write(f"  {desc}: N={len(sub)} (too few)\n")
            continue
        s = compute_stats(sub['pnl_r'].values)
        out.write(f"  {desc}: N={s['N']}, Mean={s['mean']:.4f}R, WR={s['win_rate']:.1%}, Sharpe={s['sharpe']:.3f}, PF={s['profit_factor']:.2f}\n")

    mean_val, ci_lo, ci_hi = bootstrap_ci(swing_df['pnl_r'].values, n_boot=10000)
    out.write(f"\nBootstrap 95% CI (overall): Mean={mean_val:.4f}R, CI=[{ci_lo:.4f}, {ci_hi:.4f}]\n")
    if ci_lo > 0:
        out.write("  >>> SIGNIFICANT <<<\n")
    else:
        out.write("  CI includes zero\n")

    return swing_df


# ============================================================
# MAIN
# ============================================================

def main():
    t_start = time.time()
    print("=" * 70)
    print("D13 Late-Day Swing Strategy v2 (corrected follow-up data)")
    print("=" * 70)

    # ---- Load data ----
    print("\nLoading metadata and mapping...")
    meta, is_events, oos_events = load_and_split()
    mapping = pd.read_parquet(MAPPING_PATH)
    followup_lookup = build_followup_lookup(mapping)
    print(f"  Metadata: {len(meta)} total, IS={len(is_events)}, OOS={len(oos_events)}")
    print(f"  Mapping: {len(mapping)} rows, {len(followup_lookup)} unique events")

    # ---- Pre-load ALL bar summaries into cache ----
    print("\n--- Pre-loading bar data (all events, IS + OOS) ---")
    all_events = pd.concat([is_events, oos_events], ignore_index=True)
    cache = batch_preload_summaries(all_events, followup_lookup, need_gap_day=True)
    print(f"  Cache size: {len(cache)} entries")

    # ---- Open output file ----
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    out = open(OUTPUT_PATH, 'w', encoding='utf-8')

    out.write("D13 LATE-DAY SWING STRATEGY v2 RESULTS (CORRECTED)\n")
    out.write("=" * 70 + "\n")
    out.write(f"Generated: {pd.Timestamp.now()}\n")
    out.write(f"IS period: {IS_START} to {IS_END} ({len(is_events)} events)\n")
    out.write(f"OOS period: {OOS_START} to {OOS_END} ({len(oos_events)} events)\n")
    out.write(f"Follow-up mapping: {len(followup_lookup)} events, offsets 1-10\n")
    out.write(f"Bar data cache: {len(cache)} files loaded\n")
    out.write(f"Entry: RTH close of gap day (also 14:30 variant)\n")
    out.write(f"Trade direction: with gap (long for gap-up, short for gap-down)\n")
    out.write(f"PnL normalized by ADR-20\n\n")

    # ==================================================================
    # PHASE 1: IS PnL
    # ==================================================================
    print("\n--- Phase 1: IS PnL ---")
    is_pnl = extract_pnl_all(is_events, followup_lookup, cache, label="IS")
    phase1_basic_pnl(is_pnl, out)

    # ==================================================================
    # PHASE 2: Filter combinations (IS)
    # ==================================================================
    print("\n--- Phase 2: Filter Combinations (IS) ---")
    is_combos = phase2_filter_combos(is_pnl, out)

    # ==================================================================
    # PHASE 3: OOS validation of top-5
    # ==================================================================
    print("\n--- Phase 3: OOS PnL ---")
    oos_pnl = extract_pnl_all(oos_events, followup_lookup, cache, label="OOS")

    print("\n--- Phase 3: OOS Validation ---")
    phase3_oos_validation(is_combos, oos_pnl, out)

    # ==================================================================
    # PHASE 4: Swing-Trail (IS + OOS)
    # ==================================================================
    print("\n--- Phase 4: Swing-Trail IS ---")
    is_swing = phase4_swing_trail(is_events, followup_lookup, cache, "IS", out)

    print("\n--- Phase 4: Swing-Trail OOS ---")
    oos_swing = phase4_swing_trail(oos_events, followup_lookup, cache, "OOS", out)

    # ==================================================================
    # FINAL SUMMARY
    # ==================================================================
    out.write("\n\n" + "=" * 70 + "\n")
    out.write("FINAL SUMMARY\n")
    out.write("=" * 70 + "\n\n")

    top_is = [c for c in is_combos if c['N'] >= 30 and c['mean'] > 0]
    top_is.sort(key=lambda x: x['sharpe'], reverse=True)
    out.write("Top 10 IS Filter Combinations (by Sharpe, N>=30, Mean>0):\n")
    out.write(f"{'#':<4} {'N':<6} {'Mean':<8} {'WR':<7} {'Sharpe':<8} {'PF':<7} Description\n")
    out.write("-" * 110 + "\n")
    for i, c in enumerate(top_is[:10]):
        out.write(f"{i+1:<4} {c['N']:<6} {c['mean']:<8.4f} {c['win_rate']:<7.1%} {c['sharpe']:<8.3f} {c['profit_factor']:<7.2f} {c['desc']}\n")

    out.write(f"\nSwing-Trail Summary:\n")
    if len(is_swing) > 0:
        is_sr = compute_stats(is_swing['pnl_r'].values)
        out.write(f"  IS:  N={is_sr['N']}, Mean={is_sr['mean']:.4f}R, WR={is_sr['win_rate']:.1%}, Sharpe={is_sr['sharpe']:.3f}\n")
    if len(oos_swing) > 0:
        oos_sr = compute_stats(oos_swing['pnl_r'].values)
        out.write(f"  OOS: N={oos_sr['N']}, Mean={oos_sr['mean']:.4f}R, WR={oos_sr['win_rate']:.1%}, Sharpe={oos_sr['sharpe']:.3f}\n")

        mean_val, ci_lo, ci_hi = bootstrap_ci(oos_swing['pnl_r'].values, n_boot=10000)
        out.write(f"  OOS Bootstrap 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]\n")

    elapsed = time.time() - t_start
    out.write(f"\nTotal runtime: {elapsed:.1f}s\n")
    out.close()

    print(f"\n{'='*70}")
    print(f"DONE. Results saved to: {OUTPUT_PATH}")
    print(f"Total runtime: {elapsed:.1f}s")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
