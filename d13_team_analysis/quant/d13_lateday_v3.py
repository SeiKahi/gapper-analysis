"""
D13 Late-Day Pattern-Confirmed Swing v3
=========================================
NEW HYPOTHESIS: Enter in last 90 min of Gap Day (Day 0)
ONLY if intraday price action confirms sustained interest.
Hold overnight or multi-day using follow-up day data.

DIFFERENT FROM:
- d13_lateday_v2.py: Blind close entry, no pattern confirmation -> FAILED
- d13_discretionary_led.py: Pattern-based but same-day TRAIL_D exit

KEY INNOVATION: Combine intraday pattern confirmation WITH overnight/multi-day hold.

Phases:
  1. Pattern Recognition + Fixed-Hold PnL (IS only)
  2. Filter Sweep with metadata (IS only)
  3. Devils Advocate Robustheit (IS only)
  4. OOS Validation (survivors only)
  5. Multi-Day Trailing Stop TRAIL_D (IS + OOS)
"""
import pandas as pd
import numpy as np
import os, sys, warnings, time
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy import stats as sp_stats

warnings.filterwarnings('ignore')

# ============================================================
# PATHS & CONSTANTS
# ============================================================
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASE = PROJECT_ROOT
RAW_DIR = BASE / 'data/raw_1min'
VWAP_DIR = BASE / 'data/vwap'
META_PATH = BASE / 'data/metadata/metadata_v9.parquet'
MAPPING_PATH = BASE / 'data/metadata/followup_day_mapping.parquet'
OUT_PATH = BASE / 'd13_team_analysis/results/d13_lateday_v3_results.txt'

IS_START = '2021-02-21'
IS_END   = '2023-12-31'
OOS_START = '2024-01-01'
OOS_END   = '2026-02-21'
IS_H1_END = '2022-06-30'
IS_H2_START = '2022-07-01'

ENTRY_TIMES = ['14:30', '15:00', '15:30', '15:59']
HOLD_PERIODS = [
    ('overnight', 1, 'open'),
    ('D1', 1, 'close'), ('D2', 2, 'close'), ('D3', 3, 'close'),
    ('D5', 5, 'close'), ('D10', 10, 'close'),
]
REF_SL_ADR = 0.50  # Reference SL for R conversion in fixed-hold analysis

lines = []
np.random.seed(42)


# ============================================================
# HELPERS
# ============================================================
def log(msg):
    lines.append(str(msg))
    print(msg)

def log_header(title):
    log(f"\n{'='*80}")
    log(f"  {title}")
    log(f"{'='*80}")

def compute_full_stats(pnl_arr, label=""):
    """Comprehensive stats: N, EV, WR, PF, Sharpe, MaxDD, MaxConsec, p-value."""
    arr = np.asarray(pnl_arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 5:
        return {'N': len(arr), 'label': label, 'valid': False}
    ev = np.mean(arr)
    med = np.median(arr)
    std = np.std(arr, ddof=1)
    wr = np.mean(arr > 0)
    se = std / np.sqrt(len(arr))
    t_stat = ev / se if se > 0 else 0
    p_val = 2 * (1 - sp_stats.t.cdf(abs(t_stat), df=len(arr)-1))
    sharpe = ev / std if std > 0 else 0
    wins = arr[arr > 0]
    losses = arr[arr <= 0]
    pf = abs(np.sum(wins) / np.sum(losses)) if len(losses) > 0 and np.sum(losses) != 0 else float('inf')
    # Max Drawdown
    cum = np.cumsum(arr)
    peak = np.maximum.accumulate(cum)
    max_dd = np.max(peak - cum) if len(cum) > 0 else 0
    # Max Consecutive Wins/Losses
    mcw, mcl, cw, cl = 0, 0, 0, 0
    for r in arr:
        if r > 0: cw += 1; cl = 0
        else: cl += 1; cw = 0
        mcw = max(mcw, cw); mcl = max(mcl, cl)
    return {
        'N': len(arr), 'EV': ev, 'Med': med, 'Std': std, 'WR': wr, 'PF': pf,
        'Sharpe': sharpe, 'MaxDD': max_dd, 'MaxConsecW': mcw, 'MaxConsecL': mcl,
        't_stat': t_stat, 'p_val': p_val, 'label': label, 'valid': True,
        'AvgWin': np.mean(wins) if len(wins) > 0 else 0,
        'AvgLoss': np.mean(losses) if len(losses) > 0 else 0,
        'Max': np.max(arr), 'Min': np.min(arr),
    }

def fmt(s):
    """Compact single-line format."""
    if not s['valid']:
        return f"  {s['label']:<55} | N={s['N']:>4} | [INSUFFICIENT]"
    sig = "***" if s['p_val'] < 0.01 else "**" if s['p_val'] < 0.05 else "*" if s['p_val'] < 0.10 else ""
    return (
        f"  {s['label']:<55} | N={s['N']:>5} | EV={s['EV']:>+7.3f}R | "
        f"WR={s['WR']:>5.1%} | PF={s['PF']:>5.2f} | Sh={s['Sharpe']:>+6.3f} | "
        f"DD={s['MaxDD']:>5.2f}R | CW={s['MaxConsecW']:>2} CL={s['MaxConsecL']:>2} | "
        f"p={s['p_val']:>.4f}{sig}"
    )

def fmt_detail(s):
    """Multi-line detailed format for top results."""
    if not s['valid']:
        return f"  {s['label']}: N={s['N']} [INSUFFICIENT]"
    sig = "***" if s['p_val'] < 0.01 else "**" if s['p_val'] < 0.05 else "*" if s['p_val'] < 0.10 else ""
    return (
        f"  {s['label']}:\n"
        f"    N={s['N']}, EV={s['EV']:+.4f}R, Med={s['Med']:+.4f}R, WR={s['WR']:.1%}, PF={s['PF']:.2f}\n"
        f"    Sharpe={s['Sharpe']:+.4f}, MaxDD={s['MaxDD']:.3f}R, ConsecW={s['MaxConsecW']}, ConsecL={s['MaxConsecL']}\n"
        f"    AvgWin={s['AvgWin']:+.4f}R, AvgLoss={s['AvgLoss']:+.4f}R, Best={s['Max']:+.3f}R, Worst={s['Min']:+.3f}R\n"
        f"    t={s['t_stat']:+.3f}, p={s['p_val']:.6f} {sig}"
    )

def bootstrap_ci(arr, n_boot=10000, ci=0.95):
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 20:
        return (np.nan, np.nan)
    boot = np.array([np.mean(np.random.choice(arr, len(arr), replace=True)) for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    return (np.percentile(boot, 100*alpha), np.percentile(boot, 100*(1-alpha)))


# ============================================================
# DATA LOADING & CACHING
# ============================================================
def build_followup_lookup(mapping_df):
    lookup = {}
    for _, row in mapping_df.iterrows():
        key = (row['ticker'], str(row['gap_date']))
        if key not in lookup:
            lookup[key] = {}
        lookup[key][int(row['offset'])] = str(row['actual_date'])
    return lookup

def extract_bar_summary(ticker, date_str):
    fpath = RAW_DIR / ticker / f'{date_str}.parquet'
    if not fpath.exists():
        return None
    try:
        df = pd.read_parquet(fpath)
        rth = df[df['session'] == 'rth'] if 'session' in df.columns else df
        if len(rth) == 0:
            return None
        return {
            'rth_open': rth.iloc[0]['open'],
            'rth_close': rth.iloc[-1]['close'],
            'rth_high': rth['high'].max(),
            'rth_low': rth['low'].min(),
        }
    except:
        return None

def batch_preload_summaries(events, followup_lookup):
    needed = set()
    for _, row in events.iterrows():
        key = (row['ticker'], str(row['date']))
        fu_map = followup_lookup.get(key, {})
        for offset in range(1, 11):
            fu_date = fu_map.get(offset)
            if fu_date:
                needed.add((row['ticker'], fu_date))

    log(f"  Preloading {len(needed)} followup day summaries...")
    cache = {}
    def _load(item):
        return item, extract_bar_summary(item[0], item[1])
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_load, it): it for it in needed}
        for f in tqdm(as_completed(futures), total=len(futures),
                      desc="  Followup summaries", ncols=100):
            item, result = f.result()
            if result is not None:
                cache[item] = result
    log(f"  Loaded {len(cache)}/{len(needed)} summaries")
    return cache

def batch_preload_day0(events):
    needed = set()
    for _, row in events.iterrows():
        needed.add((row['ticker'], str(row['date'])))

    log(f"  Preloading {len(needed)} Day 0 bar files...")
    cache = {}
    def _load(item):
        ticker, date = item
        fpath = RAW_DIR / ticker / f"{date}.parquet"
        if not fpath.exists():
            return item, None
        try:
            df = pd.read_parquet(fpath)
            rth = df[df['session'] == 'rth'].copy() if 'session' in df.columns else df.copy()
            return item, rth if len(rth) >= 10 else None
        except:
            return item, None
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_load, it): it for it in needed}
        for f in tqdm(as_completed(futures), total=len(futures),
                      desc="  Day0 bars", ncols=100):
            item, result = f.result()
            cache[item] = result
    loaded = sum(1 for v in cache.values() if v is not None)
    log(f"  Loaded {loaded}/{len(needed)} Day 0 bar files")
    return cache

def batch_preload_vwap(events):
    needed = set()
    for _, row in events.iterrows():
        needed.add((row['ticker'], str(row['date'])))

    log(f"  Preloading {len(needed)} VWAP files...")
    cache = {}
    def _load(item):
        ticker, date = item
        fpath = VWAP_DIR / ticker / f"{date}.parquet"
        if not fpath.exists():
            return item, None
        try:
            vd = pd.read_parquet(fpath)
            return item, vd if 'z_score' in vd.columns else None
        except:
            return item, None
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_load, it): it for it in needed}
        for f in tqdm(as_completed(futures), total=len(futures),
                      desc="  VWAP data", ncols=100):
            item, result = f.result()
            cache[item] = result
    loaded = sum(1 for v in cache.values() if v is not None)
    log(f"  Loaded {loaded}/{len(needed)} VWAP files")
    return cache


# ============================================================
# PATTERN DETECTION ON DAY 0
# ============================================================
def analyze_day0_signals(rth, vwap_data, gap_dir, adr):
    """
    Analyze Day 0 intraday price action at each ENTRY_TIME.
    Returns dict: {entry_time: signal_dict_or_None}.
    """
    sign = 1.0 if gap_dir == 'up' else -1.0
    day_open = rth.iloc[0]['open']

    # Detect sell-off in first hour (09:36-10:30)
    selloff_detected = False
    selloff_depth = 0.0
    od_bars = rth[(rth['time_et'] >= '09:30') & (rth['time_et'] <= '09:35')]
    if len(od_bars) > 0:
        if gap_dir == 'up':
            od_peak = od_bars['high'].max()
            so_bars = rth[(rth['time_et'] >= '09:36') & (rth['time_et'] <= '10:30')]
            if len(so_bars) > 0:
                selloff_depth = (od_peak - so_bars['low'].min()) / adr
                selloff_detected = selloff_depth >= 0.10
        else:
            od_peak = od_bars['low'].min()
            so_bars = rth[(rth['time_et'] >= '09:36') & (rth['time_et'] <= '10:30')]
            if len(so_bars) > 0:
                selloff_depth = (so_bars['high'].max() - od_peak) / adr
                selloff_detected = selloff_depth >= 0.10

    results = {}
    for et in ENTRY_TIMES:
        bars_to = rth if et == '15:59' else rth[rth['time_et'] <= et]
        if len(bars_to) < 10:
            results[et] = None
            continue

        entry_price = bars_to.iloc[-1]['close']
        hod = bars_to['high'].max()
        lod = bars_to['low'].min()
        day_range = max(hod - lod, 0.001 * entry_price)
        price_vs_range = ((entry_price - lod) / day_range if gap_dir == 'up'
                          else (hod - entry_price) / day_range)
        price_vs_open = (entry_price - day_open) / adr * sign

        # VWAP z-score at entry
        vwap_z = np.nan
        if vwap_data is not None and 'time_et' in vwap_data.columns:
            vbar = vwap_data if et == '15:59' else vwap_data[vwap_data['time_et'] <= et]
            if len(vbar) > 0:
                vwap_z = vbar.iloc[-1]['z_score']

        above_vwap = False
        if not np.isnan(vwap_z):
            above_vwap = (vwap_z > 0 and gap_dir == 'up') or (vwap_z < 0 and gap_dir == 'down')

        # Day 0 remaining bars (for trailing SL check)
        if et == '15:59':
            rem_h, rem_l, rem_c = np.nan, np.nan, np.nan
        else:
            remaining = rth[rth['time_et'] > et]
            if len(remaining) > 0:
                rem_h = remaining['high'].max()
                rem_l = remaining['low'].min()
                rem_c = remaining.iloc[-1]['close']
            else:
                rem_h, rem_l, rem_c = np.nan, np.nan, np.nan

        results[et] = {
            'entry_price': entry_price, 'vwap_z': vwap_z,
            'above_vwap': above_vwap, 'price_vs_range': price_vs_range,
            'price_vs_open': price_vs_open,
            'selloff_detected': selloff_detected, 'selloff_depth': selloff_depth,
            'recovery_vwap': selloff_detected and above_vwap,
            'strong_position': price_vs_range > 0.60,
            'positive_day': price_vs_open > 0,
            'day0_rem_high': rem_h, 'day0_rem_low': rem_l, 'day0_rem_close': rem_c,
        }
    return results


# ============================================================
# BUILD TRADES DATAFRAME
# ============================================================
def build_trades_df(events, followup_lookup, fu_cache, day0_cache, vwap_cache, label):
    """One row per (event x entry_time) with signals + fixed-hold PnL."""
    records = []
    skipped = 0

    for _, row in tqdm(events.iterrows(), total=len(events),
                       desc=f"  {label} trades", ncols=100):
        ticker = row['ticker']
        date = str(row['date'])
        gap_dir = row['gap_direction']
        adr = row['adr_10']
        sign = 1.0 if gap_dir == 'up' else -1.0

        rth = day0_cache.get((ticker, date))
        if rth is None:
            skipped += 1
            continue
        vwap = vwap_cache.get((ticker, date))
        signals = analyze_day0_signals(rth, vwap, gap_dir, adr)

        key = (ticker, date)
        fu_map = followup_lookup.get(key, {})

        for et in ENTRY_TIMES:
            sig = signals.get(et)
            if sig is None:
                continue

            entry_price = sig['entry_price']

            # Compute fixed-hold PnL
            pnl = {}
            for hold_label, offset, price_type in HOLD_PERIODS:
                fu_date = fu_map.get(offset)
                if fu_date is None:
                    pnl[hold_label] = np.nan
                    continue
                summary = fu_cache.get((ticker, fu_date))
                if summary is None:
                    pnl[hold_label] = np.nan
                    continue
                exit_p = summary['rth_open'] if price_type == 'open' else summary['rth_close']
                pnl[hold_label] = (exit_p - entry_price) / adr * sign

            rec = {
                'ticker': ticker, 'date': date, 'gap_dir': gap_dir,
                'sign': sign, 'adr': adr,
                'gap_size_in_adr': row.get('gap_size_in_adr', np.nan),
                'od_strength': row.get('od_strength', np.nan),
                'od_direction': row.get('od_direction', ''),
                'gap_filled': row.get('gap_filled', None),
                'rvol_5': row.get('rvol_5', np.nan),
                'sector': row.get('sector', ''),
                'entry_time': et, 'entry_price': entry_price,
            }
            for k2, v2 in sig.items():
                if k2 != 'entry_price':
                    rec[k2] = v2
            for k2, v2 in pnl.items():
                rec[f'pnl_{k2}'] = v2

            records.append(rec)

    log(f"  {label}: {len(records)} trade rows from {len(events)} events ({skipped} skipped)")
    return pd.DataFrame(records)


# ============================================================
# MULTI-DAY TRAILING STOP (TRAIL_D ADAPTED)
# ============================================================
def simulate_multiday_trail(entry_price, sl_adr_mult, adr, sign,
                            day0_rem_high, day0_rem_low,
                            followup_sums, max_days=10):
    """
    TRAIL_D for multi-day: +1R -> BE, trail = peak - 0.5R (daily bars).
    Returns dict with pnl_r, exit_reason, exit_day, etc. or None.
    """
    sl_dist = sl_adr_mult * adr
    if sl_dist <= 0:
        return None
    sl_level = entry_price - sign * sl_dist
    trail_active = False
    highest_r = 0.0

    # Day 0 remaining check (entry to close)
    if not np.isnan(day0_rem_high) and not np.isnan(day0_rem_low):
        if sign > 0:
            if day0_rem_low <= sl_level:
                return {'pnl_r': -1.0, 'exit_reason': 'SL_day0', 'exit_day': 0,
                        'max_fav_r': 0.0, 'trail_active': False}
            highest_r = max(highest_r, (day0_rem_high - entry_price) / sl_dist)
        else:
            if day0_rem_high >= sl_level:
                return {'pnl_r': -1.0, 'exit_reason': 'SL_day0', 'exit_day': 0,
                        'max_fav_r': 0.0, 'trail_active': False}
            highest_r = max(highest_r, (entry_price - day0_rem_low) / sl_dist)

        if highest_r >= 1.0:
            trail_active = True
            sl_level = entry_price
        if trail_active:
            if sign > 0:
                sl_level = max(sl_level, entry_price + (highest_r - 0.5) * sl_dist)
            else:
                sl_level = min(sl_level, entry_price - (highest_r - 0.5) * sl_dist)

    # Follow-up days
    last_close, last_day = None, None
    for i, s in enumerate(followup_sums[:max_days]):
        if s is None:
            continue
        offset = i + 1
        dh, dl, dc = s['rth_high'], s['rth_low'], s['rth_close']
        if np.isnan(dh) or np.isnan(dl) or np.isnan(dc):
            continue
        last_close, last_day = dc, offset

        if sign > 0:  # Long
            if dl <= sl_level:
                return {'pnl_r': (sl_level - entry_price) / sl_dist, 'exit_reason': 'SL_hit',
                        'exit_day': offset, 'max_fav_r': highest_r, 'trail_active': trail_active}
            highest_r = max(highest_r, (dh - entry_price) / sl_dist)
            if highest_r >= 1.0 and not trail_active:
                trail_active = True
                sl_level = entry_price
            if trail_active:
                sl_level = max(sl_level, entry_price + (highest_r - 0.5) * sl_dist)
            if trail_active and dc < sl_level:
                return {'pnl_r': (dc - entry_price) / sl_dist, 'exit_reason': 'trail_close',
                        'exit_day': offset, 'max_fav_r': highest_r, 'trail_active': True}
        else:  # Short
            if dh >= sl_level:
                return {'pnl_r': (entry_price - sl_level) / sl_dist, 'exit_reason': 'SL_hit',
                        'exit_day': offset, 'max_fav_r': highest_r, 'trail_active': trail_active}
            highest_r = max(highest_r, (entry_price - dl) / sl_dist)
            if highest_r >= 1.0 and not trail_active:
                trail_active = True
                sl_level = entry_price
            if trail_active:
                sl_level = min(sl_level, entry_price - (highest_r - 0.5) * sl_dist)
            if trail_active and dc > sl_level:
                return {'pnl_r': (entry_price - dc) / sl_dist, 'exit_reason': 'trail_close',
                        'exit_day': offset, 'max_fav_r': highest_r, 'trail_active': True}

    if last_close is not None:
        pnl = (last_close - entry_price) * sign
        return {'pnl_r': pnl / sl_dist, 'exit_reason': 'max_hold', 'exit_day': last_day,
                'max_fav_r': highest_r, 'trail_active': trail_active}
    return None


# ============================================================
# PATTERN DEFINITIONS
# ============================================================
PATTERNS = {
    'BASELINE':    lambda df: pd.Series(True, index=df.index),
    'VWAP_ABOVE':  lambda df: df['above_vwap'] == True,
    'RECOVERY':    lambda df: df['recovery_vwap'] == True,
    'STRONG_POS':  lambda df: df['strong_position'] == True,
    'POS_DAY':     lambda df: df['positive_day'] == True,
    'FAILED_FADE': lambda df: ((df['od_direction'] == 'against_gap') &
                               (df['gap_filled'] == False) & (df['above_vwap'] == True)),
    'TREND_DAY':   lambda df: ((df['od_direction'] == 'with_gap') &
                               (df['above_vwap'] == True)),
}


# ============================================================
# TRAILING HELPER: Run trail for a subset
# ============================================================
def run_trail_for_subset(df_subset, followup_lookup, fu_cache, sl_mult, max_days):
    """Run multi-day trailing for a DataFrame subset. Returns list of pnl_r."""
    trail_pnl = []
    exit_reasons = []
    exit_days = []
    for _, row in df_subset.iterrows():
        key = (row['ticker'], str(row['date']))
        fu_map = followup_lookup.get(key, {})
        fu_sums = [fu_cache.get((row['ticker'], fu_map.get(o))) if fu_map.get(o) else None
                    for o in range(1, max_days + 1)]

        result = simulate_multiday_trail(
            row['entry_price'], sl_mult, row['adr'], row['sign'],
            row['day0_rem_high'], row['day0_rem_low'], fu_sums, max_days=max_days)
        if result is not None:
            trail_pnl.append(result['pnl_r'])
            exit_reasons.append(result['exit_reason'])
            exit_days.append(result['exit_day'])
    return trail_pnl, exit_reasons, exit_days


# ============================================================
# MAIN
# ============================================================
def main():
    t_start = time.time()

    log("=" * 80)
    log("  D13 LATE-DAY PATTERN-CONFIRMED SWING v3")
    log("=" * 80)
    log(f"  Datum: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    log(f"  IS: {IS_START} bis {IS_END}")
    log(f"  OOS: {OOS_START} bis {OOS_END}")
    log(f"  Entry Times: {ENTRY_TIMES}")
    log(f"  Hold Periods: {[h[0] for h in HOLD_PERIODS]}")
    log(f"  Patterns: {list(PATTERNS.keys())}")
    log(f"  Ref SL: {REF_SL_ADR} ADR (R = PnL_ADR / {REF_SL_ADR})")
    log("")
    log("  HYPOTHESE: Entry in letzten 90 Min des Gap-Tags")
    log("  NUR wenn Intraday-Pattern nachhaltiges Interesse bestaetigt.")
    log("  Hold: Overnight bis Multi-Day (D+1 bis D+10).")
    log("  ANDERS als v2: Pattern-Confirmation statt blindem Close-Entry.")

    # ---- Load data ----
    log_header("DATA LOADING")
    meta = pd.read_parquet(META_PATH)
    IS = meta[(meta['date'] >= IS_START) & (meta['date'] <= IS_END) &
              meta['adr_10'].notna() & (meta['adr_10'] > 0) &
              meta['close_935'].notna()].copy().reset_index(drop=True)
    OOS = meta[(meta['date'] >= OOS_START) & (meta['date'] <= OOS_END) &
               meta['adr_10'].notna() & (meta['adr_10'] > 0) &
               meta['close_935'].notna()].copy().reset_index(drop=True)

    mapping = pd.read_parquet(MAPPING_PATH)
    followup_lookup = build_followup_lookup(mapping)
    log(f"  IS: {len(IS)} events, OOS: {len(OOS)} events")
    log(f"  Followup mapping: {len(followup_lookup)} unique events")

    # Preload followup summaries
    all_events = pd.concat([IS, OOS], ignore_index=True)
    fu_cache = batch_preload_summaries(all_events, followup_lookup)

    # ================================================================
    # IS PROCESSING
    # ================================================================
    log_header("IS: PRELOADING DAY 0 DATA")
    day0_is = batch_preload_day0(IS)
    vwap_is = batch_preload_vwap(IS)

    log_header("PHASE 1: IS PATTERN RECOGNITION + FIXED-HOLD PnL")
    is_trades = build_trades_df(IS, followup_lookup, fu_cache, day0_is, vwap_is, "IS")
    del day0_is, vwap_is

    if len(is_trades) == 0:
        log("  ERROR: No valid IS trades. Aborting.")
        return

    # ---- Phase 1 Analysis ----
    log(f"\n  PnL in R-Vielfachen (1R = {REF_SL_ADR} ADR)")
    hold_cols = [f'pnl_{h[0]}' for h in HOLD_PERIODS]
    total_hyp = 0
    phase1_promising = []

    for pname, pfn in PATTERNS.items():
        log(f"\n  --- Pattern: {pname} ---")
        for et in ENTRY_TIMES:
            mask = pfn(is_trades) & (is_trades['entry_time'] == et)
            subset = is_trades[mask]
            if len(subset) < 10:
                continue
            for hcol in hold_cols:
                vals = subset[hcol].dropna().values / REF_SL_ADR
                label = f"{pname}|{et}|{hcol.replace('pnl_', '')}"
                stats = compute_full_stats(vals, label)
                total_hyp += 1
                if stats['valid'] and stats['EV'] > 0 and stats['N'] >= 30:
                    log(fmt(stats))
                    phase1_promising.append((pname, et, hcol, stats, vals))

    log(f"\n  Phase 1: {total_hyp} Hypothesen, {len(phase1_promising)} vielversprechend (EV>0, N>=30)")

    phase1_promising.sort(key=lambda x: x[3]['EV'], reverse=True)
    log(f"\n  TOP-20 IS (nach EV):")
    for _, (pn, et, hc, s, _) in zip(range(20), phase1_promising):
        log(fmt(s))

    # Direction split
    log(f"\n  --- Richtungs-Split (Top-10 je) ---")
    for gd in ['up', 'down']:
        log(f"\n  Gap {gd.upper()}:")
        dp = []
        for pname, pfn in PATTERNS.items():
            for et in ENTRY_TIMES:
                mask = pfn(is_trades) & (is_trades['entry_time'] == et) & (is_trades['gap_dir'] == gd)
                subset = is_trades[mask]
                if len(subset) < 10:
                    continue
                for hcol in hold_cols:
                    vals = subset[hcol].dropna().values / REF_SL_ADR
                    label = f"{gd}|{pname}|{et}|{hcol.replace('pnl_', '')}"
                    stats = compute_full_stats(vals, label)
                    if stats['valid'] and stats['EV'] > 0 and stats['N'] >= 20:
                        dp.append(stats)
        dp.sort(key=lambda x: x['EV'], reverse=True)
        for s in dp[:10]:
            log(fmt(s))

    # ================================================================
    # PHASE 2: FILTER SWEEP (IS)
    # ================================================================
    log_header("PHASE 2: FILTER SWEEP (IS)")

    base_combos = phase1_promising[:15]
    gap_sizes = [1.0, 2.0, 3.0]
    od_threshs = [0.3, 0.5, 0.7]
    rvol_threshs = [3, 5]
    phase2_results = []

    for pname, et, hcol, base_stats, _ in base_combos:
        pfn = PATTERNS[pname]
        base_mask = pfn(is_trades) & (is_trades['entry_time'] == et)

        # Metadata filter combos
        for gs_min in gap_sizes:
            for od_min in od_threshs:
                for rv_min in rvol_threshs:
                    mask = (base_mask &
                            (is_trades['gap_size_in_adr'] >= gs_min) &
                            (is_trades['od_strength'] >= od_min) &
                            (is_trades['rvol_5'] >= rv_min))
                    vals = is_trades[mask][hcol].dropna().values / REF_SL_ADR
                    label = f"{pname}|{et}|{hcol.replace('pnl_','')}|g>={gs_min}|od>={od_min}|rv>={rv_min}"
                    stats = compute_full_stats(vals, label)
                    total_hyp += 1
                    if stats['valid'] and stats['EV'] > 0.25 and stats['N'] >= 30 and stats['p_val'] < 0.05:
                        phase2_results.append({
                            'pattern': pname, 'entry_time': et, 'hold': hcol,
                            'gap_min': gs_min, 'od_min': od_min, 'rvol_min': rv_min,
                            'stats': stats, 'pnl': vals,
                        })

        # Gap-filled filter
        for gf in [True, False]:
            mask = base_mask & (is_trades['gap_filled'] == gf)
            vals = is_trades[mask][hcol].dropna().values / REF_SL_ADR
            label = f"{pname}|{et}|{hcol.replace('pnl_','')}|filled={gf}"
            stats = compute_full_stats(vals, label)
            total_hyp += 1
            if stats['valid'] and stats['EV'] > 0.25 and stats['N'] >= 30 and stats['p_val'] < 0.05:
                phase2_results.append({
                    'pattern': pname, 'entry_time': et, 'hold': hcol,
                    'gap_min': 0, 'od_min': 0, 'rvol_min': 0, 'gap_filled_filter': gf,
                    'stats': stats, 'pnl': vals,
                })

    phase2_results.sort(key=lambda x: x['stats']['EV'], reverse=True)
    log(f"  Total Hypothesen (Phase 1+2): {total_hyp}")
    log(f"  Phase 2 Survivors (EV>0.25R, N>=30, p<0.05): {len(phase2_results)}")
    log(f"\n  TOP-20 PHASE 2:")
    for r in phase2_results[:20]:
        log(fmt(r['stats']))

    # ================================================================
    # PHASE 3: DEVILS ADVOCATE (IS)
    # ================================================================
    log_header("PHASE 3: DEVILS ADVOCATE ROBUSTHEIT-CHECKS (IS)")
    bonferroni_alpha = 0.05 / max(1, total_hyp)
    log(f"  Total Hypothesen: {total_hyp}, Bonferroni Alpha: {bonferroni_alpha:.8f}")

    phase3_survivors = []
    for r in phase2_results[:30]:
        stats = r['stats']
        pname, et, hcol = r['pattern'], r['entry_time'], r['hold']
        pfn = PATTERNS[pname]
        base_mask = pfn(is_trades) & (is_trades['entry_time'] == et)

        if r.get('gap_min', 0) > 0:
            base_mask = (base_mask &
                         (is_trades['gap_size_in_adr'] >= r['gap_min']) &
                         (is_trades['od_strength'] >= r['od_min']) &
                         (is_trades['rvol_5'] >= r['rvol_min']))
        if 'gap_filled_filter' in r:
            base_mask = base_mask & (is_trades['gap_filled'] == r['gap_filled_filter'])

        checks = {}
        # 1. Bonferroni
        checks['bonf'] = stats['p_val'] < bonferroni_alpha

        # 2. IS H1/H2 Stability
        h1_v = is_trades[base_mask & (is_trades['date'] <= IS_H1_END)][hcol].dropna().values / REF_SL_ADR
        h2_v = is_trades[base_mask & (is_trades['date'] >= IS_H2_START)][hcol].dropna().values / REF_SL_ADR
        ev_h1 = np.mean(h1_v) if len(h1_v) > 0 else -999
        ev_h2 = np.mean(h2_v) if len(h2_v) > 0 else -999
        checks['stab'] = ev_h1 > 0 and ev_h2 > 0
        stab_str = f"H1: {ev_h1:+.3f}R(N={len(h1_v)}), H2: {ev_h2:+.3f}R(N={len(h2_v)})"

        # 3. Concentration
        sub = is_trades[base_mask]
        if len(sub) > 0:
            mt = sub['ticker'].value_counts(normalize=True).iloc[0]
            ms = sub['sector'].value_counts(normalize=True).iloc[0] if 'sector' in sub.columns else 0
        else:
            mt, ms = 0, 0
        checks['conc'] = mt < 0.20 and ms < 0.25
        conc_str = f"MaxTicker={mt:.1%}, MaxSector={ms:.1%}"

        n_pass = sum([checks['bonf'], checks['stab'], checks['conc']])
        log(f"\n  {stats['label']}:")
        log(f"    Bonferroni: {'PASS' if checks['bonf'] else 'FAIL'} (p={stats['p_val']:.8f})")
        log(f"    Stability:  {'PASS' if checks['stab'] else 'FAIL'} ({stab_str})")
        log(f"    Concentr.:  {'PASS' if checks['conc'] else 'FAIL'} ({conc_str})")
        log(f"    DA Score: {n_pass}/3")

        if n_pass >= 2:
            phase3_survivors.append(r)

    log(f"\n  Phase 3 Survivors: {len(phase3_survivors)}")

    # ================================================================
    # PHASE 4: OOS VALIDATION
    # ================================================================
    log_header("PHASE 4: OOS VALIDATION")
    log(f"  Preloading OOS Day 0 data...")
    day0_oos = batch_preload_day0(OOS)
    vwap_oos = batch_preload_vwap(OOS)
    oos_trades = build_trades_df(OOS, followup_lookup, fu_cache, day0_oos, vwap_oos, "OOS")
    del day0_oos, vwap_oos

    # Validate Phase 3 survivors
    if len(phase3_survivors) > 0:
        log(f"\n  --- OOS Validierung fuer {len(phase3_survivors)} Phase-3 Survivors ---")
        for r in phase3_survivors[:15]:
            pname, et, hcol = r['pattern'], r['entry_time'], r['hold']
            pfn = PATTERNS[pname]
            oos_mask = pfn(oos_trades) & (oos_trades['entry_time'] == et)
            if r.get('gap_min', 0) > 0:
                oos_mask = (oos_mask &
                            (oos_trades['gap_size_in_adr'] >= r['gap_min']) &
                            (oos_trades['od_strength'] >= r['od_min']) &
                            (oos_trades['rvol_5'] >= r['rvol_min']))
            if 'gap_filled_filter' in r:
                oos_mask = oos_mask & (oos_trades['gap_filled'] == r['gap_filled_filter'])

            oos_vals = oos_trades[oos_mask][hcol].dropna().values / REF_SL_ADR
            label_oos = f"OOS: {r['stats']['label']}"
            oos_stats = compute_full_stats(oos_vals, label_oos)
            log(fmt_detail(oos_stats))

            if len(oos_vals) >= 20:
                ci_lo, ci_hi = bootstrap_ci(oos_vals)
                log(f"    Bootstrap 95% CI: [{ci_lo:+.4f}R, {ci_hi:+.4f}R]")
                robust = ci_lo > 0
                log(f"    >>> {'ROBUST' if robust else 'NOT ROBUST'}")
                is_ev = r['stats']['EV']
                oos_ev = oos_stats['EV'] if oos_stats['valid'] else 0
                deg = (oos_ev - is_ev) / abs(is_ev) * 100 if is_ev != 0 else 0
                log(f"    IS: {is_ev:+.4f}R -> OOS: {oos_ev:+.4f}R (Degradation: {deg:+.0f}%)")
    else:
        log(f"  Keine Phase-3 Survivors. Zeige Top-10 IS Combos auf OOS:")

    # Always show best IS patterns on OOS (even if they didn't survive Phase 3)
    log(f"\n  --- TOP-10 IS Patterns auf OOS (ohne Metadata-Filter) ---")
    for pn, et, hc, is_s, _ in phase1_promising[:10]:
        pfn = PATTERNS[pn]
        mask = pfn(oos_trades) & (oos_trades['entry_time'] == et)
        vals = oos_trades[mask][hc].dropna().values / REF_SL_ADR
        label = f"OOS_raw: {is_s['label']}"
        s = compute_full_stats(vals, label)
        if s['valid']:
            log(fmt(s))
            if len(vals) >= 20:
                ci_lo, ci_hi = bootstrap_ci(vals)
                log(f"    CI: [{ci_lo:+.4f}, {ci_hi:+.4f}] {'ROBUST' if ci_lo > 0 else ''}")

    # ================================================================
    # PHASE 5: MULTI-DAY TRAILING STOP
    # ================================================================
    log_header("PHASE 5: MULTI-DAY TRAILING STOP (TRAIL_D)")
    log("  +1R -> BE, trail = peak - 0.5R (daily OHLC)")
    log("  Day 0 remaining bars also checked for SL")

    trail_patterns = ['BASELINE', 'VWAP_ABOVE', 'RECOVERY', 'TREND_DAY', 'FAILED_FADE', 'POS_DAY']
    trail_entry_times = ['14:30', '15:00', '15:59']
    trail_sl_mults = [0.25, 0.50]
    trail_max_days_list = [5, 10]

    for ds_label, df_trades in [("IS", is_trades), ("OOS", oos_trades)]:
        if len(df_trades) == 0:
            continue
        log(f"\n  ===== {ds_label} TRAILING =====")

        for sl_mult in trail_sl_mults:
            for max_d in trail_max_days_list:
                log(f"\n  --- SL={sl_mult}ADR, MaxDays={max_d} ---")

                for pname in trail_patterns:
                    pfn = PATTERNS[pname]
                    for et in trail_entry_times:
                        mask = pfn(df_trades) & (df_trades['entry_time'] == et)
                        subset = df_trades[mask]
                        if len(subset) < 20:
                            continue

                        pnl_list, reasons, days = run_trail_for_subset(
                            subset, followup_lookup, fu_cache, sl_mult, max_d)

                        if len(pnl_list) < 10:
                            continue

                        label = f"{ds_label}|{pname}|{et}|SL{sl_mult}|D{max_d}"
                        stats = compute_full_stats(pnl_list, label)
                        if not stats['valid']:
                            continue

                        log(fmt(stats))

                        # For OOS: show CI if significant
                        if ds_label == 'OOS' and stats['EV'] > 0 and len(pnl_list) >= 20:
                            ci_lo, ci_hi = bootstrap_ci(pnl_list)
                            if ci_lo > 0:
                                log(f"    >>> TRAIL ROBUST: CI=[{ci_lo:+.3f}R, {ci_hi:+.3f}R]")

    # Detail for best trailing combos
    log(f"\n  --- Beste Trail-Kombos: Exit-Reason-Verteilung ---")
    for ds_label, df_trades in [("IS", is_trades)]:
        for pname in ['VWAP_ABOVE', 'RECOVERY', 'POS_DAY']:
            pfn = PATTERNS[pname]
            for et in ['14:30', '15:00']:
                mask = pfn(df_trades) & (df_trades['entry_time'] == et)
                subset = df_trades[mask]
                if len(subset) < 30:
                    continue
                pnl_list, reasons, days = run_trail_for_subset(
                    subset, followup_lookup, fu_cache, 0.50, 10)
                if len(pnl_list) < 20:
                    continue
                log(f"\n  {ds_label}|{pname}|{et}|SL0.5|D10 (N={len(pnl_list)}):")
                from collections import Counter
                rc = Counter(reasons)
                for reason, cnt in rc.most_common():
                    idx = [i for i, r in enumerate(reasons) if r == reason]
                    avg_r = np.mean([pnl_list[i] for i in idx])
                    log(f"    {reason}: N={cnt} ({100*cnt/len(pnl_list):.0f}%), AvgPnL={avg_r:+.3f}R")
                dc = Counter(days)
                log(f"    Exit-Day: " + ", ".join(f"D{d}={n}" for d, n in sorted(dc.items())))

    # ================================================================
    # GRAND SUMMARY
    # ================================================================
    log_header("GRAND SUMMARY")
    log(f"""
  HYPOTHESE: Late-Day Entry (letzte 90 Min) mit Pattern-Confirmation
  Hold: Overnight bis Multi-Day

  METHODIK:
  - 7 Patterns: {list(PATTERNS.keys())}
  - 4 Entry Times: {ENTRY_TIMES}
  - 6 Hold Periods: {[h[0] for h in HOLD_PERIODS]}
  - Trailing: TRAIL_D multi-day (+1R->BE, trail peak-0.5R)
  - Total Hypothesen: {total_hyp}
  - Bonferroni Alpha: {bonferroni_alpha:.8f}

  IS: {len(IS)} events -> {len(is_trades)} trade rows
  OOS: {len(OOS)} events -> {len(oos_trades)} trade rows

  Phase 1 vielversprechend (EV>0, N>=30): {len(phase1_promising)}
  Phase 2 Survivors (EV>0.25R, N>=30, p<0.05): {len(phase2_results)}
  Phase 3 DA Survivors (2/3 checks): {len(phase3_survivors)}
""")

    if len(phase1_promising) > 0:
        log("  TOP-5 IS FIXED-HOLD (nach EV):")
        for pn, et, hc, s, _ in phase1_promising[:5]:
            log(f"    {s['label']}: EV={s['EV']:+.4f}R, WR={s['WR']:.1%}, N={s['N']}, "
                f"Sharpe={s['Sharpe']:+.4f}, MaxDD={s['MaxDD']:.2f}R, p={s['p_val']:.6f}")

    if len(phase3_survivors) > 0:
        log(f"\n  PHASE 3 SURVIVORS (OOS-validiert):")
        for r in phase3_survivors[:5]:
            s = r['stats']
            log(f"    {s['label']}: EV={s['EV']:+.4f}R, WR={s['WR']:.1%}, N={s['N']}, p={s['p_val']:.6f}")

    elapsed = time.time() - t_start
    log(f"\n  Laufzeit: {elapsed:.1f}s ({elapsed/60:.1f}min)")

    # Save
    output = "\n".join(lines)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        f.write(output)

    print(f"\n\nResults saved to {OUT_PATH}", file=sys.stderr)
    print("DONE!", file=sys.stderr)


if __name__ == '__main__':
    main()
