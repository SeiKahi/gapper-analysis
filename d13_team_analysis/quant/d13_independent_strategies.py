"""
D13 PHASE 2: UNABHAENGIGE STRATEGIEN - QUANT IS-BACKTEST
=========================================================
6 Strategien die NICHT auf dem Opening Drive basieren.
IS-Periode: 2021-02-21 bis 2023-12-31
Exit: TRAIL_D fuer alle Strategien.

S1: VWAP Sigma-2 Mean Reversion
S2: No-Fill Power Move
S3: Late-Session Momentum (13:00 Entry)
S4: PM/RTH Ratio Extreme
S5: ORB-Aligned Breakout (15-min Range)
S6: VPOC Migration + Gap Continuation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PATHS
# ============================================================
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASE_DIR = PROJECT_ROOT
RAW_DIR = BASE_DIR / 'data' / 'raw_1min'
VWAP_DIR = BASE_DIR / 'data' / 'vwap'
VP_DIR = BASE_DIR / 'data' / 'volume_profile'
META_PATH = BASE_DIR / 'data' / 'metadata' / 'metadata_v9.parquet'
OUT_DIR = BASE_DIR / 'd13_team_analysis' / 'results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# TRAILING STOP SIMULATION (from d10_trailing.py)
# ============================================================

def simulate_trailing(bars_rth, entry_price, sl_dist, trade_dir, adr, trail_type='D'):
    """
    Trailing-stop simulation. TRAIL_D default.
    bars_rth must have columns: time_et, high, low, close
    Filters to time_et >= '09:36' up to '15:55' internally.
    """
    if sl_dist <= 0 or pd.isna(sl_dist) or adr <= 0:
        return None

    post_entry = bars_rth[(bars_rth['time_et'] >= '09:36') & (bars_rth['time_et'] <= '15:55')]
    if len(post_entry) == 0:
        return None

    if trade_dir == 'long':
        sl_level = entry_price - sl_dist
    else:
        sl_level = entry_price + sl_dist

    highest_r = 0.0
    trail_active = False
    exit_price = None
    exit_type = 'timeout'
    exit_time = None

    for _, bar in post_entry.iterrows():
        if trade_dir == 'long':
            current_r = max(0, (bar['high'] - entry_price) / sl_dist)
        else:
            current_r = max(0, (entry_price - bar['low']) / sl_dist)
        highest_r = max(highest_r, current_r)

        if trail_type == 'D':
            # After +1R: trail to BE, then trail = peak - 0.5R (tight)
            if highest_r >= 1.0 and not trail_active:
                trail_active = True
                sl_level = entry_price  # BE
            if trail_active:
                if trade_dir == 'long':
                    new_sl = entry_price + (highest_r - 0.5) * sl_dist
                    sl_level = max(sl_level, new_sl)
                else:
                    new_sl = entry_price - (highest_r - 0.5) * sl_dist
                    sl_level = min(sl_level, new_sl)

        # SL check
        if trade_dir == 'long' and bar['low'] <= sl_level:
            exit_price = sl_level
            exit_type = 'trail_sl' if trail_active else 'initial_sl'
            exit_time = bar['time_et']
            break
        elif trade_dir == 'short' and bar['high'] >= sl_level:
            exit_price = sl_level
            exit_type = 'trail_sl' if trail_active else 'initial_sl'
            exit_time = bar['time_et']
            break

    if exit_price is None:
        last_bar = post_entry.iloc[-1]
        exit_price = last_bar['close']
        exit_type = 'timeout'

    if trade_dir == 'long':
        pnl = exit_price - entry_price
    else:
        pnl = entry_price - exit_price

    return {
        'pnl': pnl,
        'pnl_r': pnl / sl_dist if sl_dist > 0 else 0,
        'pnl_adr': pnl / adr if adr > 0 else 0,
        'exit_type': exit_type,
        'exit_time': exit_time,
        'highest_r': highest_r,
        'trail_active': trail_active,
        'winner': pnl > 0,
    }


def simulate_trailing_from(bars_rth, entry_time, entry_price, sl_dist, trade_dir, adr):
    """Wrapper for late entries - filters bars to start from entry_time + 1 min."""
    filtered = bars_rth[bars_rth['time_et'] > entry_time].copy()
    if len(filtered) == 0:
        return None
    # Override the internal 09:36 filter by ensuring we pass bars already filtered
    # We replicate the trailing logic here directly
    post_entry = filtered[filtered['time_et'] <= '15:55']
    if len(post_entry) == 0:
        return None

    if sl_dist <= 0 or pd.isna(sl_dist) or adr <= 0:
        return None

    if trade_dir == 'long':
        sl_level = entry_price - sl_dist
    else:
        sl_level = entry_price + sl_dist

    highest_r = 0.0
    trail_active = False
    exit_price = None
    exit_type = 'timeout'
    exit_time = None

    for _, bar in post_entry.iterrows():
        if trade_dir == 'long':
            current_r = max(0, (bar['high'] - entry_price) / sl_dist)
        else:
            current_r = max(0, (entry_price - bar['low']) / sl_dist)
        highest_r = max(highest_r, current_r)

        # TRAIL_D
        if highest_r >= 1.0 and not trail_active:
            trail_active = True
            sl_level = entry_price
        if trail_active:
            if trade_dir == 'long':
                new_sl = entry_price + (highest_r - 0.5) * sl_dist
                sl_level = max(sl_level, new_sl)
            else:
                new_sl = entry_price - (highest_r - 0.5) * sl_dist
                sl_level = min(sl_level, new_sl)

        if trade_dir == 'long' and bar['low'] <= sl_level:
            exit_price = sl_level
            exit_type = 'trail_sl' if trail_active else 'initial_sl'
            exit_time = bar['time_et']
            break
        elif trade_dir == 'short' and bar['high'] >= sl_level:
            exit_price = sl_level
            exit_type = 'trail_sl' if trail_active else 'initial_sl'
            exit_time = bar['time_et']
            break

    if exit_price is None:
        last_bar = post_entry.iloc[-1]
        exit_price = last_bar['close']
        exit_type = 'timeout'

    if trade_dir == 'long':
        pnl = exit_price - entry_price
    else:
        pnl = entry_price - exit_price

    return {
        'pnl': pnl,
        'pnl_r': pnl / sl_dist if sl_dist > 0 else 0,
        'pnl_adr': pnl / adr if adr > 0 else 0,
        'exit_type': exit_type,
        'exit_time': exit_time,
        'highest_r': highest_r,
        'trail_active': trail_active,
        'winner': pnl > 0,
    }


# ============================================================
# BOOTSTRAP P-VALUE
# ============================================================

def bootstrap_pvalue(pnl_series, n_resamples=1000, rng_seed=42):
    """Bootstrap test: is mean PnL significantly > 0?"""
    arr = np.array(pnl_series)
    n = len(arr)
    if n < 5:
        return np.nan
    observed_mean = np.mean(arr)
    rng = np.random.RandomState(rng_seed)
    count_below = 0
    for _ in range(n_resamples):
        sample = rng.choice(arr, size=n, replace=True)
        if np.mean(sample) <= 0:
            count_below += 1
    p_value = count_below / n_resamples
    return p_value


# ============================================================
# SUMMARIZE STRATEGY
# ============================================================

def summarize(trades_df, label, lines, show_variants=False, variant_label=""):
    """Full summary of a strategy."""
    n = len(trades_df)

    prefix = f"  {variant_label}" if variant_label else "  "

    if n < 30:
        lines.append(f"{prefix}N: {n}  --> N < 30, keine belastbare Aussage moeglich")
        if n > 0:
            lines.append(f"{prefix}EV: {trades_df['pnl_adr'].mean():+.4f} ADR (nicht signifikant)")
        return {'n': n, 'ev': np.nan, 'wr': np.nan, 'pf': np.nan}

    ev = trades_df['pnl_adr'].mean()
    wr = trades_df['winner'].mean()
    med_r = trades_df['pnl_r'].median()
    mean_r = trades_df['pnl_r'].mean()

    winners = trades_df[trades_df['pnl_adr'] > 0]['pnl_adr']
    losers = trades_df[trades_df['pnl_adr'] < 0]['pnl_adr']
    pf = abs(winners.sum() / losers.sum()) if len(losers) > 0 and losers.sum() != 0 else np.inf

    # Exit types
    n_init = (trades_df['exit_type'] == 'initial_sl').sum()
    n_trail = (trades_df['exit_type'] == 'trail_sl').sum()
    n_to = (trades_df['exit_type'] == 'timeout').sum()

    # Bootstrap
    p_val = bootstrap_pvalue(trades_df['pnl_adr'].values)

    lines.append(f"{prefix}N: {n}")
    lines.append(f"{prefix}EV: {ev:+.4f} ADR  (p={p_val:.3f})")
    lines.append(f"{prefix}WR: {wr*100:.1f}%")
    lines.append(f"{prefix}Median PnL: {med_r:+.3f}R | Mean PnL: {mean_r:+.3f}R")
    lines.append(f"{prefix}Profit Factor: {pf:.2f}")
    lines.append(f"{prefix}Exit Types: InitSL={n_init}({100*n_init/n:.0f}%), TrailSL={n_trail}({100*n_trail/n:.0f}%), Timeout={n_to}({100*n_to/n:.0f}%)")
    lines.append(f"{prefix}Highest R (mean): {trades_df['highest_r'].mean():.2f}")

    # GapUp/GapDn split if available
    if 'gap_direction' in trades_df.columns:
        for gd, gl in [('up', 'GapUp'), ('down', 'GapDn')]:
            sub = trades_df[trades_df['gap_direction'] == gd]
            if len(sub) >= 10:
                lines.append(f"{prefix}  {gl}: N={len(sub)}, EV={sub['pnl_adr'].mean():+.4f} ADR, WR={sub['winner'].mean()*100:.1f}%")
            else:
                lines.append(f"{prefix}  {gl}: N={len(sub)} (zu wenig)")

    # Yearly split
    if 'year' in trades_df.columns:
        for yr in sorted(trades_df['year'].unique()):
            sub = trades_df[trades_df['year'] == yr]
            if len(sub) >= 5:
                lines.append(f"{prefix}  {yr}: N={len(sub)}, EV={sub['pnl_adr'].mean():+.4f} ADR, WR={sub['winner'].mean()*100:.1f}%")

    return {'n': n, 'ev': ev, 'wr': wr, 'pf': pf, 'p': p_val, 'label': label}


# ============================================================
# LOAD DATA HELPERS
# ============================================================

def load_metadata_is():
    meta = pd.read_parquet(META_PATH)
    is_df = meta[(meta['date'] >= '2021-02-21') & (meta['date'] <= '2023-12-31')].copy()
    is_df['date_str'] = is_df['date'].astype(str)
    is_df['year'] = pd.to_datetime(is_df['date']).dt.year
    return is_df


def load_1min(ticker, date_str):
    fpath = RAW_DIR / ticker / f"{date_str}.parquet"
    if not fpath.exists():
        return None
    try:
        bars = pd.read_parquet(fpath)
        rth = bars[bars['session'] == 'rth'].copy()
        if len(rth) < 10:
            return None
        return rth
    except:
        return None


def load_vwap(ticker, date_str):
    fpath = VWAP_DIR / ticker / f"{date_str}.parquet"
    if not fpath.exists():
        return None
    try:
        return pd.read_parquet(fpath)
    except:
        return None


def load_vp(ticker, date_str):
    fpath = VP_DIR / ticker / f"{date_str}.parquet"
    if not fpath.exists():
        return None
    try:
        return pd.read_parquet(fpath)
    except:
        return None


# ============================================================
# S1: VWAP Sigma-2 Mean Reversion
# ============================================================

def run_s1(is_df, z_threshold=2.0):
    """
    VWAP Sigma-2 Mean Reversion
    - Load VWAP for each IS event
    - Find FIRST bar where |z_score| >= z_threshold (09:45 to 15:00)
    - Entry: close of NEXT bar
    - Direction: AGAINST extension (z>=2 -> short, z<=-2 -> long)
    - SL: 0.25 * ADR
    - Exit: TRAIL_D from entry time
    """
    print(f"\n=== S1: VWAP Sigma-2 Mean Reversion (z>={z_threshold}) ===", file=sys.stderr)
    results = []
    n_total = len(is_df)

    for idx, (_, row) in enumerate(is_df.iterrows()):
        if idx % 500 == 0:
            print(f"  S1 progress: {idx}/{n_total}", file=sys.stderr)

        ticker = row['ticker']
        date_str = row['date_str']
        adr = row['adr_10']
        if pd.isna(adr) or adr <= 0:
            continue

        vwap_df = load_vwap(ticker, date_str)
        if vwap_df is None:
            continue

        rth = load_1min(ticker, date_str)
        if rth is None:
            continue

        # Filter VWAP to 09:45 - 15:00
        vwap_window = vwap_df[(vwap_df['time_et'] >= '09:45') & (vwap_df['time_et'] <= '15:00')].copy()
        if len(vwap_window) == 0:
            continue

        # Find first bar where |z_score| >= threshold
        trigger_bar = None
        for _, vbar in vwap_window.iterrows():
            z = vbar['z_score']
            if pd.notna(z) and abs(z) >= z_threshold:
                trigger_bar = vbar
                break

        if trigger_bar is None:
            continue

        trigger_time = trigger_bar['time_et']
        z_val = trigger_bar['z_score']

        # Entry = close of next bar in 1-min data
        next_bars = rth[rth['time_et'] > trigger_time]
        if len(next_bars) == 0:
            continue
        entry_bar = next_bars.iloc[0]
        entry_price = entry_bar['close']
        entry_time = entry_bar['time_et']

        if pd.isna(entry_price) or entry_price <= 0:
            continue

        # Direction: against extension
        trade_dir = 'short' if z_val >= z_threshold else 'long'
        sl_dist = 0.25 * adr

        # Simulate from entry time
        result = simulate_trailing_from(rth, entry_time, entry_price, sl_dist, trade_dir, adr)
        if result is None:
            continue

        result['ticker'] = ticker
        result['date'] = date_str
        result['gap_direction'] = row['gap_direction']
        result['year'] = row['year']
        result['z_trigger'] = z_val
        result['entry_time'] = entry_time
        results.append(result)

    print(f"  S1 done: {len(results)} trades", file=sys.stderr)
    return pd.DataFrame(results)


# ============================================================
# S2: No-Fill Power Move
# ============================================================

def run_s2(is_df, min_gap_adr=2.0, min_rvol60=3.0):
    """
    No-Fill Power Move
    - gap_filled == False OR gap_fill_time_minutes > 90
    - gap_size_in_adr >= min_gap_adr
    - rvol_at_time_60min >= min_rvol60
    - Entry: close_1100
    - Direction: gap_direction
    - SL: distance from entry to LOD(GapUp)/HOD(GapDn) until 11:00, min 0.15 ADR
    - Exit: TRAIL_D from 11:01
    """
    print(f"\n=== S2: No-Fill Power Move (gap>={min_gap_adr} ADR, rvol60>={min_rvol60}) ===", file=sys.stderr)

    # Pre-filter
    cond_fill = (is_df['gap_filled'] == False) | (is_df['gap_fill_time_minutes'] > 90)
    cond_gap = is_df['gap_size_in_adr'] >= min_gap_adr
    cond_rvol = is_df['rvol_at_time_60min'] >= min_rvol60
    subset = is_df[cond_fill & cond_gap & cond_rvol].copy()
    print(f"  S2 filtered: {len(subset)} events", file=sys.stderr)

    results = []
    for idx, (_, row) in enumerate(subset.iterrows()):
        if idx % 500 == 0 and idx > 0:
            print(f"  S2 progress: {idx}/{len(subset)}", file=sys.stderr)

        ticker = row['ticker']
        date_str = row['date_str']
        adr = row['adr_10']
        entry_price = row['close_1100']
        gap_dir = row['gap_direction']

        if pd.isna(adr) or adr <= 0 or pd.isna(entry_price) or entry_price <= 0:
            continue

        trade_dir = 'long' if gap_dir == 'up' else 'short'

        # Load 1-min bars for LOD/HOD calculation
        rth = load_1min(ticker, date_str)
        if rth is None:
            continue

        # Calculate LOD/HOD until 11:00
        morning = rth[(rth['time_et'] >= '09:30') & (rth['time_et'] <= '11:00')]
        if len(morning) == 0:
            continue

        if trade_dir == 'long':
            lod = morning['low'].min()
            sl_dist = max(entry_price - lod, 0.15 * adr)
        else:
            hod = morning['high'].max()
            sl_dist = max(hod - entry_price, 0.15 * adr)

        # Simulate from 11:01
        result = simulate_trailing_from(rth, '11:00', entry_price, sl_dist, trade_dir, adr)
        if result is None:
            continue

        result['ticker'] = ticker
        result['date'] = date_str
        result['gap_direction'] = gap_dir
        result['year'] = row['year']
        result['sl_dist_adr'] = sl_dist / adr
        results.append(result)

    print(f"  S2 done: {len(results)} trades", file=sys.stderr)
    return pd.DataFrame(results)


# ============================================================
# S3: Late-Session Momentum (13:00 Entry)
# ============================================================

def run_s3(is_df):
    """
    Late-Session Momentum
    - At 13:00: GapUp -> close > vwap AND close > rth_open -> Long
    -           GapDn -> close < vwap AND close < rth_open -> Short
    - Entry: close of 13:00 bar
    - SL: 0.25 * ADR
    - Exit: TRAIL_D from 13:01
    """
    print(f"\n=== S3: Late-Session Momentum (13:00) ===", file=sys.stderr)
    results = []
    n_total = len(is_df)

    for idx, (_, row) in enumerate(is_df.iterrows()):
        if idx % 500 == 0:
            print(f"  S3 progress: {idx}/{n_total}", file=sys.stderr)

        ticker = row['ticker']
        date_str = row['date_str']
        adr = row['adr_10']
        gap_dir = row['gap_direction']
        rth_open = row['rth_open']

        if pd.isna(adr) or adr <= 0 or pd.isna(rth_open):
            continue

        # Load 1-min bars and VWAP
        rth = load_1min(ticker, date_str)
        if rth is None:
            continue

        vwap_df = load_vwap(ticker, date_str)
        if vwap_df is None:
            continue

        # Get 13:00 bar from 1-min
        bar_1300 = rth[rth['time_et'] == '13:00']
        if len(bar_1300) == 0:
            continue
        close_1300 = bar_1300.iloc[0]['close']

        # Get VWAP at 13:00
        vwap_1300 = vwap_df[vwap_df['time_et'] == '13:00']
        if len(vwap_1300) == 0:
            continue
        vwap_val = vwap_1300.iloc[0]['vwap']

        if pd.isna(close_1300) or pd.isna(vwap_val):
            continue

        # Entry condition
        trade_dir = None
        if gap_dir == 'up' and close_1300 > vwap_val and close_1300 > rth_open:
            trade_dir = 'long'
        elif gap_dir == 'down' and close_1300 < vwap_val and close_1300 < rth_open:
            trade_dir = 'short'

        if trade_dir is None:
            continue

        entry_price = close_1300
        sl_dist = 0.25 * adr

        result = simulate_trailing_from(rth, '13:00', entry_price, sl_dist, trade_dir, adr)
        if result is None:
            continue

        result['ticker'] = ticker
        result['date'] = date_str
        result['gap_direction'] = gap_dir
        result['year'] = row['year']
        results.append(result)

    print(f"  S3 done: {len(results)} trades", file=sys.stderr)
    return pd.DataFrame(results)


# ============================================================
# S4: PM/RTH Ratio Extreme
# ============================================================

def run_s4(is_df, pm_rth5_max=0.10, min_gap_adr=1.5):
    """
    PM/RTH Ratio Extreme
    - pm_rth5 < pm_rth5_max (very low premarket volume)
    - gap_size_in_adr >= min_gap_adr
    - Entry: close_935
    - Direction: gap_direction
    - SL: 0.25 * ADR
    - Exit: TRAIL_D
    """
    print(f"\n=== S4: PM/RTH Ratio Extreme (pm_rth5<{pm_rth5_max}, gap>={min_gap_adr}) ===", file=sys.stderr)

    subset = is_df[(is_df['pm_rth5'] < pm_rth5_max) & (is_df['gap_size_in_adr'] >= min_gap_adr)].copy()
    print(f"  S4 filtered: {len(subset)} events", file=sys.stderr)

    results = []
    for idx, (_, row) in enumerate(subset.iterrows()):
        ticker = row['ticker']
        date_str = row['date_str']
        adr = row['adr_10']
        entry_price = row['close_935']
        gap_dir = row['gap_direction']

        if pd.isna(adr) or adr <= 0 or pd.isna(entry_price) or entry_price <= 0:
            continue

        trade_dir = 'long' if gap_dir == 'up' else 'short'
        sl_dist = 0.25 * adr

        # Load 1-min bars
        rth = load_1min(ticker, date_str)
        if rth is None:
            continue

        # Standard TRAIL_D from 09:36
        result = simulate_trailing(rth, entry_price, sl_dist, trade_dir, adr, trail_type='D')
        if result is None:
            continue

        result['ticker'] = ticker
        result['date'] = date_str
        result['gap_direction'] = gap_dir
        result['year'] = row['year']
        results.append(result)

    print(f"  S4 done: {len(results)} trades", file=sys.stderr)
    return pd.DataFrame(results)


# ============================================================
# S5: ORB-Aligned Breakout (15-min Range)
# ============================================================

def run_s5(is_df):
    """
    ORB-Aligned Breakout
    - Opening Range = High/Low of first 15 bars (09:30-09:44)
    - First break (close > OR_high or close < OR_low) from bar 15 (09:45)
    - Direction must align with gap_direction
    - SL: OR width (capped 0.1 to 0.5 ADR)
    - Exit: TRAIL_D from break time
    """
    print(f"\n=== S5: ORB-Aligned Breakout (15-min) ===", file=sys.stderr)
    results = []
    n_total = len(is_df)

    for idx, (_, row) in enumerate(is_df.iterrows()):
        if idx % 500 == 0:
            print(f"  S5 progress: {idx}/{n_total}", file=sys.stderr)

        ticker = row['ticker']
        date_str = row['date_str']
        adr = row['adr_10']
        gap_dir = row['gap_direction']

        if pd.isna(adr) or adr <= 0:
            continue

        rth = load_1min(ticker, date_str)
        if rth is None:
            continue

        # Opening Range: first 15 bars (09:30-09:44)
        or_bars = rth[(rth['time_et'] >= '09:30') & (rth['time_et'] <= '09:44')]
        if len(or_bars) < 10:
            continue

        or_high = or_bars['high'].max()
        or_low = or_bars['low'].min()
        or_width = or_high - or_low

        if or_width <= 0:
            continue

        # Clamp SL to [0.1, 0.5] ADR
        sl_dist = max(0.1 * adr, min(or_width, 0.5 * adr))

        # Search for first break from 09:45
        post_or = rth[rth['time_et'] >= '09:45']
        if len(post_or) == 0:
            continue

        break_bar = None
        break_dir = None
        for _, bar in post_or.iterrows():
            if bar['close'] > or_high:
                break_bar = bar
                break_dir = 'long'
                break
            elif bar['close'] < or_low:
                break_bar = bar
                break_dir = 'short'
                break

        if break_bar is None or break_dir is None:
            continue

        # Must align with gap direction
        expected_dir = 'long' if gap_dir == 'up' else 'short'
        if break_dir != expected_dir:
            continue

        entry_price = break_bar['close']
        entry_time = break_bar['time_et']

        if pd.isna(entry_price) or entry_price <= 0:
            continue

        result = simulate_trailing_from(rth, entry_time, entry_price, sl_dist, break_dir, adr)
        if result is None:
            continue

        result['ticker'] = ticker
        result['date'] = date_str
        result['gap_direction'] = gap_dir
        result['year'] = row['year']
        result['entry_time'] = entry_time
        result['or_width_adr'] = or_width / adr
        results.append(result)

    print(f"  S5 done: {len(results)} trades", file=sys.stderr)
    return pd.DataFrame(results)


# ============================================================
# S6: VPOC Migration + Gap Continuation
# ============================================================

def run_s6(is_df):
    """
    VPOC Migration + Gap Continuation
    - VPOC at 10:00 (minutes_since_open=30) vs VPOC at 11:00 (minutes_since_open=90)
    - Migration > 0.3 * ADR in gap direction
    - Entry: close_1100
    - SL: VAL@11:00 (GapUp) or VAH@11:00 (GapDn) + 0.1 ADR buffer
    - Exit: TRAIL_D from 11:01
    """
    print(f"\n=== S6: VPOC Migration + Gap Continuation ===", file=sys.stderr)
    results = []
    n_total = len(is_df)

    for idx, (_, row) in enumerate(is_df.iterrows()):
        if idx % 500 == 0:
            print(f"  S6 progress: {idx}/{n_total}", file=sys.stderr)

        ticker = row['ticker']
        date_str = row['date_str']
        adr = row['adr_10']
        gap_dir = row['gap_direction']
        entry_price = row['close_1100']

        if pd.isna(adr) or adr <= 0 or pd.isna(entry_price) or entry_price <= 0:
            continue

        vp_df = load_vp(ticker, date_str)
        if vp_df is None:
            continue

        # Get VPOC at 10:00 (minutes_since_open=30) and 11:00 (minutes_since_open=90)
        vp_10 = vp_df[vp_df['minutes_since_open'] == 30]
        vp_11 = vp_df[vp_df['minutes_since_open'] == 90]

        if len(vp_10) == 0 or len(vp_11) == 0:
            continue

        vpoc_10 = vp_10.iloc[0]['vpoc']
        vpoc_11 = vp_11.iloc[0]['vpoc']
        val_11 = vp_11.iloc[0]['val']
        vah_11 = vp_11.iloc[0]['vah']

        if pd.isna(vpoc_10) or pd.isna(vpoc_11):
            continue

        migration = vpoc_11 - vpoc_10

        # Check migration condition
        trade_dir = None
        if gap_dir == 'up' and migration > 0.3 * adr:
            trade_dir = 'long'
        elif gap_dir == 'down' and migration < -0.3 * adr:
            trade_dir = 'short'

        if trade_dir is None:
            continue

        # SL calculation
        if trade_dir == 'long':
            sl_dist = max(entry_price - val_11 + 0.1 * adr, 0.1 * adr)
        else:
            sl_dist = max(vah_11 - entry_price + 0.1 * adr, 0.1 * adr)

        # Load 1-min bars
        rth = load_1min(ticker, date_str)
        if rth is None:
            continue

        result = simulate_trailing_from(rth, '11:00', entry_price, sl_dist, trade_dir, adr)
        if result is None:
            continue

        result['ticker'] = ticker
        result['date'] = date_str
        result['gap_direction'] = gap_dir
        result['year'] = row['year']
        result['migration_adr'] = migration / adr
        result['sl_dist_adr'] = sl_dist / adr
        results.append(result)

    print(f"  S6 done: {len(results)} trades", file=sys.stderr)
    return pd.DataFrame(results)


# ============================================================
# MAIN
# ============================================================

def main():
    start_time = time.time()
    lines = []

    lines.append("=" * 70)
    lines.append("D13 PHASE 2: UNABHAENGIGE STRATEGIEN - QUANT IS-BACKTEST")
    lines.append("IS-Periode: 2021-02-21 bis 2023-12-31")
    lines.append("Exit: TRAIL_D (+1R->BE, trail peak-0.5R)")
    lines.append("Bootstrap: 1000 Resamples fuer P-Wert")
    lines.append("=" * 70)

    # Load metadata
    is_df = load_metadata_is()
    lines.append(f"\nTotal IS Events: {len(is_df)}")
    lines.append(f"GapUp: {(is_df['gap_direction']=='up').sum()}, GapDn: {(is_df['gap_direction']=='down').sum()}")
    lines.append("")

    all_rankings = []

    # ========================================
    # S1: VWAP Sigma-2 Mean Reversion
    # ========================================
    lines.append("\n" + "=" * 70)
    lines.append("--- S1: VWAP Sigma-2 Mean Reversion ---")
    lines.append("  Logik: Erster |z_score| >= 2.0 -> Entry gegen Extension")
    lines.append("  SL: 0.25 ADR | Exit: TRAIL_D ab Entry-Zeit")
    lines.append("=" * 70)

    s1_trades = run_s1(is_df, z_threshold=2.0)
    s1_stats = summarize(s1_trades, "S1", lines)
    all_rankings.append(('S1', 'VWAP Sigma-2 Mean Reversion', s1_stats))

    # Variant: z >= 2.5
    lines.append("\n  --- Variante: z >= 2.5 ---")
    s1v_trades = run_s1(is_df, z_threshold=2.5)
    s1v_stats = summarize(s1v_trades, "S1v", lines, variant_label="  [z>=2.5] ")

    # ========================================
    # S2: No-Fill Power Move
    # ========================================
    lines.append("\n" + "=" * 70)
    lines.append("--- S2: No-Fill Power Move ---")
    lines.append("  Logik: Gap nicht gefuellt + grosse Gap + hohes RVOL -> Continuation ab 11:00")
    lines.append("  SL: Entry bis LOD/HOD (min 0.15 ADR) | Exit: TRAIL_D ab 11:01")
    lines.append("=" * 70)

    s2_trades = run_s2(is_df, min_gap_adr=2.0, min_rvol60=3.0)
    s2_stats = summarize(s2_trades, "S2", lines)
    all_rankings.append(('S2', 'No-Fill Power Move', s2_stats))

    # Variant: gap >= 3.0 ADR
    lines.append("\n  --- Variante: gap >= 3.0 ADR ---")
    s2v_trades = run_s2(is_df, min_gap_adr=3.0, min_rvol60=3.0)
    s2v_stats = summarize(s2v_trades, "S2v", lines, variant_label="  [gap>=3.0] ")

    # ========================================
    # S3: Late-Session Momentum (13:00)
    # ========================================
    lines.append("\n" + "=" * 70)
    lines.append("--- S3: Late-Session Momentum (13:00 Entry) ---")
    lines.append("  Logik: Um 13:00 close > VWAP und > Open -> Long (GapUp), umgekehrt fuer GapDn")
    lines.append("  SL: 0.25 ADR | Exit: TRAIL_D ab 13:01")
    lines.append("=" * 70)

    s3_trades = run_s3(is_df)
    s3_stats = summarize(s3_trades, "S3", lines)
    all_rankings.append(('S3', 'Late-Session Momentum', s3_stats))

    # ========================================
    # S4: PM/RTH Ratio Extreme
    # ========================================
    lines.append("\n" + "=" * 70)
    lines.append("--- S4: PM/RTH Ratio Extreme ---")
    lines.append("  Logik: Sehr niedriges PM-Volumen (<10% RTH5) + Gap >= 1.5 ADR")
    lines.append("  Entry: close_935 | SL: 0.25 ADR | Exit: TRAIL_D")
    lines.append("=" * 70)

    s4_trades = run_s4(is_df, pm_rth5_max=0.10, min_gap_adr=1.5)
    s4_stats = summarize(s4_trades, "S4", lines)
    all_rankings.append(('S4', 'PM/RTH Ratio Extreme', s4_stats))

    # Variant: pm_rth5 < 0.05
    lines.append("\n  --- Variante: pm_rth5 < 0.05 ---")
    s4v_trades = run_s4(is_df, pm_rth5_max=0.05, min_gap_adr=1.5)
    s4v_stats = summarize(s4v_trades, "S4v", lines, variant_label="  [pm<0.05] ")

    # ========================================
    # S5: ORB-Aligned Breakout
    # ========================================
    lines.append("\n" + "=" * 70)
    lines.append("--- S5: ORB-Aligned Breakout (15-min Range) ---")
    lines.append("  Logik: 15-min OR Breakout in Gap-Richtung")
    lines.append("  SL: OR-Breite (0.1-0.5 ADR) | Exit: TRAIL_D ab Break-Zeit")
    lines.append("=" * 70)

    s5_trades = run_s5(is_df)
    s5_stats = summarize(s5_trades, "S5", lines)
    all_rankings.append(('S5', 'ORB-Aligned Breakout', s5_stats))

    # ========================================
    # S6: VPOC Migration + Gap Continuation
    # ========================================
    lines.append("\n" + "=" * 70)
    lines.append("--- S6: VPOC Migration + Gap Continuation ---")
    lines.append("  Logik: VPOC wandert >0.3 ADR in Gap-Richtung (10:00->11:00)")
    lines.append("  Entry: close_1100 | SL: VAL/VAH + 0.1 ADR | Exit: TRAIL_D ab 11:01")
    lines.append("=" * 70)

    s6_trades = run_s6(is_df)
    s6_stats = summarize(s6_trades, "S6", lines)
    all_rankings.append(('S6', 'VPOC Migration + Gap Continuation', s6_stats))

    # ========================================
    # RANKING
    # ========================================
    lines.append("\n\n" + "=" * 70)
    lines.append("RANKING (nach EV in ADR)")
    lines.append("=" * 70)

    # Sort by EV, handle NaN
    valid_rankings = [(code, name, stats) for code, name, stats in all_rankings
                       if stats.get('ev') is not None and not np.isnan(stats.get('ev', np.nan))]
    valid_rankings.sort(key=lambda x: x[2].get('ev', -999), reverse=True)

    for rank, (code, name, stats) in enumerate(valid_rankings, 1):
        ev = stats.get('ev', np.nan)
        n = stats.get('n', 0)
        wr = stats.get('wr', np.nan)
        pf = stats.get('pf', np.nan)
        p = stats.get('p', np.nan)
        ev_s = f"{ev:+.4f}" if not np.isnan(ev) else "N/A"
        wr_s = f"{wr*100:.1f}%" if not np.isnan(wr) else "N/A"
        pf_s = f"{pf:.2f}" if not np.isnan(pf) else "N/A"
        p_s = f"{p:.3f}" if not np.isnan(p) else "N/A"
        lines.append(f"  #{rank}: {code} - {name} - EV={ev_s} ADR, N={n}, WR={wr_s}, PF={pf_s}, p={p_s}")

    # Also list N<30 strategies
    invalid_rankings = [(code, name, stats) for code, name, stats in all_rankings
                         if stats.get('ev') is None or np.isnan(stats.get('ev', np.nan))]
    for code, name, stats in invalid_rankings:
        n = stats.get('n', 0)
        lines.append(f"  N/A: {code} - {name} - N={n} (zu wenig fuer Bewertung)")

    # ========================================
    # EMPFEHLUNG
    # ========================================
    lines.append("\n\n" + "=" * 70)
    lines.append("EMPFEHLUNG FUER OOS")
    lines.append("=" * 70)

    promising = [x for x in valid_rankings if x[2].get('ev', 0) > 0 and x[2].get('p', 1) < 0.10]
    marginal = [x for x in valid_rankings if x[2].get('ev', 0) > 0 and x[2].get('p', 1) >= 0.10]
    negative = [x for x in valid_rankings if x[2].get('ev', 0) <= 0]

    if promising:
        lines.append("\n  VIELVERSPRECHEND (EV>0, p<0.10):")
        for code, name, stats in promising:
            lines.append(f"    -> {code}: {name} (EV={stats['ev']:+.4f} ADR, p={stats['p']:.3f})")
        lines.append("    --> Empfehlung: OOS-Validierung durchfuehren!")
    else:
        lines.append("\n  Keine Strategie mit statistisch signifikantem positivem EV (p<0.10).")

    if marginal:
        lines.append("\n  MARGINAL (EV>0, aber p>=0.10):")
        for code, name, stats in marginal:
            lines.append(f"    -> {code}: {name} (EV={stats['ev']:+.4f} ADR, p={stats['p']:.3f})")
        lines.append("    --> Vorsicht: Koennte Zufall sein. OOS mit Vorbehalt.")

    if negative:
        lines.append("\n  NICHT EMPFOHLEN (EV<=0):")
        for code, name, stats in negative:
            lines.append(f"    -> {code}: {name} (EV={stats['ev']:+.4f} ADR)")
        lines.append("    --> Kein OOS-Test noetig.")

    elapsed = time.time() - start_time
    lines.append(f"\n\nLaufzeit: {elapsed:.0f} Sekunden")

    # Write output
    output = "\n".join(lines)
    out_path = OUT_DIR / 'd13_independent_quant_results.txt'
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(output)

    print(output)
    print(f"\n\nGespeichert: {out_path}", file=sys.stderr)


if __name__ == '__main__':
    main()
