"""
D13 PREMARKET MOMENTUM — Intraday-Momentum-Strategie (Entry VOR 10:00 Uhr)
============================================================================
5 Strategie-Ideen:
  A) Momentum Confirmation (first_candle_dir == gap_direction)
  B) Dip & Rip (Pullback-Entry nach starkem Open)
  C) Gap + VWAP Hold (Preis haelt ueber/unter VWAP)
  D) MFE-basierter Exit (Fixed Target vs Trail Hybrid)
  E) Volume-Burst Entry (Volume-Spike in ersten 30 Min)

Phase 1: IS-Backtest aller 5 Ideen
Phase 2: Parameter-Matrix fuer Top-Ideen
Phase 3: Top-5 IS-Kombinationen (N>=30, p<0.05)
Phase 4: OOS-Validierung mit Bootstrap CI

IS: 2021-02-21 bis 2023-12-31
OOS: 2024-01-01 bis 2026-02-13
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import warnings
import time
import sys
from itertools import product

warnings.filterwarnings('ignore')

# ============================================================
# PATHS
# ============================================================
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROJECT = PROJECT_ROOT
META_PATH = PROJECT / "data" / "metadata" / "metadata_v9.parquet"
RAW_1MIN = PROJECT / "data" / "raw_1min"
RESULTS_DIR = PROJECT / "d13_team_analysis" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = RESULTS_DIR / "d13_premarket_momentum_results.txt"

# ============================================================
# SIMULATE TRAIL_D
# ============================================================
def simulate_trail_d(bars_rth, entry_price, sl_dist, trade_dir, adr, entry_time='09:36'):
    """TRAIL_D: +1R -> BE, dann Trail = Peak - 0.5R"""
    if sl_dist <= 0 or pd.isna(sl_dist) or adr <= 0:
        return None
    post_entry = bars_rth[(bars_rth['time_et'] >= entry_time) & (bars_rth['time_et'] <= '15:55')]
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
        exit_price = post_entry.iloc[-1]['close']
        exit_type = 'timeout'
        exit_time = post_entry.iloc[-1]['time_et']
    if trade_dir == 'long':
        pnl = exit_price - entry_price
    else:
        pnl = entry_price - exit_price
    return {
        'pnl': pnl, 'pnl_r': pnl / sl_dist, 'pnl_adr': pnl / adr,
        'exit_type': exit_type, 'exit_time': exit_time,
        'highest_r': highest_r, 'trail_active': trail_active, 'winner': pnl > 0,
    }


def simulate_hybrid(bars_rth, entry_price, sl_dist, trade_dir, adr, target_r, entry_time='09:36'):
    """Hybrid: Fixed target OR Trail_D, whichever comes first."""
    if sl_dist <= 0 or pd.isna(sl_dist) or adr <= 0:
        return None
    post_entry = bars_rth[(bars_rth['time_et'] >= entry_time) & (bars_rth['time_et'] <= '15:55')]
    if len(post_entry) == 0:
        return None
    if trade_dir == 'long':
        sl_level = entry_price - sl_dist
        target_level = entry_price + target_r * sl_dist
    else:
        sl_level = entry_price + sl_dist
        target_level = entry_price - target_r * sl_dist
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
        # Check target first
        if trade_dir == 'long' and bar['high'] >= target_level:
            exit_price = target_level
            exit_type = 'target'
            exit_time = bar['time_et']
            break
        elif trade_dir == 'short' and bar['low'] <= target_level:
            exit_price = target_level
            exit_type = 'target'
            exit_time = bar['time_et']
            break
        # Trail logic
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
        exit_price = post_entry.iloc[-1]['close']
        exit_type = 'timeout'
        exit_time = post_entry.iloc[-1]['time_et']
    if trade_dir == 'long':
        pnl = exit_price - entry_price
    else:
        pnl = entry_price - exit_price
    return {
        'pnl': pnl, 'pnl_r': pnl / sl_dist, 'pnl_adr': pnl / adr,
        'exit_type': exit_type, 'exit_time': exit_time,
        'highest_r': highest_r, 'trail_active': trail_active, 'winner': pnl > 0,
    }


# ============================================================
# HELPER: Load 1-min bars
# ============================================================
def load_bars(ticker, date):
    """Load 1-min bars for a ticker/date. Returns RTH bars or None."""
    fpath = RAW_1MIN / ticker / f"{date}.parquet"
    if not fpath.exists():
        return None
    bars = pd.read_parquet(fpath)
    rth = bars[bars['session'] == 'rth'].copy()
    if len(rth) < 10:
        return None
    return rth


# ============================================================
# HELPER: Summary stats
# ============================================================
def summarize(pnl_r_array, label="", min_n=10):
    """Summary stats from an array of PnL in R."""
    arr = np.array(pnl_r_array)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n < min_n:
        return f"  {label:<40} | N={n:>4} [INSUFFICIENT]", n, np.nan, np.nan
    mean_r = np.mean(arr)
    med_r = np.median(arr)
    std_r = np.std(arr, ddof=1)
    wr = np.mean(arr > 0)
    t_stat, p_val = stats.ttest_1samp(arr, 0)
    sharpe = mean_r / std_r if std_r > 0 else 0
    line = (f"  {label:<40} | N={n:>4} | EV={mean_r:>+6.2f}R | Med={med_r:>+6.2f}R | "
            f"WR={wr:>5.1%} | Sharpe={sharpe:>+5.2f} | p={p_val:.4f}")
    return line, n, mean_r, p_val


def bootstrap_ci(arr, n_boot=10000, ci=0.95):
    """Bootstrap CI for mean."""
    arr = np.array(arr)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 10:
        return np.nan, np.nan, np.nan
    boot_means = np.array([np.mean(np.random.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    lo = np.percentile(boot_means, alpha * 100)
    hi = np.percentile(boot_means, (1 - alpha) * 100)
    return np.mean(arr), lo, hi


# ============================================================
# MAIN
# ============================================================
def main():
    t_start = time.time()
    lines = []
    lines.append("=" * 90)
    lines.append("D13 PREMARKET MOMENTUM — Intraday-Momentum-Strategie (Entry VOR 10:00 Uhr)")
    lines.append("=" * 90)
    lines.append("")

    # Load metadata
    print("[LOAD] Loading metadata...", file=sys.stderr)
    meta = pd.read_parquet(META_PATH)
    is_data = meta[(meta['date'] >= '2021-02-21') & (meta['date'] <= '2023-12-31')].copy()
    oos_data = meta[(meta['date'] >= '2024-01-01') & (meta['date'] <= '2026-02-13')].copy()
    print(f"[LOAD] IS: {len(is_data)} events, OOS: {len(oos_data)} events", file=sys.stderr)

    lines.append(f"IS-Periode: 2021-02-21 bis 2023-12-31  ({len(is_data)} Events)")
    lines.append(f"OOS-Periode: 2024-01-01 bis 2026-02-13 ({len(oos_data)} Events)")
    lines.append("")

    # ================================================================
    # PHASE 1: ALL 5 IDEAS ON IS
    # ================================================================
    lines.append("=" * 90)
    lines.append("PHASE 1: IS-BACKTEST ALLER 5 IDEEN")
    lines.append("=" * 90)

    # Pre-filter: need valid close_935, adr_10, gap_direction
    valid = is_data.dropna(subset=['close_935', 'adr_10', 'gap_direction']).copy()
    valid = valid[valid['adr_10'] > 0].copy()
    print(f"[PHASE1] Valid IS events: {len(valid)}", file=sys.stderr)

    # ===========================================
    # IDEE A: Momentum Confirmation
    # ===========================================
    lines.append("")
    lines.append("-" * 90)
    lines.append("IDEE A: MOMENTUM CONFIRMATION")
    lines.append("  Entry: close_935 | Bedingung: first_candle_dir == gap_direction")
    lines.append("  Richtung: gap_direction | Exit: TRAIL_D")
    lines.append("-" * 90)

    idea_a_results = {}
    for sl_frac in [0.15, 0.20, 0.25, 0.30]:
        # Filter: first_candle confirms gap
        subset = valid[valid['first_candle_dir'] == 'with_gap'].copy()
        pnl_list = []
        processed = 0
        for _, row in subset.iterrows():
            rth = load_bars(row['ticker'], str(row['date']))
            if rth is None:
                continue
            entry_price = row['close_935']
            adr = row['adr_10']
            sl_dist = sl_frac * adr
            trade_dir = 'long' if row['gap_direction'] == 'up' else 'short'
            result = simulate_trail_d(rth, entry_price, sl_dist, trade_dir, adr)
            if result:
                pnl_list.append(result['pnl_r'])
            processed += 1
            if processed % 200 == 0:
                print(f"  [A] SL={sl_frac} ... {processed}/{len(subset)}", file=sys.stderr)

        label = f"SL={sl_frac:.2f}ADR"
        line, n, ev, pval = summarize(pnl_list, label)
        lines.append(line)
        idea_a_results[sl_frac] = {'pnl_list': pnl_list, 'n': n, 'ev': ev, 'pval': pval}
        print(f"  [A] SL={sl_frac}: N={n}, EV={ev if ev is not np.nan else 'N/A'}", file=sys.stderr)

    # Also test without confirmation (baseline)
    subset_all = valid.copy()
    pnl_list_base = []
    processed = 0
    for _, row in subset_all.iterrows():
        rth = load_bars(row['ticker'], str(row['date']))
        if rth is None:
            continue
        entry_price = row['close_935']
        adr = row['adr_10']
        sl_dist = 0.25 * adr
        trade_dir = 'long' if row['gap_direction'] == 'up' else 'short'
        result = simulate_trail_d(rth, entry_price, sl_dist, trade_dir, adr)
        if result:
            pnl_list_base.append(result['pnl_r'])
        processed += 1
        if processed % 500 == 0:
            print(f"  [A-Baseline] ... {processed}/{len(subset_all)}", file=sys.stderr)

    line_base, _, _, _ = summarize(pnl_list_base, "Baseline (no filter)")
    lines.append(line_base)

    # ===========================================
    # IDEE B: Dip & Rip (1-min bars needed)
    # ===========================================
    lines.append("")
    lines.append("-" * 90)
    lines.append("IDEE B: DIP & RIP (Pullback-Entry nach starkem Open)")
    lines.append("  Scanne 09:35-10:00 fuer Pullback-Low (GapUp) / Pullback-High (GapDown)")
    lines.append("  Entry: Preis nach Pullback + neues 5-min High/Low | SL: Pullback-Extrem")
    lines.append("-" * 90)

    # Sample: strong OD events with confirmation
    idea_b_pool = valid[
        (valid['od_strength'] > 0.3) &
        (valid['od_direction'] == 'with_gap')
    ].copy()
    # Limit to 800 for speed
    if len(idea_b_pool) > 800:
        idea_b_pool = idea_b_pool.sample(800, random_state=42)

    pnl_b = []
    processed = 0
    for _, row in idea_b_pool.iterrows():
        rth = load_bars(row['ticker'], str(row['date']))
        if rth is None:
            continue
        adr = row['adr_10']
        gap_dir = row['gap_direction']

        # Get bars from 09:35 to 10:00
        scan_bars = rth[(rth['time_et'] >= '09:35') & (rth['time_et'] <= '10:00')].copy()
        if len(scan_bars) < 5:
            continue

        if gap_dir == 'up':
            # Find the low point (dip)
            dip_idx = scan_bars['low'].idxmin()
            dip_price = scan_bars.loc[dip_idx, 'low']
            dip_time = scan_bars.loc[dip_idx, 'time_et']

            # After the dip, look for recovery: highest high in a 5-bar window
            post_dip = scan_bars.loc[dip_idx:]
            if len(post_dip) < 3:
                continue

            # Entry = close of bar when we get a new 3-bar high after dip
            recovery_bars = post_dip.iloc[1:]  # skip the dip bar itself
            if len(recovery_bars) < 1:
                continue
            # Find first bar where close > dip_price + some threshold (confirm bounce)
            entry_bar = None
            for ridx, rbar in recovery_bars.iterrows():
                if rbar['close'] > dip_price + 0.02 * adr:
                    entry_bar = rbar
                    break
            if entry_bar is None:
                continue
            entry_price = entry_bar['close']
            entry_time = entry_bar['time_et']
            sl_dist = entry_price - dip_price
            if sl_dist <= 0 or sl_dist > 0.5 * adr:
                continue
            trade_dir = 'long'
        else:
            # GapDown: find the high point (rip up)
            rip_idx = scan_bars['high'].idxmax()
            rip_price = scan_bars.loc[rip_idx, 'high']
            rip_time = scan_bars.loc[rip_idx, 'time_et']

            post_rip = scan_bars.loc[rip_idx:]
            if len(post_rip) < 3:
                continue
            recovery_bars = post_rip.iloc[1:]
            if len(recovery_bars) < 1:
                continue
            entry_bar = None
            for ridx, rbar in recovery_bars.iterrows():
                if rbar['close'] < rip_price - 0.02 * adr:
                    entry_bar = rbar
                    break
            if entry_bar is None:
                continue
            entry_price = entry_bar['close']
            entry_time = entry_bar['time_et']
            sl_dist = rip_price - entry_price
            if sl_dist <= 0 or sl_dist > 0.5 * adr:
                continue
            trade_dir = 'short'

        # Simulate from entry time
        result = simulate_trail_d(rth, entry_price, sl_dist, trade_dir, adr, entry_time=entry_time)
        if result:
            pnl_b.append(result['pnl_r'])
        processed += 1
        if processed % 100 == 0:
            print(f"  [B] ... {processed}/{len(idea_b_pool)}", file=sys.stderr)

    line_b, n_b, ev_b, pval_b = summarize(pnl_b, "Dip&Rip (OD>0.3, with_gap)")
    lines.append(line_b)
    print(f"  [B] Done: N={n_b}, EV={ev_b}", file=sys.stderr)

    # ===========================================
    # IDEE C: Gap + VWAP Hold
    # ===========================================
    lines.append("")
    lines.append("-" * 90)
    lines.append("IDEE C: GAP + VWAP HOLD (Entry um 10:00)")
    lines.append("  Bedingung: vwap_z_at_10am > 0 (GapUp) oder < 0 (GapDown)")
    lines.append("  Entry: close_1000 | SL: 0.20/0.25/0.30 ADR | Exit: TRAIL_D")
    lines.append("-" * 90)

    valid_c = valid.dropna(subset=['close_1000', 'vwap_z_at_10am']).copy()
    # VWAP hold: z > 0 for gap_up, z < 0 for gap_down
    vwap_hold = valid_c[
        ((valid_c['gap_direction'] == 'up') & (valid_c['vwap_z_at_10am'] > 0)) |
        ((valid_c['gap_direction'] == 'down') & (valid_c['vwap_z_at_10am'] < 0))
    ].copy()

    for sl_frac in [0.20, 0.25, 0.30]:
        pnl_c = []
        processed = 0
        for _, row in vwap_hold.iterrows():
            rth = load_bars(row['ticker'], str(row['date']))
            if rth is None:
                continue
            entry_price = row['close_1000']
            adr = row['adr_10']
            sl_dist = sl_frac * adr
            trade_dir = 'long' if row['gap_direction'] == 'up' else 'short'
            result = simulate_trail_d(rth, entry_price, sl_dist, trade_dir, adr, entry_time='10:01')
            if result:
                pnl_c.append(result['pnl_r'])
            processed += 1
            if processed % 200 == 0:
                print(f"  [C] SL={sl_frac} ... {processed}/{len(vwap_hold)}", file=sys.stderr)

        label_c = f"VWAP Hold, SL={sl_frac:.2f}ADR"
        line_c, n_c, ev_c, _ = summarize(pnl_c, label_c)
        lines.append(line_c)
        print(f"  [C] SL={sl_frac}: N={n_c}, EV={ev_c}", file=sys.stderr)

    # Also: strong VWAP hold (z > 1 or z < -1)
    strong_vwap = valid_c[
        ((valid_c['gap_direction'] == 'up') & (valid_c['vwap_z_at_10am'] > 1.0)) |
        ((valid_c['gap_direction'] == 'down') & (valid_c['vwap_z_at_10am'] < -1.0))
    ].copy()
    pnl_cs = []
    for _, row in strong_vwap.iterrows():
        rth = load_bars(row['ticker'], str(row['date']))
        if rth is None:
            continue
        entry_price = row['close_1000']
        adr = row['adr_10']
        sl_dist = 0.25 * adr
        trade_dir = 'long' if row['gap_direction'] == 'up' else 'short'
        result = simulate_trail_d(rth, entry_price, sl_dist, trade_dir, adr, entry_time='10:01')
        if result:
            pnl_cs.append(result['pnl_r'])
    line_cs, n_cs, ev_cs, _ = summarize(pnl_cs, "Strong VWAP (|z|>1), SL=0.25")
    lines.append(line_cs)

    # ===========================================
    # IDEE D: MFE-basierter Exit (Hybrid)
    # ===========================================
    lines.append("")
    lines.append("-" * 90)
    lines.append("IDEE D: MFE-BASIERTER EXIT (Fixed Target + Trail_D Hybrid)")
    lines.append("  Entry: close_935 | first_candle = with_gap | SL: 0.25 ADR")
    lines.append("  Vergleich: Trail_D only vs Hybrid (Target OR Trail)")
    lines.append("-" * 90)

    subset_d = valid[valid['first_candle_dir'] == 'with_gap'].copy()
    # Limit for speed
    if len(subset_d) > 1200:
        subset_d_sample = subset_d.sample(1200, random_state=42)
    else:
        subset_d_sample = subset_d

    for target_r in [1.0, 1.5, 2.0, 3.0]:
        pnl_hybrid = []
        processed = 0
        for _, row in subset_d_sample.iterrows():
            rth = load_bars(row['ticker'], str(row['date']))
            if rth is None:
                continue
            entry_price = row['close_935']
            adr = row['adr_10']
            sl_dist = 0.25 * adr
            trade_dir = 'long' if row['gap_direction'] == 'up' else 'short'
            result = simulate_hybrid(rth, entry_price, sl_dist, trade_dir, adr, target_r=target_r)
            if result:
                pnl_hybrid.append(result['pnl_r'])
            processed += 1
            if processed % 200 == 0:
                print(f"  [D] Target={target_r}R ... {processed}/{len(subset_d_sample)}", file=sys.stderr)

        label_d = f"Hybrid Target={target_r:.1f}R"
        line_d, n_d, ev_d, _ = summarize(pnl_hybrid, label_d)
        lines.append(line_d)
        print(f"  [D] Target={target_r}R: N={n_d}, EV={ev_d}", file=sys.stderr)

    # Trail_D only on same subset for comparison
    pnl_trail_only = []
    for _, row in subset_d_sample.iterrows():
        rth = load_bars(row['ticker'], str(row['date']))
        if rth is None:
            continue
        entry_price = row['close_935']
        adr = row['adr_10']
        sl_dist = 0.25 * adr
        trade_dir = 'long' if row['gap_direction'] == 'up' else 'short'
        result = simulate_trail_d(rth, entry_price, sl_dist, trade_dir, adr)
        if result:
            pnl_trail_only.append(result['pnl_r'])
    line_dt, _, _, _ = summarize(pnl_trail_only, "Trail_D only (comparison)")
    lines.append(line_dt)

    # ===========================================
    # IDEE E: Volume-Burst Entry
    # ===========================================
    lines.append("")
    lines.append("-" * 90)
    lines.append("IDEE E: VOLUME-BURST ENTRY")
    lines.append("  Scanne 09:31-10:00: Bar mit Volume > 3x avg der vorherigen Bars")
    lines.append("  Entry: Close des Burst-Bars | SL: Burst-Bar Low/High")
    lines.append("-" * 90)

    idea_e_pool = valid.copy()
    if len(idea_e_pool) > 800:
        idea_e_pool = idea_e_pool.sample(800, random_state=42)

    pnl_e = []
    processed = 0
    for _, row in idea_e_pool.iterrows():
        rth = load_bars(row['ticker'], str(row['date']))
        if rth is None:
            continue
        adr = row['adr_10']

        # Scan first 30 min
        scan = rth[(rth['time_et'] >= '09:31') & (rth['time_et'] <= '10:00')].copy()
        if len(scan) < 5:
            continue

        # Calculate rolling avg volume
        scan = scan.reset_index(drop=True)
        burst_bar = None
        for i in range(3, len(scan)):
            avg_vol = scan.loc[:i-1, 'volume'].mean()
            if avg_vol > 0 and scan.loc[i, 'volume'] > 3 * avg_vol:
                burst_bar = scan.loc[i]
                break

        if burst_bar is None:
            continue

        # Determine direction from burst bar
        if burst_bar['close'] > burst_bar['open']:
            trade_dir = 'long'
            entry_price = burst_bar['close']
            sl_dist = entry_price - burst_bar['low']
        else:
            trade_dir = 'short'
            entry_price = burst_bar['close']
            sl_dist = burst_bar['high'] - entry_price

        if sl_dist <= 0 or sl_dist > 0.5 * adr:
            continue

        entry_time = burst_bar['time_et']
        result = simulate_trail_d(rth, entry_price, sl_dist, trade_dir, adr, entry_time=entry_time)
        if result:
            pnl_e.append(result['pnl_r'])
        processed += 1
        if processed % 100 == 0:
            print(f"  [E] ... {processed}/{len(idea_e_pool)}", file=sys.stderr)

    line_e, n_e, ev_e, pval_e = summarize(pnl_e, "Volume Burst (3x avg)")
    lines.append(line_e)
    print(f"  [E] Done: N={n_e}, EV={ev_e}", file=sys.stderr)

    # ================================================================
    # PHASE 1 SUMMARY
    # ================================================================
    lines.append("")
    lines.append("=" * 90)
    lines.append("PHASE 1 ZUSAMMENFASSUNG — Ranking nach EV")
    lines.append("=" * 90)

    phase1_summary = []
    # Collect best from each idea
    best_a = max(idea_a_results.items(), key=lambda x: x[1]['ev'] if not np.isnan(x[1]['ev']) else -999)
    phase1_summary.append(('A: Momentum Confirm', best_a[1]['ev'], best_a[1]['n'], best_a[1]['pval'], f"SL={best_a[0]}"))
    phase1_summary.append(('B: Dip & Rip', ev_b, n_b, pval_b, "OD>0.3"))
    phase1_summary.append(('E: Volume Burst', ev_e, n_e, pval_e, "3x avg"))

    # Sort by EV
    phase1_summary.sort(key=lambda x: x[1] if not np.isnan(x[1]) else -999, reverse=True)
    lines.append(f"  {'Idee':<25} | {'EV':>8} | {'N':>6} | {'p-val':>8} | {'Params':<15}")
    lines.append(f"  {'-'*75}")
    for name, ev, n, pval, params in phase1_summary:
        ev_s = f"{ev:>+8.3f}" if not np.isnan(ev) else "     N/A"
        p_s = f"{pval:>8.4f}" if not np.isnan(pval) else "     N/A"
        lines.append(f"  {name:<25} | {ev_s}R | {n:>6} | {p_s} | {params:<15}")

    lines.append("")
    lines.append("  Hinweis: Ideen C und D sind oben mit Details aufgelistet.")
    lines.append(f"  Phase 1 Laufzeit: {time.time() - t_start:.0f}s")

    # ================================================================
    # PHASE 2: PARAMETER MATRIX fuer Top-Ideen
    # ================================================================
    t_phase2 = time.time()
    lines.append("")
    lines.append("=" * 90)
    lines.append("PHASE 2: PARAMETER-MATRIX (Fokus auf beste Ideen)")
    lines.append("=" * 90)
    lines.append("  Kombiniere Strategie-Ideen mit Feature-Filtern")
    lines.append("")

    # We will test IDEE A (Momentum Confirmation) with parameter combos
    # and IDEE C (VWAP Hold) with parameter combos,
    # as these are metadata-driven and fastest to iterate.

    # Strategy variants:
    # S1: Entry=close_935, Dir=gap_dir, Confirm=first_candle==with_gap, Exit=TRAIL_D
    # S2: Entry=close_1000, Dir=gap_dir, VWAP_hold, Exit=TRAIL_D
    # S3: Entry=close_935, Dir=gap_dir, NO filter, Exit=TRAIL_D (baseline for filtering)

    # Filters to combine:
    gap_size_cuts = [('gap>=1ADR', 1.0), ('gap>=1.5ADR', 1.5), ('gap>=2ADR', 2.0)]
    od_strength_cuts = [('od>=0.3', 0.3), ('od>=0.5', 0.5), ('od>=0.7', 0.7)]
    rvol_cuts = [('rvol>=3', 3.0), ('rvol>=5', 5.0)]
    gap_dir_opts = [('both', 'both'), ('up', 'up'), ('down', 'down')]
    rsi_cuts = [('rsi<50', '<50'), ('rsi>=50', '>=50'), ('rsi_any', 'any')]
    pm_rth_cuts = [('pm<0.30', 0.30), ('pm<0.50', 0.50), ('pm_any', 999)]

    lines.append("-" * 90)
    lines.append("STRATEGY S1: Momentum Confirmation (close_935, first_candle=with_gap)")
    lines.append("-" * 90)

    # Pre-compute bars cache for speed
    print("[PHASE2] Pre-loading all IS bars into cache...", file=sys.stderr)
    bars_cache = {}
    unique_events = valid[['ticker', 'date']].drop_duplicates()
    loaded = 0
    for _, ev in unique_events.iterrows():
        key = (ev['ticker'], str(ev['date']))
        rth = load_bars(key[0], key[1])
        if rth is not None:
            bars_cache[key] = rth
        loaded += 1
        if loaded % 500 == 0:
            print(f"  [CACHE] {loaded}/{len(unique_events)} loaded", file=sys.stderr)
    print(f"  [CACHE] Done: {len(bars_cache)} bars loaded", file=sys.stderr)

    def run_strategy_s1(subset, sl_frac=0.25):
        """Strategy S1: Momentum Confirmation"""
        pnl_list = []
        for _, row in subset.iterrows():
            key = (row['ticker'], str(row['date']))
            rth = bars_cache.get(key)
            if rth is None:
                continue
            entry_price = row['close_935']
            adr = row['adr_10']
            sl_dist = sl_frac * adr
            trade_dir = 'long' if row['gap_direction'] == 'up' else 'short'
            result = simulate_trail_d(rth, entry_price, sl_dist, trade_dir, adr)
            if result:
                pnl_list.append(result['pnl_r'])
        return pnl_list

    def run_strategy_s2(subset, sl_frac=0.25):
        """Strategy S2: VWAP Hold (entry at close_1000)"""
        pnl_list = []
        for _, row in subset.iterrows():
            key = (row['ticker'], str(row['date']))
            rth = bars_cache.get(key)
            if rth is None:
                continue
            entry_price = row['close_1000']
            if pd.isna(entry_price):
                continue
            adr = row['adr_10']
            sl_dist = sl_frac * adr
            trade_dir = 'long' if row['gap_direction'] == 'up' else 'short'
            result = simulate_trail_d(rth, entry_price, sl_dist, trade_dir, adr, entry_time='10:01')
            if result:
                pnl_list.append(result['pnl_r'])
        return pnl_list

    def apply_filters(df, gap_dir, gap_min, od_min, rvol_min, rsi_cut, pm_max,
                      require_confirm=False, require_vwap_hold=False):
        """Apply filter combination to dataframe."""
        s = df.copy()
        if gap_dir != 'both':
            s = s[s['gap_direction'] == gap_dir]
        s = s[s['gap_size_in_adr'] >= gap_min]
        s = s[s['od_strength'] >= od_min]
        s = s[s['rvol_5'] >= rvol_min]
        if rsi_cut == '<50':
            s = s[s['rsi_14_prev'] < 50]
        elif rsi_cut == '>=50':
            s = s[s['rsi_14_prev'] >= 50]
        if pm_max < 999:
            s = s[s['pm_rth5'] < pm_max]
        if require_confirm:
            s = s[s['first_candle_dir'] == 'with_gap']
        if require_vwap_hold:
            s = s.dropna(subset=['vwap_z_at_10am'])
            s = s[
                ((s['gap_direction'] == 'up') & (s['vwap_z_at_10am'] > 0)) |
                ((s['gap_direction'] == 'down') & (s['vwap_z_at_10am'] < 0))
            ]
        return s

    # Run matrix for S1
    all_combos_s1 = []
    combo_count = 0
    total_combos_s1 = len(gap_size_cuts) * len(od_strength_cuts) * len(rvol_cuts) * len(gap_dir_opts) * len(rsi_cuts) * len(pm_rth_cuts)
    print(f"[PHASE2] S1: {total_combos_s1} combinations to test", file=sys.stderr)

    for (gl, gv), (ol, ov), (rl, rv), (dl, dv), (rsil, rsiv), (pl, pv) in product(
        gap_size_cuts, od_strength_cuts, rvol_cuts, gap_dir_opts, rsi_cuts, pm_rth_cuts
    ):
        combo_count += 1
        subset = apply_filters(valid, dv, gv, ov, rv, rsiv, pv, require_confirm=True)
        if len(subset) < 30:
            continue
        pnl = run_strategy_s1(subset)
        if len(pnl) < 30:
            continue
        arr = np.array(pnl)
        ev = np.mean(arr)
        wr = np.mean(arr > 0)
        t_stat, p_val = stats.ttest_1samp(arr, 0)
        label = f"{gl}|{ol}|{rl}|{dl}|{rsil}|{pl}"
        all_combos_s1.append({
            'label': label, 'n': len(arr), 'ev': ev, 'wr': wr, 'pval': p_val,
            'gap_min': gv, 'od_min': ov, 'rvol_min': rv, 'gap_dir': dv,
            'rsi': rsiv, 'pm_max': pv, 'pnl_list': pnl
        })
        if combo_count % 50 == 0:
            print(f"  [S1] {combo_count}/{total_combos_s1} combos tested...", file=sys.stderr)

    print(f"  [S1] Done: {len(all_combos_s1)} valid combos (N>=30)", file=sys.stderr)

    # S2: VWAP Hold matrix
    lines.append("")
    lines.append("-" * 90)
    lines.append("STRATEGY S2: VWAP Hold (close_1000, vwap_z confirmation)")
    lines.append("-" * 90)

    all_combos_s2 = []
    combo_count = 0
    for (gl, gv), (ol, ov), (rl, rv), (dl, dv), (rsil, rsiv), (pl, pv) in product(
        gap_size_cuts, od_strength_cuts, rvol_cuts, gap_dir_opts, rsi_cuts, pm_rth_cuts
    ):
        combo_count += 1
        subset = apply_filters(valid, dv, gv, ov, rv, rsiv, pv,
                               require_confirm=False, require_vwap_hold=True)
        if len(subset) < 30:
            continue
        pnl = run_strategy_s2(subset)
        if len(pnl) < 30:
            continue
        arr = np.array(pnl)
        ev = np.mean(arr)
        wr = np.mean(arr > 0)
        t_stat, p_val = stats.ttest_1samp(arr, 0)
        label = f"{gl}|{ol}|{rl}|{dl}|{rsil}|{pl}"
        all_combos_s2.append({
            'label': label, 'n': len(arr), 'ev': ev, 'wr': wr, 'pval': p_val,
            'gap_min': gv, 'od_min': ov, 'rvol_min': rv, 'gap_dir': dv,
            'rsi': rsiv, 'pm_max': pv, 'pnl_list': pnl
        })
        if combo_count % 50 == 0:
            print(f"  [S2] {combo_count} combos tested...", file=sys.stderr)

    print(f"  [S2] Done: {len(all_combos_s2)} valid combos (N>=30)", file=sys.stderr)

    # S3: Baseline (no entry confirmation, just gap_dir + filters)
    lines.append("")
    lines.append("-" * 90)
    lines.append("STRATEGY S3: Baseline (close_935, gap_dir, no confirmation)")
    lines.append("-" * 90)

    all_combos_s3 = []
    combo_count = 0
    for (gl, gv), (ol, ov), (rl, rv), (dl, dv), (rsil, rsiv), (pl, pv) in product(
        gap_size_cuts, od_strength_cuts, rvol_cuts, gap_dir_opts, rsi_cuts, pm_rth_cuts
    ):
        combo_count += 1
        subset = apply_filters(valid, dv, gv, ov, rv, rsiv, pv)
        if len(subset) < 30:
            continue
        pnl = run_strategy_s1(subset)  # Same entry logic, just no confirm filter
        if len(pnl) < 30:
            continue
        arr = np.array(pnl)
        ev = np.mean(arr)
        wr = np.mean(arr > 0)
        t_stat, p_val = stats.ttest_1samp(arr, 0)
        label = f"{gl}|{ol}|{rl}|{dl}|{rsil}|{pl}"
        all_combos_s3.append({
            'label': label, 'n': len(arr), 'ev': ev, 'wr': wr, 'pval': p_val,
            'gap_min': gv, 'od_min': ov, 'rvol_min': rv, 'gap_dir': dv,
            'rsi': rsiv, 'pm_max': pv, 'pnl_list': pnl
        })

    print(f"  [S3] Done: {len(all_combos_s3)} valid combos (N>=30)", file=sys.stderr)

    # ================================================================
    # PHASE 3: TOP-5 IS COMBINATIONS (N>=30, p<0.05)
    # ================================================================
    lines.append("")
    lines.append("=" * 90)
    lines.append("PHASE 3: TOP-5 IS-KOMBINATIONEN (N>=30, p<0.05)")
    lines.append("=" * 90)

    # Merge all combos
    all_combos = []
    for c in all_combos_s1:
        c['strategy'] = 'S1:MomConfirm'
        all_combos.append(c)
    for c in all_combos_s2:
        c['strategy'] = 'S2:VWAPHold'
        all_combos.append(c)
    for c in all_combos_s3:
        c['strategy'] = 'S3:Baseline'
        all_combos.append(c)

    # Filter significant and EV >= 0.25R
    significant = [c for c in all_combos if c['pval'] < 0.05 and c['ev'] >= 0.25 and c['n'] >= 30]
    significant.sort(key=lambda x: x['ev'], reverse=True)

    lines.append(f"  Gesamt Kombinationen getestet: {len(all_combos)}")
    lines.append(f"  Signifikant (p<0.05, EV>=0.25R, N>=30): {len(significant)}")
    lines.append("")

    if len(significant) == 0:
        # Fallback: show top 5 by EV regardless of significance
        lines.append("  KEINE signifikanten Kombinationen mit EV>=0.25R gefunden!")
        lines.append("  Zeige Top-10 nach EV (unabhaengig von Signifikanz):")
        lines.append("")
        all_combos_sorted = sorted(all_combos, key=lambda x: x['ev'], reverse=True)
        top_combos = all_combos_sorted[:10]
    else:
        top_combos = significant[:10]

    lines.append(f"  {'Rank':<5} | {'Strategy':<15} | {'Filter':<50} | {'N':>5} | {'EV':>7} | {'WR':>6} | {'p-val':>8}")
    lines.append(f"  {'-'*110}")
    for i, c in enumerate(top_combos):
        lines.append(f"  {i+1:<5} | {c['strategy']:<15} | {c['label']:<50} | {c['n']:>5} | {c['ev']:>+7.3f}R | {c['wr']:>5.1%} | {c['pval']:>8.4f}")

    # Select top 5 for OOS (even if not significant, to show them)
    top5_for_oos = top_combos[:5]

    lines.append("")
    lines.append(f"  Phase 2+3 Laufzeit: {time.time() - t_phase2:.0f}s")

    # ================================================================
    # PHASE 4: OOS VALIDATION
    # ================================================================
    t_phase4 = time.time()
    lines.append("")
    lines.append("=" * 90)
    lines.append("PHASE 4: OOS-VALIDIERUNG (Bootstrap 10000, 95% CI)")
    lines.append("=" * 90)

    # Pre-load OOS bars
    valid_oos = oos_data.dropna(subset=['close_935', 'adr_10', 'gap_direction']).copy()
    valid_oos = valid_oos[valid_oos['adr_10'] > 0].copy()

    print("[PHASE4] Pre-loading OOS bars...", file=sys.stderr)
    oos_cache = {}
    unique_oos = valid_oos[['ticker', 'date']].drop_duplicates()
    loaded = 0
    for _, ev in unique_oos.iterrows():
        key = (ev['ticker'], str(ev['date']))
        rth = load_bars(key[0], key[1])
        if rth is not None:
            oos_cache[key] = rth
        loaded += 1
        if loaded % 500 == 0:
            print(f"  [OOS CACHE] {loaded}/{len(unique_oos)} loaded", file=sys.stderr)
    print(f"  [OOS CACHE] Done: {len(oos_cache)} bars", file=sys.stderr)

    def run_oos_strategy(subset, strategy_type, sl_frac=0.25):
        """Run strategy on OOS data."""
        pnl_list = []
        for _, row in subset.iterrows():
            key = (row['ticker'], str(row['date']))
            rth = oos_cache.get(key)
            if rth is None:
                continue
            adr = row['adr_10']
            sl_dist = sl_frac * adr
            trade_dir = 'long' if row['gap_direction'] == 'up' else 'short'

            if strategy_type in ('S1:MomConfirm', 'S3:Baseline'):
                entry_price = row['close_935']
                if pd.isna(entry_price):
                    continue
                result = simulate_trail_d(rth, entry_price, sl_dist, trade_dir, adr)
            elif strategy_type == 'S2:VWAPHold':
                entry_price = row['close_1000']
                if pd.isna(entry_price):
                    continue
                result = simulate_trail_d(rth, entry_price, sl_dist, trade_dir, adr, entry_time='10:01')
            else:
                continue

            if result:
                pnl_list.append(result['pnl_r'])
        return pnl_list

    for i, combo in enumerate(top5_for_oos):
        lines.append("")
        lines.append(f"  --- OOS Combo #{i+1}: {combo['strategy']} | {combo['label']} ---")

        # Apply same filters to OOS
        oos_subset = apply_filters(
            valid_oos, combo['gap_dir'], combo['gap_min'], combo['od_min'],
            combo['rvol_min'], combo['rsi'], combo['pm_max'],
            require_confirm=(combo['strategy'] == 'S1:MomConfirm'),
            require_vwap_hold=(combo['strategy'] == 'S2:VWAPHold')
        )

        # IS results
        is_pnl = combo['pnl_list']
        is_mean, is_lo, is_hi = bootstrap_ci(is_pnl)
        lines.append(f"  IS:  N={combo['n']:>4} | EV={combo['ev']:>+.3f}R | 95%CI=[{is_lo:>+.3f}, {is_hi:>+.3f}] | WR={combo['wr']:.1%} | p={combo['pval']:.4f}")

        # OOS results
        oos_pnl = run_oos_strategy(oos_subset, combo['strategy'])
        if len(oos_pnl) >= 10:
            oos_arr = np.array(oos_pnl)
            oos_ev = np.mean(oos_arr)
            oos_wr = np.mean(oos_arr > 0)
            oos_t, oos_p = stats.ttest_1samp(oos_arr, 0)
            oos_mean, oos_lo, oos_hi = bootstrap_ci(oos_pnl)

            # Degradation
            if combo['ev'] != 0:
                degradation = 1 - (oos_ev / combo['ev'])
            else:
                degradation = np.nan

            lines.append(f"  OOS: N={len(oos_arr):>4} | EV={oos_ev:>+.3f}R | 95%CI=[{oos_lo:>+.3f}, {oos_hi:>+.3f}] | WR={oos_wr:.1%} | p={oos_p:.4f}")
            lines.append(f"  Degradation: {degradation:>+.1%}" + (" [BESTAETIGT]" if oos_ev > 0 and oos_p < 0.10 else " [NICHT BESTAETIGT]" if oos_ev <= 0 else " [MARGINAL]"))

            # Check if CI lower bound > 0
            if oos_lo > 0:
                lines.append(f"  >>> CI-Lower > 0: ROBUST <<<")
            elif oos_ev > 0:
                lines.append(f"  >>> EV positiv, aber CI inkludiert 0 <<<")
        else:
            lines.append(f"  OOS: N={len(oos_pnl):>4} [INSUFFICIENT DATA]")

        print(f"  [OOS] Combo #{i+1}: N_oos={len(oos_pnl)}", file=sys.stderr)

    lines.append("")
    lines.append(f"  Phase 4 Laufzeit: {time.time() - t_phase4:.0f}s")

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    lines.append("")
    lines.append("=" * 90)
    lines.append("FINAL SUMMARY")
    lines.append("=" * 90)

    # Find best OOS-validated combo
    lines.append("")
    lines.append("  Beste IS-Kombinationen nach EV (Top 5):")
    for i, c in enumerate(top5_for_oos):
        lines.append(f"    #{i+1}: {c['strategy']} | {c['label']} | EV={c['ev']:>+.3f}R | N={c['n']} | p={c['pval']:.4f}")

    lines.append("")
    lines.append("  Methodologie:")
    lines.append("  - 5 Strategie-Ideen mit Entry VOR 10:00 Uhr getestet")
    lines.append("  - 3 Strategievarianten (S1: Momentum Confirm, S2: VWAP Hold, S3: Baseline)")
    lines.append(f"  - {len(all_combos)} Filter-Kombinationen gescannt")
    lines.append(f"  - {len(significant)} signifikante Kombinationen (p<0.05, EV>=0.25R, N>=30)")
    lines.append("  - Top 5 OOS-validiert mit Bootstrap 10000 CI")
    lines.append("")
    lines.append(f"  Gesamt-Laufzeit: {time.time() - t_start:.0f}s")

    # Write output
    output = "\n".join(lines)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(output)
    print(output)
    print(f"\n[DONE] Saved to {OUTPUT_FILE}", file=sys.stderr)


if __name__ == '__main__':
    main()
