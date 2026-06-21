"""
D9 Aufgabe 2: R:R-basierte Trade-Simulation — Basis-Analyse (IS)
Bar-by-bar simulation with SL_FULL, SL_HALF, SL_FIX025.
WR@1R/2R/3R, EV, SL-hit timing.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

RAW_DIR = Path('data/raw_1min')


def simulate_rr_trade(bars_rth, entry_price, sl_level, trade_dir, adr, rr_targets=[1,2,3]):
    """
    Bar-by-bar R:R trade simulation.
    Returns dict with hit flags for each R:R target, SL hit, timeout info.
    """
    sl_dist = abs(entry_price - sl_level)
    if sl_dist <= 0 or pd.isna(sl_dist):
        return None

    # Target levels
    targets = {}
    for rr in rr_targets:
        if trade_dir == 'long':
            targets[rr] = entry_price + rr * sl_dist
        else:
            targets[rr] = entry_price - rr * sl_dist

    results = {f'hit_{rr}R': False for rr in rr_targets}
    results['sl_hit'] = False
    results['sl_time'] = None
    results['exit_price'] = np.nan
    results['exit_type'] = 'timeout'
    results['sl_dist_adr'] = sl_dist / adr if adr > 0 else np.nan

    # Max favorable excursion tracking
    best_rr_reached = 0

    post_entry = bars_rth[bars_rth['time_et'] >= '09:36']

    for _, bar in post_entry.iterrows():
        if bar['time_et'] > '15:55':
            break

        if trade_dir == 'long':
            # Check SL first (conservative)
            if bar['low'] <= sl_level:
                results['sl_hit'] = True
                results['exit_price'] = sl_level
                results['exit_type'] = 'sl'
                results['sl_time'] = bar['time_et']
                break
            # Check targets (best case: high reaches target)
            for rr in sorted(rr_targets):
                if bar['high'] >= targets[rr] and not results[f'hit_{rr}R']:
                    results[f'hit_{rr}R'] = True
                    best_rr_reached = max(best_rr_reached, rr)
        else:  # short
            if bar['high'] >= sl_level:
                results['sl_hit'] = True
                results['exit_price'] = sl_level
                results['exit_type'] = 'sl'
                results['sl_time'] = bar['time_et']
                break
            for rr in sorted(rr_targets):
                if bar['low'] <= targets[rr] and not results[f'hit_{rr}R']:
                    results[f'hit_{rr}R'] = True
                    best_rr_reached = max(best_rr_reached, rr)

    # Timeout: use last bar close
    if not results['sl_hit']:
        timeout_bars = post_entry[post_entry['time_et'] <= '15:55']
        if len(timeout_bars) > 0:
            results['exit_price'] = timeout_bars.iloc[-1]['close']

    # PnL in ADR
    if pd.notna(results['exit_price']) and adr > 0:
        if trade_dir == 'long':
            results['pnl_adr'] = (results['exit_price'] - entry_price) / adr
        else:
            results['pnl_adr'] = (entry_price - results['exit_price']) / adr

    # PnL in R
    if pd.notna(results['exit_price']) and sl_dist > 0:
        if trade_dir == 'long':
            results['pnl_r'] = (results['exit_price'] - entry_price) / sl_dist
        else:
            results['pnl_r'] = (entry_price - results['exit_price']) / sl_dist

    results['best_rr'] = best_rr_reached
    return results


def run_simulation(meta_subset, sl_type='sl_full'):
    """Run trade simulation for a metadata subset with given SL type."""
    all_results = []

    for _, row in meta_subset.iterrows():
        ticker = row['ticker']
        date = str(row['date'])
        fpath = RAW_DIR / ticker / f"{date}.parquet"
        if not fpath.exists():
            continue

        bars = pd.read_parquet(fpath)
        rth = bars[bars['session'] == 'rth']
        if len(rth) < 10:
            continue

        entry_price = row['close_935']
        adr = row.get('adr_10', np.nan)
        trade_dir = row.get('trade_direction', None)

        if pd.isna(entry_price) or pd.isna(adr) or adr <= 0 or trade_dir is None:
            continue

        # Determine SL level
        if sl_type == 'sl_full':
            sl_level = row.get('sl_full_level', np.nan)
        elif sl_type == 'sl_half':
            sl_level = row.get('sl_half_level', np.nan)
        elif sl_type == 'sl_fix025':
            if trade_dir == 'long':
                sl_level = entry_price - 0.25 * adr
            else:
                sl_level = entry_price + 0.25 * adr
        else:
            continue

        if pd.isna(sl_level):
            continue

        result = simulate_rr_trade(rth, entry_price, sl_level, trade_dir, adr)
        if result is None:
            continue

        result['ticker'] = ticker
        result['date'] = date
        result['sl_type'] = sl_type
        result['od_direction'] = row['od_direction']
        result['gap_direction'] = row['gap_direction']
        all_results.append(result)

    return pd.DataFrame(all_results)


def summarize_rr(trades, label=""):
    """Summarize R:R trade results."""
    n = len(trades)
    if n < 10:
        return f"  {label:<30} | N={n:>4} [N<10]"

    med_sl = trades['sl_dist_adr'].median()
    wr1 = trades['hit_1R'].mean()
    wr2 = trades['hit_2R'].mean()
    wr3 = trades['hit_3R'].mean()
    sl_pct = trades['sl_hit'].mean()
    to_pct = (trades['exit_type'] == 'timeout').mean()

    # EV calculations (using actual PnL including timeouts)
    ev_adr = trades['pnl_adr'].mean() if 'pnl_adr' in trades.columns else np.nan

    # Simplified EV at 2R target (assume take profit at 2R, SL at 1R)
    ev_2r_simple = (wr2 * 2 - (1-wr2) * 1) * med_sl if pd.notna(med_sl) else np.nan

    low_n = " [LOW N]" if n < 20 else ""
    return (f"  {label:<30} | N={n:>4} | MedSL={med_sl:.3f} | "
            f"WR@1R={wr1:>5.1%} | WR@2R={wr2:>5.1%} | WR@3R={wr3:>5.1%} | "
            f"EV@2R={ev_2r_simple:>+.3f} | SL%={sl_pct:>5.1%} | TO%={to_pct:>5.1%}{low_n}")


def sl_timing_analysis(trades, label=""):
    """Analyze when SL hits occur."""
    sl_trades = trades[trades['sl_hit'] == True].copy()
    if len(sl_trades) < 10:
        return f"  {label}: N_SL={len(sl_trades)} [zu wenig]"

    total_sl = len(sl_trades)
    lines = [f"  {label}: N_SL={total_sl}"]

    time_buckets = [
        ('< 5min (bis 09:40)', '09:36', '09:40'),
        ('5-15min (09:41-09:50)', '09:41', '09:50'),
        ('15-30min (09:51-10:05)', '09:51', '10:05'),
        ('30-60min (10:06-10:35)', '10:06', '10:35'),
        ('> 60min (10:36+)', '10:36', '15:55'),
    ]

    for blabel, t_lo, t_hi in time_buckets:
        count = len(sl_trades[(sl_trades['sl_time'] >= t_lo) & (sl_trades['sl_time'] <= t_hi)])
        pct = count / total_sl if total_sl > 0 else 0
        lines.append(f"    {blabel:<30}: {count:>4} ({pct:>5.1%})")

    return "\n".join(lines)


def main():
    lines = []
    lines.append("=" * 70)
    lines.append("D9 AUFGABE 2: R:R-BASIERTE TRADE-SIMULATION — BASIS (IS)")
    lines.append("=" * 70)
    lines.append("\nEntry: 9:35 | Timeout: 15:55 | R:R-Targets: 1R, 2R, 3R")
    lines.append("Break-Even WRs: 1R=50.0%, 2R=33.3%, 3R=25.0%")

    meta = pd.read_parquet('data/metadata/metadata_v9.parquet')
    h1 = meta[(meta['date'] >= '2021-02-21') & (meta['date'] <= '2023-12-31')].copy()
    h1 = h1[h1['od_body_pct'].notna() & h1['sl_full_adr'].notna()].copy()

    # ===== 2a: OD > 0.5 with Gap =====
    lines.append(f"\n\n{'='*70}")
    lines.append("2a: OD > 0.5 WITH GAP (Continuation)")
    lines.append(f"{'='*70}")

    for gap_dir, gap_label in [('up', 'GapUp'), ('down', 'GapDown')]:
        lines.append(f"\n--- {gap_label} ---")
        sub = h1[(h1['od_strength'] > 0.5) & (h1['od_direction'] == 'with_gap') & (h1['gap_direction'] == gap_dir)]
        print(f"  {gap_label} with: N={len(sub)}", file=sys.stderr)

        for sl_type, sl_label in [('sl_full', 'SL_FULL'), ('sl_half', 'SL_HALF'), ('sl_fix025', 'SL_FIX025')]:
            trades = run_simulation(sub, sl_type)
            lines.append(summarize_rr(trades, sl_label))

    # ===== 2b: OD > 0.5 against Gap =====
    lines.append(f"\n\n{'='*70}")
    lines.append("2b: OD > 0.5 AGAINST GAP (Fade, trade in OD-Richtung)")
    lines.append(f"{'='*70}")

    for gap_dir, gap_label in [('up', 'GapUp'), ('down', 'GapDown')]:
        lines.append(f"\n--- {gap_label} ---")
        sub = h1[(h1['od_strength'] > 0.5) & (h1['od_direction'] == 'against_gap') & (h1['gap_direction'] == gap_dir)]
        print(f"  {gap_label} against: N={len(sub)}", file=sys.stderr)

        for sl_type, sl_label in [('sl_full', 'SL_FULL'), ('sl_half', 'SL_HALF'), ('sl_fix025', 'SL_FIX025')]:
            trades = run_simulation(sub, sl_type)
            lines.append(summarize_rr(trades, sl_label))

    # ===== 2c: SL-Hit Timing =====
    lines.append(f"\n\n{'='*70}")
    lines.append("2c: SL-HIT TIMING (OD > 0.5 with Gap)")
    lines.append(f"{'='*70}")

    all_with = h1[(h1['od_strength'] > 0.5) & (h1['od_direction'] == 'with_gap')]
    for sl_type, sl_label in [('sl_full', 'SL_FULL'), ('sl_half', 'SL_HALF'), ('sl_fix025', 'SL_FIX025')]:
        trades = run_simulation(all_with, sl_type)
        lines.append(f"\n{sl_timing_analysis(trades, sl_label)}")

    lines.append(f"\n\n{'='*70}")
    lines.append("2c: SL-HIT TIMING (OD > 0.5 against Gap)")
    lines.append(f"{'='*70}")

    all_against = h1[(h1['od_strength'] > 0.5) & (h1['od_direction'] == 'against_gap')]
    for sl_type, sl_label in [('sl_full', 'SL_FULL'), ('sl_half', 'SL_HALF'), ('sl_fix025', 'SL_FIX025')]:
        trades = run_simulation(all_against, sl_type)
        lines.append(f"\n{sl_timing_analysis(trades, sl_label)}")

    output = "\n".join(lines)
    with open('results/d9_rr_basis.txt', 'w', encoding='utf-8') as f:
        f.write(output)
    print(output)
    print(f"\nSaved to results/d9_rr_basis.txt", file=sys.stderr)


if __name__ == '__main__':
    main()
