"""
D10 Aufgabe 4: Trailing-Stop Simulation (IS)
TRAIL_A/B/C/D, Fix comparison, Setup B (RVOL_30>5x).
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

RAW_DIR = Path('data/raw_1min')


def simulate_trailing(bars_rth, entry_price, sl_dist, trade_dir, adr, trail_type='A'):
    """
    Trailing-stop simulation.
    trail_type: A, B, C, D
    Returns dict with PnL and exit info.
    """
    if sl_dist <= 0 or pd.isna(sl_dist) or adr <= 0:
        return None

    post_entry = bars_rth[(bars_rth['time_et'] >= '09:36') & (bars_rth['time_et'] <= '15:55')]
    if len(post_entry) == 0:
        return None

    # Initial SL
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
        # Current favorable excursion in R
        if trade_dir == 'long':
            current_r = max(0, (bar['high'] - entry_price) / sl_dist)
        else:
            current_r = max(0, (entry_price - bar['low']) / sl_dist)

        highest_r = max(highest_r, current_r)

        # Trail activation and level update
        if trail_type == 'A':
            # After +1R: trail to BE, then trail = peak - 1R
            if highest_r >= 1.0 and not trail_active:
                trail_active = True
                if trade_dir == 'long':
                    sl_level = entry_price  # BE
                else:
                    sl_level = entry_price
            if trail_active:
                if trade_dir == 'long':
                    new_sl = entry_price + (highest_r - 1.0) * sl_dist
                    sl_level = max(sl_level, new_sl)
                else:
                    new_sl = entry_price - (highest_r - 1.0) * sl_dist
                    sl_level = min(sl_level, new_sl)

        elif trail_type == 'B':
            # After +1R: trail to +0.5R, then trail = peak - 1R
            if highest_r >= 1.0 and not trail_active:
                trail_active = True
                if trade_dir == 'long':
                    sl_level = entry_price + 0.5 * sl_dist
                else:
                    sl_level = entry_price - 0.5 * sl_dist
            if trail_active:
                if trade_dir == 'long':
                    new_sl = entry_price + (highest_r - 1.0) * sl_dist
                    sl_level = max(sl_level, new_sl)
                else:
                    new_sl = entry_price - (highest_r - 1.0) * sl_dist
                    sl_level = min(sl_level, new_sl)

        elif trail_type == 'C':
            # After +2R: trail to +1R, then trail = peak - 1R
            if highest_r >= 2.0 and not trail_active:
                trail_active = True
                if trade_dir == 'long':
                    sl_level = entry_price + 1.0 * sl_dist
                else:
                    sl_level = entry_price - 1.0 * sl_dist
            if trail_active:
                if trade_dir == 'long':
                    new_sl = entry_price + (highest_r - 1.0) * sl_dist
                    sl_level = max(sl_level, new_sl)
                else:
                    new_sl = entry_price - (highest_r - 1.0) * sl_dist
                    sl_level = min(sl_level, new_sl)

        elif trail_type == 'D':
            # After +1R: trail to BE, then trail = peak - 0.5R (tight)
            if highest_r >= 1.0 and not trail_active:
                trail_active = True
                if trade_dir == 'long':
                    sl_level = entry_price
                else:
                    sl_level = entry_price
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

    # Timeout
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
        'pnl_r': pnl / sl_dist,
        'pnl_adr': pnl / adr,
        'exit_type': exit_type,
        'exit_time': exit_time,
        'highest_r': highest_r,
        'trail_active': trail_active,
        'winner': pnl > 0,
    }


def simulate_fix_target(bars_rth, entry_price, sl_dist, trade_dir, adr, target_r):
    """Fixed target simulation: exit at target_r or SL."""
    if sl_dist <= 0 or pd.isna(sl_dist) or adr <= 0:
        return None

    post_entry = bars_rth[(bars_rth['time_et'] >= '09:36') & (bars_rth['time_et'] <= '15:55')]
    if len(post_entry) == 0:
        return None

    if trade_dir == 'long':
        sl_level = entry_price - sl_dist
        target_level = entry_price + target_r * sl_dist
    else:
        sl_level = entry_price + sl_dist
        target_level = entry_price - target_r * sl_dist

    exit_price = None
    exit_type = 'timeout'

    for _, bar in post_entry.iterrows():
        if trade_dir == 'long':
            # SL first (conservative)
            if bar['low'] <= sl_level:
                exit_price = sl_level
                exit_type = 'sl'
                break
            if bar['high'] >= target_level:
                exit_price = target_level
                exit_type = 'target'
                break
        else:
            if bar['high'] >= sl_level:
                exit_price = sl_level
                exit_type = 'sl'
                break
            if bar['low'] <= target_level:
                exit_price = target_level
                exit_type = 'target'
                break

    if exit_price is None:
        exit_price = post_entry.iloc[-1]['close']

    if trade_dir == 'long':
        pnl = exit_price - entry_price
    else:
        pnl = entry_price - exit_price

    return {
        'pnl': pnl,
        'pnl_r': pnl / sl_dist,
        'pnl_adr': pnl / adr,
        'exit_type': exit_type,
        'winner': pnl > 0,
    }


def run_strategy(meta_subset, strategy_type, **kwargs):
    """Run a strategy on a meta subset. Returns DataFrame."""
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
        gap_dir = row['gap_direction']

        if pd.isna(entry_price) or pd.isna(adr) or adr <= 0:
            continue

        trade_dir = 'long' if gap_dir == 'up' else 'short'
        sl_dist = 0.25 * adr

        if strategy_type == 'trail':
            result = simulate_trailing(rth, entry_price, sl_dist, trade_dir, adr,
                                       trail_type=kwargs.get('trail_type', 'A'))
        elif strategy_type == 'fix':
            result = simulate_fix_target(rth, entry_price, sl_dist, trade_dir, adr,
                                         target_r=kwargs.get('target_r', 2.0))
        else:
            continue

        if result is None:
            continue

        result['ticker'] = ticker
        result['date'] = date
        result['gap_direction'] = gap_dir
        result['rvol_30'] = row.get('rvol_open_30min', np.nan)
        all_results.append(result)

    return pd.DataFrame(all_results)


def summarize_strategy(trades, label=""):
    """Summary stats for a strategy."""
    n = len(trades)
    if n < 10:
        return f"  {label:<25} | N={n:>4} [N<10]"

    med_pnl_r = trades['pnl_r'].median()
    mean_pnl_r = trades['pnl_r'].mean()
    std_pnl_r = trades['pnl_r'].std()
    wr = trades['winner'].mean()
    ev_adr = trades['pnl_adr'].mean()
    med_pnl_adr = trades['pnl_adr'].median()

    low_n = " [LOW N]" if n < 20 else ""
    return (f"  {label:<25} | N={n:>4} | MedPnL={med_pnl_r:>+6.2f}R | "
            f"MeanPnL={mean_pnl_r:>+6.2f}R | StdDev={std_pnl_r:>5.2f} | "
            f"WR={wr:>5.1%} | EV={ev_adr:>+6.3f}ADR{low_n}")


def main():
    lines = []
    lines.append("=" * 70)
    lines.append("D10 AUFGABE 4: TRAILING-STOP SIMULATION (IS)")
    lines.append("=" * 70)
    lines.append("\nEntry: 9:35 | SL: 0.25 ADR | Timeout: 15:55")
    lines.append("TRAIL_A: +1R -> BE, trail peak-1R")
    lines.append("TRAIL_B: +1R -> +0.5R, trail peak-1R")
    lines.append("TRAIL_C: +2R -> +1R, trail peak-1R")
    lines.append("TRAIL_D: +1R -> BE, trail peak-0.5R (eng)")

    meta = pd.read_parquet('data/metadata/metadata_v8_5.parquet')
    h1 = meta[(meta['date'] >= '2021-02-21') & (meta['date'] <= '2023-12-31')].copy()
    base = h1[(h1['od_strength'] > 0.5) & (h1['od_direction'] == 'with_gap')]
    print(f"Base: {len(base)}", file=sys.stderr)

    # ===== 4a: Trail simulations =====
    lines.append(f"\n\n{'='*70}")
    lines.append("4a: TRAILING-STOP ERGEBNISSE")
    lines.append(f"{'='*70}")

    for gap_dir, gap_label in [('up', 'GapUp'), ('down', 'GapDown')]:
        lines.append(f"\n--- {gap_label} ---")
        gap_base = base[base['gap_direction'] == gap_dir]

        for trail_type in ['A', 'B', 'C', 'D']:
            print(f"  Running TRAIL_{trail_type} {gap_label}...", file=sys.stderr)
            trades = run_strategy(gap_base, 'trail', trail_type=trail_type)
            lines.append(summarize_strategy(trades, f"TRAIL_{trail_type}"))

            # Exit type breakdown
            if len(trades) >= 10:
                n_init = (trades['exit_type'] == 'initial_sl').sum()
                n_trail = (trades['exit_type'] == 'trail_sl').sum()
                n_to = (trades['exit_type'] == 'timeout').sum()
                lines.append(f"    Exit: InitSL={n_init} ({100*n_init/len(trades):.0f}%), "
                             f"TrailSL={n_trail} ({100*n_trail/len(trades):.0f}%), "
                             f"Timeout={n_to} ({100*n_to/len(trades):.0f}%)")

    # ===== 4b: Fix vs Trail Comparison =====
    lines.append(f"\n\n{'='*70}")
    lines.append("4b: FIX vs TRAIL VERGLEICH")
    lines.append(f"{'='*70}")

    for gap_dir, gap_label in [('up', 'GapUp'), ('down', 'GapDown')]:
        lines.append(f"\n--- {gap_label} ---")
        gap_base = base[base['gap_direction'] == gap_dir]

        # Fix targets
        for target_r in [1.0, 2.0, 3.0]:
            print(f"  Running FIX_{target_r}R {gap_label}...", file=sys.stderr)
            trades = run_strategy(gap_base, 'fix', target_r=target_r)
            lines.append(summarize_strategy(trades, f"Fix {target_r:.0f}R"))

        # Trail types
        for trail_type in ['A', 'B', 'C', 'D']:
            trades = run_strategy(gap_base, 'trail', trail_type=trail_type)
            lines.append(summarize_strategy(trades, f"TRAIL_{trail_type}"))

    # Summary table
    lines.append(f"\n\n  === KOMPAKT-VERGLEICH (EV in ADR) ===\n")
    lines.append(f"  {'Strategie':<20} | {'EV GapUp':>10} | {'EV GapDn':>10} | {'WR GapUp':>10} | {'WR GapDn':>10}")
    lines.append(f"  {'-'*68}")

    summary_data = {}
    for gap_dir, gap_label in [('up', 'GapUp'), ('down', 'GapDn')]:
        gap_base = base[base['gap_direction'] == gap_dir]

        for target_r in [1.0, 2.0, 3.0]:
            trades = run_strategy(gap_base, 'fix', target_r=target_r)
            key = f"Fix {target_r:.0f}R"
            if key not in summary_data:
                summary_data[key] = {}
            summary_data[key][gap_dir] = trades

        for trail_type in ['A', 'B', 'C', 'D']:
            trades = run_strategy(gap_base, 'trail', trail_type=trail_type)
            key = f"TRAIL_{trail_type}"
            if key not in summary_data:
                summary_data[key] = {}
            summary_data[key][gap_dir] = trades

    for key in ['Fix 1R', 'Fix 2R', 'Fix 3R', 'TRAIL_A', 'TRAIL_B', 'TRAIL_C', 'TRAIL_D']:
        ev_up = summary_data[key]['up']['pnl_adr'].mean() if len(summary_data[key].get('up', [])) >= 10 else np.nan
        ev_dn = summary_data[key]['down']['pnl_adr'].mean() if len(summary_data[key].get('down', [])) >= 10 else np.nan
        wr_up = summary_data[key]['up']['winner'].mean() if len(summary_data[key].get('up', [])) >= 10 else np.nan
        wr_dn = summary_data[key]['down']['winner'].mean() if len(summary_data[key].get('down', [])) >= 10 else np.nan

        ev_up_s = f"{ev_up:>+10.3f}" if pd.notna(ev_up) else f"{'---':>10}"
        ev_dn_s = f"{ev_dn:>+10.3f}" if pd.notna(ev_dn) else f"{'---':>10}"
        wr_up_s = f"{wr_up:>10.1%}" if pd.notna(wr_up) else f"{'---':>10}"
        wr_dn_s = f"{wr_dn:>10.1%}" if pd.notna(wr_dn) else f"{'---':>10}"
        lines.append(f"  {key:<20} | {ev_up_s} | {ev_dn_s} | {wr_up_s} | {wr_dn_s}")

    # Sharpe-like metric
    lines.append(f"\n  === SHARPE-AEHNLICH (EV/StdDev) ===\n")
    lines.append(f"  {'Strategie':<20} | {'Sharpe GapUp':>12} | {'Sharpe GapDn':>12}")
    lines.append(f"  {'-'*50}")

    for key in ['Fix 1R', 'Fix 2R', 'Fix 3R', 'TRAIL_A', 'TRAIL_B', 'TRAIL_C', 'TRAIL_D']:
        for gap_dir, gap_label in [('up', 'GapUp'), ('down', 'GapDn')]:
            t = summary_data[key].get(gap_dir, pd.DataFrame())
            if len(t) >= 10:
                ev = t['pnl_adr'].mean()
                std = t['pnl_adr'].std()
                summary_data[key][f'sharpe_{gap_dir}'] = ev / std if std > 0 else 0
            else:
                summary_data[key][f'sharpe_{gap_dir}'] = np.nan

        s_up = summary_data[key].get('sharpe_up', np.nan)
        s_dn = summary_data[key].get('sharpe_down', np.nan)
        s_up_s = f"{s_up:>+12.3f}" if pd.notna(s_up) else f"{'---':>12}"
        s_dn_s = f"{s_dn:>+12.3f}" if pd.notna(s_dn) else f"{'---':>12}"
        lines.append(f"  {key:<20} | {s_up_s} | {s_dn_s}")

    # ===== 4c: Setup B (RVOL_30 > 5x) =====
    lines.append(f"\n\n{'='*70}")
    lines.append("4c: TRAILING BEI SETUP B (RVOL_30 > 5x)")
    lines.append(f"{'='*70}")

    setup_b = base[base['rvol_open_30min'] >= 5]
    print(f"Setup B: {len(setup_b)}", file=sys.stderr)

    for gap_dir, gap_label in [('up', 'GapUp'), ('down', 'GapDown')]:
        lines.append(f"\n--- {gap_label} ---")
        gap_sub = setup_b[setup_b['gap_direction'] == gap_dir]

        for target_r in [1.0, 2.0, 3.0]:
            trades = run_strategy(gap_sub, 'fix', target_r=target_r)
            lines.append(summarize_strategy(trades, f"Fix {target_r:.0f}R"))

        for trail_type in ['A', 'B', 'C', 'D']:
            trades = run_strategy(gap_sub, 'trail', trail_type=trail_type)
            lines.append(summarize_strategy(trades, f"TRAIL_{trail_type}"))

    output = "\n".join(lines)
    with open('results/d10_trailing.txt', 'w', encoding='utf-8') as f:
        f.write(output)
    print(output)
    print(f"\nSaved to results/d10_trailing.txt", file=sys.stderr)


if __name__ == '__main__':
    main()
