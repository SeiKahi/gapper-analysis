"""
D10 Aufgabe 5: OOS-Validierung MFE + Trailing (H2)
Reproduces key IS findings on OOS data.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

RAW_DIR = Path('data/raw_1min')


def compute_mfe_mae(bars_rth, entry_price, sl_level, trade_dir, adr):
    """Compute MFE_LIVE, MFE_FULL, MAE."""
    sl_dist = abs(entry_price - sl_level)
    if sl_dist <= 0 or pd.isna(sl_dist) or adr <= 0:
        return None

    post_entry = bars_rth[(bars_rth['time_et'] >= '09:36') & (bars_rth['time_et'] <= '15:55')]
    if len(post_entry) == 0:
        return None

    mfe_full = 0.0
    mae = 0.0
    sl_hit = False
    mfe_at_sl = 0.0

    for _, bar in post_entry.iterrows():
        if trade_dir == 'long':
            fav = max(0, bar['high'] - entry_price)
            adv = max(0, entry_price - bar['low'])
        else:
            fav = max(0, entry_price - bar['low'])
            adv = max(0, bar['high'] - entry_price)

        mfe_full = max(mfe_full, fav)
        mae = max(mae, adv)

        if not sl_hit:
            mfe_at_sl = mfe_full
            if trade_dir == 'long' and bar['low'] <= sl_level:
                sl_hit = True
            elif trade_dir == 'short' and bar['high'] >= sl_level:
                sl_hit = True

    mfe_live = mfe_at_sl if sl_hit else mfe_full

    return {
        'mfe_live': mfe_live,
        'mfe_full': mfe_full,
        'mae': mae,
        'mfe_live_r': mfe_live / sl_dist,
        'mfe_full_r': mfe_full / sl_dist,
        'mae_r': mae / sl_dist,
        'mfe_live_adr': mfe_live / adr,
        'mfe_full_adr': mfe_full / adr,
        'mae_adr': mae / adr,
        'sl_hit': sl_hit,
        'sl_dist': sl_dist,
        'sl_dist_adr': sl_dist / adr,
    }


def simulate_trailing(bars_rth, entry_price, sl_dist, trade_dir, adr, trail_type='D'):
    """Trailing-stop simulation."""
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

    for _, bar in post_entry.iterrows():
        if trade_dir == 'long':
            current_r = max(0, (bar['high'] - entry_price) / sl_dist)
        else:
            current_r = max(0, (entry_price - bar['low']) / sl_dist)

        highest_r = max(highest_r, current_r)

        if trail_type == 'D':
            if highest_r >= 1.0 and not trail_active:
                trail_active = True
                sl_level = entry_price if trade_dir == 'long' else entry_price
            if trail_active:
                if trade_dir == 'long':
                    new_sl = entry_price + (highest_r - 0.5) * sl_dist
                    sl_level = max(sl_level, new_sl)
                else:
                    new_sl = entry_price - (highest_r - 0.5) * sl_dist
                    sl_level = min(sl_level, new_sl)
        elif trail_type == 'A':
            if highest_r >= 1.0 and not trail_active:
                trail_active = True
                sl_level = entry_price if trade_dir == 'long' else entry_price
            if trail_active:
                if trade_dir == 'long':
                    new_sl = entry_price + (highest_r - 1.0) * sl_dist
                    sl_level = max(sl_level, new_sl)
                else:
                    new_sl = entry_price - (highest_r - 1.0) * sl_dist
                    sl_level = min(sl_level, new_sl)

        if trade_dir == 'long' and bar['low'] <= sl_level:
            exit_price = sl_level
            exit_type = 'trail_sl' if trail_active else 'initial_sl'
            break
        elif trade_dir == 'short' and bar['high'] >= sl_level:
            exit_price = sl_level
            exit_type = 'trail_sl' if trail_active else 'initial_sl'
            break

    if exit_price is None:
        exit_price = post_entry.iloc[-1]['close']

    pnl = (exit_price - entry_price) if trade_dir == 'long' else (entry_price - exit_price)

    return {
        'pnl': pnl,
        'pnl_r': pnl / sl_dist,
        'pnl_adr': pnl / adr,
        'exit_type': exit_type,
        'winner': pnl > 0,
    }


def simulate_fix(bars_rth, entry_price, sl_dist, trade_dir, adr, target_r=2.0):
    """Fixed target simulation."""
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

    for _, bar in post_entry.iterrows():
        if trade_dir == 'long':
            if bar['low'] <= sl_level:
                exit_price = sl_level
                break
            if bar['high'] >= target_level:
                exit_price = target_level
                break
        else:
            if bar['high'] >= sl_level:
                exit_price = sl_level
                break
            if bar['low'] <= target_level:
                exit_price = target_level
                break

    if exit_price is None:
        exit_price = post_entry.iloc[-1]['close']

    pnl = (exit_price - entry_price) if trade_dir == 'long' else (entry_price - exit_price)

    return {
        'pnl': pnl,
        'pnl_r': pnl / sl_dist,
        'pnl_adr': pnl / adr,
        'winner': pnl > 0,
    }


def run_all(meta_subset, compute_type='mfe'):
    """Run analysis on a subset, return DataFrame."""
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
        sl_level = entry_price - sl_dist if trade_dir == 'long' else entry_price + sl_dist

        if compute_type == 'mfe':
            result = compute_mfe_mae(rth, entry_price, sl_level, trade_dir, adr)
        elif compute_type.startswith('trail_'):
            tt = compute_type.split('_')[1]
            result = simulate_trailing(rth, entry_price, sl_dist, trade_dir, adr, trail_type=tt)
        elif compute_type.startswith('fix_'):
            tr = float(compute_type.split('_')[1])
            result = simulate_fix(rth, entry_price, sl_dist, trade_dir, adr, target_r=tr)
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


def main():
    lines = []
    lines.append("=" * 70)
    lines.append("D10 AUFGABE 5: OOS-VALIDIERUNG MFE + TRAILING")
    lines.append("=" * 70)
    lines.append("\nH2 (OOS): 2024-01-01 bis 2026-02-06")

    meta = pd.read_parquet('data/metadata/metadata_v8_5.parquet')
    h1 = meta[(meta['date'] >= '2021-02-21') & (meta['date'] <= '2023-12-31')].copy()
    h2 = meta[(meta['date'] >= '2024-01-01') & (meta['date'] <= '2026-02-06')].copy()

    base_h1 = h1[(h1['od_strength'] > 0.5) & (h1['od_direction'] == 'with_gap')]
    base_h2 = h2[(h2['od_strength'] > 0.5) & (h2['od_direction'] == 'with_gap')]
    print(f"H1: {len(base_h1)}, H2: {len(base_h2)}", file=sys.stderr)

    # Load IS MFE data
    is_mfe = pd.read_parquet('results/d10_mfe_raw_is.parquet')

    # Compute OOS MFE
    print("Computing OOS MFE...", file=sys.stderr)
    oos_mfe = run_all(base_h2, 'mfe')
    oos_mfe.to_parquet('results/d10_mfe_raw_oos.parquet', index=False)
    print(f"OOS: {len(oos_mfe)} trades", file=sys.stderr)

    # ===== 5a: MFE Distribution IS vs OOS =====
    lines.append(f"\n\n{'='*70}")
    lines.append("5a: MFE-VERTEILUNG IS vs OOS")
    lines.append(f"{'='*70}")

    for gap_dir, gap_label in [('up', 'GapUp'), ('down', 'GapDn')]:
        lines.append(f"\n--- {gap_label} ---")
        is_sub = is_mfe[is_mfe['gap_direction'] == gap_dir]
        oos_sub = oos_mfe[oos_mfe['gap_direction'] == gap_dir]

        lines.append(f"  {'Metrik':<20} | {'IS MFE_R':>10} | {'IS MFE_ADR':>10} | {'OOS MFE_R':>10} | {'OOS MFE_ADR':>10}")
        lines.append(f"  {'-'*68}")

        for label, fn in [('N', lambda s: len(s)),
                          ('Survival%', lambda s: f"{(s['sl_hit']==False).mean():.1%}"),
                          ('Median LIVE', lambda s: f"{s['mfe_live_r'].median():.2f}"),
                          ('Mean LIVE', lambda s: f"{s['mfe_live_r'].mean():.2f}"),
                          ('StdDev LIVE', lambda s: f"{s['mfe_live_r'].std():.2f}"),
                          ('P25', lambda s: f"{s['mfe_live_r'].quantile(0.25):.2f}"),
                          ('P75', lambda s: f"{s['mfe_live_r'].quantile(0.75):.2f}"),
                          ('P90', lambda s: f"{s['mfe_live_r'].quantile(0.90):.2f}")]:

            if label == 'N':
                lines.append(f"  {label:<20} | {fn(is_sub):>10} | {'':>10} | {fn(oos_sub):>10} | {'':>10}")
            elif label == 'Survival%':
                lines.append(f"  {label:<20} | {fn(is_sub):>10} | {'':>10} | {fn(oos_sub):>10} | {'':>10}")
            else:
                is_r = fn(is_sub)
                oos_r = fn(oos_sub)
                # Also compute ADR versions
                if 'Median' in label:
                    is_adr = f"{is_sub['mfe_live_adr'].median():.3f}"
                    oos_adr = f"{oos_sub['mfe_live_adr'].median():.3f}"
                elif 'Mean' in label:
                    is_adr = f"{is_sub['mfe_live_adr'].mean():.3f}"
                    oos_adr = f"{oos_sub['mfe_live_adr'].mean():.3f}"
                elif 'StdDev' in label:
                    is_adr = f"{is_sub['mfe_live_adr'].std():.3f}"
                    oos_adr = f"{oos_sub['mfe_live_adr'].std():.3f}"
                else:
                    pct = float(label[1:]) / 100
                    is_adr = f"{is_sub['mfe_live_adr'].quantile(pct):.3f}"
                    oos_adr = f"{oos_sub['mfe_live_adr'].quantile(pct):.3f}"
                lines.append(f"  {label:<20} | {is_r:>10} | {is_adr:>10} | {oos_r:>10} | {oos_adr:>10}")

        # Winners only
        is_w = is_sub[is_sub['sl_hit'] == False]
        oos_w = oos_sub[oos_sub['sl_hit'] == False]
        lines.append(f"\n  Nur Gewinner:")
        lines.append(f"  {'N':>20} | {len(is_w):>10} | {'':>10} | {len(oos_w):>10}")
        if len(is_w) >= 10 and len(oos_w) >= 10:
            lines.append(f"  {'Median MFE_R':>20} | {is_w['mfe_live_r'].median():>10.2f} | {is_w['mfe_live_adr'].median():>10.3f} | {oos_w['mfe_live_r'].median():>10.2f} | {oos_w['mfe_live_adr'].median():>10.3f}")
            lines.append(f"  {'Mean MFE_R':>20} | {is_w['mfe_live_r'].mean():>10.2f} | {is_w['mfe_live_adr'].mean():>10.3f} | {oos_w['mfe_live_r'].mean():>10.2f} | {oos_w['mfe_live_adr'].mean():>10.3f}")
        lines.append("")

    # ===== 5b: Survival Curve IS vs OOS =====
    lines.append(f"\n{'='*70}")
    lines.append("5b: SURVIVAL-KURVE IS vs OOS (MFE_LIVE)")
    lines.append(f"{'='*70}")

    r_levels = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]

    for gap_dir, gap_label in [('up', 'GapUp'), ('down', 'GapDn')]:
        lines.append(f"\n  --- {gap_label} ---")
        is_sub = is_mfe[is_mfe['gap_direction'] == gap_dir]
        oos_sub = oos_mfe[oos_mfe['gap_direction'] == gap_dir]

        lines.append(f"  {'R-Level':>8} | {'IS %':>8} | {'OOS %':>8} | {'Delta':>8}")
        lines.append(f"  {'-'*40}")

        for r in r_levels:
            is_pct = (is_sub['mfe_live_r'] >= r).mean()
            oos_pct = (oos_sub['mfe_live_r'] >= r).mean()
            delta = oos_pct - is_pct
            lines.append(f"  {r:>7.1f}R | {is_pct:>7.1%} | {oos_pct:>7.1%} | {delta:>+7.1%}")

    # ===== 5c: Conditional Probabilities IS vs OOS =====
    lines.append(f"\n\n{'='*70}")
    lines.append("5c: BEDINGTE WK IS vs OOS")
    lines.append(f"{'='*70}")

    for gap_dir, gap_label in [('up', 'GapUp'), ('down', 'GapDn')]:
        lines.append(f"\n  --- {gap_label} ---")
        is_sub = is_mfe[is_mfe['gap_direction'] == gap_dir]
        oos_sub = oos_mfe[oos_mfe['gap_direction'] == gap_dir]

        cond_levels = [1.0, 2.0, 3.0]
        target_levels = [2.0, 3.0, 5.0]

        lines.append(f"  {'Condition':>12} | {'Target':>8} | {'IS P':>8} | {'IS N':>6} | {'OOS P':>8} | {'OOS N':>6}")
        lines.append(f"  {'-'*58}")

        for c in cond_levels:
            for t in target_levels:
                if t <= c:
                    continue
                is_cond = is_sub[is_sub['mfe_live_r'] >= c]
                oos_cond = oos_sub[oos_sub['mfe_live_r'] >= c]

                is_p = (is_cond['mfe_live_r'] >= t).mean() if len(is_cond) >= 10 else np.nan
                oos_p = (oos_cond['mfe_live_r'] >= t).mean() if len(oos_cond) >= 10 else np.nan

                is_p_s = f"{is_p:>7.1%}" if pd.notna(is_p) else " N<10  "
                oos_p_s = f"{oos_p:>7.1%}" if pd.notna(oos_p) else " N<10  "
                is_n = len(is_cond)
                oos_n = len(oos_cond)

                lines.append(f"  {'>= '+str(c)+'R':>12} | {str(t)+'R':>8} | {is_p_s} | {is_n:>6} | {oos_p_s} | {oos_n:>6}")

    # ===== 5d: MFE by RVOL_30 OOS =====
    lines.append(f"\n\n{'='*70}")
    lines.append("5d: MFE NACH RVOL_30 IS vs OOS (alle Trades)")
    lines.append(f"{'='*70}")

    rvol_buckets = [(0, 3, '<3x'), (3, 5, '3-5x'), (5, 10, '5-10x'), (10, 999, '>10x')]

    for gap_dir, gap_label in [('up', 'GapUp'), ('down', 'GapDn')]:
        lines.append(f"\n  --- {gap_label} ---")
        lines.append(f"  {'RVOL_30':<8} | {'IS Med':>8} | {'IS Surv':>8} | {'IS N':>6} | {'OOS Med':>8} | {'OOS Surv':>8} | {'OOS N':>6}")
        lines.append(f"  {'-'*65}")

        is_sub = is_mfe[is_mfe['gap_direction'] == gap_dir]
        oos_sub = oos_mfe[oos_mfe['gap_direction'] == gap_dir]

        for lo, hi, blabel in rvol_buckets:
            is_b = is_sub[(is_sub['rvol_30'] >= lo) & (is_sub['rvol_30'] < hi)]
            oos_b = oos_sub[(oos_sub['rvol_30'] >= lo) & (oos_sub['rvol_30'] < hi)]

            is_med = f"{is_b['mfe_live_r'].median():.2f}" if len(is_b) >= 10 else "N<10"
            is_surv = f"{(is_b['sl_hit']==False).mean():.1%}" if len(is_b) >= 10 else "N<10"
            oos_med = f"{oos_b['mfe_live_r'].median():.2f}" if len(oos_b) >= 10 else "N<10"
            oos_surv = f"{(oos_b['sl_hit']==False).mean():.1%}" if len(oos_b) >= 10 else "N<10"

            lines.append(f"  {blabel:<8} | {is_med:>8} | {is_surv:>8} | {len(is_b):>6} | {oos_med:>8} | {oos_surv:>8} | {len(oos_b):>6}")

    # ===== 5e: Trailing OOS (Top-3 strategies) =====
    lines.append(f"\n\n{'='*70}")
    lines.append("5e: TRAILING OOS (Top Strategien aus IS)")
    lines.append(f"{'='*70}")
    lines.append("\nIS Top: TRAIL_D (GapUp), Fix 2R (GapDn)\n")

    strategies = [
        ('Fix 2R', 'fix_2.0'),
        ('Fix 3R', 'fix_3.0'),
        ('TRAIL_A', 'trail_A'),
        ('TRAIL_D', 'trail_D'),
    ]

    for gap_dir, gap_label in [('up', 'GapUp'), ('down', 'GapDn')]:
        lines.append(f"  --- {gap_label} ---")
        lines.append(f"  {'Strategy':<15} | {'IS EV':>10} | {'IS WR':>8} | {'IS N':>6} | {'OOS EV':>10} | {'OOS WR':>8} | {'OOS N':>6}")
        lines.append(f"  {'-'*73}")

        for slabel, stype in strategies:
            is_gap = base_h1[base_h1['gap_direction'] == gap_dir]
            oos_gap = base_h2[base_h2['gap_direction'] == gap_dir]

            print(f"  {slabel} {gap_label}...", file=sys.stderr)
            is_trades = run_all(is_gap, stype)
            oos_trades = run_all(oos_gap, stype)

            is_ev = f"{is_trades['pnl_adr'].mean():>+10.3f}" if len(is_trades) >= 10 else f"{'N<10':>10}"
            is_wr = f"{is_trades['winner'].mean():>7.1%}" if len(is_trades) >= 10 else f"{'N<10':>8}"
            oos_ev = f"{oos_trades['pnl_adr'].mean():>+10.3f}" if len(oos_trades) >= 10 else f"{'N<10':>10}"
            oos_wr = f"{oos_trades['winner'].mean():>7.1%}" if len(oos_trades) >= 10 else f"{'N<10':>8}"

            lines.append(f"  {slabel:<15} | {is_ev} | {is_wr} | {len(is_trades):>6} | {oos_ev} | {oos_wr} | {len(oos_trades):>6}")

        lines.append("")

    # Setup B OOS
    lines.append(f"\n  === SETUP B (RVOL_30 > 5x) OOS ===\n")

    setup_b_h1 = base_h1[base_h1['rvol_open_30min'] >= 5]
    setup_b_h2 = base_h2[base_h2['rvol_open_30min'] >= 5]

    for gap_dir, gap_label in [('up', 'GapUp'), ('down', 'GapDn')]:
        lines.append(f"  --- {gap_label} ---")
        lines.append(f"  {'Strategy':<15} | {'IS EV':>10} | {'IS WR':>8} | {'IS N':>6} | {'OOS EV':>10} | {'OOS WR':>8} | {'OOS N':>6}")
        lines.append(f"  {'-'*73}")

        for slabel, stype in strategies:
            is_gap = setup_b_h1[setup_b_h1['gap_direction'] == gap_dir]
            oos_gap = setup_b_h2[setup_b_h2['gap_direction'] == gap_dir]

            is_trades = run_all(is_gap, stype)
            oos_trades = run_all(oos_gap, stype)

            is_ev = f"{is_trades['pnl_adr'].mean():>+10.3f}" if len(is_trades) >= 10 else f"{'N<10':>10}"
            is_wr = f"{is_trades['winner'].mean():>7.1%}" if len(is_trades) >= 10 else f"{'N<10':>8}"
            oos_ev = f"{oos_trades['pnl_adr'].mean():>+10.3f}" if len(oos_trades) >= 10 else f"{'N<10':>10}"
            oos_wr = f"{oos_trades['winner'].mean():>7.1%}" if len(oos_trades) >= 10 else f"{'N<10':>8}"

            lines.append(f"  {slabel:<15} | {is_ev} | {is_wr} | {len(is_trades):>6} | {oos_ev} | {oos_wr} | {len(oos_trades):>6}")

        lines.append("")

    # ===== 5f: Finale Antwort =====
    lines.append(f"\n{'='*70}")
    lines.append("5f: FINALE ANTWORT")
    lines.append(f"{'='*70}")

    # Compute final stats
    for gap_dir, gap_label in [('up', 'GapUp'), ('down', 'GapDn')]:
        is_sub = is_mfe[is_mfe['gap_direction'] == gap_dir]
        oos_sub = oos_mfe[oos_mfe['gap_direction'] == gap_dir]

        lines.append(f"\n  --- {gap_label} ---")
        lines.append(f"  1. MFE (alle Trades, MFE_LIVE):")
        lines.append(f"     IS:  Median = {is_sub['mfe_live_r'].median():.2f}R = {is_sub['mfe_live_adr'].median():.3f} ADR "
                     f"(+/- {is_sub['mfe_live_r'].std():.2f}R StdDev)")
        lines.append(f"     OOS: Median = {oos_sub['mfe_live_r'].median():.2f}R = {oos_sub['mfe_live_adr'].median():.3f} ADR "
                     f"(+/- {oos_sub['mfe_live_r'].std():.2f}R StdDev)")

        is_w = is_sub[is_sub['sl_hit'] == False]
        oos_w = oos_sub[oos_sub['sl_hit'] == False]
        lines.append(f"  2. MFE (nur Gewinner):")
        if len(is_w) >= 10:
            lines.append(f"     IS:  Median = {is_w['mfe_live_r'].median():.2f}R = {is_w['mfe_live_adr'].median():.3f} ADR")
        if len(oos_w) >= 10:
            lines.append(f"     OOS: Median = {oos_w['mfe_live_r'].median():.2f}R = {oos_w['mfe_live_adr'].median():.3f} ADR")

        lines.append(f"  3. Survival-Rate: IS={100*(is_sub['sl_hit']==False).mean():.1f}%, OOS={100*(oos_sub['sl_hit']==False).mean():.1f}%")

    output = "\n".join(lines)
    with open('results/d10_oos.txt', 'w', encoding='utf-8') as f:
        f.write(output)
    print(output)
    print(f"\nSaved to results/d10_oos.txt", file=sys.stderr)


if __name__ == '__main__':
    main()
