"""
D9 Aufgabe 6: OOS-Validierung mit R:R-Metriken
Teste IS-Findings auf H2 (2024-01-01 bis 2026-02-06).
Gleiche Simulation wie IS, nur andere Daten.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

RAW_DIR = Path('data/raw_1min')


def simulate_rr_trade(bars_rth, entry_price, sl_level, trade_dir, adr,
                      entry_time='09:36', timeout='15:55', rr_targets=[1, 2, 3]):
    """Bar-by-bar R:R trade simulation."""
    sl_dist = abs(entry_price - sl_level)
    if sl_dist <= 0 or pd.isna(sl_dist):
        return None

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

    post_entry = bars_rth[bars_rth['time_et'] >= entry_time]

    for _, bar in post_entry.iterrows():
        if bar['time_et'] > timeout:
            break

        if trade_dir == 'long':
            if bar['low'] <= sl_level:
                results['sl_hit'] = True
                results['exit_price'] = sl_level
                results['exit_type'] = 'sl'
                results['sl_time'] = bar['time_et']
                break
            for rr in sorted(rr_targets):
                if bar['high'] >= targets[rr] and not results[f'hit_{rr}R']:
                    results[f'hit_{rr}R'] = True
        else:
            if bar['high'] >= sl_level:
                results['sl_hit'] = True
                results['exit_price'] = sl_level
                results['exit_type'] = 'sl'
                results['sl_time'] = bar['time_et']
                break
            for rr in sorted(rr_targets):
                if bar['low'] <= targets[rr] and not results[f'hit_{rr}R']:
                    results[f'hit_{rr}R'] = True

    if not results['sl_hit']:
        timeout_bars = post_entry[post_entry['time_et'] <= timeout]
        if len(timeout_bars) > 0:
            results['exit_price'] = timeout_bars.iloc[-1]['close']

    if pd.notna(results['exit_price']) and adr > 0:
        if trade_dir == 'long':
            results['pnl_adr'] = (results['exit_price'] - entry_price) / adr
        else:
            results['pnl_adr'] = (entry_price - results['exit_price']) / adr

    if pd.notna(results['exit_price']) and sl_dist > 0:
        if trade_dir == 'long':
            results['pnl_r'] = (results['exit_price'] - entry_price) / sl_dist
        else:
            results['pnl_r'] = (entry_price - results['exit_price']) / sl_dist

    return results


def run_simulation(meta_subset, sl_type='sl_half'):
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
        return f"  {label:<45} | N={n:>4} [N<10]"

    med_sl = trades['sl_dist_adr'].median()
    wr1 = trades['hit_1R'].mean()
    wr2 = trades['hit_2R'].mean()
    wr3 = trades['hit_3R'].mean()
    sl_pct = trades['sl_hit'].mean()

    ev_2r = (wr2 * 2 - (1 - wr2) * 1) * med_sl if pd.notna(med_sl) else np.nan

    low_n = " [LOW N]" if n < 20 else ""
    return (f"  {label:<45} | N={n:>4} | SL={med_sl:.3f} | "
            f"WR@1R={wr1:>5.1%} | WR@2R={wr2:>5.1%} | WR@3R={wr3:>5.1%} | "
            f"EV@2R={ev_2r:>+.3f}{low_n}")


def ev_curve(trades, label=""):
    """EV curve for multiple R:R targets."""
    rr_range = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    n = len(trades)
    if n < 10:
        return f"    {label:<25}: N={n} [N<10]"

    med_sl = trades['sl_dist_adr'].median()
    parts = []
    best_rr = None
    best_ev = -999

    for rr in rr_range:
        col = f'hit_{rr}R' if rr == int(rr) else None
        if col and col in trades.columns:
            wr = trades[col].mean()
        elif 'pnl_r' in trades.columns:
            wr = (trades['pnl_r'] >= rr).mean()
        else:
            continue

        ev = (wr * rr - (1 - wr)) * med_sl
        rr_label = f"{rr:.1f}R" if rr != int(rr) else f"{int(rr)}.0R"
        parts.append(f"{rr_label}={wr:.0%}/{ev:+.3f}")
        if ev > best_ev:
            best_ev = ev
            best_rr = rr

    line = f"    {label:<25}: " + " ".join(parts)
    if best_rr is not None:
        rr_label = f"{best_rr:.1f}R" if best_rr != int(best_rr) else f"{int(best_rr)}.0R"
        line += f"\n{'':>29}-> Optimal: {rr_label}, EV={best_ev:+.3f} ADR (N={n})"
    return line


def main():
    lines = []
    lines.append("=" * 70)
    lines.append("D9 AUFGABE 6: OOS-VALIDIERUNG MIT R:R-METRIKEN")
    lines.append("=" * 70)
    lines.append("\nH2 (OOS): 2024-01-01 bis 2026-02-06")
    lines.append("Entry: 9:35 | Timeout: 15:55 | Break-Even: 1R=50%, 2R=33.3%, 3R=25%")

    meta = pd.read_parquet('data/metadata/metadata_v9.parquet')

    # H1 (IS) for reference
    h1 = meta[(meta['date'] >= '2021-02-21') & (meta['date'] <= '2023-12-31')].copy()
    h1 = h1[h1['od_body_pct'].notna() & h1['sl_full_adr'].notna()].copy()

    # H2 (OOS)
    h2 = meta[(meta['date'] >= '2024-01-01') & (meta['date'] <= '2026-02-06')].copy()
    h2 = h2[h2['od_body_pct'].notna() & h2['sl_full_adr'].notna()].copy()
    print(f"H1: {len(h1)}, H2: {len(h2)}", file=sys.stderr)

    # =====================================================================
    # 6a: BASIS-SETUP OOS (OD > 0.5 with_gap)
    # =====================================================================
    lines.append(f"\n\n{'='*70}")
    lines.append("6a: BASIS-SETUP OOS (OD > 0.5 with_gap)")
    lines.append(f"{'='*70}")

    for gap_dir, gap_label in [('up', 'GapUp'), ('down', 'GapDown')]:
        lines.append(f"\n--- {gap_label} ---")

        for half_name, half_df in [('IS', h1), ('OOS', h2)]:
            sub = half_df[(half_df['od_strength'] > 0.5) & (half_df['od_direction'] == 'with_gap') & (half_df['gap_direction'] == gap_dir)]
            print(f"  {gap_label} {half_name}: N={len(sub)}", file=sys.stderr)

            for sl_type, sl_label in [('sl_half', 'SL_HALF'), ('sl_fix025', 'SL_FIX025')]:
                trades = run_simulation(sub, sl_type)
                lines.append(summarize_rr(trades, f"{half_name} {sl_label}"))

    # =====================================================================
    # 6b: Q_HIGH FILTER OOS
    # =====================================================================
    lines.append(f"\n\n{'='*70}")
    lines.append("6b: Q_HIGH FILTER OOS")
    lines.append(f"{'='*70}")
    lines.append("\nOD > 0.5 + with_gap + QUALITY_HIGH\n")

    for gap_dir, gap_label in [('up', 'GapUp'), ('down', 'GapDown')]:
        lines.append(f"--- {gap_label} ---")

        for half_name, half_df in [('IS', h1), ('OOS', h2)]:
            sub = half_df[(half_df['od_strength'] > 0.5) & (half_df['od_direction'] == 'with_gap') &
                          (half_df['gap_direction'] == gap_dir) & (half_df['quality_high'] == True)]

            trades = run_simulation(sub, 'sl_half')
            lines.append(summarize_rr(trades, f"{half_name} QH SL_HALF"))

            trades_fix = run_simulation(sub, 'sl_fix025')
            lines.append(summarize_rr(trades_fix, f"{half_name} QH SL_FIX025"))

        lines.append("")

    # =====================================================================
    # 6c: KONTEXT-FILTER OOS (SL_HALF)
    # =====================================================================
    lines.append(f"\n{'='*70}")
    lines.append("6c: KONTEXT-FILTER OOS (SL_HALF)")
    lines.append(f"{'='*70}")
    lines.append("\nBasis: QH + OD > 0.5 + with_gap\n")

    filter_configs = [
        ('RVOL_30 > 5', lambda df: df[df['rvol_open_30min'] >= 5]),
        ('RVOL_30 > 10', lambda df: df[df['rvol_open_30min'] >= 10]),
        ('PM/RTH30 < 10%', lambda df: df[df['pm_rth30_computed'] < 0.10]),
        ('Gap < 1 ADR', lambda df: df[df['gap_size_in_adr'] < 1]),
        ('Gap 1-2 ADR', lambda df: df[(df['gap_size_in_adr'] >= 1) & (df['gap_size_in_adr'] < 2)]),
        ('RV30>5 + PM30<10', lambda df: df[(df['rvol_open_30min'] >= 5) & (df['pm_rth30_computed'] < 0.10)]),
        ('RV30>5 + PM30<30', lambda df: df[(df['rvol_open_30min'] >= 5) & (df['pm_rth30_computed'] < 0.30)]),
    ]

    for gap_dir, gap_label in [('up', 'GapUp'), ('down', 'GapDown')]:
        lines.append(f"--- {gap_label} ---")

        for half_name, half_df in [('IS', h1), ('OOS', h2)]:
            qh_base = half_df[(half_df['od_strength'] > 0.5) & (half_df['od_direction'] == 'with_gap') &
                              (half_df['gap_direction'] == gap_dir) & (half_df['quality_high'] == True)]

            # Baseline
            trades = run_simulation(qh_base, 'sl_half')
            lines.append(summarize_rr(trades, f"{half_name} Baseline"))

            for flabel, ffn in filter_configs:
                sub = ffn(qh_base)
                trades = run_simulation(sub, 'sl_half')
                lines.append(summarize_rr(trades, f"{half_name} +{flabel}"))

            lines.append("")

    # =====================================================================
    # 6d: KUMULATIVE FILTER OOS (SL_HALF)
    # =====================================================================
    lines.append(f"\n{'='*70}")
    lines.append("6d: KUMULATIVE FILTER OOS (SL_HALF, SL_FIX025)")
    lines.append(f"{'='*70}")
    lines.append("\nBeste IS-Layer auf OOS testen\n")

    layer_configs = [
        ('L0: OD>0.5 with', lambda df: df[(df['od_strength'] > 0.5) & (df['od_direction'] == 'with_gap')]),
        ('L1: + QH', lambda df: df[(df['od_strength'] > 0.5) & (df['od_direction'] == 'with_gap') & (df['quality_high'] == True)]),
        ('L2d: +RV30>5', lambda df: df[(df['od_strength'] > 0.5) & (df['od_direction'] == 'with_gap') & (df['quality_high'] == True) & (df['rvol_open_30min'] >= 5)]),
        ('L2e: +PM30<10', lambda df: df[(df['od_strength'] > 0.5) & (df['od_direction'] == 'with_gap') & (df['quality_high'] == True) & (df['pm_rth30_computed'] < 0.10)]),
        ('L3b: +RV30>5+PM30<10', lambda df: df[(df['od_strength'] > 0.5) & (df['od_direction'] == 'with_gap') & (df['quality_high'] == True) & (df['rvol_open_30min'] >= 5) & (df['pm_rth30_computed'] < 0.10)]),
    ]

    for gap_dir, gap_label in [('up', 'GapUp'), ('down', 'GapDown')]:
        lines.append(f"--- {gap_label} ---\n")

        for sl_type, sl_label in [('sl_half', 'SL_HALF'), ('sl_fix025', 'SL_FIX025')]:
            lines.append(f"  {sl_label}:")

            for half_name, half_df in [('IS', h1), ('OOS', h2)]:
                for llabel, lfn in layer_configs:
                    sub = lfn(half_df[half_df['gap_direction'] == gap_dir])
                    trades = run_simulation(sub, sl_type)
                    lines.append(summarize_rr(trades, f"{half_name} {llabel}"))
                lines.append("")
            lines.append("")

    # =====================================================================
    # 6e: AGAINST-GAP OOS
    # =====================================================================
    lines.append(f"\n{'='*70}")
    lines.append("6e: AGAINST-GAP OOS (SL_HALF)")
    lines.append(f"{'='*70}")

    for gap_dir, gap_label in [('up', 'GapUp'), ('down', 'GapDown')]:
        lines.append(f"\n--- {gap_label} ---")

        for half_name, half_df in [('IS', h1), ('OOS', h2)]:
            sub = half_df[(half_df['od_strength'] > 0.5) & (half_df['od_direction'] == 'against_gap') & (half_df['gap_direction'] == gap_dir)]
            trades = run_simulation(sub, 'sl_half')
            lines.append(summarize_rr(trades, f"{half_name} OD>0.5 against"))

            sub_qh = sub[sub['quality_high'] == True]
            trades_qh = run_simulation(sub_qh, 'sl_half')
            lines.append(summarize_rr(trades_qh, f"{half_name} +QH"))

    # =====================================================================
    # 6f: EV-KURVEN IS vs OOS
    # =====================================================================
    lines.append(f"\n\n{'='*70}")
    lines.append("6f: EV-KURVEN IS vs OOS (Beste Setups, SL_HALF)")
    lines.append(f"{'='*70}")

    for gap_dir, gap_label in [('up', 'GapUp'), ('down', 'GapDown')]:
        lines.append(f"\n--- {gap_label} ---")

        for half_name, half_df in [('IS', h1), ('OOS', h2)]:
            # L1 baseline
            qh_with = half_df[(half_df['od_strength'] > 0.5) & (half_df['od_direction'] == 'with_gap') &
                              (half_df['gap_direction'] == gap_dir) & (half_df['quality_high'] == True)]
            trades = run_simulation(qh_with, 'sl_half')
            lines.append(ev_curve(trades, f"{half_name} L1 QH"))

            # L2d
            sub = qh_with[qh_with['rvol_open_30min'] >= 5]
            trades = run_simulation(sub, 'sl_half')
            lines.append(ev_curve(trades, f"{half_name} L2d RV30>5"))

            # L2e
            sub = qh_with[qh_with['pm_rth30_computed'] < 0.10]
            trades = run_simulation(sub, 'sl_half')
            lines.append(ev_curve(trades, f"{half_name} L2e PM30<10"))

        lines.append("")

    # =====================================================================
    # 6g: ZUSAMMENFASSUNG IS vs OOS
    # =====================================================================
    lines.append(f"\n{'='*70}")
    lines.append("6g: ZUSAMMENFASSUNG IS vs OOS")
    lines.append(f"{'='*70}")
    lines.append("""
  Shrinkage-Tabelle: WR@2R IS -> OOS

  Setup                      | IS WR@2R | OOS WR@2R | Shrinkage | Repliziert?
  ---------------------------------------------------------------------------""")

    # Compute shrinkage for key setups
    shrinkage_setups = [
        ('L0: OD>0.5 with', lambda df: df[(df['od_strength'] > 0.5) & (df['od_direction'] == 'with_gap')]),
        ('L1: + QH', lambda df: df[(df['od_strength'] > 0.5) & (df['od_direction'] == 'with_gap') & (df['quality_high'] == True)]),
        ('L2d: +RV30>5', lambda df: df[(df['od_strength'] > 0.5) & (df['od_direction'] == 'with_gap') & (df['quality_high'] == True) & (df['rvol_open_30min'] >= 5)]),
        ('L2e: +PM30<10', lambda df: df[(df['od_strength'] > 0.5) & (df['od_direction'] == 'with_gap') & (df['quality_high'] == True) & (df['pm_rth30_computed'] < 0.10)]),
    ]

    for gap_dir, gap_label in [('up', 'GapUp'), ('down', 'GapDown')]:
        lines.append(f"\n  {gap_label}:")
        for llabel, lfn in shrinkage_setups:
            is_sub = lfn(h1[h1['gap_direction'] == gap_dir])
            oos_sub = lfn(h2[h2['gap_direction'] == gap_dir])

            is_trades = run_simulation(is_sub, 'sl_half')
            oos_trades = run_simulation(oos_sub, 'sl_half')

            if len(is_trades) >= 10 and len(oos_trades) >= 10:
                is_wr2 = is_trades['hit_2R'].mean()
                oos_wr2 = oos_trades['hit_2R'].mean()
                shrink = (is_wr2 - oos_wr2) / is_wr2 * 100 if is_wr2 > 0 else 0
                repl = "JA" if oos_wr2 > 0.333 else "NEIN"
                lines.append(f"  {llabel:<26} | {is_wr2:>7.1%} | {oos_wr2:>8.1%} | {shrink:>8.1f}% | {repl}")
            elif len(oos_trades) < 10:
                is_wr2 = is_trades['hit_2R'].mean() if len(is_trades) >= 10 else np.nan
                lines.append(f"  {llabel:<26} | {is_wr2:>7.1%} |  N<10    |    ---   | ---")
            else:
                lines.append(f"  {llabel:<26} |  N<10   |  N<10    |    ---   | ---")

    output = "\n".join(lines)
    with open('results/d9_oos.txt', 'w', encoding='utf-8') as f:
        f.write(output)
    print(output)
    print(f"\nSaved to results/d9_oos.txt", file=sys.stderr)


if __name__ == '__main__':
    main()
