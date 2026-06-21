"""
D8.5 Aufgabe 6: OOS-Validierung (H2: 2024-01-01 bis 2026-02-06)
"""
import pandas as pd
import numpy as np
from pathlib import Path


def wr_at(drift, tgt):
    v = drift.dropna()
    return (v >= tgt).mean() if len(v) > 0 else np.nan


def bootstrap_ci(data, func, n_boot=1000, ci=0.95):
    """Bootstrap confidence interval."""
    vals = []
    data = data.dropna().values
    n = len(data)
    if n < 5:
        return np.nan, np.nan, np.nan
    rng = np.random.RandomState(42)
    for _ in range(n_boot):
        sample = rng.choice(data, n, replace=True)
        vals.append(func(sample))
    vals = sorted(vals)
    lo = vals[int((1-ci)/2 * n_boot)]
    hi = vals[int((1+ci)/2 * n_boot)]
    return np.mean(vals), lo, hi


def simulate_trade(bars_rth, entry_price, adr, direction, sl_adr=0.25):
    sl_dist = sl_adr * adr
    targets = [0.25, 0.50, 0.75, 1.00]

    if direction == 'long':
        sl_price = entry_price - sl_dist
        target_prices = [entry_price + t * adr for t in targets]
    else:
        sl_price = entry_price + sl_dist
        target_prices = [entry_price - t * adr for t in targets]

    results = {f'hit_{t}': False for t in targets}
    results['sl_hit'] = False
    results['exit_price'] = np.nan

    post_entry = bars_rth[bars_rth['time_et'] >= '09:35']

    for _, bar in post_entry.iterrows():
        if direction == 'long':
            if bar['low'] <= sl_price:
                results['sl_hit'] = True
                results['exit_price'] = sl_price
                break
            for t, tp in zip(targets, target_prices):
                if bar['high'] >= tp:
                    results[f'hit_{t}'] = True
        else:
            if bar['high'] >= sl_price:
                results['sl_hit'] = True
                results['exit_price'] = sl_price
                break
            for t, tp in zip(targets, target_prices):
                if bar['low'] <= tp:
                    results[f'hit_{t}'] = True

    if not results['sl_hit']:
        timeout_bars = post_entry[post_entry['time_et'] <= '15:55']
        if len(timeout_bars) > 0:
            results['exit_price'] = timeout_bars.iloc[-1]['close']

    if pd.notna(results['exit_price']) and adr > 0:
        if direction == 'long':
            results['pnl_adr'] = (results['exit_price'] - entry_price) / adr
        else:
            results['pnl_adr'] = (entry_price - results['exit_price']) / adr

    return results


def run_sim_batch(meta_sub, entry_time='09:35'):
    raw_dir = Path('data/raw_1min')
    results = []

    for _, row in meta_sub.iterrows():
        ticker = row['ticker']
        date = str(row['date'])
        fpath = raw_dir / ticker / f"{date}.parquet"
        if not fpath.exists():
            continue

        bars = pd.read_parquet(fpath)
        rth = bars[bars['session'] == 'rth']
        if len(rth) < 10:
            continue

        adr = row.get('adr_10', row.get('adr_5', np.nan))
        entry_price = row['close_935']
        if pd.isna(entry_price) or pd.isna(adr) or adr <= 0:
            continue

        od_dir = row['od_direction']
        gap_dir = row['gap_direction']
        if od_dir == 'with_gap':
            direction = 'long' if gap_dir == 'up' else 'short'
        else:
            direction = 'short' if gap_dir == 'up' else 'long'

        trade = simulate_trade(rth, entry_price, adr, direction)
        trade['ticker'] = ticker
        trade['date'] = date
        trade['gap_direction'] = gap_dir
        trade['od_direction'] = od_dir
        results.append(trade)

    return pd.DataFrame(results)


def summarize(trades, label):
    n = len(trades)
    if n < 10:
        return f"  {label:<50} | N={n:>4} [N<10]"
    wr25 = trades['hit_0.25'].mean()
    wr50 = trades['hit_0.5'].mean()
    sl = trades['sl_hit'].mean()
    ev = wr50 * 0.50 - sl * 0.25
    low = " [LOW N]" if n < 20 else ""
    return f"  {label:<50} | N={n:>4} | WR@0.25={wr25:>5.1%} | WR@0.50={wr50:>5.1%} | EV@0.50={ev:>+.3f} | SL%={sl:>5.1%}{low}"


def main():
    lines = []
    lines.append("=" * 70)
    lines.append("D8.5 AUFGABE 6: OOS-VALIDIERUNG (H2: 2024-2026)")
    lines.append("=" * 70)

    meta = pd.read_parquet('data/metadata/metadata_v8_5.parquet')

    h1 = meta[(meta['date'] >= '2021-02-21') & (meta['date'] <= '2023-12-31')].copy()
    h2 = meta[(meta['date'] >= '2024-01-01') & (meta['date'] <= '2026-02-06')].copy()

    h1 = h1[h1['od_body_pct'].notna()].copy()
    h2 = h2[h2['od_body_pct'].notna()].copy()

    h1['gap_abs'] = h1['gap_size_in_adr'].abs()
    h2['gap_abs'] = h2['gap_size_in_adr'].abs()

    lines.append(f"\nH1 (IS): {len(h1)} Gapper")
    lines.append(f"H2 (OOS): {len(h2)} Gapper")

    # ===== 6a: Feature-Verteilung OOS =====
    lines.append(f"\n\n{'='*70}")
    lines.append("6a: FEATURE-VERTEILUNG OOS")
    lines.append(f"{'='*70}")

    for half_name, half_df in [('H1 (IS)', h1), ('H2 (OOS)', h2)]:
        lines.append(f"\n  {half_name}:")
        for col, label in [('od_body_pct', 'Body%'), ('od_wick_ratio', 'Wick Ratio')]:
            s = half_df[col].dropna()
            lines.append(f"    {label}: Median={s.median():.3f}, Q25={s.quantile(0.25):.3f}, Q75={s.quantile(0.75):.3f}")

        qh = half_df[half_df['quality_high'] == True]
        lines.append(f"    QUALITY_HIGH: {len(qh)} ({100*len(qh)/len(half_df):.1f}%)")
        qh_od = qh[qh['od_strength'] > 0.5]
        lines.append(f"    QH + OD>0.5: {len(qh_od)} ({100*len(qh_od)/len(half_df):.1f}%)")

    # ===== 6b: QUALITY_HIGH vs QUALITY_LOW (OOS) =====
    lines.append(f"\n\n{'='*70}")
    lines.append("6b: QUALITY_HIGH vs QUALITY_LOW bei OD > 0.5 (OOS)")
    lines.append(f"{'='*70}")

    directions = [
        ('OD with GapUp', 'with_gap', 'up'),
        ('OD with GapDown', 'with_gap', 'down'),
        ('OD against GapUp', 'against_gap', 'up'),
        ('OD against GapDown', 'against_gap', 'down'),
    ]

    for half_name, half_df in [('IS', h1), ('OOS', h2)]:
        lines.append(f"\n  === {half_name} ===")
        od05 = half_df[half_df['od_strength'] > 0.5]
        qh_od = od05[od05['quality_high'] == True]
        ql_od = od05[od05['quality_high'] == False]

        lines.append(f"  {'Direction':<25} | {'Q_HIGH WR@0.50':>14} | {'Q_LOW WR@0.50':>14} | {'Delta':>7} | {'N_H':>4} | {'N_L':>4}")
        lines.append(f"  {'-'*85}")

        for dir_label, od_dir, gap_dir in directions:
            sub_h = qh_od[(qh_od['od_direction'] == od_dir) & (qh_od['gap_direction'] == gap_dir)]
            sub_l = ql_od[(ql_od['od_direction'] == od_dir) & (ql_od['gap_direction'] == gap_dir)]

            wr_h = wr_at(sub_h['rest_drift'], 0.50) if len(sub_h) >= 10 else np.nan
            wr_l = wr_at(sub_l['rest_drift'], 0.50) if len(sub_l) >= 10 else np.nan

            if pd.notna(wr_h) and pd.notna(wr_l):
                delta = wr_h - wr_l
                lines.append(f"  {dir_label:<25} | {wr_h:>14.1%} | {wr_l:>14.1%} | {delta:>+7.1%} | {len(sub_h):>4} | {len(sub_l):>4}")
            elif pd.notna(wr_h):
                lines.append(f"  {dir_label:<25} | {wr_h:>14.1%} | {'N<10':>14} | {'N/A':>7} | {len(sub_h):>4} | {len(sub_l):>4}")
            else:
                lines.append(f"  {dir_label:<25} | {'N<10':>14} | {'N<10':>14} | {'N/A':>7} | {len(sub_h):>4} | {len(sub_l):>4}")

    # ===== 6c: Top-5 Kreuzungen aus IS auf OOS =====
    lines.append(f"\n\n{'='*70}")
    lines.append("6c: TOP-5 IS-KREUZUNGEN AUF OOS REPRODUZIERT")
    lines.append(f"{'='*70}")

    # Top-5 from IS (Aufgabe 3): the cells with highest rest_drift or WR
    # Based on IS results, the top cells were:
    top_cells = [
        ('QH+OD>0.5 with GapUp + RVOL_30 5-10x', 'with_gap', 'up', 'rvol_open_30min', 5, 10),
        ('QH+OD>0.5 with GapUp + PM/RTH30<10%', 'with_gap', 'up', 'pm_rth30_computed', 0, 0.10),
        ('QH+OD>0.5 with GapDn + RVOL_30>10x', 'with_gap', 'down', 'rvol_open_30min', 10, 999),
        ('QH+OD>0.5 with GapDn + PM/RTH30<10%', 'with_gap', 'down', 'pm_rth30_computed', 0, 0.10),
        ('QH+OD>0.5 with GapDn + PM/RTH5 50-100%', 'with_gap', 'down', 'pm_rth5', 0.50, 1.00),
    ]

    for cell_label, od_dir, gap_dir, col, lo, hi in top_cells:
        lines.append(f"\n  {cell_label}:")

        for half_name, half_df in [('IS', h1), ('OOS', h2)]:
            sub = half_df[(half_df['quality_high'] == True) &
                          (half_df['od_strength'] > 0.5) &
                          (half_df['od_direction'] == od_dir) &
                          (half_df['gap_direction'] == gap_dir)]
            if hi == 999:
                sub = sub[sub[col] >= lo]
            else:
                sub = sub[(sub[col] >= lo) & (sub[col] < hi)]

            n = len(sub)
            if n < 5:
                lines.append(f"    {half_name}: N={n} [zu wenig]")
                continue

            fd = sub['full_drift'].mean()
            rd = sub['rest_drift'].mean()
            wr50 = wr_at(sub['rest_drift'], 0.50)
            lines.append(f"    {half_name}: N={n}, fd={fd:+.3f}, rd={rd:+.3f}, WR@0.50={wr50:.1%}")

    # ===== 6d: Trade-Sim Layers OOS =====
    lines.append(f"\n\n{'='*70}")
    lines.append("6d: TRADE-SIM LAYERS (OOS)")
    lines.append(f"{'='*70}")
    lines.append("\nEntry: 9:35 | SL: 0.25 ADR | Timeout: 15:55")

    # WITH GAP
    lines.append(f"\n--- OD > 0.5 WITH GAP ---")

    for gap_dir, gap_label in [('up', 'GapUp'), ('down', 'GapDown')]:
        lines.append(f"\n  {gap_label}:")
        base = h2[h2['gap_direction'] == gap_dir]

        layers = [
            ('L0: OD>0.5 with', base[(base['od_strength'] > 0.5) & (base['od_direction'] == 'with_gap')]),
            ('L1: + QUALITY_HIGH', base[(base['od_strength'] > 0.5) & (base['od_direction'] == 'with_gap') & (base['quality_high'] == True)]),
            ('L2c: + PM/RTH30<10%', base[(base['od_strength'] > 0.5) & (base['od_direction'] == 'with_gap') & (base['quality_high'] == True) & (base['pm_rth30_computed'] < 0.10)]),
            ('L2a: + RVOL_30>5x', base[(base['od_strength'] > 0.5) & (base['od_direction'] == 'with_gap') & (base['quality_high'] == True) & (base['rvol_open_30min'] > 5)]),
            ('L2b: + Gap>2ADR', base[(base['od_strength'] > 0.5) & (base['od_direction'] == 'with_gap') & (base['quality_high'] == True) & (base['gap_abs'] > 2)]),
            ('L2d: + RVOL_5>5x', base[(base['od_strength'] > 0.5) & (base['od_direction'] == 'with_gap') & (base['quality_high'] == True) & (base['rvol_5'] > 5)]),
            ('L2e: + PM/RTH5<50%', base[(base['od_strength'] > 0.5) & (base['od_direction'] == 'with_gap') & (base['quality_high'] == True) & (base['pm_rth5'] < 0.50)]),
        ]

        for layer_label, subset in layers:
            print(f"  OOS {gap_label} {layer_label}: N={len(subset)}")
            if len(subset) < 10:
                lines.append(f"  {layer_label:<50} | N={len(subset):>4} [N<10]")
                continue
            trades = run_sim_batch(subset)
            lines.append(summarize(trades, layer_label))

    # AGAINST GAP
    lines.append(f"\n--- OD > 0.5 AGAINST GAP ---")
    for gap_dir, gap_label in [('up', 'GapUp'), ('down', 'GapDown')]:
        lines.append(f"\n  {gap_label}:")
        base = h2[h2['gap_direction'] == gap_dir]

        layers = [
            ('L0: OD>0.5 against', base[(base['od_strength'] > 0.5) & (base['od_direction'] == 'against_gap')]),
            ('L1: + QUALITY_HIGH', base[(base['od_strength'] > 0.5) & (base['od_direction'] == 'against_gap') & (base['quality_high'] == True)]),
        ]

        for layer_label, subset in layers:
            print(f"  OOS {gap_label} {layer_label}: N={len(subset)}")
            if len(subset) < 10:
                lines.append(f"  {layer_label:<50} | N={len(subset):>4} [N<10]")
                continue
            trades = run_sim_batch(subset)
            lines.append(summarize(trades, layer_label))

    # ===== 6e: Bootstrap CIs for top setups =====
    lines.append(f"\n\n{'='*70}")
    lines.append("6e: BOOTSTRAP CIs FUER TOP-SETUPS (OOS)")
    lines.append(f"{'='*70}")

    # Top setups: QH + OD>0.5 with + PM/RTH30<10%
    for gap_dir, gap_label in [('up', 'GapUp'), ('down', 'GapDown')]:
        base = h2[(h2['gap_direction'] == gap_dir) &
                   (h2['od_strength'] > 0.5) &
                   (h2['od_direction'] == 'with_gap') &
                   (h2['quality_high'] == True)]

        lines.append(f"\n  {gap_label} QH + OD>0.5 with (N={len(base)}):")
        if len(base) >= 20:
            rd_mean, rd_lo, rd_hi = bootstrap_ci(base['rest_drift'], np.mean)
            wr50 = wr_at(base['rest_drift'], 0.50)
            wr50_mean, wr50_lo, wr50_hi = bootstrap_ci(
                (base['rest_drift'] >= 0.50).astype(float), np.mean)
            lines.append(f"    rest_drift: {rd_mean:+.3f} [{rd_lo:+.3f}, {rd_hi:+.3f}]")
            lines.append(f"    WR@0.50: {wr50_mean:.1%} [{wr50_lo:.1%}, {wr50_hi:.1%}]")

        # With PM/RTH30<10%
        sub_pm = base[base['pm_rth30_computed'] < 0.10]
        lines.append(f"\n  {gap_label} QH + OD>0.5 with + PM/RTH30<10% (N={len(sub_pm)}):")
        if len(sub_pm) >= 10:
            rd_mean, rd_lo, rd_hi = bootstrap_ci(sub_pm['rest_drift'], np.mean)
            wr50_mean, wr50_lo, wr50_hi = bootstrap_ci(
                (sub_pm['rest_drift'] >= 0.50).astype(float), np.mean)
            lines.append(f"    rest_drift: {rd_mean:+.3f} [{rd_lo:+.3f}, {rd_hi:+.3f}]")
            lines.append(f"    WR@0.50: {wr50_mean:.1%} [{wr50_lo:.1%}, {wr50_hi:.1%}]")

    # ===== 6e finale Antworten =====
    lines.append(f"\n\n{'='*70}")
    lines.append("6e: FINALE ANTWORTEN")
    lines.append(f"{'='*70}")

    # 1. Reversal-Filter OOS
    lines.append("\n  1. Bringt der Reversal-Filter OOS EXTRA-Value auf OD > 0.5?")

    for half_name, half_df in [('IS', h1), ('OOS', h2)]:
        od05 = half_df[half_df['od_strength'] > 0.5]
        qh_with = od05[(od05['quality_high'] == True) & (od05['od_direction'] == 'with_gap')]
        ql_with = od05[(od05['quality_high'] == False) & (od05['od_direction'] == 'with_gap')]

        wr_h = wr_at(qh_with['rest_drift'], 0.50) if len(qh_with) >= 10 else np.nan
        wr_l = wr_at(ql_with['rest_drift'], 0.50) if len(ql_with) >= 10 else np.nan

        if pd.notna(wr_h) and pd.notna(wr_l):
            lines.append(f"    {half_name}: Q_HIGH WR@0.50={wr_h:.1%} (N={len(qh_with)}), Q_LOW WR@0.50={wr_l:.1%} (N={len(ql_with)}), Delta={wr_h-wr_l:+.1%}")
        elif pd.notna(wr_h):
            lines.append(f"    {half_name}: Q_HIGH WR@0.50={wr_h:.1%} (N={len(qh_with)}), Q_LOW N={len(ql_with)} [zu wenig]")

    # 2. Kontext-Parameter OOS
    lines.append("\n  2. Bringt RVOL/Gap/PM EXTRA-Value auf OD>0.5 + QH (OOS)?")
    qh_od_oos = h2[(h2['quality_high'] == True) & (h2['od_strength'] > 0.5) & (h2['od_direction'] == 'with_gap')]

    for col, label, threshold, direction in [
        ('rvol_open_30min', 'RVOL_30>5x', 5, '>'),
        ('gap_abs', 'Gap>2ADR', 2, '>'),
        ('pm_rth30_computed', 'PM/RTH30<10%', 0.10, '<'),
        ('rvol_5', 'RVOL_5>5x', 5, '>'),
        ('pm_rth5', 'PM/RTH5<50%', 0.50, '<'),
    ]:
        if direction == '>':
            sub_yes = qh_od_oos[qh_od_oos[col] > threshold]
            sub_no = qh_od_oos[qh_od_oos[col] <= threshold]
        else:
            sub_yes = qh_od_oos[qh_od_oos[col] < threshold]
            sub_no = qh_od_oos[qh_od_oos[col] >= threshold]

        wr_yes = wr_at(sub_yes['rest_drift'], 0.50) if len(sub_yes) >= 10 else np.nan
        wr_no = wr_at(sub_no['rest_drift'], 0.50) if len(sub_no) >= 10 else np.nan

        if pd.notna(wr_yes) and pd.notna(wr_no):
            lines.append(f"    {label}: YES WR@0.50={wr_yes:.1%} (N={len(sub_yes)}), NO WR@0.50={wr_no:.1%} (N={len(sub_no)}), Delta={wr_yes-wr_no:+.1%}")
        else:
            lines.append(f"    {label}: YES N={len(sub_yes)}, NO N={len(sub_no)} [sample zu klein fuer Vergleich]")

    # 3. Beste Kombi OOS
    lines.append("\n  3. Was ist die BESTE Filter-Kombination mit positivem EV OOS?")
    lines.append("     (Siehe Trade-Sim Layers oben)")

    # 4. 9:35 vs 10:00
    lines.append("\n  4. 9:35 oder 10:00 -- was ist besser NACH Reversal-Filter?")
    lines.append("     (Siehe Aufgabe 5 Ergebnisse -- 9:35 war klar besser IS)")
    lines.append("     OOS-Vergleich folgt in Synthese.")

    output = "\n".join(lines)
    with open('results/d8_5_oos.txt', 'w', encoding='utf-8') as f:
        f.write(output)
    print(output)
    print(f"\nSaved to results/d8_5_oos.txt")


if __name__ == '__main__':
    main()
