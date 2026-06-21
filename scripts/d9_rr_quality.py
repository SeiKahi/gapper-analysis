"""
D9 Aufgabe 3: Q_HIGH vs Q_LOW R:R comparison (IS)
+ Aufgabe 4: Context parameters R:R analysis + cumulative layers
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

RAW_DIR = Path('data/raw_1min')


def simulate_rr_trade(bars_rth, entry_price, sl_level, trade_dir, adr):
    sl_dist = abs(entry_price - sl_level)
    if sl_dist <= 0 or pd.isna(sl_dist):
        return None

    rr_targets = [1, 2, 3]
    targets = {}
    for rr in rr_targets:
        if trade_dir == 'long':
            targets[rr] = entry_price + rr * sl_dist
        else:
            targets[rr] = entry_price - rr * sl_dist

    results = {f'hit_{rr}R': False for rr in rr_targets}
    results['sl_hit'] = False
    results['sl_dist_adr'] = sl_dist / adr if adr > 0 else np.nan
    results['exit_price'] = np.nan
    results['exit_type'] = 'timeout'

    post_entry = bars_rth[bars_rth['time_et'] >= '09:36']
    for _, bar in post_entry.iterrows():
        if bar['time_et'] > '15:55':
            break
        if trade_dir == 'long':
            if bar['low'] <= sl_level:
                results['sl_hit'] = True
                results['exit_price'] = sl_level
                results['exit_type'] = 'sl'
                break
            for rr in rr_targets:
                if bar['high'] >= targets[rr]:
                    results[f'hit_{rr}R'] = True
        else:
            if bar['high'] >= sl_level:
                results['sl_hit'] = True
                results['exit_price'] = sl_level
                results['exit_type'] = 'sl'
                break
            for rr in rr_targets:
                if bar['low'] <= targets[rr]:
                    results[f'hit_{rr}R'] = True

    if not results['sl_hit']:
        timeout = post_entry[post_entry['time_et'] <= '15:55']
        if len(timeout) > 0:
            results['exit_price'] = timeout.iloc[-1]['close']

    if pd.notna(results['exit_price']) and adr > 0:
        if trade_dir == 'long':
            results['pnl_adr'] = (results['exit_price'] - entry_price) / adr
        else:
            results['pnl_adr'] = (entry_price - results['exit_price']) / adr

    return results


def run_sim(meta_sub, sl_type='sl_half'):
    results = []
    for _, row in meta_sub.iterrows():
        fpath = RAW_DIR / row['ticker'] / f"{row['date']}.parquet"
        if not fpath.exists():
            continue
        bars = pd.read_parquet(fpath)
        rth = bars[bars['session'] == 'rth']
        if len(rth) < 10:
            continue

        entry = row['close_935']
        adr = row.get('adr_10', np.nan)
        td = row.get('trade_direction', None)
        if pd.isna(entry) or pd.isna(adr) or adr <= 0 or td is None:
            continue

        if sl_type == 'sl_full':
            sl_level = row.get('sl_full_level', np.nan)
        elif sl_type == 'sl_half':
            sl_level = row.get('sl_half_level', np.nan)
        elif sl_type == 'sl_fix025':
            sl_level = entry - 0.25 * adr if td == 'long' else entry + 0.25 * adr
        else:
            continue

        if pd.isna(sl_level):
            continue

        r = simulate_rr_trade(rth, entry, sl_level, td, adr)
        if r:
            r['ticker'] = row['ticker']
            r['date'] = row['date']
            results.append(r)
    return pd.DataFrame(results)


def fmt_rr(trades, label=""):
    n = len(trades)
    if n < 10:
        return f"  {label:<45} | N={n:>4} [N<10]"
    med_sl = trades['sl_dist_adr'].median()
    wr1 = trades['hit_1R'].mean()
    wr2 = trades['hit_2R'].mean()
    wr3 = trades['hit_3R'].mean()
    sl_pct = trades['sl_hit'].mean()
    ev2 = (wr2 * 2 - (1-wr2) * 1) * med_sl if pd.notna(med_sl) else np.nan
    low = " [LOW N]" if n < 20 else ""
    return (f"  {label:<45} | N={n:>4} | SL={med_sl:.3f} | "
            f"WR@1R={wr1:>5.1%} | WR@2R={wr2:>5.1%} | WR@3R={wr3:>5.1%} | "
            f"EV@2R={ev2:>+.3f}{low}")


def main():
    lines = []
    lines.append("=" * 70)
    lines.append("D9 AUFGABEN 3+4: REVERSAL-FILTER + KONTEXT R:R (IS)")
    lines.append("=" * 70)
    lines.append("\nBreak-Even: 1R=50%, 2R=33.3%, 3R=25%")

    meta = pd.read_parquet('data/metadata/metadata_v9.parquet')
    h1 = meta[(meta['date'] >= '2021-02-21') & (meta['date'] <= '2023-12-31')].copy()
    h1 = h1[h1['od_body_pct'].notna() & h1['sl_full_adr'].notna()].copy()
    h1['gap_abs'] = h1['gap_size_in_adr'].abs()

    od05 = h1[h1['od_strength'] > 0.5]

    # ===== AUFGABE 3: Q_HIGH vs Q_LOW =====
    lines.append(f"\n\n{'='*70}")
    lines.append("AUFGABE 3a: Q_HIGH vs Q_LOW bei OD > 0.5")
    lines.append(f"{'='*70}")

    directions = [
        ('with GapUp', 'with_gap', 'up'),
        ('with GapDn', 'with_gap', 'down'),
        ('agst GapUp', 'against_gap', 'up'),
        ('agst GapDn', 'against_gap', 'down'),
    ]

    for sl_type in ['sl_full', 'sl_half', 'sl_fix025']:
        lines.append(f"\n  === SL-Typ: {sl_type.upper()} ===")

        for dir_label, od_dir, gap_dir in directions:
            sub = od05[(od05['od_direction'] == od_dir) & (od05['gap_direction'] == gap_dir)]
            qh = sub[sub['quality_high'] == True]
            ql = sub[sub['quality_high'] == False]

            print(f"  3a {sl_type} {dir_label}: QH={len(qh)}, QL={len(ql)}", file=sys.stderr)
            trades_h = run_sim(qh, sl_type)
            trades_l = run_sim(ql, sl_type)

            lines.append(f"\n  {dir_label}:")
            lines.append(fmt_rr(trades_h, f"Q_HIGH"))
            lines.append(fmt_rr(trades_l, f"Q_LOW"))

            # Delta
            if len(trades_h) >= 10 and len(trades_l) >= 10:
                d2 = trades_h['hit_2R'].mean() - trades_l['hit_2R'].mean()
                lines.append(f"  {'Delta WR@2R':<45} | {d2:>+.1%}")

    # 3b: Which SL type has best EV@2R for Q_HIGH with-gap?
    lines.append(f"\n\n{'='*70}")
    lines.append("AUFGABE 3b: Optimaler SL-Typ bei Q_HIGH with_gap")
    lines.append(f"{'='*70}")

    qh_with = od05[(od05['quality_high'] == True) & (od05['od_direction'] == 'with_gap')]
    for sl_type in ['sl_full', 'sl_half', 'sl_fix025']:
        trades = run_sim(qh_with, sl_type)
        lines.append(fmt_rr(trades, f"{sl_type.upper()} (Q_HIGH with)"))

    # ===== AUFGABE 4a: Marginal R:R effects =====
    lines.append(f"\n\n{'='*70}")
    lines.append("AUFGABE 4a: MARGINALE R:R-EFFEKTE (QH + OD>0.5)")
    lines.append(f"{'='*70}")
    lines.append("\nNur SL_HALF (bester Kandidat fuer positives R:R)")

    qh_od = od05[od05['quality_high'] == True]

    param_defs = [
        ('rvol_5', [0, 2, 5, 10, 999], ['<2x', '2-5x', '5-10x', '>10x'], 'RVOL_5'),
        ('rvol_open_30min', [0, 3, 5, 10, 999], ['<3x', '3-5x', '5-10x', '>10x'], 'RVOL_30'),
        ('gap_abs', [0, 1, 2, 4, 999], ['<1ADR', '1-2ADR', '2-4ADR', '>4ADR'], 'Gap_ADR'),
        ('pm_rth5', [0, 0.50, 1.00, 999], ['<50%', '50-100%', '>100%'], 'PM/RTH5'),
        ('pm_rth30_computed', [0, 0.10, 0.30, 999], ['<10%', '10-30%', '>30%'], 'PM/RTH30'),
    ]

    spread_results = {}  # For 4b

    for col, bks, bls, pname in param_defs:
        lines.append(f"\n  --- {pname} ---")

        for dir_label, od_dir, gap_dir in [('with GapUp', 'with_gap', 'up'), ('with GapDn', 'with_gap', 'down')]:
            sub = qh_od[(qh_od['od_direction'] == od_dir) & (qh_od['gap_direction'] == gap_dir)]
            lines.append(f"\n    {dir_label}:")

            best_wr2, worst_wr2 = -999, 999
            for i, bl in enumerate(bls):
                if i == 0:
                    mask = sub[col] < bks[1]
                elif i < len(bls) - 1:
                    mask = (sub[col] >= bks[i]) & (sub[col] < bks[i+1])
                else:
                    mask = sub[col] >= bks[-2]

                bsub = sub[mask]
                if len(bsub) < 10:
                    lines.append(f"    {bl:<15} | N={len(bsub):>3} [N<10]")
                    continue

                trades = run_sim(bsub, 'sl_half')
                wr2 = trades['hit_2R'].mean() if len(trades) >= 10 else np.nan
                if pd.notna(wr2):
                    if wr2 > best_wr2: best_wr2 = wr2
                    if wr2 < worst_wr2: worst_wr2 = wr2
                lines.append(fmt_rr(trades, f"{bl}"))

            spread = best_wr2 - worst_wr2 if best_wr2 > -999 and worst_wr2 < 999 else np.nan
            key = f"{pname}_{dir_label}"
            spread_results[key] = spread
            if pd.notna(spread):
                lines.append(f"    {'Spread WR@2R':<45} | {spread:>+.1%}")

    # ===== 4b: Spread summary =====
    lines.append(f"\n\n{'='*70}")
    lines.append("AUFGABE 4b: SPREAD-ZUSAMMENFASSUNG")
    lines.append(f"{'='*70}")

    lines.append(f"\n  {'Parameter':<12} | {'Spread with GU':>15} | {'Spread with GD':>15} | Bewertung")
    lines.append(f"  {'-'*65}")
    for pname in ['RVOL_5', 'RVOL_30', 'Gap_ADR', 'PM/RTH5', 'PM/RTH30']:
        sp_gu = spread_results.get(f"{pname}_with GapUp", np.nan)
        sp_gd = spread_results.get(f"{pname}_with GapDn", np.nan)
        avg_sp = np.nanmean([sp_gu, sp_gd]) if pd.notna(sp_gu) or pd.notna(sp_gd) else np.nan
        if pd.notna(avg_sp):
            bew = "EXTRA" if abs(avg_sp) > 0.05 else ("SCHWACH" if abs(avg_sp) > 0.03 else "REDUNDANT")
        else:
            bew = "N/A"
        sp_gu_s = f"{sp_gu:>+.1%}" if pd.notna(sp_gu) else "N/A"
        sp_gd_s = f"{sp_gd:>+.1%}" if pd.notna(sp_gd) else "N/A"
        lines.append(f"  {pname:<12} | {sp_gu_s:>15} | {sp_gd_s:>15} | {bew}")

    # ===== 4c: Cumulative layers =====
    lines.append(f"\n\n{'='*70}")
    lines.append("AUFGABE 4c: KUMULATIVE FILTER-SCHICHTEN (SL_HALF)")
    lines.append(f"{'='*70}")

    layer_defs = [
        ('L0: OD>0.5 with', lambda df: df[(df['od_strength'] > 0.5) & (df['od_direction'] == 'with_gap')]),
        ('L1: + QH', lambda df: df[(df['od_strength'] > 0.5) & (df['od_direction'] == 'with_gap') & (df['quality_high'] == True)]),
        ('L2a: +RV5>5', lambda df: df[(df['od_strength'] > 0.5) & (df['od_direction'] == 'with_gap') & (df['quality_high'] == True) & (df['rvol_5'] > 5)]),
        ('L2b: +Gap>2', lambda df: df[(df['od_strength'] > 0.5) & (df['od_direction'] == 'with_gap') & (df['quality_high'] == True) & (df['gap_abs'] > 2)]),
        ('L2c: +PM5<50', lambda df: df[(df['od_strength'] > 0.5) & (df['od_direction'] == 'with_gap') & (df['quality_high'] == True) & (df['pm_rth5'] < 0.50)]),
        ('L2d: +RV30>5', lambda df: df[(df['od_strength'] > 0.5) & (df['od_direction'] == 'with_gap') & (df['quality_high'] == True) & (df['rvol_open_30min'] > 5)]),
        ('L2e: +PM30<10', lambda df: df[(df['od_strength'] > 0.5) & (df['od_direction'] == 'with_gap') & (df['quality_high'] == True) & (df['pm_rth30_computed'] < 0.10)]),
        ('L3a: +RV5>5+Gap>2', lambda df: df[(df['od_strength'] > 0.5) & (df['od_direction'] == 'with_gap') & (df['quality_high'] == True) & (df['rvol_5'] > 5) & (df['gap_abs'] > 2)]),
        ('L3b: +RV30>5+PM30<10', lambda df: df[(df['od_strength'] > 0.5) & (df['od_direction'] == 'with_gap') & (df['quality_high'] == True) & (df['rvol_open_30min'] > 5) & (df['pm_rth30_computed'] < 0.10)]),
    ]

    for gap_dir, gap_label in [('up', 'GapUp'), ('down', 'GapDown')]:
        lines.append(f"\n  --- {gap_label} ---")
        base = h1[h1['gap_direction'] == gap_dir]

        for sl_type in ['sl_half', 'sl_fix025']:
            lines.append(f"\n    SL: {sl_type.upper()}")
            for layer_label, layer_fn in layer_defs:
                subset = layer_fn(base)
                n = len(subset)
                if n < 10:
                    lines.append(f"    {layer_label:<25} | N={n:>3} [N<10]")
                    continue
                print(f"  4c {gap_label} {sl_type} {layer_label}: N={n}", file=sys.stderr)
                trades = run_sim(subset, sl_type)
                lines.append(fmt_rr(trades, layer_label))

    # Also against-gap layers
    lines.append(f"\n  --- AGAINST GAP (SL_HALF only) ---")
    layer_defs_ag = [
        ('L0: OD>0.5 agst', lambda df: df[(df['od_strength'] > 0.5) & (df['od_direction'] == 'against_gap')]),
        ('L1: + QH', lambda df: df[(df['od_strength'] > 0.5) & (df['od_direction'] == 'against_gap') & (df['quality_high'] == True)]),
    ]
    for gap_dir, gap_label in [('up', 'GapUp'), ('down', 'GapDown')]:
        lines.append(f"\n    {gap_label}:")
        base = h1[h1['gap_direction'] == gap_dir]
        for layer_label, layer_fn in layer_defs_ag:
            subset = layer_fn(base)
            if len(subset) < 10:
                lines.append(f"    {layer_label:<25} | N={len(subset):>3} [N<10]")
                continue
            trades = run_sim(subset, 'sl_half')
            lines.append(fmt_rr(trades, layer_label))

    # ===== 4d: Optimal R:R per setup =====
    lines.append(f"\n\n{'='*70}")
    lines.append("AUFGABE 4d: OPTIMALER R:R PRO SETUP (SL_HALF)")
    lines.append(f"{'='*70}")
    lines.append("\n  EV fuer R:R = 0.5 bis 5.0 in 0.5er Schritten")
    lines.append("  EV(XR) = (WR@XR * X - (1-WR@XR)) * med_SL")

    # For key layers, compute optimal R:R
    key_layers = [
        ('L0: OD>0.5 with', lambda df: df[(df['od_strength'] > 0.5) & (df['od_direction'] == 'with_gap')]),
        ('L1: + QH', lambda df: df[(df['od_strength'] > 0.5) & (df['od_direction'] == 'with_gap') & (df['quality_high'] == True)]),
        ('L2d: +RV30>5', lambda df: df[(df['od_strength'] > 0.5) & (df['od_direction'] == 'with_gap') & (df['quality_high'] == True) & (df['rvol_open_30min'] > 5)]),
        ('L2e: +PM30<10', lambda df: df[(df['od_strength'] > 0.5) & (df['od_direction'] == 'with_gap') & (df['quality_high'] == True) & (df['pm_rth30_computed'] < 0.10)]),
    ]

    rr_range = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

    for gap_dir, gap_label in [('up', 'GapUp'), ('down', 'GapDown')]:
        lines.append(f"\n  --- {gap_label} ---")
        base = h1[h1['gap_direction'] == gap_dir]

        for layer_label, layer_fn in key_layers:
            subset = layer_fn(base)
            if len(subset) < 10:
                continue

            # Extended simulation with more targets
            all_results = []
            for _, row in subset.iterrows():
                fpath = RAW_DIR / row['ticker'] / f"{row['date']}.parquet"
                if not fpath.exists():
                    continue
                bars = pd.read_parquet(fpath)
                rth = bars[bars['session'] == 'rth']
                if len(rth) < 10:
                    continue
                entry = row['close_935']
                adr = row.get('adr_10', np.nan)
                td = row.get('trade_direction', None)
                sl_level = row.get('sl_half_level', np.nan)
                if pd.isna(entry) or pd.isna(adr) or adr <= 0 or td is None or pd.isna(sl_level):
                    continue

                sl_dist = abs(entry - sl_level)
                if sl_dist <= 0:
                    continue

                # Walk through bars
                hit_rrs = {rr: False for rr in rr_range}
                sl_hit = False
                post_entry = rth[rth['time_et'] >= '09:36']

                for _, bar in post_entry.iterrows():
                    if bar['time_et'] > '15:55':
                        break
                    if td == 'long':
                        if bar['low'] <= sl_level:
                            sl_hit = True
                            break
                        for rr in rr_range:
                            tgt = entry + rr * sl_dist
                            if bar['high'] >= tgt:
                                hit_rrs[rr] = True
                    else:
                        if bar['high'] >= sl_level:
                            sl_hit = True
                            break
                        for rr in rr_range:
                            tgt = entry - rr * sl_dist
                            if bar['low'] <= tgt:
                                hit_rrs[rr] = True

                res = {'sl_hit': sl_hit, 'sl_dist_adr': sl_dist / adr}
                for rr in rr_range:
                    res[f'hit_{rr}R'] = hit_rrs[rr]
                all_results.append(res)

            if len(all_results) < 10:
                continue

            df_res = pd.DataFrame(all_results)
            med_sl = df_res['sl_dist_adr'].median()

            best_ev = -999
            best_rr = 0
            rr_line = f"    {layer_label:<25}: "
            for rr in rr_range:
                wr = df_res[f'hit_{rr}R'].mean()
                ev = (wr * (rr+1) - 1) * med_sl  # (WR*(R+1) - 1) * SL
                if ev > best_ev:
                    best_ev = ev
                    best_rr = rr
                rr_line += f"{rr}R={wr:.0%}/{ev:+.3f} "

            lines.append(rr_line)
            lines.append(f"    {'':>25}  -> Optimal: {best_rr}R, EV={best_ev:+.3f} ADR (N={len(df_res)})")

    output = "\n".join(lines)
    with open('results/d9_rr_quality.txt', 'w', encoding='utf-8') as f:
        f.write(output)
    print(output)
    print(f"\nSaved to results/d9_rr_quality.txt", file=sys.stderr)


if __name__ == '__main__':
    main()
