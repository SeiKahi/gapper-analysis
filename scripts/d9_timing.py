"""
D9 Aufgabe 5: 9:35 vs 10:00 Timing mit R:R-Metriken (IS)
5a: 9:35-Parameter (sofort verfuegbar): OD > 0.5 + QH + [RVOL_5 / PM_RTH5 / Gap_in_ADR]
5b: 10:00-Parameter (muss warten): OD > 0.5 + QH + [RVOL_30 / PM_RTH30]
5c: Inkrementeller Wert des Wartens (5b vs 5a)
5d: 10:00-Entry mit SL_30MIN (|Entry_1000 - 30min_Extremum|)
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

RAW_DIR = Path('data/raw_1min')


def simulate_rr_trade(bars_rth, entry_price, sl_level, trade_dir, adr,
                      entry_time='09:36', timeout='15:55', rr_targets=[1, 2, 3]):
    """
    Bar-by-bar R:R trade simulation.
    entry_time: first bar to check (inclusive).
    """
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


def run_sim_935(meta_subset, sl_type='sl_half'):
    """Run 9:35 entry simulation."""
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

        if sl_type == 'sl_half':
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

        result = simulate_rr_trade(rth, entry_price, sl_level, trade_dir, adr,
                                   entry_time='09:36')
        if result is None:
            continue

        result['ticker'] = ticker
        result['date'] = date
        result['sl_type'] = sl_type
        result['entry_type'] = '935'
        all_results.append(result)

    return pd.DataFrame(all_results)


def run_sim_1000(meta_subset, sl_type='sl_30min'):
    """
    Run 10:00 entry simulation.
    sl_type='sl_30min': SL at 30min extremum (09:30-09:59 H/L)
    sl_type='sl_fix025': SL at 0.25 ADR from entry_1000
    """
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

        entry_price = row.get('close_1000', np.nan)
        adr = row.get('adr_10', np.nan)
        trade_dir = row.get('trade_direction', None)

        if pd.isna(entry_price) or pd.isna(adr) or adr <= 0 or trade_dir is None:
            continue

        if sl_type == 'sl_30min':
            # 30min extremum: 09:30-09:59
            bars_30 = rth[(rth['time_et'] >= '09:30') & (rth['time_et'] <= '09:59')]
            if len(bars_30) == 0:
                continue
            high_30 = bars_30['high'].max()
            low_30 = bars_30['low'].min()

            if trade_dir == 'long':
                sl_level = low_30
            else:
                sl_level = high_30
        elif sl_type == 'sl_fix025':
            if trade_dir == 'long':
                sl_level = entry_price - 0.25 * adr
            else:
                sl_level = entry_price + 0.25 * adr
        else:
            continue

        if pd.isna(sl_level):
            continue

        result = simulate_rr_trade(rth, entry_price, sl_level, trade_dir, adr,
                                   entry_time='10:01')
        if result is None:
            continue

        result['ticker'] = ticker
        result['date'] = date
        result['sl_type'] = sl_type
        result['entry_type'] = '1000'
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


def ev_curve(trades, label="", rr_range=None):
    """EV curve for multiple R:R targets."""
    if rr_range is None:
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
        # For non-integer RR, compute from pnl_r
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
    lines.append("D9 AUFGABE 5: 9:35 vs 10:00 TIMING MIT R:R (IS)")
    lines.append("=" * 70)
    lines.append("\n9:35-Entry: close_935, SL=OD-basiert, Sim ab 09:36")
    lines.append("10:00-Entry: close_1000, SL=30min-Extremum, Sim ab 10:01")
    lines.append("Break-Even: 1R=50%, 2R=33.3%, 3R=25%")

    meta = pd.read_parquet('data/metadata/metadata_v9.parquet')
    h1 = meta[(meta['date'] >= '2021-02-21') & (meta['date'] <= '2023-12-31')].copy()
    h1 = h1[h1['od_body_pct'].notna() & h1['sl_full_adr'].notna()].copy()
    print(f"H1 with valid features: {len(h1)}", file=sys.stderr)

    # Base subset: OD > 0.5 with_gap + QUALITY_HIGH
    base = h1[(h1['od_strength'] > 0.5) & (h1['od_direction'] == 'with_gap') & (h1['quality_high'] == True)]
    print(f"Base (QH+OD>0.5 with): {len(base)}", file=sys.stderr)

    # =====================================================================
    # 5a: 9:35-PARAMETER (sofort verfuegbar)
    # =====================================================================
    lines.append(f"\n\n{'='*70}")
    lines.append("5a: 9:35-PARAMETER (sofort verfuegbar)")
    lines.append(f"{'='*70}")
    lines.append("\nFilter: OD > 0.5 + with_gap + QH + 9:35-Parameter")
    lines.append("SL: SL_HALF | Entry: 9:35\n")

    for gap_dir, gap_label in [('up', 'GapUp'), ('down', 'GapDown')]:
        lines.append(f"--- {gap_label} ---")
        gap_base = base[base['gap_direction'] == gap_dir]

        # Baseline
        trades_base = run_sim_935(gap_base, 'sl_half')
        lines.append(summarize_rr(trades_base, "Baseline (QH+OD>0.5 with)"))

        # RVOL_5 filters
        for rvol_lo, rvol_hi, rlabel in [(5, 999, '+RVOL5>5'), (0, 2, '+RVOL5<2')]:
            sub = gap_base[(gap_base['rvol_5'] >= rvol_lo) & (gap_base['rvol_5'] < rvol_hi)]
            trades = run_sim_935(sub, 'sl_half')
            lines.append(summarize_rr(trades, rlabel))

        # PM/RTH5 filters
        for pm_lo, pm_hi, plabel in [(0, 0.50, '+PM5<50%'), (0.50, 999, '+PM5>50%')]:
            sub = gap_base[(gap_base['pm_rth5'] >= pm_lo) & (gap_base['pm_rth5'] < pm_hi)]
            trades = run_sim_935(sub, 'sl_half')
            lines.append(summarize_rr(trades, plabel))

        # Gap size filters
        for glo, ghi, glabel in [(0, 1, '+Gap<1ADR'), (1, 2, '+Gap 1-2ADR'), (2, 999, '+Gap>2ADR')]:
            sub = gap_base[(gap_base['gap_size_in_adr'] >= glo) & (gap_base['gap_size_in_adr'] < ghi)]
            trades = run_sim_935(sub, 'sl_half')
            lines.append(summarize_rr(trades, glabel))

        # Best 9:35 combos (N >= 30)
        lines.append(f"\n  Beste 9:35-Kombinationen (N>=30):")
        combos_935 = [
            ('RV5>5 + Gap<1', lambda df: df[(df['rvol_5'] >= 5) & (df['gap_size_in_adr'] < 1)]),
            ('RV5>5 + PM5<50', lambda df: df[(df['rvol_5'] >= 5) & (df['pm_rth5'] < 0.50)]),
            ('Gap<1 + PM5<50', lambda df: df[(df['gap_size_in_adr'] < 1) & (df['pm_rth5'] < 0.50)]),
            ('RV5>5+Gap<1+PM5<50', lambda df: df[(df['rvol_5'] >= 5) & (df['gap_size_in_adr'] < 1) & (df['pm_rth5'] < 0.50)]),
        ]
        for clabel, cfn in combos_935:
            sub = cfn(gap_base)
            trades = run_sim_935(sub, 'sl_half')
            lines.append(summarize_rr(trades, f"  {clabel}"))

        lines.append("")

    # =====================================================================
    # 5b: 10:00-PARAMETER (muss warten)
    # =====================================================================
    lines.append(f"\n{'='*70}")
    lines.append("5b: 10:00-PARAMETER (muss warten)")
    lines.append(f"{'='*70}")
    lines.append("\nFilter: OD > 0.5 + with_gap + QH + 10:00-Parameter")
    lines.append("SL: SL_HALF | Entry: 9:35 (gleicher Entry, nur andere Filter)\n")

    for gap_dir, gap_label in [('up', 'GapUp'), ('down', 'GapDown')]:
        lines.append(f"--- {gap_label} ---")
        gap_base = base[base['gap_direction'] == gap_dir]

        trades_base = run_sim_935(gap_base, 'sl_half')
        lines.append(summarize_rr(trades_base, "Baseline (QH+OD>0.5 with)"))

        # RVOL_30 filters
        for rvol_lo, rvol_hi, rlabel in [(5, 999, '+RV30>5'), (10, 999, '+RV30>10'), (0, 3, '+RV30<3')]:
            sub = gap_base[(gap_base['rvol_open_30min'] >= rvol_lo) & (gap_base['rvol_open_30min'] < rvol_hi)]
            trades = run_sim_935(sub, 'sl_half')
            lines.append(summarize_rr(trades, rlabel))

        # PM/RTH30 filters
        for pm_lo, pm_hi, plabel in [(0, 0.10, '+PM30<10%'), (0.10, 0.30, '+PM30 10-30%'), (0.30, 999, '+PM30>30%')]:
            sub = gap_base[(gap_base['pm_rth30_computed'] >= pm_lo) & (gap_base['pm_rth30_computed'] < pm_hi)]
            trades = run_sim_935(sub, 'sl_half')
            lines.append(summarize_rr(trades, plabel))

        # Best 10:00 combos
        lines.append(f"\n  Beste 10:00-Kombinationen (N>=20):")
        combos_1000 = [
            ('RV30>5 + PM30<10', lambda df: df[(df['rvol_open_30min'] >= 5) & (df['pm_rth30_computed'] < 0.10)]),
            ('RV30>5 + PM30<30', lambda df: df[(df['rvol_open_30min'] >= 5) & (df['pm_rth30_computed'] < 0.30)]),
            ('RV30>10 + PM30<10', lambda df: df[(df['rvol_open_30min'] >= 10) & (df['pm_rth30_computed'] < 0.10)]),
        ]
        for clabel, cfn in combos_1000:
            sub = cfn(gap_base)
            trades = run_sim_935(sub, 'sl_half')
            lines.append(summarize_rr(trades, f"  {clabel}"))

        lines.append("")

    # =====================================================================
    # 5c: INKREMENTELLER WERT DES WARTENS
    # =====================================================================
    lines.append(f"\n{'='*70}")
    lines.append("5c: INKREMENTELLER WERT DES WARTENS (10:00 vs 9:35 Parameter)")
    lines.append(f"{'='*70}")
    lines.append("\nVergleich: Beste 9:35-Kombi vs Beste 10:00-Kombi")
    lines.append("Beide mit SL_HALF, Entry 9:35\n")

    for gap_dir, gap_label in [('up', 'GapUp'), ('down', 'GapDown')]:
        lines.append(f"--- {gap_label} ---")
        gap_base = base[base['gap_direction'] == gap_dir]

        # Best 9:35 combo: use most promising from 5a
        # RVOL5>5 is the main 9:35 filter
        best_935 = gap_base[gap_base['rvol_5'] >= 5]
        trades_935 = run_sim_935(best_935, 'sl_half')

        # Best 10:00 combo: RV30>5 + PM30<10
        best_1000 = gap_base[(gap_base['rvol_open_30min'] >= 5) & (gap_base['pm_rth30_computed'] < 0.10)]
        trades_1000 = run_sim_935(best_1000, 'sl_half')

        lines.append(summarize_rr(trades_935, "Best 9:35 (QH+RV5>5)"))
        lines.append(summarize_rr(trades_1000, "Best 10:00 (QH+RV30>5+PM30<10)"))

        # Delta
        if len(trades_935) >= 10 and len(trades_1000) >= 10:
            wr2_935 = trades_935['hit_2R'].mean()
            wr2_1000 = trades_1000['hit_2R'].mean()
            delta = wr2_1000 - wr2_935
            lines.append(f"  Delta WR@2R: {delta:>+.1%} (Wert des Wartens)")
            lines.append(f"  N-Verlust: {len(trades_935)} -> {len(trades_1000)} ({len(trades_1000)/len(trades_935):.0%})")
        lines.append("")

    # =====================================================================
    # 5d: 10:00-ENTRY MIT SL_30MIN
    # =====================================================================
    lines.append(f"\n{'='*70}")
    lines.append("5d: 10:00-ENTRY MIT SL_30MIN")
    lines.append(f"{'='*70}")
    lines.append("\nEntry: close_1000 | SL: 30min-Extremum (09:30-09:59 H/L)")
    lines.append("Sim ab 10:01 | Timeout: 15:55\n")

    for gap_dir, gap_label in [('up', 'GapUp'), ('down', 'GapDown')]:
        lines.append(f"--- {gap_label} ---")
        gap_base = base[base['gap_direction'] == gap_dir]
        print(f"  5d {gap_label}: N={len(gap_base)}", file=sys.stderr)

        # SL_30MIN
        trades_30min = run_sim_1000(gap_base, 'sl_30min')
        lines.append(summarize_rr(trades_30min, "SL_30MIN (Entry 10:00)"))

        # SL_FIX025 for comparison
        trades_fix = run_sim_1000(gap_base, 'sl_fix025')
        lines.append(summarize_rr(trades_fix, "SL_FIX025 (Entry 10:00)"))

        # Compare with 9:35 baseline
        trades_935_half = run_sim_935(gap_base, 'sl_half')
        trades_935_fix = run_sim_935(gap_base, 'sl_fix025')
        lines.append(summarize_rr(trades_935_half, "SL_HALF (Entry 9:35, Referenz)"))
        lines.append(summarize_rr(trades_935_fix, "SL_FIX025 (Entry 9:35, Referenz)"))

        # SL_30MIN distribution
        if len(trades_30min) >= 10:
            sl_vals = trades_30min['sl_dist_adr']
            lines.append(f"\n  SL_30MIN Verteilung: Med={sl_vals.median():.3f}, "
                         f"Mean={sl_vals.mean():.3f}, Q25={sl_vals.quantile(0.25):.3f}, "
                         f"Q75={sl_vals.quantile(0.75):.3f}")

        # With best 10:00 filter combo
        best_filt = gap_base[(gap_base['rvol_open_30min'] >= 5) & (gap_base['pm_rth30_computed'] < 0.10)]
        trades_best = run_sim_1000(best_filt, 'sl_30min')
        lines.append(summarize_rr(trades_best, "SL_30MIN + RV30>5+PM30<10"))

        # EV curve for best setups
        lines.append(f"\n  EV-Kurven (SL_30MIN):")
        lines.append(ev_curve(trades_30min, "Baseline"))
        if len(trades_best) >= 10:
            lines.append(ev_curve(trades_best, "Best Filter"))

        lines.append("")

    # =====================================================================
    # 5e: ZUSAMMENFASSUNG
    # =====================================================================
    lines.append(f"\n{'='*70}")
    lines.append("5e: ZUSAMMENFASSUNG TIMING")
    lines.append(f"{'='*70}")
    lines.append("""
  Vergleich aller Timing-Varianten (SL_HALF, mit Filter):

  Setup                          | Entry | Filter-Basis  | Bemerkung
  -----------------------------------------------------------------------
  QH+OD>0.5 with (Baseline)     | 9:35  | 9:35          | Sofort handelbar
  + Best 9:35 Filter             | 9:35  | 9:35          | RVOL_5 verfuegbar
  + Best 10:00 Filter            | 9:35  | 10:00         | Warten auf 30min Daten
  + 10:00-Entry + SL_30MIN       | 10:00 | 10:00         | Spaeterer Einstieg
""")

    output = "\n".join(lines)
    with open('results/d9_timing.txt', 'w', encoding='utf-8') as f:
        f.write(output)
    print(output)
    print(f"\nSaved to results/d9_timing.txt", file=sys.stderr)


if __name__ == '__main__':
    main()
