"""
D9s: R:R-basierte Trade-Simulation mit 1-Sekunden-Aufloesung
Hybrid: 1-min Basis + 1-sec Zoom bei ambigen Bars.
Repliziert D9 Aufgaben 2-6 + Synthese.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
from tqdm import tqdm

RAW_1MIN_DIR = Path('data/raw_1min')
RAW_1SEC_DIR = Path('data/raw_1sec')

# ============================================================
# 1-sec cache
# ============================================================
_sec_cache = {}

def load_1sec(ticker, date):
    """Load 1-sec bars with caching."""
    key = (ticker, date)
    if key not in _sec_cache:
        path = RAW_1SEC_DIR / ticker / f"{date}.parquet"
        if path.exists():
            _sec_cache[key] = pd.read_parquet(path)
        else:
            _sec_cache[key] = None
    return _sec_cache[key]


# ============================================================
# Hybrid R:R simulation
# ============================================================
def simulate_rr_trade_1sec(bars_rth, entry_price, sl_level, trade_dir, adr,
                            ticker, date, rr_targets=[1, 2, 3]):
    """
    Hybrid 1-min/1-sec R:R trade simulation.
    - Iterates over 1-min bars (as before)
    - When BOTH SL-hit AND unfilled-target are possible in same bar:
      zoom to 1-sec data and walk second-by-second.
    - Within a 1-sec bar, SL is checked first (conservative at 1-sec level).
    - Across 1-sec bars, targets hit in earlier seconds count even if SL hits later.
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
    results['zoom_used'] = False
    results['zoom_changed'] = False

    sec_bars = load_1sec(ticker, date)

    post_entry = bars_rth[bars_rth['time_et'] >= '09:36']

    for _, bar in post_entry.iterrows():
        if bar['time_et'] > '15:55':
            break

        bar_time = str(bar['time_et'])[:5]  # "09:36"

        if trade_dir == 'long':
            sl_possible = bar['low'] <= sl_level
            unfilled = [rr for rr in sorted(rr_targets)
                        if not results[f'hit_{rr}R'] and bar['high'] >= targets[rr]]
            target_possible = len(unfilled) > 0

            if sl_possible and target_possible:
                # AMBIGUOUS bar â€” try 1-sec zoom
                resolved = False
                if sec_bars is not None and bar_time >= '09:30' and bar_time < '09:45':
                    minute_secs = sec_bars[sec_bars['time_et'].str[:5] == bar_time]
                    if len(minute_secs) > 0:
                        results['zoom_used'] = True
                        # Walk second by second
                        for _, sec in minute_secs.iterrows():
                            # SL first within same second (conservative)
                            if sec['low'] <= sl_level:
                                results['sl_hit'] = True
                                results['exit_price'] = sl_level
                                results['exit_type'] = 'sl'
                                results['sl_time'] = sec['time_et']
                                resolved = True
                                break
                            # Targets (no SL in this second â†’ targets are definite)
                            for rr in sorted(unfilled):
                                if sec['high'] >= targets[rr] and not results[f'hit_{rr}R']:
                                    results[f'hit_{rr}R'] = True
                                    results['zoom_changed'] = True
                            unfilled = [rr for rr in unfilled if not results[f'hit_{rr}R']]

                        if not resolved:
                            # 1-sec didn't trigger SL (OHLC gap). Mark targets hit, then SL.
                            for rr in unfilled:
                                if bar['high'] >= targets[rr]:
                                    results[f'hit_{rr}R'] = True
                                    results['zoom_changed'] = True
                            results['sl_hit'] = True
                            results['exit_price'] = sl_level
                            results['exit_type'] = 'sl'
                            results['sl_time'] = bar_time
                            resolved = True

                        if resolved:
                            break

                if not resolved:
                    # No 1-sec data: conservative (SL first, targets NOT counted)
                    results['sl_hit'] = True
                    results['exit_price'] = sl_level
                    results['exit_type'] = 'sl'
                    results['sl_time'] = bar_time
                    break

            elif sl_possible:
                results['sl_hit'] = True
                results['exit_price'] = sl_level
                results['exit_type'] = 'sl'
                results['sl_time'] = bar_time
                break

            elif target_possible:
                for rr in unfilled:
                    results[f'hit_{rr}R'] = True

        else:  # short
            sl_possible = bar['high'] >= sl_level
            unfilled = [rr for rr in sorted(rr_targets)
                        if not results[f'hit_{rr}R'] and bar['low'] <= targets[rr]]
            target_possible = len(unfilled) > 0

            if sl_possible and target_possible:
                resolved = False
                if sec_bars is not None and bar_time >= '09:30' and bar_time < '09:45':
                    minute_secs = sec_bars[sec_bars['time_et'].str[:5] == bar_time]
                    if len(minute_secs) > 0:
                        results['zoom_used'] = True
                        for _, sec in minute_secs.iterrows():
                            if sec['high'] >= sl_level:
                                results['sl_hit'] = True
                                results['exit_price'] = sl_level
                                results['exit_type'] = 'sl'
                                results['sl_time'] = sec['time_et']
                                resolved = True
                                break
                            for rr in sorted(unfilled):
                                if sec['low'] <= targets[rr] and not results[f'hit_{rr}R']:
                                    results[f'hit_{rr}R'] = True
                                    results['zoom_changed'] = True
                            unfilled = [rr for rr in unfilled if not results[f'hit_{rr}R']]

                        if not resolved:
                            for rr in unfilled:
                                if bar['low'] <= targets[rr]:
                                    results[f'hit_{rr}R'] = True
                                    results['zoom_changed'] = True
                            results['sl_hit'] = True
                            results['exit_price'] = sl_level
                            results['exit_type'] = 'sl'
                            results['sl_time'] = bar_time
                            resolved = True

                        if resolved:
                            break

                if not resolved:
                    results['sl_hit'] = True
                    results['exit_price'] = sl_level
                    results['exit_type'] = 'sl'
                    results['sl_time'] = bar_time
                    break

            elif sl_possible:
                results['sl_hit'] = True
                results['exit_price'] = sl_level
                results['exit_type'] = 'sl'
                results['sl_time'] = bar_time
                break

            elif target_possible:
                for rr in unfilled:
                    results[f'hit_{rr}R'] = True

    # Timeout
    if not results['sl_hit']:
        timeout_bars = post_entry[post_entry['time_et'] <= '15:55']
        if len(timeout_bars) > 0:
            results['exit_price'] = timeout_bars.iloc[-1]['close']

    # PnL
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


# ============================================================
# Run simulation
# ============================================================
def run_simulation(meta_subset, sl_type='sl_fix025'):
    """Run hybrid trade simulation for a metadata subset."""
    all_results = []
    for _, row in meta_subset.iterrows():
        ticker = row['ticker']
        date = str(row['date'])
        fpath = RAW_1MIN_DIR / ticker / f"{date}.parquet"
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
            sl_level = entry_price - 0.25 * adr if trade_dir == 'long' else entry_price + 0.25 * adr
        else:
            continue

        if pd.isna(sl_level):
            continue

        result = simulate_rr_trade_1sec(rth, entry_price, sl_level, trade_dir, adr,
                                         ticker, date)
        if result is None:
            continue

        result['ticker'] = ticker
        result['date'] = date
        result['sl_type'] = sl_type
        result['od_direction'] = row['od_direction']
        result['gap_direction'] = row['gap_direction']
        all_results.append(result)

    return pd.DataFrame(all_results)


# ============================================================
# Formatting helpers
# ============================================================
def summarize_rr(trades, label=""):
    n = len(trades)
    if n < 10:
        return f"  {label:<45} | N={n:>4} [N<10]"

    med_sl = trades['sl_dist_adr'].median()
    wr1 = trades['hit_1R'].mean()
    wr2 = trades['hit_2R'].mean()
    wr3 = trades['hit_3R'].mean()
    sl_pct = trades['sl_hit'].mean()
    to_pct = (trades['exit_type'] == 'timeout').mean()
    ev2 = (wr2 * 2 - (1-wr2) * 1) * med_sl if pd.notna(med_sl) else np.nan
    low = " [LOW N]" if n < 20 else ""

    # Zoom stats
    zoom_used_pct = trades['zoom_used'].mean() if 'zoom_used' in trades.columns else 0
    zoom_changed_pct = trades['zoom_changed'].mean() if 'zoom_changed' in trades.columns else 0

    return (f"  {label:<45} | N={n:>4} | SL={med_sl:.3f} | "
            f"WR@1R={wr1:>5.1%} | WR@2R={wr2:>5.1%} | WR@3R={wr3:>5.1%} | "
            f"EV@2R={ev2:>+.3f} | Zoom={zoom_used_pct:.0%}/{zoom_changed_pct:.0%}{low}")


def sl_timing_analysis(trades, label=""):
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


def ev_curve(trades, label=""):
    rr_range = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    n = len(trades)
    if n < 10:
        return f"    {label:<25}: N={n} [N<10]"

    med_sl = trades['sl_dist_adr'].median()
    parts = []
    best_rr = None
    best_ev = -999

    for rr in rr_range:
        col = f'hit_{int(rr)}R' if rr == int(rr) else None
        if col and col in trades.columns:
            wr = trades[col].mean()
        else:
            continue

        ev = (wr * rr - (1 - wr)) * med_sl
        parts.append(f"{int(rr)}R={wr:.0%}/{ev:+.3f}")
        if ev > best_ev:
            best_ev = ev
            best_rr = rr

    line = f"    {label:<25}: " + " ".join(parts)
    if best_rr is not None:
        line += f"\n{'':>29}-> Optimal: {int(best_rr)}R, EV={best_ev:+.3f} ADR (N={n})"
    return line


# ============================================================
# Main
# ============================================================
def main():
    lines = []
    lines.append("=" * 70)
    lines.append("D9s: R:R-BASIERTE TRADE-SIMULATION MIT 1-SEC ZOOM")
    lines.append("=" * 70)
    lines.append("\nEntry: 9:35 | Timeout: 15:55 | R:R-Targets: 1R, 2R, 3R")
    lines.append("Break-Even WRs: 1R=50.0%, 2R=33.3%, 3R=25.0%")
    lines.append("HYBRID: 1-min base + 1-sec zoom bei ambigen Bars (09:30-09:45)")
    lines.append("Zoom-Stats: Zoom=used%/changed% (wie oft 1-sec genutzt / Ergebnis geaendert)")

    meta = pd.read_parquet('data/metadata/metadata_v9.parquet')

    # IS = H1
    h1 = meta[(meta['date'] >= '2021-02-21') & (meta['date'] <= '2023-12-31')].copy()
    h1 = h1[h1['od_body_pct'].notna() & h1['sl_full_adr'].notna()].copy()
    h1['gap_abs'] = h1['gap_size_in_adr'].abs()

    # OOS = H2 (extended to 2026-02-13)
    h2 = meta[(meta['date'] >= '2024-01-01')].copy()
    h2 = h2[h2['od_body_pct'].notna() & h2['sl_full_adr'].notna()].copy()
    h2['gap_abs'] = h2['gap_size_in_adr'].abs()

    print(f"H1 (IS): {len(h1)}, H2 (OOS): {len(h2)}", file=sys.stderr)

    # ===== 2a: OD > 0.5 with Gap â€” IS =====
    lines.append(f"\n\n{'='*70}")
    lines.append("2a: OD > 0.5 WITH GAP â€” IS (1-sec Zoom)")
    lines.append(f"{'='*70}")

    for gap_dir, gap_label in [('up', 'GapUp'), ('down', 'GapDown')]:
        lines.append(f"\n--- {gap_label} ---")
        sub = h1[(h1['od_strength'] > 0.5) & (h1['od_direction'] == 'with_gap') & (h1['gap_direction'] == gap_dir)]
        print(f"  {gap_label} with IS: N={len(sub)}", file=sys.stderr)

        for sl_type, sl_label in [('sl_full', 'SL_FULL'), ('sl_half', 'SL_HALF'), ('sl_fix025', 'SL_FIX025')]:
            trades = run_simulation(sub, sl_type)
            lines.append(summarize_rr(trades, sl_label))

    # ===== 2b: OD > 0.5 against Gap â€” IS =====
    lines.append(f"\n\n{'='*70}")
    lines.append("2b: OD > 0.5 AGAINST GAP â€” IS (1-sec Zoom)")
    lines.append(f"{'='*70}")

    for gap_dir, gap_label in [('up', 'GapUp'), ('down', 'GapDown')]:
        lines.append(f"\n--- {gap_label} ---")
        sub = h1[(h1['od_strength'] > 0.5) & (h1['od_direction'] == 'against_gap') & (h1['gap_direction'] == gap_dir)]
        print(f"  {gap_label} against IS: N={len(sub)}", file=sys.stderr)

        for sl_type, sl_label in [('sl_full', 'SL_FULL'), ('sl_half', 'SL_HALF'), ('sl_fix025', 'SL_FIX025')]:
            trades = run_simulation(sub, sl_type)
            lines.append(summarize_rr(trades, sl_label))

    # ===== 2c: SL-Hit Timing =====
    lines.append(f"\n\n{'='*70}")
    lines.append("2c: SL-HIT TIMING (OD > 0.5 with Gap, IS)")
    lines.append(f"{'='*70}")

    all_with = h1[(h1['od_strength'] > 0.5) & (h1['od_direction'] == 'with_gap')]
    for sl_type, sl_label in [('sl_full', 'SL_FULL'), ('sl_half', 'SL_HALF'), ('sl_fix025', 'SL_FIX025')]:
        trades = run_simulation(all_with, sl_type)
        lines.append(f"\n{sl_timing_analysis(trades, sl_label)}")

    # ===== ZOOM STATISTICS =====
    lines.append(f"\n\n{'='*70}")
    lines.append("ZOOM-STATISTIK (1-sec Aufloesung)")
    lines.append(f"{'='*70}")

    for sl_type, sl_label in [('sl_full', 'SL_FULL'), ('sl_half', 'SL_HALF'), ('sl_fix025', 'SL_FIX025')]:
        trades = run_simulation(all_with, sl_type)
        n = len(trades)
        if n < 10:
            continue
        n_zoom = trades['zoom_used'].sum()
        n_changed = trades['zoom_changed'].sum()
        lines.append(f"\n  {sl_label}: N={n}")
        lines.append(f"    Zoom eingesetzt: {n_zoom}/{n} ({100*n_zoom/n:.1f}%)")
        lines.append(f"    Ergebnis geaendert: {n_changed}/{n} ({100*n_changed/n:.1f}%)")
        if n_changed > 0:
            lines.append(f"    â†’ {n_changed} Trades hatten Target VOR SL (in 1-min als SL-first gezaehlt)")

    # ===== 6a: OOS Basis =====
    lines.append(f"\n\n{'='*70}")
    lines.append("6a: OOS-VALIDIERUNG (OD > 0.5 with_gap, 1-sec Zoom)")
    lines.append(f"{'='*70}")

    for gap_dir, gap_label in [('up', 'GapUp'), ('down', 'GapDown')]:
        lines.append(f"\n--- {gap_label} ---")

        for half_name, half_df in [('IS', h1), ('OOS', h2)]:
            sub = half_df[(half_df['od_strength'] > 0.5) & (half_df['od_direction'] == 'with_gap') & (half_df['gap_direction'] == gap_dir)]
            print(f"  {gap_label} {half_name}: N={len(sub)}", file=sys.stderr)

            for sl_type, sl_label in [('sl_half', 'SL_HALF'), ('sl_fix025', 'SL_FIX025')]:
                trades = run_simulation(sub, sl_type)
                lines.append(summarize_rr(trades, f"{half_name} {sl_label}"))

    # ===== 6b: Q_HIGH OOS =====
    lines.append(f"\n\n{'='*70}")
    lines.append("6b: Q_HIGH FILTER OOS (1-sec Zoom)")
    lines.append(f"{'='*70}")

    for gap_dir, gap_label in [('up', 'GapUp'), ('down', 'GapDown')]:
        lines.append(f"\n--- {gap_label} ---")
        for half_name, half_df in [('IS', h1), ('OOS', h2)]:
            sub = half_df[(half_df['od_strength'] > 0.5) & (half_df['od_direction'] == 'with_gap') &
                          (half_df['gap_direction'] == gap_dir) & (half_df['quality_high'] == True)]
            for sl_type, sl_label in [('sl_half', 'SL_HALF'), ('sl_fix025', 'SL_FIX025')]:
                trades = run_simulation(sub, sl_type)
                lines.append(summarize_rr(trades, f"{half_name} QH {sl_label}"))
        lines.append("")

    # ===== 6d: Kumulative Filter OOS =====
    lines.append(f"\n{'='*70}")
    lines.append("6d: KUMULATIVE FILTER OOS (1-sec Zoom)")
    lines.append(f"{'='*70}")

    layer_configs = [
        ('L0: OD>0.5 with', lambda df: df[(df['od_strength'] > 0.5) & (df['od_direction'] == 'with_gap')]),
        ('L1: + QH', lambda df: df[(df['od_strength'] > 0.5) & (df['od_direction'] == 'with_gap') & (df['quality_high'] == True)]),
        ('L2d: +RV30>5', lambda df: df[(df['od_strength'] > 0.5) & (df['od_direction'] == 'with_gap') & (df['quality_high'] == True) & (df['rvol_open_30min'] >= 5)]),
        ('L2e: +PM30<10', lambda df: df[(df['od_strength'] > 0.5) & (df['od_direction'] == 'with_gap') & (df['quality_high'] == True) & (df['pm_rth30_computed'] < 0.10)]),
        ('L3b: +RV30>5+PM30<10', lambda df: df[(df['od_strength'] > 0.5) & (df['od_direction'] == 'with_gap') & (df['quality_high'] == True) & (df['rvol_open_30min'] >= 5) & (df['pm_rth30_computed'] < 0.10)]),
    ]

    for gap_dir, gap_label in [('up', 'GapUp'), ('down', 'GapDown')]:
        lines.append(f"\n--- {gap_label} ---")
        for sl_type, sl_label in [('sl_half', 'SL_HALF'), ('sl_fix025', 'SL_FIX025')]:
            lines.append(f"\n  {sl_label}:")
            for half_name, half_df in [('IS', h1), ('OOS', h2)]:
                for llabel, lfn in layer_configs:
                    sub = lfn(half_df[half_df['gap_direction'] == gap_dir])
                    trades = run_simulation(sub, sl_type)
                    lines.append(summarize_rr(trades, f"{half_name} {llabel}"))
                lines.append("")

    # ===== 6f: EV-Kurven IS vs OOS =====
    lines.append(f"\n{'='*70}")
    lines.append("6f: EV-KURVEN IS vs OOS (1-sec Zoom)")
    lines.append(f"{'='*70}")

    for gap_dir, gap_label in [('up', 'GapUp'), ('down', 'GapDown')]:
        lines.append(f"\n--- {gap_label} ---")
        for half_name, half_df in [('IS', h1), ('OOS', h2)]:
            qh_with = half_df[(half_df['od_strength'] > 0.5) & (half_df['od_direction'] == 'with_gap') &
                              (half_df['gap_direction'] == gap_dir) & (half_df['quality_high'] == True)]
            trades = run_simulation(qh_with, 'sl_half')
            lines.append(ev_curve(trades, f"{half_name} L1 QH"))

            sub = qh_with[qh_with['rvol_open_30min'] >= 5]
            trades = run_simulation(sub, 'sl_half')
            lines.append(ev_curve(trades, f"{half_name} L2d RV30>5"))
        lines.append("")

    # ===== 6g: Shrinkage Table =====
    lines.append(f"\n{'='*70}")
    lines.append("6g: SHRINKAGE-TABELLE IS â†’ OOS (1-sec Zoom)")
    lines.append(f"{'='*70}")
    lines.append(f"\n  {'Setup':<26} | {'IS WR@2R':>8} | {'OOS WR@2R':>9} | {'Shrink':>8} | Repliziert?")
    lines.append(f"  {'-'*70}")

    shrinkage_setups = [
        ('L0: OD>0.5 with', lambda df: df[(df['od_strength'] > 0.5) & (df['od_direction'] == 'with_gap')]),
        ('L1: + QH', lambda df: df[(df['od_strength'] > 0.5) & (df['od_direction'] == 'with_gap') & (df['quality_high'] == True)]),
        ('L2d: +RV30>5', lambda df: df[(df['od_strength'] > 0.5) & (df['od_direction'] == 'with_gap') & (df['quality_high'] == True) & (df['rvol_open_30min'] >= 5)]),
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
                lines.append(f"  {llabel:<26} | {is_wr2:>7.1%} | {oos_wr2:>8.1%} | {shrink:>7.1f}% | {repl}")
            else:
                lines.append(f"  {llabel:<26} | N<10")

    # Write main output
    output = "\n".join(lines)
    os.makedirs("results", exist_ok=True)
    with open('results/d9s_rr_basis.txt', 'w', encoding='utf-8') as f:
        f.write(output)
    print(output)

    # ===== SYNTHESE =====
    synth = []
    synth.append("=" * 70)
    synth.append("D9s SYNTHESE: R:R MIT 1-SEC AUFLOESUNG")
    synth.append("=" * 70)
    synth.append("\nMethodik: Hybrid 1-min/1-sec Simulation")
    synth.append("  - Basis-Iteration: 1-min Bars (wie D9)")
    synth.append("  - Zoom: Bei ambigen Bars (SL + Target moeglich), 1-sec Aufloesung 09:30-09:45")
    synth.append("  - Ausserhalb 1-sec-Fenster: 1-min konservativ (SL first)")
    synth.append("")
    synth.append("ZOOM-STATISTIK:")

    # Compute zoom stats for synthesis
    for label, half_df in [('IS', h1), ('OOS', h2)]:
        base = half_df[(half_df['od_strength'] > 0.5) & (half_df['od_direction'] == 'with_gap')]
        for sl_type, sl_label in [('sl_fix025', 'SL_FIX025')]:
            trades = run_simulation(base, sl_type)
            n = len(trades)
            if n < 10:
                continue
            n_zoom = trades['zoom_used'].sum()
            n_changed = trades['zoom_changed'].sum()
            synth.append(f"  {label} {sl_label}: Zoom={n_zoom}/{n} ({100*n_zoom/n:.1f}%), Changed={n_changed}/{n} ({100*n_changed/n:.1f}%)")

    synth.append("")
    synth.append("KERNFRAGE: Aendert die 1-sec Aufloesung die D9-Ergebnisse signifikant?")
    synth.append("")
    synth.append("ANTWORT: Siehe Delta-Analyse in d_synthese_1sec.txt")

    synth_output = "\n".join(synth)
    with open('results/d9s_synthese.txt', 'w', encoding='utf-8') as f:
        f.write(synth_output)

    print(f"\nSaved to results/d9s_rr_basis.txt + d9s_synthese.txt", file=sys.stderr)


if __name__ == '__main__':
    main()
