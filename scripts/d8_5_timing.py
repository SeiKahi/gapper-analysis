"""
D8.5 Aufgabe 5: 9:35-Actionable vs 10:00-Actionable (IS)
Compare what you can see at 9:35 vs what you know at 10:00.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product


def simulate_trade(bars_rth, entry_price, adr, direction, entry_time='09:35', sl_adr=0.25):
    """Simulate trade from given entry_time."""
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

    post_entry = bars_rth[bars_rth['time_et'] >= entry_time]

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
    """Run simulation for a batch, with optional 10:00 entry."""
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
        if pd.isna(adr) or adr <= 0:
            continue

        # Determine entry price
        if entry_time == '09:35':
            entry_price = row['close_935']
        else:
            # Use close_1000 from metadata or from 1min data
            entry_price = row.get('close_1000', np.nan)
            if pd.isna(entry_price):
                bar_10 = rth[rth['time_et'] == '10:00']
                if len(bar_10) > 0:
                    entry_price = bar_10.iloc[0]['close']
                else:
                    continue

        if pd.isna(entry_price):
            continue

        # Determine direction: OD with_gap = trade in gap direction
        od_dir = row['od_direction']
        gap_dir = row['gap_direction']
        if od_dir == 'with_gap':
            direction = 'long' if gap_dir == 'up' else 'short'
        else:
            direction = 'short' if gap_dir == 'up' else 'long'

        trade = simulate_trade(rth, entry_price, adr, direction, entry_time)
        trade['ticker'] = ticker
        trade['date'] = date
        trade['gap_direction'] = gap_dir
        results.append(trade)

    return pd.DataFrame(results)


def summarize(trades, label):
    n = len(trades)
    if n < 10:
        return f"  {label:<55} | N={n:>3} [N<10]"
    wr25 = trades['hit_0.25'].mean()
    wr50 = trades['hit_0.5'].mean()
    sl = trades['sl_hit'].mean()
    ev = wr50 * 0.50 - sl * 0.25
    low = " [LOW N]" if n < 20 else ""
    return f"  {label:<55} | N={n:>3} | WR@0.25={wr25:>5.1%} | WR@0.50={wr50:>5.1%} | EV@0.50={ev:>+.3f} | SL%={sl:>5.1%}{low}"


def main():
    lines = []
    lines.append("=" * 70)
    lines.append("D8.5 AUFGABE 5: 9:35-ACTIONABLE vs 10:00-ACTIONABLE (IS)")
    lines.append("=" * 70)

    meta = pd.read_parquet('data/metadata/metadata_v8_5.parquet')
    h1 = meta[(meta['date'] >= '2021-02-21') & (meta['date'] <= '2023-12-31')].copy()
    h1 = h1[h1['od_body_pct'].notna()].copy()
    h1['gap_abs'] = h1['gap_size_in_adr'].abs()

    # Only WITH GAP (continuation), since against-gap was negative EV
    base_with = h1[(h1['od_strength'] > 0.5) & (h1['od_direction'] == 'with_gap') & (h1['quality_high'] == True)]

    # ===== 5a: 9:35-Parameter (sofort verfuegbar) =====
    lines.append(f"\n\n{'='*70}")
    lines.append("5a: NUR 9:35-PARAMETER (sofort verfuegbar)")
    lines.append(f"{'='*70}")
    lines.append("\nBasis: OD > 0.5 + QUALITY_HIGH + with_gap")
    lines.append("Zusaetzliche 9:35-Filter: RVOL_5, PM/RTH5, Gap_in_ADR")

    combos_935 = [
        ('Basis (QH + OD>0.5 with)', {}),
        ('+ RVOL_5 > 5x', {'rvol_5': ('>', 5)}),
        ('+ RVOL_5 > 3x', {'rvol_5': ('>', 3)}),
        ('+ Gap > 2 ADR', {'gap_abs': ('>', 2)}),
        ('+ Gap > 1 ADR', {'gap_abs': ('>', 1)}),
        ('+ PM/RTH5 < 50%', {'pm_rth5': ('<', 0.50)}),
        ('+ PM/RTH5 < 100%', {'pm_rth5': ('<', 1.00)}),
        ('+ RVOL_5>5x + Gap>2', {'rvol_5': ('>', 5), 'gap_abs': ('>', 2)}),
        ('+ RVOL_5>3x + PM/RTH5<50%', {'rvol_5': ('>', 3), 'pm_rth5': ('<', 0.50)}),
        ('+ Gap>2 + PM/RTH5<50%', {'gap_abs': ('>', 2), 'pm_rth5': ('<', 0.50)}),
        ('+ RVOL_5>5x + Gap>2 + PM/RTH5<50%', {'rvol_5': ('>', 5), 'gap_abs': ('>', 2), 'pm_rth5': ('<', 0.50)}),
    ]

    for gap_dir, gap_label in [('up', 'GapUp'), ('down', 'GapDown')]:
        lines.append(f"\n--- {gap_label} ---")
        base = base_with[base_with['gap_direction'] == gap_dir]

        for combo_label, filters in combos_935:
            sub = base.copy()
            for col, (op, val) in filters.items():
                if op == '>':
                    sub = sub[sub[col] > val]
                elif op == '<':
                    sub = sub[sub[col] < val]

            print(f"  9:35 {gap_label} {combo_label}: N={len(sub)}")
            if len(sub) < 10:
                lines.append(f"  {combo_label:<55} | N={len(sub):>3} [N<10]")
                continue

            trades = run_sim_batch(sub, entry_time='09:35')
            lines.append(summarize(trades, combo_label))

    # ===== 5b: 10:00-Parameter (muss warten) =====
    lines.append(f"\n\n{'='*70}")
    lines.append("5b: 10:00-PARAMETER (erst um 10:00 bekannt)")
    lines.append(f"{'='*70}")
    lines.append("\nBasis: OD > 0.5 + QUALITY_HIGH + with_gap")
    lines.append("Zusaetzliche 10:00-Filter: RVOL_30, PM/RTH30")
    lines.append("Entry um 10:00 (NICHT 9:35!)")

    combos_1000 = [
        ('Basis (QH + OD>0.5 with) @ 10:00', {}),
        ('+ RVOL_30 > 5x', {'rvol_open_30min': ('>', 5)}),
        ('+ RVOL_30 > 3x', {'rvol_open_30min': ('>', 3)}),
        ('+ PM/RTH30 < 10%', {'pm_rth30_computed': ('<', 0.10)}),
        ('+ PM/RTH30 < 30%', {'pm_rth30_computed': ('<', 0.30)}),
        ('+ RVOL_30>5x + PM/RTH30<10%', {'rvol_open_30min': ('>', 5), 'pm_rth30_computed': ('<', 0.10)}),
        ('+ RVOL_30>3x + PM/RTH30<10%', {'rvol_open_30min': ('>', 3), 'pm_rth30_computed': ('<', 0.10)}),
    ]

    for gap_dir, gap_label in [('up', 'GapUp'), ('down', 'GapDown')]:
        lines.append(f"\n--- {gap_label} ---")
        base = base_with[base_with['gap_direction'] == gap_dir]

        for combo_label, filters in combos_1000:
            sub = base.copy()
            for col, (op, val) in filters.items():
                if op == '>':
                    sub = sub[sub[col] > val]
                elif op == '<':
                    sub = sub[sub[col] < val]

            print(f"  10:00 {gap_label} {combo_label}: N={len(sub)}")
            if len(sub) < 10:
                lines.append(f"  {combo_label:<55} | N={len(sub):>3} [N<10]")
                continue

            trades = run_sim_batch(sub, entry_time='10:00')
            lines.append(summarize(trades, combo_label))

    # ===== 5c: Inkrementeller Wert des Wartens =====
    lines.append(f"\n\n{'='*70}")
    lines.append("5c: INKREMENTELLER WERT DES WARTENS")
    lines.append(f"{'='*70}")
    lines.append("\nVergleich: Gleicher Filter-Set, Entry 9:35 vs 10:00")

    # Run same PM/RTH30<10% filter but compare entry times
    for gap_dir, gap_label in [('up', 'GapUp'), ('down', 'GapDown')]:
        lines.append(f"\n--- {gap_label} ---")
        base = base_with[base_with['gap_direction'] == gap_dir]

        # Best IS combo from 5a: we'll use basis + PM/RTH30<10%
        sub_pm = base[base['pm_rth30_computed'] < 0.10]

        if len(sub_pm) >= 10:
            trades_935 = run_sim_batch(sub_pm, '09:35')
            trades_1000 = run_sim_batch(sub_pm, '10:00')

            if len(trades_935) >= 10 and len(trades_1000) >= 10:
                wr50_935 = trades_935['hit_0.5'].mean()
                wr50_1000 = trades_1000['hit_0.5'].mean()
                sl_935 = trades_935['sl_hit'].mean()
                sl_1000 = trades_1000['sl_hit'].mean()
                ev_935 = wr50_935 * 0.50 - sl_935 * 0.25
                ev_1000 = wr50_1000 * 0.50 - sl_1000 * 0.25

                lines.append(f"  PM/RTH30<10% Filter (N_935={len(trades_935)}, N_1000={len(trades_1000)}):")
                lines.append(f"    Entry 9:35:  WR@0.50={wr50_935:.1%}, EV@0.50={ev_935:+.3f}, SL%={sl_935:.1%}")
                lines.append(f"    Entry 10:00: WR@0.50={wr50_1000:.1%}, EV@0.50={ev_1000:+.3f}, SL%={sl_1000:.1%}")
                lines.append(f"    Delta WR:  {wr50_1000 - wr50_935:+.1%}")
                lines.append(f"    Delta EV:  {ev_1000 - ev_935:+.3f}")

        # Also compare basis (no additional filter) at both times
        if len(base) >= 10:
            trades_935_base = run_sim_batch(base, '09:35')
            trades_1000_base = run_sim_batch(base, '10:00')

            if len(trades_935_base) >= 10 and len(trades_1000_base) >= 10:
                wr50_935 = trades_935_base['hit_0.5'].mean()
                wr50_1000 = trades_1000_base['hit_0.5'].mean()
                sl_935 = trades_935_base['sl_hit'].mean()
                sl_1000 = trades_1000_base['sl_hit'].mean()
                ev_935 = wr50_935 * 0.50 - sl_935 * 0.25
                ev_1000 = wr50_1000 * 0.50 - sl_1000 * 0.25

                # Missed drift: mean move from 9:35 to 10:00
                missed = base['rest_drift'].mean() - base['rest_drift_1000'].mean() if 'rest_drift_1000' in base.columns else np.nan

                lines.append(f"\n  Basis (QH + OD>0.5 with, N={len(base)}):")
                lines.append(f"    Entry 9:35:  WR@0.50={wr50_935:.1%}, EV@0.50={ev_935:+.3f}")
                lines.append(f"    Entry 10:00: WR@0.50={wr50_1000:.1%}, EV@0.50={ev_1000:+.3f}")
                lines.append(f"    Delta WR:  {wr50_1000 - wr50_935:+.1%}")
                lines.append(f"    Delta EV:  {ev_1000 - ev_935:+.3f}")
                if pd.notna(missed):
                    lines.append(f"    Missed drift (9:35->10:00): {missed:+.3f} ADR")

    lines.append(f"\n\nFazit: Lohnt sich Warten bis 10:00 wenn Reversal-Filter bestanden?")
    lines.append("(Antwort basiert auf Delta WR und Delta EV oben)")

    output = "\n".join(lines)
    with open('results/d8_5_timing.txt', 'w', encoding='utf-8') as f:
        f.write(output)
    print(output)
    print(f"\nSaved to results/d8_5_timing.txt")


if __name__ == '__main__':
    main()
