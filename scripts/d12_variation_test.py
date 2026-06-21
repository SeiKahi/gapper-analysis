"""
D12: Trade Setup Variation Test (IS Only)
Tests 7 variations of OD-direction, entry type, and SL type.
All use TRAIL_D exit. IS period: 2021-02-21 to 2023-12-31.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

RAW_DIR = Path('data/raw_1min')
IS_START = '2021-02-21'
IS_END = '2023-12-31'


def simulate_trail_d(bars_post_entry, entry_price, sl_dist, trade_dir, adr):
    """
    TRAIL_D simulation: +1R → BE, then trail = peak - 0.5R.
    Timeout at 15:55.
    Returns dict with PnL and exit info, or None.
    """
    if sl_dist <= 0 or pd.isna(sl_dist) or adr <= 0 or pd.isna(adr):
        return None
    if len(bars_post_entry) == 0:
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

    for _, bar in bars_post_entry.iterrows():
        if trade_dir == 'long':
            current_r = max(0, (bar['high'] - entry_price) / sl_dist)
        else:
            current_r = max(0, (entry_price - bar['low']) / sl_dist)

        highest_r = max(highest_r, current_r)

        # TRAIL_D: +1R → BE, trail = peak - 0.5R
        if highest_r >= 1.0 and not trail_active:
            trail_active = True
            sl_level = entry_price  # BE
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
        last_bar = bars_post_entry.iloc[-1]
        exit_price = last_bar['close']
        exit_type = 'timeout'

    pnl = (exit_price - entry_price) if trade_dir == 'long' else (entry_price - exit_price)

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


def find_breakout_entry(bars_rth, close_935, trade_dir):
    """
    Scan 1-min bars 09:36-10:00 for breakout past close_935.
    Long: first bar with high > close_935 → entry at close_935
    Short: first bar with low < close_935 → entry at close_935
    Returns (entry_bar_idx, hit_sl_same_bar: bool) or None if no breakout.
    """
    scan_bars = bars_rth[
        (bars_rth['time_et'] >= '09:36') & (bars_rth['time_et'] <= '10:00')
    ]
    for idx, bar in scan_bars.iterrows():
        if trade_dir == 'long' and bar['high'] > close_935:
            return idx, bar
        elif trade_dir == 'short' and bar['low'] < close_935:
            return idx, bar
    return None


def run_variation(meta_is, variation_name):
    """
    Run one variation. Returns list of trade dicts.
    Variation names: A1, A2, A3, B, C1, C2, C3
    """
    results = []
    skip_no_breakout = 0
    skip_entry_bar_sl = 0
    skip_sl_dist_invalid = 0
    skip_od_long_nan = 0
    skip_data_missing = 0

    for _, row in meta_is.iterrows():
        ticker = row['ticker']
        date = str(row['date'])
        close_935 = row['close_935']
        adr = row['adr_10']
        od_long = row['od_long']
        gap_dir = row['gap_direction']

        # Common skips
        if pd.isna(close_935) or pd.isna(adr) or adr <= 0:
            skip_data_missing += 1
            continue

        # od_long determines trade direction for all variations
        if pd.isna(od_long):
            skip_od_long_nan += 1
            continue

        # Trade direction logic
        if variation_name in ('A1', 'A2', 'A3', 'B'):
            # Long wenn od_long, Short wenn !od_long
            trade_dir = 'long' if od_long else 'short'
        else:
            # C1, C2, C3: Long wenn od_long (only od_long==True trades)
            if not od_long:
                continue
            trade_dir = 'long'

        # Load 1-min data
        fpath = RAW_DIR / ticker / f"{date}.parquet"
        if not fpath.exists():
            skip_data_missing += 1
            continue
        bars = pd.read_parquet(fpath)
        rth = bars[bars['session'] == 'rth'].copy()
        if len(rth) < 10:
            skip_data_missing += 1
            continue

        # Entry + SL logic
        if variation_name in ('A1', 'A2', 'A3'):
            # Baseline: entry = close_935, SL = 0.25 ADR fix
            entry_price = close_935
            sl_dist = 0.25 * adr
            # Post-entry bars: 09:36 onwards
            post_entry = rth[
                (rth['time_et'] >= '09:36') & (rth['time_et'] <= '15:55')
            ]
        else:
            # Breakout entry (B, C1, C2, C3)
            od_low5 = row['od_low5']
            od_high5 = row['od_high5']
            if pd.isna(od_low5) or pd.isna(od_high5):
                skip_data_missing += 1
                continue

            bo = find_breakout_entry(rth, close_935, trade_dir)
            if bo is None:
                skip_no_breakout += 1
                continue

            bo_idx, bo_bar = bo
            entry_price = close_935

            # OD-based SL
            if trade_dir == 'long':
                sl_level = od_low5 - 0.01
                sl_dist = entry_price - sl_level
            else:
                sl_level = od_high5 + 0.01
                sl_dist = sl_level - entry_price

            if sl_dist <= 0:
                skip_sl_dist_invalid += 1
                continue

            # Check if breakout bar itself hits SL (conservative: SL first → -1R)
            if trade_dir == 'long' and bo_bar['low'] <= sl_level:
                skip_entry_bar_sl += 1
                results.append({
                    'ticker': ticker,
                    'date': date,
                    'gap_direction': gap_dir,
                    'trade_dir': trade_dir,
                    'entry_price': entry_price,
                    'sl_dist': sl_dist,
                    'sl_dist_adr': sl_dist / adr,
                    'pnl': -sl_dist,
                    'pnl_r': -1.0,
                    'pnl_adr': -sl_dist / adr,
                    'exit_type': 'entry_bar_sl',
                    'exit_time': bo_bar['time_et'],
                    'highest_r': 0.0,
                    'trail_active': False,
                    'winner': False,
                })
                continue
            elif trade_dir == 'short' and bo_bar['high'] >= sl_level:
                skip_entry_bar_sl += 1
                results.append({
                    'ticker': ticker,
                    'date': date,
                    'gap_direction': gap_dir,
                    'trade_dir': trade_dir,
                    'entry_price': entry_price,
                    'sl_dist': sl_dist,
                    'sl_dist_adr': sl_dist / adr,
                    'pnl': -sl_dist,
                    'pnl_r': -1.0,
                    'pnl_adr': -sl_dist / adr,
                    'exit_type': 'entry_bar_sl',
                    'exit_time': bo_bar['time_et'],
                    'highest_r': 0.0,
                    'trail_active': False,
                    'winner': False,
                })
                continue

            # Post-entry: bars after breakout bar until 15:55
            # Get numeric position of breakout bar and take next bars
            bo_pos = rth.index.get_loc(bo_idx)
            post_entry = rth.iloc[bo_pos + 1:]
            post_entry = post_entry[post_entry['time_et'] <= '15:55']

        if len(post_entry) == 0:
            skip_data_missing += 1
            continue

        # Simulate TRAIL_D
        res = simulate_trail_d(post_entry, entry_price, sl_dist, trade_dir, adr)
        if res is None:
            skip_data_missing += 1
            continue

        res['ticker'] = ticker
        res['date'] = date
        res['gap_direction'] = gap_dir
        res['trade_dir'] = trade_dir
        res['entry_price'] = entry_price
        res['sl_dist'] = sl_dist
        res['sl_dist_adr'] = sl_dist / adr
        results.append(res)

    return results, {
        'skip_no_breakout': skip_no_breakout,
        'skip_entry_bar_sl': skip_entry_bar_sl,
        'skip_sl_dist_invalid': skip_sl_dist_invalid,
        'skip_od_long_nan': skip_od_long_nan,
        'skip_data_missing': skip_data_missing,
    }


def summarize(trades_df, label, show_sl_dist=False):
    """Stats line: N, WR, EV(ADR), Median PnL(R), SL-Rate, Exit-Breakdown."""
    n = len(trades_df)
    if n == 0:
        return f"  {label:<28s} |  N={0:>4d}  [NO DATA]"
    if n < 10:
        return f"  {label:<28s} |  N={n:>4d}  [LOW N]"

    med_pnl_r = trades_df['pnl_r'].median()
    wr = trades_df['winner'].mean()
    ev_adr = trades_df['pnl_adr'].mean()

    n_init = (trades_df['exit_type'] == 'initial_sl').sum()
    n_trail = (trades_df['exit_type'] == 'trail_sl').sum()
    n_to = (trades_df['exit_type'] == 'timeout').sum()
    n_entry_sl = (trades_df['exit_type'] == 'entry_bar_sl').sum()

    sl_pct = (n_init + n_entry_sl) / n * 100

    low_n = " [LOW N]" if n < 20 else ""

    line = (f"  {label:<28s} | {n:>4d} | {med_pnl_r:>+6.2f}R | "
            f"{wr:>5.1%} | {ev_adr:>+6.3f} | {sl_pct:>5.1f}% | "
            f"{n_init:>3d}({100*n_init/n:>4.0f}%) | "
            f"{n_trail:>3d}({100*n_trail/n:>4.0f}%) | "
            f"{n_to:>3d}({100*n_to/n:>4.0f}%)")

    if n_entry_sl > 0:
        line += f" | EntryBarSL={n_entry_sl}"

    if show_sl_dist and 'sl_dist_adr' in trades_df.columns:
        med_sl = trades_df['sl_dist_adr'].median()
        line += f" | MedSL={med_sl:.3f}ADR"

    line += low_n
    return line


def main():
    lines = []

    def w(text=""):
        lines.append(text)

    w("=" * 70)
    w("D12: TRADE SETUP VARIATION TEST (IS ONLY)")
    w("=" * 70)
    w(f"IS Period: {IS_START} to {IS_END}")
    w("Trail: TRAIL_D (+1R->BE, trail peak-0.5R) | Timeout: 15:55")
    w()

    # Load metadata
    meta = pd.read_parquet('data/metadata/metadata_v9.parquet')
    h1 = meta[(meta['date'] >= IS_START) & (meta['date'] <= IS_END)].copy()
    print(f"IS total: {len(h1)}", file=sys.stderr)

    # Prepare filter subsets
    od_with = h1[(h1['od_strength'] > 0.5) & (h1['od_direction'] == 'with_gap')].copy()
    od_against = h1[(h1['od_strength'] > 0.5) & (h1['od_direction'] == 'against_gap')].copy()
    od_all = h1[h1['od_strength'] > 0.5].copy()

    w(f"Filters: OD>0.5+with_gap N={len(od_with)}, OD>0.5+against_gap N={len(od_against)}, OD>0.5 all N={len(od_all)}")
    w()

    # ===================================================================
    # Define variations
    # ===================================================================
    variations = {
        'A1': ('with_gap (Baseline)', od_with),
        'A2': ('against_gap', od_against),
        'A3': ('all OD>0.5', od_all),
        'B':  ('with_gap Breakout+OD-SL', od_with),
        'C1': ('with_gap BO Long-only', od_with),
        'C2': ('against_gap BO Long-only', od_against),
        'C3': ('all OD>0.5 BO Long-only', od_all),
    }

    all_trades = {}
    all_skips = {}

    for var_name, (var_label, var_data) in variations.items():
        print(f"Running {var_name}: {var_label} (N={len(var_data)})...", file=sys.stderr)
        trades, skips = run_variation(var_data, var_name)
        all_trades[var_name] = pd.DataFrame(trades) if trades else pd.DataFrame()
        all_skips[var_name] = skips
        print(f"  → {len(trades)} trades, skips: {skips}", file=sys.stderr)

    # ===================================================================
    # VARIATION A: OD-Richtung
    # ===================================================================
    header_line = (f"  {'Variation':<28s} | {'N':>4s} | {'MedPnL':>6s} | "
                   f"{'WR':>5s} | {'EV_ADR':>6s} | {'SL%':>5s} | "
                   f"{'InitSL':>10s} | {'TrailSL':>10s} | {'Timeout':>10s}")
    sep_line = "  " + "-" * 105

    w("=" * 70)
    w("VARIATION A: OD-RICHTUNG (Baseline Entry)")
    w("=" * 70)
    w("Entry: close_935 @09:35 | SL: 0.25 ADR fix")
    w("Trade-Richtung: Long wenn od_long, Short wenn !od_long")
    w()

    for gap_dir, gap_label in [('up', 'GapUp'), ('down', 'GapDn')]:
        w(f"--- {gap_label} ---")
        w(header_line)
        w(sep_line)
        for var_name in ['A1', 'A2', 'A3']:
            df = all_trades[var_name]
            label = f"{var_name} {variations[var_name][0]}"
            if len(df) > 0:
                sub = df[df['gap_direction'] == gap_dir]
            else:
                sub = df
            w(summarize(sub, label))
        w()

    # Verification: A3 ⊇ A1 ∪ A2
    n_a1 = len(all_trades['A1'])
    n_a2 = len(all_trades['A2'])
    n_a3 = len(all_trades['A3'])
    check = "OK" if n_a3 == n_a1 + n_a2 else f"MISMATCH (A1={n_a1}+A2={n_a2}={n_a1+n_a2} vs A3={n_a3})"
    w(f"  Verifikation: N(A3)={n_a3} == N(A1)+N(A2)={n_a1}+{n_a2}={n_a1+n_a2} → {check}")
    w()

    # ===================================================================
    # VARIATION B/C: Breakout Entry
    # ===================================================================
    w("=" * 70)
    w("VARIATION B/C: BREAKOUT ENTRY + OD-BASIERTER SL")
    w("=" * 70)
    w("Entry: Breakout close_935 (09:36-10:00)")
    w("SL Long: od_low5 - 0.01 | SL Short: od_high5 + 0.01")
    w("B: Long wenn od_long, Short wenn !od_long")
    w("C1-C3: Long wenn od_long (Long-only)")
    w()

    for gap_dir, gap_label in [('up', 'GapUp'), ('down', 'GapDn')]:
        w(f"--- {gap_label} ---")
        w(header_line)
        w(sep_line)
        for var_name in ['B', 'C1', 'C2', 'C3']:
            df = all_trades[var_name]
            label = f"{var_name} {variations[var_name][0]}"
            if len(df) > 0:
                sub = df[df['gap_direction'] == gap_dir]
            else:
                sub = df
            w(summarize(sub, label, show_sl_dist=True))
        w()

    # Skip statistics for B/C
    w("  --- Skip-Statistik (Breakout-Variationen) ---")
    for var_name in ['B', 'C1', 'C2', 'C3']:
        sk = all_skips[var_name]
        total_input = len(variations[var_name][1])
        n_trades = len(all_trades[var_name])
        no_bo = sk['skip_no_breakout']
        entry_sl = sk['skip_entry_bar_sl']
        sl_inv = sk['skip_sl_dist_invalid']
        data_miss = sk['skip_data_missing']
        skip_rate = no_bo / max(total_input, 1) * 100
        w(f"  {var_name}: Input={total_input}, Trades={n_trades}, "
          f"NoBreakout={no_bo} ({skip_rate:.1f}%), "
          f"EntryBarSL={entry_sl}, SL_Invalid={sl_inv}, DataMiss={data_miss}")
        if skip_rate > 50:
            w(f"    WARNING: Skip-Rate >50% -- Variation fragwuerdig!")
    w()

    # SL-Distanz Vergleich
    w("  --- SL-Distanz Vergleich (Breakout vs Baseline) ---")
    for var_name in ['A1', 'B', 'C1']:
        df = all_trades[var_name]
        if len(df) > 0 and 'sl_dist_adr' in df.columns:
            med_sl = df['sl_dist_adr'].median()
            mean_sl = df['sl_dist_adr'].mean()
            w(f"  {var_name}: Median SL = {med_sl:.3f} ADR, Mean SL = {mean_sl:.3f} ADR (N={len(df)})")
    w()

    # ===================================================================
    # VERGLEICHSTABELLE
    # ===================================================================
    w("=" * 70)
    w("VERGLEICHSTABELLE (alle 7 Variationen)")
    w("=" * 70)
    w()

    comp_header = (f"  {'Var':<5s} {'Beschreibung':<28s} | {'Gap':>4s} | {'N':>4s} | "
                   f"{'MedR':>5s} | {'WR%':>5s} | {'EV_ADR':>7s} | {'SL%':>5s}")
    comp_sep = "  " + "-" * 80

    w(comp_header)
    w(comp_sep)

    for gap_dir, gap_label in [('up', 'Up'), ('down', 'Dn')]:
        for var_name in ['A1', 'A2', 'A3', 'B', 'C1', 'C2', 'C3']:
            df = all_trades[var_name]
            desc = variations[var_name][0]
            if len(df) > 0:
                sub = df[df['gap_direction'] == gap_dir]
            else:
                sub = pd.DataFrame()
            n = len(sub)
            if n < 10:
                low = " [LOW N]" if 0 < n < 10 else ""
                w(f"  {var_name:<5s} {desc:<28s} | {gap_label:>4s} | {n:>4d} | {'---':>5s} | {'---':>5s} | {'---':>7s} | {'---':>5s}{low}")
            else:
                med_r = sub['pnl_r'].median()
                wr = sub['winner'].mean() * 100
                ev = sub['pnl_adr'].mean()
                n_init = (sub['exit_type'] == 'initial_sl').sum()
                n_entry_sl = (sub['exit_type'] == 'entry_bar_sl').sum()
                sl_pct = (n_init + n_entry_sl) / n * 100
                w(f"  {var_name:<5s} {desc:<28s} | {gap_label:>4s} | {n:>4d} | {med_r:>+5.2f} | {wr:>5.1f} | {ev:>+7.3f} | {sl_pct:>5.1f}")
        w(comp_sep)

    w()

    # ===================================================================
    # Baseline-Verifikation
    # ===================================================================
    w("=" * 70)
    w("BASELINE-VERIFIKATION (A1 vs d10_trailing.py TRAIL_D)")
    w("=" * 70)
    w("Erwartung: A1 GapUp ~N=205, EV~+0.149 ADR, WR~55.6%")
    w("           A1 GapDn ~N=181, EV~+0.026 ADR, WR~54.1%")
    w("(Kleine Abweichungen moeglich: metadata_v9 vs v8_5, trade_dir Logik)")
    w()
    a1 = all_trades['A1']
    if len(a1) > 0:
        for gap_dir, gap_label in [('up', 'GapUp'), ('down', 'GapDn')]:
            sub = a1[a1['gap_direction'] == gap_dir]
            n = len(sub)
            if n >= 10:
                wr = sub['winner'].mean() * 100
                ev = sub['pnl_adr'].mean()
                w(f"  A1 {gap_label}: N={n}, WR={wr:.1f}%, EV={ev:+.3f} ADR")
            else:
                w(f"  A1 {gap_label}: N={n} [LOW N]")
    w()

    # ===================================================================
    # Save outputs
    # ===================================================================
    os.makedirs('results', exist_ok=True)
    output_txt = "\n".join(lines)

    with open('results/d12_variation_test.txt', 'w', encoding='utf-8') as f:
        f.write(output_txt + "\n")

    # Save raw trades parquet
    all_raw = []
    for var_name, df in all_trades.items():
        if len(df) > 0:
            df_copy = df.copy()
            df_copy['variation'] = var_name
            all_raw.append(df_copy)
    if all_raw:
        raw_df = pd.concat(all_raw, ignore_index=True)
        raw_df.to_parquet('results/d12_variation_raw.parquet', index=False)
        print(f"Saved {len(raw_df)} raw trades to results/d12_variation_raw.parquet", file=sys.stderr)

    sys.stdout.buffer.write((output_txt + "\n").encode('utf-8'))
    print(f"\nSaved to results/d12_variation_test.txt", file=sys.stderr)


if __name__ == '__main__':
    main()
