"""
D8.0 Aufgabe 5: Post-Reversal Verhalten & Trade-Simulation (IS)
================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
META_DIR = DATA_DIR / "metadata"
RAW_DIR = DATA_DIR / "raw_1min"
RESULTS_DIR = BASE_DIR / "results"
IS_END = "2023-12-31"


def load_1min_data(ticker, date_str):
    path = RAW_DIR / ticker / f"{date_str}.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        if 'time_et' in df.columns:
            df = df.sort_values('time_et').reset_index(drop=True)
        elif 'datetime_et' in df.columns:
            df['time_et'] = pd.to_datetime(df['datetime_et']).dt.strftime('%H:%M')
            df = df.sort_values('time_et').reset_index(drop=True)
        return df
    except Exception:
        return None


def simulate_trade(bars, entry_time, direction, entry_price, adr, sl_adr, target_adr, timeout_time='15:55'):
    """
    Simulate a trade bar-by-bar.
    direction: 'long' or 'short'
    Returns: outcome ('target', 'sl', 'timeout'), pnl_adr, exit_time
    """
    sl_price = entry_price - sl_adr * adr if direction == 'long' else entry_price + sl_adr * adr
    target_price = entry_price + target_adr * adr if direction == 'long' else entry_price - target_adr * adr

    trade_bars = bars[(bars['time_et'] > entry_time) & (bars['time_et'] <= timeout_time)]
    if len(trade_bars) == 0:
        return 'no_data', 0, entry_time

    for _, bar in trade_bars.iterrows():
        if direction == 'long':
            # Check SL first (conservative)
            if bar['low'] <= sl_price:
                pnl = (sl_price - entry_price) / adr
                return 'sl', pnl, bar['time_et']
            # Check target
            if bar['high'] >= target_price:
                pnl = (target_price - entry_price) / adr
                return 'target', pnl, bar['time_et']
        else:  # short
            if bar['high'] >= sl_price:
                pnl = (entry_price - sl_price) / adr
                return 'sl', pnl, bar['time_et']
            if bar['low'] <= target_price:
                pnl = (entry_price - target_price) / adr
                return 'target', pnl, bar['time_et']

    # Timeout
    last_price = trade_bars.iloc[-1]['close']
    if direction == 'long':
        pnl = (last_price - entry_price) / adr
    else:
        pnl = (entry_price - last_price) / adr
    return 'timeout', pnl, trade_bars.iloc[-1]['time_et']


def bootstrap_mean_ci(data, n_boot=1000):
    data = np.array([x for x in data if not np.isnan(x)])
    if len(data) < 5:
        return np.nan, np.nan
    rng = np.random.RandomState(42)
    means = [rng.choice(data, len(data), replace=True).mean() for _ in range(n_boot)]
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def main():
    # Load reversal raw data from Aufgabe 4
    rev_path = RESULTS_DIR / "d8_4_reversal_raw.parquet"
    if not rev_path.exists():
        print("ERROR: Run d8_reversal_predict.py first!", file=sys.stderr)
        return

    rev_df = pd.read_parquet(rev_path)

    lines = []
    lines.append("=" * 90)
    lines.append("AUFGABE 5: POST-REVERSAL VERHALTEN & TRADE-SIMULATION (IS)")
    lines.append("=" * 90)

    # Filter to IS period
    rev_df['date_dt'] = pd.to_datetime(rev_df['date'])
    rev_df = rev_df[rev_df['date_dt'] <= IS_END].copy()

    earn_mask = rev_df['is_earnings'] == True
    non_mask = (~rev_df['is_earnings']) & (~rev_df['earnings_unknown'])

    # 5a: Post-Reversal Drift
    lines.append("\n--- 5a: Post-Reversal Drift ---")
    lines.append("  Drift from 10:00 in reversal direction (= against OD, back to gap direction)")

    categories = [
        ("Full Rev, Earnings", rev_df['reversal_full'] & earn_mask),
        ("Full Rev, Non-E", rev_df['reversal_full'] & non_mask),
        ("Partial Rev, Earn", rev_df['reversal_partial'] & ~rev_df['reversal_full'] & earn_mask),
        ("Partial Rev, Non-E", rev_df['reversal_partial'] & ~rev_df['reversal_full'] & non_mask),
        ("No Reversal, Earn", ~rev_df['reversal_partial'] & earn_mask),
        ("No Reversal, Non-E", ~rev_df['reversal_partial'] & non_mask),
    ]

    header = f"  {'Type':<25} | {'Drift→Gap med':>14} | {'Drift→OD med':>14} | {'N':>6}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for label, mask in categories:
        sub = rev_df[mask]
        n = len(sub)
        if n < 10:
            lines.append(f"  {label:<25} | {'N/A':>14} | {'N/A':>14} | {n:>6}")
            continue
        dg = sub['rest_drift_gap'].median()
        do = sub['rest_drift_od'].median()
        tag = " [LOW]" if n < 20 else ""
        lines.append(f"  {label:<25} | {dg:>+14.4f} | {do:>+14.4f} | {n:>6}{tag}")

    # 5b: Trade Simulation
    lines.append("\n\n--- 5b: Trade-Simulation fuer Reversal-Trade ---")
    lines.append("  Entry: 10:00, Richtung: gegen OD (= in Reversal/Gap-Richtung)")

    # Only process full reversals for trade simulation
    rev_full = rev_df[rev_df['reversal_full']].copy()
    print(f"Processing {len(rev_full)} full reversal trades...", file=sys.stderr)

    # For each reversal, load bars and simulate trades
    trade_results = []

    for _, row in tqdm(rev_full.iterrows(), total=len(rev_full),
                       desc="Simulating trades", file=sys.stderr):
        bars = load_1min_data(row['ticker'], row['date'])
        if bars is None:
            continue

        # Entry at 10:00 close
        entry_bar = bars[bars['time_et'] == '10:00']
        if len(entry_bar) == 0:
            continue
        entry_price = entry_bar.iloc[0]['close']
        adr = row['adr']

        # Direction: against OD
        if row['od_bullish']:
            direction = 'short'  # OD was bullish, we go against → short
        else:
            direction = 'long'  # OD was bearish, we go against → long

        # SL options
        sl_configs = {
            'SL_1000': None,  # Will use bar high/low
            'SL_025': 0.25,
            'SL_050': 0.50,
        }

        # For SL_1000: use the 10:00 bar's high (for short) or low (for long)
        bar_1000 = bars[bars['time_et'] == '10:00'].iloc[0]
        if direction == 'short':
            sl_1000_adr = (bar_1000['high'] - entry_price) / adr + 0.02  # small buffer
        else:
            sl_1000_adr = (entry_price - bar_1000['low']) / adr + 0.02
        sl_configs['SL_1000'] = max(sl_1000_adr, 0.05)  # minimum 0.05 ADR

        targets = [0.25, 0.50, 0.75, 1.00]

        for sl_name, sl_val in sl_configs.items():
            for tgt in targets:
                outcome, pnl, exit_time = simulate_trade(
                    bars, '10:00', direction, entry_price, adr, sl_val, tgt
                )
                trade_results.append({
                    'ticker': row['ticker'],
                    'date': row['date'],
                    'is_earnings': row['is_earnings'],
                    'earnings_unknown': row['earnings_unknown'],
                    'od_with_gap': row['od_with_gap'],
                    'od_bullish': row['od_bullish'],
                    'sl_name': sl_name,
                    'sl_val': sl_val,
                    'target': tgt,
                    'outcome': outcome,
                    'pnl_adr': pnl,
                    'direction': direction,
                })

    trade_df = pd.DataFrame(trade_results)
    trade_df.to_parquet(RESULTS_DIR / "d8_5_trade_sim.parquet", index=False)

    # Report results
    for sl_name in ['SL_1000', 'SL_025', 'SL_050']:
        for tgt in [0.25, 0.50, 0.75, 1.00]:
            sub = trade_df[(trade_df['sl_name'] == sl_name) & (trade_df['target'] == tgt)]
            if len(sub) == 0:
                continue

            lines.append(f"\n  {sl_name} / Target {tgt:.2f} ADR (N={len(sub)}):")

            for earn_label, emask in [("Earnings", sub['is_earnings'] == True),
                                       ("Non-E", (~sub['is_earnings']) & (~sub['earnings_unknown'])),
                                       ("ALL", sub.index == sub.index)]:
                esub = sub[emask]
                n = len(esub)
                if n < 10:
                    continue

                wr = (esub['outcome'] == 'target').mean() * 100
                sl_hit = (esub['outcome'] == 'sl').mean() * 100
                timeout = (esub['outcome'] == 'timeout').mean() * 100
                ev = esub['pnl_adr'].mean()
                ci_lo, ci_hi = bootstrap_mean_ci(esub['pnl_adr'].values)

                tag = " [LOW]" if n < 20 else ""
                lines.append(f"    {earn_label:>8} | WR={wr:5.1f}% SL={sl_hit:5.1f}% TO={timeout:5.1f}% | EV={ev:+.4f} [{ci_lo:+.4f},{ci_hi:+.4f}] | N={n}{tag}")

    # 5c: Vergleich mit Continuation und Fade Trade
    lines.append("\n\n--- 5c: Vergleich: Reversal vs Continuation vs Fade ---")
    lines.append("  (Bei denselben Tagen mit Full Reversal)")
    lines.append("  Continuation: Entry 9:35, in OD-Richtung, SL=0.25, Tgt=0.50")
    lines.append("  Fade: Entry 9:35, gegen Gap, SL=0.25, Tgt=0.50")
    lines.append("  Reversal: Entry 10:00, gegen OD, SL=0.25, Tgt=0.50")

    # Simulate continuation and fade for comparison
    comp_results = []
    for _, row in tqdm(rev_full.iterrows(), total=len(rev_full),
                       desc="Comparison trades", file=sys.stderr):
        bars = load_1min_data(row['ticker'], row['date'])
        if bars is None:
            continue

        adr = row['adr']

        # 9:35 entry
        entry_935 = bars[bars['time_et'] == '09:35']
        if len(entry_935) == 0:
            continue
        price_935 = entry_935.iloc[0]['close']

        # 10:00 entry
        entry_1000 = bars[bars['time_et'] == '10:00']
        if len(entry_1000) == 0:
            continue
        price_1000 = entry_1000.iloc[0]['close']

        # Continuation: in OD direction
        cont_dir = 'long' if row['od_bullish'] else 'short'
        outcome_c, pnl_c, _ = simulate_trade(bars, '09:35', cont_dir, price_935, adr, 0.25, 0.50)

        # Fade: against gap direction
        gap_dir = row['gap_direction']
        fade_dir = 'short' if gap_dir == 'up' else 'long'
        outcome_f, pnl_f, _ = simulate_trade(bars, '09:35', fade_dir, price_935, adr, 0.25, 0.50)

        # Reversal: against OD at 10:00
        rev_dir = 'short' if row['od_bullish'] else 'long'
        outcome_r, pnl_r, _ = simulate_trade(bars, '10:00', rev_dir, price_1000, adr, 0.25, 0.50)

        comp_results.append({
            'ticker': row['ticker'],
            'date': row['date'],
            'is_earnings': row['is_earnings'],
            'earnings_unknown': row['earnings_unknown'],
            'cont_outcome': outcome_c, 'cont_pnl': pnl_c,
            'fade_outcome': outcome_f, 'fade_pnl': pnl_f,
            'rev_outcome': outcome_r, 'rev_pnl': pnl_r,
        })

    comp_df = pd.DataFrame(comp_results)

    if len(comp_df) >= 20:
        for trade_type, pnl_col, outcome_col in [
            ('Continuation', 'cont_pnl', 'cont_outcome'),
            ('Fade', 'fade_pnl', 'fade_outcome'),
            ('Reversal@10:00', 'rev_pnl', 'rev_outcome'),
        ]:
            valid = comp_df[comp_df[outcome_col] != 'no_data']
            ev = valid[pnl_col].mean()
            wr = (valid[outcome_col] == 'target').mean() * 100
            ci_lo, ci_hi = bootstrap_mean_ci(valid[pnl_col].values)
            lines.append(f"  {trade_type:<20} | EV={ev:+.4f} [{ci_lo:+.4f},{ci_hi:+.4f}] | WR={wr:.1f}% | N={len(valid)}")

        # Split by earnings
        for earn_label, emask in [("Earnings", comp_df['is_earnings'] == True),
                                   ("Non-E", (~comp_df['is_earnings']) & (~comp_df['earnings_unknown']))]:
            esub = comp_df[emask]
            if len(esub) < 10:
                continue
            lines.append(f"\n  --- {earn_label} (N={len(esub)}) ---")
            for trade_type, pnl_col, outcome_col in [
                ('Continuation', 'cont_pnl', 'cont_outcome'),
                ('Fade', 'fade_pnl', 'fade_outcome'),
                ('Reversal@10:00', 'rev_pnl', 'rev_outcome'),
            ]:
                valid = esub[esub[outcome_col] != 'no_data']
                if len(valid) < 5:
                    continue
                ev = valid[pnl_col].mean()
                wr = (valid[outcome_col] == 'target').mean() * 100
                lines.append(f"    {trade_type:<20} | EV={ev:+.4f} | WR={wr:.1f}% | N={len(valid)}")

    text = "\n".join(lines)
    out_file = RESULTS_DIR / "d8_5_reversal_trade.txt"
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(text)
    print(text.replace('\u2192', '->'))


if __name__ == "__main__":
    main()
