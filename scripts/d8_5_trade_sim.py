"""
D8.5 Aufgabe 4: Trade-Simulation mit kumulativen Filter-Schichten (IS)
Entry: 9:35, SL: 0.25 ADR fix, Targets: 0.25/0.50/0.75/1.00 ADR, Timeout: 15:55
"""
import pandas as pd
import numpy as np
from pathlib import Path


def simulate_trade(bars_rth, entry_price, adr, direction, sl_adr=0.25):
    """
    Simulate a trade from 9:35 onward.
    direction: 'long' or 'short'
    Returns dict with WR at various targets, SL hit, and timeout info.
    """
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
    results['exit_time'] = None

    # Walk through bars from 9:35 onward
    post_entry = bars_rth[bars_rth['time_et'] >= '09:35'].copy()

    for _, bar in post_entry.iterrows():
        if direction == 'long':
            # Check SL first (conservative)
            if bar['low'] <= sl_price:
                results['sl_hit'] = True
                results['exit_price'] = sl_price
                results['exit_time'] = bar['time_et']
                break
            # Check targets
            for t, tp in zip(targets, target_prices):
                if bar['high'] >= tp and not results[f'hit_{t}']:
                    results[f'hit_{t}'] = True
        else:
            if bar['high'] >= sl_price:
                results['sl_hit'] = True
                results['exit_price'] = sl_price
                results['exit_time'] = bar['time_et']
                break
            for t, tp in zip(targets, target_prices):
                if bar['low'] <= tp and not results[f'hit_{t}']:
                    results[f'hit_{t}'] = True

    # Timeout at 15:55 if not stopped out
    if not results['sl_hit']:
        timeout_bars = post_entry[post_entry['time_et'] <= '15:55']
        if len(timeout_bars) > 0:
            results['exit_price'] = timeout_bars.iloc[-1]['close']
            results['exit_time'] = '15:55'

    # Calculate PnL for EV calculation
    if pd.notna(results['exit_price']) and adr > 0:
        if direction == 'long':
            results['pnl_adr'] = (results['exit_price'] - entry_price) / adr
        else:
            results['pnl_adr'] = (entry_price - results['exit_price']) / adr

    return results


def run_simulation(meta_subset, lines, label):
    """Run trade simulation on a subset, loading 1min data for each."""
    raw_dir = Path('data/raw_1min')
    results = []

    for _, row in meta_subset.iterrows():
        ticker = row['ticker']
        date = str(row['date'])
        fpath = raw_dir / ticker / f"{date}.parquet"

        if not fpath.exists():
            continue

        bars = pd.read_parquet(fpath)
        rth = bars[bars['session'] == 'rth']

        if len(rth) < 10:
            continue

        entry_price = row['close_935']
        adr = row.get('adr_10', row.get('adr_5', np.nan))
        if pd.isna(entry_price) or pd.isna(adr) or adr <= 0:
            continue

        # Determine trade direction based on OD
        od_dir = row['od_direction']
        gap_dir = row['gap_direction']

        # "with gap" trade: trade in gap direction (OD confirms gap)
        # "against gap" trade: trade against gap (OD opposes gap, trade in OD direction)
        if od_dir == 'with_gap':
            direction = 'long' if gap_dir == 'up' else 'short'
        else:
            direction = 'short' if gap_dir == 'up' else 'long'

        trade = simulate_trade(rth, entry_price, adr, direction)
        trade['ticker'] = ticker
        trade['date'] = date
        trade['od_direction'] = od_dir
        trade['gap_direction'] = gap_dir
        results.append(trade)

    return pd.DataFrame(results)


def summarize_trades(trades_df, label):
    """Summarize trade results."""
    n = len(trades_df)
    if n < 10:
        return f"  {label:<45} | N={n:>4} [N<10, nicht berichtet]"

    wr25 = trades_df['hit_0.25'].mean()
    wr50 = trades_df['hit_0.5'].mean()
    wr75 = trades_df['hit_0.75'].mean()
    wr100 = trades_df['hit_1.0'].mean()
    sl_pct = trades_df['sl_hit'].mean()

    # EV @ 0.50 target with 0.25 SL
    # Win: +0.50, Loss: -0.25
    ev_050 = wr50 * 0.50 - sl_pct * 0.25

    low_n = " [LOW N]" if n < 20 else ""

    return (f"  {label:<45} | N={n:>4} | WR@0.25={wr25:>5.1%} | WR@0.50={wr50:>5.1%} | "
            f"WR@1.00={wr100:>5.1%} | EV@0.50={ev_050:>+.3f} | SL%={sl_pct:>5.1%}{low_n}")


def main():
    lines = []
    lines.append("=" * 70)
    lines.append("D8.5 AUFGABE 4: TRADE-SIMULATION MIT FILTER-SCHICHTEN (IS)")
    lines.append("=" * 70)
    lines.append("\nEntry: 9:35 | SL: 0.25 ADR | Targets: 0.25/0.50/0.75/1.00 ADR | Timeout: 15:55")

    meta = pd.read_parquet('data/metadata/metadata_v8_5.parquet')
    h1 = meta[(meta['date'] >= '2021-02-21') & (meta['date'] <= '2023-12-31')].copy()
    h1 = h1[h1['od_body_pct'].notna()].copy()
    h1['gap_abs'] = h1['gap_size_in_adr'].abs()

    # Define filter layers
    def layer0_with(df):
        return df[(df['od_strength'] > 0.5) & (df['od_direction'] == 'with_gap')]

    def layer0_against(df):
        return df[(df['od_strength'] > 0.5) & (df['od_direction'] == 'against_gap')]

    def layer1(df):
        return df[df['quality_high'] == True]

    def layer2a(df):  # RVOL_30 > 5x
        return df[df['rvol_open_30min'] > 5]

    def layer2b(df):  # Gap > 2 ADR
        return df[df['gap_abs'] > 2]

    def layer2c(df):  # PM/RTH30 < 10%
        return df[df['pm_rth30_computed'] < 0.10]

    def layer2d(df):  # RVOL_5 > 5x
        return df[df['rvol_5'] > 5]

    def layer2e(df):  # PM/RTH5 < 50%
        return df[df['pm_rth5'] < 0.50]

    def layer3(df):  # RVOL_30 > 5x + Gap > 2 ADR
        return df[(df['rvol_open_30min'] > 5) & (df['gap_abs'] > 2)]

    # ====== WITH GAP ======
    lines.append(f"\n\n{'='*70}")
    lines.append("OD > 0.5 WITH GAP (Trade in Gap-Richtung)")
    lines.append(f"{'='*70}")

    for gap_dir, gap_label in [('up', 'GapUp'), ('down', 'GapDown')]:
        lines.append(f"\n--- {gap_label} ---")
        base = h1[h1['gap_direction'] == gap_dir]

        layers = [
            ('L0: OD>0.5 with', lambda df: layer0_with(df)),
            ('L1: + QUALITY_HIGH', lambda df: layer1(layer0_with(df))),
            ('L2a: + RVOL_30>5x', lambda df: layer2a(layer1(layer0_with(df)))),
            ('L2b: + Gap>2ADR', lambda df: layer2b(layer1(layer0_with(df)))),
            ('L2c: + PM/RTH30<10%', lambda df: layer2c(layer1(layer0_with(df)))),
            ('L2d: + RVOL_5>5x', lambda df: layer2d(layer1(layer0_with(df)))),
            ('L2e: + PM/RTH5<50%', lambda df: layer2e(layer1(layer0_with(df)))),
            ('L3: + RVOL_30>5x + Gap>2ADR', lambda df: layer3(layer1(layer0_with(df)))),
        ]

        for layer_label, layer_fn in layers:
            subset = layer_fn(base)
            print(f"  Simulating {gap_label} {layer_label}: N={len(subset)}")
            if len(subset) < 10:
                lines.append(f"  {layer_label:<45} | N={len(subset):>4} [N<10]")
                continue
            trades = run_simulation(subset, lines, layer_label)
            lines.append(summarize_trades(trades, layer_label))

    # ====== AGAINST GAP ======
    lines.append(f"\n\n{'='*70}")
    lines.append("OD > 0.5 AGAINST GAP (Trade in OD-Richtung = gegen Gap)")
    lines.append(f"{'='*70}")

    for gap_dir, gap_label in [('up', 'GapUp'), ('down', 'GapDown')]:
        lines.append(f"\n--- {gap_label} ---")
        base = h1[h1['gap_direction'] == gap_dir]

        layers = [
            ('L0: OD>0.5 against', lambda df: layer0_against(df)),
            ('L1: + QUALITY_HIGH', lambda df: layer1(layer0_against(df))),
            ('L2a: + RVOL_30>5x', lambda df: layer2a(layer1(layer0_against(df)))),
            ('L2b: + Gap>2ADR', lambda df: layer2b(layer1(layer0_against(df)))),
            ('L2c: + PM/RTH30<10%', lambda df: layer2c(layer1(layer0_against(df)))),
            ('L2d: + RVOL_5>5x', lambda df: layer2d(layer1(layer0_against(df)))),
            ('L2e: + PM/RTH5<50%', lambda df: layer2e(layer1(layer0_against(df)))),
            ('L3: + RVOL_30>5x + Gap>2ADR', lambda df: layer3(layer1(layer0_against(df)))),
        ]

        for layer_label, layer_fn in layers:
            subset = layer_fn(base)
            print(f"  Simulating {gap_label} {layer_label}: N={len(subset)}")
            if len(subset) < 10:
                lines.append(f"  {layer_label:<45} | N={len(subset):>4} [N<10]")
                continue
            trades = run_simulation(subset, lines, layer_label)
            lines.append(summarize_trades(trades, layer_label))

    # Summary questions
    lines.append(f"\n\n{'='*70}")
    lines.append("ZUSAMMENFASSUNG")
    lines.append(f"{'='*70}")
    lines.append("\nFrage 1: Ab welchem Layer wird N zu duenn?")
    lines.append("Frage 2: Ab welchem Layer kommt kein Zusatz-EV mehr?")
    lines.append("(Antworten siehe Tabellen oben)")

    output = "\n".join(lines)
    with open('results/d8_5_trade_sim.txt', 'w', encoding='utf-8') as f:
        f.write(output)
    print(output)
    print(f"\nSaved to results/d8_5_trade_sim.txt")


if __name__ == '__main__':
    main()
