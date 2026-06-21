"""
D8.5 Aufgabe 1: Reversal-Filter Features berechnen
- od_body_pct: |close_935 - rth_open| / (high_5min - low_5min)
- od_wick_ratio: Wick gegen OD-Richtung / (high_5min - low_5min)
- candle5_vs_candle1: Schliesst Kerze 5 (09:34) in OD-Richtung? (Boolean)
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path

def compute_od_features(meta):
    """Compute od_body_pct and od_wick_ratio from 1min data."""
    results = []
    raw_dir = Path('data/raw_1min')

    total = len(meta)
    errors = 0
    skipped = 0

    for i, (idx, row) in enumerate(meta.iterrows()):
        if i % 1000 == 0:
            print(f"  Processing {i}/{total}...")

        ticker = row['ticker']
        date = str(row['date'])
        rth_open = row['rth_open']
        close_935 = row['close_935']

        fpath = raw_dir / ticker / f"{date}.parquet"
        if not fpath.exists():
            skipped += 1
            results.append({'idx': idx, 'od_body_pct': np.nan, 'od_wick_ratio': np.nan, 'candle5_vs_candle1': np.nan})
            continue

        try:
            bars = pd.read_parquet(fpath)
            rth = bars[bars['session'] == 'rth']

            # First 5 minutes: 09:30, 09:31, 09:32, 09:33, 09:34
            od_bars = rth[rth['time_et'].isin(['09:30', '09:31', '09:32', '09:33', '09:34'])]

            if len(od_bars) == 0:
                skipped += 1
                results.append({'idx': idx, 'od_body_pct': np.nan, 'od_wick_ratio': np.nan, 'candle5_vs_candle1': np.nan})
                continue

            high_5min = od_bars['high'].max()
            low_5min = od_bars['low'].min()
            od_range = high_5min - low_5min

            if od_range <= 0 or pd.isna(rth_open) or pd.isna(close_935):
                results.append({'idx': idx, 'od_body_pct': np.nan, 'od_wick_ratio': np.nan, 'candle5_vs_candle1': np.nan})
                continue

            # od_body_pct
            od_body_pct = abs(close_935 - rth_open) / od_range

            # od_wick_ratio: wick against OD direction
            od_bullish = close_935 > rth_open
            if od_bullish:
                # Bullish OD: wick against = lower tail (open - low)
                od_wick_ratio = (rth_open - low_5min) / od_range
            else:
                # Bearish OD: wick against = upper tail (high - open)
                od_wick_ratio = (high_5min - rth_open) / od_range

            # candle5_vs_candle1: Does candle 5 (09:34) close in OD direction?
            candle1 = od_bars[od_bars['time_et'] == '09:30']
            candle5 = od_bars[od_bars['time_et'] == '09:34']

            c5_vs_c1 = np.nan
            if len(candle5) > 0:
                c5_close = candle5.iloc[0]['close']
                if od_bullish:
                    c5_vs_c1 = c5_close > rth_open  # Candle 5 closes above open = confirms bullish
                else:
                    c5_vs_c1 = c5_close < rth_open  # Candle 5 closes below open = confirms bearish

            results.append({
                'idx': idx,
                'od_body_pct': od_body_pct,
                'od_wick_ratio': od_wick_ratio,
                'candle5_vs_candle1': c5_vs_c1
            })

        except Exception as e:
            errors += 1
            results.append({'idx': idx, 'od_body_pct': np.nan, 'od_wick_ratio': np.nan, 'candle5_vs_candle1': np.nan})

    print(f"  Done. Errors: {errors}, Skipped (no file): {skipped}")
    return pd.DataFrame(results).set_index('idx')


def main():
    lines = []
    lines.append("=" * 70)
    lines.append("D8.5 AUFGABE 1: REVERSAL-FILTER FEATURES & BUCKETS")
    lines.append("=" * 70)

    # Load metadata
    meta = pd.read_parquet('data/metadata/metadata_v8.parquet')
    print(f"Loaded metadata_v8: {len(meta)} rows")

    # Compute features
    print("Computing OD features from 1min data...")
    feats = compute_od_features(meta)

    # Merge into metadata
    meta = meta.join(feats, how='left')

    # Save as v8_5
    meta.to_parquet('data/metadata/metadata_v8_5.parquet', index=False)
    print(f"Saved metadata_v8_5.parquet with {len(meta)} rows, {len(meta.columns)} columns")

    # ===== 1a: Distribution =====
    lines.append("\n\n--- 1a: Verteilung od_body_pct und od_wick_ratio ---")

    valid = meta[meta['od_body_pct'].notna() & meta['od_wick_ratio'].notna()]
    lines.append(f"\nGesamt: {len(meta)} Gapper, davon {len(valid)} mit gueltigem od_body_pct/wick_ratio")
    lines.append(f"NaN/dropped: {len(meta) - len(valid)} (od_range=0 oder fehlende 1min-Daten)")

    for col, label in [('od_body_pct', 'OD Body%'), ('od_wick_ratio', 'OD Wick Ratio')]:
        s = valid[col]
        lines.append(f"\n  {label}:")
        lines.append(f"    Mean:   {s.mean():.3f}")
        lines.append(f"    Median: {s.median():.3f}")
        lines.append(f"    Q25:    {s.quantile(0.25):.3f}")
        lines.append(f"    Q75:    {s.quantile(0.75):.3f}")
        lines.append(f"    Min:    {s.min():.3f}")
        lines.append(f"    Max:    {s.max():.3f}")
        lines.append(f"    Std:    {s.std():.3f}")

    # ===== 1b: Define QUALITY_HIGH / QUALITY_LOW =====
    lines.append("\n\n--- 1b: QUALITY_HIGH vs QUALITY_LOW Definition ---")
    lines.append("\n  QUALITY_HIGH: od_body_pct >= 0.50 AND od_wick_ratio < 0.35")
    lines.append("  QUALITY_LOW:  od_body_pct < 0.50 OR  od_wick_ratio >= 0.35")

    valid['quality_high'] = (valid['od_body_pct'] >= 0.50) & (valid['od_wick_ratio'] < 0.35)

    n_high = valid['quality_high'].sum()
    n_low = (~valid['quality_high']).sum()
    lines.append(f"\n  QUALITY_HIGH: {n_high} ({100*n_high/len(valid):.1f}%)")
    lines.append(f"  QUALITY_LOW:  {n_low} ({100*n_low/len(valid):.1f}%)")

    # Also save quality_high to parquet
    meta['quality_high'] = (meta['od_body_pct'] >= 0.50) & (meta['od_wick_ratio'] < 0.35)
    meta.to_parquet('data/metadata/metadata_v8_5.parquet', index=False)

    # ===== 1c: QUALITY_HIGH + OD > 0.5 ADR breakdown =====
    lines.append("\n\n--- 1c: QUALITY_HIGH + OD > 0.5 ADR Breakdown ---")

    # Use IS only (H1)
    h1 = valid[(valid['date'] >= '2021-02-21') & (valid['date'] <= '2023-12-31')].copy()
    h2 = valid[(valid['date'] >= '2024-01-01') & (valid['date'] <= '2026-02-06')].copy()

    lines.append(f"\n  H1 (IS): {len(h1)} Gapper mit gueltigem Body%/Wick")
    lines.append(f"  H2 (OOS): {len(h2)} Gapper mit gueltigem Body%/Wick")

    for half_name, half_df in [('H1 (IS)', h1), ('H2 (OOS)', h2)]:
        lines.append(f"\n  === {half_name} ===")

        qh = half_df[half_df['quality_high']].copy()
        qh_od = qh[qh['od_strength'] > 0.5].copy()

        lines.append(f"  QUALITY_HIGH: {len(qh)} ({100*len(qh)/len(half_df):.1f}%)")
        lines.append(f"  QUALITY_HIGH + OD > 0.5 ADR: {len(qh_od)} ({100*len(qh_od)/len(half_df):.1f}%)")

        # By direction
        for direction in ['with_gap', 'against_gap']:
            for gap_dir in ['up', 'down']:
                subset = qh_od[(qh_od['od_direction'] == direction) & (qh_od['gap_direction'] == gap_dir)]
                label = f"OD {direction} Gap{gap_dir.capitalize()}"
                lines.append(f"    {label}: N={len(subset)}")

    # Also show QUALITY_LOW + OD > 0.5 for comparison
    lines.append(f"\n  === Vergleich: QUALITY_LOW + OD > 0.5 (H1) ===")
    ql = h1[~h1['quality_high']].copy()
    ql_od = ql[ql['od_strength'] > 0.5].copy()
    lines.append(f"  QUALITY_LOW + OD > 0.5 ADR: {len(ql_od)}")
    for direction in ['with_gap', 'against_gap']:
        for gap_dir in ['up', 'down']:
            subset = ql_od[(ql_od['od_direction'] == direction) & (ql_od['gap_direction'] == gap_dir)]
            label = f"OD {direction} Gap{gap_dir.capitalize()}"
            lines.append(f"    {label}: N={len(subset)}")

    # Additional: Distribution of od_body_pct and wick_ratio for OD>0.5
    lines.append(f"\n  === Verteilung bei OD > 0.5 (H1) ===")
    od05 = h1[h1['od_strength'] > 0.5]
    for col, label in [('od_body_pct', 'Body%'), ('od_wick_ratio', 'Wick Ratio')]:
        s = od05[col].dropna()
        lines.append(f"  {label}: Median={s.median():.3f}, Q25={s.quantile(0.25):.3f}, Q75={s.quantile(0.75):.3f}, N={len(s)}")

    # Save results
    output = "\n".join(lines)
    with open('results/d8_5_features.txt', 'w') as f:
        f.write(output)
    print(output)
    print(f"\nSaved to results/d8_5_features.txt")


if __name__ == '__main__':
    main()
