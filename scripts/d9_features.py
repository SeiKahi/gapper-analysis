"""
D9 Aufgabe 1: Reversal-Filter Buckets + SL-Distanz Analyse
Reuse od_body_pct/od_wick_ratio from v8_5.
Compute SL_FULL, SL_HALF, OD extremes from 1min data.
Save as metadata_v9.parquet.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

def compute_sl_features(meta):
    """Compute SL_FULL and SL_HALF from 1min data."""
    raw_dir = Path('data/raw_1min')
    results = []
    total = len(meta)
    errors = 0
    skipped = 0

    for i, (idx, row) in enumerate(meta.iterrows()):
        if i % 2000 == 0:
            print(f"  Processing {i}/{total}...", file=sys.stderr)

        ticker = row['ticker']
        date = str(row['date'])
        rth_open = row['rth_open']
        close_935 = row['close_935']
        adr = row.get('adr_10', np.nan)
        od_dir = row.get('od_direction', None)
        gap_dir = row.get('gap_direction', None)

        fpath = raw_dir / ticker / f"{date}.parquet"
        if not fpath.exists() or pd.isna(close_935) or pd.isna(adr) or adr <= 0:
            skipped += 1
            results.append({'idx': idx, 'od_high5': np.nan, 'od_low5': np.nan,
                            'sl_full_dist': np.nan, 'sl_half_dist': np.nan,
                            'sl_full_level': np.nan, 'sl_half_level': np.nan,
                            'trade_direction': None})
            continue

        try:
            bars = pd.read_parquet(fpath)
            rth = bars[bars['session'] == 'rth']

            # First 5 minutes: 09:30-09:34
            od_bars = rth[rth['time_et'].isin(['09:30', '09:31', '09:32', '09:33', '09:34'])]

            if len(od_bars) == 0:
                skipped += 1
                results.append({'idx': idx, 'od_high5': np.nan, 'od_low5': np.nan,
                                'sl_full_dist': np.nan, 'sl_half_dist': np.nan,
                                'sl_full_level': np.nan, 'sl_half_level': np.nan,
                                'trade_direction': None})
                continue

            od_high5 = od_bars['high'].max()
            od_low5 = od_bars['low'].min()
            od_range = od_high5 - od_low5

            # Determine trade direction and SL
            if od_dir == 'with_gap':
                if gap_dir == 'up':
                    # Long trade, SL at OD low
                    trade_dir = 'long'
                    sl_full_level = od_low5
                    sl_full_dist = close_935 - od_low5
                else:
                    # Short trade, SL at OD high
                    trade_dir = 'short'
                    sl_full_level = od_high5
                    sl_full_dist = od_high5 - close_935
            elif od_dir == 'against_gap':
                if gap_dir == 'up':
                    # Short trade (against gap up = OD is bearish), SL at OD high
                    trade_dir = 'short'
                    sl_full_level = od_high5
                    sl_full_dist = od_high5 - close_935
                else:
                    # Long trade (against gap down = OD is bullish), SL at OD low
                    trade_dir = 'long'
                    sl_full_level = od_low5
                    sl_full_dist = close_935 - od_low5
            else:
                results.append({'idx': idx, 'od_high5': od_high5, 'od_low5': od_low5,
                                'sl_full_dist': np.nan, 'sl_half_dist': np.nan,
                                'sl_full_level': np.nan, 'sl_half_level': np.nan,
                                'trade_direction': None})
                continue

            # SL_HALF: half the OD range
            sl_half_dist = 0.5 * od_range

            # SL_HALF level
            if trade_dir == 'long':
                sl_half_level = close_935 - sl_half_dist
            else:
                sl_half_level = close_935 + sl_half_dist

            # Normalize to ADR
            sl_full_adr = sl_full_dist / adr if adr > 0 else np.nan
            sl_half_adr = sl_half_dist / adr if adr > 0 else np.nan

            results.append({
                'idx': idx,
                'od_high5': od_high5,
                'od_low5': od_low5,
                'sl_full_dist': sl_full_dist,
                'sl_half_dist': sl_half_dist,
                'sl_full_adr': sl_full_adr,
                'sl_half_adr': sl_half_adr,
                'sl_full_level': sl_full_level,
                'sl_half_level': sl_half_level,
                'trade_direction': trade_dir,
            })

        except Exception as e:
            errors += 1
            results.append({'idx': idx, 'od_high5': np.nan, 'od_low5': np.nan,
                            'sl_full_dist': np.nan, 'sl_half_dist': np.nan,
                            'sl_full_level': np.nan, 'sl_half_level': np.nan,
                            'trade_direction': None})

    print(f"  Done. Errors: {errors}, Skipped: {skipped}", file=sys.stderr)
    return pd.DataFrame(results).set_index('idx')


def main():
    lines = []
    lines.append("=" * 70)
    lines.append("D9 AUFGABE 1: REVERSAL-FILTER BUCKETS + SL-ANALYSE")
    lines.append("=" * 70)

    # Load existing metadata with features
    meta = pd.read_parquet('data/metadata/metadata_v8_5.parquet')
    print(f"Loaded metadata_v8_5: {len(meta)} rows, {len(meta.columns)} cols", file=sys.stderr)

    # Compute SL features
    print("Computing SL features from 1min data...", file=sys.stderr)
    sl_feats = compute_sl_features(meta)

    # Merge
    for col in sl_feats.columns:
        if col not in meta.columns:
            meta[col] = sl_feats[col].values

    # Save as v9
    meta.to_parquet('data/metadata/metadata_v9.parquet', index=False)
    print(f"Saved metadata_v9.parquet: {len(meta)} rows, {len(meta.columns)} cols", file=sys.stderr)

    # ===== 1a: Distribution =====
    lines.append("\n\n--- 1a: Verteilung od_body_pct und od_wick_ratio ---")
    valid = meta[meta['od_body_pct'].notna() & meta['od_wick_ratio'].notna()]
    lines.append(f"\nGesamt: {len(meta)} Gapper, {len(valid)} mit gueltigem Body%/Wick")

    for col, label in [('od_body_pct', 'OD Body%'), ('od_wick_ratio', 'OD Wick Ratio')]:
        s = valid[col]
        lines.append(f"\n  {label}: Mean={s.mean():.3f}, Median={s.median():.3f}, Q25={s.quantile(0.25):.3f}, Q75={s.quantile(0.75):.3f}")

    # ===== 1b: QUALITY_HIGH / QUALITY_LOW =====
    lines.append("\n\n--- 1b: QUALITY_HIGH vs QUALITY_LOW ---")
    lines.append("  QUALITY_HIGH: od_body_pct >= 0.50 AND od_wick_ratio < 0.35")
    lines.append("  QUALITY_LOW:  od_body_pct < 0.50 OR  od_wick_ratio >= 0.35")

    n_high = (valid['quality_high'] == True).sum()
    n_low = (valid['quality_high'] == False).sum()
    lines.append(f"\n  QUALITY_HIGH: {n_high} ({100*n_high/len(valid):.1f}%)")
    lines.append(f"  QUALITY_LOW:  {n_low} ({100*n_low/len(valid):.1f}%)")

    # ===== 1c: QUALITY_HIGH + OD > 0.5 =====
    lines.append("\n\n--- 1c: QUALITY_HIGH + OD > 0.5 ADR Breakdown ---")

    h1 = valid[(valid['date'] >= '2021-02-21') & (valid['date'] <= '2023-12-31')]
    h2 = valid[(valid['date'] >= '2024-01-01') & (valid['date'] <= '2026-02-06')]

    for half_name, half_df in [('H1 (IS)', h1), ('H2 (OOS)', h2)]:
        lines.append(f"\n  === {half_name} ===")
        qh = half_df[half_df['quality_high'] == True]
        qh_od = qh[qh['od_strength'] > 0.5]
        lines.append(f"  QUALITY_HIGH: {len(qh)} ({100*len(qh)/len(half_df):.1f}%)")
        lines.append(f"  QH + OD > 0.5: {len(qh_od)} ({100*len(qh_od)/len(half_df):.1f}%)")

        for direction in ['with_gap', 'against_gap']:
            for gap_dir in ['up', 'down']:
                subset = qh_od[(qh_od['od_direction'] == direction) & (qh_od['gap_direction'] == gap_dir)]
                lines.append(f"    OD {direction} Gap{gap_dir.capitalize()}: N={len(subset)}")

    # ===== 1d: SL-Distanz Analyse =====
    lines.append("\n\n--- 1d: SL-DISTANZ ANALYSE ---")

    # IS only
    h1_od = h1[h1['od_strength'] > 0.5].copy()
    h1_od = h1_od[h1_od['sl_full_adr'].notna()].copy()

    lines.append(f"\n  OD > 0.5 (IS): N={len(h1_od)} mit gueltiger SL-Distanz")

    od_buckets = [
        ('0.5-0.7 ADR', 0.5, 0.7),
        ('0.7-1.0 ADR', 0.7, 1.0),
        ('1.0-1.5 ADR', 1.0, 1.5),
        ('> 1.5 ADR', 1.5, 999),
    ]

    for filter_label, filter_fn_name in [
        ('ALL OD > 0.5', lambda df: df),
        ('QUALITY_HIGH', lambda df: df[df['quality_high'] == True]),
    ]:
        sub = filter_fn_name(h1_od)
        lines.append(f"\n  === {filter_label} (N={len(sub)}) ===")
        lines.append(f"  {'OD Bucket':<15} | {'Med SL_FULL':>12} | {'Med SL_HALF':>12} | {'Med SL_FIX':>12} | {'N':>5}")
        lines.append(f"  {'-'*65}")

        for blabel, lo, hi in od_buckets:
            bsub = sub[(sub['od_strength'] >= lo) & (sub['od_strength'] < hi)]
            if len(bsub) < 10:
                lines.append(f"  {blabel:<15} | {'[N<10]':>12} | {'':>12} | {'':>12} | {len(bsub):>5}")
                continue
            med_full = bsub['sl_full_adr'].median()
            med_half = bsub['sl_half_adr'].median()
            lines.append(f"  {blabel:<15} | {med_full:>12.3f} | {med_half:>12.3f} | {'0.250':>12} | {len(bsub):>5}")

    # Additional: SL distribution for key subset
    lines.append("\n  === SL-Verteilung: QH + OD > 0.5 with_gap (IS) ===")
    qh_with = h1_od[(h1_od['quality_high'] == True) & (h1_od['od_direction'] == 'with_gap')]
    for sl_col, sl_label in [('sl_full_adr', 'SL_FULL'), ('sl_half_adr', 'SL_HALF')]:
        s = qh_with[sl_col].dropna()
        lines.append(f"  {sl_label}: Mean={s.mean():.3f}, Median={s.median():.3f}, Q25={s.quantile(0.25):.3f}, Q75={s.quantile(0.75):.3f}, Min={s.min():.3f}, Max={s.max():.3f}")

    lines.append(f"\n  Fazit: SL_HALF (Median ~{qh_with['sl_half_adr'].median():.2f} ADR) ist ca. halb so gross wie SL_FULL ({qh_with['sl_full_adr'].median():.2f} ADR).")
    lines.append(f"  SL_FIX025 = 0.25 ADR liegt {'unter' if 0.25 < qh_with['sl_half_adr'].median() else 'ueber'} SL_HALF Median.")

    output = "\n".join(lines)
    with open('results/d9_features.txt', 'w', encoding='utf-8') as f:
        f.write(output)
    print(output)
    print(f"\nSaved to results/d9_features.txt", file=sys.stderr)


if __name__ == '__main__':
    main()
