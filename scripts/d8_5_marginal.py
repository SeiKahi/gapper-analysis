"""
D8.5 Aufgabe 2: Marginale Effekte der Kontext-Parameter bei QUALITY_HIGH + OD > 0.5
Nur H1 (IS): 2021-02-21 bis 2023-12-31
"""
import pandas as pd
import numpy as np

def wr_at_target(drift_series, target):
    """WR = fraction where drift >= target (for with-gap) or drift >= target (gap-signed)."""
    valid = drift_series.dropna()
    if len(valid) == 0:
        return np.nan
    return (valid >= target).mean()

def compute_metrics(sub, label=""):
    """Compute standard metrics for a subset."""
    n = len(sub)
    if n < 10:
        return None
    fd = sub['full_drift'].mean()
    rd = sub['rest_drift'].mean()
    wr25 = wr_at_target(sub['rest_drift'], 0.25)
    wr50 = wr_at_target(sub['rest_drift'], 0.50)
    return {
        'label': label,
        'N': n,
        'full_drift': fd,
        'rest_drift': rd,
        'WR@0.25': wr25,
        'WR@0.50': wr50,
    }

def format_row(m):
    if m is None:
        return "  [N < 10, nicht berichtet]"
    low_n = " [LOW N]" if m['N'] < 20 else ""
    return (f"  N={m['N']:>4}{low_n:>8} | fd={m['full_drift']:>+.3f} | rd={m['rest_drift']:>+.3f} | "
            f"WR@0.25={m['WR@0.25']:.1%} | WR@0.50={m['WR@0.50']:.1%}")


def bucket_analysis(df, col, buckets, bucket_labels, lines, title):
    """Analyze buckets for a given column, split by direction."""
    lines.append(f"\n\n=== {title} ===")
    lines.append(f"  Spalte: {col}")

    directions = [
        ('OD with GapUp', 'with_gap', 'up'),
        ('OD with GapDown', 'with_gap', 'down'),
        ('OD against GapUp', 'against_gap', 'up'),
        ('OD against GapDown', 'against_gap', 'down'),
    ]

    for dir_label, od_dir, gap_dir in directions:
        sub = df[(df['od_direction'] == od_dir) & (df['gap_direction'] == gap_dir)]
        lines.append(f"\n  --- {dir_label} (N={len(sub)}) ---")
        lines.append(f"  {'Bucket':<20} | {'N':>4} | {'full_drift':>10} | {'rest_drift':>10} | {'WR@0.25':>8} | {'WR@0.50':>8}")
        lines.append(f"  {'-'*80}")

        for i, label in enumerate(bucket_labels):
            if i < len(buckets) - 1:
                mask = (sub[col] >= buckets[i]) & (sub[col] < buckets[i+1])
            else:
                mask = sub[col] >= buckets[i]

            # Handle first bucket: include everything below
            if i == 0:
                mask = sub[col] < buckets[1]

            bsub = sub[mask]
            m = compute_metrics(bsub, label)
            if m:
                lines.append(f"  {label:<20} | {m['N']:>4} | {m['full_drift']:>+10.3f} | {m['rest_drift']:>+10.3f} | {m['WR@0.25']:>8.1%} | {m['WR@0.50']:>8.1%}")
            else:
                lines.append(f"  {label:<20} | {'N<10':>4} |")


def main():
    lines = []
    lines.append("=" * 70)
    lines.append("D8.5 AUFGABE 2: MARGINALE EFFEKTE BEI QUALITY_HIGH + OD > 0.5 (IS)")
    lines.append("=" * 70)

    meta = pd.read_parquet('data/metadata/metadata_v8_5.parquet')

    # H1 only
    h1 = meta[(meta['date'] >= '2021-02-21') & (meta['date'] <= '2023-12-31')].copy()
    h1 = h1[h1['od_body_pct'].notna() & h1['od_wick_ratio'].notna()].copy()

    # QUALITY_HIGH + OD > 0.5
    qh = h1[(h1['quality_high'] == True) & (h1['od_strength'] > 0.5)].copy()
    lines.append(f"\nQUALITY_HIGH + OD > 0.5 (IS): N={len(qh)}")

    # ===== 2a: Marginal effects =====
    lines.append("\n\n" + "=" * 70)
    lines.append("2a: MARGINALE EFFEKTE JEDES KONTEXT-PARAMETERS")
    lines.append("=" * 70)

    # RVOL_30 (= rvol_open_30min)
    bucket_analysis(qh, 'rvol_open_30min',
                    [0, 3, 5, 10, 999],
                    ['< 3x', '3-5x', '5-10x', '> 10x'],
                    lines, 'RVOL_30')

    # Gap_in_ADR (absolute)
    qh['gap_abs'] = qh['gap_size_in_adr'].abs()
    bucket_analysis(qh, 'gap_abs',
                    [0, 1, 2, 4, 999],
                    ['< 1 ADR', '1-2 ADR', '2-4 ADR', '> 4 ADR'],
                    lines, 'Gap in ADR')

    # PM/RTH30
    bucket_analysis(qh, 'pm_rth30_computed',
                    [0, 0.10, 0.30, 999],
                    ['< 10%', '10-30%', '> 30%'],
                    lines, 'PM/RTH30')

    # PM/RTH5
    bucket_analysis(qh, 'pm_rth5',
                    [0, 0.50, 1.00, 999],
                    ['< 50%', '50-100%', '> 100%'],
                    lines, 'PM/RTH5')

    # RVOL_5
    bucket_analysis(qh, 'rvol_5',
                    [0, 2, 5, 10, 999],
                    ['< 2x', '2-5x', '5-10x', '> 10x'],
                    lines, 'RVOL_5')

    # ===== 2b: QUALITY_HIGH vs QUALITY_LOW comparison =====
    lines.append("\n\n" + "=" * 70)
    lines.append("2b: QUALITY_HIGH vs QUALITY_LOW bei OD > 0.5")
    lines.append("=" * 70)

    od05 = h1[h1['od_strength'] > 0.5].copy()
    qh_od = od05[od05['quality_high'] == True]
    ql_od = od05[od05['quality_high'] == False]

    lines.append(f"\n  {'':>25} | {'Q_HIGH':>40} | {'Q_LOW':>40} | {'Delta':>12}")
    lines.append(f"  {'':>25} | {'N':>4} | {'fd':>7} | {'rd':>7} | {'WR25':>6} | {'WR50':>6} | {'N':>4} | {'fd':>7} | {'rd':>7} | {'WR25':>6} | {'WR50':>6} | {'dWR50':>6}")
    lines.append(f"  {'-'*140}")

    directions = [
        ('OD with GapUp', 'with_gap', 'up'),
        ('OD with GapDown', 'with_gap', 'down'),
        ('OD against GapUp', 'against_gap', 'up'),
        ('OD against GapDown', 'against_gap', 'down'),
    ]

    for dir_label, od_dir, gap_dir in directions:
        sub_h = qh_od[(qh_od['od_direction'] == od_dir) & (qh_od['gap_direction'] == gap_dir)]
        sub_l = ql_od[(ql_od['od_direction'] == od_dir) & (ql_od['gap_direction'] == gap_dir)]

        m_h = compute_metrics(sub_h)
        m_l = compute_metrics(sub_l)

        if m_h and m_l:
            delta_wr50 = m_h['WR@0.50'] - m_l['WR@0.50']
            lines.append(
                f"  {dir_label:<25} | {m_h['N']:>4} | {m_h['full_drift']:>+7.3f} | {m_h['rest_drift']:>+7.3f} | {m_h['WR@0.25']:>6.1%} | {m_h['WR@0.50']:>6.1%}"
                f" | {m_l['N']:>4} | {m_l['full_drift']:>+7.3f} | {m_l['rest_drift']:>+7.3f} | {m_l['WR@0.25']:>6.1%} | {m_l['WR@0.50']:>6.1%}"
                f" | {delta_wr50:>+6.1%}"
            )
        elif m_h:
            lines.append(f"  {dir_label:<25} | {m_h['N']:>4} | {m_h['full_drift']:>+7.3f} | Q_LOW N<10")
        else:
            lines.append(f"  {dir_label:<25} | N<10 for one or both groups")

    # Summary
    lines.append(f"\n  Zusammenfassung:")
    lines.append(f"  QUALITY_HIGH + OD > 0.5: N={len(qh_od)}")
    lines.append(f"  QUALITY_LOW + OD > 0.5:  N={len(ql_od)}")

    mh_all = compute_metrics(qh_od)
    ml_all = compute_metrics(ql_od)
    if mh_all and ml_all:
        lines.append(f"  Gesamt Q_HIGH: fd={mh_all['full_drift']:+.3f}, rd={mh_all['rest_drift']:+.3f}, WR@0.50={mh_all['WR@0.50']:.1%}")
        lines.append(f"  Gesamt Q_LOW:  fd={ml_all['full_drift']:+.3f}, rd={ml_all['rest_drift']:+.3f}, WR@0.50={ml_all['WR@0.50']:.1%}")
        lines.append(f"  Delta WR@0.50: {mh_all['WR@0.50'] - ml_all['WR@0.50']:+.1%}")

    output = "\n".join(lines)
    with open('results/d8_5_marginal.txt', 'w') as f:
        f.write(output)
    print(output)
    print(f"\nSaved to results/d8_5_marginal.txt")


if __name__ == '__main__':
    main()
