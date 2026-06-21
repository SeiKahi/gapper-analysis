"""
D8.5 Aufgabe 3: Kreuzung Reversal-Filter x Kontext-Parameter (IS)
+ Spread-Analyse + Regression
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def wr_at(drift, tgt):
    v = drift.dropna()
    return (v >= tgt).mean() if len(v) > 0 else np.nan

def cross_table(df, col, buckets, blabels, lines, title):
    """2er-Kreuzung: QUALITY_HIGH + OD>0.5 + context-param."""
    lines.append(f"\n\n{'='*70}")
    lines.append(f"3: KREUZUNG QUALITY_HIGH + OD > 0.5 + {title}")
    lines.append(f"{'='*70}")

    directions = [
        ('with GapUp', 'with_gap', 'up'),
        ('with GapDn', 'with_gap', 'down'),
        ('agst GapUp', 'against_gap', 'up'),
        ('agst GapDn', 'against_gap', 'down'),
    ]

    header = f"  {'Direction':<14} {'Bucket':<15} | {'N':>4} | {'full_drift':>10} | {'rest_drift':>10} | {'WR@0.50':>8}"
    lines.append(f"\n{header}")
    lines.append(f"  {'-'*75}")

    spreads = {}

    for dir_label, od_dir, gap_dir in directions:
        sub = df[(df['od_direction'] == od_dir) & (df['gap_direction'] == gap_dir)]

        best_rd, worst_rd = -999, 999
        best_lbl, worst_lbl = "", ""

        for i, label in enumerate(blabels):
            if i == 0:
                mask = sub[col] < buckets[1]
            elif i < len(blabels) - 1:
                mask = (sub[col] >= buckets[i]) & (sub[col] < buckets[i+1])
            else:
                mask = sub[col] >= buckets[-2]

            bsub = sub[mask]
            n = len(bsub)

            if n < 10:
                lines.append(f"  {dir_label:<14} {label:<15} | {n:>4} | {'[N<10]':>10}")
                continue

            fd = bsub['full_drift'].mean()
            rd = bsub['rest_drift'].mean()
            w50 = wr_at(bsub['rest_drift'], 0.50)
            low = " [LOW N]" if n < 20 else ""
            lines.append(f"  {dir_label:<14} {label:<15} | {n:>4} | {fd:>+10.3f} | {rd:>+10.3f} | {w50:>8.1%}{low}")

            if rd > best_rd:
                best_rd = rd
                best_lbl = label
            if rd < worst_rd:
                worst_rd = rd
                worst_lbl = label

        if best_rd > -999 and worst_rd < 999:
            spread = best_rd - worst_rd
            spreads[dir_label] = spread
            lines.append(f"  {dir_label:<14} {'SPREAD':<15} | {'':>4} | {'':>10} | {spread:>+10.3f} | (best={best_lbl}, worst={worst_lbl})")

        lines.append("")  # blank line between directions

    return spreads


def main():
    lines = []
    lines.append("=" * 70)
    lines.append("D8.5 AUFGABE 3: KREUZUNG REVERSAL-FILTER x KONTEXT (IS)")
    lines.append("=" * 70)

    meta = pd.read_parquet('data/metadata/metadata_v8_5.parquet')
    h1 = meta[(meta['date'] >= '2021-02-21') & (meta['date'] <= '2023-12-31')].copy()
    h1 = h1[h1['od_body_pct'].notna()].copy()

    # QUALITY_HIGH + OD > 0.5
    qh = h1[(h1['quality_high'] == True) & (h1['od_strength'] > 0.5)].copy()
    qh['gap_abs'] = qh['gap_size_in_adr'].abs()

    # Also prepare "only OD>0.5" (no quality filter) for spread comparison
    od05 = h1[h1['od_strength'] > 0.5].copy()
    od05['gap_abs'] = od05['gap_size_in_adr'].abs()

    lines.append(f"\nQH + OD>0.5: N={len(qh)}")
    lines.append(f"OD>0.5 (no quality filter): N={len(od05)}")

    # 3a-3e: Cross tables
    params = [
        ('rvol_open_30min', [0, 3, 5, 10, 999], ['< 3x', '3-5x', '5-10x', '> 10x'], 'RVOL_30'),
        ('gap_abs', [0, 1, 2, 4, 999], ['< 1 ADR', '1-2 ADR', '2-4 ADR', '> 4 ADR'], 'Gap_in_ADR'),
        ('pm_rth30_computed', [0, 0.10, 0.30, 999], ['< 10%', '10-30%', '> 30%'], 'PM/RTH30'),
        ('pm_rth5', [0, 0.50, 1.00, 999], ['< 50%', '50-100%', '> 100%'], 'PM/RTH5'),
        ('rvol_5', [0, 2, 5, 10, 999], ['< 2x', '2-5x', '5-10x', '> 10x'], 'RVOL_5'),
    ]

    all_spreads_qh = {}
    all_spreads_od = {}

    for col, bks, bls, title in params:
        sp_qh = cross_table(qh, col, bks, bls, lines, title)
        all_spreads_qh[title] = sp_qh

        # Same for OD>0.5 only (no quality filter)
        sp_od_lines = []
        sp_od = cross_table(od05, col, bks, bls, sp_od_lines, f"{title} [NUR OD>0.5, kein QH]")
        all_spreads_od[title] = sp_od

    # 3f: Spread analysis
    lines.append(f"\n\n{'='*70}")
    lines.append("3f: SPREAD-ANALYSE (Redundanz-Test)")
    lines.append(f"{'='*70}")
    lines.append("\n  Frage: Ist der Spread bei QH + OD>0.5 KLEINER als bei nur OD>0.5?")
    lines.append("  -> Kleiner = Kontext redundant nach Reversal-Filter")
    lines.append("  -> Gleich/groesser = Kontext bringt Extra-Info")

    lines.append(f"\n  {'Parameter':<15} | {'Direction':<14} | {'Spread QH+OD':>12} | {'Spread OD>0.5':>13} | {'Differenz':>10} | Bewertung")
    lines.append(f"  {'-'*90}")

    for title in [p[3] for p in params]:
        for dir_label in ['with GapUp', 'with GapDn', 'agst GapUp', 'agst GapDn']:
            sp_qh = all_spreads_qh.get(title, {}).get(dir_label, np.nan)
            sp_od = all_spreads_od.get(title, {}).get(dir_label, np.nan)

            if pd.notna(sp_qh) and pd.notna(sp_od):
                diff = sp_qh - sp_od
                if abs(sp_qh) < abs(sp_od) * 0.7:
                    bewert = "REDUNDANT"
                elif abs(sp_qh) > abs(sp_od) * 1.1:
                    bewert = "EXTRA-INFO"
                else:
                    bewert = "AEHNLICH"
                lines.append(f"  {title:<15} | {dir_label:<14} | {sp_qh:>+12.3f} | {sp_od:>+13.3f} | {diff:>+10.3f} | {bewert}")
            else:
                lines.append(f"  {title:<15} | {dir_label:<14} | {'N/A':>12} | {'N/A':>13} |")

    # 3g: Regression R2
    lines.append(f"\n\n{'='*70}")
    lines.append("3g: INFORMATIONSGEWINN (Regression)")
    lines.append(f"{'='*70}")

    directions = [
        ('with GapUp', 'with_gap', 'up'),
        ('with GapDn', 'with_gap', 'down'),
        ('agst GapUp', 'against_gap', 'up'),
        ('agst GapDn', 'against_gap', 'down'),
        ('ALLE', None, None),
    ]

    feat_cols = ['rvol_open_30min', 'gap_abs', 'pm_rth30_computed', 'pm_rth5', 'rvol_5']

    for dir_label, od_dir, gap_dir in directions:
        if od_dir:
            sub = qh[(qh['od_direction'] == od_dir) & (qh['gap_direction'] == gap_dir)].copy()
        else:
            sub = qh.copy()

        valid = sub.dropna(subset=feat_cols + ['rest_drift'])
        if len(valid) < 20:
            lines.append(f"\n  {dir_label}: N={len(valid)} — zu wenig fuer Regression")
            continue

        X = valid[feat_cols].values
        y = valid['rest_drift'].values
        reg = LinearRegression().fit(X, y)
        r2 = reg.score(X, y)

        lines.append(f"\n  {dir_label} (N={len(valid)}): R2 = {r2:.4f} ({r2*100:.2f}%)")
        for fn, coef in zip(feat_cols, reg.coef_):
            lines.append(f"    {fn:<25}: coef = {coef:+.4f}")

    lines.append(f"\n  Vergleich: D7.5 Dreieck (alle Gapper) R2 = 0.7%")

    output = "\n".join(lines)
    with open('results/d8_5_kreuzung.txt', 'w', encoding='utf-8') as f:
        f.write(output)
    print(output)
    print(f"\nSaved to results/d8_5_kreuzung.txt")


if __name__ == '__main__':
    main()
