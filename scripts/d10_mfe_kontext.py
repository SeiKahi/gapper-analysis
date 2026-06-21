"""
D10 Aufgabe 3: MFE nach Kontext-Parametern (IS, nur Gewinner)
"""
import pandas as pd
import numpy as np
import sys


def bucket_table(df, col, buckets, gap_dirs=['up', 'down']):
    """MFE stats by bucket for a given column."""
    lines = []
    for gap_dir, gap_label in [('up', 'GapUp'), ('down', 'GapDn')]:
        if gap_dir not in gap_dirs:
            continue
        lines.append(f"\n    {gap_label}:")
        lines.append(f"    {'Bucket':<12} | {'Med MFE_R':>10} | {'Mean MFE_R':>10} | {'StdDev':>8} | {'Med MFE_ADR':>11} | {'N':>4}")
        lines.append(f"    {'-'*65}")

        sub = df[df['gap_direction'] == gap_dir]

        for blabel, lo, hi in buckets:
            bsub = sub[(sub[col] >= lo) & (sub[col] < hi)]
            n = len(bsub)
            if n < 10:
                low_n = " [LOW N]" if n >= 1 else ""
                lines.append(f"    {blabel:<12} | {'':>10} | {'':>10} | {'':>8} | {'':>11} | {n:>4}{low_n}")
                continue

            med_r = bsub['mfe_live_r'].median()
            mean_r = bsub['mfe_live_r'].mean()
            std_r = bsub['mfe_live_r'].std()
            med_adr = bsub['mfe_live_adr'].median()
            low_n = " [LOW N]" if n < 20 else ""
            lines.append(f"    {blabel:<12} | {med_r:>10.2f} | {mean_r:>10.2f} | {std_r:>8.2f} | {med_adr:>11.3f} | {n:>4}{low_n}")

    return "\n".join(lines)


def main():
    lines = []
    lines.append("=" * 70)
    lines.append("D10 AUFGABE 3: MFE NACH KONTEXT-PARAMETERN (IS)")
    lines.append("=" * 70)
    lines.append("\nNur GEWINNER (SL ueberlebt). MFE_LIVE in R und ADR.")

    df = pd.read_parquet('results/d10_mfe_raw_is.parquet')
    winners = df[df['sl_hit'] == False].copy()
    all_trades = df.copy()
    print(f"Loaded {len(df)} trades, {len(winners)} winners", file=sys.stderr)

    # ===== 3a: RVOL_30 =====
    lines.append(f"\n\n{'='*70}")
    lines.append("3a: MFE nach RVOL_30 (nur Gewinner)")
    lines.append(f"{'='*70}")

    rvol30_buckets = [
        ('< 3x', 0, 3),
        ('3-5x', 3, 5),
        ('5-10x', 5, 10),
        ('> 10x', 10, 999),
    ]
    lines.append(bucket_table(winners, 'rvol_30', rvol30_buckets))

    # Also show ALL trades to see if high RVOL changes survival rate
    lines.append("\n\n  Survival-Rate nach RVOL_30 (alle Trades):")
    for gap_dir, gap_label in [('up', 'GapUp'), ('down', 'GapDn')]:
        sub = all_trades[all_trades['gap_direction'] == gap_dir]
        lines.append(f"    {gap_label}:")
        for blabel, lo, hi in rvol30_buckets:
            bsub = sub[(sub['rvol_30'] >= lo) & (sub['rvol_30'] < hi)]
            n = len(bsub)
            if n < 10:
                continue
            surv = (bsub['sl_hit'] == False).mean()
            med_mfe = bsub['mfe_live_r'].median()
            lines.append(f"      {blabel:<12}: Survival={surv:>5.1%}, Med MFE_R={med_mfe:.2f}, N={n}")

    # ===== 3b: Gap-Groesse =====
    lines.append(f"\n\n{'='*70}")
    lines.append("3b: MFE nach Gap-Groesse (nur Gewinner)")
    lines.append(f"{'='*70}")

    gap_buckets = [
        ('< 1 ADR', 0, 1),
        ('1-2 ADR', 1, 2),
        ('2-4 ADR', 2, 4),
        ('> 4 ADR', 4, 999),
    ]
    lines.append(bucket_table(winners, 'gap_adr', gap_buckets))

    lines.append("\n\n  Survival-Rate nach Gap-Groesse (alle Trades):")
    for gap_dir, gap_label in [('up', 'GapUp'), ('down', 'GapDn')]:
        sub = all_trades[all_trades['gap_direction'] == gap_dir]
        lines.append(f"    {gap_label}:")
        for blabel, lo, hi in gap_buckets:
            bsub = sub[(sub['gap_adr'] >= lo) & (sub['gap_adr'] < hi)]
            n = len(bsub)
            if n < 10:
                lines.append(f"      {blabel:<12}: N={n} [LOW N]")
                continue
            surv = (bsub['sl_hit'] == False).mean()
            med_mfe = bsub['mfe_live_r'].median()
            lines.append(f"      {blabel:<12}: Survival={surv:>5.1%}, Med MFE_R={med_mfe:.2f}, N={n}")

    # ===== 3c: OD-Staerke =====
    lines.append(f"\n\n{'='*70}")
    lines.append("3c: MFE nach OD-Staerke (nur Gewinner)")
    lines.append(f"{'='*70}")

    od_buckets = [
        ('0.5-0.7', 0.5, 0.7),
        ('0.7-1.0', 0.7, 1.0),
        ('1.0-1.5', 1.0, 1.5),
        ('> 1.5', 1.5, 999),
    ]
    lines.append(bucket_table(winners, 'od_strength', od_buckets))

    lines.append("\n\n  Survival-Rate nach OD-Staerke (alle Trades):")
    for gap_dir, gap_label in [('up', 'GapUp'), ('down', 'GapDn')]:
        sub = all_trades[all_trades['gap_direction'] == gap_dir]
        lines.append(f"    {gap_label}:")
        for blabel, lo, hi in od_buckets:
            bsub = sub[(sub['od_strength'] >= lo) & (sub['od_strength'] < hi)]
            n = len(bsub)
            if n < 10:
                lines.append(f"      {blabel:<12}: N={n} [LOW N]")
                continue
            surv = (bsub['sl_hit'] == False).mean()
            med_mfe = bsub['mfe_live_r'].median()
            lines.append(f"      {blabel:<12}: Survival={surv:>5.1%}, Med MFE_R={med_mfe:.2f}, N={n}")

    # ===== 3d: RVOL_5 =====
    lines.append(f"\n\n{'='*70}")
    lines.append("3d: MFE nach RVOL_5 (nur Gewinner)")
    lines.append(f"{'='*70}")

    rvol5_buckets = [
        ('< 2x', 0, 2),
        ('2-5x', 2, 5),
        ('5-10x', 5, 10),
        ('> 10x', 10, 999),
    ]
    lines.append(bucket_table(winners, 'rvol_5', rvol5_buckets))

    # ===== 3e: ZUSAMMENFASSUNG =====
    lines.append(f"\n\n{'='*70}")
    lines.append("3e: ZUSAMMENFASSUNG KONTEXT-EFFEKTE AUF MFE")
    lines.append(f"{'='*70}")

    lines.append("""
  Welcher Parameter beeinflusst wie weit Gewinner laufen?

  Parameter     | MFE-Spread (Med, Gewinner) | Survival-Effekt  | Urteil
  --------------------------------------------------------------------------""")

    # Compute spreads for each parameter
    for param_name, col, buckets in [
        ('RVOL_30', 'rvol_30', rvol30_buckets),
        ('Gap_ADR', 'gap_adr', gap_buckets),
        ('OD_Strength', 'od_strength', od_buckets),
        ('RVOL_5', 'rvol_5', rvol5_buckets),
    ]:
        mfe_vals = []
        surv_vals = []
        for gap_dir in ['up', 'down']:
            for _, lo, hi in buckets:
                sub_w = winners[(winners['gap_direction'] == gap_dir) & (winners[col] >= lo) & (winners[col] < hi)]
                sub_a = all_trades[(all_trades['gap_direction'] == gap_dir) & (all_trades[col] >= lo) & (all_trades[col] < hi)]
                if len(sub_w) >= 10:
                    mfe_vals.append(sub_w['mfe_live_r'].median())
                if len(sub_a) >= 10:
                    surv_vals.append((sub_a['sl_hit'] == False).mean())

        mfe_spread = max(mfe_vals) - min(mfe_vals) if len(mfe_vals) >= 2 else np.nan
        surv_spread = max(surv_vals) - min(surv_vals) if len(surv_vals) >= 2 else np.nan

        mfe_str = f"{mfe_spread:.1f}R" if pd.notna(mfe_spread) else "---"
        surv_str = f"{surv_spread:.1%}" if pd.notna(surv_spread) else "---"
        urteil = "STARK" if (pd.notna(mfe_spread) and mfe_spread > 3) else "MODERAT" if (pd.notna(mfe_spread) and mfe_spread > 1) else "SCHWACH"
        lines.append(f"  {param_name:<14}| {mfe_str:>28} | {surv_str:>16} | {urteil}")

    output = "\n".join(lines)
    with open('results/d10_mfe_kontext.txt', 'w', encoding='utf-8') as f:
        f.write(output)
    print(output)
    print(f"\nSaved to results/d10_mfe_kontext.txt", file=sys.stderr)


if __name__ == '__main__':
    main()
