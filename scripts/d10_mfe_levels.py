"""
D10 Aufgabe 2: MFE nach R-Stufen + bedingte Wahrscheinlichkeiten (IS)
Survival curves and conditional probabilities for trailing decisions.
"""
import pandas as pd
import numpy as np
import sys


def main():
    lines = []
    lines.append("=" * 70)
    lines.append("D10 AUFGABE 2: MFE NACH R-STUFEN + BEDINGTE WK (IS)")
    lines.append("=" * 70)
    lines.append("\nSetup: OD > 0.5 ADR + with_gap + SL 0.25 ADR fix")
    lines.append("1R = 0.25 ADR = SL-Distanz")

    # Load pre-computed MFE results
    df = pd.read_parquet('results/d10_mfe_raw_is.parquet')
    print(f"Loaded {len(df)} trades", file=sys.stderr)

    # ===== 2a: Kumulative Verteilung ALLE =====
    lines.append(f"\n\n{'='*70}")
    lines.append("2a: KUMULATIVE VERTEILUNG ALLE TRADES (MFE_LIVE)")
    lines.append(f"{'='*70}")

    r_levels = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
    lines.append(f"\n  {'R-Level':>8} | {'% GapUp':>8} | {'% GapDn':>8} | {'N GapUp':>8} | {'N GapDn':>8}")
    lines.append(f"  {'-'*50}")

    for gap_dir, gap_label in [('up', 'GapUp'), ('down', 'GapDn')]:
        sub = df[df['gap_direction'] == gap_dir]
        globals()[f'sub_{gap_dir}'] = sub  # store for later

    sub_up = df[df['gap_direction'] == 'up']
    sub_dn = df[df['gap_direction'] == 'down']

    for r in r_levels:
        n_up = (sub_up['mfe_live_r'] >= r).sum()
        n_dn = (sub_dn['mfe_live_r'] >= r).sum()
        pct_up = 100 * n_up / len(sub_up) if len(sub_up) > 0 else 0
        pct_dn = 100 * n_dn / len(sub_dn) if len(sub_dn) > 0 else 0
        lines.append(f"  {r:>7.1f}R | {pct_up:>7.1f}% | {pct_dn:>7.1f}% | {n_up:>8} | {n_dn:>8}")

    # ADR levels
    adr_levels = [0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00]
    lines.append(f"\n  {'ADR-Lvl':>8} | {'% GapUp':>8} | {'% GapDn':>8}")
    lines.append(f"  {'-'*30}")

    for a in adr_levels:
        pct_up = 100 * (sub_up['mfe_live_adr'] >= a).sum() / len(sub_up) if len(sub_up) > 0 else 0
        pct_dn = 100 * (sub_dn['mfe_live_adr'] >= a).sum() / len(sub_dn) if len(sub_dn) > 0 else 0
        lines.append(f"  {a:>6.2f}ADR | {pct_up:>7.1f}% | {pct_dn:>7.1f}%")

    # ===== 2b: Kumulative Verteilung NUR GEWINNER =====
    lines.append(f"\n\n{'='*70}")
    lines.append("2b: KUMULATIVE VERTEILUNG NUR GEWINNER (SL ueberlebt)")
    lines.append(f"{'='*70}")

    winners = df[df['sl_hit'] == False]
    win_up = winners[winners['gap_direction'] == 'up']
    win_dn = winners[winners['gap_direction'] == 'down']

    lines.append(f"\n  Gewinner: GapUp N={len(win_up)}, GapDn N={len(win_dn)}")

    lines.append(f"\n  {'R-Level':>8} | {'% GapUp':>8} | {'% GapDn':>8} | {'N GapUp':>8} | {'N GapDn':>8}")
    lines.append(f"  {'-'*50}")

    for r in r_levels:
        n_up = (win_up['mfe_live_r'] >= r).sum()
        n_dn = (win_dn['mfe_live_r'] >= r).sum()
        pct_up = 100 * n_up / len(win_up) if len(win_up) > 0 else 0
        pct_dn = 100 * n_dn / len(win_dn) if len(win_dn) > 0 else 0
        lines.append(f"  {r:>7.1f}R | {pct_up:>7.1f}% | {pct_dn:>7.1f}% | {n_up:>8} | {n_dn:>8}")

    lines.append(f"\n  {'ADR-Lvl':>8} | {'% GapUp':>8} | {'% GapDn':>8}")
    lines.append(f"  {'-'*30}")

    for a in adr_levels:
        pct_up = 100 * (win_up['mfe_live_adr'] >= a).sum() / len(win_up) if len(win_up) > 0 else 0
        pct_dn = 100 * (win_dn['mfe_live_adr'] >= a).sum() / len(win_dn) if len(win_dn) > 0 else 0
        lines.append(f"  {a:>6.2f}ADR | {pct_up:>7.1f}% | {pct_dn:>7.1f}%")

    # ===== 2c: Bedingte Wahrscheinlichkeiten =====
    lines.append(f"\n\n{'='*70}")
    lines.append("2c: BEDINGTE WAHRSCHEINLICHKEITEN (MFE_LIVE)")
    lines.append(f"{'='*70}")
    lines.append("\n  P(YR | >= XR) = 'Wenn Trade XR erreicht hat, wie oft erreicht er YR?'")

    for gap_dir, gap_label in [('up', 'GapUp'), ('down', 'GapDn')]:
        lines.append(f"\n  --- {gap_label} ---")
        sub = df[df['gap_direction'] == gap_dir]

        cond_levels = [1.0, 2.0, 3.0, 4.0, 5.0]
        target_levels = [2.0, 3.0, 4.0, 5.0, 8.0, 10.0]

        header = f"  {'Bereits':>10}"
        for t in target_levels:
            header += f" | P({t:.0f}R)"
        header += f" |   N"
        lines.append(header)
        lines.append(f"  {'-'*75}")

        for c in cond_levels:
            reached_c = sub[sub['mfe_live_r'] >= c]
            n_c = len(reached_c)
            if n_c < 10:
                line = f"  {'>= '+str(c)+'R':>10}"
                line += " | N<10" * len(target_levels) + f" | {n_c:>3}"
                lines.append(line)
                continue

            line = f"  {'>= '+str(c)+'R':>10}"
            for t in target_levels:
                if t <= c:
                    line += " |   ---"
                else:
                    p = (reached_c['mfe_live_r'] >= t).sum() / n_c if n_c > 0 else 0
                    line += f" | {p:>5.1%}"
            low_n = " [LOW N]" if n_c < 20 else ""
            line += f" | {n_c:>3}{low_n}"
            lines.append(line)

    # Same for ADR units
    lines.append(f"\n\n  Bedingte WK in ADR-Einheiten:")

    for gap_dir, gap_label in [('up', 'GapUp'), ('down', 'GapDn')]:
        lines.append(f"\n  --- {gap_label} ---")
        sub = df[df['gap_direction'] == gap_dir]

        cond_adr = [0.25, 0.50, 1.00, 1.50]
        target_adr = [0.50, 1.00, 1.50, 2.00, 3.00]

        header = f"  {'Bereits':>10}"
        for t in target_adr:
            header += f" | P({t:.2f})"
        header += f" |   N"
        lines.append(header)
        lines.append(f"  {'-'*65}")

        for c in cond_adr:
            reached_c = sub[sub['mfe_live_adr'] >= c]
            n_c = len(reached_c)
            if n_c < 10:
                line = f"  {'>= '+f'{c:.2f}':>10}"
                line += " | N<10" * len(target_adr) + f" | {n_c:>3}"
                lines.append(line)
                continue

            line = f"  {'>= '+f'{c:.2f}':>10}"
            for t in target_adr:
                if t <= c:
                    line += " |   ---"
                else:
                    p = (reached_c['mfe_live_adr'] >= t).sum() / n_c if n_c > 0 else 0
                    line += f" | {p:>5.1%}"
            low_n = " [LOW N]" if n_c < 20 else ""
            line += f" | {n_c:>3}{low_n}"
            lines.append(line)

    output = "\n".join(lines)
    with open('results/d10_mfe_levels.txt', 'w', encoding='utf-8') as f:
        f.write(output)
    print(output)
    print(f"\nSaved to results/d10_mfe_levels.txt", file=sys.stderr)


if __name__ == '__main__':
    main()
