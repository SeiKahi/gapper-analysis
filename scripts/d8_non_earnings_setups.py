"""
D8.0 Aufgabe 6: Non-Earnings Top-Setups (IS)
==============================================
Separate 9:35-Dreieck fuer Non-Earnings, Continuation fuer Earnings.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

BASE_DIR = Path(__file__).parent.parent
META_DIR = BASE_DIR / "data" / "metadata"
RESULTS_DIR = BASE_DIR / "results"
IS_END = "2023-12-31"


def bootstrap_median_ci(data, n_boot=1000):
    data = data.dropna().values
    if len(data) < 5:
        return np.nan, np.nan
    rng = np.random.RandomState(42)
    meds = [np.median(rng.choice(data, len(data), replace=True)) for _ in range(n_boot)]
    return np.percentile(meds, 2.5), np.percentile(meds, 97.5)


def compute_od_with_gap(row):
    od = row.get('od_direction', None)
    if pd.isna(od) or od is None:
        return np.nan
    return od == 'with_gap'


def sign_drift(row, col='full_drift'):
    val = row[col]
    if pd.isna(val):
        return np.nan
    return val if row['gap_direction'] == 'up' else -val


def make_buckets(series, n=3, labels=None):
    """Create n quantile buckets."""
    try:
        return pd.qcut(series, n, labels=labels, duplicates='drop')
    except ValueError:
        return pd.cut(series, n, labels=labels, duplicates='drop')


def main():
    meta = pd.read_parquet(META_DIR / "metadata_v8.parquet")
    meta['date_dt'] = pd.to_datetime(meta['date'])
    meta = meta[meta['date_dt'] <= IS_END].copy()

    meta['od_with_gap'] = meta.apply(compute_od_with_gap, axis=1)
    meta['full_drift_signed'] = meta.apply(lambda r: sign_drift(r, 'full_drift'), axis=1)
    meta['rest_drift_signed'] = meta.apply(lambda r: sign_drift(r, 'rest_drift'), axis=1)

    earn = meta['is_earnings'] == True
    non_earn = (~meta['is_earnings']) & (~meta['earnings_unknown'])
    gap_up = meta['gap_direction'] == 'up'
    gap_dn = meta['gap_direction'] == 'down'

    lines = []
    lines.append("=" * 90)
    lines.append("AUFGABE 6: NON-EARNINGS TOP-SETUPS (IS)")
    lines.append("=" * 90)

    # 6a: 9:35-Dreieck NUR fuer Non-Earnings
    lines.append("\n--- 6a: 9:35-Dreieck (PM/RTH5 x RVOL_5 x Gap_in_ADR) NUR Non-Earnings ---")

    ne = meta[non_earn].copy()
    pm_col = 'pm_rth5' if 'pm_rth5' in ne.columns else 'pm_rth30_computed'
    rvol_col = 'rvol_5' if 'rvol_5' in ne.columns else 'rvol_at_time_30min'

    for gap_label, gmask in [("GapUp", gap_up), ("GapDn", gap_dn)]:
        sub = ne[gmask[ne.index]].copy()
        if len(sub) < 50:
            lines.append(f"\n  {gap_label}: N={len(sub)} [INSUFFICIENT]")
            continue

        lines.append(f"\n  {gap_label} Non-Earnings (N={len(sub)}):")

        # Create 3x3x3 buckets
        valid = sub.dropna(subset=[pm_col, rvol_col, 'gap_size_in_adr']).copy()
        if len(valid) < 50:
            lines.append(f"    After dropna: N={len(valid)} [INSUFFICIENT]")
            continue

        valid['pm_bucket'] = make_buckets(valid[pm_col], 3, ['PM_LO', 'PM_MID', 'PM_HI'])
        valid['rv_bucket'] = make_buckets(valid[rvol_col], 3, ['RV_LO', 'RV_MID', 'RV_HI'])
        valid['gap_bucket'] = make_buckets(valid['gap_size_in_adr'].abs(), 3, ['GAP_SM', 'GAP_MD', 'GAP_LG'])

        # Report each cell
        header = f"    {'PM':>6} {'RV':>6} {'Gap':>6} | {'fd med':>10} {'rd med':>10} | {'N':>5}"
        lines.append(header)
        lines.append("    " + "-" * (len(header) - 4))

        best_fade = None
        best_cont = None

        for pm in ['PM_LO', 'PM_MID', 'PM_HI']:
            for rv in ['RV_LO', 'RV_MID', 'RV_HI']:
                for gap in ['GAP_SM', 'GAP_MD', 'GAP_LG']:
                    cell = valid[(valid['pm_bucket'] == pm) &
                                 (valid['rv_bucket'] == rv) &
                                 (valid['gap_bucket'] == gap)]
                    n = len(cell)
                    if n < 5:
                        continue

                    fd = cell['full_drift_signed'].median()
                    rd = cell['rest_drift_signed'].median()
                    tag = " [LOW]" if n < 20 else ""

                    lines.append(f"    {pm:>6} {rv:>6} {gap:>6} | {fd:>+10.4f} {rd:>+10.4f} | {n:>5}{tag}")

                    # Track best fade (most negative drift) and best continuation (most positive)
                    if n >= 20:
                        if best_fade is None or fd < best_fade[3]:
                            best_fade = (pm, rv, gap, fd, rd, n)
                        if best_cont is None or fd > best_cont[3]:
                            best_cont = (pm, rv, gap, fd, rd, n)

        if best_fade:
            lines.append(f"\n    Best Fade Cell: {best_fade[0]} {best_fade[1]} {best_fade[2]} fd={best_fade[3]:+.4f} N={best_fade[5]}")
        if best_cont:
            lines.append(f"    Best Cont Cell: {best_cont[0]} {best_cont[1]} {best_cont[2]} fd={best_cont[3]:+.4f} N={best_cont[5]}")

    # Compare with ALL gappers triangle
    lines.append("\n\n  Vergleich: Non-Earnings vs ALL (Fade-Zellen stabiler bei NE?)")
    for gap_label, gmask in [("GapUp", gap_up), ("GapDn", gap_dn)]:
        all_sub = meta[gmask].dropna(subset=[pm_col, rvol_col, 'gap_size_in_adr']).copy()
        ne_sub = ne[gmask[ne.index]].dropna(subset=[pm_col, rvol_col, 'gap_size_in_adr']).copy()

        if len(ne_sub) < 20 or len(all_sub) < 20:
            continue

        # Low PM + Low RV bucket (typical fade cell)
        for sub_label, sub_df in [("ALL", all_sub), ("Non-E", ne_sub)]:
            lo_pm = sub_df[pm_col] < sub_df[pm_col].quantile(0.33)
            lo_rv = sub_df[rvol_col] < sub_df[rvol_col].quantile(0.33)
            cell = sub_df[lo_pm & lo_rv]
            if len(cell) >= 10:
                fd = cell['full_drift_signed'].median()
                ci_lo, ci_hi = bootstrap_median_ci(cell['full_drift_signed'])
                lines.append(f"    {gap_label} {sub_label} PM_LO+RV_LO: fd={fd:+.4f} [{ci_lo:+.4f},{ci_hi:+.4f}] N={len(cell)}")

    # 6b: Continuation-Setups NUR fuer Earnings
    lines.append("\n\n--- 6b: Continuation-Setups NUR Earnings ---")

    earn_df = meta[earn].copy()
    for gap_label, gmask in [("GapUp", gap_up), ("GapDn", gap_dn)]:
        sub = earn_df[gmask[earn_df.index]].copy()
        lines.append(f"\n  {gap_label} Earnings (N={len(sub)}):")

        # OD with + strong
        od_with_strong = (sub['od_with_gap'] == True) & (sub['od_strength'].abs() > 0.5)
        cell = sub[od_with_strong]
        if len(cell) >= 5:
            fd = cell['full_drift_signed'].median()
            rd = cell['rest_drift_signed'].median()
            ci_lo, ci_hi = bootstrap_median_ci(cell['full_drift_signed'])
            tag = " [LOW]" if len(cell) < 20 else ""
            lines.append(f"    OD>0.5 with: fd={fd:+.4f} [{ci_lo:+.4f},{ci_hi:+.4f}] rd={rd:+.4f} N={len(cell)}{tag}")

        # OD with + any strength
        od_with_any = sub['od_with_gap'] == True
        cell = sub[od_with_any]
        if len(cell) >= 10:
            fd = cell['full_drift_signed'].median()
            lines.append(f"    OD any with: fd={fd:+.4f} N={len(cell)}")

        # High RVOL + OD with
        if rvol_col in sub.columns:
            hi_rv = sub[rvol_col] > sub[rvol_col].quantile(0.67)
            cell = sub[od_with_any & hi_rv]
            if len(cell) >= 5:
                fd = cell['full_drift_signed'].median()
                lines.append(f"    OD with + HiRV: fd={fd:+.4f} N={len(cell)}")

    # 6c: Separate Entscheidungsbaeume
    lines.append("\n\n--- 6c: Entscheidungsbaeume ---")

    lines.append("\n  WENN EARNINGS:")
    lines.append("    → OD with Gap + stark (>0.5 ADR) → CONTINUATION (in Gap-Richtung)")
    lines.append("    → OD against Gap → VORSICHT! Oft Flush, nicht shorten")
    lines.append("    → Warten auf 10:00 Reversal bei OD-against")

    lines.append("\n  WENN NON-EARNINGS:")
    lines.append("    → OD against Gap + PM_LO + RV_LO → FADE moeglich")
    lines.append("    → OD with Gap → Continuation, aber schwaecher als Earnings")
    lines.append("    → 9:35-Dreieck funktioniert (besser) bei Non-Earnings")

    text = "\n".join(lines)
    out_file = RESULTS_DIR / "d8_6_non_earnings_setups.txt"
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(text)
    print(text.replace('\u2192', '->'))


if __name__ == "__main__":
    main()
