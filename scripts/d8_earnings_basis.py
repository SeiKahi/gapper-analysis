"""
D8.0 Aufgabe 1: Earnings vs Non-Earnings Basis-Statistik (IS)
==============================================================
Vergleiche Earnings-Gapper und Non-Earnings-Gapper auf allen Basismetriken.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm

BASE_DIR = Path(__file__).parent.parent
META_DIR = BASE_DIR / "data" / "metadata"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

IS_END = "2023-12-31"


def bootstrap_ci(data, stat_func=np.nanmedian, n_boot=1000, ci=0.95):
    """Bootstrap confidence interval."""
    data = data.dropna().values
    if len(data) < 5:
        return np.nan, np.nan
    rng = np.random.RandomState(42)
    stats = []
    for _ in range(n_boot):
        sample = rng.choice(data, size=len(data), replace=True)
        stats.append(stat_func(sample))
    alpha = (1 - ci) / 2
    return np.percentile(stats, 100*alpha), np.percentile(stats, 100*(1-alpha))


def fmt_ci(val, ci_lo, ci_hi, decimals=4):
    """Format value with CI."""
    if np.isnan(val):
        return "N/A"
    return f"{val:.{decimals}f} [{ci_lo:.{decimals}f}, {ci_hi:.{decimals}f}]"


def compute_od_with_gap(row):
    """Check if OD direction is 'with gap'."""
    od = row.get('od_direction', None)
    if pd.isna(od) or od is None:
        return np.nan
    return od == 'with_gap'


def main():
    meta = pd.read_parquet(META_DIR / "metadata_v8.parquet")
    meta['date_dt'] = pd.to_datetime(meta['date'])

    # IS only
    meta = meta[meta['date_dt'] <= IS_END].copy()
    print(f"IS dataset: {len(meta)} rows", file=sys.stderr)

    # Compute OD with gap flag
    meta['od_with_gap'] = meta.apply(compute_od_with_gap, axis=1)

    # Split groups
    earn_mask = meta['is_earnings'] == True
    non_earn_mask = (~meta['is_earnings']) & (~meta['earnings_unknown'])
    unknown_mask = meta['earnings_unknown'] == True

    gap_up = meta['gap_direction'] == 'up'
    gap_dn = meta['gap_direction'] == 'down'

    lines = []
    lines.append("=" * 80)
    lines.append("AUFGABE 1: EARNINGS vs NON-EARNINGS BASIS-STATISTIK (IS)")
    lines.append("=" * 80)

    # 1a: Grundlegende Verteilung
    lines.append("\n--- 1a: Grundlegende Verteilung ---\n")

    groups = {
        'Earnings GapUp': meta[earn_mask & gap_up],
        'Earnings GapDn': meta[earn_mask & gap_dn],
        'Non-E GapUp': meta[non_earn_mask & gap_up],
        'Non-E GapDn': meta[non_earn_mask & gap_dn],
        'Unknown GapUp': meta[unknown_mask & gap_up],
        'Unknown GapDn': meta[unknown_mask & gap_dn],
    }

    metrics = [
        ('N', lambda df: len(df), False),
        ('Median Gap%', lambda df: df['gap_pct'].abs().median(), True),
        ('Median Gap in ADR', lambda df: df['gap_size_in_adr'].abs().median(), True),
        ('Median RVOL_30', lambda df: df['rvol_at_time_30min'].median(), True),
        ('Median RVOL_5', lambda df: df['rvol_5'].median() if 'rvol_5' in df.columns else np.nan, True),
        ('Median PM/RTH30', lambda df: df['pm_rth30_computed'].median() if 'pm_rth30_computed' in df.columns else np.nan, True),
        ('Median PM/RTH5', lambda df: df['pm_rth5'].median() if 'pm_rth5' in df.columns else np.nan, True),
        ('Median OD Staerke', lambda df: df['od_strength'].abs().median(), True),
        ('OD with Gap %', lambda df: 100 * df['od_with_gap'].mean() if df['od_with_gap'].notna().any() else np.nan, False),
        ('Median full_drift', lambda df: df['full_drift'].median(), True),
        ('Median rest_drift', lambda df: df['rest_drift'].median(), True),
        ('Fill-Rate %', lambda df: 100 * df['gap_filled'].mean() if 'gap_filled' in df.columns and df['gap_filled'].notna().any() else np.nan, False),
    ]

    # Header
    header = f"{'Metrik':<22}"
    for gname in groups:
        header += f" | {gname:>16}"
    lines.append(header)
    lines.append("-" * len(header))

    for mname, mfunc, do_ci in metrics:
        row_str = f"{mname:<22}"
        for gname, gdf in groups.items():
            if len(gdf) == 0:
                row_str += f" | {'N/A':>16}"
                continue
            val = mfunc(gdf)
            if mname == 'N':
                row_str += f" | {int(val):>16}"
            elif do_ci and not np.isnan(val):
                row_str += f" | {val:>16.4f}"
            elif not np.isnan(val):
                row_str += f" | {val:>16.1f}"
            else:
                row_str += f" | {'N/A':>16}"
        lines.append(row_str)

    # 1b: OD-Richtung bei Earnings vs Non-Earnings
    lines.append("\n\n--- 1b: OD-Richtung bei Earnings vs Non-Earnings ---\n")

    for earn_label, emask in [("Earnings", earn_mask), ("Non-Earnings", non_earn_mask), ("Unknown", unknown_mask)]:
        sub = meta[emask]
        if len(sub) == 0:
            continue
        od_valid = sub['od_with_gap'].dropna()
        od_with = (od_valid == True).sum()
        od_against = (od_valid == False).sum()
        od_na = sub['od_with_gap'].isna().sum()
        total_valid = od_with + od_against
        lines.append(f"  {earn_label} (N={len(sub)}):")
        if total_valid > 0:
            lines.append(f"    OD with Gap:    {od_with:5d} ({100*od_with/total_valid:.1f}%)")
            lines.append(f"    OD against Gap: {od_against:5d} ({100*od_against/total_valid:.1f}%)")
        lines.append(f"    OD unknown:     {od_na:5d}")

        # Split by gap direction
        for gap_label, gmask in [("GapUp", gap_up), ("GapDn", gap_dn)]:
            sub2 = meta[emask & gmask]
            if len(sub2) == 0:
                continue
            od_v2 = sub2['od_with_gap'].dropna()
            ow = (od_v2 == True).sum()
            oa = (od_v2 == False).sum()
            tv = ow + oa
            if tv > 0:
                lines.append(f"    {gap_label}: with={ow} ({100*ow/tv:.1f}%), against={oa} ({100*oa/tv:.1f}%)")
        lines.append("")

    # 1c: Drift-Vergleich
    lines.append("\n--- 1c: Drift-Vergleich ---\n")
    lines.append("Drift in ADR, signiert in Gap-Richtung (positiv = Gap-Richtung)")
    lines.append("")

    # Compute gap-signed drifts
    meta['full_drift_signed'] = meta.apply(
        lambda r: r['full_drift'] if r['gap_direction'] == 'up' else -r['full_drift']
        if pd.notna(r['full_drift']) else np.nan, axis=1)
    meta['rest_drift_signed'] = meta.apply(
        lambda r: r['rest_drift'] if r['gap_direction'] == 'up' else -r['rest_drift']
        if pd.notna(r['rest_drift']) else np.nan, axis=1)

    setups = [
        ('Alle Gapper GapUp', gap_up, None),
        ('Alle Gapper GapDn', gap_dn, None),
        ('OD with, GapUp', gap_up, meta['od_with_gap'] == True),
        ('OD with, GapDn', gap_dn, meta['od_with_gap'] == True),
        ('OD against, GapUp', gap_up, meta['od_with_gap'] == False),
        ('OD against, GapDn', gap_dn, meta['od_with_gap'] == False),
        ('OD against+stark GapUp', gap_up, (meta['od_with_gap'] == False) & (meta['od_strength'].abs() > 0.5)),
        ('OD against+stark GapDn', gap_dn, (meta['od_with_gap'] == False) & (meta['od_strength'].abs() > 0.5)),
    ]

    header = f"{'Setup':<25} | {'Earnings fd':>14} | {'Non-E fd':>14} | {'Delta':>10} | {'Earn N':>7} | {'NonE N':>7}"
    lines.append(header)
    lines.append("-" * len(header))

    for sname, gap_mask, extra_mask in setups:
        combined = gap_mask
        if extra_mask is not None:
            combined = combined & extra_mask

        earn_sub = meta[combined & earn_mask]
        non_sub = meta[combined & non_earn_mask]

        earn_fd = earn_sub['full_drift_signed'].median() if len(earn_sub) >= 10 else np.nan
        non_fd = non_sub['full_drift_signed'].median() if len(non_sub) >= 10 else np.nan

        delta = earn_fd - non_fd if not (np.isnan(earn_fd) or np.isnan(non_fd)) else np.nan

        e_str = f"{earn_fd:+.4f}" if not np.isnan(earn_fd) else "N/A"
        n_str = f"{non_fd:+.4f}" if not np.isnan(non_fd) else "N/A"
        d_str = f"{delta:+.4f}" if not np.isnan(delta) else "N/A"
        n_tag_e = f"{'[LOW]' if len(earn_sub) < 20 else ''}"
        n_tag_n = f"{'[LOW]' if len(non_sub) < 20 else ''}"

        lines.append(f"{sname:<25} | {e_str:>14} | {n_str:>14} | {d_str:>10} | {len(earn_sub):>5}{n_tag_e:>2} | {len(non_sub):>5}{n_tag_n:>2}")

    # Same for rest_drift
    lines.append("\n\nRest Drift (9:35 → Close) in Gap-Richtung:")
    header = f"{'Setup':<25} | {'Earnings rd':>14} | {'Non-E rd':>14} | {'Delta':>10} | {'Earn N':>7} | {'NonE N':>7}"
    lines.append(header)
    lines.append("-" * len(header))

    for sname, gap_mask, extra_mask in setups:
        combined = gap_mask
        if extra_mask is not None:
            combined = combined & extra_mask

        earn_sub = meta[combined & earn_mask]
        non_sub = meta[combined & non_earn_mask]

        earn_rd = earn_sub['rest_drift_signed'].median() if len(earn_sub) >= 10 else np.nan
        non_rd = non_sub['rest_drift_signed'].median() if len(non_sub) >= 10 else np.nan
        delta = earn_rd - non_rd if not (np.isnan(earn_rd) or np.isnan(non_rd)) else np.nan

        e_str = f"{earn_rd:+.4f}" if not np.isnan(earn_rd) else "N/A"
        n_str = f"{non_rd:+.4f}" if not np.isnan(non_rd) else "N/A"
        d_str = f"{delta:+.4f}" if not np.isnan(delta) else "N/A"

        lines.append(f"{sname:<25} | {e_str:>14} | {n_str:>14} | {d_str:>10} | {len(earn_sub):>5} | {len(non_sub):>5}")

    text = "\n".join(lines)
    out_file = RESULTS_DIR / "d8_1_earnings_basis.txt"
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(text)
    # Replace unicode arrows for console output
    print(text.replace('\u2192', '->'))


if __name__ == "__main__":
    main()
