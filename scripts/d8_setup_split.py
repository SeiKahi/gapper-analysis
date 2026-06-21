"""
D8.0 Aufgabe 2: Alle bisherigen Top-Setups, getrennt nach Earnings (IS)
========================================================================
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
    """Sign drift so positive = gap direction."""
    val = row[col]
    if pd.isna(val):
        return np.nan
    return val if row['gap_direction'] == 'up' else -val


def fmt_row(label, earn_sub, non_sub, drift_col='full_drift_signed', rd_col='rest_drift_signed'):
    """Format a comparison row."""
    def stats(df, col):
        if len(df) < 10:
            return "N/A", "N/A", len(df), True
        med = df[col].median()
        ci_lo, ci_hi = bootstrap_median_ci(df[col])
        return f"{med:+.4f}", f"[{ci_lo:+.4f},{ci_hi:+.4f}]", len(df), len(df) < 20

    efd, eci, en, elow = stats(earn_sub, drift_col)
    nfd, nci, nn, nlow = stats(non_sub, drift_col)
    erd, _, _, _ = stats(earn_sub, rd_col)
    nrd, _, _, _ = stats(non_sub, rd_col)

    low_e = " [LOW]" if elow and en >= 10 else ""
    low_n = " [LOW]" if nlow and nn >= 10 else ""

    return (f"  {label:<30} | E fd={efd:>8} rd={erd:>8} N={en:>4}{low_e} "
            f"| NE fd={nfd:>8} rd={nrd:>8} N={nn:>4}{low_n}")


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
    lines.append("AUFGABE 2: ALLE BISHERIGEN TOP-SETUPS, GETRENNT (IS)")
    lines.append("=" * 90)

    # 2a: OD > 0.5 with Gap
    lines.append("\n--- 2a: OD > 0.5 with Gap ---")
    od_with_strong = (meta['od_with_gap'] == True) & (meta['od_strength'].abs() > 0.5)

    for gap_label, gmask in [("GapUp", gap_up), ("GapDn", gap_dn)]:
        mask = od_with_strong & gmask
        lines.append(fmt_row(f"OD>0.5 with {gap_label}", meta[mask & earn], meta[mask & non_earn]))

    # 2b: Combo E (grosse Kerze + OD > 0.5 with)
    lines.append("\n--- 2b: Combo E (grosse Kerze + OD > 0.5 with) ---")
    if 'first_candle_size' in meta.columns:
        big_candle = meta['first_candle_size'] > meta['first_candle_size'].quantile(0.75)
    else:
        big_candle = meta['od_strength'].abs() > 0.7  # proxy

    combo_e = od_with_strong & big_candle
    for gap_label, gmask in [("GapUp", gap_up), ("GapDn", gap_dn)]:
        mask = combo_e & gmask
        lines.append(fmt_row(f"Combo E {gap_label}", meta[mask & earn], meta[mask & non_earn]))

    # 2c: PM/RTH30 < 10% + OD > 0.5
    lines.append("\n--- 2c: PM/RTH30 < 10% + OD > 0.5 ---")
    pm_col = 'pm_rth5' if 'pm_rth5' in meta.columns else 'pm_rth30_computed'
    if pm_col in meta.columns:
        low_pm = meta[pm_col] < 0.10
        od_strong = meta['od_strength'].abs() > 0.5
        mask = low_pm & od_strong
        for gap_label, gmask in [("GapUp", gap_up), ("GapDn", gap_dn)]:
            m = mask & gmask
            lines.append(fmt_row(f"LowPM+OD>0.5 {gap_label}", meta[m & earn], meta[m & non_earn]))

    # 2d: Top-5 Fade-Zellen aus D7.5 (approximation based on gap/pm/rvol buckets)
    lines.append("\n--- 2d: Top Fade-Zellen (GapUp, OD-against, low RVOL, low PM) ---")
    lines.append("  Approximation using: OD against + gap_size_in_adr > median + low rvol_5")

    od_against = meta['od_with_gap'] == False
    if 'rvol_5' in meta.columns:
        rvol_lo = meta['rvol_5'] < meta['rvol_5'].quantile(0.33)
        rvol_hi = meta['rvol_5'] > meta['rvol_5'].quantile(0.67)
    else:
        rvol_lo = meta['rvol_at_time_30min'] < meta['rvol_at_time_30min'].quantile(0.33)
        rvol_hi = meta['rvol_at_time_30min'] > meta['rvol_at_time_30min'].quantile(0.67)

    gap_big = meta['gap_size_in_adr'].abs() > meta['gap_size_in_adr'].abs().quantile(0.67)
    gap_med = (meta['gap_size_in_adr'].abs() > meta['gap_size_in_adr'].abs().quantile(0.33)) & \
              (meta['gap_size_in_adr'].abs() <= meta['gap_size_in_adr'].abs().quantile(0.67))

    if pm_col in meta.columns:
        pm_lo = meta[pm_col] < meta[pm_col].quantile(0.33)
        pm_hi = meta[pm_col] > meta[pm_col].quantile(0.67)
    else:
        pm_lo = pd.Series(True, index=meta.index)
        pm_hi = pd.Series(True, index=meta.index)

    fade_cells = [
        ("GapUp OD-ag LowRV LowPM", gap_up & od_against & rvol_lo & pm_lo),
        ("GapUp OD-ag MedGap LowRV", gap_up & od_against & gap_med & rvol_lo),
        ("GapUp OD-ag BigGap LowRV", gap_up & od_against & gap_big & rvol_lo),
        ("GapDn OD-ag LowRV LowPM", gap_dn & od_against & rvol_lo & pm_lo),
        ("GapDn OD-ag MedGap LowRV", gap_dn & od_against & gap_med & rvol_lo),
    ]

    for label, mask in fade_cells:
        lines.append(fmt_row(label, meta[mask & earn], meta[mask & non_earn]))

    # 2e: Top-5 Continuation-Zellen
    lines.append("\n--- 2e: Top Continuation-Zellen (OD with, high RVOL, strong OD) ---")

    cont_cells = [
        ("GapUp OD-with HiRV", gap_up & (meta['od_with_gap'] == True) & rvol_hi),
        ("GapUp OD-with HiRV BigGap", gap_up & (meta['od_with_gap'] == True) & rvol_hi & gap_big),
        ("GapDn OD-with HiRV", gap_dn & (meta['od_with_gap'] == True) & rvol_hi),
        ("GapDn OD-with HiRV BigGap", gap_dn & (meta['od_with_gap'] == True) & rvol_hi & gap_big),
        ("GapUp OD>0.7 with HiRV", gap_up & (meta['od_with_gap'] == True) & (meta['od_strength'].abs() > 0.7) & rvol_hi),
    ]

    for label, mask in cont_cells:
        lines.append(fmt_row(label, meta[mask & earn], meta[mask & non_earn]))

    # Summary
    lines.append("\n\n--- ZUSAMMENFASSUNG ---")
    lines.append("Kernfrage: Funktionieren Fade-Setups NUR bei Non-Earnings?")
    lines.append("           Continuation-Setups BESSER bei Earnings?")

    text = "\n".join(lines)
    out_file = RESULTS_DIR / "d8_2_setup_split.txt"
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
