"""
D11s DISTANCE: Wie weit laufen Big Runner?

Verwendet BR_RANGE und BR_ENTRY Flags (gleiche Definition wie d11s_fix.py).
Berechnet Distance-Metriken ab Entry (close_935) und ab Open (rth_open).

Output: results/d11s_distance.txt
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
from tqdm import tqdm

RAW_1MIN_DIR = Path('data/raw_1min')


def compute_br_range_vec(df):
    """Vectorized BR_RANGE: ab RTH Open, >= 1.5 ADR."""
    with np.errstate(divide='ignore', invalid='ignore'):
        adr = df['adr_10'].values.astype(float)
        ro = df['rth_open'].values.astype(float)
        rh = df['rth_high'].values.astype(float)
        rl = df['rth_low'].values.astype(float)
        rc = df['rth_close'].values.astype(float)
        is_up = (df['gap_direction'] == 'up').values

        range_up = rh - ro
        cpct_up = np.where(range_up > 0, (rc - ro) / range_up, 0)
        br_up = (range_up / adr >= 1.5) & (cpct_up >= 0.80)

        range_dn = ro - rl
        cpct_dn = np.where(range_dn > 0, (ro - rc) / range_dn, 0)
        br_dn = (range_dn / adr >= 1.5) & (cpct_dn >= 0.80)

        br = np.where(is_up, br_up, br_dn).astype(float)
        nan_mask = (np.isnan(adr) | (adr <= 0) | np.isnan(ro) |
                    np.isnan(rh) | np.isnan(rl) | np.isnan(rc))
        br[nan_mask] = np.nan
    return br


def compute_br_entry(row, bars_rth):
    """BR_ENTRY: ab close_935, Bars >= 09:36, >= 0.7 ADR."""
    adr = row.get('adr_10', np.nan)
    if pd.isna(adr) or adr <= 0:
        return np.nan
    entry = row.get('close_935', np.nan)
    if pd.isna(entry):
        return np.nan

    post = bars_rth[(bars_rth['time_et'] >= '09:36') & (bars_rth['time_et'] <= '15:55')]
    if len(post) == 0:
        return np.nan

    if row['gap_direction'] == 'up':
        rg = post['high'].max() - entry
        if rg <= 0:
            return 0
        cpct = (post.iloc[-1]['close'] - entry) / rg
    else:
        rg = entry - post['low'].min()
        if rg <= 0:
            return 0
        cpct = (entry - post.iloc[-1]['close']) / rg

    return 1 if (rg / adr >= 0.7 and max(0, cpct) >= 0.80) else 0


def build_dataset(base_df, label):
    """Build dataset with BR flags + distance metrics."""
    results = []
    for _, row in tqdm(base_df.iterrows(), total=len(base_df),
                       desc=f"Building {label}", file=sys.stderr):
        ticker = row['ticker']
        date = str(row['date'])
        gap_dir = row['gap_direction']
        adr = row.get('adr_10', np.nan)
        c935 = row.get('close_935', np.nan)
        ro = row.get('rth_open', np.nan)
        rc = row.get('rth_close', np.nan)
        rh = row.get('rth_high', np.nan)
        rl = row.get('rth_low', np.nan)

        br_range_val = row['br_range']
        valid_adr = pd.notna(adr) and adr > 0

        # Read 1min bars for BR_ENTRY + mfe_from_entry
        br_entry_val = np.nan
        mfe_entry = np.nan
        fpath = RAW_1MIN_DIR / ticker / f"{date}.parquet"
        if fpath.exists():
            bars = pd.read_parquet(fpath)
            rth_bars = bars[bars['session'] == 'rth']
            if len(rth_bars) >= 10:
                br_entry_val = compute_br_entry(row, rth_bars)
                post = rth_bars[(rth_bars['time_et'] >= '09:36') &
                                (rth_bars['time_et'] <= '15:55')]
                if len(post) > 0 and pd.notna(c935) and valid_adr:
                    if gap_dir == 'up':
                        mfe_entry = max(0, (post['high'].max() - c935) / adr)
                    else:
                        mfe_entry = max(0, (c935 - post['low'].min()) / adr)

        if pd.isna(br_range_val) and pd.isna(br_entry_val):
            continue

        # rest_drift (ab Entry → EOD)
        rest_drift = np.nan
        if pd.notna(c935) and pd.notna(rc) and valid_adr:
            rest_drift = ((rc - c935) / adr) if gap_dir == 'up' else ((c935 - rc) / adr)

        # full_drift, mfe_open, od_consumed (ab Open)
        full_drift = np.nan
        mfe_open = np.nan
        od_consumed = np.nan
        if pd.notna(ro) and pd.notna(rc) and valid_adr:
            if gap_dir == 'up':
                full_drift = (rc - ro) / adr
                if pd.notna(rh):
                    mfe_open = (rh - ro) / adr
                if pd.notna(c935):
                    od_consumed = (c935 - ro) / adr
            else:
                full_drift = (ro - rc) / adr
                if pd.notna(rl):
                    mfe_open = (ro - rl) / adr
                if pd.notna(c935):
                    od_consumed = (ro - c935) / adr

        # close retentions
        close_ret_entry = np.nan
        if pd.notna(rest_drift) and pd.notna(mfe_entry) and mfe_entry > 0:
            close_ret_entry = rest_drift / mfe_entry
        close_ret_open = np.nan
        if pd.notna(full_drift) and pd.notna(mfe_open) and mfe_open > 0:
            close_ret_open = full_drift / mfe_open

        results.append({
            'ticker': ticker, 'date': date,
            'gap_direction': gap_dir,
            'br_range': int(br_range_val) if pd.notna(br_range_val) else np.nan,
            'br_entry': int(br_entry_val) if pd.notna(br_entry_val) else np.nan,
            'rest_drift': rest_drift,
            'mfe_entry': mfe_entry,
            'close_ret_entry': close_ret_entry,
            'full_drift': full_drift,
            'mfe_open': mfe_open,
            'close_ret_open': close_ret_open,
            'od_consumed': od_consumed,
        })

    return pd.DataFrame(results)


def print_stats_table(lines, df, br_col, gap_dir, glabel, dlabel, include_open):
    """Print stats table for one BR-definition / gap-direction / IS-OOS group."""
    g = df[(df['gap_direction'] == gap_dir) & df[br_col].notna()].copy()
    br = g[g[br_col] == 1]
    nr = g[g[br_col] == 0]
    n_br = len(br)
    n_nr = len(nr)
    low = " [LOW N]" if n_br < 20 else ""

    lines.append(f"\n  {glabel} {dlabel} (BR N={n_br}{low}, Non-Runner N={n_nr}):")
    lines.append(f"    {'Metrik':<22} | {'BR Med':>8} | {'BR Mean':>8} | {'BR Std':>8} "
                 f"| {'BR P25':>8} | {'BR P75':>8} | {'NR Med':>8} | {'NR Mean':>8}")
    lines.append(f"    {'-'*99}")

    def fmt_row(metric, col):
        br_s = br[col].dropna()
        nr_s = nr[col].dropna()
        if len(br_s) > 0:
            parts = (f"    {metric:<22} | {br_s.median():>+8.3f} | {br_s.mean():>+8.3f} | "
                     f"{br_s.std():>8.3f} | {br_s.quantile(0.25):>+8.3f} | "
                     f"{br_s.quantile(0.75):>+8.3f} | ")
        else:
            parts = f"    {metric:<22} | {'—':>8} | {'—':>8} | {'—':>8} | {'—':>8} | {'—':>8} | "
        if len(nr_s) > 0:
            parts += f"{nr_s.median():>+8.3f} | {nr_s.mean():>+8.3f}"
        else:
            parts += f"{'—':>8} | {'—':>8}"
        lines.append(parts)

    lines.append(f"    AB ENTRY (close_935):")
    fmt_row('rest_drift (ADR)', 'rest_drift')
    fmt_row('mfe_entry (ADR)', 'mfe_entry')
    fmt_row('close_ret_entry', 'close_ret_entry')

    if include_open:
        lines.append(f"    AB OPEN (rth_open):")
        fmt_row('full_drift (ADR)', 'full_drift')
        fmt_row('mfe_open (ADR)', 'mfe_open')
        fmt_row('close_ret_open', 'close_ret_open')
        fmt_row('od_consumed (ADR)', 'od_consumed')


def main():
    lines = []
    lines.append("=" * 70)
    lines.append("D11s DISTANCE: WIE WEIT LAUFEN BIG RUNNER?")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Alle Metriken in ADR-Einheiten, positiv = in Gap-Richtung")
    lines.append("AB ENTRY = ab close_935 (was der Trader erlebt)")
    lines.append("AB OPEN  = ab rth_open  (Gesamtbewegung des Tages)")
    lines.append("")
    lines.append("Basis: OD > 0.5 + with_gap")

    # Load metadata
    meta = pd.read_parquet('data/metadata/metadata_v8_5.parquet')
    h1 = meta[(meta['date'] >= '2021-02-21') & (meta['date'] <= '2023-12-31')].copy()
    h2 = meta[(meta['date'] >= '2024-01-01')].copy()

    base_h1 = h1[(h1['od_strength'] > 0.5) & (h1['od_direction'] == 'with_gap')].copy()
    base_h2 = h2[(h2['od_strength'] > 0.5) & (h2['od_direction'] == 'with_gap')].copy()
    print(f"H1: {len(base_h1)}, H2: {len(base_h2)}", file=sys.stderr)

    # Pre-compute BR_RANGE
    base_h1['br_range'] = compute_br_range_vec(base_h1)
    base_h2['br_range'] = compute_br_range_vec(base_h2)

    # Build datasets
    df_is = build_dataset(base_h1, "IS")
    df_oos = build_dataset(base_h2, "OOS")
    print(f"IS: {len(df_is)} trades, OOS: {len(df_oos)} trades", file=sys.stderr)

    # === BR_RANGE ===
    lines.append(f"\n\n{'='*70}")
    lines.append("=== BR_RANGE (ab Open, >= 1.5 ADR) ===")
    lines.append(f"{'='*70}")

    for gap_dir, glabel in [('up', 'GapUp'), ('down', 'GapDn')]:
        for dlabel, ddf in [('IS', df_is), ('OOS', df_oos)]:
            print_stats_table(lines, ddf, 'br_range', gap_dir, glabel, dlabel,
                              include_open=True)

    # === BR_ENTRY ===
    lines.append(f"\n\n{'='*70}")
    lines.append("=== BR_ENTRY (ab Entry, >= 0.7 ADR) ===")
    lines.append(f"{'='*70}")

    for gap_dir, glabel in [('up', 'GapUp'), ('down', 'GapDn')]:
        for dlabel, ddf in [('IS', df_is), ('OOS', df_oos)]:
            print_stats_table(lines, ddf, 'br_entry', gap_dir, glabel, dlabel,
                              include_open=False)

    # === VERGLEICH ===
    lines.append(f"\n\n{'='*70}")
    lines.append("VERGLEICH: BIG RUNNER DISTANCE (nur BRs)")
    lines.append(f"{'='*70}")

    cw = 22  # column width for range strings

    comp_header = (f"  {'Metrik':<24} | {'RANGE GapUp':<{cw}} | {'ENTRY GapUp':<{cw}} "
                   f"| {'RANGE GapDn':<{cw}} | {'ENTRY GapDn':<{cw}}")

    def stat_str(ddf, br_col, gap_dir, col):
        """Return 'Median [Mean-Std .. Mean+Std]' string."""
        g = ddf[(ddf['gap_direction'] == gap_dir) & (ddf[br_col] == 1)]
        s = g[col].dropna()
        if len(s) == 0:
            return "—"
        med = s.median()
        mean = s.mean()
        std = s.std()
        lo = mean - std
        hi = mean + std
        return f"{med:+.2f} [{lo:+.2f}..{hi:+.2f}]"

    def med_only(ddf, br_col, gap_dir, col):
        g = ddf[(ddf['gap_direction'] == gap_dir) & (ddf[br_col] == 1)]
        s = g[col].dropna()
        if len(s) == 0:
            return "—"
        med = s.median()
        mean = s.mean()
        std = s.std()
        lo = mean - std
        hi = mean + std
        return f"{med:+.2f} [{lo:+.2f}..{hi:+.2f}]"

    for period_label, ddf in [('OOS', df_oos), ('IS', df_is)]:
        lines.append(f"\n  {period_label} — Format: Median [Mean-1Std .. Mean+1Std]")
        lines.append(comp_header)
        lines.append(f"  {'-'*(28 + 4*(cw+3))}")

        for metric, col in [('rest_drift (ADR)', 'rest_drift'),
                             ('mfe_entry (ADR)', 'mfe_entry'),
                             ('retention_entry', 'close_ret_entry')]:
            vru = stat_str(ddf, 'br_range', 'up', col)
            veu = stat_str(ddf, 'br_entry', 'up', col)
            vrd = stat_str(ddf, 'br_range', 'down', col)
            ved = stat_str(ddf, 'br_entry', 'down', col)
            lines.append(f"  {metric:<24} | {vru:<{cw}} | {veu:<{cw}} "
                         f"| {vrd:<{cw}} | {ved:<{cw}}")

        for metric, col in [('od_consumed (ADR)', 'od_consumed'),
                             ('full_drift (ADR)', 'full_drift')]:
            vru = stat_str(ddf, 'br_range', 'up', col)
            vrd = stat_str(ddf, 'br_range', 'down', col)
            lines.append(f"  {metric:<24} | {vru:<{cw}} | {'—':<{cw}} "
                         f"| {vrd:<{cw}} | {'—':<{cw}}")

    # Write output
    output = "\n".join(lines)
    os.makedirs("results", exist_ok=True)
    with open('results/d11s_distance.txt', 'w', encoding='utf-8') as f:
        f.write(output)
    print(output)
    print(f"\nSaved to results/d11s_distance.txt", file=sys.stderr)


if __name__ == '__main__':
    main()
