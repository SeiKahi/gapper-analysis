"""
D11s FIX: Dual Big Runner Definition (BR_RANGE + BR_ENTRY)

Fixes bug where d11s compute_br_flag() used close_935 instead of rth_open,
halving BR rate (19.5% -> 10.2%).

Two definitions computed in parallel:
  BR_RANGE: ab RTH Open, Range >= 1.5 ADR, alle RTH Bars (= Original D11)
  BR_ENTRY: ab Entry (close_935), Bars ab 09:36, Range >= 0.7 ADR

Output: results/d11s_fix_synthese.txt
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import sys
import os
from tqdm import tqdm

RAW_1MIN_DIR = Path('data/raw_1min')


def compute_br_range_vec(df):
    """Vectorized BR_RANGE: ab RTH Open, >= 1.5 ADR, alle RTH Bars."""
    with np.errstate(divide='ignore', invalid='ignore'):
        adr = df['adr_10'].values.astype(float)
        rth_open = df['rth_open'].values.astype(float)
        rth_high = df['rth_high'].values.astype(float)
        rth_low = df['rth_low'].values.astype(float)
        rth_close = df['rth_close'].values.astype(float)
        is_up = (df['gap_direction'] == 'up').values

        range_up = rth_high - rth_open
        cpct_up = np.where(range_up > 0, (rth_close - rth_open) / range_up, 0)
        br_up = (range_up / adr >= 1.5) & (cpct_up >= 0.80)

        range_dn = rth_open - rth_low
        cpct_dn = np.where(range_dn > 0, (rth_open - rth_close) / range_dn, 0)
        br_dn = (range_dn / adr >= 1.5) & (cpct_dn >= 0.80)

        br = np.where(is_up, br_up, br_dn).astype(float)
        nan_mask = (np.isnan(adr) | (adr <= 0) | np.isnan(rth_open) |
                    np.isnan(rth_high) | np.isnan(rth_low) | np.isnan(rth_close))
        br[nan_mask] = np.nan

    return br


def compute_br_entry(row, bars_rth):
    """BR_ENTRY: ab close_935, Bars >= 09:36, Range >= 0.7 ADR."""
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
        range_gap = post['high'].max() - entry
        if range_gap <= 0:
            return 0
        close_pct = (post.iloc[-1]['close'] - entry) / range_gap
    else:
        range_gap = entry - post['low'].min()
        if range_gap <= 0:
            return 0
        close_pct = (entry - post.iloc[-1]['close']) / range_gap

    range_adr = range_gap / adr
    return 1 if (range_adr >= 0.7 and max(0, close_pct) >= 0.80) else 0


def build_dataset(base_df, label):
    """Build dataset with both BR flags + features."""
    results = []
    for _, row in tqdm(base_df.iterrows(), total=len(base_df),
                       desc=f"Building {label}", file=sys.stderr):
        ticker = row['ticker']
        date = str(row['date'])

        br_range_val = row['br_range']

        br_entry_val = np.nan
        fpath = RAW_1MIN_DIR / ticker / f"{date}.parquet"
        if fpath.exists():
            bars = pd.read_parquet(fpath)
            rth = bars[bars['session'] == 'rth']
            if len(rth) >= 10:
                br_entry_val = compute_br_entry(row, rth)

        if pd.isna(br_range_val) and pd.isna(br_entry_val):
            continue

        rec = {
            'ticker': ticker, 'date': date,
            'gap_direction': row['gap_direction'],
            'br_range': int(br_range_val) if pd.notna(br_range_val) else np.nan,
            'br_entry': int(br_entry_val) if pd.notna(br_entry_val) else np.nan,
            'od_strength': row['od_strength'],
            'rvol_5': row.get('rvol_5', np.nan),
            'rvol_open_30min': row.get('rvol_open_30min', np.nan),
            'gap_size_in_adr': row.get('gap_size_in_adr', np.nan),
            'pm_rth5': row.get('pm_rth5', np.nan),
            'pm_rth30_computed': row.get('pm_rth30_computed', np.nan),
            'first_candle_size': row.get('first_candle_size', np.nan),
            'od_body_pct': row.get('od_body_pct', np.nan),
        }

        adr = row.get('adr_10', np.nan)
        if pd.notna(adr) and adr > 0:
            c935 = row.get('close_935', np.nan)
            c1000 = row.get('close_1000', np.nan)
            if pd.notna(c935) and pd.notna(c1000):
                drift_raw = (c1000 - c935) / adr
                rec['drift_30min'] = drift_raw if row['gap_direction'] == 'up' else -drift_raw

        results.append(rec)

    return pd.DataFrame(results)


PARAMS_TO_TEST = [
    ('od_strength', 'OD_strength'),
    ('rvol_5', 'RVOL_5'),
    ('rvol_open_30min', 'RVOL_30'),
    ('gap_size_in_adr', 'Gap_in_ADR'),
    ('pm_rth5', 'PM/RTH5'),
    ('pm_rth30_computed', 'PM/RTH30'),
    ('first_candle_size', 'first_candle_size'),
    ('od_body_pct', 'od_body_pct'),
    ('drift_30min', '30min_drift'),
]

RULES_935 = [
    ('RV5>5+OD>1+Gap>2', 'up',
     lambda d: d[(d['gap_direction'] == 'up') & (d['rvol_5'] > 5) &
                 (d['od_strength'] > 1.0) & (d['gap_size_in_adr'].abs() > 2)]),
    ('RV5>5+OD>1', 'down',
     lambda d: d[(d['gap_direction'] == 'down') & (d['rvol_5'] > 5) &
                 (d['od_strength'] > 1.0)]),
]

RULES_1000 = [
    ('drift>0.25+RV30>5+OD>1', 'up',
     lambda d: d[(d['gap_direction'] == 'up') & (d['drift_30min'] > 0.25) &
                 (d['rvol_open_30min'] > 5) & (d['od_strength'] > 1.0)]),
    ('drift>0.50+RV30>5', 'up',
     lambda d: d[(d['gap_direction'] == 'up') & (d['drift_30min'] > 0.50) &
                 (d['rvol_open_30min'] > 5)]),
    ('drift>0.50', 'down',
     lambda d: d[(d['gap_direction'] == 'down') & (d['drift_30min'] > 0.50)]),
]


def run_analysis(df_is, df_oos, br_col, def_label, lines):
    """Run full analysis for one BR definition. Returns top_rules dict."""
    lines.append(f"\n\n{'='*70}")
    lines.append(f"=== {def_label} ===")
    lines.append(f"{'='*70}")

    # --- Basisstatistik ---
    lines.append(f"\n  Basisstatistik:")
    for dlabel, ddf in [('IS', df_is), ('OOS', df_oos)]:
        valid = ddf[ddf[br_col].notna()]
        for gap_dir, glabel in [('up', 'GapUp'), ('down', 'GapDn')]:
            g = valid[valid['gap_direction'] == gap_dir]
            n = len(g)
            n_br = int(g[br_col].sum())
            rate = n_br / n * 100 if n > 0 else 0
            lines.append(f"    {dlabel} {glabel}: BR = {n_br}/{n} = {rate:.1f}%")

    # --- Spearman Ranking (IS) ---
    lines.append(f"\n  Ranking nach Trennschaerfe (Spearman rho, IS):")

    valid_is = df_is[df_is[br_col].notna()]
    for gap_dir, glabel in [('up', 'GapUp'), ('down', 'GapDn')]:
        g = valid_is[valid_is['gap_direction'] == gap_dir].copy()
        y = g[br_col].astype(int)

        rankings = []
        for col, label in PARAMS_TO_TEST:
            if col not in g.columns:
                continue
            valid_mask = g[col].notna()
            if valid_mask.sum() < 30:
                continue
            rho, pval = stats.spearmanr(g.loc[valid_mask, col], y[valid_mask])
            rankings.append((label, rho, pval, valid_mask.sum()))

        rankings.sort(key=lambda x: abs(x[1]), reverse=True)

        lines.append(f"\n    --- {glabel} (Base BR Rate: {y.mean():.1%}, N={len(g)}) ---")
        lines.append(f"    {'Rank':>4} | {'Parameter':<18} | {'rho':>8} | {'p-value':>10} | {'N':>5}")
        lines.append(f"    {'-'*55}")
        for i, (label, rho, pval, n) in enumerate(rankings):
            sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else ""))
            lines.append(f"    {i+1:>4} | {label:<18} | {rho:>+8.3f} | {pval:>10.4f} | {n:>5} {sig}")

    # --- OOS Regeln (9:35) ---
    lines.append(f"\n  OOS Regeln (ab 9:35):")

    top_rules = {}

    for rlabel, gap_dir, rfn in RULES_935:
        glabel = 'GapUp' if gap_dir == 'up' else 'GapDn'
        lines.append(f"\n    {glabel}: {rlabel}:")
        for dlabel, ddf in [('IS', df_is), ('OOS', df_oos)]:
            valid = ddf[ddf[br_col].notna()]
            sub = rfn(valid)
            n = len(sub)
            if n < 5:
                lines.append(f"      {dlabel}: N={n} [zu wenig]")
                continue
            br_rate = sub[br_col].mean()
            base_rate = valid[valid['gap_direction'] == gap_dir][br_col].mean()
            lift = br_rate / base_rate if base_rate > 0 else 0
            low = " [LOW N]" if n < 20 else ""
            lines.append(f"      {dlabel}: BR={br_rate:.1%} (Basis: {base_rate:.1%}), Lift={lift:.1f}x, N={n}{low}")

            if dlabel == 'OOS':
                key = glabel
                if key not in top_rules or br_rate > top_rules[key][1]:
                    top_rules[key] = (f"{glabel}: {rlabel}", br_rate, n, lift)

    # 10:00 rules
    if 'drift_30min' in df_is.columns:
        lines.append(f"\n    --- 10:00 Regeln ---")
        for rlabel, gap_dir, rfn in RULES_1000:
            glabel = 'GapUp' if gap_dir == 'up' else 'GapDn'
            lines.append(f"\n    {glabel}: {rlabel}:")
            for dlabel, ddf in [('IS', df_is), ('OOS', df_oos)]:
                valid = ddf[ddf[br_col].notna()]
                sub = rfn(valid)
                n = len(sub)
                if n < 5:
                    lines.append(f"      {dlabel}: N={n} [zu wenig]")
                    continue
                br_rate = sub[br_col].mean()
                base_rate = valid[valid['gap_direction'] == gap_dir][br_col].mean()
                lift = br_rate / base_rate if base_rate > 0 else 0
                low = " [LOW N]" if n < 20 else ""
                lines.append(f"      {dlabel}: BR={br_rate:.1%} (Basis: {base_rate:.1%}), Lift={lift:.1f}x, N={n}{low}")

                if dlabel == 'OOS':
                    key = glabel
                    if key not in top_rules or br_rate > top_rules[key][1]:
                        top_rules[key] = (f"{glabel}: {rlabel}", br_rate, n, lift)

    return top_rules


def main():
    lines = []
    lines.append("=" * 70)
    lines.append("D11s FIX: DUAL BIG RUNNER DEFINITION")
    lines.append("=" * 70)
    lines.append("")
    lines.append("BR_RANGE: ab RTH Open, Range >= 1.5 ADR, alle RTH Bars (= Original D11)")
    lines.append("BR_ENTRY: ab Entry (close_935), Bars ab 09:36, Range >= 0.7 ADR")
    lines.append("")
    lines.append("Basis: OD > 0.5 + with_gap, IS: 2021-02-21 bis 2023-12-31, OOS: ab 2024-01-01")

    # Load metadata
    meta = pd.read_parquet('data/metadata/metadata_v8_5.parquet')
    h1 = meta[(meta['date'] >= '2021-02-21') & (meta['date'] <= '2023-12-31')].copy()
    h2 = meta[(meta['date'] >= '2024-01-01')].copy()

    base_h1 = h1[(h1['od_strength'] > 0.5) & (h1['od_direction'] == 'with_gap')].copy()
    base_h2 = h2[(h2['od_strength'] > 0.5) & (h2['od_direction'] == 'with_gap')].copy()

    print(f"H1: {len(base_h1)}, H2: {len(base_h2)}", file=sys.stderr)

    # Pre-compute BR_RANGE vectorized
    base_h1['br_range'] = compute_br_range_vec(base_h1)
    base_h2['br_range'] = compute_br_range_vec(base_h2)

    # Build datasets with BR_ENTRY from 1min bars
    df_is = build_dataset(base_h1, "IS")
    df_oos = build_dataset(base_h2, "OOS")

    n_range_is = int(df_is['br_range'].sum())
    n_entry_is = int(df_is['br_entry'].sum())
    n_range_oos = int(df_oos['br_range'].sum())
    n_entry_oos = int(df_oos['br_entry'].sum())
    print(f"IS: {len(df_is)} trades, BR_RANGE={n_range_is}, BR_ENTRY={n_entry_is}", file=sys.stderr)
    print(f"OOS: {len(df_oos)} trades, BR_RANGE={n_range_oos}, BR_ENTRY={n_entry_oos}", file=sys.stderr)

    # Sanity check: BR_RANGE IS rates should be ~19-24%
    for gap_dir, glabel, expected_lo, expected_hi in [('up', 'GapUp', 15, 28), ('down', 'GapDn', 18, 30)]:
        g = df_is[(df_is['gap_direction'] == gap_dir) & df_is['br_range'].notna()]
        rate = g['br_range'].mean() * 100
        if rate < expected_lo or rate > expected_hi:
            print(f"WARNING: BR_RANGE IS {glabel} rate {rate:.1f}% outside expected range "
                  f"[{expected_lo}-{expected_hi}%]!", file=sys.stderr)

    # Run analyses for both definitions
    top_range = run_analysis(df_is, df_oos, 'br_range',
                             "BR_RANGE (ab Open, >= 1.5 ADR)", lines)
    top_entry = run_analysis(df_is, df_oos, 'br_entry',
                             "BR_ENTRY (ab Entry, >= 0.7 ADR)", lines)

    # === VERGLEICHSTABELLE ===
    lines.append(f"\n\n{'='*70}")
    lines.append("VERGLEICHSTABELLE")
    lines.append(f"{'='*70}")

    comp_rows = []
    for gap_dir, glabel in [('up', 'GapUp'), ('down', 'GapDn')]:
        for dlabel, ddf in [('IS', df_is), ('OOS', df_oos)]:
            vr = ddf[ddf['br_range'].notna() & (ddf['gap_direction'] == gap_dir)]
            ve = ddf[ddf['br_entry'].notna() & (ddf['gap_direction'] == gap_dir)]
            rate_r = vr['br_range'].mean() * 100 if len(vr) > 0 else 0
            rate_e = ve['br_entry'].mean() * 100 if len(ve) > 0 else 0
            comp_rows.append((f"BR Rate {dlabel} {glabel}",
                              f"{rate_r:.1f}% (N={len(vr)})",
                              f"{rate_e:.1f}% (N={len(ve)})"))

    for gk in ['GapUp', 'GapDn']:
        tr = top_range.get(gk, None)
        te = top_entry.get(gk, None)
        val_r = f"{tr[0].split(': ', 1)[1]} ({tr[1]:.0%}, N={tr[2]})" if tr else "n/a"
        val_e = f"{te[0].split(': ', 1)[1]} ({te[1]:.0%}, N={te[2]})" if te else "n/a"
        comp_rows.append((f"Top Regel OOS {gk}", val_r, val_e))

    # Overlap
    both_valid = df_is[df_is['br_range'].notna() & df_is['br_entry'].notna()]
    both_br_is = len(both_valid[(both_valid['br_range'] == 1) & (both_valid['br_entry'] == 1)])
    only_range_is = len(both_valid[(both_valid['br_range'] == 1) & (both_valid['br_entry'] == 0)])
    only_entry_is = len(both_valid[(both_valid['br_range'] == 0) & (both_valid['br_entry'] == 1)])
    comp_rows.append(("Overlap IS",
                      f"beide: {both_br_is}, nur Range: {only_range_is}",
                      f"nur Entry: {only_entry_is}"))

    both_valid_oos = df_oos[df_oos['br_range'].notna() & df_oos['br_entry'].notna()]
    both_br_oos = len(both_valid_oos[(both_valid_oos['br_range'] == 1) & (both_valid_oos['br_entry'] == 1)])
    only_range_oos = len(both_valid_oos[(both_valid_oos['br_range'] == 1) & (both_valid_oos['br_entry'] == 0)])
    only_entry_oos = len(both_valid_oos[(both_valid_oos['br_range'] == 0) & (both_valid_oos['br_entry'] == 1)])
    comp_rows.append(("Overlap OOS",
                      f"beide: {both_br_oos}, nur Range: {only_range_oos}",
                      f"nur Entry: {only_entry_oos}"))

    # Format table
    c1w = max(len(r[0]) for r in comp_rows) + 2
    c2w = max(len(r[1]) for r in comp_rows) + 2
    c3w = max(len(r[2]) for r in comp_rows) + 2

    lines.append(f"\n  {'Metrik':<{c1w}} | {'BR_RANGE':<{c2w}} | {'BR_ENTRY':<{c3w}}")
    lines.append(f"  {'-'*(c1w + c2w + c3w + 6)}")
    for label, val_r, val_e in comp_rows:
        lines.append(f"  {label:<{c1w}} | {val_r:<{c2w}} | {val_e:<{c3w}}")

    # Write output
    output = "\n".join(lines)
    os.makedirs("results", exist_ok=True)
    with open('results/d11s_fix_synthese.txt', 'w', encoding='utf-8') as f:
        f.write(output)
    print(output)
    print(f"\nSaved to results/d11s_fix_synthese.txt", file=sys.stderr)


if __name__ == '__main__':
    main()
