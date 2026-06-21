###############################################################################
# OOS VALIDATION — Opening Drive on Half 2 (2024-2026)
# Durchlauf 3.0 — FINAL ONE-SHOT TEST
#
# Pre-registered hypotheses:
# H1: GapUp + OD_WITH (continuation) has positive Drift240 > 0
# H2: OD direction spread (WITH - AGAINST) > 0 for GapUp
# H3: ORB aligned with OD has positive Drift60 > 0
# H4: Overall OD direction predicts rest-of-day (all trades)
#
# Bekannte Vorbehalte:
# - Signal war 2021/2022 NEGATIV, erst 2023 positiv
# - Outlier-getrieben (Median ≈ 0, Top 10% treiben Mean)
# - Koennte 2023-spezifisch sein
#
# Run: .\gapper_env\Scripts\python.exe scripts\oos_opening_drive_half2.py
###############################################################################

import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE_DIR = str(PROJECT_ROOT)
RAW_1MIN_DIR = os.path.join(BASE_DIR, 'data', 'raw_1min')
METADATA_PATH = os.path.join(BASE_DIR, 'data', 'metadata', 'metadata_master.parquet')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# HALF 2: LOCKED DATA — first time accessing
OOS_START = '2024-01-01'
OOS_END = '2026-12-31'

np.random.seed(42)


def load_raw_1min(ticker, date_str):
    path = os.path.join(RAW_1MIN_DIR, ticker, f"{date_str}.parquet")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_parquet(path)
        if 'session' in df.columns:
            df = df[df['session'] == 'rth'].reset_index(drop=True)
        elif 'time_et' in df.columns:
            df = df[(df['time_et'] >= '09:30') & (df['time_et'] <= '15:59')].reset_index(drop=True)
        return df
    except:
        return None


def compute_opening_drive(df, period_bars=15):
    if len(df) < period_bars:
        return None, None, None
    first_open = df.iloc[0]['open']
    od_close = df.iloc[period_bars - 1]['close']
    if first_open == 0:
        return None, None, None
    move = (od_close - first_open) / first_open
    direction = 'up' if od_close > first_open else 'down'
    return direction, move, od_close


def compute_opening_range(df, period_bars=15):
    if len(df) < period_bars:
        return None, None
    subset = df.iloc[:period_bars]
    return subset['high'].max(), subset['low'].min()


def bootstrap_ci(vals, n_boot=5000, alpha=0.05):
    boots = [np.mean(np.random.choice(vals, size=len(vals), replace=True)) for _ in range(n_boot)]
    ci = np.percentile(boots, [100*alpha/2, 100*(1-alpha/2)])
    p_neg = np.mean([b <= 0 for b in boots])
    return ci, p_neg, boots


def main():
    # Load metadata — HALF 2 ONLY
    print("Loading metadata (HALF 2: 2024-2026)...", file=sys.stderr)
    meta = pd.read_parquet(METADATA_PATH)
    meta['date'] = pd.to_datetime(meta['date']).dt.strftime('%Y-%m-%d')
    meta = meta[(meta['date'] >= OOS_START) & (meta['date'] <= OOS_END)]
    meta = meta.dropna(subset=['adr_10'])
    meta = meta[meta['adr_10'] > 0]
    print(f"  {len(meta)} gappers in Half 2", file=sys.stderr)
    print(f"  GapUp: {(meta['gap_direction']=='up').sum()}, GapDn: {(meta['gap_direction']=='down').sum()}", file=sys.stderr)

    all_results = []

    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Processing OOS", file=sys.stderr):
        ticker = row['ticker']
        date_str = row['date']
        adr = row['adr_10']
        gap_dir = row['gap_direction']

        df = load_raw_1min(ticker, date_str)
        if df is None or len(df) < 60:
            continue

        required = ['open', 'high', 'low', 'close', 'volume']
        if not all(c in df.columns for c in required):
            continue

        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        rth_open = opens[0]

        if rth_open <= 0 or adr <= 0:
            continue

        # Opening Drive
        od_dir, od_move, od_close_price = compute_opening_drive(df, 15)
        if od_dir is None:
            continue

        od_move_adr = (od_close_price - rth_open) / adr
        od_against_gap = (gap_dir == 'up' and od_dir == 'down') or (gap_dir == 'down' and od_dir == 'up')

        # Opening Range
        or15_hi, or15_lo = compute_opening_range(df, 15)
        if or15_hi is None:
            continue

        # ORB breakout
        orb_break_dir = None
        for i in range(15, min(len(df), 60)):
            if highs[i] > or15_hi:
                orb_break_dir = 'up'
                break
            elif lows[i] < or15_lo:
                orb_break_dir = 'down'
                break

        orb_aligns_od = orb_break_dir == od_dir if orb_break_dir else None

        # Behavioral metrics (same sign convention as v1)
        if od_against_gap:
            sign = -1 if gap_dir == 'up' else 1
        else:
            sign = 1 if gap_dir == 'up' else -1

        windows = {'30m': 30, '60m': 60, '120m': 120, '240m': 240}
        mfe_w = {}
        drift_w = {}
        for w_name, w_bars in windows.items():
            end = min(15 + w_bars, len(closes))
            future = closes[15:end]
            if len(future) > 0:
                w_moves = sign * (future - od_close_price) / adr
                mfe_w[f'mfe_{w_name}'] = np.max(w_moves)
                drift_w[f'drift_{w_name}'] = w_moves[-1]
            else:
                mfe_w[f'mfe_{w_name}'] = np.nan
                drift_w[f'drift_{w_name}'] = np.nan

        result = {
            'ticker': ticker,
            'date': date_str,
            'year': date_str[:4],
            'gap_direction': gap_dir,
            'adr': adr,
            'od_direction': od_dir,
            'od_move_adr': od_move_adr,
            'od_against_gap': od_against_gap,
            'orb_break_dir': orb_break_dir,
            'orb_aligns_od': orb_aligns_od,
            'spy_return': row.get('spy_return_day', np.nan),
        }
        result.update(mfe_w)
        result.update(drift_w)
        all_results.append(result)

    df_all = pd.DataFrame(all_results)
    print(f"\nTotal OOS results: {len(df_all)}", file=sys.stderr)
    if len(df_all) == 0:
        print("ERROR: No results!", file=sys.stderr)
        return

    df_all.to_parquet(os.path.join(RESULTS_DIR, 'oos_opening_drive_half2_events.parquet'), index=False)

    # ====================================================================
    # ANALYSIS
    # ====================================================================
    out = []
    out.append("=" * 90)
    out.append("OOS VALIDATION — Opening Drive on Half 2 (2024-2026)")
    out.append("EINMALIGER TEST — Durchlauf 3.0")
    out.append("=" * 90)
    out.append("")
    out.append(f"Total OOS: {len(df_all)}")
    for yr in sorted(df_all['year'].unique()):
        n = (df_all['year'] == yr).sum()
        out.append(f"  {yr}: {n}")
    out.append(f"GapUp: {(df_all['gap_direction']=='up').sum()}, "
               f"GapDn: {(df_all['gap_direction']=='down').sum()}")
    out.append("")

    # ===================================================================
    # H1: GapUp + OD_WITH has positive Drift240
    # ===================================================================
    out.append("=" * 90)
    out.append("H1: GapUp + OD_WITH → Drift240 > 0?")
    out.append("  In-sample (2023 Val): +0.2073, P=0.0006")
    out.append("  In-sample (2021-22 Train): -0.0166 (NEGATIVE!)")
    out.append("=" * 90)
    out.append("")

    groups = [
        ('GapUp+OD_WITH', lambda d: (d['gap_direction']=='up') & (~d['od_against_gap'])),
        ('GapUp+OD_AGAINST', lambda d: (d['gap_direction']=='up') & (d['od_against_gap'])),
        ('GapDn+OD_WITH', lambda d: (d['gap_direction']=='down') & (~d['od_against_gap'])),
        ('GapDn+OD_AGAINST', lambda d: (d['gap_direction']=='down') & (d['od_against_gap'])),
    ]

    out.append(f"{'Group':<20} {'N':>5} {'Drift60':>9} {'Drift240':>10} {'MFE60':>8} {'P(<=0)':>8}")
    out.append("-" * 65)

    for label, mask_fn in groups:
        sub = df_all[mask_fn(df_all)]
        if len(sub) < 30:
            out.append(f"{label:<20} {len(sub):>5} (too small)")
            continue
        d60 = sub['drift_60m'].dropna()
        d240 = sub['drift_240m'].dropna()
        mfe60 = sub['mfe_60m'].dropna()

        ci240, p240, boots240 = bootstrap_ci(d240.values)
        out.append(f"{label:<20} {len(sub):>5} {d60.mean():>9.4f} {d240.mean():>10.4f} "
                   f"{mfe60.mean():>8.4f} {p240:>8.4f}")

    out.append("")

    # Detailed bootstrap CIs
    out.append("--- Bootstrap CIs (5000 iterations) ---")
    for label, mask_fn in groups:
        sub = df_all[mask_fn(df_all)]
        for target in ['drift_60m', 'drift_240m']:
            vals = sub[target].dropna().values
            if len(vals) < 30:
                continue
            ci, p, _ = bootstrap_ci(vals)
            out.append(f"  {label} {target}: mean={np.mean(vals):.4f}, "
                       f"CI [{ci[0]:.4f}, {ci[1]:.4f}], P(<=0)={p:.4f}")
        out.append("")

    # ===================================================================
    # H2: OD direction spread (WITH - AGAINST) > 0 for GapUp
    # ===================================================================
    out.append("=" * 90)
    out.append("H2: OD Spread (WITH - AGAINST) > 0 for GapUp?")
    out.append("  In-sample (2023 Val): Spread = +0.247 ADR")
    out.append("=" * 90)
    out.append("")

    gap_up = df_all[df_all['gap_direction'] == 'up']
    with_vals = gap_up[~gap_up['od_against_gap']]['drift_240m'].dropna().values
    against_vals = gap_up[gap_up['od_against_gap']]['drift_240m'].dropna().values

    real_spread = np.mean(with_vals) - np.mean(against_vals)
    out.append(f"  GapUp WITH mean: {np.mean(with_vals):.4f}")
    out.append(f"  GapUp AGAINST mean: {np.mean(against_vals):.4f}")
    out.append(f"  Spread: {real_spread:.4f}")

    # Permutation test on the spread
    all_gapu_vals = gap_up['drift_240m'].dropna().values
    all_gapu_labels = gap_up.loc[gap_up['drift_240m'].notna(), 'od_against_gap'].values

    perm_spreads = []
    for _ in range(2000):
        shuffled = np.random.permutation(all_gapu_labels)
        w = all_gapu_vals[~shuffled].mean()
        a = all_gapu_vals[shuffled].mean()
        perm_spreads.append(w - a)
    perm_spreads = np.array(perm_spreads)
    p_perm = np.mean(perm_spreads >= real_spread)

    out.append(f"  Permutation P: {p_perm:.4f}")
    out.append(f"  VERDICT: {'SIGNIFICANT' if p_perm < 0.05 else 'NOT significant'}")
    out.append("")

    # Same for GapDown
    gap_dn = df_all[df_all['gap_direction'] == 'down']
    dn_with = gap_dn[~gap_dn['od_against_gap']]['drift_240m'].dropna().values
    dn_against = gap_dn[gap_dn['od_against_gap']]['drift_240m'].dropna().values
    if len(dn_with) > 20 and len(dn_against) > 20:
        dn_spread = np.mean(dn_with) - np.mean(dn_against)
        out.append(f"  GapDn WITH mean: {np.mean(dn_with):.4f}")
        out.append(f"  GapDn AGAINST mean: {np.mean(dn_against):.4f}")
        out.append(f"  GapDn Spread: {dn_spread:.4f}")
    out.append("")

    # ===================================================================
    # H3: ORB aligned with OD → positive Drift60
    # ===================================================================
    out.append("=" * 90)
    out.append("H3: ORB aligned with OD → Drift60 > 0?")
    out.append("  In-sample (2023 Val): Aligned MFE=0.827, Drift=+0.042")
    out.append("=" * 90)
    out.append("")

    orb_events = df_all[df_all['orb_break_dir'].notna()]
    out.append(f"  ORB breaks: {len(orb_events)} / {len(df_all)} ({len(orb_events)/len(df_all)*100:.1f}%)")

    for aligns in [True, False]:
        sub = orb_events[orb_events['orb_aligns_od'] == aligns]
        if len(sub) < 30:
            continue
        d60 = sub['drift_60m'].dropna()
        mfe60 = sub['mfe_60m'].dropna()
        out.append(f"  ORB {'aligned' if aligns else 'opposing'}: N={len(sub)}, "
                   f"Drift60={d60.mean():.4f}, MFE60={mfe60.mean():.4f}")
    out.append("")

    # ===================================================================
    # H4: Overall OD direction predicts rest-of-day
    # ===================================================================
    out.append("=" * 90)
    out.append("H4: OD direction predicts rest-of-day (all trades)?")
    out.append("=" * 90)
    out.append("")

    for od_d in ['up', 'down']:
        sub = df_all[df_all['od_direction'] == od_d]
        d240 = sub['drift_240m'].dropna()
        out.append(f"  OD={od_d}: N={len(sub)}, Drift240={d240.mean():.4f}")
    out.append("")

    with_all = df_all[~df_all['od_against_gap']]['drift_240m'].dropna().values
    against_all = df_all[df_all['od_against_gap']]['drift_240m'].dropna().values
    out.append(f"  ALL WITH: mean={np.mean(with_all):.4f}")
    out.append(f"  ALL AGAINST: mean={np.mean(against_all):.4f}")
    out.append(f"  Spread: {np.mean(with_all) - np.mean(against_all):.4f}")
    out.append("")

    # ===================================================================
    # YEAR-BY-YEAR BREAKDOWN
    # ===================================================================
    out.append("=" * 90)
    out.append("YEAR-BY-YEAR BREAKDOWN")
    out.append("=" * 90)
    out.append("")

    for label, mask_fn in groups:
        out.append(f"  {label}:")
        for yr in sorted(df_all['year'].unique()):
            sub = df_all[(df_all['year'] == yr) & mask_fn(df_all)]
            if len(sub) < 15:
                out.append(f"    {yr}: N={len(sub)} (too small)")
                continue
            d60 = sub['drift_60m'].dropna()
            d240 = sub['drift_240m'].dropna()
            mfe60 = sub['mfe_60m'].dropna()
            out.append(f"    {yr}: N={len(sub)}, Drift60={d60.mean():.4f}, "
                       f"Drift240={d240.mean():.4f}, MFE60={mfe60.mean():.4f}")
        out.append("")

    # ===================================================================
    # DISTRIBUTION OF RETURNS (GapUp+WITH)
    # ===================================================================
    out.append("=" * 90)
    out.append("RETURN DISTRIBUTION — GapUp+WITH (OOS)")
    out.append("=" * 90)
    out.append("")

    sub_guw = df_all[(df_all['gap_direction'] == 'up') & (~df_all['od_against_gap'])]
    for col in ['drift_60m', 'drift_240m']:
        vals = sub_guw[col].dropna().values
        if len(vals) < 20:
            continue
        out.append(f"  {col} (N={len(vals)}):")
        out.append(f"    Mean: {np.mean(vals):.4f}")
        out.append(f"    Median: {np.median(vals):.4f}")
        pcts = np.percentile(vals, [5, 10, 25, 50, 75, 90, 95])
        out.append(f"    P5={pcts[0]:.4f}, P10={pcts[1]:.4f}, P25={pcts[2]:.4f}, "
                   f"P50={pcts[3]:.4f}, P75={pcts[4]:.4f}, P90={pcts[5]:.4f}, P95={pcts[6]:.4f}")
        wr = np.mean(vals > 0)
        out.append(f"    Win Rate: {wr:.1%}")
        # Trimmed mean
        cutoff90 = np.percentile(vals, 90)
        trimmed = vals[vals <= cutoff90]
        out.append(f"    Mean (excl top 10%): {np.mean(trimmed):.4f}")
        out.append("")

    # ===================================================================
    # COMPARISON: In-Sample vs OOS
    # ===================================================================
    out.append("=" * 90)
    out.append("IN-SAMPLE vs OOS COMPARISON")
    out.append("=" * 90)
    out.append("")

    # Load in-sample events for comparison
    is_path = os.path.join(RESULTS_DIR, 'orb_opening_drive_events_v1.parquet')
    if os.path.exists(is_path):
        is_ev = pd.read_parquet(is_path)
        is_val = is_ev[is_ev['split'] == 'val']

        out.append(f"{'Group':<20} {'IS_Val_D240':>12} {'OOS_D240':>10} {'Delta':>8} {'Replicated?':>12}")
        out.append("-" * 65)

        for label, mask_fn in groups:
            is_sub = is_val[mask_fn(is_val)]
            oos_sub = df_all[mask_fn(df_all)]
            is_d240 = is_sub['drift_240m'].mean() if len(is_sub) > 0 else np.nan
            oos_d240 = oos_sub['drift_240m'].mean() if len(oos_sub) > 0 else np.nan
            delta = oos_d240 - is_d240 if not np.isnan(oos_d240) and not np.isnan(is_d240) else np.nan
            # Replicated = same sign and OOS > 50% of IS
            replicated = "YES" if (is_d240 > 0 and oos_d240 > is_d240 * 0.3) else \
                         "PARTIAL" if (is_d240 > 0 and oos_d240 > 0) else "NO"
            out.append(f"{label:<20} {is_d240:>12.4f} {oos_d240:>10.4f} {delta:>+8.4f} {replicated:>12}")
        out.append("")

    # ===================================================================
    # SLIPPAGE-ADJUSTED
    # ===================================================================
    out.append("=" * 90)
    out.append("SLIPPAGE-ADJUSTED DRIFT (0.005 / 0.010 ADR)")
    out.append("=" * 90)
    out.append("")

    for label, mask_fn in groups:
        sub = df_all[mask_fn(df_all)]
        d60 = sub['drift_60m'].mean()
        d240 = sub['drift_240m'].mean()
        out.append(f"  {label}:")
        out.append(f"    Raw:        Drift60={d60:.4f}, Drift240={d240:.4f}")
        out.append(f"    Slip 0.005: Drift60={d60-0.005:.4f}, Drift240={d240-0.005:.4f}")
        out.append(f"    Slip 0.010: Drift60={d60-0.010:.4f}, Drift240={d240-0.010:.4f}")

    out.append("")

    # ===================================================================
    # FINAL VERDICT
    # ===================================================================
    out.append("=" * 90)
    out.append("FINAL VERDICT — OOS Validation")
    out.append("=" * 90)
    out.append("")

    # Get key numbers
    guw_oos = df_all[(df_all['gap_direction'] == 'up') & (~df_all['od_against_gap'])]
    guw_d240 = guw_oos['drift_240m'].dropna().values
    if len(guw_d240) > 30:
        ci, p, _ = bootstrap_ci(guw_d240)
        out.append(f"  H1 (GapUp+WITH Drift240 > 0):")
        out.append(f"    OOS: mean={np.mean(guw_d240):.4f}, CI [{ci[0]:.4f}, {ci[1]:.4f}], P(<=0)={p:.4f}")
        if p < 0.05 and np.mean(guw_d240) > 0:
            out.append(f"    RESULT: CONFIRMED — Edge replicates OOS!")
        elif np.mean(guw_d240) > 0:
            out.append(f"    RESULT: WEAK — Positive but not significant")
        else:
            out.append(f"    RESULT: FAILED — Edge does not replicate")
    out.append("")

    out.append(f"  H2 (GapUp Spread WITH-AGAINST > 0):")
    out.append(f"    OOS Spread: {real_spread:.4f}, Perm P: {p_perm:.4f}")
    if p_perm < 0.05 and real_spread > 0:
        out.append(f"    RESULT: CONFIRMED")
    else:
        out.append(f"    RESULT: {'FAILED' if real_spread <= 0 else 'WEAK — not significant'}")
    out.append("")

    out.append("OVERALL: Does Opening Drive predict rest-of-day on new data?")
    # Summarize
    all_with_d240 = np.mean(with_all)
    all_against_d240 = np.mean(against_all)
    total_spread = all_with_d240 - all_against_d240
    out.append(f"  WITH mean: {all_with_d240:.4f}, AGAINST mean: {all_against_d240:.4f}")
    out.append(f"  Total spread: {total_spread:.4f}")
    if total_spread > 0.05:
        out.append("  Opening Drive carries REAL predictive value on new data.")
    elif total_spread > 0:
        out.append("  Opening Drive shows WEAK predictive value on new data.")
    else:
        out.append("  Opening Drive has NO predictive value on new data.")
    out.append("")

    # Write
    output_path = os.path.join(RESULTS_DIR, 'oos_opening_drive_half2.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out))
    print(f"Ergebnisse: {output_path}", file=sys.stderr)


if __name__ == '__main__':
    main()
