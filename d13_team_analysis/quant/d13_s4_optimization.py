"""
QUANT-Agent: N-dimensionale Parameter-Optimierung der S4-Strategie
==================================================================
Phase 1: Univariate Scans (IS)
Phase 2: Top-Variablen identifizieren
Phase 3: Multi-dimensionale Matrix
Phase 4: Top-5 IS Kombinationen
Phase 5: OOS-Validierung
"""
import pandas as pd
import numpy as np
import os
import sys
import time
from itertools import product

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))
from d10_trailing import simulate_trailing

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = os.path.join(os.path.dirname(__file__), '..', '..')
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw_1min')
META_PATH = os.path.join(BASE_DIR, 'data', 'metadata', 'metadata_v9.parquet')
OUTPUT_PATH = os.path.join(BASE_DIR, 'd13_team_analysis', 'results', 'd13_s4_optimization_results.txt')

MIN_N = 30
BOOTSTRAP_IS = 1000
BOOTSTRAP_OOS = 10000


# ============================================================
# BACKTESTING
# ============================================================
def backtest_s4(events_df, label="", silent=False):
    """Run S4 backtest: entry close_935, SL 0.25*ADR, TRAIL_D, gap direction."""
    results = []
    for _, row in events_df.iterrows():
        ticker, date = row['ticker'], str(row['date'])
        path = os.path.join(RAW_DIR, ticker, f"{date}.parquet")
        if not os.path.exists(path):
            continue
        bars = pd.read_parquet(path)
        rth = bars[bars['session'] == 'rth'] if 'session' in bars.columns else bars
        if len(rth) < 10:
            continue
        entry_price = row['close_935']
        adr = row['adr_10']
        if pd.isna(entry_price) or pd.isna(adr) or adr <= 0:
            continue
        trade_dir = 'long' if row['gap_direction'] == 'up' else 'short'
        sl_dist = 0.25 * adr
        result = simulate_trailing(rth, entry_price, sl_dist, trade_dir, adr, trail_type='D')
        if result is None:
            continue
        result['ticker'] = ticker
        result['date'] = date
        results.append(result)
    if not silent and len(results) > 0:
        print(f"  {label}: N={len(results)}", file=sys.stderr)
    return pd.DataFrame(results)


def compute_stats(trades_df):
    """Compute EV(R), WR, PF, N from trades."""
    if len(trades_df) == 0:
        return {'N': 0, 'EV_R': np.nan, 'EV_ADR': np.nan, 'WR': np.nan, 'PF': np.nan}
    n = len(trades_df)
    ev_adr = trades_df['pnl_adr'].mean()
    ev_r = trades_df['pnl_r'].mean()
    wr = trades_df['winner'].mean()
    winners = trades_df[trades_df['pnl_r'] > 0]['pnl_r'].sum()
    losers = abs(trades_df[trades_df['pnl_r'] < 0]['pnl_r'].sum())
    pf = winners / losers if losers > 0 else np.inf
    return {'N': n, 'EV_R': ev_r, 'EV_ADR': ev_adr, 'WR': wr, 'PF': pf}


def bootstrap_pvalue(trades_df, n_resamples=1000):
    """Bootstrap p-value: H0 = EV <= 0."""
    if len(trades_df) < 10:
        return 1.0
    pnl_r = trades_df['pnl_r'].values
    n = len(pnl_r)
    rng = np.random.RandomState(42)
    means = np.array([rng.choice(pnl_r, size=n, replace=True).mean() for _ in range(n_resamples)])
    p_val = (means <= 0).sum() / n_resamples
    return p_val


def bootstrap_ci(trades_df, n_resamples=10000, ci=0.95):
    """Bootstrap confidence interval for EV(R)."""
    if len(trades_df) < 10:
        return np.nan, np.nan, 1.0
    pnl_r = trades_df['pnl_r'].values
    n = len(pnl_r)
    rng = np.random.RandomState(42)
    means = np.array([rng.choice(pnl_r, size=n, replace=True).mean() for _ in range(n_resamples)])
    alpha = (1 - ci) / 2
    lo = np.percentile(means, alpha * 100)
    hi = np.percentile(means, (1 - alpha) * 100)
    p_val = (means <= 0).sum() / n_resamples
    return lo, hi, p_val


# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time.time()
    lines = []

    # Load data
    print("Loading metadata...", file=sys.stderr)
    meta = pd.read_parquet(META_PATH)

    # IS / OOS split
    is_all = meta[(meta['date'] >= '2021-02-21') & (meta['date'] <= '2023-12-31')].copy()
    oos_all = meta[(meta['date'] >= '2024-01-01') & (meta['date'] <= '2026-02-13')].copy()

    # S4 Baseline filter
    is_base = is_all[(is_all['pm_rth5'] < 0.30) & (is_all['gap_size_in_adr'] >= 1.5)].copy()
    oos_base = oos_all[(oos_all['pm_rth5'] < 0.30) & (oos_all['gap_size_in_adr'] >= 1.5)].copy()

    print(f"IS Baseline: N={len(is_base)}", file=sys.stderr)
    print(f"OOS Baseline: N={len(oos_base)}", file=sys.stderr)

    # ============================================================
    # Pre-compute IS baseline trades (once)
    # ============================================================
    print("\n=== Pre-computing IS Baseline trades ===", file=sys.stderr)
    is_trades = backtest_s4(is_base, label="IS Baseline")
    baseline_stats = compute_stats(is_trades)
    print(f"IS Baseline: N={baseline_stats['N']}, EV={baseline_stats['EV_R']:.3f}R, WR={baseline_stats['WR']:.1%}", file=sys.stderr)

    # Merge metadata back into trades for filtering
    # We need to keep track of which metadata row produced which trade
    # Rebuild: map (ticker, date) -> metadata row
    meta_lookup = {}
    for _, row in is_base.iterrows():
        key = (row['ticker'], str(row['date']))
        meta_lookup[key] = row

    # Enrich trades with metadata columns needed for filtering
    enrich_cols = ['pm_rth5', 'gap_size_in_adr', 'od_strength', 'rvol_5', 'rsi_14_prev',
                   'gap_direction', 'is_earnings', 'first_candle_size', 'dist_from_52w_high',
                   'market_cap_bucket']

    for col in enrich_cols:
        vals = []
        for _, tr in is_trades.iterrows():
            key = (tr['ticker'], tr['date'])
            if key in meta_lookup:
                vals.append(meta_lookup[key].get(col, np.nan))
            else:
                vals.append(np.nan)
        is_trades[col] = vals

    # Same for OOS
    print("\n=== Pre-computing OOS Baseline trades ===", file=sys.stderr)
    oos_trades = backtest_s4(oos_base, label="OOS Baseline")
    oos_baseline_stats = compute_stats(oos_trades)
    print(f"OOS Baseline: N={oos_baseline_stats['N']}, EV={oos_baseline_stats['EV_R']:.3f}R, WR={oos_baseline_stats['WR']:.1%}", file=sys.stderr)

    oos_meta_lookup = {}
    for _, row in oos_base.iterrows():
        key = (row['ticker'], str(row['date']))
        oos_meta_lookup[key] = row

    for col in enrich_cols:
        vals = []
        for _, tr in oos_trades.iterrows():
            key = (tr['ticker'], tr['date'])
            if key in oos_meta_lookup:
                vals.append(oos_meta_lookup[key].get(col, np.nan))
            else:
                vals.append(np.nan)
        oos_trades[col] = vals

    # ============================================================
    # HEADER
    # ============================================================
    lines.append("=" * 70)
    lines.append("S4 PARAMETER-OPTIMIERUNG (QUANT-Agent)")
    lines.append("=" * 70)
    lines.append(f"IS: 2021-02-21 bis 2023-12-31")
    lines.append(f"OOS: 2024-01-01 bis 2026-02-13")
    lines.append(f"Baseline: pm_rth5 < 0.30, gap_size_in_adr >= 1.5")
    lines.append(f"Entry: close_935, SL: 0.25*ADR, TRAIL_D, gap_direction")
    lines.append(f"1R = SL = 0.25 ADR, EV(R) = EV(ADR) / 0.25")
    lines.append(f"")
    lines.append(f"IS Baseline: N={baseline_stats['N']}, EV={baseline_stats['EV_R']:+.3f}R, WR={baseline_stats['WR']:.1%}, PF={baseline_stats['PF']:.2f}")
    lines.append(f"OOS Baseline: N={oos_baseline_stats['N']}, EV={oos_baseline_stats['EV_R']:+.3f}R, WR={oos_baseline_stats['WR']:.1%}, PF={oos_baseline_stats['PF']:.2f}")

    # ============================================================
    # PHASE 1: UNIVARIATE SCANS
    # ============================================================
    lines.append(f"\n\n{'='*70}")
    lines.append("PHASE 1: UNIVARIATE SCANS (IS-Daten)")
    lines.append(f"{'='*70}")
    lines.append("Basis: S4-Baseline-Trades, dann zusaetzlicher Filter.")
    lines.append("Alle Werte: EV in R (1R = 0.25 ADR)")

    phase1_results = {}  # {variable: [(label, filter_mask, stats_dict), ...]}

    def scan_and_report(var_name, filters, description=""):
        """Run univariate scan. filters = [(label, boolean_mask_on_is_trades), ...]"""
        lines.append(f"\n--- {var_name} ({description}) ---")
        header = f"  {'Filter':<35} | {'N':>5} | {'EV(R)':>8} | {'WR':>6} | {'PF':>6} | {'vs BL':>8}"
        lines.append(header)
        lines.append(f"  {'-'*80}")

        results_list = []
        for label, mask in filters:
            subset = is_trades[mask]
            st = compute_stats(subset)
            delta = st['EV_R'] - baseline_stats['EV_R'] if not np.isnan(st['EV_R']) else np.nan

            n_str = f"{st['N']:>5}"
            ev_str = f"{st['EV_R']:>+8.3f}" if not np.isnan(st['EV_R']) else f"{'---':>8}"
            wr_str = f"{st['WR']:>6.1%}" if not np.isnan(st['WR']) else f"{'---':>6}"
            pf_str = f"{st['PF']:>6.2f}" if not np.isnan(st['PF']) and st['PF'] != np.inf else f"{'inf':>6}" if st.get('PF') == np.inf else f"{'---':>6}"
            delta_str = f"{delta:>+8.3f}" if not np.isnan(delta) else f"{'---':>8}"

            line = f"  {label:<35} | {n_str} | {ev_str} | {wr_str} | {pf_str} | {delta_str}"
            if st['N'] < MIN_N:
                line += " [N<30]"
            lines.append(line)
            results_list.append((label, mask, st, delta))

        phase1_results[var_name] = results_list

    # 1. pm_rth5
    print("\nPhase 1.1: pm_rth5...", file=sys.stderr)
    pm_thresholds = [0.10, 0.15, 0.20, 0.25, 0.30]
    pm_filters = []
    for t in pm_thresholds:
        mask = is_trades['pm_rth5'] < t
        pm_filters.append((f"pm_rth5 < {t:.2f}", mask))
    scan_and_report("pm_rth5", pm_filters, "Premarket/RTH5 Volume Ratio")

    # 2. gap_size_in_adr
    print("Phase 1.2: gap_size_in_adr...", file=sys.stderr)
    gap_thresholds = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    gap_filters = []
    for t in gap_thresholds:
        mask = is_trades['gap_size_in_adr'] >= t
        gap_filters.append((f"gap >= {t:.1f} ADR", mask))
    scan_and_report("gap_size_in_adr", gap_filters, "Gap Size in ADR")

    # 3. od_strength
    print("Phase 1.3: od_strength...", file=sys.stderr)
    od_thresholds = [0.0, 0.3, 0.5, 0.7, 1.0]
    od_filters = []
    for t in od_thresholds:
        mask = is_trades['od_strength'] >= t
        od_filters.append((f"od_strength >= {t:.1f}", mask))
    scan_and_report("od_strength", od_filters, "Opening Drive Strength")

    # 4. rvol_5
    print("Phase 1.4: rvol_5...", file=sys.stderr)
    rvol_thresholds = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
    rvol_filters = []
    for t in rvol_thresholds:
        mask = is_trades['rvol_5'] >= t
        rvol_filters.append((f"rvol_5 >= {t:.1f}", mask))
    scan_and_report("rvol_5", rvol_filters, "Relative Volume (5d)")

    # 5. rsi_14_prev
    print("Phase 1.5: rsi_14_prev...", file=sys.stderr)
    rsi_filters = [
        ("RSI 0-30 (oversold)", (is_trades['rsi_14_prev'] >= 0) & (is_trades['rsi_14_prev'] < 30)),
        ("RSI 30-50", (is_trades['rsi_14_prev'] >= 30) & (is_trades['rsi_14_prev'] < 50)),
        ("RSI 50-70", (is_trades['rsi_14_prev'] >= 50) & (is_trades['rsi_14_prev'] < 70)),
        ("RSI 70-100 (overbought)", (is_trades['rsi_14_prev'] >= 70) & (is_trades['rsi_14_prev'] <= 100)),
        ("RSI < 50", is_trades['rsi_14_prev'] < 50),
        ("RSI >= 50", is_trades['rsi_14_prev'] >= 50),
    ]
    scan_and_report("rsi_14_prev", rsi_filters, "RSI(14) previous day")

    # 6. gap_direction
    print("Phase 1.6: gap_direction...", file=sys.stderr)
    dir_filters = [
        ("up only (long)", is_trades['gap_direction'] == 'up'),
        ("down only (short)", is_trades['gap_direction'] == 'down'),
        ("both (baseline)", is_trades['gap_direction'].isin(['up', 'down'])),
    ]
    scan_and_report("gap_direction", dir_filters, "Gap Direction")

    # 7. is_earnings
    print("Phase 1.7: is_earnings...", file=sys.stderr)
    earn_filters = [
        ("earnings=True", is_trades['is_earnings'] == True),
        ("earnings=False", is_trades['is_earnings'] == False),
        ("both (baseline)", is_trades['is_earnings'].isin([True, False])),
    ]
    scan_and_report("is_earnings", earn_filters, "Earnings Gap")

    # 8. first_candle_size
    print("Phase 1.8: first_candle_size...", file=sys.stderr)
    fcs = is_trades['first_candle_size'].dropna()
    fcs_median = fcs.median()
    fcs_q25, fcs_q75 = fcs.quantile(0.25), fcs.quantile(0.75)
    fc_filters = [
        (f"Q1 (< {fcs_q25:.3f})", is_trades['first_candle_size'] < fcs_q25),
        (f"Q2 ({fcs_q25:.3f}-{fcs_median:.3f})", (is_trades['first_candle_size'] >= fcs_q25) & (is_trades['first_candle_size'] < fcs_median)),
        (f"Q3 ({fcs_median:.3f}-{fcs_q75:.3f})", (is_trades['first_candle_size'] >= fcs_median) & (is_trades['first_candle_size'] < fcs_q75)),
        (f"Q4 (>= {fcs_q75:.3f})", is_trades['first_candle_size'] >= fcs_q75),
        (f"below median (< {fcs_median:.3f})", is_trades['first_candle_size'] < fcs_median),
        (f"above median (>= {fcs_median:.3f})", is_trades['first_candle_size'] >= fcs_median),
    ]
    scan_and_report("first_candle_size", fc_filters, "First 5min Candle Size (ADR)")

    # 9. dist_from_52w_high
    print("Phase 1.9: dist_from_52w_high...", file=sys.stderr)
    d52_filters = [
        ("within 5% of 52w high", is_trades['dist_from_52w_high'] >= -5),
        ("within 10% of 52w high", is_trades['dist_from_52w_high'] >= -10),
        ("within 20% of 52w high", is_trades['dist_from_52w_high'] >= -20),
        ("more than 20% from high", is_trades['dist_from_52w_high'] < -20),
        ("more than 30% from high", is_trades['dist_from_52w_high'] < -30),
        ("more than 50% from high", is_trades['dist_from_52w_high'] < -50),
    ]
    scan_and_report("dist_from_52w_high", d52_filters, "Distance from 52-week High (%)")

    # 10. market_cap_bucket
    print("Phase 1.10: market_cap_bucket...", file=sys.stderr)
    mc_filters = [
        ("2-10B", is_trades['market_cap_bucket'] == '2-10B'),
        ("10-50B", is_trades['market_cap_bucket'] == '10-50B'),
        ("50-200B", is_trades['market_cap_bucket'] == '50-200B'),
        ("200B+", is_trades['market_cap_bucket'] == '200B+'),
        (">=50B (large)", is_trades['market_cap_bucket'].isin(['50-200B', '200B+'])),
        ("<=50B (small/mid)", is_trades['market_cap_bucket'].isin(['2-10B', '10-50B'])),
    ]
    scan_and_report("market_cap_bucket", mc_filters, "Market Cap Bucket")

    # ============================================================
    # PHASE 2: TOP-VARIABLEN IDENTIFIZIEREN
    # ============================================================
    lines.append(f"\n\n{'='*70}")
    lines.append("PHASE 2: TOP-VARIABLEN (staerkster EV-Anstieg vs Baseline)")
    lines.append(f"{'='*70}")
    lines.append(f"\nBaseline IS EV: {baseline_stats['EV_R']:+.3f}R")
    lines.append("")

    # Collect best filter per variable (with N>=30)
    best_per_var = []
    for var_name, results_list in phase1_results.items():
        best_delta = -999
        best_entry = None
        for label, mask, st, delta in results_list:
            if st['N'] >= MIN_N and not np.isnan(delta) and delta > best_delta:
                best_delta = delta
                best_entry = (var_name, label, mask, st, delta)
        if best_entry is not None:
            best_per_var.append(best_entry)

    best_per_var.sort(key=lambda x: x[4], reverse=True)

    lines.append(f"  {'Rank':>4} | {'Variable':<20} | {'Best Filter':<35} | {'N':>5} | {'EV(R)':>8} | {'Delta':>8}")
    lines.append(f"  {'-'*95}")
    for i, (var, label, mask, st, delta) in enumerate(best_per_var):
        lines.append(f"  {i+1:>4} | {var:<20} | {label:<35} | {st['N']:>5} | {st['EV_R']:>+8.3f} | {delta:>+8.3f}")

    print(f"\nPhase 2 complete. Top variables:", file=sys.stderr)
    for i, (var, label, mask, st, delta) in enumerate(best_per_var[:5]):
        print(f"  {i+1}. {var}: {label} -> EV={st['EV_R']:+.3f}R (delta={delta:+.3f}R, N={st['N']})", file=sys.stderr)

    # ============================================================
    # PHASE 3: MULTI-DIMENSIONALE MATRIX
    # ============================================================
    lines.append(f"\n\n{'='*70}")
    lines.append("PHASE 3: MULTI-DIMENSIONALE MATRIX")
    lines.append(f"{'='*70}")

    # Select top variables: pick those with positive delta and N>=30
    # Use top 4-5 variables with meaningful levels
    # We build a grid of filter combinations

    # Define filter levels for top variables (based on Phase 1 findings)
    # We'll use the top-ranked variables and define 2-3 levels each

    # Build grid dynamically based on top variables
    # For each top variable, define levels that showed improvement

    grid_vars = {}

    # For each of the top variables, define grid levels
    # pm_rth5 levels
    grid_vars['pm_rth5'] = [
        ("pm<0.10", is_trades['pm_rth5'] < 0.10),
        ("pm<0.15", is_trades['pm_rth5'] < 0.15),
        ("pm<0.20", is_trades['pm_rth5'] < 0.20),
        ("pm<0.30", is_trades['pm_rth5'] < 0.30),  # baseline = all
    ]

    # gap_size_in_adr levels
    grid_vars['gap_size'] = [
        ("gap>=1.5", is_trades['gap_size_in_adr'] >= 1.5),  # baseline
        ("gap>=2.0", is_trades['gap_size_in_adr'] >= 2.0),
        ("gap>=2.5", is_trades['gap_size_in_adr'] >= 2.5),
        ("gap>=3.0", is_trades['gap_size_in_adr'] >= 3.0),
    ]

    # od_strength levels
    grid_vars['od_str'] = [
        ("od>=0.0", is_trades['od_strength'] >= 0.0),  # all
        ("od>=0.3", is_trades['od_strength'] >= 0.3),
        ("od>=0.5", is_trades['od_strength'] >= 0.5),
        ("od>=0.7", is_trades['od_strength'] >= 0.7),
    ]

    # rvol_5 levels
    grid_vars['rvol5'] = [
        ("rv>=1", is_trades['rvol_5'] >= 1.0),  # all
        ("rv>=3", is_trades['rvol_5'] >= 3.0),
        ("rv>=5", is_trades['rvol_5'] >= 5.0),
        ("rv>=7", is_trades['rvol_5'] >= 7.0),
    ]

    # is_earnings levels
    grid_vars['earnings'] = [
        ("any", is_trades['is_earnings'].isin([True, False])),
        ("earn=T", is_trades['is_earnings'] == True),
        ("earn=F", is_trades['is_earnings'] == False),
    ]

    # For the multi-dim matrix, select the top 3-4 variables that showed best deltas
    # to keep combinations manageable
    # Take variables ranked by delta
    top_var_names = [v[0] for v in best_per_var[:5]]
    print(f"\nTop-5 variables for grid: {top_var_names}", file=sys.stderr)

    # Map variable names to grid keys
    var_to_grid = {
        'pm_rth5': 'pm_rth5',
        'gap_size_in_adr': 'gap_size',
        'od_strength': 'od_str',
        'rvol_5': 'rvol5',
        'is_earnings': 'earnings',
        'rsi_14_prev': None,  # will add if in top
        'gap_direction': None,
        'first_candle_size': None,
        'dist_from_52w_high': None,
        'market_cap_bucket': None,
    }

    # Add RSI if in top
    if 'rsi_14_prev' in top_var_names:
        grid_vars['rsi'] = [
            ("rsi_any", is_trades['rsi_14_prev'].notna()),
            ("rsi<50", is_trades['rsi_14_prev'] < 50),
            ("rsi>=50", is_trades['rsi_14_prev'] >= 50),
        ]
        var_to_grid['rsi_14_prev'] = 'rsi'

    if 'gap_direction' in top_var_names:
        grid_vars['gap_dir'] = [
            ("both", is_trades['gap_direction'].isin(['up', 'down'])),
            ("up", is_trades['gap_direction'] == 'up'),
            ("down", is_trades['gap_direction'] == 'down'),
        ]
        var_to_grid['gap_direction'] = 'gap_dir'

    if 'first_candle_size' in top_var_names:
        fcs_med = is_trades['first_candle_size'].median()
        grid_vars['fc_size'] = [
            ("fc_any", is_trades['first_candle_size'].notna()),
            (f"fc<{fcs_med:.3f}", is_trades['first_candle_size'] < fcs_med),
            (f"fc>={fcs_med:.3f}", is_trades['first_candle_size'] >= fcs_med),
        ]
        var_to_grid['first_candle_size'] = 'fc_size'

    if 'dist_from_52w_high' in top_var_names:
        grid_vars['d52w'] = [
            ("d52_any", is_trades['dist_from_52w_high'].notna()),
            ("d52>=-20", is_trades['dist_from_52w_high'] >= -20),
            ("d52<-20", is_trades['dist_from_52w_high'] < -20),
        ]
        var_to_grid['dist_from_52w_high'] = 'd52w'

    if 'market_cap_bucket' in top_var_names:
        grid_vars['mcap'] = [
            ("mcap_any", is_trades['market_cap_bucket'].notna()),
            ("mcap>=50B", is_trades['market_cap_bucket'].isin(['50-200B', '200B+'])),
            ("mcap<50B", is_trades['market_cap_bucket'].isin(['2-10B', '10-50B'])),
        ]
        var_to_grid['market_cap_bucket'] = 'mcap'

    # Select top 3 for the main grid (to keep combinations reasonable)
    # Then do a secondary check with top 4-5
    selected_grid_keys = []
    for vname in top_var_names[:4]:
        gk = var_to_grid.get(vname)
        if gk and gk in grid_vars:
            selected_grid_keys.append(gk)

    if len(selected_grid_keys) < 3:
        # Fallback: use pm_rth5, gap_size, od_str
        for gk in ['pm_rth5', 'gap_size', 'od_str', 'rvol5', 'earnings']:
            if gk not in selected_grid_keys:
                selected_grid_keys.append(gk)
            if len(selected_grid_keys) >= 4:
                break

    print(f"\nGrid variables: {selected_grid_keys}", file=sys.stderr)
    lines.append(f"\nGrid-Variablen: {', '.join(selected_grid_keys)}")

    # Generate all combinations
    grid_levels = [grid_vars[k] for k in selected_grid_keys]
    all_combos = list(product(*grid_levels))
    print(f"Total combinations: {len(all_combos)}", file=sys.stderr)
    lines.append(f"Anzahl Kombinationen: {len(all_combos)}")
    lines.append("")

    # Header
    var_headers = " | ".join([f"{k:<10}" for k in selected_grid_keys])
    full_header = f"  {var_headers} | {'N':>5} | {'EV(R)':>8} | {'WR':>6} | {'PF':>6} | {'p-val':>6}"
    lines.append(full_header)
    lines.append(f"  {'-'*len(full_header)}")

    combo_results = []
    for idx, combo in enumerate(all_combos):
        if (idx + 1) % 50 == 0:
            print(f"  Combo {idx+1}/{len(all_combos)}...", file=sys.stderr)

        # Combined mask
        combined_mask = pd.Series(True, index=is_trades.index)
        labels = []
        for (label, mask) in combo:
            combined_mask = combined_mask & mask
            labels.append(label)

        subset = is_trades[combined_mask]
        st = compute_stats(subset)

        if st['N'] >= MIN_N:
            p_val = bootstrap_pvalue(subset, n_resamples=BOOTSTRAP_IS)
        else:
            p_val = np.nan

        combo_key = " | ".join([f"{l:<10}" for l in labels])
        n_str = f"{st['N']:>5}"
        ev_str = f"{st['EV_R']:>+8.3f}" if not np.isnan(st['EV_R']) else f"{'---':>8}"
        wr_str = f"{st['WR']:>6.1%}" if not np.isnan(st['WR']) else f"{'---':>6}"
        pf_str = f"{st['PF']:>6.2f}" if not np.isnan(st['PF']) and st['PF'] != np.inf else f"{'---':>6}"
        p_str = f"{p_val:>6.3f}" if not np.isnan(p_val) else f"{'---':>6}"

        suffix = ""
        if st['N'] < MIN_N:
            suffix = " [N<30]"

        lines.append(f"  {combo_key} | {n_str} | {ev_str} | {wr_str} | {pf_str} | {p_str}{suffix}")

        combo_results.append({
            'labels': labels,
            'combo': combo,
            'stats': st,
            'p_val': p_val,
            'combined_mask': combined_mask,
        })

    # ============================================================
    # PHASE 4: TOP-5 IS KOMBINATIONEN
    # ============================================================
    lines.append(f"\n\n{'='*70}")
    lines.append("PHASE 4: TOP-5 IS KOMBINATIONEN")
    lines.append(f"{'='*70}")
    lines.append(f"Filter: N >= {MIN_N} und p < 0.05")
    lines.append("")

    # Filter valid combos
    valid_combos = [c for c in combo_results if c['stats']['N'] >= MIN_N and not np.isnan(c['p_val']) and c['p_val'] < 0.05]
    valid_combos.sort(key=lambda x: x['stats']['EV_R'], reverse=True)

    if len(valid_combos) == 0:
        # Relax p-value constraint
        lines.append("HINWEIS: Keine Kombination mit p<0.05. Zeige Top-5 mit N>=30:")
        valid_combos = [c for c in combo_results if c['stats']['N'] >= MIN_N]
        valid_combos.sort(key=lambda x: x['stats']['EV_R'], reverse=True)

    top5_is = valid_combos[:5]

    for i, c in enumerate(top5_is):
        st = c['stats']
        lines.append(f"  #{i+1}: {' + '.join(c['labels'])}")
        lines.append(f"       N={st['N']}, EV={st['EV_R']:+.3f}R, WR={st['WR']:.1%}, PF={st['PF']:.2f}, p={c['p_val']:.4f}")
        lines.append("")

    print(f"\nPhase 4: Top 5 IS combinations identified", file=sys.stderr)
    for i, c in enumerate(top5_is):
        print(f"  #{i+1}: {' + '.join(c['labels'])} -> EV={c['stats']['EV_R']:+.3f}R, N={c['stats']['N']}", file=sys.stderr)

    # ============================================================
    # PHASE 5: OOS-VALIDIERUNG
    # ============================================================
    lines.append(f"\n\n{'='*70}")
    lines.append("PHASE 5: OOS-VALIDIERUNG")
    lines.append(f"{'='*70}")
    lines.append(f"Bootstrap: {BOOTSTRAP_OOS} resamples, 95% CI")
    lines.append("")

    # Build OOS masks for each grid variable
    oos_grid_vars = {}

    oos_grid_vars['pm_rth5'] = {
        "pm<0.10": oos_trades['pm_rth5'] < 0.10,
        "pm<0.15": oos_trades['pm_rth5'] < 0.15,
        "pm<0.20": oos_trades['pm_rth5'] < 0.20,
        "pm<0.30": oos_trades['pm_rth5'] < 0.30,
    }
    oos_grid_vars['gap_size'] = {
        "gap>=1.5": oos_trades['gap_size_in_adr'] >= 1.5,
        "gap>=2.0": oos_trades['gap_size_in_adr'] >= 2.0,
        "gap>=2.5": oos_trades['gap_size_in_adr'] >= 2.5,
        "gap>=3.0": oos_trades['gap_size_in_adr'] >= 3.0,
    }
    oos_grid_vars['od_str'] = {
        "od>=0.0": oos_trades['od_strength'] >= 0.0,
        "od>=0.3": oos_trades['od_strength'] >= 0.3,
        "od>=0.5": oos_trades['od_strength'] >= 0.5,
        "od>=0.7": oos_trades['od_strength'] >= 0.7,
    }
    oos_grid_vars['rvol5'] = {
        "rv>=1": oos_trades['rvol_5'] >= 1.0,
        "rv>=3": oos_trades['rvol_5'] >= 3.0,
        "rv>=5": oos_trades['rvol_5'] >= 5.0,
        "rv>=7": oos_trades['rvol_5'] >= 7.0,
    }
    oos_grid_vars['earnings'] = {
        "any": oos_trades['is_earnings'].isin([True, False]),
        "earn=T": oos_trades['is_earnings'] == True,
        "earn=F": oos_trades['is_earnings'] == False,
    }

    if 'rsi' in grid_vars:
        oos_grid_vars['rsi'] = {
            "rsi_any": oos_trades['rsi_14_prev'].notna(),
            "rsi<50": oos_trades['rsi_14_prev'] < 50,
            "rsi>=50": oos_trades['rsi_14_prev'] >= 50,
        }
    if 'gap_dir' in grid_vars:
        oos_grid_vars['gap_dir'] = {
            "both": oos_trades['gap_direction'].isin(['up', 'down']),
            "up": oos_trades['gap_direction'] == 'up',
            "down": oos_trades['gap_direction'] == 'down',
        }
    if 'fc_size' in grid_vars:
        fcs_med = is_trades['first_candle_size'].median()  # use IS median for OOS
        oos_grid_vars['fc_size'] = {
            "fc_any": oos_trades['first_candle_size'].notna(),
            f"fc<{fcs_med:.3f}": oos_trades['first_candle_size'] < fcs_med,
            f"fc>={fcs_med:.3f}": oos_trades['first_candle_size'] >= fcs_med,
        }
    if 'd52w' in grid_vars:
        oos_grid_vars['d52w'] = {
            "d52_any": oos_trades['dist_from_52w_high'].notna(),
            "d52>=-20": oos_trades['dist_from_52w_high'] >= -20,
            "d52<-20": oos_trades['dist_from_52w_high'] < -20,
        }
    if 'mcap' in grid_vars:
        oos_grid_vars['mcap'] = {
            "mcap_any": oos_trades['market_cap_bucket'].notna(),
            "mcap>=50B": oos_trades['market_cap_bucket'].isin(['50-200B', '200B+']),
            "mcap<50B": oos_trades['market_cap_bucket'].isin(['2-10B', '10-50B']),
        }

    for i, c in enumerate(top5_is):
        labels = c['labels']
        lines.append(f"  --- IS #{i+1}: {' + '.join(labels)} ---")
        lines.append(f"  IS: N={c['stats']['N']}, EV={c['stats']['EV_R']:+.3f}R, WR={c['stats']['WR']:.1%}")

        # Build OOS mask
        oos_mask = pd.Series(True, index=oos_trades.index)
        for gk_idx, gk in enumerate(selected_grid_keys):
            label = labels[gk_idx]
            if gk in oos_grid_vars and label in oos_grid_vars[gk]:
                oos_mask = oos_mask & oos_grid_vars[gk][label]
            else:
                lines.append(f"  WARNING: Could not map label '{label}' for grid key '{gk}' in OOS")

        oos_subset = oos_trades[oos_mask]
        oos_st = compute_stats(oos_subset)

        if oos_st['N'] >= 10:
            lo, hi, p_val = bootstrap_ci(oos_subset, n_resamples=BOOTSTRAP_OOS)
            lines.append(f"  OOS: N={oos_st['N']}, EV={oos_st['EV_R']:+.3f}R, WR={oos_st['WR']:.1%}, PF={oos_st['PF']:.2f}")
            lines.append(f"       95% CI: [{lo:+.3f}R, {hi:+.3f}R], p={p_val:.4f}")

            # IS/OOS consistency
            if not np.isnan(c['stats']['EV_R']) and not np.isnan(oos_st['EV_R']):
                ratio = oos_st['EV_R'] / c['stats']['EV_R'] if c['stats']['EV_R'] != 0 else np.nan
                if not np.isnan(ratio):
                    lines.append(f"       OOS/IS Ratio: {ratio:.2f}")
                    if ratio >= 0.5:
                        lines.append(f"       --> ROBUST (OOS >= 50% of IS)")
                    elif ratio > 0:
                        lines.append(f"       --> PARTIAL (OOS positive but < 50% of IS)")
                    else:
                        lines.append(f"       --> FAILED (OOS negative)")
        else:
            lines.append(f"  OOS: N={oos_st['N']} [INSUFFICIENT DATA]")

        lines.append("")
        print(f"  OOS #{i+1}: N={oos_st['N']}, EV={oos_st['EV_R']:+.3f}R" if oos_st['N'] > 0 else f"  OOS #{i+1}: N=0", file=sys.stderr)

    # ============================================================
    # ADDITIONAL: Also test some promising 2-variable combos explicitly
    # ============================================================
    lines.append(f"\n\n{'='*70}")
    lines.append("BONUS: GEZIELTE 2-VARIABLEN-KOMBIS (explorativ)")
    lines.append(f"{'='*70}")
    lines.append("")

    # Test specific promising combinations
    targeted_combos = [
        # Tighter pm_rth5 + larger gaps
        ("pm<0.15 + gap>=2.0",
         (is_trades['pm_rth5'] < 0.15) & (is_trades['gap_size_in_adr'] >= 2.0),
         (oos_trades['pm_rth5'] < 0.15) & (oos_trades['gap_size_in_adr'] >= 2.0)),
        ("pm<0.20 + gap>=2.0",
         (is_trades['pm_rth5'] < 0.20) & (is_trades['gap_size_in_adr'] >= 2.0),
         (oos_trades['pm_rth5'] < 0.20) & (oos_trades['gap_size_in_adr'] >= 2.0)),
        ("pm<0.15 + gap>=2.5",
         (is_trades['pm_rth5'] < 0.15) & (is_trades['gap_size_in_adr'] >= 2.5),
         (oos_trades['pm_rth5'] < 0.15) & (oos_trades['gap_size_in_adr'] >= 2.5)),
        # od_strength combos
        ("pm<0.20 + od>=0.5",
         (is_trades['pm_rth5'] < 0.20) & (is_trades['od_strength'] >= 0.5),
         (oos_trades['pm_rth5'] < 0.20) & (oos_trades['od_strength'] >= 0.5)),
        ("pm<0.20 + od>=0.7",
         (is_trades['pm_rth5'] < 0.20) & (is_trades['od_strength'] >= 0.7),
         (oos_trades['pm_rth5'] < 0.20) & (oos_trades['od_strength'] >= 0.7)),
        ("gap>=2.0 + od>=0.5",
         (is_trades['gap_size_in_adr'] >= 2.0) & (is_trades['od_strength'] >= 0.5),
         (oos_trades['gap_size_in_adr'] >= 2.0) & (oos_trades['od_strength'] >= 0.5)),
        # earnings filter
        ("pm<0.20 + earn=T",
         (is_trades['pm_rth5'] < 0.20) & (is_trades['is_earnings'] == True),
         (oos_trades['pm_rth5'] < 0.20) & (oos_trades['is_earnings'] == True)),
        ("pm<0.20 + earn=F",
         (is_trades['pm_rth5'] < 0.20) & (is_trades['is_earnings'] == False),
         (oos_trades['pm_rth5'] < 0.20) & (oos_trades['is_earnings'] == False)),
        ("gap>=2.0 + earn=T",
         (is_trades['gap_size_in_adr'] >= 2.0) & (is_trades['is_earnings'] == True),
         (oos_trades['gap_size_in_adr'] >= 2.0) & (oos_trades['is_earnings'] == True)),
        # rvol combos
        ("pm<0.20 + rv>=5",
         (is_trades['pm_rth5'] < 0.20) & (is_trades['rvol_5'] >= 5.0),
         (oos_trades['pm_rth5'] < 0.20) & (oos_trades['rvol_5'] >= 5.0)),
        ("gap>=2.0 + rv>=5",
         (is_trades['gap_size_in_adr'] >= 2.0) & (is_trades['rvol_5'] >= 5.0),
         (oos_trades['gap_size_in_adr'] >= 2.0) & (oos_trades['rvol_5'] >= 5.0)),
        # 3-way combos
        ("pm<0.20 + gap>=2.0 + od>=0.5",
         (is_trades['pm_rth5'] < 0.20) & (is_trades['gap_size_in_adr'] >= 2.0) & (is_trades['od_strength'] >= 0.5),
         (oos_trades['pm_rth5'] < 0.20) & (oos_trades['gap_size_in_adr'] >= 2.0) & (oos_trades['od_strength'] >= 0.5)),
        ("pm<0.15 + gap>=2.0 + od>=0.3",
         (is_trades['pm_rth5'] < 0.15) & (is_trades['gap_size_in_adr'] >= 2.0) & (is_trades['od_strength'] >= 0.3),
         (oos_trades['pm_rth5'] < 0.15) & (oos_trades['gap_size_in_adr'] >= 2.0) & (oos_trades['od_strength'] >= 0.3)),
        ("pm<0.20 + gap>=2.0 + earn=T",
         (is_trades['pm_rth5'] < 0.20) & (is_trades['gap_size_in_adr'] >= 2.0) & (is_trades['is_earnings'] == True),
         (oos_trades['pm_rth5'] < 0.20) & (oos_trades['gap_size_in_adr'] >= 2.0) & (oos_trades['is_earnings'] == True)),
        ("pm<0.20 + gap>=2.0 + rv>=5",
         (is_trades['pm_rth5'] < 0.20) & (is_trades['gap_size_in_adr'] >= 2.0) & (is_trades['rvol_5'] >= 5.0),
         (oos_trades['pm_rth5'] < 0.20) & (oos_trades['gap_size_in_adr'] >= 2.0) & (oos_trades['rvol_5'] >= 5.0)),
    ]

    header = f"  {'Combo':<40} | {'IS_N':>5} | {'IS_EV':>8} | {'IS_WR':>6} | {'OOS_N':>5} | {'OOS_EV':>8} | {'OOS_WR':>6} | {'p-val':>6}"
    lines.append(header)
    lines.append(f"  {'-'*len(header)}")

    for label, is_mask, oos_mask in targeted_combos:
        is_sub = is_trades[is_mask]
        oos_sub = oos_trades[oos_mask]
        is_st = compute_stats(is_sub)
        oos_st = compute_stats(oos_sub)

        if oos_st['N'] >= 10:
            _, _, p_val = bootstrap_ci(oos_sub, n_resamples=BOOTSTRAP_OOS)
        else:
            p_val = np.nan

        is_n = f"{is_st['N']:>5}"
        is_ev = f"{is_st['EV_R']:>+8.3f}" if not np.isnan(is_st['EV_R']) else f"{'---':>8}"
        is_wr = f"{is_st['WR']:>6.1%}" if not np.isnan(is_st['WR']) else f"{'---':>6}"
        oos_n = f"{oos_st['N']:>5}"
        oos_ev = f"{oos_st['EV_R']:>+8.3f}" if not np.isnan(oos_st['EV_R']) else f"{'---':>8}"
        oos_wr = f"{oos_st['WR']:>6.1%}" if not np.isnan(oos_st['WR']) else f"{'---':>6}"
        p_str = f"{p_val:>6.3f}" if not np.isnan(p_val) else f"{'---':>6}"

        suffix = ""
        if is_st['N'] < MIN_N:
            suffix += " [IS N<30]"
        if oos_st['N'] < MIN_N:
            suffix += " [OOS N<30]"

        lines.append(f"  {label:<40} | {is_n} | {is_ev} | {is_wr} | {oos_n} | {oos_ev} | {oos_wr} | {p_str}{suffix}")

    # ============================================================
    # FAZIT
    # ============================================================
    lines.append(f"\n\n{'='*70}")
    lines.append("FAZIT")
    lines.append(f"{'='*70}")

    # Find overall best OOS-validated combo
    lines.append(f"\nBaseline S4:")
    lines.append(f"  IS:  N={baseline_stats['N']}, EV={baseline_stats['EV_R']:+.3f}R, WR={baseline_stats['WR']:.1%}")
    lines.append(f"  OOS: N={oos_baseline_stats['N']}, EV={oos_baseline_stats['EV_R']:+.3f}R, WR={oos_baseline_stats['WR']:.1%}")
    lines.append("")
    lines.append("Top Phase-4 IS-Kombinationen mit OOS-Validierung:")
    for i, c in enumerate(top5_is):
        lines.append(f"  #{i+1}: {' + '.join(c['labels'])}")
        lines.append(f"       IS: N={c['stats']['N']}, EV={c['stats']['EV_R']:+.3f}R")

    lines.append("")
    lines.append("Empfehlung: Siehe detaillierte OOS-Ergebnisse oben.")
    lines.append("Nur Kombinationen mit OOS-EV > 0.25R und p < 0.05 sollten live getradet werden.")

    elapsed = time.time() - t0
    lines.append(f"\n\nLaufzeit: {elapsed:.1f} Sekunden")

    # Write output
    output = "\n".join(lines)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write(output)

    print(f"\n{'='*50}", file=sys.stderr)
    print(f"FERTIG. Ergebnis gespeichert: {OUTPUT_PATH}", file=sys.stderr)
    print(f"Laufzeit: {elapsed:.1f} Sekunden", file=sys.stderr)
    print(output)


if __name__ == '__main__':
    main()
