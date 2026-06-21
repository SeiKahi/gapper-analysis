"""
D15 — Fees & Slippage in 3 Szenarien
======================================
Reads D14 trade data, enriches with sl_dist/adr, applies cost model,
re-computes all KPIs, equity curves, and Monte Carlo.

Phases:
  0. Cat-2 sl_dist Recovery (re-run 2.3 to collect sl_dist per trade)
  1. Trade Enrichment (join metadata for adr, compute sl_dist)
  2. Cost Calculation (3 scenarios: optimistic, base, pessimistic)
  3. KPI Table (raw + 3 scenarios)
  4. Sensitivity Analysis (breakeven slippage per strategy)
  5. Equity Curves (3 cost lines per strategy)
  6. Monte Carlo (base scenario)
  7. Sensitivity Overview Chart

Usage:
  .\\gapper_env\\Scripts\\python.exe d15_bias_free/d15_enrich_and_cost.py
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
import sys
import time
from pathlib import Path

warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ============================================================
# CONSTANTS
# ============================================================
TRADES_PATH = Path('d14_bias_free/results/d14_all_trades.parquet')
META_PATH = Path('data/metadata/metadata_v9.parquet')
RESULTS_DIR = Path('d15_bias_free/results')
CHARTS_DIR = RESULTS_DIR / 'charts'

IS_START = '2021-02-21'
IS_END = '2023-12-31'
OOS_START = '2024-01-01'
OOS_END = '2026-02-21'

IS_CALENDAR_DAYS = (pd.Timestamp(IS_END) - pd.Timestamp(IS_START)).days
OOS_CALENDAR_DAYS = (pd.Timestamp(OOS_END) - pd.Timestamp(OOS_START)).days
TOTAL_CALENDAR_DAYS = (pd.Timestamp(OOS_END) - pd.Timestamp(IS_START)).days

MC_RUNS = 400
MC_SEED = 42

# Fee: $0.0025 per share roundtrip
FEE_PER_SHARE = 0.0025

# Slippage scenarios (in ADR fraction, entry+exit combined)
SLIPPAGE_SCENARIOS = {
    'opt':  0.02,   # Optimistic: limit orders, high liquidity
    'base': 0.05,   # Base: market orders, realistic
    'pess': 0.10,   # Pessimistic: bad execution
}

# SL multiplier per Cat-1 strategy (adr fraction)
SL_MULT = {
    '1.1_S4_Optimiert':       0.25,
    '1.2_Pre10am_Momentum':   0.15,
    '1.3_Earnings_OD_RVOL':   0.25,
    '1.4_Power_Setup':         0.25,
    '1.5_Big_First_Candle':   0.25,
    '1.6_Low_Wick':           0.25,
    '1.7_Big_Body':           0.25,
    '1.8_Gap_3ADR':           0.25,
    '1.9_RVOL5_RSI50':        0.25,
}

# The 10 surviving strategies with OOS EV >= 0.2R
SURVIVING_STRATEGIES = [
    '1.2_Pre10am_Momentum',
    '2.3_Dip_and_Rip',
    '1.4_Power_Setup',
    '1.3_Earnings_OD_RVOL',
    '1.8_Gap_3ADR',
    '1.5_Big_First_Candle',
    '1.1_S4_Optimiert',
    '1.9_RVOL5_RSI50',
    '1.6_Low_Wick',
    '1.7_Big_Body',
]

lines = []


def log(msg=''):
    lines.append(str(msg))
    print(msg)


def log_header(title):
    log(f"\n{'='*80}")
    log(f"  {title}")
    log(f"{'='*80}")


# ============================================================
# KPI FUNCTIONS (from D14.5, unchanged)
# ============================================================
def bootstrap_ci(arr, n_boot=10000, ci=0.95):
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 20:
        return (np.nan, np.nan)
    rng = np.random.RandomState(MC_SEED)
    boot = np.array([np.mean(rng.choice(arr, len(arr), replace=True)) for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    return (np.percentile(boot, 100 * alpha), np.percentile(boot, 100 * (1 - alpha)))


def compute_kpis(pnl_arr, calendar_days):
    """Compute all KPIs for a PnL array."""
    arr = np.asarray(pnl_arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n < 5:
        return None

    ev = np.mean(arr)
    std = np.std(arr, ddof=1)
    wr = np.mean(arr > 0)

    per_trade_sharpe = ev / std if std > 0 else 0
    trades_per_year = n / (calendar_days / 365.25) if calendar_days > 0 else 0
    ann_sharpe = per_trade_sharpe * np.sqrt(trades_per_year) if trades_per_year > 0 else 0

    cum = np.cumsum(arr)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    max_dd = dd.min()

    in_dd = dd < 0
    dd_depths = []
    dd_durations = []
    current_dd_start = None
    for i in range(n):
        if in_dd[i]:
            if current_dd_start is None:
                current_dd_start = i
            dd_depths.append(dd[i])
        else:
            if current_dd_start is not None:
                dd_durations.append(i - current_dd_start)
                current_dd_start = None
    if current_dd_start is not None:
        dd_durations.append(n - current_dd_start)
    avg_dd = np.mean(dd_depths) if dd_depths else 0

    max_consec = 0
    current_streak = 0
    for p in arr:
        if p <= 0:
            current_streak += 1
            max_consec = max(max_consec, current_streak)
        else:
            current_streak = 0

    start_cap = 100.0
    final_equity = start_cap + cum[-1]
    years = calendar_days / 365.25
    if final_equity > 0 and years > 0:
        cagr = (final_equity / start_cap) ** (1 / years) - 1
    else:
        cagr = -1.0

    wins = arr[arr > 0]
    losses = arr[arr <= 0]
    pf = abs(np.sum(wins) / np.sum(losses)) if len(losses) > 0 and np.sum(losses) != 0 else float('inf')

    ci_lo, ci_hi = bootstrap_ci(arr)

    return {
        'N': n, 'EV': ev, 'WR': wr, 'PF': pf,
        'Sharpe_ann': ann_sharpe, 'MaxDD': max_dd, 'AvgDD': avg_dd,
        'MaxConsecLoss': max_consec, 'CAGR_pct': cagr * 100,
        'Trades_per_yr': trades_per_year, 'Final_R': cum[-1],
        'CI_lo': ci_lo, 'CI_hi': ci_hi,
    }


# ============================================================
# PHASE 0: CAT-2 SL_DIST RECOVERY
# ============================================================
def recover_cat2_sl_dist(meta):
    """
    Re-run Cat-2 Dip&Rip strategy to recover sl_dist and adr per trade.
    Returns dict: (ticker, date) -> (sl_dist, adr)
    """
    from d14_bias_free.engine.streaming_trail import simulate_trail_d_streaming
    from d14_bias_free.strategies.cat2_dip_and_rip import run_cat2_dip_and_rip

    def simulate_fn(bars_rth, entry_price, sl_dist, trade_dir, adr,
                    entry_time='09:36', ticker=None, date=None):
        return simulate_trail_d_streaming(
            bars_rth, entry_price, sl_dist, trade_dir, adr,
            entry_time=entry_time, ticker=ticker, date=date,
            use_1sec=False  # No need for 1-sec resolution, just collecting sl_dist
        )

    log("  Running 2.3_Dip_and_Rip on full dataset to collect sl_dist...")
    results = run_cat2_dip_and_rip(meta, simulate_fn, label='ALL')

    lookup = {}
    for r in results:
        key = (r['ticker'], r['date'])
        lookup[key] = (r['sl_dist'], r['adr'])

    log(f"  Recovered sl_dist for {len(lookup)} Dip&Rip trades")
    return lookup


# ============================================================
# PLOTTING FUNCTIONS
# ============================================================
def plot_equity_3costs(trades_sub, strategy, kpis_by_scenario, out_path):
    """Plot equity curve with 3 cost scenario lines + IS/OOS split."""
    sub = trades_sub.sort_values('date').reset_index(drop=True)
    if len(sub) < 5:
        return

    is_mask = sub['sample'] == 'IS'
    oos_mask = sub['sample'] == 'OOS'
    is_idx = sub.index[is_mask].tolist()
    oos_idx = sub.index[oos_mask].tolist()
    split_idx = max(is_idx) if is_idx else 0

    dates = pd.to_datetime(sub['date'], errors='coerce')

    fig, ax1 = plt.subplots(figsize=(14, 7))

    colors = {'opt': '#4CAF50', 'base': '#2196F3', 'pess': '#F44336'}
    labels = {'opt': 'Optimistic (0.02 ADR)', 'base': 'Base (0.05 ADR)', 'pess': 'Pessimistic (0.10 ADR)'}

    for scenario in ['opt', 'base', 'pess']:
        col = f'pnl_r_{scenario}'
        if col not in sub.columns:
            continue
        cum = sub[col].cumsum().values
        ax1.plot(range(len(sub)), cum, color=colors[scenario], linewidth=1.5,
                 label=labels[scenario], alpha=0.9)

    # Raw equity (dashed)
    cum_raw = sub['pnl_r'].cumsum().values
    ax1.plot(range(len(sub)), cum_raw, color='black', linewidth=1.0,
             linestyle='--', label='Raw (no costs)', alpha=0.6)

    # IS/OOS split line
    if is_idx and oos_idx:
        ax1.axvline(x=split_idx, color='gray', linestyle='--', alpha=0.5, label='IS/OOS Split')

    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax1.set_xlabel('Trade #')
    ax1.set_ylabel('Cumulative PnL (R)')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Top x-axis: dates
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    n_ticks = min(8, len(sub))
    tick_positions = np.linspace(0, len(sub) - 1, n_ticks, dtype=int)
    ax2.set_xticks(tick_positions)
    tick_labels = []
    for tp in tick_positions:
        d = dates.iloc[tp]
        tick_labels.append(d.strftime('%Y-%m') if pd.notna(d) else '')
    ax2.set_xticklabels(tick_labels, fontsize=8)

    # Title with base-scenario KPIs
    k = kpis_by_scenario.get('base')
    if k:
        ax1.set_title(
            f'{strategy}  |  EV_base={k["EV"]:+.3f}R  Sharpe={k["Sharpe_ann"]:.2f}  '
            f'MaxDD={k["MaxDD"]:.1f}R  N={k["N"]}',
            fontsize=11, pad=25
        )
    else:
        ax1.set_title(strategy, fontsize=11, pad=25)

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_monte_carlo(pnl_arr, strategy, out_path):
    """Monte Carlo simulation on net PnL (base scenario)."""
    arr = np.asarray(pnl_arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n < 20:
        return

    rng = np.random.RandomState(MC_SEED)
    mc_curves = np.zeros((MC_RUNS, n))
    mc_maxdd = np.zeros(MC_RUNS)

    for i in range(MC_RUNS):
        shuffled = rng.permutation(arr)
        mc_curves[i] = np.cumsum(shuffled)
        peak = np.maximum.accumulate(mc_curves[i])
        mc_maxdd[i] = (mc_curves[i] - peak).min()

    p10 = np.percentile(mc_curves, 10, axis=0)
    p50 = np.percentile(mc_curves, 50, axis=0)
    p90 = np.percentile(mc_curves, 90, axis=0)
    actual = np.cumsum(arr)

    fig, ax = plt.subplots(figsize=(14, 7))

    for i in range(MC_RUNS):
        ax.plot(mc_curves[i], color='grey', alpha=0.03, linewidth=0.5)

    x = np.arange(n)
    ax.plot(x, p10, color='black', linestyle='--', linewidth=1.0, label='10th / 90th pctl')
    ax.plot(x, p90, color='black', linestyle='--', linewidth=1.0)
    ax.plot(x, p50, color='black', linestyle='-', linewidth=1.0, alpha=0.5, label='Median')
    ax.plot(x, actual, color='red', linewidth=2, label='Actual', zorder=5)

    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Trade #')
    ax.set_ylabel('Cumulative PnL (R)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    final_10, final_50, final_90 = p10[-1], p50[-1], p90[-1]
    actual_final = actual[-1]
    dd_10 = np.percentile(mc_maxdd, 10)
    dd_50 = np.percentile(mc_maxdd, 50)
    dd_90 = np.percentile(mc_maxdd, 90)

    anno = (f"Final R:  Actual={actual_final:+.1f}  "
            f"[10%={final_10:+.1f}, Med={final_50:+.1f}, 90%={final_90:+.1f}]\n"
            f"MaxDD:  [10%={dd_10:.1f}, Med={dd_50:.1f}, 90%={dd_90:.1f}]R")
    ax.annotate(anno, xy=(0.02, 0.02), xycoords='axes fraction',
                fontsize=9, fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax.set_title(f'{strategy}  —  Monte Carlo Base ({MC_RUNS} runs, N={n})',
                 fontsize=11)

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_sensitivity_overview(ev_data, out_path):
    """
    Sensitivity chart: X = scenario, Y = EV, one line per strategy.
    ev_data: dict of {strategy: {'raw': ev, 'opt': ev, 'base': ev, 'pess': ev}}
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    x_labels = ['Raw', 'Optimistic\n(0.02 ADR)', 'Base\n(0.05 ADR)', 'Pessimistic\n(0.10 ADR)']
    x = np.arange(4)

    # Color cycle
    cmap = plt.cm.tab10
    for i, (strat, evs) in enumerate(ev_data.items()):
        y = [evs['raw'], evs['opt'], evs['base'], evs['pess']]
        short_name = strat[:20]
        color = cmap(i % 10)
        ax.plot(x, y, marker='o', linewidth=2, markersize=6, label=short_name, color=color)

        # Annotate base EV
        ax.annotate(f'{evs["base"]:+.2f}', xy=(2, evs['base']),
                    fontsize=7, ha='left', va='bottom', color=color)

    ax.axhline(y=0, color='red', linestyle='-', linewidth=1.5, alpha=0.7, label='Breakeven')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_ylabel('Expected Value (R)', fontsize=11)
    ax.set_title('D15 Sensitivity: EV at 3 Cost Levels (OOS)', fontsize=13)
    ax.legend(loc='upper right', fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================
def main():
    t_start = time.time()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    log("=" * 80)
    log("  D15 — FEES & SLIPPAGE IN 3 SZENARIEN")
    log("=" * 80)
    log(f"  Datum: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    log(f"  IS:  {IS_START} bis {IS_END}")
    log(f"  OOS: {OOS_START} bis {OOS_END}")
    log(f"  Fee: ${FEE_PER_SHARE}/share roundtrip")
    log(f"  Slippage: {SLIPPAGE_SCENARIOS}")
    log(f"  Strategies: {len(SURVIVING_STRATEGIES)}")
    log("")

    # ============================================================
    # PHASE 0: CAT-2 SL_DIST RECOVERY
    # ============================================================
    log_header("PHASE 0: CAT-2 SL_DIST RECOVERY")

    log("  Loading metadata_v9...")
    meta = pd.read_parquet(META_PATH)
    meta['date'] = meta['date'].astype(str)
    log(f"  Total events: {len(meta)}")

    cat2_lookup = recover_cat2_sl_dist(meta)

    # ============================================================
    # PHASE 1: TRADE ENRICHMENT
    # ============================================================
    log_header("PHASE 1: TRADE ENRICHMENT")

    log("  Loading d14_all_trades.parquet...")
    df = pd.read_parquet(TRADES_PATH)
    log(f"  Total trades: {len(df)}")

    # Filter to surviving strategies
    df = df[df['strategy'].isin(SURVIVING_STRATEGIES)].copy()
    log(f"  After filter to {len(SURVIVING_STRATEGIES)} strategies: {len(df)} trades")

    # Join metadata for adr_10
    meta_adr = meta[['ticker', 'date', 'adr_10']].copy()
    meta_adr = meta_adr.drop_duplicates(subset=['ticker', 'date'])
    df = df.merge(meta_adr, on=['ticker', 'date'], how='left')
    adr_ok = df['adr_10'].notna().sum()
    log(f"  adr_10 joined: {adr_ok}/{len(df)} ({adr_ok/len(df)*100:.1f}%)")

    # Compute sl_dist for each trade
    def compute_sl_dist(row):
        strat = row['strategy']
        if strat in SL_MULT:
            # Cat-1: fixed SL multiplier
            if pd.notna(row['adr_10']) and row['adr_10'] > 0:
                return SL_MULT[strat] * row['adr_10']
            return np.nan
        elif strat == '2.3_Dip_and_Rip':
            # Cat-2: variable sl_dist from Phase 0 lookup
            key = (row['ticker'], row['date'])
            if key in cat2_lookup:
                return cat2_lookup[key][0]
            return np.nan
        return np.nan

    df['sl_dist'] = df.apply(compute_sl_dist, axis=1)
    sl_ok = df['sl_dist'].notna().sum()
    log(f"  sl_dist computed: {sl_ok}/{len(df)} ({sl_ok/len(df)*100:.1f}%)")

    # Drop rows without valid sl_dist or adr_10
    before = len(df)
    df = df[df['sl_dist'].notna() & (df['sl_dist'] > 0) & df['adr_10'].notna() & (df['adr_10'] > 0)].copy()
    log(f"  After dropping invalid sl_dist/adr: {len(df)} trades (dropped {before - len(df)})")

    # ============================================================
    # PHASE 2: COST CALCULATION
    # ============================================================
    log_header("PHASE 2: COST CALCULATION (3 SCENARIOS)")

    # Fee in R: fee_per_share / sl_dist_dollars
    # sl_dist is already in dollars (adr_10 is in price units)
    df['fee_r'] = FEE_PER_SHARE / df['sl_dist']

    log(f"  fee_r: mean={df['fee_r'].mean():.4f}R, median={df['fee_r'].median():.4f}R, "
        f"max={df['fee_r'].max():.4f}R")

    for scenario, slip_adr_frac in SLIPPAGE_SCENARIOS.items():
        # slippage_r = slip_adr_frac * adr / sl_dist
        df[f'slippage_r_{scenario}'] = slip_adr_frac * df['adr_10'] / df['sl_dist']
        df[f'cost_r_{scenario}'] = df['fee_r'] + df[f'slippage_r_{scenario}']
        df[f'pnl_r_{scenario}'] = df['pnl_r'] - df[f'cost_r_{scenario}']

        mean_slip = df[f'slippage_r_{scenario}'].mean()
        mean_cost = df[f'cost_r_{scenario}'].mean()
        log(f"  {scenario:>5}: mean_slippage={mean_slip:.4f}R, mean_total_cost={mean_cost:.4f}R")

    # Verify specific example: SL=0.25 ADR → slippage_base should be 0.20R
    cat1_025 = df[df['strategy'].isin([s for s, m in SL_MULT.items() if m == 0.25])]
    if len(cat1_025) > 0:
        mean_base_slip = cat1_025['slippage_r_base'].mean()
        log(f"\n  Verification (SL=0.25 ADR strategies):")
        log(f"    mean slippage_r_base = {mean_base_slip:.4f}R (expected: 0.2000R)")

    cat1_015 = df[df['strategy'] == '1.2_Pre10am_Momentum']
    if len(cat1_015) > 0:
        mean_base_slip_15 = cat1_015['slippage_r_base'].mean()
        log(f"    1.2 (SL=0.15 ADR): mean slippage_r_base = {mean_base_slip_15:.4f}R (expected: 0.3333R)")

    # ============================================================
    # PHASE 3: KPI TABLE
    # ============================================================
    log_header("PHASE 3: KPI TABLE (RAW + 3 SCENARIOS)")

    oos_df = df[df['sample'] == 'OOS']

    kpi_rows = []
    kpi_lookup = {}  # strategy -> {scenario: kpis}

    for strat in SURVIVING_STRATEGIES:
        sub_oos = oos_df[oos_df['strategy'] == strat]
        sub_is = df[(df['strategy'] == strat) & (df['sample'] == 'IS')]
        sub_all = df[df['strategy'] == strat]

        if len(sub_oos) < 5:
            log(f"  {strat}: insufficient OOS trades ({len(sub_oos)}), skipping")
            continue

        row = {'Strategy': strat}
        kpi_lookup[strat] = {}

        # Raw
        k_raw = compute_kpis(sub_oos['pnl_r'].values, OOS_CALENDAR_DAYS)
        if k_raw:
            row['RAW_N'] = k_raw['N']
            row['RAW_EV'] = k_raw['EV']
            row['RAW_WR'] = k_raw['WR']
            row['RAW_Sharpe'] = k_raw['Sharpe_ann']
            row['RAW_MaxDD'] = k_raw['MaxDD']
            row['RAW_PF'] = k_raw['PF']
            row['RAW_CAGR'] = k_raw['CAGR_pct']
            row['RAW_CI_lo'] = k_raw['CI_lo']
            row['RAW_CI_hi'] = k_raw['CI_hi']
            kpi_lookup[strat]['raw'] = k_raw

        for scenario in ['opt', 'base', 'pess']:
            col = f'pnl_r_{scenario}'
            k = compute_kpis(sub_oos[col].values, OOS_CALENDAR_DAYS)
            if k:
                pref = scenario.upper()
                row[f'{pref}_EV'] = k['EV']
                row[f'{pref}_WR'] = k['WR']
                row[f'{pref}_Sharpe'] = k['Sharpe_ann']
                row[f'{pref}_MaxDD'] = k['MaxDD']
                row[f'{pref}_PF'] = k['PF']
                row[f'{pref}_CAGR'] = k['CAGR_pct']
                row[f'{pref}_CI_lo'] = k['CI_lo']
                row[f'{pref}_CI_hi'] = k['CI_hi']
                kpi_lookup[strat][scenario] = k

        kpi_rows.append(row)

    kpi_df = pd.DataFrame(kpi_rows)

    # Write KPI table
    tbl_lines = []
    tbl_lines.append("D15 — KPI TABLE: RAW + 3 COST SCENARIOS (OOS)")
    tbl_lines.append("=" * 160)
    tbl_lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    tbl_lines.append(f"OOS: {OOS_START} to {OOS_END}")
    tbl_lines.append(f"Fee: ${FEE_PER_SHARE}/share roundtrip | Slippage: Opt={SLIPPAGE_SCENARIOS['opt']} ADR, "
                     f"Base={SLIPPAGE_SCENARIOS['base']} ADR, Pess={SLIPPAGE_SCENARIOS['pess']} ADR")
    tbl_lines.append("")

    header = (f"  {'Strategy':<28} | {'N':>5} | "
              f"{'EV_raw':>7} {'EV_opt':>7} {'EV_base':>7} {'EV_pess':>7} | "
              f"{'WR_base':>7} | {'Sha_base':>8} | {'MaxDD_b':>7} | "
              f"{'PF_base':>7} | {'CAGR_b%':>7} | {'CI_lo_b':>7} {'CI_hi_b':>7} | {'Robust':>6}")
    tbl_lines.append(header)
    tbl_lines.append("  " + "-" * 155)

    # Sort by base EV
    kpi_df = kpi_df.sort_values('BASE_EV', ascending=False)

    for _, r in kpi_df.iterrows():
        def fmt(val, width=7, fmt_str='+.3f'):
            if pd.isna(val):
                return 'N/A'.rjust(width)
            return f'{val:{fmt_str}}'.rjust(width)

        robust = 'ROBUST' if (not pd.isna(r.get('BASE_CI_lo', np.nan)) and r.get('BASE_CI_lo', -1) > 0) else 'NO'

        tbl_lines.append(
            f"  {r['Strategy']:<28} | {r['RAW_N']:>5.0f} | "
            f"{fmt(r.get('RAW_EV'))} {fmt(r.get('OPT_EV'))} {fmt(r.get('BASE_EV'))} {fmt(r.get('PESS_EV'))} | "
            f"{fmt(r.get('BASE_WR'), 7, '.1%')} | {fmt(r.get('BASE_Sharpe'), 8, '.2f')} | "
            f"{fmt(r.get('BASE_MaxDD'), 7, '.1f')} | "
            f"{fmt(r.get('BASE_PF'), 7, '.2f')} | {fmt(r.get('BASE_CAGR'), 7, '+.1f')} | "
            f"{fmt(r.get('BASE_CI_lo'))} {fmt(r.get('BASE_CI_hi'))} | {robust:>6}"
        )

    tbl_lines.append("")
    tbl_lines.append("Legend:")
    tbl_lines.append("  EV_raw  = Gross EV (no costs)")
    tbl_lines.append("  EV_opt  = EV after optimistic costs (0.02 ADR slippage)")
    tbl_lines.append("  EV_base = EV after base costs (0.05 ADR slippage)")
    tbl_lines.append("  EV_pess = EV after pessimistic costs (0.10 ADR slippage)")
    tbl_lines.append("  Sha_base= Annualized Sharpe (base scenario)")
    tbl_lines.append("  CI      = 95% Bootstrap CI of EV (base scenario)")
    tbl_lines.append("  ROBUST  = CI_lo > 0 (base scenario)")

    kpi_text = '\n'.join(tbl_lines)
    (RESULTS_DIR / 'd15_kpi_table.txt').write_text(kpi_text, encoding='utf-8')
    log(f"  KPI table saved: {RESULTS_DIR / 'd15_kpi_table.txt'}")

    # Print summary
    for _, r in kpi_df.iterrows():
        ev_raw = r.get('RAW_EV', np.nan)
        ev_base = r.get('BASE_EV', np.nan)
        ev_pess = r.get('PESS_EV', np.nan)
        ci_lo = r.get('BASE_CI_lo', np.nan)
        robust = 'ROBUST' if (not pd.isna(ci_lo) and ci_lo > 0) else 'NO'
        log(f"  {r['Strategy']:<28} | raw={ev_raw:+.3f} opt={r.get('OPT_EV', np.nan):+.3f} "
            f"base={ev_base:+.3f} pess={ev_pess:+.3f} | {robust}")

    # ============================================================
    # PHASE 4: SENSITIVITY ANALYSIS
    # ============================================================
    log_header("PHASE 4: SENSITIVITY ANALYSIS")

    sens_lines = []
    sens_lines.append("D15 — SENSITIVITY ANALYSIS: BREAKEVEN SLIPPAGE")
    sens_lines.append("=" * 100)
    sens_lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    sens_lines.append("")
    sens_lines.append("Breakeven Slippage = max slippage (in ADR) where EV remains > 0")
    sens_lines.append("  Formula: breakeven_adr = (EV_raw - fee_r_mean) * sl_adr_ratio")
    sens_lines.append("  Higher = more robust against execution costs")
    sens_lines.append("")

    sens_lines.append(f"  {'Strategy':<28} | {'EV_raw':>7} | {'SL_ADR':>6} | {'MeanFee':>7} | "
                      f"{'BE_ADR':>7} | {'BE vs Base':>10} | {'Survives_Pess':>13}")
    sens_lines.append("  " + "-" * 100)

    ev_data = {}  # For sensitivity chart

    for strat in SURVIVING_STRATEGIES:
        sub_oos = oos_df[oos_df['strategy'] == strat]
        if len(sub_oos) < 5:
            continue

        ev_raw = sub_oos['pnl_r'].mean()
        mean_fee_r = sub_oos['fee_r'].mean()

        # sl_adr_ratio: sl_dist / adr  (= SL_MULT for Cat-1, variable for Cat-2)
        sl_adr_ratio = (sub_oos['sl_dist'] / sub_oos['adr_10']).mean()

        # Breakeven: EV_raw = fee_r + slip_adr * adr / sl_dist
        # → slip_adr_breakeven = (EV_raw - mean_fee_r) * sl_adr_ratio
        be_adr = (ev_raw - mean_fee_r) * sl_adr_ratio
        be_vs_base = be_adr / SLIPPAGE_SCENARIOS['base']  # Multiples of base scenario

        ev_pess = sub_oos[f'pnl_r_pess'].mean()
        survives = 'YES' if ev_pess > 0 else 'NO'

        sens_lines.append(
            f"  {strat:<28} | {ev_raw:>+7.3f} | {sl_adr_ratio:>6.3f} | {mean_fee_r:>7.4f} | "
            f"{be_adr:>7.4f} | {be_vs_base:>9.1f}x | {survives:>13}"
        )

        ev_data[strat] = {
            'raw': ev_raw,
            'opt': sub_oos['pnl_r_opt'].mean(),
            'base': sub_oos['pnl_r_base'].mean(),
            'pess': ev_pess,
        }

        log(f"  {strat:<28}: BE={be_adr:.4f} ADR ({be_vs_base:.1f}x base), "
            f"survives_pess={'YES' if ev_pess > 0 else 'NO'}")

    sens_lines.append("")
    sens_lines.append("Ranking by Breakeven (most robust to least):")
    ranked = sorted(ev_data.items(), key=lambda x: x[1]['raw'] - x[1]['base'], reverse=True)
    # Actually rank by breakeven ADR value
    be_ranking = []
    for strat in SURVIVING_STRATEGIES:
        sub_oos_s = oos_df[oos_df['strategy'] == strat]
        if len(sub_oos_s) < 5:
            continue
        ev_raw = sub_oos_s['pnl_r'].mean()
        mean_fee_r = sub_oos_s['fee_r'].mean()
        sl_adr_ratio = (sub_oos_s['sl_dist'] / sub_oos_s['adr_10']).mean()
        be_adr = (ev_raw - mean_fee_r) * sl_adr_ratio
        be_ranking.append((strat, be_adr))

    be_ranking.sort(key=lambda x: x[1], reverse=True)
    for i, (strat, be) in enumerate(be_ranking, 1):
        sens_lines.append(f"  {i:>2}. {strat:<28} BE = {be:.4f} ADR")

    sens_text = '\n'.join(sens_lines)
    (RESULTS_DIR / 'd15_sensitivity.txt').write_text(sens_text, encoding='utf-8')
    log(f"\n  Sensitivity saved: {RESULTS_DIR / 'd15_sensitivity.txt'}")

    # ============================================================
    # PHASE 5: EQUITY CURVES (3 cost lines)
    # ============================================================
    log_header("PHASE 5: EQUITY CURVES (10 STRATEGIES x 3 COST LEVELS)")

    eq_count = 0
    for strat in SURVIVING_STRATEGIES:
        sub = df[df['strategy'] == strat].copy()
        if len(sub) < 5:
            continue
        safe_name = strat.replace(':', '').replace(' ', '_')
        out_path = CHARTS_DIR / f"eq_{safe_name}.png"
        kpis_by_scenario = kpi_lookup.get(strat, {})
        plot_equity_3costs(sub, strat, kpis_by_scenario, out_path)
        eq_count += 1
        log(f"  eq_{safe_name}.png")

    log(f"  Equity curves saved: {eq_count} PNGs")

    # ============================================================
    # PHASE 6: MONTE CARLO (BASE SCENARIO)
    # ============================================================
    log_header("PHASE 6: MONTE CARLO (BASE SCENARIO)")

    mc_count = 0
    for strat in SURVIVING_STRATEGIES:
        sub_oos = oos_df[oos_df['strategy'] == strat]
        pnl_base = sub_oos['pnl_r_base'].dropna().values
        if len(pnl_base) < 20:
            continue
        safe_name = strat.replace(':', '').replace(' ', '_')
        out_path = CHARTS_DIR / f"mc_{safe_name}.png"
        plot_monte_carlo(pnl_base, strat, out_path)
        mc_count += 1
        log(f"  mc_{safe_name}.png")

    log(f"  Monte Carlo saved: {mc_count} PNGs")

    # ============================================================
    # PHASE 7: SENSITIVITY OVERVIEW CHART
    # ============================================================
    log_header("PHASE 7: SENSITIVITY OVERVIEW CHART")

    if ev_data:
        plot_sensitivity_overview(ev_data, CHARTS_DIR / 'd15_sensitivity.png')
        log(f"  d15_sensitivity.png saved")

    # ============================================================
    # SUMMARY
    # ============================================================
    log_header("SUMMARY")

    surviving_base = sum(1 for s, e in ev_data.items() if e['base'] > 0)
    surviving_pess = sum(1 for s, e in ev_data.items() if e['pess'] > 0)

    log(f"  Strategies tested: {len(ev_data)}")
    log(f"  Survive Base (0.05 ADR):        {surviving_base}/{len(ev_data)}")
    log(f"  Survive Pessimistic (0.10 ADR):  {surviving_pess}/{len(ev_data)}")
    log(f"")
    log(f"  Top 5 by Base EV:")
    sorted_base = sorted(ev_data.items(), key=lambda x: x[1]['base'], reverse=True)
    for i, (strat, evs) in enumerate(sorted_base[:5], 1):
        log(f"    {i}. {strat:<28} base={evs['base']:+.3f}R  pess={evs['pess']:+.3f}R")

    elapsed = time.time() - t_start
    log(f"\n  Total runtime: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Save full log
    full_log = '\n'.join(lines)
    (RESULTS_DIR / 'd15_full_log.txt').write_text(full_log, encoding='utf-8')
    log(f"  Full log saved: {RESULTS_DIR / 'd15_full_log.txt'}")

    log("\n  DONE.")


if __name__ == '__main__':
    main()
