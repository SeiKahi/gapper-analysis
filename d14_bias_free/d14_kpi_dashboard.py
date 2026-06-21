"""
D14.5 — KPI Dashboard, Equity Curves & Monte Carlo
====================================================
Reads d14_all_trades.parquet (after gap-through fix) and produces:
  1. KPI table for ALL strategies (d14_kpi_table.txt)
  2. Correlation analysis for 5 representative strategies (d14_correlation.txt)
  3. Equity curve PNGs for strategies with OOS EV >= 0.2R
  4. Monte Carlo PNGs for strategies with OOS EV >= 0.2R

Usage:
  .\\gapper_env\\Scripts\\python.exe d14_bias_free/d14_kpi_dashboard.py
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
import time
from pathlib import Path

warnings.filterwarnings('ignore')

# ============================================================
# CONSTANTS
# ============================================================
TRADES_PATH = Path('d14_bias_free/results/d14_all_trades.parquet')
RESULTS_DIR = Path('d14_bias_free/results')
CHARTS_DIR = RESULTS_DIR / 'charts'

IS_START = '2021-02-21'
IS_END = '2023-12-31'
OOS_START = '2024-01-01'
OOS_END = '2026-02-21'

OOS_EV_THRESHOLD = 0.2  # Only plot strategies with OOS EV >= this

# Calendar days for annualization
IS_CALENDAR_DAYS = (pd.Timestamp(IS_END) - pd.Timestamp(IS_START)).days
OOS_CALENDAR_DAYS = (pd.Timestamp(OOS_END) - pd.Timestamp(OOS_START)).days
TOTAL_CALENDAR_DAYS = (pd.Timestamp(OOS_END) - pd.Timestamp(IS_START)).days

MC_RUNS = 400
MC_SEED = 42


# ============================================================
# KPI FUNCTIONS
# ============================================================
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

    # Per-trade Sharpe + annualized
    per_trade_sharpe = ev / std if std > 0 else 0
    trades_per_year = n / (calendar_days / 365.25) if calendar_days > 0 else 0
    ann_sharpe = per_trade_sharpe * np.sqrt(trades_per_year) if trades_per_year > 0 else 0

    # Drawdowns (R-based)
    cum = np.cumsum(arr)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak  # always <= 0
    max_dd = dd.min()

    # Avg drawdown depth (all DD periods)
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
    avg_dd_duration = np.mean(dd_durations) if dd_durations else 0

    # Max consecutive losers
    max_consec = 0
    current_streak = 0
    for p in arr:
        if p <= 0:
            current_streak += 1
            max_consec = max(max_consec, current_streak)
        else:
            current_streak = 0

    # CAGR (R-based, start at 100R)
    start_cap = 100.0
    final_equity = start_cap + cum[-1]
    years = calendar_days / 365.25
    if final_equity > 0 and years > 0:
        cagr = (final_equity / start_cap) ** (1 / years) - 1
    else:
        cagr = -1.0

    # Profit factor
    wins = arr[arr > 0]
    losses = arr[arr <= 0]
    pf = abs(np.sum(wins) / np.sum(losses)) if len(losses) > 0 and np.sum(losses) != 0 else float('inf')

    return {
        'N': n,
        'EV': ev,
        'WR': wr,
        'PF': pf,
        'Sharpe_ann': ann_sharpe,
        'MaxDD': max_dd,
        'AvgDD': avg_dd,
        'AvgDD_Dur': avg_dd_duration,
        'MaxConsecLoss': max_consec,
        'CAGR_pct': cagr * 100,
        'Trades_per_yr': trades_per_year,
        'Final_R': cum[-1],
    }


# ============================================================
# CORRELATION
# ============================================================
def compute_daily_pnl(trades_df, strategy):
    """Aggregate PnL per day for a strategy."""
    sub = trades_df[trades_df['strategy'] == strategy].copy()
    if len(sub) == 0:
        return pd.Series(dtype=float)
    daily = sub.groupby('date')['pnl_r'].sum()
    return daily


def correlation_matrix(trades_df, strat_list):
    """Compute Pearson correlation on daily PnL between strategies."""
    daily_pnls = {}
    for s in strat_list:
        daily_pnls[s] = compute_daily_pnl(trades_df, s)

    # Build DataFrame: all dates where at least one strategy had a trade
    all_dates = set()
    for dp in daily_pnls.values():
        all_dates.update(dp.index)
    all_dates = sorted(all_dates)

    df = pd.DataFrame(index=all_dates)
    for s in strat_list:
        df[s] = daily_pnls[s].reindex(all_dates).fillna(0)

    corr_full = df.corr()

    # Also: correlation only on days where BOTH had trades
    corr_shared = pd.DataFrame(np.nan, index=strat_list, columns=strat_list)
    for i, s1 in enumerate(strat_list):
        for j, s2 in enumerate(strat_list):
            if i == j:
                corr_shared.loc[s1, s2] = 1.0
                continue
            shared_dates = daily_pnls[s1].index.intersection(daily_pnls[s2].index)
            if len(shared_dates) >= 20:
                corr_shared.loc[s1, s2] = daily_pnls[s1][shared_dates].corr(daily_pnls[s2][shared_dates])

    return corr_full, corr_shared


# ============================================================
# EQUITY CURVE PLOT
# ============================================================
def plot_equity_curve(trades_df, strategy, kpis_is, kpis_oos, out_path):
    """Plot equity curve with IS/OOS split and dual x-axis."""
    sub = trades_df[trades_df['strategy'] == strategy].copy()
    sub = sub.sort_values('date').reset_index(drop=True)
    if len(sub) < 5:
        return

    cum_pnl = sub['pnl_r'].cumsum().values
    is_mask = sub['sample'] == 'IS'
    oos_mask = sub['sample'] == 'OOS'

    # Find split index
    is_idx = sub.index[is_mask].tolist()
    oos_idx = sub.index[oos_mask].tolist()
    split_idx = max(is_idx) if is_idx else 0

    # Parse dates for top x-axis
    dates = pd.to_datetime(sub['date'], errors='coerce')

    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Plot IS portion
    if is_idx:
        ax1.plot(is_idx, cum_pnl[is_idx], color='#2196F3', linewidth=1.2, label='IS')
    # Plot OOS portion
    if oos_idx:
        ax1.plot(oos_idx, cum_pnl[oos_idx], color='#4CAF50', linewidth=1.2, label='OOS')

    # Split line
    if is_idx and oos_idx:
        ax1.axvline(x=split_idx, color='red', linestyle='--', alpha=0.7, label='IS/OOS Split')

    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax1.set_xlabel('Trade #')
    ax1.set_ylabel('Cumulative PnL (R)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Top x-axis: dates
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    # Pick ~8 evenly spaced tick positions
    n_ticks = min(8, len(sub))
    tick_positions = np.linspace(0, len(sub) - 1, n_ticks, dtype=int)
    ax2.set_xticks(tick_positions)
    tick_labels = []
    for tp in tick_positions:
        d = dates.iloc[tp]
        tick_labels.append(d.strftime('%Y-%m') if pd.notna(d) else '')
    ax2.set_xticklabels(tick_labels, fontsize=8)

    # Title with KPIs
    ev_is = f"{kpis_is['EV']:+.3f}" if kpis_is else 'N/A'
    ev_oos = f"{kpis_oos['EV']:+.3f}" if kpis_oos else 'N/A'
    sharpe = f"{kpis_oos['Sharpe_ann']:.2f}" if kpis_oos else 'N/A'
    maxdd = f"{kpis_oos['MaxDD']:.1f}" if kpis_oos else 'N/A'
    n_total = len(sub)
    ax1.set_title(f'{strategy}  |  EV(IS)={ev_is}  EV(OOS)={ev_oos}  '
                   f'Sharpe={sharpe}  MaxDD={maxdd}R  N={n_total}',
                   fontsize=11, pad=25)

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# MONTE CARLO PLOT
# ============================================================
def plot_monte_carlo(trades_df, strategy, out_path):
    """Monte Carlo simulation: shuffle trades 400x, plot percentile bands."""
    sub = trades_df[trades_df['strategy'] == strategy]
    pnl_arr = sub['pnl_r'].dropna().values
    n = len(pnl_arr)
    if n < 20:
        return

    rng = np.random.RandomState(MC_SEED)
    mc_curves = np.zeros((MC_RUNS, n))
    mc_maxdd = np.zeros(MC_RUNS)

    for i in range(MC_RUNS):
        shuffled = rng.permutation(pnl_arr)
        mc_curves[i] = np.cumsum(shuffled)
        peak = np.maximum.accumulate(mc_curves[i])
        mc_maxdd[i] = (mc_curves[i] - peak).min()

    # Percentiles at each trade index
    p10 = np.percentile(mc_curves, 10, axis=0)
    p50 = np.percentile(mc_curves, 50, axis=0)
    p90 = np.percentile(mc_curves, 90, axis=0)

    # Actual curve
    actual = np.cumsum(pnl_arr)

    fig, ax = plt.subplots(figsize=(14, 7))

    # MC curves (transparent grey)
    for i in range(MC_RUNS):
        ax.plot(mc_curves[i], color='grey', alpha=0.03, linewidth=0.5)

    # Percentile bands
    x = np.arange(n)
    ax.plot(x, p10, color='black', linestyle='--', linewidth=1.0, label='10th / 90th pctl')
    ax.plot(x, p90, color='black', linestyle='--', linewidth=1.0)
    ax.plot(x, p50, color='black', linestyle='-', linewidth=1.0, alpha=0.5, label='Median')

    # Actual curve
    ax.plot(x, actual, color='red', linewidth=2, label='Actual', zorder=5)

    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Trade #')
    ax.set_ylabel('Cumulative PnL (R)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # Annotation: final R range + MaxDD range
    final_10 = p10[-1]
    final_50 = p50[-1]
    final_90 = p90[-1]
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

    ax.set_title(f'{strategy}  —  Monte Carlo ({MC_RUNS} runs, N={n})',
                 fontsize=11)

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time.time()
    print("=" * 70)
    print("  D14.5 — KPI DASHBOARD, EQUITY CURVES & MONTE CARLO")
    print("=" * 70)

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load trades
    df = pd.read_parquet(TRADES_PATH)
    print(f"  Loaded {len(df)} trades, {df['strategy'].nunique()} strategies")

    strategies = sorted(df['strategy'].unique())

    # ================================================================
    # 1. KPI TABLE — ALL STRATEGIES
    # ================================================================
    print("\n  [1/4] Computing KPIs...")
    kpi_rows = []

    for strat in strategies:
        sub = df[df['strategy'] == strat]
        sub_is = sub[sub['sample'] == 'IS']
        sub_oos = sub[sub['sample'] == 'OOS']
        sub_all = sub

        kpis_is = compute_kpis(sub_is['pnl_r'].values, IS_CALENDAR_DAYS)
        kpis_oos = compute_kpis(sub_oos['pnl_r'].values, OOS_CALENDAR_DAYS)
        kpis_all = compute_kpis(sub_all['pnl_r'].values, TOTAL_CALENDAR_DAYS)

        row = {'Strategy': strat}
        for prefix, k in [('IS', kpis_is), ('OOS', kpis_oos), ('ALL', kpis_all)]:
            if k:
                for metric in ['N', 'EV', 'WR', 'PF', 'Sharpe_ann', 'MaxDD',
                               'AvgDD', 'AvgDD_Dur', 'MaxConsecLoss', 'CAGR_pct',
                               'Trades_per_yr', 'Final_R']:
                    row[f'{prefix}_{metric}'] = k[metric]
            else:
                for metric in ['N', 'EV', 'WR', 'PF', 'Sharpe_ann', 'MaxDD',
                               'AvgDD', 'AvgDD_Dur', 'MaxConsecLoss', 'CAGR_pct',
                               'Trades_per_yr', 'Final_R']:
                    row[f'{prefix}_{metric}'] = np.nan
        kpi_rows.append(row)

    kpi_df = pd.DataFrame(kpi_rows)

    # Write KPI table as formatted text
    lines = []
    lines.append("D14.5 — KPI TABLE (ALL STRATEGIES)")
    lines.append("=" * 140)
    lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Trades: {len(df)} total | IS: {IS_START} to {IS_END} | OOS: {OOS_START} to {OOS_END}")
    lines.append("")

    # Sort by OOS EV descending
    kpi_df = kpi_df.sort_values('OOS_EV', ascending=False)

    header = (f"  {'Strategy':<35} | {'N_IS':>5} {'N_OOS':>6} | "
              f"{'EV_IS':>7} {'EV_OOS':>7} | {'WR_OOS':>6} | "
              f"{'Sharpe':>6} | {'MaxDD':>7} | {'AvgDD':>7} | "
              f"{'MCL':>3} | {'CAGR%':>7} | {'T/yr':>5} | {'FinalR':>8}")
    lines.append(header)
    lines.append("  " + "-" * 135)

    for _, r in kpi_df.iterrows():
        ev_is = f"{r['IS_EV']:+.3f}" if not np.isnan(r.get('IS_EV', np.nan)) else "  N/A"
        ev_oos = f"{r['OOS_EV']:+.3f}" if not np.isnan(r.get('OOS_EV', np.nan)) else "  N/A"
        wr_oos = f"{r['OOS_WR']:.1%}" if not np.isnan(r.get('OOS_WR', np.nan)) else " N/A"
        sharpe = f"{r['OOS_Sharpe_ann']:.2f}" if not np.isnan(r.get('OOS_Sharpe_ann', np.nan)) else " N/A"
        maxdd = f"{r['OOS_MaxDD']:.1f}" if not np.isnan(r.get('OOS_MaxDD', np.nan)) else "  N/A"
        avgdd = f"{r['OOS_AvgDD']:.2f}" if not np.isnan(r.get('OOS_AvgDD', np.nan)) else "  N/A"
        mcl = f"{r['OOS_MaxConsecLoss']:.0f}" if not np.isnan(r.get('OOS_MaxConsecLoss', np.nan)) else "N/A"
        cagr = f"{r['OOS_CAGR_pct']:+.1f}" if not np.isnan(r.get('OOS_CAGR_pct', np.nan)) else "  N/A"
        tpy = f"{r['OOS_Trades_per_yr']:.0f}" if not np.isnan(r.get('OOS_Trades_per_yr', np.nan)) else " N/A"
        finalr = f"{r['ALL_Final_R']:+.1f}" if not np.isnan(r.get('ALL_Final_R', np.nan)) else "  N/A"
        n_is = f"{r['IS_N']:.0f}" if not np.isnan(r.get('IS_N', np.nan)) else "  N/A"
        n_oos = f"{r['OOS_N']:.0f}" if not np.isnan(r.get('OOS_N', np.nan)) else "  N/A"

        lines.append(f"  {r['Strategy']:<35} | {n_is:>5} {n_oos:>6} | "
                     f"{ev_is:>7} {ev_oos:>7} | {wr_oos:>6} | "
                     f"{sharpe:>6} | {maxdd:>7} | {avgdd:>7} | "
                     f"{mcl:>3} | {cagr:>7} | {tpy:>5} | {finalr:>8}")

    kpi_text = '\n'.join(lines)
    (RESULTS_DIR / 'd14_kpi_table.txt').write_text(kpi_text, encoding='utf-8')
    print(f"  KPI table saved: {RESULTS_DIR / 'd14_kpi_table.txt'}")

    # ================================================================
    # 2. CORRELATION ANALYSIS
    # ================================================================
    print("\n  [2/4] Computing correlations...")

    # Pick 5 representative strategies (must exist in data with N >= 300 OOS)
    oos_df = df[df['sample'] == 'OOS']
    oos_counts = oos_df.groupby('strategy').size()

    # Candidate list in preference order
    candidates = [
        '2.3_Dip_and_Rip',
        '3_FAILED_FADE_15:59',
        '3_BASELINE_15:59',
        '3_TREND_DAY_14:30',
        '3_STRONG_POS_15:00',
        '3_POS_DAY_15:59',
        '3_VWAP_ABOVE_15:59',
        '1.8_Gap3ADR',
        '1.9_RVOL5_RSI50',
    ]
    corr_strats = []
    for c in candidates:
        if c in oos_counts and oos_counts[c] >= 100:
            corr_strats.append(c)
        if len(corr_strats) >= 5:
            break

    # Fallback: if not enough, pick top-N by count
    if len(corr_strats) < 5:
        remaining = [s for s in oos_counts.sort_values(ascending=False).index
                     if s not in corr_strats and oos_counts[s] >= 100]
        corr_strats.extend(remaining[:5 - len(corr_strats)])

    corr_lines = []
    corr_lines.append("D14.5 — STRATEGY CORRELATION ANALYSIS")
    corr_lines.append("=" * 100)
    corr_lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    corr_lines.append(f"Sample: OOS only ({OOS_START} to {OOS_END})")
    corr_lines.append(f"\nRepresentative Strategies:")
    for i, s in enumerate(corr_strats, 1):
        n = oos_counts.get(s, 0)
        ev = oos_df[oos_df['strategy'] == s]['pnl_r'].mean()
        corr_lines.append(f"  {i}. {s} (N={n}, EV={ev:+.3f}R)")

    if len(corr_strats) >= 2:
        oos_trades = df[df['sample'] == 'OOS']
        corr_full, corr_shared = correlation_matrix(oos_trades, corr_strats)

        # Short labels
        short = {s: s[:25] for s in corr_strats}

        corr_lines.append(f"\n--- Pearson Correlation (daily PnL, 0-filled) ---")
        # Header
        hdr = f"  {'':>25} | " + " | ".join(f"{short[s]:>25}" for s in corr_strats)
        corr_lines.append(hdr)
        corr_lines.append("  " + "-" * (28 + 28 * len(corr_strats)))
        for s1 in corr_strats:
            vals = []
            for s2 in corr_strats:
                v = corr_full.loc[s1, s2]
                vals.append(f"{v:>25.3f}" if not np.isnan(v) else f"{'N/A':>25}")
            corr_lines.append(f"  {short[s1]:>25} | " + " | ".join(vals))

        corr_lines.append(f"\n--- Pearson Correlation (shared trading days only) ---")
        hdr = f"  {'':>25} | " + " | ".join(f"{short[s]:>25}" for s in corr_strats)
        corr_lines.append(hdr)
        corr_lines.append("  " + "-" * (28 + 28 * len(corr_strats)))
        for s1 in corr_strats:
            vals = []
            for s2 in corr_strats:
                v = corr_shared.loc[s1, s2]
                vals.append(f"{v:>25.3f}" if pd.notna(v) else f"{'N/A':>25}")
            corr_lines.append(f"  {short[s1]:>25} | " + " | ".join(vals))

        # Summary
        off_diag = []
        for i, s1 in enumerate(corr_strats):
            for j, s2 in enumerate(corr_strats):
                if i < j:
                    v = corr_full.loc[s1, s2]
                    if not np.isnan(v):
                        off_diag.append(v)
        if off_diag:
            corr_lines.append(f"\nOff-diagonal mean: {np.mean(off_diag):.3f}")
            corr_lines.append(f"Off-diagonal max:  {np.max(off_diag):.3f}")
            corr_lines.append(f"Off-diagonal min:  {np.min(off_diag):.3f}")
    else:
        corr_lines.append("\n  [Insufficient strategies for correlation analysis]")

    corr_text = '\n'.join(corr_lines)
    (RESULTS_DIR / 'd14_correlation.txt').write_text(corr_text, encoding='utf-8')
    print(f"  Correlation saved: {RESULTS_DIR / 'd14_correlation.txt'}")

    # ================================================================
    # 3. IDENTIFY STRATEGIES WITH OOS EV >= 0.2R
    # ================================================================
    oos_ev = oos_df.groupby('strategy')['pnl_r'].mean()
    qualifying = oos_ev[oos_ev >= OOS_EV_THRESHOLD].sort_values(ascending=False)
    print(f"\n  Strategies with OOS EV >= {OOS_EV_THRESHOLD}R: {len(qualifying)}")
    for s in qualifying.index:
        print(f"    {s:<35} EV={qualifying[s]:+.3f}R")

    # Build IS/OOS KPI lookup
    kpi_lookup_is = {}
    kpi_lookup_oos = {}
    for _, r in kpi_df.iterrows():
        strat = r['Strategy']
        kpi_lookup_is[strat] = {m: r.get(f'IS_{m}', np.nan) for m in
                                ['N', 'EV', 'WR', 'PF', 'Sharpe_ann', 'MaxDD']}
        kpi_lookup_oos[strat] = {m: r.get(f'OOS_{m}', np.nan) for m in
                                 ['N', 'EV', 'WR', 'PF', 'Sharpe_ann', 'MaxDD']}

    # ================================================================
    # 4. EQUITY CURVES
    # ================================================================
    print(f"\n  [3/4] Plotting equity curves ({len(qualifying)} strategies)...")
    eq_count = 0
    for strat in qualifying.index:
        safe_name = strat.replace(':', '').replace(' ', '_')
        out_path = CHARTS_DIR / f"eq_{safe_name}.png"
        kpis_is = kpi_lookup_is.get(strat)
        kpis_oos = kpi_lookup_oos.get(strat)
        plot_equity_curve(df, strat, kpis_is, kpis_oos, out_path)
        eq_count += 1
    print(f"  Equity curves saved: {eq_count} PNGs in {CHARTS_DIR}")

    # ================================================================
    # 5. MONTE CARLO
    # ================================================================
    print(f"\n  [4/4] Plotting Monte Carlo ({len(qualifying)} strategies)...")
    mc_count = 0
    for strat in qualifying.index:
        safe_name = strat.replace(':', '').replace(' ', '_')
        out_path = CHARTS_DIR / f"mc_{safe_name}.png"
        plot_monte_carlo(df, strat, out_path)
        mc_count += 1
    print(f"  Monte Carlo saved: {mc_count} PNGs in {CHARTS_DIR}")

    # ================================================================
    # SUMMARY
    # ================================================================
    elapsed = time.time() - t0
    print(f"\n  {'='*60}")
    print(f"  DONE in {elapsed:.1f}s")
    print(f"  Outputs:")
    print(f"    {RESULTS_DIR / 'd14_kpi_table.txt'}")
    print(f"    {RESULTS_DIR / 'd14_correlation.txt'}")
    print(f"    {CHARTS_DIR}/eq_*.png  ({eq_count} files)")
    print(f"    {CHARTS_DIR}/mc_*.png  ({mc_count} files)")
    print(f"  {'='*60}")


if __name__ == '__main__':
    main()
