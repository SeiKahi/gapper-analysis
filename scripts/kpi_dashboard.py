"""
KPI Dashboard — CTRL Strategy (TRAIL_D)
Berechnet alle relevanten KPIs und erstellt Grafiken.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
R_DOLLAR = 1000          # 1R = $1000
START_CAPITAL = 50_000
TRADING_DAYS_YEAR = 252
RISK_FREE = 0.0
MC_SIMS = 1000
MC_SEED = 42
ENTRY_TIME = "09:35:00"  # Feste Entry-Zeit

OUT_DIR = Path("results/kpi_dashboard")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Chart-Style
DARK_BG = '#1a1a2e'
CYAN = '#00b4d8'
GREEN = '#00ff88'
RED = '#ff4466'
YELLOW = '#ffd60a'
GREY = '#888888'

# ============================================================
# LOAD DATA
# ============================================================
print("=" * 70)
print("LOADING DATA")
print("=" * 70)

try:
    df = pd.read_parquet('results/backtest_v3/all_trades_v3.parquet')
    print(f"Loaded: all_trades_v3.parquet ({len(df)} rows)")
except FileNotFoundError:
    df = pd.read_parquet('results/backtest_v2/all_trades_v2.parquet')
    print(f"Loaded: all_trades_v2.parquet ({len(df)} rows)")

# Filter CTRL
ctrl = df[df['approach'] == 'CTRL'].copy()
print(f"CTRL trades: {len(ctrl)}")

# Parse date
ctrl['date'] = pd.to_datetime(ctrl['date'])

# Parse exit_time -> duration in minutes from 09:35
def parse_exit_time(et):
    """Parse exit_time string to timedelta from 09:35."""
    try:
        parts = str(et).split(':')
        h, m = int(parts[0]), int(parts[1])
        s = int(parts[2]) if len(parts) > 2 else 0
        exit_td = h * 60 + m + s / 60.0
        entry_td = 9 * 60 + 35
        return exit_td - entry_td
    except:
        return np.nan

ctrl['duration_min'] = ctrl['exit_time'].apply(parse_exit_time)

# Weekday (0=Mon, 4=Fri)
ctrl['weekday'] = ctrl['date'].dt.dayofweek
ctrl['year_month'] = ctrl['date'].dt.to_period('M')
ctrl['year'] = ctrl['date'].dt.year
ctrl['month'] = ctrl['date'].dt.month
ctrl['quarter'] = ctrl['date'].dt.to_period('Q')

# ============================================================
# DEFINE SEGMENTS
# ============================================================
segments = {
    'Tier2_GapUp_OOS': ctrl[(ctrl['period'] == 'OOS') & (ctrl['gap_direction'] == 'up')],
    'Tier2_GapDn_OOS': ctrl[(ctrl['period'] == 'OOS') & (ctrl['gap_direction'] == 'down')],
    'Tier1_GapUp_OOS': ctrl[(ctrl['period'] == 'OOS') & (ctrl['gap_direction'] == 'up') & (ctrl['is_tier1'] == True)],
    'Tier1_GapDn_OOS': ctrl[(ctrl['period'] == 'OOS') & (ctrl['gap_direction'] == 'down') & (ctrl['is_tier1'] == True)],
    'GESAMT_OOS':      ctrl[ctrl['period'] == 'OOS'],
    'GESAMT_IS':       ctrl[ctrl['period'] == 'IS'],
}

for name, seg in segments.items():
    print(f"  {name}: N={len(seg)}")

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def fmt_dollar(v):
    """Format dollar value with comma separator."""
    if pd.isna(v):
        return "N/A"
    sign = "-" if v < 0 else ""
    return f"{sign}${abs(v):,.0f}"

def fmt_pct(v):
    if pd.isna(v):
        return "N/A"
    return f"{v:.1f}%"

def fmt_r(v):
    if pd.isna(v):
        return "N/A"
    return f"{v:+.3f}R"

def fmt_num(v, decimals=2):
    if pd.isna(v):
        return "N/A"
    return f"{v:.{decimals}f}"

def compute_equity(pnl_series, start=START_CAPITAL):
    """Compute equity curve from PnL series."""
    equity = start + pnl_series.cumsum()
    return equity

def compute_drawdown(equity):
    """Compute drawdown series."""
    running_max = equity.cummax()
    dd = equity - running_max
    return dd

def compute_daily_returns(seg_df, start_date=None, end_date=None):
    """Compute daily return series including zero-days."""
    if start_date is None:
        start_date = seg_df['date'].min()
    if end_date is None:
        end_date = seg_df['date'].max()

    bdays = pd.bdate_range(start_date, end_date)
    daily_pnl = seg_df.groupby('date')['pnl_dollar'].sum()
    daily_pnl = daily_pnl.reindex(bdays, fill_value=0.0)
    daily_ret = daily_pnl / START_CAPITAL
    return daily_ret, daily_pnl


# ============================================================
# KPI COMPUTATION
# ============================================================

def compute_all_kpis(seg_df, seg_name, oos_start='2024-01-01', oos_end='2026-02-13'):
    """Compute all KPIs for a segment."""
    kpis = {}
    kpis['segment'] = seg_name
    n = len(seg_df)
    kpis['N'] = n

    if n == 0:
        return kpis

    pnl = seg_df['pnl_dollar'].values
    pnl_r = seg_df['pnl_r'].values

    # ---- BLOCK 1: TRADE-STATISTIK ----
    winners = pnl > 0
    losers = pnl < 0
    breakeven = np.abs(pnl) < 1.0

    kpis['win_rate'] = winners.sum() / n * 100
    kpis['loss_rate'] = losers.sum() / n * 100
    kpis['be_rate'] = breakeven.sum() / n * 100
    kpis['wl_ratio'] = winners.sum() / max(losers.sum(), 1)

    # Trades per month / week
    months_spanned = seg_df['year_month'].nunique()
    date_range_days = (seg_df['date'].max() - seg_df['date'].min()).days
    weeks_spanned = max(date_range_days / 7, 1)

    trades_per_month = seg_df.groupby('year_month').size()
    trades_per_week = seg_df.groupby(seg_df['date'].dt.isocalendar().week.astype(int) + seg_df['date'].dt.year * 100).size()

    kpis['trades_per_month_mean'] = trades_per_month.mean()
    kpis['trades_per_month_median'] = trades_per_month.median()
    kpis['trades_per_week_mean'] = trades_per_week.mean()
    kpis['trades_per_week_median'] = trades_per_week.median()

    # ---- BLOCK 2: PnL-VERTEILUNG ----
    kpis['total_pnl'] = pnl.sum()
    kpis['mean_pnl'] = pnl.mean()
    kpis['median_pnl'] = np.median(pnl)
    kpis['std_pnl'] = pnl.std()
    kpis['skew_pnl'] = stats.skew(pnl)
    kpis['kurt_pnl'] = stats.kurtosis(pnl)

    best_idx = seg_df['pnl_dollar'].idxmax()
    worst_idx = seg_df['pnl_dollar'].idxmin()
    kpis['best_trade'] = seg_df.loc[best_idx, 'pnl_dollar']
    kpis['best_ticker'] = seg_df.loc[best_idx, 'ticker']
    kpis['best_date'] = str(seg_df.loc[best_idx, 'date'].date())
    kpis['worst_trade'] = seg_df.loc[worst_idx, 'pnl_dollar']
    kpis['worst_ticker'] = seg_df.loc[worst_idx, 'ticker']
    kpis['worst_date'] = str(seg_df.loc[worst_idx, 'date'].date())

    for p in [5, 10, 25, 50, 75, 90, 95]:
        kpis[f'P{p}_pnl'] = np.percentile(pnl, p)

    # ---- BLOCK 3: R-MULTIPLE VERTEILUNG ----
    kpis['mean_r'] = pnl_r.mean()
    kpis['median_r'] = np.median(pnl_r)
    kpis['std_r'] = pnl_r.std()
    kpis['skew_r'] = stats.skew(pnl_r)

    for threshold in [1, 2, 3, 5, 10]:
        kpis[f'WR_at_{threshold}R'] = (pnl_r >= threshold).sum() / n * 100

    kpis['expectancy_r'] = pnl_r.mean()

    # ---- BLOCK 4: RISIKO-METRIKEN ----
    gross_profit = pnl[pnl > 0].sum() if (pnl > 0).any() else 0
    gross_loss = abs(pnl[pnl < 0].sum()) if (pnl < 0).any() else 1
    kpis['profit_factor'] = gross_profit / max(gross_loss, 1)

    mean_win = pnl[pnl > 0].mean() if (pnl > 0).any() else 0
    mean_loss = abs(pnl[pnl < 0].mean()) if (pnl < 0).any() else 1
    kpis['payoff_ratio'] = mean_win / max(mean_loss, 1)

    equity = compute_equity(pd.Series(pnl))
    dd = compute_drawdown(equity)
    kpis['max_dd_dollar'] = dd.min()
    kpis['max_dd_pct'] = dd.min() / START_CAPITAL * 100

    # Max DD Dauer (in Trades)
    running_max = equity.cummax()
    in_dd = equity < running_max
    dd_groups = (~in_dd).cumsum()
    if in_dd.any():
        dd_durations = in_dd.groupby(dd_groups).sum()
        kpis['max_dd_duration'] = int(dd_durations.max())
    else:
        kpis['max_dd_duration'] = 0

    # Recovery (trades to new equity high after max DD)
    max_dd_idx = dd.idxmin()
    post_dd = equity.iloc[max_dd_idx:]
    recovery_mask = post_dd >= running_max.iloc[max_dd_idx]
    if recovery_mask.any():
        recovery_idx = recovery_mask.idxmax()
        kpis['max_dd_recovery'] = int(recovery_idx - max_dd_idx)
    else:
        kpis['max_dd_recovery'] = len(post_dd)  # Not recovered

    # Annualized return
    years = max(date_range_days / 365.25, 0.1)
    annual_return = kpis['total_pnl'] / years
    kpis['annual_return'] = annual_return
    kpis['calmar'] = annual_return / max(abs(kpis['max_dd_dollar']), 1)

    # Ulcer Index
    dd_pct = dd / START_CAPITAL * 100
    kpis['ulcer_index'] = np.sqrt((dd_pct ** 2).mean())

    # ---- BLOCK 5: SHARPE & RISK-ADJUSTED ----
    if 'OOS' in seg_name:
        daily_ret, daily_pnl = compute_daily_returns(seg_df, oos_start, oos_end)
    else:
        daily_ret, daily_pnl = compute_daily_returns(seg_df)

    if daily_ret.std() > 0:
        kpis['sharpe'] = daily_ret.mean() / daily_ret.std() * np.sqrt(TRADING_DAYS_YEAR)
    else:
        kpis['sharpe'] = 0.0

    downside = daily_ret[daily_ret < 0]
    if len(downside) > 1 and downside.std() > 1e-10:
        sortino_raw = daily_ret.mean() / downside.std() * np.sqrt(TRADING_DAYS_YEAR)
        kpis['sortino'] = min(sortino_raw, 99.99)  # Cap at 99.99
    else:
        kpis['sortino'] = np.nan  # Not computable

    kpis['mar'] = annual_return / max(abs(kpis['max_dd_dollar']), 1)
    kpis['risk_adj_return'] = kpis['total_pnl'] / max(abs(kpis['max_dd_dollar']), 1)

    # Kelly
    wr = winners.sum() / n
    if mean_loss > 0 and kpis['payoff_ratio'] > 0:
        kelly = (wr * kpis['payoff_ratio'] - (1 - wr)) / kpis['payoff_ratio']
    else:
        kelly = 0
    kpis['kelly'] = kelly
    kpis['half_kelly'] = kelly / 2

    # Expectancy * Frequency
    trades_per_year = n / max(years, 0.1)
    kpis['trades_per_year'] = trades_per_year
    kpis['exp_x_freq'] = kpis['mean_pnl'] * trades_per_year

    # ---- BLOCK 6: STREAKS ----
    signs = np.sign(pnl)
    signs[np.abs(pnl) < 1] = 0  # breakeven

    def streak_stats(signs, target):
        streaks = []
        current = 0
        for s in signs:
            if s == target:
                current += 1
            else:
                if current > 0:
                    streaks.append(current)
                current = 0
        if current > 0:
            streaks.append(current)
        return streaks

    win_streaks = streak_stats(signs, 1)
    loss_streaks = streak_stats(signs, -1)
    be_streaks = streak_stats(signs, 0)

    kpis['max_consec_wins'] = max(win_streaks) if win_streaks else 0
    kpis['max_consec_losses'] = max(loss_streaks) if loss_streaks else 0
    kpis['max_consec_be'] = max(be_streaks) if be_streaks else 0
    kpis['mean_win_streak'] = np.mean(win_streaks) if win_streaks else 0
    kpis['mean_loss_streak'] = np.mean(loss_streaks) if loss_streaks else 0

    # Longest flat phase (no new equity high, in trading days)
    eq_dates = seg_df[['date', 'pnl_dollar']].sort_values('date')
    eq_by_day = eq_dates.groupby('date')['pnl_dollar'].sum().cumsum() + START_CAPITAL
    running_max_daily = eq_by_day.cummax()
    flat = eq_by_day < running_max_daily
    if flat.any():
        flat_groups = (~flat).cumsum()
        flat_durations = flat.groupby(flat_groups).sum()
        kpis['longest_flat_days'] = int(flat_durations.max())
    else:
        kpis['longest_flat_days'] = 0

    # Comeback trade (biggest win after a loss streak)
    biggest_comeback = 0
    in_loss_streak = False
    for i, p in enumerate(pnl):
        if p < 0:
            in_loss_streak = True
        elif in_loss_streak and p > 0:
            biggest_comeback = max(biggest_comeback, p)
            in_loss_streak = False
        else:
            in_loss_streak = False
    kpis['biggest_comeback'] = biggest_comeback

    # ---- BLOCK 7: ZEITLICHE ANALYSE ----
    monthly_pnl = seg_df.groupby('year_month')['pnl_dollar'].sum()
    kpis['monthly_mean'] = monthly_pnl.mean()
    kpis['monthly_median'] = monthly_pnl.median()
    kpis['monthly_std'] = monthly_pnl.std()

    best_month_idx = monthly_pnl.idxmax()
    worst_month_idx = monthly_pnl.idxmin()
    kpis['best_month'] = monthly_pnl.max()
    kpis['best_month_date'] = str(best_month_idx)
    kpis['worst_month'] = monthly_pnl.min()
    kpis['worst_month_date'] = str(worst_month_idx)
    kpis['pct_profitable_months'] = (monthly_pnl > 0).sum() / len(monthly_pnl) * 100

    quarterly_pnl = seg_df.groupby('quarter')['pnl_dollar'].sum()
    kpis['pct_profitable_quarters'] = (quarterly_pnl > 0).sum() / len(quarterly_pnl) * 100

    kpis['annualized_return_pct'] = annual_return / START_CAPITAL * 100
    total_return = kpis['total_pnl'] / START_CAPITAL
    kpis['cagr'] = ((1 + total_return) ** (1 / max(years, 0.1)) - 1) * 100

    # Daily stats
    daily_trade_counts = seg_df.groupby('date').size()
    kpis['mean_pnl_trade_days'] = daily_pnl[daily_pnl != 0].mean() if (daily_pnl != 0).any() else 0
    bdays_total = len(daily_ret)
    kpis['pct_days_with_trade'] = (daily_pnl != 0).sum() / max(bdays_total, 1) * 100
    kpis['max_trades_per_day'] = int(daily_trade_counts.max())

    # Correlation trades/day vs daily PnL
    day_counts = seg_df.groupby('date').size()
    day_pnl = seg_df.groupby('date')['pnl_dollar'].sum()
    merged = pd.DataFrame({'count': day_counts, 'pnl': day_pnl}).dropna()
    if len(merged) > 2:
        kpis['corr_trades_day_pnl'] = merged['count'].corr(merged['pnl'])
    else:
        kpis['corr_trades_day_pnl'] = np.nan

    # ---- BLOCK 8: TRADE-DAUER ----
    dur = seg_df['duration_min'].dropna()
    if len(dur) > 0:
        kpis['median_duration'] = dur.median()
        kpis['mean_duration'] = dur.mean()
        kpis['p90_duration'] = dur.quantile(0.9)
        kpis['max_duration'] = dur.max()

        winners_dur = seg_df.loc[seg_df['pnl_dollar'] > 0, 'duration_min'].dropna()
        losers_dur = seg_df.loc[seg_df['pnl_dollar'] < 0, 'duration_min'].dropna()
        kpis['median_dur_winners'] = winners_dur.median() if len(winners_dur) > 0 else np.nan
        kpis['median_dur_losers'] = losers_dur.median() if len(losers_dur) > 0 else np.nan

        if len(dur) > 2:
            kpis['corr_dur_pnl'] = seg_df[['duration_min', 'pnl_dollar']].dropna().corr().iloc[0, 1]
        else:
            kpis['corr_dur_pnl'] = np.nan

    return kpis


# ============================================================
# COMPUTE ALL SEGMENTS
# ============================================================
print("\n" + "=" * 70)
print("COMPUTING KPIs")
print("=" * 70)

all_kpis = {}
for seg_name, seg_df in segments.items():
    print(f"  Computing: {seg_name} (N={len(seg_df)})")
    kpis = compute_all_kpis(seg_df, seg_name)
    all_kpis[seg_name] = kpis

# ============================================================
# BLOCK 9: SETUP-VERGLEICH (separate)
# ============================================================
print("  Computing: Setup comparisons")

def welch_ttest(a, b):
    """Welch t-test on two arrays."""
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return t, p

comparisons = {}

# GapUp vs GapDn OOS
oos_up = ctrl[(ctrl['period'] == 'OOS') & (ctrl['gap_direction'] == 'up')]
oos_dn = ctrl[(ctrl['period'] == 'OOS') & (ctrl['gap_direction'] == 'down')]
t, p = welch_ttest(oos_up['pnl_r'].values, oos_dn['pnl_r'].values)
comparisons['GapUp_vs_GapDn'] = {'t': t, 'p': p}

# Tier1 vs Tier2 OOS
oos_t1 = ctrl[(ctrl['period'] == 'OOS') & (ctrl['is_tier1'] == True)]
oos_t2 = ctrl[(ctrl['period'] == 'OOS')]
t, p = welch_ttest(oos_t1['pnl_r'].values, oos_t2['pnl_r'].values)
comparisons['Tier1_vs_Tier2'] = {'t': t, 'p': p}

# IS vs OOS Shrinkage
shrinkage = {}
is_kpis = all_kpis['GESAMT_IS']
oos_kpis = all_kpis['GESAMT_OOS']
for metric in ['mean_r', 'win_rate', 'profit_factor', 'sharpe', 'mean_pnl']:
    if metric in is_kpis and metric in oos_kpis:
        is_val = is_kpis[metric]
        oos_val = oos_kpis[metric]
        if is_val != 0:
            shrinkage[metric] = (oos_val - is_val) / abs(is_val) * 100
        else:
            shrinkage[metric] = np.nan

# ============================================================
# BLOCK 10: MONTE CARLO
# ============================================================
print("  Computing: Monte Carlo (1000 sims)...")

mc_seg = segments['Tier2_GapUp_OOS']
mc_pnl = mc_seg['pnl_dollar'].values.copy()
np.random.seed(MC_SEED)

mc_max_dd = []
mc_max_consec_losses = []
mc_equity_paths = []
mc_min_equity = []  # Tiefster Punkt auf dem Pfad

for i in range(MC_SIMS):
    shuffled = np.random.permutation(mc_pnl)
    eq = START_CAPITAL + np.cumsum(shuffled)
    rm = np.maximum.accumulate(eq)
    dd = eq - rm

    mc_max_dd.append(dd.min())
    mc_min_equity.append(eq.min())

    # Max consecutive losses
    signs = np.sign(shuffled)
    max_cl = 0
    current = 0
    for s in signs:
        if s < 0:
            current += 1
            max_cl = max(max_cl, current)
        else:
            current = 0
    mc_max_consec_losses.append(max_cl)

    if i < 100:  # Store first 100 paths for chart
        mc_equity_paths.append(eq)

mc_final_equity_fixed = START_CAPITAL + mc_pnl.sum()  # Always the same
mc_max_dd = np.array(mc_max_dd)
mc_max_consec_losses = np.array(mc_max_consec_losses)
mc_min_equity = np.array(mc_min_equity)

mc_results = {
    'final_equity': mc_final_equity_fixed,
    'median_min_equity': np.median(mc_min_equity),
    'p5_min_equity': np.percentile(mc_min_equity, 5),
    'p25_min_equity': np.percentile(mc_min_equity, 25),
    'p75_min_equity': np.percentile(mc_min_equity, 75),
    'p95_min_equity': np.percentile(mc_min_equity, 95),
    'median_max_dd': np.median(mc_max_dd),
    'p5_max_dd': np.percentile(mc_max_dd, 5),
    'p25_max_dd': np.percentile(mc_max_dd, 25),
    'p75_max_dd': np.percentile(mc_max_dd, 75),
    'p95_max_dd': np.percentile(mc_max_dd, 95),
    'worst_max_dd': mc_max_dd.min(),
    'p_dd_10k': (mc_max_dd < -10_000).mean() * 100,
    'p_dd_15k': (mc_max_dd < -15_000).mean() * 100,
    'p_dd_20k': (mc_max_dd < -20_000).mean() * 100,
    'median_max_cl': np.median(mc_max_consec_losses),
    'p95_max_cl': np.percentile(mc_max_consec_losses, 95),
}

# MC Sharpe CI
mc_sharpes = []
np.random.seed(MC_SEED)
oos_daily_ret, _ = compute_daily_returns(mc_seg, '2024-01-01', '2026-02-13')
daily_ret_arr = oos_daily_ret.values
for i in range(MC_SIMS):
    boot = np.random.choice(daily_ret_arr, size=len(daily_ret_arr), replace=True)
    if boot.std() > 0:
        mc_sharpes.append(boot.mean() / boot.std() * np.sqrt(252))
mc_sharpes = np.array(mc_sharpes)
mc_results['sharpe_ci_lo'] = np.percentile(mc_sharpes, 2.5)
mc_results['sharpe_ci_hi'] = np.percentile(mc_sharpes, 97.5)

print("  Monte Carlo done.")

# ============================================================
# FORMAT REPORT
# ============================================================
print("\n" + "=" * 70)
print("GENERATING REPORT")
print("=" * 70)

lines = []
def L(text=""):
    lines.append(text)

L("=" * 78)
L("  KPI DASHBOARD — CTRL STRATEGY (TRAIL_D)")
L("  Generated: " + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'))
L("  1R = $1,000 | Start Capital = $50,000 | 252 Trading Days/Year")
L("=" * 78)

# ---- Block 1 ----
L()
L("=" * 78)
L("  BLOCK 1: TRADE-STATISTIK")
L("=" * 78)

header = f"{'Metric':<30}"
for seg_name in segments:
    header += f"  {seg_name:>14}"
L(header)
L("-" * (30 + 16 * len(segments)))

block1_metrics = [
    ('N', 'N', lambda v: f"{v:>14,d}" if not pd.isna(v) else f"{'N/A':>14}"),
    ('Win Rate %', 'win_rate', lambda v: f"{v:>13.1f}%" if not pd.isna(v) else f"{'N/A':>14}"),
    ('Loss Rate %', 'loss_rate', lambda v: f"{v:>13.1f}%" if not pd.isna(v) else f"{'N/A':>14}"),
    ('Breakeven %', 'be_rate', lambda v: f"{v:>13.1f}%" if not pd.isna(v) else f"{'N/A':>14}"),
    ('W/L Ratio', 'wl_ratio', lambda v: f"{v:>14.2f}" if not pd.isna(v) else f"{'N/A':>14}"),
    ('Trades/Month Mean', 'trades_per_month_mean', lambda v: f"{v:>14.1f}" if not pd.isna(v) else f"{'N/A':>14}"),
    ('Trades/Month Med', 'trades_per_month_median', lambda v: f"{v:>14.1f}" if not pd.isna(v) else f"{'N/A':>14}"),
    ('Trades/Week Mean', 'trades_per_week_mean', lambda v: f"{v:>14.1f}" if not pd.isna(v) else f"{'N/A':>14}"),
    ('Trades/Week Med', 'trades_per_week_median', lambda v: f"{v:>14.1f}" if not pd.isna(v) else f"{'N/A':>14}"),
]

for label, key, fmt in block1_metrics:
    row = f"{label:<30}"
    for seg_name in segments:
        val = all_kpis[seg_name].get(key, np.nan)
        row += f"  {fmt(val)}"
    L(row)

# ---- Block 2 ----
L()
L("=" * 78)
L("  BLOCK 2: PnL-VERTEILUNG ($)")
L("=" * 78)

L(f"{'Metric':<30}" + "".join(f"  {s:>14}" for s in segments))
L("-" * (30 + 16 * len(segments)))

def fmt_d(v):
    if pd.isna(v): return f"{'N/A':>14}"
    sign = "-" if v < 0 else " "
    return f"{sign}${abs(v):>12,.0f}"

def fmt_d2(v):
    if pd.isna(v): return f"{'N/A':>14}"
    sign = "-" if v < 0 else " "
    return f"{sign}${abs(v):>12,.2f}"

block2_metrics = [
    ('Total PnL', 'total_pnl', fmt_d),
    ('Mean PnL/Trade', 'mean_pnl', fmt_d2),
    ('Median PnL/Trade', 'median_pnl', fmt_d2),
    ('StdDev PnL', 'std_pnl', fmt_d),
    ('Skewness', 'skew_pnl', lambda v: f"{v:>14.2f}" if not pd.isna(v) else f"{'N/A':>14}"),
    ('Kurtosis', 'kurt_pnl', lambda v: f"{v:>14.2f}" if not pd.isna(v) else f"{'N/A':>14}"),
]

for label, key, fmt in block2_metrics:
    row = f"{label:<30}"
    for seg_name in segments:
        val = all_kpis[seg_name].get(key, np.nan)
        row += f"  {fmt(val)}"
    L(row)

# Best/Worst trades
L()
for seg_name in segments:
    k = all_kpis[seg_name]
    if 'best_trade' in k:
        L(f"  {seg_name}:")
        L(f"    Best:  {fmt_dollar(k['best_trade'])} ({k.get('best_ticker','?')} {k.get('best_date','?')})")
        L(f"    Worst: {fmt_dollar(k['worst_trade'])} ({k.get('worst_ticker','?')} {k.get('worst_date','?')})")

L()
L("  Percentiles ($):")
L(f"{'Pctl':<10}" + "".join(f"  {s:>14}" for s in segments))
L("-" * (10 + 16 * len(segments)))
for p in [5, 10, 25, 50, 75, 90, 95]:
    row = f"{'P'+str(p):<10}"
    for seg_name in segments:
        val = all_kpis[seg_name].get(f'P{p}_pnl', np.nan)
        row += f"  {fmt_d(val)}"
    L(row)

# ---- Block 3 ----
L()
L("=" * 78)
L("  BLOCK 3: R-MULTIPLE VERTEILUNG")
L("=" * 78)

L(f"{'Metric':<30}" + "".join(f"  {s:>14}" for s in segments))
L("-" * (30 + 16 * len(segments)))

def fmt_rf(v):
    if pd.isna(v): return f"{'N/A':>14}"
    return f"{v:>+13.3f}R"

block3_metrics = [
    ('Mean R', 'mean_r', fmt_rf),
    ('Median R', 'median_r', fmt_rf),
    ('StdDev R', 'std_r', lambda v: f"{v:>13.3f}R" if not pd.isna(v) else f"{'N/A':>14}"),
    ('Skewness R', 'skew_r', lambda v: f"{v:>14.2f}" if not pd.isna(v) else f"{'N/A':>14}"),
    ('WR@1R', 'WR_at_1R', lambda v: f"{v:>13.1f}%" if not pd.isna(v) else f"{'N/A':>14}"),
    ('WR@2R', 'WR_at_2R', lambda v: f"{v:>13.1f}%" if not pd.isna(v) else f"{'N/A':>14}"),
    ('WR@3R', 'WR_at_3R', lambda v: f"{v:>13.1f}%" if not pd.isna(v) else f"{'N/A':>14}"),
    ('WR@5R', 'WR_at_5R', lambda v: f"{v:>13.1f}%" if not pd.isna(v) else f"{'N/A':>14}"),
    ('WR@10R', 'WR_at_10R', lambda v: f"{v:>13.1f}%" if not pd.isna(v) else f"{'N/A':>14}"),
    ('Expectancy', 'expectancy_r', fmt_rf),
]

for label, key, fmt in block3_metrics:
    row = f"{label:<30}"
    for seg_name in segments:
        val = all_kpis[seg_name].get(key, np.nan)
        row += f"  {fmt(val)}"
    L(row)

# ---- Block 4 ----
L()
L("=" * 78)
L("  BLOCK 4: RISIKO-METRIKEN")
L("=" * 78)

L(f"{'Metric':<30}" + "".join(f"  {s:>14}" for s in segments))
L("-" * (30 + 16 * len(segments)))

block4_metrics = [
    ('Profit Factor', 'profit_factor', lambda v: f"{v:>14.2f}" if not pd.isna(v) else f"{'N/A':>14}"),
    ('Payoff Ratio', 'payoff_ratio', lambda v: f"{v:>14.2f}" if not pd.isna(v) else f"{'N/A':>14}"),
    ('Max DD $', 'max_dd_dollar', fmt_d),
    ('Max DD % Capital', 'max_dd_pct', lambda v: f"{v:>13.1f}%" if not pd.isna(v) else f"{'N/A':>14}"),
    ('Max DD Duration', 'max_dd_duration', lambda v: f"{v:>11d} tr" if not pd.isna(v) else f"{'N/A':>14}"),
    ('Max DD Recovery', 'max_dd_recovery', lambda v: f"{v:>11d} tr" if not pd.isna(v) else f"{'N/A':>14}"),
    ('Calmar Ratio', 'calmar', lambda v: f"{v:>14.3f}" if not pd.isna(v) else f"{'N/A':>14}"),
    ('Ulcer Index', 'ulcer_index', lambda v: f"{v:>14.2f}" if not pd.isna(v) else f"{'N/A':>14}"),
]

for label, key, fmt in block4_metrics:
    row = f"{label:<30}"
    for seg_name in segments:
        val = all_kpis[seg_name].get(key, np.nan)
        row += f"  {fmt(val)}"
    L(row)

# ---- Block 5 ----
L()
L("=" * 78)
L("  BLOCK 5: SHARPE & RISIKO-ADJUSTIERT")
L("=" * 78)

L(f"{'Metric':<30}" + "".join(f"  {s:>14}" for s in segments))
L("-" * (30 + 16 * len(segments)))

block5_metrics = [
    ('Sharpe Ratio', 'sharpe', lambda v: f"{v:>14.3f}" if not pd.isna(v) else f"{'N/A':>14}"),
    ('Sortino Ratio', 'sortino', lambda v: f"{v:>14.3f}" if not pd.isna(v) else f"{'N/A':>14}"),
    ('MAR Ratio', 'mar', lambda v: f"{v:>14.3f}" if not pd.isna(v) else f"{'N/A':>14}"),
    ('Risk-Adj Return', 'risk_adj_return', lambda v: f"{v:>14.2f}" if not pd.isna(v) else f"{'N/A':>14}"),
    ('Kelly f*', 'kelly', lambda v: f"{v*100:>13.1f}%" if not pd.isna(v) else f"{'N/A':>14}"),
    ('Half-Kelly', 'half_kelly', lambda v: f"{v*100:>13.1f}%" if not pd.isna(v) else f"{'N/A':>14}"),
    ('Trades/Year', 'trades_per_year', lambda v: f"{v:>14.1f}" if not pd.isna(v) else f"{'N/A':>14}"),
    ('Exp x Freq $/yr', 'exp_x_freq', fmt_d),
]

for label, key, fmt in block5_metrics:
    row = f"{label:<30}"
    for seg_name in segments:
        val = all_kpis[seg_name].get(key, np.nan)
        row += f"  {fmt(val)}"
    L(row)

# ---- Block 6 ----
L()
L("=" * 78)
L("  BLOCK 6: STREAKS & SERIEN")
L("=" * 78)

L(f"{'Metric':<30}" + "".join(f"  {s:>14}" for s in segments))
L("-" * (30 + 16 * len(segments)))

block6_metrics = [
    ('Max Consec Wins', 'max_consec_wins', lambda v: f"{v:>14d}" if not pd.isna(v) else f"{'N/A':>14}"),
    ('Max Consec Losses', 'max_consec_losses', lambda v: f"{v:>14d}" if not pd.isna(v) else f"{'N/A':>14}"),
    ('Max Consec BE', 'max_consec_be', lambda v: f"{v:>14d}" if not pd.isna(v) else f"{'N/A':>14}"),
    ('Mean Win Streak', 'mean_win_streak', lambda v: f"{v:>14.1f}" if not pd.isna(v) else f"{'N/A':>14}"),
    ('Mean Loss Streak', 'mean_loss_streak', lambda v: f"{v:>14.1f}" if not pd.isna(v) else f"{'N/A':>14}"),
    ('Longest Flat (days)', 'longest_flat_days', lambda v: f"{v:>14d}" if not pd.isna(v) else f"{'N/A':>14}"),
    ('Biggest Comeback $', 'biggest_comeback', fmt_d),
]

for label, key, fmt in block6_metrics:
    row = f"{label:<30}"
    for seg_name in segments:
        val = all_kpis[seg_name].get(key, np.nan)
        row += f"  {fmt(val)}"
    L(row)

# ---- Block 7 ----
L()
L("=" * 78)
L("  BLOCK 7: ZEITLICHE ANALYSE")
L("=" * 78)

L(f"{'Metric':<30}" + "".join(f"  {s:>14}" for s in segments))
L("-" * (30 + 16 * len(segments)))

block7_metrics = [
    ('Monthly Mean $', 'monthly_mean', fmt_d),
    ('Monthly Median $', 'monthly_median', fmt_d),
    ('Monthly StdDev $', 'monthly_std', fmt_d),
    ('Best Month $', 'best_month', fmt_d),
    ('Worst Month $', 'worst_month', fmt_d),
    ('% Profit Months', 'pct_profitable_months', lambda v: f"{v:>13.1f}%" if not pd.isna(v) else f"{'N/A':>14}"),
    ('% Profit Quarters', 'pct_profitable_quarters', lambda v: f"{v:>13.1f}%" if not pd.isna(v) else f"{'N/A':>14}"),
    ('Annualized Ret %', 'annualized_return_pct', lambda v: f"{v:>13.1f}%" if not pd.isna(v) else f"{'N/A':>14}"),
    ('CAGR %', 'cagr', lambda v: f"{v:>13.1f}%" if not pd.isna(v) else f"{'N/A':>14}"),
    ('Mean PnL Trade Days', 'mean_pnl_trade_days', fmt_d2),
    ('% Days w/ Trade', 'pct_days_with_trade', lambda v: f"{v:>13.1f}%" if not pd.isna(v) else f"{'N/A':>14}"),
    ('Max Trades/Day', 'max_trades_per_day', lambda v: f"{v:>14d}" if not pd.isna(v) else f"{'N/A':>14}"),
    ('Corr Trades/Day~PnL', 'corr_trades_day_pnl', lambda v: f"{v:>14.3f}" if not pd.isna(v) else f"{'N/A':>14}"),
]

for label, key, fmt in block7_metrics:
    row = f"{label:<30}"
    for seg_name in segments:
        val = all_kpis[seg_name].get(key, np.nan)
        row += f"  {fmt(val)}"
    L(row)

# Best/Worst month details
L()
for seg_name in segments:
    k = all_kpis[seg_name]
    L(f"  {seg_name}: Best Month {k.get('best_month_date','?')} | Worst Month {k.get('worst_month_date','?')}")

# ---- Block 8 ----
L()
L("=" * 78)
L("  BLOCK 8: TRADE-DAUER (Minuten ab 09:35)")
L("=" * 78)

L(f"{'Metric':<30}" + "".join(f"  {s:>14}" for s in segments))
L("-" * (30 + 16 * len(segments)))

block8_metrics = [
    ('Median Duration', 'median_duration', lambda v: f"{v:>11.0f}min" if not pd.isna(v) else f"{'N/A':>14}"),
    ('Mean Duration', 'mean_duration', lambda v: f"{v:>11.0f}min" if not pd.isna(v) else f"{'N/A':>14}"),
    ('P90 Duration', 'p90_duration', lambda v: f"{v:>11.0f}min" if not pd.isna(v) else f"{'N/A':>14}"),
    ('Max Duration', 'max_duration', lambda v: f"{v:>11.0f}min" if not pd.isna(v) else f"{'N/A':>14}"),
    ('Median Dur Winners', 'median_dur_winners', lambda v: f"{v:>11.0f}min" if not pd.isna(v) else f"{'N/A':>14}"),
    ('Median Dur Losers', 'median_dur_losers', lambda v: f"{v:>11.0f}min" if not pd.isna(v) else f"{'N/A':>14}"),
    ('Corr Duration~PnL', 'corr_dur_pnl', lambda v: f"{v:>14.3f}" if not pd.isna(v) else f"{'N/A':>14}"),
]

for label, key, fmt in block8_metrics:
    row = f"{label:<30}"
    for seg_name in segments:
        val = all_kpis[seg_name].get(key, np.nan)
        row += f"  {fmt(val)}"
    L(row)

# ---- Block 9 ----
L()
L("=" * 78)
L("  BLOCK 9: SETUP-VERGLEICH")
L("=" * 78)

L()
L("  GapUp vs GapDn (OOS) — Welch t-test on Mean R:")
c = comparisons['GapUp_vs_GapDn']
L(f"    t = {c['t']:.3f}, p = {c['p']:.4f}")
L(f"    {'*** SIGNIFIKANT ***' if c['p'] < 0.05 else 'Nicht signifikant (p >= 0.05)'}")

L()
L("  Tier1 vs Tier2 (OOS) — Welch t-test on Mean R:")
c = comparisons['Tier1_vs_Tier2']
L(f"    t = {c['t']:.3f}, p = {c['p']:.4f}")
L(f"    {'*** SIGNIFIKANT ***' if c['p'] < 0.05 else 'Nicht signifikant (p >= 0.05)'}")

L()
L("  IS vs OOS Shrinkage:")
L(f"    {'Metric':<25} {'IS':>12} {'OOS':>12} {'Shrinkage':>12}")
L("    " + "-" * 61)
for metric in ['mean_r', 'win_rate', 'profit_factor', 'sharpe', 'mean_pnl']:
    is_val = is_kpis.get(metric, np.nan)
    oos_val = oos_kpis.get(metric, np.nan)
    shr = shrinkage.get(metric, np.nan)
    is_str = f"{is_val:>12.3f}" if not pd.isna(is_val) else f"{'N/A':>12}"
    oos_str = f"{oos_val:>12.3f}" if not pd.isna(oos_val) else f"{'N/A':>12}"
    shr_str = f"{shr:>+11.1f}%" if not pd.isna(shr) else f"{'N/A':>12}"
    L(f"    {metric:<25} {is_str} {oos_str} {shr_str}")

# ---- Block 10 ----
L()
L("=" * 78)
L("  BLOCK 10: MONTE CARLO (Tier2 GapUp OOS, 1000 Simulationen)")
L("=" * 78)

L()
L(f"  Finale Equity (alle Pfade):       {fmt_dollar(mc_results['final_equity']):>12}  (Summe ist kommutativ)")
L()
L(f"  {'Min Equity auf Pfad':<35} {'Median':>12} {'[P5':>12} {'P25':>8} {'P75':>8} {'P95]':>8}")
L(f"  {'':<35} {fmt_dollar(mc_results['median_min_equity']):>12} "
  f"{fmt_dollar(mc_results['p5_min_equity']):>12} "
  f"{fmt_dollar(mc_results['p25_min_equity']):>8} "
  f"{fmt_dollar(mc_results['p75_min_equity']):>8} "
  f"{fmt_dollar(mc_results['p95_min_equity']):>8}")

L()
L(f"  {'Max Drawdown':<35} {'Median':>12} {'[P5':>12} {'P25':>8} {'P75':>8} {'P95]':>8}")
L(f"  {'':<35} {fmt_dollar(mc_results['median_max_dd']):>12} "
  f"{fmt_dollar(mc_results['p5_max_dd']):>12} "
  f"{fmt_dollar(mc_results['p25_max_dd']):>8} "
  f"{fmt_dollar(mc_results['p75_max_dd']):>8} "
  f"{fmt_dollar(mc_results['p95_max_dd']):>8}")

L(f"  Worst-Case Max DD:                {fmt_dollar(mc_results['worst_max_dd']):>12}")

L()
L(f"  P(DD > $10K):                     {mc_results['p_dd_10k']:>11.1f}%")
L(f"  P(DD > $15K):                     {mc_results['p_dd_15k']:>11.1f}%")
L(f"  P(DD > $20K):                     {mc_results['p_dd_20k']:>11.1f}%")

L()
L(f"  Max Consec Losses:   Median = {mc_results['median_max_cl']:.0f}  |  P95 = {mc_results['p95_max_cl']:.0f}")
L(f"  Sharpe 95% CI:       [{mc_results['sharpe_ci_lo']:.3f}, {mc_results['sharpe_ci_hi']:.3f}]")

L()
L("=" * 78)
L("  END OF REPORT")
L("=" * 78)

report = "\n".join(lines)
print(report)

# Save report
with open(OUT_DIR / "kpi_report.txt", "w", encoding="utf-8") as f:
    f.write(report)
print(f"\nReport saved to {OUT_DIR / 'kpi_report.txt'}")


# ============================================================
# CHARTS
# ============================================================
print("\n" + "=" * 70)
print("GENERATING CHARTS")
print("=" * 70)

plt.rcParams.update({
    'figure.facecolor': DARK_BG,
    'axes.facecolor': DARK_BG,
    'axes.edgecolor': GREY,
    'axes.labelcolor': 'white',
    'text.color': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'grid.color': '#333355',
    'grid.alpha': 0.5,
    'font.size': 10,
})

# --- CHART 1: EQUITY CURVE ---
fig, ax = plt.subplots(figsize=(14, 7))

oos_all = ctrl[ctrl['period'] == 'OOS'].sort_values('date').reset_index(drop=True)
oos_up = oos_all[oos_all['gap_direction'] == 'up']
oos_t1_up = oos_all[(oos_all['gap_direction'] == 'up') & (oos_all['is_tier1'] == True)]

eq_all = START_CAPITAL + oos_all['pnl_dollar'].cumsum()
eq_up = START_CAPITAL + oos_up['pnl_dollar'].cumsum()
eq_t1 = START_CAPITAL + oos_t1_up['pnl_dollar'].cumsum()

# Drawdown shading for all
dd_all = eq_all - eq_all.cummax()
ax.fill_between(range(len(dd_all)), eq_all.values, eq_all.cummax().values,
                alpha=0.3, color=RED, label='Drawdown')

ax.plot(range(len(eq_all)), eq_all.values, color=CYAN, linewidth=1.5, label=f'GESAMT OOS (N={len(oos_all)})')
ax.plot(range(len(eq_up)), eq_up.values, color=GREEN, linewidth=1.2, label=f'GapUp (N={len(oos_up)})', alpha=0.8)
ax.plot(range(len(eq_t1)), eq_t1.values, color=YELLOW, linewidth=1.0, label=f'Tier1 GapUp (N={len(oos_t1_up)})', alpha=0.7)

ax.axhline(START_CAPITAL, color=GREY, linestyle='--', alpha=0.5)
ax.set_xlabel('Trade #')
ax.set_ylabel('Equity ($)')
ax.set_title('CTRL Strategy — Equity Curve (OOS)', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', framealpha=0.7)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / 'chart1_equity_curve.png', dpi=300, facecolor=DARK_BG)
plt.close()
print("  Chart 1: Equity Curve saved")

# --- CHART 2: MONTHLY RETURNS HEATMAP ---
fig, ax = plt.subplots(figsize=(14, 5))

oos_all_sorted = ctrl[ctrl['period'] == 'OOS'].copy()
oos_all_sorted['ym'] = oos_all_sorted['date'].dt.to_period('M')
monthly = oos_all_sorted.groupby('ym')['pnl_dollar'].sum()

years = sorted(oos_all_sorted['year'].unique())
months = range(1, 13)
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

data_matrix = np.full((len(years), 12), np.nan)
for ym, val in monthly.items():
    y_idx = years.index(ym.year)
    m_idx = ym.month - 1
    data_matrix[y_idx, m_idx] = val

# Custom colormap
max_abs = np.nanmax(np.abs(data_matrix[~np.isnan(data_matrix)])) if not np.all(np.isnan(data_matrix)) else 1
cmap = mcolors.LinearSegmentedColormap.from_list('rg', [RED, DARK_BG, GREEN])
im = ax.imshow(data_matrix, cmap=cmap, aspect='auto', vmin=-max_abs, vmax=max_abs)

ax.set_xticks(range(12))
ax.set_xticklabels(month_names)
ax.set_yticks(range(len(years)))
ax.set_yticklabels(years)

# Annotate cells
for i in range(len(years)):
    for j in range(12):
        val = data_matrix[i, j]
        if not np.isnan(val):
            color = 'white' if abs(val) > max_abs * 0.5 else 'white'
            ax.text(j, i, f"${val:,.0f}", ha='center', va='center', fontsize=8,
                    color=color, fontweight='bold')

ax.set_title('CTRL Strategy — Monthly Returns Heatmap (OOS)', fontsize=14, fontweight='bold')
fig.colorbar(im, ax=ax, label='PnL ($)', shrink=0.8)
fig.tight_layout()
fig.savefig(OUT_DIR / 'chart2_monthly_heatmap.png', dpi=300, facecolor=DARK_BG)
plt.close()
print("  Chart 2: Monthly Heatmap saved")

# --- CHART 3: R-DISTRIBUTION ---
fig, ax = plt.subplots(figsize=(12, 6))

r_data = segments['GESAMT_OOS']['pnl_r'].values
bins_edges = [-np.inf, -0.999, 0.001, 1.0, 2.0, 3.0, 5.0, 10.0, np.inf]
bin_labels = ['=-1R\n(SL Hit)', '(-1,0)R', '[0,1)R', '[1,2)R', '[2,3)R', '[3,5)R', '[5,10)R', '>=10R']
counts = np.histogram(r_data, bins=bins_edges)[0]
pcts = counts / len(r_data) * 100

colors = [RED if i < 2 else (GREY if i == 2 else CYAN) for i in range(len(counts))]
bars = ax.bar(range(len(counts)), counts, color=colors, edgecolor='white', linewidth=0.5)

for i, (c, p) in enumerate(zip(counts, pcts)):
    ax.text(i, c + max(counts) * 0.02, f"N={c}\n{p:.1f}%", ha='center', va='bottom',
            fontsize=9, color='white')

ax.set_xticks(range(len(bin_labels)))
ax.set_xticklabels(bin_labels, fontsize=9)
ax.set_ylabel('Count')
ax.set_title('CTRL Strategy — R-Multiple Distribution (OOS GESAMT)', fontsize=14, fontweight='bold')
ax.grid(True, axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / 'chart3_r_distribution.png', dpi=300, facecolor=DARK_BG)
plt.close()
print("  Chart 3: R-Distribution saved")

# --- CHART 4: DRAWDOWN OVER TIME ---
fig, ax = plt.subplots(figsize=(14, 6))

oos_sorted = ctrl[ctrl['period'] == 'OOS'].sort_values('date').reset_index(drop=True)
eq = START_CAPITAL + oos_sorted['pnl_dollar'].cumsum()
dd = compute_drawdown(eq)

ax.fill_between(range(len(dd)), dd.values, 0, color=RED, alpha=0.5)
ax.plot(range(len(dd)), dd.values, color=RED, linewidth=0.8)

# Max DD line
ax.axhline(dd.min(), color=YELLOW, linestyle='--', linewidth=1.5,
           label=f'Max DD: {fmt_dollar(dd.min())}')

# MC P95 line
ax.axhline(mc_results['p95_max_dd'], color=CYAN, linestyle=':', linewidth=1.5,
           label=f'MC P95 DD: {fmt_dollar(mc_results["p95_max_dd"])}')

ax.set_xlabel('Trade #')
ax.set_ylabel('Drawdown ($)')
ax.set_title('CTRL Strategy — Drawdown Over Time (OOS)', fontsize=14, fontweight='bold')
ax.legend(loc='lower left', framealpha=0.7)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / 'chart4_drawdown.png', dpi=300, facecolor=DARK_BG)
plt.close()
print("  Chart 4: Drawdown saved")

# --- CHART 5: PnL BY WEEKDAY ---
fig, ax = plt.subplots(figsize=(10, 6))

oos_ctrl = ctrl[ctrl['period'] == 'OOS']
weekday_pnl = oos_ctrl.groupby('weekday')['pnl_dollar']
means = weekday_pnl.mean()
stds = weekday_pnl.std()

day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
colors_wd = [GREEN if m > 0 else RED for m in means.values]

bars = ax.bar(range(5), means.values, color=colors_wd, edgecolor='white', linewidth=0.5,
              yerr=stds.values, capsize=5, error_kw={'color': GREY, 'linewidth': 1.5})

for i, (m, s) in enumerate(zip(means.values, stds.values)):
    ax.text(i, m + s + 20, f"${m:,.0f}", ha='center', va='bottom', fontsize=10, color='white')

ax.set_xticks(range(5))
ax.set_xticklabels(day_names)
ax.axhline(0, color=GREY, linewidth=0.5)
ax.set_ylabel('Mean PnL ($)')
ax.set_title('CTRL Strategy — PnL by Weekday (OOS)', fontsize=14, fontweight='bold')
ax.grid(True, axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / 'chart5_weekday_pnl.png', dpi=300, facecolor=DARK_BG)
plt.close()
print("  Chart 5: Weekday PnL saved")

# --- CHART 6: ROLLING SHARPE ---
fig, ax = plt.subplots(figsize=(14, 6))

oos_sorted = ctrl[ctrl['period'] == 'OOS'].sort_values('date').reset_index(drop=True)
pnl_series = oos_sorted['pnl_dollar']
ret_series = pnl_series / START_CAPITAL
window = 50

rolling_mean = ret_series.rolling(window).mean()
rolling_std = ret_series.rolling(window).std()
rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)

ax.plot(range(len(rolling_sharpe)), rolling_sharpe.values, color=CYAN, linewidth=1.5, label=f'{window}-Trade Rolling Sharpe')

overall_sharpe = all_kpis['GESAMT_OOS'].get('sharpe', 0)
ax.axhline(overall_sharpe, color=YELLOW, linestyle='--', linewidth=1.5,
           label=f'Overall Sharpe: {overall_sharpe:.3f}')
ax.axhline(0, color=GREY, linewidth=0.5)

ax.set_xlabel('Trade #')
ax.set_ylabel('Sharpe Ratio (annualized)')
ax.set_title('CTRL Strategy — 50-Trade Rolling Sharpe (OOS)', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', framealpha=0.7)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / 'chart6_rolling_sharpe.png', dpi=300, facecolor=DARK_BG)
plt.close()
print("  Chart 6: Rolling Sharpe saved")

# --- CHART 7: WIN/LOSS STREAKS ---
fig, ax = plt.subplots(figsize=(14, 5))

oos_sorted = ctrl[ctrl['period'] == 'OOS'].sort_values('date').reset_index(drop=True)
pnl_arr = oos_sorted['pnl_dollar'].values
signs = np.sign(pnl_arr)
signs[np.abs(pnl_arr) < 1] = 0

# Build streak series
streaks = []
current_sign = signs[0]
current_len = 1
for i in range(1, len(signs)):
    if signs[i] == current_sign:
        current_len += 1
    else:
        streaks.append((current_sign, current_len))
        current_sign = signs[i]
        current_len = 1
streaks.append((current_sign, current_len))

# Plot as bars
x = 0
for sign, length in streaks:
    color = GREEN if sign > 0 else (RED if sign < 0 else GREY)
    height = length if sign >= 0 else -length
    ax.bar(x, height, color=color, width=1.0, edgecolor='none')
    x += 1

ax.axhline(0, color=GREY, linewidth=0.5)
ax.set_xlabel('Streak #')
ax.set_ylabel('Streak Length (pos=wins, neg=losses)')
ax.set_title('CTRL Strategy — Win/Loss Streaks (OOS)', fontsize=14, fontweight='bold')
ax.grid(True, axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / 'chart7_streaks.png', dpi=300, facecolor=DARK_BG)
plt.close()
print("  Chart 7: Streaks saved")

# --- CHART 8: MONTE CARLO EQUITY FAN ---
fig, ax = plt.subplots(figsize=(14, 7))

# Plot 100 random paths
for path in mc_equity_paths[:100]:
    ax.plot(range(len(path)), path, color=CYAN, alpha=0.05, linewidth=0.5)

# Median and percentile paths
n_trades = len(mc_pnl)
all_paths_arr = np.array([START_CAPITAL + np.cumsum(np.random.permutation(mc_pnl))
                          for _ in range(MC_SIMS)])
np.random.seed(MC_SEED)  # Reset for reproducibility

median_path = np.median(np.array([START_CAPITAL + np.cumsum(np.random.permutation(mc_pnl))
                                   for _ in range(MC_SIMS)]), axis=0)
np.random.seed(MC_SEED)
all_mc = np.array([START_CAPITAL + np.cumsum(np.random.permutation(mc_pnl))
                    for _ in range(MC_SIMS)])
p5_path = np.percentile(all_mc, 5, axis=0)
p95_path = np.percentile(all_mc, 95, axis=0)
median_path = np.median(all_mc, axis=0)

ax.plot(range(n_trades), median_path, color=YELLOW, linewidth=2.5, label='Median')
ax.plot(range(n_trades), p5_path, color=RED, linewidth=1.5, linestyle='--', label='P5')
ax.plot(range(n_trades), p95_path, color=GREEN, linewidth=1.5, linestyle='--', label='P95')

# Actual path
actual_eq = START_CAPITAL + mc_seg.sort_values('date')['pnl_dollar'].cumsum().values
ax.plot(range(len(actual_eq)), actual_eq, color='white', linewidth=1.5, alpha=0.8, label='Actual')

ax.axhline(START_CAPITAL, color=GREY, linestyle=':', alpha=0.5)
ax.set_xlabel('Trade #')
ax.set_ylabel('Equity ($)')
ax.set_title('Monte Carlo Equity Fan — Tier2 GapUp OOS (1000 Sims)', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', framealpha=0.7)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / 'chart8_monte_carlo.png', dpi=300, facecolor=DARK_BG)
plt.close()
print("  Chart 8: Monte Carlo Fan saved")

print("\n" + "=" * 70)
print(f"ALL DONE. Output in: {OUT_DIR}/")
print("=" * 70)
