"""
D15 Quant Review Analysis — Comprehensive statistical checks.
Outputs all results to stdout for capture.
"""
import pandas as pd
import numpy as np
import warnings
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_parquet('d14_bias_free/results/d14_all_trades.parquet')
meta = pd.read_parquet('data/metadata/metadata_v9.parquet')
meta['date'] = meta['date'].astype(str)

STRATS = [
    '1.2_Pre10am_Momentum', '2.3_Dip_and_Rip', '1.4_Power_Setup',
    '1.3_Earnings_OD_RVOL', '1.8_Gap_3ADR', '1.5_Big_First_Candle',
    '1.1_S4_Optimiert', '1.9_RVOL5_RSI50', '1.6_Low_Wick', '1.7_Big_Body',
]

SL_MULT = {
    '1.1_S4_Optimiert': 0.25, '1.2_Pre10am_Momentum': 0.15,
    '1.3_Earnings_OD_RVOL': 0.25, '1.4_Power_Setup': 0.25,
    '1.5_Big_First_Candle': 0.25, '1.6_Low_Wick': 0.25,
    '1.7_Big_Body': 0.25, '1.8_Gap_3ADR': 0.25, '1.9_RVOL5_RSI50': 0.25,
}

df_all = df.copy()
df = df[df['strategy'].isin(STRATS)].copy()
oos = df[df['sample'] == 'OOS'].copy()
oos['date_dt'] = pd.to_datetime(oos['date'])

meta_adr = meta[['ticker', 'date', 'adr_10', 'prev_close']].drop_duplicates(subset=['ticker', 'date'])
oos2 = oos.merge(meta_adr, on=['ticker', 'date'], how='left')

calendar = pd.bdate_range(start='2024-01-01', end='2026-02-21')
n_cal = len(calendar)
OOS_DAYS = (pd.Timestamp('2026-02-21') - pd.Timestamp('2024-01-01')).days
OOS_YEARS = OOS_DAYS / 365.25

# ============================================================
# 1. DAILY SHARPE vs PER-TRADE SHARPE (RAW)
# ============================================================
print('=' * 80)
print('SECTION 1: DAILY SHARPE vs PER-TRADE SHARPE (RAW, NO COSTS)')
print('=' * 80)
print(f'Trading days in OOS period: {n_cal}')
print()
print(f'{"Strategy":<28} | {"N":>5} | {"TrDays":>6} | {"%Act":>5} | {"D15_Sha":>8} | {"DailySha":>8} | {"Ratio":>6}')
print('-' * 80)

for strat in STRATS:
    sub = oos[oos['strategy'] == strat]
    n = len(sub)
    if n < 5:
        continue
    ev = sub['pnl_r'].mean()
    std_t = sub['pnl_r'].std(ddof=1)
    per_trade_sha = ev / std_t if std_t > 0 else 0
    tpy = n / (OOS_DAYS / 365.25)
    d15_sha = per_trade_sha * np.sqrt(tpy)

    daily = sub.groupby('date_dt')['pnl_r'].sum().reindex(calendar, fill_value=0.0)
    sha_daily = daily.mean() / daily.std(ddof=1) * np.sqrt(252) if daily.std() > 0 else 0
    n_trade_days = (daily != 0).sum()
    pct_active = n_trade_days / n_cal * 100
    ratio = sha_daily / d15_sha if d15_sha != 0 else 0

    print(f'{strat:<28} | {n:>5} | {n_trade_days:>6} | {pct_active:>4.1f}% | {d15_sha:>8.2f} | {sha_daily:>8.2f} | {ratio:>6.2f}')

# ============================================================
# 2. DAILY SHARPE WITH BASE COSTS (0.05 ADR)
# ============================================================
print()
print('=' * 80)
print('SECTION 2: DAILY SHARPE -- NET OF BASE COSTS (0.05 ADR)')
print('=' * 80)

# Compute costs
def get_sl_dist(row):
    strat = row['strategy']
    if strat in SL_MULT:
        if pd.notna(row['adr_10']) and row['adr_10'] > 0:
            return SL_MULT[strat] * row['adr_10']
    elif strat == '2.3_Dip_and_Rip':
        if pd.notna(row['adr_10']) and row['adr_10'] > 0:
            return 0.225 * row['adr_10']
    return np.nan

oos2['sl_dist'] = oos2.apply(get_sl_dist, axis=1)
oos2_valid = oos2[oos2['sl_dist'].notna() & (oos2['sl_dist'] > 0)].copy()
oos2_valid['fee_r'] = 0.0025 / oos2_valid['sl_dist']
oos2_valid['slip_r_base'] = 0.05 * oos2_valid['adr_10'] / oos2_valid['sl_dist']
oos2_valid['cost_r_base'] = oos2_valid['fee_r'] + oos2_valid['slip_r_base']
oos2_valid['pnl_r_base'] = oos2_valid['pnl_r'] - oos2_valid['cost_r_base']

print(f'{"Strategy":<28} | {"N":>5} | {"EV_raw":>7} | {"EV_net":>7} | {"D15_Sha":>8} | {"DailyRaw":>8} | {"DailyNet":>8} | {"Net/D15":>7}')
print('-' * 100)

sharpe_results = []
for strat in STRATS:
    sub_raw = oos[oos['strategy'] == strat]
    sub_net = oos2_valid[oos2_valid['strategy'] == strat]
    n = len(sub_raw)
    if n < 5:
        continue

    ev_raw = sub_raw['pnl_r'].mean()
    ev_net = sub_net['pnl_r_base'].mean() if len(sub_net) > 0 else np.nan
    std_t = sub_raw['pnl_r'].std(ddof=1)
    per_trade_sha = ev_raw / std_t if std_t > 0 else 0
    tpy = n / (OOS_DAYS / 365.25)
    d15_sha = per_trade_sha * np.sqrt(tpy)

    daily_raw = sub_raw.groupby('date_dt')['pnl_r'].sum().reindex(calendar, fill_value=0.0)
    sha_daily_raw = daily_raw.mean() / daily_raw.std(ddof=1) * np.sqrt(252) if daily_raw.std() > 0 else 0

    daily_net = sub_net.groupby('date_dt')['pnl_r_base'].sum().reindex(calendar, fill_value=0.0)
    sha_daily_net = daily_net.mean() / daily_net.std(ddof=1) * np.sqrt(252) if daily_net.std() > 0 else 0

    ratio = sha_daily_net / d15_sha if d15_sha != 0 else 0
    n_trade_days = (daily_raw != 0).sum()

    print(f'{strat:<28} | {n:>5} | {ev_raw:>+7.3f} | {ev_net:>+7.3f} | {d15_sha:>8.2f} | {sha_daily_raw:>8.2f} | {sha_daily_net:>8.2f} | {ratio:>7.2f}')

    sharpe_results.append({
        'strategy': strat, 'n': n, 'n_trade_days': int(n_trade_days),
        'ev_raw': ev_raw, 'ev_net': ev_net,
        'd15_sharpe': d15_sha, 'daily_raw': sha_daily_raw, 'daily_net': sha_daily_net,
        'ratio': ratio,
    })

# ============================================================
# 3. AUTOCORRELATION
# ============================================================
print()
print('=' * 80)
print('SECTION 3: RETURN AUTOCORRELATION (DAILY PNL, LAG 1-5)')
print('=' * 80)

from scipy import stats

print(f'{"Strategy":<28} | {"AC(1)":>7} | {"AC(2)":>7} | {"AC(3)":>7} | {"AC(5)":>7} | {"LB(10)p":>8} | {"IID?":>5}')
print('-' * 85)

for strat in STRATS:
    sub = oos[oos['strategy'] == strat]
    daily = sub.groupby('date_dt')['pnl_r'].sum().reindex(calendar, fill_value=0.0)

    ac = [daily.autocorr(lag=i) for i in range(1, 6)]
    n = len(daily)
    lb_stat = n * (n + 2) * sum([(daily.autocorr(lag=k)**2) / (n - k) for k in range(1, 11)])
    lb_p = 1 - stats.chi2.cdf(lb_stat, df=10)
    iid = 'YES' if lb_p > 0.05 else 'NO'

    print(f'{strat:<28} | {ac[0]:>+7.3f} | {ac[1]:>+7.3f} | {ac[2]:>+7.3f} | {ac[4]:>+7.3f} | {lb_p:>8.4f} | {iid:>5}')

# ============================================================
# 4. TRADE CLUSTERING
# ============================================================
print()
print('=' * 80)
print('SECTION 4: TRADE CLUSTERING (MULTIPLE TRADES PER DAY)')
print('=' * 80)

print(f'{"Strategy":<28} | {"N":>5} | {"TrDays":>6} | {"Multi":>5} | {"%Multi":>6} | {"Max/Day":>7} | {"Avg/TrD":>7}')
print('-' * 80)

for strat in STRATS:
    sub = oos[oos['strategy'] == strat]
    daily_count = sub.groupby('date_dt').size()
    n_trade_days = len(daily_count)
    multi = (daily_count > 1).sum()
    pct_multi = multi / n_trade_days * 100 if n_trade_days > 0 else 0
    max_day = daily_count.max()
    avg_per = daily_count.mean()

    print(f'{strat:<28} | {len(sub):>5} | {n_trade_days:>6} | {multi:>5} | {pct_multi:>5.1f}% | {max_day:>7} | {avg_per:>7.2f}')

# ============================================================
# 5. DAILY PNL CORRELATION MATRIX
# ============================================================
print()
print('=' * 80)
print('SECTION 5: DAILY PNL CORRELATION MATRIX (10 STRATEGIES, OOS)')
print('=' * 80)

daily_pnl_matrix = pd.DataFrame(index=calendar)
for strat in STRATS:
    sub = oos[oos['strategy'] == strat]
    daily = sub.groupby('date_dt')['pnl_r'].sum()
    daily_pnl_matrix[strat] = daily.reindex(calendar, fill_value=0.0)

corr = daily_pnl_matrix.corr()
short_names = [s[:10] for s in STRATS]

print(f'{"":>12} | {" | ".join([f"{s:>10}" for s in short_names])}')
print('-' * (13 + 13 * len(STRATS)))
for i, s1 in enumerate(STRATS):
    row_vals = []
    for j, s2 in enumerate(STRATS):
        c = corr.loc[s1, s2]
        row_vals.append(f'{c:>10.3f}')
    print(f'{short_names[i]:>12} | {" | ".join(row_vals)}')

mask = np.ones(corr.shape, dtype=bool)
np.fill_diagonal(mask, False)
off_diag = corr.values[mask]
print(f'\nOff-diagonal: mean={np.mean(off_diag):.3f}, std={np.std(off_diag):.3f}, '
      f'min={np.min(off_diag):.3f}, max={np.max(off_diag):.3f}')

# ============================================================
# 6. TICKER+DATE OVERLAP (Jaccard)
# ============================================================
print()
print('=' * 80)
print('SECTION 6: TICKER+DATE OVERLAP (JACCARD SIMILARITY)')
print('=' * 80)

sets = {}
for strat in STRATS:
    sub = oos[oos['strategy'] == strat]
    sets[strat] = set(zip(sub['ticker'], sub['date']))

short6 = [s[:12] for s in STRATS]
print(f'{"":>14} | {" | ".join([f"{s:>12}" for s in short6])}')
print('-' * (15 + 15 * len(STRATS)))
for i, s1 in enumerate(STRATS):
    row = []
    for j, s2 in enumerate(STRATS):
        if i == j:
            row.append(f'{"---":>12}')
        else:
            inter = len(sets[s1] & sets[s2])
            union = len(sets[s1] | sets[s2])
            jacc = inter / union if union > 0 else 0
            row.append(f'{inter:>4}/{jacc:.2f}'.rjust(12))
    print(f'{short6[i]:>14} | {" | ".join(row)}')

# ============================================================
# 7. REGIME ANALYSIS (yearly EV)
# ============================================================
print()
print('=' * 80)
print('SECTION 7: REGIME ANALYSIS -- YEARLY EV (OOS)')
print('=' * 80)

oos['year'] = oos['date_dt'].dt.year
print(f'{"Strategy":<28} | {"2024 N":>6} {"2024 EV":>8} | {"2025 N":>6} {"2025 EV":>8} | {"2026 N":>6} {"2026 EV":>8} | {"EV_Std":>6}')
print('-' * 100)

for strat in STRATS:
    sub = oos[oos['strategy'] == strat]
    vals = []
    parts = []
    for yr in [2024, 2025, 2026]:
        yr_sub = sub[sub['year'] == yr]
        n = len(yr_sub)
        ev = yr_sub['pnl_r'].mean() if n > 0 else np.nan
        vals.append(ev)
        parts.append(f'{n:>6} {ev:>+8.3f}' if n > 0 else f'{n:>6} {"N/A":>8}')

    valid_evs = [v for v in vals if not np.isnan(v)]
    std_ev = np.std(valid_evs) if len(valid_evs) > 1 else np.nan
    std_str = f'{std_ev:.3f}' if not np.isnan(std_ev) else 'N/A'

    print(f'{strat:<28} | {parts[0]} | {parts[1]} | {parts[2]} | {std_str:>6}')

# ============================================================
# 8. IS vs OOS COMPARISON
# ============================================================
print()
print('=' * 80)
print('SECTION 8: IS vs OOS COMPARISON')
print('=' * 80)

is_data = df[df['sample'] == 'IS']
oos_data = df[df['sample'] == 'OOS']

print(f'{"Strategy":<28} | {"IS_N":>5} {"IS_EV":>8} {"IS_WR":>6} | {"OOS_N":>5} {"OOS_EV":>8} {"OOS_WR":>6} | {"Decay":>6}')
print('-' * 95)

for strat in STRATS:
    is_sub = is_data[is_data['strategy'] == strat]
    oos_sub = oos_data[oos_data['strategy'] == strat]

    is_n = len(is_sub)
    oos_n = len(oos_sub)
    is_ev = is_sub['pnl_r'].mean() if is_n > 0 else np.nan
    oos_ev = oos_sub['pnl_r'].mean() if oos_n > 0 else np.nan
    is_wr = is_sub['winner'].mean() * 100 if is_n > 0 else np.nan
    oos_wr = oos_sub['winner'].mean() * 100 if oos_n > 0 else np.nan

    if not np.isnan(is_ev) and is_ev > 0 and not np.isnan(oos_ev):
        if oos_ev < is_ev:
            decay = f'{(1 - oos_ev/is_ev)*100:.0f}%'
        else:
            decay = f'+{(oos_ev/is_ev - 1)*100:.0f}%'
    else:
        decay = 'N/A'

    print(f'{strat:<28} | {is_n:>5} {is_ev:>+8.3f} {is_wr:>5.1f}% | {oos_n:>5} {oos_ev:>+8.3f} {oos_wr:>5.1f}% | {decay:>6}')

# ============================================================
# 9. WIN RATE + KELLY
# ============================================================
print()
print('=' * 80)
print('SECTION 9: WIN RATE PLAUSIBILITY + KELLY')
print('=' * 80)

print(f'{"Strategy":<28} | {"N":>5} | {"WR":>6} | {"AvgW":>7} | {"AvgL":>7} | {"PF":>6} | {"Kelly%":>7}')
print('-' * 80)

for strat in STRATS:
    sub = oos[oos['strategy'] == strat]
    n = len(sub)
    wr = sub['winner'].mean() * 100
    wins = sub[sub['pnl_r'] > 0]['pnl_r']
    losses = sub[sub['pnl_r'] <= 0]['pnl_r']
    avg_w = wins.mean() if len(wins) > 0 else 0
    avg_l = losses.mean() if len(losses) > 0 else 0
    pf = abs(wins.sum() / losses.sum()) if len(losses) > 0 and losses.sum() != 0 else float('inf')
    w = wr / 100
    b = abs(avg_w / avg_l) if avg_l != 0 else 0
    kelly = (w * b - (1 - w)) / b if b > 0 else 0

    print(f'{strat:<28} | {n:>5} | {wr:>5.1f}% | {avg_w:>+7.3f} | {avg_l:>+7.3f} | {pf:>6.2f} | {kelly*100:>6.1f}%')

# ============================================================
# 10. MAX DRAWDOWN PLAUSIBILITY
# ============================================================
print()
print('=' * 80)
print('SECTION 10: MAX DRAWDOWN PLAUSIBILITY')
print('=' * 80)

print(f'{"Strategy":<28} | {"N":>5} | {"EV":>7} | {"Std":>7} | {"MaxDD":>7} | {"DD/Sqrt(N)":>10}')
print('-' * 80)

for strat in STRATS:
    sub = oos[oos['strategy'] == strat].sort_values('date')
    n = len(sub)
    ev = sub['pnl_r'].mean()
    std = sub['pnl_r'].std(ddof=1)
    cum = sub['pnl_r'].cumsum()
    maxdd = (cum - cum.cummax()).min()
    dd_sqrtn = maxdd / np.sqrt(n)

    print(f'{strat:<28} | {n:>5} | {ev:>+7.3f} | {std:>7.3f} | {maxdd:>7.1f} | {dd_sqrtn:>10.3f}')

# ============================================================
# 11. CAGR PLAUSIBILITY
# ============================================================
print()
print('=' * 80)
print('SECTION 11: CAGR PLAUSIBILITY (100R start)')
print('=' * 80)

print(f'{"Strategy":<28} | {"N":>5} | {"EV":>7} | {"TotalR":>7} | {"Final":>7} | {"CAGR":>6} | {"TotalR/MaxDD":>12}')
print('-' * 90)

for strat in STRATS:
    sub = oos[oos['strategy'] == strat].sort_values('date')
    n = len(sub)
    ev = sub['pnl_r'].mean()
    total_r = sub['pnl_r'].sum()
    final = 100 + total_r
    cagr = (final / 100) ** (1 / OOS_YEARS) - 1 if final > 0 else -1
    cum = sub['pnl_r'].cumsum()
    maxdd = (cum - cum.cummax()).min()
    ratio = total_r / abs(maxdd) if maxdd != 0 else float('inf')

    print(f'{strat:<28} | {n:>5} | {ev:>+7.3f} | {total_r:>+7.1f} | {final:>7.1f} | {cagr*100:>5.1f}% | {ratio:>12.2f}')

# ============================================================
# 12. COST MODEL ASSESSMENT
# ============================================================
print()
print('=' * 80)
print('SECTION 12: COST MODEL ASSESSMENT')
print('=' * 80)

print('ADR statistics (OOS):')
print(f'  Mean ADR: ${oos2["adr_10"].mean():.2f}')
print(f'  Median ADR: ${oos2["adr_10"].median():.2f}')
print(f'  Mean prev_close: ${oos2["prev_close"].mean():.2f}')
adr_pct = (oos2["adr_10"] / oos2["prev_close"]).mean() * 100
print(f'  ADR as % of price: {adr_pct:.2f}%')
print()

mean_adr = oos2['adr_10'].mean()
mean_price = oos2['prev_close'].mean()
print('Slippage reality check:')
print(f'  Base slippage = 0.05 ADR = ${mean_adr * 0.05:.3f}/share on avg ${mean_price:.0f} stock')
print(f'  = {0.05 * mean_adr / mean_price * 100:.3f}% of share price (~20 bps)')
print(f'  Typical market order slippage for liquid stocks: 5-20 bps')
print(f'  Base scenario is CONSERVATIVE for large-caps, REALISTIC for mid/small-caps')
print()

print('Cost breakdown per strategy:')
for strat in STRATS:
    sub = oos2_valid[oos2_valid['strategy'] == strat]
    if len(sub) == 0:
        continue
    ev_raw = sub['pnl_r'].mean()
    total_cost = sub['cost_r_base'].mean()
    fee_mean = sub['fee_r'].mean()
    slip_mean = sub['slip_r_base'].mean()

    print(f'  {strat:<28}: fee={fee_mean:.4f}R, slip={slip_mean:.3f}R, '
          f'total={total_cost:.3f}R ({total_cost/ev_raw*100:.1f}% of EV)')

# ============================================================
# 13. SURVIVORSHIP BIAS CHECK
# ============================================================
print()
print('=' * 80)
print('SECTION 13: SURVIVORSHIP BIAS CHECK')
print('=' * 80)

is_tickers = set(df[df['sample'] == 'IS']['ticker'].unique())
oos_tickers = set(df[df['sample'] == 'OOS']['ticker'].unique())
print(f'IS tickers: {len(is_tickers)}')
print(f'OOS tickers: {len(oos_tickers)}')
print(f'Only in IS (potential delistings): {len(is_tickers - oos_tickers)}')
print(f'Only in OOS (new listings): {len(oos_tickers - is_tickers)}')
print(f'In both: {len(is_tickers & oos_tickers)}')
print()

is_only = sorted(is_tickers - oos_tickers)
print(f'IS-only tickers (first 20): {is_only[:20]}')
all_meta_tickers = set(meta['ticker'].unique())
print(f'Total tickers in metadata: {len(all_meta_tickers)}')
is_only_trades = df[(df['sample'] == 'IS') & (df['ticker'].isin(is_tickers - oos_tickers))]
is_total = len(df[df['sample'] == 'IS'])
print(f'IS trades from IS-only tickers: {len(is_only_trades)} / {is_total} ({len(is_only_trades)/is_total*100:.1f}%)')

# ============================================================
# 14. PORTFOLIO EFFECT
# ============================================================
print()
print('=' * 80)
print('SECTION 14: PORTFOLIO EFFECT -- COMBINED DAILY PNL')
print('=' * 80)

daily_combined = pd.Series(0.0, index=calendar)
for strat in STRATS:
    sub = oos[oos['strategy'] == strat]
    daily = sub.groupby('date_dt')['pnl_r'].sum().reindex(calendar, fill_value=0.0)
    daily_combined += daily

port_mean = daily_combined.mean()
port_std = daily_combined.std(ddof=1)
port_sharpe = port_mean / port_std * np.sqrt(252) if port_std > 0 else 0

active_days = int((daily_combined != 0).sum())
print(f'Combined portfolio (equal weight, all 10 strategies):')
print(f'  Trading days: {n_cal}')
print(f'  Active days (>=1 trade): {active_days} ({active_days/n_cal*100:.1f}%)')
print(f'  Mean daily PnL: {port_mean:+.3f}R')
print(f'  Std daily PnL: {port_std:.3f}R')
print(f'  Daily Sharpe (ann): {port_sharpe:.2f}')
print(f'  Total PnL: {daily_combined.sum():+.1f}R')

cum = daily_combined.cumsum()
peak = cum.cummax()
dd = cum - peak
maxdd = dd.min()
print(f'  MaxDD: {maxdd:.1f}R')
calmar = port_mean * 252 / abs(maxdd) if maxdd != 0 else float('inf')
print(f'  Calmar ratio: {calmar:.2f}')

ac1 = daily_combined.autocorr(lag=1)
ac2 = daily_combined.autocorr(lag=2)
print(f'  Portfolio AC(1): {ac1:+.3f}')
print(f'  Portfolio AC(2): {ac2:+.3f}')

# Reduced portfolio
print()
print('Reduced portfolio (remove highly correlated 1.6, 1.7):')
REDUCED = [s for s in STRATS if s not in ['1.6_Low_Wick', '1.7_Big_Body']]
daily_reduced = pd.Series(0.0, index=calendar)
for strat in REDUCED:
    sub = oos[oos['strategy'] == strat]
    daily = sub.groupby('date_dt')['pnl_r'].sum().reindex(calendar, fill_value=0.0)
    daily_reduced += daily

red_mean = daily_reduced.mean()
red_std = daily_reduced.std(ddof=1)
red_sharpe = red_mean / red_std * np.sqrt(252) if red_std > 0 else 0
cum_r = daily_reduced.cumsum()
maxdd_r = (cum_r - cum_r.cummax()).min()
print(f'  Strategies: {len(REDUCED)}')
print(f'  Daily Sharpe (ann): {red_sharpe:.2f}')
print(f'  Total PnL: {daily_reduced.sum():+.1f}R')
print(f'  MaxDD: {maxdd_r:.1f}R')

# ============================================================
# 15. QUARTERLY BREAKDOWN
# ============================================================
print()
print('=' * 80)
print('SECTION 15: QUARTERLY BREAKDOWN (PORTFOLIO)')
print('=' * 80)

daily_df = daily_combined.to_frame('pnl')
daily_df['quarter'] = daily_df.index.to_period('Q')

qtr_stats = daily_df.groupby('quarter').agg(
    total_pnl=('pnl', 'sum'),
    mean_pnl=('pnl', 'mean'),
    std_pnl=('pnl', 'std'),
    n_days=('pnl', 'count'),
    active_days=('pnl', lambda x: int((x != 0).sum())),
)
qtr_stats['sharpe'] = qtr_stats['mean_pnl'] / qtr_stats['std_pnl'] * np.sqrt(252)

print(f'{"Quarter":>8} | {"TotalR":>7} | {"Mean":>7} | {"Std":>7} | {"Days":>5} | {"Active":>6} | {"Sharpe":>7}')
print('-' * 65)
for idx, r in qtr_stats.iterrows():
    print(f'{str(idx):>8} | {r["total_pnl"]:>+7.1f} | {r["mean_pnl"]:>+7.3f} | {r["std_pnl"]:>7.3f} | '
          f'{r["n_days"]:>5.0f} | {r["active_days"]:>6.0f} | {r["sharpe"]:>7.2f}')

# ============================================================
# 16. N-PROBLEMATIK (minimum sample size)
# ============================================================
print()
print('=' * 80)
print('SECTION 16: N-PROBLEMATIK -- MINIMUM SAMPLE SIZE')
print('=' * 80)

print('Rule of thumb: need N >= 100 for narrow CI, N >= 30 for usable estimates')
print('With fat tails (typical for trading), need even more: N >= 200 for robust inference')
print()
print(f'{"Strategy":<28} | {"N_OOS":>6} | {"SE(EV)":>7} | {"CI_width":>8} | {"N_needed_10%":>13} | {"Assessment":>12}')
print('-' * 95)

for strat in STRATS:
    sub = oos[oos['strategy'] == strat]
    n = len(sub)
    ev = sub['pnl_r'].mean()
    std = sub['pnl_r'].std(ddof=1)
    se = std / np.sqrt(n)
    ci_width = 2 * 1.96 * se

    # N needed for 10% precision: EV +/- 0.1*EV
    # se_target = 0.1 * abs(ev) / 1.96
    # N_needed = (std / se_target)^2
    if ev != 0:
        se_target = 0.1 * abs(ev) / 1.96
        n_needed = (std / se_target) ** 2 if se_target > 0 else float('inf')
    else:
        n_needed = float('inf')

    if n >= 200:
        assessment = 'ADEQUATE'
    elif n >= 100:
        assessment = 'MARGINAL'
    elif n >= 50:
        assessment = 'LOW'
    else:
        assessment = 'INSUFFICIENT'

    print(f'{strat:<28} | {n:>6} | {se:>7.3f} | {ci_width:>8.3f} | {n_needed:>13.0f} | {assessment:>12}')

print()
print('DONE.')
