"""
D16v2: VWAP-Band Swing Strategies — OOS Validation
=====================================================
Uses EXACT IS parameters (no refit). Adds cost scenarios from D15.
Outputs: KPI table (raw + 3 cost levels), equity curves, Monte Carlo,
IS vs OOS comparison.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from d14_bias_free.engine.streaming_trail import simulate_multiday_trail_d_streaming
from d16v2_swing.engine.rvol30_precompute import precompute_rvol30
from d16v2_swing.engine.yesterday_hl import precompute_yesterday_hl
from d16v2_swing.strategies.detect_all import detect_strategies

# === CONSTANTS ===
IS_START = '2021-02-21'
IS_END = '2023-12-31'
OOS_START = '2024-01-01'
OOS_END = '2026-02-06'

FEE_PER_SHARE = 0.0025  # $/share roundtrip
SLIPPAGE_SCENARIOS = {
    'opt': 0.02,
    'base': 0.05,
    'pess': 0.10,
}

MC_RUNS = 400
MC_SEED = 42
RAW_1MIN_DIR = Path('data/raw_1min')
RESULTS_DIR = Path('d16v2_swing/results')
CHARTS_DIR = RESULTS_DIR / 'charts'

STRATEGY_NAMES = ['S1_GapUp_Recovery', 'S2_GapDown_Recovery',
                  'S3_GapUp_Fade', 'S4_GapDown_Fade']


def bootstrap_ci(arr, n_boot=10000, ci=0.95):
    arr = np.array([x for x in arr if not np.isnan(x)])
    if len(arr) < 20:
        return np.nan, np.nan
    rng = np.random.RandomState(MC_SEED)
    means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)]
    alpha = (1 - ci) / 2
    return np.percentile(means, alpha * 100), np.percentile(means, (1 - alpha) * 100)


def compute_kpis(pnl_arr, label=''):
    arr = np.array(pnl_arr)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n < 5:
        return {'N': n, 'label': label, 'valid': False}

    ev = arr.mean()
    std = arr.std(ddof=1)
    wr = (arr > 0).mean()
    wins = arr[arr > 0]
    losses = arr[arr < 0]
    pf = abs(wins.sum() / losses.sum()) if len(losses) > 0 and losses.sum() != 0 else np.inf

    ci_lo, ci_hi = bootstrap_ci(arr)

    cum = np.cumsum(arr)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    max_dd = dd.min()

    max_consec = 0
    current = 0
    for x in (arr <= 0):
        if x:
            current += 1
            max_consec = max(max_consec, current)
        else:
            current = 0

    return {
        'N': n, 'EV': ev, 'Std': std, 'WR': wr, 'PF': pf,
        'CI_lo': ci_lo, 'CI_hi': ci_hi,
        'MaxDD': max_dd, 'MaxConsecLoss': max_consec,
        'AvgWin': wins.mean() if len(wins) > 0 else 0,
        'AvgLoss': losses.mean() if len(losses) > 0 else 0,
        'label': label, 'valid': True,
    }


def fmt_kpi(k):
    if not k.get('valid', False):
        return f"  {k.get('label','')}: N={k.get('N',0)} (insufficient data)"
    robust = 'ROBUST' if k['CI_lo'] > 0 else 'NOT ROBUST'
    return (f"  {k['label']}: N={k['N']}, EV={k['EV']:+.3f}R, WR={k['WR']:.1%}, "
            f"PF={k['PF']:.2f}, MaxDD={k['MaxDD']:.2f}R, "
            f"CI=[{k['CI_lo']:+.3f}, {k['CI_hi']:+.3f}] {robust}")


def apply_costs(trades_df):
    """Apply fee + slippage cost scenarios to trades."""
    if len(trades_df) == 0:
        return trades_df

    df = trades_df.copy()
    df['sl_dist_price'] = df['sl_adr_mult'] * df['adr']
    df['fee_r'] = FEE_PER_SHARE / df['sl_dist_price']
    df['fee_r'] = df['fee_r'].fillna(0)

    for scenario, slip_adr in SLIPPAGE_SCENARIOS.items():
        slip_r = slip_adr * df['adr'] / df['sl_dist_price']
        cost_r = df['fee_r'] + slip_r
        df[f'pnl_r_{scenario}'] = df['pnl_r'] - cost_r

    return df


def plot_equity_combined(is_trades, oos_trades, strategy, kpis_is, kpis_oos, out_path):
    """Equity curve with IS/OOS split."""
    fig, ax = plt.subplots(figsize=(14, 5))

    all_pnl = []
    if len(is_trades) > 0:
        is_sorted = is_trades.sort_values('date')
        all_pnl.extend(is_sorted['pnl_r'].values)
    split_idx = len(all_pnl)
    if len(oos_trades) > 0:
        oos_sorted = oos_trades.sort_values('date')
        all_pnl.extend(oos_sorted['pnl_r'].values)

    if len(all_pnl) == 0:
        plt.close(fig)
        return

    cum = np.cumsum(all_pnl)
    ax.plot(range(len(cum)), cum, color='steelblue', linewidth=1.2)
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    if split_idx > 0:
        ax.axvline(split_idx, color='red', linewidth=1, linestyle='--', label='IS/OOS split')

    is_ev = f"IS EV={kpis_is['EV']:+.3f}" if kpis_is.get('valid') else 'IS N/A'
    oos_ev = f"OOS EV={kpis_oos['EV']:+.3f}" if kpis_oos.get('valid') else 'OOS N/A'
    ax.set_title(f"{strategy} — {is_ev} | {oos_ev}")
    ax.set_xlabel('Trade #')
    ax.set_ylabel('Cumulative R')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_equity_3costs(oos_trades, strategy, kpis_by_scenario, out_path):
    """Equity curves for raw + 3 cost scenarios (OOS only)."""
    if len(oos_trades) == 0:
        return
    df = oos_trades.sort_values('date')

    fig, ax = plt.subplots(figsize=(14, 5))

    cum_raw = df['pnl_r'].cumsum()
    ax.plot(range(len(cum_raw)), cum_raw.values, color='black', linestyle='--',
            linewidth=1, label='Raw', alpha=0.7)

    colors = {'opt': 'green', 'base': 'blue', 'pess': 'red'}
    for scenario in ['opt', 'base', 'pess']:
        col = f'pnl_r_{scenario}'
        if col in df.columns:
            cum = df[col].cumsum()
            ax.plot(range(len(cum)), cum.values, color=colors[scenario],
                    linewidth=1.2, label=f'{scenario.title()} ({SLIPPAGE_SCENARIOS[scenario]} ADR)')

    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    base_kpi = kpis_by_scenario.get('base', {})
    base_ev = f"Base EV={base_kpi['EV']:+.3f}" if base_kpi.get('valid') else 'N/A'
    ax.set_title(f"{strategy} OOS Equity (3 Cost Scenarios) — {base_ev}")
    ax.set_xlabel('Trade #')
    ax.set_ylabel('Cumulative R')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_monte_carlo(pnl_arr, strategy, out_path):
    arr = np.array(pnl_arr)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 10:
        return
    rng = np.random.RandomState(MC_SEED)
    fig, ax = plt.subplots(figsize=(12, 5))

    finals = []
    for _ in range(MC_RUNS):
        shuffled = rng.permutation(arr)
        cum = np.cumsum(shuffled)
        ax.plot(range(len(cum)), cum, color='grey', alpha=0.03, linewidth=0.5)
        finals.append(cum[-1])

    actual_cum = np.cumsum(arr)
    ax.plot(range(len(actual_cum)), actual_cum, color='red', linewidth=1.5, label='Actual')

    p10 = np.percentile(finals, 10)
    p50 = np.percentile(finals, 50)
    p90 = np.percentile(finals, 90)

    ax.set_title(f"{strategy} OOS Monte Carlo ({MC_RUNS} runs)\n"
                 f"Final R: p10={p10:.1f}, p50={p50:.1f}, p90={p90:.1f}")
    ax.set_xlabel('Trade #')
    ax.set_ylabel('Cumulative R')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _get_day0_bars_after_entry(ticker, date):
    """Get 15:59 bar (1 bar remaining after 15:58 entry)."""
    path = RAW_1MIN_DIR / ticker / f"{date}.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        rth = df[df['session'] == 'rth']
        bars_after = rth[rth['time_et'] >= '15:59']
        if len(bars_after) == 0:
            return None
        return bars_after
    except Exception:
        return None


def _get_followup_sums(ticker, date, fmap, max_days):
    """Get followup day OHLC summaries for D+1..D+max_days."""
    fu = fmap[(fmap['ticker'] == ticker) & (fmap['gap_date'] == date)]
    fu = fu.sort_values('offset')

    sums = []
    for _, row in fu.iterrows():
        if row['offset'] < 1 or row['offset'] > max_days:
            continue
        actual_date = row['actual_date']
        path = RAW_1MIN_DIR / ticker / f"{actual_date}.parquet"
        if not path.exists():
            sums.append(None)
            continue
        try:
            df = pd.read_parquet(path)
            rth = df[df['session'] == 'rth']
            if len(rth) == 0:
                sums.append(None)
                continue
            sums.append({
                'rth_open': float(rth.iloc[0]['open']),
                'rth_close': float(rth.iloc[-1]['close']),
                'rth_high': float(rth['high'].max()),
                'rth_low': float(rth['low'].min()),
            })
        except Exception:
            sums.append(None)

    return sums


def run_all_strategies(meta_subset, fmap, rvol30_dict, yesterday_hl_dict,
                       configs_by_strategy, label=''):
    """
    Run strategy detection + multiday simulation using per-strategy configs.
    configs_by_strategy: {strategy_name: {'sl_mult': float, 'max_days': int}}
    Returns dict: {strategy_name: [trade_dicts]}
    """
    trades_by_strategy = {sn: [] for sn in configs_by_strategy}

    total = len(meta_subset)
    for idx, (_, row) in enumerate(meta_subset.iterrows()):
        if (idx + 1) % 500 == 0:
            print(f"      Processing {idx+1}/{total}...")

        ticker = row['ticker']
        date = str(row['date'])
        gap_dir = row['gap_direction']
        adr = row.get('adr_10', np.nan)
        prev_close = row.get('prev_close', np.nan)

        if pd.isna(adr) or adr <= 0:
            continue

        rvol30 = rvol30_dict.get((ticker, date), 0)
        yhl = yesterday_hl_dict.get((ticker, date))
        if yhl is None:
            continue
        yest_high, yest_low = yhl

        triggers = detect_strategies(
            ticker, date, gap_dir, adr, prev_close,
            rvol30, yest_high, yest_low
        )

        if not triggers:
            continue

        # Only load data once per (ticker, date)
        day0_bars = None
        followup_cache = {}

        for trig in triggers:
            strat_name = trig['strategy']
            if strat_name not in configs_by_strategy:
                continue

            cfg = configs_by_strategy[strat_name]
            sl_mult = cfg['sl_mult']
            max_days = cfg['max_days']

            if day0_bars is None:
                day0_bars = _get_day0_bars_after_entry(ticker, date)

            if max_days not in followup_cache:
                followup_cache[max_days] = _get_followup_sums(ticker, date, fmap, max_days)
            followup_sums = followup_cache[max_days]

            entry_price = trig['entry_price']
            sign = 1 if trig['trade_dir'] == 'long' else -1

            result = simulate_multiday_trail_d_streaming(
                entry_price=entry_price,
                sl_adr_mult=sl_mult,
                adr=adr,
                sign=sign,
                day0_bars_after_entry=day0_bars,
                followup_sums=followup_sums,
                max_days=max_days,
            )

            if result is None:
                continue

            trade = {
                'ticker': ticker,
                'date': date,
                'strategy': strat_name,
                'trade_dir': trig['trade_dir'],
                'entry_price': entry_price,
                'entry_time': trig['entry_time'],
                'sl_adr_mult': sl_mult,
                'max_days': max_days,
                'adr': adr,
                'rvol30': rvol30,
                'gap_direction': gap_dir,
                'pnl_r': result['pnl_r'],
                'exit_reason': result['exit_reason'],
                'exit_day': result['exit_day'],
                'max_fav_r': result['max_fav_r'],
                'trail_active': result['trail_active'],
            }
            trades_by_strategy[strat_name].append(trade)

    return trades_by_strategy


def main():
    t0 = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(msg)

    log("=" * 70)
    log("D16v2 VWAP-Band Swing Strategies — OOS Validation")
    log("=" * 70)

    # [1] Load best configs from IS
    log("\n[1] Loading IS best configs...")
    cfg_path = RESULTS_DIR / 'd16v2_best_configs.txt'
    if not cfg_path.exists():
        log("  ERROR: d16v2_best_configs.txt not found. Run d16v2_run_is.py first!")
        return

    best_configs = {}
    for line in cfg_path.read_text(encoding='utf-8').strip().split('\n'):
        if line.startswith('#'):
            continue
        if ':' not in line:
            continue
        strat, params = line.split(':', 1)
        strat = strat.strip()
        parts = dict(p.strip().split('=') for p in params.split(','))
        best_configs[strat] = {
            'sl_mult': float(parts['sl_mult']),
            'max_days': int(parts['max_days']),
        }
    log(f"  Loaded configs: {best_configs}")

    # [2] Load data
    log("\n[2] Loading data...")
    meta = pd.read_parquet('data/metadata/metadata_v9.parquet')
    meta['date'] = meta['date'].astype(str)
    fmap = pd.read_parquet('data/metadata/followup_day_mapping.parquet')
    fmap['gap_date'] = fmap['gap_date'].astype(str)
    fmap['actual_date'] = fmap['actual_date'].astype(str)

    is_meta = meta[(meta['date'] >= IS_START) & (meta['date'] <= IS_END)].copy()
    oos_meta = meta[(meta['date'] >= OOS_START) & (meta['date'] <= OOS_END)].copy()
    log(f"  IS: {len(is_meta)} rows | OOS: {len(oos_meta)} rows")

    # [3] Precompute for both IS + OOS
    full_meta = pd.concat([is_meta, oos_meta])

    log("\n[3] Precomputing RVOL_30 (IS + OOS)...")
    t_pre = time.time()
    rvol30_dict = precompute_rvol30(full_meta)
    log(f"  RVOL_30: {len(rvol30_dict)} entries ({time.time()-t_pre:.0f}s)")

    log("\n[4] Precomputing Yesterday H/L (IS + OOS)...")
    t_pre = time.time()
    yesterday_hl_dict = precompute_yesterday_hl(full_meta)
    log(f"  Yesterday H/L: {len(yesterday_hl_dict)} entries ({time.time()-t_pre:.0f}s)")

    # [5] Run IS + OOS with best configs
    log("\n[5] Running IS + OOS with best configs (no refit)...")

    all_trades = []
    kpi_table = []

    # IS run
    log("\n  --- IS Run ---")
    t_run = time.time()
    is_trades_by_strat = run_all_strategies(
        is_meta, fmap, rvol30_dict, yesterday_hl_dict,
        best_configs, label='IS'
    )
    log(f"  IS run took {time.time()-t_run:.0f}s")

    # OOS run
    log("\n  --- OOS Run ---")
    t_run = time.time()
    oos_trades_by_strat = run_all_strategies(
        oos_meta, fmap, rvol30_dict, yesterday_hl_dict,
        best_configs, label='OOS'
    )
    log(f"  OOS run took {time.time()-t_run:.0f}s")

    # [6] Compute KPIs and generate charts per strategy
    log("\n[6] Computing KPIs and generating charts...")

    for strat_name in STRATEGY_NAMES:
        if strat_name not in best_configs:
            log(f"\n  {strat_name}: SKIPPED (no config)")
            continue

        cfg = best_configs[strat_name]
        sl_mult = cfg['sl_mult']
        max_days = cfg['max_days']

        is_trades = is_trades_by_strat.get(strat_name, [])
        oos_trades = oos_trades_by_strat.get(strat_name, [])

        log(f"\n  {strat_name}: SL={sl_mult}, MaxDays={max_days}")

        # Tag samples
        for t in is_trades:
            t['sample'] = 'IS'
        for t in oos_trades:
            t['sample'] = 'OOS'

        # KPIs
        is_pnl = [t['pnl_r'] for t in is_trades]
        oos_pnl = [t['pnl_r'] for t in oos_trades]
        kpis_is = compute_kpis(is_pnl, label=f'{strat_name}_IS')
        kpis_oos = compute_kpis(oos_pnl, label=f'{strat_name}_OOS_raw')

        log(f"    IS:  {fmt_kpi(kpis_is)}")
        log(f"    OOS: {fmt_kpi(kpis_oos)}")

        # EV decay
        if kpis_is.get('valid') and kpis_oos.get('valid') and kpis_is['EV'] != 0:
            decay = 1 - kpis_oos['EV'] / kpis_is['EV']
            log(f"    EV Decay: {decay:.1%}")

        # Apply costs to OOS
        oos_df = pd.DataFrame(oos_trades)
        if len(oos_df) > 0:
            oos_df = apply_costs(oos_df)

            kpis_by_scenario = {'raw': kpis_oos}
            for scenario in ['opt', 'base', 'pess']:
                col = f'pnl_r_{scenario}'
                if col in oos_df.columns:
                    k = compute_kpis(oos_df[col].values,
                                     label=f'{strat_name}_OOS_{scenario}')
                    kpis_by_scenario[scenario] = k
                    log(f"    OOS {scenario}: {fmt_kpi(k)}")

            # Exit distribution
            if 'exit_day' in oos_df.columns:
                exit_days = oos_df['exit_day'].values
                exit_reasons = oos_df['exit_reason'].value_counts().to_dict()
                log(f"    Exit day mean={np.mean(exit_days):.1f}, med={np.median(exit_days):.0f}")
                log(f"    Exit reasons: {exit_reasons}")

            # Charts
            is_df = pd.DataFrame(is_trades) if is_trades else pd.DataFrame()
            sn_lower = strat_name.lower()

            plot_equity_combined(is_df, oos_df,
                                 f"{strat_name} (SL={sl_mult}, D={max_days})",
                                 kpis_is, kpis_oos,
                                 CHARTS_DIR / f"eq_{sn_lower}.png")
            plot_equity_3costs(oos_df,
                               f"{strat_name} (SL={sl_mult}, D={max_days})",
                               kpis_by_scenario,
                               CHARTS_DIR / f"eq_{sn_lower}_costs.png")
            plot_monte_carlo(oos_df['pnl_r'].values,
                             f"{strat_name} (SL={sl_mult}, D={max_days})",
                             CHARTS_DIR / f"mc_{sn_lower}.png")
            log(f"    Charts saved: eq, costs, mc")
        else:
            kpis_by_scenario = {}

        # Collect trades
        all_trades.extend(is_trades)
        all_trades.extend(oos_trades)

        # KPI table row
        row = {
            'Strategy': strat_name,
            'SL': sl_mult,
            'MaxDays': max_days,
            'IS_N': kpis_is.get('N', 0),
            'IS_EV': kpis_is.get('EV', np.nan),
            'IS_WR': kpis_is.get('WR', np.nan),
            'IS_CI_lo': kpis_is.get('CI_lo', np.nan),
            'IS_CI_hi': kpis_is.get('CI_hi', np.nan),
            'OOS_N': kpis_oos.get('N', 0),
            'OOS_EV_raw': kpis_oos.get('EV', np.nan),
            'OOS_WR': kpis_oos.get('WR', np.nan),
            'OOS_CI_lo': kpis_oos.get('CI_lo', np.nan),
            'OOS_CI_hi': kpis_oos.get('CI_hi', np.nan),
            'OOS_MaxDD': kpis_oos.get('MaxDD', np.nan),
        }
        for scenario in ['opt', 'base', 'pess']:
            k = kpis_by_scenario.get(scenario, {})
            row[f'OOS_EV_{scenario}'] = k.get('EV', np.nan)
            row[f'OOS_CI_lo_{scenario}'] = k.get('CI_lo', np.nan)
            row[f'OOS_CI_hi_{scenario}'] = k.get('CI_hi', np.nan)

        kpi_table.append(row)

    # [7] Save Results
    log("\n[7] Saving results...")

    # KPI text file
    kpi_text = "D16v2 VWAP-Band Swing Strategies — KPI Table (IS + OOS + Costs)\n"
    kpi_text += "=" * 80 + "\n\n"
    kpi_text += f"IS:  {IS_START} to {IS_END}\n"
    kpi_text += f"OOS: {OOS_START} to {OOS_END}\n\n"

    for row in kpi_table:
        s = row['Strategy']
        kpi_text += f"{s} (SL={row['SL']}, MaxDays={row['MaxDays']})\n"

        is_ev = row['IS_EV']
        is_wr = row['IS_WR']
        is_ci_lo = row['IS_CI_lo']
        is_ci_hi = row['IS_CI_hi']
        if not np.isnan(is_ev):
            kpi_text += f"  IS:  N={row['IS_N']}, EV={is_ev:+.3f}R, WR={is_wr:.1%}, "
            kpi_text += f"CI=[{is_ci_lo:+.3f}, {is_ci_hi:+.3f}]\n"
        else:
            kpi_text += f"  IS:  N={row['IS_N']} (insufficient data)\n"

        oos_ev = row['OOS_EV_raw']
        if not np.isnan(oos_ev):
            kpi_text += (f"  OOS Raw:  N={row['OOS_N']}, EV={oos_ev:+.3f}R, "
                         f"WR={row['OOS_WR']:.1%}, "
                         f"CI=[{row['OOS_CI_lo']:+.3f}, {row['OOS_CI_hi']:+.3f}], "
                         f"MaxDD={row['OOS_MaxDD']:.2f}R\n")
        else:
            kpi_text += f"  OOS Raw:  N={row['OOS_N']} (insufficient data)\n"

        for scenario in ['opt', 'base', 'pess']:
            ev = row.get(f'OOS_EV_{scenario}', np.nan)
            ci_lo = row.get(f'OOS_CI_lo_{scenario}', np.nan)
            ci_hi = row.get(f'OOS_CI_hi_{scenario}', np.nan)
            if not np.isnan(ev):
                robust = 'ROBUST' if ci_lo > 0 else 'NOT ROBUST'
                slip = SLIPPAGE_SCENARIOS[scenario]
                kpi_text += f"  OOS {scenario:>4} ({slip:.2f} ADR): EV={ev:+.3f}R, "
                kpi_text += f"CI=[{ci_lo:+.3f}, {ci_hi:+.3f}] {robust}\n"
        kpi_text += "\n"

    # IS vs OOS comparison
    kpi_text += "\n" + "=" * 80 + "\n"
    kpi_text += "IS vs OOS Comparison\n"
    kpi_text += "-" * 80 + "\n"
    for row in kpi_table:
        s = row['Strategy']
        is_ev = row['IS_EV']
        oos_ev = row['OOS_EV_raw']
        if is_ev != 0 and not np.isnan(is_ev) and not np.isnan(oos_ev):
            decay = 1 - oos_ev / is_ev
            kpi_text += f"  {s}: IS={is_ev:+.3f}R -> OOS={oos_ev:+.3f}R (decay={decay:+.1%})\n"

    (RESULTS_DIR / 'd16v2_kpi_table.txt').write_text(kpi_text, encoding='utf-8')
    log(f"  KPI table: {RESULTS_DIR / 'd16v2_kpi_table.txt'}")

    # Trades parquet
    if all_trades:
        trades_out = pd.DataFrame(all_trades)
        trades_out.to_parquet(RESULTS_DIR / 'd16v2_all_trades.parquet', index=False)
        log(f"  All trades: {RESULTS_DIR / 'd16v2_all_trades.parquet'} ({len(trades_out)} records)")

    # Log
    elapsed = time.time() - t0
    log(f"\n[DONE] OOS validation complete in {elapsed:.0f}s")
    (RESULTS_DIR / 'd16v2_oos_log.txt').write_text('\n'.join(log_lines), encoding='utf-8')


if __name__ == '__main__':
    main()
