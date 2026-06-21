"""
D16v2: VWAP-Band Swing Strategies — IS Analysis
=================================================
Runs S1-S4 on IS period (2021-02-21 to 2023-12-31).
Sensitivity grid: SL width [0.25, 0.50, 0.75] x max_days [5, 10].
Selects best config per strategy by EV/|MaxDD| ratio.
Outputs: KPI table, equity curves, Monte Carlo, trade parquet, best configs.
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
SL_GRID = [0.25, 0.50, 0.75]
DAYS_GRID = [5, 10]
MC_RUNS = 400
MC_SEED = 42
RAW_1MIN_DIR = Path('data/raw_1min')
RESULTS_DIR = Path('d16v2_swing/results')
CHARTS_DIR = RESULTS_DIR / 'charts'


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


def plot_equity(trades_df, strategy, kpis, out_path):
    if len(trades_df) == 0:
        return
    cum_r = trades_df['pnl_r'].cumsum()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(range(len(cum_r)), cum_r.values, color='steelblue', linewidth=1.2)
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Trade #')
    ax.set_ylabel('Cumulative R')
    ev_str = f"EV={kpis['EV']:+.3f}R" if kpis.get('valid') else 'N/A'
    ax.set_title(f"{strategy} IS Equity — {ev_str}, N={kpis.get('N', 0)}")
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
    max_dds = []
    for _ in range(MC_RUNS):
        shuffled = rng.permutation(arr)
        cum = np.cumsum(shuffled)
        ax.plot(range(len(cum)), cum, color='grey', alpha=0.03, linewidth=0.5)
        finals.append(cum[-1])
        peak = np.maximum.accumulate(cum)
        max_dds.append((cum - peak).min())

    actual_cum = np.cumsum(arr)
    ax.plot(range(len(actual_cum)), actual_cum, color='red', linewidth=1.5, label='Actual')

    p10 = np.percentile(finals, 10)
    p50 = np.percentile(finals, 50)
    p90 = np.percentile(finals, 90)
    dd_p10 = np.percentile(max_dds, 10)

    ax.set_title(f"{strategy} Monte Carlo ({MC_RUNS} runs)\n"
                 f"Final R: p10={p10:.1f}, p50={p50:.1f}, p90={p90:.1f} | "
                 f"MaxDD p10={dd_p10:.1f}R")
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
                       sl_adr_mult, max_days, label=''):
    """
    Run strategy detection + multiday simulation for all (ticker, date)
    in meta_subset. Returns dict: {strategy_name: [trade_dicts]}
    """
    trades_by_strategy = {}

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

        # Detect which strategies fire
        triggers = detect_strategies(
            ticker, date, gap_dir, adr, prev_close,
            rvol30, yest_high, yest_low
        )

        if not triggers:
            continue

        # For each triggered strategy, simulate multiday exit
        day0_bars = _get_day0_bars_after_entry(ticker, date)
        followup_sums = _get_followup_sums(ticker, date, fmap, max_days)

        for trig in triggers:
            strat_name = trig['strategy']
            entry_price = trig['entry_price']
            sign = 1 if trig['trade_dir'] == 'long' else -1

            result = simulate_multiday_trail_d_streaming(
                entry_price=entry_price,
                sl_adr_mult=sl_adr_mult,
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
                'sl_adr_mult': sl_adr_mult,
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

            if strat_name not in trades_by_strategy:
                trades_by_strategy[strat_name] = []
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
    log("D16v2 VWAP-Band Swing Strategies — IS Analysis")
    log("=" * 70)

    # [1] Load data
    log("\n[1] Loading data...")
    meta = pd.read_parquet('data/metadata/metadata_v9.parquet')
    meta['date'] = meta['date'].astype(str)
    fmap = pd.read_parquet('data/metadata/followup_day_mapping.parquet')
    fmap['gap_date'] = fmap['gap_date'].astype(str)
    fmap['actual_date'] = fmap['actual_date'].astype(str)

    # IS split
    is_meta = meta[(meta['date'] >= IS_START) & (meta['date'] <= IS_END)].copy()
    log(f"  IS period: {IS_START} to {IS_END}, N={len(is_meta)} rows")

    # [2] Precompute RVOL_30 and Yesterday H/L
    log("\n[2] Precomputing RVOL_30...")
    t_pre = time.time()
    rvol30_dict = precompute_rvol30(is_meta)
    log(f"  RVOL_30: {len(rvol30_dict)} entries ({time.time()-t_pre:.0f}s)")

    log("\n[3] Precomputing Yesterday H/L...")
    t_pre = time.time()
    yesterday_hl_dict = precompute_yesterday_hl(is_meta)
    log(f"  Yesterday H/L: {len(yesterday_hl_dict)} entries ({time.time()-t_pre:.0f}s)")

    # [4] Sensitivity Grid
    log("\n[4] Running sensitivity grid: SL x MaxDays")
    log(f"  SL grid: {SL_GRID}")
    log(f"  Days grid: {DAYS_GRID}")

    STRATEGY_NAMES = ['S1_GapUp_Recovery', 'S2_GapDown_Recovery',
                      'S3_GapUp_Fade', 'S4_GapDown_Fade']

    all_grid_results = {}  # {(strategy, sl, days): [trades]}

    for sl_mult in SL_GRID:
        for max_d in DAYS_GRID:
            config_key = f"SL={sl_mult}_D={max_d}"
            log(f"\n  --- Config: {config_key} ---")
            t_cfg = time.time()

            trades_by_strat = run_all_strategies(
                is_meta, fmap, rvol30_dict, yesterday_hl_dict,
                sl_adr_mult=sl_mult, max_days=max_d, label='IS'
            )

            for sn in STRATEGY_NAMES:
                trades = trades_by_strat.get(sn, [])
                all_grid_results[(sn, sl_mult, max_d)] = trades
                if trades:
                    pnl_arr = [t['pnl_r'] for t in trades]
                    kpis = compute_kpis(pnl_arr, label=f"{sn}_SL{sl_mult}_D{max_d}")
                    log(f"    {fmt_kpi(kpis)}")
                else:
                    log(f"    {sn}: N=0")

            log(f"    (config took {time.time()-t_cfg:.0f}s)")

    # [5] Select Best Config per Strategy
    log("\n[5] Selecting best config per strategy (EV/|MaxDD| ratio)")

    best_configs = {}
    for strat_name in STRATEGY_NAMES:
        best_ratio = -np.inf
        best_key = None
        best_kpis = None

        log(f"\n  {strat_name} Grid Summary:")
        for sl_mult in SL_GRID:
            for max_d in DAYS_GRID:
                trades = all_grid_results[(strat_name, sl_mult, max_d)]
                if len(trades) == 0:
                    continue

                pnl_list = [t['pnl_r'] for t in trades]
                kpis = compute_kpis(pnl_list, label=f"{strat_name}_SL{sl_mult}_D{max_d}")

                if kpis.get('valid') and kpis['MaxDD'] < 0:
                    ratio = kpis['EV'] / abs(kpis['MaxDD'])
                elif kpis.get('valid') and kpis['EV'] > 0:
                    ratio = kpis['EV'] * 100
                else:
                    ratio = -np.inf

                if ratio > best_ratio:
                    best_ratio = ratio
                    best_key = (sl_mult, max_d)
                    best_kpis = kpis

        if best_key:
            best_configs[strat_name] = {
                'sl_mult': best_key[0],
                'max_days': best_key[1],
                'trades': all_grid_results[(strat_name, best_key[0], best_key[1])],
                'kpis': best_kpis,
            }
            log(f"  >>> {strat_name} BEST: SL={best_key[0]}, Days={best_key[1]}, "
                f"EV/MaxDD={best_ratio:.4f}")
        else:
            log(f"  >>> {strat_name}: NO VALID CONFIG")

    # [6] Final IS KPIs
    log("\n[6] Final IS KPIs (best config per strategy)")
    log("-" * 70)

    all_trades = []
    kpi_rows = []

    for strat_name in STRATEGY_NAMES:
        if strat_name not in best_configs:
            log(f"  {strat_name}: SKIPPED (no valid config)")
            continue

        cfg = best_configs[strat_name]
        kpis = cfg['kpis']
        trades = cfg['trades']
        log(fmt_kpi(kpis))

        for t in trades:
            t['sample'] = 'IS'
            t['best_sl_mult'] = cfg['sl_mult']
            t['best_max_days'] = cfg['max_days']
        all_trades.extend(trades)

        if trades:
            exit_days = [t.get('exit_day', 0) for t in trades]
            exit_reasons = pd.Series([t.get('exit_reason', 'unknown') for t in trades])
            log(f"    Exit day mean={np.mean(exit_days):.1f}, "
                f"median={np.median(exit_days):.0f}")
            log(f"    Exit reasons: {dict(exit_reasons.value_counts())}")

        kpi_rows.append({
            'Strategy': strat_name,
            'SL': cfg['sl_mult'],
            'MaxDays': cfg['max_days'],
            **{k: v for k, v in kpis.items() if k != 'label'},
        })

    # [7] Charts
    log("\n[7] Generating charts...")

    for strat_name in STRATEGY_NAMES:
        if strat_name not in best_configs:
            continue
        cfg = best_configs[strat_name]
        trades_df = pd.DataFrame(cfg['trades'])
        kpis = cfg['kpis']

        if len(trades_df) > 0:
            trades_df = trades_df.sort_values('date')
            sn_lower = strat_name.lower()
            plot_equity(trades_df, f"{strat_name} (SL={cfg['sl_mult']}, D={cfg['max_days']})",
                        kpis, CHARTS_DIR / f"eq_{sn_lower}_is.png")
            plot_monte_carlo(trades_df['pnl_r'].values,
                             f"{strat_name} (SL={cfg['sl_mult']}, D={cfg['max_days']})",
                             CHARTS_DIR / f"mc_{sn_lower}_is.png")
            log(f"  {strat_name}: equity + MC charts saved")

    # [8] Save Results
    log("\n[8] Saving results...")

    # KPI table
    kpi_text = "D16v2 VWAP-Band Swing Strategies — IS KPI Table\n"
    kpi_text += "=" * 70 + "\n\n"
    kpi_text += f"IS Period: {IS_START} to {IS_END}\n"
    kpi_text += f"SL Grid: {SL_GRID}\n"
    kpi_text += f"Days Grid: {DAYS_GRID}\n\n"
    for row in kpi_rows:
        strat = row['Strategy']
        kpi_text += f"{strat} (SL={row['SL']}, MaxDays={row['MaxDays']}):\n"
        kpi_text += f"  N={row['N']}, EV={row['EV']:+.3f}R, WR={row['WR']:.1%}\n"
        kpi_text += f"  PF={row['PF']:.2f}, MaxDD={row['MaxDD']:.2f}R\n"
        kpi_text += f"  CI=[{row['CI_lo']:+.3f}, {row['CI_hi']:+.3f}]\n"
        kpi_text += f"  AvgWin={row['AvgWin']:+.3f}, AvgLoss={row['AvgLoss']:+.3f}\n"
        robust = 'ROBUST' if row['CI_lo'] > 0 else 'NOT ROBUST'
        kpi_text += f"  Status: {robust}\n\n"

    (RESULTS_DIR / 'd16v2_is_kpi_table.txt').write_text(kpi_text, encoding='utf-8')
    log(f"  KPI table: {RESULTS_DIR / 'd16v2_is_kpi_table.txt'}")

    # Trades parquet
    if all_trades:
        trades_out = pd.DataFrame(all_trades)
        trades_out.to_parquet(RESULTS_DIR / 'd16v2_is_trades.parquet', index=False)
        log(f"  Trades: {RESULTS_DIR / 'd16v2_is_trades.parquet'} ({len(trades_out)} records)")

    # Best configs for OOS
    best_cfg_text = "# D16v2 Best IS Configs (for OOS)\n"
    for strat_name in STRATEGY_NAMES:
        if strat_name in best_configs:
            cfg = best_configs[strat_name]
            best_cfg_text += f"{strat_name}: sl_mult={cfg['sl_mult']}, max_days={cfg['max_days']}\n"
    (RESULTS_DIR / 'd16v2_best_configs.txt').write_text(best_cfg_text, encoding='utf-8')
    log(f"  Best configs: {RESULTS_DIR / 'd16v2_best_configs.txt'}")

    # Full sensitivity grid table
    grid_text = "D16v2 Full Sensitivity Grid — IS\n"
    grid_text += "=" * 80 + "\n\n"
    for strat_name in STRATEGY_NAMES:
        grid_text += f"\n{strat_name}:\n"
        grid_text += f"  {'SL':>6} {'Days':>5} {'N':>5} {'EV':>8} {'WR':>7} {'MaxDD':>8} {'CI_lo':>8} {'CI_hi':>8} {'Status':>12}\n"
        grid_text += "  " + "-" * 75 + "\n"
        for sl_mult in SL_GRID:
            for max_d in DAYS_GRID:
                trades = all_grid_results.get((strat_name, sl_mult, max_d), [])
                if len(trades) == 0:
                    grid_text += f"  {sl_mult:>6.2f} {max_d:>5d} {'0':>5}\n"
                    continue
                pnl_list = [t['pnl_r'] for t in trades]
                k = compute_kpis(pnl_list)
                if k.get('valid'):
                    robust = 'ROBUST' if k['CI_lo'] > 0 else 'NOT ROBUST'
                    grid_text += (f"  {sl_mult:>6.2f} {max_d:>5d} {k['N']:>5d} "
                                  f"{k['EV']:>+8.3f} {k['WR']:>6.1%} {k['MaxDD']:>8.2f} "
                                  f"{k['CI_lo']:>+8.3f} {k['CI_hi']:>+8.3f} {robust:>12}\n")
                else:
                    grid_text += f"  {sl_mult:>6.2f} {max_d:>5d} {k.get('N',0):>5d} (insufficient)\n"
    (RESULTS_DIR / 'd16v2_sensitivity_grid.txt').write_text(grid_text, encoding='utf-8')

    # Log
    elapsed = time.time() - t0
    log(f"\n[DONE] IS analysis complete in {elapsed:.0f}s")
    (RESULTS_DIR / 'd16v2_is_log.txt').write_text('\n'.join(log_lines), encoding='utf-8')

    return best_configs


if __name__ == '__main__':
    main()
