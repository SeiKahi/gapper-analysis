"""
D16: Pattern-Based Swing Strategies — IS Analysis
===================================================
Runs S1, S2, S3 on IS period (2021-02-21 to 2023-12-31).
Sensitivity grid: SL width [0.35, 0.50, 0.75] x max_days [3, 5].
Selects best config per strategy by EV/MaxDD ratio.
Outputs: KPI table, equity curves, Monte Carlo, trade parquet.
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
from d16_swing.strategies.s1_trend_continuation import run_s1
from d16_swing.strategies.s2_power_breakout import run_s2
from d16_swing.strategies.s3_dip_and_hold import run_s3

# === CONSTANTS ===
IS_START = '2021-02-21'
IS_END = '2023-12-31'
SL_GRID = [0.35, 0.50, 0.75]
DAYS_GRID = [3, 5]
MC_RUNS = 400
MC_SEED = 42
RESULTS_DIR = Path('d16_swing/results')
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

    # Bootstrap CI
    ci_lo, ci_hi = bootstrap_ci(arr)

    # Drawdown
    cum = np.cumsum(arr)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    max_dd = dd.min()

    # Max consecutive losses
    is_loss = arr <= 0
    max_consec = 0
    current = 0
    for x in is_loss:
        if x:
            current += 1
            max_consec = max(max_consec, current)
        else:
            current = 0

    # Exit day distribution
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

    # Actual
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


def main():
    t0 = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(msg)

    log("=" * 70)
    log("D16 Swing Strategies — IS Analysis")
    log("=" * 70)

    # Load data
    log("\n[1] Loading data...")
    meta = pd.read_parquet('data/metadata/metadata_v9.parquet')
    meta['date'] = meta['date'].astype(str)
    fmap = pd.read_parquet('data/metadata/followup_day_mapping.parquet')
    fmap['gap_date'] = fmap['gap_date'].astype(str)
    fmap['actual_date'] = fmap['actual_date'].astype(str)

    # IS split
    is_meta = meta[(meta['date'] >= IS_START) & (meta['date'] <= IS_END)].copy()
    log(f"  IS period: {IS_START} to {IS_END}, N={len(is_meta)} rows")

    # === Sensitivity Grid ===
    log("\n[2] Running sensitivity grid: SL x MaxDays")
    log(f"  SL grid: {SL_GRID}")
    log(f"  Days grid: {DAYS_GRID}")

    all_grid_results = {}  # {(strategy, sl, days): [trades]}

    for sl_mult in SL_GRID:
        for max_d in DAYS_GRID:
            config_key = f"SL={sl_mult}_D={max_d}"
            log(f"\n  --- Config: {config_key} ---")

            trades_s1 = run_s1(is_meta, fmap, simulate_multiday_trail_d_streaming,
                               sl_adr_mult=sl_mult, max_days=max_d, label='IS')
            trades_s2 = run_s2(is_meta, fmap, simulate_multiday_trail_d_streaming,
                               sl_adr_mult=sl_mult, max_days=max_d, label='IS')
            trades_s3 = run_s3(is_meta, fmap, simulate_multiday_trail_d_streaming,
                               sl_adr_mult=sl_mult, max_days=max_d, label='IS')

            all_grid_results[('S1', sl_mult, max_d)] = trades_s1
            all_grid_results[('S2', sl_mult, max_d)] = trades_s2
            all_grid_results[('S3', sl_mult, max_d)] = trades_s3

    # === Select Best Config per Strategy ===
    log("\n[3] Selecting best config per strategy (EV/MaxDD ratio)")

    best_configs = {}
    for strat_name in ['S1', 'S2', 'S3']:
        best_ratio = -np.inf
        best_key = None
        best_kpis = None

        log(f"\n  {strat_name} Grid Results:")
        for sl_mult in SL_GRID:
            for max_d in DAYS_GRID:
                trades = all_grid_results[(strat_name, sl_mult, max_d)]
                if len(trades) == 0:
                    log(f"    SL={sl_mult}, D={max_d}: N=0")
                    continue

                pnl_list = [t['pnl_r'] for t in trades]
                kpis = compute_kpis(pnl_list, label=f"{strat_name}_SL{sl_mult}_D{max_d}")
                log(f"    {fmt_kpi(kpis)}")

                if kpis.get('valid') and kpis['MaxDD'] < 0:
                    ratio = kpis['EV'] / abs(kpis['MaxDD'])
                elif kpis.get('valid') and kpis['EV'] > 0:
                    ratio = kpis['EV'] * 100  # no drawdown = very good
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
            log(f"\n  >>> {strat_name} BEST: SL={best_key[0]}, Days={best_key[1]}, "
                f"EV/MaxDD={best_ratio:.3f}")
        else:
            log(f"\n  >>> {strat_name}: NO VALID CONFIG")

    # === Final KPIs ===
    log("\n[4] Final IS KPIs (best config per strategy)")
    log("-" * 70)

    all_trades = []
    kpi_rows = []

    for strat_name in ['S1', 'S2', 'S3']:
        if strat_name not in best_configs:
            log(f"  {strat_name}: SKIPPED (no valid config)")
            continue

        cfg = best_configs[strat_name]
        kpis = cfg['kpis']
        trades = cfg['trades']
        log(fmt_kpi(kpis))

        # Add to all trades
        for t in trades:
            t['sample'] = 'IS'
            t['best_sl_mult'] = cfg['sl_mult']
            t['best_max_days'] = cfg['max_days']
        all_trades.extend(trades)

        # Exit day distribution
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

    # === Charts ===
    log("\n[5] Generating charts...")

    for strat_name in ['S1', 'S2', 'S3']:
        if strat_name not in best_configs:
            continue
        cfg = best_configs[strat_name]
        trades_df = pd.DataFrame(cfg['trades'])
        kpis = cfg['kpis']

        if len(trades_df) > 0:
            trades_df = trades_df.sort_values('date')
            # Equity curve
            plot_equity(trades_df, f"{strat_name} (SL={cfg['sl_mult']}, D={cfg['max_days']})",
                        kpis, CHARTS_DIR / f"eq_{strat_name.lower()}_is.png")
            # Monte Carlo
            plot_monte_carlo(trades_df['pnl_r'].values,
                             f"{strat_name} (SL={cfg['sl_mult']}, D={cfg['max_days']})",
                             CHARTS_DIR / f"mc_{strat_name.lower()}_is.png")
            log(f"  {strat_name}: equity + MC charts saved")

    # === Save Results ===
    log("\n[6] Saving results...")

    # KPI table
    kpi_text = "D16 Swing Strategies — IS KPI Table\n"
    kpi_text += "=" * 70 + "\n\n"
    kpi_text += f"IS Period: {IS_START} to {IS_END}\n\n"
    for row in kpi_rows:
        strat = row['Strategy']
        kpi_text += f"{strat} (SL={row['SL']}, MaxDays={row['MaxDays']}):\n"
        kpi_text += f"  N={row['N']}, EV={row['EV']:+.3f}R, WR={row['WR']:.1%}\n"
        kpi_text += f"  PF={row['PF']:.2f}, MaxDD={row['MaxDD']:.2f}R\n"
        kpi_text += f"  CI=[{row['CI_lo']:+.3f}, {row['CI_hi']:+.3f}]\n"
        kpi_text += f"  AvgWin={row['AvgWin']:+.3f}, AvgLoss={row['AvgLoss']:+.3f}\n"
        robust = 'ROBUST' if row['CI_lo'] > 0 else 'NOT ROBUST'
        kpi_text += f"  Status: {robust}\n\n"

    (RESULTS_DIR / 'd16_is_kpi_table.txt').write_text(kpi_text, encoding='utf-8')
    log(f"  KPI table: {RESULTS_DIR / 'd16_is_kpi_table.txt'}")

    # Trades parquet
    if all_trades:
        trades_out = pd.DataFrame(all_trades)
        trades_out.to_parquet(RESULTS_DIR / 'd16_is_trades.parquet', index=False)
        log(f"  Trades: {RESULTS_DIR / 'd16_is_trades.parquet'} ({len(trades_out)} records)")

    # Best configs for OOS
    best_cfg_text = "# D16 Best IS Configs (for OOS)\n"
    for strat_name in ['S1', 'S2', 'S3']:
        if strat_name in best_configs:
            cfg = best_configs[strat_name]
            best_cfg_text += f"{strat_name}: sl_mult={cfg['sl_mult']}, max_days={cfg['max_days']}\n"
    (RESULTS_DIR / 'd16_best_configs.txt').write_text(best_cfg_text, encoding='utf-8')

    # Full log
    elapsed = time.time() - t0
    log(f"\n[DONE] IS analysis complete in {elapsed:.0f}s")
    (RESULTS_DIR / 'd16_is_log.txt').write_text('\n'.join(log_lines), encoding='utf-8')

    return best_configs


if __name__ == '__main__':
    main()
