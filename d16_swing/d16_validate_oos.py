"""
D16: Pattern-Based Swing Strategies — OOS Validation
======================================================
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
from d16_swing.strategies.s1_trend_continuation import run_s1
from d16_swing.strategies.s2_power_breakout import run_s2
from d16_swing.strategies.s3_dip_and_hold import run_s3

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

    # sl_dist in price terms = sl_adr_mult * adr
    df['sl_dist_price'] = df['sl_adr_mult'] * df['adr']

    # Fee per R
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

    # Raw
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


def main():
    t0 = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(msg)

    log("=" * 70)
    log("D16 Swing Strategies — OOS Validation")
    log("=" * 70)

    # Load best configs from IS
    log("\n[1] Loading IS best configs...")
    cfg_path = RESULTS_DIR / 'd16_best_configs.txt'
    if not cfg_path.exists():
        log("  ERROR: d16_best_configs.txt not found. Run d16_run_is.py first!")
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

    # Load data
    log("\n[2] Loading data...")
    meta = pd.read_parquet('data/metadata/metadata_v9.parquet')
    meta['date'] = meta['date'].astype(str)
    fmap = pd.read_parquet('data/metadata/followup_day_mapping.parquet')
    fmap['gap_date'] = fmap['gap_date'].astype(str)
    fmap['actual_date'] = fmap['actual_date'].astype(str)

    is_meta = meta[(meta['date'] >= IS_START) & (meta['date'] <= IS_END)].copy()
    oos_meta = meta[(meta['date'] >= OOS_START) & (meta['date'] <= OOS_END)].copy()
    log(f"  IS: {len(is_meta)} rows | OOS: {len(oos_meta)} rows")

    # Strategy runner map
    strat_runners = {
        'S1': run_s1,
        'S2': run_s2,
        'S3': run_s3,
    }

    # === Run IS + OOS with best configs ===
    log("\n[3] Running IS + OOS with best configs (no refit)...")

    all_trades = []
    kpi_table = []

    for strat_name, runner in strat_runners.items():
        if strat_name not in best_configs:
            log(f"  {strat_name}: SKIPPED (no config)")
            continue

        cfg = best_configs[strat_name]
        sl_mult = cfg['sl_mult']
        max_days = cfg['max_days']
        log(f"\n  {strat_name}: SL={sl_mult}, MaxDays={max_days}")

        # IS run
        is_trades = runner(is_meta, fmap, simulate_multiday_trail_d_streaming,
                           sl_adr_mult=sl_mult, max_days=max_days, label='IS')
        # OOS run
        oos_trades = runner(oos_meta, fmap, simulate_multiday_trail_d_streaming,
                            sl_adr_mult=sl_mult, max_days=max_days, label='OOS')

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
            exit_days = oos_df['exit_day'].values if 'exit_day' in oos_df.columns else []
            exit_reasons = oos_df['exit_reason'].value_counts().to_dict() if 'exit_reason' in oos_df.columns else {}
            if len(exit_days) > 0:
                log(f"    Exit day mean={np.mean(exit_days):.1f}, med={np.median(exit_days):.0f}")
                log(f"    Exit reasons: {exit_reasons}")

            # Charts
            is_df = pd.DataFrame(is_trades) if is_trades else pd.DataFrame()
            plot_equity_combined(is_df, oos_df, f"{strat_name} (SL={sl_mult}, D={max_days})",
                                 kpis_is, kpis_oos,
                                 CHARTS_DIR / f"eq_{strat_name.lower()}.png")
            plot_equity_3costs(oos_df, f"{strat_name} (SL={sl_mult}, D={max_days})",
                               kpis_by_scenario,
                               CHARTS_DIR / f"eq_{strat_name.lower()}_costs.png")
            plot_monte_carlo(oos_df['pnl_r'].values,
                             f"{strat_name} (SL={sl_mult}, D={max_days})",
                             CHARTS_DIR / f"mc_{strat_name.lower()}.png")
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

    # === Save Results ===
    log("\n[4] Saving results...")

    # KPI text file
    kpi_text = "D16 Swing Strategies — KPI Table (IS + OOS + Costs)\n"
    kpi_text += "=" * 80 + "\n\n"
    kpi_text += f"IS:  {IS_START} to {IS_END}\n"
    kpi_text += f"OOS: {OOS_START} to {OOS_END}\n\n"

    for row in kpi_table:
        s = row['Strategy']
        kpi_text += f"{s} (SL={row['SL']}, MaxDays={row['MaxDays']})\n"
        kpi_text += f"  IS:  N={row['IS_N']}, EV={row['IS_EV']:+.3f}R, WR={row['IS_WR']:.1%}, "
        kpi_text += f"CI=[{row['IS_CI_lo']:+.3f}, {row['IS_CI_hi']:+.3f}]\n"
        kpi_text += f"  OOS Raw:  N={row['OOS_N']}, EV={row['OOS_EV_raw']:+.3f}R, WR={row['OOS_WR']:.1%}, "
        kpi_text += f"CI=[{row['OOS_CI_lo']:+.3f}, {row['OOS_CI_hi']:+.3f}], MaxDD={row['OOS_MaxDD']:.2f}R\n"
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

    (RESULTS_DIR / 'd16_kpi_table.txt').write_text(kpi_text, encoding='utf-8')
    log(f"  KPI table: {RESULTS_DIR / 'd16_kpi_table.txt'}")

    # Trades parquet
    if all_trades:
        trades_out = pd.DataFrame(all_trades)
        trades_out.to_parquet(RESULTS_DIR / 'd16_all_trades.parquet', index=False)
        log(f"  All trades: {RESULTS_DIR / 'd16_all_trades.parquet'} ({len(trades_out)} records)")

    # Full log
    elapsed = time.time() - t0
    log(f"\n[DONE] OOS validation complete in {elapsed:.0f}s")
    (RESULTS_DIR / 'd16_oos_log.txt').write_text('\n'.join(log_lines), encoding='utf-8')


if __name__ == '__main__':
    main()
