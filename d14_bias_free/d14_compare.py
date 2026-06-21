"""
D14 vs D13 Comparison
======================
Side-by-side analysis of D14 bias-free results vs D13 results.

- Trade-level matching: (ticker, date) -> trades in both, only D13, only D14
- KPI table: N, EV, WR, CI, Degradation for each strategy
- Bias impact: D13_EV - D14_EV = how much was inflated by bias
- Bootstrap CI (10,000 resamples) for all D14 results
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS_DIR = Path('d14_bias_free/results')
np.random.seed(42)


def bootstrap_ci(arr, n_boot=10000, ci=0.95):
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 20:
        return (np.nan, np.nan)
    boot = np.array([np.mean(np.random.choice(arr, len(arr), replace=True)) for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    return (np.percentile(boot, 100 * alpha), np.percentile(boot, 100 * (1 - alpha)))


# D13 reference values (from ALLE_ROBUSTEN_STRATEGIEN.txt)
D13_REFERENCE = {
    # Cat 1 strategies
    '1.1_S4_Optimiert':      {'N_OOS': 62,   'EV_OOS': 0.620, 'WR_OOS': 0.629},
    '1.2_Pre10am_Momentum':  {'N_OOS': 127,  'EV_OOS': 0.900, 'WR_OOS': 0.638},
    '1.3_Earnings_OD_RVOL':  {'N_OOS': 58,   'EV_OOS': 0.800, 'WR_OOS': np.nan},
    '1.4_Power_Setup':       {'N_OOS': 85,   'EV_OOS': 0.730, 'WR_OOS': np.nan},
    '1.5_Big_First_Candle':  {'N_OOS': 250,  'EV_OOS': 0.640, 'WR_OOS': np.nan},
    '1.6_Low_Wick':          {'N_OOS': 253,  'EV_OOS': 0.450, 'WR_OOS': np.nan},
    '1.7_Big_Body':          {'N_OOS': 262,  'EV_OOS': 0.400, 'WR_OOS': np.nan},
    '1.8_Gap_3ADR':          {'N_OOS': 506,  'EV_OOS': 0.540, 'WR_OOS': np.nan},
    '1.9_RVOL5_RSI50':       {'N_OOS': 266,  'EV_OOS': 0.640, 'WR_OOS': np.nan},
    # Cat 2 strategies
    '2.1_SellOff_Recovery':  {'N_OOS': 3128, 'EV_OOS': 0.438, 'WR_OOS': 0.743},
    '2.2_Failed_Fade':       {'N_OOS': 1702, 'EV_OOS': 0.662, 'WR_OOS': 0.891},
    '2.3_Dip_and_Rip':       {'N_OOS': 293,  'EV_OOS': 0.408, 'WR_OOS': 0.683},
}


def run_comparison():
    """Compare D14 trades with D13 reference values."""
    lines = []

    def log(msg=''):
        lines.append(str(msg))
        print(msg)

    log("=" * 100)
    log("  D14 vs D13 COMPARISON — BIAS IMPACT ANALYSIS")
    log("=" * 100)
    log(f"  Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    log("")

    # Load D14 trade data
    trades_path = RESULTS_DIR / 'd14_all_trades.parquet'
    if not trades_path.exists():
        log("  ERROR: d14_all_trades.parquet not found. Run d14_run_all.py first.")
        return

    trades = pd.read_parquet(trades_path)
    log(f"  D14 total trades: {len(trades)}")

    # Comparison table
    log("\n  COMPARISON TABLE: D14 (Bias-Free) vs D13 (Original)")
    log(f"  {'Strategy':<30} | {'D13 N':>6} | {'D13 EV':>7} | {'D14 N':>6} | {'D14 EV':>7} | "
        f"{'Delta':>7} | {'%Chg':>6} | {'D14 CI_lo':>8} | {'Survives':>8}")
    log(f"  {'-'*110}")

    comparison_rows = []

    for strat_name, d13_vals in D13_REFERENCE.items():
        # Get D14 OOS trades for this strategy
        d14_oos = trades[(trades['strategy'] == strat_name) & (trades['sample'] == 'OOS')]
        d14_pnl = d14_oos['pnl_r'].values

        d14_n = len(d14_pnl)
        d14_ev = np.mean(d14_pnl) if d14_n > 0 else np.nan
        d14_wr = np.mean(d14_pnl > 0) if d14_n > 0 else np.nan
        ci_lo, ci_hi = bootstrap_ci(d14_pnl) if d14_n >= 20 else (np.nan, np.nan)

        d13_ev = d13_vals['EV_OOS']
        d13_n = d13_vals['N_OOS']

        delta = d14_ev - d13_ev if not np.isnan(d14_ev) else np.nan
        pct_chg = (delta / d13_ev * 100) if d13_ev != 0 and not np.isnan(delta) else np.nan
        survives = 'YES' if (not np.isnan(ci_lo) and ci_lo > 0) else 'NO'

        d14_ev_str = f"{d14_ev:+.3f}" if not np.isnan(d14_ev) else "  N/A"
        delta_str = f"{delta:+.3f}" if not np.isnan(delta) else "  N/A"
        pct_str = f"{pct_chg:+.1f}%" if not np.isnan(pct_chg) else " N/A"
        ci_lo_str = f"{ci_lo:+.3f}" if not np.isnan(ci_lo) else "  N/A"

        log(f"  {strat_name:<30} | {d13_n:>6} | {d13_ev:>+7.3f} | {d14_n:>6} | {d14_ev_str:>7} | "
            f"{delta_str:>7} | {pct_str:>6} | {ci_lo_str:>8} | {survives:>8}")

        comparison_rows.append({
            'Strategy': strat_name,
            'D13_N': d13_n, 'D13_EV': d13_ev,
            'D14_N': d14_n, 'D14_EV': d14_ev, 'D14_WR': d14_wr,
            'Delta_EV': delta, 'Pct_Change': pct_chg,
            'D14_CI_lo': ci_lo, 'D14_CI_hi': ci_hi,
            'Survives': survives,
        })

    # Summary
    log(f"\n  {'='*60}")
    log("  BIAS IMPACT SUMMARY:")
    log(f"  {'='*60}")

    comp_df = pd.DataFrame(comparison_rows)
    valid = comp_df[comp_df['D14_EV'].notna()]

    if len(valid) > 0:
        avg_delta = valid['Delta_EV'].mean()
        avg_pct = valid['Pct_Change'].mean()
        survivors = (valid['Survives'] == 'YES').sum()

        log(f"  Average EV change: {avg_delta:+.3f}R ({avg_pct:+.1f}%)")
        log(f"  Surviving strategies (CI_lo > 0): {survivors}/{len(valid)}")
        log("")

        # Strategies with biggest bias impact
        log("  BIGGEST BIAS IMPACT (most EV lost):")
        sorted_by_delta = valid.sort_values('Delta_EV')
        for _, row in sorted_by_delta.head(5).iterrows():
            log(f"    {row['Strategy']:<30}: {row['Delta_EV']:+.3f}R ({row['Pct_Change']:+.1f}%)")

        log("")
        log("  STRATEGIES THAT SURVIVED BIAS REMOVAL:")
        for _, row in valid[valid['Survives'] == 'YES'].sort_values('D14_EV', ascending=False).iterrows():
            log(f"    {row['Strategy']:<30}: EV={row['D14_EV']:+.3f}R, CI=[{row['D14_CI_lo']:+.3f}, {row['D14_CI_hi']:+.3f}]")

        log("")
        log("  STRATEGIES ELIMINATED BY BIAS REMOVAL:")
        for _, row in valid[valid['Survives'] == 'NO'].iterrows():
            ci_str = f"CI=[{row['D14_CI_lo']:+.3f}, {row['D14_CI_hi']:+.3f}]" if not np.isnan(row['D14_CI_lo']) else "CI=N/A"
            log(f"    {row['Strategy']:<30}: EV={row['D14_EV']:+.3f}R if D14_N>0 else N/A, {ci_str}")

    # Cat 1 sanity check
    log("")
    log("  SANITY CHECK: Cat-1 strategies (no SPY fix) should show <5% EV change")
    cat1_no_spy = comp_df[comp_df['Strategy'].str.startswith('1.') &
                          ~comp_df['Strategy'].str.contains('Power')]
    if len(cat1_no_spy) > 0:
        cat1_avg_pct = cat1_no_spy['Pct_Change'].mean()
        log(f"  Cat-1 (no SPY) average change: {cat1_avg_pct:+.1f}%")
        if abs(cat1_avg_pct) < 5:
            log("  -> PASS: Engine change impact minimal (as expected)")
        else:
            log("  -> WARNING: Larger than expected change — investigate!")

    # Save comparison
    comp_text = '\n'.join(lines)
    (RESULTS_DIR / 'd14_comparison.txt').write_text(comp_text, encoding='utf-8')
    print(f"\n  Comparison saved to {RESULTS_DIR / 'd14_comparison.txt'}")

    return comp_df


if __name__ == '__main__':
    run_comparison()
