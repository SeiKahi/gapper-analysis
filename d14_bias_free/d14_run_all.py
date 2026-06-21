"""
D14 Bias-Free Re-Evaluation — Main Orchestrator
=================================================
Runs all D13-robust strategies with bias-free engine.
NO new parameters, NO optimization — only bias fixes.

Phases:
  0. Load metadata, precompute SPY returns, build 1-sec index
  1. Category 1: 9 close_935 strategies (IS + OOS)
  2. Category 2: 3 pattern-based strategies (IS + OOS)
  3. Category 3: 7 late-day swing patterns x 3 entry times (IS + OOS)
  4. Comparison: D14 vs D13 side-by-side

Usage:
  .\\gapper_env\\Scripts\\python.exe d14_bias_free/d14_run_all.py
"""
import pandas as pd
import numpy as np
import sys
import time
import warnings
from pathlib import Path
from scipy import stats as sp_stats

warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from d14_bias_free.engine.streaming_trail import simulate_trail_d_streaming
from d14_bias_free.engine.spy_realtime import build_spy_return_935
from d14_bias_free.engine.intrabar_resolver import IntrabarResolver
from d14_bias_free.strategies.cat1_close935 import run_cat1_strategies
from d14_bias_free.strategies.cat2_selloff_recovery import run_cat2_selloff
from d14_bias_free.strategies.cat2_failed_fade import run_cat2_failed_fade
from d14_bias_free.strategies.cat2_dip_and_rip import run_cat2_dip_and_rip
from d14_bias_free.strategies.cat3_lateday_swing import run_cat3_lateday


# ============================================================
# CONSTANTS
# ============================================================
META_PATH = Path('data/metadata/metadata_v9.parquet')
RESULTS_DIR = Path('d14_bias_free/results')
MEMORY_DIR = Path('d14_bias_free/memory')

IS_START = '2021-02-21'
IS_END = '2023-12-31'
OOS_START = '2024-01-01'
OOS_END = '2026-02-21'

np.random.seed(42)
lines = []


# ============================================================
# HELPERS
# ============================================================
def log(msg=''):
    lines.append(str(msg))
    print(msg)


def log_header(title):
    log(f"\n{'='*80}")
    log(f"  {title}")
    log(f"{'='*80}")


def bootstrap_ci(arr, n_boot=10000, ci=0.95):
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 20:
        return (np.nan, np.nan)
    boot = np.array([np.mean(np.random.choice(arr, len(arr), replace=True)) for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    return (np.percentile(boot, 100 * alpha), np.percentile(boot, 100 * (1 - alpha)))


def compute_stats(pnl_list, label=''):
    arr = np.asarray(pnl_list, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 5:
        return {'N': len(arr), 'label': label, 'valid': False}
    ev = np.mean(arr)
    med = np.median(arr)
    std = np.std(arr, ddof=1)
    wr = np.mean(arr > 0)
    se = std / np.sqrt(len(arr))
    t_stat = ev / se if se > 0 else 0
    p_val = 2 * (1 - sp_stats.t.cdf(abs(t_stat), df=len(arr) - 1))
    ci_lo, ci_hi = bootstrap_ci(arr)
    wins = arr[arr > 0]
    losses = arr[arr <= 0]
    pf = abs(np.sum(wins) / np.sum(losses)) if len(losses) > 0 and np.sum(losses) != 0 else float('inf')
    return {
        'N': len(arr), 'EV': ev, 'Med': med, 'Std': std, 'WR': wr,
        'PF': pf, 't_stat': t_stat, 'p_val': p_val,
        'CI_lo': ci_lo, 'CI_hi': ci_hi,
        'AvgWin': np.mean(wins) if len(wins) > 0 else 0,
        'AvgLoss': np.mean(losses) if len(losses) > 0 else 0,
        'label': label, 'valid': True,
    }


def fmt_stats(s):
    if not s['valid']:
        return f"  {s['label']:<50} | N={s['N']:>5} | [INSUFFICIENT]"
    sig = "***" if s['p_val'] < 0.01 else "**" if s['p_val'] < 0.05 else "*" if s['p_val'] < 0.10 else ""
    ci_str = f"[{s['CI_lo']:+.3f}, {s['CI_hi']:+.3f}]" if not np.isnan(s['CI_lo']) else "[N/A]"
    robust = "ROBUST" if (not np.isnan(s['CI_lo']) and s['CI_lo'] > 0) else "NOT ROBUST"
    return (
        f"  {s['label']:<50} | N={s['N']:>5} | EV={s['EV']:>+7.3f}R | "
        f"WR={s['WR']:>5.1%} | PF={s['PF']:>5.2f} | p={s['p_val']:>.4f}{sig} | "
        f"CI={ci_str} | {robust}"
    )


# ============================================================
# MAIN
# ============================================================
def main():
    t_start = time.time()

    log("=" * 80)
    log("  D14 BIAS-FREE RE-EVALUATION — ALLE D13-ROBUSTEN STRATEGIEN")
    log("=" * 80)
    log(f"  Datum: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    log(f"  IS:  {IS_START} bis {IS_END}")
    log(f"  OOS: {OOS_START} bis {OOS_END}")
    log(f"  Engine: Streaming TRAIL_D (SL-first, conservative)")
    log(f"  Bias Fixes: SPY open-return, bar-by-bar gap fill, streaming LOD/HOD")
    log(f"  Data Snooping: ZERO new parameters, exact D13 values")
    log("")

    # ============================================================
    # PHASE 0: DATA LOADING
    # ============================================================
    log_header("PHASE 0: DATA LOADING")

    log("  Loading metadata_v9.parquet...")
    meta = pd.read_parquet(META_PATH)
    meta['date'] = meta['date'].astype(str)
    log(f"  Total events: {len(meta)}")

    # Split IS/OOS
    IS = meta[(meta['date'] >= IS_START) & (meta['date'] <= IS_END)].copy()
    OOS = meta[(meta['date'] >= OOS_START) & (meta['date'] <= OOS_END)].copy()
    log(f"  IS:  {len(IS)} events ({IS_START} to {IS_END})")
    log(f"  OOS: {len(OOS)} events ({OOS_START} to {OOS_END})")

    # SPY returns
    log("  Computing SPY open returns (spy_return_935)...")
    all_dates = set(IS['date'].tolist() + OOS['date'].tolist())
    spy_returns = build_spy_return_935(all_dates)
    valid_spy = sum(1 for v in spy_returns.values() if not np.isnan(v))
    log(f"  SPY returns: {valid_spy}/{len(spy_returns)} dates with valid data")

    # 1-sec index
    log("  Building 1-sec data index...")
    resolver = IntrabarResolver(use_1sec=True)
    n_sec = resolver.build_index()
    log(f"  1-sec files available: {n_sec}")

    # Simulate function wrapper (uses streaming trail)
    def simulate_fn(bars_rth, entry_price, sl_dist, trade_dir, adr,
                    entry_time='09:36', ticker=None, date=None):
        return simulate_trail_d_streaming(
            bars_rth, entry_price, sl_dist, trade_dir, adr,
            entry_time=entry_time, ticker=ticker, date=date,
            use_1sec=True
        )

    # ============================================================
    # PHASE 1: CATEGORY 1 — 9 CLOSE_935 STRATEGIES
    # ============================================================
    log_header("PHASE 1: CATEGORY 1 — CLOSE_935 STRATEGIES (9)")

    cat1_results = {}
    for sample_label, sample_df in [('IS', IS), ('OOS', OOS)]:
        log(f"\n  --- {sample_label} ---")
        results = run_cat1_strategies(sample_df, spy_returns, simulate_fn, label=sample_label)
        for strat_name, trade_list in results.items():
            key = (strat_name, sample_label)
            pnl_list = [t['pnl_r'] for t in trade_list]
            cat1_results[key] = {
                'trades': trade_list,
                'pnl': pnl_list,
                'stats': compute_stats(pnl_list, f"{strat_name} ({sample_label})"),
            }
            log(fmt_stats(cat1_results[key]['stats']))

    # Save Cat 1 results
    cat1_lines = ["D14 BIAS-FREE — CATEGORY 1 RESULTS (CLOSE_935 STRATEGIES)", "=" * 80, ""]
    for key, data in sorted(cat1_results.items()):
        cat1_lines.append(fmt_stats(data['stats']))
    cat1_text = '\n'.join(cat1_lines)
    (RESULTS_DIR / 'd14_cat1_results.txt').write_text(cat1_text, encoding='utf-8')
    log(f"\n  Cat 1 results saved to {RESULTS_DIR / 'd14_cat1_results.txt'}")

    # ============================================================
    # PHASE 2: CATEGORY 2 — 3 PATTERN-BASED STRATEGIES
    # ============================================================
    log_header("PHASE 2: CATEGORY 2 — PATTERN-BASED STRATEGIES (3)")

    cat2_results = {}
    for sample_label, sample_df in [('IS', IS), ('OOS', OOS)]:
        log(f"\n  --- {sample_label} ---")

        # 2.1 Sell-Off & Recovery
        so_trades = run_cat2_selloff(sample_df, simulate_fn, label=sample_label)
        so_pnl = [t['pnl_r'] for t in so_trades]
        cat2_results[('2.1_SellOff_Recovery', sample_label)] = {
            'trades': so_trades, 'pnl': so_pnl,
            'stats': compute_stats(so_pnl, f"2.1_SellOff_Recovery ({sample_label})")
        }
        log(fmt_stats(cat2_results[('2.1_SellOff_Recovery', sample_label)]['stats']))

        # 2.2 Failed Fade
        ff_trades = run_cat2_failed_fade(sample_df, simulate_fn, label=sample_label)
        ff_pnl = [t['pnl_r'] for t in ff_trades]
        cat2_results[('2.2_Failed_Fade', sample_label)] = {
            'trades': ff_trades, 'pnl': ff_pnl,
            'stats': compute_stats(ff_pnl, f"2.2_Failed_Fade ({sample_label})")
        }
        log(fmt_stats(cat2_results[('2.2_Failed_Fade', sample_label)]['stats']))

        # 2.3 Dip & Rip
        dr_trades = run_cat2_dip_and_rip(sample_df, simulate_fn, label=sample_label)
        dr_pnl = [t['pnl_r'] for t in dr_trades]
        cat2_results[('2.3_Dip_and_Rip', sample_label)] = {
            'trades': dr_trades, 'pnl': dr_pnl,
            'stats': compute_stats(dr_pnl, f"2.3_Dip_and_Rip ({sample_label})")
        }
        log(fmt_stats(cat2_results[('2.3_Dip_and_Rip', sample_label)]['stats']))

    # Save Cat 2 results
    cat2_lines = ["D14 BIAS-FREE — CATEGORY 2 RESULTS (PATTERN-BASED)", "=" * 80, ""]
    for key, data in sorted(cat2_results.items()):
        cat2_lines.append(fmt_stats(data['stats']))
    cat2_text = '\n'.join(cat2_lines)
    (RESULTS_DIR / 'd14_cat2_results.txt').write_text(cat2_text, encoding='utf-8')
    log(f"\n  Cat 2 results saved to {RESULTS_DIR / 'd14_cat2_results.txt'}")

    # ============================================================
    # PHASE 3: CATEGORY 3 — LATE-DAY SWING
    # ============================================================
    log_header("PHASE 3: CATEGORY 3 — LATE-DAY SWING (7 PATTERNS x 3 ENTRIES)")

    cat3_results = {}
    for sample_label, sample_df in [('IS', IS), ('OOS', OOS)]:
        log(f"\n  --- {sample_label} ---")
        pattern_results = run_cat3_lateday(sample_df, label=sample_label)

        for (pattern, et), trade_list in pattern_results.items():
            if len(trade_list) == 0:
                continue
            strat_name = f"3_{pattern}_{et}"
            pnl_list = [t['pnl_r'] for t in trade_list]
            key = (strat_name, sample_label)
            cat3_results[key] = {
                'trades': trade_list, 'pnl': pnl_list,
                'stats': compute_stats(pnl_list, f"{strat_name} ({sample_label})")
            }

        # Print only top results
        oos_keys = [(k, v) for k, v in cat3_results.items()
                    if k[1] == sample_label and v['stats']['valid']]
        oos_keys.sort(key=lambda x: x[1]['stats']['EV'], reverse=True)
        for k, v in oos_keys[:10]:
            log(fmt_stats(v['stats']))
        if len(oos_keys) > 10:
            log(f"  ... and {len(oos_keys) - 10} more pattern/time combos")

    # Save Cat 3 results
    cat3_lines = ["D14 BIAS-FREE — CATEGORY 3 RESULTS (LATE-DAY SWING)", "=" * 80, ""]
    for key, data in sorted(cat3_results.items()):
        if data['stats']['valid']:
            cat3_lines.append(fmt_stats(data['stats']))
    cat3_text = '\n'.join(cat3_lines)
    (RESULTS_DIR / 'd14_cat3_lateday_results.txt').write_text(cat3_text, encoding='utf-8')
    log(f"\n  Cat 3 results saved to {RESULTS_DIR / 'd14_cat3_lateday_results.txt'}")

    # ============================================================
    # PHASE 4: SUMMARY & COMPARISON SETUP
    # ============================================================
    log_header("PHASE 4: SUMMARY")

    all_results = {}
    all_results.update(cat1_results)
    all_results.update(cat2_results)
    all_results.update(cat3_results)

    # Collect all OOS results
    oos_summary = []
    for key, data in all_results.items():
        strat_name, sample = key
        if sample != 'OOS' or not data['stats']['valid']:
            continue
        s = data['stats']
        oos_summary.append({
            'Strategy': strat_name,
            'N': s['N'], 'EV': s['EV'], 'WR': s['WR'], 'PF': s['PF'],
            'CI_lo': s['CI_lo'], 'CI_hi': s['CI_hi'], 'p_val': s['p_val'],
            'Robust': 'YES' if (not np.isnan(s['CI_lo']) and s['CI_lo'] > 0) else 'NO',
        })

    oos_df = pd.DataFrame(oos_summary).sort_values('EV', ascending=False)

    log(f"\n  OOS Results Summary ({len(oos_df)} strategies tested):")
    log(f"  {'Strategy':<45} | {'N':>5} | {'EV':>7} | {'WR':>5} | {'CI_lo':>7} | {'Robust':>6}")
    log(f"  {'-'*90}")
    for _, row in oos_df.iterrows():
        ci_lo_str = f"{row['CI_lo']:+.3f}" if not np.isnan(row['CI_lo']) else "  N/A"
        log(f"  {row['Strategy']:<45} | {row['N']:>5} | {row['EV']:>+7.3f} | {row['WR']:>5.1%} | {ci_lo_str:>7} | {row['Robust']:>6}")

    robust_count = (oos_df['Robust'] == 'YES').sum()
    log(f"\n  ROBUST strategies (CI_lo > 0): {robust_count}/{len(oos_df)}")

    # ============================================================
    # SAVE SUMMARY
    # ============================================================
    summary_lines = [
        "D14 BIAS-FREE RE-EVALUATION — EXECUTIVE SUMMARY",
        "=" * 80,
        f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        f"IS: {IS_START} to {IS_END} ({len(IS)} events)",
        f"OOS: {OOS_START} to {OOS_END} ({len(OOS)} events)",
        "",
        "ENGINE CHANGES:",
        "  1. Streaming TRAIL_D: SL check BEFORE peak update (conservative)",
        "  2. SPY open-return replaces spy_return_day (Strategies 1.4)",
        "  3. Bar-by-bar gap fill tracking (Strategies 2.2, Late-Day FAILED_FADE)",
        "  4. Streaming LOD/HOD/PB_Low (Strategies 2.1, 2.2, 2.3)",
        "  5. 1-sec intra-bar resolution for ambiguous bars",
        "",
        "DATA SNOOPING MITIGATION:",
        "  - ZERO new parameters (exact D13 values)",
        "  - ZERO new filters",
        "  - ZERO optimization",
        "",
        "OOS RESULTS:",
        f"  {'Strategy':<45} | {'N':>5} | {'EV':>7} | {'WR':>5} | {'CI_lo':>7} | {'Robust':>6}",
        f"  {'-'*90}",
    ]
    for _, row in oos_df.iterrows():
        ci_lo_str = f"{row['CI_lo']:+.3f}" if not np.isnan(row['CI_lo']) else "  N/A"
        summary_lines.append(
            f"  {row['Strategy']:<45} | {row['N']:>5} | {row['EV']:>+7.3f} | {row['WR']:>5.1%} | {ci_lo_str:>7} | {row['Robust']:>6}"
        )
    summary_lines.append(f"\n  ROBUST strategies: {robust_count}/{len(oos_df)}")

    elapsed = time.time() - t_start
    summary_lines.append(f"\n  Total runtime: {elapsed/60:.1f} minutes")

    summary_text = '\n'.join(summary_lines)
    (RESULTS_DIR / 'd14_summary.txt').write_text(summary_text, encoding='utf-8')
    log(f"\n  Summary saved to {RESULTS_DIR / 'd14_summary.txt'}")

    # Save full log
    full_log = '\n'.join(lines)
    (RESULTS_DIR / 'd14_full_log.txt').write_text(full_log, encoding='utf-8')

    # Save trade-level data for comparison
    all_trades = []
    for key, data in all_results.items():
        strat_name, sample = key
        for t in data['trades']:
            rec = {
                'strategy': strat_name, 'sample': sample,
                'ticker': t.get('ticker', ''), 'date': t.get('date', ''),
                'pnl_r': t.get('pnl_r', np.nan),
                'exit_type': t.get('exit_type', t.get('exit_reason', '')),
                'winner': t.get('winner', t.get('pnl_r', 0) > 0),
            }
            all_trades.append(rec)
    trades_df = pd.DataFrame(all_trades)
    trades_df.to_parquet(RESULTS_DIR / 'd14_all_trades.parquet', index=False)
    log(f"  Trade data saved: {len(trades_df)} records")

    elapsed = time.time() - t_start
    log(f"\n  TOTAL RUNTIME: {elapsed/60:.1f} minutes")
    log("  DONE.")


if __name__ == '__main__':
    main()
