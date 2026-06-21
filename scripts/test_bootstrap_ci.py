###############################################################################
# TEST 4: Bootstrap Confidence Intervals for Top-5 Setups
#
# For each top-5 setup: draw 10,000 bootstrap samples (with replacement)
# Calculate 95% CI for AvgPnL and WR
# Check if CI contains zero
###############################################################################

import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from tqdm import tqdm

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE_DIR = str(PROJECT_ROOT)
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

N_BOOTSTRAP = 10000

# ============================================================
# Load data
# ============================================================
print("Loading trades...", file=sys.stderr)
trades_all = pd.read_parquet(os.path.join(RESULTS_DIR, 'all_trades_v3.parquet'))
val = trades_all[trades_all['split'] == 'val'].copy()
val['label'] = (val['outcome'] == 'target_hit').astype(int)
print(f"Val trades: {len(val)}", file=sys.stderr)

# Load setup rankings
rankings = pd.read_csv(os.path.join(RESULTS_DIR, 'setup_rankings_v3_val.csv'))
top5 = rankings.head(5)

# ============================================================
# Bootstrap function
# ============================================================
def bootstrap_ci(values, n_boot=10000, alpha=0.05, seed=42):
    """Calculate bootstrap confidence interval for the mean"""
    rng = np.random.RandomState(seed)
    n = len(values)
    boot_means = np.empty(n_boot)

    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boot_means[i] = np.mean(sample)

    lower = np.percentile(boot_means, 100 * alpha / 2)
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    median = np.percentile(boot_means, 50)

    return lower, median, upper, boot_means

# ============================================================
# Run bootstrap for each top-5 setup
# ============================================================
output_lines = []
output_lines.append("=" * 80)
output_lines.append("TEST 4: BOOTSTRAP CONFIDENCE INTERVALS FOR TOP-5 SETUPS")
output_lines.append("=" * 80)
output_lines.append(f"")
output_lines.append(f"Method: 10,000 bootstrap resamples (with replacement)")
output_lines.append(f"CI level: 95% (2.5th to 97.5th percentile)")
output_lines.append(f"")

output_lines.append("=" * 80)
output_lines.append("A) AvgPnL BOOTSTRAP CONFIDENCE INTERVALS")
output_lines.append("=" * 80)
output_lines.append(f"")

all_results = []

for idx, row in top5.iterrows():
    scenario = row['scenario']
    entry = row['entry']
    sl = row['sl']
    tgt = row['target']
    setup_name = f"{scenario} {entry} {sl} {tgt}"

    mask = (
        (val['scenario'] == scenario) &
        (val['entry_type'] == entry) &
        (val['sl_method'] == sl) &
        (val['target_method'] == tgt)
    )
    sub = val[mask]
    n = len(sub)

    if n < 10:
        continue

    pnl_values = sub['pnl_adr'].values
    wr_values = sub['label'].values.astype(float)

    print(f"Bootstrap #{idx+1}: {setup_name} (N={n})...", file=sys.stderr)

    # PnL CI
    pnl_lo, pnl_med, pnl_hi, pnl_boots = bootstrap_ci(pnl_values, N_BOOTSTRAP, seed=42+idx)

    # WR CI
    wr_lo, wr_med, wr_hi, wr_boots = bootstrap_ci(wr_values, N_BOOTSTRAP, seed=1000+idx)

    # Check if CI contains zero
    pnl_contains_zero = pnl_lo <= 0 <= pnl_hi

    # P-value: proportion of bootstrap samples with mean <= 0
    p_value = np.mean(pnl_boots <= 0)

    result = {
        'rank': idx + 1,
        'setup': setup_name,
        'n': n,
        'point_pnl': row['avg_pnl_val'],
        'pnl_lo': pnl_lo,
        'pnl_med': pnl_med,
        'pnl_hi': pnl_hi,
        'pnl_contains_zero': pnl_contains_zero,
        'p_value': p_value,
        'point_wr': row['win_rate_val'],
        'wr_lo': wr_lo,
        'wr_med': wr_med,
        'wr_hi': wr_hi,
        'pnl_std': np.std(pnl_boots),
    }
    all_results.append(result)

    output_lines.append(f"--- #{idx+1}: {setup_name} ---")
    output_lines.append(f"  N = {n}")
    output_lines.append(f"  AvgPnL:")
    output_lines.append(f"    Point estimate:  {row['avg_pnl_val']:+.4f} ADR")
    output_lines.append(f"    95% CI:          [{pnl_lo:+.4f}, {pnl_hi:+.4f}]")
    output_lines.append(f"    Bootstrap median: {pnl_med:+.4f}")
    output_lines.append(f"    CI width:         {pnl_hi - pnl_lo:.4f}")
    output_lines.append(f"    Contains zero:    {'YES - NOT SIGNIFICANT' if pnl_contains_zero else 'NO - SIGNIFICANT'}")
    output_lines.append(f"    P-value (PnL<=0): {p_value:.4f}")
    output_lines.append(f"  Win Rate:")
    output_lines.append(f"    Point estimate:  {row['win_rate_val']:.1%}")
    output_lines.append(f"    95% CI:          [{wr_lo:.1%}, {wr_hi:.1%}]")
    output_lines.append(f"")

# ============================================================
# B) Summary Table
# ============================================================
output_lines.append("=" * 80)
output_lines.append("B) SUMMARY TABLE")
output_lines.append("=" * 80)
output_lines.append(f"")

header = f"{'#':>2s} | {'N':>5s} | {'AvgPnL':>7s} | {'CI Low':>7s} | {'CI High':>7s} | {'Width':>7s} | {'Zero?':>6s} | {'P-val':>6s} | {'WR':>5s} | {'WR CI':>15s}"
output_lines.append(header)
output_lines.append("-" * len(header))

for r in all_results:
    zero_flag = "YES" if r['pnl_contains_zero'] else "NO"
    output_lines.append(
        f"{r['rank']:>2d} | {r['n']:>5d} | {r['point_pnl']:>+6.4f} | {r['pnl_lo']:>+6.4f} | {r['pnl_hi']:>+6.4f} | "
        f"{r['pnl_hi'] - r['pnl_lo']:>6.4f} | {zero_flag:>6s} | {r['p_value']:>5.3f} | {r['point_wr']:>4.1%} | "
        f"[{r['wr_lo']:.1%}, {r['wr_hi']:.1%}]"
    )

output_lines.append(f"")

# ============================================================
# C) Additional: MedPnL Bootstrap for robustness check
# ============================================================
output_lines.append("=" * 80)
output_lines.append("C) MEDIAN PnL BOOTSTRAP (robustness check)")
output_lines.append("=" * 80)
output_lines.append(f"")

for idx, row in top5.iterrows():
    scenario = row['scenario']
    entry = row['entry']
    sl = row['sl']
    tgt = row['target']
    setup_name = f"{scenario} {entry} {sl} {tgt}"

    mask = (
        (val['scenario'] == scenario) &
        (val['entry_type'] == entry) &
        (val['sl_method'] == sl) &
        (val['target_method'] == tgt)
    )
    sub = val[mask]
    pnl_values = sub['pnl_adr'].values

    rng = np.random.RandomState(2000 + idx)
    boot_medians = np.empty(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        sample = rng.choice(pnl_values, size=len(pnl_values), replace=True)
        boot_medians[i] = np.median(sample)

    med_lo = np.percentile(boot_medians, 2.5)
    med_hi = np.percentile(boot_medians, 97.5)
    med_point = np.median(pnl_values)
    med_contains_zero = med_lo <= 0 <= med_hi

    output_lines.append(f"  #{idx+1} {setup_name}")
    output_lines.append(f"    MedPnL = {med_point:+.4f}, 95% CI = [{med_lo:+.4f}, {med_hi:+.4f}]")
    output_lines.append(f"    Contains zero: {'YES' if med_contains_zero else 'NO'}")
    output_lines.append(f"    Interpretation: {'Median trade is profitable' if med_point > 0 else 'Median trade LOSES money'}")
    output_lines.append(f"")

# ============================================================
# D) Verdict
# ============================================================
output_lines.append("=" * 80)
output_lines.append("D) VERDICT")
output_lines.append("=" * 80)
output_lines.append(f"")

n_significant = sum(1 for r in all_results if not r['pnl_contains_zero'])
output_lines.append(f"  {n_significant}/{len(all_results)} Top-5 setups have 95% CI that does NOT contain zero")
output_lines.append(f"")

for r in all_results:
    if r['pnl_contains_zero']:
        output_lines.append(f"  WARNING: #{r['rank']} ({r['setup']})")
        output_lines.append(f"    CI contains zero -> edge is NOT statistically significant at 95%")
        output_lines.append(f"    N={r['n']}, P-value={r['p_value']:.3f}")
    else:
        output_lines.append(f"  OK: #{r['rank']} ({r['setup']})")
        output_lines.append(f"    CI does NOT contain zero -> edge is statistically significant")
        output_lines.append(f"    N={r['n']}, P-value={r['p_value']:.4f}")

output_lines.append(f"")
output_lines.append("NOTE: Statistical significance does NOT imply economic significance.")
output_lines.append("Even a significant edge can be erased by slippage (see Test 2).")
output_lines.append("Multiple testing correction (Bonferroni for 288 combos) would require")
output_lines.append("p < 0.000174 for significance — a much higher bar.")

# Write to file
output_path = os.path.join(RESULTS_DIR, 'test_bootstrap_ci.txt')
with open(output_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(output_lines))

print(f"\nResults written to {output_path}", file=sys.stderr)
print("Done!", file=sys.stderr)
