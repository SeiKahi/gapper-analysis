###############################################################################
# TEST 2: SL-Slippage Stress Test
#
# For each trade with outcome='sl_hit': add slippage to loss
# For each trade with outcome='target_hit': subtract slippage from gain
# Recalculate PnL and check when edge dies
###############################################################################

import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import numpy as np

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE_DIR = str(PROJECT_ROOT)
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# ============================================================
# Load data
# ============================================================
print("Loading trades...", file=sys.stderr)
trades_all = pd.read_parquet(os.path.join(RESULTS_DIR, 'all_trades_v3.parquet'))
val = trades_all[trades_all['split'] == 'val'].copy()
print(f"Val trades: {len(val)}", file=sys.stderr)

# Load top-10 setups
print("Loading setup rankings...", file=sys.stderr)
rankings = pd.read_csv(os.path.join(RESULTS_DIR, 'setup_rankings_v3_val.csv'))
top10 = rankings.head(10)
print(f"Top 10 setups loaded", file=sys.stderr)

# Also get Top-25 for extended analysis
top25 = rankings.head(25)

# ============================================================
# Slippage levels to test
# ============================================================
SLIPPAGE_LEVELS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.10]

# Slippage for SL hits (added to loss)
SL_SLIPPAGE = [0.01, 0.02, 0.05, 0.10]
# Slippage for Target hits (subtracted from gain - fill slippage)
TGT_SLIPPAGE = [0.01, 0.02]

output_lines = []
output_lines.append("=" * 80)
output_lines.append("TEST 2: SL-SLIPPAGE STRESS TEST")
output_lines.append("=" * 80)
output_lines.append(f"")
output_lines.append(f"Val trades: {len(val)}")
output_lines.append(f"Method: For SL-hit trades, ADD slippage to loss (in ADR units)")
output_lines.append(f"         For Target-hit trades, SUBTRACT fill-slippage from gain")
output_lines.append(f"         Timeout trades: unchanged")
output_lines.append(f"")

# ============================================================
# A) Top-10 Setups at each slippage level
# ============================================================
output_lines.append("=" * 80)
output_lines.append("A) TOP-10 SETUPS: AvgPnL AT EACH SLIPPAGE LEVEL")
output_lines.append("=" * 80)
output_lines.append(f"")

# Header
header = f"{'#':>2s} | {'Scenario':>20s} | {'Entry':>5s} | {'SL':>10s} | {'Target':>8s} | {'N':>5s} | {'WR':>5s}"
for slip in SLIPPAGE_LEVELS:
    header += f" | {f'Slip={slip:.2f}':>10s}"
output_lines.append(header)
output_lines.append("-" * len(header))

for idx, row in top10.iterrows():
    scenario = row['scenario']
    entry = row['entry']
    sl = row['sl']
    tgt = row['target']
    n_val = row['n_val']
    wr = row['win_rate_val']

    mask = (
        (val['scenario'] == scenario) &
        (val['entry_type'] == entry) &
        (val['sl_method'] == sl) &
        (val['target_method'] == tgt)
    )
    sub = val[mask].copy()

    line = f"{idx+1:>2d} | {scenario:>20s} | {entry:>5s} | {sl:>10s} | {tgt:>8s} | {len(sub):>5d} | {wr:>4.1%}"

    for total_slip in SLIPPAGE_LEVELS:
        # Apply slippage
        adjusted_pnl = sub['pnl_adr'].copy()

        # SL-hit trades: loss increases by slippage
        sl_mask = sub['outcome'] == 'sl_hit'
        adjusted_pnl.loc[sl_mask] -= total_slip

        # Target-hit trades: gain decreases by min of total_slip and TGT slippage cap
        tgt_mask = sub['outcome'] == 'target_hit'
        tgt_slip = min(total_slip, 0.02)  # Target fill slippage capped at 0.02
        adjusted_pnl.loc[tgt_mask] -= tgt_slip

        avg_pnl = adjusted_pnl.mean()
        line += f" | {avg_pnl:>+9.4f}"

    output_lines.append(line)

output_lines.append(f"")

# ============================================================
# B) Detailed: When does each setup's edge die?
# ============================================================
output_lines.append("=" * 80)
output_lines.append("B) EDGE DEATH ANALYSIS: At what slippage does PnL turn negative?")
output_lines.append("=" * 80)
output_lines.append(f"")

fine_slippage = np.arange(0.0, 0.15, 0.005)

for idx, row in top10.iterrows():
    scenario = row['scenario']
    entry = row['entry']
    sl = row['sl']
    tgt = row['target']

    mask = (
        (val['scenario'] == scenario) &
        (val['entry_type'] == entry) &
        (val['sl_method'] == sl) &
        (val['target_method'] == tgt)
    )
    sub = val[mask].copy()

    death_slip = None
    for total_slip in fine_slippage:
        adjusted_pnl = sub['pnl_adr'].copy()
        sl_mask = sub['outcome'] == 'sl_hit'
        adjusted_pnl.loc[sl_mask] -= total_slip
        tgt_mask = sub['outcome'] == 'target_hit'
        tgt_slip = min(total_slip, 0.02)
        adjusted_pnl.loc[tgt_mask] -= tgt_slip

        if adjusted_pnl.mean() <= 0:
            death_slip = total_slip
            break

    setup_name = f"{scenario} {entry} {sl} {tgt}"
    original_pnl = row['avg_pnl_val']
    sl_pct = (sub['outcome'] == 'sl_hit').mean() * 100

    if death_slip is not None:
        output_lines.append(f"  #{idx+1:2d} {setup_name:50s} | Orig={original_pnl:+.4f} | SL%={sl_pct:5.1f}% | DIES at slip={death_slip:.3f} ADR")
    else:
        output_lines.append(f"  #{idx+1:2d} {setup_name:50s} | Orig={original_pnl:+.4f} | SL%={sl_pct:5.1f}% | SURVIVES all tested slippage")

output_lines.append(f"")

# ============================================================
# C) Extended: Top-25 at 0.03 ADR slippage (realistic)
# ============================================================
output_lines.append("=" * 80)
output_lines.append("C) TOP-25 AT REALISTIC SLIPPAGE (0.03 ADR on SL, 0.02 on Target)")
output_lines.append("=" * 80)
output_lines.append(f"")

realistic_results = []
for idx, row in top25.iterrows():
    scenario = row['scenario']
    entry = row['entry']
    sl = row['sl']
    tgt = row['target']

    mask = (
        (val['scenario'] == scenario) &
        (val['entry_type'] == entry) &
        (val['sl_method'] == sl) &
        (val['target_method'] == tgt)
    )
    sub = val[mask].copy()

    adjusted_pnl = sub['pnl_adr'].copy()
    sl_mask = sub['outcome'] == 'sl_hit'
    adjusted_pnl.loc[sl_mask] -= 0.03
    tgt_mask = sub['outcome'] == 'target_hit'
    adjusted_pnl.loc[tgt_mask] -= 0.02

    new_avg = adjusted_pnl.mean()
    realistic_results.append({
        'rank': idx + 1,
        'setup': f"{scenario} {entry} {sl} {tgt}",
        'n': len(sub),
        'orig_pnl': row['avg_pnl_val'],
        'adj_pnl': new_avg,
        'pnl_change': new_avg - row['avg_pnl_val'],
        'still_positive': new_avg > 0,
    })

# Sort by adjusted pnl
realistic_results.sort(key=lambda x: -x['adj_pnl'])

output_lines.append(f"{'Orig#':>5s} | {'Setup':>55s} | {'N':>5s} | {'OrigPnL':>8s} | {'AdjPnL':>8s} | {'Status':>12s}")
output_lines.append("-" * 120)

n_survive = 0
for r in realistic_results:
    status = "SURVIVES" if r['still_positive'] else "DEAD"
    if r['still_positive']:
        n_survive += 1
    output_lines.append(f"  {r['rank']:>3d} | {r['setup']:>55s} | {r['n']:>5d} | {r['orig_pnl']:>+7.4f} | {r['adj_pnl']:>+7.4f} | {status:>12s}")

output_lines.append(f"")
output_lines.append(f"Setups surviving realistic slippage: {n_survive}/{len(realistic_results)}")

# ============================================================
# D) Sensitivity: How much does the best scenario aggregate change?
# ============================================================
output_lines.append(f"")
output_lines.append("=" * 80)
output_lines.append("D) SCENARIO-LEVEL SLIPPAGE SENSITIVITY (SL5_035 only)")
output_lines.append("=" * 80)
output_lines.append(f"")

scenarios = sorted(val['scenario'].unique())
for scenario in scenarios:
    s_mask = (val['scenario'] == scenario) & (val['sl_method'] == 'SL5_035')
    s_sub = val[s_mask]
    if len(s_sub) < 100:
        continue

    line = f"  {scenario:20s} (N={len(s_sub):5d}): "
    parts = []
    for slip in [0.0, 0.02, 0.05, 0.10]:
        adj = s_sub['pnl_adr'].copy()
        sl_m = s_sub['outcome'] == 'sl_hit'
        adj.loc[sl_m] -= slip
        tgt_m = s_sub['outcome'] == 'target_hit'
        adj.loc[tgt_m] -= min(slip, 0.02)
        parts.append(f"slip={slip:.2f}→{adj.mean():+.4f}")
    line += " | ".join(parts)
    output_lines.append(line)

output_lines.append(f"")

# ============================================================
# E) Summary
# ============================================================
output_lines.append("=" * 80)
output_lines.append("E) SUMMARY")
output_lines.append("=" * 80)
output_lines.append(f"")

# Count setups that survive at each level
for slip in SLIPPAGE_LEVELS[1:]:
    n_alive = 0
    for idx, row in top10.iterrows():
        mask = (
            (val['scenario'] == row['scenario']) &
            (val['entry_type'] == row['entry']) &
            (val['sl_method'] == row['sl']) &
            (val['target_method'] == row['target'])
        )
        sub = val[mask].copy()
        adj = sub['pnl_adr'].copy()
        sl_m = sub['outcome'] == 'sl_hit'
        adj.loc[sl_m] -= slip
        tgt_m = sub['outcome'] == 'target_hit'
        adj.loc[tgt_m] -= min(slip, 0.02)
        if adj.mean() > 0:
            n_alive += 1
    output_lines.append(f"  At slippage={slip:.2f} ADR: {n_alive}/10 Top setups still profitable")

output_lines.append(f"")
output_lines.append("INTERPRETATION:")
output_lines.append("  0.01 ADR slippage = ~$0.20 on a $50 stock with 5% ADR (minimal)")
output_lines.append("  0.02 ADR slippage = ~$0.40 (realistic for limit orders)")
output_lines.append("  0.05 ADR slippage = ~$1.00 (realistic for market orders on gappers)")
output_lines.append("  0.10 ADR slippage = ~$2.00 (worst case, thin book)")

# Write to file
output_path = os.path.join(RESULTS_DIR, 'test_slippage_stress.txt')
with open(output_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(output_lines))

print(f"\nResults written to {output_path}", file=sys.stderr)
print("Done!", file=sys.stderr)
