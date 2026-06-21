###############################################################################
# TEST 1: Random Entry Baseline
#
# For each unique (ticker, date) in Val set: pick 3 random bars as entry
# Apply same SL/Target logic (SL5_035, T4_3R and T4_2R)
# Bar-by-bar simulation using VWAP files
# Compare: Random WR + PnL vs real pipeline WR + PnL
###############################################################################

import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import glob
from tqdm import tqdm

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE_DIR = str(PROJECT_ROOT)
VWAP_DIR = os.path.join(BASE_DIR, 'data', 'vwap')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
METADATA_PATH = os.path.join(BASE_DIR, 'data', 'metadata', 'metadata_master.parquet')

# ============================================================
# Load metadata and filter to val period
# ============================================================
print("Loading metadata...", file=sys.stderr)
meta = pd.read_parquet(METADATA_PATH)
meta = meta[(meta['date'] >= '2021-02-21') & (meta['date'] <= '2023-12-31')]
val_meta = meta[(meta['date'] > '2022-12-31') & (meta['date'] <= '2023-12-31')].copy()
print(f"Val metadata: {len(val_meta)} rows", file=sys.stderr)

# ============================================================
# Load real pipeline trades (val only) for comparison
# ============================================================
print("Loading all_trades_v3...", file=sys.stderr)
trades_all = pd.read_parquet(os.path.join(RESULTS_DIR, 'all_trades_v3.parquet'))
val_trades = trades_all[trades_all['split'] == 'val'].copy()
print(f"Val trades: {len(val_trades)}", file=sys.stderr)

# ============================================================
# Sample 500 random ticker-dates from val
# ============================================================
np.random.seed(42)
val_meta_keys = val_meta[['ticker', 'date']].drop_duplicates()
if len(val_meta_keys) > 500:
    sample_idx = np.random.choice(len(val_meta_keys), 500, replace=False)
    val_sample = val_meta_keys.iloc[sample_idx]
else:
    val_sample = val_meta_keys
print(f"Sampled {len(val_sample)} ticker-dates for random baseline", file=sys.stderr)

# Create lookup for adr_10 and gap_direction
meta_lookup = {}
for _, row in val_meta.iterrows():
    key = f"{row['ticker']}_{row['date']}"
    meta_lookup[key] = {
        'adr_10': row.get('adr_10', np.nan),
        'gap_direction': row.get('gap_direction', ''),
        'gap_pct': abs(row.get('gap_pct', 0)) if not pd.isna(row.get('gap_pct', 0)) else 0,
    }

# ============================================================
# Trade simulation function (same logic as pipeline)
# ============================================================
def simulate_trade(closes, e_bar, e_price, sl_dist, tgt_mult, is_short, adr_10, max_bars=240):
    """Simulate a single trade bar-by-bar, same logic as ml_pipeline_v3"""
    n = len(closes)

    if is_short:
        sl_price = e_price + sl_dist
        tgt_price = e_price - tgt_mult * sl_dist
    else:
        sl_price = e_price - sl_dist
        tgt_price = e_price + tgt_mult * sl_dist

    outcome = 'timeout'
    exit_price = closes[min(e_bar + max_bars, n - 1)]

    for j in range(e_bar + 1, min(n, e_bar + max_bars + 1)):
        # Check SL (close-based)
        sl_hit = False
        if is_short and closes[j] >= sl_price:
            sl_hit = True
        elif not is_short and closes[j] <= sl_price:
            sl_hit = True

        # Check Target (close-based)
        tgt_hit = False
        if is_short and closes[j] <= tgt_price:
            tgt_hit = True
        elif not is_short and closes[j] >= tgt_price:
            tgt_hit = True

        if sl_hit and tgt_hit:
            outcome = 'sl_hit'
            exit_price = sl_price
            break
        elif tgt_hit:
            outcome = 'target_hit'
            exit_price = tgt_price
            break
        elif sl_hit:
            outcome = 'sl_hit'
            exit_price = sl_price
            break

    if is_short:
        pnl_adr = (e_price - exit_price) / adr_10
    else:
        pnl_adr = (exit_price - e_price) / adr_10

    return outcome, pnl_adr

# ============================================================
# Run random baseline
# ============================================================
N_RANDOM_PER_DAY = 3
SL_CONFIGS = {'SL5_035': 0.35}  # SL5 = 0.35 * ADR
TGT_CONFIGS = {'T4_3R': 3.0, 'T4_2R': 2.0}

# Determine trade direction based on gap_direction and scenario type
# GapDn scenarios -> long, GapUp -> short
# For random baseline: use gap_direction to determine trade direction

random_results = []
processed = 0
errors = 0

rng = np.random.RandomState(42)

for _, row in tqdm(val_sample.iterrows(), total=len(val_sample), desc="Random baseline", file=sys.stderr):
    ticker = row['ticker']
    date = row['date']
    key = f"{ticker}_{date}"

    info = meta_lookup.get(key)
    if info is None:
        errors += 1
        continue

    adr_10 = info['adr_10']
    if pd.isna(adr_10) or adr_10 <= 0:
        errors += 1
        continue

    gap_dir = info['gap_direction']

    # Determine trade direction: GapDn -> long (fade), GapUp -> short (fade)
    is_short = (gap_dir == 'up')

    # Load VWAP file
    vwap_path = os.path.join(VWAP_DIR, ticker, f"{date}.parquet")
    if not os.path.exists(vwap_path):
        errors += 1
        continue

    try:
        df = pd.read_parquet(vwap_path)
    except Exception:
        errors += 1
        continue

    if len(df) < 50:
        errors += 1
        continue

    closes = df['close'].values
    n = len(closes)

    # Find market start
    times = df['time_et'].values
    market_start_idx = 0
    for i in range(n):
        if times[i] >= '09:30':
            market_start_idx = i
            break

    # Random entry bars: between bar 20 and bar 200 (relative to market start)
    min_bar = market_start_idx + 20
    max_bar = min(market_start_idx + 200, n - 50)

    if max_bar <= min_bar:
        errors += 1
        continue

    random_bars = rng.randint(min_bar, max_bar, size=N_RANDOM_PER_DAY)

    for e_bar in random_bars:
        e_price = closes[e_bar]

        for sl_name, sl_mult in SL_CONFIGS.items():
            sl_dist = sl_mult * adr_10

            for tgt_name, tgt_mult in TGT_CONFIGS.items():
                outcome, pnl_adr = simulate_trade(
                    closes, e_bar, e_price, sl_dist, tgt_mult, is_short, adr_10
                )

                random_results.append({
                    'ticker': ticker,
                    'date': date,
                    'gap_direction': gap_dir,
                    'entry_bar': e_bar,
                    'sl_method': sl_name,
                    'target_method': tgt_name,
                    'outcome': outcome,
                    'pnl_adr': round(pnl_adr, 4),
                    'is_short': is_short,
                })

    processed += 1

print(f"\nProcessed: {processed}, Errors: {errors}", file=sys.stderr)

# ============================================================
# Compute random baseline stats
# ============================================================
df_random = pd.DataFrame(random_results)

# ============================================================
# Compute real pipeline stats on the SAME ticker-dates
# ============================================================
sample_keys = set(val_sample['ticker'] + '_' + val_sample['date'])
val_trades['key'] = val_trades['ticker'] + '_' + val_trades['date']
real_subset = val_trades[val_trades['key'].isin(sample_keys)].copy()

# ============================================================
# Write results
# ============================================================
output_lines = []
output_lines.append("=" * 80)
output_lines.append("TEST 1: RANDOM ENTRY BASELINE")
output_lines.append("=" * 80)
output_lines.append(f"")
output_lines.append(f"Sampled {len(val_sample)} ticker-dates from Val period (2023)")
output_lines.append(f"Random entries per day: {N_RANDOM_PER_DAY}")
output_lines.append(f"Total random trades: {len(df_random)}")
output_lines.append(f"Processed: {processed}, Errors: {errors}")
output_lines.append(f"")

# Overall comparison
output_lines.append("=" * 80)
output_lines.append("A) OVERALL COMPARISON: Random vs Real Pipeline")
output_lines.append("=" * 80)
output_lines.append(f"")

for sl_name in SL_CONFIGS:
    for tgt_name in TGT_CONFIGS:
        # Random stats
        r_mask = (df_random['sl_method'] == sl_name) & (df_random['target_method'] == tgt_name)
        r_sub = df_random[r_mask]
        r_wr = (r_sub['outcome'] == 'target_hit').mean() if len(r_sub) > 0 else 0
        r_pnl = r_sub['pnl_adr'].mean() if len(r_sub) > 0 else 0
        r_med = r_sub['pnl_adr'].median() if len(r_sub) > 0 else 0
        r_n = len(r_sub)

        # Real stats (same SL/Target, same ticker-dates, all scenarios)
        p_mask = (real_subset['sl_method'] == sl_name) & (real_subset['target_method'] == tgt_name)
        p_sub = real_subset[p_mask]
        p_wr = (p_sub['outcome'] == 'target_hit').mean() if len(p_sub) > 0 else 0
        p_pnl = p_sub['pnl_adr'].mean() if len(p_sub) > 0 else 0
        p_med = p_sub['pnl_adr'].median() if len(p_sub) > 0 else 0
        p_n = len(p_sub)

        output_lines.append(f"--- {sl_name} + {tgt_name} ---")
        output_lines.append(f"  Random:   N={r_n:5d}, WR={r_wr:.1%}, AvgPnL={r_pnl:+.4f}, MedPnL={r_med:+.4f}")
        output_lines.append(f"  Pipeline: N={p_n:5d}, WR={p_wr:.1%}, AvgPnL={p_pnl:+.4f}, MedPnL={p_med:+.4f}")
        output_lines.append(f"  Delta WR: {(p_wr - r_wr)*100:+.2f}pp")
        output_lines.append(f"  Delta PnL: {(p_pnl - r_pnl):+.4f} ADR")
        output_lines.append(f"")

# Breakdown by gap direction
output_lines.append("=" * 80)
output_lines.append("B) BREAKDOWN BY GAP DIRECTION")
output_lines.append("=" * 80)
output_lines.append(f"")

for gap_dir in ['down', 'up']:
    output_lines.append(f"--- Gap {gap_dir.upper()} ---")
    for sl_name in SL_CONFIGS:
        for tgt_name in TGT_CONFIGS:
            # Random
            r_mask = (df_random['sl_method'] == sl_name) & (df_random['target_method'] == tgt_name) & (df_random['gap_direction'] == gap_dir)
            r_sub = df_random[r_mask]
            r_wr = (r_sub['outcome'] == 'target_hit').mean() if len(r_sub) > 0 else 0
            r_pnl = r_sub['pnl_adr'].mean() if len(r_sub) > 0 else 0
            r_n = len(r_sub)

            # Pipeline
            p_mask = (real_subset['sl_method'] == sl_name) & (real_subset['target_method'] == tgt_name) & (real_subset['gap_direction'] == gap_dir)
            p_sub = real_subset[p_mask]
            p_wr = (p_sub['outcome'] == 'target_hit').mean() if len(p_sub) > 0 else 0
            p_pnl = p_sub['pnl_adr'].mean() if len(p_sub) > 0 else 0
            p_n = len(p_sub)

            output_lines.append(f"  {sl_name} + {tgt_name}:")
            output_lines.append(f"    Random:   N={r_n:5d}, WR={r_wr:.1%}, AvgPnL={r_pnl:+.4f}")
            output_lines.append(f"    Pipeline: N={p_n:5d}, WR={p_wr:.1%}, AvgPnL={p_pnl:+.4f}")
            output_lines.append(f"    Delta PnL: {(p_pnl - r_pnl):+.4f} ADR")
    output_lines.append(f"")

# Breakdown by scenario for pipeline
output_lines.append("=" * 80)
output_lines.append("C) PIPELINE BREAKDOWN BY SCENARIO (same ticker-dates)")
output_lines.append("=" * 80)
output_lines.append(f"")

for tgt_name in TGT_CONFIGS:
    output_lines.append(f"--- SL5_035 + {tgt_name} ---")
    for scenario in sorted(real_subset['scenario'].unique()):
        p_mask = (real_subset['sl_method'] == 'SL5_035') & (real_subset['target_method'] == tgt_name) & (real_subset['scenario'] == scenario)
        p_sub = real_subset[p_mask]
        if len(p_sub) < 10:
            continue
        p_wr = (p_sub['outcome'] == 'target_hit').mean()
        p_pnl = p_sub['pnl_adr'].mean()
        output_lines.append(f"  {scenario:20s}: N={len(p_sub):5d}, WR={p_wr:.1%}, AvgPnL={p_pnl:+.4f}")
    output_lines.append(f"")

# Outcome distribution comparison
output_lines.append("=" * 80)
output_lines.append("D) OUTCOME DISTRIBUTION")
output_lines.append("=" * 80)
output_lines.append(f"")

for tgt_name in TGT_CONFIGS:
    r_mask = (df_random['sl_method'] == 'SL5_035') & (df_random['target_method'] == tgt_name)
    r_sub = df_random[r_mask]
    p_mask = (real_subset['sl_method'] == 'SL5_035') & (real_subset['target_method'] == tgt_name)
    p_sub = real_subset[p_mask]

    output_lines.append(f"--- SL5_035 + {tgt_name} ---")
    for outcome in ['target_hit', 'sl_hit', 'timeout']:
        r_pct = (r_sub['outcome'] == outcome).mean() * 100 if len(r_sub) > 0 else 0
        p_pct = (p_sub['outcome'] == outcome).mean() * 100 if len(p_sub) > 0 else 0
        output_lines.append(f"  {outcome:12s}: Random={r_pct:5.1f}%  Pipeline={p_pct:5.1f}%")
    output_lines.append(f"")

# Summary / Verdict
output_lines.append("=" * 80)
output_lines.append("E) VERDICT")
output_lines.append("=" * 80)
output_lines.append(f"")

# Check if pipeline edge is significantly better than random
for tgt_name in TGT_CONFIGS:
    r_mask = (df_random['sl_method'] == 'SL5_035') & (df_random['target_method'] == tgt_name)
    r_sub = df_random[r_mask]
    p_mask = (real_subset['sl_method'] == 'SL5_035') & (real_subset['target_method'] == tgt_name)
    p_sub = real_subset[p_mask]

    r_pnl = r_sub['pnl_adr'].mean() if len(r_sub) > 0 else 0
    p_pnl = p_sub['pnl_adr'].mean() if len(p_sub) > 0 else 0
    delta = p_pnl - r_pnl

    if delta > 0.02:
        verdict = "PIPELINE CLEARLY BETTER than random"
    elif delta > 0.005:
        verdict = "PIPELINE SLIGHTLY BETTER than random"
    elif delta > -0.005:
        verdict = "NO MEANINGFUL DIFFERENCE — Edge may be artifact"
    else:
        verdict = "RANDOM IS BETTER — Pipeline entry adds no value"

    output_lines.append(f"SL5_035 + {tgt_name}: Delta PnL = {delta:+.4f} ADR => {verdict}")

output_lines.append(f"")
output_lines.append("NOTE: Random baseline uses same trade direction as gap (GapDn->Long, GapUp->Short)")
output_lines.append("but picks bars randomly instead of waiting for VWAP cross signal.")
output_lines.append("If random performs similarly, the 'edge' comes from the Gapper+SL/Target structure,")
output_lines.append("not from the VWAP cross timing.")

# Write to file
output_path = os.path.join(RESULTS_DIR, 'test_random_baseline.txt')
with open(output_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(output_lines))

print(f"\nResults written to {output_path}", file=sys.stderr)
print("Done!", file=sys.stderr)
