"""
D14.5 — Recompute Cat-3 Trades after Gap-Through Fix
=====================================================
Only re-runs Category 3 (late-day swing) with the fixed
simulate_multiday_trail_d_streaming that accounts for gap-through slippage.
Then merges with existing Cat-1/Cat-2 trades.
"""
import pandas as pd
import numpy as np
import sys
import time
import warnings
from pathlib import Path
from scipy import stats as sp_stats

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from d14_bias_free.strategies.cat3_lateday_swing import run_cat3_lateday

META_PATH = Path('data/metadata/metadata_v9.parquet')
RESULTS_DIR = Path('d14_bias_free/results')
TRADES_PATH = RESULTS_DIR / 'd14_all_trades.parquet'

IS_START = '2021-02-21'
IS_END = '2023-12-31'
OOS_START = '2024-01-01'
OOS_END = '2026-02-21'


def main():
    t0 = time.time()
    print("=" * 70)
    print("  D14.5 — RECOMPUTE CAT-3 (GAP-THROUGH FIX)")
    print("=" * 70)

    # Load existing trades
    old = pd.read_parquet(TRADES_PATH)
    non_cat3 = old[~old['strategy'].str.startswith('3_')].copy()
    old_cat3 = old[old['strategy'].str.startswith('3_')]
    print(f"  Existing trades: {len(old)} total, {len(non_cat3)} non-Cat3, {len(old_cat3)} Cat3")

    # Load metadata
    meta = pd.read_parquet(META_PATH)
    meta['date'] = meta['date'].astype(str)
    IS = meta[(meta['date'] >= IS_START) & (meta['date'] <= IS_END)].copy()
    OOS = meta[(meta['date'] >= OOS_START) & (meta['date'] <= OOS_END)].copy()
    print(f"  IS: {len(IS)} events, OOS: {len(OOS)} events")

    # Re-run Cat 3
    new_cat3_rows = []
    for sample_label, sample_df in [('IS', IS), ('OOS', OOS)]:
        print(f"\n  --- Re-running Cat 3: {sample_label} ---")
        pattern_results = run_cat3_lateday(sample_df, label=sample_label)

        for (pattern, et), trade_list in pattern_results.items():
            strat_name = f"3_{pattern}_{et}"
            for t in trade_list:
                new_cat3_rows.append({
                    'strategy': strat_name,
                    'sample': sample_label,
                    'ticker': t.get('ticker', ''),
                    'date': t.get('date', ''),
                    'pnl_r': t.get('pnl_r', np.nan),
                    'exit_type': t.get('exit_reason', ''),
                    'winner': t.get('pnl_r', 0) > 0,
                })

    new_cat3 = pd.DataFrame(new_cat3_rows)
    print(f"\n  New Cat-3 trades: {len(new_cat3)}")

    # Compare old vs new
    if len(new_cat3) > 0:
        old_sl = old_cat3[old_cat3['exit_type'] == 'SL_hit']
        new_sl = new_cat3[new_cat3['exit_type'] == 'SL_hit']
        print(f"  Old SL_hit trades: {len(old_sl)}, mean pnl_r: {old_sl['pnl_r'].mean():.4f}")
        print(f"  New SL_hit trades: {len(new_sl)}, mean pnl_r: {new_sl['pnl_r'].mean():.4f}")

        # Detailed gap-through impact
        old_mean = old_cat3.groupby('strategy')['pnl_r'].mean()
        new_mean = new_cat3.groupby('strategy')['pnl_r'].mean()
        shared = old_mean.index.intersection(new_mean.index)
        if len(shared) > 0:
            diff = new_mean[shared] - old_mean[shared]
            print(f"\n  Gap-Through Impact (EV change per strategy):")
            for s in diff.sort_values().index:
                print(f"    {s:<35} old={old_mean[s]:+.4f}  new={new_mean[s]:+.4f}  delta={diff[s]:+.4f}")

    # Merge
    merged = pd.concat([non_cat3, new_cat3], ignore_index=True)
    merged.to_parquet(TRADES_PATH, index=False)
    print(f"\n  Saved {len(merged)} trades to {TRADES_PATH}")

    elapsed = time.time() - t0
    print(f"  Runtime: {elapsed/60:.1f} minutes")
    print("  DONE.")


if __name__ == '__main__':
    main()
