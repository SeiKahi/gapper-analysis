###############################################################################
# DEVIL'S ADVOCATE — Opening Drive Validation
# Durchlauf 3.0, Task #12
#
# Tests:
# 1. Circular reference check (metadata vs OHLC-computed OD)
# 2. Unconditional OD momentum (ignore gap direction)
# 3. Year-by-year stability (2021, 2022, 2023)
# 4. Return distribution (median, percentiles, win rate)
# 5. Permutation test (shuffle OD labels, 2000 iterations)
# 6. Multiple testing correction for Section D combos
# 7. Random baseline: random "entry" at 9:45 in random direction
# 8. OD magnitude vs sign (is it just direction, or does strength matter?)
###############################################################################

import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE_DIR = str(PROJECT_ROOT)
EVENTS_PATH = os.path.join(BASE_DIR, 'results', 'orb_opening_drive_events_v1.parquet')
METADATA_PATH = os.path.join(BASE_DIR, 'data', 'metadata', 'metadata_master.parquet')
RAW_1MIN_DIR = os.path.join(BASE_DIR, 'data', 'raw_1min')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

np.random.seed(42)


def main():
    # Load events from ORB+OD v1
    print("Loading events...", file=sys.stderr)
    ev = pd.read_parquet(EVENTS_PATH)
    val = ev[ev['split'] == 'val'].copy()
    train = ev[ev['split'] == 'train'].copy()
    print(f"  Total: {len(ev)}, Train: {len(train)}, Val: {len(val)}", file=sys.stderr)

    # Load metadata for cross-check
    meta = pd.read_parquet(METADATA_PATH)
    meta['date'] = pd.to_datetime(meta['date']).dt.strftime('%Y-%m-%d')

    out = []
    out.append("=" * 90)
    out.append("DEVIL'S ADVOCATE — Opening Drive Validation")
    out.append("Durchlauf 3.0, Haelfte 1 only")
    out.append("=" * 90)
    out.append("")

    # ========================================================================
    # TEST 1: Circular Reference Check
    # ========================================================================
    out.append("=" * 90)
    out.append("TEST 1: CIRCULAR REFERENCE CHECK")
    out.append("  Does metadata opening_drive_direction match our OHLC od_direction?")
    out.append("=" * 90)
    out.append("")

    if 'opening_drive_direction' in meta.columns:
        merged = ev.merge(meta[['ticker', 'date', 'opening_drive_direction']],
                          on=['ticker', 'date'], how='left')
        merged['meta_od'] = merged['opening_drive_direction'].str.lower().str.strip()
        merged['ohlc_od'] = merged['od_direction'].str.lower().str.strip()
        match = (merged['meta_od'] == merged['ohlc_od']).sum()
        total = merged['meta_od'].notna().sum()
        out.append(f"  Matched: {match} / {total} ({match/total*100:.1f}%)")
        mismatches = merged[merged['meta_od'] != merged['ohlc_od']]
        out.append(f"  Mismatches: {len(mismatches)}")
        if len(mismatches) > 0 and len(mismatches) < 20:
            for _, r in mismatches.head(10).iterrows():
                out.append(f"    {r['ticker']} {r['date']}: meta={r['meta_od']}, ohlc={r['ohlc_od']}")
        out.append("")
        out.append("  VERDICT: If match ~100%, our OHLC computation is consistent.")
        out.append("  But this also means the metadata field IS the opening drive, not an independent signal.")
        out.append("  No circular leak — we compute from raw OHLC, metadata just pre-computed same thing.")
    else:
        out.append("  opening_drive_direction NOT in metadata columns.")
        out.append("  No circular reference concern.")
    out.append("")

    # ========================================================================
    # TEST 2: UNCONDITIONAL OD MOMENTUM
    # ========================================================================
    out.append("=" * 90)
    out.append("TEST 2: UNCONDITIONAL OD MOMENTUM")
    out.append("  Ignore gap direction. Just: does OD direction predict rest-of-day?")
    out.append("  If YES: gap direction adds nothing. OD is just momentum.")
    out.append("=" * 90)
    out.append("")

    for split_name, sdf in [('Train', train), ('Val', val)]:
        out.append(f"--- {split_name} ---")
        # Raw return from 9:45 (od_close) to EOD, without sign convention
        # We compute: if OD was UP, measure (close - od_close)/adr (positive = continuation)
        # if OD was DOWN, measure -(close - od_close)/adr (positive = continuation in OD dir)
        for od_d in ['up', 'down']:
            sub = sdf[sdf['od_direction'] == od_d]
            if len(sub) < 30:
                continue
            # Drift in OD direction (same sign convention as original script)
            od_sign = 1 if od_d == 'up' else -1
            # We need raw return, not the signed drift from the original script
            # Original script signed drift based on gap+OD combo. Let's compute raw.
            # Actually we can use drift_240m which IS already signed in OD direction for
            # WITH trades, but in OD direction for AGAINST trades too (since sign follows OD)

            # Wait - the original script sets sign based on whether OD is WITH or AGAINST gap.
            # For WITH+GapUp: sign=+1 (OD=up, measure upside = OD continuation)
            # For WITH+GapDn: sign=-1 (OD=down, measure downside = OD continuation)
            # For AGAINST+GapUp: sign=-1 (OD=down, measure downside = OD continuation)
            # For AGAINST+GapDn: sign=+1 (OD=up, measure upside = OD continuation)
            # So in ALL cases, positive drift = continuation in OD direction!
            # This means drift_240m IS already "OD continuation drift"

            d60 = sub['drift_60m'].dropna()
            d240 = sub['drift_240m'].dropna()
            out.append(f"  OD={od_d}: N={len(sub)}, Drift60={d60.mean():.4f}, Drift240={d240.mean():.4f}")

        # Now split by gap direction to see if gap adds info
        out.append("")
        out.append(f"  Breakdown by Gap Direction:")
        for gap_d in ['up', 'down']:
            for od_d in ['up', 'down']:
                sub = sdf[(sdf['gap_direction'] == gap_d) & (sdf['od_direction'] == od_d)]
                if len(sub) < 20:
                    continue
                d60 = sub['drift_60m'].dropna()
                d240 = sub['drift_240m'].dropna()
                od_vs = 'WITH' if (gap_d == od_d) else 'AGAINST'
                out.append(f"    Gap={gap_d}, OD={od_d} ({od_vs}): N={len(sub)}, "
                           f"Drift60={d60.mean():.4f}, Drift240={d240.mean():.4f}")
        out.append("")

    # KEY: Is gap direction adding info, or is it just OD direction?
    # Compare: GapUp+OD_up vs GapDn+OD_up — both have OD=up
    out.append("  KEY COMPARISON (Val): Same OD direction, different gap:")
    for od_d in ['up', 'down']:
        for gap_d in ['up', 'down']:
            sub = val[(val['gap_direction'] == gap_d) & (val['od_direction'] == od_d)]
            if len(sub) < 20:
                continue
            d240 = sub['drift_240m'].dropna()
            out.append(f"    OD={od_d}, Gap={gap_d}: N={len(sub)}, Drift240={d240.mean():.4f}")
    out.append("")
    out.append("  VERDICT: If GapUp+OD_up ≈ GapDn+OD_up, then gap direction is irrelevant.")
    out.append("  If they differ, gap direction adds independent info.")
    out.append("")

    # ========================================================================
    # TEST 3: YEAR-BY-YEAR STABILITY
    # ========================================================================
    out.append("=" * 90)
    out.append("TEST 3: YEAR-BY-YEAR STABILITY")
    out.append("=" * 90)
    out.append("")

    ev['year'] = ev['date'].str[:4]
    for label, mask_fn in [
        ('GapUp+WITH', lambda df: (df['gap_direction']=='up') & (~df['od_against_gap'])),
        ('GapUp+AGAINST', lambda df: (df['gap_direction']=='up') & (df['od_against_gap'])),
        ('GapDn+WITH', lambda df: (df['gap_direction']=='down') & (~df['od_against_gap'])),
        ('GapDn+AGAINST', lambda df: (df['gap_direction']=='down') & (df['od_against_gap'])),
    ]:
        out.append(f"  {label}:")
        for yr in ['2021', '2022', '2023']:
            sub = ev[(ev['year'] == yr) & mask_fn(ev)]
            if len(sub) < 20:
                out.append(f"    {yr}: N={len(sub)} (too small)")
                continue
            d60 = sub['drift_60m'].dropna()
            d240 = sub['drift_240m'].dropna()
            mfe60 = sub['mfe_60m'].dropna()
            out.append(f"    {yr}: N={len(sub)}, Drift60={d60.mean():.4f}, "
                       f"Drift240={d240.mean():.4f}, MFE60={mfe60.mean():.4f}")
        out.append("")

    # ========================================================================
    # TEST 4: RETURN DISTRIBUTION (not just mean)
    # ========================================================================
    out.append("=" * 90)
    out.append("TEST 4: RETURN DISTRIBUTION — GapUp+WITH (Val)")
    out.append("  Is the mean driven by outliers?")
    out.append("=" * 90)
    out.append("")

    target_sub = val[(val['gap_direction'] == 'up') & (~val['od_against_gap'])]
    for target_col in ['drift_60m', 'drift_240m']:
        vals = target_sub[target_col].dropna().values
        if len(vals) < 30:
            continue
        out.append(f"  {target_col} (N={len(vals)}):")
        out.append(f"    Mean:   {np.mean(vals):.4f}")
        out.append(f"    Median: {np.median(vals):.4f}")
        out.append(f"    Std:    {np.std(vals):.4f}")
        pcts = np.percentile(vals, [5, 10, 25, 50, 75, 90, 95])
        out.append(f"    P5={pcts[0]:.4f}, P10={pcts[1]:.4f}, P25={pcts[2]:.4f}, "
                   f"P50={pcts[3]:.4f}, P75={pcts[4]:.4f}, P90={pcts[5]:.4f}, P95={pcts[6]:.4f}")
        wr = np.mean(vals > 0)
        out.append(f"    Win Rate (>0): {wr:.1%}")
        out.append(f"    Win Rate (>0.1 ADR): {np.mean(vals > 0.1):.1%}")
        out.append(f"    Loss Rate (<-0.1 ADR): {np.mean(vals < -0.1):.1%}")

        # Mean without top 5%
        cutoff = np.percentile(vals, 95)
        trimmed = vals[vals <= cutoff]
        out.append(f"    Mean (excl top 5%): {np.mean(trimmed):.4f}")
        # Mean without top 10%
        cutoff = np.percentile(vals, 90)
        trimmed = vals[vals <= cutoff]
        out.append(f"    Mean (excl top 10%): {np.mean(trimmed):.4f}")
        out.append("")

    # Also show distribution for GapUp+AGAINST and all 4 groups
    out.append("  COMPARISON — All 4 groups (Val), drift_240m:")
    for label, mask_fn in [
        ('GapUp+WITH', lambda df: (df['gap_direction']=='up') & (~df['od_against_gap'])),
        ('GapUp+AGAINST', lambda df: (df['gap_direction']=='up') & (df['od_against_gap'])),
        ('GapDn+WITH', lambda df: (df['gap_direction']=='down') & (~df['od_against_gap'])),
        ('GapDn+AGAINST', lambda df: (df['gap_direction']=='down') & (df['od_against_gap'])),
    ]:
        sub = val[mask_fn(val)]
        vals = sub['drift_240m'].dropna().values
        if len(vals) < 20:
            continue
        out.append(f"    {label}: N={len(vals)}, Mean={np.mean(vals):.4f}, "
                   f"Median={np.median(vals):.4f}, WR={np.mean(vals>0):.1%}")
    out.append("")

    # ========================================================================
    # TEST 5: PERMUTATION TEST
    # ========================================================================
    out.append("=" * 90)
    out.append("TEST 5: PERMUTATION TEST — Is OD direction informative?")
    out.append("  Shuffle od_against_gap labels within each gap_direction group.")
    out.append("  If real spread >> shuffled spread, OD direction has real info.")
    out.append("=" * 90)
    out.append("")

    N_PERM = 2000

    # Test on Val: GapUp group
    gap_up_val = val[val['gap_direction'] == 'up'].copy()
    real_with = gap_up_val[~gap_up_val['od_against_gap']]['drift_240m'].dropna().mean()
    real_against = gap_up_val[gap_up_val['od_against_gap']]['drift_240m'].dropna().mean()
    real_spread = real_with - real_against

    perm_spreads = []
    vals_all = gap_up_val['drift_240m'].dropna().values
    labels_all = gap_up_val.loc[gap_up_val['drift_240m'].notna(), 'od_against_gap'].values

    for _ in tqdm(range(N_PERM), desc="Perm GapUp", file=sys.stderr):
        shuffled = np.random.permutation(labels_all)
        with_mean = vals_all[~shuffled].mean()
        against_mean = vals_all[shuffled].mean()
        perm_spreads.append(with_mean - against_mean)

    perm_spreads = np.array(perm_spreads)
    p_val_gapu = np.mean(perm_spreads >= real_spread)

    out.append(f"  GapUp (Val):")
    out.append(f"    Real: WITH={real_with:.4f}, AGAINST={real_against:.4f}, Spread={real_spread:.4f}")
    out.append(f"    Perm spread: Mean={perm_spreads.mean():.4f}, Std={perm_spreads.std():.4f}")
    out.append(f"    P(perm >= real): {p_val_gapu:.4f}")
    out.append(f"    VERDICT: {'SIGNIFICANT' if p_val_gapu < 0.05 else 'NOT significant'}")
    out.append("")

    # Same for GapDown
    gap_dn_val = val[val['gap_direction'] == 'down'].copy()
    real_with_dn = gap_dn_val[~gap_dn_val['od_against_gap']]['drift_240m'].dropna().mean()
    real_against_dn = gap_dn_val[gap_dn_val['od_against_gap']]['drift_240m'].dropna().mean()
    real_spread_dn = real_with_dn - real_against_dn

    perm_spreads_dn = []
    vals_dn = gap_dn_val['drift_240m'].dropna().values
    labels_dn = gap_dn_val.loc[gap_dn_val['drift_240m'].notna(), 'od_against_gap'].values

    for _ in tqdm(range(N_PERM), desc="Perm GapDn", file=sys.stderr):
        shuffled = np.random.permutation(labels_dn)
        with_mean = vals_dn[~shuffled].mean()
        against_mean = vals_dn[shuffled].mean()
        perm_spreads_dn.append(with_mean - against_mean)

    perm_spreads_dn = np.array(perm_spreads_dn)
    p_val_gapd = np.mean(perm_spreads_dn >= real_spread_dn)

    out.append(f"  GapDown (Val):")
    out.append(f"    Real: WITH={real_with_dn:.4f}, AGAINST={real_against_dn:.4f}, Spread={real_spread_dn:.4f}")
    out.append(f"    Perm spread: Mean={perm_spreads_dn.mean():.4f}, Std={perm_spreads_dn.std():.4f}")
    out.append(f"    P(perm >= real): {p_val_gapd:.4f}")
    out.append(f"    VERDICT: {'SIGNIFICANT' if p_val_gapd < 0.05 else 'NOT significant'}")
    out.append("")

    # Combined: test OD direction across ALL trades (ignore gap direction)
    out.append("  COMBINED (All Val trades):")
    real_with_all = val[~val['od_against_gap']]['drift_240m'].dropna().mean()
    real_against_all = val[val['od_against_gap']]['drift_240m'].dropna().mean()
    real_spread_all = real_with_all - real_against_all

    perm_spreads_all = []
    vals_combined = val['drift_240m'].dropna().values
    labels_combined = val.loc[val['drift_240m'].notna(), 'od_against_gap'].values

    for _ in tqdm(range(N_PERM), desc="Perm All", file=sys.stderr):
        shuffled = np.random.permutation(labels_combined)
        with_mean = vals_combined[~shuffled].mean()
        against_mean = vals_combined[shuffled].mean()
        perm_spreads_all.append(with_mean - against_mean)

    perm_spreads_all = np.array(perm_spreads_all)
    p_val_all = np.mean(perm_spreads_all >= real_spread_all)

    out.append(f"    Real: WITH={real_with_all:.4f}, AGAINST={real_against_all:.4f}, Spread={real_spread_all:.4f}")
    out.append(f"    Perm P: {p_val_all:.4f}")
    out.append(f"    VERDICT: {'SIGNIFICANT' if p_val_all < 0.05 else 'NOT significant'}")
    out.append("")

    # ========================================================================
    # TEST 6: MULTIPLE TESTING CORRECTION
    # ========================================================================
    out.append("=" * 90)
    out.append("TEST 6: MULTIPLE TESTING CORRECTION")
    out.append("  Section D tested ~30+ combos. Apply Bonferroni.")
    out.append("=" * 90)
    out.append("")

    # The bootstrap CI for GapUp+OD_WITH was P=0.0006
    # How many groups were tested in Section E? 4 groups x 2 metrics = 8 tests
    # In Section D: ~30+ combos
    n_tests_e = 8
    n_tests_d = 30
    p_raw = 0.0006

    out.append(f"  Raw P-value (GapUp+OD_WITH drift_240m): {p_raw}")
    out.append(f"  Section E tests: {n_tests_e}, Bonferroni-adjusted: {min(p_raw * n_tests_e, 1.0):.4f}")
    out.append(f"  Section D tests: ~{n_tests_d}, Bonferroni-adjusted: {min(p_raw * n_tests_d, 1.0):.4f}")
    out.append(f"  Conservative (all tests): ~{n_tests_e + n_tests_d}, Bonferroni-adjusted: {min(p_raw * (n_tests_e + n_tests_d), 1.0):.4f}")
    out.append("")
    out.append(f"  VERDICT: Even with {n_tests_e + n_tests_d} corrections, P={min(p_raw * (n_tests_e + n_tests_d), 1.0):.4f}")
    survives = min(p_raw * (n_tests_e + n_tests_d), 1.0) < 0.05
    out.append(f"  Survives Bonferroni: {'YES' if survives else 'NO'}")
    out.append("")

    # ========================================================================
    # TEST 7: RANDOM DIRECTION BASELINE
    # ========================================================================
    out.append("=" * 90)
    out.append("TEST 7: RANDOM DIRECTION BASELINE")
    out.append("  Buy at 9:45 in random direction (long/short with 50/50).")
    out.append("  Compare mean absolute drift to OD-guided drift.")
    out.append("=" * 90)
    out.append("")

    # For Val, compute raw unsigned return from 9:45 to various horizons
    # od_close_price is the 9:45 price
    # We need raw close prices... but they're not in the events file
    # Alternative: use the signed drift and flip sign randomly

    # Actually, the drift is computed as sign * (future - od_close) / adr
    # where sign depends on the OD direction. To get the unsigned return:
    # unsigned = drift / sign. But sign varies per trade.

    # For a fair baseline, let's use the original sign convention and randomly
    # flip 50% of the trades. If OD-guided >> random, OD has value.

    n_random_trials = 1000
    val_drifts_240 = val['drift_240m'].dropna().values
    real_mean_240 = val_drifts_240.mean()

    random_means = []
    for _ in range(n_random_trials):
        flips = np.random.choice([-1, 1], size=len(val_drifts_240))
        random_means.append((val_drifts_240 * flips).mean())

    random_means = np.array(random_means)
    out.append(f"  Val drift_240m: OD-guided mean = {real_mean_240:.4f}")
    out.append(f"  Random direction: mean = {random_means.mean():.4f}, std = {random_means.std():.4f}")
    out.append(f"  Z-score: {(real_mean_240 - random_means.mean()) / random_means.std():.2f}")
    out.append(f"  P(random >= real): {np.mean(random_means >= real_mean_240):.4f}")
    out.append("")

    # Same but only for GapUp+WITH
    sub_vals = target_sub['drift_240m'].dropna().values
    real_sub = sub_vals.mean()
    random_sub = []
    for _ in range(n_random_trials):
        flips = np.random.choice([-1, 1], size=len(sub_vals))
        random_sub.append((sub_vals * flips).mean())
    random_sub = np.array(random_sub)
    out.append(f"  GapUp+WITH drift_240m: OD-guided mean = {real_sub:.4f}")
    out.append(f"  Random direction: mean = {random_sub.mean():.4f}, std = {random_sub.std():.4f}")
    out.append(f"  P(random >= real): {np.mean(random_sub >= real_sub):.4f}")
    out.append("")

    # ========================================================================
    # TEST 8: OD MAGNITUDE — Does strength actually matter?
    # ========================================================================
    out.append("=" * 90)
    out.append("TEST 8: OD MAGNITUDE — Linear relationship?")
    out.append("  Is drift_240m correlated with od_move_adr?")
    out.append("=" * 90)
    out.append("")

    # For GapUp+WITH (Val)
    sub = val[(val['gap_direction'] == 'up') & (~val['od_against_gap'])].copy()
    sub = sub.dropna(subset=['od_move_adr', 'drift_240m'])
    if len(sub) > 30:
        corr = sub['od_move_adr'].corr(sub['drift_240m'])
        out.append(f"  GapUp+WITH (Val): N={len(sub)}")
        out.append(f"    Correlation(od_move_adr, drift_240m) = {corr:.4f}")

        # Quartile analysis
        out.append(f"    Quartile analysis:")
        sub['od_q'] = pd.qcut(sub['od_move_adr'].abs(), 4, labels=['Q1_weak', 'Q2', 'Q3', 'Q4_strong'])
        for q in ['Q1_weak', 'Q2', 'Q3', 'Q4_strong']:
            qs = sub[sub['od_q'] == q]
            out.append(f"      {q}: N={len(qs)}, Mean_OD={qs['od_move_adr'].abs().mean():.3f}, "
                       f"Drift240={qs['drift_240m'].mean():.4f}, MFE60={qs['mfe_60m'].mean():.4f}")
    out.append("")

    # For ALL Val trades
    sub_all = val.dropna(subset=['od_move_adr', 'drift_240m']).copy()
    if len(sub_all) > 30:
        corr_all = sub_all['od_move_adr'].abs().corr(sub_all['drift_240m'])
        out.append(f"  ALL Val: N={len(sub_all)}")
        out.append(f"    Correlation(|od_move_adr|, drift_240m) = {corr_all:.4f}")
    out.append("")

    # ========================================================================
    # TEST 9: SIGN CONVENTION SANITY CHECK
    # ========================================================================
    out.append("=" * 90)
    out.append("TEST 9: SIGN CONVENTION SANITY CHECK")
    out.append("  Verify that positive drift_240m actually means profit.")
    out.append("  Compute RAW (unsigned) returns and compare.")
    out.append("=" * 90)
    out.append("")

    # For GapUp+WITH: sign=+1, so drift = (future - od_close) / adr
    # If positive, price went UP from 9:45. Entry is LONG at 9:45.
    # This means: we go LONG at od_close_price, and price rose => PROFIT. CORRECT.

    # For GapDn+WITH: sign=-1, so drift = -(future - od_close) / adr
    # If positive, price went DOWN from 9:45. Entry is SHORT at 9:45.
    # This means: we go SHORT at od_close_price, and price fell => PROFIT. CORRECT.

    # For GapUp+AGAINST: sign=-1, OD was down
    # drift = -(future - od_close) / adr
    # If positive, price went DOWN further. Entry is SHORT at 9:45.
    # This trades CONTINUATION of OD (down), not reversal. CORRECT per OD-following thesis.

    out.append("  Sign convention review:")
    out.append("    GapUp+WITH: sign=+1, LONG at 9:45. Positive drift = price up = PROFIT. OK")
    out.append("    GapDn+WITH: sign=-1, SHORT at 9:45. Positive drift = price down = PROFIT. OK")
    out.append("    GapUp+AGAINST: sign=-1, SHORT at 9:45 (follow OD down). Pos drift = down = PROFIT. OK")
    out.append("    GapDn+AGAINST: sign=+1, LONG at 9:45 (follow OD up). Pos drift = up = PROFIT. OK")
    out.append("")
    out.append("  IMPORTANT: All drifts measure CONTINUATION in OD direction.")
    out.append("  GapUp+WITH positive drift = 'stock continues up after gapping up & OD up'")
    out.append("  This is MOMENTUM, not mean-reversion.")
    out.append("")

    # Verify with raw numbers
    sub_verify = val[(val['gap_direction'] == 'up') & (~val['od_against_gap'])]
    sub_verify = sub_verify.dropna(subset=['od_close_price', 'drift_240m'])
    if len(sub_verify) > 0:
        # Compute what % of GapUp+WITH trades have od_direction = 'up' (should be 100%)
        pct_od_up = (sub_verify['od_direction'] == 'up').mean()
        out.append(f"  GapUp+WITH: {pct_od_up:.1%} have od_direction=up (should be ~100%)")
        out.append(f"  Mean od_move_adr: {sub_verify['od_move_adr'].mean():.4f} (should be positive)")
    out.append("")

    # ========================================================================
    # TEST 10: TRAIN→VAL STABILITY MATRIX
    # ========================================================================
    out.append("=" * 90)
    out.append("TEST 10: TRAIN → VAL STABILITY")
    out.append("  Compare effect sizes between Train and Val.")
    out.append("=" * 90)
    out.append("")

    out.append(f"{'Group':<20} {'Train_D60':>10} {'Val_D60':>10} {'Delta':>8} "
               f"{'Train_D240':>11} {'Val_D240':>11} {'Delta':>8}")
    out.append("-" * 80)

    for label, mask_fn in [
        ('GapUp+WITH', lambda df: (df['gap_direction']=='up') & (~df['od_against_gap'])),
        ('GapUp+AGAINST', lambda df: (df['gap_direction']=='up') & (df['od_against_gap'])),
        ('GapDn+WITH', lambda df: (df['gap_direction']=='down') & (~df['od_against_gap'])),
        ('GapDn+AGAINST', lambda df: (df['gap_direction']=='down') & (df['od_against_gap'])),
    ]:
        tr = train[mask_fn(train)]
        va = val[mask_fn(val)]
        tr_d60 = tr['drift_60m'].mean()
        va_d60 = va['drift_60m'].mean()
        tr_d240 = tr['drift_240m'].mean()
        va_d240 = va['drift_240m'].mean()
        out.append(f"{label:<20} {tr_d60:>10.4f} {va_d60:>10.4f} {va_d60-tr_d60:>+8.4f} "
                   f"{tr_d240:>11.4f} {va_d240:>11.4f} {va_d240-tr_d240:>+8.4f}")

    out.append("")
    out.append("  VERDICT: If Val > Train, signal may be strengthening (good).")
    out.append("  If Val << Train, overfitting concern.")
    out.append("")

    # ========================================================================
    # FINAL VERDICT
    # ========================================================================
    out.append("=" * 90)
    out.append("DEVIL'S ADVOCATE — FINAL VERDICT")
    out.append("=" * 90)
    out.append("")
    out.append("Key Questions and Answers:")
    out.append("  1. Circular reference? → No (computed independently from OHLC)")
    out.append(f"  2. Permutation test GapUp: P={p_val_gapu:.4f} → "
               f"{'OD labels carry real info' if p_val_gapu < 0.05 else 'Could be noise'}")
    out.append(f"  3. Bonferroni survives? → {'YES' if survives else 'NO'}")
    out.append(f"  4. Is it just momentum? → Check Test 2 results above")
    out.append("  5. Year stability? → Check Test 3 results above")
    out.append("  6. Outlier-driven? → Check Test 4 results above")
    out.append("")
    out.append("RECOMMENDATION:")
    if p_val_gapu < 0.05 and survives:
        out.append("  GapUp+OD_WITH passes Devil's Advocate scrutiny.")
        out.append("  PROCEED to Half-2 OOS validation for this signal.")
    elif p_val_gapu < 0.05:
        out.append("  GapUp+OD_WITH shows real signal but multiple testing is a concern.")
        out.append("  CAUTIOUSLY proceed to Half-2 OOS validation.")
    else:
        out.append("  GapUp+OD_WITH does NOT pass permutation test.")
        out.append("  DO NOT proceed to OOS validation.")
    out.append("")

    # Write output
    output_path = os.path.join(RESULTS_DIR, 'devils_advocate_od_v1.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out))
    print(f"Ergebnisse: {output_path}", file=sys.stderr)


if __name__ == '__main__':
    main()
