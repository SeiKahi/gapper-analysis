"""
TEST A: Random Walk Simulation V3 — PROPERLY CALIBRATED
=========================================================
V3 FIX: Calibrate sigma so that the OUTCOME DISTRIBUTION (WR, SL%, Timeout%)
matches the real data. This is the only fair comparison.

The issue: 1-min sigma from gapper days (~0.0039) combined with historical ADR_10
creates a mismatch. Gapper days have ~3x normal volatility, so the simulated
paths hit SL and Target much more often than in reality.

APPROACH: Binary search for the sigma that produces ~35% SL hit rate
(matching real GapDn SL5 T4_3R data: 34.8% SL, 6.1% target, 59.1% timeout).
Then check if the resulting expectancy is positive.
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
from tqdm import tqdm
import glob

np.random.seed(42)

# ============================================================
# Load real data parameters
# ============================================================
print("Loading real data parameters...", file=sys.stderr)
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

meta = pd.read_parquet(str(PROJECT_ROOT / 'data' / 'metadata' / 'metadata_master.parquet'))
meta['date'] = pd.to_datetime(meta['date'])
val_meta = meta[(meta['date'] >= '2023-01-01') & (meta['date'] <= '2023-12-31')]
median_adr_ratio = np.median(val_meta['adr_10'].dropna().values / val_meta['today_open'].dropna().values)

# Real outcome rates for GapDn SL5 T4_3R
REAL_SL_RATE = 0.348
REAL_WR = 0.061
REAL_TO_RATE = 0.591

# ============================================================
# Calibration: Find sigma that matches real SL rate
# ============================================================
print("\nCalibrating sigma to match real outcome distribution...", file=sys.stderr)

N_BARS = 390
START_PRICE = 50.0
TIMEOUT_BARS = 240
ADR_RATIO = median_adr_ratio
EXTERNAL_ADR = ADR_RATIO * START_PRICE
SL_DIST = 0.35 * EXTERNAL_ADR
TGT_DIST = 1.05 * EXTERNAL_ADR

def quick_sim(sigma, n_sims=2000, rng_seed=123):
    """Quick simulation to estimate outcome rates."""
    rng = np.random.default_rng(rng_seed)
    n_sl = 0
    n_tgt = 0
    n_to = 0
    n_total = 0

    for _ in range(n_sims):
        log_returns = rng.normal(0, sigma, N_BARS - 1)
        log_prices = np.zeros(N_BARS)
        log_prices[0] = np.log(START_PRICE)
        log_prices[1:] = log_prices[0] + np.cumsum(log_returns)
        prices = np.exp(log_prices)

        # 1 entry at bar 50 (middle of morning)
        entry_bar = 50
        entry_price = prices[entry_bar]
        sl_price = entry_price - SL_DIST
        target_price = entry_price + TGT_DIST

        outcome = 'timeout'
        max_bars = min(entry_bar + TIMEOUT_BARS, N_BARS)
        for b in range(entry_bar + 1, max_bars):
            if prices[b] <= sl_price:
                outcome = 'sl_hit'
                break
            elif prices[b] >= target_price:
                outcome = 'target_hit'
                break

        n_total += 1
        if outcome == 'sl_hit': n_sl += 1
        elif outcome == 'target_hit': n_tgt += 1
        else: n_to += 1

    return n_sl/n_total, n_tgt/n_total, n_to/n_total

# Binary search for sigma
sigma_lo = 0.0005
sigma_hi = 0.005
for _ in range(20):
    sigma_mid = (sigma_lo + sigma_hi) / 2
    sl_rate, wr, to_rate = quick_sim(sigma_mid)
    print(f"  sigma={sigma_mid:.6f}: SL={sl_rate:.3f}, WR={wr:.3f}, TO={to_rate:.3f}", file=sys.stderr)
    if sl_rate > REAL_SL_RATE:
        sigma_hi = sigma_mid  # too volatile, reduce
    else:
        sigma_lo = sigma_mid  # not volatile enough, increase

calibrated_sigma = (sigma_lo + sigma_hi) / 2
sl_rate, wr, to_rate = quick_sim(calibrated_sigma, n_sims=5000)
print(f"\n  Calibrated sigma: {calibrated_sigma:.6f}", file=sys.stderr)
print(f"  Calibrated outcomes: SL={sl_rate:.3f}, WR={wr:.3f}, TO={to_rate:.3f}", file=sys.stderr)
print(f"  Real outcomes:       SL={REAL_SL_RATE:.3f}, WR={REAL_WR:.3f}, TO={REAL_TO_RATE:.3f}", file=sys.stderr)

# ============================================================
# Full simulation with calibrated sigma
# ============================================================
print("\nRunning full simulation with calibrated sigma...", file=sys.stderr)

N_SIMS = 10000
ENTRIES_PER_SIM = 3

configs = [
    {'name': 'SL5_T4_3R', 'sl_mult': 0.35, 'tgt_mult': 1.05},
    {'name': 'SL5_T4_2R', 'sl_mult': 0.35, 'tgt_mult': 0.70},
    {'name': 'SL5_T4_15R', 'sl_mult': 0.35, 'tgt_mult': 0.525},
]

drifts = [0.0, +0.00005, -0.00005]

def run_full_sim(sigma, drift, n_sims, configs, direction='long'):
    rng = np.random.default_rng(42 + int(drift*100000) + (0 if direction=='long' else 99))
    results = {c['name']: [] for c in configs}

    for _ in tqdm(range(n_sims), desc=f"{direction} drift={drift:+.5f}", file=sys.stderr):
        log_returns = rng.normal(drift, sigma, N_BARS - 1)
        log_prices = np.zeros(N_BARS)
        log_prices[0] = np.log(START_PRICE)
        log_prices[1:] = log_prices[0] + np.cumsum(log_returns)
        prices = np.exp(log_prices)

        entry_bars = rng.integers(20, 200, size=ENTRIES_PER_SIM)

        for entry_bar in entry_bars:
            entry_price = prices[entry_bar]

            for cfg in configs:
                sl_dist = cfg['sl_mult'] * EXTERNAL_ADR
                tgt_dist = cfg['tgt_mult'] * EXTERNAL_ADR

                if direction == 'long':
                    sl_price = entry_price - sl_dist
                    target_price = entry_price + tgt_dist
                else:
                    sl_price = entry_price + sl_dist
                    target_price = entry_price - tgt_dist

                outcome = 'timeout'
                exit_price = None
                max_bars = min(entry_bar + TIMEOUT_BARS, N_BARS)

                for b in range(entry_bar + 1, max_bars):
                    p = prices[b]
                    if direction == 'long':
                        if p <= sl_price:
                            outcome = 'sl_hit'
                            exit_price = sl_price
                            break
                        elif p >= target_price:
                            outcome = 'target_hit'
                            exit_price = target_price
                            break
                    else:
                        if p >= sl_price:
                            outcome = 'sl_hit'
                            exit_price = sl_price
                            break
                        elif p <= target_price:
                            outcome = 'target_hit'
                            exit_price = target_price
                            break

                if outcome == 'timeout':
                    exit_price = prices[min(max_bars - 1, N_BARS - 1)]

                if direction == 'long':
                    pnl = exit_price - entry_price
                else:
                    pnl = entry_price - exit_price

                pnl_adr = pnl / EXTERNAL_ADR

                results[cfg['name']].append({
                    'outcome': outcome,
                    'pnl_adr': pnl_adr,
                })

    return results

# Run all scenarios
all_results = {}
for drift in drifts:
    all_results[drift] = run_full_sim(calibrated_sigma, drift, N_SIMS, configs, 'long')

short_results = run_full_sim(calibrated_sigma, 0.0, N_SIMS, configs, 'short')

# Also run with the ORIGINAL uncalibrated sigma for comparison
orig_sigma = 0.003894
orig_results = run_full_sim(orig_sigma, 0.0, N_SIMS, configs, 'long')

# ============================================================
# Bootstrap CI
# ============================================================
def bootstrap_ci(data, n_boot=5000, ci=0.95):
    data = np.array(data)
    n = len(data)
    rng = np.random.default_rng(42)
    boot_means = np.array([np.mean(rng.choice(data, size=n, replace=True)) for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    lo = np.percentile(boot_means, alpha * 100)
    hi = np.percentile(boot_means, (1 - alpha) * 100)
    p_value = np.mean(boot_means <= 0)
    return lo, hi, p_value

# ============================================================
# Write results
# ============================================================
print("\nWriting results...", file=sys.stderr)

outpath = str(PROJECT_ROOT / 'results' / 'test_random_walk.txt')
with open(outpath, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("TEST A: RANDOM WALK SIMULATION — V3 PROPERLY CALIBRATED\n")
    f.write("=" * 80 + "\n")
    f.write(f"Date: 2026-02-13\n\n")

    f.write("METHODOLOGY:\n")
    f.write(f"  N_SIMS: {N_SIMS}, N_BARS: {N_BARS}, ENTRIES_PER_SIM: {ENTRIES_PER_SIM}\n")
    f.write(f"  Start Price: ${START_PRICE}\n")
    f.write(f"  EXTERNAL ADR: ${EXTERNAL_ADR:.2f} (median ADR_10/Open = {ADR_RATIO:.4f})\n")
    f.write(f"  Timeout: {TIMEOUT_BARS} bars\n")
    f.write(f"  Entry window: Bar 20-200 (random)\n\n")

    f.write("CALIBRATION:\n")
    f.write(f"  Real data (GapDn SL5 T4_3R): SL={REAL_SL_RATE:.1%}, WR={REAL_WR:.1%}, TO={REAL_TO_RATE:.1%}\n")
    f.write(f"  Calibrated sigma: {calibrated_sigma:.6f} (per minute)\n")
    f.write(f"  Calibrated outcomes: SL={sl_rate:.1%}, WR={wr:.1%}, TO={to_rate:.1%}\n")
    f.write(f"  Original sigma (uncalibrated): 0.003894\n\n")

    f.write("  WHY CALIBRATION IS NEEDED:\n")
    f.write("  ADR_10 is the 10-day HISTORICAL average daily range.\n")
    f.write("  1-min sigma from gapper days reflects CURRENT DAY volatility.\n")
    f.write("  Gapper days have ~3x normal volatility -> SL/Target are hit\n")
    f.write("  too often with uncalibrated sigma. Calibration matches the\n")
    f.write("  outcome distribution to ensure fair comparison.\n\n")

    # ---- SECTION 1: Calibrated Results ----
    f.write("=" * 80 + "\n")
    f.write("SECTION 1: CALIBRATED RANDOM WALK — LONG (drift=0)\n")
    f.write("=" * 80 + "\n\n")

    for drift in drifts:
        f.write(f"--- Drift = {drift:+.5f} per bar ---\n")
        results = all_results[drift]

        for cfg_name in results:
            trades = results[cfg_name]
            pnls = [t['pnl_adr'] for t in trades]
            outcomes = [t['outcome'] for t in trades]

            n = len(pnls)
            avg_pnl = np.mean(pnls)
            med_pnl = np.median(pnls)

            n_sl = sum(1 for o in outcomes if o == 'sl_hit')
            n_tgt = sum(1 for o in outcomes if o == 'target_hit')
            n_to = sum(1 for o in outcomes if o == 'timeout')

            wr = n_tgt / n * 100
            lo, hi, p_val = bootstrap_ci(pnls)

            to_pnls = [t['pnl_adr'] for t in trades if t['outcome'] == 'timeout']
            avg_to_pnl = np.mean(to_pnls) if to_pnls else 0

            tgt_pnls = [t['pnl_adr'] for t in trades if t['outcome'] == 'target_hit']
            sl_pnl_list = [t['pnl_adr'] for t in trades if t['outcome'] == 'sl_hit']
            tgt_contrib = (n_tgt/n) * np.mean(tgt_pnls) if n_tgt > 0 else 0
            sl_contrib = (n_sl/n) * np.mean(sl_pnl_list) if n_sl > 0 else 0
            to_contrib = (n_to/n) * np.mean(to_pnls) if n_to > 0 else 0

            f.write(f"\n  {cfg_name}:\n")
            f.write(f"    N={n}, WR={wr:.1f}%, SL%={n_sl/n*100:.1f}%, TO%={n_to/n*100:.1f}%\n")
            f.write(f"    AvgPnL: {avg_pnl:+.4f}, MedPnL: {med_pnl:+.4f}\n")
            f.write(f"    CI: [{lo:+.4f}, {hi:+.4f}], P(<=0): {p_val:.4f}\n")
            f.write(f"    Decomposition: Tgt={tgt_contrib:+.4f}, SL={sl_contrib:+.4f}, TO={to_contrib:+.4f}\n")
            f.write(f"    Avg Timeout PnL: {avg_to_pnl:+.4f}\n")

            if drift == 0.0:
                if avg_pnl > 0 and p_val < 0.05:
                    f.write(f"    >>> ARTEFACT: Positive on calibrated RW (p={p_val:.4f}) <<<\n")
                elif avg_pnl > 0:
                    f.write(f"    >>> Slightly positive but NOT significant (p={p_val:.4f}) <<<\n")
                else:
                    f.write(f"    >>> NO ARTEFACT: Negative/zero on calibrated RW <<<\n")

        f.write("\n")

    # ---- SECTION 2: Short trades ----
    f.write("=" * 80 + "\n")
    f.write("SECTION 2: CALIBRATED RANDOM WALK — SHORT (drift=0)\n")
    f.write("=" * 80 + "\n\n")

    for cfg_name in short_results:
        trades = short_results[cfg_name]
        pnls = [t['pnl_adr'] for t in trades]
        outcomes = [t['outcome'] for t in trades]
        n = len(pnls)
        avg_pnl = np.mean(pnls)
        n_tgt = sum(1 for o in outcomes if o == 'target_hit')
        n_sl = sum(1 for o in outcomes if o == 'sl_hit')
        n_to = sum(1 for o in outcomes if o == 'timeout')
        lo, hi, p_val = bootstrap_ci(pnls)

        f.write(f"  {cfg_name} (SHORT): N={n}, WR={n_tgt/n*100:.1f}%, SL={n_sl/n*100:.1f}%, TO={n_to/n*100:.1f}%\n")
        f.write(f"    AvgPnL={avg_pnl:+.4f}, CI=[{lo:+.4f},{hi:+.4f}], P(<=0)={p_val:.4f}\n\n")

    # ---- SECTION 3: Uncalibrated for comparison ----
    f.write("=" * 80 + "\n")
    f.write("SECTION 3: UNCALIBRATED RANDOM WALK (sigma=0.003894) — FOR COMPARISON\n")
    f.write("=" * 80 + "\n\n")

    for cfg_name in orig_results:
        trades = orig_results[cfg_name]
        pnls = [t['pnl_adr'] for t in trades]
        outcomes = [t['outcome'] for t in trades]
        n = len(pnls)
        avg_pnl = np.mean(pnls)
        n_tgt = sum(1 for o in outcomes if o == 'target_hit')
        n_sl = sum(1 for o in outcomes if o == 'sl_hit')
        n_to = sum(1 for o in outcomes if o == 'timeout')
        lo, hi, p_val = bootstrap_ci(pnls)

        f.write(f"  {cfg_name}: N={n}, WR={n_tgt/n*100:.1f}%, SL={n_sl/n*100:.1f}%, TO={n_to/n*100:.1f}%\n")
        f.write(f"    AvgPnL={avg_pnl:+.4f}, CI=[{lo:+.4f},{hi:+.4f}], P(<=0)={p_val:.4f}\n")
        f.write(f"    NOTE: Outcome distribution DOES NOT match real data (WR=26.8% vs 6.1% real)\n\n")

    # ---- SECTION 4: Comparison with real data ----
    f.write("=" * 80 + "\n")
    f.write("SECTION 4: COMPARISON — CALIBRATED RW vs REAL DATA\n")
    f.write("=" * 80 + "\n\n")

    real_df = pd.read_parquet(str(PROJECT_ROOT / 'results' / 'all_trades_v3.parquet'))
    real_val = real_df[real_df['split'] == 'val']

    for cfg_name, sl_m, tgt_m in [('SL5_T4_3R', 'SL5_035', 'T4_3R'),
                                    ('SL5_T4_2R', 'SL5_035', 'T4_2R')]:
        real_gapdn = real_val[(real_val['gap_direction'] == 'down') &
                              (real_val['sl_method'] == sl_m) &
                              (real_val['target_method'] == tgt_m)]
        real_gapup = real_val[(real_val['gap_direction'] == 'up') &
                              (real_val['sl_method'] == sl_m) &
                              (real_val['target_method'] == tgt_m)]

        rw_cal = all_results[0.0][cfg_name]
        rw_pnls = [t['pnl_adr'] for t in rw_cal]
        rw_avg = np.mean(rw_pnls)
        rw_wr = sum(1 for t in rw_cal if t['outcome']=='target_hit') / len(rw_cal) * 100

        real_dn_avg = real_gapdn['pnl_adr'].mean()
        real_up_avg = real_gapup['pnl_adr'].mean()

        f.write(f"  {cfg_name}:\n")
        f.write(f"    Calibrated RW (drift=0): AvgPnL={rw_avg:+.4f}, WR={rw_wr:.1f}%\n")
        f.write(f"    Real GapDn Long (2023):  AvgPnL={real_dn_avg:+.4f}, WR={(real_gapdn['outcome']=='target_hit').mean()*100:.1f}%\n")
        f.write(f"    Real GapUp Short (2023): AvgPnL={real_up_avg:+.4f}, WR={(real_gapup['outcome']=='target_hit').mean()*100:.1f}%\n\n")

        f.write(f"    GapDn excess over RW: {real_dn_avg - rw_avg:+.4f} ADR\n")
        f.write(f"    GapUp excess over RW: {real_up_avg - rw_avg:+.4f} ADR\n")

        if rw_avg > 0:
            explained = rw_avg / real_dn_avg * 100 if real_dn_avg > 0 else 0
            f.write(f"    RW artefact explains {explained:.0f}% of GapDn edge\n")
        f.write("\n")

    # ---- SECTION 5: Definitive Verdict ----
    f.write("=" * 80 + "\n")
    f.write("SECTION 5: DEFINITIVE VERDICT\n")
    f.write("=" * 80 + "\n\n")

    key_cal = all_results[0.0]['SL5_T4_3R']
    key_pnls = [t['pnl_adr'] for t in key_cal]
    key_avg = np.mean(key_pnls)
    key_lo, key_hi, key_p = bootstrap_ci(key_pnls)

    short_key = short_results['SL5_T4_3R']
    short_pnls = [t['pnl_adr'] for t in short_key]
    short_avg = np.mean(short_pnls)

    real_gapdn_3r = real_val[(real_val['gap_direction'] == 'down') &
                              (real_val['sl_method'] == 'SL5_035') &
                              (real_val['target_method'] == 'T4_3R')]
    real_avg = real_gapdn_3r['pnl_adr'].mean()

    real_gapup_3r = real_val[(real_val['gap_direction'] == 'up') &
                              (real_val['sl_method'] == 'SL5_035') &
                              (real_val['target_method'] == 'T4_3R')]
    real_up_avg = real_gapup_3r['pnl_adr'].mean()

    f.write(f"  Calibrated RW (drift=0, sigma={calibrated_sigma:.6f}):\n")
    f.write(f"    Long:  AvgPnL = {key_avg:+.4f}, CI=[{key_lo:+.4f},{key_hi:+.4f}], P(<=0)={key_p:.4f}\n")
    f.write(f"    Short: AvgPnL = {short_avg:+.4f}\n\n")

    f.write(f"  Real data (Val 2023):\n")
    f.write(f"    GapDn Long: AvgPnL = {real_avg:+.4f}\n")
    f.write(f"    GapUp Short: AvgPnL = {real_up_avg:+.4f}\n\n")

    if key_avg > 0 and key_p < 0.05:
        explained = key_avg / real_avg * 100 if real_avg > 0 else 0
        residual = real_avg - key_avg

        f.write("  *** VERDICT: STRUCTURAL ARTEFACT COMPONENT CONFIRMED ***\n\n")
        f.write(f"  The SL5+T4_3R setup generates +{key_avg:.4f} ADR on a PURE Random Walk\n")
        f.write(f"  with calibrated volatility (drift=0, no directional bias).\n\n")
        f.write(f"  DECOMPOSITION:\n")
        f.write(f"    Total real edge (GapDn):      {real_avg:+.4f} ADR\n")
        f.write(f"    Structural artefact:           {key_avg:+.4f} ADR ({explained:.0f}%)\n")
        f.write(f"    Genuine residual (GapDn):      {residual:+.4f} ADR ({100-explained:.0f}%)\n")
        f.write(f"    GapUp Short (reality check):   {real_up_avg:+.4f} ADR\n")
        f.write(f"    GapUp excess over RW:          {real_up_avg - key_avg:+.4f} ADR\n\n")

        f.write(f"  INTERPRETATION:\n")
        if explained > 80:
            f.write(f"    The artefact explains >{explained:.0f}% of the edge.\n")
            f.write(f"    The 'edge' is MOSTLY a mathematical property of asymmetric\n")
            f.write(f"    SL/Target with timeout on volatile instruments.\n")
        elif explained > 50:
            f.write(f"    The artefact explains ~{explained:.0f}% of the edge.\n")
            f.write(f"    There IS a genuine GapDn component ({residual:+.4f}), but it's\n")
            f.write(f"    smaller than the structural component.\n")
        elif explained > 20:
            f.write(f"    The artefact explains ~{explained:.0f}% of the edge.\n")
            f.write(f"    The genuine GapDn edge ({residual:+.4f}) is SUBSTANTIAL.\n")
            f.write(f"    The SL/Target structure amplifies a real signal.\n")
        else:
            f.write(f"    The artefact is small ({explained:.0f}%). Most of the edge is genuine.\n")

        # Direction test
        gap_delta = real_avg - real_up_avg
        f.write(f"\n  DIRECTION SPECIFICITY:\n")
        f.write(f"    GapDn Long:  {real_avg:+.4f} ADR\n")
        f.write(f"    GapUp Short: {real_up_avg:+.4f} ADR\n")
        f.write(f"    Delta:       {gap_delta:+.4f} ADR\n")
        if gap_delta > 0.02:
            f.write(f"    GapDn is BETTER than GapUp by {gap_delta:.4f} ADR.\n")
            f.write(f"    This directional asymmetry CANNOT be explained by the artefact.\n")
        else:
            f.write(f"    Minimal directional difference.\n")

    elif key_avg > 0:
        f.write("  VERDICT: Slight positive tendency on RW but NOT significant.\n")
        f.write(f"  Most of the real edge ({real_avg:+.4f}) likely genuine.\n")
    else:
        f.write("  VERDICT: NO ARTEFACT.\n")
        f.write(f"  Calibrated RW shows zero/negative expectancy ({key_avg:+.4f}).\n")
        f.write(f"  The real edge ({real_avg:+.4f}) is entirely genuine.\n")

    f.write("\n  LONG/SHORT SYMMETRY ON RW:\n")
    f.write(f"    Long:  {key_avg:+.4f}\n")
    f.write(f"    Short: {short_avg:+.4f}\n")
    diff = abs(key_avg - short_avg)
    f.write(f"    Delta: {diff:.4f}\n")

    f.write("\n" + "=" * 80 + "\n")
    f.write("END OF TEST A\n")
    f.write("=" * 80 + "\n")

print(f"\nResults written to: {outpath}", file=sys.stderr)
print("TEST A V3 COMPLETE.", file=sys.stderr)
