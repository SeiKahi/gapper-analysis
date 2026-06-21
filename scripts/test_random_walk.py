"""
TEST A: Random Walk Simulation — DER KILLER-TEST (V2 — FIXED ADR)
===================================================================
Prueft ob SL5+T4_3R auf SYNTHETISCHEN Random Walks (drift=0) positive Expectancy erzeugt.

CRITICAL FIX vs V1: ADR is now an EXTERNAL parameter (set to match real gapper ADR),
NOT computed from the simulated path itself. Computing ADR from the path creates
look-ahead bias because ADR = high-low of the SAME path the trade is on,
making target = 1.05 * (high-low) nearly unreachable by construction.

Real gapper ADR_10 is the 10-day HISTORICAL ADR, NOT the current day's range.
So it must be an external constant for each simulation.
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
# STEP 1: Estimate real 1-min volatility AND real ADR_10 from data
# ============================================================
print("Step 1: Estimating real parameters from data...", file=sys.stderr)

# Get real ADR_10 distribution from metadata
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

meta = pd.read_parquet(str(PROJECT_ROOT / 'data' / 'metadata' / 'metadata_master.parquet'))
meta['date'] = pd.to_datetime(meta['date'])
val_meta = meta[(meta['date'] >= '2023-01-01') & (meta['date'] <= '2023-12-31')]
real_adr_values = val_meta['adr_10'].dropna().values
real_open_prices = val_meta['today_open'].dropna().values

median_adr = np.median(real_adr_values)
mean_adr = np.mean(real_adr_values)
median_open = np.median(real_open_prices)
median_adr_ratio = np.median(real_adr_values / real_open_prices)

print(f"  Median ADR_10: ${median_adr:.2f}", file=sys.stderr)
print(f"  Mean ADR_10: ${mean_adr:.2f}", file=sys.stderr)
print(f"  Median Open Price: ${median_open:.2f}", file=sys.stderr)
print(f"  Median ADR/Open ratio: {median_adr_ratio:.4f}", file=sys.stderr)

# Get real 1-min volatility
vwap_files = glob.glob(str(PROJECT_ROOT / 'data' / 'vwap' / '**' / '*.parquet'), recursive=True)
sample_files = np.random.choice(vwap_files, size=min(500, len(vwap_files)), replace=False)

all_1min_returns = []
for f in tqdm(sample_files, desc="Reading VWAP files", file=sys.stderr):
    try:
        df = pd.read_parquet(f, columns=['close'])
        prices = df['close'].dropna().values
        if len(prices) < 50:
            continue
        returns = np.diff(np.log(prices))
        all_1min_returns.extend(returns.tolist())
    except:
        continue

real_1min_std = np.std(all_1min_returns)
real_1min_mean = np.mean(all_1min_returns)
print(f"  Real 1-min log-return std: {real_1min_std:.6f}", file=sys.stderr)
print(f"  Real 1-min log-return mean: {real_1min_mean:.8f}", file=sys.stderr)

# ============================================================
# STEP 2: Simulate Random Walks with EXTERNAL ADR
# ============================================================
print("\nStep 2: Simulating Random Walks with external ADR...", file=sys.stderr)

N_SIMS = 10000
N_BARS = 390
START_PRICE = 50.0
ENTRIES_PER_SIM = 3
TIMEOUT_BARS = 240

sigma = real_1min_std

# External ADR: use median_adr_ratio * START_PRICE
# This simulates: "yesterday's ADR was X, today we use it as SL/Target reference"
EXTERNAL_ADR = median_adr_ratio * START_PRICE
print(f"  External ADR for simulation: ${EXTERNAL_ADR:.2f} ({median_adr_ratio:.4f} of ${START_PRICE})", file=sys.stderr)

# SL/Target configurations
configs = [
    {'name': 'SL5_T4_3R', 'sl_adr_mult': 0.35, 'tgt_adr_mult': 1.05},
    {'name': 'SL5_T4_2R', 'sl_adr_mult': 0.35, 'tgt_adr_mult': 0.70},
    {'name': 'SL5_T4_15R', 'sl_adr_mult': 0.35, 'tgt_adr_mult': 0.525},
]

drifts = [0.0, +0.0001, -0.0001]

def simulate_trades_fixed_adr(n_sims, n_bars, start_price, sigma, drift, external_adr, configs, entries_per_sim, timeout_bars, rng):
    """Simulate random walk price paths with EXTERNAL ADR for SL/Target."""
    results = {c['name']: [] for c in configs}

    for _ in tqdm(range(n_sims), desc=f"Simulating (drift={drift:.4f})", file=sys.stderr):
        log_returns = rng.normal(drift, sigma, n_bars - 1)
        log_prices = np.zeros(n_bars)
        log_prices[0] = np.log(start_price)
        log_prices[1:] = log_prices[0] + np.cumsum(log_returns)
        prices = np.exp(log_prices)

        entry_bars = rng.integers(20, 200, size=entries_per_sim)

        for entry_bar in entry_bars:
            entry_price = prices[entry_bar]

            for cfg in configs:
                sl_dist = cfg['sl_adr_mult'] * external_adr
                target_dist = cfg['tgt_adr_mult'] * external_adr

                sl_price = entry_price - sl_dist       # Long
                target_price = entry_price + target_dist

                outcome = 'timeout'
                exit_price = None
                max_bars = min(entry_bar + timeout_bars, n_bars)

                for b in range(entry_bar + 1, max_bars):
                    p = prices[b]
                    if p <= sl_price:
                        outcome = 'sl_hit'
                        exit_price = sl_price
                        break
                    elif p >= target_price:
                        outcome = 'target_hit'
                        exit_price = target_price
                        break

                if outcome == 'timeout':
                    exit_price = prices[min(max_bars - 1, n_bars - 1)]

                pnl = exit_price - entry_price
                pnl_adr = pnl / external_adr if external_adr > 0 else 0

                results[cfg['name']].append({
                    'outcome': outcome,
                    'pnl': pnl,
                    'pnl_adr': pnl_adr,
                    'entry_bar': entry_bar,
                })

    return results

def simulate_short_fixed_adr(n_sims, n_bars, start_price, sigma, drift, external_adr, configs, entries_per_sim, timeout_bars, rng):
    """Same but SHORT direction."""
    results = {c['name']: [] for c in configs}

    for _ in tqdm(range(n_sims), desc=f"Short (drift={drift:.4f})", file=sys.stderr):
        log_returns = rng.normal(drift, sigma, n_bars - 1)
        log_prices = np.zeros(n_bars)
        log_prices[0] = np.log(start_price)
        log_prices[1:] = log_prices[0] + np.cumsum(log_returns)
        prices = np.exp(log_prices)

        entry_bars = rng.integers(20, 200, size=entries_per_sim)

        for entry_bar in entry_bars:
            entry_price = prices[entry_bar]

            for cfg in configs:
                sl_dist = cfg['sl_adr_mult'] * external_adr
                target_dist = cfg['tgt_adr_mult'] * external_adr

                sl_price = entry_price + sl_dist
                target_price = entry_price - target_dist

                outcome = 'timeout'
                exit_price = None
                max_bars = min(entry_bar + timeout_bars, n_bars)

                for b in range(entry_bar + 1, max_bars):
                    p = prices[b]
                    if p >= sl_price:
                        outcome = 'sl_hit'
                        exit_price = sl_price
                        break
                    elif p <= target_price:
                        outcome = 'target_hit'
                        exit_price = target_price
                        break

                if outcome == 'timeout':
                    exit_price = prices[min(max_bars - 1, n_bars - 1)]

                pnl = entry_price - exit_price
                pnl_adr = pnl / external_adr if external_adr > 0 else 0

                results[cfg['name']].append({
                    'outcome': outcome,
                    'pnl': pnl,
                    'pnl_adr': pnl_adr,
                    'entry_bar': entry_bar,
                })

    return results

# ============================================================
# STEP 3: Run all simulations
# ============================================================
all_results = {}
rng = np.random.default_rng(42)

for drift in drifts:
    results = simulate_trades_fixed_adr(N_SIMS, N_BARS, START_PRICE, sigma, drift, EXTERNAL_ADR, configs, ENTRIES_PER_SIM, TIMEOUT_BARS, rng)
    all_results[drift] = results

print("\nSimulating SHORT trades (drift=0)...", file=sys.stderr)
short_results = simulate_short_fixed_adr(N_SIMS, N_BARS, START_PRICE, sigma, 0.0, EXTERNAL_ADR, configs, ENTRIES_PER_SIM, TIMEOUT_BARS, rng)

# ============================================================
# STEP 4: Bootstrap CI
# ============================================================
print("\nComputing Bootstrap CIs...", file=sys.stderr)

def bootstrap_ci(data, n_boot=5000, ci=0.95):
    data = np.array(data)
    n = len(data)
    boot_means = np.array([np.mean(np.random.choice(data, size=n, replace=True)) for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    lo = np.percentile(boot_means, alpha * 100)
    hi = np.percentile(boot_means, (1 - alpha) * 100)
    p_value = np.mean(boot_means <= 0)
    return lo, hi, p_value

# ============================================================
# STEP 5: Write results
# ============================================================
print("\nWriting results...", file=sys.stderr)

outpath = str(PROJECT_ROOT / 'results' / 'test_random_walk.txt')
with open(outpath, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("TEST A: RANDOM WALK SIMULATION — DER KILLER-TEST (V2 — FIXED ADR)\n")
    f.write("=" * 80 + "\n")
    f.write(f"Date: 2026-02-13\n")
    f.write(f"N_SIMS: {N_SIMS}, N_BARS: {N_BARS}, ENTRIES_PER_SIM: {ENTRIES_PER_SIM}\n")
    f.write(f"Start Price: ${START_PRICE}\n")
    f.write(f"1-min Volatility (sigma): {sigma:.6f} (measured from {len(sample_files)} real VWAP files)\n")
    f.write(f"  Real 1-min mean return: {real_1min_mean:.8f}\n")
    f.write(f"  Real 1-min std return: {real_1min_std:.6f}\n")
    f.write(f"EXTERNAL ADR: ${EXTERNAL_ADR:.2f} (= {median_adr_ratio:.4f} * ${START_PRICE})\n")
    f.write(f"  Based on median ADR_10/Open from 2023 val gappers\n")
    f.write(f"  Real median ADR_10: ${median_adr:.2f}, Median Open: ${median_open:.2f}\n")
    f.write(f"Timeout: {TIMEOUT_BARS} bars\n")
    f.write(f"Entry window: Bar 20-200 (random)\n")
    f.write(f"Trade direction: LONG (unless noted SHORT)\n")
    f.write(f"\nCRITICAL FIX: ADR is EXTERNAL (not computed from simulation path).\n")
    f.write(f"This matches how real ADR_10 works: it's yesterday's value, not today's.\n\n")

    # ---- MAIN RESULTS ----
    f.write("=" * 80 + "\n")
    f.write("SECTION 1: LONG TRADES ON RANDOM WALKS\n")
    f.write("=" * 80 + "\n\n")

    for drift in drifts:
        f.write(f"--- Drift = {drift:+.4f} per bar ---\n")
        results = all_results[drift]

        for cfg_name in results:
            trades = results[cfg_name]
            pnls = [t['pnl_adr'] for t in trades]
            outcomes = [t['outcome'] for t in trades]

            n = len(pnls)
            avg_pnl = np.mean(pnls)
            med_pnl = np.median(pnls)
            std_pnl = np.std(pnls)

            n_sl = sum(1 for o in outcomes if o == 'sl_hit')
            n_tgt = sum(1 for o in outcomes if o == 'target_hit')
            n_to = sum(1 for o in outcomes if o == 'timeout')

            wr = n_tgt / n * 100 if n > 0 else 0
            sl_pct = n_sl / n * 100 if n > 0 else 0
            to_pct = n_to / n * 100 if n > 0 else 0

            lo, hi, p_val = bootstrap_ci(pnls)

            to_pnls = [t['pnl_adr'] for t in trades if t['outcome'] == 'timeout']
            avg_to_pnl = np.mean(to_pnls) if to_pnls else 0

            # Decomposition
            sl_pnls = [t['pnl_adr'] for t in trades if t['outcome'] == 'sl_hit']
            tgt_pnls = [t['pnl_adr'] for t in trades if t['outcome'] == 'target_hit']

            tgt_contrib = (n_tgt / n) * np.mean(tgt_pnls) if n_tgt > 0 else 0
            sl_contrib = (n_sl / n) * np.mean(sl_pnls) if n_sl > 0 else 0
            to_contrib = (n_to / n) * np.mean(to_pnls) if n_to > 0 else 0

            f.write(f"\n  {cfg_name}:\n")
            f.write(f"    N Trades:       {n}\n")
            f.write(f"    Target Hit:     {n_tgt} ({wr:.1f}%)\n")
            f.write(f"    SL Hit:         {n_sl} ({sl_pct:.1f}%)\n")
            f.write(f"    Timeout:        {n_to} ({to_pct:.1f}%)\n")
            f.write(f"    AvgPnL (ADR):   {avg_pnl:+.4f}\n")
            f.write(f"    MedPnL (ADR):   {med_pnl:+.4f}\n")
            f.write(f"    StdPnL (ADR):   {std_pnl:.4f}\n")
            f.write(f"    Bootstrap 95% CI: [{lo:+.4f}, {hi:+.4f}]\n")
            f.write(f"    P(mean<=0):     {p_val:.4f}\n")
            f.write(f"    --- PnL Decomposition ---\n")
            f.write(f"    Target contrib: {tgt_contrib:+.4f}\n")
            f.write(f"    SL contrib:     {sl_contrib:+.4f}\n")
            f.write(f"    Timeout contrib:{to_contrib:+.4f}\n")
            f.write(f"    Avg Timeout PnL: {avg_to_pnl:+.4f}\n")

            if drift == 0.0:
                if avg_pnl > 0 and p_val < 0.05:
                    f.write(f"    >>> ARTEFACT CONFIRMED: POSITIVE EXPECTANCY ON RANDOM WALK (p={p_val:.4f}) <<<\n")
                elif avg_pnl > 0:
                    f.write(f"    >>> Slightly positive but NOT significant (p={p_val:.4f}) <<<\n")
                else:
                    f.write(f"    >>> NO artefact: NEGATIVE or ZERO on Random Walk <<<\n")

        f.write("\n")

    # ---- SHORT RESULTS ----
    f.write("=" * 80 + "\n")
    f.write("SECTION 2: SHORT TRADES ON RANDOM WALKS (drift=0)\n")
    f.write("=" * 80 + "\n\n")

    for cfg_name in short_results:
        trades = short_results[cfg_name]
        pnls = [t['pnl_adr'] for t in trades]
        outcomes = [t['outcome'] for t in trades]

        n = len(pnls)
        avg_pnl = np.mean(pnls)
        med_pnl = np.median(pnls)

        n_sl = sum(1 for o in outcomes if o == 'sl_hit')
        n_tgt = sum(1 for o in outcomes if o == 'target_hit')
        n_to = sum(1 for o in outcomes if o == 'timeout')

        wr = n_tgt / n * 100 if n > 0 else 0
        lo, hi, p_val = bootstrap_ci(pnls)

        f.write(f"  {cfg_name} (SHORT):\n")
        f.write(f"    N={n}, Target={n_tgt} ({wr:.1f}%), SL={n_sl} ({n_sl/n*100:.1f}%), TO={n_to} ({n_to/n*100:.1f}%)\n")
        f.write(f"    AvgPnL={avg_pnl:+.4f}, MedPnL={med_pnl:+.4f}, CI=[{lo:+.4f},{hi:+.4f}], P(<=0)={p_val:.4f}\n\n")

    # ---- COMPARISON ----
    f.write("=" * 80 + "\n")
    f.write("SECTION 3: COMPARISON — RANDOM WALK vs REAL DATA\n")
    f.write("=" * 80 + "\n\n")

    real_df = pd.read_parquet(str(PROJECT_ROOT / 'results' / 'all_trades_v3.parquet'))
    real_val = real_df[real_df['split'] == 'val']

    for cfg_name, sl_m, tgt_m in [('SL5_T4_3R', 'SL5_035', 'T4_3R'),
                                    ('SL5_T4_2R', 'SL5_035', 'T4_2R'),
                                    ('SL5_T4_15R', 'SL5_035', 'T4_15R')]:
        real_gapdn = real_val[(real_val['gap_direction'] == 'down') &
                              (real_val['sl_method'] == sl_m) &
                              (real_val['target_method'] == tgt_m)]
        real_gapup = real_val[(real_val['gap_direction'] == 'up') &
                              (real_val['sl_method'] == sl_m) &
                              (real_val['target_method'] == tgt_m)]

        rw_trades = all_results[0.0][cfg_name]
        rw_pnls = [t['pnl_adr'] for t in rw_trades]
        rw_avg = np.mean(rw_pnls)

        rw_wr = sum(1 for t in rw_trades if t['outcome'] == 'target_hit') / len(rw_trades) * 100
        real_dn_wr = (real_gapdn['outcome'] == 'target_hit').mean() * 100
        real_up_wr = (real_gapup['outcome'] == 'target_hit').mean() * 100

        f.write(f"  {cfg_name}:\n")
        f.write(f"    Random Walk (drift=0): N={len(rw_pnls)}, AvgPnL={rw_avg:+.4f}, WR={rw_wr:.1f}%\n")
        f.write(f"    Real GapDn (Val 2023): N={len(real_gapdn)}, AvgPnL={real_gapdn['pnl_adr'].mean():+.4f}, WR={real_dn_wr:.1f}%\n")
        f.write(f"    Real GapUp (Val 2023): N={len(real_gapup)}, AvgPnL={real_gapup['pnl_adr'].mean():+.4f}, WR={real_up_wr:.1f}%\n")
        f.write(f"    Real GapDn - RW = {real_gapdn['pnl_adr'].mean() - rw_avg:+.4f} ADR\n")
        f.write(f"    Real GapUp - RW = {real_gapup['pnl_adr'].mean() - rw_avg:+.4f} ADR\n")

        if rw_avg > 0:
            f.write(f"    >>> Random Walk POSITIVE — structural artefact component exists <<<\n")
        else:
            f.write(f"    >>> Random Walk NEGATIVE — no structural artefact <<<\n")
        f.write("\n")

    # ---- VERDICT ----
    f.write("=" * 80 + "\n")
    f.write("SECTION 4: DEFINITIVE VERDICT\n")
    f.write("=" * 80 + "\n\n")

    key_rw = all_results[0.0]['SL5_T4_3R']
    key_pnls = [t['pnl_adr'] for t in key_rw]
    key_avg = np.mean(key_pnls)
    key_lo, key_hi, key_p = bootstrap_ci(key_pnls)

    short_key = short_results['SL5_T4_3R']
    short_pnls = [t['pnl_adr'] for t in short_key]
    short_avg = np.mean(short_pnls)

    real_gapdn_3r = real_val[(real_val['gap_direction'] == 'down') &
                              (real_val['sl_method'] == 'SL5_035') &
                              (real_val['target_method'] == 'T4_3R')]
    real_avg = real_gapdn_3r['pnl_adr'].mean()

    f.write(f"  SL5+T4_3R on Random Walk (drift=0, external ADR):\n")
    f.write(f"    Long:  AvgPnL = {key_avg:+.4f} ADR, CI=[{key_lo:+.4f},{key_hi:+.4f}], P(<=0)={key_p:.4f}\n")
    f.write(f"    Short: AvgPnL = {short_avg:+.4f} ADR\n\n")

    f.write(f"  Real GapDn Long SL5+T4_3R (Val 2023):\n")
    f.write(f"    AvgPnL = {real_avg:+.4f} ADR\n\n")

    if key_avg > 0 and key_p < 0.05:
        f.write("  *** VERDICT: ARTEFACT CONFIRMED ***\n")
        f.write(f"  SL5+T4_3R generates POSITIVE expectancy ({key_avg:+.4f}) on PURE Random Walks.\n")
        explained = key_avg / real_avg * 100 if real_avg > 0 else 0
        residual = real_avg - key_avg
        f.write(f"  Structural artefact explains: {explained:.0f}% of the real edge\n")
        f.write(f"  Residual (potentially genuine): {residual:+.4f} ADR\n")
        if residual > 0:
            # Bootstrap the residual
            f.write(f"  The residual ({residual:+.4f}) may still represent a real GapDn mean-reversion effect.\n")
        if explained > 80:
            f.write(f"  >>> MOST of the edge is an artefact. Very little genuine signal. <<<\n")
        elif explained > 50:
            f.write(f"  >>> MAJORITY artefact, small genuine component. <<<\n")
        elif explained > 20:
            f.write(f"  >>> Significant artefact component, but real edge also substantial. <<<\n")
        else:
            f.write(f"  >>> Small artefact component. Most of the edge appears genuine. <<<\n")

    elif key_avg > 0:
        f.write("  VERDICT: Random Walk slightly positive but NOT significant.\n")
        f.write("  The artefact hypothesis is POSSIBLE but not proven.\n")
        f.write(f"  Most of the real edge ({real_avg:+.4f}) appears genuine.\n")

    else:
        f.write("  *** VERDICT: NO ARTEFACT ***\n")
        f.write(f"  SL5+T4_3R generates NEGATIVE expectancy ({key_avg:+.4f}) on Random Walks.\n")
        f.write(f"  The real GapDn edge ({real_avg:+.4f}) is entirely from market dynamics.\n")
        f.write("  This is strong evidence for a genuine mean-reversion effect in GapDn stocks.\n")

    f.write(f"\n  Long/Short symmetry on Random Walk (drift=0):\n")
    f.write(f"    Long:  {key_avg:+.4f}\n")
    f.write(f"    Short: {short_avg:+.4f}\n")
    diff = abs(key_avg - short_avg)
    f.write(f"    Difference: {diff:.4f} ({'symmetric' if diff < 0.005 else 'slight asymmetry from sampling'})\n")

    f.write("\n" + "=" * 80 + "\n")
    f.write("END OF TEST A\n")
    f.write("=" * 80 + "\n")

print(f"\nResults written to: {outpath}", file=sys.stderr)
print("TEST A V2 COMPLETE.", file=sys.stderr)
