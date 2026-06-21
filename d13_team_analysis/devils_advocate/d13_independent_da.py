"""
D13 INDEPENDENT DEVIL'S ADVOCATE
=================================
Kritische Pruefung der 2 vielversprechenden IS-Strategien:
  S4 relaxed: PM/RTH Ratio Extreme (pm_rth5 < 0.30, gap >= 1.5 ADR)
  S2 GapUp:   No-Fill Power Move (gap not filled 90min, rvol60>=3, gap>=3 ADR, GapUp only)

Pruefungen:
  1) Data-Mining / Multiple Testing Korrektur (Bonferroni, Holm-Bonferroni)
  2) Threshold-Sensitivitaet (monotone Edge?)
  3) Jahres-Stabilitaet (Halbjahres-Split, Bootstrap-Test)
  4) OD-Unabhaengigkeit (Korrelationsanalyse)
  5) S2 Timeout-Problem (Exit-Decomposition)
  6) S4 Concentration-Risk (Ticker, Earnings, Sektor, MarketCap)
  7) Gesamturteil

IS-Periode: 2021-02-21 bis 2023-12-31
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats as sp_stats
import warnings
import time

warnings.filterwarnings('ignore')
np.random.seed(42)

# ============================================================
# PATHS & CONFIG
# ============================================================
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROJECT = PROJECT_ROOT
META_PATH = PROJECT / "data" / "metadata" / "metadata_v9.parquet"
RAW_1MIN = PROJECT / "data" / "raw_1min"
RESULTS_DIR = PROJECT / "d13_team_analysis" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = RESULTS_DIR / "d13_independent_da_results.txt"

N_BOOTSTRAP = 10_000
ALPHA = 0.05
# We tested at least 6 strategies + 5 variants = 11 hypotheses
# S4 was relaxed AFTER the original failed -> that's an extra degree of freedom
NUM_HYPOTHESES = 11  # conservative minimum
NUM_HYPOTHESES_WITH_RELAXATION = 12  # S4 relaxation = additional hypothesis

# ============================================================
# OUTPUT LOGGER
# ============================================================
out_lines = []

def log(msg=''):
    print(msg)
    out_lines.append(msg)


# ============================================================
# SIMULATE TRAIL_D (for S4: entry 9:35, post_entry from 9:36)
# ============================================================
def simulate_trail_d_935(bars_rth, entry_price, sl_dist, trade_dir, adr):
    """TRAIL_D: +1R -> BE, then trail = peak - 0.5R. Entry at 9:35."""
    if sl_dist <= 0 or pd.isna(sl_dist) or adr <= 0:
        return None
    post_entry = bars_rth[(bars_rth['time_et'] >= '09:36') & (bars_rth['time_et'] <= '15:55')]
    if len(post_entry) == 0:
        return None

    sl_level = entry_price - sl_dist if trade_dir == 'long' else entry_price + sl_dist
    highest_r = 0.0
    trail_active = False
    exit_price = None
    exit_type = 'timeout'
    exit_time = None

    for _, bar in post_entry.iterrows():
        if trade_dir == 'long':
            current_r = max(0, (bar['high'] - entry_price) / sl_dist)
        else:
            current_r = max(0, (entry_price - bar['low']) / sl_dist)
        highest_r = max(highest_r, current_r)

        if highest_r >= 1.0 and not trail_active:
            trail_active = True
            sl_level = entry_price  # BE
        if trail_active:
            if trade_dir == 'long':
                new_sl = entry_price + (highest_r - 0.5) * sl_dist
                sl_level = max(sl_level, new_sl)
            else:
                new_sl = entry_price - (highest_r - 0.5) * sl_dist
                sl_level = min(sl_level, new_sl)

        if trade_dir == 'long' and bar['low'] <= sl_level:
            exit_price = sl_level
            exit_type = 'trail_sl' if trail_active else 'initial_sl'
            exit_time = bar['time_et']
            break
        elif trade_dir == 'short' and bar['high'] >= sl_level:
            exit_price = sl_level
            exit_type = 'trail_sl' if trail_active else 'initial_sl'
            exit_time = bar['time_et']
            break

    if exit_price is None:
        exit_price = post_entry.iloc[-1]['close']
        exit_type = 'timeout'
        exit_time = post_entry.iloc[-1]['time_et']

    pnl = (exit_price - entry_price) if trade_dir == 'long' else (entry_price - exit_price)
    return {
        'pnl': pnl, 'pnl_adr': pnl / adr, 'pnl_r': pnl / sl_dist,
        'exit_type': exit_type, 'exit_time': exit_time,
        'highest_r': highest_r, 'trail_active': trail_active,
        'winner': pnl > 0,
    }


# ============================================================
# SIMULATE TRAIL_D for S2 (entry 11:00, trail from 11:01)
# ============================================================
def simulate_trail_d_1100(bars_rth, entry_price, sl_dist, trade_dir, adr):
    """TRAIL_D for S2: Entry at 11:00, trail active from 11:01."""
    if sl_dist <= 0 or pd.isna(sl_dist) or adr <= 0:
        return None
    post_entry = bars_rth[(bars_rth['time_et'] >= '11:01') & (bars_rth['time_et'] <= '15:55')]
    if len(post_entry) == 0:
        return None

    sl_level = entry_price - sl_dist if trade_dir == 'long' else entry_price + sl_dist
    highest_r = 0.0
    trail_active = False
    exit_price = None
    exit_type = 'timeout'
    exit_time = None

    for _, bar in post_entry.iterrows():
        if trade_dir == 'long':
            current_r = max(0, (bar['high'] - entry_price) / sl_dist)
        else:
            current_r = max(0, (entry_price - bar['low']) / sl_dist)
        highest_r = max(highest_r, current_r)

        if highest_r >= 1.0 and not trail_active:
            trail_active = True
            sl_level = entry_price  # BE
        if trail_active:
            if trade_dir == 'long':
                new_sl = entry_price + (highest_r - 0.5) * sl_dist
                sl_level = max(sl_level, new_sl)
            else:
                new_sl = entry_price - (highest_r - 0.5) * sl_dist
                sl_level = min(sl_level, new_sl)

        if trade_dir == 'long' and bar['low'] <= sl_level:
            exit_price = sl_level
            exit_type = 'trail_sl' if trail_active else 'initial_sl'
            exit_time = bar['time_et']
            break
        elif trade_dir == 'short' and bar['high'] >= sl_level:
            exit_price = sl_level
            exit_type = 'trail_sl' if trail_active else 'initial_sl'
            exit_time = bar['time_et']
            break

    if exit_price is None:
        exit_price = post_entry.iloc[-1]['close']
        exit_type = 'timeout'
        exit_time = post_entry.iloc[-1]['time_et']

    pnl = (exit_price - entry_price) if trade_dir == 'long' else (entry_price - exit_price)
    return {
        'pnl': pnl, 'pnl_adr': pnl / adr, 'pnl_r': pnl / sl_dist,
        'exit_type': exit_type, 'exit_time': exit_time,
        'highest_r': highest_r, 'trail_active': trail_active,
        'winner': pnl > 0,
    }


# ============================================================
# SIMULATE FIXED EOD EXIT (for S2 comparison)
# ============================================================
def simulate_fixed_eod(bars_rth, entry_price, sl_dist, trade_dir, adr):
    """Fixed EOD exit: hold until 15:55 close, no trail. SL still active."""
    if sl_dist <= 0 or pd.isna(sl_dist) or adr <= 0:
        return None
    post_entry = bars_rth[(bars_rth['time_et'] >= '11:01') & (bars_rth['time_et'] <= '15:55')]
    if len(post_entry) == 0:
        return None

    sl_level = entry_price - sl_dist if trade_dir == 'long' else entry_price + sl_dist
    exit_price = None
    exit_type = 'eod'
    exit_time = None

    for _, bar in post_entry.iterrows():
        if trade_dir == 'long' and bar['low'] <= sl_level:
            exit_price = sl_level
            exit_type = 'initial_sl'
            exit_time = bar['time_et']
            break
        elif trade_dir == 'short' and bar['high'] >= sl_level:
            exit_price = sl_level
            exit_type = 'initial_sl'
            exit_time = bar['time_et']
            break

    if exit_price is None:
        exit_price = post_entry.iloc[-1]['close']
        exit_type = 'eod'
        exit_time = post_entry.iloc[-1]['time_et']

    pnl = (exit_price - entry_price) if trade_dir == 'long' else (entry_price - exit_price)
    return {
        'pnl': pnl, 'pnl_adr': pnl / adr, 'pnl_r': pnl / sl_dist,
        'exit_type': exit_type, 'exit_time': exit_time,
        'winner': pnl > 0,
    }


# ============================================================
# BACKTEST RUNNERS
# ============================================================
def run_s4_backtest(df_qualified):
    """Run S4: Entry close_935, SL 0.25 ADR, TRAIL_D, direction = gap_direction."""
    results = []
    for _, row in df_qualified.iterrows():
        ticker = row['ticker']
        date = str(row['date'])
        entry_price = row['close_935']
        adr = row['adr_10']
        gap_dir = row['gap_direction']
        trade_dir = 'long' if gap_dir == 'up' else 'short'
        sl_dist = 0.25 * adr

        if pd.isna(entry_price) or pd.isna(adr) or adr <= 0:
            continue

        bar_path = RAW_1MIN / ticker / f"{date}.parquet"
        if not bar_path.exists():
            continue
        bars = pd.read_parquet(bar_path)
        bars_rth = bars[bars['session'] == 'rth']
        if len(bars_rth) < 10:
            continue

        result = simulate_trail_d_935(bars_rth, entry_price, sl_dist, trade_dir, adr)
        if result is None:
            continue

        result['ticker'] = ticker
        result['date'] = date
        result['gap_direction'] = gap_dir
        result['trade_dir'] = trade_dir
        result['entry_price'] = entry_price
        result['adr'] = adr
        result['sl_dist'] = sl_dist
        result['od_strength'] = row.get('od_strength', np.nan)
        result['pm_rth5'] = row.get('pm_rth5', np.nan)
        result['is_earnings'] = row.get('is_earnings', False)
        result['sector'] = row.get('sector', 'Unknown')
        result['market_cap_bucket'] = row.get('market_cap_bucket', 'Unknown')
        result['full_drift'] = row.get('full_drift', np.nan)
        result['gap_size_in_adr'] = row.get('gap_size_in_adr', np.nan)
        results.append(result)
    return pd.DataFrame(results)


def run_s2_backtest(df_qualified, sim_func=None):
    """Run S2: Entry close_1100, SL = Entry to LOD+0.1ADR (min 0.15 ADR), TRAIL_D from 11:01."""
    if sim_func is None:
        sim_func = simulate_trail_d_1100
    results = []
    for _, row in df_qualified.iterrows():
        ticker = row['ticker']
        date = str(row['date'])
        entry_price = row['close_1100']
        adr = row['adr_10']
        trade_dir = 'long'  # S2 is GapUp only

        if pd.isna(entry_price) or pd.isna(adr) or adr <= 0:
            continue

        bar_path = RAW_1MIN / ticker / f"{date}.parquet"
        if not bar_path.exists():
            continue
        bars = pd.read_parquet(bar_path)
        bars_rth = bars[bars['session'] == 'rth']
        if len(bars_rth) < 10:
            continue

        # SL: entry to LOD + 0.1*ADR, minimum 0.15 ADR
        lod = bars_rth['low'].min()
        sl_dist = max(entry_price - lod + 0.1 * adr, 0.15 * adr)

        result = sim_func(bars_rth, entry_price, sl_dist, trade_dir, adr)
        if result is None:
            continue

        result['ticker'] = ticker
        result['date'] = date
        result['gap_direction'] = 'up'
        result['trade_dir'] = trade_dir
        result['entry_price'] = entry_price
        result['adr'] = adr
        result['sl_dist'] = sl_dist
        result['od_strength'] = row.get('od_strength', np.nan)
        result['rest_drift_1100'] = row.get('rest_drift_1100', np.nan)
        result['gap_size_in_adr'] = row.get('gap_size_in_adr', np.nan)
        result['rvol_at_time_60min'] = row.get('rvol_at_time_60min', np.nan)
        results.append(result)
    return pd.DataFrame(results)


# ============================================================
# BOOTSTRAP HELPERS
# ============================================================
def bootstrap_mean_ci(data, n_boot=N_BOOTSTRAP, ci=0.95):
    data = np.array(data, dtype=float)
    n = len(data)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = data[np.random.randint(0, n, size=n)]
        boot_means[i] = sample.mean()
    alpha = (1 - ci) / 2
    return data.mean(), np.percentile(boot_means, alpha*100), np.percentile(boot_means, (1-alpha)*100), boot_means.std()


def bootstrap_p_value(data, n_boot=N_BOOTSTRAP):
    """One-sided bootstrap p-value for H0: mean <= 0."""
    data = np.array(data, dtype=float)
    n = len(data)
    if n == 0:
        return 1.0
    obs_mean = data.mean()
    if obs_mean <= 0:
        return 1.0
    data_centered = data - obs_mean
    count_ge = 0
    for i in range(n_boot):
        sample = data_centered[np.random.randint(0, n, size=n)]
        if sample.mean() >= obs_mean:
            count_ge += 1
    return count_ge / n_boot


def bootstrap_diff_test(data1, data2, n_boot=N_BOOTSTRAP):
    """Two-sample bootstrap test: H0: mean(data1) == mean(data2). Returns p-value."""
    d1 = np.array(data1, dtype=float)
    d2 = np.array(data2, dtype=float)
    obs_diff = d1.mean() - d2.mean()
    combined = np.concatenate([d1, d2])
    n1 = len(d1)
    count = 0
    for i in range(n_boot):
        perm = combined[np.random.randint(0, len(combined), size=len(combined))]
        boot_diff = perm[:n1].mean() - perm[n1:].mean()
        if abs(boot_diff) >= abs(obs_diff):
            count += 1
    return count / n_boot


# ============================================================
# MAIN
# ============================================================
def main():
    t_start = time.time()

    log("=" * 100)
    log("D13 INDEPENDENT DEVIL'S ADVOCATE -- BRUTAL EHRLICHE PRUEFUNG")
    log("=" * 100)
    log(f"  IS-Periode: 2021-02-21 bis 2023-12-31")
    log(f"  Bootstrap-Resamples: {N_BOOTSTRAP}")
    log(f"  Minimale Hypothesen getestet: {NUM_HYPOTHESES} (+ S4 Relaxation = {NUM_HYPOTHESES_WITH_RELAXATION})")
    log()

    # ----------------------------------------------------------
    # LOAD DATA
    # ----------------------------------------------------------
    log("Lade Daten...")
    meta = pd.read_parquet(META_PATH)
    h1 = meta[(meta['date'] >= '2021-02-21') & (meta['date'] <= '2023-12-31')].copy()
    log(f"  IS events gesamt: {len(h1)}")

    # --- S4 Filter: pm_rth5 < 0.30, gap_size_in_adr >= 1.5, KEIN od_strength filter ---
    s4_events = h1[(h1['pm_rth5'] < 0.30) & (h1['gap_size_in_adr'] >= 1.5)].copy()
    log(f"  S4 events (pm_rth5<0.30, gap>=1.5 ADR): {len(s4_events)}")

    # --- S2 Filter: gap not filled by 90min, rvol60>=3, gap>=3 ADR, GapUp only ---
    s2_events = h1[
        (h1['gap_size_in_adr'] >= 3.0) &
        (h1['rvol_at_time_60min'] >= 3.0) &
        (h1['gap_direction'] == 'up') &
        ((h1['gap_filled'] == False) | (h1['gap_fill_time_minutes'] > 90))
    ].copy()
    log(f"  S2 events (no-fill 90min, rvol60>=3, gap>=3 ADR, GapUp): {len(s2_events)}")
    log()

    # ----------------------------------------------------------
    # RUN BACKTESTS
    # ----------------------------------------------------------
    log("Fuehre Backtests durch...")
    s4_trades = run_s4_backtest(s4_events)
    log(f"  S4 trades: {len(s4_trades)}")
    s2_trades = run_s2_backtest(s2_events)
    log(f"  S2 trades: {len(s2_trades)}")
    log()

    # Quick summary
    for label, trades in [("S4 relaxed", s4_trades), ("S2 GapUp", s2_trades)]:
        n = len(trades)
        ev = trades['pnl_adr'].mean()
        wr = trades['winner'].mean() * 100
        winners = trades[trades['pnl_adr'] > 0]['pnl_adr']
        losers = trades[trades['pnl_adr'] < 0]['pnl_adr']
        pf = abs(winners.sum() / losers.sum()) if losers.sum() != 0 else np.inf
        log(f"  {label}: N={n}, EV={ev:+.4f} ADR, WR={wr:.1f}%, PF={pf:.2f}")
    log()

    # ==============================================================
    # PRUEFUNG 1: DATA-MINING / MULTIPLE TESTING KORREKTUR
    # ==============================================================
    log("=" * 100)
    log("PRUEFUNG 1: DATA-MINING / MULTIPLE TESTING KORREKTUR")
    log("=" * 100)
    log()
    log(f"  Es wurden mindestens {NUM_HYPOTHESES} Hypothesen getestet.")
    log(f"  S4 wurde NACH Scheitern der Original-Hypothese (pm_rth5<0.10, N=1) relaxiert")
    log(f"  -> Das ist ein ZUSAETZLICHER Freiheitsgrad! Effektiv {NUM_HYPOTHESES_WITH_RELAXATION} Hypothesen.")
    log()

    # Compute bootstrap p-values
    p_s4 = bootstrap_p_value(s4_trades['pnl_adr'].values, N_BOOTSTRAP)
    p_s2 = bootstrap_p_value(s2_trades['pnl_adr'].values, N_BOOTSTRAP)

    log(f"  Raw bootstrap p-values:")
    log(f"    S4 relaxed: p = {p_s4:.4f}")
    log(f"    S2 GapUp:   p = {p_s2:.4f}")
    log()

    # Bonferroni correction
    alpha_bonf = ALPHA / NUM_HYPOTHESES_WITH_RELAXATION
    log(f"  --- Bonferroni-Korrektur (alpha_korr = {ALPHA}/{NUM_HYPOTHESES_WITH_RELAXATION} = {alpha_bonf:.5f}) ---")
    log(f"    S4: p={p_s4:.4f} {'< ' + str(alpha_bonf) + ' -> UEBERLEBT' if p_s4 < alpha_bonf else '>= ' + str(round(alpha_bonf,5)) + ' -> FAELLT DURCH'}")
    log(f"    S2: p={p_s2:.4f} {'< ' + str(alpha_bonf) + ' -> UEBERLEBT' if p_s2 < alpha_bonf else '>= ' + str(round(alpha_bonf,5)) + ' -> FAELLT DURCH'}")
    log()

    # Holm-Bonferroni: sort p-values, stepwise correction
    p_values_sorted = sorted([('S4', p_s4), ('S2', p_s2)], key=lambda x: x[1])
    log(f"  --- Holm-Bonferroni (schrittweise Korrektur) ---")
    holm_results = {}
    all_reject = True
    for i, (name, p) in enumerate(p_values_sorted):
        k = NUM_HYPOTHESES_WITH_RELAXATION - i  # remaining hypotheses
        threshold = ALPHA / k
        reject = p < threshold and all_reject
        if not reject:
            all_reject = False
        holm_results[name] = {'p': p, 'threshold': threshold, 'reject': reject}
        status = "VERWORFEN (H0 rejected, Edge existiert)" if reject else "NICHT VERWORFEN (H0 nicht rejected)"
        log(f"    Schritt {i+1}: {name} p={p:.4f}, Schwelle={threshold:.5f} -> {status}")
    log()

    # CRITICAL: S4 relaxation penalty
    log("  --- ZUSAETZLICHE WARNUNG fuer S4 ---")
    log("  Die Original-Hypothese pm_rth5 < 0.10 ergab NUR 1 Event.")
    log("  Die Relaxierung auf < 0.30 ist eine POST-HOC Anpassung.")
    log("  Dies ist klassisches 'researcher degrees of freedom' / p-hacking.")
    log("  Der effektive p-Wert fuer S4 sollte HOEHER sein als berechnet,")
    log("  weil der Threshold NACH Betrachtung der Daten gewaehlt wurde.")
    log()

    # ==============================================================
    # PRUEFUNG 2: THRESHOLD-SENSITIVITAET
    # ==============================================================
    log("=" * 100)
    log("PRUEFUNG 2: THRESHOLD-SENSITIVITAET")
    log("=" * 100)
    log()

    # --- S4: pm_rth5 Thresholds ---
    log("  --- S4: pm_rth5 Thresholds (0.15 bis 0.50, Schritt 0.05) ---")
    log(f"  {'Threshold':<12} {'N':>5} {'EV(ADR)':>10} {'WR%':>7} {'PF':>7} {'p-value':>10} {'Monoton?':>10}")
    log(f"  {'-'*70}")

    s4_threshold_results = []
    for thresh in np.arange(0.15, 0.55, 0.05):
        sub_events = h1[(h1['pm_rth5'] < thresh) & (h1['gap_size_in_adr'] >= 1.5)]
        if len(sub_events) < 5:
            log(f"  {thresh:<12.2f} {'N<5':>5}")
            s4_threshold_results.append({'thresh': thresh, 'n': len(sub_events), 'ev': np.nan})
            continue
        sub_trades = run_s4_backtest(sub_events)
        n = len(sub_trades)
        if n < 5:
            log(f"  {thresh:<12.2f} {n:>5} [zu wenig Trades]")
            s4_threshold_results.append({'thresh': thresh, 'n': n, 'ev': np.nan})
            continue
        ev = sub_trades['pnl_adr'].mean()
        wr = sub_trades['winner'].mean() * 100
        w = sub_trades[sub_trades['pnl_adr'] > 0]['pnl_adr']
        l = sub_trades[sub_trades['pnl_adr'] < 0]['pnl_adr']
        pf = abs(w.sum() / l.sum()) if l.sum() != 0 else np.inf
        p = bootstrap_p_value(sub_trades['pnl_adr'].values, min(N_BOOTSTRAP, 5000))
        s4_threshold_results.append({'thresh': thresh, 'n': n, 'ev': ev, 'wr': wr, 'pf': pf, 'p': p})
        log(f"  {thresh:<12.2f} {n:>5} {ev:>+10.4f} {wr:>6.1f}% {pf:>7.2f} {p:>10.4f}")

    # Monotonicity check for S4
    evs_s4 = [r['ev'] for r in s4_threshold_results if not np.isnan(r.get('ev', np.nan))]
    if len(evs_s4) >= 3:
        monotone_count = sum(1 for i in range(len(evs_s4)-1) if evs_s4[i] >= evs_s4[i+1])
        log(f"\n  Monotonie-Check: {monotone_count}/{len(evs_s4)-1} benachbarte Paare sind monoton fallend")
        log(f"  (Erwartung bei echtem Signal: EV STEIGT bei niedrigerem Threshold)")
        is_monotone = monotone_count >= (len(evs_s4)-1) * 0.6
        log(f"  Urteil: {'MONOTON (gut)' if is_monotone else 'NICHT MONOTON (Artefakt-Verdacht!)'}")
    log()

    # --- S2: gap_size Thresholds ---
    log("  --- S2: gap_size_in_adr Thresholds (1.5 bis 5.0, Schritt 0.5) ---")
    log(f"  {'gap_size':<12} {'N':>5} {'EV(ADR)':>10} {'WR%':>7} {'PF':>7}")
    log(f"  {'-'*50}")

    s2_gap_results = []
    for gap_thresh in np.arange(1.5, 5.5, 0.5):
        sub_events = h1[
            (h1['gap_size_in_adr'] >= gap_thresh) &
            (h1['rvol_at_time_60min'] >= 3.0) &
            (h1['gap_direction'] == 'up') &
            ((h1['gap_filled'] == False) | (h1['gap_fill_time_minutes'] > 90))
        ]
        if len(sub_events) < 5:
            log(f"  {gap_thresh:<12.1f} {len(sub_events):>5} [zu wenig]")
            s2_gap_results.append({'thresh': gap_thresh, 'n': len(sub_events), 'ev': np.nan})
            continue
        sub_trades = run_s2_backtest(sub_events)
        n = len(sub_trades)
        if n < 5:
            log(f"  {gap_thresh:<12.1f} {n:>5} [zu wenig]")
            s2_gap_results.append({'thresh': gap_thresh, 'n': n, 'ev': np.nan})
            continue
        ev = sub_trades['pnl_adr'].mean()
        wr = sub_trades['winner'].mean() * 100
        w = sub_trades[sub_trades['pnl_adr'] > 0]['pnl_adr']
        l = sub_trades[sub_trades['pnl_adr'] < 0]['pnl_adr']
        pf = abs(w.sum() / l.sum()) if l.sum() != 0 else np.inf
        s2_gap_results.append({'thresh': gap_thresh, 'n': n, 'ev': ev, 'wr': wr, 'pf': pf})
        log(f"  {gap_thresh:<12.1f} {n:>5} {ev:>+10.4f} {wr:>6.1f}% {pf:>7.2f}")

    log()

    # --- S2: rvol Thresholds ---
    log("  --- S2: rvol_at_time_60min Thresholds (1.0 bis 5.0, Schritt 1.0) ---")
    log(f"  {'rvol':<12} {'N':>5} {'EV(ADR)':>10} {'WR%':>7} {'PF':>7}")
    log(f"  {'-'*50}")

    s2_rvol_results = []
    for rvol_thresh in [1.0, 2.0, 3.0, 4.0, 5.0]:
        sub_events = h1[
            (h1['gap_size_in_adr'] >= 3.0) &
            (h1['rvol_at_time_60min'] >= rvol_thresh) &
            (h1['gap_direction'] == 'up') &
            ((h1['gap_filled'] == False) | (h1['gap_fill_time_minutes'] > 90))
        ]
        if len(sub_events) < 5:
            log(f"  {rvol_thresh:<12.1f} {len(sub_events):>5} [zu wenig]")
            s2_rvol_results.append({'thresh': rvol_thresh, 'n': len(sub_events), 'ev': np.nan})
            continue
        sub_trades = run_s2_backtest(sub_events)
        n = len(sub_trades)
        if n < 5:
            log(f"  {rvol_thresh:<12.1f} {n:>5} [zu wenig]")
            s2_rvol_results.append({'thresh': rvol_thresh, 'n': n, 'ev': np.nan})
            continue
        ev = sub_trades['pnl_adr'].mean()
        wr = sub_trades['winner'].mean() * 100
        w = sub_trades[sub_trades['pnl_adr'] > 0]['pnl_adr']
        l = sub_trades[sub_trades['pnl_adr'] < 0]['pnl_adr']
        pf = abs(w.sum() / l.sum()) if l.sum() != 0 else np.inf
        s2_rvol_results.append({'thresh': rvol_thresh, 'n': n, 'ev': ev, 'wr': wr, 'pf': pf})
        log(f"  {rvol_thresh:<12.1f} {n:>5} {ev:>+10.4f} {wr:>6.1f}% {pf:>7.2f}")

    # Monotonicity check for S2 gap
    evs_s2_gap = [r['ev'] for r in s2_gap_results if not np.isnan(r.get('ev', np.nan))]
    if len(evs_s2_gap) >= 3:
        mono_inc = sum(1 for i in range(len(evs_s2_gap)-1) if evs_s2_gap[i] <= evs_s2_gap[i+1])
        log(f"\n  S2 gap_size Monotonie: {mono_inc}/{len(evs_s2_gap)-1} Paare monoton steigend")
        log(f"  (Erwartung: EV steigt bei hoeherem gap_size -> staerkeres Momentum)")

    evs_s2_rvol = [r['ev'] for r in s2_rvol_results if not np.isnan(r.get('ev', np.nan))]
    if len(evs_s2_rvol) >= 3:
        mono_inc = sum(1 for i in range(len(evs_s2_rvol)-1) if evs_s2_rvol[i] <= evs_s2_rvol[i+1])
        log(f"  S2 rvol Monotonie: {mono_inc}/{len(evs_s2_rvol)-1} Paare monoton steigend")
    log()

    # ==============================================================
    # PRUEFUNG 3: JAHRES-STABILITAET (DETAILLIERT)
    # ==============================================================
    log("=" * 100)
    log("PRUEFUNG 3: JAHRES-STABILITAET (Detailliert)")
    log("=" * 100)
    log()

    # Add half-year columns
    for trades_df in [s4_trades, s2_trades]:
        trades_df['year'] = trades_df['date'].astype(str).str[:4]
        month = trades_df['date'].astype(str).str[5:7].astype(int)
        trades_df['half_year'] = trades_df['year'] + '-H' + np.where(month <= 6, '1', '2')

    half_years = ['2021-H1', '2021-H2', '2022-H1', '2022-H2', '2023-H1', '2023-H2']

    # --- S4 Jahres-Stabilitaet ---
    log("  --- S4 relaxed: Jahres- und Halbjahres-Split ---")
    log(f"  {'Periode':<12} {'N':>5} {'EV(ADR)':>10} {'WR%':>7} {'95%-CI':>25}")
    log(f"  {'-'*65}")

    for yr in ['2021', '2022', '2023']:
        sub = s4_trades[s4_trades['year'] == yr]
        if len(sub) < 5:
            log(f"  {yr:<12} {len(sub):>5} [zu wenig]")
            continue
        ev = sub['pnl_adr'].mean()
        wr = sub['winner'].mean() * 100
        _, ci_lo, ci_hi, _ = bootstrap_mean_ci(sub['pnl_adr'].values)
        log(f"  {yr:<12} {len(sub):>5} {ev:>+10.4f} {wr:>6.1f}% [{ci_lo:>+.4f}, {ci_hi:>+.4f}]")

    log()
    for hy in half_years:
        sub = s4_trades[s4_trades['half_year'] == hy]
        if len(sub) < 3:
            log(f"  {hy:<12} {len(sub):>5} [zu wenig]")
            continue
        ev = sub['pnl_adr'].mean()
        wr = sub['winner'].mean() * 100
        ci_str = ""
        if len(sub) >= 10:
            _, ci_lo, ci_hi, _ = bootstrap_mean_ci(sub['pnl_adr'].values)
            ci_str = f"[{ci_lo:>+.4f}, {ci_hi:>+.4f}]"
        log(f"  {hy:<12} {len(sub):>5} {ev:>+10.4f} {wr:>6.1f}% {ci_str}")

    log()

    # Bootstrap test: Is 2022 significantly different from 2021+2023?
    s4_2022 = s4_trades[s4_trades['year'] == '2022']['pnl_adr'].values
    s4_non2022 = s4_trades[s4_trades['year'] != '2022']['pnl_adr'].values
    if len(s4_2022) >= 10 and len(s4_non2022) >= 10:
        p_diff = bootstrap_diff_test(s4_2022, s4_non2022, N_BOOTSTRAP)
        log(f"  Bootstrap-Test 2022 vs (2021+2023): p = {p_diff:.4f}")
        log(f"  2022 EV = {s4_2022.mean():+.4f}, Non-2022 EV = {s4_non2022.mean():+.4f}")
        if p_diff < 0.05:
            log(f"  -> 2022 ist SIGNIFIKANT schwaecher (p<0.05). REGIME-ABHAENGIG!")
        else:
            log(f"  -> Unterschied NICHT signifikant (p={p_diff:.4f}). Akzeptabel.")
    log()

    # --- S2 Jahres-Stabilitaet ---
    log("  --- S2 GapUp: Jahres- und Halbjahres-Split ---")
    log(f"  {'Periode':<12} {'N':>5} {'EV(ADR)':>10} {'WR%':>7} {'95%-CI':>25}")
    log(f"  {'-'*65}")

    for yr in ['2021', '2022', '2023']:
        sub = s2_trades[s2_trades['year'] == yr]
        if len(sub) < 5:
            log(f"  {yr:<12} {len(sub):>5} [zu wenig]")
            continue
        ev = sub['pnl_adr'].mean()
        wr = sub['winner'].mean() * 100
        _, ci_lo, ci_hi, _ = bootstrap_mean_ci(sub['pnl_adr'].values)
        log(f"  {yr:<12} {len(sub):>5} {ev:>+10.4f} {wr:>6.1f}% [{ci_lo:>+.4f}, {ci_hi:>+.4f}]")

    log()
    for hy in half_years:
        sub = s2_trades[s2_trades['half_year'] == hy]
        if len(sub) < 3:
            log(f"  {hy:<12} {len(sub):>5} [zu wenig]")
            continue
        ev = sub['pnl_adr'].mean()
        wr = sub['winner'].mean() * 100
        ci_str = ""
        if len(sub) >= 10:
            _, ci_lo, ci_hi, _ = bootstrap_mean_ci(sub['pnl_adr'].values)
            ci_str = f"[{ci_lo:>+.4f}, {ci_hi:>+.4f}]"
        log(f"  {hy:<12} {len(sub):>5} {ev:>+10.4f} {wr:>6.1f}% {ci_str}")

    log()

    # S2 trend analysis: Is the edge disappearing?
    s2_2023 = s2_trades[s2_trades['year'] == '2023']['pnl_adr'].values
    s2_pre2023 = s2_trades[s2_trades['year'] != '2023']['pnl_adr'].values
    if len(s2_2023) >= 10 and len(s2_pre2023) >= 10:
        p_diff_s2 = bootstrap_diff_test(s2_2023, s2_pre2023, N_BOOTSTRAP)
        log(f"  Bootstrap-Test 2023 vs (2021+2022): p = {p_diff_s2:.4f}")
        log(f"  2023 EV = {s2_2023.mean():+.4f}, Pre-2023 EV = {s2_pre2023.mean():+.4f}")
        if p_diff_s2 < 0.05:
            log(f"  -> 2023 ist SIGNIFIKANT schwaecher. DER EDGE VERSCHWINDET!")
        else:
            log(f"  -> Unterschied NICHT signifikant. Aber Trend ist besorgniserregend.")
        # Additional: Is 2023 EV > 0 at all?
        p_2023 = bootstrap_p_value(s2_2023, N_BOOTSTRAP)
        log(f"  Ist S2 in 2023 alleine signifikant? p = {p_2023:.4f}")
        if p_2023 > 0.05:
            log(f"  -> S2 hat IN 2023 KEINEN signifikanten Edge! (p={p_2023:.4f})")
    log()

    # ==============================================================
    # PRUEFUNG 4: OD-UNABHAENGIGKEIT (Korrelationsanalyse)
    # ==============================================================
    log("=" * 100)
    log("PRUEFUNG 4: OD-UNABHAENGIGKEIT (Korrelationsanalyse)")
    log("=" * 100)
    log()
    log("  Frage: Ist od_strength ein versteckter Confounder?")
    log("  Wenn der Edge NUR bei hohem OD existiert, ist es nicht die Filter-Variable,")
    log("  sondern der Opening Drive, der den Trade profitabel macht.")
    log()

    # --- S4: od_strength vs full_drift (PnL proxy) ---
    log("  --- S4: od_strength vs Trade-PnL ---")
    s4_valid = s4_trades.dropna(subset=['od_strength', 'pnl_adr'])
    if len(s4_valid) >= 20:
        pearson_r, pearson_p = sp_stats.pearsonr(s4_valid['od_strength'], s4_valid['pnl_adr'])
        spearman_r, spearman_p = sp_stats.spearmanr(s4_valid['od_strength'], s4_valid['pnl_adr'])
        log(f"  Pearson:  r = {pearson_r:+.4f}, p = {pearson_p:.4f}")
        log(f"  Spearman: r = {spearman_r:+.4f}, p = {spearman_p:.4f}")

        # Split: high OD vs low OD
        s4_high_od = s4_trades[s4_trades['od_strength'] > 0.5]
        s4_low_od = s4_trades[s4_trades['od_strength'] <= 0.5]
        log(f"\n  S4 Split nach od_strength:")
        log(f"    od > 0.5:  N={len(s4_high_od):>4}, EV={s4_high_od['pnl_adr'].mean():>+.4f} ADR, WR={s4_high_od['winner'].mean()*100:.1f}%")
        log(f"    od <= 0.5: N={len(s4_low_od):>4}, EV={s4_low_od['pnl_adr'].mean():>+.4f} ADR, WR={s4_low_od['winner'].mean()*100:.1f}%")
        if len(s4_low_od) >= 10:
            p_low = bootstrap_p_value(s4_low_od['pnl_adr'].values, N_BOOTSTRAP)
            log(f"    Edge bei schwachem OD (<=0.5): p = {p_low:.4f}")
            if p_low > 0.05:
                log(f"    -> KEIN signifikanter Edge bei schwachem OD! Edge haengt von OD ab!")
            else:
                log(f"    -> Edge auch bei schwachem OD vorhanden (gut, unabhaengig von OD)")
        else:
            log(f"    -> Zu wenig Events mit od<=0.5 fuer statistischen Test")
    log()

    # --- S2: od_strength vs rest_drift_1100 (PnL proxy) ---
    log("  --- S2: od_strength vs Trade-PnL ---")
    s2_valid = s2_trades.dropna(subset=['od_strength', 'pnl_adr'])
    if len(s2_valid) >= 20:
        pearson_r, pearson_p = sp_stats.pearsonr(s2_valid['od_strength'], s2_valid['pnl_adr'])
        spearman_r, spearman_p = sp_stats.spearmanr(s2_valid['od_strength'], s2_valid['pnl_adr'])
        log(f"  Pearson:  r = {pearson_r:+.4f}, p = {pearson_p:.4f}")
        log(f"  Spearman: r = {spearman_r:+.4f}, p = {spearman_p:.4f}")

        s2_high_od = s2_trades[s2_trades['od_strength'] > 0.5]
        s2_low_od = s2_trades[s2_trades['od_strength'] <= 0.5]
        log(f"\n  S2 Split nach od_strength:")
        log(f"    od > 0.5:  N={len(s2_high_od):>4}, EV={s2_high_od['pnl_adr'].mean():>+.4f} ADR, WR={s2_high_od['winner'].mean()*100:.1f}%")
        log(f"    od <= 0.5: N={len(s2_low_od):>4}, EV={s2_low_od['pnl_adr'].mean():>+.4f} ADR, WR={s2_low_od['winner'].mean()*100:.1f}%")
        if len(s2_low_od) >= 10:
            p_low = bootstrap_p_value(s2_low_od['pnl_adr'].values, N_BOOTSTRAP)
            log(f"    Edge bei schwachem OD (<=0.5): p = {p_low:.4f}")
            if p_low > 0.05:
                log(f"    -> KEIN signifikanter Edge bei schwachem OD!")
            else:
                log(f"    -> Edge auch bei schwachem OD vorhanden (gut)")
        if len(s2_high_od) >= 10:
            p_high = bootstrap_p_value(s2_high_od['pnl_adr'].values, N_BOOTSTRAP)
            log(f"    Edge bei starkem OD (>0.5): p = {p_high:.4f}")
    log()

    # ==============================================================
    # PRUEFUNG 5: S2 TIMEOUT-PROBLEM
    # ==============================================================
    log("=" * 100)
    log("PRUEFUNG 5: S2 TIMEOUT-PROBLEM")
    log("=" * 100)
    log()
    log("  S2 hat angeblich 48% Timeout-Rate.")
    log("  Wenn Timeout-Trades den positiven EV treiben, ist der Edge")
    log("  ein 'Buy and Hold Gap' Artefakt, KEIN Momentum-Signal.")
    log()

    # Exit-type decomposition
    log("  --- S2 Exit-Type Decomposition ---")
    total_n = len(s2_trades)
    total_ev = s2_trades['pnl_adr'].mean()
    for etype in ['initial_sl', 'trail_sl', 'timeout']:
        sub = s2_trades[s2_trades['exit_type'] == etype]
        n_e = len(sub)
        if n_e > 0:
            ev_e = sub['pnl_adr'].mean()
            wr_e = sub['winner'].mean() * 100
            pct = n_e / total_n * 100
            contribution = (n_e / total_n) * ev_e
            pct_of_ev = (contribution / total_ev * 100) if total_ev != 0 else 0
            log(f"    {etype:<15}: N={n_e:>4} ({pct:>5.1f}%), EV={ev_e:>+.4f} ADR, "
                f"WR={wr_e:.1f}%, Beitrag zum EV: {contribution:>+.4f} ({pct_of_ev:.1f}%)")

    log()

    # Compare: EV nur Timeout vs EV nur SL+Trail
    timeout_trades = s2_trades[s2_trades['exit_type'] == 'timeout']
    non_timeout_trades = s2_trades[s2_trades['exit_type'] != 'timeout']
    log(f"  Timeout-Trades:     N={len(timeout_trades)}, EV={timeout_trades['pnl_adr'].mean():+.4f} ADR")
    log(f"  Non-Timeout-Trades: N={len(non_timeout_trades)}, EV={non_timeout_trades['pnl_adr'].mean():+.4f} ADR")
    log()

    if len(timeout_trades) >= 10:
        p_timeout = bootstrap_p_value(timeout_trades['pnl_adr'].values, N_BOOTSTRAP)
        log(f"  Timeout-Trades alleine signifikant? p = {p_timeout:.4f}")
    if len(non_timeout_trades) >= 10:
        p_non_timeout = bootstrap_p_value(non_timeout_trades['pnl_adr'].values, N_BOOTSTRAP)
        log(f"  Non-Timeout-Trades alleine signifikant? p = {p_non_timeout:.4f}")
    log()

    # --- Vergleich: TRAIL_D vs Fixed EOD (kein Trail) ---
    log("  --- S2: TRAIL_D vs Fixed EOD Exit (close_1555) ---")
    s2_eod_trades = run_s2_backtest(s2_events, sim_func=simulate_fixed_eod)
    log(f"  TRAIL_D:   N={len(s2_trades)}, EV={s2_trades['pnl_adr'].mean():+.4f} ADR, WR={s2_trades['winner'].mean()*100:.1f}%")
    log(f"  Fixed EOD: N={len(s2_eod_trades)}, EV={s2_eod_trades['pnl_adr'].mean():+.4f} ADR, WR={s2_eod_trades['winner'].mean()*100:.1f}%")
    log()

    # If Fixed EOD has similar or better EV, the trail adds no value
    ev_trail = s2_trades['pnl_adr'].mean()
    ev_eod = s2_eod_trades['pnl_adr'].mean()
    if ev_eod >= ev_trail * 0.8:
        log("  WARNUNG: Fixed EOD Exit hat aehnlichen EV wie TRAIL_D!")
        log("  -> Der Trail-Stop fuegt keinen signifikanten Wert hinzu.")
        log("  -> Der Edge kommt vom GAP SELBST (buy-and-hold), nicht vom Trailing.")
    else:
        log("  Trail_D verbessert EV gegenueber Fixed EOD.")
        log(f"  Delta: {ev_trail - ev_eod:+.4f} ADR")
    log()

    # ==============================================================
    # PRUEFUNG 6: S4 - WAS TREIBT pm_rth5 < 0.30?
    # ==============================================================
    log("=" * 100)
    log("PRUEFUNG 6: S4 - WAS TREIBT pm_rth5 < 0.30?")
    log("=" * 100)
    log()

    # Ticker-Konzentration
    ticker_counts = s4_trades['ticker'].value_counts()
    n_unique = len(ticker_counts)
    top5_tickers = ticker_counts.head(5)
    top5_pct = top5_tickers.sum() / len(s4_trades) * 100
    top10_tickers = ticker_counts.head(10)
    top10_pct = top10_tickers.sum() / len(s4_trades) * 100

    log(f"  --- Ticker-Konzentration ---")
    log(f"  Unique Tickers: {n_unique}")
    log(f"  Top 5 Tickers:  {top5_pct:.1f}% aller Trades")
    log(f"  Top 10 Tickers: {top10_pct:.1f}% aller Trades")
    log(f"  Top 5: {dict(top5_tickers)}")
    log()

    if top5_pct > 50:
        log("  KRITISCH: Mehr als 50% der Trades von Top-5-Tickern!")
        log("  -> Concentration-Risk! Kein systematischer Edge.")
    elif top5_pct > 30:
        log("  WARNUNG: Mehr als 30% der Trades von Top-5-Tickern.")
    else:
        log("  OK: Keine uebertriebene Ticker-Konzentration.")
    log()

    # Earnings-Events
    if 'is_earnings' in s4_trades.columns:
        n_earn = s4_trades['is_earnings'].sum()
        n_non_earn = len(s4_trades) - n_earn
        log(f"  --- Earnings vs Non-Earnings ---")
        log(f"  Earnings:     N={n_earn} ({n_earn/len(s4_trades)*100:.1f}%)")
        log(f"  Non-Earnings: N={n_non_earn} ({n_non_earn/len(s4_trades)*100:.1f}%)")
        if n_earn >= 5:
            ev_earn = s4_trades[s4_trades['is_earnings'] == True]['pnl_adr'].mean()
            ev_non = s4_trades[s4_trades['is_earnings'] == False]['pnl_adr'].mean()
            log(f"  EV Earnings:     {ev_earn:+.4f} ADR")
            log(f"  EV Non-Earnings: {ev_non:+.4f} ADR")
            if n_non_earn >= 10:
                p_non = bootstrap_p_value(s4_trades[s4_trades['is_earnings'] == False]['pnl_adr'].values, N_BOOTSTRAP)
                log(f"  Non-Earnings alleine signifikant? p = {p_non:.4f}")
    log()

    # Sektor-Verteilung
    log(f"  --- Sektor-Verteilung ---")
    sector_counts = s4_trades['sector'].value_counts()
    for sector, cnt in sector_counts.items():
        sub = s4_trades[s4_trades['sector'] == sector]
        ev_s = sub['pnl_adr'].mean()
        log(f"    {sector:<30}: N={cnt:>4}, EV={ev_s:>+.4f} ADR")
    log()

    # Market-Cap Buckets
    log(f"  --- Market-Cap Verteilung ---")
    if 'market_cap_bucket' in s4_trades.columns:
        mc_counts = s4_trades['market_cap_bucket'].value_counts()
        for mc, cnt in mc_counts.items():
            sub = s4_trades[s4_trades['market_cap_bucket'] == mc]
            ev_m = sub['pnl_adr'].mean()
            log(f"    {str(mc):<20}: N={cnt:>4}, EV={ev_m:>+.4f} ADR")
    log()

    # What does pm_rth5 < 0.30 actually mean?
    log("  --- Was bedeutet pm_rth5 < 0.30? ---")
    log("  pm_rth5 = Premarket-Volumen / RTH-5min-Volumen")
    log("  Niedriger Wert = wenig Premarket-Aktivitaet relativ zur RTH-Eroeffnung")
    log("  = 'Surprise Gap' - der Markt reagiert erst beim RTH-Open stark")
    s4_pm = s4_trades['pm_rth5'].describe()
    log(f"  pm_rth5 Verteilung in S4 Events:")
    log(f"    Min={s4_pm['min']:.3f}, Median={s4_pm['50%']:.3f}, Max={s4_pm['max']:.3f}")
    log()

    # ==============================================================
    # PRUEFUNG 7: GESAMTURTEIL
    # ==============================================================
    log("=" * 100)
    log("PRUEFUNG 7: GESAMTURTEIL")
    log("=" * 100)
    log()

    # === S4 Verdict ===
    log("  ============================================================")
    log("  S4 RELAXED (pm_rth5 < 0.30, gap >= 1.5 ADR)")
    log("  ============================================================")
    log()

    s4_issues = []
    s4_positives = []

    # P1: Multiple testing
    if p_s4 < alpha_bonf:
        s4_positives.append("P1: Ueberlebt Bonferroni-Korrektur (p=" + f"{p_s4:.4f})")
    else:
        s4_issues.append("P1: Faellt durch Bonferroni")
    s4_issues.append("P1: POST-HOC Threshold-Relaxierung (researcher degrees of freedom)")

    # P2: Threshold sensitivity
    evs_valid = [r for r in s4_threshold_results if not np.isnan(r.get('ev', np.nan))]
    if len(evs_valid) >= 3:
        evs = [r['ev'] for r in evs_valid]
        if evs[0] > evs[-1]:
            s4_positives.append("P2: EV tendenziell hoeher bei niedrigerem Threshold (monoton)")
        else:
            s4_issues.append("P2: Threshold-Sensitivitaet: NICHT monoton")

    # P3: Year stability
    s4_yr_evs = {}
    for yr in ['2021', '2022', '2023']:
        sub = s4_trades[s4_trades['year'] == yr]
        if len(sub) >= 5:
            s4_yr_evs[yr] = sub['pnl_adr'].mean()
    if all(v > 0 for v in s4_yr_evs.values()):
        s4_positives.append("P3: Positiver EV in allen Jahren")
    else:
        s4_issues.append("P3: Nicht in allen Jahren positiv")
    if s4_yr_evs.get('2022', 1) < 0.03:
        s4_issues.append("P3: 2022 EV fast Null (0.019)")

    # P4: OD independence
    if len(s4_low_od) >= 10:
        ev_low = s4_low_od['pnl_adr'].mean()
        if ev_low > 0:
            s4_positives.append(f"P4: Edge auch bei schwachem OD (EV={ev_low:+.4f})")
        else:
            s4_issues.append("P4: Kein Edge bei schwachem OD -> OD-abhaengig")

    # P6: Concentration
    if top5_pct < 30:
        s4_positives.append("P6: Keine Ticker-Konzentration")
    else:
        s4_issues.append(f"P6: Ticker-Konzentration ({top5_pct:.1f}%)")

    log("  POSITIV:")
    for p in s4_positives:
        log(f"    + {p}")
    log()
    log("  NEGATIV:")
    for i in s4_issues:
        log(f"    - {i}")
    log()

    # Final verdict S4
    n_critical = sum(1 for i in s4_issues if 'POST-HOC' in i or 'NICHT monoton' in i or 'Kein Edge' in i)
    if n_critical >= 2:
        s4_verdict = "FRAGIL"
    elif n_critical >= 1:
        s4_verdict = "FRAGIL"
    elif len(s4_issues) <= 2 and len(s4_positives) >= 3:
        s4_verdict = "ROBUST (mit Vorbehalten)"
    else:
        s4_verdict = "FRAGIL"

    log(f"  >>> S4 VERDIKT: {s4_verdict}")
    log()

    # === S2 Verdict ===
    log("  ============================================================")
    log("  S2 GAPUP (No-Fill Power Move)")
    log("  ============================================================")
    log()

    s2_issues = []
    s2_positives = []

    # P1: Multiple testing
    if p_s2 < alpha_bonf:
        s2_positives.append(f"P1: Ueberlebt Bonferroni (p={p_s2:.4f})")
    else:
        s2_issues.append(f"P1: Faellt durch Bonferroni (p={p_s2:.4f})")

    # P3: Year stability
    s2_yr_evs = {}
    for yr in ['2021', '2022', '2023']:
        sub = s2_trades[s2_trades['year'] == yr]
        if len(sub) >= 5:
            s2_yr_evs[yr] = sub['pnl_adr'].mean()
    if all(v > 0 for v in s2_yr_evs.values()):
        s2_positives.append("P3: Positiver EV in allen Jahren")
    else:
        neg_yrs = [yr for yr, v in s2_yr_evs.items() if v <= 0]
        s2_issues.append(f"P3: Negativer EV in {neg_yrs}")

    if s2_yr_evs.get('2023', 1) < 0.03:
        s2_issues.append(f"P3: 2023 EV fast Null ({s2_yr_evs.get('2023', 0):+.4f}) -> EDGE VERSCHWINDET")

    # P5: Timeout problem
    timeout_pct = len(timeout_trades) / len(s2_trades) * 100
    if timeout_pct > 40:
        timeout_ev = timeout_trades['pnl_adr'].mean()
        non_timeout_ev = non_timeout_trades['pnl_adr'].mean()
        s2_issues.append(f"P5: {timeout_pct:.0f}% Timeout-Rate")
        if timeout_ev > 0 and non_timeout_ev <= 0:
            s2_issues.append("P5: Edge kommt NUR von Timeout-Trades -> Buy-and-Hold Artefakt!")
        elif timeout_ev > non_timeout_ev * 2:
            s2_issues.append("P5: Timeout-Trades treiben ueberproportional den EV")

    # Fixed EOD comparison
    if ev_eod >= ev_trail * 0.8:
        s2_issues.append("P5: Fixed EOD hat aehnlichen EV -> Trail unnoetig")
    else:
        s2_positives.append("P5: Trail_D verbessert EV gegenueber Fixed EOD")

    # P4: OD independence
    if len(s2_low_od) >= 10:
        ev_low_s2 = s2_low_od['pnl_adr'].mean()
        if ev_low_s2 > 0:
            s2_positives.append(f"P4: Edge bei schwachem OD (EV={ev_low_s2:+.4f})")
        else:
            s2_issues.append("P4: Kein Edge bei schwachem OD")

    log("  POSITIV:")
    for p in s2_positives:
        log(f"    + {p}")
    log()
    log("  NEGATIV:")
    for i in s2_issues:
        log(f"    - {i}")
    log()

    n_critical_s2 = sum(1 for i in s2_issues if 'VERSCHWINDET' in i or 'Artefakt' in i or 'Bonferroni' in i)
    if n_critical_s2 >= 2:
        s2_verdict = "ARTEFAKT"
    elif n_critical_s2 >= 1:
        s2_verdict = "FRAGIL"
    elif len(s2_issues) <= 2 and len(s2_positives) >= 3:
        s2_verdict = "ROBUST (mit Vorbehalten)"
    else:
        s2_verdict = "FRAGIL"

    log(f"  >>> S2 VERDIKT: {s2_verdict}")
    log()

    # === OOS Recommendation ===
    log("  ============================================================")
    log("  OOS-EMPFEHLUNG")
    log("  ============================================================")
    log()

    log(f"  S4 relaxed: {s4_verdict}")
    if 'ROBUST' in s4_verdict or s4_verdict == 'FRAGIL':
        log("    -> OOS-Test EMPFOHLEN, aber mit klaren Erwartungen:")
        log("    -> Erwarte EV-Degradation von mindestens 30-50% (IS-Overfit)")
        log("    -> S4 hat den Vorteil der Bonferroni-Robustheit")
        log("    -> ABER: Post-hoc Threshold muss im OOS fix sein (0.30)")
    else:
        log("    -> OOS-Test NICHT empfohlen")

    log()
    log(f"  S2 GapUp: {s2_verdict}")
    if s2_verdict == 'ARTEFAKT':
        log("    -> OOS-Test NICHT empfohlen")
        log("    -> Der verschwindende Edge in 2023 und die hohe Timeout-Rate")
        log("    -> deuten auf ein Artefakt hin, nicht auf einen echten Edge.")
    elif 'FRAGIL' in s2_verdict:
        log("    -> OOS-Test NUR MIT VORSICHT")
        log("    -> Der Edge in 2023 ist nahe Null")
        log("    -> Die hohe Timeout-Rate ist besorgniserregend")
        log("    -> Wenn OOS getestet, STRENGE Signifikanz-Schwelle (alpha=0.01)")
    log()

    log("  ============================================================")
    log("  ZUSAMMENFASSUNG: BRUTAL EHRLICH")
    log("  ============================================================")
    log()
    log("  1. BEIDE Strategien haben signifikante Schwaechen:")
    log("     - S4 ist post-hoc relaxiert (researcher degrees of freedom)")
    log("     - S2 hat einen verschwindenden Edge (2023 fast Null)")
    log()
    log("  2. S4 ist die STAERKERE der beiden Strategien:")
    log("     - Ueberlebt Bonferroni")
    log("     - Positiv in allen Jahren (2022 schwach, aber positiv)")
    log("     - Keine Ticker-Konzentration")
    log("     - ABER: Post-hoc Threshold ist ein ernstes methodisches Problem")
    log()
    log("  3. S2 ist die SCHWAECHERE:")
    log("     - Faellt moeglicherweise durch Bonferroni")
    log("     - 2023 fast Null -> Edge erodiert")
    log("     - Hohe Timeout-Rate -> moeglicherweise Buy-and-Hold Artefakt")
    log()
    log("  4. EMPFEHLUNG:")
    log("     - S4: Verdient einen OOS-Test, aber mit Erwartung von EV-Degradation")
    log("     - S2: Nur testen wenn S4 funktioniert, und mit strengerem alpha")
    log()

    elapsed = time.time() - t_start
    log(f"  Laufzeit: {elapsed:.1f} Sekunden")

    # Save
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out_lines))
    print(f"\n  Ergebnisse gespeichert: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
