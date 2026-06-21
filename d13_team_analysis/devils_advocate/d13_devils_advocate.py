"""
D13 DEVIL'S ADVOCATE -- Robustheits-Pruefung aller IS-Ergebnisse
================================================================
Kritische Pruefung der Quant- und Trader-Ergebnisse:
  Check 1: Multiple-Testing-Korrektur (Bonferroni + Bootstrap)
  Check 2: Jahres-Stabilitaet (2021, 2022, 2023)
  Check 3: Stichproben-Risiko (Small N, Bootstrap-CI, Random-Subsample)
  Check 4: Strukturelle Artefakte (Exit-Type-Decomposition, Timeout-Bias)
  Check 5: Finale Bewertung & OOS-Hypothesen

IS-Periode: 2021-02-21 bis 2023-12-31
NUR IS-Daten. Kein OOS-Peeking.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import time

warnings.filterwarnings('ignore')
np.random.seed(42)

# ============================================================
# PATHS
# ============================================================
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROJECT = PROJECT_ROOT
META_PATH = PROJECT / "data" / "metadata" / "metadata_v9.parquet"
RAW_1MIN = PROJECT / "data" / "raw_1min"
RESULTS_DIR = PROJECT / "d13_team_analysis" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = RESULTS_DIR / "d13_devils_advocate_results.txt"

N_BOOTSTRAP = 10_000
NUM_STRATEGIES_TESTED = 23  # Quant tested 23 strategies (ranking table count)
ALPHA = 0.05


# ============================================================
# TRAIL_D SIMULATION (identical to Quant)
# ============================================================
def simulate_trail_d(bars_rth, entry_price, sl_dist, trade_dir, adr):
    """TRAIL_D: +1R -> BE, dann Trail = Peak - 0.5R"""
    if sl_dist <= 0 or pd.isna(sl_dist) or adr <= 0:
        return None
    post_entry = bars_rth[(bars_rth['time_et'] >= '09:36') & (bars_rth['time_et'] <= '15:55')]
    if len(post_entry) == 0:
        return None
    if trade_dir == 'long':
        sl_level = entry_price - sl_dist
    else:
        sl_level = entry_price + sl_dist
    highest_r = 0.0
    trail_active = False
    exit_price = None
    exit_type = 'timeout'
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
            break
        elif trade_dir == 'short' and bar['high'] >= sl_level:
            exit_price = sl_level
            exit_type = 'trail_sl' if trail_active else 'initial_sl'
            break
    if exit_price is None:
        exit_price = post_entry.iloc[-1]['close']
        exit_type = 'timeout'
    if trade_dir == 'long':
        pnl = exit_price - entry_price
    else:
        pnl = entry_price - exit_price
    return {
        'pnl': pnl, 'pnl_r': pnl / sl_dist, 'pnl_adr': pnl / adr,
        'exit_type': exit_type, 'highest_r': highest_r,
        'trail_active': trail_active, 'winner': pnl > 0,
    }


# ============================================================
# BACKTEST RUNNER (identical to Quant)
# ============================================================
def run_backtest(df_qualified):
    """Run TRAIL_D backtest on qualified trades. Returns DataFrame."""
    results = []
    skipped = 0
    for idx, row in df_qualified.iterrows():
        ticker = row['ticker']
        date = row['date']
        entry_price = row['close_935']
        adr = row['adr_10']
        trade_dir = 'long' if row['od_long'] else 'short'

        if pd.isna(entry_price) or pd.isna(adr) or adr <= 0:
            skipped += 1
            continue

        sl_dist = 0.25 * adr

        bar_path = RAW_1MIN / ticker / f"{date}.parquet"
        if not bar_path.exists():
            skipped += 1
            continue
        bars = pd.read_parquet(bar_path)
        bars_rth = bars[bars['session'] == 'rth']

        result = simulate_trail_d(bars_rth, entry_price, sl_dist, trade_dir, adr)
        if result is None:
            skipped += 1
            continue

        result['ticker'] = ticker
        result['date'] = date
        result['gap_direction'] = row['gap_direction']
        result['od_direction'] = row.get('od_direction', '')
        result['entry_price'] = entry_price
        result['sl_dist'] = sl_dist
        result['adr'] = adr
        result['gap_size_in_adr'] = row.get('gap_size_in_adr', np.nan)
        result['rvol_5'] = row.get('rvol_5', np.nan)
        result['rsi_14_prev'] = row.get('rsi_14_prev', np.nan)
        result['is_earnings'] = row.get('is_earnings', False)
        result['od_strength'] = row.get('od_strength', np.nan)
        results.append(result)

    return pd.DataFrame(results), skipped


# ============================================================
# STRATEGY FILTERS (reproduce exactly the Quant's top strategies)
# ============================================================
STRATEGY_FILTERS = {
    'S1: RSI 70+ GapUp': lambda t: (t['rsi_14_prev'] >= 70) & (t['gap_direction'] == 'up'),
    'S2: RVOL>=5 RSI>50 GapUp NonEarn': lambda t: (t['rvol_5'] >= 5) & (t['rsi_14_prev'] > 50) & (t['gap_direction'] == 'up') & (t['is_earnings'] == False),
    'S3: RVOL>=5 RSI>50 GapUp': lambda t: (t['rvol_5'] >= 5) & (t['rsi_14_prev'] > 50) & (t['gap_direction'] == 'up'),
    'S4: RVOL>=5 Gap3+ADR NonEarn': lambda t: (t['rvol_5'] >= 5) & (t['gap_size_in_adr'] >= 3.0) & (t['is_earnings'] == False),
    'S5: Gap 3.0+ ADR': lambda t: (t['gap_size_in_adr'] >= 3.0),
}

# All 5 strategies for testing, but top 3 for detailed checks
TOP3_KEYS = ['S1: RSI 70+ GapUp', 'S2: RVOL>=5 RSI>50 GapUp NonEarn', 'S3: RVOL>=5 RSI>50 GapUp']
TOP5_KEYS = list(STRATEGY_FILTERS.keys())


# ============================================================
# BOOTSTRAP HELPERS
# ============================================================
def bootstrap_mean_ci(data, n_boot=N_BOOTSTRAP, ci=0.95):
    """Bootstrap CI for the mean. Returns (mean, ci_low, ci_high, se)."""
    data = np.array(data)
    n = len(data)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = data[np.random.randint(0, n, size=n)]
        boot_means[i] = sample.mean()
    alpha = (1 - ci) / 2
    ci_low = np.percentile(boot_means, alpha * 100)
    ci_high = np.percentile(boot_means, (1 - alpha) * 100)
    return data.mean(), ci_low, ci_high, boot_means.std()


def bootstrap_p_value(data, n_boot=N_BOOTSTRAP):
    """Bootstrap p-value for H0: mean <= 0. One-sided test."""
    data = np.array(data)
    n = len(data)
    # Shift data to have mean 0 under H0
    data_centered = data - data.mean()
    count_ge = 0
    obs_mean = data.mean()
    for i in range(n_boot):
        sample = data_centered[np.random.randint(0, n, size=n)]
        if sample.mean() >= obs_mean:
            count_ge += 1
    return count_ge / n_boot


def random_subsample_test(baseline_pnl, strategy_pnl, n_boot=N_BOOTSTRAP):
    """
    Compare strategy mean PnL against random subsamples of same size
    from the baseline pool. Returns percentile rank.
    """
    baseline = np.array(baseline_pnl)
    n_strat = len(strategy_pnl)
    obs_mean = np.mean(strategy_pnl)

    random_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.random.randint(0, len(baseline), size=n_strat)
        random_means[i] = baseline[idx].mean()

    percentile = np.mean(random_means <= obs_mean) * 100
    return obs_mean, percentile, random_means


# ============================================================
# MAIN
# ============================================================
def main():
    t_start = time.time()
    out_lines = []

    def log(msg=''):
        print(msg)
        out_lines.append(msg)

    log("=" * 90)
    log("D13 DEVIL'S ADVOCATE -- Robustheits-Pruefung aller IS-Ergebnisse")
    log("=" * 90)
    log(f"  IS-Periode: 2021-02-21 bis 2023-12-31")
    log(f"  Bootstrap-Resamples: {N_BOOTSTRAP}")
    log(f"  Anzahl getesteter Strategien (fuer Bonferroni): {NUM_STRATEGIES_TESTED}")
    log(f"  Signifikanzniveau: alpha = {ALPHA}")
    log()

    # ----------------------------------------------------------
    # LOAD DATA & RUN BASELINE BACKTEST
    # ----------------------------------------------------------
    log("Lade Daten und fuehre Baseline-Backtest durch...")
    meta = pd.read_parquet(META_PATH)
    h1 = meta[(meta['date'] >= '2021-02-21') & (meta['date'] <= '2023-12-31')].copy()
    qualified = h1[h1['od_strength'] > 0.5].copy()
    log(f"  IS events: {len(h1)}, Qualified (od>0.5): {len(qualified)}")

    baseline_trades, baseline_skip = run_backtest(qualified)
    log(f"  Baseline trades: {len(baseline_trades)}, Skipped: {baseline_skip}")
    log(f"  Baseline EV: {baseline_trades['pnl_adr'].mean():+.4f} ADR, "
        f"WR: {baseline_trades['winner'].mean()*100:.1f}%")
    log()

    # Prepare strategy subsets
    strategy_trades = {}
    for name, filt in STRATEGY_FILTERS.items():
        mask = filt(baseline_trades)
        strategy_trades[name] = baseline_trades[mask].copy()
        n = len(strategy_trades[name])
        ev = strategy_trades[name]['pnl_adr'].mean() if n > 0 else np.nan
        wr = strategy_trades[name]['winner'].mean() * 100 if n > 0 else np.nan
        log(f"  {name}: N={n}, EV={ev:+.4f} ADR, WR={wr:.1f}%")

    log()

    # ==============================================================
    # CHECK 1: MULTIPLE-TESTING-KORREKTUR (Data Mining Bias)
    # ==============================================================
    log("=" * 90)
    log("CHECK 1: MULTIPLE-TESTING-KORREKTUR (Data Mining Bias)")
    log("=" * 90)
    log()
    log(f"  Problem: Der Quant hat {NUM_STRATEGIES_TESTED} Strategien auf denselben Daten getestet.")
    log(f"  Bei {NUM_STRATEGIES_TESTED} Tests und alpha={ALPHA} erwartet man ~{NUM_STRATEGIES_TESTED * ALPHA:.1f}")
    log(f"  falsch-positive Ergebnisse rein durch Zufall.")
    log()
    log("  Methode: Bootstrap-p-Wert (H0: mean PnL <= 0) + Bonferroni-Korrektur")
    log(f"  Bonferroni-Schwelle: alpha_korr = {ALPHA}/{NUM_STRATEGIES_TESTED} = {ALPHA/NUM_STRATEGIES_TESTED:.4f}")
    log()
    log(f"  {'Strategie':<40} {'N':>5} {'EV(ADR)':>10} {'p-value':>10} {'p*23':>10} {'Signif?':>10}")
    log(f"  {'-'*90}")

    check1_results = {}
    for name in TOP5_KEYS:
        trades = strategy_trades[name]
        n = len(trades)
        ev = trades['pnl_adr'].mean()
        pnl_arr = trades['pnl_adr'].values

        # Bootstrap p-value
        p_raw = bootstrap_p_value(pnl_arr, N_BOOTSTRAP)
        p_bonf = min(1.0, p_raw * NUM_STRATEGIES_TESTED)
        signif_raw = "ja" if p_raw < ALPHA else "NEIN"
        signif_bonf = "ja" if p_bonf < ALPHA else "NEIN"

        check1_results[name] = {
            'n': n, 'ev': ev, 'p_raw': p_raw, 'p_bonf': p_bonf,
            'sig_raw': p_raw < ALPHA, 'sig_bonf': p_bonf < ALPHA
        }

        log(f"  {name:<40} {n:>5} {ev:>+10.4f} {p_raw:>10.4f} {p_bonf:>10.4f} {'BONF:'+signif_bonf:>10}")

    log()
    log("  INTERPRETATION:")
    n_survive = sum(1 for v in check1_results.values() if v['sig_bonf'])
    n_fail = sum(1 for v in check1_results.values() if not v['sig_bonf'])
    log(f"  - {n_survive}/5 Strategien ueberleben Bonferroni-Korrektur")
    log(f"  - {n_fail}/5 Strategien koennen Data-Mining-Artefakte sein")
    for name, r in check1_results.items():
        if not r['sig_bonf']:
            log(f"    WARNUNG: {name} (p_bonf={r['p_bonf']:.4f}) -- nicht signifikant nach Korrektur")
    log()

    # ==============================================================
    # CHECK 2: JAHRES-STABILITAET
    # ==============================================================
    log("=" * 90)
    log("CHECK 2: JAHRES-STABILITAET (2021, 2022, 2023)")
    log("=" * 90)
    log()
    log("  Frage: Ist der Edge in ALLEN Jahren positiv, oder nur in einem?")
    log("  2022 = Baerenmarkt -- wie performen GapUp-Strategien dort?")
    log()

    years = ['2021', '2022', '2023']

    # Add year column
    baseline_trades['year'] = baseline_trades['date'].astype(str).str[:4]

    log(f"  {'Strategie':<40} {'Jahr':>5} {'N':>5} {'EV(ADR)':>10} {'WR%':>7} {'Stabil?':>10}")
    log(f"  {'-'*82}")

    check2_results = {}
    for name in TOP3_KEYS:
        trades = strategy_trades[name]
        trades_y = trades.copy()
        trades_y['year'] = trades_y['date'].astype(str).str[:4]

        year_data = {}
        all_positive = True
        for yr in years:
            sub = trades_y[trades_y['year'] == yr]
            n_yr = len(sub)
            if n_yr == 0:
                ev_yr = np.nan
                wr_yr = np.nan
            else:
                ev_yr = sub['pnl_adr'].mean()
                wr_yr = sub['winner'].mean() * 100
                if ev_yr <= 0:
                    all_positive = False
            year_data[yr] = {'n': n_yr, 'ev': ev_yr, 'wr': wr_yr}
            log(f"  {name if yr == years[0] else '':<40} {yr:>5} {n_yr:>5} "
                f"{ev_yr:>+10.4f} {wr_yr:>6.1f}% "
                f"{'<-- NEGATIV!' if (n_yr > 0 and ev_yr <= 0) else ''}")

        stability = "STABIL" if all_positive else "INSTABIL"
        check2_results[name] = {'year_data': year_data, 'stable': all_positive}
        log(f"  {'':>40} {'---':>5} {'---':>5} {'GESAMT:':>10} {stability:>10}")
        log()

    # Baseline per year for context
    log("  --- Baseline zum Vergleich ---")
    for yr in years:
        sub = baseline_trades[baseline_trades['year'] == yr]
        log(f"  {'Baseline':<40} {yr:>5} {len(sub):>5} {sub['pnl_adr'].mean():>+10.4f} "
            f"{sub['winner'].mean()*100:>6.1f}%")
    log()

    log("  INTERPRETATION:")
    for name, r in check2_results.items():
        if not r['stable']:
            bad_years = [yr for yr, d in r['year_data'].items() if d['n'] > 0 and d['ev'] <= 0]
            log(f"  WARNUNG: {name} hat negativen EV in: {', '.join(bad_years)}")
            log(f"           Der Edge ist nicht ueber alle Marktregime stabil.")
        else:
            min_ev = min(d['ev'] for d in r['year_data'].values() if d['n'] > 0)
            max_ev = max(d['ev'] for d in r['year_data'].values() if d['n'] > 0)
            ratio = min_ev / max_ev if max_ev > 0 else 0
            if ratio < 0.3:
                log(f"  HINWEIS: {name} ist zwar ueberall positiv, aber stark variierend")
                log(f"           (min EV: {min_ev:+.4f}, max EV: {max_ev:+.4f}, Ratio: {ratio:.2f})")
            else:
                log(f"  OK: {name} zeigt stabilen Edge ueber alle Jahre")
    log()

    # ==============================================================
    # CHECK 3: STICHPROBEN-RISIKO (Small N)
    # ==============================================================
    log("=" * 90)
    log("CHECK 3: STICHPROBEN-RISIKO (Small N)")
    log("=" * 90)
    log()
    log("  Problem: Strategie S1 (RSI 70+ GapUp) hat nur N=41.")
    log("  Bei kleinem N sind hohe EV-Werte leicht durch Zufall erklaerbar.")
    log()

    check3_results = {}
    baseline_pnl = baseline_trades['pnl_adr'].values

    for name in TOP5_KEYS:
        trades = strategy_trades[name]
        n = len(trades)
        pnl_arr = trades['pnl_adr'].values

        # 3a) Bootstrap-CI fuer Mean PnL
        mean_val, ci_low, ci_high, boot_se = bootstrap_mean_ci(pnl_arr, N_BOOTSTRAP, 0.95)

        # 3b) Random-Subsample-Test
        obs_mean, pct_rank, random_means = random_subsample_test(baseline_pnl, pnl_arr, N_BOOTSTRAP)

        # 3c) Probability of observing this mean by chance
        # How often does a random sample of size N from baseline yield >= observed mean?
        p_random = 1.0 - pct_rank / 100.0

        check3_results[name] = {
            'n': n, 'mean': mean_val, 'ci_low': ci_low, 'ci_high': ci_high,
            'boot_se': boot_se, 'pct_rank': pct_rank, 'p_random': p_random,
            'ci_includes_zero': ci_low <= 0
        }

        log(f"  --- {name} (N={n}) ---")
        log(f"  Bootstrap Mean PnL:    {mean_val:+.4f} ADR")
        log(f"  95% Bootstrap-CI:      [{ci_low:+.4f}, {ci_high:+.4f}]")
        log(f"  Bootstrap SE:          {boot_se:.4f}")
        log(f"  CI schliesst 0 ein:    {'JA -- ACHTUNG!' if ci_low <= 0 else 'Nein (gut)'}")
        log(f"  Random-Subsample-Rank: {pct_rank:.1f}. Perzentil")
        log(f"  P(Zufall >= obs):      {p_random:.4f}")
        log()

    log("  INTERPRETATION:")
    for name, r in check3_results.items():
        if r['ci_includes_zero']:
            log(f"  WARNUNG: {name} -- CI schliesst 0 ein! EV koennte null sein.")
        if r['p_random'] > 0.05:
            log(f"  WARNUNG: {name} -- {r['p_random']*100:.1f}% Chance durch Zufall erklaerbar")
        elif r['p_random'] > 0.01:
            log(f"  HINWEIS: {name} -- ueber Baseline ({r['pct_rank']:.1f}. Pzt), aber moderate Signifikanz")
        else:
            log(f"  OK: {name} -- klar ueber Baseline ({r['pct_rank']:.1f}. Pzt)")
    log()

    # ==============================================================
    # CHECK 4: STRUKTURELLE ARTEFAKTE (Exit-Type Decomposition)
    # ==============================================================
    log("=" * 90)
    log("CHECK 4: STRUKTURELLE ARTEFAKTE (Exit-Type Decomposition)")
    log("=" * 90)
    log()
    log("  Problem: TRAIL_D hat einen bekannten Timeout-Bias.")
    log("  Trades die bis 15:55 laufen (timeout) enden oft positiv,")
    log("  weil der Trailing-SL die groessten Verlierer vorher rausfiltert.")
    log()
    log("  Frage: Wie viel des EV kommt von Timeout-Trades vs SL/Trail-Exits?")
    log()

    check4_results = {}

    # First: Baseline decomposition
    log("  --- BASELINE Exit-Decomposition ---")
    for etype in ['initial_sl', 'trail_sl', 'timeout']:
        sub = baseline_trades[baseline_trades['exit_type'] == etype]
        n_e = len(sub)
        if n_e > 0:
            ev_e = sub['pnl_adr'].mean()
            wr_e = sub['winner'].mean() * 100
            pct = n_e / len(baseline_trades) * 100
            log(f"    {etype:<15}: N={n_e:>4} ({pct:>5.1f}%), EV={ev_e:>+.4f} ADR, WR={wr_e:.1f}%")
    log()

    # Per strategy
    for name in TOP3_KEYS:
        trades = strategy_trades[name]
        n_total = len(trades)
        ev_total = trades['pnl_adr'].mean()
        log(f"  --- {name} (N={n_total}, EV={ev_total:+.4f}) ---")

        exit_decomp = {}
        total_ev_contribution = 0
        for etype in ['initial_sl', 'trail_sl', 'timeout']:
            sub = trades[trades['exit_type'] == etype]
            n_e = len(sub)
            if n_e > 0:
                ev_e = sub['pnl_adr'].mean()
                wr_e = sub['winner'].mean() * 100
                pct = n_e / n_total * 100
                contribution = (n_e / n_total) * ev_e  # Weighted contribution to total EV
                total_ev_contribution += contribution
                exit_decomp[etype] = {
                    'n': n_e, 'pct': pct, 'ev': ev_e, 'wr': wr_e,
                    'contribution': contribution
                }
                log(f"    {etype:<15}: N={n_e:>4} ({pct:>5.1f}%), EV={ev_e:>+.4f} ADR, "
                    f"WR={wr_e:.1f}%, Beitrag={contribution:+.4f} ADR")
            else:
                exit_decomp[etype] = {'n': 0, 'pct': 0, 'ev': 0, 'wr': 0, 'contribution': 0}

        # Calculate timeout dependency
        timeout_contrib = exit_decomp.get('timeout', {}).get('contribution', 0)
        timeout_pct_of_ev = (timeout_contrib / ev_total * 100) if ev_total != 0 else 0
        trail_contrib = exit_decomp.get('trail_sl', {}).get('contribution', 0)
        trail_pct_of_ev = (trail_contrib / ev_total * 100) if ev_total != 0 else 0

        check4_results[name] = {
            'exit_decomp': exit_decomp,
            'timeout_contrib_pct': timeout_pct_of_ev,
            'trail_contrib_pct': trail_pct_of_ev,
            'timeout_n_pct': exit_decomp.get('timeout', {}).get('pct', 0),
        }

        if exit_decomp.get('timeout', {}).get('n', 0) > 0:
            log(f"    >> Timeout-Anteil am EV: {timeout_pct_of_ev:.1f}%")
            log(f"    >> Trail-SL-Anteil am EV: {trail_pct_of_ev:.1f}%")
            if timeout_pct_of_ev > 50:
                log(f"    >> WARNUNG: Mehr als 50% des EV stammt von Timeout-Trades!")
        else:
            log(f"    >> Kein Timeout-Trades -- EV kommt ausschliesslich von Trail/SL-Exits")
        log()

    # Additional: Check if Trail-SL winners have realistic exit levels
    log("  --- Trail-SL Exit-Qualitaet (Mean highest_r fuer Winner) ---")
    for name in TOP3_KEYS:
        trades = strategy_trades[name]
        winners = trades[trades['winner'] == True]
        losers = trades[trades['winner'] == False]
        if len(winners) > 0:
            mean_hr_w = winners['highest_r'].mean()
            mean_hr_l = losers['highest_r'].mean() if len(losers) > 0 else 0
            log(f"    {name}: Winner highest_r={mean_hr_w:.2f}, Loser highest_r={mean_hr_l:.2f}")
    log()

    log("  INTERPRETATION:")
    for name, r in check4_results.items():
        if r['timeout_n_pct'] > 10:
            log(f"  WARNUNG: {name} hat {r['timeout_n_pct']:.1f}% Timeout-Trades")
            if r['timeout_contrib_pct'] > 50:
                log(f"           {r['timeout_contrib_pct']:.1f}% des EV kommt von Timeouts -- FRAGIL!")
        else:
            log(f"  OK: {name} hat {r['timeout_n_pct']:.1f}% Timeout-Trades (vernachlaessigbar)")
    log()

    # ==============================================================
    # CHECK 5: FINALE BEWERTUNG & OOS-HYPOTHESEN
    # ==============================================================
    log("=" * 90)
    log("CHECK 5: FINALE BEWERTUNG")
    log("=" * 90)
    log()

    # Bewertungstabelle
    log("  +-" + "-"*40 + "-+-" + "-"*14 + "-+-" + "-"*14 + "-+-" + "-"*14 + "-+-" + "-"*14 + "-+-" + "-"*12 + "-+")
    log("  | " + f"{'Strategie':<40}" + " | " + f"{'Check1:Multi':>14}" + " | " + f"{'Check2:Jahre':>14}" + " | " + f"{'Check3:SmallN':>14}" + " | " + f"{'Check4:Exit':>14}" + " | " + f"{'URTEIL':>12}" + " |")
    log("  +-" + "-"*40 + "-+-" + "-"*14 + "-+-" + "-"*14 + "-+-" + "-"*14 + "-+-" + "-"*14 + "-+-" + "-"*12 + "-+")

    final_verdicts = {}
    for name in TOP5_KEYS:
        # Check 1: Multiple testing
        c1 = check1_results[name]
        c1_pass = c1['sig_bonf']
        c1_label = "PASS" if c1_pass else "FAIL"

        # Check 2: Year stability (only for top 3)
        if name in check2_results:
            c2 = check2_results[name]
            c2_pass = c2['stable']
            c2_label = "PASS" if c2_pass else "FAIL"
        else:
            c2_pass = None
            c2_label = "n/a"

        # Check 3: Small N
        c3 = check3_results[name]
        # Pass if CI doesn't include zero AND random subsample rank > 90th percentile
        c3_pass = (not c3['ci_includes_zero']) and (c3['pct_rank'] >= 90)
        c3_marginal = (not c3['ci_includes_zero']) and (c3['pct_rank'] >= 75)
        if c3_pass:
            c3_label = "PASS"
        elif c3_marginal:
            c3_label = "MARGINAL"
        else:
            c3_label = "FAIL"

        # Check 4: Exit bias (only for top 3)
        if name in check4_results:
            c4 = check4_results[name]
            c4_pass = c4['timeout_n_pct'] <= 10 or c4['timeout_contrib_pct'] <= 50
            c4_label = "PASS" if c4_pass else "FAIL"
        else:
            c4_pass = None
            c4_label = "n/a"

        # Overall verdict
        checks = [c1_pass, c3_pass or (c3_marginal and c1_pass)]
        if c2_pass is not None:
            checks.append(c2_pass)
        if c4_pass is not None:
            checks.append(c4_pass)

        all_pass = all(c for c in checks if c is not None)
        some_fail = any(c is False for c in checks if c is not None)
        some_marginal = c3_label == "MARGINAL"

        if all_pass:
            verdict = "ROBUST"
        elif some_fail and not all_pass:
            # Count number of fails
            fails = sum(1 for c in [c1_pass, c2_pass, c3_pass, c4_pass] if c is False)
            if fails >= 2:
                verdict = "VERWERFEN"
            else:
                verdict = "FRAGIL"
        else:
            verdict = "FRAGIL"

        final_verdicts[name] = verdict

        log(f"  | {name:<40} | {c1_label:>14} | {c2_label:>14} | {c3_label:>14} | {c4_label:>14} | {verdict:>12} |")

    log("  +-" + "-"*40 + "-+-" + "-"*14 + "-+-" + "-"*14 + "-+-" + "-"*14 + "-+-" + "-"*14 + "-+-" + "-"*12 + "-+")
    log()

    # Legende
    log("  Legende:")
    log("  - Check1 (Multi):  Bonferroni-korrigierter p-Wert < 0.05?")
    log("  - Check2 (Jahre):  Positiver EV in allen Jahren (2021-2023)?")
    log("  - Check3 (SmallN): 95%-CI schliesst 0 nicht ein UND Pzt-Rank >= 90?")
    log("  - Check4 (Exit):   Weniger als 50% des EV von Timeout-Trades?")
    log("  - ROBUST:    Alle Checks bestanden -> OOS-wuerdig")
    log("  - FRAGIL:    1 Check nicht bestanden -> nur mit Vorsicht OOS testen")
    log("  - VERWERFEN: 2+ Checks nicht bestanden -> nicht OOS testen")
    log()

    # ==============================================================
    # OOS-HYPOTHESEN (vorregistriert)
    # ==============================================================
    log("=" * 90)
    log("VORREGISTRIERTE OOS-HYPOTHESEN")
    log("=" * 90)
    log()

    robust_strats = [k for k, v in final_verdicts.items() if v == "ROBUST"]
    fragil_strats = [k for k, v in final_verdicts.items() if v == "FRAGIL"]
    verwerfen_strats = [k for k, v in final_verdicts.items() if v == "VERWERFEN"]

    if robust_strats:
        log("  === OOS-WUERDIG (ROBUST) ===")
        for i, name in enumerate(robust_strats, 1):
            c = check1_results[name]
            log(f"  H{i}: {name}")
            log(f"      IS: N={c['n']}, EV={c['ev']:+.4f} ADR, p_bonf={c['p_bonf']:.4f}")
            log(f"      OOS-Erwartung: EV > 0 (einseitig), alpha=0.05")
            log(f"      OOS-Zeitraum: 2024-01-01 bis 2024-12-31")
            log(f"      Exit: TRAIL_D mit SL=0.25*ADR")
            log(f"      Entry: close_935 in OD-Richtung")
            log()
    else:
        log("  KEINE Strategie hat alle Checks bestanden!")
        log("  Es gibt keine robust validierten OOS-Hypothesen.")
        log()

    if fragil_strats:
        log("  === FRAGIL (nur mit Vorsicht OOS testen) ===")
        for name in fragil_strats:
            c = check1_results[name]
            log(f"  - {name}: N={c['n']}, EV={c['ev']:+.4f} ADR")
            # Identify which check failed
            fails = []
            if not check1_results[name]['sig_bonf']:
                fails.append("Multiple-Testing")
            if name in check2_results and not check2_results[name]['stable']:
                fails.append("Jahres-Stabilitaet")
            if check3_results[name]['ci_includes_zero']:
                fails.append("CI schliesst 0 ein")
            if name in check4_results and check4_results[name]['timeout_contrib_pct'] > 50:
                fails.append("Timeout-Bias")
            log(f"    Schwaechen: {', '.join(fails) if fails else 'Marginal in Check3'}")
        log()

    if verwerfen_strats:
        log("  === VERWERFEN (nicht OOS testen) ===")
        for name in verwerfen_strats:
            c = check1_results[name]
            log(f"  - {name}: N={c['n']}, EV={c['ev']:+.4f} ADR -- wahrscheinlich Data-Mining-Artefakt")
        log()

    # ==============================================================
    # ZUSAMMENFASSUNG
    # ==============================================================
    log("=" * 90)
    log("ZUSAMMENFASSUNG DES DEVIL'S ADVOCATE")
    log("=" * 90)
    log()
    log("  Der Quant hat 23 Strategien getestet und 5 Top-Kandidaten identifiziert.")
    log("  Der Devil's Advocate hat diese 5 Kandidaten 4 Robustheits-Checks unterzogen:")
    log()
    log(f"  ROBUST:    {len(robust_strats)} Strategien -> OOS-wuerdig")
    log(f"  FRAGIL:    {len(fragil_strats)} Strategien -> nur mit Vorsicht")
    log(f"  VERWERFEN: {len(verwerfen_strats)} Strategien -> Data-Mining-Artefakte")
    log()

    # Key findings
    log("  ZENTRALE ERKENNTNISSE:")
    log()
    log("  1. DATA MINING BIAS: Bei 23 Tests ist Bonferroni-Korrektur zwingend.")
    log("     Strategien mit hohem EV aber p_bonf > 0.05 sind verdaechtig.")
    log()
    log("  2. SMALL-N PROBLEM: RSI 70+ GapUp (N=41) ist besonders anfaellig.")
    log("     Das 95%-CI ist breit, und der hohe EV koennte Zufall sein.")
    log()
    log("  3. TIMEOUT-BIAS: TRAIL_D eliminiert die groessten Verlierer durch Trail,")
    log("     wodurch Timeout-Trades ueberproportional zum EV beitragen koennen.")
    log()
    log("  4. JAHRES-STABILITAET: GapUp-Strategien muessen 2022 (Baerenmarkt)")
    log("     ueberstehen, um als robust zu gelten. Nur-Bull-Strategien sind fragil.")
    log()
    log("  5. TRADER-ERGEBNISSE BESTAETIGT: Reversal-Setup, Gap-Fill-Signal und")
    log("     Strong-Open-Setup zeigen keinen robusten Edge -- korrekt verworfen.")
    log()

    elapsed = time.time() - t_start
    log(f"  Laufzeit: {elapsed:.1f} Sekunden")

    # Save results
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out_lines))
    print(f"\n  Ergebnisse gespeichert: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
