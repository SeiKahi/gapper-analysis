"""
D13 QUANT EXPLORATION — Systematische Hypothesen-Entwicklung (IS only)
======================================================================
Drei Teile:
  1. Feature-Trennschaerfe (Spearman + Bucket-Analyse)
  2. Neue Strategie-Hypothesen (5 Backtests mit TRAIL_D)
  3. Kombinierte Strategien (Top-Filter zusammen)

IS-Periode: 2021-02-21 bis 2023-12-31
Datenquelle: metadata_v9.parquet + raw_1min/
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import warnings
import time

warnings.filterwarnings('ignore')

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
OUTPUT_FILE = RESULTS_DIR / "d13_quant_results.txt"

# ============================================================
# TRAIL_D SIMULATION
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
# HELPER: Run backtest on a filtered set
# ============================================================
def run_backtest(df_qualified, sl_mode='fix_025', label=''):
    """
    Run TRAIL_D backtest on qualified trades.
    sl_mode: 'fix_025' = 0.25*ADR, 'od_based' = OD-extremum SL
    Returns DataFrame with trade results.
    """
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

        # SL distance
        if sl_mode == 'fix_025':
            sl_dist = 0.25 * adr
        elif sl_mode == 'od_based':
            if trade_dir == 'long':
                sl_dist = abs(entry_price - row.get('od_low5', np.nan))
                if pd.isna(sl_dist) or sl_dist <= 0:
                    sl_dist = 0.25 * adr
            else:
                sl_dist = abs(row.get('od_high5', np.nan) - entry_price)
                if pd.isna(sl_dist) or sl_dist <= 0:
                    sl_dist = 0.25 * adr
        else:
            sl_dist = 0.25 * adr

        # Load 1min bars
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
        result['sl_adr'] = sl_dist / adr
        result['adr'] = adr
        result['gap_size_in_adr'] = row.get('gap_size_in_adr', np.nan)
        result['rvol_5'] = row.get('rvol_5', np.nan)
        result['rsi_14_prev'] = row.get('rsi_14_prev', np.nan)
        result['is_earnings'] = row.get('is_earnings', False)
        result['od_strength'] = row.get('od_strength', np.nan)
        results.append(result)

    return pd.DataFrame(results), skipped


def summarize_trades(df, label=''):
    """Return summary dict for a set of trades."""
    if len(df) == 0:
        return {'label': label, 'N': 0, 'WR%': np.nan, 'EV_ADR': np.nan,
                'Med_PnL_R': np.nan, 'SL_rate': np.nan}
    n = len(df)
    wr = df['winner'].mean() * 100
    ev_adr = df['pnl_adr'].mean()
    med_r = df['pnl_r'].median()
    sl_rate = (df['exit_type'] == 'initial_sl').mean() * 100
    trail_rate = (df['exit_type'] == 'trail_sl').mean() * 100
    timeout_rate = (df['exit_type'] == 'timeout').mean() * 100
    return {
        'label': label, 'N': n, 'WR%': round(wr, 1),
        'EV_ADR': round(ev_adr, 4), 'Med_PnL_R': round(med_r, 3),
        'SL_rate%': round(sl_rate, 1), 'Trail%': round(trail_rate, 1),
        'Timeout%': round(timeout_rate, 1),
    }


def format_summary(s):
    """Format summary dict into a readable string."""
    if s['N'] == 0:
        return f"  {s['label']}: N=0 [NO DATA]"
    warn = " [LOW N]" if s['N'] < 10 else ""
    return (f"  {s['label']}: N={s['N']}, WR={s['WR%']:.1f}%, "
            f"EV={s['EV_ADR']:+.4f} ADR, Med PnL={s['Med_PnL_R']:+.3f}R, "
            f"SL={s['SL_rate%']:.1f}%, Trail={s['Trail%']:.1f}%, "
            f"TO={s['Timeout%']:.1f}%{warn}")


# ============================================================
# MAIN
# ============================================================
def main():
    t_start = time.time()
    out_lines = []

    def log(msg=''):
        print(msg)
        out_lines.append(msg)

    log("=" * 80)
    log("D13 QUANT EXPLORATION — IS-Analyse (2021-02-21 bis 2023-12-31)")
    log("=" * 80)
    log()

    # ----------------------------------------------------------
    # LOAD DATA
    # ----------------------------------------------------------
    log("Loading metadata_v9.parquet...")
    meta = pd.read_parquet(META_PATH)
    h1 = meta[(meta['date'] >= '2021-02-21') & (meta['date'] <= '2023-12-31')].copy()
    log(f"  Total events: {len(meta)}, IS events: {len(h1)}")
    log(f"  IS date range: {h1['date'].min()} to {h1['date'].max()}")
    log()

    # ===========================================================
    # TEIL 1: FEATURE-TRENNSCHAERFE
    # ===========================================================
    log("=" * 80)
    log("TEIL 1: FEATURE-TRENNSCHAERFE (Spearman-Korrelation mit rest_drift)")
    log("=" * 80)
    log()

    # Numerische Spalten ermitteln
    numeric_cols = h1.select_dtypes(include=[np.number]).columns.tolist()
    # Entferne Target-Variable und ID-Spalten
    exclude = ['rest_drift', 'rest_drift_1000', 'rest_drift_1100', 'full_drift',
               'cl', 'day_of_week', 'vix_level']
    feature_cols = [c for c in numeric_cols if c not in exclude]

    # Spearman-Korrelation mit rest_drift
    target = h1['rest_drift'].dropna()
    correlations = {}
    for col in feature_cols:
        valid = h1[[col, 'rest_drift']].dropna()
        if len(valid) < 50:
            continue
        rho, pval = stats.spearmanr(valid[col], valid['rest_drift'])
        correlations[col] = {'rho': rho, 'pval': pval, 'abs_rho': abs(rho), 'n_valid': len(valid)}

    corr_df = pd.DataFrame(correlations).T.sort_values('abs_rho', ascending=False)

    log("Top 20 Features nach abs(Spearman-Korrelation) mit rest_drift:")
    log("-" * 75)
    log(f"  {'Feature':<30} {'rho':>8} {'p-value':>12} {'N':>6}")
    log("-" * 75)
    for i, (feat, row) in enumerate(corr_df.head(20).iterrows()):
        sig = "***" if row['pval'] < 0.001 else "**" if row['pval'] < 0.01 else "*" if row['pval'] < 0.05 else ""
        log(f"  {feat:<30} {row['rho']:>+8.4f} {row['pval']:>12.2e} {int(row['n_valid']):>6} {sig}")
    log()
    log("  Legende: *** p<0.001, ** p<0.01, * p<0.05")
    log()

    # Bucket-Analyse fuer Top-10 Features
    log("BUCKET-ANALYSE (Quintile) fuer Top-10 Features:")
    log("=" * 75)
    top_features = corr_df.head(10).index.tolist()

    for feat in top_features:
        valid = h1[[feat, 'rest_drift']].dropna()
        if len(valid) < 50:
            continue
        try:
            valid['quintile'] = pd.qcut(valid[feat], 5, labels=['Q1(low)', 'Q2', 'Q3', 'Q4', 'Q5(high)'],
                                         duplicates='drop')
        except ValueError:
            continue

        log(f"\n  Feature: {feat} (rho={corr_df.loc[feat, 'rho']:+.4f})")
        log(f"  {'Quintile':<12} {'N':>6} {'Mean drift':>12} {'WR%':>8} {'Median':>10} {'Range':>20}")
        log(f"  {'-'*70}")

        for q in valid['quintile'].unique():
            subset = valid[valid['quintile'] == q]
            n = len(subset)
            mean_d = subset['rest_drift'].mean()
            wr = (subset['rest_drift'] > 0).mean() * 100
            med_d = subset['rest_drift'].median()
            feat_min = subset[feat].min()
            feat_max = subset[feat].max()
            log(f"  {str(q):<12} {n:>6} {mean_d:>+12.4f} {wr:>7.1f}% {med_d:>+10.4f} [{feat_min:.3f} - {feat_max:.3f}]")
    log()

    # ===========================================================
    # TEIL 2: HYPOTHESEN-BACKTESTS
    # ===========================================================
    log("=" * 80)
    log("TEIL 2: HYPOTHESEN-BACKTESTS (TRAIL_D auf 1-Min-Daten)")
    log("=" * 80)
    log()

    # Baseline: od_strength > 0.5, close_935 entry, SL=0.25*ADR, TRAIL_D
    qualified = h1[h1['od_strength'] > 0.5].copy()
    log(f"Baseline-Qualifikation: od_strength > 0.5 -> N={len(qualified)}")
    log("Running baseline backtest...")
    baseline_trades, baseline_skip = run_backtest(qualified, sl_mode='fix_025', label='Baseline')
    log(f"  Completed: {len(baseline_trades)} trades, {baseline_skip} skipped")

    base_summary = summarize_trades(baseline_trades, 'Baseline (all)')
    log(format_summary(base_summary))

    # Split by gap direction
    for gd in ['up', 'down']:
        sub = baseline_trades[baseline_trades['gap_direction'] == gd]
        s = summarize_trades(sub, f'Baseline Gap{gd.title()}')
        log(format_summary(s))
    log()

    # ----------------------------------------------------------
    # HYPOTHESE 1: GAP-SIZE-OPTIMIERUNG
    # ----------------------------------------------------------
    log("-" * 80)
    log("HYPOTHESE 1: Gap-Size-Optimierung")
    log("  Basis: od_strength>0.5, Entry=close_935, SL=0.25*ADR, TRAIL_D")
    log("-" * 80)

    gap_buckets = [
        ('0.5-1.0 ADR', 0.5, 1.0),
        ('1.0-2.0 ADR', 1.0, 2.0),
        ('2.0-3.0 ADR', 2.0, 3.0),
        ('3.0+ ADR', 3.0, 999),
    ]

    h1_gap_results = {}
    for label, lo, hi in gap_buckets:
        mask = (baseline_trades['gap_size_in_adr'] >= lo) & (baseline_trades['gap_size_in_adr'] < hi)
        sub = baseline_trades[mask]
        s = summarize_trades(sub, label)
        log(format_summary(s))
        h1_gap_results[label] = s

        # Split by gap direction
        for gd in ['up', 'down']:
            sub_gd = sub[sub['gap_direction'] == gd]
            s_gd = summarize_trades(sub_gd, f'  {label} Gap{gd.title()}')
            log(format_summary(s_gd))
    log()

    # ----------------------------------------------------------
    # HYPOTHESE 2: RVOL-FILTER ENHANCEMENT
    # ----------------------------------------------------------
    log("-" * 80)
    log("HYPOTHESE 2: RVOL-Filter Enhancement")
    log("  Basis: od_strength>0.5, verschiedene rvol_5 Thresholds")
    log("-" * 80)

    rvol_thresholds = [2, 5, 8, 10]
    h2_results = {}
    for thr in rvol_thresholds:
        mask = baseline_trades['rvol_5'] >= thr
        sub = baseline_trades[mask]
        label = f'RVOL_5 >= {thr}'
        s = summarize_trades(sub, label)
        log(format_summary(s))
        h2_results[thr] = s

        for gd in ['up', 'down']:
            sub_gd = sub[sub['gap_direction'] == gd]
            s_gd = summarize_trades(sub_gd, f'  {label} Gap{gd.title()}')
            log(format_summary(s_gd))
    log()

    # ----------------------------------------------------------
    # HYPOTHESE 3: PREMARKET-AKTIVITAET (Volume + Range)
    # ----------------------------------------------------------
    log("-" * 80)
    log("HYPOTHESE 3: Premarket-Aktivitaet")
    log("  today_open == rth_open (Gap = prev_close vs 09:30-Open)")
    log("  -> PM-Drift nicht sinnvoll berechenbar aus Metadata")
    log("  Stattdessen: PM-Volume-Intensitaet (pm_rth5, pm_rth30_computed)")
    log("  und PM-Bar-Fill-Rate als Proxy fuer PM-Aktivitaet")
    log("-" * 80)

    # PM-Volume-Ratio: pm_rth5 = PM-Volume / RTH-5min-Volume
    # Hohe pm_rth5 = starke PM-Partizipation relativ zur Eroeffnung
    pm_rth5_lookup = qualified.set_index(['ticker', 'date'])['pm_rth5'].to_dict()
    baseline_trades['pm_rth5'] = baseline_trades.apply(
        lambda r: pm_rth5_lookup.get((r['ticker'], r['date']), np.nan), axis=1
    )
    pm_fill_lookup = qualified.set_index(['ticker', 'date'])['premarket_bar_fill_rate'].to_dict()
    baseline_trades['pm_fill_rate'] = baseline_trades.apply(
        lambda r: pm_fill_lookup.get((r['ticker'], r['date']), np.nan), axis=1
    )

    valid_pm = baseline_trades['pm_rth5'].dropna()
    log(f"\n  pm_rth5 Statistik (N={len(valid_pm)}):")
    log(f"    Mean: {valid_pm.mean():.3f}, Median: {valid_pm.median():.3f}")
    log(f"    Q25: {valid_pm.quantile(0.25):.3f}, Q75: {valid_pm.quantile(0.75):.3f}")

    # pm_rth5 Buckets
    pm_rth5_buckets = [
        ('pm_rth5 < 0.2', 0, 0.2),
        ('pm_rth5 0.2-0.5', 0.2, 0.5),
        ('pm_rth5 0.5-1.0', 0.5, 1.0),
        ('pm_rth5 1.0-2.0', 1.0, 2.0),
        ('pm_rth5 > 2.0', 2.0, 999),
    ]
    log("\n  PM-Volume-Ratio Buckets (pm_rth5 = PM_vol / RTH_5min_vol):")
    for label, lo, hi in pm_rth5_buckets:
        mask = (baseline_trades['pm_rth5'] >= lo) & (baseline_trades['pm_rth5'] < hi)
        sub = baseline_trades[mask]
        s = summarize_trades(sub, f'  {label}')
        log(format_summary(s))
        for gd in ['up', 'down']:
            sub_gd = sub[sub['gap_direction'] == gd]
            if len(sub_gd) >= 5:
                log(format_summary(summarize_trades(sub_gd, f'    {label} Gap{gd.title()}')))

    # PM-Bar-Fill-Rate Buckets
    log("\n  PM-Bar-Fill-Rate Buckets (Anteil PM-Minuten mit Handelsdaten):")
    pm_fill_buckets = [
        ('fill < 0.5', 0, 0.5),
        ('fill 0.5-0.7', 0.5, 0.7),
        ('fill 0.7-0.85', 0.7, 0.85),
        ('fill 0.85-0.95', 0.85, 0.95),
        ('fill > 0.95', 0.95, 1.01),
    ]
    for label, lo, hi in pm_fill_buckets:
        mask = (baseline_trades['pm_fill_rate'] >= lo) & (baseline_trades['pm_fill_rate'] < hi)
        sub = baseline_trades[mask]
        s = summarize_trades(sub, f'  {label}')
        log(format_summary(s))
    log()

    # ----------------------------------------------------------
    # HYPOTHESE 4: RSI-KONTEXT
    # ----------------------------------------------------------
    log("-" * 80)
    log("HYPOTHESE 4: RSI-Kontext (rsi_14_prev)")
    log("  Getrennt fuer GapUp und GapDn")
    log("  Hypothese: Oversold + GapDn = staerkeres Bounce?")
    log("-" * 80)

    rsi_buckets = [
        ('<30 Oversold', 0, 30),
        ('30-50', 30, 50),
        ('50-70', 50, 70),
        ('70+ Overbought', 70, 100),
    ]

    for gd in ['up', 'down']:
        log(f"\n  Gap {gd.title()}:")
        gd_trades = baseline_trades[baseline_trades['gap_direction'] == gd]
        for label, lo, hi in rsi_buckets:
            sub = gd_trades[(gd_trades['rsi_14_prev'] >= lo) & (gd_trades['rsi_14_prev'] < hi)]
            s = summarize_trades(sub, f'  RSI {label}')
            log(format_summary(s))
    log()

    # ----------------------------------------------------------
    # HYPOTHESE 5: EARNINGS VS NON-EARNINGS
    # ----------------------------------------------------------
    log("-" * 80)
    log("HYPOTHESE 5: Earnings vs Non-Earnings")
    log("  Getrennt fuer with_gap und against_gap")
    log("-" * 80)

    # Merge od_direction info from metadata
    q5 = qualified.copy()

    for earn_status in [True, False]:
        earn_label = "EARNINGS" if earn_status else "NON-EARNINGS"
        q5_sub = q5[q5['is_earnings'] == earn_status]
        log(f"\n  {earn_label}: N_qualified={len(q5_sub)}")

        # Run backtest
        if len(q5_sub) == 0:
            log(f"    [NO DATA]")
            continue

        trades_earn, skip_e = run_backtest(q5_sub, sl_mode='fix_025')
        s_all = summarize_trades(trades_earn, f'{earn_label} (all)')
        log(format_summary(s_all))

        # with_gap vs against_gap
        for od_dir in ['with_gap', 'against_gap']:
            sub = trades_earn[trades_earn['od_direction'] == od_dir]
            s = summarize_trades(sub, f'  {earn_label} {od_dir}')
            log(format_summary(s))

        # by gap direction
        for gd in ['up', 'down']:
            sub = trades_earn[trades_earn['gap_direction'] == gd]
            s = summarize_trades(sub, f'  {earn_label} Gap{gd.title()}')
            log(format_summary(s))
    log()

    # ===========================================================
    # TEIL 3: KOMBINIERTE STRATEGIEN
    # ===========================================================
    log("=" * 80)
    log("TEIL 3: KOMBINIERTE STRATEGIEN")
    log("  Teste die besten Einzel-Filter in Kombination")
    log("=" * 80)
    log()

    # Identify best individual filters from Teil 2 results
    # Based on typical findings: RVOL>=5, Gap 1-2 ADR, Earnings
    # We'll test multiple combinations

    combos = [
        # (label, filter_func)
        ('Combo A: RVOL_5>=5 + Gap 1-2 ADR',
         lambda t: (t['rvol_5'] >= 5) & (t['gap_size_in_adr'] >= 1.0) & (t['gap_size_in_adr'] < 2.0)),
        ('Combo B: RVOL_5>=5 + Gap 2+ ADR',
         lambda t: (t['rvol_5'] >= 5) & (t['gap_size_in_adr'] >= 2.0)),
        ('Combo C: RVOL_5>=8 + Gap 1+ ADR',
         lambda t: (t['rvol_5'] >= 8) & (t['gap_size_in_adr'] >= 1.0)),
        ('Combo D: RVOL_5>=5 + Earnings',
         lambda t: (t['rvol_5'] >= 5) & (t['is_earnings'] == True)),
        ('Combo E: RVOL_5>=5 + Non-Earnings',
         lambda t: (t['rvol_5'] >= 5) & (t['is_earnings'] == False)),
        ('Combo F: Gap 2+ ADR + Earnings',
         lambda t: (t['gap_size_in_adr'] >= 2.0) & (t['is_earnings'] == True)),
        ('Combo G: RVOL_5>=5 + RSI<50 + GapDn',
         lambda t: (t['rvol_5'] >= 5) & (t['rsi_14_prev'] < 50) & (t['gap_direction'] == 'down')),
        ('Combo H: RVOL_5>=5 + RSI>50 + GapUp',
         lambda t: (t['rvol_5'] >= 5) & (t['rsi_14_prev'] > 50) & (t['gap_direction'] == 'up')),
        ('Combo I: RVOL_5>=10 + Gap 1+ ADR',
         lambda t: (t['rvol_5'] >= 10) & (t['gap_size_in_adr'] >= 1.0)),
        ('Combo J: RVOL_5>=5 + Gap 1-3 ADR + Earnings',
         lambda t: (t['rvol_5'] >= 5) & (t['gap_size_in_adr'] >= 1.0) & (t['gap_size_in_adr'] < 3.0) & (t['is_earnings'] == True)),
        ('Combo K: RVOL_5>=5 + RSI>50 + GapUp + Non-Earn',
         lambda t: (t['rvol_5'] >= 5) & (t['rsi_14_prev'] > 50) & (t['gap_direction'] == 'up') & (t['is_earnings'] == False)),
        ('Combo L: RVOL_5>=5 + Gap 3+ ADR + Non-Earn',
         lambda t: (t['rvol_5'] >= 5) & (t['gap_size_in_adr'] >= 3.0) & (t['is_earnings'] == False)),
        ('Combo M: RSI 70+ + GapUp (overbought momentum)',
         lambda t: (t['rsi_14_prev'] >= 70) & (t['gap_direction'] == 'up')),
        ('Combo N: RSI<30 + GapDn (oversold bounce)',
         lambda t: (t['rsi_14_prev'] < 30) & (t['gap_direction'] == 'down')),
    ]

    combo_summaries = []
    for label, filt in combos:
        try:
            mask = filt(baseline_trades)
            sub = baseline_trades[mask]
            s = summarize_trades(sub, label)
            log(format_summary(s))
            combo_summaries.append(s)

            # Split by gap direction
            for gd in ['up', 'down']:
                sub_gd = sub[sub['gap_direction'] == gd]
                s_gd = summarize_trades(sub_gd, f'  {label} Gap{gd.title()}')
                log(format_summary(s_gd))
        except Exception as e:
            log(f"  {label}: ERROR - {e}")
        log()

    # ===========================================================
    # RANKING: Alle Strategien nach EV sortiert
    # ===========================================================
    log("=" * 80)
    log("RANKING: Alle Strategien nach EV(ADR) sortiert")
    log("  (nur Strategien mit N >= 10)")
    log("=" * 80)
    log()

    all_summaries = [base_summary]
    # Add H1 gap buckets
    for k, v in h1_gap_results.items():
        all_summaries.append(v)
    # Add H2 RVOL
    for k, v in h2_results.items():
        all_summaries.append(v)
    # Add combos
    all_summaries.extend(combo_summaries)

    # Filter N >= 10 and sort by EV
    valid_strats = [s for s in all_summaries if s['N'] >= 10 and not np.isnan(s['EV_ADR'])]
    valid_strats.sort(key=lambda x: x['EV_ADR'], reverse=True)

    log(f"  {'Rank':>4} {'Strategy':<50} {'N':>5} {'WR%':>7} {'EV(ADR)':>10} {'Med R':>8}")
    log(f"  {'-'*84}")
    for i, s in enumerate(valid_strats, 1):
        log(f"  {i:>4} {s['label']:<50} {s['N']:>5} {s['WR%']:>6.1f}% {s['EV_ADR']:>+10.4f} {s['Med_PnL_R']:>+8.3f}")
    log()

    # ===========================================================
    # ZUSAMMENFASSUNG & EMPFEHLUNGEN
    # ===========================================================
    log("=" * 80)
    log("ZUSAMMENFASSUNG & EMPFEHLUNGEN")
    log("=" * 80)
    log()
    log("Baseline (od_strength > 0.5, SL=0.25*ADR, TRAIL_D):")
    log(format_summary(base_summary))
    log()

    if len(valid_strats) >= 3:
        log("Top 3 Strategien nach EV(ADR):")
        for i, s in enumerate(valid_strats[:3], 1):
            log(f"  #{i}: {format_summary(s)}")
        log()

    log("HINWEIS: Alle Ergebnisse sind IS (In-Sample). OOS-Validierung erforderlich!")
    log("HINWEIS: Strategien mit N < 30 sollten mit Vorsicht betrachtet werden.")

    elapsed = time.time() - t_start
    log(f"\n  Laufzeit: {elapsed:.1f} Sekunden")

    # Save results
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out_lines))
    print(f"\n  Ergebnisse gespeichert: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
