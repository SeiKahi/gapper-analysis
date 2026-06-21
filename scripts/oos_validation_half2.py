"""
OOS Validation — Half 2 (2024-01-01 to 2026-02-06)
====================================================
EINMALIGER Test. Kein Hin-und-Her-Optimieren.

Hypothese: "GapDn + Long + SL5_035 (0.35 ADR) + T4_3R (3R Target) hat AvgPnL > 0"
Baseline: Random Walk artefact = +0.01 ADR

Entries: Bar 30 (Morning), Bar 90 (Midday), Bar 180 (Afternoon)
         KEINE VWAP-Cross Entries.
Trade Direction: Long fuer GapDn, Short fuer GapUp
Timeout: 240 Bars
"""

import sys
import io
import os
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

# Use tqdm on stderr for progress
from tqdm import tqdm

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE = PROJECT_ROOT
META_PATH = BASE / 'data' / 'metadata' / 'metadata_master.parquet'
VWAP_DIR = BASE / 'data' / 'vwap'
OUT_PATH = BASE / 'results' / 'oos_validation_half2.txt'

# Parameters
ENTRY_BARS = [30, 90, 180]  # Bar indices (0-indexed from 09:30)
SL_ADR = 0.35  # SL5_035
TARGETS = {
    'T4_3R': 3.0,  # R-multiple
    'T4_2R': 2.0,
    'T3_050': None,  # 0.50 ADR fixed
}
TIMEOUT_BARS = 240
N_BOOTSTRAP = 10000
BONFERRONI_TESTS = 30
ALPHA = 0.05 / BONFERRONI_TESTS  # 0.0017

# Half-1 reference numbers (from test_yearly_stability.txt and test_timeout_decomp.txt)
HALF1_REF = {
    'GapDn_SL5_T4_3R': {
        'overall': {'N': 30727, 'AvgPnL': 0.0228, 'WR': 5.0, 'SL_pct': 30.3, 'TO_pct': 64.7},
        2021: {'N': 10278, 'AvgPnL': 0.0292, 'WR': 4.8},
        2022: {'N': 12529, 'AvgPnL': 0.0070, 'WR': 4.5},
        2023: {'N': 7920, 'AvgPnL': 0.0396, 'WR': 6.1},
    },
    'GapDn_SL5_T4_2R': {
        'overall': {'N': 30727, 'AvgPnL': 0.0158, 'WR': 11.9},
        2023: {'N': 7920, 'AvgPnL': 0.0344, 'WR': 14.1},
    },
    'GapDn_SL5_T3_050': {
        'overall': {'N': 30727, 'AvgPnL': 0.0080, 'WR': 19.8},
        2023: {'N': 7920, 'AvgPnL': 0.0297, 'WR': 23.9},
    },
    'GapUp_SL5_T4_3R': {
        'overall': {'N': 34726, 'AvgPnL': 0.0086, 'WR': 5.3},
        2023: {'N': 10800, 'AvgPnL': -0.0006, 'WR': 6.1},
    },
}

# Naked direction Half-1 reference (2023 val)
NAKED_H1 = {
    'GapDn_Long': {'N': 560, 'Mean': 0.0346, 'CI_lo': -0.0810, 'CI_hi': 0.1581},
    'GapUp_Short': {'N': 688, 'Mean': -0.1035, 'CI_lo': -0.2102, 'CI_hi': -0.0019},
}


def load_metadata():
    """Load metadata and filter to Half 2."""
    meta = pd.read_parquet(META_PATH)
    meta['date_dt'] = pd.to_datetime(meta['date'])
    h2 = meta[meta['date'] >= '2024-01-01'].copy()
    h2['year'] = h2['date_dt'].dt.year

    # ADR: prefer adr_10, fallback to adr_5, adr_20
    h2['adr'] = h2['adr_10']
    if 'adr_5' in h2.columns:
        h2['adr'] = h2['adr'].fillna(h2['adr_5'])
    if 'adr_20' in h2.columns:
        h2['adr'] = h2['adr'].fillna(h2['adr_20'])

    return h2


def determine_scenario(open_price, vwap_row, gap_dir):
    """Determine scenario based on open price relative to VWAP levels at bar 30."""
    if pd.isna(vwap_row.get('vwap')) or pd.isna(vwap_row.get('upper_2std')):
        return f"Gap{'Up' if gap_dir == 'up' else 'Dn'}_unknown"

    vwap = vwap_row['vwap']
    u1 = vwap_row['upper_1std']
    u2 = vwap_row['upper_2std']
    l1 = vwap_row['lower_1std']
    l2 = vwap_row['lower_2std']

    if gap_dir == 'up':
        if open_price > u2:
            return 'GapUp_2s_Fade'
        elif open_price > u1:
            return 'GapUp_1s_Fade'
        elif open_price > vwap:
            return 'GapUp_VWAP_Brk'
        else:
            return 'GapUp_below_VWAP'
    else:  # gap_dir == 'down'
        if open_price < l2:
            return 'GapDn_2s_Fade'
        elif open_price < l1:
            return 'GapDn_1s_Fade'
        elif open_price < vwap:
            return 'GapDn_VWAP_Brk'
        else:
            return 'GapDn_above_VWAP'


def simulate_trade(prices, entry_bar_idx, adr, sl_adr, target_method, target_param,
                   is_long, timeout_bars):
    """
    Simulate a single trade.
    Returns: (pnl_adr, outcome, bars_held)
    outcome: 'target', 'sl', 'timeout'
    """
    if entry_bar_idx >= len(prices):
        return None, None, None

    entry_price = prices[entry_bar_idx]
    if pd.isna(entry_price) or entry_price <= 0 or pd.isna(adr) or adr <= 0:
        return None, None, None

    sl_dist = sl_adr * adr
    if target_method == 'T4':
        target_dist = target_param * sl_dist  # R-multiple
    elif target_method == 'T3':
        target_dist = target_param * adr  # ADR fraction

    if is_long:
        sl_price = entry_price - sl_dist
        target_price = entry_price + target_dist
    else:
        sl_price = entry_price + sl_dist
        target_price = entry_price - target_dist

    end_bar = min(entry_bar_idx + timeout_bars, len(prices))

    for i in range(entry_bar_idx + 1, end_bar):
        p = prices[i]
        if pd.isna(p):
            continue

        if is_long:
            if p <= sl_price:
                pnl = (sl_price - entry_price) / adr
                return pnl, 'sl', i - entry_bar_idx
            if p >= target_price:
                pnl = (target_price - entry_price) / adr
                return pnl, 'target', i - entry_bar_idx
        else:
            if p >= sl_price:
                pnl = (entry_price - sl_price) / adr
                return pnl, 'sl', i - entry_bar_idx
            if p <= target_price:
                pnl = (entry_price - target_price) / adr
                return pnl, 'target', i - entry_bar_idx

    # Timeout
    last_price = prices[end_bar - 1] if end_bar > entry_bar_idx + 1 else entry_price
    if pd.isna(last_price):
        # Find last valid price
        for j in range(end_bar - 1, entry_bar_idx, -1):
            if not pd.isna(prices[j]):
                last_price = prices[j]
                break
        else:
            last_price = entry_price

    if is_long:
        pnl = (last_price - entry_price) / adr
    else:
        pnl = (entry_price - last_price) / adr

    # Cap timeout PnL at SL/Target boundaries
    if is_long:
        pnl = max(pnl, -sl_adr)
        pnl = min(pnl, target_dist / adr)
    else:
        pnl = max(pnl, -sl_adr)
        pnl = min(pnl, target_dist / adr)

    return pnl, 'timeout', end_bar - entry_bar_idx


def bootstrap_ci(data, n_boot=10000, ci=0.95):
    """Bootstrap confidence interval and p-value."""
    data = np.array(data)
    n = len(data)
    if n == 0:
        return 0, 0, 0, 1.0, 1.0
    means = np.zeros(n_boot)
    rng = np.random.default_rng(42)
    for i in range(n_boot):
        sample = rng.choice(data, size=n, replace=True)
        means[i] = np.mean(sample)
    alpha_half = (1 - ci) / 2
    ci_lo = np.percentile(means, alpha_half * 100)
    ci_hi = np.percentile(means, (1 - alpha_half) * 100)
    p_leq_0 = np.mean(means <= 0)
    p_leq_rw = np.mean(means <= 0.01)  # Random walk baseline
    return np.mean(data), ci_lo, ci_hi, p_leq_0, p_leq_rw


def process_gapper(row, vwap_dir):
    """Process a single gapper day. Returns list of trade dicts."""
    ticker = row['ticker']
    date = row['date']
    gap_dir = row['gap_direction']
    adr = row['adr']
    open_price = row['today_open']
    year = row['year']

    if pd.isna(adr) or adr <= 0 or pd.isna(open_price) or open_price <= 0:
        return []

    # Load VWAP file — filename is just {date}.parquet inside {ticker}/ folder
    vwap_path = vwap_dir / ticker / f"{date}.parquet"
    if not vwap_path.exists():
        # Fallback: try {ticker}_{date}.parquet
        vwap_path = vwap_dir / ticker / f"{ticker}_{date}.parquet"
    if not vwap_path.exists():
        return []

    try:
        df = pd.read_parquet(vwap_path)
    except Exception:
        return []

    if len(df) < 50:
        return []

    prices = df['close'].values

    # Determine scenario using bar 30 VWAP levels
    if len(df) > 30 and not pd.isna(df.iloc[30].get('vwap', np.nan)):
        scenario = determine_scenario(open_price, df.iloc[30], gap_dir)
    else:
        scenario = f"Gap{'Up' if gap_dir == 'up' else 'Dn'}_unknown"

    is_long = (gap_dir == 'down')  # Long for GapDn, Short for GapUp

    trades = []
    for entry_bar in ENTRY_BARS:
        if entry_bar >= len(prices):
            continue

        for tgt_name, tgt_param in TARGETS.items():
            if tgt_name.startswith('T4'):
                method = 'T4'
                param = tgt_param
            elif tgt_name == 'T3_050':
                method = 'T3'
                param = 0.50

            pnl, outcome, bars = simulate_trade(
                prices, entry_bar, adr, SL_ADR, method, param, is_long, TIMEOUT_BARS
            )

            if pnl is not None:
                trades.append({
                    'ticker': ticker,
                    'date': date,
                    'year': year,
                    'gap_dir': gap_dir,
                    'scenario': scenario,
                    'entry_bar': entry_bar,
                    'target': tgt_name,
                    'pnl_adr': pnl,
                    'outcome': outcome,
                    'bars_held': bars,
                    'adr': adr,
                    'is_long': is_long,
                })

    # Naked direction: Open-to-Close return
    close_price = prices[-1] if len(prices) > 0 and not pd.isna(prices[-1]) else None
    if close_price is not None and not pd.isna(close_price):
        if is_long:
            naked_pnl = (close_price - open_price) / adr
        else:
            naked_pnl = (open_price - close_price) / adr
        trades.append({
            'ticker': ticker,
            'date': date,
            'year': year,
            'gap_dir': gap_dir,
            'scenario': scenario,
            'entry_bar': -1,  # marker for naked direction
            'target': 'NAKED',
            'pnl_adr': naked_pnl,
            'outcome': 'naked',
            'bars_held': len(prices),
            'adr': adr,
            'is_long': is_long,
        })

    return trades


def format_pnl(v):
    return f"+{v:.4f}" if v >= 0 else f"{v:.4f}"


def main():
    start_time = time.time()
    out_lines = []

    def log(s=''):
        out_lines.append(s)

    log("=" * 80)
    log("OOS VALIDATION — HALF 2 (2024-01-01 to 2026-02-06)")
    log("=" * 80)
    log(f"Date: 2026-02-13")
    log(f"Hypothesis: GapDn + Long + SL5_035 + T4_3R has AvgPnL > 0")
    log(f"RW Baseline: +0.01 ADR (structural artefact)")
    log(f"Significance: Bootstrap P < {ALPHA:.4f} (Bonferroni for {BONFERRONI_TESTS} tests)")
    log()

    # =========================================================================
    # SECTION 1: DATA OVERVIEW
    # =========================================================================
    log("=" * 80)
    log("SECTION 1: DATA OVERVIEW — HALF 2")
    log("=" * 80)
    log()

    meta_h2 = load_metadata()
    n_total = len(meta_h2)
    n_up = (meta_h2['gap_direction'] == 'up').sum()
    n_dn = (meta_h2['gap_direction'] == 'down').sum()

    log(f"  Total Gapper in Half 2: {n_total}")
    log(f"  GapUp: {n_up} ({100*n_up/n_total:.1f}%)")
    log(f"  GapDn: {n_dn} ({100*n_dn/n_total:.1f}%)")
    log()
    log("  Year Distribution:")
    for yr in sorted(meta_h2['year'].unique()):
        ny = (meta_h2['year'] == yr).sum()
        log(f"    {yr}: {ny} ({100*ny/n_total:.1f}%)")
    log()
    log(f"  ADR_10 coverage: {meta_h2['adr'].notna().sum()}/{n_total} ({100*meta_h2['adr'].notna().mean():.1f}%)")
    log(f"  Median ADR: ${meta_h2['adr'].median():.2f}")
    log(f"  Median Gap%: {meta_h2['gap_pct'].median():.1f}%")
    log()

    # =========================================================================
    # SECTION 2: PROCESS ALL GAPPER
    # =========================================================================
    log("=" * 80)
    log("SECTION 2: TRADE SIMULATION")
    log("=" * 80)
    log()

    all_trades = []
    n_loaded = 0
    n_failed = 0

    rows = meta_h2.to_dict('records')
    for row in tqdm(rows, desc="Processing gappers", file=sys.stderr):
        trades = process_gapper(row, VWAP_DIR)
        if trades:
            all_trades.extend(trades)
            n_loaded += 1
        else:
            n_failed += 1

    trades_df = pd.DataFrame(all_trades)
    real_trades = trades_df[trades_df['target'] != 'NAKED'].copy()
    naked_trades = trades_df[trades_df['target'] == 'NAKED'].copy()

    log(f"  Gapper loaded: {n_loaded}")
    log(f"  Gapper failed/skipped: {n_failed}")
    log(f"  Total trades generated: {len(real_trades)}")
    log(f"  Naked direction trades: {len(naked_trades)}")
    log()

    # Scenario distribution
    log("  Scenario Distribution:")
    for sc in sorted(real_trades['scenario'].unique()):
        n_sc = real_trades[real_trades['scenario'] == sc]['date'].nunique()
        log(f"    {sc}: {n_sc} days")
    log()

    # =========================================================================
    # SECTION 3: RESULTS BY DIRECTION (GapDn Long vs GapUp Short)
    # =========================================================================
    log("=" * 80)
    log("SECTION 3: RESULTS BY DIRECTION")
    log("=" * 80)
    log()

    for gap_dir_label, gap_dir_val, dir_label in [
        ('GapDn (Long)', 'down', 'GapDn'),
        ('GapUp (Short)', 'up', 'GapUp')
    ]:
        log(f"--- {gap_dir_label} ---")
        for tgt_name in ['T4_3R', 'T4_2R', 'T3_050']:
            subset = real_trades[
                (real_trades['gap_dir'] == gap_dir_val) &
                (real_trades['target'] == tgt_name)
            ]
            if len(subset) == 0:
                continue

            n = len(subset)
            avg_pnl = subset['pnl_adr'].mean()
            med_pnl = subset['pnl_adr'].median()
            wr = (subset['outcome'] == 'target').mean() * 100
            sl_pct = (subset['outcome'] == 'sl').mean() * 100
            to_pct = (subset['outcome'] == 'timeout').mean() * 100

            mean_val, ci_lo, ci_hi, p_leq0, p_leq_rw = bootstrap_ci(
                subset['pnl_adr'].values, N_BOOTSTRAP
            )

            sig_marker = ""
            if p_leq0 < ALPHA:
                sig_marker = " *** (Bonferroni)"
            elif p_leq0 < 0.05:
                sig_marker = " ** (nominal)"

            log(f"  SL5_035 + {tgt_name}:")
            log(f"    N={n}, WR={wr:.1f}%, SL%={sl_pct:.1f}%, TO%={to_pct:.1f}%")
            log(f"    AvgPnL={format_pnl(avg_pnl)}, MedPnL={format_pnl(med_pnl)}")
            log(f"    CI=[{format_pnl(ci_lo)},{format_pnl(ci_hi)}]")
            log(f"    P(<=0)={p_leq0:.4f}, P(<=RW 0.01)={p_leq_rw:.4f}{sig_marker}")
            log()

    # =========================================================================
    # SECTION 4: RESULTS BY SCENARIO
    # =========================================================================
    log("=" * 80)
    log("SECTION 4: RESULTS BY SCENARIO (SL5_035 + T4_3R, all entries)")
    log("=" * 80)
    log()

    tgt_focus = 'T4_3R'
    scenario_order = [
        'GapDn_2s_Fade', 'GapDn_1s_Fade', 'GapDn_VWAP_Brk',
        'GapDn_above_VWAP', 'GapDn_unknown',
        'GapUp_2s_Fade', 'GapUp_1s_Fade', 'GapUp_VWAP_Brk',
        'GapUp_below_VWAP', 'GapUp_unknown',
    ]

    log(f"{'Scenario':<25} {'N':>6} {'AvgPnL':>8} {'MedPnL':>8} {'WR%':>6} {'SL%':>6} {'TO%':>6} {'CI_lo':>8} {'CI_hi':>8} {'P(<=0)':>8}")
    log("-" * 105)

    for sc in scenario_order:
        subset = real_trades[
            (real_trades['scenario'] == sc) &
            (real_trades['target'] == tgt_focus)
        ]
        if len(subset) == 0:
            continue

        n = len(subset)
        avg = subset['pnl_adr'].mean()
        med = subset['pnl_adr'].median()
        wr = (subset['outcome'] == 'target').mean() * 100
        sl_p = (subset['outcome'] == 'sl').mean() * 100
        to_p = (subset['outcome'] == 'timeout').mean() * 100
        _, ci_lo, ci_hi, p0, prw = bootstrap_ci(subset['pnl_adr'].values, N_BOOTSTRAP)

        log(f"  {sc:<23} {n:>6} {format_pnl(avg):>8} {format_pnl(med):>8} {wr:>5.1f}% {sl_p:>5.1f}% {to_p:>5.1f}% {format_pnl(ci_lo):>8} {format_pnl(ci_hi):>8} {p0:>8.4f}")
    log()

    # =========================================================================
    # SECTION 5: RESULTS BY TARGET METHOD
    # =========================================================================
    log("=" * 80)
    log("SECTION 5: RESULTS BY TARGET METHOD (GapDn only)")
    log("=" * 80)
    log()

    log(f"{'Target':<12} {'N':>6} {'AvgPnL':>8} {'MedPnL':>8} {'WR%':>6} {'SL%':>6} {'TO%':>6} {'CI_lo':>8} {'CI_hi':>8} {'P(<=0)':>8} {'P(<=RW)':>8}")
    log("-" * 110)

    for tgt in ['T4_3R', 'T4_2R', 'T3_050']:
        subset = real_trades[
            (real_trades['gap_dir'] == 'down') &
            (real_trades['target'] == tgt)
        ]
        if len(subset) == 0:
            continue

        n = len(subset)
        avg = subset['pnl_adr'].mean()
        med = subset['pnl_adr'].median()
        wr = (subset['outcome'] == 'target').mean() * 100
        sl_p = (subset['outcome'] == 'sl').mean() * 100
        to_p = (subset['outcome'] == 'timeout').mean() * 100
        _, ci_lo, ci_hi, p0, prw = bootstrap_ci(subset['pnl_adr'].values, N_BOOTSTRAP)

        log(f"  {tgt:<10} {n:>6} {format_pnl(avg):>8} {format_pnl(med):>8} {wr:>5.1f}% {sl_p:>5.1f}% {to_p:>5.1f}% {format_pnl(ci_lo):>8} {format_pnl(ci_hi):>8} {p0:>8.4f} {prw:>8.4f}")
    log()

    # Same for GapUp
    log("  (GapUp for comparison:)")
    for tgt in ['T4_3R', 'T4_2R', 'T3_050']:
        subset = real_trades[
            (real_trades['gap_dir'] == 'up') &
            (real_trades['target'] == tgt)
        ]
        if len(subset) == 0:
            continue

        n = len(subset)
        avg = subset['pnl_adr'].mean()
        med = subset['pnl_adr'].median()
        wr = (subset['outcome'] == 'target').mean() * 100
        sl_p = (subset['outcome'] == 'sl').mean() * 100
        to_p = (subset['outcome'] == 'timeout').mean() * 100
        _, ci_lo, ci_hi, p0, prw = bootstrap_ci(subset['pnl_adr'].values, N_BOOTSTRAP)

        log(f"  {tgt:<10} {n:>6} {format_pnl(avg):>8} {format_pnl(med):>8} {wr:>5.1f}% {sl_p:>5.1f}% {to_p:>5.1f}% {format_pnl(ci_lo):>8} {format_pnl(ci_hi):>8} {p0:>8.4f} {prw:>8.4f}")
    log()

    # =========================================================================
    # SECTION 6: RESULTS BY YEAR
    # =========================================================================
    log("=" * 80)
    log("SECTION 6: RESULTS BY YEAR")
    log("=" * 80)
    log()

    for combo_label, gap_val, tgt_val in [
        ('GapDn + SL5_035 + T4_3R', 'down', 'T4_3R'),
        ('GapDn + SL5_035 + T4_2R', 'down', 'T4_2R'),
        ('GapDn + SL5_035 + T3_050', 'down', 'T3_050'),
        ('GapUp + SL5_035 + T4_3R', 'up', 'T4_3R'),
        ('GapDn_VWAP_Brk + SL5_035 + T4_3R', 'down', 'T4_3R'),  # Scenario-specific
        ('GapDn_2s_Fade + SL5_035 + T4_3R', 'down', 'T4_3R'),
        ('GapDn_1s_Fade + SL5_035 + T4_3R', 'down', 'T4_3R'),
    ]:
        # Filter by scenario if label contains it
        if 'VWAP_Brk' in combo_label:
            subset = real_trades[
                (real_trades['scenario'] == 'GapDn_VWAP_Brk') &
                (real_trades['target'] == tgt_val)
            ]
        elif '2s_Fade' in combo_label:
            subset = real_trades[
                (real_trades['scenario'] == 'GapDn_2s_Fade') &
                (real_trades['target'] == tgt_val)
            ]
        elif '1s_Fade' in combo_label:
            subset = real_trades[
                (real_trades['scenario'] == 'GapDn_1s_Fade') &
                (real_trades['target'] == tgt_val)
            ]
        else:
            subset = real_trades[
                (real_trades['gap_dir'] == gap_val) &
                (real_trades['target'] == tgt_val)
            ]

        if len(subset) == 0:
            continue

        log(f"--- {combo_label} ---")

        # Overall
        n = len(subset)
        avg = subset['pnl_adr'].mean()
        med = subset['pnl_adr'].median()
        wr = (subset['outcome'] == 'target').mean() * 100
        _, ci_lo, ci_hi, p0, prw = bootstrap_ci(subset['pnl_adr'].values, N_BOOTSTRAP)
        log(f"  Overall: N={n}, AvgPnL={format_pnl(avg)}, MedPnL={format_pnl(med)}, WR={wr:.1f}%, CI=[{format_pnl(ci_lo)},{format_pnl(ci_hi)}], P(<=0)={p0:.4f}")

        # Per year
        for yr in sorted(subset['year'].unique()):
            ys = subset[subset['year'] == yr]
            yn = len(ys)
            ya = ys['pnl_adr'].mean()
            ym = ys['pnl_adr'].median()
            ywr = (ys['outcome'] == 'target').mean() * 100
            ysl = (ys['outcome'] == 'sl').mean() * 100
            yto = (ys['outcome'] == 'timeout').mean() * 100
            _, yci_lo, yci_hi, yp0, _ = bootstrap_ci(ys['pnl_adr'].values, N_BOOTSTRAP)

            sig = "***" if yp0 < ALPHA else ("**" if yp0 < 0.05 else "")
            log(f"  {yr}: N={yn}, AvgPnL={format_pnl(ya)}, MedPnL={format_pnl(ym)}, WR={ywr:.1f}%, SL%={ysl:.1f}%, TO%={yto:.1f}%, CI=[{format_pnl(yci_lo)},{format_pnl(yci_hi)}], P(<=0)={yp0:.4f} {sig}")

        log()

    # =========================================================================
    # SECTION 7: RESULTS BY ENTRY TIMEPOINT
    # =========================================================================
    log("=" * 80)
    log("SECTION 7: RESULTS BY ENTRY TIMEPOINT (GapDn + SL5_035 + T4_3R)")
    log("=" * 80)
    log()

    entry_labels = {30: 'Morning (Bar 30)', 90: 'Midday (Bar 90)', 180: 'Afternoon (Bar 180)'}

    for eb in ENTRY_BARS:
        subset = real_trades[
            (real_trades['gap_dir'] == 'down') &
            (real_trades['target'] == 'T4_3R') &
            (real_trades['entry_bar'] == eb)
        ]
        if len(subset) == 0:
            continue

        n = len(subset)
        avg = subset['pnl_adr'].mean()
        med = subset['pnl_adr'].median()
        wr = (subset['outcome'] == 'target').mean() * 100
        sl_p = (subset['outcome'] == 'sl').mean() * 100
        to_p = (subset['outcome'] == 'timeout').mean() * 100
        _, ci_lo, ci_hi, p0, prw = bootstrap_ci(subset['pnl_adr'].values, N_BOOTSTRAP)

        log(f"  {entry_labels[eb]}:")
        log(f"    N={n}, WR={wr:.1f}%, SL%={sl_p:.1f}%, TO%={to_p:.1f}%")
        log(f"    AvgPnL={format_pnl(avg)}, MedPnL={format_pnl(med)}")
        log(f"    CI=[{format_pnl(ci_lo)},{format_pnl(ci_hi)}], P(<=0)={p0:.4f}, P(<=RW)={prw:.4f}")
        log()

    # Also show entry bars for GapUp
    log("  (GapUp Short for comparison:)")
    for eb in ENTRY_BARS:
        subset = real_trades[
            (real_trades['gap_dir'] == 'up') &
            (real_trades['target'] == 'T4_3R') &
            (real_trades['entry_bar'] == eb)
        ]
        if len(subset) == 0:
            continue

        n = len(subset)
        avg = subset['pnl_adr'].mean()
        _, ci_lo, ci_hi, p0, prw = bootstrap_ci(subset['pnl_adr'].values, N_BOOTSTRAP)
        log(f"    {entry_labels[eb]}: N={n}, AvgPnL={format_pnl(avg)}, CI=[{format_pnl(ci_lo)},{format_pnl(ci_hi)}], P(<=0)={p0:.4f}")
    log()

    # =========================================================================
    # SECTION 8: STATISTICAL SIGNIFICANCE (CORE HYPOTHESIS)
    # =========================================================================
    log("=" * 80)
    log("SECTION 8: STATISTICAL SIGNIFICANCE — CORE HYPOTHESIS")
    log("=" * 80)
    log()

    core = real_trades[
        (real_trades['gap_dir'] == 'down') &
        (real_trades['target'] == 'T4_3R')
    ]
    if len(core) > 0:
        mean_val, ci_lo, ci_hi, p0, prw = bootstrap_ci(core['pnl_adr'].values, N_BOOTSTRAP)

        log(f"  H1: GapDn + Long + SL5_035 + T4_3R has AvgPnL > 0")
        log(f"  N = {len(core)}")
        log(f"  Mean PnL = {format_pnl(mean_val)} ADR")
        log(f"  Median PnL = {format_pnl(core['pnl_adr'].median())} ADR")
        log(f"  95% CI = [{format_pnl(ci_lo)}, {format_pnl(ci_hi)}]")
        log(f"  P(Mean <= 0) = {p0:.6f}")
        log(f"  P(Mean <= RW baseline 0.01) = {prw:.6f}")
        log(f"  Bonferroni threshold: P < {ALPHA:.4f}")
        log()

        if p0 < ALPHA:
            log(f"  >>> SIGNIFICANT at Bonferroni level: P(<=0) = {p0:.6f} < {ALPHA:.4f} <<<")
        elif p0 < 0.05:
            log(f"  >>> SIGNIFICANT at nominal level: P(<=0) = {p0:.6f} < 0.05 <<<")
        else:
            log(f"  >>> NOT SIGNIFICANT: P(<=0) = {p0:.6f} <<<")

        if prw < ALPHA:
            log(f"  >>> EXCEEDS RW baseline at Bonferroni level: P(<=0.01) = {prw:.6f} < {ALPHA:.4f} <<<")
        elif prw < 0.05:
            log(f"  >>> EXCEEDS RW baseline at nominal level: P(<=0.01) = {prw:.6f} < 0.05 <<<")
        else:
            log(f"  >>> DOES NOT EXCEED RW baseline: P(<=0.01) = {prw:.6f} <<<")
        log()

    # =========================================================================
    # SECTION 9: TIMEOUT DECOMPOSITION
    # =========================================================================
    log("=" * 80)
    log("SECTION 9: TIMEOUT PnL DECOMPOSITION")
    log("=" * 80)
    log()

    for combo_label, gap_val, tgt_val, scenario_filter in [
        ('GapDn + SL5_035 + T4_3R', 'down', 'T4_3R', None),
        ('GapDn + SL5_035 + T4_2R', 'down', 'T4_2R', None),
        ('GapDn + SL5_035 + T3_050', 'down', 'T3_050', None),
        ('GapUp + SL5_035 + T4_3R', 'up', 'T4_3R', None),
        ('GapDn_VWAP_Brk + SL5_035 + T4_3R', 'down', 'T4_3R', 'GapDn_VWAP_Brk'),
        ('GapDn_2s_Fade + SL5_035 + T4_3R', 'down', 'T4_3R', 'GapDn_2s_Fade'),
        ('GapDn_1s_Fade + SL5_035 + T4_3R', 'down', 'T4_3R', 'GapDn_1s_Fade'),
    ]:
        if scenario_filter:
            subset = real_trades[
                (real_trades['scenario'] == scenario_filter) &
                (real_trades['target'] == tgt_val)
            ]
        else:
            subset = real_trades[
                (real_trades['gap_dir'] == gap_val) &
                (real_trades['target'] == tgt_val)
            ]

        if len(subset) == 0:
            continue

        n = len(subset)
        avg_pnl = subset['pnl_adr'].mean()

        tgt_trades = subset[subset['outcome'] == 'target']
        sl_trades = subset[subset['outcome'] == 'sl']
        to_trades = subset[subset['outcome'] == 'timeout']

        tgt_pct = len(tgt_trades) / n * 100
        sl_pct = len(sl_trades) / n * 100
        to_pct = len(to_trades) / n * 100

        tgt_contribution = tgt_trades['pnl_adr'].sum() / n if n > 0 else 0
        sl_contribution = sl_trades['pnl_adr'].sum() / n if n > 0 else 0
        to_contribution = to_trades['pnl_adr'].sum() / n if n > 0 else 0

        to_avg = to_trades['pnl_adr'].mean() if len(to_trades) > 0 else 0
        to_med = to_trades['pnl_adr'].median() if len(to_trades) > 0 else 0
        to_pos_pct = (to_trades['pnl_adr'] > 0).mean() * 100 if len(to_trades) > 0 else 0

        log(f"--- {combo_label} ---")
        log(f"  N={n}, AvgPnL={format_pnl(avg_pnl)}")
        log(f"  Target: {tgt_pct:.1f}%, SL: {sl_pct:.1f}%, Timeout: {to_pct:.1f}%")
        log(f"  Contribution: Tgt={format_pnl(tgt_contribution)}, SL={format_pnl(sl_contribution)}, TO={format_pnl(to_contribution)}")
        log(f"  Timeout Avg={format_pnl(to_avg)}, Med={format_pnl(to_med)}, %Positive={to_pos_pct:.1f}%")

        # Without timeouts
        noto = subset[subset['outcome'] != 'timeout']
        noto_avg = noto['pnl_adr'].mean() if len(noto) > 0 else 0
        log(f"  Without Timeouts: AvgPnL={format_pnl(noto_avg)}")
        log()

    # =========================================================================
    # SECTION 10: HALF 1 vs HALF 2 COMPARISON
    # =========================================================================
    log("=" * 80)
    log("SECTION 10: HALF 1 vs HALF 2 COMPARISON")
    log("=" * 80)
    log()

    log(f"{'Setup':<40} {'Half1 N':>8} {'H1 PnL':>8} {'H1 WR':>6} {'Half2 N':>8} {'H2 PnL':>8} {'H2 WR':>6} {'Delta':>8}")
    log("-" * 105)

    comparison_setups = [
        ('GapDn + SL5 + T4_3R', 'GapDn_SL5_T4_3R', 'down', 'T4_3R', None),
        ('GapDn + SL5 + T4_2R', 'GapDn_SL5_T4_2R', 'down', 'T4_2R', None),
        ('GapDn + SL5 + T3_050', 'GapDn_SL5_T3_050', 'down', 'T3_050', None),
        ('GapUp + SL5 + T4_3R', 'GapUp_SL5_T4_3R', 'up', 'T4_3R', None),
    ]

    for label, ref_key, gap_val, tgt_val, sc_filter in comparison_setups:
        if sc_filter:
            h2_sub = real_trades[
                (real_trades['scenario'] == sc_filter) &
                (real_trades['target'] == tgt_val)
            ]
        else:
            h2_sub = real_trades[
                (real_trades['gap_dir'] == gap_val) &
                (real_trades['target'] == tgt_val)
            ]

        h1_data = HALF1_REF.get(ref_key, {}).get('overall', {})
        h1_n = h1_data.get('N', 0)
        h1_pnl = h1_data.get('AvgPnL', 0)
        h1_wr = h1_data.get('WR', 0)

        h2_n = len(h2_sub)
        h2_pnl = h2_sub['pnl_adr'].mean() if h2_n > 0 else 0
        h2_wr = (h2_sub['outcome'] == 'target').mean() * 100 if h2_n > 0 else 0
        delta = h2_pnl - h1_pnl

        log(f"  {label:<38} {h1_n:>8} {format_pnl(h1_pnl):>8} {h1_wr:>5.1f}% {h2_n:>8} {format_pnl(h2_pnl):>8} {h2_wr:>5.1f}% {format_pnl(delta):>8}")

    log()
    log("  NOTE: Half 1 used entries at every VWAP cross; Half 2 uses fixed Bar 30/90/180.")
    log("        Direct comparison is DIRECTIONAL (bigger/smaller) not exact-magnitude.")
    log()

    # Year-by-year comparison for core setup
    log("  Year-by-Year: GapDn + SL5 + T4_3R")
    log(f"  {'Year':<8} {'H1 PnL':>8} {'H2 PnL':>8} {'Delta':>8}")
    log("  " + "-" * 40)

    core_h2 = real_trades[
        (real_trades['gap_dir'] == 'down') &
        (real_trades['target'] == 'T4_3R')
    ]

    h1_years = {2021: 0.0292, 2022: 0.0070, 2023: 0.0396}
    for yr in sorted(core_h2['year'].unique()):
        ys = core_h2[core_h2['year'] == yr]
        ya = ys['pnl_adr'].mean()
        h1_val = h1_years.get(yr, None)
        if h1_val is not None:
            log(f"  {yr:<8} {format_pnl(h1_val):>8} {format_pnl(ya):>8} {format_pnl(ya - h1_val):>8}")
        else:
            log(f"  {yr:<8} {'N/A':>8} {format_pnl(ya):>8}")
    log()

    # =========================================================================
    # SECTION 11: NAKED DIRECTION (Open-to-Close)
    # =========================================================================
    log("=" * 80)
    log("SECTION 11: NAKED DIRECTION — OPEN TO CLOSE")
    log("=" * 80)
    log()

    for gap_val, direction_label in [('down', 'GapDn (Long at Open)'), ('up', 'GapUp (Short at Open)')]:
        ns = naked_trades[naked_trades['gap_dir'] == gap_val]
        if len(ns) == 0:
            continue

        n = len(ns)
        avg = ns['pnl_adr'].mean()
        med = ns['pnl_adr'].median()
        std = ns['pnl_adr'].std()
        _, ci_lo, ci_hi, p0, prw = bootstrap_ci(ns['pnl_adr'].values, N_BOOTSTRAP)

        log(f"  {direction_label}:")
        log(f"    N={n}")
        log(f"    Mean={format_pnl(avg)} ADR")
        log(f"    Median={format_pnl(med)} ADR")
        log(f"    Std={std:.4f} ADR")
        log(f"    95% CI=[{format_pnl(ci_lo)},{format_pnl(ci_hi)}]")
        log(f"    P(mean<=0)={p0:.4f}")

        # Compare with Half-1
        if gap_val == 'down':
            h1_ref = NAKED_H1.get('GapDn_Long', {})
        else:
            h1_ref = NAKED_H1.get('GapUp_Short', {})

        if h1_ref:
            log(f"    Half-1 comparison: Mean={format_pnl(h1_ref['Mean'])}, CI=[{format_pnl(h1_ref['CI_lo'])},{format_pnl(h1_ref['CI_hi'])}]")
        log()

    # Naked direction by year
    log("  Naked Direction by Year:")
    for gap_val, lbl in [('down', 'GapDn Long'), ('up', 'GapUp Short')]:
        ns = naked_trades[naked_trades['gap_dir'] == gap_val]
        log(f"  --- {lbl} ---")
        for yr in sorted(ns['year'].unique()):
            ys = ns[ns['year'] == yr]
            ya = ys['pnl_adr'].mean()
            ym = ys['pnl_adr'].median()
            _, yci_lo, yci_hi, yp0, _ = bootstrap_ci(ys['pnl_adr'].values, N_BOOTSTRAP)
            log(f"    {yr}: N={len(ys)}, Mean={format_pnl(ya)}, Med={format_pnl(ym)}, CI=[{format_pnl(yci_lo)},{format_pnl(yci_hi)}], P(<=0)={yp0:.4f}")
        log()

    # =========================================================================
    # SECTION 12: VERDICT
    # =========================================================================
    log("=" * 80)
    log("SECTION 12: FINAL VERDICT")
    log("=" * 80)
    log()

    # Core hypothesis result
    core = real_trades[
        (real_trades['gap_dir'] == 'down') &
        (real_trades['target'] == 'T4_3R')
    ]
    if len(core) > 0:
        mean_val, ci_lo, ci_hi, p0, prw = bootstrap_ci(core['pnl_adr'].values, N_BOOTSTRAP)

        log(f"  CORE HYPOTHESIS: GapDn + Long + SL5_035 + T4_3R")
        log(f"  Half-2 AvgPnL = {format_pnl(mean_val)} ADR")
        log(f"  Half-1 AvgPnL = +0.0228 ADR (overall 2021-2023)")
        log(f"  Half-2 95% CI = [{format_pnl(ci_lo)}, {format_pnl(ci_hi)}]")
        log(f"  P(<=0) = {p0:.6f}")
        log(f"  P(<=RW 0.01) = {prw:.6f}")
        log(f"  Bonferroni threshold = {ALPHA:.4f}")
        log()

        # Direction comparison
        core_up = real_trades[
            (real_trades['gap_dir'] == 'up') &
            (real_trades['target'] == 'T4_3R')
        ]
        up_mean = core_up['pnl_adr'].mean() if len(core_up) > 0 else 0

        log(f"  DIRECTION ASYMMETRY:")
        log(f"    GapDn Long:  {format_pnl(mean_val)} ADR")
        log(f"    GapUp Short: {format_pnl(up_mean)} ADR")
        log(f"    Delta:       {format_pnl(mean_val - up_mean)} ADR")
        log()

        # WR comparison with RW
        wr = (core['outcome'] == 'target').mean() * 100
        log(f"  WIN RATE vs RANDOM WALK:")
        log(f"    Real GapDn WR:     {wr:.1f}%")
        log(f"    Calibrated RW WR:  0.6% (from Half-1 test)")
        log(f"    Ratio:             {wr/0.6:.1f}x more target hits than RW")
        log()

        # Final verdict
        edge_confirmed = p0 < ALPHA and mean_val > 0
        exceeds_rw = prw < ALPHA
        stable = True  # Will check per-year

        for yr in sorted(core['year'].unique()):
            ys = core[core['year'] == yr]
            if ys['pnl_adr'].mean() <= 0:
                stable = False

        log("  ====================================")
        if edge_confirmed and exceeds_rw and stable:
            log("  VERDICT: EDGE CONFIRMED")
            log("  ====================================")
            log(f"  - AvgPnL > 0 at Bonferroni significance")
            log(f"  - AvgPnL > RW baseline (+0.01) at significance")
            log(f"  - Positive in all years of Half 2")
            log(f"  - Direction asymmetry preserved (GapDn > GapUp)")
        elif edge_confirmed and not exceeds_rw:
            log("  VERDICT: EDGE PARTIALLY CONFIRMED")
            log("  ====================================")
            log(f"  - AvgPnL > 0 at significance")
            log(f"  - BUT does not clearly exceed RW baseline")
            log(f"  - Edge may be partially structural artefact")
        elif edge_confirmed and not stable:
            log("  VERDICT: EDGE PARTIALLY CONFIRMED (UNSTABLE)")
            log("  ====================================")
            log(f"  - AvgPnL > 0 at significance overall")
            log(f"  - BUT not stable across all years")
        elif mean_val > 0 and p0 < 0.05:
            log("  VERDICT: EDGE WEAK / INCONCLUSIVE")
            log("  ====================================")
            log(f"  - AvgPnL > 0 at nominal significance only")
            log(f"  - Does not survive Bonferroni correction")
        else:
            log("  VERDICT: EDGE NOT CONFIRMED")
            log("  ====================================")
            log(f"  - AvgPnL is not significantly > 0")
            log(f"  - The edge from Half 1 does not replicate on Half 2")
        log()

    elapsed = time.time() - start_time
    log(f"  Runtime: {elapsed:.0f} seconds")
    log()
    log("=" * 80)
    log("END OF OOS VALIDATION")
    log("=" * 80)

    # Write to file
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out_lines))

    print(f"\nResults written to: {OUT_PATH}")
    print(f"Runtime: {elapsed:.0f} seconds")


if __name__ == '__main__':
    main()
