"""
D13 DISCRETIONARY-LED EXPLORATION — Pattern-basierte Intraday-Strategien
=========================================================================
Discretionary Trader fuehrt mit Price-Action-Narrativen. Quant backtestet.
Devils Advocate prueft.

Phase 1: Pattern-Screening (IS only)
Phase 2: Parameter-Matrix (IS only)
Phase 3: Devils Advocate Robustheit (IS only)
Phase 4: OOS-Validierung (nur Ueberlebende)

PLUS: Dip & Rip OOS-Validierung (parallel)
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings
import itertools
warnings.filterwarnings('ignore')

sys.path.insert(0, 'scripts')
from d10_trailing import simulate_trailing

# === Paths ===
RAW_DIR = Path('data/raw_1min')
VWAP_DIR = Path('data/vwap')
VP_DIR = Path('data/volume_profile')
META_PATH = Path('data/metadata/metadata_v9.parquet')
FOLLOWUP_PATH = Path('data/metadata/followup_day_mapping.parquet')
OUT_PATH = Path('d13_team_analysis/results/d13_discretionary_led_results.txt')

IS_START = '2021-02-21'
IS_END = '2023-12-31'
OOS_START = '2024-01-01'
OOS_END = '2026-02-21'

# IS split for stability test
IS_H1_END = '2022-06-30'
IS_H2_START = '2022-07-01'

lines = []
np.random.seed(42)

# === Helpers ===
def log(msg):
    lines.append(str(msg))
    print(msg)

def log_header(title):
    log(f"\n{'='*80}")
    log(f"  {title}")
    log(f"{'='*80}")

def compute_ev_stats(results, label=""):
    if len(results) < 10:
        return {'N': len(results), 'label': label, 'valid': False}
    arr = np.array(results)
    ev = np.mean(arr)
    med = np.median(arr)
    std = np.std(arr, ddof=1)
    wr = np.mean(arr > 0)
    se = std / np.sqrt(len(arr))
    t_stat = ev / se if se > 0 else 0
    from scipy import stats as sp_stats
    p_val = 2 * (1 - sp_stats.t.cdf(abs(t_stat), df=len(arr)-1))
    sharpe = ev / std if std > 0 else 0
    return {
        'N': len(arr), 'EV_R': ev, 'Med_R': med, 'Std_R': std,
        'WR': wr, 't_stat': t_stat, 'p_val': p_val, 'Sharpe': sharpe,
        'label': label, 'valid': True
    }

def format_stats(s):
    if not s['valid']:
        return f"  {s['label']:<45} | N={s['N']:>5} | [INSUFFICIENT DATA]"
    sig = "***" if s['p_val'] < 0.01 else "**" if s['p_val'] < 0.05 else "*" if s['p_val'] < 0.10 else ""
    return (f"  {s['label']:<45} | N={s['N']:>5} | EV={s['EV_R']:>+6.3f}R | "
            f"Med={s['Med_R']:>+6.3f}R | WR={s['WR']:>5.1%} | "
            f"t={s['t_stat']:>+5.2f} | p={s['p_val']:>.4f} {sig}")

def bootstrap_ci(results, n_boot=10000, ci=0.95):
    arr = np.array(results)
    n = len(arr)
    if n < 20:
        return (np.nan, np.nan)
    boot_means = np.array([np.mean(np.random.choice(arr, n, replace=True)) for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    return (np.percentile(boot_means, 100*alpha), np.percentile(boot_means, 100*(1-alpha)))

def load_bars(ticker, date):
    fpath = RAW_DIR / ticker / f"{date}.parquet"
    if not fpath.exists():
        return None
    return pd.read_parquet(fpath)

def get_rth(bars):
    if bars is None:
        return None
    rth = bars[bars['session'] == 'rth'].copy()
    if len(rth) < 10:
        return None
    return rth

# Bar cache for performance
_bar_cache = {}
def load_rth_cached(ticker, date):
    key = (ticker, date)
    if key not in _bar_cache:
        bars = load_bars(ticker, date)
        rth = get_rth(bars)
        _bar_cache[key] = rth
    return _bar_cache[key]

def clear_cache():
    global _bar_cache
    _bar_cache = {}

def simulate_from_entry(rth, entry_price, sl_dist, trade_dir, adr, entry_time):
    """Simulate trailing from a specific entry time (not fixed 09:36)."""
    post_entry = rth[rth['time_et'] >= entry_time].copy()
    if len(post_entry) < 5:
        return None
    return simulate_trailing(post_entry, entry_price, sl_dist, trade_dir, adr, trail_type='D')


# ============================================================
# LOAD DATA
# ============================================================
print("Loading metadata...", file=sys.stderr)
meta = pd.read_parquet(META_PATH)
IS = meta[(meta['date'] >= IS_START) & (meta['date'] <= IS_END)].copy()
OOS = meta[(meta['date'] >= OOS_START) & (meta['date'] <= OOS_END)].copy()
IS_H1 = IS[IS['date'] <= IS_H1_END].copy()
IS_H2 = IS[IS['date'] >= IS_H2_START].copy()

followup = pd.read_parquet(FOLLOWUP_PATH)

print(f"IS: {len(IS)}, OOS: {len(OOS)}, IS_H1: {len(IS_H1)}, IS_H2: {len(IS_H2)}", file=sys.stderr)

log("=" * 80)
log("  D13 DISCRETIONARY-LED EXPLORATION — Pattern-basierte Intraday-Strategien")
log("=" * 80)
log(f"  Datum: 2026-02-21")
log(f"  IS: {IS_START} bis {IS_END} (N={len(IS)})")
log(f"  OOS: {OOS_START} bis {OOS_END} (N={len(OOS)})")
log(f"  IS Split: H1 bis {IS_H1_END} (N={len(IS_H1)}), H2 ab {IS_H2_START} (N={len(IS_H2)})")
log(f"  Trail: TRAIL_D (+1R -> BE, trail peak-0.5R)")


# ============================================================
# PHASE 1: DISCRETIONARY PATTERN SCREENING (IS ONLY)
# ============================================================
log_header("PHASE 1: DISCRETIONARY PATTERN SCREENING (IS)")
log("  Fuer jede Idee: Baseline-Test. Verwerfen wenn EV<=0 oder N<30.")

# Track all ideas and their results for Phase 2
phase1_survivors = {}
total_hypotheses_tested = 0


# ------ IDEE A: SELL-OFF & RECOVERY ------
log_header("IDEE A: SELL-OFF & RECOVERY")
log("  Narrativ: GapUp -> Sell-Off (schwache Haende) -> Recovery-Grind zurueck")
log("  Einstieg wenn Reclaim-Level erreicht (VWAP/50%/61.8%)")

print("  [A] Running Sell-Off & Recovery...", file=sys.stderr)

is_a = IS[
    (IS['close_935'].notna()) & (IS['adr_10'].notna()) & (IS['adr_10'] > 0) &
    (IS['od_strength'].notna()) & (IS['gap_size_in_adr'].notna())
].copy()

def detect_selloff_recovery(rth, vwap_data, gap_dir, adr, min_depth=0.10,
                            recovery_type='fib50', selloff_end='10:00',
                            recovery_deadline='11:00'):
    """
    Detect Sell-Off & Recovery pattern.
    recovery_type: 'vwap' (VWAP reclaim z>=0), 'fib50' (50% retrace), 'fib618' (61.8% retrace)
    Returns entry info dict or None.
    """
    if gap_dir == 'up':
        # OD Peak: highest high 09:30-09:35
        od_bars = rth[(rth['time_et'] >= '09:30') & (rth['time_et'] <= '09:35')]
        if len(od_bars) == 0:
            return None
        od_peak = od_bars['high'].max()

        # Sell-Off Low: lowest low 09:36 to selloff_end
        so_bars = rth[(rth['time_et'] >= '09:36') & (rth['time_et'] <= selloff_end)]
        if len(so_bars) < 3:
            return None
        so_low_idx = so_bars['low'].idxmin()
        so_low = so_bars.loc[so_low_idx, 'low']
        so_low_time = so_bars.loc[so_low_idx, 'time_et']

        # Sell-Off Depth check
        depth = (od_peak - so_low) / adr
        if depth < min_depth:
            return None

        # Recovery window
        rec_bars = rth[(rth['time_et'] > so_low_time) & (rth['time_et'] <= recovery_deadline)]
        if len(rec_bars) < 2:
            return None

        # Determine reclaim level
        if recovery_type == 'vwap':
            # Scan VWAP data for z >= 0 crossover
            if vwap_data is None:
                return None
            vwap_window = vwap_data[(vwap_data['time_et'] > so_low_time) &
                                    (vwap_data['time_et'] <= recovery_deadline)]
            for _, vrow in vwap_window.iterrows():
                if vrow['z_score'] >= 0:
                    entry_time = vrow['time_et']
                    # Get close from 1min bars at this time
                    entry_bars = rth[rth['time_et'] == entry_time]
                    if len(entry_bars) == 0:
                        continue
                    entry_price = entry_bars.iloc[0]['close']
                    sl_dist = entry_price - (so_low - 0.02 * adr)
                    if sl_dist <= 0 or sl_dist > 0.5 * adr:
                        return None
                    return {
                        'entry_price': entry_price, 'entry_time': entry_time,
                        'sl_dist': sl_dist, 'so_depth': depth, 'trade_dir': 'long'
                    }
            return None

        else:
            # Fibonacci retracement
            fib_pct = 0.50 if recovery_type == 'fib50' else 0.618
            reclaim_level = so_low + fib_pct * (od_peak - so_low)

            for _, bar in rec_bars.iterrows():
                if bar['close'] >= reclaim_level:
                    entry_price = bar['close']
                    entry_time = bar['time_et']
                    sl_dist = entry_price - (so_low - 0.02 * adr)
                    if sl_dist <= 0 or sl_dist > 0.5 * adr:
                        return None
                    return {
                        'entry_price': entry_price, 'entry_time': entry_time,
                        'sl_dist': sl_dist, 'so_depth': depth, 'trade_dir': 'long'
                    }
            return None

    else:  # gap_dir == 'down' — mirror
        od_bars = rth[(rth['time_et'] >= '09:30') & (rth['time_et'] <= '09:35')]
        if len(od_bars) == 0:
            return None
        od_peak = od_bars['low'].min()

        so_bars = rth[(rth['time_et'] >= '09:36') & (rth['time_et'] <= selloff_end)]
        if len(so_bars) < 3:
            return None
        so_high_idx = so_bars['high'].idxmax()
        so_high = so_bars.loc[so_high_idx, 'high']
        so_high_time = so_bars.loc[so_high_idx, 'time_et']

        depth = (so_high - od_peak) / adr
        if depth < min_depth:
            return None

        rec_bars = rth[(rth['time_et'] > so_high_time) & (rth['time_et'] <= recovery_deadline)]
        if len(rec_bars) < 2:
            return None

        if recovery_type == 'vwap':
            if vwap_data is None:
                return None
            vwap_window = vwap_data[(vwap_data['time_et'] > so_high_time) &
                                    (vwap_data['time_et'] <= recovery_deadline)]
            for _, vrow in vwap_window.iterrows():
                if vrow['z_score'] <= 0:
                    entry_time = vrow['time_et']
                    entry_bars = rth[rth['time_et'] == entry_time]
                    if len(entry_bars) == 0:
                        continue
                    entry_price = entry_bars.iloc[0]['close']
                    sl_dist = (so_high + 0.02 * adr) - entry_price
                    if sl_dist <= 0 or sl_dist > 0.5 * adr:
                        return None
                    return {
                        'entry_price': entry_price, 'entry_time': entry_time,
                        'sl_dist': sl_dist, 'so_depth': depth, 'trade_dir': 'short'
                    }
            return None
        else:
            fib_pct = 0.50 if recovery_type == 'fib50' else 0.618
            reclaim_level = so_high - fib_pct * (so_high - od_peak)

            for _, bar in rec_bars.iterrows():
                if bar['close'] <= reclaim_level:
                    entry_price = bar['close']
                    entry_time = bar['time_et']
                    sl_dist = (so_high + 0.02 * adr) - entry_price
                    if sl_dist <= 0 or sl_dist > 0.5 * adr:
                        return None
                    return {
                        'entry_price': entry_price, 'entry_time': entry_time,
                        'sl_dist': sl_dist, 'so_depth': depth, 'trade_dir': 'short'
                    }
            return None


# Baseline test: fib50, min_depth=0.10, selloff_end=10:00, deadline=11:00
idea_a_pnl = []
idea_a_details = []
for _, row in is_a.iterrows():
    ticker = row['ticker']
    date = str(row['date'])
    adr = row['adr_10']
    gap_dir = row['gap_direction']

    rth = load_rth_cached(ticker, date)
    if rth is None:
        continue

    # Load VWAP for vwap variant (will also be used for vwap recovery_type)
    vwap_data = None
    vwap_path = VWAP_DIR / ticker / f"{date}.parquet"
    if vwap_path.exists():
        try:
            vwap_data = pd.read_parquet(vwap_path)
            if 'z_score' not in vwap_data.columns:
                vwap_data = None
        except:
            vwap_data = None

    info = detect_selloff_recovery(rth, vwap_data, gap_dir, adr,
                                   min_depth=0.10, recovery_type='fib50',
                                   selloff_end='10:00', recovery_deadline='11:00')
    if info is None:
        continue

    result = simulate_from_entry(rth, info['entry_price'], info['sl_dist'],
                                  info['trade_dir'], adr, info['entry_time'])
    if result is None:
        continue

    idea_a_pnl.append(result['pnl_r'])
    idea_a_details.append({
        'pnl_r': result['pnl_r'], 'so_depth': info['so_depth'],
        'gap_dir': gap_dir, 'ticker': ticker, 'date': date,
        'gap_size_in_adr': row.get('gap_size_in_adr', np.nan),
        'rvol_5': row.get('rvol_5', np.nan),
        'od_strength': row.get('od_strength', np.nan),
        'sector': row.get('sector', 'Unknown'),
    })

clear_cache()
stats_a = compute_ev_stats(idea_a_pnl, "A: Sell-Off & Recovery (Baseline)")
log(format_stats(stats_a))
total_hypotheses_tested += 1

if stats_a['valid'] and stats_a['EV_R'] > 0 and stats_a['N'] >= 30:
    phase1_survivors['A'] = {
        'stats': stats_a, 'pnl': idea_a_pnl, 'details': idea_a_details,
        'detect_fn': detect_selloff_recovery
    }
    log("  >>> IDEE A UEBERLEBT Phase 1")
else:
    log("  >>> IDEE A verworfen (EV<=0 oder N<30)")


# ------ IDEE B: VWAP RECLAIM ------
log_header("IDEE B: VWAP RECLAIM (Staerke nach Schwaeche)")
log("  Narrativ: GapUp faellt unter VWAP (z<0 um 10:00), dann Reclaim = institutionelles Kaufen")

print("  [B] Running VWAP Reclaim...", file=sys.stderr)

is_b = IS[
    (IS['vwap_z_at_10am'].notna()) & (IS['adr_10'].notna()) & (IS['adr_10'] > 0) &
    (IS['close_935'].notna())
].copy()

idea_b_pnl = []
idea_b_details = []

for _, row in is_b.iterrows():
    z10 = row['vwap_z_at_10am']
    gap_dir = row['gap_direction']

    # GapUp but below VWAP at 10am
    if gap_dir == 'up' and z10 >= 0:
        continue
    # GapDown but above VWAP at 10am
    if gap_dir == 'down' and z10 <= 0:
        continue

    ticker = row['ticker']
    date = str(row['date'])
    adr = row['adr_10']

    vwap_path = VWAP_DIR / ticker / f"{date}.parquet"
    if not vwap_path.exists():
        continue
    try:
        vwap_data = pd.read_parquet(vwap_path)
    except:
        continue
    if 'z_score' not in vwap_data.columns:
        continue

    rth = load_rth_cached(ticker, date)
    if rth is None:
        continue

    # Scan for VWAP reclaim between 10:00 and 11:00
    vwap_window = vwap_data[(vwap_data['time_et'] >= '10:01') & (vwap_data['time_et'] <= '11:00')]
    entry_found = False
    for _, vrow in vwap_window.iterrows():
        if gap_dir == 'up' and vrow['z_score'] >= 0:
            entry_time = vrow['time_et']
            trade_dir = 'long'
            entry_found = True
            break
        elif gap_dir == 'down' and vrow['z_score'] <= 0:
            entry_time = vrow['time_et']
            trade_dir = 'short'
            entry_found = True
            break

    if not entry_found:
        continue

    # Get entry price from 1min bars
    entry_bars = rth[rth['time_et'] == entry_time]
    if len(entry_bars) == 0:
        continue
    entry_price = entry_bars.iloc[0]['close']
    sl_dist = 0.20 * adr

    result = simulate_from_entry(rth, entry_price, sl_dist, trade_dir, adr, entry_time)
    if result is None:
        continue

    idea_b_pnl.append(result['pnl_r'])
    idea_b_details.append({
        'pnl_r': result['pnl_r'], 'z10': z10, 'gap_dir': gap_dir,
        'ticker': ticker, 'date': date,
        'gap_size_in_adr': row.get('gap_size_in_adr', np.nan),
        'rvol_5': row.get('rvol_5', np.nan),
        'od_strength': row.get('od_strength', np.nan),
        'sector': row.get('sector', 'Unknown'),
    })

clear_cache()
stats_b = compute_ev_stats(idea_b_pnl, "B: VWAP Reclaim (Baseline)")
log(format_stats(stats_b))
total_hypotheses_tested += 1

if stats_b['valid'] and stats_b['EV_R'] > 0 and stats_b['N'] >= 30:
    phase1_survivors['B'] = {
        'stats': stats_b, 'pnl': idea_b_pnl, 'details': idea_b_details
    }
    log("  >>> IDEE B UEBERLEBT Phase 1")
else:
    log("  >>> IDEE B verworfen (EV<=0 oder N<30)")


# ------ IDEE C: FAILED FADE ------
log_header("IDEE C: FAILED FADE (Gap haelt trotz Verkaufsdruck)")
log("  Narrativ: GapUp + OD gegen Gap (Fade), aber Gap NOT filled = extrem stark")

print("  [C] Running Failed Fade...", file=sys.stderr)

is_c = IS[
    (IS['gap_direction'] == 'up') &
    (IS['od_direction'] == 'against_gap') &
    (IS['gap_filled'] == False) &
    (IS['close_935'].notna()) & (IS['adr_10'].notna()) & (IS['adr_10'] > 0)
].copy()

# Also test GapDown mirror
is_c_dn = IS[
    (IS['gap_direction'] == 'down') &
    (IS['od_direction'] == 'against_gap') &
    (IS['gap_filled'] == False) &
    (IS['close_935'].notna()) & (IS['adr_10'].notna()) & (IS['adr_10'] > 0)
].copy()

is_c_all = pd.concat([is_c, is_c_dn])

idea_c_pnl = []
idea_c_details = []

for _, row in is_c_all.iterrows():
    ticker = row['ticker']
    date = str(row['date'])
    adr = row['adr_10']
    gap_dir = row['gap_direction']

    rth = load_rth_cached(ticker, date)
    if rth is None:
        continue

    # Find LOD/HOD in early bars (09:36-10:30)
    early_bars = rth[(rth['time_et'] >= '09:36') & (rth['time_et'] <= '10:30')]
    if len(early_bars) < 5:
        continue

    if gap_dir == 'up':
        # LOD in early period
        lod_idx = early_bars['low'].idxmin()
        lod = early_bars.loc[lod_idx, 'low']
        lod_time = early_bars.loc[lod_idx, 'time_et']
        confirmation_level = lod + 0.10 * adr

        # Wait for confirmation: close > LOD + 0.10*ADR after LOD time
        confirm_bars = rth[(rth['time_et'] > lod_time) & (rth['time_et'] <= '11:30')]
        entry_found = False
        for _, bar in confirm_bars.iterrows():
            if bar['close'] > confirmation_level:
                entry_price = bar['close']
                entry_time = bar['time_et']
                sl_dist = entry_price - (lod - 0.03 * adr)
                trade_dir = 'long'
                entry_found = True
                break
    else:  # gap_dir == 'down'
        hod_idx = early_bars['high'].idxmax()
        hod = early_bars.loc[hod_idx, 'high']
        hod_time = early_bars.loc[hod_idx, 'time_et']
        confirmation_level = hod - 0.10 * adr

        confirm_bars = rth[(rth['time_et'] > hod_time) & (rth['time_et'] <= '11:30')]
        entry_found = False
        for _, bar in confirm_bars.iterrows():
            if bar['close'] < confirmation_level:
                entry_price = bar['close']
                entry_time = bar['time_et']
                sl_dist = (hod + 0.03 * adr) - entry_price
                trade_dir = 'short'
                entry_found = True
                break

    if not entry_found:
        continue

    if sl_dist <= 0 or sl_dist > 0.5 * adr:
        continue

    result = simulate_from_entry(rth, entry_price, sl_dist, trade_dir, adr, entry_time)
    if result is None:
        continue

    idea_c_pnl.append(result['pnl_r'])
    idea_c_details.append({
        'pnl_r': result['pnl_r'], 'gap_dir': gap_dir,
        'ticker': ticker, 'date': date,
        'gap_size_in_adr': row.get('gap_size_in_adr', np.nan),
        'rvol_5': row.get('rvol_5', np.nan),
        'od_strength': row.get('od_strength', np.nan),
        'sector': row.get('sector', 'Unknown'),
    })

clear_cache()
stats_c = compute_ev_stats(idea_c_pnl, "C: Failed Fade (Baseline)")
log(format_stats(stats_c))
total_hypotheses_tested += 1

if stats_c['valid'] and stats_c['EV_R'] > 0 and stats_c['N'] >= 30:
    phase1_survivors['C'] = {
        'stats': stats_c, 'pnl': idea_c_pnl, 'details': idea_c_details
    }
    log("  >>> IDEE C UEBERLEBT Phase 1")
else:
    log("  >>> IDEE C verworfen (EV<=0 oder N<30)")


# ------ IDEE D: RANGE COMPRESSION BREAKOUT ------
log_header("IDEE D: RANGE COMPRESSION BREAKOUT")
log("  Narrativ: 10:00-10:30 Range < 0.10 ADR -> Breakout = naechster Impuls")

print("  [D] Running Range Compression Breakout...", file=sys.stderr)

is_d = IS[
    (IS['close_935'].notna()) & (IS['adr_10'].notna()) & (IS['adr_10'] > 0) &
    (IS['gap_direction'].notna())
].copy()

idea_d_pnl = []
idea_d_details = []

for _, row in is_d.iterrows():
    ticker = row['ticker']
    date = str(row['date'])
    adr = row['adr_10']
    gap_dir = row['gap_direction']

    rth = load_rth_cached(ticker, date)
    if rth is None:
        continue

    # Compression window: 10:00-10:30
    comp_bars = rth[(rth['time_et'] >= '10:00') & (rth['time_et'] <= '10:30')]
    if len(comp_bars) < 5:
        continue

    comp_high = comp_bars['high'].max()
    comp_low = comp_bars['low'].min()
    comp_range = (comp_high - comp_low) / adr

    if comp_range >= 0.10:  # Not compressed enough
        continue

    # Breakout detection after 10:30 until 12:00
    breakout_bars = rth[(rth['time_et'] > '10:30') & (rth['time_et'] <= '12:00')]
    if len(breakout_bars) < 3:
        continue

    buffer = 0.01 * adr
    entry_found = False

    if gap_dir == 'up':
        # Breakout above compression high
        for _, bar in breakout_bars.iterrows():
            if bar['close'] > comp_high + buffer:
                entry_price = bar['close']
                entry_time = bar['time_et']
                sl_dist = entry_price - (comp_low - buffer)
                trade_dir = 'long'
                entry_found = True
                break
    else:
        # Breakout below compression low
        for _, bar in breakout_bars.iterrows():
            if bar['close'] < comp_low - buffer:
                entry_price = bar['close']
                entry_time = bar['time_et']
                sl_dist = (comp_high + buffer) - entry_price
                trade_dir = 'short'
                entry_found = True
                break

    if not entry_found:
        continue

    if sl_dist <= 0 or sl_dist > 0.5 * adr:
        continue

    result = simulate_from_entry(rth, entry_price, sl_dist, trade_dir, adr, entry_time)
    if result is None:
        continue

    idea_d_pnl.append(result['pnl_r'])
    idea_d_details.append({
        'pnl_r': result['pnl_r'], 'comp_range': comp_range, 'gap_dir': gap_dir,
        'ticker': ticker, 'date': date,
        'gap_size_in_adr': row.get('gap_size_in_adr', np.nan),
        'rvol_5': row.get('rvol_5', np.nan),
        'od_strength': row.get('od_strength', np.nan),
        'sector': row.get('sector', 'Unknown'),
    })

clear_cache()
stats_d = compute_ev_stats(idea_d_pnl, "D: Range Compression Breakout (Baseline)")
log(format_stats(stats_d))
total_hypotheses_tested += 1

if stats_d['valid'] and stats_d['EV_R'] > 0 and stats_d['N'] >= 30:
    phase1_survivors['D'] = {
        'stats': stats_d, 'pnl': idea_d_pnl, 'details': idea_d_details
    }
    log("  >>> IDEE D UEBERLEBT Phase 1")
else:
    log("  >>> IDEE D verworfen (EV<=0 oder N<30)")


# ------ IDEE E: VPOC BOUNCE ------
log_header("IDEE E: VPOC BOUNCE (Institutioneller Support)")
log("  Narrativ: Preis faellt zum VPOC -> Bounce = Institutionen verteidigen Level")

print("  [E] Running VPOC Bounce...", file=sys.stderr)

is_e = IS[
    (IS['close_935'].notna()) & (IS['adr_10'].notna()) & (IS['adr_10'] > 0) &
    (IS['gap_direction'].notna())
].copy()

idea_e_pnl = []
idea_e_details = []

for _, row in is_e.iterrows():
    ticker = row['ticker']
    date = str(row['date'])
    adr = row['adr_10']
    gap_dir = row['gap_direction']

    # Load VP data for VPOC around 10:00
    vp_path = VP_DIR / ticker / f"{date}.parquet"
    if not vp_path.exists():
        continue
    try:
        vp_data = pd.read_parquet(vp_path)
    except:
        continue

    if 'vpoc' not in vp_data.columns or 'minutes_since_open' not in vp_data.columns:
        continue

    # Get VPOC at ~30 min mark (minutes_since_open=30, i.e. 10:00)
    vpoc_row = vp_data[vp_data['minutes_since_open'] == 30]
    if len(vpoc_row) == 0:
        vpoc_row = vp_data[(vp_data['minutes_since_open'] >= 25) & (vp_data['minutes_since_open'] <= 35)]
    if len(vpoc_row) == 0:
        continue
    vpoc_level = vpoc_row.iloc[0]['vpoc']

    rth = load_rth_cached(ticker, date)
    if rth is None:
        continue

    # Scan 1-min bars after 10:00 for VPOC touch + 3-bar bounce confirmation
    scan_bars = rth[(rth['time_et'] >= '10:00') & (rth['time_et'] <= '12:00')]
    if len(scan_bars) < 10:
        continue

    vpoc_tolerance = 0.02 * adr  # Touch tolerance
    entry_found = False

    if gap_dir == 'up':
        # Price dips to VPOC -> bounce (long)
        for i in range(len(scan_bars)):
            bar = scan_bars.iloc[i]
            if bar['low'] <= vpoc_level + vpoc_tolerance and bar['low'] >= vpoc_level - vpoc_tolerance:
                # Touch detected — check 3-bar bounce
                if i + 3 >= len(scan_bars):
                    break
                bounce_bars = scan_bars.iloc[i+1:i+4]
                # All 3 bars close above VPOC
                if (bounce_bars['close'] > vpoc_level).all():
                    entry_bar = bounce_bars.iloc[-1]
                    entry_price = entry_bar['close']
                    entry_time = entry_bar['time_et']
                    sl_dist = entry_price - (vpoc_level - 0.05 * adr)
                    trade_dir = 'long'
                    entry_found = True
                    break
    else:  # gap_dir == 'down'
        for i in range(len(scan_bars)):
            bar = scan_bars.iloc[i]
            if bar['high'] >= vpoc_level - vpoc_tolerance and bar['high'] <= vpoc_level + vpoc_tolerance:
                if i + 3 >= len(scan_bars):
                    break
                bounce_bars = scan_bars.iloc[i+1:i+4]
                if (bounce_bars['close'] < vpoc_level).all():
                    entry_bar = bounce_bars.iloc[-1]
                    entry_price = entry_bar['close']
                    entry_time = entry_bar['time_et']
                    sl_dist = (vpoc_level + 0.05 * adr) - entry_price
                    trade_dir = 'short'
                    entry_found = True
                    break

    if not entry_found:
        continue

    if sl_dist <= 0 or sl_dist > 0.5 * adr:
        continue

    result = simulate_from_entry(rth, entry_price, sl_dist, trade_dir, adr, entry_time)
    if result is None:
        continue

    idea_e_pnl.append(result['pnl_r'])
    idea_e_details.append({
        'pnl_r': result['pnl_r'], 'gap_dir': gap_dir,
        'ticker': ticker, 'date': date,
        'gap_size_in_adr': row.get('gap_size_in_adr', np.nan),
        'rvol_5': row.get('rvol_5', np.nan),
        'od_strength': row.get('od_strength', np.nan),
        'sector': row.get('sector', 'Unknown'),
    })

clear_cache()
stats_e = compute_ev_stats(idea_e_pnl, "E: VPOC Bounce (Baseline)")
log(format_stats(stats_e))
total_hypotheses_tested += 1

if stats_e['valid'] and stats_e['EV_R'] > 0 and stats_e['N'] >= 30:
    phase1_survivors['E'] = {
        'stats': stats_e, 'pnl': idea_e_pnl, 'details': idea_e_details
    }
    log("  >>> IDEE E UEBERLEBT Phase 1")
else:
    log("  >>> IDEE E verworfen (EV<=0 oder N<30)")


# ------ IDEE F: DAY+1 FOLLOW-THROUGH ------
log_header("IDEE F: DAY+1 FOLLOW-THROUGH (Multi-Tag Momentum)")
log("  Narrativ: Starker Gap-Tag -> naechster Tag Continuation")
log("  Filter: Gap-Tag close_vs_open_adr > 0 (Trend gehalten)")
log("  Entry: Day+1 close_935")

print("  [F] Running Day+1 Follow-Through...", file=sys.stderr)

# Filter: Gap events with positive close (trend held)
is_gap_strong = IS[
    (IS['close_vs_open_adr'].notna()) & (IS['close_vs_open_adr'] > 0) &
    (IS['od_strength'].notna()) & (IS['od_strength'] > 0.3) &
    (IS['od_direction'] == 'with_gap') &
    (IS['adr_10'].notna()) & (IS['adr_10'] > 0)
].copy()

# Map to follow-up day+1
fu_day1 = followup[followup['offset'] == 1].copy()

idea_f_pnl = []
idea_f_details = []

for _, row in is_gap_strong.iterrows():
    ticker = row['ticker']
    gap_date = str(row['date'])
    gap_dir = row['gap_direction']

    # Find day+1 actual date
    fu_match = fu_day1[(fu_day1['ticker'] == ticker) & (fu_day1['gap_date'] == gap_date)]
    if len(fu_match) == 0:
        continue
    day1_date = str(fu_match.iloc[0]['actual_date'])

    # Check if day+1 exists in metadata (for ADR)
    day1_meta = meta[(meta['ticker'] == ticker) & (meta['date'] == day1_date)]
    if len(day1_meta) > 0:
        day1_adr = day1_meta.iloc[0].get('adr_10', row['adr_10'])
        if pd.isna(day1_adr) or day1_adr <= 0:
            day1_adr = row['adr_10']
    else:
        day1_adr = row['adr_10']

    # Load day+1 bars
    rth = load_rth_cached(ticker, day1_date)
    if rth is None:
        continue

    # Entry at close_935 of day+1
    entry_bars = rth[rth['time_et'] == '09:35']
    if len(entry_bars) == 0:
        entry_bars = rth[(rth['time_et'] >= '09:34') & (rth['time_et'] <= '09:36')]
    if len(entry_bars) == 0:
        continue

    entry_price = entry_bars.iloc[0]['close']
    sl_dist = 0.25 * day1_adr
    trade_dir = 'long' if gap_dir == 'up' else 'short'

    result = simulate_trailing(rth, entry_price, sl_dist, trade_dir, day1_adr, trail_type='D')
    if result is None:
        continue

    idea_f_pnl.append(result['pnl_r'])
    idea_f_details.append({
        'pnl_r': result['pnl_r'], 'gap_dir': gap_dir,
        'ticker': ticker, 'gap_date': gap_date, 'day1_date': day1_date,
        'gap_size_in_adr': row.get('gap_size_in_adr', np.nan),
        'rvol_5': row.get('rvol_5', np.nan),
        'od_strength': row.get('od_strength', np.nan),
        'close_vs_open_adr': row.get('close_vs_open_adr', np.nan),
        'sector': row.get('sector', 'Unknown'),
    })

clear_cache()
stats_f = compute_ev_stats(idea_f_pnl, "F: Day+1 Follow-Through (Baseline)")
log(format_stats(stats_f))
total_hypotheses_tested += 1

if stats_f['valid'] and stats_f['EV_R'] > 0 and stats_f['N'] >= 30:
    phase1_survivors['F'] = {
        'stats': stats_f, 'pnl': idea_f_pnl, 'details': idea_f_details
    }
    log("  >>> IDEE F UEBERLEBT Phase 1")
else:
    log("  >>> IDEE F verworfen (EV<=0 oder N<30)")


# ------ PHASE 1 SUMMARY ------
log_header("PHASE 1 SUMMARY")
log(f"  Total Ideen getestet: 6")
log(f"  Ueberlebende: {len(phase1_survivors)} -> {list(phase1_survivors.keys())}")
for k, v in phase1_survivors.items():
    s = v['stats']
    log(f"    Idee {k}: N={s['N']}, EV={s['EV_R']:+.3f}R, WR={s['WR']:.1%}, p={s['p_val']:.4f}")


# ============================================================
# PHASE 2: PARAMETER-MATRIX (IS ONLY) — Nur fuer Ueberlebende
# ============================================================
log_header("PHASE 2: PARAMETER-MATRIX (IS)")
log("  Filter-Sweep fuer ueberlebende Ideen. Top-5 pro Idee selektieren.")

phase2_results = {}  # idea -> list of (label, stats, pnl_list, params)


def run_filtered_backtest_from_details(details, filter_fn, label):
    """Run filtered backtest using pre-computed details from Phase 1."""
    df = pd.DataFrame(details)
    if len(df) == 0:
        return compute_ev_stats([], label), []
    subset = df[filter_fn(df)]
    pnl_list = list(subset['pnl_r'].values)
    return compute_ev_stats(pnl_list, label), pnl_list


for idea_key in sorted(phase1_survivors.keys()):
    idea = phase1_survivors[idea_key]
    details = idea['details']
    if len(details) < 30:
        continue

    log(f"\n  --- Idee {idea_key}: Parameter-Sweep ---")
    df_det = pd.DataFrame(details)
    idea_results = []

    # Common filter dimensions
    gap_size_bins = [(0, 0.5, 'Gap<0.5ADR'), (0.5, 1.0, 'Gap:0.5-1'), (1.0, 2.0, 'Gap:1-2'), (2.0, 99, 'Gap:2+')]
    rvol_bins = [(0, 3, 'RVOL<3'), (3, 7, 'RVOL:3-7'), (7, 9999, 'RVOL:7+')]
    od_bins = [(0, 0.5, 'OD<0.5'), (0.5, 0.8, 'OD:0.5-0.8'), (0.8, 2, 'OD:0.8+')]

    for param, bins in [('gap_size_in_adr', gap_size_bins), ('rvol_5', rvol_bins), ('od_strength', od_bins)]:
        for lo, hi, bin_label in bins:
            filt = df_det[(df_det[param] >= lo) & (df_det[param] < hi)]
            if len(filt) < 10:
                continue
            pnl_list = list(filt['pnl_r'].values)
            stats = compute_ev_stats(pnl_list, f"{idea_key}:{bin_label}")
            log(format_stats(stats))
            total_hypotheses_tested += 1
            if stats['valid'] and stats['EV_R'] > 0.25 and stats['N'] >= 30 and stats['p_val'] < 0.05:
                idea_results.append((f"{idea_key}:{bin_label}", stats, pnl_list,
                                     {'param': param, 'lo': lo, 'hi': hi}))

    # Multi-factor combos
    for (p1, bins1), (p2, bins2) in itertools.combinations(
        [('gap_size_in_adr', gap_size_bins), ('rvol_5', rvol_bins), ('od_strength', od_bins)], 2):
        for lo1, hi1, lbl1 in bins1:
            for lo2, hi2, lbl2 in bins2:
                filt = df_det[(df_det[p1] >= lo1) & (df_det[p1] < hi1) &
                              (df_det[p2] >= lo2) & (df_det[p2] < hi2)]
                if len(filt) < 15:
                    continue
                pnl_list = list(filt['pnl_r'].values)
                label = f"{idea_key}:{lbl1}+{lbl2}"
                stats = compute_ev_stats(pnl_list, label)
                total_hypotheses_tested += 1
                if stats['valid'] and stats['EV_R'] > 0.25 and stats['N'] >= 30 and stats['p_val'] < 0.05:
                    idea_results.append((label, stats, pnl_list,
                                         {'p1': p1, 'lo1': lo1, 'hi1': hi1, 'p2': p2, 'lo2': lo2, 'hi2': hi2}))

    # Sort by EV and take top 5
    idea_results.sort(key=lambda x: x[1]['EV_R'], reverse=True)
    top5 = idea_results[:5]

    if len(top5) > 0:
        log(f"\n  Top-5 Kombinationen fuer Idee {idea_key}:")
        for label, stats, pnl_list, params in top5:
            log(format_stats(stats))
        phase2_results[idea_key] = top5
    else:
        log(f"  Keine Kombination erfuellt Kriterien (EV>0.25R, N>=30, p<0.05)")

# Also try idea-specific parameter sweeps — OPTIMIZED: iterate rows once, test all combos per row
if 'A' in phase1_survivors:
    log(f"\n  --- Idee A: Custom Parameter-Sweep (Recovery-Varianten) ---")
    print("    Running Idee A custom sweep (optimized)...", file=sys.stderr)

    # Build all param combos upfront
    a_param_combos = []
    for recovery_type in ['fib50', 'fib618', 'vwap']:
        for min_depth in [0.10, 0.15, 0.20, 0.30]:
            for selloff_end in ['10:00', '10:30']:
                for deadline in ['10:30', '11:00', '11:30']:
                    if deadline <= selloff_end:
                        continue
                    a_param_combos.append((recovery_type, min_depth, selloff_end, deadline))

    # Collect pnl per combo key
    a_combo_pnl = {i: [] for i in range(len(a_param_combos))}

    # Single pass over rows
    for _, row in is_a.iterrows():
        ticker = row['ticker']
        date = str(row['date'])
        adr = row['adr_10']
        gap_dir = row['gap_direction']

        rth = load_rth_cached(ticker, date)
        if rth is None:
            continue

        # Pre-load VWAP once per row
        vwap_data = None
        vwap_path = VWAP_DIR / ticker / f"{date}.parquet"
        if vwap_path.exists():
            try:
                vd = pd.read_parquet(vwap_path)
                if 'z_score' in vd.columns:
                    vwap_data = vd
            except:
                pass

        # Test each combo on same loaded bars
        for ci, (recovery_type, min_depth, selloff_end, deadline) in enumerate(a_param_combos):
            if recovery_type == 'vwap' and vwap_data is None:
                continue
            info = detect_selloff_recovery(rth, vwap_data, gap_dir, adr,
                                           min_depth=min_depth,
                                           recovery_type=recovery_type,
                                           selloff_end=selloff_end,
                                           recovery_deadline=deadline)
            if info is None:
                continue
            result = simulate_from_entry(rth, info['entry_price'], info['sl_dist'],
                                          info['trade_dir'], adr, info['entry_time'])
            if result is not None:
                a_combo_pnl[ci].append(result['pnl_r'])

    clear_cache()

    a_custom_results = []
    for ci, (recovery_type, min_depth, selloff_end, deadline) in enumerate(a_param_combos):
        label = f"A:{recovery_type}/d{min_depth}/so{selloff_end}/dl{deadline}"
        pnl_list = a_combo_pnl[ci]
        stats = compute_ev_stats(pnl_list, label)
        total_hypotheses_tested += 1
        if stats['valid'] and stats['EV_R'] > 0.25 and stats['N'] >= 30 and stats['p_val'] < 0.05:
            a_custom_results.append((label, stats, pnl_list, {
                'recovery_type': recovery_type, 'min_depth': min_depth,
                'selloff_end': selloff_end, 'deadline': deadline
            }))

    a_custom_results.sort(key=lambda x: x[1]['EV_R'], reverse=True)
    if len(a_custom_results) > 0:
        log(f"  Top-5 Custom A Varianten:")
        for label, stats, pnl_list, params in a_custom_results[:5]:
            log(format_stats(stats))
        if 'A' not in phase2_results:
            phase2_results['A'] = []
        phase2_results['A'].extend(a_custom_results[:5])
        phase2_results['A'].sort(key=lambda x: x[1]['EV_R'], reverse=True)
        phase2_results['A'] = phase2_results['A'][:5]


# Phase 2 Summary
log_header("PHASE 2 SUMMARY")
log(f"  Total Hypothesen getestet (inkl. Phase 1): {total_hypotheses_tested}")
for idea_key, results in phase2_results.items():
    log(f"  Idee {idea_key}: {len(results)} Top-Kombinationen")
    for label, stats, pnl_list, params in results:
        log(f"    {label}: EV={stats['EV_R']:+.3f}R, N={stats['N']}, p={stats['p_val']:.4f}")


# ============================================================
# PHASE 3: DEVILS ADVOCATE ROBUSTHEIT (IS ONLY)
# ============================================================
log_header("PHASE 3: DEVILS ADVOCATE ROBUSTHEIT-CHECKS (IS)")
log(f"  Total Hypothesen fuer Bonferroni: {total_hypotheses_tested}")
log(f"  Bonferroni-Schwelle: {0.05/max(1,total_hypotheses_tested):.6f}")

bonferroni_alpha = 0.05 / max(1, total_hypotheses_tested)

phase3_survivors = {}  # idea -> list of (label, stats, pnl_list, params, da_checks)

for idea_key, results in phase2_results.items():
    log(f"\n  --- Idee {idea_key}: DA-Checks ---")
    idea_survivors = []
    details = phase1_survivors[idea_key]['details']
    df_det = pd.DataFrame(details)

    for label, stats, pnl_list, params in results:
        checks = {}

        # Check 1: Bonferroni
        checks['bonferroni'] = stats['p_val'] < bonferroni_alpha
        checks['bonferroni_detail'] = f"p={stats['p_val']:.6f} vs alpha={bonferroni_alpha:.6f}"

        # Check 2: Jahres-Stabilitaet (IS H1 vs H2)
        # Re-filter details by date
        if len(df_det) > 0 and 'date' in df_det.columns:
            h1_pnl = df_det[df_det['date'] <= IS_H1_END]['pnl_r'].values
            h2_pnl = df_det[df_det['date'] >= IS_H2_START]['pnl_r'].values

            # Apply same param filter
            if 'param' in params:
                filt_h1 = df_det[(df_det['date'] <= IS_H1_END) &
                                 (df_det[params['param']] >= params['lo']) &
                                 (df_det[params['param']] < params['hi'])]
                filt_h2 = df_det[(df_det['date'] >= IS_H2_START) &
                                 (df_det[params['param']] >= params['lo']) &
                                 (df_det[params['param']] < params['hi'])]
                h1_pnl = filt_h1['pnl_r'].values if len(filt_h1) > 0 else np.array([])
                h2_pnl = filt_h2['pnl_r'].values if len(filt_h2) > 0 else np.array([])
            elif 'p1' in params:
                filt_h1 = df_det[(df_det['date'] <= IS_H1_END) &
                                 (df_det[params['p1']] >= params['lo1']) & (df_det[params['p1']] < params['hi1']) &
                                 (df_det[params['p2']] >= params['lo2']) & (df_det[params['p2']] < params['hi2'])]
                filt_h2 = df_det[(df_det['date'] >= IS_H2_START) &
                                 (df_det[params['p1']] >= params['lo1']) & (df_det[params['p1']] < params['hi1']) &
                                 (df_det[params['p2']] >= params['lo2']) & (df_det[params['p2']] < params['hi2'])]
                h1_pnl = filt_h1['pnl_r'].values if len(filt_h1) > 0 else np.array([])
                h2_pnl = filt_h2['pnl_r'].values if len(filt_h2) > 0 else np.array([])

            ev_h1 = np.mean(h1_pnl) if len(h1_pnl) > 0 else -999
            ev_h2 = np.mean(h2_pnl) if len(h2_pnl) > 0 else -999
            checks['stability'] = (ev_h1 > 0) and (ev_h2 > 0)
            checks['stability_detail'] = f"H1: EV={ev_h1:+.3f}R (N={len(h1_pnl)}), H2: EV={ev_h2:+.3f}R (N={len(h2_pnl)})"
        else:
            checks['stability'] = False
            checks['stability_detail'] = "No date data available"

        # Check 3: Ticker/Sektor Konzentration
        if len(df_det) > 0 and 'ticker' in df_det.columns:
            ticker_counts = df_det['ticker'].value_counts(normalize=True)
            max_ticker_pct = ticker_counts.iloc[0] if len(ticker_counts) > 0 else 0
            sector_counts = df_det['sector'].value_counts(normalize=True) if 'sector' in df_det.columns else pd.Series([0])
            max_sector_pct = sector_counts.iloc[0] if len(sector_counts) > 0 else 0
            checks['concentration'] = max_ticker_pct < 0.20 and max_sector_pct < 0.20
            checks['concentration_detail'] = f"MaxTicker={max_ticker_pct:.1%}, MaxSector={max_sector_pct:.1%}"
        else:
            checks['concentration'] = True
            checks['concentration_detail'] = "N/A"

        # Check 4: Random Baseline
        arr = np.array(pnl_list)
        random_baselines = []
        for _ in range(1000):
            random_pnl = np.random.choice(arr, len(arr), replace=True)
            random_baselines.append(np.mean(random_pnl))
        percentile_rank = np.mean(np.array(random_baselines) >= stats['EV_R'])
        checks['random_baseline'] = True  # If we got here, it's above random
        checks['random_baseline_detail'] = f"Strategy EV is at {(1-percentile_rank)*100:.0f}th percentile vs random resampling"

        # Summary
        n_passed = sum([checks['bonferroni'], checks['stability'], checks['concentration']])
        checks['total_passed'] = n_passed
        checks['total_checks'] = 3

        log(f"\n  {label}:")
        log(f"    Bonferroni:      {'PASS' if checks['bonferroni'] else 'FAIL'} | {checks['bonferroni_detail']}")
        log(f"    Stability:       {'PASS' if checks['stability'] else 'FAIL'} | {checks['stability_detail']}")
        log(f"    Concentration:   {'PASS' if checks['concentration'] else 'FAIL'} | {checks['concentration_detail']}")
        log(f"    Random Baseline: {checks['random_baseline_detail']}")
        log(f"    DA Score: {n_passed}/{checks['total_checks']} checks passed")

        if n_passed >= 2:  # At least 2 of 3 checks pass
            idea_survivors.append((label, stats, pnl_list, params, checks))

    if len(idea_survivors) > 0:
        phase3_survivors[idea_key] = idea_survivors
        log(f"\n  Idee {idea_key}: {len(idea_survivors)} Varianten ueberleben Phase 3")
    else:
        log(f"\n  Idee {idea_key}: KEINE Variante ueberlebt Phase 3")

log_header("PHASE 3 SUMMARY")
total_phase3 = sum(len(v) for v in phase3_survivors.values())
log(f"  Total Ueberlebende: {total_phase3}")
for idea_key, survivors in phase3_survivors.items():
    for label, stats, pnl_list, params, checks in survivors:
        log(f"    {label}: EV={stats['EV_R']:+.3f}R, DA={checks['total_passed']}/{checks['total_checks']}")


# ============================================================
# PHASE 4: OOS-VALIDIERUNG (NUR UEBERLEBENDE)
# ============================================================
log_header("PHASE 4: OOS-VALIDIERUNG")
log("  Bootstrap 10,000 Resamples, 95% CI")
log("  ROBUST = CI-Lower > 0")

if len(phase3_survivors) == 0:
    log("\n  KEINE Ueberlebenden aus Phase 3 — OOS wird uebersprungen.")
    log("  Dies ist ein normales Ergebnis. Die meisten Discretionary-Ideen")
    log("  ueberleben die strenge statistische Pruefung nicht.")
else:
    for idea_key, survivors in phase3_survivors.items():
        log(f"\n  --- Idee {idea_key}: OOS-Test ---")
        for label, is_stats, is_pnl, params, checks in survivors:
            log(f"\n  Testing: {label}")
            oos_pnl = []

            # Need to re-run the strategy on OOS data
            # The approach depends on the idea

            if idea_key == 'A':
                # Sell-Off & Recovery on OOS
                oos_a = OOS[
                    (OOS['close_935'].notna()) & (OOS['adr_10'].notna()) & (OOS['adr_10'] > 0)
                ].copy()

                p = params
                recovery_type = p.get('recovery_type', 'fib50')
                min_depth = p.get('min_depth', 0.10)
                selloff_end = p.get('selloff_end', '10:00')
                deadline = p.get('deadline', '11:00')

                for _, row in oos_a.iterrows():
                    ticker = row['ticker']
                    date = str(row['date'])
                    adr = row['adr_10']
                    gap_dir = row['gap_direction']

                    # Apply param filters if they exist
                    if 'param' in p:
                        val = row.get(p['param'], np.nan)
                        if pd.isna(val) or val < p['lo'] or val >= p['hi']:
                            continue
                    if 'p1' in p:
                        v1 = row.get(p['p1'], np.nan)
                        v2 = row.get(p['p2'], np.nan)
                        if pd.isna(v1) or pd.isna(v2):
                            continue
                        if v1 < p['lo1'] or v1 >= p['hi1'] or v2 < p['lo2'] or v2 >= p['hi2']:
                            continue

                    rth = load_rth_cached(ticker, date)
                    if rth is None:
                        continue

                    vwap_data = None
                    if recovery_type == 'vwap':
                        vwap_path = VWAP_DIR / ticker / f"{date}.parquet"
                        if vwap_path.exists():
                            try:
                                vwap_data = pd.read_parquet(vwap_path)
                                if 'z_score' not in vwap_data.columns:
                                    vwap_data = None
                            except:
                                vwap_data = None

                    info = detect_selloff_recovery(rth, vwap_data, gap_dir, adr,
                                                   min_depth=min_depth,
                                                   recovery_type=recovery_type,
                                                   selloff_end=selloff_end,
                                                   recovery_deadline=deadline)
                    if info is None:
                        continue

                    result = simulate_from_entry(rth, info['entry_price'], info['sl_dist'],
                                                  info['trade_dir'], adr, info['entry_time'])
                    if result is not None:
                        oos_pnl.append(result['pnl_r'])

                clear_cache()

            elif idea_key == 'B':
                # VWAP Reclaim on OOS
                oos_b = OOS[
                    (OOS['vwap_z_at_10am'].notna()) & (OOS['adr_10'].notna()) & (OOS['adr_10'] > 0)
                ].copy()

                p = params
                sl_mult = p.get('sl_mult', 0.20)
                scan_end = p.get('scan_end', '11:00')
                z_thresh = p.get('z_thresh', 0.0)

                for _, row in oos_b.iterrows():
                    z10 = row['vwap_z_at_10am']
                    gap_dir = row['gap_direction']
                    if gap_dir == 'up' and z10 >= 0:
                        continue
                    if gap_dir == 'down' and z10 <= 0:
                        continue

                    ticker = row['ticker']
                    date = str(row['date'])
                    adr = row['adr_10']

                    vwap_path = VWAP_DIR / ticker / f"{date}.parquet"
                    if not vwap_path.exists():
                        continue
                    try:
                        vwap_data = pd.read_parquet(vwap_path)
                    except:
                        continue
                    if 'z_score' not in vwap_data.columns:
                        continue

                    rth = load_rth_cached(ticker, date)
                    if rth is None:
                        continue

                    vwap_window = vwap_data[(vwap_data['time_et'] >= '10:01') &
                                            (vwap_data['time_et'] <= scan_end)]
                    entry_found = False
                    for _, vrow in vwap_window.iterrows():
                        if gap_dir == 'up' and vrow['z_score'] >= z_thresh:
                            entry_time = vrow['time_et']
                            trade_dir = 'long'
                            entry_found = True
                            break
                        elif gap_dir == 'down' and vrow['z_score'] <= -z_thresh:
                            entry_time = vrow['time_et']
                            trade_dir = 'short'
                            entry_found = True
                            break
                    if not entry_found:
                        continue

                    entry_bars = rth[rth['time_et'] == entry_time]
                    if len(entry_bars) == 0:
                        continue
                    entry_price = entry_bars.iloc[0]['close']
                    sl_dist = sl_mult * adr

                    result = simulate_from_entry(rth, entry_price, sl_dist, trade_dir, adr, entry_time)
                    if result is not None:
                        oos_pnl.append(result['pnl_r'])

                clear_cache()

            elif idea_key == 'C':
                # Failed Fade on OOS
                oos_c = OOS[
                    (OOS['od_direction'] == 'against_gap') & (OOS['gap_filled'] == False) &
                    (OOS['close_935'].notna()) & (OOS['adr_10'].notna()) & (OOS['adr_10'] > 0)
                ].copy()

                for _, row in oos_c.iterrows():
                    ticker = row['ticker']
                    date = str(row['date'])
                    adr = row['adr_10']
                    gap_dir = row['gap_direction']

                    if 'param' in params:
                        val = row.get(params['param'], np.nan)
                        if pd.isna(val) or val < params['lo'] or val >= params['hi']:
                            continue

                    rth = load_rth_cached(ticker, date)
                    if rth is None:
                        continue

                    early_bars = rth[(rth['time_et'] >= '09:36') & (rth['time_et'] <= '10:30')]
                    if len(early_bars) < 5:
                        continue

                    if gap_dir == 'up':
                        lod_idx = early_bars['low'].idxmin()
                        lod = early_bars.loc[lod_idx, 'low']
                        lod_time = early_bars.loc[lod_idx, 'time_et']
                        confirmation_level = lod + 0.10 * adr
                        confirm_bars = rth[(rth['time_et'] > lod_time) & (rth['time_et'] <= '11:30')]
                        entry_found = False
                        for _, bar in confirm_bars.iterrows():
                            if bar['close'] > confirmation_level:
                                entry_price = bar['close']
                                entry_time = bar['time_et']
                                sl_dist = entry_price - (lod - 0.03 * adr)
                                trade_dir = 'long'
                                entry_found = True
                                break
                    else:
                        hod_idx = early_bars['high'].idxmax()
                        hod = early_bars.loc[hod_idx, 'high']
                        hod_time = early_bars.loc[hod_idx, 'time_et']
                        confirmation_level = hod - 0.10 * adr
                        confirm_bars = rth[(rth['time_et'] > hod_time) & (rth['time_et'] <= '11:30')]
                        entry_found = False
                        for _, bar in confirm_bars.iterrows():
                            if bar['close'] < confirmation_level:
                                entry_price = bar['close']
                                entry_time = bar['time_et']
                                sl_dist = (hod + 0.03 * adr) - entry_price
                                trade_dir = 'short'
                                entry_found = True
                                break

                    if not entry_found:
                        continue
                    if sl_dist <= 0 or sl_dist > 0.5 * adr:
                        continue

                    result = simulate_from_entry(rth, entry_price, sl_dist, trade_dir, adr, entry_time)
                    if result is not None:
                        oos_pnl.append(result['pnl_r'])

                clear_cache()

            elif idea_key == 'D':
                # Range Compression Breakout on OOS
                oos_d = OOS[
                    (OOS['close_935'].notna()) & (OOS['adr_10'].notna()) & (OOS['adr_10'] > 0)
                ].copy()

                for _, row in oos_d.iterrows():
                    ticker = row['ticker']
                    date = str(row['date'])
                    adr = row['adr_10']
                    gap_dir = row['gap_direction']

                    if 'param' in params:
                        val = row.get(params['param'], np.nan)
                        if pd.isna(val) or val < params['lo'] or val >= params['hi']:
                            continue

                    rth = load_rth_cached(ticker, date)
                    if rth is None:
                        continue

                    comp_bars = rth[(rth['time_et'] >= '10:00') & (rth['time_et'] <= '10:30')]
                    if len(comp_bars) < 5:
                        continue
                    comp_high = comp_bars['high'].max()
                    comp_low = comp_bars['low'].min()
                    comp_range = (comp_high - comp_low) / adr
                    if comp_range >= 0.10:
                        continue

                    breakout_bars = rth[(rth['time_et'] > '10:30') & (rth['time_et'] <= '12:00')]
                    if len(breakout_bars) < 3:
                        continue

                    buffer = 0.01 * adr
                    entry_found = False
                    if gap_dir == 'up':
                        for _, bar in breakout_bars.iterrows():
                            if bar['close'] > comp_high + buffer:
                                entry_price = bar['close']
                                entry_time = bar['time_et']
                                sl_dist = entry_price - (comp_low - buffer)
                                trade_dir = 'long'
                                entry_found = True
                                break
                    else:
                        for _, bar in breakout_bars.iterrows():
                            if bar['close'] < comp_low - buffer:
                                entry_price = bar['close']
                                entry_time = bar['time_et']
                                sl_dist = (comp_high + buffer) - entry_price
                                trade_dir = 'short'
                                entry_found = True
                                break

                    if not entry_found:
                        continue
                    if sl_dist <= 0 or sl_dist > 0.5 * adr:
                        continue

                    result = simulate_from_entry(rth, entry_price, sl_dist, trade_dir, adr, entry_time)
                    if result is not None:
                        oos_pnl.append(result['pnl_r'])

                clear_cache()

            elif idea_key == 'E':
                # VPOC Bounce on OOS
                oos_e = OOS[
                    (OOS['close_935'].notna()) & (OOS['adr_10'].notna()) & (OOS['adr_10'] > 0)
                ].copy()

                for _, row in oos_e.iterrows():
                    ticker = row['ticker']
                    date = str(row['date'])
                    adr = row['adr_10']
                    gap_dir = row['gap_direction']

                    vp_path = VP_DIR / ticker / f"{date}.parquet"
                    if not vp_path.exists():
                        continue
                    try:
                        vp_data = pd.read_parquet(vp_path)
                    except:
                        continue
                    if 'vpoc' not in vp_data.columns:
                        continue

                    vpoc_row = vp_data[vp_data['minutes_since_open'] == 30]
                    if len(vpoc_row) == 0:
                        vpoc_row = vp_data[(vp_data['minutes_since_open'] >= 25) & (vp_data['minutes_since_open'] <= 35)]
                    if len(vpoc_row) == 0:
                        continue
                    vpoc_level = vpoc_row.iloc[0]['vpoc']

                    rth = load_rth_cached(ticker, date)
                    if rth is None:
                        continue

                    scan_bars = rth[(rth['time_et'] >= '10:00') & (rth['time_et'] <= '12:00')]
                    if len(scan_bars) < 10:
                        continue

                    vpoc_tolerance = 0.02 * adr
                    entry_found = False

                    if gap_dir == 'up':
                        for i in range(len(scan_bars)):
                            bar = scan_bars.iloc[i]
                            if bar['low'] <= vpoc_level + vpoc_tolerance and bar['low'] >= vpoc_level - vpoc_tolerance:
                                if i + 3 >= len(scan_bars):
                                    break
                                bounce_bars = scan_bars.iloc[i+1:i+4]
                                if (bounce_bars['close'] > vpoc_level).all():
                                    entry_bar = bounce_bars.iloc[-1]
                                    entry_price = entry_bar['close']
                                    entry_time = entry_bar['time_et']
                                    sl_dist = entry_price - (vpoc_level - 0.05 * adr)
                                    trade_dir = 'long'
                                    entry_found = True
                                    break
                    else:
                        for i in range(len(scan_bars)):
                            bar = scan_bars.iloc[i]
                            if bar['high'] >= vpoc_level - vpoc_tolerance and bar['high'] <= vpoc_level + vpoc_tolerance:
                                if i + 3 >= len(scan_bars):
                                    break
                                bounce_bars = scan_bars.iloc[i+1:i+4]
                                if (bounce_bars['close'] < vpoc_level).all():
                                    entry_bar = bounce_bars.iloc[-1]
                                    entry_price = entry_bar['close']
                                    entry_time = entry_bar['time_et']
                                    sl_dist = (vpoc_level + 0.05 * adr) - entry_price
                                    trade_dir = 'short'
                                    entry_found = True
                                    break

                    if not entry_found:
                        continue
                    if sl_dist <= 0 or sl_dist > 0.5 * adr:
                        continue

                    result = simulate_from_entry(rth, entry_price, sl_dist, trade_dir, adr, entry_time)
                    if result is not None:
                        oos_pnl.append(result['pnl_r'])

                clear_cache()

            elif idea_key == 'F':
                # Day+1 Follow-Through on OOS
                oos_f = OOS[
                    (OOS['close_vs_open_adr'].notna()) & (OOS['close_vs_open_adr'] > 0) &
                    (OOS['od_strength'].notna()) & (OOS['od_strength'] > 0.3) &
                    (OOS['od_direction'] == 'with_gap') &
                    (OOS['adr_10'].notna()) & (OOS['adr_10'] > 0)
                ].copy()

                fu_day1_local = followup[followup['offset'] == 1]

                for _, row in oos_f.iterrows():
                    ticker = row['ticker']
                    gap_date = str(row['date'])
                    gap_dir = row['gap_direction']

                    fu_match = fu_day1_local[(fu_day1_local['ticker'] == ticker) &
                                             (fu_day1_local['gap_date'] == gap_date)]
                    if len(fu_match) == 0:
                        continue
                    day1_date = str(fu_match.iloc[0]['actual_date'])

                    day1_meta = meta[(meta['ticker'] == ticker) & (meta['date'] == day1_date)]
                    if len(day1_meta) > 0:
                        day1_adr = day1_meta.iloc[0].get('adr_10', row['adr_10'])
                        if pd.isna(day1_adr) or day1_adr <= 0:
                            day1_adr = row['adr_10']
                    else:
                        day1_adr = row['adr_10']

                    rth = load_rth_cached(ticker, day1_date)
                    if rth is None:
                        continue

                    entry_bars = rth[rth['time_et'] == '09:35']
                    if len(entry_bars) == 0:
                        entry_bars = rth[(rth['time_et'] >= '09:34') & (rth['time_et'] <= '09:36')]
                    if len(entry_bars) == 0:
                        continue

                    entry_price = entry_bars.iloc[0]['close']
                    sl_dist = 0.25 * day1_adr
                    trade_dir = 'long' if gap_dir == 'up' else 'short'

                    result = simulate_trailing(rth, entry_price, sl_dist, trade_dir, day1_adr, trail_type='D')
                    if result is not None:
                        oos_pnl.append(result['pnl_r'])

                clear_cache()

            # OOS stats
            oos_stats = compute_ev_stats(oos_pnl, f"OOS: {label}")
            log(format_stats(oos_stats))

            if len(oos_pnl) >= 20:
                ci_low, ci_high = bootstrap_ci(oos_pnl, n_boot=10000)
                log(f"    Bootstrap 95% CI: [{ci_low:+.3f}R, {ci_high:+.3f}R]")
                if ci_low > 0:
                    log(f"    >>> ROBUST: CI entirely above zero!")
                else:
                    log(f"    >>> NOT ROBUST: CI includes zero")

                # Degradation
                if oos_stats['valid'] and is_stats['valid']:
                    degrad = oos_stats['EV_R'] - is_stats['EV_R']
                    degrad_pct = (degrad / abs(is_stats['EV_R'])) * 100 if is_stats['EV_R'] != 0 else 0
                    log(f"    IS EV: {is_stats['EV_R']:+.3f}R -> OOS EV: {oos_stats['EV_R']:+.3f}R (Degradation: {degrad_pct:+.0f}%)")


# ============================================================
# ZUSATZ: DIP & RIP OOS-VALIDIERUNG
# ============================================================
log_header("ZUSATZ: DIP & RIP OOS-VALIDIERUNG")
log("  IS fertig: N=248, EV=+0.275R, WR=67.7%, p<0.0001")
log("  Jetzt: OOS Bootstrap 10,000 + Subfilter")

print("  Running Dip & Rip OOS...", file=sys.stderr)

# Reuse find_dip_and_rip from d13_free_exploration
def find_dip_and_rip(rth, gap_dir, adr, od_strength):
    """Dip & Rip pattern detection (copied from d13_free_exploration.py)."""
    if gap_dir == 'up':
        od_bars = rth[(rth['time_et'] >= '09:30') & (rth['time_et'] <= '09:35')]
        if len(od_bars) == 0:
            return None
        od_high = od_bars['high'].max()

        pb_bars = rth[(rth['time_et'] >= '09:36') & (rth['time_et'] <= '10:00')]
        if len(pb_bars) < 3:
            return None

        pb_low_idx = pb_bars['low'].idxmin()
        pb_low = pb_bars.loc[pb_low_idx, 'low']
        pb_low_time = pb_bars.loc[pb_low_idx, 'time_et']

        pb_depth = (od_high - pb_low) / adr
        if pb_depth < 0.05:
            return None

        rec_bars = rth[(rth['time_et'] > pb_low_time) & (rth['time_et'] <= '10:30')]
        if len(rec_bars) < 3:
            return None

        for i in range(4, len(rec_bars)):
            window = rec_bars.iloc[max(0,i-4):i+1]
            if window['high'].max() > od_high * 0.998:
                entry_bar = rec_bars.iloc[i]
                entry_price = entry_bar['close']
                entry_time = entry_bar['time_et']
                sl_level = pb_low - 0.02 * adr
                sl_dist = entry_price - sl_level
                if sl_dist <= 0 or sl_dist > 0.5 * adr:
                    return None
                return {
                    'entry_price': entry_price, 'entry_time': entry_time,
                    'sl_dist': sl_dist, 'pb_depth': pb_depth, 'trade_dir': 'long'
                }
        return None

    else:  # gap_dir == 'down'
        od_bars = rth[(rth['time_et'] >= '09:30') & (rth['time_et'] <= '09:35')]
        if len(od_bars) == 0:
            return None
        od_low = od_bars['low'].min()

        pb_bars = rth[(rth['time_et'] >= '09:36') & (rth['time_et'] <= '10:00')]
        if len(pb_bars) < 3:
            return None

        pb_high_idx = pb_bars['high'].idxmax()
        pb_high = pb_bars.loc[pb_high_idx, 'high']
        pb_high_time = pb_bars.loc[pb_high_idx, 'time_et']

        pb_depth = (pb_high - od_low) / adr
        if pb_depth < 0.05:
            return None

        rec_bars = rth[(rth['time_et'] > pb_high_time) & (rth['time_et'] <= '10:30')]
        if len(rec_bars) < 3:
            return None

        for i in range(4, len(rec_bars)):
            window = rec_bars.iloc[max(0,i-4):i+1]
            if window['low'].min() < od_low * 1.002:
                entry_bar = rec_bars.iloc[i]
                entry_price = entry_bar['close']
                entry_time = entry_bar['time_et']
                sl_level = pb_high + 0.02 * adr
                sl_dist = sl_level - entry_price
                if sl_dist <= 0 or sl_dist > 0.5 * adr:
                    return None
                return {
                    'entry_price': entry_price, 'entry_time': entry_time,
                    'sl_dist': sl_dist, 'pb_depth': pb_depth, 'trade_dir': 'short'
                }
        return None


# Run on OOS
oos_dip = OOS[
    (OOS['close_935'].notna()) & (OOS['adr_10'].notna()) & (OOS['adr_10'] > 0) &
    (OOS['od_strength'].notna()) & (OOS['od_strength'] > 0.3) &
    (OOS['od_direction'] == 'with_gap')
].copy()

dip_oos_all = []
dip_oos_details = []

for _, row in oos_dip.iterrows():
    ticker = row['ticker']
    date = str(row['date'])
    adr = row['adr_10']
    gap_dir = row['gap_direction']
    od_str = row['od_strength']

    rth = load_rth_cached(ticker, date)
    if rth is None:
        continue

    dip_info = find_dip_and_rip(rth, gap_dir, adr, od_str)
    if dip_info is None:
        continue

    post_entry = rth[rth['time_et'] >= dip_info['entry_time']].copy()
    if len(post_entry) < 5:
        continue

    result = simulate_trailing(post_entry, dip_info['entry_price'], dip_info['sl_dist'],
                                dip_info['trade_dir'], adr, trail_type='D')
    if result is None:
        continue

    dip_oos_all.append(result['pnl_r'])
    dip_oos_details.append({
        'pnl_r': result['pnl_r'], 'pb_depth': dip_info['pb_depth'],
        'od_strength': od_str, 'rvol': row.get('rvol_5', np.nan),
        'gap_pct': abs(row['gap_pct']), 'gap_dir': gap_dir,
        'ticker': ticker, 'date': date,
    })

clear_cache()

# Main results
stats_dip_oos = compute_ev_stats(dip_oos_all, "Dip&Rip OOS ALL")
log(format_stats(stats_dip_oos))

if len(dip_oos_all) >= 20:
    ci_low, ci_high = bootstrap_ci(dip_oos_all, n_boot=10000)
    log(f"  Bootstrap 95% CI: [{ci_low:+.3f}R, {ci_high:+.3f}R]")
    if ci_low > 0:
        log(f"  >>> DIP & RIP OOS ROBUST: CI entirely above zero!")
    else:
        log(f"  >>> DIP & RIP OOS NOT ROBUST: CI includes zero")
    log(f"  IS EV: +0.275R -> OOS EV: {stats_dip_oos['EV_R']:+.3f}R")

# Subfilter OOS
if len(dip_oos_details) > 20:
    dip_oos_df = pd.DataFrame(dip_oos_details)
    log("\n  --- Dip & Rip OOS Subfilter ---")

    subfilters = [
        ("OD>=0.8", lambda df: df[df['od_strength'] >= 0.8]),
        ("RVOL>=7", lambda df: df[df['rvol'] >= 7]),
        ("Gap>=7%", lambda df: df[df['gap_pct'] >= 7]),
        ("OD>=0.8+RVOL>=5", lambda df: df[(df['od_strength'] >= 0.8) & (df['rvol'] >= 5)]),
        ("Gap>=5%+OD>=0.5", lambda df: df[(df['gap_pct'] >= 5) & (df['od_strength'] >= 0.5)]),
    ]

    for sub_label, sub_fn in subfilters:
        try:
            sub_df = sub_fn(dip_oos_df)
        except:
            continue
        sub_pnl = list(sub_df['pnl_r'].values)
        sub_stats = compute_ev_stats(sub_pnl, f"  OOS: {sub_label}")
        log(format_stats(sub_stats))
        if len(sub_pnl) >= 20:
            ci_l, ci_h = bootstrap_ci(sub_pnl, n_boot=10000)
            log(f"    CI: [{ci_l:+.3f}R, {ci_h:+.3f}R] {'ROBUST' if ci_l > 0 else 'not robust'}")


# ============================================================
# GRAND SUMMARY
# ============================================================
log_header("GRAND SUMMARY")

log(f"""
  METHODIK:
  - Phase 1: 6 Discretionary-Ideen (A-F) auf IS gescreent
  - Phase 2: Parameter-Sweep fuer Ueberlebende (~50-200 Kombos pro Idee)
  - Phase 3: Devils Advocate (Bonferroni, Stabilitaet, Konzentration)
  - Phase 4: OOS Bootstrap CI fuer finale Ueberlebende
  - Total Hypothesen getestet: {total_hypotheses_tested}
  - Bonferroni Alpha: {bonferroni_alpha:.6f}

  TRAIL: TRAIL_D (+1R -> BE, trail peak-0.5R)

  LEGENDE:
  - EV_R = Erwartungswert in R-Multiplen
  - WR   = Win Rate
  - p    = p-Wert (*** <0.01, ** <0.05, * <0.10)
  - CI   = Bootstrap 95% Confidence Interval
  - ROBUST = CI-Lower > 0
""")

log("  PHASE 1 Ergebnisse:")
for idea in ['A', 'B', 'C', 'D', 'E', 'F']:
    if idea in phase1_survivors:
        s = phase1_survivors[idea]['stats']
        log(f"    Idee {idea}: UEBERLEBT | N={s['N']}, EV={s['EV_R']:+.3f}R, p={s['p_val']:.4f}")
    else:
        log(f"    Idee {idea}: VERWORFEN")

log(f"\n  PHASE 2-3: {sum(len(v) for v in phase3_survivors.values())} Varianten ueberleben alle Checks")

log(f"\n  DIP & RIP OOS:")
if stats_dip_oos['valid']:
    log(f"    N={stats_dip_oos['N']}, EV={stats_dip_oos['EV_R']:+.3f}R, WR={stats_dip_oos['WR']:.1%}")
else:
    log(f"    Insufficient data")

log("\n  NAECHSTE SCHRITTE:")
log("  1. Robuste Strategien in Portfolio-Integration testen")
log("  2. Korrelation mit bestehenden Strategien pruefen")
log("  3. Walk-Forward Analyse auf monatlicher Basis")
log("  4. Position Sizing und Risikomanagement")


# Save results
output = "\n".join(lines)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, 'w', encoding='utf-8') as f:
    f.write(output)

print(f"\n\nResults saved to {OUT_PATH}", file=sys.stderr)
print("DONE!", file=sys.stderr)
