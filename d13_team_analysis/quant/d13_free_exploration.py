"""
D13 QUANT FREE EXPLORATION - Neue Strategien finden
=====================================================
10 neue Ideen, kreative Exploration, volle Power.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, 'scripts')
from d10_trailing import simulate_trailing, simulate_fix_target

RAW_DIR = Path('data/raw_1min')
VWAP_DIR = Path('data/vwap')
VP_DIR = Path('data/volume_profile')
META_PATH = Path('data/metadata/metadata_v9.parquet')
OUT_PATH = Path('d13_team_analysis/results/d13_free_exploration_results.txt')

IS_START = '2021-02-21'
IS_END = '2023-12-31'
OOS_START = '2024-01-01'
OOS_END = '2026-02-13'

lines = []

def log(msg):
    lines.append(msg)
    print(msg)

def log_header(title):
    log(f"\n{'='*80}")
    log(f"  {title}")
    log(f"{'='*80}")

def compute_ev_stats(results, label=""):
    """Compute EV stats from a list of pnl_r values."""
    if len(results) < 10:
        return {'N': len(results), 'label': label, 'valid': False}
    arr = np.array(results)
    ev = np.mean(arr)
    med = np.median(arr)
    std = np.std(arr, ddof=1)
    wr = np.mean(arr > 0)
    # t-test
    se = std / np.sqrt(len(arr))
    t_stat = ev / se if se > 0 else 0
    # approx p-value (2-sided) using normal approx
    from scipy import stats as sp_stats
    p_val = 2 * (1 - sp_stats.t.cdf(abs(t_stat), df=len(arr)-1))
    return {
        'N': len(arr), 'EV_R': ev, 'Med_R': med, 'Std_R': std,
        'WR': wr, 't_stat': t_stat, 'p_val': p_val,
        'label': label, 'valid': True
    }

def format_stats(s):
    if not s['valid']:
        return f"  {s['label']:<40} | N={s['N']:>5} | [INSUFFICIENT DATA]"
    sig = "***" if s['p_val'] < 0.01 else "**" if s['p_val'] < 0.05 else "*" if s['p_val'] < 0.10 else ""
    return (f"  {s['label']:<40} | N={s['N']:>5} | EV={s['EV_R']:>+6.3f}R | "
            f"Med={s['Med_R']:>+6.3f}R | WR={s['WR']:>5.1%} | "
            f"t={s['t_stat']:>+5.2f} | p={s['p_val']:>.4f} {sig}")

def bootstrap_ci(results, n_boot=10000, ci=0.95):
    """Bootstrap 95% CI for mean EV."""
    arr = np.array(results)
    n = len(arr)
    if n < 20:
        return (np.nan, np.nan)
    boot_means = np.array([np.mean(np.random.choice(arr, n, replace=True)) for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    return (np.percentile(boot_means, 100*alpha), np.percentile(boot_means, 100*(1-alpha)))

def load_bars(ticker, date):
    """Load 1min bars for ticker/date."""
    fpath = RAW_DIR / ticker / f"{date}.parquet"
    if not fpath.exists():
        return None
    bars = pd.read_parquet(fpath)
    return bars

def get_rth(bars):
    """Get RTH session bars."""
    if bars is None:
        return None
    rth = bars[bars['session'] == 'rth'].copy()
    if len(rth) < 10:
        return None
    return rth


# ============================================================
# LOAD METADATA
# ============================================================
print("Loading metadata...", file=sys.stderr)
meta = pd.read_parquet(META_PATH)
IS = meta[(meta['date'] >= IS_START) & (meta['date'] <= IS_END)].copy()
OOS = meta[(meta['date'] >= OOS_START) & (meta['date'] <= OOS_END)].copy()
print(f"IS: {len(IS)}, OOS: {len(OOS)}", file=sys.stderr)

log("=" * 80)
log("  D13 QUANT FREE EXPLORATION - NEUE STRATEGIEN")
log("=" * 80)
log(f"  Datum: 2026-02-21")
log(f"  IS: {IS_START} bis {IS_END} (N={len(IS)})")
log(f"  OOS: {OOS_START} bis {OOS_END} (N={len(OOS)})")
log(f"  Minimum EV: 0.25R")
log(f"  Bereits getestet: OD Momentum, VWAP Sigma-2, No-Fill 11:00,")
log(f"                     Late-Session 13:00, ORB, VPOC Migration,")
log(f"                     Volume-Burst, Gap-Fill 5-Day Swing,")
log(f"                     Pre-10am Momentum, Dip&Rip (rudimentaer)")


# ============================================================
# IDEE 1: REVERSAL AFTER EXHAUSTION (z>2 then drop to z<1.5)
# ============================================================
log_header("IDEE 1: REVERSAL AFTER EXHAUSTION (VWAP z-score Signal)")
log("  Hypothese: Nach VWAP z>2 um 10:00 -> Warte auf z<1.5 -> Entry GEGEN die Bewegung")
log("  Nicht blosse Mean-Reversion, sondern KONKRETES Umkehrsignal")

print("  [1/10] Running Reversal After Exhaustion...", file=sys.stderr)

# Use metadata vwap_z_at_10am to identify exhaustion
# Then load 1min to find z<1.5 entry between 10:00-10:30
exhaustion_results_up = []
exhaustion_results_dn = []
exhaustion_count = 0

is_exhaustion = IS[
    (IS['vwap_z_at_10am'].notna()) &
    (IS['adr_10'].notna()) & (IS['adr_10'] > 0) &
    (IS['od_strength'].notna())
].copy()

for _, row in is_exhaustion.iterrows():
    z10 = row['vwap_z_at_10am']
    if abs(z10) < 2.0:
        continue

    ticker = row['ticker']
    date = str(row['date'])
    adr = row['adr_10']
    gap_dir = row['gap_direction']

    # Load VWAP data to find z-score crossing below 1.5
    vwap_path = VWAP_DIR / ticker / f"{date}.parquet"
    if not vwap_path.exists():
        continue

    try:
        vwap_data = pd.read_parquet(vwap_path)
    except:
        continue

    if 'z_score' not in vwap_data.columns or 'time_et' not in vwap_data.columns:
        continue

    # Find entry: z drops below 1.5 (for z>2) or rises above -1.5 (for z<-2) between 10:00-10:30
    window = vwap_data[(vwap_data['time_et'] >= '10:00') & (vwap_data['time_et'] <= '10:30')]

    entry_bar = None
    if z10 > 2.0:
        # Was overbought, wait for z to drop below 1.5 -> short
        for _, vrow in window.iterrows():
            if vrow['z_score'] < 1.5:
                entry_bar = vrow
                break
        trade_dir = 'short'
    else:  # z10 < -2.0
        # Was oversold, wait for z to rise above -1.5 -> long
        for _, vrow in window.iterrows():
            if vrow['z_score'] > -1.5:
                entry_bar = vrow
                break
        trade_dir = 'long'

    if entry_bar is None:
        continue

    # Load 1min bars for trailing sim
    bars = load_bars(ticker, date)
    if bars is None:
        continue
    rth = get_rth(bars)
    if rth is None:
        continue

    entry_time = entry_bar['time_et']
    # Find close price at entry time
    entry_bars = rth[rth['time_et'] == entry_time]
    if len(entry_bars) == 0:
        continue

    entry_price = entry_bars.iloc[0]['close']
    sl_dist = 0.25 * adr

    # Use bars AFTER entry
    post_entry = rth[rth['time_et'] >= entry_time].copy()
    if len(post_entry) < 5:
        continue

    result = simulate_trailing(post_entry, entry_price, sl_dist, trade_dir, adr, trail_type='D')
    if result is None:
        continue

    exhaustion_count += 1
    if trade_dir == 'short':
        exhaustion_results_up.append(result['pnl_r'])
    else:
        exhaustion_results_dn.append(result['pnl_r'])

all_exhaustion = exhaustion_results_up + exhaustion_results_dn
stats = compute_ev_stats(all_exhaustion, "Reversal After Exhaustion (ALL)")
log(format_stats(stats))
stats_up = compute_ev_stats(exhaustion_results_up, "  -> Short (after z>2)")
log(format_stats(stats_up))
stats_dn = compute_ev_stats(exhaustion_results_dn, "  -> Long (after z<-2)")
log(format_stats(stats_dn))


# ============================================================
# IDEE 2: SECTOR MOMENTUM (Multiple gaps same sector same day)
# ============================================================
log_header("IDEE 2: SECTOR MOMENTUM (Multi-Stock Sektor-Gaps)")
log("  Hypothese: Wenn >= 2 Stocks im selben Sektor am selben Tag gappen -> Sektor-Impuls")

print("  [2/10] Running Sector Momentum...", file=sys.stderr)

# Count gaps per sector per day
is_valid = IS[
    (IS['sector'].notna()) & (IS['sector'] != 'Unknown') &
    (IS['close_935'].notna()) & (IS['adr_10'].notna()) & (IS['adr_10'] > 0)
].copy()

sector_day_counts = is_valid.groupby(['date', 'sector']).size().reset_index(name='sector_gap_count')
is_valid = is_valid.merge(sector_day_counts, on=['date', 'sector'], how='left')

sector_results = {}
for min_count in [2, 3, 5]:
    subset = is_valid[is_valid['sector_gap_count'] >= min_count]
    pnl_list = []
    for _, row in subset.iterrows():
        ticker = row['ticker']
        date = str(row['date'])
        entry_price = row['close_935']
        adr = row['adr_10']
        gap_dir = row['gap_direction']
        trade_dir = 'long' if gap_dir == 'up' else 'short'
        sl_dist = 0.25 * adr

        bars = load_bars(ticker, date)
        if bars is None:
            continue
        rth = get_rth(bars)
        if rth is None:
            continue

        result = simulate_trailing(rth, entry_price, sl_dist, trade_dir, adr, trail_type='D')
        if result is None:
            continue
        pnl_list.append(result['pnl_r'])

    stats = compute_ev_stats(pnl_list, f"Sector >= {min_count} stocks same day")
    log(format_stats(stats))
    sector_results[min_count] = stats

# Compare: Sector momentum vs isolated gaps
isolated = is_valid[is_valid['sector_gap_count'] == 1]
pnl_iso = []
# Sample to keep runtime manageable
iso_sample = isolated.sample(min(500, len(isolated)), random_state=42)
for _, row in iso_sample.iterrows():
    ticker = row['ticker']
    date = str(row['date'])
    entry_price = row['close_935']
    adr = row['adr_10']
    gap_dir = row['gap_direction']
    trade_dir = 'long' if gap_dir == 'up' else 'short'
    sl_dist = 0.25 * adr
    bars = load_bars(ticker, date)
    if bars is None:
        continue
    rth = get_rth(bars)
    if rth is None:
        continue
    result = simulate_trailing(rth, entry_price, sl_dist, trade_dir, adr, trail_type='D')
    if result is None:
        continue
    pnl_iso.append(result['pnl_r'])

stats_iso = compute_ev_stats(pnl_iso, "Isolated (1 stock in sector)")
log(format_stats(stats_iso))


# ============================================================
# IDEE 3: PRIOR RUN-UP + GAP (Multi-Day Momentum)
# ============================================================
log_header("IDEE 3: PRIOR RUN-UP + GAP (Double Momentum)")
log("  Hypothese: 5d-Vorlauf + Gap in gleicher Richtung = Extra-Momentum")

print("  [3/10] Running Prior Run-Up + Gap...", file=sys.stderr)

is_prior = IS[
    (IS['prior_return_5d'].notna()) &
    (IS['close_935'].notna()) & (IS['adr_10'].notna()) & (IS['adr_10'] > 0)
].copy()

# Scenario A: Prior 5d up + GapUp (Double Bull Momentum)
for label, filt_fn, tdir in [
    ("5d_UP + GapUP (Double Bull)",
     lambda df: (df['prior_return_5d'] > 5) & (df['gap_direction'] == 'up'), 'long'),
    ("5d_DN + GapDN (Double Bear)",
     lambda df: (df['prior_return_5d'] < -5) & (df['gap_direction'] == 'down'), 'short'),
    ("5d_UP>10 + GapUP (Strong Bull)",
     lambda df: (df['prior_return_5d'] > 10) & (df['gap_direction'] == 'up'), 'long'),
    ("5d_DN<-10 + GapDN (Strong Bear)",
     lambda df: (df['prior_return_5d'] < -10) & (df['gap_direction'] == 'down'), 'short'),
    ("10d_UP>15 + GapUP",
     lambda df: (df['prior_return_10d'] > 15) & (df['gap_direction'] == 'up'), 'long'),
    ("Near 52w High + GapUP",
     lambda df: (df['dist_from_52w_high'] > -5) & (df['gap_direction'] == 'up'), 'long'),
    ("Far from 52w High + GapDN",
     lambda df: (df['dist_from_52w_high'] < -40) & (df['gap_direction'] == 'down'), 'short'),
]:
    subset = is_prior[filt_fn(is_prior)]
    pnl_list = []
    sample = subset.sample(min(500, len(subset)), random_state=42) if len(subset) > 500 else subset
    for _, row in sample.iterrows():
        ticker = row['ticker']
        date = str(row['date'])
        entry_price = row['close_935']
        adr = row['adr_10']
        sl_dist = 0.25 * adr
        bars = load_bars(ticker, date)
        if bars is None:
            continue
        rth = get_rth(bars)
        if rth is None:
            continue
        result = simulate_trailing(rth, entry_price, sl_dist, tdir, adr, trail_type='D')
        if result is None:
            continue
        pnl_list.append(result['pnl_r'])

    stats = compute_ev_stats(pnl_list, label)
    log(format_stats(stats))


# ============================================================
# IDEE 4: SPY ALIGNMENT (Markt-Richtung)
# ============================================================
log_header("IDEE 4: SPY ALIGNMENT (Market Direction Filter)")
log("  Hypothese: GapUp an SPY-positiven Tagen -> besserer EV")
log("  Feature: spy_return_day")

print("  [4/10] Running SPY Alignment...", file=sys.stderr)

is_spy = IS[
    (IS['spy_return_day'].notna()) &
    (IS['close_935'].notna()) & (IS['adr_10'].notna()) & (IS['adr_10'] > 0) &
    (IS['od_strength'].notna())
].copy()

for label, filt_fn, trade_dir_fn in [
    ("GapUP + SPY_pos (aligned)",
     lambda df: (df['gap_direction']=='up') & (df['spy_return_day'] > 0),
     lambda r: 'long'),
    ("GapUP + SPY_neg (counter)",
     lambda df: (df['gap_direction']=='up') & (df['spy_return_day'] < 0),
     lambda r: 'long'),
    ("GapDN + SPY_neg (aligned)",
     lambda df: (df['gap_direction']=='down') & (df['spy_return_day'] < 0),
     lambda r: 'short'),
    ("GapDN + SPY_pos (counter)",
     lambda df: (df['gap_direction']=='down') & (df['spy_return_day'] > 0),
     lambda r: 'short'),
    ("GapUP + SPY>0.5% (strong align)",
     lambda df: (df['gap_direction']=='up') & (df['spy_return_day'] > 0.5),
     lambda r: 'long'),
    ("GapDN + SPY<-0.5% (strong align)",
     lambda df: (df['gap_direction']=='down') & (df['spy_return_day'] < -0.5),
     lambda r: 'short'),
]:
    subset = is_spy[filt_fn(is_spy)]
    pnl_list = []
    sample = subset.sample(min(500, len(subset)), random_state=42) if len(subset) > 500 else subset
    for _, row in sample.iterrows():
        ticker = row['ticker']
        date = str(row['date'])
        entry_price = row['close_935']
        adr = row['adr_10']
        tdir = trade_dir_fn(row)
        sl_dist = 0.25 * adr
        bars = load_bars(ticker, date)
        if bars is None:
            continue
        rth = get_rth(bars)
        if rth is None:
            continue
        result = simulate_trailing(rth, entry_price, sl_dist, tdir, adr, trail_type='D')
        if result is None:
            continue
        pnl_list.append(result['pnl_r'])
    stats = compute_ev_stats(pnl_list, label)
    log(format_stats(stats))

# SPY aligned + OD with_gap filter
log("\n  --- SPY Aligned + OD with_gap Filter ---")
for label, filt_fn, tdir in [
    ("GapUP+SPY_pos+OD_with (full align)",
     lambda df: (df['gap_direction']=='up') & (df['spy_return_day']>0) & (df['od_direction']=='with_gap') & (df['od_strength']>0.5),
     'long'),
    ("GapDN+SPY_neg+OD_with (full align)",
     lambda df: (df['gap_direction']=='down') & (df['spy_return_day']<0) & (df['od_direction']=='with_gap') & (df['od_strength']>0.5),
     'short'),
]:
    subset = is_spy[filt_fn(is_spy)]
    pnl_list = []
    for _, row in subset.iterrows():
        ticker = row['ticker']
        date = str(row['date'])
        entry_price = row['close_935']
        adr = row['adr_10']
        sl_dist = 0.25 * adr
        bars = load_bars(ticker, date)
        if bars is None:
            continue
        rth = get_rth(bars)
        if rth is None:
            continue
        result = simulate_trailing(rth, entry_price, sl_dist, tdir, adr, trail_type='D')
        if result is None:
            continue
        pnl_list.append(result['pnl_r'])
    stats = compute_ev_stats(pnl_list, label)
    log(format_stats(stats))


# ============================================================
# IDEE 5: EARNINGS SURPRISE CONTINUATION
# ============================================================
log_header("IDEE 5: EARNINGS SURPRISE CONTINUATION")
log("  Hypothese: Earnings-Gaps mit starkem OD haben Extra-Continuation")

print("  [5/10] Running Earnings Continuation...", file=sys.stderr)

is_earn = IS[
    (IS['is_earnings'] == True) &
    (IS['close_935'].notna()) & (IS['adr_10'].notna()) & (IS['adr_10'] > 0)
].copy()
is_noearn = IS[
    (IS['is_earnings'] == False) &
    (IS['close_935'].notna()) & (IS['adr_10'].notna()) & (IS['adr_10'] > 0)
].copy()

for label, subset in [
    ("Earnings ALL", is_earn),
    ("Non-Earnings ALL", is_noearn.sample(min(500, len(is_noearn)), random_state=42)),
]:
    pnl_list = []
    for _, row in subset.iterrows():
        ticker = row['ticker']
        date = str(row['date'])
        entry_price = row['close_935']
        adr = row['adr_10']
        gap_dir = row['gap_direction']
        trade_dir = 'long' if gap_dir == 'up' else 'short'
        sl_dist = 0.25 * adr
        bars = load_bars(ticker, date)
        if bars is None:
            continue
        rth = get_rth(bars)
        if rth is None:
            continue
        result = simulate_trailing(rth, entry_price, sl_dist, trade_dir, adr, trail_type='D')
        if result is None:
            continue
        pnl_list.append(result['pnl_r'])
    stats = compute_ev_stats(pnl_list, label)
    log(format_stats(stats))

# Earnings with strong OD
for label, filt_fn, tdir in [
    ("Earnings + OD>0.5 + GapUP",
     lambda df: (df['od_strength']>0.5) & (df['od_direction']=='with_gap') & (df['gap_direction']=='up'),
     'long'),
    ("Earnings + OD>0.5 + GapDN",
     lambda df: (df['od_strength']>0.5) & (df['od_direction']=='with_gap') & (df['gap_direction']=='down'),
     'short'),
    ("Earnings + BigGap>5% + OD>0.5",
     lambda df: (df['gap_pct'].abs()>5) & (df['od_strength']>0.5) & (df['od_direction']=='with_gap'),
     None),
    ("Earnings + RVOL>5 + OD>0.5",
     lambda df: (df['rvol_5']>5) & (df['od_strength']>0.5) & (df['od_direction']=='with_gap'),
     None),
]:
    subset = is_earn[filt_fn(is_earn)] if filt_fn is not None else is_earn
    pnl_list = []
    for _, row in subset.iterrows():
        ticker = row['ticker']
        date = str(row['date'])
        entry_price = row['close_935']
        adr = row['adr_10']
        if tdir is None:
            trade_dir = 'long' if row['gap_direction'] == 'up' else 'short'
        else:
            trade_dir = tdir
        sl_dist = 0.25 * adr
        bars = load_bars(ticker, date)
        if bars is None:
            continue
        rth = get_rth(bars)
        if rth is None:
            continue
        result = simulate_trailing(rth, entry_price, sl_dist, trade_dir, adr, trail_type='D')
        if result is None:
            continue
        pnl_list.append(result['pnl_r'])
    stats = compute_ev_stats(pnl_list, label)
    log(format_stats(stats))


# ============================================================
# IDEE 6: DAY OF WEEK EFFECT
# ============================================================
log_header("IDEE 6: DAY OF WEEK EFFECT")
log("  Hypothese: Bestimmte Wochentage haben systematisch besseren EV")

print("  [6/10] Running Day-of-Week Effect...", file=sys.stderr)

is_dow = IS[
    (IS['day_of_week'].notna()) &
    (IS['close_935'].notna()) & (IS['adr_10'].notna()) & (IS['adr_10'] > 0) &
    (IS['od_strength'].notna()) & (IS['od_strength'] > 0.5) &
    (IS['od_direction'] == 'with_gap')
].copy()

for dow in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']:
    subset = is_dow[is_dow['day_of_week'] == dow]
    pnl_list = []
    for _, row in subset.iterrows():
        ticker = row['ticker']
        date = str(row['date'])
        entry_price = row['close_935']
        adr = row['adr_10']
        gap_dir = row['gap_direction']
        trade_dir = 'long' if gap_dir == 'up' else 'short'
        sl_dist = 0.25 * adr
        bars = load_bars(ticker, date)
        if bars is None:
            continue
        rth = get_rth(bars)
        if rth is None:
            continue
        result = simulate_trailing(rth, entry_price, sl_dist, trade_dir, adr, trail_type='D')
        if result is None:
            continue
        pnl_list.append(result['pnl_r'])
    stats = compute_ev_stats(pnl_list, f"OD+with_gap | {dow}")
    log(format_stats(stats))


# ============================================================
# IDEE 7: DIP & RIP (Full Implementation)
# ============================================================
log_header("IDEE 7: DIP & RIP (Vollstaendige Implementierung)")
log("  Logik: Nach starkem OD -> Pullback -> Recovery -> Entry auf Recovery-Bar")
log("  SL: Pullback-Low - Buffer | Exit: TRAIL_D")

print("  [7/10] Running Dip & Rip (full)...", file=sys.stderr)

is_dip = IS[
    (IS['close_935'].notna()) & (IS['adr_10'].notna()) & (IS['adr_10'] > 0) &
    (IS['od_strength'].notna()) & (IS['od_strength'] > 0.3) &
    (IS['od_direction'] == 'with_gap') &
    (IS['gap_direction'] == 'up')  # Start with GapUp, then also GapDn
].copy()

def find_dip_and_rip(rth, gap_dir, adr, od_strength):
    """
    Find dip & rip pattern:
    1. After opening drive (09:35), find pullback low between 09:36-10:00
    2. Wait for recovery: new 5-bar high after the low
    3. Return entry info
    """
    if gap_dir == 'up':
        # Opening drive peak
        od_bars = rth[(rth['time_et'] >= '09:30') & (rth['time_et'] <= '09:35')]
        if len(od_bars) == 0:
            return None
        od_high = od_bars['high'].max()

        # Pullback window: 09:36 - 10:00
        pb_bars = rth[(rth['time_et'] >= '09:36') & (rth['time_et'] <= '10:00')]
        if len(pb_bars) < 3:
            return None

        # Find pullback low
        pb_low_idx = pb_bars['low'].idxmin()
        pb_low = pb_bars.loc[pb_low_idx, 'low']
        pb_low_time = pb_bars.loc[pb_low_idx, 'time_et']

        # Pullback depth (relative to OD high)
        pb_depth = (od_high - pb_low) / adr

        # Must have meaningful pullback (at least 5% of ADR)
        if pb_depth < 0.05:
            return None

        # Recovery window: after pullback low until 10:30
        rec_bars = rth[(rth['time_et'] > pb_low_time) & (rth['time_et'] <= '10:30')]
        if len(rec_bars) < 3:
            return None

        # Rolling 5-bar high
        for i in range(4, len(rec_bars)):
            window = rec_bars.iloc[max(0,i-4):i+1]
            if window['high'].max() > od_high * 0.998:  # Near or above OD high
                entry_bar = rec_bars.iloc[i]
                entry_price = entry_bar['close']
                entry_time = entry_bar['time_et']
                sl_level = pb_low - 0.02 * adr  # Small buffer below pullback low
                sl_dist = entry_price - sl_level
                if sl_dist <= 0 or sl_dist > 0.5 * adr:
                    return None
                return {
                    'entry_price': entry_price,
                    'entry_time': entry_time,
                    'sl_dist': sl_dist,
                    'pb_depth': pb_depth,
                    'trade_dir': 'long'
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
                    'entry_price': entry_price,
                    'entry_time': entry_time,
                    'sl_dist': sl_dist,
                    'pb_depth': pb_depth,
                    'trade_dir': 'short'
                }
        return None

# Run Dip & Rip for GapUp
dip_results_all = []
dip_details = []

# Also include GapDown
is_dip_all = IS[
    (IS['close_935'].notna()) & (IS['adr_10'].notna()) & (IS['adr_10'] > 0) &
    (IS['od_strength'].notna()) & (IS['od_strength'] > 0.3) &
    (IS['od_direction'] == 'with_gap')
].copy()

sample_dip = is_dip_all.sample(min(1500, len(is_dip_all)), random_state=42)
dip_count = 0
dip_results_by_depth = {'shallow': [], 'medium': [], 'deep': []}

for _, row in sample_dip.iterrows():
    ticker = row['ticker']
    date = str(row['date'])
    adr = row['adr_10']
    gap_dir = row['gap_direction']
    od_str = row['od_strength']

    bars = load_bars(ticker, date)
    if bars is None:
        continue
    rth = get_rth(bars)
    if rth is None:
        continue

    dip_info = find_dip_and_rip(rth, gap_dir, adr, od_str)
    if dip_info is None:
        continue

    dip_count += 1
    entry_price = dip_info['entry_price']
    sl_dist = dip_info['sl_dist']
    trade_dir = dip_info['trade_dir']
    entry_time = dip_info['entry_time']

    # Simulate from entry time
    post_entry = rth[rth['time_et'] >= entry_time].copy()
    if len(post_entry) < 5:
        continue

    result = simulate_trailing(post_entry, entry_price, sl_dist, trade_dir, adr, trail_type='D')
    if result is None:
        continue

    dip_results_all.append(result['pnl_r'])

    pb_depth = dip_info['pb_depth']
    if pb_depth < 0.10:
        dip_results_by_depth['shallow'].append(result['pnl_r'])
    elif pb_depth < 0.20:
        dip_results_by_depth['medium'].append(result['pnl_r'])
    else:
        dip_results_by_depth['deep'].append(result['pnl_r'])

    dip_details.append({
        'pnl_r': result['pnl_r'], 'pb_depth': pb_depth,
        'od_strength': od_str, 'gap_dir': gap_dir,
        'rvol': row.get('rvol_5', np.nan),
        'gap_pct': abs(row['gap_pct']),
    })

log(f"\n  Dip & Rip Signals found: {dip_count} (from {len(sample_dip)} scanned)")
stats = compute_ev_stats(dip_results_all, "Dip & Rip ALL")
log(format_stats(stats))

for depth_label, depth_results in dip_results_by_depth.items():
    stats = compute_ev_stats(depth_results, f"  -> Depth: {depth_label}")
    log(format_stats(stats))

# Parameter interactions for Dip & Rip
if len(dip_details) > 30:
    dip_df = pd.DataFrame(dip_details)
    log("\n  --- Dip & Rip Parameter Matrix ---")

    for param, bins, labels in [
        ('od_strength', [0, 0.5, 0.8, 2.0], ['OD:0.3-0.5', 'OD:0.5-0.8', 'OD:0.8+']),
        ('rvol', [0, 3, 7, 1000], ['RVOL:<3', 'RVOL:3-7', 'RVOL:7+']),
        ('gap_pct', [0, 3, 7, 200], ['Gap:<3%', 'Gap:3-7%', 'Gap:7%+']),
    ]:
        dip_df[f'{param}_bin'] = pd.cut(dip_df[param], bins=bins, labels=labels)
        for label in labels:
            sub = dip_df[dip_df[f'{param}_bin'] == label]['pnl_r'].values
            stats = compute_ev_stats(list(sub), f"    {label}")
            log(format_stats(stats))


# ============================================================
# IDEE 8: RSI EXTREME + GAP (Overbought/Oversold Context)
# ============================================================
log_header("IDEE 8: RSI EXTREME + GAP (Momentum vs Exhaustion)")
log("  Hypothese: RSI>70 + GapUp = Ueberhitzung ODER starkes Momentum?")
log("  Feature: rsi_14_prev (RSI vom Vortag)")

print("  [8/10] Running RSI Extreme + Gap...", file=sys.stderr)

is_rsi = IS[
    (IS['rsi_14_prev'].notna()) &
    (IS['close_935'].notna()) & (IS['adr_10'].notna()) & (IS['adr_10'] > 0)
].copy()

for label, filt_fn, tdir in [
    ("RSI>70 + GapUP (Overbought Bull)",
     lambda df: (df['rsi_14_prev']>70) & (df['gap_direction']=='up'), 'long'),
    ("RSI<30 + GapDN (Oversold Bear)",
     lambda df: (df['rsi_14_prev']<30) & (df['gap_direction']=='down'), 'short'),
    ("RSI>70 + GapUP + OD>0.5",
     lambda df: (df['rsi_14_prev']>70) & (df['gap_direction']=='up') & (df['od_strength']>0.5) & (df['od_direction']=='with_gap'), 'long'),
    ("RSI<30 + GapDN + OD>0.5",
     lambda df: (df['rsi_14_prev']<30) & (df['gap_direction']=='down') & (df['od_strength']>0.5) & (df['od_direction']=='with_gap'), 'short'),
    ("RSI 40-60 + GapUP (Neutral Cont.)",
     lambda df: (df['rsi_14_prev']>=40) & (df['rsi_14_prev']<=60) & (df['gap_direction']=='up'), 'long'),
    ("RSI 40-60 + GapDN (Neutral Cont.)",
     lambda df: (df['rsi_14_prev']>=40) & (df['rsi_14_prev']<=60) & (df['gap_direction']=='down'), 'short'),
    # Contrarian
    ("RSI>70 + GapDN (Bull Pullback Short)",
     lambda df: (df['rsi_14_prev']>70) & (df['gap_direction']=='down'), 'short'),
    ("RSI<30 + GapUP (Bear Bounce Long)",
     lambda df: (df['rsi_14_prev']<30) & (df['gap_direction']=='up'), 'long'),
]:
    subset = is_rsi[filt_fn(is_rsi)]
    pnl_list = []
    sample = subset.sample(min(500, len(subset)), random_state=42) if len(subset) > 500 else subset
    for _, row in sample.iterrows():
        ticker = row['ticker']
        date = str(row['date'])
        entry_price = row['close_935']
        adr = row['adr_10']
        sl_dist = 0.25 * adr
        bars = load_bars(ticker, date)
        if bars is None:
            continue
        rth = get_rth(bars)
        if rth is None:
            continue
        result = simulate_trailing(rth, entry_price, sl_dist, tdir, adr, trail_type='D')
        if result is None:
            continue
        pnl_list.append(result['pnl_r'])
    stats = compute_ev_stats(pnl_list, label)
    log(format_stats(stats))


# ============================================================
# IDEE 9: MARKET CAP SWEET SPOT (Systematischer Effekt)
# ============================================================
log_header("IDEE 9: MARKET CAP EFFECT (Systematisch)")
log("  Hypothese: Bestimmte Market-Cap-Buckets haben systematisch besseren EV")

print("  [9/10] Running Market Cap Effect...", file=sys.stderr)

is_mc = IS[
    (IS['market_cap_bucket'].notna()) &
    (IS['close_935'].notna()) & (IS['adr_10'].notna()) & (IS['adr_10'] > 0) &
    (IS['od_strength'].notna()) & (IS['od_strength'] > 0.5) &
    (IS['od_direction'] == 'with_gap')
].copy()

for bucket in ['2-10B', '10-50B', '50-200B', '200B+']:
    subset = is_mc[is_mc['market_cap_bucket'] == bucket]
    pnl_list = []
    for _, row in subset.iterrows():
        ticker = row['ticker']
        date = str(row['date'])
        entry_price = row['close_935']
        adr = row['adr_10']
        gap_dir = row['gap_direction']
        trade_dir = 'long' if gap_dir == 'up' else 'short'
        sl_dist = 0.25 * adr
        bars = load_bars(ticker, date)
        if bars is None:
            continue
        rth = get_rth(bars)
        if rth is None:
            continue
        result = simulate_trailing(rth, entry_price, sl_dist, trade_dir, adr, trail_type='D')
        if result is None:
            continue
        pnl_list.append(result['pnl_r'])
    stats = compute_ev_stats(pnl_list, f"MC: {bucket} (OD>0.5)")
    log(format_stats(stats))

# Market Cap x Gap Direction interaction
log("\n  --- Market Cap x Gap Direction ---")
for bucket in ['2-10B', '10-50B', '50-200B', '200B+']:
    for gap_dir, tdir in [('up', 'long'), ('down', 'short')]:
        subset = is_mc[(is_mc['market_cap_bucket']==bucket) & (is_mc['gap_direction']==gap_dir)]
        pnl_list = []
        for _, row in subset.iterrows():
            ticker = row['ticker']
            date = str(row['date'])
            entry_price = row['close_935']
            adr = row['adr_10']
            sl_dist = 0.25 * adr
            bars = load_bars(ticker, date)
            if bars is None:
                continue
            rth = get_rth(bars)
            if rth is None:
                continue
            result = simulate_trailing(rth, entry_price, sl_dist, tdir, adr, trail_type='D')
            if result is None:
                continue
            pnl_list.append(result['pnl_r'])
        stats = compute_ev_stats(pnl_list, f"  MC:{bucket} x Gap{gap_dir}")
        log(format_stats(stats))


# ============================================================
# IDEE 10: CREATIVE COMBOS (Multi-Factor Stacking)
# ============================================================
log_header("IDEE 10: CREATIVE MULTI-FACTOR COMBOS")
log("  Die besten Einzel-Effekte kombiniert stapeln")

print("  [10/10] Running Creative Multi-Factor Combos...", file=sys.stderr)

is_combo = IS[
    (IS['close_935'].notna()) & (IS['adr_10'].notna()) & (IS['adr_10'] > 0) &
    (IS['od_strength'].notna()) & (IS['spy_return_day'].notna()) &
    (IS['prior_return_5d'].notna()) & (IS['rsi_14_prev'].notna()) &
    (IS['rvol_5'].notna())
].copy()

combos = [
    ("OD>0.5+SPY_align+5d_mom GapUP",
     lambda df: (df['od_strength']>0.5) & (df['od_direction']=='with_gap') &
                (df['spy_return_day']>0) & (df['prior_return_5d']>0) &
                (df['gap_direction']=='up'),
     'long'),
    ("OD>0.5+SPY_align+5d_mom GapDN",
     lambda df: (df['od_strength']>0.5) & (df['od_direction']=='with_gap') &
                (df['spy_return_day']<0) & (df['prior_return_5d']<0) &
                (df['gap_direction']=='down'),
     'short'),
    ("OD>0.8+RVOL>5+GapUP+SPY_pos",
     lambda df: (df['od_strength']>0.8) & (df['od_direction']=='with_gap') &
                (df['rvol_5']>5) & (df['gap_direction']=='up') &
                (df['spy_return_day']>0),
     'long'),
    ("OD>0.5+RSI_neutral+SPY_pos GapUP",
     lambda df: (df['od_strength']>0.5) & (df['od_direction']=='with_gap') &
                (df['rsi_14_prev']>=40) & (df['rsi_14_prev']<=65) &
                (df['spy_return_day']>0) & (df['gap_direction']=='up'),
     'long'),
    ("Earnings+OD>0.5+RVOL>5 GapUP",
     lambda df: (df['is_earnings']==True) & (df['od_strength']>0.5) &
                (df['od_direction']=='with_gap') & (df['rvol_5']>5) &
                (df['gap_direction']=='up'),
     'long'),
    ("MC:50-200B+OD>0.5+GapUP",
     lambda df: (df['market_cap_bucket']=='50-200B') & (df['od_strength']>0.5) &
                (df['od_direction']=='with_gap') & (df['gap_direction']=='up'),
     'long'),
    ("Near52wHi+OD>0.5+GapUP+SPY_pos",
     lambda df: (df['dist_from_52w_high']>-10) & (df['od_strength']>0.5) &
                (df['od_direction']=='with_gap') & (df['gap_direction']=='up') &
                (df['spy_return_day']>0),
     'long'),
    ("Far52wHi+OD>0.5+GapDN+SPY_neg",
     lambda df: (df['dist_from_52w_high']<-30) & (df['od_strength']>0.5) &
                (df['od_direction']=='with_gap') & (df['gap_direction']=='down') &
                (df['spy_return_day']<0),
     'short'),
    # First candle size interaction
    ("OD>0.5+BigFirstCandle>0.2+GapUP",
     lambda df: (df['od_strength']>0.5) & (df['od_direction']=='with_gap') &
                (df['first_candle_size']>0.2) & (df['gap_direction']=='up'),
     'long'),
    ("OD>0.5+BigBody>0.6+GapUP",
     lambda df: (df['od_strength']>0.5) & (df['od_direction']=='with_gap') &
                (df['od_body_pct']>0.6) & (df['gap_direction']=='up'),
     'long'),
    # Low wick ratio (clean candles)
    ("OD>0.5+LowWick<0.15+GapUP",
     lambda df: (df['od_strength']>0.5) & (df['od_direction']=='with_gap') &
                (df['od_wick_ratio']<0.15) & (df['gap_direction']=='up'),
     'long'),
    # Gap size sweet spot
    ("OD>0.5+GapInADR:1-2+GapUP",
     lambda df: (df['od_strength']>0.5) & (df['od_direction']=='with_gap') &
                (df['gap_size_in_adr']>=1) & (df['gap_size_in_adr']<=2) &
                (df['gap_direction']=='up'),
     'long'),
    ("OD>0.5+GapInADR:0.5-1+GapUP",
     lambda df: (df['od_strength']>0.5) & (df['od_direction']=='with_gap') &
                (df['gap_size_in_adr']>=0.5) & (df['gap_size_in_adr']<=1) &
                (df['gap_direction']=='up'),
     'long'),
]

combo_results = {}
for label, filt_fn, tdir in combos:
    try:
        subset = is_combo[filt_fn(is_combo)]
    except:
        continue
    pnl_list = []
    for _, row in subset.iterrows():
        ticker = row['ticker']
        date = str(row['date'])
        entry_price = row['close_935']
        adr = row['adr_10']
        sl_dist = 0.25 * adr
        bars = load_bars(ticker, date)
        if bars is None:
            continue
        rth = get_rth(bars)
        if rth is None:
            continue
        result = simulate_trailing(rth, entry_price, sl_dist, tdir, adr, trail_type='D')
        if result is None:
            continue
        pnl_list.append(result['pnl_r'])
    stats = compute_ev_stats(pnl_list, label)
    log(format_stats(stats))
    if stats['valid'] and stats['EV_R'] >= 0.25:
        combo_results[label] = {'stats': stats, 'pnl_list': pnl_list}


# ============================================================
# TOP STRATEGIES: OOS VALIDATION
# ============================================================
log_header("OOS VALIDATION: TOP STRATEGIES")
log("  Kriterium: IS EV >= 0.25R | Jetzt OOS testen mit Bootstrap 95% CI")

print("  Running OOS validation for top strategies...", file=sys.stderr)

# Collect all promising strategies (EV >= 0.25R, p < 0.10)
# We'll re-define them on OOS

oos_valid = OOS[
    (OOS['close_935'].notna()) & (OOS['adr_10'].notna()) & (OOS['adr_10'] > 0) &
    (OOS['od_strength'].notna()) & (OOS['spy_return_day'].notna()) &
    (OOS['prior_return_5d'].notna()) & (OOS['rsi_14_prev'].notna()) &
    (OOS['rvol_5'].notna())
].copy()

oos_broad = OOS[
    (OOS['close_935'].notna()) & (OOS['adr_10'].notna()) & (OOS['adr_10'] > 0) &
    (OOS['od_strength'].notna())
].copy()

# Define OOS test strategies
oos_strategies = []

# Add all combo strategies that passed IS
for label, info in combo_results.items():
    # Find matching combo definition
    for clabel, filt_fn, tdir in combos:
        if clabel == label:
            oos_strategies.append((label, filt_fn, tdir, info['stats']))
            break

# Also add best individual findings from earlier tests
# We'll track additional strategies dynamically

def run_oos_test(label, filt_fn, tdir, is_stats, dataset=None):
    if dataset is None:
        dataset = oos_valid
    try:
        subset = dataset[filt_fn(dataset)]
    except:
        log(f"  {label:<40} | ERROR applying filter to OOS")
        return

    pnl_list = []
    for _, row in subset.iterrows():
        ticker = row['ticker']
        date = str(row['date'])
        entry_price = row['close_935']
        adr = row['adr_10']
        sl_dist = 0.25 * adr
        bars = load_bars(ticker, date)
        if bars is None:
            continue
        rth = get_rth(bars)
        if rth is None:
            continue
        result = simulate_trailing(rth, entry_price, sl_dist, tdir, adr, trail_type='D')
        if result is None:
            continue
        pnl_list.append(result['pnl_r'])

    oos_stats = compute_ev_stats(pnl_list, f"OOS: {label}")
    log(format_stats(oos_stats))

    if len(pnl_list) >= 20:
        ci_low, ci_high = bootstrap_ci(pnl_list, n_boot=10000)
        log(f"    Bootstrap 95% CI: [{ci_low:+.3f}R, {ci_high:+.3f}R]")
        if ci_low > 0:
            log(f"    >>> ROBUST: CI entirely above zero!")
        is_ev = is_stats['EV_R'] if is_stats['valid'] else 0
        if oos_stats['valid']:
            log(f"    IS EV: {is_ev:+.3f}R -> OOS EV: {oos_stats['EV_R']:+.3f}R")

    return oos_stats, pnl_list

for label, filt_fn, tdir, is_stats in oos_strategies:
    run_oos_test(label, filt_fn, tdir, is_stats)

# Also run some key tests even if they didn't appear in combos
log("\n  --- Additional OOS Tests (Key Findings) ---")

additional_oos = [
    ("DayOfWeek: best day (identify from IS)",
     None, None, None),  # placeholder, will handle separately
]

# Day-of-week: run all days on OOS
oos_dow = OOS[
    (OOS['day_of_week'].notna()) & (OOS['close_935'].notna()) &
    (OOS['adr_10'].notna()) & (OOS['adr_10'] > 0) &
    (OOS['od_strength'].notna()) & (OOS['od_strength'] > 0.5) &
    (OOS['od_direction'] == 'with_gap')
].copy()

log("\n  --- Day of Week OOS ---")
for dow in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']:
    subset = oos_dow[oos_dow['day_of_week'] == dow]
    pnl_list = []
    for _, row in subset.iterrows():
        ticker = row['ticker']
        date = str(row['date'])
        entry_price = row['close_935']
        adr = row['adr_10']
        gap_dir = row['gap_direction']
        trade_dir = 'long' if gap_dir == 'up' else 'short'
        sl_dist = 0.25 * adr
        bars = load_bars(ticker, date)
        if bars is None:
            continue
        rth = get_rth(bars)
        if rth is None:
            continue
        result = simulate_trailing(rth, entry_price, sl_dist, trade_dir, adr, trail_type='D')
        if result is None:
            continue
        pnl_list.append(result['pnl_r'])
    stats = compute_ev_stats(pnl_list, f"OOS: OD+with_gap | {dow}")
    log(format_stats(stats))

# Market Cap OOS
oos_mc = OOS[
    (OOS['market_cap_bucket'].notna()) & (OOS['close_935'].notna()) &
    (OOS['adr_10'].notna()) & (OOS['adr_10'] > 0) &
    (OOS['od_strength'].notna()) & (OOS['od_strength'] > 0.5) &
    (OOS['od_direction'] == 'with_gap')
].copy()

log("\n  --- Market Cap OOS ---")
for bucket in ['2-10B', '10-50B', '50-200B', '200B+']:
    subset = oos_mc[oos_mc['market_cap_bucket'] == bucket]
    pnl_list = []
    for _, row in subset.iterrows():
        ticker = row['ticker']
        date = str(row['date'])
        entry_price = row['close_935']
        adr = row['adr_10']
        gap_dir = row['gap_direction']
        trade_dir = 'long' if gap_dir == 'up' else 'short'
        sl_dist = 0.25 * adr
        bars = load_bars(ticker, date)
        if bars is None:
            continue
        rth = get_rth(bars)
        if rth is None:
            continue
        result = simulate_trailing(rth, entry_price, sl_dist, trade_dir, adr, trail_type='D')
        if result is None:
            continue
        pnl_list.append(result['pnl_r'])
    stats = compute_ev_stats(pnl_list, f"OOS: MC:{bucket} (OD>0.5)")
    log(format_stats(stats))
    if len(pnl_list) >= 20:
        ci_low, ci_high = bootstrap_ci(pnl_list, n_boot=10000)
        log(f"    Bootstrap 95% CI: [{ci_low:+.3f}R, {ci_high:+.3f}R]")

# Earnings OOS
oos_earn = OOS[
    (OOS['is_earnings']==True) & (OOS['close_935'].notna()) &
    (OOS['adr_10'].notna()) & (OOS['adr_10'] > 0) &
    (OOS['od_strength'].notna()) & (OOS['od_strength'] > 0.5) &
    (OOS['od_direction'] == 'with_gap')
].copy()

log("\n  --- Earnings OOS ---")
for gap_dir, tdir in [('up', 'long'), ('down', 'short')]:
    subset = oos_earn[oos_earn['gap_direction'] == gap_dir]
    pnl_list = []
    for _, row in subset.iterrows():
        ticker = row['ticker']
        date = str(row['date'])
        entry_price = row['close_935']
        adr = row['adr_10']
        sl_dist = 0.25 * adr
        bars = load_bars(ticker, date)
        if bars is None:
            continue
        rth = get_rth(bars)
        if rth is None:
            continue
        result = simulate_trailing(rth, entry_price, sl_dist, tdir, adr, trail_type='D')
        if result is None:
            continue
        pnl_list.append(result['pnl_r'])
    stats = compute_ev_stats(pnl_list, f"OOS: Earnings+OD>0.5 Gap{gap_dir}")
    log(format_stats(stats))
    if len(pnl_list) >= 20:
        ci_low, ci_high = bootstrap_ci(pnl_list, n_boot=10000)
        log(f"    Bootstrap 95% CI: [{ci_low:+.3f}R, {ci_high:+.3f}R]")


# ============================================================
# BONUS: UNEXPECTED PATTERN DETECTION
# ============================================================
log_header("BONUS: UNEXPECTED PATTERN DETECTION")
log("  Automatische Suche nach unerwarteten Feature-Interaktionen")

print("  Running Bonus Pattern Detection...", file=sys.stderr)

# Use IS with OD base strategy
is_base = IS[
    (IS['close_935'].notna()) & (IS['adr_10'].notna()) & (IS['adr_10'] > 0) &
    (IS['od_strength'].notna()) & (IS['od_strength'] > 0.5) &
    (IS['od_direction'] == 'with_gap')
].copy()

# Quick scan: compute PnL for entire base on IS (sample 800)
base_sample = is_base.sample(min(800, len(is_base)), random_state=42)
base_pnl = {}

for _, row in base_sample.iterrows():
    ticker = row['ticker']
    date = str(row['date'])
    entry_price = row['close_935']
    adr = row['adr_10']
    gap_dir = row['gap_direction']
    trade_dir = 'long' if gap_dir == 'up' else 'short'
    sl_dist = 0.25 * adr
    bars = load_bars(ticker, date)
    if bars is None:
        continue
    rth = get_rth(bars)
    if rth is None:
        continue
    result = simulate_trailing(rth, entry_price, sl_dist, trade_dir, adr, trail_type='D')
    if result is None:
        continue
    idx = row.name
    base_pnl[idx] = result['pnl_r']

# Merge pnl back
pnl_series = pd.Series(base_pnl, name='pnl_r')
scan_df = base_sample.loc[pnl_series.index].copy()
scan_df['pnl_r'] = pnl_series

log(f"\n  Base strategy scan: N={len(scan_df)}, EV={scan_df['pnl_r'].mean():+.3f}R, WR={100*(scan_df['pnl_r']>0).mean():.1f}%")

# Feature split analysis
log("\n  --- Feature Split Analysis (Top vs Bottom Tercile) ---")
numeric_features = [
    'gap_pct', 'gap_size_in_adr', 'od_strength', 'first_candle_size',
    'od_body_pct', 'od_wick_ratio', 'rvol_5', 'rvol_open_30min',
    'prior_return_5d', 'prior_return_10d', 'dist_from_52w_high',
    'spy_return_day', 'rsi_14_prev', 'pm_rth5', 'premarket_volume',
    'time_in_vwap_bands', 'vwap_z_at_10am',
]

interesting_splits = []
for feat in numeric_features:
    if feat not in scan_df.columns or scan_df[feat].notna().sum() < 50:
        continue

    valid = scan_df[scan_df[feat].notna()]
    try:
        q33 = valid[feat].quantile(0.33)
        q67 = valid[feat].quantile(0.67)
    except:
        continue

    bottom = valid[valid[feat] <= q33]['pnl_r']
    top = valid[valid[feat] >= q67]['pnl_r']

    if len(bottom) < 15 or len(top) < 15:
        continue

    ev_bot = bottom.mean()
    ev_top = top.mean()
    diff = ev_top - ev_bot

    if abs(diff) > 0.15:
        interesting_splits.append((feat, ev_bot, ev_top, diff, len(bottom), len(top)))

interesting_splits.sort(key=lambda x: abs(x[3]), reverse=True)

for feat, ev_bot, ev_top, diff, n_bot, n_top in interesting_splits[:15]:
    direction = "HIGH better" if diff > 0 else "LOW better"
    log(f"  {feat:<25} | Low33%: {ev_bot:+.3f}R (N={n_bot}) | High33%: {ev_top:+.3f}R (N={n_top}) | Delta: {diff:+.3f}R | {direction}")


# ============================================================
# GRAND SUMMARY
# ============================================================
log_header("GRAND SUMMARY - TOP FINDINGS")

log("""
  LEGENDE:
  - EV_R = Erwartungswert in R-Multiplen (1R = 0.25 ADR Risiko)
  - WR   = Win Rate
  - p    = p-Wert des t-Tests (*** <0.01, ** <0.05, * <0.10)
  - CI   = Bootstrap 95% Confidence Interval

  BEWERTUNG:
  - EV >= 0.25R, p < 0.05, N >= 30: STRONG SIGNAL
  - EV >= 0.25R, p < 0.10, N >= 20: MODERATE SIGNAL
  - EV >= 0.25R, p >= 0.10:         WEAK / NOISE
  - OOS CI > 0:                      ROBUST (tradebar)

  Alle Strategien verwenden:
  - Entry: close_935 (oder spezifisch angegeben)
  - SL: 0.25 ADR (oder Dip&Rip: Pullback-Low)
  - Exit: TRAIL_D (+1R -> BE, trail peak-0.5R)
""")

log("\n  Naechste Schritte fuer positive Ergebnisse:")
log("  1. Volle Parameter-Matrix (SL-Varianten, Trail-Varianten)")
log("  2. Walk-Forward Analyse (jaehrliche Stabilitaet)")
log("  3. Korrelation mit bestehenden Strategien")
log("  4. Position Sizing und Portfolio-Integration")

# Save results
output = "\n".join(lines)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, 'w', encoding='utf-8') as f:
    f.write(output)

print(f"\n\nResults saved to {OUT_PATH}", file=sys.stderr)
print("DONE!", file=sys.stderr)
