"""
D13 PHASE 2: TRADER MICROSTRUCTURE & UMSETZBARKEITS-ANALYSE
============================================================
Discretionary Trader Agent - Analyse der 6 unabhaengigen Strategien
Fokus: Slippage, Liquiditaet, Entry-Qualitaet, Filter, OD-Unabhaengigkeit
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASE_DIR = PROJECT_ROOT
DATA_DIR = BASE_DIR / "data"
META_PATH = DATA_DIR / "metadata" / "metadata_v9.parquet"
RAW_1MIN_DIR = DATA_DIR / "raw_1min"
VWAP_DIR = DATA_DIR / "vwap"
VP_DIR = DATA_DIR / "volume_profile"
RESULTS_DIR = BASE_DIR / "d13_team_analysis" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = RESULTS_DIR / "d13_independent_trader_results.txt"

IS_START = "2021-02-21"
IS_END = "2023-12-31"
SAMPLE_SIZE = 300
np.random.seed(42)

# ============================================================
# LOAD METADATA
# ============================================================
print("Loading metadata...")
meta = pd.read_parquet(META_PATH)
meta['date'] = pd.to_datetime(meta['date'])
is_data = meta[(meta['date'] >= IS_START) & (meta['date'] <= IS_END)].copy()
print(f"  IS data: {len(is_data)} events ({IS_START} to {IS_END})")

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def load_1min(ticker, date_str):
    """Load 1-min bar data for a ticker/date."""
    path = RAW_1MIN_DIR / ticker / f"{date_str}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return df

def load_vwap(ticker, date_str):
    """Load VWAP data for a ticker/date."""
    path = VWAP_DIR / ticker / f"{date_str}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return df

def load_vp(ticker, date_str):
    """Load volume profile for a ticker/date."""
    path = VP_DIR / ticker / f"{date_str}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return df

def get_rth_bars(bars_df):
    """Filter to RTH (regular trading hours) bars only."""
    if bars_df is None:
        return None
    rth = bars_df[bars_df['session'] == 'regular'].copy()
    if len(rth) == 0:
        # fallback: filter by time
        rth = bars_df[(bars_df['time_et'] >= '09:30') & (bars_df['time_et'] < '16:00')].copy()
    return rth

def time_to_idx(rth_df, target_time):
    """Find index closest to target_time in RTH bars."""
    if rth_df is None or len(rth_df) == 0:
        return None
    matches = rth_df[rth_df['time_et'] == target_time]
    if len(matches) > 0:
        return matches.index[0]
    # find closest
    times = rth_df['time_et'].values
    for i, t in enumerate(times):
        if t >= target_time:
            return rth_df.index[i]
    return None

def bar_range_adr(bar_row, adr):
    """Compute bar range as fraction of ADR."""
    if adr == 0 or pd.isna(adr):
        return np.nan
    return (bar_row['high'] - bar_row['low']) / adr

def sample_events(df, n=SAMPLE_SIZE):
    """Sample n events, or all if fewer."""
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=42)


# ============================================================
# STRATEGY FILTERS (identify qualifying events)
# ============================================================
print("\nIdentifying strategy-qualifying events...")

# S1: VWAP Sigma-2 Mean Reversion - all events qualify, we check at 1min level
s1_events = is_data.copy()
print(f"  S1 (VWAP Sigma-2 MR): {len(s1_events)} candidates (all IS, checked at 1min)")

# S2: No-Fill Power Move - gap not filled by 11:00, gap_size >= 2 ADR, rvol_60min >= 3
s2_events = is_data[
    (is_data['gap_filled'] == False) &
    (is_data['gap_size_in_adr'] >= 2.0) &
    (is_data['rvol_at_time_60min'] >= 3.0)
].copy()
print(f"  S2 (No-Fill Power): {len(s2_events)} qualifying events")

# S3: Late Session Momentum - entry 13:00, need close > VWAP and > open (for gap up)
# We'll use vwap_z_at_10am > 0 as proxy for "above VWAP trend held" for gap up
# Full check requires 13:00 data, so we use all events and filter at 1min level
s3_events = is_data.copy()
print(f"  S3 (Late Session): {len(s3_events)} candidates (filtered at 1min level)")

# S4: PM/RTH Ratio Extreme - pm_rth5 < 0.10, gap_size >= 1.5 ADR
s4_events = is_data[
    (is_data['pm_rth5'] < 0.10) &
    (is_data['gap_size_in_adr'] >= 1.5)
].copy()
print(f"  S4 (PM/RTH Extreme): {len(s4_events)} qualifying events")

# S5: ORB-Aligned Breakout - 15-min ORB aligned with gap direction
# Need 1-min data to compute ORB, so all events are candidates
s5_events = is_data.copy()
print(f"  S5 (ORB Breakout): {len(s5_events)} candidates (checked at 1min)")

# S6: VPOC Migration + Gap Continuation - entry 11:00 with VPOC migration
# vpoc_migration available in metadata
s6_events = is_data[is_data['vpoc_migration'].notna()].copy()
print(f"  S6 (VPOC Migration): {len(s6_events)} candidates")


# ============================================================
# ANALYSE 1: SLIPPAGE & LIQUIDITAET
# ============================================================
print("\n" + "="*60)
print("ANALYSE 1: Slippage & Liquiditaet")
print("="*60)

# For each strategy, we define the entry time and sample events
strategy_entry_times = {
    'S1': None,  # variable - when z_score hits +-2
    'S2': '11:00',
    'S3': '13:00',
    'S4': '09:35',
    'S5': None,  # variable - at ORB breakout
    'S6': '11:00',
}

strategy_events = {
    'S1': s1_events,
    'S2': s2_events,
    'S3': s3_events,
    'S4': s4_events,
    'S5': s5_events,
    'S6': s6_events,
}

slippage_results = {}

for strat_name, entry_time in strategy_entry_times.items():
    events = strategy_events[strat_name]
    sampled = sample_events(events, SAMPLE_SIZE)

    bar_ranges = []
    vol_ratios = []
    entry_volumes = []
    loaded = 0

    print(f"\n  {strat_name} (entry={entry_time or 'variable'}): sampling {len(sampled)} events...")

    for i, (idx, row) in enumerate(sampled.iterrows()):
        if (i+1) % 100 == 0:
            print(f"    ... processed {i+1}/{len(sampled)} events")

        ticker = row['ticker']
        date_str = row['date'].strftime('%Y-%m-%d')
        adr = row['adr_10']

        bars = load_1min(ticker, date_str)
        if bars is None:
            continue

        rth = get_rth_bars(bars)
        if rth is None or len(rth) < 30:
            continue

        loaded += 1

        # Determine entry bar index
        if strat_name == 'S1':
            # For S1, find first bar where VWAP z_score crosses +-2
            vwap_data = load_vwap(ticker, date_str)
            if vwap_data is None:
                continue
            sigma2_bars = vwap_data[(vwap_data['z_score'].abs() >= 2.0) &
                                    (vwap_data['time_et'] >= '09:35') &
                                    (vwap_data['time_et'] <= '15:30')]
            if len(sigma2_bars) == 0:
                continue
            entry_time_actual = sigma2_bars.iloc[0]['time_et']
        elif strat_name == 'S5':
            # For S5, find 15-min ORB breakout
            orb_bars = rth[rth['time_et'] < '09:45']
            if len(orb_bars) == 0:
                continue
            orb_high = orb_bars['high'].max()
            orb_low = orb_bars['low'].min()
            gap_dir = row['gap_direction']
            # Look for breakout after 09:45
            post_orb = rth[rth['time_et'] >= '09:45']
            if gap_dir == 'up':
                breakout_bars = post_orb[post_orb['high'] > orb_high]
            else:
                breakout_bars = post_orb[post_orb['low'] < orb_low]
            if len(breakout_bars) == 0:
                continue
            entry_time_actual = breakout_bars.iloc[0]['time_et']
        else:
            entry_time_actual = entry_time

        # Get entry bar and surrounding bars
        entry_idx = time_to_idx(rth, entry_time_actual)
        if entry_idx is None:
            continue

        # Position in rth
        pos = rth.index.get_loc(entry_idx)
        if isinstance(pos, slice):
            pos = pos.start

        # Get 3 bars around entry: entry-1, entry, entry+1
        for offset in [-1, 0, 1]:
            target_pos = pos + offset
            if 0 <= target_pos < len(rth):
                bar = rth.iloc[target_pos]
                br = bar_range_adr(bar, adr)
                if not np.isnan(br):
                    bar_ranges.append(br)

        # Volume at entry bar
        if 0 <= pos < len(rth):
            entry_bar = rth.iloc[pos]
            entry_vol = entry_bar['volume']
            entry_volumes.append(entry_vol)

            # Average volume across all RTH bars
            avg_vol = rth['volume'].mean()
            if avg_vol > 0:
                vol_ratios.append(entry_vol / avg_vol)

    slippage_results[strat_name] = {
        'n_loaded': loaded,
        'n_sampled': len(sampled),
        'bar_range_mean': np.nanmean(bar_ranges) if bar_ranges else np.nan,
        'bar_range_median': np.nanmedian(bar_ranges) if bar_ranges else np.nan,
        'bar_range_p90': np.nanpercentile(bar_ranges, 90) if bar_ranges else np.nan,
        'vol_ratio_mean': np.nanmean(vol_ratios) if vol_ratios else np.nan,
        'vol_ratio_median': np.nanmedian(vol_ratios) if vol_ratios else np.nan,
        'avg_entry_volume': np.nanmean(entry_volumes) if entry_volumes else np.nan,
        'n_bars_measured': len(bar_ranges),
        'n_vol_measured': len(vol_ratios),
    }

    print(f"    Loaded: {loaded}/{len(sampled)}")
    print(f"    Bar-Range/ADR: mean={slippage_results[strat_name]['bar_range_mean']:.4f}, "
          f"median={slippage_results[strat_name]['bar_range_median']:.4f}, "
          f"p90={slippage_results[strat_name]['bar_range_p90']:.4f}")
    print(f"    Volume Ratio (entry/avg): mean={slippage_results[strat_name]['vol_ratio_mean']:.2f}, "
          f"median={slippage_results[strat_name]['vol_ratio_median']:.2f}")


# ============================================================
# ANALYSE 2: ENTRY-QUALITAET
# ============================================================
print("\n" + "="*60)
print("ANALYSE 2: Entry-Qualitaet")
print("="*60)

# --- S1: VWAP Sigma-2 Mean Reversion Quality ---
print("\n  S1: VWAP Sigma-2 Mean Reversion - Bounce Quality")
s1_sampled = sample_events(s1_events, SAMPLE_SIZE)

s1_mfe_list = []  # max favorable excursion (30min)
s1_mae_list = []  # max adverse excursion (30min)
s1_bounce_count = 0
s1_total_touches = 0
s1_touch_times = []
s1_z_at_touch = []

for i, (idx, row) in enumerate(s1_sampled.iterrows()):
    if (i+1) % 100 == 0:
        print(f"    ... processed {i+1}/{len(s1_sampled)} events")

    ticker = row['ticker']
    date_str = row['date'].strftime('%Y-%m-%d')
    adr = row['adr_10']
    gap_dir = row['gap_direction']

    vwap_data = load_vwap(ticker, date_str)
    if vwap_data is None:
        continue

    bars = load_1min(ticker, date_str)
    if bars is None:
        continue
    rth = get_rth_bars(bars)
    if rth is None or len(rth) < 60:
        continue

    # Find all sigma-2 touches (first one per day only for simplicity)
    vwap_rth = vwap_data[(vwap_data['time_et'] >= '09:35') & (vwap_data['time_et'] <= '15:00')]
    sigma2_touches = vwap_rth[vwap_rth['z_score'].abs() >= 2.0]

    if len(sigma2_touches) == 0:
        continue

    # Take first touch
    first_touch = sigma2_touches.iloc[0]
    touch_time = first_touch['time_et']
    touch_z = first_touch['z_score']
    touch_price = first_touch['close']

    s1_total_touches += 1
    s1_touch_times.append(touch_time)
    s1_z_at_touch.append(touch_z)

    # Get 30 bars after touch
    touch_idx = time_to_idx(rth, touch_time)
    if touch_idx is None:
        continue

    pos = rth.index.get_loc(touch_idx)
    if isinstance(pos, slice):
        pos = pos.start

    future_bars = rth.iloc[pos:pos+30]
    if len(future_bars) < 5:
        continue

    # MFE and MAE depend on direction of expected reversion
    # If z > 2 (overextended up), we expect price to come DOWN -> short
    # If z < -2 (overextended down), we expect price to come UP -> long
    if touch_z > 0:
        # Short: favorable = price going down, adverse = price going up
        mfe = (touch_price - future_bars['low'].min()) / adr if adr > 0 else np.nan
        mae = (future_bars['high'].max() - touch_price) / adr if adr > 0 else np.nan
    else:
        # Long: favorable = price going up, adverse = price going down
        mfe = (future_bars['high'].max() - touch_price) / adr if adr > 0 else np.nan
        mae = (touch_price - future_bars['low'].min()) / adr if adr > 0 else np.nan

    if not np.isnan(mfe):
        s1_mfe_list.append(mfe)
        s1_mae_list.append(mae)
        if mfe > mae:
            s1_bounce_count += 1

s1_quality = {
    'total_touches': s1_total_touches,
    'bounce_rate': s1_bounce_count / max(s1_total_touches, 1),
    'mfe_mean': np.nanmean(s1_mfe_list) if s1_mfe_list else np.nan,
    'mfe_median': np.nanmedian(s1_mfe_list) if s1_mfe_list else np.nan,
    'mae_mean': np.nanmean(s1_mae_list) if s1_mae_list else np.nan,
    'mae_median': np.nanmedian(s1_mae_list) if s1_mae_list else np.nan,
    'reward_risk': (np.nanmean(s1_mfe_list) / np.nanmean(s1_mae_list)) if s1_mae_list and np.nanmean(s1_mae_list) > 0 else np.nan,
    'avg_touch_z': np.nanmean(s1_z_at_touch) if s1_z_at_touch else np.nan,
}

print(f"    Sigma-2 Touches found: {s1_total_touches}")
print(f"    Bounce Rate (MFE>MAE in 30min): {s1_quality['bounce_rate']:.1%}")
print(f"    MFE (30min): mean={s1_quality['mfe_mean']:.3f} ADR, median={s1_quality['mfe_median']:.3f} ADR")
print(f"    MAE (30min): mean={s1_quality['mae_mean']:.3f} ADR, median={s1_quality['mae_median']:.3f} ADR")
print(f"    Reward/Risk: {s1_quality['reward_risk']:.2f}")


# --- S2: No-Fill Power Move - Price Action at 11:00 ---
print("\n  S2: No-Fill Power Move - Price Action at 11:00")

s2_trend_strength = []
s2_exhaustion_count = 0
s2_continuation_count = 0
s2_post_1100_drift = []

for i, (idx, row) in enumerate(s2_events.iterrows()):
    adr = row['adr_10']
    gap_dir = row['gap_direction']
    c1000 = row['close_1000']
    c1100 = row['close_1100']
    rth_open = row['rth_open']

    if pd.isna(c1000) or pd.isna(c1100) or adr == 0:
        continue

    # Trend strength: how much did price move from 10:00 to 11:00
    trend = (c1100 - c1000) / adr
    s2_trend_strength.append(trend)

    # Is trend aligned with gap?
    if gap_dir == 'up':
        if trend > 0:
            s2_continuation_count += 1
        elif trend < -0.05:
            s2_exhaustion_count += 1
    else:
        if trend < 0:
            s2_continuation_count += 1
        elif trend > 0.05:
            s2_exhaustion_count += 1

    # Post-11:00 drift (close vs 11:00 price)
    rth_close = row['rth_close']
    if not pd.isna(rth_close):
        if gap_dir == 'up':
            post_drift = (rth_close - c1100) / adr
        else:
            post_drift = (c1100 - rth_close) / adr
        s2_post_1100_drift.append(post_drift)

s2_quality = {
    'n_events': len(s2_events),
    'trend_strength_mean': np.nanmean(s2_trend_strength) if s2_trend_strength else np.nan,
    'trend_strength_median': np.nanmedian(s2_trend_strength) if s2_trend_strength else np.nan,
    'continuation_rate': s2_continuation_count / max(len(s2_trend_strength), 1),
    'exhaustion_rate': s2_exhaustion_count / max(len(s2_trend_strength), 1),
    'post_1100_drift_mean': np.nanmean(s2_post_1100_drift) if s2_post_1100_drift else np.nan,
    'post_1100_drift_median': np.nanmedian(s2_post_1100_drift) if s2_post_1100_drift else np.nan,
    'post_1100_drift_winrate': np.mean([d > 0 for d in s2_post_1100_drift]) if s2_post_1100_drift else np.nan,
}

print(f"    Events: {s2_quality['n_events']}")
print(f"    Trend strength (10:00-11:00): mean={s2_quality['trend_strength_mean']:.3f}, median={s2_quality['trend_strength_median']:.3f} ADR")
print(f"    Continuation rate: {s2_quality['continuation_rate']:.1%}")
print(f"    Exhaustion rate: {s2_quality['exhaustion_rate']:.1%}")
print(f"    Post-11:00 drift (gap dir): mean={s2_quality['post_1100_drift_mean']:.3f} ADR, win={s2_quality['post_1100_drift_winrate']:.1%}")


# --- S3: Late Session Momentum - Afternoon Pattern ---
print("\n  S3: Late Session Momentum - Afternoon Pattern")

s3_sampled = sample_events(s3_events, SAMPLE_SIZE)
s3_pm_drift = []
s3_pm_positive_count = 0
s3_total_measured = 0
s3_vwap_aligned_pm_drift = []
s3_vwap_nonaligned_pm_drift = []

for i, (idx, row) in enumerate(s3_sampled.iterrows()):
    if (i+1) % 100 == 0:
        print(f"    ... processed {i+1}/{len(s3_sampled)} events")

    ticker = row['ticker']
    date_str = row['date'].strftime('%Y-%m-%d')
    adr = row['adr_10']
    gap_dir = row['gap_direction']
    rth_open = row['rth_open']

    bars = load_1min(ticker, date_str)
    if bars is None:
        continue
    rth = get_rth_bars(bars)
    if rth is None or len(rth) < 300:
        continue

    # Get close at 13:00 and 15:00
    bar_1300 = rth[rth['time_et'] == '13:00']
    bar_1500 = rth[rth['time_et'] == '15:00']

    if len(bar_1300) == 0 or len(bar_1500) == 0:
        continue

    close_1300 = bar_1300.iloc[0]['close']
    close_1500 = bar_1500.iloc[0]['close']

    # Also get VWAP at 13:00
    vwap_data = load_vwap(ticker, date_str)
    if vwap_data is None:
        continue
    vwap_1300 = vwap_data[vwap_data['time_et'] == '13:00']
    if len(vwap_1300) == 0:
        continue

    vwap_val = vwap_1300.iloc[0]['vwap']

    # Check if aligned: for gap up, close_1300 > vwap and > rth_open
    if gap_dir == 'up':
        aligned = (close_1300 > vwap_val) and (close_1300 > rth_open)
        pm_drift = (close_1500 - close_1300) / adr if adr > 0 else np.nan
    else:
        aligned = (close_1300 < vwap_val) and (close_1300 < rth_open)
        pm_drift = (close_1300 - close_1500) / adr if adr > 0 else np.nan

    if not np.isnan(pm_drift):
        s3_pm_drift.append(pm_drift)
        s3_total_measured += 1
        if pm_drift > 0:
            s3_pm_positive_count += 1

        if aligned:
            s3_vwap_aligned_pm_drift.append(pm_drift)
        else:
            s3_vwap_nonaligned_pm_drift.append(pm_drift)

s3_quality = {
    'n_measured': s3_total_measured,
    'pm_drift_mean': np.nanmean(s3_pm_drift) if s3_pm_drift else np.nan,
    'pm_drift_median': np.nanmedian(s3_pm_drift) if s3_pm_drift else np.nan,
    'pm_positive_rate': s3_pm_positive_count / max(s3_total_measured, 1),
    'aligned_drift_mean': np.nanmean(s3_vwap_aligned_pm_drift) if s3_vwap_aligned_pm_drift else np.nan,
    'aligned_drift_median': np.nanmedian(s3_vwap_aligned_pm_drift) if s3_vwap_aligned_pm_drift else np.nan,
    'aligned_count': len(s3_vwap_aligned_pm_drift),
    'aligned_winrate': np.mean([d > 0 for d in s3_vwap_aligned_pm_drift]) if s3_vwap_aligned_pm_drift else np.nan,
    'nonaligned_drift_mean': np.nanmean(s3_vwap_nonaligned_pm_drift) if s3_vwap_nonaligned_pm_drift else np.nan,
    'nonaligned_count': len(s3_vwap_nonaligned_pm_drift),
}

print(f"    Measured: {s3_quality['n_measured']}")
print(f"    PM drift (13:00-15:00 in gap dir): mean={s3_quality['pm_drift_mean']:.3f}, median={s3_quality['pm_drift_median']:.3f} ADR")
print(f"    Positive rate: {s3_quality['pm_positive_rate']:.1%}")
print(f"    VWAP-aligned: n={s3_quality['aligned_count']}, drift={s3_quality['aligned_drift_mean']:.3f}, win={s3_quality['aligned_winrate']:.1%}")
print(f"    Non-aligned: n={s3_quality['nonaligned_count']}, drift={s3_quality['nonaligned_drift_mean']:.3f}")


# --- S4: PM/RTH Ratio - Additional Entry Quality ---
print("\n  S4: PM/RTH Ratio Extreme - Entry Quality at 9:35")

s4_first5_drift = []
s4_first15_drift = []

s4_sampled = sample_events(s4_events, SAMPLE_SIZE)

for i, (idx, row) in enumerate(s4_sampled.iterrows()):
    if (i+1) % 100 == 0:
        print(f"    ... processed {i+1}/{len(s4_sampled)} events")

    ticker = row['ticker']
    date_str = row['date'].strftime('%Y-%m-%d')
    adr = row['adr_10']
    gap_dir = row['gap_direction']
    rth_open = row['rth_open']

    bars = load_1min(ticker, date_str)
    if bars is None:
        continue
    rth = get_rth_bars(bars)
    if rth is None or len(rth) < 30:
        continue

    # Price at 9:35 and 9:45
    bar_935 = rth[rth['time_et'] == '09:35']
    bar_945 = rth[rth['time_et'] == '09:45']

    if len(bar_935) == 0:
        continue

    close_935 = bar_935.iloc[0]['close']

    # 5-min drift from open
    drift_5 = (close_935 - rth_open) / adr if adr > 0 else np.nan
    if gap_dir == 'down':
        drift_5 = -drift_5
    if not np.isnan(drift_5):
        s4_first5_drift.append(drift_5)

    if len(bar_945) > 0:
        close_945 = bar_945.iloc[0]['close']
        drift_15 = (close_945 - rth_open) / adr if adr > 0 else np.nan
        if gap_dir == 'down':
            drift_15 = -drift_15
        if not np.isnan(drift_15):
            s4_first15_drift.append(drift_15)

s4_quality = {
    'n_events': len(s4_events),
    'n_sampled': len(s4_sampled),
    'drift_5min_mean': np.nanmean(s4_first5_drift) if s4_first5_drift else np.nan,
    'drift_5min_median': np.nanmedian(s4_first5_drift) if s4_first5_drift else np.nan,
    'drift_15min_mean': np.nanmean(s4_first15_drift) if s4_first15_drift else np.nan,
    'drift_15min_median': np.nanmedian(s4_first15_drift) if s4_first15_drift else np.nan,
    'drift_5min_positive': np.mean([d > 0 for d in s4_first5_drift]) if s4_first5_drift else np.nan,
}

print(f"    Events: {s4_quality['n_events']}, sampled: {s4_quality['n_sampled']}")
print(f"    First 5min drift (gap dir): mean={s4_quality['drift_5min_mean']:.3f}, median={s4_quality['drift_5min_median']:.3f} ADR")
print(f"    First 15min drift (gap dir): mean={s4_quality['drift_15min_mean']:.3f}, median={s4_quality['drift_15min_median']:.3f} ADR")
print(f"    5min positive rate: {s4_quality['drift_5min_positive']:.1%}")


# --- S5: ORB Breakout - Additional Quality ---
print("\n  S5: ORB Breakout - Breakout Quality")

s5_sampled = sample_events(s5_events, SAMPLE_SIZE)
s5_breakout_found = 0
s5_breakout_hold = 0
s5_mfe_list = []
s5_mae_list = []

for i, (idx, row) in enumerate(s5_sampled.iterrows()):
    if (i+1) % 100 == 0:
        print(f"    ... processed {i+1}/{len(s5_sampled)} events")

    ticker = row['ticker']
    date_str = row['date'].strftime('%Y-%m-%d')
    adr = row['adr_10']
    gap_dir = row['gap_direction']

    bars = load_1min(ticker, date_str)
    if bars is None:
        continue
    rth = get_rth_bars(bars)
    if rth is None or len(rth) < 60:
        continue

    # Compute 15-min ORB
    orb_bars = rth[(rth['time_et'] >= '09:30') & (rth['time_et'] < '09:45')]
    if len(orb_bars) == 0:
        continue

    orb_high = orb_bars['high'].max()
    orb_low = orb_bars['low'].min()

    # Look for aligned breakout
    post_orb = rth[rth['time_et'] >= '09:45']
    if gap_dir == 'up':
        breakout_bars = post_orb[post_orb['close'] > orb_high]
    else:
        breakout_bars = post_orb[post_orb['close'] < orb_low]

    if len(breakout_bars) == 0:
        continue

    s5_breakout_found += 1
    bo_time = breakout_bars.iloc[0]['time_et']
    bo_price = breakout_bars.iloc[0]['close']

    # MFE/MAE in next 30 bars after breakout
    bo_idx = breakout_bars.index[0]
    pos = rth.index.get_loc(bo_idx)
    if isinstance(pos, slice):
        pos = pos.start

    future = rth.iloc[pos:pos+30]
    if len(future) < 5:
        continue

    if gap_dir == 'up':
        mfe = (future['high'].max() - bo_price) / adr if adr > 0 else np.nan
        mae = (bo_price - future['low'].min()) / adr if adr > 0 else np.nan
    else:
        mfe = (bo_price - future['low'].min()) / adr if adr > 0 else np.nan
        mae = (future['high'].max() - bo_price) / adr if adr > 0 else np.nan

    if not np.isnan(mfe):
        s5_mfe_list.append(mfe)
        s5_mae_list.append(mae)
        if mfe > mae:
            s5_breakout_hold += 1

s5_quality = {
    'n_sampled': len(s5_sampled),
    'breakout_found': s5_breakout_found,
    'breakout_rate': s5_breakout_found / max(len(s5_sampled), 1),
    'hold_rate': s5_breakout_hold / max(s5_breakout_found, 1),
    'mfe_mean': np.nanmean(s5_mfe_list) if s5_mfe_list else np.nan,
    'mfe_median': np.nanmedian(s5_mfe_list) if s5_mfe_list else np.nan,
    'mae_mean': np.nanmean(s5_mae_list) if s5_mae_list else np.nan,
    'mae_median': np.nanmedian(s5_mae_list) if s5_mae_list else np.nan,
    'reward_risk': (np.nanmean(s5_mfe_list) / np.nanmean(s5_mae_list)) if s5_mae_list and np.nanmean(s5_mae_list) > 0 else np.nan,
}

print(f"    Sampled: {s5_quality['n_sampled']}")
print(f"    Aligned breakout found: {s5_quality['breakout_found']} ({s5_quality['breakout_rate']:.1%})")
print(f"    Breakout held (MFE>MAE): {s5_quality['hold_rate']:.1%}")
print(f"    MFE (30min): mean={s5_quality['mfe_mean']:.3f}, median={s5_quality['mfe_median']:.3f} ADR")
print(f"    MAE (30min): mean={s5_quality['mae_mean']:.3f}, median={s5_quality['mae_median']:.3f} ADR")


# --- S6: VPOC Migration - Quality ---
print("\n  S6: VPOC Migration + Gap Continuation")

s6_sampled = sample_events(s6_events, SAMPLE_SIZE)
s6_strong_migration = []
s6_post_drift = []
s6_migration_aligned = 0
s6_total_checked = 0

for i, (idx, row) in enumerate(s6_sampled.iterrows()):
    if (i+1) % 100 == 0:
        print(f"    ... processed {i+1}/{len(s6_sampled)} events")

    ticker = row['ticker']
    date_str = row['date'].strftime('%Y-%m-%d')
    adr = row['adr_10']
    gap_dir = row['gap_direction']
    vpoc_mig = row['vpoc_migration']
    rth_open = row['rth_open']
    c1100 = row['close_1100']
    rth_close = row['rth_close']

    if pd.isna(vpoc_mig) or adr == 0:
        continue

    s6_total_checked += 1

    # Load VP data for migration details
    vp_data = load_vp(ticker, date_str)
    if vp_data is not None and len(vp_data) >= 2:
        # VPOC at 30min vs VPOC at 90min
        vp_30 = vp_data[vp_data['minutes_since_open'] == 30]
        vp_90 = vp_data[vp_data['minutes_since_open'] == 90]

        if len(vp_30) > 0 and len(vp_90) > 0:
            vpoc_30 = vp_30.iloc[0]['vpoc']
            vpoc_90 = vp_90.iloc[0]['vpoc']
            migration = (vpoc_90 - vpoc_30) / adr
            s6_strong_migration.append(migration)

            # Check if migration aligned with gap
            if (gap_dir == 'up' and migration > 0.05) or (gap_dir == 'down' and migration < -0.05):
                s6_migration_aligned += 1

    # Post-11:00 drift
    if not pd.isna(c1100) and not pd.isna(rth_close):
        if gap_dir == 'up':
            post_drift = (rth_close - c1100) / adr
        else:
            post_drift = (c1100 - rth_close) / adr
        s6_post_drift.append(post_drift)

s6_quality = {
    'n_checked': s6_total_checked,
    'migration_aligned_rate': s6_migration_aligned / max(s6_total_checked, 1),
    'migration_mean': np.nanmean(s6_strong_migration) if s6_strong_migration else np.nan,
    'migration_median': np.nanmedian(s6_strong_migration) if s6_strong_migration else np.nan,
    'post_drift_mean': np.nanmean(s6_post_drift) if s6_post_drift else np.nan,
    'post_drift_median': np.nanmedian(s6_post_drift) if s6_post_drift else np.nan,
    'post_drift_winrate': np.mean([d > 0 for d in s6_post_drift]) if s6_post_drift else np.nan,
}

print(f"    Checked: {s6_quality['n_checked']}")
print(f"    VPOC migration aligned with gap: {s6_quality['migration_aligned_rate']:.1%}")
print(f"    Migration magnitude: mean={s6_quality['migration_mean']:.3f}, median={s6_quality['migration_median']:.3f} ADR")
print(f"    Post-11:00 drift (gap dir): mean={s6_quality['post_drift_mean']:.3f}, win={s6_quality['post_drift_winrate']:.1%}")


# ============================================================
# ANALYSE 3: FILTER-VERFEINERUNG VORSCHLAEGE
# ============================================================
print("\n" + "="*60)
print("ANALYSE 3: Filter-Verfeinerung")
print("="*60)

# S1: Check if volume confirmation helps
print("\n  S1: Volume confirmation at Sigma-2 touch")
s1_high_vol_mfe = []
s1_low_vol_mfe = []
s1_high_vol_mae = []
s1_low_vol_mae = []

s1_filter_sampled = sample_events(s1_events, SAMPLE_SIZE)
for i, (idx, row) in enumerate(s1_filter_sampled.iterrows()):
    if (i+1) % 100 == 0:
        print(f"    ... processed {i+1}/{len(s1_filter_sampled)} events")

    ticker = row['ticker']
    date_str = row['date'].strftime('%Y-%m-%d')
    adr = row['adr_10']
    rvol = row['rvol_at_time_60min']

    vwap_data = load_vwap(ticker, date_str)
    if vwap_data is None:
        continue

    bars = load_1min(ticker, date_str)
    if bars is None:
        continue
    rth = get_rth_bars(bars)
    if rth is None or len(rth) < 60:
        continue

    vwap_rth = vwap_data[(vwap_data['time_et'] >= '09:35') & (vwap_data['time_et'] <= '15:00')]
    sigma2_touches = vwap_rth[vwap_rth['z_score'].abs() >= 2.0]

    if len(sigma2_touches) == 0:
        continue

    first_touch = sigma2_touches.iloc[0]
    touch_time = first_touch['time_et']
    touch_z = first_touch['z_score']
    touch_price = first_touch['close']

    touch_idx = time_to_idx(rth, touch_time)
    if touch_idx is None:
        continue
    pos = rth.index.get_loc(touch_idx)
    if isinstance(pos, slice):
        pos = pos.start

    future = rth.iloc[pos:pos+30]
    if len(future) < 5:
        continue

    if touch_z > 0:
        mfe = (touch_price - future['low'].min()) / adr if adr > 0 else np.nan
        mae = (future['high'].max() - touch_price) / adr if adr > 0 else np.nan
    else:
        mfe = (future['high'].max() - touch_price) / adr if adr > 0 else np.nan
        mae = (touch_price - future['low'].min()) / adr if adr > 0 else np.nan

    if np.isnan(mfe) or pd.isna(rvol):
        continue

    # Split by rvol
    if rvol >= 2.0:
        s1_high_vol_mfe.append(mfe)
        s1_high_vol_mae.append(mae)
    else:
        s1_low_vol_mfe.append(mfe)
        s1_low_vol_mae.append(mae)

print(f"    High RVOL (>=2): n={len(s1_high_vol_mfe)}, MFE={np.nanmean(s1_high_vol_mfe):.3f}, MAE={np.nanmean(s1_high_vol_mae):.3f}")
print(f"    Low RVOL (<2): n={len(s1_low_vol_mfe)}, MFE={np.nanmean(s1_low_vol_mfe):.3f}, MAE={np.nanmean(s1_low_vol_mae):.3f}")

# S2: Check if gap_size threshold matters
print("\n  S2: Gap size sub-filter")
s2_big_gap = is_data[(is_data['gap_filled'] == False) & (is_data['gap_size_in_adr'] >= 3.0) & (is_data['rvol_at_time_60min'] >= 3.0)]
s2_med_gap = is_data[(is_data['gap_filled'] == False) & (is_data['gap_size_in_adr'] >= 2.0) & (is_data['gap_size_in_adr'] < 3.0) & (is_data['rvol_at_time_60min'] >= 3.0)]

def s2_continuation_metric(df):
    """Compute continuation metrics for S2 sub-groups."""
    drifts = []
    for _, r in df.iterrows():
        adr = r['adr_10']
        gd = r['gap_direction']
        c1100 = r['close_1100']
        rc = r['rth_close']
        if pd.isna(c1100) or pd.isna(rc) or adr == 0:
            continue
        if gd == 'up':
            d = (rc - c1100) / adr
        else:
            d = (c1100 - rc) / adr
        drifts.append(d)
    return drifts

s2_big_drifts = s2_continuation_metric(s2_big_gap)
s2_med_drifts = s2_continuation_metric(s2_med_gap)

print(f"    Big gap (>=3 ADR): n={len(s2_big_gap)}, post-11 drift={np.nanmean(s2_big_drifts):.3f}, win={np.mean([d>0 for d in s2_big_drifts]):.1%}" if s2_big_drifts else "    Big gap (>=3 ADR): insufficient data")
print(f"    Med gap (2-3 ADR): n={len(s2_med_gap)}, post-11 drift={np.nanmean(s2_med_drifts):.3f}, win={np.mean([d>0 for d in s2_med_drifts]):.1%}" if s2_med_drifts else "    Med gap (2-3 ADR): insufficient data")

# S3: Time-of-day filter
print("\n  S3: Afternoon drift by gap size bucket")
for gsize_min, gsize_max, label in [(1.0, 2.0, '1-2 ADR'), (2.0, 3.0, '2-3 ADR'), (3.0, 999, '3+ ADR')]:
    sub = is_data[(is_data['gap_size_in_adr'] >= gsize_min) & (is_data['gap_size_in_adr'] < gsize_max)]
    sub_drifts = []
    for _, r in sub.iterrows():
        c1100 = r['close_1100']
        rc = r['rth_close']
        adr = r['adr_10']
        gd = r['gap_direction']
        if pd.isna(c1100) or pd.isna(rc) or adr == 0:
            continue
        if gd == 'up':
            d = (rc - c1100) / adr
        else:
            d = (c1100 - rc) / adr
        sub_drifts.append(d)
    if sub_drifts:
        print(f"    {label}: n={len(sub)}, afternoon drift={np.nanmean(sub_drifts):.3f}, win={np.mean([d>0 for d in sub_drifts]):.1%}")

# S5: ORB with OD alignment
print("\n  S5: ORB with OD-strength filter")
s5_strong_od = is_data[is_data['od_strength'].abs() >= 0.5]
s5_weak_od = is_data[is_data['od_strength'].abs() < 0.5]
print(f"    Strong OD: {len(s5_strong_od)} events")
print(f"    Weak OD: {len(s5_weak_od)} events")

# S6: VPOC migration threshold
print("\n  S6: VPOC migration thresholds")
for mig_thresh in [1, 2, 3, 4]:
    sub = is_data[is_data['vpoc_migration'] >= mig_thresh]
    drifts = []
    for _, r in sub.iterrows():
        c1100 = r['close_1100']
        rc = r['rth_close']
        adr = r['adr_10']
        gd = r['gap_direction']
        if pd.isna(c1100) or pd.isna(rc) or adr == 0:
            continue
        if gd == 'up':
            d = (rc - c1100) / adr
        else:
            d = (c1100 - rc) / adr
        drifts.append(d)
    if drifts:
        print(f"    VPOC migration >= {mig_thresh}: n={len(sub)}, drift={np.nanmean(drifts):.3f}, win={np.mean([d>0 for d in drifts]):.1%}")


# ============================================================
# ANALYSE 4: OD-UNABHAENGIGKEIT (Korrelation mit od_strength)
# ============================================================
print("\n" + "="*60)
print("ANALYSE 4: OD-Unabhaengigkeit")
print("="*60)

od_corr_results = {}

# S1: VWAP MR - use full_drift as PnL proxy
s1_corr_data = is_data[['od_strength', 'full_drift', 'vwap_z_at_10am']].dropna()
if len(s1_corr_data) > 30:
    corr = s1_corr_data['od_strength'].corr(s1_corr_data['full_drift'])
    od_corr_results['S1'] = corr
    print(f"  S1 (VWAP MR): corr(od_strength, full_drift) = {corr:.3f}")

# S2: No-Fill Power - use rest_drift_1100 as PnL proxy (post-11:00)
s2_corr_data = is_data[['od_strength', 'rest_drift_1100']].dropna()
s2_corr_data = s2_corr_data.merge(
    is_data[['od_strength', 'rest_drift_1100', 'gap_filled', 'gap_size_in_adr', 'rvol_at_time_60min']],
    left_index=True, right_index=True, suffixes=('', '_dup')
)
# Filter to S2-qualifying
s2_corr_filtered = is_data[
    (is_data['gap_filled'] == False) &
    (is_data['gap_size_in_adr'] >= 2.0) &
    (is_data['rvol_at_time_60min'] >= 3.0)
][['od_strength', 'rest_drift_1100']].dropna()
if len(s2_corr_filtered) > 10:
    corr = s2_corr_filtered['od_strength'].corr(s2_corr_filtered['rest_drift_1100'])
    od_corr_results['S2'] = corr
    print(f"  S2 (No-Fill): corr(od_strength, rest_drift_1100) = {corr:.3f}")

# S3: Late Session - use rest_drift as PnL proxy
s3_corr_data = is_data[['od_strength', 'rest_drift']].dropna()
if len(s3_corr_data) > 30:
    corr = s3_corr_data['od_strength'].corr(s3_corr_data['rest_drift'])
    od_corr_results['S3'] = corr
    print(f"  S3 (Late Session): corr(od_strength, rest_drift) = {corr:.3f}")

# S4: PM/RTH - use full_drift as PnL proxy
s4_corr_filtered = is_data[
    (is_data['pm_rth5'] < 0.10) &
    (is_data['gap_size_in_adr'] >= 1.5)
][['od_strength', 'full_drift']].dropna()
if len(s4_corr_filtered) > 10:
    corr = s4_corr_filtered['od_strength'].corr(s4_corr_filtered['full_drift'])
    od_corr_results['S4'] = corr
    print(f"  S4 (PM/RTH): corr(od_strength, full_drift) = {corr:.3f}")

# S5: ORB Breakout - use full_drift as PnL proxy
s5_corr_data = is_data[['od_strength', 'full_drift']].dropna()
if len(s5_corr_data) > 30:
    corr = s5_corr_data['od_strength'].corr(s5_corr_data['full_drift'])
    od_corr_results['S5'] = corr
    print(f"  S5 (ORB): corr(od_strength, full_drift) = {corr:.3f}")

# S6: VPOC Migration - use rest_drift_1100 as PnL proxy
s6_corr_filtered = is_data[is_data['vpoc_migration'].notna()][['od_strength', 'rest_drift_1100']].dropna()
if len(s6_corr_filtered) > 30:
    corr = s6_corr_filtered['od_strength'].corr(s6_corr_filtered['rest_drift_1100'])
    od_corr_results['S6'] = corr
    print(f"  S6 (VPOC): corr(od_strength, rest_drift_1100) = {corr:.3f}")

# OD independence check
print("\n  OD-Unabhaengigkeit Check (|corr| > 0.3 = NICHT unabhaengig):")
for strat, corr in od_corr_results.items():
    status = "NICHT UNABHAENGIG" if abs(corr) > 0.3 else "UNABHAENGIG"
    print(f"    {strat}: |corr|={abs(corr):.3f} -> {status}")


# ============================================================
# WRITE RESULTS FILE
# ============================================================
print("\n" + "="*60)
print("Writing results file...")
print("="*60)

lines = []
lines.append("=" * 72)
lines.append("D13 PHASE 2: TRADER MICROSTRUCTURE & UMSETZBARKEITS-ANALYSE")
lines.append("=" * 72)
lines.append(f"Erstellt: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
lines.append(f"IS-Zeitraum: {IS_START} bis {IS_END}")
lines.append(f"IS-Events gesamt: {len(is_data)}")
lines.append(f"1-Min Sample Size: {SAMPLE_SIZE}")
lines.append("")

# --- ANALYSE 1 ---
lines.append("-" * 72)
lines.append("ANALYSE 1: SLIPPAGE & LIQUIDITAET")
lines.append("-" * 72)
lines.append("")
lines.append("Methode: Fuer jede Strategie wurden die 3 Bars um den Entry-Zeitpunkt")
lines.append("gemessen. Bar-Range/ADR gibt an, wie gross die typische Slippage sein")
lines.append("koennte. Volume-Ratio zeigt, ob am Entry genug Liquiditaet vorhanden ist.")
lines.append("")

for strat, res in slippage_results.items():
    entry_t = strategy_entry_times[strat] or 'variabel'
    lines.append(f"  {strat} (Entry: {entry_t}):")
    lines.append(f"    Events geladen: {res['n_loaded']}/{res['n_sampled']}")
    lines.append(f"    Bar-Range/ADR:  mean={res['bar_range_mean']:.4f}  median={res['bar_range_median']:.4f}  p90={res['bar_range_p90']:.4f}")
    lines.append(f"    Volume-Ratio:   mean={res['vol_ratio_mean']:.2f}  median={res['vol_ratio_median']:.2f}")
    lines.append(f"    Avg Entry Vol:  {res['avg_entry_volume']:,.0f}" if not np.isnan(res['avg_entry_volume']) else "    Avg Entry Vol:  N/A")

    # Interpretation
    if res['bar_range_mean'] > 0.05:
        lines.append(f"    -> WARNUNG: Hohe Bar-Range ({res['bar_range_mean']:.1%} ADR) = erhebliches Slippage-Risiko")
    elif res['bar_range_mean'] > 0.03:
        lines.append(f"    -> MODERAT: Bar-Range ({res['bar_range_mean']:.1%} ADR) = akzeptables Slippage")
    else:
        lines.append(f"    -> GUT: Niedrige Bar-Range ({res['bar_range_mean']:.1%} ADR) = geringes Slippage")

    if res['vol_ratio_mean'] < 0.5:
        lines.append(f"    -> WARNUNG: Duenne Liquiditaet am Entry (nur {res['vol_ratio_mean']:.0%} des Tagesdurchschnitts)")
    elif res['vol_ratio_mean'] > 2.0:
        lines.append(f"    -> SEHR GUT: Hohe Liquiditaet am Entry ({res['vol_ratio_mean']:.1f}x Durchschnitt)")
    else:
        lines.append(f"    -> OK: Ausreichende Liquiditaet ({res['vol_ratio_mean']:.1f}x Durchschnitt)")
    lines.append("")

# --- ANALYSE 2 ---
lines.append("-" * 72)
lines.append("ANALYSE 2: ENTRY-QUALITAET")
lines.append("-" * 72)
lines.append("")

# S1
lines.append("  S1: VWAP Sigma-2 Mean Reversion")
lines.append(f"    Sigma-2 Touches gefunden: {s1_quality['total_touches']}")
lines.append(f"    Bounce Rate (MFE > MAE in 30min): {s1_quality['bounce_rate']:.1%}")
lines.append(f"    Max Favorable Excursion (30min): mean={s1_quality['mfe_mean']:.3f} ADR, median={s1_quality['mfe_median']:.3f} ADR")
lines.append(f"    Max Adverse Excursion (30min):   mean={s1_quality['mae_mean']:.3f} ADR, median={s1_quality['mae_median']:.3f} ADR")
lines.append(f"    Reward/Risk Ratio: {s1_quality['reward_risk']:.2f}")
if s1_quality['bounce_rate'] >= 0.55:
    lines.append(f"    -> POSITIV: Sigma-2 bietet statistisch eine Edge ({s1_quality['bounce_rate']:.0%} Bounce)")
elif s1_quality['bounce_rate'] >= 0.48:
    lines.append(f"    -> NEUTRAL: Bounce-Rate nahe 50% - Edge unklar")
else:
    lines.append(f"    -> NEGATIV: Sigma-2 Touches halten oft nicht ({s1_quality['bounce_rate']:.0%} Bounce)")
lines.append("")

# S2
lines.append("  S2: No-Fill Power Move (11:00 Entry)")
lines.append(f"    Qualifying Events: {s2_quality['n_events']}")
lines.append(f"    Trend Staerke (10:00-11:00): mean={s2_quality['trend_strength_mean']:.3f}, median={s2_quality['trend_strength_median']:.3f} ADR")
lines.append(f"    Continuation Rate (Trend in Gap-Dir): {s2_quality['continuation_rate']:.1%}")
lines.append(f"    Exhaustion Rate (Gegenrichtung): {s2_quality['exhaustion_rate']:.1%}")
lines.append(f"    Post-11:00 Drift (Gap-Dir): mean={s2_quality['post_1100_drift_mean']:.3f} ADR")
lines.append(f"    Post-11:00 Win-Rate: {s2_quality['post_1100_drift_winrate']:.1%}")
if s2_quality['post_1100_drift_mean'] > 0.05 and s2_quality['post_1100_drift_winrate'] > 0.55:
    lines.append(f"    -> POSITIV: Power-Moves setzen sich nach 11:00 fort")
elif s2_quality['post_1100_drift_mean'] < -0.05:
    lines.append(f"    -> NEGATIV: Trend erschoepft sich typischerweise vor 11:00")
else:
    lines.append(f"    -> NEUTRAL: Schwacher Drift nach 11:00")
lines.append("")

# S3
lines.append("  S3: Late Session Momentum (13:00 Entry)")
lines.append(f"    Events gemessen: {s3_quality['n_measured']}")
lines.append(f"    PM Drift (13:00-15:00, Gap-Dir): mean={s3_quality['pm_drift_mean']:.3f}, median={s3_quality['pm_drift_median']:.3f} ADR")
lines.append(f"    Positive Rate: {s3_quality['pm_positive_rate']:.1%}")
lines.append(f"    VWAP-Aligned Events: {s3_quality['aligned_count']}")
lines.append(f"      Aligned Drift: mean={s3_quality['aligned_drift_mean']:.3f}, median={s3_quality['aligned_drift_median']:.3f} ADR")
lines.append(f"      Aligned Win-Rate: {s3_quality['aligned_winrate']:.1%}")
lines.append(f"    Non-Aligned Events: {s3_quality['nonaligned_count']}")
lines.append(f"      Non-Aligned Drift: mean={s3_quality['nonaligned_drift_mean']:.3f} ADR")
if s3_quality['aligned_winrate'] > 0.55 and s3_quality['aligned_drift_mean'] > 0.02:
    lines.append(f"    -> POSITIV: VWAP-Alignment verbessert nachmittags-Drift signifikant")
elif s3_quality['pm_drift_mean'] < 0:
    lines.append(f"    -> NEGATIV: Nachmittags typischerweise Gegenbewegung")
else:
    lines.append(f"    -> NEUTRAL: Schwacher Nachmittags-Drift")
lines.append("")

# S4
lines.append("  S4: PM/RTH Ratio Extreme (9:35 Entry)")
lines.append(f"    Qualifying Events: {s4_quality['n_events']}")
lines.append(f"    First 5min Drift (Gap-Dir): mean={s4_quality['drift_5min_mean']:.3f}, median={s4_quality['drift_5min_median']:.3f} ADR")
lines.append(f"    First 15min Drift (Gap-Dir): mean={s4_quality['drift_15min_mean']:.3f}, median={s4_quality['drift_15min_median']:.3f} ADR")
lines.append(f"    5min Positive Rate: {s4_quality['drift_5min_positive']:.1%}")
if s4_quality['drift_5min_mean'] > 0.05:
    lines.append(f"    -> POSITIV: Starker initialer Drift in Gap-Richtung")
elif s4_quality['drift_5min_mean'] < -0.05:
    lines.append(f"    -> NEGATIV: Sofortige Gegenbewegung - Entry zu frueh")
else:
    lines.append(f"    -> NEUTRAL: Kein klarer initialer Vorteil")
lines.append("")

# S5
lines.append("  S5: ORB-Aligned Breakout")
lines.append(f"    Events untersucht: {s5_quality['n_sampled']}")
lines.append(f"    Aligned Breakout gefunden: {s5_quality['breakout_found']} ({s5_quality['breakout_rate']:.1%})")
lines.append(f"    Breakout hielt (MFE>MAE, 30min): {s5_quality['hold_rate']:.1%}")
lines.append(f"    MFE (30min): mean={s5_quality['mfe_mean']:.3f}, median={s5_quality['mfe_median']:.3f} ADR")
lines.append(f"    MAE (30min): mean={s5_quality['mae_mean']:.3f}, median={s5_quality['mae_median']:.3f} ADR")
lines.append(f"    Reward/Risk: {s5_quality['reward_risk']:.2f}")
if s5_quality['hold_rate'] > 0.55 and s5_quality['reward_risk'] > 1.0:
    lines.append(f"    -> POSITIV: Solide Breakout-Qualitaet mit gutem R/R")
elif s5_quality['hold_rate'] < 0.45:
    lines.append(f"    -> NEGATIV: Breakouts halten oft nicht - viele Fakeouts")
else:
    lines.append(f"    -> NEUTRAL: Breakout-Qualitaet akzeptabel aber nicht herausragend")
lines.append("")

# S6
lines.append("  S6: VPOC Migration + Gap Continuation")
lines.append(f"    Events geprueft: {s6_quality['n_checked']}")
lines.append(f"    VPOC Migration aligned mit Gap: {s6_quality['migration_aligned_rate']:.1%}")
lines.append(f"    Migration Magnitude: mean={s6_quality['migration_mean']:.3f}, median={s6_quality['migration_median']:.3f} ADR")
lines.append(f"    Post-11:00 Drift (Gap-Dir): mean={s6_quality['post_drift_mean']:.3f} ADR")
lines.append(f"    Post-11:00 Win-Rate: {s6_quality['post_drift_winrate']:.1%}")
if s6_quality['migration_aligned_rate'] > 0.5 and s6_quality['post_drift_winrate'] > 0.55:
    lines.append(f"    -> POSITIV: VPOC Migration ist ein starker Bestaetigungs-Indikator")
elif s6_quality['post_drift_winrate'] < 0.45:
    lines.append(f"    -> NEGATIV: VPOC Migration hat keinen predikativen Wert fuer Post-11:00")
else:
    lines.append(f"    -> NEUTRAL: VPOC Migration liefert schwaches Signal")
lines.append("")

# --- ANALYSE 3 ---
lines.append("-" * 72)
lines.append("ANALYSE 3: FILTER-VERFEINERUNG VORSCHLAEGE")
lines.append("-" * 72)
lines.append("")

lines.append("  S1 - VWAP Sigma-2 Mean Reversion:")
if s1_high_vol_mfe and s1_low_vol_mfe:
    hv_rr = np.nanmean(s1_high_vol_mfe) / max(np.nanmean(s1_high_vol_mae), 0.001)
    lv_rr = np.nanmean(s1_low_vol_mfe) / max(np.nanmean(s1_low_vol_mae), 0.001)
    lines.append(f"    High RVOL (>=2): n={len(s1_high_vol_mfe)}, MFE={np.nanmean(s1_high_vol_mfe):.3f}, MAE={np.nanmean(s1_high_vol_mae):.3f}, R/R={hv_rr:.2f}")
    lines.append(f"    Low RVOL (<2):   n={len(s1_low_vol_mfe)}, MFE={np.nanmean(s1_low_vol_mfe):.3f}, MAE={np.nanmean(s1_low_vol_mae):.3f}, R/R={lv_rr:.2f}")
    if hv_rr > lv_rr * 1.2:
        lines.append(f"    -> EMPFEHLUNG: RVOL >= 2 als Zusatz-Filter verwenden (R/R verbessert sich)")
    elif lv_rr > hv_rr * 1.2:
        lines.append(f"    -> EMPFEHLUNG: Niedrige RVOL bevorzugen (weniger Volatilitaet = besserer MR)")
    else:
        lines.append(f"    -> RVOL hat keinen klaren Einfluss auf Entry-Qualitaet")
lines.append("")

lines.append("  S2 - No-Fill Power Move:")
if s2_big_drifts and s2_med_drifts:
    lines.append(f"    Big Gap (>=3 ADR): n={len(s2_big_gap)}, drift={np.nanmean(s2_big_drifts):.3f}, win={np.mean([d>0 for d in s2_big_drifts]):.1%}")
    lines.append(f"    Med Gap (2-3 ADR): n={len(s2_med_gap)}, drift={np.nanmean(s2_med_drifts):.3f}, win={np.mean([d>0 for d in s2_med_drifts]):.1%}")
    if np.nanmean(s2_big_drifts) > np.nanmean(s2_med_drifts) * 1.5:
        lines.append(f"    -> EMPFEHLUNG: Fokus auf grosse Gaps (>=3 ADR) - deutlich besserer Drift")
    else:
        lines.append(f"    -> Gap-Groesse allein kein starker Differenzierer")
lines.append("")

lines.append("  S3 - Late Session Momentum:")
lines.append(f"    -> EMPFEHLUNG: VWAP-Alignment als Pflicht-Filter (aligned vs non-aligned)")
if s3_quality['aligned_winrate'] > s3_quality['pm_positive_rate'] + 0.05:
    lines.append(f"       VWAP-Alignment erhoeht Win-Rate um {(s3_quality['aligned_winrate'] - s3_quality['pm_positive_rate']):.0%}")
lines.append("")

lines.append("  S4 - PM/RTH Ratio Extreme:")
lines.append(f"    -> Sehr kleines Universum ({s4_quality['n_events']} Events). Vorsicht vor Overfitting.")
if s4_quality['n_events'] < 50:
    lines.append(f"    -> WARNUNG: Zu wenige Events fuer robuste Schlussfolgerungen")
lines.append("")

lines.append("  S5 - ORB Breakout:")
lines.append(f"    -> Breakout-Rate: {s5_quality['breakout_rate']:.1%} - " +
             ("haeufig genug fuer Trading" if s5_quality['breakout_rate'] > 0.3 else "zu selten fuer regelmaessiges Trading"))
lines.append(f"    -> EMPFEHLUNG: Volume-Confirmation am Breakout-Bar hinzufuegen")
lines.append("")

lines.append("  S6 - VPOC Migration:")
lines.append(f"    -> EMPFEHLUNG: Nur starke Migrationen (>=3 Buckets) als Signal werten")
lines.append("")

# --- ANALYSE 4 ---
lines.append("-" * 72)
lines.append("ANALYSE 4: OD-UNABHAENGIGKEIT")
lines.append("-" * 72)
lines.append("")
lines.append("Methode: Pearson-Korrelation zwischen od_strength und PnL-Proxy.")
lines.append("|corr| > 0.3 bedeutet: Strategie ist NICHT unabhaengig vom Opening Drive.")
lines.append("")

for strat, corr in od_corr_results.items():
    status = "NICHT UNABHAENGIG" if abs(corr) > 0.3 else "UNABHAENGIG"
    marker = "!!!" if abs(corr) > 0.3 else "   "
    lines.append(f"  {marker} {strat}: corr = {corr:+.3f}  |corr| = {abs(corr):.3f}  -> {status}")

lines.append("")
n_dependent = sum(1 for c in od_corr_results.values() if abs(c) > 0.3)
lines.append(f"  Zusammenfassung: {n_dependent} von {len(od_corr_results)} Strategien zeigen OD-Abhaengigkeit")
lines.append("")

# ============================================================
# TRADER-URTEIL
# ============================================================
lines.append("=" * 72)
lines.append("TRADER-URTEIL PRO STRATEGIE")
lines.append("=" * 72)
lines.append("")

# S1 Urteil
s1_verdict = "UMSETZBAR" if (s1_quality['bounce_rate'] >= 0.52 and
                              s1_quality['reward_risk'] >= 0.9 and
                              abs(od_corr_results.get('S1', 0)) <= 0.3) else "PROBLEMATISCH"
lines.append(f"  S1 (VWAP Sigma-2 MR): [{s1_verdict}]")
lines.append(f"      Bounce-Rate: {s1_quality['bounce_rate']:.1%}, R/R: {s1_quality['reward_risk']:.2f}")
if slippage_results['S1']['vol_ratio_mean'] < 0.7:
    lines.append(f"      ABER: Liquiditaet am Touch oft duenn ({slippage_results['S1']['vol_ratio_mean']:.1f}x avg)")
if abs(od_corr_results.get('S1', 0)) > 0.3:
    lines.append(f"      PROBLEM: Korrelation mit OD = {od_corr_results.get('S1', 0):.3f}")
lines.append(f"      Slippage: {slippage_results['S1']['bar_range_mean']:.1%} ADR pro Bar")
lines.append("")

# S2 Urteil
s2_verdict = "UMSETZBAR" if (s2_quality['post_1100_drift_winrate'] >= 0.52 and
                              s2_quality['post_1100_drift_mean'] > 0 and
                              abs(od_corr_results.get('S2', 0)) <= 0.3) else "PROBLEMATISCH"
lines.append(f"  S2 (No-Fill Power): [{s2_verdict}]")
lines.append(f"      Post-11:00 Drift: {s2_quality['post_1100_drift_mean']:.3f} ADR, Win: {s2_quality['post_1100_drift_winrate']:.1%}")
lines.append(f"      Continuation: {s2_quality['continuation_rate']:.1%}, Exhaustion: {s2_quality['exhaustion_rate']:.1%}")
lines.append(f"      Universum: {s2_quality['n_events']} Events" + (" (KLEIN!)" if s2_quality['n_events'] < 100 else ""))
if abs(od_corr_results.get('S2', 0)) > 0.3:
    lines.append(f"      PROBLEM: Korrelation mit OD = {od_corr_results.get('S2', 0):.3f}")
lines.append(f"      Slippage: {slippage_results['S2']['bar_range_mean']:.4f} ADR pro Bar")
lines.append("")

# S3 Urteil
s3_verdict = "UMSETZBAR" if (s3_quality['aligned_winrate'] >= 0.52 and
                              s3_quality['aligned_drift_mean'] > 0 and
                              abs(od_corr_results.get('S3', 0)) <= 0.3) else "PROBLEMATISCH"
lines.append(f"  S3 (Late Session): [{s3_verdict}]")
lines.append(f"      PM Drift (aligned): {s3_quality['aligned_drift_mean']:.3f} ADR, Win: {s3_quality['aligned_winrate']:.1%}")
lines.append(f"      Nachmittags-Liquiditaet: {slippage_results['S3']['vol_ratio_mean']:.1f}x avg")
if slippage_results['S3']['vol_ratio_mean'] < 0.6:
    lines.append(f"      WARNUNG: Nachmittags-Liquiditaet unter Durchschnitt")
if abs(od_corr_results.get('S3', 0)) > 0.3:
    lines.append(f"      PROBLEM: Korrelation mit OD = {od_corr_results.get('S3', 0):.3f}")
lines.append(f"      Slippage: {slippage_results['S3']['bar_range_mean']:.1%} ADR pro Bar")
lines.append("")

# S4 Urteil
s4_verdict = "PROBLEMATISCH"  # Almost always problematic due to small sample
if s4_quality['n_events'] >= 50 and s4_quality['drift_5min_positive'] >= 0.52:
    s4_verdict = "UMSETZBAR"
lines.append(f"  S4 (PM/RTH Extreme): [{s4_verdict}]")
lines.append(f"      Universum: {s4_quality['n_events']} Events" + (" (ZU KLEIN!)" if s4_quality['n_events'] < 50 else ""))
lines.append(f"      First 5min Drift: {s4_quality['drift_5min_mean']:.3f} ADR, Positive: {s4_quality['drift_5min_positive']:.1%}")
if abs(od_corr_results.get('S4', 0)) > 0.3:
    lines.append(f"      PROBLEM: Korrelation mit OD = {od_corr_results.get('S4', 0):.3f}")
lines.append(f"      Slippage: {slippage_results['S4']['bar_range_mean']:.4f} ADR pro Bar")
lines.append("")

# S5 Urteil
s5_verdict = "UMSETZBAR" if (s5_quality['hold_rate'] >= 0.50 and
                              s5_quality['reward_risk'] >= 0.9 and
                              abs(od_corr_results.get('S5', 0)) <= 0.3) else "PROBLEMATISCH"
lines.append(f"  S5 (ORB Breakout): [{s5_verdict}]")
lines.append(f"      Breakout-Rate: {s5_quality['breakout_rate']:.1%}, Hold: {s5_quality['hold_rate']:.1%}")
lines.append(f"      R/R: {s5_quality['reward_risk']:.2f}")
if abs(od_corr_results.get('S5', 0)) > 0.3:
    lines.append(f"      PROBLEM: Korrelation mit OD = {od_corr_results.get('S5', 0):.3f} - NICHT unabhaengig!")
lines.append(f"      Slippage: {slippage_results['S5']['bar_range_mean']:.4f} ADR pro Bar")
lines.append("")

# S6 Urteil
s6_verdict = "UMSETZBAR" if (s6_quality['post_drift_winrate'] >= 0.52 and
                              s6_quality['post_drift_mean'] > 0 and
                              abs(od_corr_results.get('S6', 0)) <= 0.3) else "PROBLEMATISCH"
lines.append(f"  S6 (VPOC Migration): [{s6_verdict}]")
lines.append(f"      Post-11:00 Drift: {s6_quality['post_drift_mean']:.3f} ADR, Win: {s6_quality['post_drift_winrate']:.1%}")
lines.append(f"      Migration Aligned: {s6_quality['migration_aligned_rate']:.1%}")
if abs(od_corr_results.get('S6', 0)) > 0.3:
    lines.append(f"      PROBLEM: Korrelation mit OD = {od_corr_results.get('S6', 0):.3f}")
lines.append(f"      Slippage: {slippage_results['S6']['bar_range_mean']:.4f} ADR pro Bar")
lines.append("")

# ZUSAMMENFASSUNG
lines.append("=" * 72)
lines.append("ZUSAMMENFASSUNG & RANKING")
lines.append("=" * 72)
lines.append("")

verdicts = {'S1': s1_verdict, 'S2': s2_verdict, 'S3': s3_verdict,
            'S4': s4_verdict, 'S5': s5_verdict, 'S6': s6_verdict}
umsetzbar = [s for s, v in verdicts.items() if v == "UMSETZBAR"]
problematisch = [s for s, v in verdicts.items() if v == "PROBLEMATISCH"]

lines.append(f"  UMSETZBAR ({len(umsetzbar)}): {', '.join(umsetzbar) if umsetzbar else 'Keine'}")
lines.append(f"  PROBLEMATISCH ({len(problematisch)}): {', '.join(problematisch) if problematisch else 'Keine'}")
lines.append("")

lines.append("  Top-3 Empfehlungen fuer den Quant (Prioritaet):")
# Build simple ranking
rank_scores = {}
for strat in ['S1','S2','S3','S4','S5','S6']:
    score = 0
    if verdicts[strat] == 'UMSETZBAR':
        score += 3
    od_c = abs(od_corr_results.get(strat, 0))
    score -= od_c * 5  # penalize OD correlation
    if slippage_results[strat]['vol_ratio_mean'] > 1.0:
        score += 1
    if slippage_results[strat]['bar_range_mean'] < 0.03:
        score += 1
    rank_scores[strat] = score

sorted_strats = sorted(rank_scores.items(), key=lambda x: x[1], reverse=True)
for rank, (strat, score) in enumerate(sorted_strats[:3], 1):
    lines.append(f"    {rank}. {strat} (Score: {score:.1f}) - {verdicts[strat]}")

lines.append("")
lines.append("=" * 72)
lines.append("ENDE DER ANALYSE")
lines.append("=" * 72)

# Write
result_text = "\n".join(lines)
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    f.write(result_text)

print(f"\nErgebnisse geschrieben nach: {OUTPUT_FILE}")
print("\nDONE.")
