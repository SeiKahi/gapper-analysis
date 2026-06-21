"""
D14 Category 3: Late-Day Swing Strategies
===========================================
7 Patterns x 3 Entry Times, Multi-Day TRAIL_D

Patterns (from D13 lateday_v3):
  BASELINE:    All events (no filter)
  VWAP_ABOVE:  VWAP z-score > 0 (up) or < 0 (down) at entry time
  RECOVERY:    Sell-off detected + above VWAP at entry
  STRONG_POS:  Price vs range > 0.60 (near HOD for up)
  POS_DAY:     Price vs open > 0 (positive return for gap direction)
  FAILED_FADE: od=against_gap + gap NOT filled at entry time + above VWAP
  TREND_DAY:   od=with_gap + above VWAP

Entry Times: 14:30, 15:00, 15:59
SL: 0.25 ADR, TRAIL_D, MaxDays=5

Bias Fixes:
  - FAILED_FADE: gap_filled checked in real-time at entry time (not EOD)
  - TRAIL_D: Conservative SL-first on daily bars
  - Day 0 remaining: bar-by-bar instead of .min()/.max()
"""
import pandas as pd
import numpy as np
from pathlib import Path

from d14_bias_free.engine.gap_fill_tracker import is_gap_filled_at_time
from d14_bias_free.engine.streaming_trail import simulate_multiday_trail_d_streaming


RAW_DIR = Path('data/raw_1min')
VWAP_DIR = Path('data/vwap')
MAPPING_PATH = Path('data/metadata/followup_day_mapping.parquet')

ENTRY_TIMES = ['14:30', '15:00', '15:59']
SL_MULT = 0.25
MAX_DAYS = 5


def extract_bar_summary(ticker, date_str):
    """Load daily summary from 1-min bars."""
    fpath = RAW_DIR / ticker / f'{date_str}.parquet'
    if not fpath.exists():
        return None
    try:
        df = pd.read_parquet(fpath)
        rth = df[df['session'] == 'rth'] if 'session' in df.columns else df
        if len(rth) == 0:
            return None
        return {
            'rth_open': rth.iloc[0]['open'],
            'rth_close': rth.iloc[-1]['close'],
            'rth_high': rth['high'].max(),
            'rth_low': rth['low'].min(),
        }
    except Exception:
        return None


def build_followup_lookup(mapping_df):
    """Build lookup: (ticker, date) -> {offset: actual_date}."""
    lookup = {}
    for _, row in mapping_df.iterrows():
        key = (row['ticker'], str(row['gap_date']))
        if key not in lookup:
            lookup[key] = {}
        lookup[key][int(row['offset'])] = str(row['actual_date'])
    return lookup


def analyze_day0_signals(rth, vwap_data, gap_dir, adr, prev_close, od_direction):
    """
    Analyze Day 0 at each entry time — bias-free version.

    FIX: gap_filled is checked in real-time at each entry time using
    bar-by-bar scanning (not from metadata).
    """
    sign = 1.0 if gap_dir == 'up' else -1.0
    day_open = rth.iloc[0]['open']

    # Sell-off detection (09:36-10:30)
    selloff_detected = False
    od_bars = rth[(rth['time_et'] >= '09:30') & (rth['time_et'] <= '09:35')]
    if len(od_bars) > 0:
        if gap_dir == 'up':
            od_peak = od_bars['high'].max()
            so_bars = rth[(rth['time_et'] >= '09:36') & (rth['time_et'] <= '10:30')]
            if len(so_bars) > 0:
                selloff_depth = (od_peak - so_bars['low'].min()) / adr
                selloff_detected = selloff_depth >= 0.10
        else:
            od_peak = od_bars['low'].min()
            so_bars = rth[(rth['time_et'] >= '09:36') & (rth['time_et'] <= '10:30')]
            if len(so_bars) > 0:
                selloff_depth = (so_bars['high'].max() - od_peak) / adr
                selloff_detected = selloff_depth >= 0.10

    results = {}
    for et in ENTRY_TIMES:
        bars_to = rth if et == '15:59' else rth[rth['time_et'] <= et]
        if len(bars_to) < 10:
            results[et] = None
            continue

        entry_price = bars_to.iloc[-1]['close']
        hod = bars_to['high'].max()
        lod = bars_to['low'].min()
        day_range = max(hod - lod, 0.001 * entry_price)
        price_vs_range = ((entry_price - lod) / day_range if gap_dir == 'up'
                          else (hod - entry_price) / day_range)
        price_vs_open = (entry_price - day_open) / adr * sign

        # VWAP z-score at entry
        vwap_z = np.nan
        if vwap_data is not None and 'time_et' in vwap_data.columns:
            vbar = vwap_data if et == '15:59' else vwap_data[vwap_data['time_et'] <= et]
            if len(vbar) > 0:
                vwap_z = vbar.iloc[-1]['z_score']

        above_vwap = False
        if not np.isnan(vwap_z):
            above_vwap = (vwap_z > 0 and gap_dir == 'up') or (vwap_z < 0 and gap_dir == 'down')

        # FIX: Real-time gap fill check at this entry time
        gap_filled_rt = is_gap_filled_at_time(rth, gap_dir, prev_close, et)

        # Remaining bars after entry for Day 0 trail
        remaining_bars = rth[rth['time_et'] > et] if et != '15:59' else None

        results[et] = {
            'entry_price': entry_price,
            'above_vwap': above_vwap,
            'price_vs_range': price_vs_range,
            'price_vs_open': price_vs_open,
            'selloff_detected': selloff_detected,
            'recovery_vwap': selloff_detected and above_vwap,
            'strong_position': price_vs_range > 0.60,
            'positive_day': price_vs_open > 0,
            'gap_filled_rt': gap_filled_rt,
            'od_direction': od_direction,
            'remaining_bars': remaining_bars,
        }
    return results


def get_pattern_mask(signal, pattern_name):
    """Check if a signal matches a pattern."""
    if signal is None:
        return False

    if pattern_name == 'BASELINE':
        return True
    elif pattern_name == 'VWAP_ABOVE':
        return signal['above_vwap']
    elif pattern_name == 'RECOVERY':
        return signal['recovery_vwap']
    elif pattern_name == 'STRONG_POS':
        return signal['strong_position']
    elif pattern_name == 'POS_DAY':
        return signal['positive_day']
    elif pattern_name == 'FAILED_FADE':
        # FIX: Use real-time gap fill, not metadata
        return (signal['od_direction'] == 'against_gap' and
                not signal['gap_filled_rt'] and
                signal['above_vwap'])
    elif pattern_name == 'TREND_DAY':
        return (signal['od_direction'] == 'with_gap' and
                signal['above_vwap'])
    return False


PATTERNS = ['BASELINE', 'VWAP_ABOVE', 'RECOVERY', 'STRONG_POS',
            'POS_DAY', 'FAILED_FADE', 'TREND_DAY']


def run_cat3_lateday(metadata, label=''):
    """
    Run Late-Day Swing strategies.

    Returns:
        dict: {(pattern, entry_time): list of result dicts}
    """
    # Load followup mapping
    if not MAPPING_PATH.exists():
        print(f"  WARNING: {MAPPING_PATH} not found, skipping Cat 3")
        return {}

    mapping_df = pd.read_parquet(MAPPING_PATH)
    followup_lookup = build_followup_lookup(mapping_df)

    valid = metadata[
        metadata['close_935'].notna() &
        metadata['adr_10'].notna() &
        (metadata['adr_10'] > 0) &
        metadata['gap_direction'].notna() &
        metadata['prev_close'].notna()
    ].copy()

    # Pre-load followup summaries
    print(f"  {label} Cat 3: Pre-loading followup summaries...")
    fu_cache = {}
    needed = set()
    for _, row in valid.iterrows():
        key = (row['ticker'], str(row['date']))
        fu_map = followup_lookup.get(key, {})
        for offset in range(1, MAX_DAYS + 1):
            fu_date = fu_map.get(offset)
            if fu_date:
                needed.add((row['ticker'], fu_date))

    for ticker, fu_date in needed:
        summary = extract_bar_summary(ticker, fu_date)
        if summary:
            fu_cache[(ticker, fu_date)] = summary

    print(f"  {label} Cat 3: Loaded {len(fu_cache)}/{len(needed)} followup summaries")

    # Process each event
    all_results = {(p, et): [] for p in PATTERNS for et in ENTRY_TIMES}
    processed = 0
    skipped = 0

    for _, row in valid.iterrows():
        ticker = row['ticker']
        date = str(row['date'])
        adr = row['adr_10']
        gap_dir = row['gap_direction']
        sign = 1.0 if gap_dir == 'up' else -1.0
        prev_close = row['prev_close']
        od_dir = row.get('od_direction', '')

        # Load Day 0 bars
        rth_path = RAW_DIR / ticker / f"{date}.parquet"
        if not rth_path.exists():
            skipped += 1
            continue
        try:
            bars = pd.read_parquet(rth_path)
            rth = bars[bars['session'] == 'rth'].copy() if 'session' in bars.columns else bars.copy()
        except Exception:
            skipped += 1
            continue
        if len(rth) < 10:
            skipped += 1
            continue

        # Load VWAP
        vwap_path = VWAP_DIR / ticker / f"{date}.parquet"
        vwap_data = None
        if vwap_path.exists():
            try:
                vwap_data = pd.read_parquet(vwap_path)
                if 'z_score' not in vwap_data.columns:
                    vwap_data = None
            except Exception:
                pass

        # Analyze signals at each entry time
        signals = analyze_day0_signals(rth, vwap_data, gap_dir, adr, prev_close, od_dir)

        # Build followup summaries
        key = (ticker, date)
        fu_map = followup_lookup.get(key, {})
        fu_sums = []
        for offset in range(1, MAX_DAYS + 1):
            fu_date = fu_map.get(offset)
            if fu_date:
                fu_sums.append(fu_cache.get((ticker, fu_date)))
            else:
                fu_sums.append(None)

        # For each pattern x entry_time
        for pattern in PATTERNS:
            for et in ENTRY_TIMES:
                sig = signals.get(et)
                if sig is None:
                    continue
                if not get_pattern_mask(sig, pattern):
                    continue

                entry_price = sig['entry_price']
                remaining_bars = sig['remaining_bars']

                # Simulate multi-day TRAIL_D
                result = simulate_multiday_trail_d_streaming(
                    entry_price=entry_price,
                    sl_adr_mult=SL_MULT,
                    adr=adr,
                    sign=sign,
                    day0_bars_after_entry=remaining_bars,
                    followup_sums=fu_sums,
                    max_days=MAX_DAYS,
                )
                if result is None:
                    continue

                result['ticker'] = ticker
                result['date'] = date
                result['gap_dir'] = gap_dir
                result['entry_time'] = et
                result['pattern'] = pattern
                result['strategy'] = f'3_{pattern}_{et}'
                all_results[(pattern, et)].append(result)

        processed += 1

    print(f"  {label} Cat 3: Processed {processed} events ({skipped} skipped)")
    total_trades = sum(len(v) for v in all_results.values())
    print(f"  {label} Cat 3: {total_trades} total trade records across all pattern/time combos")
    return all_results
