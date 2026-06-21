"""
D16 Strategy S3: Dip & Hold (Long)
====================================
Logic: D14 Dip & Rip (EV +0.962R intraday) detected in morning,
then HOLD if still strong in afternoon => swing conversion.

Phase 1: Dip & Rip Detection (reuse D14 exact logic)
Phase 2: Afternoon Persistence Check (ab 14:00):
  - Price still above Dip & Rip entry
  - Price above VWAP
  - No new LOD since entry

If ALL met => Swing conversion (hold overnight)

Entry: Original Dip & Rip entry price
SL: min(swing_sl_adr * ADR, dip_and_rip_sl) on day0; widened to swing_sl_adr for overnight
Exit: Trail-D Multiday (max_days configurable)
"""
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from d14_bias_free.strategies.cat2_dip_and_rip import detect_dip_and_rip_streaming
from d16_swing.engine.realtime_vwap import StreamingVWAP

RAW_DIR = Path('data/raw_1min')


def check_afternoon_persistence(rth, entry_price, entry_time, gap_dir, pb_low, adr):
    """
    Check afternoon persistence after Dip & Rip entry.

    Scans bars from 14:00 onwards. Returns True on first bar where ALL conditions met:
    - Price (close) above entry price (for longs)
    - Price above VWAP
    - LOD since entry held above pullback low (the dip didn't get worse)

    Parameters:
        rth: DataFrame with time_et, open, high, low, close, volume
        entry_price: float, original Dip & Rip entry price
        entry_time: str, 'HH:MM' of original entry
        gap_dir: 'up' or 'down'
        pb_low: float, the pullback low from D&R detection
        adr: float, average daily range

    Returns:
        dict with persist_time or None
    """
    if gap_dir != 'up':
        return None  # long only for now

    # Track LOD from entry onwards
    post_entry = rth[rth['time_et'] > entry_time]
    if len(post_entry) == 0:
        return None

    # Compute VWAP from session start
    vwap = StreamingVWAP()
    lod_since_entry = float('inf')

    for _, bar in rth.iterrows():
        t = bar['time_et']
        h, l, c, v = bar['high'], bar['low'], bar['close'], bar['volume']

        current_vwap = vwap.update(h, l, c, v)

        # Track LOD since entry
        if t > entry_time:
            lod_since_entry = min(lod_since_entry, l)

        # Only check persistence from 14:00 onwards
        if t < '14:00':
            continue

        # Must be after entry
        if t <= entry_time:
            continue

        # Condition 1: Price above entry
        if c <= entry_price:
            continue

        # Condition 2: Price above VWAP
        if c <= current_vwap:
            continue

        # Condition 3: LOD since entry held above the pullback low
        # (the morning dip low is support; if broken, pattern is invalidated)
        if lod_since_entry < pb_low:
            continue

        return {'persist_time': t}

    return None


def run_s3(metadata, followup_map, simulate_multiday_fn, sl_adr_mult=0.50,
           max_days=5, label=''):
    """
    Run S3 strategy on filtered metadata.

    Phase 1: Detect Dip & Rip using D14 exact logic
    Phase 2: Check afternoon persistence
    Phase 3: Simulate multiday if persistence confirmed
    """
    # Pre-filter (same as D14 Dip & Rip)
    valid = metadata[
        (metadata['od_strength'] >= 0.30) &
        (metadata['od_direction'] == 'with_gap') &
        (metadata['adr_10'] > 0) &
        (metadata['gap_direction'] == 'up') &  # long only
        (metadata['close_935'].notna())
    ].copy()

    results = []
    skipped = 0
    dip_rip_found = 0

    for _, row in valid.iterrows():
        ticker = row['ticker']
        date = str(row['date'])
        adr = row['adr_10']
        gap_dir = row['gap_direction']
        od_str = row['od_strength']

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

        # Phase 1: Dip & Rip Detection (exact D14 logic)
        dr_info = detect_dip_and_rip_streaming(rth, gap_dir, adr, od_str)
        if dr_info is None:
            continue
        dip_rip_found += 1

        entry_price = dr_info['entry_price']
        entry_time = dr_info['entry_time']
        dr_sl_dist = dr_info['sl_dist']

        # Phase 2: Afternoon Persistence
        pb_low = dr_info.get('running_pb_low', entry_price)
        persist = check_afternoon_persistence(rth, entry_price, entry_time, gap_dir,
                                              pb_low, adr)
        if persist is None:
            continue

        # Day0: Persistence confirmed at 14:00+ => hold to close, no intraday SL.
        # The swing trail starts from D+1. Pass None for day0 bars.
        persist_time = persist['persist_time']

        followup_sums = _build_followup_sums(ticker, date, followup_map, max_days)

        result = simulate_multiday_fn(
            entry_price=entry_price,
            sl_adr_mult=sl_adr_mult,
            adr=adr,
            sign=1,  # long
            day0_bars_after_entry=None,  # no day0 SL — hold to close
            followup_sums=followup_sums,
            max_days=max_days,
        )
        if result is None:
            continue

        result['ticker'] = ticker
        result['date'] = date
        result['entry_time'] = entry_time
        result['persist_time'] = persist_time
        result['entry_price'] = entry_price
        result['dr_sl_dist'] = dr_sl_dist
        result['swing_sl_dist'] = sl_adr_mult * adr
        result['adr'] = adr
        result['sl_adr_mult'] = sl_adr_mult
        result['strategy'] = 'S3_Dip_and_Hold'
        result['trade_dir'] = 'long'
        results.append(result)

    print(f"  {label} S3_Dip_and_Hold (SL={sl_adr_mult}, days={max_days}): "
          f"N={len(results)} trades, Dip&Rip found={dip_rip_found} ({skipped} skipped)")
    return results


def _build_followup_sums(ticker, date, followup_map, max_days):
    """Build list of followup day OHLC summaries from 1-min data."""
    fdays = followup_map[
        (followup_map['ticker'] == ticker) &
        (followup_map['gap_date'] == date)
    ].sort_values('offset')

    sums = []
    for _, frow in fdays.iterrows():
        if frow['offset'] > max_days:
            break
        act_date = str(frow['actual_date'])
        fpath = RAW_DIR / ticker / f"{act_date}.parquet"
        if not fpath.exists():
            sums.append(None)
            continue
        try:
            fb = pd.read_parquet(fpath)
            frth = fb[fb['session'] == 'rth'] if 'session' in fb.columns else fb
            if len(frth) == 0:
                sums.append(None)
                continue
            sums.append({
                'rth_open': frth.iloc[0]['open'],
                'rth_close': frth.iloc[-1]['close'],
                'rth_high': frth['high'].max(),
                'rth_low': frth['low'].min(),
            })
        except Exception:
            sums.append(None)
    return sums
