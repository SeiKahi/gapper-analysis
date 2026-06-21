"""
D16 Strategy S2: Power Breakout Hold (Long)
=============================================
Logic: Stock makes new HOD in afternoon with volume => breakout momentum
carries into follow-up days.

Pre-filter (metadata):
  - gap_size_in_adr >= 0.5
  - od_direction == 'with_gap'
  - rvol_5 >= 2.0
  - gap_direction == 'up' (long only)

Streaming confirmation (bar-by-bar after 11:00):
  - New HOD (bar high > previous running high)
  - Volume of breakout bar >= 1.5x average volume of prior bars
  - Price above VWAP

Entry: Close of breakout bar
SL: max(sl_adr_mult * ADR, entry - AM_low) below entry
Exit: Trail-D Multiday (max_days configurable)
"""
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from d16_swing.engine.realtime_vwap import StreamingVWAP
from d16_swing.engine.pattern_detector import PowerBreakoutDetector

RAW_DIR = Path('data/raw_1min')


def detect_s2_power_breakout(rth, adr):
    """
    Streaming detection of Power Breakout pattern.

    Parameters:
        rth: DataFrame with time_et, open, high, low, close, volume
        adr: float, average daily range

    Returns:
        dict with entry_price, entry_time, am_low, trade_dir or None
    """
    vwap = StreamingVWAP()
    detector = PowerBreakoutDetector()

    for _, bar in rth.iterrows():
        t = bar['time_et']
        h, l, c, v = bar['high'], bar['low'], bar['close'], bar['volume']

        current_vwap = vwap.update(h, l, c, v)

        if detector.update(t, h, l, c, v, current_vwap):
            return {
                'entry_price': detector.entry_price,
                'entry_time': detector.entry_time,
                'am_low': detector.am_low,
                'trade_dir': 'long',
            }

    return None


def run_s2(metadata, followup_map, simulate_multiday_fn, sl_adr_mult=0.50,
           max_days=5, label=''):
    """
    Run S2 strategy on filtered metadata.

    SL = max(sl_adr_mult * ADR, entry - AM_low) — never tighter than morning low.
    """
    # Pre-filter
    valid = metadata[
        (metadata['gap_size_in_adr'] >= 0.5) &
        (metadata['od_direction'] == 'with_gap') &
        (metadata['rvol_5'] >= 2.0) &
        (metadata['gap_direction'] == 'up') &
        (metadata['adr_10'] > 0)
    ].copy()

    results = []
    skipped = 0

    for _, row in valid.iterrows():
        ticker = row['ticker']
        date = str(row['date'])
        adr = row['adr_10']

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

        info = detect_s2_power_breakout(rth, adr)
        if info is None:
            continue

        entry_price = info['entry_price']
        entry_time = info['entry_time']
        am_low = info['am_low']

        # SL: max(sl_adr_mult * ADR, entry - AM_low) — never tighter than AM low
        sl_dist_min = sl_adr_mult * adr
        sl_dist_am = entry_price - am_low if am_low < entry_price else sl_dist_min
        sl_dist = max(sl_dist_min, sl_dist_am)

        # SL as ADR multiple for the multiday function
        actual_sl_adr = sl_dist / adr

        # Day0 remaining bars
        day0_remaining = rth[rth['time_et'] > entry_time].copy()

        # Followup sums
        followup_sums = _build_followup_sums(ticker, date, followup_map, max_days)

        result = simulate_multiday_fn(
            entry_price=entry_price,
            sl_adr_mult=actual_sl_adr,
            adr=adr,
            sign=1,
            day0_bars_after_entry=day0_remaining,
            followup_sums=followup_sums,
            max_days=max_days,
        )
        if result is None:
            continue

        result['ticker'] = ticker
        result['date'] = date
        result['entry_time'] = entry_time
        result['entry_price'] = entry_price
        result['am_low'] = am_low
        result['adr'] = adr
        result['sl_dist'] = sl_dist
        result['sl_adr_mult'] = actual_sl_adr
        result['strategy'] = 'S2_Power_Breakout'
        result['trade_dir'] = 'long'
        results.append(result)

    print(f"  {label} S2_Power_Breakout (SL={sl_adr_mult}, days={max_days}): "
          f"N={len(results)} trades ({skipped} skipped)")
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
