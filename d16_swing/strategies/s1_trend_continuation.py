"""
D16 Strategy S1: Trend Day Continuation (Long)
===============================================
Logic: Strong gappers that stay in trend all day -> likely multi-day runner.
Entry after 13:00 when confirmed as trend day.

Pre-filter (metadata):
  - gap_size_in_adr >= 1.0
  - od_direction == 'with_gap'
  - od_strength >= 0.30
  - rvol_5 >= 3.0
  - gap_direction == 'up' (long only)

Streaming confirmation (bar-by-bar after 13:00):
  - Price in top 30% of day range
  - Price above VWAP
  - Gap NOT filled

Entry: Close of confirmation bar
SL: sl_adr_mult * ADR below entry
Exit: Trail-D Multiday (max_days configurable)
"""
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from d16_swing.engine.realtime_vwap import StreamingVWAP
from d16_swing.engine.pattern_detector import TrendContinuationDetector

RAW_DIR = Path('data/raw_1min')


def detect_s1_trend_continuation(rth, gap_dir, adr, prev_close):
    """
    Streaming detection of Trend Day Continuation pattern.

    Parameters:
        rth: DataFrame with time_et, open, high, low, close, volume (RTH bars)
        gap_dir: 'up' or 'down'
        adr: float, average daily range
        prev_close: float, previous day close price

    Returns:
        dict with entry_price, entry_time, trade_dir or None
    """
    if gap_dir != 'up':
        return None  # long only

    vwap = StreamingVWAP()
    detector = TrendContinuationDetector()

    gap_filled = False

    for _, bar in rth.iterrows():
        t = bar['time_et']
        h, l, c, v = bar['high'], bar['low'], bar['close'], bar['volume']

        # Update VWAP
        current_vwap = vwap.update(h, l, c, v)

        # Track gap fill: gap up filled when low <= prev_close
        if not gap_filled and l <= prev_close:
            gap_filled = True

        # Check pattern
        if detector.update(t, h, l, c, current_vwap, gap_filled):
            return {
                'entry_price': detector.entry_price,
                'entry_time': detector.entry_time,
                'trade_dir': 'long',
            }

    return None


def run_s1(metadata, followup_map, simulate_multiday_fn, sl_adr_mult=0.50,
           max_days=5, label=''):
    """
    Run S1 strategy on filtered metadata.

    Parameters:
        metadata: DataFrame with all columns
        followup_map: DataFrame with ticker, gap_date, offset, actual_date
        simulate_multiday_fn: function(entry_price, sl_adr_mult, adr, sign,
                                        day0_bars, followup_sums, max_days) -> dict
        sl_adr_mult: SL width as ADR fraction
        max_days: max holding days
        label: 'IS' or 'OOS'

    Returns:
        list of trade dicts
    """
    # Pre-filter
    valid = metadata[
        (metadata['gap_size_in_adr'] >= 1.0) &
        (metadata['od_direction'] == 'with_gap') &
        (metadata['od_strength'] >= 0.30) &
        (metadata['rvol_5'] >= 3.0) &
        (metadata['gap_direction'] == 'up') &
        (metadata['adr_10'] > 0) &
        (metadata['prev_close'].notna())
    ].copy()

    results = []
    skipped = 0

    for _, row in valid.iterrows():
        ticker = row['ticker']
        date = str(row['date'])
        adr = row['adr_10']
        prev_close = row['prev_close']

        # Load 1-min bars
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

        # Detect pattern
        info = detect_s1_trend_continuation(rth, row['gap_direction'], adr, prev_close)
        if info is None:
            continue

        entry_price = info['entry_price']
        entry_time = info['entry_time']

        # Day0: NO SL on entry day — this is a confirmed trend day,
        # hold to close. SL starts from D+1 only.
        # Pass None for day0_bars to skip intraday SL processing.
        followup_sums = _build_followup_sums(ticker, date, followup_map, max_days)

        result = simulate_multiday_fn(
            entry_price=entry_price,
            sl_adr_mult=sl_adr_mult,
            adr=adr,
            sign=1,  # long
            day0_bars_after_entry=None,  # no day0 SL
            followup_sums=followup_sums,
            max_days=max_days,
        )
        if result is None:
            continue

        result['ticker'] = ticker
        result['date'] = date
        result['entry_time'] = entry_time
        result['entry_price'] = entry_price
        result['adr'] = adr
        result['sl_adr_mult'] = sl_adr_mult
        result['strategy'] = 'S1_Trend_Continuation'
        result['trade_dir'] = 'long'
        results.append(result)

    print(f"  {label} S1_Trend_Continuation (SL={sl_adr_mult}, days={max_days}): "
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
