"""
D14 Category 2.3: Streaming Dip & Rip
=======================================
CHANGE vs D13:
  D13: Uses .idxmin() on 09:36-10:00 to find absolute pullback low
  D14: Running Pullback Low — tracks low bar by bar, checks recovery
       against the running low at each point

Parameters (exact D13):
  - od_strength >= 0.30
  - od_direction = with_gap
  - PB depth >= 0.05 ADR
  - PB window: 09:36-10:00
  - Recovery: Rolling 5-bar high > OD_High * 0.998 (or low < OD_Low * 1.002)
  - Recovery deadline: 10:30
  - SL: PB_Low - 0.02*ADR (long), PB_High + 0.02*ADR (short)
  - SL max: 0.50 ADR
"""
import pandas as pd
import numpy as np
from pathlib import Path


RAW_DIR = Path('data/raw_1min')


def detect_dip_and_rip_streaming(rth, gap_dir, adr, od_strength):
    """
    Streaming Dip & Rip detection.

    KEY DIFFERENCE vs D13:
    - D13: Find absolute PB_Low with .idxmin() then scan recovery after
    - D14: Track running PB low. At each bar in the pullback+recovery window:
      1. Update running extreme
      2. Check depth threshold
      3. If depth met, check 5-bar rolling recovery

    This may produce earlier entries (at a higher running low that later goes lower)
    or different entries altogether.
    """
    if od_strength < 0.30:
        return None

    # OD extreme
    od_bars = rth[(rth['time_et'] >= '09:30') & (rth['time_et'] <= '09:35')]
    if len(od_bars) == 0:
        return None

    if gap_dir == 'up':
        od_high = od_bars['high'].max()

        # Scan pullback + recovery window (09:36-10:30)
        scan_bars = rth[(rth['time_et'] >= '09:36') & (rth['time_et'] <= '10:30')]
        if len(scan_bars) < 5:
            return None

        running_pb_low = float('inf')
        depth_met = False
        highs_history = []

        for idx in range(len(scan_bars)):
            bar = scan_bars.iloc[idx]
            bar_time = bar['time_et']

            # Update running pullback low (only during 09:36-10:00)
            if bar_time <= '10:00':
                if bar['low'] < running_pb_low:
                    running_pb_low = bar['low']

            # Check depth
            if running_pb_low < float('inf'):
                pb_depth = (od_high - running_pb_low) / adr
                if pb_depth >= 0.05:
                    depth_met = True

            # Track highs for rolling window
            highs_history.append(bar['high'])

            # Check recovery (need depth met + at least 5 bars of history)
            if depth_met and len(highs_history) >= 5:
                window_high = max(highs_history[-5:])
                if window_high > od_high * 0.998:
                    entry_price = bar['close']
                    sl_level = running_pb_low - 0.02 * adr
                    sl_dist = entry_price - sl_level
                    if sl_dist <= 0 or sl_dist > 0.5 * adr:
                        return None
                    return {
                        'entry_price': entry_price,
                        'entry_time': bar_time,
                        'sl_dist': sl_dist,
                        'pb_depth': pb_depth,
                        'trade_dir': 'long',
                        'running_pb_low': running_pb_low,
                    }

        return None

    else:  # gap_dir == 'down'
        od_low = od_bars['low'].min()

        scan_bars = rth[(rth['time_et'] >= '09:36') & (rth['time_et'] <= '10:30')]
        if len(scan_bars) < 5:
            return None

        running_pb_high = float('-inf')
        depth_met = False
        lows_history = []

        for idx in range(len(scan_bars)):
            bar = scan_bars.iloc[idx]
            bar_time = bar['time_et']

            # Update running pullback high (only during 09:36-10:00)
            if bar_time <= '10:00':
                if bar['high'] > running_pb_high:
                    running_pb_high = bar['high']

            # Check depth
            if running_pb_high > float('-inf'):
                pb_depth = (running_pb_high - od_low) / adr
                if pb_depth >= 0.05:
                    depth_met = True

            lows_history.append(bar['low'])

            # Check recovery
            if depth_met and len(lows_history) >= 5:
                window_low = min(lows_history[-5:])
                if window_low < od_low * 1.002:
                    entry_price = bar['close']
                    sl_level = running_pb_high + 0.02 * adr
                    sl_dist = sl_level - entry_price
                    if sl_dist <= 0 or sl_dist > 0.5 * adr:
                        return None
                    return {
                        'entry_price': entry_price,
                        'entry_time': bar_time,
                        'sl_dist': sl_dist,
                        'pb_depth': pb_depth,
                        'trade_dir': 'short',
                        'running_pb_high': running_pb_high,
                    }

        return None


def run_cat2_dip_and_rip(metadata, simulate_fn, label=''):
    """
    Run Dip & Rip strategy.

    Pre-filter: od_strength >= 0.30, od_direction = with_gap
    """
    valid = metadata[
        metadata['close_935'].notna() &
        metadata['adr_10'].notna() &
        (metadata['adr_10'] > 0) &
        metadata['od_strength'].notna() &
        (metadata['od_strength'] >= 0.30) &
        (metadata['od_direction'] == 'with_gap')
    ].copy()

    results = []
    skipped = 0

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

        info = detect_dip_and_rip_streaming(rth, gap_dir, adr, od_str)
        if info is None:
            continue

        result = simulate_fn(rth, info['entry_price'], info['sl_dist'],
                             info['trade_dir'], adr,
                             entry_time=info['entry_time'],
                             ticker=ticker, date=date)
        if result is None:
            continue

        result['ticker'] = ticker
        result['date'] = date
        result['gap_dir'] = gap_dir
        result['pb_depth'] = info['pb_depth']
        result['entry_time_actual'] = info['entry_time']
        result['sl_dist'] = info['sl_dist']
        result['adr'] = adr
        result['strategy'] = '2.3_Dip_and_Rip'
        results.append(result)

    print(f"  {label} 2.3_Dip_and_Rip: N={len(results)} trades ({skipped} skipped)")
    return results
