"""
D14 Category 2.1: Streaming Sell-Off & Recovery
=================================================
CRITICAL CHANGE vs D13:
  D13: Uses .idxmin() on 09:36-10:30 window = looks at ALL bars to find lowest point
       (knows the future within the window)
  D14: Running Low bar-by-bar = only knows past bars, entry at FIRST recovery signal

Parameters (exact D13):
  - min_depth: 0.10 ADR
  - selloff_end: 10:30
  - recovery_deadline: 11:00
  - recovery_type: vwap (z_score >= 0)
  - SL buffer: 0.02 * ADR below running low
  - SL max: 0.50 * ADR
"""
import pandas as pd
import numpy as np
from pathlib import Path


RAW_DIR = Path('data/raw_1min')
VWAP_DIR = Path('data/vwap')


def detect_selloff_recovery_streaming(rth, vwap_data, gap_dir, adr,
                                       min_depth=0.10, selloff_end='10:30',
                                       recovery_deadline='11:00'):
    """
    Streaming Sell-Off & Recovery detection.

    KEY DIFFERENCE vs D13:
    - D13: Find absolute min/max of entire selloff window, then scan recovery after
    - D14: Track running low/high bar by bar. At each bar:
           1. Update running extreme
           2. Check if depth threshold met
           3. If so, check recovery conditions
           4. Return FIRST valid entry

    This means D14 may find entry EARLIER (running low at time T < absolute low)
    or may find DIFFERENT entries (recovery relative to running low, not absolute low).
    """
    # OD Peak: highest high / lowest low 09:30-09:35
    od_bars = rth[(rth['time_et'] >= '09:30') & (rth['time_et'] <= '09:35')]
    if len(od_bars) == 0:
        return None

    if gap_dir == 'up':
        od_peak = od_bars['high'].max()
    else:
        od_peak = od_bars['low'].min()

    # Sell-off window: 09:36 to selloff_end
    so_bars = rth[(rth['time_et'] >= '09:36') & (rth['time_et'] <= selloff_end)]
    if len(so_bars) < 3:
        return None

    # Recovery window bars (selloff_end to deadline, but we also scan during selloff)
    all_scan_bars = rth[(rth['time_et'] >= '09:36') & (rth['time_et'] <= recovery_deadline)]

    # VWAP data for recovery detection
    if vwap_data is None:
        return None
    if 'z_score' not in vwap_data.columns or 'time_et' not in vwap_data.columns:
        return None

    # Build VWAP z-score lookup by time
    vwap_lookup = {}
    for _, vrow in vwap_data.iterrows():
        vwap_lookup[vrow['time_et']] = vrow['z_score']

    # Stream through bars
    if gap_dir == 'up':
        running_low = float('inf')
        running_low_time = None
        depth_met = False

        for _, bar in all_scan_bars.iterrows():
            bar_time = bar['time_et']

            # Update running low
            if bar['low'] < running_low:
                running_low = bar['low']
                running_low_time = bar_time

            # Check depth threshold
            depth = (od_peak - running_low) / adr
            if depth >= min_depth:
                depth_met = True

            # Check recovery (only after depth met and after running low time)
            if depth_met and bar_time > running_low_time:
                z = vwap_lookup.get(bar_time, None)
                if z is not None and z >= 0:
                    entry_price = bar['close']
                    sl_dist = entry_price - (running_low - 0.02 * adr)
                    if sl_dist <= 0 or sl_dist > 0.5 * adr:
                        return None
                    return {
                        'entry_price': entry_price,
                        'entry_time': bar_time,
                        'sl_dist': sl_dist,
                        'so_depth': depth,
                        'trade_dir': 'long',
                        'running_low': running_low,
                        'running_low_time': running_low_time,
                    }

        return None

    else:  # gap_dir == 'down'
        running_high = float('-inf')
        running_high_time = None
        depth_met = False

        for _, bar in all_scan_bars.iterrows():
            bar_time = bar['time_et']

            # Update running high
            if bar['high'] > running_high:
                running_high = bar['high']
                running_high_time = bar_time

            # Check depth
            depth = (running_high - od_peak) / adr
            if depth >= min_depth:
                depth_met = True

            # Check recovery
            if depth_met and bar_time > running_high_time:
                z = vwap_lookup.get(bar_time, None)
                if z is not None and z <= 0:
                    entry_price = bar['close']
                    sl_dist = (running_high + 0.02 * adr) - entry_price
                    if sl_dist <= 0 or sl_dist > 0.5 * adr:
                        return None
                    return {
                        'entry_price': entry_price,
                        'entry_time': bar_time,
                        'sl_dist': sl_dist,
                        'so_depth': depth,
                        'trade_dir': 'short',
                        'running_high': running_high,
                        'running_high_time': running_high_time,
                    }

        return None


def run_cat2_selloff(metadata, simulate_fn, label=''):
    """
    Run Sell-Off & Recovery strategy on all qualifying events.

    Parameters:
        metadata: DataFrame with gap_direction, adr_10, close_935, ticker, date
        simulate_fn: function(bars_rth, entry_price, sl_dist, trade_dir, adr,
                              entry_time, ticker, date) -> result dict
        label: 'IS' or 'OOS'

    Returns:
        list of result dicts
    """
    valid = metadata[
        metadata['close_935'].notna() &
        metadata['adr_10'].notna() &
        (metadata['adr_10'] > 0) &
        metadata['gap_direction'].notna()
    ].copy()

    results = []
    skipped = 0

    for _, row in valid.iterrows():
        ticker = row['ticker']
        date = str(row['date'])
        adr = row['adr_10']
        gap_dir = row['gap_direction']

        # Load RTH bars
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

        # Streaming detection
        info = detect_selloff_recovery_streaming(
            rth, vwap_data, gap_dir, adr,
            min_depth=0.10, selloff_end='10:30', recovery_deadline='11:00'
        )
        if info is None:
            continue

        # Simulate TRAIL_D from entry
        result = simulate_fn(rth, info['entry_price'], info['sl_dist'],
                             info['trade_dir'], adr,
                             entry_time=info['entry_time'],
                             ticker=ticker, date=date)
        if result is None:
            continue

        result['ticker'] = ticker
        result['date'] = date
        result['gap_dir'] = gap_dir
        result['so_depth'] = info['so_depth']
        result['entry_time_actual'] = info['entry_time']
        result['sl_dist'] = info['sl_dist']
        result['adr'] = adr
        result['strategy'] = '2.1_SellOff_Recovery'
        results.append(result)

    print(f"  {label} 2.1_SellOff_Recovery: N={len(results)} trades ({skipped} skipped)")
    return results
