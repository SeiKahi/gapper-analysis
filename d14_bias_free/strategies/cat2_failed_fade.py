"""
D14 Category 2.2: Streaming Failed Fade
=========================================
CRITICAL CHANGES vs D13 (2 look-ahead biases removed):

  FIX 1: gap_filled — D13 uses metadata gap_filled (EOD boolean) as pre-filter
         D14: Track gap fill bar-by-bar. Check at each bar whether gap has been
         filled BY THAT TIME. If gap fills during the confirmation window, abort.

  FIX 2: LOD/HOD — D13 uses .idxmin()/.idxmax() on 09:36-10:30 window
         (knows the absolute extreme of the entire window)
         D14: Running LOD/HOD bar-by-bar. Confirmation is checked against
         the running extreme at each bar.

Parameters (exact D13):
  - LOD/HOD window: 09:36-10:30
  - Confirmation: close > running_LOD + 0.10*ADR (GapUp)
  - Confirmation deadline: 11:30
  - SL buffer: 0.03 * ADR below running LOD
  - SL max: 0.50 * ADR
"""
import pandas as pd
import numpy as np
from pathlib import Path


RAW_DIR = Path('data/raw_1min')


def detect_failed_fade_streaming(rth, gap_dir, adr, prev_close):
    """
    Streaming Failed Fade detection with both bias fixes.

    KEY DIFFERENCES vs D13:
    1. gap_filled check is done bar-by-bar (not from metadata EOD value)
    2. LOD/HOD is a running minimum/maximum, not .idxmin()

    Logic (GapUp/Long):
    - For each bar 09:36-10:30: update running LOD, check gap fill
    - For each bar after running LOD until 11:30:
      - Keep checking gap fill (if fills, abort)
      - Keep updating running LOD (it can go lower)
      - Check confirmation: close > running_LOD + 0.10*ADR
    - Entry on first confirmation bar

    Parameters:
        rth: DataFrame with time_et, OHLC
        gap_dir: 'up' or 'down'
        adr: Average daily range
        prev_close: Previous day's close (for gap fill check)

    Returns:
        dict with entry_price, entry_time, sl_dist, trade_dir, etc. or None
    """
    if prev_close is None or np.isnan(prev_close) or adr <= 0:
        return None

    # All bars from 09:36 to 11:30 (combined search + confirmation window)
    scan_bars = rth[(rth['time_et'] >= '09:36') & (rth['time_et'] <= '11:30')]
    if len(scan_bars) < 5:
        return None

    if gap_dir == 'up':
        # Track: running LOD, gap fill status
        running_lod = float('inf')
        running_lod_time = None
        gap_filled = False
        lod_search_ended = False  # After 10:30, no more LOD updates

        for _, bar in scan_bars.iterrows():
            bar_time = bar['time_et']

            # Check gap fill (bar-by-bar)
            if bar['low'] <= prev_close:
                gap_filled = True
                return None  # Gap filled = not a failed fade

            # Update running LOD (only during 09:36-10:30)
            if bar_time <= '10:30':
                if bar['low'] < running_lod:
                    running_lod = bar['low']
                    running_lod_time = bar_time
            else:
                lod_search_ended = True

            # Check confirmation (must have a valid LOD and be after LOD time)
            if running_lod < float('inf') and bar_time > running_lod_time:
                confirmation_level = running_lod + 0.10 * adr
                if bar['close'] > confirmation_level:
                    entry_price = bar['close']
                    sl_dist = entry_price - (running_lod - 0.03 * adr)
                    if sl_dist <= 0 or sl_dist > 0.5 * adr:
                        return None
                    return {
                        'entry_price': entry_price,
                        'entry_time': bar_time,
                        'sl_dist': sl_dist,
                        'trade_dir': 'long',
                        'running_lod': running_lod,
                        'running_lod_time': running_lod_time,
                        'gap_filled_at_entry': False,
                    }

        return None

    else:  # gap_dir == 'down'
        running_hod = float('-inf')
        running_hod_time = None
        gap_filled = False

        for _, bar in scan_bars.iterrows():
            bar_time = bar['time_et']

            # Check gap fill
            if bar['high'] >= prev_close:
                gap_filled = True
                return None

            # Update running HOD (09:36-10:30)
            if bar_time <= '10:30':
                if bar['high'] > running_hod:
                    running_hod = bar['high']
                    running_hod_time = bar_time

            # Check confirmation
            if running_hod > float('-inf') and bar_time > running_hod_time:
                confirmation_level = running_hod - 0.10 * adr
                if bar['close'] < confirmation_level:
                    entry_price = bar['close']
                    sl_dist = (running_hod + 0.03 * adr) - entry_price
                    if sl_dist <= 0 or sl_dist > 0.5 * adr:
                        return None
                    return {
                        'entry_price': entry_price,
                        'entry_time': bar_time,
                        'sl_dist': sl_dist,
                        'trade_dir': 'short',
                        'running_hod': running_hod,
                        'running_hod_time': running_hod_time,
                        'gap_filled_at_entry': False,
                    }

        return None


def run_cat2_failed_fade(metadata, simulate_fn, label=''):
    """
    Run Failed Fade strategy on all qualifying events.

    D13 pre-filters: gap_filled=False (from metadata)
    D14: We still filter for od_direction='against_gap', but check gap_filled
         in real-time during bar-by-bar scanning.

    Parameters:
        metadata: DataFrame
        simulate_fn: TRAIL_D simulation function
        label: 'IS' or 'OOS'

    Returns:
        list of result dicts
    """
    # Filter: od_direction = against_gap (this is known at 09:35)
    # NOTE: We do NOT filter on gap_filled from metadata — that's the bias!
    valid = metadata[
        metadata['close_935'].notna() &
        metadata['adr_10'].notna() &
        (metadata['adr_10'] > 0) &
        (metadata['od_direction'] == 'against_gap') &
        metadata['gap_direction'].notna() &
        metadata['prev_close'].notna()
    ].copy()

    results = []
    skipped = 0

    for _, row in valid.iterrows():
        ticker = row['ticker']
        date = str(row['date'])
        adr = row['adr_10']
        gap_dir = row['gap_direction']
        prev_close = row['prev_close']

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

        # Streaming detection
        info = detect_failed_fade_streaming(rth, gap_dir, adr, prev_close)
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
        result['entry_time_actual'] = info['entry_time']
        result['sl_dist'] = info['sl_dist']
        result['adr'] = adr
        result['strategy'] = '2.2_Failed_Fade'
        results.append(result)

    print(f"  {label} 2.2_Failed_Fade: N={len(results)} trades ({skipped} skipped)")
    return results
