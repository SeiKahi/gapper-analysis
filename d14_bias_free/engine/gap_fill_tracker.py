"""
D14 Engine: Bar-by-Bar Gap Fill Tracking
=========================================
Replaces gap_filled (EOD boolean from metadata) with realtime tracking.

For GapUp: gap filled when bar['low'] <= prev_close
For GapDown: gap filled when bar['high'] >= prev_close

Returns fill time for any query point, enabling strategies to check
"was gap filled BY this time?" instead of "was gap filled at EOD?"
"""
import pandas as pd
import numpy as np


def track_gap_fill(rth_bars, gap_direction, prev_close):
    """
    Track gap fill bar-by-bar through RTH session.

    Parameters:
        rth_bars: DataFrame with time_et, open, high, low, close
        gap_direction: 'up' or 'down'
        prev_close: Previous day's close price

    Returns:
        dict with:
            'filled': bool (True if gap filled during session)
            'fill_time': str or None (time_et when gap first filled)
            'fill_bar_idx': int or None (index of fill bar)
    """
    if prev_close is None or np.isnan(prev_close):
        return {'filled': False, 'fill_time': None, 'fill_bar_idx': None}

    for idx, (_, bar) in enumerate(rth_bars.iterrows()):
        if gap_direction == 'up':
            if bar['low'] <= prev_close:
                return {
                    'filled': True,
                    'fill_time': bar['time_et'],
                    'fill_bar_idx': idx,
                }
        else:  # gap_direction == 'down'
            if bar['high'] >= prev_close:
                return {
                    'filled': True,
                    'fill_time': bar['time_et'],
                    'fill_bar_idx': idx,
                }

    return {'filled': False, 'fill_time': None, 'fill_bar_idx': None}


def is_gap_filled_at_time(rth_bars, gap_direction, prev_close, check_time):
    """
    Check if gap was filled by a specific time.

    Parameters:
        rth_bars: DataFrame with time_et, open, high, low, close
        gap_direction: 'up' or 'down'
        prev_close: Previous day's close price
        check_time: String 'HH:MM' — check if filled before/at this time

    Returns:
        bool: True if gap was filled at or before check_time
    """
    if prev_close is None or np.isnan(prev_close):
        return False

    bars_to_check = rth_bars[rth_bars['time_et'] <= check_time]
    for _, bar in bars_to_check.iterrows():
        if gap_direction == 'up' and bar['low'] <= prev_close:
            return True
        elif gap_direction == 'down' and bar['high'] >= prev_close:
            return True
    return False


def get_prev_close(metadata_row):
    """
    Extract prev_close from metadata row.
    prev_close = close_935 adjusted by gap:
      For GapUp: prev_close = close_935 / (1 + gap_pct)
      Simpler: use open - gap_size (where gap_size = gap_size_in_adr * adr_10)

    Actually, we compute it from the first bar's open and gap info:
      GapUp:  prev_close = rth_open - (gap_size_in_adr * adr_10) approximately
      But safer: use metadata's gap data directly.

    The most reliable method: prev_close is the price level the gap references.
    For GapUp: prev_close is below the open → gap fills when price drops to prev_close
    For GapDown: prev_close is above the open → gap fills when price rises to prev_close

    We can derive prev_close from the first bar's open:
      GapUp: prev_close = first_bar_open / (1 + gap_pct)
      But gap_pct isn't in metadata directly.

    Best approach: Use close_935 and gap_size_in_adr:
      The gap is measured from prev_close to open.
      prev_close ≈ open - gap_amount (for gap up)
      gap_amount ≈ gap_size_in_adr * adr_10
    """
    # This is a helper - the actual prev_close extraction happens in the strategy
    # code using the first RTH bar's open and gap info.
    pass


def compute_prev_close_from_bars(rth_bars, gap_direction, gap_size_in_adr, adr):
    """
    Compute prev_close from first RTH bar open and gap size.

    For GapUp: open > prev_close → prev_close = open - gap_amount
    For GapDown: open < prev_close → prev_close = open + gap_amount
    """
    if len(rth_bars) == 0:
        return np.nan

    rth_open = rth_bars.iloc[0]['open']
    gap_amount = gap_size_in_adr * adr

    if gap_direction == 'up':
        return rth_open - gap_amount
    else:
        return rth_open + gap_amount
