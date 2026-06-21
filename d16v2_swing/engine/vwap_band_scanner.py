"""
VWAP Band Scanner Engine
=========================
Loads pre-computed VWAP band data + raw_1min bars.
Checks:
  1. Was +/-2sigma touched after 11:00?
  2. Where is close at 15:58 relative to bands?

No look-ahead: VWAP bands are cumulative (each bar uses only data up to itself).
"""
import pandas as pd
import numpy as np
from pathlib import Path

VWAP_DIR = Path('data/vwap')
RAW_1MIN_DIR = Path('data/raw_1min')


def scan_vwap_bands(ticker, date):
    """
    Load pre-computed VWAP data + raw_1min for high/low.
    Check touch conditions and 15:58 position.

    Returns:
        dict with:
            touch_upper_2std: bool (bar.high >= upper_2std after 11:00)
            touch_lower_2std: bool (bar.low <= lower_2std after 11:00)
            close_1558: float (close price at 15:58)
            upper_1std_1558: float
            lower_1std_1558: float
            upper_2std_1558: float
            lower_2std_1558: float
        or None if data missing
    """
    vwap_path = VWAP_DIR / ticker / f"{date}.parquet"
    raw_path = RAW_1MIN_DIR / ticker / f"{date}.parquet"

    if not vwap_path.exists() or not raw_path.exists():
        return None

    try:
        vwap_df = pd.read_parquet(vwap_path)
        raw_df = pd.read_parquet(raw_path)
    except Exception:
        return None

    # Get RTH bars from raw_1min (for high/low)
    rth = raw_df[raw_df['session'] == 'rth'].copy()
    if len(rth) == 0:
        return None

    # Merge VWAP bands onto raw_1min by time_et
    merged = rth.merge(
        vwap_df[['time_et', 'upper_1std', 'lower_1std', 'upper_2std', 'lower_2std']],
        on='time_et', how='left'
    )

    # Filter after 11:00 for touch detection
    after_11 = merged[merged['time_et'] >= '11:00'].dropna(
        subset=['upper_2std', 'lower_2std']
    )

    if len(after_11) == 0:
        return None

    # Touch detection: bar.high >= upper_2std or bar.low <= lower_2std
    touch_upper_2std = (after_11['high'] >= after_11['upper_2std']).any()
    touch_lower_2std = (after_11['low'] <= after_11['lower_2std']).any()

    # 15:58 bar position
    bar_1558 = merged[merged['time_et'] == '15:58']
    if len(bar_1558) == 0:
        return None

    bar_1558 = bar_1558.iloc[0]

    # Check for NaN band values at 15:58
    if pd.isna(bar_1558['upper_1std']) or pd.isna(bar_1558['lower_1std']):
        return None

    return {
        'touch_upper_2std': bool(touch_upper_2std),
        'touch_lower_2std': bool(touch_lower_2std),
        'close_1558': float(bar_1558['close']),
        'upper_1std_1558': float(bar_1558['upper_1std']),
        'lower_1std_1558': float(bar_1558['lower_1std']),
        'upper_2std_1558': float(bar_1558['upper_2std']),
        'lower_2std_1558': float(bar_1558['lower_2std']),
    }
