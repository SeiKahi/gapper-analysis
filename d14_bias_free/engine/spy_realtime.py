"""
D14 Engine: SPY Open-Return (Realtime-safe)
============================================
Replaces spy_return_day (EOD SPY return) with spy_return_935:
  spy_return_935 = (SPY_open - SPY_prev_close) / SPY_prev_close

This is known at 09:30 and doesn't require future information.

Data source: data/daily/SPY/{date}_context.parquet
  Columns: date, open, high, low, close, volume, vwap, transactions, ticker
"""
import pandas as pd
import numpy as np
from pathlib import Path


SPY_DIR = Path('data/daily/SPY')


def build_spy_return_935(dates):
    """
    Build spy_return_935 for all unique dates.

    Uses SPY open on gap day vs previous day's close as the signal
    (available at 09:30, no look-ahead).

    Parameters:
        dates: iterable of date strings ('YYYY-MM-DD')

    Returns:
        dict: {date_str: spy_return_935} (float, as % return)
    """
    unique_dates = sorted(set(str(d) for d in dates))

    # Load all available SPY daily data
    spy_data = {}
    for f in SPY_DIR.glob('*_context.parquet'):
        date_str = f.stem.replace('_context', '')
        try:
            df = pd.read_parquet(f)
            if len(df) > 0:
                row = df.iloc[0]
                spy_data[date_str] = {
                    'open': float(row['open']),
                    'close': float(row['close']),
                }
        except Exception:
            continue

    # Sort dates to find previous trading day
    all_spy_dates = sorted(spy_data.keys())
    date_to_idx = {d: i for i, d in enumerate(all_spy_dates)}

    result = {}
    for d in unique_dates:
        if d not in date_to_idx:
            result[d] = np.nan
            continue

        idx = date_to_idx[d]
        if idx == 0:
            result[d] = np.nan
            continue

        prev_date = all_spy_dates[idx - 1]
        prev_close = spy_data[prev_date]['close']
        today_open = spy_data[d]['open']

        if prev_close > 0:
            result[d] = (today_open - prev_close) / prev_close
        else:
            result[d] = np.nan

    return result


def get_spy_open_return(spy_returns_dict, date):
    """Get SPY open return for a specific date."""
    return spy_returns_dict.get(str(date), np.nan)
