"""
RVOL_30 Precompute Engine
=========================
RVOL_30 = Vol_heute_09:30-09:59 / mean(Vol_letzte_10_Tage_09:30-09:59)

For each (ticker, date): sliding 10-day window of 30min volumes.
Uses raw_1min files, RTH session only.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

RAW_1MIN_DIR = Path('data/raw_1min')


def _load_vol_30min(ticker, date_str):
    """Load volume sum for 09:30-09:59 from raw_1min."""
    path = RAW_1MIN_DIR / ticker / f"{date_str}.parquet"
    if not path.exists():
        return np.nan
    try:
        df = pd.read_parquet(path)
        rth = df[df['session'] == 'rth']
        first30 = rth[(rth['time_et'] >= '09:30') & (rth['time_et'] <= '09:59')]
        vol = first30['volume'].sum()
        return float(vol) if vol > 0 else np.nan
    except Exception:
        return np.nan


def precompute_rvol30(metadata):
    """
    Returns {(ticker, date_str): rvol_30_value}

    For each ticker: sort all available dates, compute sliding 10-day
    baseline of 30min volumes. RVOL_30 = today / mean(last 10).
    """
    result = {}

    # Group metadata by ticker to get required dates
    needed = defaultdict(set)
    for _, row in metadata.iterrows():
        needed[row['ticker']].add(str(row['date']))

    total_tickers = len(needed)
    for idx, (ticker, gap_dates) in enumerate(sorted(needed.items())):
        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"    RVOL30: ticker {idx+1}/{total_tickers} ({ticker})")

        # Get all available dates for this ticker (sorted)
        ticker_dir = RAW_1MIN_DIR / ticker
        if not ticker_dir.exists():
            continue

        all_files = sorted(ticker_dir.glob('*.parquet'))
        all_dates = [f.stem for f in all_files]

        if len(all_dates) == 0:
            continue

        # Load 30min volumes for all dates (cached per ticker)
        vol_cache = {}
        dates_needed_set = set(gap_dates)

        # Find which dates we need volumes for (gap dates + their lookback)
        dates_to_load = set()
        for gd in gap_dates:
            try:
                gd_idx = all_dates.index(gd)
            except ValueError:
                continue
            # Need this date + up to 10 prior dates
            start = max(0, gd_idx - 10)
            for i in range(start, gd_idx + 1):
                dates_to_load.add(all_dates[i])

        # Load volumes for needed dates
        for d in dates_to_load:
            vol_cache[d] = _load_vol_30min(ticker, d)

        # Compute RVOL_30 for each gap date
        for gd in gap_dates:
            try:
                gd_idx = all_dates.index(gd)
            except ValueError:
                continue

            today_vol = vol_cache.get(gd, np.nan)
            if np.isnan(today_vol):
                continue

            # Last 10 trading days before gap date
            lookback_dates = all_dates[max(0, gd_idx - 10):gd_idx]
            lookback_vols = [vol_cache.get(d, np.nan) for d in lookback_dates]
            lookback_vols = [v for v in lookback_vols if not np.isnan(v)]

            if len(lookback_vols) < 3:
                continue

            baseline = np.mean(lookback_vols)
            if baseline <= 0:
                continue

            result[(ticker, gd)] = today_vol / baseline

    return result
