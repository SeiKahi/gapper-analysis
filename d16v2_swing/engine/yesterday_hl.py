"""
Yesterday High/Low Precompute Engine
=====================================
For each (ticker, date): find previous trading day's RTH high/low.
Uses sorted file listing from raw_1min to find "yesterday".
"""
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

RAW_1MIN_DIR = Path('data/raw_1min')


def precompute_yesterday_hl(metadata):
    """
    Returns {(ticker, date_str): (yesterday_high, yesterday_low)}

    For each ticker: sorted file list from raw_1min.
    For each gap_date: previous file = "yesterday".
    Load RTH bars, compute high/low.
    """
    result = {}

    # Group by ticker
    needed = defaultdict(set)
    for _, row in metadata.iterrows():
        needed[row['ticker']].add(str(row['date']))

    total_tickers = len(needed)
    for idx, (ticker, gap_dates) in enumerate(sorted(needed.items())):
        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"    Yesterday H/L: ticker {idx+1}/{total_tickers} ({ticker})")

        ticker_dir = RAW_1MIN_DIR / ticker
        if not ticker_dir.exists():
            continue

        all_files = sorted(ticker_dir.glob('*.parquet'))
        all_dates = [f.stem for f in all_files]

        if len(all_dates) < 2:
            continue

        # Cache: only load yesterday's data when needed
        hl_cache = {}

        for gd in gap_dates:
            try:
                gd_idx = all_dates.index(gd)
            except ValueError:
                continue

            if gd_idx == 0:
                continue  # no previous day

            yest_date = all_dates[gd_idx - 1]

            if yest_date not in hl_cache:
                yest_path = RAW_1MIN_DIR / ticker / f"{yest_date}.parquet"
                try:
                    df = pd.read_parquet(yest_path)
                    rth = df[df['session'] == 'rth']
                    if len(rth) > 0:
                        hl_cache[yest_date] = (rth['high'].max(), rth['low'].min())
                    else:
                        hl_cache[yest_date] = (np.nan, np.nan)
                except Exception:
                    hl_cache[yest_date] = (np.nan, np.nan)

            yh, yl = hl_cache[yest_date]
            if not np.isnan(yh) and not np.isnan(yl):
                result[(ticker, gd)] = (yh, yl)

    return result
