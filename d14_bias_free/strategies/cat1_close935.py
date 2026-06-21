"""
D14 Category 1: Close-at-935 Strategies (9 Strategies)
=======================================================
All strategies: Entry = close_935, Exit = TRAIL_D (streaming, bias-free)
Exact D13 parameters — NO new optimization.

Look-Ahead Fixes:
  - All: Intra-bar resolution (SL before peak)
  - 1.4: spy_return_935 replaces spy_return_day

Strategies:
  1.1 S4 Optimiert:       od>=0.70, gap>=2.0, pm_rth5<0.30
  1.2 Pre-10am Momentum:  GapUp, od>=0.50, gap>=2.0, rvol>=3, rsi>=50, MomConfirm
  1.3 Earnings+OD+RVOL:   GapUp, earnings=True, od>=0.50, rvol>=5
  1.4 Power Setup:        GapUp, od>=0.80, rvol>=5, spy_return_935>0
  1.5 Big First Candle:   GapUp, od>=0.50, first_candle_size>0.20
  1.6 Low Wick:           GapUp, od>=0.50, wick_ratio<0.15
  1.7 Big Body:           GapUp, od>=0.50, body_pct>0.60
  1.8 Gap 3.0+ ADR:       gap>=3.0, od>=0.50, od=with_gap
  1.9 RVOL>=5+RSI>50:     GapUp, rvol>=5, rsi>50
"""
import pandas as pd
import numpy as np
from pathlib import Path


RAW_DIR = Path('data/raw_1min')


def get_cat1_strategies():
    """Return list of (name, filter_fn, sl_mult) for all Cat-1 strategies."""
    return [
        ('1.1_S4_Optimiert', filter_1_1, 0.25),
        ('1.2_Pre10am_Momentum', filter_1_2, 0.15),
        ('1.3_Earnings_OD_RVOL', filter_1_3, 0.25),
        ('1.4_Power_Setup', filter_1_4, 0.25),
        ('1.5_Big_First_Candle', filter_1_5, 0.25),
        ('1.6_Low_Wick', filter_1_6, 0.25),
        ('1.7_Big_Body', filter_1_7, 0.25),
        ('1.8_Gap_3ADR', filter_1_8, 0.25),
        ('1.9_RVOL5_RSI50', filter_1_9, 0.25),
    ]


def _base_filter(df):
    """Base filter for all Cat-1 strategies."""
    return (
        df['close_935'].notna() &
        df['adr_10'].notna() &
        (df['adr_10'] > 0) &
        df['od_strength'].notna()
    )


def filter_1_1(df, spy_returns=None):
    """S4 Optimiert: od>=0.70, gap>=2.0, pm_rth5<0.30, od=with_gap, both directions."""
    mask = _base_filter(df)
    mask &= (df['od_direction'] == 'with_gap')
    mask &= (df['od_strength'] >= 0.70)
    mask &= (df['gap_size_in_adr'] >= 2.0)
    mask &= (df['pm_rth5'] < 0.30)
    return mask


def filter_1_2(df, spy_returns=None):
    """Pre-10am Momentum: GapUp, od>=0.50, gap>=2.0, rvol>=3, rsi>=50, MomConfirm."""
    mask = _base_filter(df)
    mask &= (df['gap_direction'] == 'up')
    mask &= (df['od_direction'] == 'with_gap')
    mask &= (df['od_strength'] >= 0.50)
    mask &= (df['gap_size_in_adr'] >= 2.0)
    mask &= (df['rvol_5'] >= 3)
    mask &= (df['rsi_14_prev'] >= 50)
    mask &= (df['first_candle_dir'] == 'with_gap')  # MomConfirm
    return mask


def filter_1_3(df, spy_returns=None):
    """Earnings+OD+RVOL: GapUp, earnings=True, od>=0.50, rvol>=5."""
    mask = _base_filter(df)
    mask &= (df['gap_direction'] == 'up')
    mask &= (df['od_direction'] == 'with_gap')
    mask &= (df['od_strength'] >= 0.50)
    mask &= (df['is_earnings'] == True)
    mask &= (df['rvol_5'] >= 5)
    return mask


def filter_1_4(df, spy_returns=None):
    """Power Setup: GapUp, od>=0.80, rvol>=5, spy_return_935>0.
    FIX: Uses spy_return_935 (open return) instead of spy_return_day (EOD).
    """
    mask = _base_filter(df)
    mask &= (df['gap_direction'] == 'up')
    mask &= (df['od_direction'] == 'with_gap')
    mask &= (df['od_strength'] >= 0.80)
    mask &= (df['rvol_5'] >= 5)
    # SPY fix: use realtime spy_return_935
    if spy_returns is not None:
        spy_vals = df['date'].astype(str).map(spy_returns)
        mask &= (spy_vals > 0)
    else:
        # Fallback (should not happen in D14)
        mask &= (df['spy_return_day'] > 0)
    return mask


def filter_1_5(df, spy_returns=None):
    """Big First Candle: GapUp, od>=0.50, first_candle_size>0.20."""
    mask = _base_filter(df)
    mask &= (df['gap_direction'] == 'up')
    mask &= (df['od_direction'] == 'with_gap')
    mask &= (df['od_strength'] >= 0.50)
    mask &= (df['first_candle_size'] > 0.20)
    return mask


def filter_1_6(df, spy_returns=None):
    """Low Wick: GapUp, od>=0.50, wick_ratio<0.15."""
    mask = _base_filter(df)
    mask &= (df['gap_direction'] == 'up')
    mask &= (df['od_direction'] == 'with_gap')
    mask &= (df['od_strength'] >= 0.50)
    mask &= (df['od_wick_ratio'] < 0.15)
    return mask


def filter_1_7(df, spy_returns=None):
    """Big Body: GapUp, od>=0.50, body_pct>0.60."""
    mask = _base_filter(df)
    mask &= (df['gap_direction'] == 'up')
    mask &= (df['od_direction'] == 'with_gap')
    mask &= (df['od_strength'] >= 0.50)
    mask &= (df['od_body_pct'] > 0.60)
    return mask


def filter_1_8(df, spy_returns=None):
    """Gap 3.0+ ADR: gap>=3.0, od>=0.50, od=with_gap, both directions."""
    mask = _base_filter(df)
    mask &= (df['od_direction'] == 'with_gap')
    mask &= (df['gap_size_in_adr'] >= 3.0)
    mask &= (df['od_strength'] >= 0.50)
    return mask


def filter_1_9(df, spy_returns=None):
    """RVOL>=5+RSI>50: GapUp, rvol>=5, rsi>50."""
    mask = _base_filter(df)
    mask &= (df['gap_direction'] == 'up')
    mask &= (df['rvol_5'] >= 5)
    mask &= (df['rsi_14_prev'] > 50)
    return mask


def run_cat1_strategies(metadata, spy_returns, simulate_fn, label=''):
    """
    Run all 9 Cat-1 strategies.

    Parameters:
        metadata: DataFrame with all metadata columns
        spy_returns: dict {date_str: spy_return_935}
        simulate_fn: function(bars_rth, entry_price, sl_dist, trade_dir, adr,
                              entry_time, ticker, date) -> result dict
        label: 'IS' or 'OOS' for logging

    Returns:
        dict: {strategy_name: list of result dicts}
    """
    strategies = get_cat1_strategies()
    results = {}

    for strat_name, filter_fn, sl_mult in strategies:
        mask = filter_fn(metadata, spy_returns)
        subset = metadata[mask].copy()
        trade_results = []
        skipped = 0

        for _, row in subset.iterrows():
            ticker = row['ticker']
            date = str(row['date'])
            adr = row['adr_10']
            entry_price = row['close_935']
            gap_dir = row['gap_direction']
            trade_dir = 'long' if gap_dir == 'up' else 'short'
            sl_dist = sl_mult * adr

            if sl_dist <= 0 or pd.isna(entry_price):
                skipped += 1
                continue

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

            result = simulate_fn(rth, entry_price, sl_dist, trade_dir, adr,
                                 entry_time='09:36', ticker=ticker, date=date)
            if result is None:
                skipped += 1
                continue

            result['ticker'] = ticker
            result['date'] = date
            result['gap_dir'] = gap_dir
            result['strategy'] = strat_name
            trade_results.append(result)

        results[strat_name] = trade_results
        n = len(trade_results)
        print(f"  {label} {strat_name}: N={n} trades ({skipped} skipped)")

    return results
