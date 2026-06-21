"""
Durchlauf 5.0 — DATA ENGINEER
Downloads earnings from Polygon, builds feature vectors, creates episode dataset.
"""
import os
from dotenv import load_dotenv
import sys
import time
import json
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# ── Config ──────────────────────────────────────────────────────────
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE_DIR = PROJECT_ROOT
os.chdir(BASE_DIR)
load_dotenv()

POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY')
POLYGON_BASE = 'https://api.polygon.io'

META_PATH = BASE_DIR / 'data' / 'metadata' / 'metadata_master.parquet'
EARNINGS_OUT = BASE_DIR / 'data' / 'metadata' / 'metadata_with_earnings.parquet'
EPISODES_OUT = BASE_DIR / 'data' / 'metadata' / 'rl_episodes.parquet'

# Only Half 1 for training
TRAIN_START = '2021-02-21'
TRAIN_END = '2022-12-31'
VAL_START = '2023-01-01'
VAL_END = '2023-12-31'

# ── PART 1: EARNINGS DOWNLOAD ───────────────────────────────────────

def fetch_earnings_for_ticker(ticker, api_key):
    """Fetch all earnings for a ticker from Polygon."""
    url = f'{POLYGON_BASE}/vX/reference/financials'
    params = {
        'ticker': ticker,
        'timeframe': 'quarterly',
        'order': 'desc',
        'limit': 50,
        'apiKey': api_key,
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            return data.get('results', [])
        elif resp.status_code == 403:
            return None  # Not available on this plan
        else:
            return []
    except Exception:
        return []


def fetch_earnings_v3(ticker, api_key):
    """Try the /v3/reference/earnings endpoint."""
    # This endpoint may not exist on all plans, try it
    url = f'{POLYGON_BASE}/v3/reference/earnings'
    params = {
        'ticker': ticker,
        'order': 'desc',
        'limit': 50,
        'apiKey': api_key,
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            return data.get('results', [])
        else:
            return None
    except Exception:
        return None


def download_earnings(meta_df):
    """Download earnings for all unique tickers in metadata."""
    tickers = sorted(meta_df['ticker'].unique())
    print(f"Downloading earnings for {len(tickers)} tickers...", file=sys.stderr)

    # Try v3 endpoint first with one ticker
    test = fetch_earnings_v3(tickers[0], POLYGON_API_KEY)
    if test is not None:
        print(f"Using /v3/reference/earnings endpoint", file=sys.stderr)
        use_v3 = True
    else:
        print(f"v3 not available, trying /vX/reference/financials...", file=sys.stderr)
        test2 = fetch_earnings_for_ticker(tickers[0], POLYGON_API_KEY)
        if test2 is None:
            print("WARNING: Earnings endpoints not available on this plan.", file=sys.stderr)
            print("Continuing without earnings data.", file=sys.stderr)
            return None
        use_v3 = False

    all_earnings = {}
    success = 0
    fail = 0

    for i, ticker in enumerate(tickers):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(tickers)} tickers ({success} with data)", file=sys.stderr)

        if use_v3:
            results = fetch_earnings_v3(ticker, POLYGON_API_KEY)
        else:
            results = fetch_earnings_for_ticker(ticker, POLYGON_API_KEY)

        if results:
            all_earnings[ticker] = results
            success += 1
        else:
            fail += 1

        # Small delay to be nice (even though unlimited)
        if i % 100 == 99:
            time.sleep(0.5)

    print(f"Done: {success} tickers with earnings, {fail} without", file=sys.stderr)
    return all_earnings


def parse_earnings_to_df(all_earnings):
    """Convert raw earnings JSON to a clean DataFrame."""
    rows = []
    for ticker, results in all_earnings.items():
        for r in results:
            if isinstance(r, dict):
                # v3 format
                row = {
                    'ticker': ticker,
                    'report_date': r.get('report_date', r.get('filing_date', r.get('end_date', ''))),
                    'actual_eps': r.get('actual_eps'),
                    'estimated_eps': r.get('estimated_eps'),
                    'eps_surprise': r.get('eps_surprise'),
                    'actual_revenue': r.get('actual_revenue'),
                    'estimated_revenue': r.get('estimated_revenue'),
                    'revenue_surprise': r.get('revenue_surprise'),
                    'time_of_report': r.get('time_of_report', ''),
                }
                # vX/financials format
                if 'financials' in r:
                    fin = r['financials']
                    if 'income_statement' in fin:
                        inc = fin['income_statement']
                        eps_data = inc.get('basic_earnings_per_share', {})
                        rev_data = inc.get('revenues', {})
                        row['actual_eps'] = eps_data.get('value')
                        row['actual_revenue'] = rev_data.get('value')
                    row['report_date'] = r.get('filing_date', r.get('end_date', ''))
                rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Clean dates
    df['report_date'] = pd.to_datetime(df['report_date'], errors='coerce')
    return df.dropna(subset=['report_date'])


def merge_earnings_with_metadata(meta_df, earnings_df):
    """For each gapper day, check if there were earnings."""
    if earnings_df is None or earnings_df.empty:
        print("No earnings data available. Setting all earnings features to default.", file=sys.stderr)
        meta_df['has_earnings'] = False
        meta_df['earnings_timing'] = 'none'
        meta_df['eps_surprise'] = 0.0
        meta_df['eps_surprise_pct'] = 0.0
        meta_df['revenue_surprise'] = 0.0
        meta_df['revenue_surprise_pct'] = 0.0
        meta_df['earnings_score'] = 0
        return meta_df

    meta_df['date_dt'] = pd.to_datetime(meta_df['date'])
    earnings_df['report_date'] = pd.to_datetime(earnings_df['report_date'])

    # For each gapper, check same day or previous day earnings
    has_earnings = []
    earnings_timing = []
    eps_surprise_list = []
    eps_surprise_pct_list = []
    rev_surprise_list = []
    rev_surprise_pct_list = []
    earnings_score_list = []

    for _, row in meta_df.iterrows():
        ticker = row['ticker']
        gapper_date = row['date_dt']
        prev_date = gapper_date - timedelta(days=1)
        # Skip weekends for prev_date
        if prev_date.weekday() == 6:  # Sunday
            prev_date -= timedelta(days=2)
        elif prev_date.weekday() == 5:  # Saturday
            prev_date -= timedelta(days=1)

        # Check same day or previous day
        mask = (earnings_df['ticker'] == ticker) & (
            (earnings_df['report_date'].dt.date == gapper_date.date()) |
            (earnings_df['report_date'].dt.date == prev_date.date())
        )
        matches = earnings_df[mask]

        if len(matches) > 0:
            match = matches.iloc[0]
            has_earnings.append(True)

            # Timing
            if match['report_date'].date() == gapper_date.date():
                timing = match.get('time_of_report', 'premarket')
                if pd.isna(timing) or timing == '':
                    earnings_timing.append('premarket')
                elif 'after' in str(timing).lower() or 'close' in str(timing).lower():
                    earnings_timing.append('afterhours_prev')
                else:
                    earnings_timing.append('premarket')
            else:
                earnings_timing.append('afterhours_prev')

            # EPS surprise
            eps_s = match.get('eps_surprise', np.nan)
            if pd.isna(eps_s):
                actual = match.get('actual_eps')
                estimated = match.get('estimated_eps')
                if pd.notna(actual) and pd.notna(estimated) and estimated != 0:
                    eps_s = actual - estimated
                else:
                    eps_s = 0.0
            eps_surprise_list.append(float(eps_s) if pd.notna(eps_s) else 0.0)

            # EPS surprise pct
            estimated = match.get('estimated_eps')
            if pd.notna(estimated) and estimated != 0 and pd.notna(eps_s):
                eps_surprise_pct_list.append(float(eps_s / abs(estimated)))
            else:
                eps_surprise_pct_list.append(0.0)

            # Revenue surprise
            rev_s = match.get('revenue_surprise', np.nan)
            if pd.isna(rev_s):
                actual_r = match.get('actual_revenue')
                estimated_r = match.get('estimated_revenue')
                if pd.notna(actual_r) and pd.notna(estimated_r) and estimated_r != 0:
                    rev_s = actual_r - estimated_r
                else:
                    rev_s = 0.0
            rev_surprise_list.append(float(rev_s) if pd.notna(rev_s) else 0.0)

            # Revenue surprise pct
            estimated_r = match.get('estimated_revenue')
            if pd.notna(estimated_r) and estimated_r != 0 and pd.notna(rev_s):
                rev_surprise_pct_list.append(float(rev_s / abs(estimated_r)))
            else:
                rev_surprise_pct_list.append(0.0)

            # Earnings score
            eps_pct = eps_surprise_pct_list[-1]
            if abs(eps_pct) <= 0.05:
                earnings_score_list.append(0)  # Meet
            elif eps_pct > 0.05:
                earnings_score_list.append(1)  # Beat
            else:
                earnings_score_list.append(-1)  # Miss
        else:
            has_earnings.append(False)
            earnings_timing.append('none')
            eps_surprise_list.append(0.0)
            eps_surprise_pct_list.append(0.0)
            rev_surprise_list.append(0.0)
            rev_surprise_pct_list.append(0.0)
            earnings_score_list.append(0)

    meta_df['has_earnings'] = has_earnings
    meta_df['earnings_timing'] = earnings_timing
    meta_df['eps_surprise'] = eps_surprise_list
    meta_df['eps_surprise_pct'] = eps_surprise_pct_list
    meta_df['revenue_surprise'] = rev_surprise_list
    meta_df['revenue_surprise_pct'] = rev_surprise_pct_list
    meta_df['earnings_score'] = earnings_score_list
    meta_df.drop(columns=['date_dt'], inplace=True, errors='ignore')

    return meta_df


# ── PART 2: FEATURE VECTOR BUILDING ─────────────────────────────────

def compute_pre_open_features(meta_row):
    """Compute features available before market open (features 1-10)."""
    features = {}

    # 1. gap_direction: +1 / -1
    features['gap_direction_num'] = 1.0 if meta_row.get('gap_direction') == 'up' else -1.0

    # 2. gap_size_adr
    features['gap_size_adr'] = float(meta_row.get('gap_size_in_adr', 0.0))
    if pd.isna(features['gap_size_adr']):
        features['gap_size_adr'] = abs(float(meta_row.get('gap_pct', 0.0))) / 5.0  # rough estimate

    # 3. adr_pct
    adr = meta_row.get('adr_10', np.nan)
    prev_close = meta_row.get('prev_close', np.nan)
    if pd.notna(adr) and pd.notna(prev_close) and prev_close > 0:
        features['adr_pct'] = float(adr / prev_close)
    else:
        features['adr_pct'] = 0.05  # default 5%

    # 4. prior_10d_return
    features['prior_10d_return'] = float(meta_row.get('prior_return_10d', 0.0))
    if pd.isna(features['prior_10d_return']):
        features['prior_10d_return'] = 0.0

    # 5. prior_30d_return (use 20d as proxy)
    features['prior_30d_return'] = float(meta_row.get('prior_return_20d', 0.0))
    if pd.isna(features['prior_30d_return']):
        features['prior_30d_return'] = 0.0

    # 6. rsi_14
    features['rsi_14'] = float(meta_row.get('rsi_14_prev', 50.0))
    if pd.isna(features['rsi_14']):
        features['rsi_14'] = 50.0  # neutral default
    features['rsi_14'] /= 100.0  # normalize to 0-1

    # 7. pm_volume_ratio (will be computed from raw data later)
    pm_vol = meta_row.get('premarket_volume', np.nan)
    if pd.notna(pm_vol) and pm_vol > 0:
        features['pm_volume_ratio'] = float(pm_vol) / 500000.0  # rough normalization
    else:
        features['pm_volume_ratio'] = 1.0

    # 8. pm_range_adr (computed from raw data later, use metadata proxy)
    features['pm_range_adr'] = 0.5  # default, overridden when raw data available

    # 9. has_earnings
    features['has_earnings'] = 1.0 if meta_row.get('has_earnings', False) else 0.0

    # 10. earnings_score
    features['earnings_score'] = float(meta_row.get('earnings_score', 0))

    return features


def compute_pm_features_from_raw(ticker, date, adr):
    """Compute premarket features from raw 1-min data."""
    raw_path = BASE_DIR / 'data' / 'raw_1min' / ticker / f'{date}.parquet'
    if not raw_path.exists():
        return None

    try:
        df = pd.read_parquet(raw_path)
        pm = df[df['session'] == 'premarket']
        if pm.empty:
            return None

        pm_high = pm['high'].max()
        pm_low = pm['low'].min()
        pm_volume = pm['volume'].sum()
        pm_range = pm_high - pm_low

        result = {}
        if adr > 0:
            result['pm_range_adr'] = pm_range / adr
        else:
            result['pm_range_adr'] = 0.5
        result['pm_volume_total'] = pm_volume
        result['pm_high'] = pm_high
        result['pm_low'] = pm_low

        return result
    except Exception:
        return None


def compute_post_open_features_at_time(rth_df, time_idx, adr, gap_dir, vwap_df=None):
    """
    Compute post-open features at a specific time index in the RTH bars.
    time_idx: index into rth_df (0=9:30, 5=9:35, etc.)
    Returns dict of features.
    """
    if time_idx >= len(rth_df):
        time_idx = len(rth_df) - 1

    features = {}
    bars = rth_df.iloc[:time_idx + 1]
    if len(bars) == 0:
        return {
            'od_strength_adr': 0.0, 'od_with_gap': 0.0,
            'rvol_at_time': 1.0, 'vwap_position': 0.0,
            'pct_adr_used': 0.0, 'vwap_broken': 0.0,
            'pm_rth30_ratio': 0.0, 'vwap_test_count': 0.0,
            'spy_return_current': 0.0,
        }

    open_price = rth_df.iloc[0]['open']
    current_close = bars.iloc[-1]['close']
    current_high = bars['high'].max()
    current_low = bars['low'].min()

    # 11. od_strength_adr: |Close at time - Open| / ADR
    if adr > 0:
        features['od_strength_adr'] = abs(current_close - open_price) / adr
    else:
        features['od_strength_adr'] = 0.0

    # 12. od_with_gap: +1 if OD in gap direction, -1 if against
    od_direction = 1.0 if current_close > open_price else -1.0
    features['od_with_gap'] = od_direction * gap_dir

    # 13. rvol_at_time (simplified: cumulative vol / expected vol)
    cum_vol = bars['volume'].sum()
    # Use 500k as rough baseline for first 30 min
    expected_vol = max(1, time_idx * 50000)  # rough approximation
    features['rvol_at_time'] = min(10.0, cum_vol / expected_vol)

    # 14. vwap_position (z-score)
    if vwap_df is not None and len(vwap_df) > time_idx:
        vwap_bar = vwap_df.iloc[time_idx]
        vwap_val = vwap_bar.get('vwap', current_close)
        std_val = vwap_bar.get('std_dev', 1.0)
        if pd.notna(vwap_val) and pd.notna(std_val) and std_val > 0:
            features['vwap_position'] = (current_close - vwap_val) / std_val
        else:
            features['vwap_position'] = 0.0
    else:
        # Compute VWAP from OHLCV
        typical = (bars['high'] + bars['low'] + bars['close']) / 3
        vol = bars['volume'].replace(0, 1)
        vwap_calc = (typical * vol).sum() / vol.sum()
        if adr > 0:
            features['vwap_position'] = (current_close - vwap_calc) / (adr * 0.3)
        else:
            features['vwap_position'] = 0.0
    features['vwap_position'] = np.clip(features['vwap_position'], -3.0, 3.0)

    # 15. pct_adr_used
    intraday_range = current_high - current_low
    if adr > 0:
        features['pct_adr_used'] = intraday_range / adr
    else:
        features['pct_adr_used'] = 0.0

    # 16. vwap_broken: 3+ consecutive closes on same side
    features['vwap_broken'] = 0.0
    if vwap_df is not None and len(bars) >= 3:
        try:
            recent_closes = bars['close'].values[-3:]
            recent_vwaps = []
            for j in range(max(0, time_idx - 2), time_idx + 1):
                if j < len(vwap_df):
                    recent_vwaps.append(vwap_df.iloc[j].get('vwap', np.nan))
            if len(recent_vwaps) == 3 and all(pd.notna(v) for v in recent_vwaps):
                above = all(c > v for c, v in zip(recent_closes, recent_vwaps))
                below = all(c < v for c, v in zip(recent_closes, recent_vwaps))
                if above or below:
                    features['vwap_broken'] = 1.0
        except Exception:
            pass

    # 17. pm_rth30_ratio (only available after 30 min)
    features['pm_rth30_ratio'] = 0.0  # set externally

    # 18. vwap_test_count
    features['vwap_test_count'] = 0.0
    if vwap_df is not None:
        test_count = 0
        last_test_idx = -6  # cooldown of 5 bars
        for j in range(min(time_idx + 1, len(vwap_df))):
            if j < len(bars) and j < len(vwap_df):
                bar_high = bars.iloc[j]['high']
                bar_low = bars.iloc[j]['low']
                vwap_val = vwap_df.iloc[j].get('vwap', np.nan)
                if pd.notna(vwap_val) and bar_low <= vwap_val <= bar_high:
                    if j - last_test_idx >= 5:
                        test_count += 1
                        last_test_idx = j
        features['vwap_test_count'] = float(test_count)

    # 19. spy_return_current (set externally from metadata)
    features['spy_return_current'] = 0.0

    return features


def build_episode_dataset(meta_df):
    """
    Build the complete episode dataset.
    Each row = one gapper day with pre-open features + metadata for post-open computation.
    """
    print("Building episode dataset...", file=sys.stderr)

    # Filter to Half 1 only
    meta_df['date'] = meta_df['date'].astype(str)
    half1 = meta_df[(meta_df['date'] >= TRAIN_START) & (meta_df['date'] <= VAL_END)].copy()
    print(f"Half 1: {len(half1)} gapper days", file=sys.stderr)

    episodes = []
    skipped = 0

    for idx, (_, row) in enumerate(half1.iterrows()):
        if idx % 500 == 0:
            print(f"  Processing {idx}/{len(half1)}...", file=sys.stderr)

        ticker = row['ticker']
        date = str(row['date'])[:10]
        adr = row.get('adr_10', np.nan)
        if pd.isna(adr) or adr <= 0:
            skipped += 1
            continue

        # Check raw data exists
        raw_path = BASE_DIR / 'data' / 'raw_1min' / ticker / f'{date}.parquet'
        if not raw_path.exists():
            skipped += 1
            continue

        # Pre-open features
        pre_feats = compute_pre_open_features(row)

        # PM features from raw
        pm_feats = compute_pm_features_from_raw(ticker, date, adr)
        if pm_feats:
            pre_feats['pm_range_adr'] = pm_feats['pm_range_adr']
            # pm_volume_ratio: normalize by median PM volume across all gappers
            pre_feats['pm_volume_total'] = pm_feats.get('pm_volume_total', 0)

        # Compute PM/RTH30 ratio for metadata
        try:
            raw_df = pd.read_parquet(raw_path)
            rth = raw_df[raw_df['session'] == 'rth'].reset_index(drop=True)
            pm = raw_df[raw_df['session'] == 'premarket']
            pm_vol = pm['volume'].sum() if len(pm) > 0 else 0
            rth30_vol = rth.iloc[:30]['volume'].sum() if len(rth) >= 30 else rth['volume'].sum()
            if rth30_vol > 0:
                pre_feats['pm_rth30_ratio_val'] = pm_vol / rth30_vol
            else:
                pre_feats['pm_rth30_ratio_val'] = 0.0
        except Exception:
            pre_feats['pm_rth30_ratio_val'] = 0.0

        episode = {
            'ticker': ticker,
            'date': date,
            'adr': float(adr),
            'split': 'train' if date <= TRAIN_END else 'val',
            'spy_return_day': float(row.get('spy_return_day', 0.0)) if pd.notna(row.get('spy_return_day')) else 0.0,
            'prev_close': float(row.get('prev_close', 0.0)) if pd.notna(row.get('prev_close')) else 0.0,
            'today_open': float(row.get('today_open', 0.0)) if pd.notna(row.get('today_open')) else 0.0,
            'rth_open': float(row.get('rth_open', 0.0)) if pd.notna(row.get('rth_open')) else 0.0,
        }
        episode.update(pre_feats)
        episodes.append(episode)

    print(f"Built {len(episodes)} episodes, skipped {skipped}", file=sys.stderr)

    df = pd.DataFrame(episodes)

    # Normalize pm_volume_ratio by median
    if 'pm_volume_total' in df.columns:
        median_pm = df['pm_volume_total'].median()
        if median_pm > 0:
            df['pm_volume_ratio'] = df['pm_volume_total'] / median_pm
        df['pm_volume_ratio'] = df['pm_volume_ratio'].clip(0, 10)
        df.drop(columns=['pm_volume_total'], inplace=True, errors='ignore')

    return df


# ── MAIN ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60, file=sys.stderr)
    print("DURCHLAUF 5.0 — DATA ENGINEER", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    # Load metadata
    meta = pd.read_parquet(META_PATH)
    print(f"Loaded metadata: {len(meta)} rows", file=sys.stderr)

    # ── Aufgabe 0: Earnings ──
    print("\n--- AUFGABE 0: EARNINGS DOWNLOAD ---", file=sys.stderr)
    earnings_data = download_earnings(meta)

    if earnings_data:
        earnings_df = parse_earnings_to_df(earnings_data)
        print(f"Parsed {len(earnings_df)} earnings records", file=sys.stderr)
    else:
        earnings_df = None

    meta = merge_earnings_with_metadata(meta, earnings_df)

    # Save metadata with earnings
    meta.to_parquet(EARNINGS_OUT, index=False)
    earnings_count = meta['has_earnings'].sum()
    print(f"Saved metadata_with_earnings.parquet: {earnings_count} gappers with earnings", file=sys.stderr)

    # ── Feature Vector ──
    print("\n--- FEATURE VECTOR BUILD ---", file=sys.stderr)
    episodes = build_episode_dataset(meta)

    # Stats
    train_n = (episodes['split'] == 'train').sum()
    val_n = (episodes['split'] == 'val').sum()
    print(f"\nEpisode dataset:", file=sys.stderr)
    print(f"  Train: {train_n}", file=sys.stderr)
    print(f"  Val: {val_n}", file=sys.stderr)
    print(f"  Total: {len(episodes)}", file=sys.stderr)

    # Save
    episodes.to_parquet(EPISODES_OUT, index=False)
    print(f"Saved rl_episodes.parquet", file=sys.stderr)

    # Write summary
    os.makedirs(BASE_DIR / 'results', exist_ok=True)
    with open(BASE_DIR / 'results' / 'rl_data_engineer.txt', 'w') as f:
        f.write("DURCHLAUF 5.0 — DATA ENGINEER REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Metadata: {len(meta)} gappers total\n")
        f.write(f"Earnings: {earnings_count} gappers with earnings ({earnings_count/len(meta)*100:.1f}%)\n")
        f.write(f"Episodes: {len(episodes)} (Train={train_n}, Val={val_n})\n\n")
        f.write("Feature columns:\n")
        feat_cols = [c for c in episodes.columns if c not in ['ticker', 'date', 'split', 'adr',
                     'spy_return_day', 'prev_close', 'today_open', 'rth_open', 'pm_rth30_ratio_val']]
        for c in feat_cols:
            f.write(f"  {c}: mean={episodes[c].mean():.4f}, std={episodes[c].std():.4f}, "
                   f"min={episodes[c].min():.4f}, max={episodes[c].max():.4f}\n")

        f.write(f"\nEarnings breakdown:\n")
        f.write(f"  has_earnings: {meta['has_earnings'].sum()}\n")
        f.write(f"  earnings_score +1 (beat): {(meta['earnings_score'] == 1).sum()}\n")
        f.write(f"  earnings_score 0 (meet): {((meta['earnings_score'] == 0) & meta['has_earnings']).sum()}\n")
        f.write(f"  earnings_score -1 (miss): {(meta['earnings_score'] == -1).sum()}\n")

    print("\nDone! Check results/rl_data_engineer.txt", file=sys.stderr)


if __name__ == '__main__':
    main()
