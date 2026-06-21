"""
Phase 1: Fetch 1-second intraday data from Polygon for critical (ticker, date) pairs.
Window: 09:30-09:45 ET (first 15 minutes of RTH).
Resume-capable: skips existing parquet files.
"""
import pandas as pd
import numpy as np
import requests
import json
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from tqdm import tqdm

# ============================================================
# Config
# ============================================================
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import POLYGON_API_KEY

BASE_DIR = Path(__file__).parent.parent
RAW_1SEC_DIR = BASE_DIR / 'data' / 'raw_1sec'
PROGRESS_FILE = RAW_1SEC_DIR / '_progress.json'
MAX_WORKERS = 10
WINDOW_START_MS = None  # Computed per date
WINDOW_END_MS = None

# Eastern timezone offset handling
# Polygon returns timestamps in UTC milliseconds
# We need 09:30-09:45 ET which is:
#   EST (Nov-Mar): UTC-5 -> 14:30-14:45 UTC
#   EDT (Mar-Nov): UTC-4 -> 13:30-13:45 UTC


def is_dst(date_str):
    """Check if a date falls in US Eastern Daylight Time."""
    from datetime import datetime
    import calendar
    year, month, day = int(date_str[:4]), int(date_str[5:7]), int(date_str[8:10])

    if month < 3 or month > 11:
        return False
    if month > 3 and month < 11:
        return True

    # March: DST starts 2nd Sunday
    if month == 3:
        # Find 2nd Sunday
        first_day = calendar.weekday(year, 3, 1)
        second_sunday = 8 + (6 - first_day) % 7
        return day >= second_sunday

    # November: DST ends 1st Sunday
    if month == 11:
        first_day = calendar.weekday(year, 11, 1)
        first_sunday = 1 + (6 - first_day) % 7
        return day < first_sunday

    return False


def get_window_utc_ms(date_str, start_et="09:30:00", end_et="09:45:00"):
    """Convert ET time window to UTC milliseconds for a specific date."""
    import calendar
    offset_hours = 4 if is_dst(date_str) else 5

    y, m, d = int(date_str[:4]), int(date_str[5:7]), int(date_str[8:10])

    start_h, start_m, start_s = [int(x) for x in start_et.split(':')]
    end_h, end_m, end_s = [int(x) for x in end_et.split(':')]

    # Use calendar.timegm for UTC (not datetime.timestamp which uses local TZ!)
    start_ms = int(calendar.timegm((y, m, d, start_h + offset_hours, start_m, start_s, 0, 0, 0)) * 1000)
    end_ms = int(calendar.timegm((y, m, d, end_h + offset_hours, end_m, end_s, 0, 0, 0)) * 1000)

    return start_ms, end_ms


def fetch_1sec(ticker, date_str, api_key):
    """Fetch 1-second bars from Polygon for a specific ticker/date window."""
    out_dir = RAW_1SEC_DIR / ticker
    out_path = out_dir / f"{date_str}.parquet"

    # Skip if already exists
    if out_path.exists():
        return ('skip', ticker, date_str)

    url = (f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/second/"
           f"{date_str}/{date_str}"
           f"?adjusted=true&sort=asc&limit=50000&apiKey={api_key}")

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        results = data.get('results', [])
        if not results:
            return ('empty', ticker, date_str)

        # Parse results
        rows = []
        start_ms, end_ms = get_window_utc_ms(date_str, "09:30:00", "09:45:00")

        for bar in results:
            t = bar['t']
            # Filter to 09:30-09:45 ET window
            if t < start_ms or t >= end_ms:
                continue
            rows.append({
                'timestamp': t,
                'open': bar.get('o', np.nan),
                'high': bar.get('h', np.nan),
                'low': bar.get('l', np.nan),
                'close': bar.get('c', np.nan),
                'volume': bar.get('v', 0),
                'vwap': bar.get('vw', np.nan),
                'n_trades': bar.get('n', 0),
            })

        if not rows:
            return ('no_window', ticker, date_str)

        df = pd.DataFrame(rows)

        # Convert timestamp to datetime
        df['datetime_utc'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df['datetime_et'] = df['datetime_utc'].dt.tz_convert('US/Eastern')
        df['time_et'] = df['datetime_et'].dt.strftime('%H:%M:%S')

        # Save
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)

        return ('ok', ticker, date_str, len(df))

    except requests.exceptions.HTTPError as e:
        if resp.status_code == 429:
            time.sleep(1)
            return ('rate_limit', ticker, date_str)
        return ('http_error', ticker, date_str, str(e))
    except Exception as e:
        return ('error', ticker, date_str, str(e))


def load_progress():
    """Load progress tracking file."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {'done': 0, 'total': 0, 'skipped': 0, 'empty': 0, 'errors': 0}


def save_progress(progress):
    """Save progress tracking file."""
    RAW_1SEC_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


def main():
    print("=" * 70, file=sys.stderr)
    print("PHASE 1: FETCH 1-SECOND DATA FROM POLYGON", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    api_key = POLYGON_API_KEY
    if not api_key:
        print("ERROR: POLYGON_API_KEY not set!", file=sys.stderr)
        sys.exit(1)

    # Load metadata_v9 and filter to OD > 0.5 + with_gap
    meta = pd.read_parquet('data/metadata/metadata_v9.parquet')
    mask = (meta['od_strength'] > 0.5) & (meta['od_direction'] == 'with_gap')
    trades = meta[mask][['ticker', 'date']].copy()
    trades['date'] = trades['date'].astype(str)

    # Also include against_gap trades (D9 tests these too)
    mask_against = (meta['od_strength'] > 0.5) & (meta['od_direction'] == 'against_gap')
    trades_against = meta[mask_against][['ticker', 'date']].copy()
    trades_against['date'] = trades_against['date'].astype(str)

    all_pairs = pd.concat([trades, trades_against]).drop_duplicates()
    all_pairs = all_pairs.sort_values(['date', 'ticker']).reset_index(drop=True)

    print(f"\nTotal (ticker, date) pairs to fetch: {len(all_pairs)}", file=sys.stderr)
    print(f"  with_gap: {len(trades)}", file=sys.stderr)
    print(f"  against_gap: {len(trades_against)}", file=sys.stderr)
    print(f"  Unique after dedup: {len(all_pairs)}", file=sys.stderr)

    # Check how many already exist
    existing = 0
    to_fetch = []
    for _, row in all_pairs.iterrows():
        path = RAW_1SEC_DIR / row['ticker'] / f"{row['date']}.parquet"
        if path.exists():
            existing += 1
        else:
            to_fetch.append((row['ticker'], row['date']))

    print(f"\n  Already downloaded: {existing}", file=sys.stderr)
    print(f"  Still to fetch: {len(to_fetch)}", file=sys.stderr)

    if len(to_fetch) == 0:
        print("\nAll data already fetched! Nothing to do.", file=sys.stderr)
        save_progress({'done': len(all_pairs), 'total': len(all_pairs),
                       'skipped': existing, 'empty': 0, 'errors': 0})
        return

    # Fetch with parallel requests
    progress = load_progress()
    progress['total'] = len(all_pairs)

    ok_count = 0
    empty_count = 0
    error_count = 0
    rate_limited = []

    RAW_1SEC_DIR.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(fetch_1sec, ticker, date, api_key): (ticker, date)
            for ticker, date in to_fetch
        }

        pbar = tqdm(as_completed(futures), total=len(futures),
                     desc="Fetching 1sec", file=sys.stderr)

        for future in pbar:
            try:
                result = future.result()
                status = result[0]

                if status == 'ok':
                    ok_count += 1
                    bars = result[3]
                    pbar.set_postfix(ok=ok_count, empty=empty_count,
                                     err=error_count, last=f"{result[1]}/{result[2]} ({bars})")
                elif status == 'skip':
                    existing += 1
                elif status in ('empty', 'no_window'):
                    empty_count += 1
                elif status == 'rate_limit':
                    rate_limited.append((result[1], result[2]))
                else:
                    error_count += 1
                    if error_count <= 10:
                        print(f"  ERROR {result[1]}/{result[2]}: {result[3] if len(result) > 3 else 'unknown'}",
                              file=sys.stderr)

            except Exception as e:
                error_count += 1
                print(f"  FUTURE ERROR: {e}", file=sys.stderr)

    # Retry rate-limited ones sequentially
    if rate_limited:
        print(f"\nRetrying {len(rate_limited)} rate-limited requests...", file=sys.stderr)
        for ticker, date in tqdm(rate_limited, desc="Retrying", file=sys.stderr):
            time.sleep(0.5)
            result = fetch_1sec(ticker, date, api_key)
            if result[0] == 'ok':
                ok_count += 1
            elif result[0] in ('empty', 'no_window'):
                empty_count += 1
            else:
                error_count += 1

    # Save progress
    progress['done'] = existing + ok_count
    progress['skipped'] = existing
    progress['empty'] = empty_count
    progress['errors'] = error_count
    save_progress(progress)

    # Summary
    print(f"\n{'='*70}", file=sys.stderr)
    print(f"FETCH COMPLETE", file=sys.stderr)
    print(f"  Total pairs: {len(all_pairs)}", file=sys.stderr)
    print(f"  Already existed: {existing}", file=sys.stderr)
    print(f"  Newly fetched: {ok_count}", file=sys.stderr)
    print(f"  Empty/no data: {empty_count}", file=sys.stderr)
    print(f"  Errors: {error_count}", file=sys.stderr)
    print(f"  Coverage: {(existing + ok_count) / len(all_pairs) * 100:.1f}%", file=sys.stderr)
    print(f"{'='*70}", file=sys.stderr)


if __name__ == '__main__':
    main()
