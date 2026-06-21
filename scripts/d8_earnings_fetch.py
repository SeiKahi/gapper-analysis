"""
D8.0 Aufgabe 0: Earnings-Daten beschaffen
==========================================
- Extrahiert unique Ticker aus metadata_v7
- Holt Earnings-Dates via Polygon vX/reference/financials (filing_date)
- Erstellt earnings_dates.parquet
- Merged zu metadata_v8.parquet mit is_earnings Flag
- Plausibilitaets-Checks
"""

import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from pathlib import Path
from polygon import RESTClient
import time
import sys
from tqdm import tqdm
from datetime import datetime, timedelta

load_dotenv()

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
META_DIR = DATA_DIR / "metadata"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

API_KEY = os.getenv("POLYGON_API_KEY")

# IS/OOS split dates
IS_END = "2023-12-31"
OOS_START = "2024-01-01"

def fetch_earnings_dates(tickers, output_path):
    """Fetch filing dates for all tickers via Polygon financials API."""
    client = RESTClient(API_KEY)

    all_records = []
    success_count = 0
    no_data_count = 0
    error_count = 0
    error_tickers = []

    for ticker in tqdm(tickers, desc="Fetching earnings", file=sys.stderr):
        try:
            results = list(client.vx.list_stock_financials(
                ticker=ticker,
                limit=100,
                # Get all filings
            ))

            if not results:
                no_data_count += 1
                continue

            ticker_records = []
            for r in results:
                # Skip TTM (trailing twelve months) - not a real filing
                if r.fiscal_period == 'TTM':
                    continue

                filing_date = r.filing_date
                if filing_date is None:
                    # For Q4 without filing_date, try to use FY filing
                    continue

                ticker_records.append({
                    'ticker': ticker,
                    'earnings_date': filing_date,
                    'fiscal_period': r.fiscal_period,
                    'fiscal_year': r.fiscal_year,
                    'end_date': r.end_date,
                })

            if ticker_records:
                all_records.extend(ticker_records)
                success_count += 1
            else:
                no_data_count += 1

            # Rate limiting: ~5 requests per second
            time.sleep(0.22)

        except Exception as e:
            error_count += 1
            error_tickers.append((ticker, str(e)))
            # Don't stop on individual errors
            time.sleep(0.5)
            continue

    df = pd.DataFrame(all_records)
    if len(df) > 0:
        df['earnings_date'] = pd.to_datetime(df['earnings_date'])
        df = df.sort_values(['ticker', 'earnings_date']).reset_index(drop=True)

    df.to_parquet(output_path, index=False)

    return df, success_count, no_data_count, error_count, error_tickers


def compute_trading_day_before(dates_series, all_trading_days):
    """For each date, find the previous trading day."""
    td_set = set(all_trading_days)
    td_sorted = sorted(all_trading_days)

    result = {}
    for d in td_sorted:
        # Find previous trading day
        prev = d - timedelta(days=1)
        while prev not in td_set and prev > td_sorted[0] - timedelta(days=10):
            prev -= timedelta(days=1)
        result[d] = prev
    return result


def create_metadata_v8(meta_v7, earnings_df, output_path):
    """Merge earnings data with metadata to create is_earnings flag."""

    meta = meta_v7.copy()
    meta['date_dt'] = pd.to_datetime(meta['date'])

    # Get all unique trading days from metadata
    all_trading_days = sorted(meta['date_dt'].unique())

    # Build set of (ticker, earnings_date) for fast lookup
    if len(earnings_df) > 0:
        earnings_dates_set = set()
        for _, row in earnings_df.iterrows():
            earnings_dates_set.add((row['ticker'], row['earnings_date']))

        # Tickers that have ANY earnings data
        tickers_with_data = set(earnings_df['ticker'].unique())
    else:
        earnings_dates_set = set()
        tickers_with_data = set()

    # All unique tickers in metadata
    all_tickers = set(meta['ticker'].unique())

    # For each gapper row, check if:
    # - filing_date == gapper_date (BMO: reported before market, gap happens same day)
    # - filing_date == gapper_date - 1 trading day (AMC: reported after close, gap next day)
    # Since we don't have BMO/AMC timing, we use both conditions.

    is_earnings = []
    earnings_unknown = []
    nearest_earnings = []

    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Computing is_earnings", file=sys.stderr):
        ticker = row['ticker']
        gapper_date = row['date_dt']

        if ticker not in tickers_with_data:
            # No earnings data for this ticker
            is_earnings.append(False)
            earnings_unknown.append(True)
            nearest_earnings.append(pd.NaT)
            continue

        # Check: filing_date == gapper_date (BMO)
        match_same_day = (ticker, gapper_date) in earnings_dates_set

        # Check: filing_date == gapper_date - 1 trading day (AMC previous day)
        # Find previous trading day
        prev_day = gapper_date - timedelta(days=1)
        while prev_day.weekday() >= 5:  # Skip weekends
            prev_day -= timedelta(days=1)
        # Also check 2-3 days back for long weekends
        candidates = [prev_day]
        prev_day2 = prev_day - timedelta(days=1)
        while prev_day2.weekday() >= 5:
            prev_day2 -= timedelta(days=1)
        candidates.append(prev_day2)

        match_prev_day = any((ticker, pd.Timestamp(c)) in earnings_dates_set for c in candidates)

        if match_same_day or match_prev_day:
            is_earnings.append(True)
            earnings_unknown.append(False)
        else:
            is_earnings.append(False)
            earnings_unknown.append(False)

        # Find nearest earnings date
        ticker_earnings = earnings_df[earnings_df['ticker'] == ticker]['earnings_date'].values
        if len(ticker_earnings) > 0:
            diffs = np.abs(ticker_earnings - gapper_date.to_numpy())
            nearest_idx = np.argmin(diffs)
            nearest_earnings.append(pd.Timestamp(ticker_earnings[nearest_idx]))
        else:
            nearest_earnings.append(pd.NaT)

    meta['is_earnings'] = is_earnings
    meta['earnings_unknown'] = earnings_unknown
    meta['nearest_earnings_date'] = nearest_earnings

    # Drop helper column
    meta = meta.drop(columns=['date_dt'])

    meta.to_parquet(output_path, index=False)
    return meta


def plausibility_check(meta_v8, out_file):
    """Check that earnings flag makes sense."""
    lines = []
    lines.append("=" * 70)
    lines.append("AUFGABE 0: EARNINGS-DATEN PLAUSIBILITAETS-CHECK")
    lines.append("=" * 70)

    # Overall distribution
    lines.append("\n--- 0c: Verteilung ---")

    for label, mask in [("ALL", meta_v8.index == meta_v8.index),
                        ("GapUp", meta_v8['gap_direction'] == 'up'),
                        ("GapDown", meta_v8['gap_direction'] == 'down')]:
        sub = meta_v8[mask]
        n_earn = sub['is_earnings'].sum()
        n_non = (~sub['is_earnings'] & ~sub['earnings_unknown']).sum()
        n_unk = sub['earnings_unknown'].sum()
        lines.append(f"\n  {label} (N={len(sub)}):")
        lines.append(f"    Earnings:     {n_earn:5d} ({100*n_earn/len(sub):.1f}%)")
        lines.append(f"    Non-Earnings: {n_non:5d} ({100*n_non/len(sub):.1f}%)")
        lines.append(f"    Unknown:      {n_unk:5d} ({100*n_unk/len(sub):.1f}%)")

    # 0d: Plausibility - Earnings should have larger gaps and higher RVOL
    lines.append("\n--- 0d: Plausibilitaet (Earnings vs Non-Earnings) ---")

    earn_mask = meta_v8['is_earnings'] == True
    non_earn_mask = (~meta_v8['is_earnings']) & (~meta_v8['earnings_unknown'])

    for metric, col in [("Gap in ADR", "gap_size_in_adr"),
                        ("Gap %", "gap_pct"),
                        ("RVOL 30min", "rvol_at_time_30min"),
                        ("RVOL 5min", "rvol_5"),
                        ("ADR 10", "adr_10" if "adr_10" in meta_v8.columns else "adr_5")]:
        if col not in meta_v8.columns:
            continue
        earn_med = meta_v8.loc[earn_mask, col].median()
        non_med = meta_v8.loc[non_earn_mask, col].median()
        lines.append(f"\n  {metric}:")
        lines.append(f"    Earnings median:     {earn_med:.4f}")
        lines.append(f"    Non-Earnings median: {non_med:.4f}")
        lines.append(f"    Ratio (E/NE):        {earn_med/non_med:.2f}x" if non_med != 0 else "    Ratio: N/A")

    # IS / OOS split
    lines.append("\n--- IS/OOS Split ---")
    meta_v8_dt = pd.to_datetime(meta_v8['date'])
    is_mask = meta_v8_dt <= IS_END
    oos_mask = meta_v8_dt >= OOS_START

    for period, pmask in [("IS (H1)", is_mask), ("OOS (H2)", oos_mask)]:
        sub = meta_v8[pmask]
        n_earn = sub['is_earnings'].sum()
        n_non = (~sub['is_earnings'] & ~sub['earnings_unknown']).sum()
        n_unk = sub['earnings_unknown'].sum()
        lines.append(f"\n  {period} (N={len(sub)}):")
        lines.append(f"    Earnings:     {n_earn:5d} ({100*n_earn/len(sub):.1f}%)")
        lines.append(f"    Non-Earnings: {n_non:5d} ({100*n_non/len(sub):.1f}%)")
        lines.append(f"    Unknown:      {n_unk:5d} ({100*n_unk/len(sub):.1f}%)")

    text = "\n".join(lines)
    with open(out_file, 'w') as f:
        f.write(text)
    print(text)
    return text


def main():
    print("=" * 70)
    print("D8.0 AUFGABE 0: EARNINGS-DATEN BESCHAFFEN")
    print("=" * 70)

    # 0a: Extract unique tickers
    meta_v7 = pd.read_parquet(META_DIR / "metadata_v7.parquet")
    tickers = sorted(meta_v7['ticker'].unique())
    print(f"\n0a: {len(tickers)} unique Ticker in metadata_v7")
    print(f"    Dataset: {len(meta_v7)} Gapper-Tage")

    # 0b: Fetch earnings dates
    earnings_path = META_DIR / "earnings_dates.parquet"

    if earnings_path.exists():
        print(f"\n0b: earnings_dates.parquet existiert bereits, lade...")
        earnings_df = pd.read_parquet(earnings_path)
        print(f"    {len(earnings_df)} Earnings-Records fuer {earnings_df['ticker'].nunique()} Ticker")
    else:
        print(f"\n0b: Starte Polygon API Abruf fuer {len(tickers)} Ticker...")
        print(f"    Geschaetzte Dauer: ~{len(tickers)*0.25/60:.1f} Minuten")

        earnings_df, success, no_data, errors, error_tickers = fetch_earnings_dates(
            tickers, earnings_path
        )

        print(f"\n    API Ergebnis:")
        print(f"    Erfolgreich:   {success}")
        print(f"    Ohne Daten:    {no_data}")
        print(f"    Fehler:        {errors}")
        if error_tickers:
            print(f"    Fehler-Ticker: {error_tickers[:10]}")
        print(f"    Total Records: {len(earnings_df)}")
        print(f"    Gespeichert:   {earnings_path}")

    # 0c: Merge to metadata_v8
    v8_path = META_DIR / "metadata_v8.parquet"
    print(f"\n0c: Erstelle metadata_v8.parquet...")
    meta_v8 = create_metadata_v8(meta_v7, earnings_df, v8_path)
    print(f"    Shape: {meta_v8.shape}")
    print(f"    Gespeichert: {v8_path}")

    # 0d: Plausibility check
    print(f"\n0d: Plausibilitaets-Check...")
    plausibility_check(meta_v8, RESULTS_DIR / "d8_0_earnings_check.txt")

    print("\n" + "=" * 70)
    print("AUFGABE 0 ABGESCHLOSSEN")
    print("=" * 70)


if __name__ == "__main__":
    main()
