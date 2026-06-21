# 02b_download_baseline.py — Baseline 1-Min Bars für RVOL-at-Time Berechnung

import sys
import time
import logging
import threading
from pathlib import Path
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import METADATA_DIR, BASELINE_1MIN_DIR
from scripts.utils import (
    get_polygon_client, timestamp_to_et, retry_api_call,
    save_parquet, ensure_dir, setup_logging, parse_date_args,
    get_trading_days
)


def download_baseline_day(client, ticker: str, baseline_date: str, force: bool = False) -> bool:
    """Lädt 1-Min RTH Bars für einen einzelnen Baseline-Tag."""
    output_dir = BASELINE_1MIN_DIR / ticker
    output_path = output_dir / f"{baseline_date}.parquet"

    if output_path.exists() and not force:
        logging.debug(f"  Überspringe {ticker}/{baseline_date}: existiert")
        return True

    try:
        aggs = retry_api_call(
            client.get_aggs,
            ticker=ticker,
            multiplier=1,
            timespan="minute",
            from_=baseline_date,
            to=baseline_date,
            adjusted=True,
            sort="asc",
            limit=50000
        )
    except Exception as e:
        logging.warning(f"  Baseline Download fehlgeschlagen für {ticker} am {baseline_date}: {e}")
        return False

    if not aggs:
        logging.debug(f"  Keine Baseline-Daten für {ticker} am {baseline_date}")
        return False

    rows = []
    for bar in aggs:
        dt_et = timestamp_to_et(bar.timestamp)
        time_str = dt_et.strftime("%H:%M")

        # Nur RTH: 09:30-16:00 ET
        if "09:30" <= time_str < "16:00":
            rows.append({
                "timestamp_utc": bar.timestamp,
                "datetime_et": dt_et.strftime("%Y-%m-%d %H:%M:%S"),
                "time_et": time_str,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume or 0,
                "ticker": ticker,
                "date": baseline_date,
            })

    if not rows:
        logging.debug(f"  Keine RTH Bars für {ticker} am {baseline_date}")
        return False

    df = pd.DataFrame(rows)
    df["timestamp_utc"] = df["timestamp_utc"].astype("int64")
    df["volume"] = df["volume"].astype("int64")
    save_parquet(df, output_path)
    logging.debug(f"  {ticker}/{baseline_date}: {len(df)} RTH Bars gespeichert")
    return True


def get_baseline_dates(scan_date: str, n_days: int = 10) -> list:
    """Findet die letzten n Handelstage VOR dem scan_date."""
    dt = pd.Timestamp(scan_date)
    # Gehe weit genug zurück um sicher n Handelstage zu finden
    start_dt = dt - timedelta(days=n_days * 2 + 5)
    all_days = get_trading_days(start_dt.strftime("%Y-%m-%d"), (dt - timedelta(days=1)).strftime("%Y-%m-%d"))
    # Nimm die letzten n Tage
    return all_days[-n_days:] if len(all_days) >= n_days else all_days


def process_date(date_str: str, client, force: bool = False):
    """Lädt Baseline-Daten für alle Gapper eines Tages."""
    gapper_file = METADATA_DIR / f"gappers_{date_str}.csv"
    if not gapper_file.exists():
        logging.warning(f"Keine Gapper-Datei für {date_str}: {gapper_file}")
        return

    df_gappers = pd.read_csv(gapper_file)
    if df_gappers.empty:
        logging.info(f"Keine Gapper für {date_str}")
        return

    tickers = df_gappers["ticker"].tolist()
    baseline_dates = get_baseline_dates(date_str, n_days=10)

    if not baseline_dates:
        logging.warning(f"Keine Baseline-Tage gefunden vor {date_str}")
        return

    logging.info(f"{date_str}: Lade Baseline-Daten für {len(tickers)} Ticker × {len(baseline_dates)} Tage")

    api_lock = threading.Lock()
    success = 0
    errors = 0

    # Alle Ticker×Baseline-Tag Kombinationen sammeln
    tasks = [(ticker, bd) for ticker in tickers for bd in baseline_dates]

    def _download_one(args):
        ticker, bd = args
        with api_lock:
            time.sleep(0.15)
        return download_baseline_day(client, ticker, bd, force)

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(_download_one, task): task for task in tasks}
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc=f"  {date_str} baseline", leave=False):
            try:
                ok = future.result()
                if ok:
                    success += 1
                else:
                    errors += 1
            except Exception as e:
                logging.error(f"  Baseline Fehler: {e}")
                errors += 1

    logging.info(f"  {date_str}: {success} erfolgreich, {errors} Fehler (Baseline)")


def main():
    dates, args = parse_date_args()
    setup_logging("02b_download_baseline",
                  dates[0] if len(dates) == 1 else f"{dates[0]}_to_{dates[-1]}")

    client = get_polygon_client()
    logging.info(f"=== Baseline Download ===")
    logging.info(f"Verarbeite {len(dates)} Handelstag(e)")

    for date_str in tqdm(dates, desc="Downloading baseline"):
        try:
            process_date(date_str, client, force=args.force)
        except Exception as e:
            logging.error(f"Fehler bei {date_str}: {e}", exc_info=True)

    logging.info("=== Baseline Download abgeschlossen ===")


if __name__ == "__main__":
    main()
