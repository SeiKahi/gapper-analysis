# 02_download_intraday.py — 1-Min Intraday-Daten herunterladen

import sys
import time
import logging
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import METADATA_DIR, RAW_1MIN_DIR
from scripts.utils import (
    get_polygon_client, timestamp_to_et, retry_api_call,
    save_parquet, ensure_dir, setup_logging, parse_date_args
)


def download_intraday_for_ticker(client, ticker: str, date_str: str, force: bool = False) -> bool:
    """Lädt 1-Min Bars für einen Ticker und speichert als Parquet."""
    output_dir = RAW_1MIN_DIR / ticker
    output_path = output_dir / f"{date_str}.parquet"

    if output_path.exists() and not force:
        logging.debug(f"  Überspringe {ticker}: {output_path.name} existiert")
        return True

    try:
        aggs = retry_api_call(
            client.get_aggs,
            ticker=ticker,
            multiplier=1,
            timespan="minute",
            from_=date_str,
            to=date_str,
            adjusted=True,
            sort="asc",
            limit=50000
        )
    except Exception as e:
        logging.warning(f"  Download fehlgeschlagen für {ticker} am {date_str}: {e}")
        return False

    if not aggs:
        logging.warning(f"  Keine Intraday-Daten für {ticker} am {date_str}")
        return False

    rows = []
    for bar in aggs:
        dt_et = timestamp_to_et(bar.timestamp)
        time_str = dt_et.strftime("%H:%M")

        if "04:00" <= time_str < "09:30":
            session = "premarket"
        elif "09:30" <= time_str < "16:00":
            session = "rth"
        elif "16:00" <= time_str <= "20:00":
            session = "afterhours"
        else:
            continue

        rows.append({
            "timestamp_utc": bar.timestamp,
            "datetime_et": dt_et.strftime("%Y-%m-%d %H:%M:%S"),
            "time_et": time_str,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume or 0,
            "vwap": getattr(bar, "vwap", None),
            "transactions": getattr(bar, "transactions", None),
            "ticker": ticker,
            "date": date_str,
            "session": session,
        })

    if not rows:
        logging.warning(f"  Keine gültigen Bars für {ticker} am {date_str}")
        return False

    df = pd.DataFrame(rows)
    df["timestamp_utc"] = df["timestamp_utc"].astype("int64")
    df["volume"] = df["volume"].astype("int64")
    save_parquet(df, output_path)
    logging.debug(f"  {ticker}: {len(df)} Bars gespeichert")
    return True


def process_date(date_str: str, client, force: bool = False):
    """Verarbeitet alle Gapper eines Tages."""
    gapper_file = METADATA_DIR / f"gappers_{date_str}.csv"
    if not gapper_file.exists():
        logging.warning(f"Keine Gapper-Datei für {date_str}: {gapper_file}")
        return

    df_gappers = pd.read_csv(gapper_file)
    if df_gappers.empty:
        logging.info(f"Keine Gapper für {date_str}")
        return

    tickers = df_gappers["ticker"].tolist()
    logging.info(f"{date_str}: Lade Intraday-Daten für {len(tickers)} Gapper")

    api_lock = threading.Lock()
    success = 0
    errors = 0

    def _download_one(ticker):
        with api_lock:
            time.sleep(0.15)
        return ticker, download_intraday_for_ticker(client, ticker, date_str, force)

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(_download_one, t): t for t in tickers}
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc=f"  {date_str}", leave=False):
            try:
                ticker, ok = future.result()
                if ok:
                    success += 1
                else:
                    errors += 1
            except Exception as e:
                logging.error(f"  Fehler bei {futures[future]}: {e}")
                errors += 1

    logging.info(f"  {date_str}: {success} erfolgreich, {errors} Fehler")


def main():
    dates, args = parse_date_args()
    setup_logging("02_download_intraday", dates[0] if len(dates) == 1 else f"{dates[0]}_to_{dates[-1]}")

    client = get_polygon_client()
    logging.info(f"=== Intraday Download ===")
    logging.info(f"Verarbeite {len(dates)} Handelstag(e)")

    for date_str in tqdm(dates, desc="Downloading intraday"):
        try:
            process_date(date_str, client, force=args.force)
        except Exception as e:
            logging.error(f"Fehler bei {date_str}: {e}", exc_info=True)

    logging.info("=== Intraday Download abgeschlossen ===")


if __name__ == "__main__":
    main()
