# 03_download_daily.py — Daily Bars der letzten 90 Tage herunterladen

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
from config import METADATA_DIR, DAILY_DIR, DAILY_CONTEXT_LOOKBACK
from scripts.utils import (
    get_polygon_client, retry_api_call, save_parquet,
    ensure_dir, setup_logging, parse_date_args, get_previous_trading_day
)


def download_daily_for_ticker(client, ticker: str, scan_date: str, force: bool = False) -> bool:
    """Lädt ~90 Tage Daily Bars und speichert als Parquet."""
    output_dir = DAILY_DIR / ticker
    output_path = output_dir / f"{scan_date}_context.parquet"

    if output_path.exists() and not force:
        logging.debug(f"  Überspringe {ticker}: {output_path.name} existiert")
        return True

    # 130 Kalendertage zurück ergibt ~90 Handelstage
    end_dt = pd.Timestamp(scan_date)
    start_dt = end_dt - timedelta(days=130)
    start_str = start_dt.strftime("%Y-%m-%d")

    try:
        aggs = retry_api_call(
            client.get_aggs,
            ticker=ticker,
            multiplier=1,
            timespan="day",
            from_=start_str,
            to=scan_date,
            adjusted=True,
            sort="asc",
            limit=50000
        )
    except Exception as e:
        logging.warning(f"  Daily Download fehlgeschlagen für {ticker}: {e}")
        return False

    if not aggs:
        logging.warning(f"  Keine Daily-Daten für {ticker}")
        return False

    rows = []
    for bar in aggs:
        # Timestamp to date
        from scripts.utils import timestamp_to_et
        dt_et = timestamp_to_et(bar.timestamp)
        rows.append({
            "date": dt_et.strftime("%Y-%m-%d"),
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume or 0,
            "vwap": getattr(bar, "vwap", None),
            "transactions": getattr(bar, "transactions", None),
            "ticker": ticker,
        })

    df = pd.DataFrame(rows)
    df["volume"] = df["volume"].astype("int64")
    save_parquet(df, output_path)
    logging.debug(f"  {ticker}: {len(df)} Daily Bars gespeichert")
    return True


def download_spy_data(client, scan_date: str, force: bool = False):
    """Lädt SPY Daily-Daten für Marktkontext."""
    output_dir = DAILY_DIR / "SPY"
    output_path = output_dir / f"{scan_date}_context.parquet"

    if output_path.exists() and not force:
        return

    end_dt = pd.Timestamp(scan_date)
    start_dt = end_dt - timedelta(days=10)
    start_str = start_dt.strftime("%Y-%m-%d")

    try:
        aggs = retry_api_call(
            client.get_aggs,
            ticker="SPY",
            multiplier=1,
            timespan="day",
            from_=start_str,
            to=scan_date,
            adjusted=True,
            sort="asc",
            limit=50000
        )
        if aggs:
            from scripts.utils import timestamp_to_et
            rows = []
            for bar in aggs:
                dt_et = timestamp_to_et(bar.timestamp)
                rows.append({
                    "date": dt_et.strftime("%Y-%m-%d"),
                    "open": bar.open, "high": bar.high,
                    "low": bar.low, "close": bar.close,
                    "volume": bar.volume or 0,
                    "vwap": getattr(bar, "vwap", None),
                    "transactions": getattr(bar, "transactions", None),
                    "ticker": "SPY",
                })
            df = pd.DataFrame(rows)
            save_parquet(df, output_path)
            logging.debug(f"  SPY: {len(df)} Daily Bars gespeichert")
    except Exception as e:
        logging.warning(f"  SPY Download fehlgeschlagen: {e}")


def download_vix_data(client, scan_date: str, force: bool = False):
    """Versucht VIX-Daten zu laden (optional)."""
    output_dir = DAILY_DIR / "VIX"
    output_path = output_dir / f"{scan_date}_context.parquet"

    if output_path.exists() and not force:
        return

    end_dt = pd.Timestamp(scan_date)
    start_dt = end_dt - timedelta(days=10)
    start_str = start_dt.strftime("%Y-%m-%d")

    for vix_ticker in ["VIX", "I:VIX"]:
        try:
            aggs = retry_api_call(
                client.get_aggs,
                ticker=vix_ticker,
                multiplier=1,
                timespan="day",
                from_=start_str,
                to=scan_date,
                adjusted=True,
                sort="asc",
                limit=50000
            )
            if aggs:
                from scripts.utils import timestamp_to_et
                rows = []
                for bar in aggs:
                    dt_et = timestamp_to_et(bar.timestamp)
                    rows.append({
                        "date": dt_et.strftime("%Y-%m-%d"),
                        "open": bar.open, "high": bar.high,
                        "low": bar.low, "close": bar.close,
                        "volume": bar.volume or 0,
                        "ticker": vix_ticker,
                    })
                df = pd.DataFrame(rows)
                save_parquet(df, output_path)
                logging.debug(f"  VIX ({vix_ticker}): {len(df)} Daily Bars gespeichert")
                return
        except Exception as e:
            logging.debug(f"  VIX Ticker '{vix_ticker}' nicht verfügbar: {e}")

    logging.warning(f"  VIX-Daten nicht verfügbar (Starter Plan Limitation)")


def process_date(date_str: str, client, force: bool = False):
    """Verarbeitet alle Gapper eines Tages."""
    gapper_file = METADATA_DIR / f"gappers_{date_str}.csv"
    if not gapper_file.exists():
        logging.warning(f"Keine Gapper-Datei für {date_str}")
        return

    df_gappers = pd.read_csv(gapper_file)
    if df_gappers.empty:
        logging.info(f"Keine Gapper für {date_str}")
        return

    tickers = df_gappers["ticker"].tolist()
    logging.info(f"{date_str}: Lade Daily-Daten für {len(tickers)} Gapper + SPY/VIX")

    # SPY & VIX
    download_spy_data(client, date_str, force)
    time.sleep(0.15)
    download_vix_data(client, date_str, force)
    time.sleep(0.15)

    api_lock = threading.Lock()
    success = 0
    errors = 0

    def _download_one(ticker):
        with api_lock:
            time.sleep(0.15)
        return ticker, download_daily_for_ticker(client, ticker, date_str, force)

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
    setup_logging("03_download_daily", dates[0] if len(dates) == 1 else f"{dates[0]}_to_{dates[-1]}")

    client = get_polygon_client()
    logging.info(f"=== Daily Download ===")
    logging.info(f"Verarbeite {len(dates)} Handelstag(e)")

    for date_str in tqdm(dates, desc="Downloading daily"):
        try:
            process_date(date_str, client, force=args.force)
        except Exception as e:
            logging.error(f"Fehler bei {date_str}: {e}", exc_info=True)

    logging.info("=== Daily Download abgeschlossen ===")


if __name__ == "__main__":
    main()
