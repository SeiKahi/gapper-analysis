# 06_normalize.py — ATR-normalisierte Intraday-Profile

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import METADATA_DIR, RAW_1MIN_DIR, DAILY_DIR, NORMALIZED_DIR
from scripts.utils import load_parquet, save_parquet, setup_logging, parse_date_args


def aggregate_to_interval(df: pd.DataFrame, interval_minutes: int) -> pd.DataFrame:
    """Aggregiert 1-Min Bars auf beliebiges Intervall."""
    df = df.copy()
    df["group"] = df["minutes_since_open"] // interval_minutes

    agg = df.groupby("group").agg({
        "datetime_et": "first",
        "time_et": "first",
        "minutes_since_open": "first",
        "open": "first" if "open" in df.columns else "first",
        "close": "last",
        "volume": "sum",
        "rth_open": "first",
        "adr_5": "first",
        "adr_20": "first",
        "normalized_adr5": "last",
        "normalized_adr20": "last",
        "ticker": "first",
        "date": "first",
    }).reset_index(drop=True)

    if "high" in df.columns:
        agg["high"] = df.groupby("group")["high"].max().values
    if "low" in df.columns:
        agg["low"] = df.groupby("group")["low"].min().values

    return agg


def normalize_for_ticker(ticker: str, date_str: str, force: bool = False) -> bool:
    """Erstellt ATR-normalisiertes Intraday-Profil."""
    intraday_path = RAW_1MIN_DIR / ticker / f"{date_str}.parquet"
    daily_path = DAILY_DIR / ticker / f"{date_str}_context.parquet"
    output_path = NORMALIZED_DIR / ticker / f"{date_str}.parquet"

    if output_path.exists() and not force:
        logging.debug(f"  Überspringe {ticker}: Normalisiert existiert")
        return True

    if not intraday_path.exists():
        logging.warning(f"  Keine Intraday-Daten für {ticker}")
        return False
    if not daily_path.exists():
        logging.warning(f"  Keine Daily-Daten für {ticker}")
        return False

    df_intraday = load_parquet(intraday_path)
    df_daily = load_parquet(daily_path)

    rth = df_intraday[df_intraday["session"] == "rth"].copy().reset_index(drop=True)
    if len(rth) < 10:
        logging.warning(f"  Zu wenige RTH-Bars für {ticker}: {len(rth)}")
        return False

    # Daily-Daten: Nur Tage VOR dem gap_date
    df_daily = df_daily[df_daily["date"] < date_str].copy()
    df_daily = df_daily.sort_values("date").reset_index(drop=True)

    if len(df_daily) < 5:
        logging.warning(f"  Zu wenige Daily-Daten für {ticker}: {len(df_daily)}")
        return False

    # ADR berechnen
    df_daily["daily_range"] = df_daily["high"] - df_daily["low"]
    adr_5 = df_daily["daily_range"].iloc[-5:].mean()
    adr_20 = df_daily["daily_range"].iloc[-20:].mean() if len(df_daily) >= 20 else df_daily["daily_range"].mean()

    if adr_5 == 0 or np.isnan(adr_5):
        logging.warning(f"  ADR(5) ist 0/NaN für {ticker}")
        return False

    # RTH Open
    rth_open = rth.iloc[0]["open"]

    # Normalisierung
    result = pd.DataFrame({
        "datetime_et": rth["datetime_et"].values,
        "time_et": rth["time_et"].values,
        "minutes_since_open": np.arange(len(rth)),
        "close": rth["close"].values,
        "rth_open": rth_open,
        "adr_5": adr_5,
        "adr_20": adr_20,
        "normalized_adr5": (rth["close"].values - rth_open) / adr_5,
        "normalized_adr20": (rth["close"].values - rth_open) / adr_20,
        "volume": rth["volume"].values,
        "ticker": ticker,
        "date": date_str,
    })

    save_parquet(result, output_path)
    logging.debug(f"  {ticker}: Normalisiert ({len(result)} Bars, ADR5={adr_5:.2f})")
    return True


def process_date(date_str: str, force: bool = False):
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
    logging.info(f"{date_str}: Normalisiere {len(tickers)} Gapper")

    success = 0
    errors = 0
    for ticker in tqdm(tickers, desc=f"  {date_str}", leave=False):
        try:
            if normalize_for_ticker(ticker, date_str, force):
                success += 1
            else:
                errors += 1
        except Exception as e:
            logging.error(f"  Normalisierung Fehler bei {ticker}: {e}")
            errors += 1

    logging.info(f"  {date_str}: {success} erfolgreich, {errors} Fehler")


def main():
    dates, args = parse_date_args()
    setup_logging("06_normalize", dates[0] if len(dates) == 1 else f"{dates[0]}_to_{dates[-1]}")

    logging.info(f"=== Normalisierung ===")
    logging.info(f"Verarbeite {len(dates)} Handelstag(e)")

    for date_str in tqdm(dates, desc="Normalizing"):
        try:
            process_date(date_str, force=args.force)
        except Exception as e:
            logging.error(f"Fehler bei {date_str}: {e}", exc_info=True)

    logging.info("=== Normalisierung abgeschlossen ===")


if __name__ == "__main__":
    main()
