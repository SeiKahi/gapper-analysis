# utils.py — Shared Utilities for Gapper Analysis Pipeline

import time
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytz
import pyarrow as pa
import pyarrow.parquet as pq
from polygon import RESTClient

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import POLYGON_API_KEY

# Timezone
EASTERN = pytz.timezone("US/Eastern")
UTC = pytz.utc


def get_polygon_client() -> RESTClient:
    """Erstellt und gibt einen Polygon REST Client zurück."""
    if not POLYGON_API_KEY:
        raise ValueError("POLYGON_API_KEY nicht gesetzt! Prüfe .env Datei.")
    return RESTClient(api_key=POLYGON_API_KEY)


def timestamp_to_et(ts_ms: int) -> datetime:
    """Konvertiert Unix-Millisekunden-Timestamp nach US/Eastern datetime."""
    dt_utc = datetime.fromtimestamp(ts_ms / 1000, tz=UTC)
    return dt_utc.astimezone(EASTERN)


def et_to_timestamp(dt: datetime) -> int:
    """Konvertiert ET datetime nach Unix-Millisekunden-Timestamp."""
    if dt.tzinfo is None:
        dt = EASTERN.localize(dt)
    return int(dt.timestamp() * 1000)


def get_trading_days(start_date: str, end_date: str) -> list:
    """Gibt Liste aller US-Handelstage (NYSE Kalender) zwischen start und end zurück."""
    try:
        import exchange_calendars as xcals
        nyse = xcals.get_calendar("XNYS")
        sessions = nyse.sessions_in_range(start_date, end_date)
        return [s.strftime("%Y-%m-%d") for s in sessions]
    except Exception:
        # Fallback: pandas business days minus US holidays
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        bdays = pd.bdate_range(start, end)
        holidays = _get_us_holidays(start.year, end.year)
        trading_days = [d for d in bdays if d.strftime("%Y-%m-%d") not in holidays]
        return [d.strftime("%Y-%m-%d") for d in trading_days]


def _get_us_holidays(start_year: int, end_year: int) -> set:
    """Manuelle US-Feiertage als Fallback."""
    holidays = set()
    for year in range(start_year, end_year + 1):
        holidays.update([
            f"{year}-01-01",  # New Year
            f"{year}-06-19",  # Juneteenth
            f"{year}-07-04",  # Independence Day
            f"{year}-12-25",  # Christmas
        ])
        # MLK Day: 3rd Monday of January
        jan1 = datetime(year, 1, 1)
        mlk = jan1 + timedelta(days=(7 - jan1.weekday()) % 7 + 14)
        holidays.add(mlk.strftime("%Y-%m-%d"))
        # Presidents Day: 3rd Monday of February
        feb1 = datetime(year, 2, 1)
        pres = feb1 + timedelta(days=(7 - feb1.weekday()) % 7 + 14)
        holidays.add(pres.strftime("%Y-%m-%d"))
        # Memorial Day: Last Monday of May
        may31 = datetime(year, 5, 31)
        mem = may31 - timedelta(days=may31.weekday())
        holidays.add(mem.strftime("%Y-%m-%d"))
        # Labor Day: 1st Monday of September
        sep1 = datetime(year, 9, 1)
        labor = sep1 + timedelta(days=(7 - sep1.weekday()) % 7)
        holidays.add(labor.strftime("%Y-%m-%d"))
        # Thanksgiving: 4th Thursday of November
        nov1 = datetime(year, 11, 1)
        first_thu = nov1 + timedelta(days=(3 - nov1.weekday()) % 7)
        thanks = first_thu + timedelta(weeks=3)
        holidays.add(thanks.strftime("%Y-%m-%d"))
    return holidays


def get_previous_trading_day(date_str: str) -> str:
    """Gibt den vorherigen Handelstag zurück."""
    dt = pd.Timestamp(date_str)
    candidate = dt - timedelta(days=1)
    for _ in range(10):
        days = get_trading_days(candidate.strftime("%Y-%m-%d"), candidate.strftime("%Y-%m-%d"))
        if days:
            return days[0]
        candidate -= timedelta(days=1)
    raise ValueError(f"Kein Handelstag gefunden vor {date_str}")


def save_parquet(df: pd.DataFrame, path: Path):
    """Speichert DataFrame als Parquet mit pyarrow engine."""
    ensure_dir(path.parent)
    df.to_parquet(path, engine="pyarrow", index=False)


def load_parquet(path: Path) -> pd.DataFrame:
    """Lädt Parquet-Datei, gibt DataFrame zurück."""
    return pd.read_parquet(path, engine="pyarrow")


def ensure_dir(path: Path):
    """Erstellt Ordner falls nicht vorhanden."""
    path.mkdir(parents=True, exist_ok=True)


def retry_api_call(func, *args, max_retries=3, **kwargs):
    """Wrapper für API Calls mit Exponential Backoff."""
    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            error_str = str(e)
            # Permanent errors — no point retrying
            if any(msg in error_str for msg in ("NOT_FOUND", "not found", "Ticker not found")):
                raise
            if attempt < max_retries:
                wait_time = 0.5 * (2 ** attempt)  # 0.5s, 1s, 2s
                logging.warning(f"API call failed (attempt {attempt + 1}/{max_retries + 1}): {error_str}. "
                                f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logging.error(f"API call failed after {max_retries + 1} attempts: {error_str}")
    raise last_exception


def sic_to_sector(sic_code: str) -> str:
    """Mappt SIC Code auf vereinfachten Sektor-String."""
    if not sic_code:
        return "Unknown"
    try:
        code = int(sic_code)
    except (ValueError, TypeError):
        return "Unknown"

    if 1000 <= code < 1500:
        return "Mining/Energy"
    elif 1500 <= code < 1800:
        return "Construction"
    elif 2000 <= code < 4000:
        return "Manufacturing"
    elif 4000 <= code < 5000:
        return "Transportation/Utilities"
    elif 5000 <= code < 5200:
        return "Wholesale"
    elif 5200 <= code < 6000:
        return "Retail"
    elif 6000 <= code < 6800:
        return "Finance"
    elif 7000 <= code < 9000:
        return "Services"
    elif 9000 <= code < 10000:
        return "Government"
    else:
        return "Unknown"


def setup_logging(script_name: str, date_str: str = None, log_dir: Path = None):
    """Konfiguriert Logging für ein Script."""
    if log_dir is None:
        from config import METADATA_DIR
        log_dir = METADATA_DIR / "logs"
    ensure_dir(log_dir)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # Console handler: INFO
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(console)

    # File handler: DEBUG
    suffix = f"_{date_str}" if date_str else ""
    log_file = log_dir / f"{script_name}{suffix}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(file_handler)

    return logger


def parse_date_args():
    """Parst --date, --from, --to CLI Argumente."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="Single date YYYY-MM-DD")
    parser.add_argument("--from", dest="from_date", help="Start date for backfill")
    parser.add_argument("--to", dest="to_date", help="End date for backfill")
    parser.add_argument("--force", action="store_true", help="Überschreibe existierende Dateien")
    args = parser.parse_args()

    if args.date:
        dates = [args.date]
    elif args.from_date and args.to_date:
        dates = get_trading_days(args.from_date, args.to_date)
    else:
        yesterday = get_previous_trading_day(datetime.now().strftime("%Y-%m-%d"))
        dates = [yesterday]
        logging.info(f"Kein Datum angegeben, nutze: {yesterday}")

    return dates, args
