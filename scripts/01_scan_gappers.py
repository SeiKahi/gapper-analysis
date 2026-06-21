# 01_scan_gappers.py — Täglicher Gapper-Scanner (Up UND Down Gaps)

import sys
import json
import time
import logging
import threading
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    MIN_GAP_PCT, MIN_PREMARKET_VOLUME, MIN_MARKET_CAP, MIN_PRICE,
    ALLOWED_EXCHANGES, ALLOWED_TYPES, METADATA_DIR, PREMARKET_START
)
from scripts.utils import (
    get_polygon_client, timestamp_to_et, get_previous_trading_day,
    get_trading_days, ensure_dir, retry_api_call, sic_to_sector,
    setup_logging, parse_date_args, EASTERN
)


def load_ticker_cache(cache_path: Path) -> dict:
    """Lädt Ticker-Detail-Cache aus JSON."""
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_ticker_cache(cache: dict, cache_path: Path):
    """Speichert Ticker-Detail-Cache als JSON."""
    ensure_dir(cache_path.parent)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def is_cache_fresh(entry: dict, max_age_days: int = 7) -> bool:
    """Prüft ob ein Cache-Eintrag noch frisch genug ist."""
    if "last_updated" not in entry:
        return False
    last = datetime.strptime(entry["last_updated"], "%Y-%m-%d")
    return (datetime.now() - last).days <= max_age_days


# Global rate-limit lock for API calls
_api_lock = threading.Lock()
_cache_lock = threading.Lock()


def get_ticker_details_cached(client, ticker: str, cache: dict, cache_path: Path, scan_date: str) -> dict:
    """Holt Ticker Details mit Caching."""
    if ticker in cache:
        entry = cache[ticker]
        # Skip tickers previously marked as not found
        if entry.get("not_found"):
            return None
        if is_cache_fresh(entry):
            return entry

    try:
        with _api_lock:
            time.sleep(0.15)
        details = retry_api_call(client.get_ticker_details, ticker, date=scan_date)
        entry = {
            "market_cap": getattr(details, "market_cap", None),
            "primary_exchange": getattr(details, "primary_exchange", None),
            "type": getattr(details, "type", None),
            "sic_code": getattr(details, "sic_code", None),
            "sic_description": getattr(details, "sic_description", None),
            "name": getattr(details, "name", None),
            "last_updated": scan_date,
        }
        with _cache_lock:
            cache[ticker] = entry
            save_ticker_cache(cache, cache_path)
        return entry
    except Exception as e:
        error_str = str(e)
        if any(msg in error_str for msg in ("NOT_FOUND", "not found", "Ticker not found")):
            with _cache_lock:
                cache[ticker] = {"not_found": True, "last_updated": scan_date}
                save_ticker_cache(cache, cache_path)
        logging.warning(f"Ticker Details für {ticker} fehlgeschlagen: {e}")
        return None


def get_premarket_volume(client, ticker: str, scan_date: str) -> int:
    """Berechnet PreMarket-Volumen (04:00-09:29 ET) aus 1-Min Bars."""
    try:
        aggs = retry_api_call(
            client.get_aggs,
            ticker=ticker,
            multiplier=1,
            timespan="minute",
            from_=scan_date,
            to=scan_date,
            adjusted=True,
            sort="asc",
            limit=50000
        )
        if not aggs:
            return 0

        pm_volume = 0
        for bar in aggs:
            dt_et = timestamp_to_et(bar.timestamp)
            t = dt_et.strftime("%H:%M")
            if "04:00" <= t < "09:30":
                pm_volume += (bar.volume or 0)
        return pm_volume
    except Exception as e:
        logging.warning(f"PreMarket-Daten für {ticker} fehlgeschlagen: {e}")
        return 0


def scan_single_day(scan_date: str, client, cache: dict, cache_path: Path, force: bool = False):
    """Scannt einen einzelnen Tag auf Gapper."""
    output_path = METADATA_DIR / f"gappers_{scan_date}.csv"
    if output_path.exists() and not force:
        logging.info(f"Überspringe {scan_date}: {output_path.name} existiert bereits")
        return

    prev_date = get_previous_trading_day(scan_date)
    logging.info(f"Scanne {scan_date} (Vortag: {prev_date})")

    # Grouped Daily für heute und gestern
    try:
        today_grouped = retry_api_call(client.get_grouped_daily_aggs, date=scan_date)
    except Exception as e:
        logging.warning(f"Keine Grouped Daily Daten für {scan_date}: {e}")
        return

    if not today_grouped:
        logging.warning(f"Keine Daten für {scan_date} (Feiertag/Wochenende?)")
        return

    try:
        prev_grouped = retry_api_call(client.get_grouped_daily_aggs, date=prev_date)
    except Exception as e:
        logging.warning(f"Keine Grouped Daily Daten für {prev_date}: {e}")
        return

    # Build lookup dicts
    today_data = {}
    for bar in today_grouped:
        today_data[bar.ticker] = {
            "open": bar.open, "high": bar.high, "low": bar.low,
            "close": bar.close, "volume": bar.volume, "vwap": getattr(bar, "vwap", None)
        }

    prev_data = {}
    for bar in prev_grouped:
        prev_data[bar.ticker] = {"close": bar.close}

    # Step 1: Gap filter
    gap_candidates = []
    n_total = 0
    for ticker in today_data:
        if ticker not in prev_data:
            continue
        n_total += 1
        prev_close = prev_data[ticker]["close"]
        today_open = today_data[ticker]["open"]

        if prev_close is None or prev_close == 0 or today_open is None:
            continue
        if today_open < MIN_PRICE:
            continue

        gap_pct = (today_open - prev_close) / prev_close * 100
        if abs(gap_pct) < MIN_GAP_PCT:
            continue

        gap_candidates.append({
            "ticker": ticker,
            "gap_pct": round(gap_pct, 2),
            "gap_direction": "up" if gap_pct > 0 else "down",
            "prev_close": prev_close,
            "today_open": today_open,
            "today_close": today_data[ticker]["close"],
            "today_volume": today_data[ticker]["volume"],
        })

    n_gap = len(gap_candidates)
    logging.info(f"  {n_total} Ticker insgesamt")
    logging.info(f"  {n_gap} bestehen Gap-Filter (>={MIN_GAP_PCT}%)")

    # Step 2: Volume pre-filter (kostenlos, aus Grouped Daily Bars)
    MIN_DAILY_VOLUME = 500_000
    volume_filtered = [c for c in gap_candidates if (c.get("today_volume") or 0) >= MIN_DAILY_VOLUME]
    n_vol = len(volume_filtered)
    logging.info(f"  {n_vol} bestehen Volumen-Vorfilter (>={MIN_DAILY_VOLUME:,} daily)")

    # Step 3: Exchange/Type/MarketCap filter (parallelisiert, thread-safe)
    def _fetch_and_filter(cand):
        """Holt Ticker-Details und filtert. Gibt ein NEUES Dict zurück (kein Mutieren des Originals)."""
        details = get_ticker_details_cached(client, cand["ticker"], cache, cache_path, scan_date)
        if details is None:
            return None
        if details.get("primary_exchange") not in ALLOWED_EXCHANGES:
            return None
        if details.get("type") not in ALLOWED_TYPES:
            return None
        mc = details.get("market_cap")
        if mc is None or mc < MIN_MARKET_CAP:
            return None

        # Neues Dict erstellen statt cand in-place zu mutieren (Thread-Safety)
        enriched = dict(cand)
        enriched["name"] = details.get("name", "")
        enriched["market_cap"] = mc
        enriched["primary_exchange"] = details.get("primary_exchange", "")
        enriched["type"] = details.get("type", "")
        enriched["sic_code"] = details.get("sic_code", "")
        enriched["sic_description"] = details.get("sic_description", "")
        enriched["sector"] = sic_to_sector(details.get("sic_code", ""))
        return enriched

    exchange_filtered = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(_fetch_and_filter, cand): cand for cand in volume_filtered}
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc=f"  Ticker Details", leave=False):
            try:
                result = future.result()
            except Exception as e:
                logging.warning(f"  Ticker-Detail-Fehler: {e}")
                continue
            if result is not None:
                exchange_filtered.append(result)

    n_exchange = len(exchange_filtered)
    logging.info(f"  {n_exchange} bestehen Exchange/Type/MarketCap Filter")

    # Sanity check: Jeder Filterschritt darf die Anzahl nur reduzieren
    assert n_exchange <= n_vol, (
        f"BUG: Exchange-Filter ({n_exchange}) > Volumen-Vorfilter ({n_vol})")

    # Step 4: PreMarket Volume filter (teuer, 1 API Call pro Ticker)
    final_gappers = []
    for cand in tqdm(exchange_filtered, desc=f"  PreMarket Volume", leave=False):
        pm_vol = get_premarket_volume(client, cand["ticker"], scan_date)
        cand["premarket_volume"] = pm_vol
        if pm_vol >= MIN_PREMARKET_VOLUME:
            final_gappers.append(cand)
        time.sleep(0.12)

    n_final = len(final_gappers)
    logging.info(f"  {n_final} finale Gapper nach PreMarket-Volume Filter (>={MIN_PREMARKET_VOLUME:,})")

    # Save results
    if final_gappers:
        df = pd.DataFrame(final_gappers)
        df.insert(0, "date", scan_date)
        cols = ["date", "ticker", "name", "gap_pct", "gap_direction", "prev_close",
                "today_open", "premarket_volume", "market_cap", "primary_exchange",
                "type", "sic_code", "sic_description", "sector"]
        for c in cols:
            if c not in df.columns:
                df[c] = None
        df = df[cols]
        ensure_dir(METADATA_DIR)
        df.to_csv(output_path, index=False)
        logging.info(f"  Gespeichert: {output_path.name} ({n_final} Gapper)")
    else:
        # Save empty CSV with headers
        df = pd.DataFrame(columns=["date", "ticker", "name", "gap_pct", "gap_direction",
                                   "prev_close", "today_open", "premarket_volume", "market_cap",
                                   "primary_exchange", "type", "sic_code", "sic_description", "sector"])
        ensure_dir(METADATA_DIR)
        df.to_csv(output_path, index=False)
        logging.info(f"  Keine Gapper gefunden für {scan_date}")

    logging.info(f"Scanned {scan_date}: {n_total} total → {n_gap} gap → {n_vol} volume "
                 f"→ {n_exchange} exchange → {n_final} final gappers")


def main():
    dates, args = parse_date_args()
    setup_logging("01_scan_gappers", dates[0] if len(dates) == 1 else f"{dates[0]}_to_{dates[-1]}")

    client = get_polygon_client()
    cache_path = METADATA_DIR / "ticker_cache.json"
    cache = load_ticker_cache(cache_path)

    logging.info(f"=== Gapper Scanner ===")
    logging.info(f"Scanne {len(dates)} Handelstag(e)")

    success = 0
    errors = 0
    for scan_date in tqdm(dates, desc="Scanning days"):
        try:
            scan_single_day(scan_date, client, cache, cache_path, force=args.force)
            success += 1
        except Exception as e:
            logging.error(f"Fehler bei {scan_date}: {e}", exc_info=True)
            errors += 1

    logging.info(f"\n=== Zusammenfassung: {success} erfolgreich, {errors} Fehler ===")


if __name__ == "__main__":
    main()
