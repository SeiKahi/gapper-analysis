# 06_download_followup_days.py — Follow-Up-Tage nach Gap-Events herunterladen
#
# Fuer jedes Gap-Event in metadata_v9.parquet:
#   1. Die naechsten 10 Handelstage bestimmen (NYSE Kalender)
#   2. 1-Min Bars von Polygon API holen (identisches Format wie Gap-Tag)
#   3. VWAP + Volume Profile berechnen
#   4. Event-Mapping speichern (Gap-Tag -> Follow-Up-Tage Zuordnung)
#
# Nutzung:
#   python scripts/06_download_followup_days.py                    # Alles
#   python scripts/06_download_followup_days.py --phase mapping    # Nur Mapping erstellen
#   python scripts/06_download_followup_days.py --phase download   # Nur Downloads
#   python scripts/06_download_followup_days.py --phase compute    # Nur VWAP+VP berechnen
#   python scripts/06_download_followup_days.py --workers 10       # Mehr Threads
#   python scripts/06_download_followup_days.py --force             # Existierende ueberschreiben
#   python scripts/06_download_followup_days.py --days 5           # Nur 5 Follow-Up-Tage
#   python scripts/06_download_followup_days.py --dry-run          # Nur zaehlen, nicht laden

import sys
import time
import logging
import argparse
import threading
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import exchange_calendars as xcals
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    METADATA_DIR, RAW_1MIN_DIR, VWAP_DIR, VOLUME_PROFILE_DIR,
    VWAP_SHIFT, VWAP_STD_DEVS, VWAP_BAND_WIDTH, VP_VALUE_AREA_PCT
)
from scripts.utils import (
    get_polygon_client, timestamp_to_et, retry_api_call,
    save_parquet, load_parquet, ensure_dir, setup_logging
)

# ============================================================
# Konstanten
# ============================================================
FOLLOWUP_DAYS = 10           # Default: 10 Handelstage nach Gap
MAPPING_FILE = METADATA_DIR / "followup_day_mapping.parquet"
PROGRESS_FILE = METADATA_DIR / "followup_download_progress.parquet"

# NYSE Kalender (gecacht)
NYSE = xcals.get_calendar("XNYS")


# ============================================================
# PHASE 1: Mapping erstellen
# ============================================================

def get_next_n_trading_days(date_str: str, n: int) -> list:
    """Gibt die naechsten n NYSE-Handelstage nach date_str zurueck."""
    start = pd.Timestamp(date_str) + pd.Timedelta(days=1)
    # Suche bis zu 30 Tage voraus (genug fuer 10 Handelstage + Feiertage)
    end = start + pd.Timedelta(days=n * 3)
    try:
        sessions = NYSE.sessions_in_range(
            start.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d")
        )
        return [s.strftime("%Y-%m-%d") for s in sessions[:n]]
    except Exception:
        # Fallback: Business Days
        bdays = pd.bdate_range(start, end)
        return [d.strftime("%Y-%m-%d") for d in bdays[:n]]


def create_followup_mapping(metadata_path: Path, n_days: int) -> pd.DataFrame:
    """Erstellt Mapping: Fuer jedes Event die naechsten n Handelstage."""
    logging.info("=== Phase 1: Erstelle Follow-Up-Tag-Mapping ===")

    meta = pd.read_parquet(metadata_path)
    logging.info(f"  Events in metadata: {len(meta)}")

    rows = []
    for _, event in tqdm(meta.iterrows(), total=len(meta), desc="Mapping erstellen"):
        ticker = event["ticker"]
        gap_date = str(event["date"])
        followup_dates = get_next_n_trading_days(gap_date, n_days)

        for offset, actual_date in enumerate(followup_dates, start=1):
            rows.append({
                "ticker": ticker,
                "gap_date": gap_date,
                "offset": offset,
                "actual_date": actual_date,
            })

    mapping = pd.DataFrame(rows)
    save_parquet(mapping, MAPPING_FILE)
    logging.info(f"  Mapping erstellt: {len(mapping)} Zeilen ({len(meta)} Events x bis zu {n_days} Tage)")

    # Statistik
    unique_pairs = mapping.drop_duplicates(subset=["ticker", "actual_date"])
    logging.info(f"  Unique (Ticker, Datum) Paare: {len(unique_pairs)}")

    # Check was schon existiert
    existing = 0
    for _, row in unique_pairs.iterrows():
        path = RAW_1MIN_DIR / row["ticker"] / f"{row['actual_date']}.parquet"
        if path.exists():
            existing += 1

    logging.info(f"  Bereits vorhanden: {existing}")
    logging.info(f"  Noch herunterzuladen: {len(unique_pairs) - existing}")

    return mapping


# ============================================================
# PHASE 2: Download von Polygon API
# ============================================================

def download_one_ticker_date(client, ticker: str, date_str: str,
                              api_lock: threading.Lock, force: bool = False) -> tuple:
    """Laedt 1-Min Bars fuer einen Ticker/Datum. Gibt (ticker, date, success) zurueck."""
    output_dir = RAW_1MIN_DIR / ticker
    output_path = output_dir / f"{date_str}.parquet"

    if output_path.exists() and not force:
        return (ticker, date_str, "skipped")

    # Rate Limiting
    with api_lock:
        time.sleep(0.15)

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
        error_str = str(e)
        if "NOT_FOUND" in error_str or "not found" in error_str:
            return (ticker, date_str, "not_found")
        if "NOT_AUTHORIZED" in error_str:
            return (ticker, date_str, "not_authorized")
        logging.warning(f"  Download fehlgeschlagen: {ticker} {date_str}: {e}")
        return (ticker, date_str, "error")

    if not aggs:
        return (ticker, date_str, "no_data")

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
        return (ticker, date_str, "no_bars")

    df = pd.DataFrame(rows)
    df["timestamp_utc"] = df["timestamp_utc"].astype("int64")
    df["volume"] = df["volume"].astype("int64")
    save_parquet(df, output_path)
    return (ticker, date_str, "success")


def get_polygon_min_date() -> str:
    """Bestimmt das frueheste verfuegbare Datum im Polygon-Plan (5 Jahre zurueck)."""
    from datetime import datetime, timedelta
    # Polygon 5-Jahres-Plan: Daten ab heute - 5 Jahre
    min_date = datetime.now() - timedelta(days=5*365)
    return min_date.strftime("%Y-%m-%d")


def download_followup_data(mapping: pd.DataFrame, max_workers: int = 5,
                            force: bool = False, dry_run: bool = False):
    """Laedt alle Follow-Up-Tage herunter."""
    logging.info("=== Phase 2: Download Follow-Up-Daten ===")

    # Polygon Plan-Limit bestimmen
    min_date = get_polygon_min_date()
    logging.info(f"  Polygon Plan-Minimum-Datum: {min_date}")

    # Deduplizieren: jede (ticker, date) Kombination nur einmal
    unique_pairs = mapping[["ticker", "actual_date"]].drop_duplicates()
    unique_pairs = unique_pairs.rename(columns={"actual_date": "date"})

    # Filtern: nur was noch nicht existiert UND im Plan-Fenster liegt
    to_download = []
    skipped_plan = 0
    for _, row in unique_pairs.iterrows():
        if row["date"] < min_date:
            skipped_plan += 1
            continue
        path = RAW_1MIN_DIR / row["ticker"] / f"{row['date']}.parquet"
        if not path.exists() or force:
            to_download.append((row["ticker"], row["date"]))

    logging.info(f"  Unique Paare gesamt: {len(unique_pairs)}")
    logging.info(f"  Ausserhalb Polygon-Plan (vor {min_date}): {skipped_plan}")
    logging.info(f"  Noch herunterzuladen: {len(to_download)}")
    logging.info(f"  Geschaetzte Zeit: {len(to_download) * 0.15 / 60:.0f} Minuten "
                 f"(bei {1/0.15:.0f} req/s, {max_workers} workers)")

    if dry_run:
        logging.info("  [DRY RUN] Kein Download durchgefuehrt.")
        ticker_counts = defaultdict(int)
        for t, d in to_download:
            ticker_counts[t] += 1
        top10 = sorted(ticker_counts.items(), key=lambda x: -x[1])[:10]
        logging.info(f"  Top 10 Ticker nach Downloads:")
        for t, c in top10:
            logging.info(f"    {t}: {c} Tage")
        return

    if not to_download:
        logging.info("  Nichts herunterzuladen — alles vorhanden!")
        return

    client = get_polygon_client()
    api_lock = threading.Lock()

    stats = {"success": 0, "skipped": 0, "error": 0, "no_data": 0,
             "not_found": 0, "no_bars": 0, "not_authorized": 0}
    save_interval = 1000

    def _download_one(args):
        ticker, date_str = args
        return download_one_ticker_date(client, ticker, date_str, api_lock, force)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_download_one, pair): pair for pair in to_download}

        for i, future in enumerate(tqdm(as_completed(futures), total=len(futures),
                                         desc="Downloading")):
            try:
                ticker, date_str, status = future.result()
                stats[status] += 1
            except Exception as e:
                logging.error(f"  Unerwarteter Fehler: {e}")
                stats["error"] += 1

            if (i + 1) % save_interval == 0:
                logging.info(f"  Fortschritt: {i+1}/{len(to_download)} — "
                             f"OK={stats['success']}, Skip={stats['skipped']}, "
                             f"Err={stats['error']}, NoData={stats['no_data']}, "
                             f"NotAuth={stats['not_authorized']}")

    logging.info(f"  === Download abgeschlossen ===")
    logging.info(f"  Erfolgreich:      {stats['success']}")
    logging.info(f"  Uebersprungen:    {stats['skipped']}")
    logging.info(f"  Keine Daten:      {stats['no_data']}")
    logging.info(f"  Nicht gefunden:   {stats['not_found']}")
    logging.info(f"  Nicht autorisiert:{stats['not_authorized']}")
    logging.info(f"  Fehler:           {stats['error']}")


# ============================================================
# PHASE 3: VWAP + Volume Profile berechnen
# ============================================================

def compute_vwap_for_file(ticker: str, date_str: str, force: bool = False) -> bool:
    """Berechnet VWAP + StdDev Baender (identisch zu 04_compute_vwap.py)."""
    input_path = RAW_1MIN_DIR / ticker / f"{date_str}.parquet"
    output_path = VWAP_DIR / ticker / f"{date_str}.parquet"

    if output_path.exists() and not force:
        return True

    if not input_path.exists():
        return False

    df = load_parquet(input_path)
    rth = df[df["session"] == "rth"].copy().reset_index(drop=True)

    if len(rth) < 10:
        return False

    # Kumulative VWAP
    volumes = rth["volume"].values.astype(float)
    bar_vwaps = rth["vwap"].values.astype(float)

    nan_mask = np.isnan(bar_vwaps)
    if nan_mask.any():
        typical = (rth["high"].values + rth["low"].values + rth["close"].values) / 3
        bar_vwaps[nan_mask] = typical[nan_mask]

    cum_volume = np.cumsum(volumes)
    cum_pv = np.cumsum(bar_vwaps * volumes)
    cum_volume_safe = np.where(cum_volume == 0, 1, cum_volume)
    vwap_values = cum_pv / cum_volume_safe

    n_bars = len(rth)
    std_values = np.zeros(n_bars)
    for i in range(n_bars):
        if cum_volume[i] == 0:
            std_values[i] = 0
            continue
        diffs_sq = volumes[:i+1] * (bar_vwaps[:i+1] - vwap_values[i]) ** 2
        variance = np.sum(diffs_sq) / cum_volume[i]
        std_values[i] = np.sqrt(variance)

    # SHIFT um 1 Bar
    vwap_shifted = np.full(n_bars, np.nan)
    std_shifted = np.full(n_bars, np.nan)
    vwap_shifted[1:] = vwap_values[:-1]
    std_shifted[1:] = std_values[:-1]

    result = pd.DataFrame({
        "datetime_et": rth["datetime_et"].values,
        "time_et": rth["time_et"].values,
        "open": rth["open"].values,
        "high": rth["high"].values,
        "low": rth["low"].values,
        "close": rth["close"].values,
        "volume": rth["volume"].values,
        "vwap": vwap_shifted,
        "std_dev": std_shifted,
    })

    for level in VWAP_STD_DEVS:
        result[f"upper_{level}std"] = vwap_shifted + level * std_shifted
        result[f"lower_{level}std"] = vwap_shifted - level * std_shifted

    result["z_score"] = np.where(
        std_shifted > 0,
        (rth["close"].values - vwap_shifted) / std_shifted,
        np.nan
    )

    band_w = VWAP_BAND_WIDTH
    result["vwap_band_upper"] = vwap_shifted + band_w * std_shifted
    result["vwap_band_lower"] = vwap_shifted - band_w * std_shifted

    for level in VWAP_STD_DEVS:
        upper_line = vwap_shifted + level * std_shifted
        lower_line = vwap_shifted - level * std_shifted
        result[f"upper_{level}std_band_upper"] = upper_line + band_w * std_shifted
        result[f"upper_{level}std_band_lower"] = upper_line - band_w * std_shifted
        result[f"lower_{level}std_band_upper"] = lower_line + band_w * std_shifted
        result[f"lower_{level}std_band_lower"] = lower_line - band_w * std_shifted

    result["ticker"] = ticker
    result["date"] = date_str

    save_parquet(result, output_path)
    return True


def compute_volume_profile_for_file(ticker: str, date_str: str, force: bool = False) -> bool:
    """Berechnet Volume Profile (identisch zu 05_compute_volume_profile.py)."""
    input_path = RAW_1MIN_DIR / ticker / f"{date_str}.parquet"
    output_path = VOLUME_PROFILE_DIR / ticker / f"{date_str}.parquet"

    if output_path.exists() and not force:
        return True

    if not input_path.exists():
        return False

    df = load_parquet(input_path)
    rth = df[df["session"] == "rth"].copy().reset_index(drop=True)

    if len(rth) < 10:
        return False

    day_high = rth["high"].max()
    day_low = rth["low"].min()
    price_range = day_high - day_low

    # Dynamische Bucket-Groesse
    if price_range < 5:
        bucket_size = 0.05
    elif price_range < 20:
        bucket_size = 0.10
    elif price_range < 100:
        bucket_size = 0.25
    else:
        bucket_size = 0.50

    bar_vwaps = rth["vwap"].values.astype(float)
    nan_mask = np.isnan(bar_vwaps)
    if nan_mask.any():
        typical = (rth["high"].values + rth["low"].values + rth["close"].values) / 3
        bar_vwaps[nan_mask] = typical[nan_mask]

    volumes = rth["volume"].values
    snapshots = []
    cumulative_profile = defaultdict(float)

    for i in range(len(rth)):
        bucket = round(bar_vwaps[i] / bucket_size) * bucket_size
        cumulative_profile[bucket] += volumes[i]

        minute_num = i + 1
        if minute_num % 5 == 0 or minute_num == len(rth):
            # VPOC, VAH, VAL berechnen
            if not cumulative_profile:
                continue
            vpoc = max(cumulative_profile, key=cumulative_profile.get)
            total_volume = sum(cumulative_profile.values())

            if total_volume == 0:
                vah, val = vpoc, vpoc
            else:
                target_volume = total_volume * (VP_VALUE_AREA_PCT / 100.0)
                sorted_buckets = sorted(cumulative_profile.keys())
                vpoc_idx = sorted_buckets.index(vpoc)
                va_volume = cumulative_profile[vpoc]
                lower_idx = vpoc_idx
                upper_idx = vpoc_idx

                while va_volume < target_volume:
                    can_go_up = upper_idx + 1 < len(sorted_buckets)
                    can_go_down = lower_idx - 1 >= 0
                    if not can_go_up and not can_go_down:
                        break
                    vol_up = cumulative_profile[sorted_buckets[upper_idx + 1]] if can_go_up else -1
                    vol_down = cumulative_profile[sorted_buckets[lower_idx - 1]] if can_go_down else -1
                    if vol_up >= vol_down:
                        upper_idx += 1
                        va_volume += cumulative_profile[sorted_buckets[upper_idx]]
                    else:
                        lower_idx -= 1
                        va_volume += cumulative_profile[sorted_buckets[lower_idx]]

                vah = sorted_buckets[upper_idx]
                val = sorted_buckets[lower_idx]

            snapshots.append({
                "datetime_et": rth.iloc[i]["datetime_et"],
                "minutes_since_open": minute_num,
                "vpoc": vpoc,
                "vah": vah,
                "val": val,
                "total_volume": total_volume,
                "num_buckets": len(cumulative_profile),
                "bucket_size": bucket_size,
                "ticker": ticker,
                "date": date_str,
            })

    if not snapshots:
        return False

    result = pd.DataFrame(snapshots)
    save_parquet(result, output_path)
    return True


def compute_derived_data(mapping: pd.DataFrame, force: bool = False):
    """Berechnet VWAP und Volume Profile fuer alle heruntergeladenen Follow-Up-Tage."""
    logging.info("=== Phase 3: VWAP + Volume Profile berechnen ===")

    unique_pairs = mapping[["ticker", "actual_date"]].drop_duplicates()

    # Nur Paare wo raw_1min existiert aber VWAP/VP fehlt
    to_compute_vwap = []
    to_compute_vp = []

    for _, row in unique_pairs.iterrows():
        ticker, date_str = row["ticker"], row["actual_date"]
        raw_path = RAW_1MIN_DIR / ticker / f"{date_str}.parquet"
        if not raw_path.exists():
            continue

        vwap_path = VWAP_DIR / ticker / f"{date_str}.parquet"
        vp_path = VOLUME_PROFILE_DIR / ticker / f"{date_str}.parquet"

        if not vwap_path.exists() or force:
            to_compute_vwap.append((ticker, date_str))
        if not vp_path.exists() or force:
            to_compute_vp.append((ticker, date_str))

    logging.info(f"  VWAP zu berechnen: {len(to_compute_vwap)}")
    logging.info(f"  Volume Profile zu berechnen: {len(to_compute_vp)}")

    # VWAP berechnen
    if to_compute_vwap:
        vwap_ok = 0
        vwap_err = 0
        for ticker, date_str in tqdm(to_compute_vwap, desc="Computing VWAP"):
            try:
                if compute_vwap_for_file(ticker, date_str, force):
                    vwap_ok += 1
                else:
                    vwap_err += 1
            except Exception as e:
                logging.debug(f"  VWAP Fehler {ticker} {date_str}: {e}")
                vwap_err += 1

        logging.info(f"  VWAP: {vwap_ok} erfolgreich, {vwap_err} Fehler")

    # Volume Profile berechnen
    if to_compute_vp:
        vp_ok = 0
        vp_err = 0
        for ticker, date_str in tqdm(to_compute_vp, desc="Computing VP"):
            try:
                if compute_volume_profile_for_file(ticker, date_str, force):
                    vp_ok += 1
                else:
                    vp_err += 1
            except Exception as e:
                logging.debug(f"  VP Fehler {ticker} {date_str}: {e}")
                vp_err += 1

        logging.info(f"  VP: {vp_ok} erfolgreich, {vp_err} Fehler")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Follow-Up-Tage nach Gap-Events herunterladen")
    parser.add_argument("--phase", choices=["mapping", "download", "compute", "all"],
                        default="all", help="Welche Phase ausfuehren")
    parser.add_argument("--days", type=int, default=FOLLOWUP_DAYS,
                        help=f"Anzahl Follow-Up-Handelstage (default: {FOLLOWUP_DAYS})")
    parser.add_argument("--workers", type=int, default=5,
                        help="Anzahl Download-Threads (default: 5)")
    parser.add_argument("--force", action="store_true",
                        help="Existierende Dateien ueberschreiben")
    parser.add_argument("--dry-run", action="store_true",
                        help="Nur zaehlen, nichts herunterladen")
    args = parser.parse_args()

    setup_logging("06_download_followup", f"days{args.days}")

    metadata_path = METADATA_DIR / "metadata_v9.parquet"
    if not metadata_path.exists():
        logging.error(f"metadata_v9.parquet nicht gefunden: {metadata_path}")
        return

    logging.info("=" * 70)
    logging.info("FOLLOW-UP-TAGE DOWNLOAD PIPELINE")
    logging.info(f"  Follow-Up-Tage: {args.days}")
    logging.info(f"  Workers: {args.workers}")
    logging.info(f"  Force: {args.force}")
    logging.info(f"  Phase: {args.phase}")
    logging.info("=" * 70)

    t0 = time.time()

    # Phase 1: Mapping
    if args.phase in ("mapping", "all"):
        mapping = create_followup_mapping(metadata_path, args.days)
    else:
        if MAPPING_FILE.exists():
            mapping = pd.read_parquet(MAPPING_FILE)
            logging.info(f"  Mapping geladen: {len(mapping)} Zeilen")
        else:
            logging.error("Kein Mapping vorhanden. Bitte zuerst --phase mapping ausfuehren.")
            return

    # Phase 2: Download
    if args.phase in ("download", "all"):
        download_followup_data(mapping, max_workers=args.workers,
                                force=args.force, dry_run=args.dry_run)

    # Phase 3: VWAP + Volume Profile
    if args.phase in ("compute", "all") and not args.dry_run:
        compute_derived_data(mapping, force=args.force)

    elapsed = time.time() - t0
    logging.info(f"\n  Gesamtzeit: {elapsed/60:.1f} Minuten")
    logging.info("=" * 70)


if __name__ == "__main__":
    main()
