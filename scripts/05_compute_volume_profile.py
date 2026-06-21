# 05_compute_volume_profile.py — Developing Volume Profile

import sys
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import METADATA_DIR, RAW_1MIN_DIR, VOLUME_PROFILE_DIR, VP_VALUE_AREA_PCT
from scripts.utils import load_parquet, save_parquet, setup_logging, parse_date_args


def determine_bucket_size(price_range: float) -> float:
    """Bestimmt dynamische Bucket-Größe basierend auf Preisspanne."""
    if price_range < 5:
        return 0.05
    elif price_range < 20:
        return 0.10
    elif price_range < 100:
        return 0.25
    else:
        return 0.50


def compute_value_area(volume_profile: dict, target_pct: float = 70.0):
    """Berechnet VPOC, VAH, VAL aus einem Volume Profile."""
    if not volume_profile:
        return None, None, None

    # VPOC = Bucket mit höchstem Volume
    vpoc = max(volume_profile, key=volume_profile.get)
    total_volume = sum(volume_profile.values())

    if total_volume == 0:
        return vpoc, vpoc, vpoc

    target_volume = total_volume * (target_pct / 100.0)

    # Starte bei VPOC, expandiere nach oben und unten
    sorted_buckets = sorted(volume_profile.keys())
    vpoc_idx = sorted_buckets.index(vpoc)

    va_volume = volume_profile[vpoc]
    lower_idx = vpoc_idx
    upper_idx = vpoc_idx

    while va_volume < target_volume:
        can_go_up = upper_idx + 1 < len(sorted_buckets)
        can_go_down = lower_idx - 1 >= 0

        if not can_go_up and not can_go_down:
            break

        vol_up = volume_profile[sorted_buckets[upper_idx + 1]] if can_go_up else -1
        vol_down = volume_profile[sorted_buckets[lower_idx - 1]] if can_go_down else -1

        if vol_up >= vol_down:
            upper_idx += 1
            va_volume += volume_profile[sorted_buckets[upper_idx]]
        else:
            lower_idx -= 1
            va_volume += volume_profile[sorted_buckets[lower_idx]]

    vah = sorted_buckets[upper_idx]
    val = sorted_buckets[lower_idx]

    return vpoc, vah, val


def compute_volume_profile_for_ticker(ticker: str, date_str: str, force: bool = False) -> bool:
    """Berechnet Developing Volume Profile mit 5-Min Snapshots."""
    input_path = RAW_1MIN_DIR / ticker / f"{date_str}.parquet"
    output_path = VOLUME_PROFILE_DIR / ticker / f"{date_str}.parquet"

    if output_path.exists() and not force:
        logging.debug(f"  Überspringe {ticker}: VP existiert")
        return True

    if not input_path.exists():
        logging.warning(f"  Keine Intraday-Daten für {ticker} am {date_str}")
        return False

    df = load_parquet(input_path)
    rth = df[df["session"] == "rth"].copy().reset_index(drop=True)

    if len(rth) < 10:
        logging.warning(f"  Zu wenige RTH-Bars für {ticker}: {len(rth)}")
        return False

    # Bestimme Bucket-Größe
    day_high = rth["high"].max()
    day_low = rth["low"].min()
    price_range = day_high - day_low
    bucket_size = determine_bucket_size(price_range)

    # Bar-Level VWAP für Bucket-Zuordnung
    bar_vwaps = rth["vwap"].values.astype(float)
    nan_mask = np.isnan(bar_vwaps)
    if nan_mask.any():
        typical = (rth["high"].values + rth["low"].values + rth["close"].values) / 3
        bar_vwaps[nan_mask] = typical[nan_mask]

    volumes = rth["volume"].values

    # Snapshots alle 5 Minuten
    snapshots = []
    cumulative_profile = defaultdict(float)

    for i in range(len(rth)):
        bucket = round(bar_vwaps[i] / bucket_size) * bucket_size
        cumulative_profile[bucket] += volumes[i]

        minute_num = i + 1  # 1-basiert
        if minute_num % 5 == 0 or minute_num == len(rth):
            vpoc, vah, val = compute_value_area(dict(cumulative_profile), VP_VALUE_AREA_PCT)
            snapshots.append({
                "datetime_et": rth.iloc[i]["datetime_et"],
                "minutes_since_open": minute_num,
                "vpoc": vpoc,
                "vah": vah,
                "val": val,
                "total_volume": sum(cumulative_profile.values()),
                "num_buckets": len(cumulative_profile),
                "bucket_size": bucket_size,
                "ticker": ticker,
                "date": date_str,
            })

    if not snapshots:
        logging.warning(f"  Keine VP-Snapshots für {ticker}")
        return False

    result = pd.DataFrame(snapshots)
    save_parquet(result, output_path)
    logging.debug(f"  {ticker}: {len(result)} VP-Snapshots gespeichert")
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
    logging.info(f"{date_str}: Berechne Volume Profile für {len(tickers)} Gapper")

    success = 0
    errors = 0
    for ticker in tqdm(tickers, desc=f"  {date_str}", leave=False):
        try:
            if compute_volume_profile_for_ticker(ticker, date_str, force):
                success += 1
            else:
                errors += 1
        except Exception as e:
            logging.error(f"  VP Fehler bei {ticker}: {e}")
            errors += 1

    logging.info(f"  {date_str}: {success} erfolgreich, {errors} Fehler")


def main():
    dates, args = parse_date_args()
    setup_logging("05_compute_volume_profile", dates[0] if len(dates) == 1 else f"{dates[0]}_to_{dates[-1]}")

    logging.info(f"=== Volume Profile Berechnung ===")
    logging.info(f"Verarbeite {len(dates)} Handelstag(e)")

    for date_str in tqdm(dates, desc="Computing VP"):
        try:
            process_date(date_str, force=args.force)
        except Exception as e:
            logging.error(f"Fehler bei {date_str}: {e}", exc_info=True)

    logging.info("=== Volume Profile Berechnung abgeschlossen ===")


if __name__ == "__main__":
    main()
