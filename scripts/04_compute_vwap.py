# 04_compute_vwap.py — VWAP + StdDev Bänder berechnen

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    METADATA_DIR, RAW_1MIN_DIR, VWAP_DIR,
    VWAP_SHIFT, VWAP_STD_DEVS, VWAP_BAND_WIDTH
)
from scripts.utils import (
    load_parquet, save_parquet, setup_logging, parse_date_args
)


def compute_vwap_for_ticker(ticker: str, date_str: str, force: bool = False) -> bool:
    """Berechnet VWAP + StdDev Bänder aus RTH 1-Min Bars."""
    input_path = RAW_1MIN_DIR / ticker / f"{date_str}.parquet"
    output_path = VWAP_DIR / ticker / f"{date_str}.parquet"

    if output_path.exists() and not force:
        logging.debug(f"  Überspringe {ticker}: VWAP existiert")
        return True

    if not input_path.exists():
        logging.warning(f"  Keine Intraday-Daten für {ticker} am {date_str}")
        return False

    df = load_parquet(input_path)
    rth = df[df["session"] == "rth"].copy().reset_index(drop=True)

    if len(rth) < 10:
        logging.warning(f"  Zu wenige RTH-Bars für {ticker} am {date_str}: {len(rth)}")
        return False

    # Kumulative VWAP-Berechnung mit bar-level VWAP
    volumes = rth["volume"].values.astype(float)
    bar_vwaps = rth["vwap"].values.astype(float)

    # Handle NaN vwaps: fallback to (H+L+C)/3
    nan_mask = np.isnan(bar_vwaps)
    if nan_mask.any():
        typical = (rth["high"].values + rth["low"].values + rth["close"].values) / 3
        bar_vwaps[nan_mask] = typical[nan_mask]

    cum_volume = np.cumsum(volumes)
    cum_pv = np.cumsum(bar_vwaps * volumes)

    # Avoid division by zero
    cum_volume_safe = np.where(cum_volume == 0, 1, cum_volume)
    vwap_values = cum_pv / cum_volume_safe

    # Kumulative Standardabweichung (volumengewichtet)
    n_bars = len(rth)
    std_values = np.zeros(n_bars)
    for i in range(n_bars):
        if cum_volume[i] == 0:
            std_values[i] = 0
            continue
        # Volumengewichtete Varianz
        diffs_sq = volumes[:i+1] * (bar_vwaps[:i+1] - vwap_values[i]) ** 2
        variance = np.sum(diffs_sq) / cum_volume[i]
        std_values[i] = np.sqrt(variance)

    # SHIFT um 1 Bar nach rechts (Look-Ahead-Bias)
    vwap_shifted = np.full(n_bars, np.nan)
    std_shifted = np.full(n_bars, np.nan)
    vwap_shifted[1:] = vwap_values[:-1]
    std_shifted[1:] = std_values[:-1]

    # Bänder berechnen
    # UPDATE: High und Low hinzugefügt!
    result = pd.DataFrame({
        "datetime_et": rth["datetime_et"].values,
        "time_et": rth["time_et"].values,
        "open": rth["open"].values,   # NEU
        "high": rth["high"].values,   # NEU
        "low": rth["low"].values,     # NEU
        "close": rth["close"].values,
        "volume": rth["volume"].values,
        "vwap": vwap_shifted,
        "std_dev": std_shifted,
    })

    for level in VWAP_STD_DEVS:
        result[f"upper_{level}std"] = vwap_shifted + level * std_shifted
        result[f"lower_{level}std"] = vwap_shifted - level * std_shifted

    # Z-Score
    result["z_score"] = np.where(
        std_shifted > 0,
        (rth["close"].values - vwap_shifted) / std_shifted,
        np.nan
    )

    # 0.25σ Sub-Bands für alle 7 Linien
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
    logging.debug(f"  {ticker}: VWAP berechnet ({len(result)} Bars)")
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
    logging.info(f"{date_str}: Berechne VWAP für {len(tickers)} Gapper")

    success = 0
    errors = 0
    for ticker in tqdm(tickers, desc=f"  {date_str}", leave=False):
        try:
            if compute_vwap_for_ticker(ticker, date_str, force):
                success += 1
            else:
                errors += 1
        except Exception as e:
            logging.error(f"  VWAP Fehler bei {ticker}: {e}")
            errors += 1

    logging.info(f"  {date_str}: {success} erfolgreich, {errors} Fehler")


def main():
    dates, args = parse_date_args()
    setup_logging("04_compute_vwap", dates[0] if len(dates) == 1 else f"{dates[0]}_to_{dates[-1]}")

    logging.info(f"=== VWAP Berechnung ===")
    logging.info(f"Verarbeite {len(dates)} Handelstag(e)")

    for date_str in tqdm(dates, desc="Computing VWAP"):
        try:
            process_date(date_str, force=args.force)
        except Exception as e:
            logging.error(f"Fehler bei {date_str}: {e}", exc_info=True)

    logging.info("=== VWAP Berechnung abgeschlossen ===")


if __name__ == "__main__":
    main()