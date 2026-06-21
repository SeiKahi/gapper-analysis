# 07_compute_metadata.py — Kontext-Metadaten berechnen (~40 Features pro Gapper)

import sys
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    METADATA_DIR, RAW_1MIN_DIR, DAILY_DIR, VWAP_DIR, VOLUME_PROFILE_DIR,
    NORMALIZED_DIR, WATCHLIST_DIR, BASELINE_1MIN_DIR, OPENING_DRIVE_MINUTES,
    VWAP_BAND_WIDTH, ADR_LOOKBACK, SMA_PERIOD, RSI_PERIOD
)
from scripts.utils import (
    load_parquet, save_parquet, ensure_dir, sic_to_sector,
    setup_logging, parse_date_args, get_previous_trading_day, get_trading_days
)


def _safe_rsi(close_series: pd.Series, period: int = 14):
    """Berechnet RSI mit pandas_ta falls verfügbar, sonst manuell."""
    try:
        import pandas_ta as ta
        rsi = ta.rsi(close_series, length=period)
        return rsi
    except Exception:
        delta = close_series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))


def _load_ticker_cache() -> dict:
    """Lädt den Ticker-Details-Cache."""
    cache_path = METADATA_DIR / "ticker_cache.json"
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _get_spy_return(scan_date: str) -> float:
    """Berechnet SPY Tagesreturn."""
    spy_path = DAILY_DIR / "SPY" / f"{scan_date}_context.parquet"
    if not spy_path.exists():
        return np.nan
    try:
        df_spy = load_parquet(spy_path)
        df_spy = df_spy.sort_values("date").reset_index(drop=True)
        if len(df_spy) < 2:
            return np.nan
        # Finde scan_date und Vortag
        spy_today = df_spy[df_spy["date"] == scan_date]
        spy_prev = df_spy[df_spy["date"] < scan_date]
        if spy_today.empty or spy_prev.empty:
            return np.nan
        spy_close = spy_today.iloc[-1]["close"]
        spy_prev_close = spy_prev.iloc[-1]["close"]
        return (spy_close - spy_prev_close) / spy_prev_close * 100
    except Exception:
        return np.nan


def _get_vix_level(scan_date: str) -> float:
    """Holt VIX-Level vom Vortag."""
    vix_path = DAILY_DIR / "VIX" / f"{scan_date}_context.parquet"
    if not vix_path.exists():
        return np.nan
    try:
        df_vix = load_parquet(vix_path)
        df_vix = df_vix.sort_values("date").reset_index(drop=True)
        vix_prev = df_vix[df_vix["date"] < scan_date]
        if vix_prev.empty:
            # Fallback: letzter verfügbarer Wert
            return df_vix.iloc[-1]["close"] if not df_vix.empty else np.nan
        return vix_prev.iloc[-1]["close"]
    except Exception:
        return np.nan


def _check_watchlist_catalyst(ticker: str, scan_date: str) -> str:
    """Prüft ob ein Katalysator aus der Watchlist bekannt ist."""
    try:
        watchlist_files = list(WATCHLIST_DIR.glob("*.csv"))
        for wf in watchlist_files:
            df_wl = pd.read_csv(wf)
            match = df_wl[
                (df_wl["ticker"] == ticker) &
                (df_wl["date"] == scan_date)
            ]
            if not match.empty and "catalyst_type" in match.columns:
                val = match.iloc[0]["catalyst_type"]
                if pd.notna(val) and str(val).strip():
                    return str(val).strip()
    except Exception:
        pass
    return "unknown"


def compute_metadata_for_ticker(
    ticker: str, date_str: str, gapper_row: pd.Series,
    ticker_cache: dict, spy_return: float, vix_level: float
) -> dict:
    """Berechnet alle ~40 Metadaten für einen einzelnen Gapper."""

    meta = {
        "date": date_str,
        "ticker": ticker,
    }

    # === Daten laden ===
    intraday_path = RAW_1MIN_DIR / ticker / f"{date_str}.parquet"
    daily_path = DAILY_DIR / ticker / f"{date_str}_context.parquet"
    vwap_path = VWAP_DIR / ticker / f"{date_str}.parquet"
    vp_path = VOLUME_PROFILE_DIR / ticker / f"{date_str}.parquet"
    norm_path = NORMALIZED_DIR / ticker / f"{date_str}.parquet"

    df_intraday = load_parquet(intraday_path) if intraday_path.exists() else None
    df_daily = load_parquet(daily_path) if daily_path.exists() else None
    df_vwap = load_parquet(vwap_path) if vwap_path.exists() else None
    df_vp = load_parquet(vp_path) if vp_path.exists() else None
    df_norm = load_parquet(norm_path) if norm_path.exists() else None

    # RTH Bars
    rth = None
    if df_intraday is not None:
        rth = df_intraday[df_intraday["session"] == "rth"].copy().reset_index(drop=True)
        if rth.empty:
            rth = None

    # Daily-Daten vor gap_date
    daily_before = None
    if df_daily is not None:
        daily_before = df_daily[df_daily["date"] < date_str].sort_values("date").reset_index(drop=True)
        if daily_before.empty:
            daily_before = None

    # ================================================================
    # 1. GAP-KONTEXT (aus gapper CSV)
    # ================================================================
    gap_pct = gapper_row.get("gap_pct", np.nan)
    gap_direction = gapper_row.get("gap_direction", "up")
    prev_close = gapper_row.get("prev_close", np.nan)
    today_open = gapper_row.get("today_open", np.nan)
    premarket_volume = gapper_row.get("premarket_volume", 0)

    meta["gap_pct"] = gap_pct
    meta["gap_direction"] = gap_direction
    meta["prev_close"] = prev_close
    meta["today_open"] = today_open
    meta["premarket_volume"] = premarket_volume
    meta["name"] = gapper_row.get("name", "")

    # ================================================================
    # 2. PRE-GAP KONTEXT (aus Daily-Daten)
    # ================================================================
    if daily_before is not None and len(daily_before) >= 5:
        closes = daily_before["close"].values
        pc = closes[-1]  # prev_close from daily

        # Prior Returns
        if len(closes) >= 5:
            meta["prior_return_5d"] = (pc - closes[-5]) / closes[-5] * 100
        else:
            meta["prior_return_5d"] = np.nan

        if len(closes) >= 10:
            meta["prior_return_10d"] = (pc - closes[-10]) / closes[-10] * 100
        else:
            meta["prior_return_10d"] = np.nan

        if len(closes) >= 20:
            meta["prior_return_20d"] = (pc - closes[-20]) / closes[-20] * 100
        else:
            meta["prior_return_20d"] = np.nan

        # SMA(20)
        if len(closes) >= SMA_PERIOD:
            sma_20 = np.mean(closes[-SMA_PERIOD:])
            meta["dist_from_20sma"] = (pc - sma_20) / sma_20 * 100
        else:
            meta["dist_from_20sma"] = np.nan

        # 52-Week High (aus verfügbaren Daten)
        highs = daily_before["high"].values
        high_52w = np.max(highs)
        meta["dist_from_52w_high"] = (pc - high_52w) / high_52w * 100

        # RSI(14)
        close_series = daily_before["close"]
        if len(close_series) >= RSI_PERIOD + 1:
            rsi_vals = _safe_rsi(close_series, RSI_PERIOD)
            last_rsi = rsi_vals.dropna()
            meta["rsi_14_prev"] = last_rsi.iloc[-1] if not last_rsi.empty else np.nan
        else:
            meta["rsi_14_prev"] = np.nan

        # ADR
        daily_before_copy = daily_before.copy()
        daily_before_copy["daily_range"] = daily_before_copy["high"] - daily_before_copy["low"]
        meta["adr_5"] = daily_before_copy["daily_range"].iloc[-5:].mean()
        if len(daily_before_copy) >= 20:
            meta["adr_20"] = daily_before_copy["daily_range"].iloc[-20:].mean()
        else:
            meta["adr_20"] = daily_before_copy["daily_range"].mean()

        # Avg Daily Volume (20d)
        if len(daily_before) >= 20:
            meta["avg_daily_volume_20"] = daily_before["volume"].iloc[-20:].mean()
        else:
            meta["avg_daily_volume_20"] = daily_before["volume"].mean()
    else:
        for col in ["prior_return_5d", "prior_return_10d", "prior_return_20d",
                     "dist_from_20sma", "dist_from_52w_high", "rsi_14_prev",
                     "adr_5", "adr_20", "avg_daily_volume_20"]:
            meta[col] = np.nan

    # Fallback ADR aus normalized
    if np.isnan(meta.get("adr_5", np.nan)) and df_norm is not None:
        meta["adr_5"] = df_norm.iloc[0]["adr_5"] if "adr_5" in df_norm.columns else np.nan
        meta["adr_20"] = df_norm.iloc[0]["adr_20"] if "adr_20" in df_norm.columns else np.nan

    adr_5 = meta.get("adr_5", np.nan)
    if np.isnan(adr_5) or adr_5 == 0:
        adr_5 = 1.0  # Fallback um Division by zero zu vermeiden

    # ================================================================
    # 3. SEKTOR / MARKET CAP
    # ================================================================
    tc = ticker_cache.get(ticker, {})
    meta["sector"] = sic_to_sector(tc.get("sic_code", ""))
    mc = tc.get("market_cap") or gapper_row.get("market_cap")
    meta["market_cap"] = mc
    if mc is not None and not np.isnan(mc):
        if mc < 10e9:
            meta["market_cap_bucket"] = "2-10B"
        elif mc < 50e9:
            meta["market_cap_bucket"] = "10-50B"
        elif mc < 200e9:
            meta["market_cap_bucket"] = "50-200B"
        else:
            meta["market_cap_bucket"] = "200B+"
    else:
        meta["market_cap_bucket"] = "Unknown"

    # Catalyst
    meta["catalyst_type"] = _check_watchlist_catalyst(ticker, date_str)

    # Marktkontext
    meta["spy_return_day"] = spy_return
    meta["vix_level"] = vix_level
    meta["day_of_week"] = pd.Timestamp(date_str).strftime("%a")

    # ================================================================
    # 4. INTRADAY-VOLUMEN
    # ================================================================
    if rth is not None and len(rth) > 0:
        # Erste 30 Minuten: 09:30 - 09:59
        first_30 = rth[rth["time_et"] < "10:00"]
        volume_first_30min = first_30["volume"].sum() if not first_30.empty else 0
        meta["volume_first_30min"] = volume_first_30min

        avg_vol_20 = meta.get("avg_daily_volume_20", np.nan)
        if not np.isnan(avg_vol_20) and avg_vol_20 > 0:
            avg_volume_first_30min = avg_vol_20 * 0.22
            meta["rvol_open_30min"] = volume_first_30min / avg_volume_first_30min
        else:
            meta["rvol_open_30min"] = np.nan

        total_rth_volume = rth["volume"].sum()
        meta["total_rth_volume"] = total_rth_volume
        if not np.isnan(avg_vol_20) and avg_vol_20 > 0:
            meta["rvol_full_day"] = total_rth_volume / avg_vol_20
        else:
            meta["rvol_full_day"] = np.nan
    else:
        for col in ["volume_first_30min", "rvol_open_30min", "total_rth_volume", "rvol_full_day"]:
            meta[col] = np.nan

    # ================================================================
    # 5. INTRADAY-ERGEBNIS
    # ================================================================
    if rth is not None and len(rth) >= 10:
        rth_open = rth.iloc[0]["open"]
        rth_close = rth.iloc[-1]["close"]
        rth_high = rth["high"].max()
        rth_low = rth["low"].min()

        meta["rth_open"] = rth_open
        meta["rth_close"] = rth_close
        meta["rth_high"] = rth_high
        meta["rth_low"] = rth_low

        # --- Opening Drive ---
        first_n = rth.iloc[:OPENING_DRIVE_MINUTES]
        if gap_direction == "up":
            max_move_15 = first_n["high"].max() - rth_open
        else:
            max_move_15 = rth_open - first_n["low"].min()
        meta["opening_drive_atr"] = max_move_15 / adr_5

        price_at_15 = first_n.iloc[-1]["close"] if len(first_n) > 0 else rth_open
        if gap_direction == "up":
            meta["opening_drive_direction"] = "with_gap" if price_at_15 > rth_open else "against_gap"
        else:
            meta["opening_drive_direction"] = "with_gap" if price_at_15 < rth_open else "against_gap"

        # --- HOD/LOD Timing ---
        hod_idx = rth["high"].idxmax()
        lod_idx = rth["low"].idxmin()
        meta["hod_time"] = int(hod_idx)
        meta["lod_time"] = int(lod_idx)

        # --- Gap Fill ---
        if gap_direction == "up":
            fill_bars = rth[rth["low"] <= prev_close]
            meta["gap_filled"] = not fill_bars.empty
            if not fill_bars.empty:
                meta["gap_fill_time_minutes"] = int(fill_bars.index[0])
            else:
                meta["gap_fill_time_minutes"] = np.nan
        else:
            fill_bars = rth[rth["high"] >= prev_close]
            meta["gap_filled"] = not fill_bars.empty
            if not fill_bars.empty:
                meta["gap_fill_time_minutes"] = int(fill_bars.index[0])
            else:
                meta["gap_fill_time_minutes"] = np.nan

        # --- Close vs Open ---
        meta["close_vs_open_adr"] = (rth_close - rth_open) / adr_5

        # --- Max Extension & Adverse ---
        if gap_direction == "up":
            meta["max_extension_adr"] = (rth_high - rth_open) / adr_5
            meta["max_adverse_adr"] = (rth_open - rth_low) / adr_5
        else:
            meta["max_extension_adr"] = (rth_open - rth_low) / adr_5
            meta["max_adverse_adr"] = (rth_high - rth_open) / adr_5

        # --- Reversal Detection ---
        if gap_direction == "up":
            extension_peak = rth_high
            hod_pos = int(hod_idx)
            bars_after = rth.iloc[hod_pos + 1:]
            if not bars_after.empty:
                lowest_after = bars_after["low"].min()
                retracement = extension_peak - lowest_after
            else:
                retracement = 0
            meta["reversal_occurred"] = bool(retracement >= adr_5)
            if meta["reversal_occurred"] and not bars_after.empty:
                rev_idx = bars_after["low"].idxmin()
                meta["reversal_time_minutes"] = int(rev_idx)
            else:
                meta["reversal_time_minutes"] = np.nan
        else:
            extension_peak = rth_low
            lod_pos = int(lod_idx)
            bars_after = rth.iloc[lod_pos + 1:]
            if not bars_after.empty:
                highest_after = bars_after["high"].max()
                retracement = highest_after - extension_peak
            else:
                retracement = 0
            meta["reversal_occurred"] = bool(retracement >= adr_5)
            if meta["reversal_occurred"] and not bars_after.empty:
                rev_idx = bars_after["high"].idxmax()
                meta["reversal_time_minutes"] = int(rev_idx)
            else:
                meta["reversal_time_minutes"] = np.nan

        # --- Close vs Gap ---
        if gap_direction == "up":
            if rth_close > rth_open:
                meta["close_vs_gap"] = "extended"
            elif rth_close >= prev_close:
                meta["close_vs_gap"] = "faded"
            else:
                meta["close_vs_gap"] = "reversed"
        else:
            if rth_close < rth_open:
                meta["close_vs_gap"] = "extended"
            elif rth_close <= prev_close:
                meta["close_vs_gap"] = "faded"
            else:
                meta["close_vs_gap"] = "reversed"
    else:
        for col in ["rth_open", "rth_close", "rth_high", "rth_low",
                     "opening_drive_atr", "opening_drive_direction",
                     "hod_time", "lod_time", "gap_filled", "gap_fill_time_minutes",
                     "close_vs_open_adr", "max_extension_adr", "max_adverse_adr",
                     "reversal_occurred", "reversal_time_minutes", "close_vs_gap"]:
            meta[col] = np.nan

    # ================================================================
    # 6. VWAP-METRIKEN
    # ================================================================
    if df_vwap is not None and len(df_vwap) > 0 and rth is not None and len(rth) >= 10:
        vwap_data = df_vwap.copy().reset_index(drop=True)
        rth_open_price = rth.iloc[0]["open"]

        # VWAP Cross
        valid_vwap = vwap_data.dropna(subset=["vwap"])
        if not valid_vwap.empty:
            first_valid = valid_vwap.iloc[0]
            initial_side = "above" if rth_open_price > first_valid["vwap"] else "below"

            cross_found = False
            for idx, row in valid_vwap.iterrows():
                current_side = "above" if row["close"] > row["vwap"] else "below"
                if current_side != initial_side:
                    meta["vwap_cross_time_minutes"] = int(idx)
                    cross_found = True
                    break
            if not cross_found:
                meta["vwap_cross_time_minutes"] = np.nan

            # VWAP Held
            last_valid = valid_vwap.iloc[-1]
            close_side = "above" if last_valid["close"] > last_valid["vwap"] else "below"
            meta["vwap_held"] = bool(initial_side == close_side)
        else:
            meta["vwap_cross_time_minutes"] = np.nan
            meta["vwap_held"] = np.nan

        # Close Location relative to VWAP
        last_row = vwap_data.iloc[-1]
        vwap_at_close = last_row.get("vwap", np.nan)
        std_at_close = last_row.get("std_dev", np.nan)
        if not np.isnan(vwap_at_close) and not np.isnan(std_at_close) and std_at_close > 0:
            rth_close_price = rth.iloc[-1]["close"]
            tolerance = VWAP_BAND_WIDTH * std_at_close
            if abs(rth_close_price - vwap_at_close) <= tolerance:
                meta["close_location"] = "at_vwap"
            elif rth_close_price > vwap_at_close:
                meta["close_location"] = "above_vwap"
            else:
                meta["close_location"] = "below_vwap"
        else:
            meta["close_location"] = np.nan

        # Z-Scores at specific times
        if "z_score" in vwap_data.columns and "time_et" in vwap_data.columns:
            z_10 = vwap_data[vwap_data["time_et"] == "10:00"]
            meta["vwap_z_at_10am"] = z_10.iloc[0]["z_score"] if not z_10.empty else np.nan

            z_1030 = vwap_data[vwap_data["time_et"] == "10:30"]
            meta["vwap_z_at_1030am"] = z_1030.iloc[0]["z_score"] if not z_1030.empty else np.nan

            # Z-Score at max extension
            if gap_direction == "up":
                ext_idx = rth["high"].idxmax()
            else:
                ext_idx = rth["low"].idxmin()
            if ext_idx < len(vwap_data):
                meta["vwap_z_at_max_extension"] = vwap_data.iloc[ext_idx]["z_score"]
            else:
                meta["vwap_z_at_max_extension"] = np.nan

            # Time in VWAP bands (|z_score| <= 1.0)
            valid_z = vwap_data["z_score"].dropna()
            if len(valid_z) > 0:
                bars_in_1std = (valid_z.abs() <= 1.0).sum()
                meta["time_in_vwap_bands"] = bars_in_1std / len(vwap_data)
            else:
                meta["time_in_vwap_bands"] = np.nan
        else:
            for col in ["vwap_z_at_10am", "vwap_z_at_1030am",
                         "vwap_z_at_max_extension", "time_in_vwap_bands"]:
                meta[col] = np.nan
    else:
        for col in ["vwap_cross_time_minutes", "vwap_held", "close_location",
                     "vwap_z_at_10am", "vwap_z_at_1030am",
                     "vwap_z_at_max_extension", "time_in_vwap_bands"]:
            meta[col] = np.nan

    # ================================================================
    # 7a. RVOL-AT-TIME (aus Baseline-Daten)
    # ================================================================
    for x_min, label in [(5, "5min"), (15, "15min"), (30, "30min"), (60, "60min")]:
        meta[f"rvol_at_time_{label}"] = np.nan

    try:
        baseline_dir = BASELINE_1MIN_DIR / ticker
        if baseline_dir.exists():
            baseline_files = sorted(baseline_dir.glob("*.parquet"))
            if baseline_files and rth is not None and len(rth) > 0:
                for x_min, label in [(5, "5min"), (15, "15min"), (30, "30min"), (60, "60min")]:
                    baseline_cum_vols = []
                    for bf in baseline_files:
                        try:
                            df_bl = load_parquet(bf)
                            if df_bl.empty or "time_et" not in df_bl.columns:
                                continue
                            # Kumulatives Volumen von 09:30 bis 09:30+X min
                            cutoff_hour = 9 + (30 + x_min) // 60
                            cutoff_minute = (30 + x_min) % 60
                            cutoff_time = f"{cutoff_hour:02d}:{cutoff_minute:02d}"
                            bl_window = df_bl[(df_bl["time_et"] >= "09:30") & (df_bl["time_et"] < cutoff_time)]
                            if not bl_window.empty:
                                baseline_cum_vols.append(bl_window["volume"].sum())
                        except Exception:
                            continue

                    if baseline_cum_vols:
                        baseline_avg = np.mean(baseline_cum_vols)
                        if baseline_avg > 0:
                            # Today's cumulative volume for same window
                            cutoff_hour = 9 + (30 + x_min) // 60
                            cutoff_minute = (30 + x_min) % 60
                            cutoff_time = f"{cutoff_hour:02d}:{cutoff_minute:02d}"
                            today_window = rth[(rth["time_et"] >= "09:30") & (rth["time_et"] < cutoff_time)]
                            if not today_window.empty:
                                today_cum_vol = today_window["volume"].sum()
                                meta[f"rvol_at_time_{label}"] = today_cum_vol / baseline_avg
    except Exception as e:
        logging.debug(f"  RVOL-at-time Fehler für {ticker}: {e}")

    # ================================================================
    # 7b. PREMARKET BAR FILL RATE
    # ================================================================
    meta["premarket_bar_fill_rate"] = np.nan
    if df_intraday is not None and "time_et" in df_intraday.columns:
        try:
            pm_window = df_intraday[
                (df_intraday["time_et"] >= "07:00") & (df_intraday["time_et"] <= "09:25")
            ]
            bars_with_volume = (pm_window["volume"] > 0).sum() if not pm_window.empty else 0
            meta["premarket_bar_fill_rate"] = bars_with_volume / 145.0
        except Exception as e:
            logging.debug(f"  Premarket fill rate Fehler für {ticker}: {e}")

    # ================================================================
    # 7c. GAP VS PRIOR RANGE
    # ================================================================
    meta["gap_vs_prior_range"] = np.nan
    if daily_before is not None and len(daily_before) >= 1:
        try:
            yesterday = daily_before.iloc[-1]
            yesterday_high = yesterday["high"]
            yesterday_low = yesterday["low"]
            if gap_direction == "up":
                if today_open > yesterday_high:
                    meta["gap_vs_prior_range"] = "above_range"
                else:
                    meta["gap_vs_prior_range"] = "within_range"
            else:
                if today_open < yesterday_low:
                    meta["gap_vs_prior_range"] = "below_range"
                else:
                    meta["gap_vs_prior_range"] = "within_range"
        except Exception as e:
            logging.debug(f"  Gap vs prior range Fehler für {ticker}: {e}")

    # ================================================================
    # 7d. ADR(10), GAP SIZE IN ADR, GAP DIRECTION
    # ================================================================
    meta["adr_10"] = np.nan
    meta["gap_size_in_adr"] = np.nan
    meta["gap_direction"] = gap_direction  # bereits vorhanden, sicherstellen

    if daily_before is not None and len(daily_before) >= 1:
        try:
            db = daily_before.copy()
            db["daily_range"] = db["high"] - db["low"]
            last_10 = db["daily_range"].iloc[-10:] if len(db) >= 10 else db["daily_range"]
            adr_10_val = last_10.mean()
            meta["adr_10"] = adr_10_val

            if adr_10_val > 0 and not np.isnan(prev_close) and not np.isnan(today_open):
                meta["gap_size_in_adr"] = abs(today_open - prev_close) / adr_10_val
        except Exception as e:
            logging.debug(f"  ADR/Gap size Fehler für {ticker}: {e}")

    # ================================================================
    # 7. VOLUME PROFILE METRIKEN
    # ================================================================
    if df_vp is not None and len(df_vp) > 0:
        vp = df_vp.sort_values("minutes_since_open").reset_index(drop=True)

        # VPOC at close (letzter Snapshot)
        meta["vpoc_at_close"] = vp.iloc[-1]["vpoc"]

        # VPOC Migration
        vpoc_changes = 0
        for i in range(1, len(vp)):
            if vp.iloc[i]["vpoc"] != vp.iloc[i - 1]["vpoc"]:
                vpoc_changes += 1
        meta["vpoc_migration"] = vpoc_changes

        # Close vs VPOC
        if rth is not None and len(rth) > 0:
            rth_close_val = rth.iloc[-1]["close"]
            meta["close_vs_vpoc"] = rth_close_val - vp.iloc[-1]["vpoc"]

            # Close in Value Area
            val_close = vp.iloc[-1]["val"]
            vah_close = vp.iloc[-1]["vah"]
            meta["close_in_value_area"] = bool(val_close <= rth_close_val <= vah_close)
        else:
            meta["close_vs_vpoc"] = np.nan
            meta["close_in_value_area"] = np.nan
    else:
        for col in ["vpoc_at_close", "vpoc_migration", "close_vs_vpoc", "close_in_value_area"]:
            meta[col] = np.nan

    return meta


def process_date(date_str: str, force: bool = False):
    """Berechnet Metadaten für alle Gapper eines Tages."""
    gapper_file = METADATA_DIR / f"gappers_{date_str}.csv"
    if not gapper_file.exists():
        logging.warning(f"Keine Gapper-Datei für {date_str}")
        return

    df_gappers = pd.read_csv(gapper_file)
    if df_gappers.empty:
        logging.info(f"Keine Gapper für {date_str}")
        return

    ticker_cache = _load_ticker_cache()
    spy_return = _get_spy_return(date_str)
    vix_level = _get_vix_level(date_str)

    logging.info(f"{date_str}: Berechne Metadaten für {len(df_gappers)} Gapper "
                 f"(SPY={spy_return:.2f}%, VIX={vix_level:.1f})" if not np.isnan(spy_return) and not np.isnan(vix_level)
                 else f"{date_str}: Berechne Metadaten für {len(df_gappers)} Gapper")

    all_meta = []
    success = 0
    errors = 0

    for _, row in tqdm(df_gappers.iterrows(), total=len(df_gappers),
                       desc=f"  {date_str}", leave=False):
        ticker = row["ticker"]
        try:
            meta = compute_metadata_for_ticker(ticker, date_str, row, ticker_cache, spy_return, vix_level)
            all_meta.append(meta)
            success += 1
        except Exception as e:
            logging.error(f"  Metadata Fehler bei {ticker}: {e}", exc_info=True)
            errors += 1

    if all_meta:
        df_new = pd.DataFrame(all_meta)

        # Master-Datei laden und aktualisieren
        master_path = METADATA_DIR / "metadata_master.parquet"
        if master_path.exists() and not force:
            df_master = load_parquet(master_path)
            # Entferne existierende Einträge für diesen Tag
            df_master = df_master[~(
                (df_master["date"] == date_str) &
                (df_master["ticker"].isin(df_new["ticker"]))
            )]
            df_combined = pd.concat([df_master, df_new], ignore_index=True)
        else:
            df_combined = df_new

        # Sortieren
        df_combined = df_combined.sort_values(["date", "ticker"]).reset_index(drop=True)
        save_parquet(df_combined, master_path)
        logging.info(f"  {date_str}: {success} erfolgreich, {errors} Fehler — "
                     f"Master hat jetzt {len(df_combined)} Einträge")
    else:
        logging.warning(f"  {date_str}: Keine Metadaten berechnet")


def main():
    dates, args = parse_date_args()
    setup_logging("07_compute_metadata",
                  dates[0] if len(dates) == 1 else f"{dates[0]}_to_{dates[-1]}")

    logging.info(f"=== Metadata Berechnung ===")
    logging.info(f"Verarbeite {len(dates)} Handelstag(e)")

    for date_str in tqdm(dates, desc="Computing metadata"):
        try:
            process_date(date_str, force=args.force)
        except Exception as e:
            logging.error(f"Fehler bei {date_str}: {e}", exc_info=True)

    logging.info("=== Metadata Berechnung abgeschlossen ===")


if __name__ == "__main__":
    main()
