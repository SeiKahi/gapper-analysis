# test_pipeline.py — Validiert die komplette Pipeline-Infrastruktur

import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

passed = 0
failed = 0
warnings = 0


def check(label: str, condition: bool, warn_only: bool = False):
    global passed, failed, warnings
    if condition:
        print(f"  [OK] {label}")
        passed += 1
    elif warn_only:
        print(f"  [WARN] {label}")
        warnings += 1
    else:
        print(f"  [FAIL] {label}")
        failed += 1


def main():
    global passed, failed, warnings

    print("=" * 60)
    print("  GAPPER ANALYSIS — PIPELINE VALIDATION")
    print("=" * 60)

    # ================================================================
    # 1. Config Imports
    # ================================================================
    print("\n--- 1. Config ---")
    try:
        from config import (
            POLYGON_API_KEY, BASE_DIR as BD, DATA_DIR,
            RAW_1MIN_DIR, DAILY_DIR, NORMALIZED_DIR, VWAP_DIR,
            VOLUME_PROFILE_DIR, METADATA_DIR, WATCHLIST_DIR,
            MIN_GAP_PCT, MIN_PREMARKET_VOLUME, MIN_MARKET_CAP, MIN_PRICE,
            ALLOWED_EXCHANGES, ALLOWED_TYPES,
            BARS_PER_DAY, ADR_LOOKBACK, RSI_PERIOD, SMA_PERIOD,
            VWAP_SHIFT, VWAP_STD_DEVS, VWAP_BAND_WIDTH,
            VP_VALUE_AREA_PCT, OPENING_DRIVE_MINUTES
        )
        check("config.py importiert", True)
        check(f"POLYGON_API_KEY gesetzt ({len(POLYGON_API_KEY)} Zeichen)", bool(POLYGON_API_KEY))
        check(f"MIN_GAP_PCT = {MIN_GAP_PCT}", MIN_GAP_PCT == 3.0)
        check(f"MIN_MARKET_CAP = {MIN_MARKET_CAP:,.0f}", MIN_MARKET_CAP == 2_000_000_000)
        check(f"BARS_PER_DAY = {BARS_PER_DAY}", BARS_PER_DAY == 390)
        check(f"ALLOWED_EXCHANGES = {ALLOWED_EXCHANGES}", "XNYS" in ALLOWED_EXCHANGES)
    except Exception as e:
        check(f"config.py Import FEHLER: {e}", False)

    # ================================================================
    # 2. Utils Imports
    # ================================================================
    print("\n--- 2. Utils ---")
    try:
        from scripts.utils import (
            get_polygon_client, timestamp_to_et, et_to_timestamp,
            get_trading_days, get_previous_trading_day,
            save_parquet, load_parquet, ensure_dir,
            retry_api_call, sic_to_sector,
            setup_logging, parse_date_args, EASTERN
        )
        check("utils.py importiert", True)

        # Test sic_to_sector
        check("sic_to_sector('3571') = Manufacturing", sic_to_sector("3571") == "Manufacturing")
        check("sic_to_sector('6020') = Finance", sic_to_sector("6020") == "Finance")
        check("sic_to_sector('') = Unknown", sic_to_sector("") == "Unknown")

        # Test timestamp conversion roundtrip
        from datetime import datetime
        import pytz
        et = pytz.timezone("US/Eastern")
        dt = et.localize(datetime(2026, 2, 6, 9, 30, 0))
        ts = et_to_timestamp(dt)
        dt_back = timestamp_to_et(ts)
        check(f"Timestamp roundtrip: {dt.strftime('%H:%M')} -> {dt_back.strftime('%H:%M')}",
              dt_back.strftime("%H:%M") == "09:30")

    except Exception as e:
        check(f"utils.py Import FEHLER: {e}", False)

    # ================================================================
    # 3. Script Imports
    # ================================================================
    print("\n--- 3. Script Imports ---")
    scripts = [
        ("scripts.01_scan_gappers", "01_scan_gappers.py"),
        ("scripts.02_download_intraday", "02_download_intraday.py"),
        ("scripts.02b_download_baseline", "02b_download_baseline.py"),
        ("scripts.03_download_daily", "03_download_daily.py"),
        ("scripts.04_compute_vwap", "04_compute_vwap.py"),
        ("scripts.05_compute_volume_profile", "05_compute_volume_profile.py"),
        ("scripts.06_normalize", "06_normalize.py"),
        ("scripts.07_compute_metadata", "07_compute_metadata.py"),
        ("scripts.run_daily_pipeline", "run_daily_pipeline.py"),
    ]
    # We import each module to validate syntax
    import importlib
    for module_name, file_name in scripts:
        try:
            # Reset argv to avoid argparse conflicts
            old_argv = sys.argv
            sys.argv = [file_name]
            mod = importlib.import_module(module_name.replace("-", "_"))
            check(f"{file_name} importiert", True)
            # Check for main function
            has_main = hasattr(mod, "main")
            check(f"  {file_name} hat main()", has_main)
            sys.argv = old_argv
        except Exception as e:
            sys.argv = old_argv
            check(f"{file_name} Import FEHLER: {e}", False)

    # ================================================================
    # 4. Verzeichnisstruktur
    # ================================================================
    print("\n--- 4. Verzeichnisse ---")
    from config import (RAW_1MIN_DIR, DAILY_DIR, NORMALIZED_DIR, VWAP_DIR,
                        VOLUME_PROFILE_DIR, METADATA_DIR, BASELINE_1MIN_DIR)
    dirs = {
        "data/": DATA_DIR,
        "data/raw_1min/": RAW_1MIN_DIR,
        "data/daily/": DAILY_DIR,
        "data/baseline_1min/": BASELINE_1MIN_DIR,
        "data/normalized/": NORMALIZED_DIR,
        "data/vwap/": VWAP_DIR,
        "data/volume_profile/": VOLUME_PROFILE_DIR,
        "data/metadata/": METADATA_DIR,
    }
    for name, path in dirs.items():
        exists = path.exists()
        if not exists:
            path.mkdir(parents=True, exist_ok=True)
            check(f"{name} erstellt", True)
        else:
            check(f"{name} existiert", True)

    # ================================================================
    # 5. Paket-Abhängigkeiten
    # ================================================================
    print("\n--- 5. Pakete ---")
    packages = [
        ("polygon", "polygon-api-client"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("pyarrow", "pyarrow"),
        ("pytz", "pytz"),
        ("tqdm", "tqdm"),
        ("dotenv", "python-dotenv"),
        ("exchange_calendars", "exchange_calendars"),
        ("pandas_ta", "pandas-ta"),
        ("scipy", "scipy"),
        ("sklearn", "scikit-learn"),
    ]
    for import_name, pip_name in packages:
        try:
            mod = importlib.import_module(import_name)
            version = getattr(mod, "__version__", "?")
            check(f"{pip_name} ({version})", True)
        except ImportError:
            check(f"{pip_name} NICHT installiert", False)

    # ================================================================
    # 6. Trading Days Test
    # ================================================================
    print("\n--- 6. Trading Days ---")
    try:
        days = get_trading_days("2026-01-02", "2026-01-10")
        check(f"Handelstage 2.-10. Jan 2026: {len(days)} Tage", len(days) > 0)
        prev = get_previous_trading_day("2026-02-08")
        check(f"Vortag von 2026-02-08: {prev}", bool(prev))
    except Exception as e:
        check(f"Trading Days FEHLER: {e}", False)

    # ================================================================
    # 7. Polygon Client Test
    # ================================================================
    print("\n--- 7. Polygon Client ---")
    try:
        client = get_polygon_client()
        check("Polygon Client erstellt", client is not None)
    except Exception as e:
        check(f"Polygon Client FEHLER: {e}", False)

    # ================================================================
    # 8. Existierende Daten
    # ================================================================
    print("\n--- 8. Existierende Daten ---")
    gapper_files = list(METADATA_DIR.glob("gappers_*.csv"))
    check(f"Gapper-CSVs gefunden: {len(gapper_files)}", True, warn_only=True)

    raw_tickers = [d.name for d in RAW_1MIN_DIR.iterdir() if d.is_dir()] if RAW_1MIN_DIR.exists() else []
    check(f"Raw 1-Min Ticker-Ordner: {len(raw_tickers)}", True, warn_only=True)

    daily_tickers = [d.name for d in DAILY_DIR.iterdir() if d.is_dir()] if DAILY_DIR.exists() else []
    check(f"Daily Ticker-Ordner: {len(daily_tickers)}", True, warn_only=True)

    master_path = METADATA_DIR / "metadata_master.parquet"
    if master_path.exists():
        import pandas as pd
        df = pd.read_parquet(master_path)
        check(f"metadata_master.parquet: {len(df)} Einträge, {len(df.columns)} Spalten", True)
        print(f"         Spalten: {', '.join(df.columns[:10])}...")
    else:
        check("metadata_master.parquet existiert noch nicht", True, warn_only=True)

    cache_path = METADATA_DIR / "ticker_cache.json"
    if cache_path.exists():
        import json
        with open(cache_path) as f:
            cache = json.load(f)
        check(f"ticker_cache.json: {len(cache)} Ticker", True)
    else:
        check("ticker_cache.json existiert noch nicht", True, warn_only=True)

    # ================================================================
    # Zusammenfassung
    # ================================================================
    print(f"\n{'='*60}")
    total = passed + failed + warnings
    print(f"  ERGEBNIS: {passed}/{total} bestanden, {failed} fehlgeschlagen, {warnings} Warnungen")
    if failed == 0:
        print(f"  Pipeline ist bereit!")
    else:
        print(f"  {failed} Fehler muessen behoben werden.")
    print(f"{'='*60}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
