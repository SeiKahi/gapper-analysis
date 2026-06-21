# run_daily_pipeline.py — Master-Script: führt alle Pipeline-Schritte nacheinander aus

import sys
import subprocess
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import BASE_DIR
from scripts.utils import get_previous_trading_day, setup_logging

SCRIPTS_DIR = BASE_DIR / "scripts"


def run_script(script_name: str, date_args: list, python_exe: str):
    """Führe ein Script aus und prüfe Exit-Code."""
    cmd = [python_exe, str(SCRIPTS_DIR / script_name)] + date_args
    print(f"\n{'='*60}")
    print(f"  Running: {script_name}")
    print(f"{'='*60}")
    logging.info(f"Starte: {script_name} {' '.join(date_args)}")

    result = subprocess.run(cmd, cwd=str(BASE_DIR))

    if result.returncode != 0:
        logging.error(f"{script_name} fehlgeschlagen (Exit Code {result.returncode})")
        print(f"\n  {script_name} failed with exit code {result.returncode}")
        try:
            response = input("  Continue with next script? (y/n): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            response = "n"
        if response != "y":
            print("  Pipeline abgebrochen.")
            sys.exit(1)
    else:
        logging.info(f"{script_name} erfolgreich abgeschlossen")
        print(f"  {script_name} completed successfully")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Gapper Analysis Pipeline")
    parser.add_argument("--date", help="Single date YYYY-MM-DD")
    parser.add_argument("--from", dest="from_date", help="Start date for backfill")
    parser.add_argument("--to", dest="to_date", help="End date for backfill")
    parser.add_argument("--skip-download", action="store_true",
                        help="Überspringe Download-Schritte (wenn Daten bereits vorhanden)")
    parser.add_argument("--force", action="store_true",
                        help="Überschreibe existierende Dateien")
    args = parser.parse_args()

    # Date-Argumente bestimmen
    if args.date:
        date_args = ["--date", args.date]
        display_date = args.date
    elif args.from_date and args.to_date:
        date_args = ["--from", args.from_date, "--to", args.to_date]
        display_date = f"{args.from_date} bis {args.to_date}"
    else:
        yesterday = get_previous_trading_day(datetime.now().strftime("%Y-%m-%d"))
        date_args = ["--date", yesterday]
        display_date = yesterday
        print(f"Kein Datum angegeben, nutze letzten Handelstag: {yesterday}")

    if args.force:
        date_args.append("--force")

    setup_logging("run_daily_pipeline", display_date.replace(" ", "_"))

    python_exe = sys.executable

    print(f"\n{'='*60}")
    print(f"  GAPPER ANALYSIS PIPELINE")
    print(f"  Datum: {display_date}")
    print(f"  Python: {python_exe}")
    print(f"  Skip Download: {args.skip_download}")
    print(f"  Force: {args.force}")
    print(f"{'='*60}")

    # Pipeline Schritte
    if not args.skip_download:
        run_script("01_scan_gappers.py", date_args, python_exe)
        run_script("02_download_intraday.py", date_args, python_exe)
        run_script("02b_download_baseline.py", date_args, python_exe)
        run_script("03_download_daily.py", date_args, python_exe)

    run_script("04_compute_vwap.py", date_args, python_exe)
    run_script("05_compute_volume_profile.py", date_args, python_exe)
    run_script("06_normalize.py", date_args, python_exe)
    run_script("07_compute_metadata.py", date_args, python_exe)

    print(f"\n{'='*60}")
    print(f"  Pipeline complete!")
    print(f"{'='*60}")
    logging.info("Pipeline abgeschlossen")


if __name__ == "__main__":
    main()
