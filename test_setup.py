"""Test-Script: Prueft ob alle Pakete installiert sind und die Projektstruktur stimmt."""
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
passed = 0
failed = 0

def check(name, ok, detail=""):
    global passed, failed
    if ok:
        passed += 1
        print(f"  [OK]   {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name}  {detail}")

print("=" * 60)
print("1. PAKET-IMPORTS")
print("=" * 60)

packages = {
    "polygon": "polygon",
    "pyarrow": "pyarrow",
    "duckdb": "duckdb",
    "pandas": "pandas",
    "numpy": "numpy",
    "scipy": "scipy",
    "polars": "polars",
    "pandas_ta": "pandas_ta",
    "sklearn": "sklearn",
    "tslearn": "tslearn",
    "statsmodels": "statsmodels",
    "matplotlib": "matplotlib",
    "plotly": "plotly",
    "seaborn": "seaborn",
    "streamlit": "streamlit",
    "dotenv": "dotenv",
    "tqdm": "tqdm",
    "requests": "requests",
    "pytz": "pytz",
    "jupyter": "jupyter",
}

for display_name, import_name in packages.items():
    try:
        __import__(import_name)
        check(display_name, True)
    except ImportError as e:
        check(display_name, False, str(e))

print()
print("=" * 60)
print("2. ORDNERSTRUKTUR")
print("=" * 60)

dirs = [
    "data/raw_1min",
    "data/daily",
    "data/normalized",
    "data/vwap",
    "data/volume_profile",
    "data/metadata",
    "scripts",
    "notebooks",
    "watchlist",
    "dashboard",
]

for d in dirs:
    path = os.path.join(BASE_DIR, d)
    check(d, os.path.isdir(path), f"Ordner nicht gefunden: {path}")

print()
print("=" * 60)
print("3. DATEIEN")
print("=" * 60)

files = [
    "config.py",
    ".env",
    ".gitignore",
    "requirements.txt",
    "README.md",
    "watchlist/template.csv",
    "notebooks/exploration.ipynb",
    "dashboard/app.py",
    "scripts/01_scan_gappers.py",
    "scripts/02_download_intraday.py",
    "scripts/03_download_daily.py",
    "scripts/04_compute_vwap.py",
    "scripts/05_compute_volume_profile.py",
    "scripts/06_normalize.py",
    "scripts/07_compute_metadata.py",
    "scripts/08_analyze.py",
    "scripts/09_cluster.py",
    "scripts/run_daily_pipeline.py",
    "scripts/utils.py",
]

for f in files:
    path = os.path.join(BASE_DIR, f)
    check(f, os.path.isfile(path), f"Datei nicht gefunden: {path}")

print()
print("=" * 60)
print("4. CONFIG IMPORT")
print("=" * 60)

sys.path.insert(0, BASE_DIR)
try:
    import config
    check("config.py importierbar", True)
    check("POLYGON_API_KEY definiert", hasattr(config, "POLYGON_API_KEY"))
    check("BASE_DIR korrekt", str(config.BASE_DIR) == BASE_DIR)
    check("MIN_GAP_PCT = 3.0", config.MIN_GAP_PCT == 3.0)
    check("MIN_PREMARKET_VOLUME = 300000", config.MIN_PREMARKET_VOLUME == 300_000)
    check("MIN_PRICE = 10.0", config.MIN_PRICE == 10.0)
    check("BARS_PER_DAY = 390", config.BARS_PER_DAY == 390)
    check("NORMALIZED_BAR_INTERVAL = 1", config.NORMALIZED_BAR_INTERVAL == 1)
except Exception as e:
    check("config.py importierbar", False, str(e))

print()
print("=" * 60)
print(f"ERGEBNIS: {passed} OK, {failed} FEHLGESCHLAGEN")
print("=" * 60)

if failed > 0:
    sys.exit(1)
else:
    print("\nAlle Tests bestanden!")
