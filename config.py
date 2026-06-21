import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# ============================================================
# API
# ============================================================
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

# ============================================================
# Pfade
# ============================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_1MIN_DIR = DATA_DIR / "raw_1min"
DAILY_DIR = DATA_DIR / "daily"
NORMALIZED_DIR = DATA_DIR / "normalized"
VWAP_DIR = DATA_DIR / "vwap"
VOLUME_PROFILE_DIR = DATA_DIR / "volume_profile"
BASELINE_1MIN_DIR = DATA_DIR / "baseline_1min"
METADATA_DIR = DATA_DIR / "metadata"
WATCHLIST_DIR = BASE_DIR / "watchlist"

# ============================================================
# Gapper-Kriterien (Schicht 1 — objektiv)
# Scannt sowohl UP-Gaps als auch DOWN-Gaps!
# ============================================================
MIN_GAP_PCT = 3.0               # Mindest-Gap in Prozent (Absolutwert, gilt für Up UND Down)
MIN_PREMARKET_VOLUME = 300_000   # Mindest-PreMarket-Volumen (Shares)
MIN_MARKET_CAP = 2_000_000_000   # $2B Minimum Market Cap
MIN_PRICE = 10.0                  # Mindestpreis

# Exchange & Typ Filter
# Polygon nutzt MIC Exchange Codes: XNYS = NYSE, XNAS = NASDAQ
# Ticker Types: CS = Common Stock
# Ausgeschlossen: ETFs, ADRs, Warrants, Rights, Units, OTC, Pink Sheets
ALLOWED_EXCHANGES = ["XNYS", "XNAS"]   # NYSE und NASDAQ only
ALLOWED_TYPES = ["CS"]           # Common Stock und Preferred Stock

# ============================================================
# Rohdaten (von Polygon)
# ============================================================
RAW_BAR_INTERVAL = 1             # 1-Minute Bars als Rohdaten
PREMARKET_START = "04:00"        # PreMarket Start (ET)
RTH_START = "09:30"              # Regular Trading Hours Start (ET)
RTH_END = "16:00"                # Regular Trading Hours End (ET)
AFTERHOURS_END = "20:00"         # After Hours End (ET)
DAILY_CONTEXT_LOOKBACK = 90      # Tage Daily-Bars Kontext vor dem Gap-Tag

# ============================================================
# Normalisierte Profile (für Overlay & Statistik)
# Rohdaten = 1-Min, Profile = 1-Min, optional aggregierbar
# ============================================================
NORMALIZED_BAR_INTERVAL = 1      # Volle 1-Min Auflösung für normalisierte Profile
BARS_PER_DAY = 390               # 390 RTH-Minuten (9:30-16:00) = 390 Bars
OPTIONAL_AGGREGATION = [5, 15]   # Kann bei Bedarf auf 5-Min oder 15-Min aggregiert werden

# ============================================================
# Normalisierung — Y-Achse
# ============================================================
ADR_LOOKBACK = 5                 # Tage für Average Daily Range (primäre Normalisierung)
ATR_LOOKBACK = 14                # Tage für ATR (sekundär / Backup)

# ============================================================
# VWAP Berechnung
# ============================================================
VWAP_SHIFT = 1                   # Bars Verschiebung nach rechts (vermeidet Look-Ahead-Bias)
VWAP_STD_DEVS = [1, 2, 3]       # Standardabweichungs-Level (±1σ, ±2σ, ±3σ)
VWAP_BAND_WIDTH = 0.25           # ±0.25σ Band um jede der 7 Linien (VWAP, ±1σ, ±2σ, ±3σ)
# Ergebnis: 7 Linien × (Linie + oberes Band + unteres Band) = 21 Werte pro Zeitpunkt
# Linien: VWAP, +1σ, -1σ, +2σ, -2σ, +3σ, -3σ
# Jede Linie hat ein Band von ±0.25σ drumherum

# ============================================================
# Volume Profile
# ============================================================
VP_TICK_SIZE = 0.01              # Preis-Bucket-Größe für Volume Profile (wird dynamisch angepasst)
VP_VALUE_AREA_PCT = 70           # Value Area = 70% des Gesamtvolumens
VP_SNAPSHOT_INTERVAL = 1         # Alle 5 Minuten ein Snapshot des developing 1
# Berechnet: VPOC (höchstes Volumen-Bucket), VAH, VAL — sich entwickelnd über den Tag

# ============================================================
# Technische Indikatoren (für Kontext-Metadaten)
# ============================================================
RSI_PERIOD = 14
SMA_PERIOD = 20

# ============================================================
# Analyse-Zeitfenster (ET)
# ============================================================
OPENING_DRIVE_MINUTES = 15       # Erste 15 Min nach Open
MORNING_SESSION_END = "10:30"
LUNCH_START = "12:00"
LUNCH_END = "14:00"
POWER_HOUR_START = "15:00"

# ============================================================
# Kontext-Metadaten (berechnet für jeden Gapper-Tag)
# ============================================================
# Pre-Gap Kontext (aus Daily-Daten der letzten 90 Tage):
#   - gap_pct: (Open - PrevClose) / PrevClose × 100 (positiv ODER negativ)
#   - gap_direction: "up" oder "down"
#   - prior_return_5d: Kum. Return 5 Tage vor Gap
#   - prior_return_10d: Kum. Return 10 Tage vor Gap
#   - prior_return_20d: Kum. Return 20 Tage vor Gap
#   - dist_from_20sma: (PrevClose - SMA20) / SMA20 × 100
#   - dist_from_52w_high: Abstand vom 52-Wochen-Hoch (braucht ~63 Tage der 90)
#   - rsi_14_prev: RSI(14) am Vortag
#   - adr_5: Average Daily Range (5 Tage)
#   - adr_20: Average Daily Range (20 Tage)
#   - avg_daily_volume_20: Durchschn. Tagesvolumen (20 Tage)
#   - sector: GICS Sektor
#   - market_cap_bucket: "2-10B", "10-50B", "50-200B", "200B+"
#   - catalyst_type: "earnings", "guidance", "fda", "upgrade", "downgrade", "macro", "unknown"
#   - spy_return_day: S&P 500 Return am Gap-Tag (Marktkontext)
#   - vix_level: VIX Schlusskurs am Vortag (Volatilitäts-Regime)
#   - day_of_week: Mo-Fr
#
# Intraday-Volumendaten:
#   - rvol_open_30min: Volumen erste 30 Min / 20-Tage-Avg
#   - rvol_full_day: Tagesvolumen / 20-Tage-Avg
#
# Intraday-Ergebnis (nach Marktschluss berechenbar):
#   - opening_drive_atr: Max Bewegung erste 15 Min / ADR(5)
#   - opening_drive_direction: "with_gap" oder "against_gap"
#   - vwap_cross_time_minutes: Minuten nach Open bis erster VWAP-Kreuz (oder NaN)
#   - vwap_held: Boolean — Schlusskurs auf gleicher Seite wie Open relativ zu VWAP?
#   - hod_time: Zeitpunkt Tageshoch (Minuten nach Open)
#   - lod_time: Zeitpunkt Tagestief (Minuten nach Open)
#   - gap_filled: Boolean — wurde der Gap intraday komplett gefüllt?
#   - gap_fill_time_minutes: Minuten bis Gap-Fill (oder NaN)
#   - close_vs_open_adr: (Close - Open) / ADR(5)
#   - max_extension_adr: Max Distanz vom Open / ADR(5)
#   - max_adverse_adr: Max Gegenrichtung vom Open / ADR(5)
#   - reversal_occurred: Boolean — Preis mind. 1 ADR vom Intraday-Extremum zurück?
#   - reversal_time_minutes: Minuten nach Open bis Reversal (oder NaN)
#   - close_location: "above_vwap", "below_vwap", "at_vwap" (±0.25σ)
#   - close_vs_gap: "extended" (weiter in Gap-Richtung), "faded" (zurück), "reversed" (über Gap hinaus)
#
# VWAP-basierte Metriken (pro Zeitpunkt, für Conditional Probabilities):
#   - vwap_z_at_10am: VWAP Z-Score um 10:00 ET
#   - vwap_z_at_1030am: VWAP Z-Score um 10:30 ET
#   - vwap_z_at_max_extension: VWAP Z-Score beim Tages-Extremum
#   - time_in_vwap_bands: Anteil RTH-Zeit die Preis innerhalb ±1σ verbringt
#
# Volume Profile Metriken:
#   - vpoc_at_close: VPOC am Ende des Tages
#   - vpoc_migration: Wie oft hat sich die VPOC im Laufe des Tages verschoben?
#   - close_vs_vpoc: Schlusskurs relativ zum VPOC
#   - close_in_value_area: Boolean — Schlusskurs innerhalb VAH/VAL?

# ============================================================
# Statistik
# ============================================================
MIN_SAMPLE_SIZE = 30             # Minimum N für statistische Aussagen
CONFIDENCE_LEVEL = 0.95          # 95% Konfidenzintervall

# ============================================================
# Clustering
# ============================================================
DEFAULT_N_CLUSTERS = 6           # Startpunkt für K-Means
