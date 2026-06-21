# Intraday Gapper Analysis

I built this to answer one question: do intraday stock gaps contain a tradeable, statistically real edge, or do the patterns fall apart once you test them honestly? The system scans, stores and analyses gap events in US large-cap stocks (gaps of 3% or more, up and down). The dataset covers about 10,250 events across roughly 1,000 tickers and five years, at one-minute and one-second resolution, with 98 features per event.

The point of the repo is the method, not a trading signal. It is built to attack its own findings before trusting them.

## What it shows

- Experimental discipline. Every backtest uses a fixed in-sample / out-of-sample split, and the out-of-sample years (2024 to 2026) are never touched during parameter search. Hypotheses get written down first, then tested once.
- Bias hunting. The repo has explicit tests for look-ahead and data leakage, plus a Bonferroni correction and year-by-year stability checks.
- Data engineering at some scale. A reproducible pipeline pulls millions of one-minute bars from the Polygon.io API into Parquet, with a one-second subset for closer work.
- Feature work. 98 features per event, covering VWAP bands, volume profile, relative volume, opening-drive microstructure and drift, all normalised to daily range.
- Some ML and stats. Logistic regression with feature importance, clustering, a "big runner" classifier, and a reinforcement-learning data pipeline.
- Honesty about negative results. Dead ends stay in. Two whole runs (d14 and d15) exist only to re-test earlier findings for bias.

## What I found (the honest version)

The project went through several arcs and I kept all of them, including the ones that failed.

Arc 1 used ML on VWAP-cross entries. It found no real edge. An in-sample signal of +0.029 ADR dropped 85% out of sample and stopped being significant. I wrote it up as a null result instead of burying it (see docs/FINAL_REPORT.md).

The re-audit caught my own mistakes. One setup looked great at +0.66 R and an 89% win rate. It turned out to be 97% look-ahead bias, because it used end-of-day and low-of-day information you would not have at entry. Another "edge" was a backtest bug where stops filled at the stop price instead of the gapping open. Both got removed.

After that, I put every remaining candidate through five gates: bias-free recomputation, a realistic fill model, a positive out-of-sample result on 2024 to 2026, a bootstrap confidence interval with its lower bound above zero, and survival under pessimistic costs (0.10 ADR slippage plus fees). A few setups cleared all five. The most reliable, by sample size and stability:

| Setup | EV per trade | N | Note |
|---|---|---|---|
| Dip and Rip | +0.53 R | 625 | largest sample, the only uncorrelated one |
| Gap 3.0+ ADR | +0.57 R | 258 | most stable across years |
| Pre-10am momentum | +1.02 R | 110 | highest EV after costs |

The catch, which I will not hide: these survivors tend to do better out of sample than in sample. That points to regime dependence (the out-of-sample window is mostly the 2024 to 2025 bull market) or to bias I have not fully removed. A bootstrap interval above zero means the result is unlikely to be noise inside that window, not that it holds in every regime. Walk-forward testing is still on the to-do list, and I have not traded these live. So I treat them as candidate edges that cleared a hard backtest, not as proven money. The point on display is the filtering, not a promise of returns.

## Stack

Python, pandas, NumPy, scikit-learn, Streamlit, Parquet, the Polygon.io API, and a multi-agent LLM workflow for the research runs.

## How the repo is laid out

```
scripts/             data pipeline (01-07) and research runs (d9 to d16)
d13_team_analysis/   multi-agent study: quant, trader, and a devil's-advocate reviewer
d14_bias_free/       re-runs that re-test earlier findings for bias
d15_bias_free/       cost-stressed re-evaluation
d16_swing/           swing-trading engine and strategies
docs/                data dictionary, findings log, project brief
dashboard/           Streamlit dashboard
config.py            config; the API key lives in a local .env (not committed)
```

The raw data, trained models and large result dumps are not in the repo. This is the code and the write-up, not the dataset.

## Data

You need a Polygon.io key (their Starter plan covers one-minute and pre/after-hours data). Put it in a local .env as POLYGON_API_KEY and run the pipeline.

## About

I am Seihan Kahirov. I have an M.Sc. in renewable energy engineering and I am moving into data and quantitative work. This was self-directed. The full methodology, data dictionary and per-run findings are in the docs folder and further down this page, in German.

---

# Detailed Technical Documentation (German)

## System-Ueberblick

Automatisches Scannen, Archivieren und statistisches Auswerten von Gapper-Aktien (Up UND Down Gaps >= 3%) mit normalisierter Overlay-Analyse, VWAP-Baendern und Volume Profile.

## Universum

- **Boersen:** NYSE (XNYS) und NASDAQ (XNAS)
- **Typ:** Common Stock (CS)
- **Market Cap:** > $2B (US Large Caps)
- **Mindestpreis:** $10
- **Ausgeschlossen:** ETFs, ADRs, Warrants, Rights, Units, OTC, Pink Sheets

## Setup

### 1. Virtual Environment aktivieren

```bash
# Windows
gapper_env\Scripts\activate

# macOS/Linux
source gapper_env/bin/activate
```

### 2. Pakete installieren

```bash
pip install -r requirements.txt
```

### 3. API Key eintragen

Bearbeite `.env` und trage deinen Polygon.io API Key ein:

```
POLYGON_API_KEY=dein_echter_api_key
```

> Benoetigt: Polygon.io Starter Plan ($29/Monat) fuer 1-Minuten-Daten und Premarket/Afterhours.

## Taeglicher Workflow

### Morgens (vor Markteroeffnung, ~8:00 ET)

1. Gapper-Scanner laufen lassen: `python scripts/01_scan_gappers.py`
2. Watchlist-Eintraege in `watchlist/YYYY-MM-DD.csv` eintragen (Schicht 2: subjektive Bewertung)

### Abends (nach Marktschluss, ~17:00 ET)

1. Pipeline ausfuehren: `python scripts/run_daily_pipeline.py`
   - Laedt 1-Min Intraday-Daten herunter (04:00-20:00 ET)
   - Laedt 90 Tage Daily-Bars herunter
   - Berechnet VWAP + StdDev-Baender
   - Berechnet Developing Volume Profile
   - Normalisiert auf 1-Min-Profile (optional aggregierbar auf 5-Min/15-Min)
   - Berechnet Kontext-Metadaten
   - Fuehrt statistische Auswertung durch
   - Fuehrt Clustering durch

### Dashboard

```bash
streamlit run dashboard/app.py
```

## Datenfluss

```
Polygon.io API
    |
    v
1-Min Rohdaten (data/raw_1min/) + 90 Tage Daily (data/daily/)
    |
    v
VWAP/StdDev-Baender (data/vwap/) + Volume Profile (data/volume_profile/)
    |
    v
1-Min Normalisierte Profile (data/normalized/)
    |
    v
Kontext-Metadaten (data/metadata/)
    |
    v
Statistische Analyse + Clustering -> Dashboard
```

## Projektstruktur

- `scripts/` -- Pipeline-Skripte (01-07) + Analyse-Durchlaeufe (d9-d12) + Hilfsfunktionen
- `data/` -- Alle Daten (nicht in Git)
- `results/` -- Ergebnisse aller Durchlaeufe (.txt Reports + .parquet Rohdaten)
- `notebooks/` -- Jupyter Notebooks fuer Exploration
- `watchlist/` -- Taegliche Watchlists (Schicht 2)
- `dashboard/` -- Streamlit Dashboard
- `config.py` -- Zentrale Konfiguration
- `CLAUDE.md` -- Aktueller Durchlauf-Kontext fuer AI-Assistenten

---

## Ausfuehrung (Windows)

Alle Scripts werden aus dem Projektordner heraus gestartet:

```bash
cd gapper-analysis
.\gapper_env\Scripts\python.exe scripts/SCRIPT_NAME.py
```

---

# DETAILLIERTE DOKUMENTATION

## 1. Datenstruktur

### 1.1 Ordnerstruktur `data/`

```
data/
  raw_1min/              1-Min-Intraday-OHLCV pro Ticker/Datum
    {TICKER}/
      {YYYY-MM-DD}.parquet
  raw_1sec/              1-Sekunden-Daten (Subset, fuer Zoom-Analysen)
    {TICKER}/
      {YYYY-MM-DD}.parquet
  daily/                 90 Tage Daily-OHLCV-Kontext pro Ticker
    {TICKER}/
      {YYYY-MM-DD}_context.parquet
  baseline_1min/         Baseline-1-Min-Daten der letzten 20 Tage (fuer RVOL)
    {TICKER}/
      {YYYY-MM-DD}.parquet
  vwap/                  Intraday-VWAP + StdDev-Baender
    {TICKER}/
      {YYYY-MM-DD}.parquet
  volume_profile/        Developing Volume Profile (VPOC, VAH, VAL)
    {TICKER}/
      {YYYY-MM-DD}.parquet
  normalized/            ATR-normalisierte Intraday-Profile
    {TICKER}/
      {YYYY-MM-DD}.parquet
  metadata/              Metadata-Parquets + taegliche Gapper-CSVs
    metadata_v9.parquet      <- AKTUELLE VERSION (98 Spalten)
    metadata_v8_5.parquet    <- Vorversion (89 Spalten)
    metadata_v8.parquet      <- mit is_earnings Flag (85 Spalten)
    metadata_v7.parquet      <- Basis (82 Spalten)
    metadata_master.parquet  <- Rohdaten (63 Spalten)
    earnings_dates.parquet   <- Earnings-Kalender (Polygon API)
    gappers_YYYY-MM-DD.csv   <- ~1260 taegliche Scanner-CSVs
```

### 1.2 Datenmenge

| Datensatz | Anzahl | Beschreibung |
|-----------|--------|-------------|
| Ticker | 1.059 | Einzigartige Aktien im Universum |
| Gap-Events | 10.252 | Ein parquet pro Event in raw_1min |
| IS-Periode (H1) | 4.254 | 2021-02-22 bis 2023-12-28 |
| OOS-Periode (H2) | 5.903 | 2024-01-02 bis 2026-02-13 (laufend) |
| 1-Min-Bars pro Event | ~400-700 | Premarket 04:00 bis Afterhours 20:00 |
| 1-Sek-Daten | 664 Ticker | Subset fuer Zoom-Analysen |

### 1.3 1-Minuten-Bar Format (`data/raw_1min/{TICKER}/{DATE}.parquet`)

| Spalte | Typ | Beschreibung |
|--------|-----|-------------|
| `timestamp_utc` | int64 | Unix-Timestamp in Millisekunden |
| `datetime_et` | string | Datetime in Eastern Time |
| `time_et` | string | Uhrzeit in ET (z.B. "09:35") |
| `open` | float64 | Open-Preis der Minute |
| `high` | float64 | High-Preis der Minute |
| `low` | float64 | Low-Preis der Minute |
| `close` | float64 | Close-Preis der Minute |
| `volume` | int64 | Handelsvolumen der Minute |
| `vwap` | float64 | VWAP der Minute (von Polygon) |
| `transactions` | int64 | Anzahl Transaktionen |
| `ticker` | string | Ticker-Symbol |
| `date` | string | Datum (YYYY-MM-DD) |
| `session` | string | `premarket` / `rth` / `afterhours` |

**Sessions:**
- `premarket`: 04:00 - 09:29 ET
- `rth`: 09:30 - 16:00 ET (Regular Trading Hours)
- `afterhours`: 16:00 - 20:00 ET

---

## 2. Metadata: Alle Indikatoren und Features

Die zentrale Datei `data/metadata/metadata_v9.parquet` enthaelt 98 Spalten pro Gap-Event.
Sie wird von `scripts/07_compute_metadata.py` erzeugt und in spaeteren Durchlaeufen erweitert.

### 2.1 Identifikation

| Spalte | Typ | Beschreibung |
|--------|-----|-------------|
| `date` | string | Handelstag (YYYY-MM-DD) |
| `ticker` | string | Ticker-Symbol (z.B. AAPL, TSLA) |
| `name` | string | Firmenname |

### 2.2 Gap-Informationen

| Spalte | Typ | Beschreibung | Beispiel |
|--------|-----|-------------|---------|
| `gap_pct` | float | Gap in Prozent (positiv = up, negativ = down) | +5.2, -8.1 |
| `gap_direction` | string | `up` oder `down` | |
| `gap_size_in_adr` | float | Gap normalisiert auf ADR_10 | Median: 1.15 |
| `prev_close` | float | Schlusskurs Vortag (USD) | |
| `today_open` | float | Eroeffnungskurs (USD) | |
| `gap_filled` | bool | Wurde die Gap im Laufe des Tages gefuellt? | |
| `gap_fill_time_minutes` | float | Minuten bis Gap-Fill (NaN wenn nicht gefuellt) | |

### 2.3 Premarket-Daten

| Spalte | Typ | Beschreibung |
|--------|-----|-------------|
| `premarket_volume` | float | Gesamtvolumen Premarket (04:00-09:29) |
| `pm_vol` | float | Premarket-Volumen (Kurzform) |
| `pm_rth5` | float | Ratio: Premarket-Volume / RTH-First-5-Min-Volume |
| `pm_rth30_computed` | float | Ratio: Premarket-Volume / RTH-First-30-Min-Volume |

### 2.4 Preis-Kontext (aus Daily-Bars)

| Spalte | Typ | Beschreibung |
|--------|-----|-------------|
| `prior_return_5d` | float | Rendite der letzten 5 Handelstage |
| `prior_return_10d` | float | Rendite der letzten 10 Handelstage |
| `prior_return_20d` | float | Rendite der letzten 20 Handelstage |
| `dist_from_20sma` | float | Abstand zum 20-Tage-SMA (normalisiert) |
| `dist_from_52w_high` | float | Abstand zum 52-Wochen-Hoch (negativ = darunter) |
| `rsi_14_prev` | float | RSI(14) des Vortags |

### 2.5 Volatilitaet: ADR (Average Daily Range)

ADR = Durchschnitt von (High - Low) der letzten N Tage. Nicht zu verwechseln mit ATR.

| Spalte | Typ | Beschreibung | Median IS |
|--------|-----|-------------|-----------|
| `adr_5` | float | ADR ueber 5 Tage (USD) | 1.98 |
| `adr_10` | float | ADR ueber 10 Tage (USD) -- **Hauptreferenz** | 1.95 |
| `adr_20` | float | ADR ueber 20 Tage (USD) | 1.88 |

**Verwendung:** ADR_10 ist die zentrale Normalisierungsgroesse fuer:
- Gap-Groesse (`gap_size_in_adr = gap_pct * prev_close / adr_10`)
- PnL-Messung (`pnl_adr = pnl / adr_10`)
- SL-Distanz (`sl = 0.25 * adr_10` im Baseline-Setup)
- Alle "in ADR"-Metriken

### 2.6 Relative Volume (RVOL)

RVOL = Volumen / Durchschnitt der letzten 20 Tage zum gleichen Zeitpunkt.
Misst ungewoehnliche Handelsaktivitaet.

| Spalte | Typ | Beschreibung | Median IS |
|--------|-----|-------------|-----------|
| `rvol_5` | float | RVOL nach 5 Min (09:30-09:34) | 2.82 |
| `rvol_at_time_5min` | float | RVOL-at-time bei 5 Min | |
| `rvol_at_time_15min` | float | RVOL-at-time bei 15 Min | |
| `rvol_at_time_30min` | float | RVOL-at-time bei 30 Min | |
| `rvol_at_time_60min` | float | RVOL-at-time bei 60 Min | |
| `rvol_open_30min` | float | RVOL der ersten 30 Min (Kurzform: RVOL_30) | 1.96 |
| `rvol_full_day` | float | RVOL des gesamten RTH-Tages | 1.93 |

**Interpretation:**
- RVOL < 1: Unterdurchschnittliches Volumen
- RVOL 1-3: Normal
- RVOL 3-5: Erhoehtes Interesse
- RVOL > 5: Starkes Interesse (wichtiger Filter in Backtests)
- RVOL > 10: Extremes Interesse

### 2.7 Opening Drive (OD) -- Erste 5 Minuten (09:30-09:34)

Der Opening Drive ist die Kursbewegung der ersten 5 Minuten nach Marktoeffnung.
Er bestimmt die Trade-Richtung.

| Spalte | Typ | Beschreibung | Median IS |
|--------|-----|-------------|-----------|
| `od_strength` | float | OD-Staerke = abs(close_935 - rth_open) / adr_10 | 0.22 |
| `od_direction` | string | `with_gap` oder `against_gap` | |
| `od_long` | bool | True = OD bullish, False = OD bearish | |
| `od_body_pct` | float | Koerper-Anteil der 5-Min-OD-Kerze (0-1) | 0.46 |
| `od_wick_ratio` | float | Docht-Verhaeltnis der OD-Kerze | 0.22 |
| `od_high5` | float | Hoechstkurs der ersten 5 Minuten (USD) | |
| `od_low5` | float | Tiefstkurs der ersten 5 Minuten (USD) | |
| `opening_drive_atr` | float | OD-Range / ADR (aelterer Indikator) | |
| `opening_drive_direction` | string | `with_gap` / `against_gap` (aeltere Version) | |
| `candle5_vs_candle1` | float | Verhaeltnis 5. Kerze zu 1. Kerze | |

**od_direction vs gap_direction:**
- `with_gap`: OD geht in Gap-Richtung (GapUp + OD bullish, oder GapDn + OD bearish)
- `against_gap`: OD geht gegen die Gap (GapUp + OD bearish, oder GapDn + OD bullish)

**Wichtiger Filter:** `od_strength > 0.5` filtert auf starke Opening Drives (ca. 18% der Events).

### 2.8 Erste Kerze (09:30)

| Spalte | Typ | Beschreibung |
|--------|-----|-------------|
| `first_candle_dir` | string | `with_gap` oder `against_gap` |
| `first_candle_size` | float | Groesse der 1. Kerze / ADR_10 |

### 2.9 Schluessel-Preislevels

| Spalte | Typ | Beschreibung |
|--------|-----|-------------|
| `close_935` | float | Close-Preis um 09:35 (Ende OD) -- **Entry-Preis** |
| `close_1000` | float | Close-Preis um 10:00 |
| `close_1100` | float | Close-Preis um 11:00 |
| `rth_open` | float | RTH-Eroeffnungskurs (09:30) |
| `rth_close` | float | RTH-Schlusskurs (16:00) |
| `rth_high` | float | RTH-Tageshoch |
| `rth_low` | float | RTH-Tagestief |
| `hod_time` | string | Uhrzeit des Tageshochs |
| `lod_time` | string | Uhrzeit des Tagestiefs |

### 2.10 Stop-Loss-Levels (vorberechnet in v9)

| Spalte | Typ | Beschreibung |
|--------|-----|-------------|
| `sl_full_dist` | float | SL-Distanz "Full OD" = abs(close_935 - od_extreme) in USD |
| `sl_half_dist` | float | SL-Distanz "Half OD" in USD |
| `sl_full_level` | float | SL-Level "Full OD" in USD |
| `sl_half_level` | float | SL-Level "Half OD" in USD |
| `sl_full_adr` | float | SL Full in ADR (Median: 0.37) |
| `sl_half_adr` | float | SL Half in ADR (Median: 0.26) |
| `trade_direction` | string | `long` oder `short` (folgt od_long) |

### 2.11 Drift-Metriken (Kursbewegung nach Entry)

Alle Drift-Werte sind normalisiert auf ADR_10 und richtungsbereinigt (positiv = in Trade-Richtung).

| Spalte | Typ | Beschreibung |
|--------|-----|-------------|
| `rest_drift` | float | Drift von 09:35 bis Close, normalisiert auf ADR |
| `rest_drift_1000` | float | Drift von 10:00 bis Close |
| `rest_drift_1100` | float | Drift von 11:00 bis Close |
| `full_drift` | float | Drift von RTH Open bis Close |
| `cl` | float | Close Location (0-1): wo hat der Kurs im Tagesrange geschlossen |

### 2.12 VWAP-Metriken

| Spalte | Typ | Beschreibung |
|--------|-----|-------------|
| `vwap_z_at_10am` | float | VWAP Z-Score um 10:00 |
| `vwap_z_at_1030am` | float | VWAP Z-Score um 10:30 |
| `vwap_z_at_max_extension` | float | VWAP Z-Score am Maximum |
| `time_in_vwap_bands` | float | % der RTH-Zeit innerhalb +/- 1 StdDev |
| `vwap_held` | bool | Blieb Preis ueber/unter VWAP (je nach Gap-Richtung) |
| `vwap_cross_time_minutes` | float | Minuten bis VWAP gekreuzt wurde |

### 2.13 Volume Profile

| Spalte | Typ | Beschreibung |
|--------|-----|-------------|
| `vpoc_at_close` | float | Volume Point of Control bei Tagesschluss |
| `vpoc_migration` | float | VPOC-Wanderung im Tagesverlauf |
| `close_vs_vpoc` | float | Abstand Close zu VPOC |
| `close_in_value_area` | bool | Close innerhalb der Value Area (70% Volumen) |

### 2.14 Tages-Outcome

| Spalte | Typ | Beschreibung |
|--------|-----|-------------|
| `close_vs_open_adr` | float | (Close - Open) / ADR |
| `max_extension_adr` | float | Maximale Ausdehnung in Gap-Richtung / ADR |
| `max_adverse_adr` | float | Maximale Gegenbewegung / ADR |
| `reversal_occurred` | bool | Reversal (>50% der Gap gefadet) |
| `reversal_time_minutes` | float | Minuten bis Reversal |
| `close_vs_gap` | string | `extended` / `faded` / `reversed` |
| `close_location` | string | Wo Close relativ zu VWAP liegt |

### 2.15 Fundamentaldaten

| Spalte | Typ | Beschreibung |
|--------|-----|-------------|
| `sector` | string | Sektor (Manufacturing, Services, Finance, ...) |
| `market_cap` | float | Market Cap in USD |
| `market_cap_bucket` | string | `2-10B` / `10-50B` / `50-200B` / `200B+` |
| `is_earnings` | bool | Gap durch Earnings ausgeloest? |
| `earnings_unknown` | bool | Earnings-Status unklar |
| `nearest_earnings_date` | string | Naechstes Earnings-Datum |
| `catalyst_type` | string | Katalysator-Typ (aktuell alle "unknown") |
| `spy_return_day` | float | S&P 500 Rendite am Gap-Tag |
| `vix_level` | float | VIX-Level (aktuell leer/NaN) |
| `day_of_week` | int | Wochentag (0=Mo, 4=Fr) |

### 2.16 Qualitaets-Flags

| Spalte | Typ | Beschreibung |
|--------|-----|-------------|
| `quality_high` | bool | OD>0.5 AND with_gap AND RVOL_5>2 |
| `premarket_bar_fill_rate` | float | Anteil der Premarket-Minuten mit Daten |
| `gap_vs_prior_range` | float | Gap relativ zur Vortags-Range |

---

## 3. IS/OOS-Split (In-Sample / Out-of-Sample)

Alle Backtests verwenden einen festen zeitlichen Split:

| Bezeichnung | Zeitraum | Events | Zweck |
|-------------|----------|--------|-------|
| **H1 (IS)** | 2021-02-21 bis 2023-12-31 | 4.254 | Hypothesenbildung, Parameteroptimierung |
| **H2 (OOS)** | 2024-01-01 bis 2026-02-13 | 5.903 | Validierung, keine Optimierung erlaubt |

**Regel:** OOS-Daten werden NIEMALS fuer Parametersuche verwendet. Jeder IS-Befund wird
separat auf OOS getestet. Nur replizierte Befunde gelten als stabil.

---

## 4. Trade-Setup (aktueller Stand nach D12)

### 4.1 Baseline-Setup (TRAIL_D)

```
Qualifikation:  od_strength > 0.5 (starker Opening Drive)
Entry:          close_935 um 09:35 ET
Trade-Richtung: Long wenn od_long=True, Short wenn od_long=False
Stop-Loss:      0.25 x ADR_10 (fix)
Trail:          TRAIL_D: +1R erreicht -> SL auf Breakeven
                Danach: Trail = Peak - 0.5R
Timeout:        15:55 ET (Market Close)
```

**IS-Ergebnisse (Baseline, with_gap):**
- GapUp: N=205, WR=55.6%, EV=+0.149 ADR
- GapDn: N=181, WR=54.1%, EV=+0.026 ADR

### 4.2 Breakout + OD-basierter SL (D12-Variante)

```
Qualifikation:  od_strength > 0.5
Entry:          Breakout ueber/unter close_935 (09:36-10:00)
                Kein Breakout bis 10:00 -> kein Trade
Trade-Richtung: Long wenn od_long=True, Short wenn od_long=False
Stop-Loss:      Long: od_low5 - 0.01 (1 Tick unter 5-Min-Tief)
                Short: od_high5 + 0.01 (1 Tick ueber 5-Min-Hoch)
Trail:          TRAIL_D (wie oben)
Timeout:        15:55 ET
```

**IS-Ergebnisse (alle 4 Quadranten):**

| Quadrant | N | WR% | EV (ADR) | Med SL (ADR) |
|----------|---|-----|----------|-------------|
| GapUp + OD_Long (with_gap) | 188 | 59.0% | +0.172 | 0.952 |
| GapUp + OD_Short (against_gap) | 215 | 60.0% | +0.177 | 0.944 |
| GapDn + OD_Short (with_gap) | 167 | 57.5% | +0.096 | 0.864 |
| GapDn + OD_Long (against_gap) | 134 | 61.9% | +0.159 | 0.889 |

### 4.3 Trail-Typen (getestet in D10)

| Trail | Aktivierung | Initiales SL | Trail-Formel | Ergebnis |
|-------|-------------|-------------|--------------|----------|
| TRAIL_A | +1R | Breakeven | Peak - 1.0R | Gut |
| TRAIL_B | +1R | +0.5R | Peak - 1.0R | Gut |
| TRAIL_C | +2R | +1.0R | Peak - 1.0R | Weniger Trades |
| **TRAIL_D** | **+1R** | **Breakeven** | **Peak - 0.5R** | **Bester EV** |

---

## 5. Pipeline-Scripts (scripts/)

### 5.1 Daten-Pipeline (01-07)

| Script | Funktion | Input | Output |
|--------|----------|-------|--------|
| `01_scan_gappers.py` | Taeglicher Gap-Scanner | Polygon API | `metadata/gappers_YYYY-MM-DD.csv` |
| `02_download_intraday.py` | 1-Min-Bars herunterladen | Gapper-Liste | `raw_1min/{ticker}/{date}.parquet` |
| `02b_download_baseline.py` | Baseline fuer RVOL | Ticker-Liste | `baseline_1min/{ticker}/{date}.parquet` |
| `03_download_daily.py` | 90-Tage Daily-Kontext | Ticker-Liste | `daily/{ticker}/{date}_context.parquet` |
| `04_compute_vwap.py` | VWAP + StdDev-Baender | raw_1min | `vwap/{ticker}/{date}.parquet` |
| `05_compute_volume_profile.py` | Volume Profile (VPOC/VAH/VAL) | raw_1min | `volume_profile/{ticker}/{date}.parquet` |
| `06_normalize.py` | ATR-Normalisierung | raw_1min + daily | `normalized/{ticker}/{date}.parquet` |
| `07_compute_metadata.py` | Feature-Berechnung (98 Spalten) | Alle obigen | `metadata/metadata_vX.parquet` |
| `run_daily_pipeline.py` | Master-Orchestrator | CLI-Argumente | Alle obigen |

### 5.2 Analyse-Durchlaeufe (D9-D12)

Jeder "Durchlauf" (D) ist eine Forschungsiteration mit eigenen Scripts und Ergebnissen.

#### D9: R:R-Basis-Simulation
| Script | Funktion |
|--------|----------|
| `d9s_rr_basis.py` | R:R-Simulation mit 1-Sek-Zoom. Testet SL_FULL, SL_HALF, SL_FIX025 bei 1R/2R/3R Targets. |

#### D10: MFE-Analyse + Trailing-Stops
| Script | Funktion |
|--------|----------|
| `d10_trailing.py` | **Kern-Script:** TRAIL_A/B/C/D Simulation + Fix-Target-Vergleich |
| `d10_mfe_kontext.py` | MFE-Verhalten nach Kontext-Parametern (RVOL, Gap-Groesse, OD) |
| `d10_synthese.py` | Synthese: TRAIL_D als bestes Modell, IS vs OOS Shrinkage |
| `d10_oos.py` | OOS-Validierung der MFE- und Trail-Strategien |
| `d10s_mfe_trailing.py` | Hybrid 1-Min/1-Sek MFE-Simulation |

**Kernergebnis D10:** TRAIL_D (Peak - 0.5R nach +1R) ist das beste Exit-Modell.

#### D11: Big-Runner-Vorhersage
| Script | Funktion |
|--------|----------|
| `d11_target.py` | Big-Runner-Definition: Range >= 1.5 ADR + Close-Retention >= 80% |
| `d11_existing_params.py` | Trennschaerfe bestehender Parameter (Spearman, Buckets) |
| `d11_new_params.py` | Neue Features aus 1-Min-Daten (Pullback, Continuation, etc.) |
| `d11_multivariate.py` | Logistic Regression, Feature-Importance, Kombinations-Regeln |
| `d11_timing.py` | Wann erkennt man Big Runners? (09:35 vs 10:00 vs im Trade) |
| `d11_synthese.py` | Synthese: drift_30min staerkster Praediktor (rho ~0.25) |
| `d11_oos.py` | OOS-Validierung aller D11-Befunde |
| `d11s_distance.py` | Wie weit laufen Big Runners tatsaechlich? |
| `d11s_big_runners.py` | Big-Runner-Analyse mit 1-Sek-Zoom |
| `d11s_fix.py` | Duale BR-Definition: Range-basiert vs Entry-basiert |

**Kernergebnis D11:** Bester OOS-Filter: RVOL_30 > 5x + drift > 0.25 + OD > 1.0 (~50% BR-Rate).

#### D12: Trade-Setup-Variationen
| Script | Funktion |
|--------|----------|
| `d12_variation_test.py` | 7 Variationen: OD-Richtung, Breakout-Entry, OD-basierter SL |
| `d12_ergebnis.py` | Saubere Ergebnisaufbereitung mit allen 4 Quadranten |

**Kernergebnis D12:** Breakout-Entry + OD-basierter SL verbessert EV in allen 4 Quadranten.

---

## 6. Ergebnisse (results/)

Alle Analyse-Ergebnisse liegen in `results/`. Konvention:
- `.txt` -- Menschenlesbarer Report
- `.parquet` -- Rohdaten fuer Weiterverarbeitung

### 6.1 Durchlauf-Ergebnisse

| Datei | Inhalt |
|-------|--------|
| `d10_trailing.txt` | TRAIL_A/B/C/D Ergebnisse IS |
| `d10_mfe.txt` / `d10_mfe_kontext.txt` | MFE-Analyse |
| `d10_synthese.txt` | D10-Zusammenfassung |
| `d10_oos.txt` | D10 OOS-Validierung |
| `d10_mfe_raw_is.parquet` | MFE-Rohdaten IS |
| `d10_mfe_raw_oos.parquet` | MFE-Rohdaten OOS |
| `d11_target.txt` | Big-Runner-Definition + Basis-Statistik |
| `d11_existing_params.txt` | Bestehende Parameter-Trennschaerfe |
| `d11_new_params.txt` | Neue Parameter-Trennschaerfe |
| `d11_multivariate.txt` | Multivariate Analyse |
| `d11_timing.txt` | Temporale Erkennbarkeit |
| `d11_synthese.txt` | D11-Zusammenfassung |
| `d11_oos.txt` | D11 OOS-Validierung |
| `d11_is_enriched.parquet` | IS-Daten mit allen neuen Features |
| `d11_oos_enriched.parquet` | OOS-Daten mit allen neuen Features |
| `d12_variation_test.txt` | D12 alle 7 Variationen |
| `d12_variation_raw.parquet` | D12 Trade-Rohdaten |
| `D12_Ergebnis.txt` | D12 saubere Zusammenfassung (im Projektroot) |

### 6.2 Aeltere Analysen

| Ordner/Datei | Inhalt |
|------|--------|
| `backtest/`, `backtest_v2/`, `backtest_v3/` | Fruehere Backtest-Versionen |
| `cheat_cat1_*.txt` | Cheat-Sheet Kategorien |
| `ml_pipeline_v3_results.txt` | ML-Pipeline-Ergebnisse |
| `10am_analysis/` | 10:00-Uhr Analyse |
| `dipgo_analysis/`, `dipgo_phase2/`, `dipgo_phase3/` | Dip-and-Go Analyse |

---

## 7. Metadata-Versionen

| Version | Spalten | Aenderungen gegenueber Vorversion |
|---------|---------|-----------------------------------|
| v7 | 82 | Basis-Features, VWAP, Volume Profile, OD |
| v8 | 85 | + `is_earnings`, `earnings_unknown`, `nearest_earnings_date` |
| v8.5 | 89 | + `od_body_pct`, `od_wick_ratio`, `candle5_vs_candle1`, `quality_high` |
| **v9** | **98** | + `od_high5`, `od_low5`, `sl_full_dist`, `sl_half_dist`, `sl_full_level`, `sl_half_level`, `trade_direction`, `sl_full_adr`, `sl_half_adr` |

---

## 8. Glossar

| Begriff | Bedeutung |
|---------|-----------|
| **ADR** | Average Daily Range = Durchschnitt(High-Low) der letzten N Tage |
| **OD** | Opening Drive = Kursbewegung in den ersten 5 Min (09:30-09:34) |
| **RVOL** | Relative Volume = aktuelles Volumen / Durchschnitt gleicher Zeitpunkt |
| **SL** | Stop-Loss |
| **R** | Risikoeinheit = Abstand Entry zu Stop-Loss |
| **PnL** | Profit and Loss |
| **EV** | Expected Value = durchschnittlicher PnL pro Trade |
| **WR** | Win Rate = Anteil gewonnener Trades |
| **MFE** | Maximum Favorable Excursion = maximaler Buchgewinn waehrend Trade |
| **MAE** | Maximum Adverse Excursion = maximaler Buchverlust waehrend Trade |
| **BR** | Big Runner = Trade mit Range >= 1.5 ADR + Close >= 80% Extremum |
| **IS** | In-Sample = Trainingsdaten (H1) |
| **OOS** | Out-of-Sample = Validierungsdaten (H2) |
| **TRAIL_D** | Bester Trail: +1R -> BE, dann Peak - 0.5R |
| **with_gap** | OD geht in Gap-Richtung |
| **against_gap** | OD geht gegen Gap-Richtung |
| **VPOC** | Volume Point of Control = Preis mit meistem Volumen |
| **VAH/VAL** | Value Area High/Low = Grenzen der 70%-Volumen-Zone |
| **RTH** | Regular Trading Hours (09:30-16:00 ET) |

---

## 9. Wie man neue Backtests baut

### 9.1 Schnellstart

```python
import pandas as pd
from pathlib import Path

# 1. Metadata laden
meta = pd.read_parquet('data/metadata/metadata_v9.parquet')

# 2. IS-Daten filtern
h1 = meta[(meta['date'] >= '2021-02-21') & (meta['date'] <= '2023-12-31')]

# 3. Qualifikation anwenden
qualified = h1[h1['od_strength'] > 0.5]

# 4. 1-Min-Bars fuer einen Trade laden
row = qualified.iloc[0]
bars = pd.read_parquet(f"data/raw_1min/{row['ticker']}/{row['date']}.parquet")
rth = bars[bars['session'] == 'rth']
post_entry = rth[rth['time_et'] >= '09:36']

# 5. Trade simulieren
entry_price = row['close_935']
sl_dist = 0.25 * row['adr_10']
trade_dir = 'long' if row['od_long'] else 'short'
```

### 9.2 Konventionen

1. **Immer IS/OOS trennen.** Parameter NIEMALS auf OOS optimieren.
2. **EV in ADR messen** (nicht in USD oder Prozent), da ADR-normalisiert vergleichbar ist.
3. **N < 10 -> "[LOW N]" Warnung.** Keine Aussage bei kleiner Stichprobe.
4. **Ergebnisse als .txt + .parquet speichern** in `results/`.
5. **Scripts benennen:** `d{N}_{thema}.py` (N = Durchlauf-Nummer).
6. **Trail-Simulation:** `simulate_trail_d()` aus `d10_trailing.py` ist die Referenz-Implementierung.

### 9.3 Typischer Analyse-Workflow

```
1. Hypothese formulieren (z.B. "RVOL > 5x verbessert WR")
2. IS-Daten filtern
3. Simulation laufen lassen (d10_trailing.py als Vorlage)
4. Ergebnisse nach Gap-Richtung splitten (GapUp / GapDn)
5. Statistik: N, WR%, EV(ADR), Median PnL(R), SL-Rate
6. Wenn IS vielversprechend: OOS-Validierung
7. Ergebnis dokumentieren in results/
```

### 9.4 Wichtige Filter-Kombinationen (aus bisheriger Forschung)

| Filter | Beschreibung | IS-Effekt |
|--------|-------------|-----------|
| `od_strength > 0.5` | Starker Opening Drive | Basis-Qualifikation |
| `od_direction == 'with_gap'` | OD in Gap-Richtung | Besserer EV (Baseline) |
| `rvol_open_30min >= 5` | Hohes 30-Min-Volumen | Staerkerer EV |
| `rvol_5 > 5` | Hohes 5-Min-Volumen | Frueh-Filter |
| `gap_size_in_adr > 2` | Grosse Gap | Mehr Big Runners |
| `is_earnings == True` | Earnings-Gap | Andere Mikrostruktur |

---

## 10. Bekannte Probleme und Hinweise

- `vix_level` ist in metadata_v9 komplett NaN (nie befuellt)
- `catalyst_type` ist immer "unknown" (Watchlist-Daten nicht systematisch gepflegt)
- `pm_rth5` kann NaN sein wenn RTH5-Volumen = 0
- `rvol_5` kann NaN sein wenn < 10 Baseline-Tage vorhanden
- 1-Min-Daten: manche Ticker haben Luecken (fehlende Minuten-Bars)
- `close_location` ist als String gespeichert (nicht Category)
- Polygon API Rate Limits beachten (5 Requests/Minute bei Free, mehr bei Paid)
- `od_long` in metadata_v9 ist Python bool (True/False), nicht String

---

## 11. D13: Agent-Team Analyse (2026-02-21)

### 11.1 Methodik

D13 wurde als **Agent-Team** durchgefuehrt mit drei spezialisierten Agenten:
- **Quant:** Statistische Feature-Exploration, 5 Hypothesen-Backtests, 23 Strategie-Kombinationen
- **Discretionary Trader:** Intraday-Microstructure, Time-of-Day-Analyse, kreative Setups (Reversal, Gap-Fill, Strong Open)
- **Devils Advocate:** Bonferroni-Korrektur, Jahres-Stabilitaet, Small-N-Checks, Timeout-Bias-Pruefung

Workflow: IS-Analyse -> Devils-Advocate-Review -> Vorregistrierung -> OOS-Validierung

### 11.2 IS-Ergebnisse (Quant)

**Feature-Trennschaerfe:** Vorhersagbare Features haben max rho=0.14 mit rest_drift. Staerkster Praediktor: opening_drive_atr.

**Top-Strategien (IS, od_strength > 0.5, SL=0.25*ADR, TRAIL_D):**

| Rang | Strategie | N | WR% | EV (ADR) |
|------|-----------|---|-----|----------|
| 1 | RSI 70+ + GapUp | 41 | 58.5% | +0.426 |
| 2 | RVOL>=5 + RSI>50 + GapUp + Non-Earnings | 99 | 54.5% | +0.248 |
| 3 | RVOL>=5 + RSI>50 + GapUp | 134 | 53.0% | +0.198 |
| 4 | RVOL>=5 + Gap 3+ ADR + Non-Earnings | 168 | 52.4% | +0.186 |
| 5 | Gap 3.0+ ADR | 297 | 50.8% | +0.147 |
| Baseline | od_strength > 0.5 | 767 | 52.3% | +0.071 |

**Kern-Erkenntnisse:**
- GapUp dominiert GapDn massiv in fast allen Strategien
- RSI>70 + GapUp = Overbought-Momentum (staerkster Einzelfilter, aber N=41)
- Non-Earnings > Earnings (konsistent besserer EV)
- RVOL und Gap-Size wirken additiv

### 11.3 IS-Ergebnisse (Trader)

| Analyse | Ergebnis | Edge? |
|---------|----------|-------|
| Erste-30-Min Microstructure | HOD/LOD in 37% der Faelle in ersten 5 Min. WithGap Pullback nur 0.13 ADR vs AgainstGap 1.24 ADR | Informativ |
| Time-of-Day PnL-Kurve | GapDn steigt ueber den Tag (+0.15 bis +0.26 ADR). WithGap bester Early-PnL (+0.22 @10:00) | Informativ |
| Reversal-Setup (VWAP-Cross nach 10:00) | EV = +0.001 ADR, WR 49.3% | **Kein Edge** |
| Gap-Fill als Signal | Mean PnL = -0.097 ADR, WR 43.6% | **Kein Edge** |
| Strong Open Setup | EV = +0.005 ADR, underperformt Referenz | **Kein Edge** |

### 11.4 Devils Advocate Review

| Strategie | Bonferroni | Jahres-Stabil | Small-N | Timeout-Bias | Urteil |
|-----------|-----------|---------------|---------|-------------|--------|
| RSI 70+ GapUp | FAIL (p=1.0) | 2022: N=4 | PASS | Kein | FRAGIL |
| RVOL>=5+RSI>50+GapUp+NonEarn | FAIL (p=0.43) | Ratio 0.16 | PASS | Kein | FRAGIL |
| RVOL>=5+RSI>50+GapUp | FAIL (p=0.33) | Ratio 0.28 | PASS | Kein | FRAGIL |
| RVOL>=5+Gap3+ADR+NonEarn | FAIL (p=0.19) | n/a | PASS | Kein | FRAGIL |
| **Gap 3.0+ ADR** | **PASS (p=0.035)** | n/a | PASS | Kein | **ROBUST** |

Positiv: TRAIL_D erzeugt 0% Timeout-Trades bei diesen Strategien (kein Timeout-Artefakt).

### 11.5 OOS-Validierung (2024-01-01 bis 2026-02-13)

**Vorregistrierte Hypothese H1 = BESTAETIGT:**

| Metrik | IS (2021-2023) | OOS (2024-2026) |
|--------|---------------|-----------------|
| N | 297 | 506 |
| WR | 50.8% | 54.3% |
| EV (ADR) | +0.147 | **+0.135** |
| 95% CI | -- | [+0.094, +0.182] |
| P-Wert | 0.035 | **0.0000** |
| Shrinkage | -- | +8.1% |

**OOS-Details H1 (Gap 3.0+ ADR):**
- GapUp: N=288, EV=+0.198 ADR, P=0.0000
- GapDn: N=218, EV=+0.052 ADR, P=0.007
- 2024: N=232, EV=+0.124 ADR
- 2025: N=242, EV=+0.123 ADR
- 2026: N=32, EV=+0.313 ADR
- Exit-Types: 45.7% initial_sl, 54.3% trail_sl, 0% timeout
- Kumulative PnL: +273.73R ueber 506 Trades, Max Drawdown -10.26R

**Explorative Hypothesen (alle OOS-signifikant):**

| Hyp | Name | N_IS | EV_IS | N_OOS | EV_OOS | Shrinkage | P_OOS |
|-----|------|------|-------|-------|--------|-----------|-------|
| H1 | Gap 3.0+ ADR (VORREGISTRIERT) | 297 | +0.147 | 506 | +0.135 | +8% | 0.0000 |
| H2 | Baseline (od>0.5) | 767 | +0.071 | 1372 | +0.072 | -3% | 0.0000 |
| H3 | GapUp only (od>0.5) | 436 | +0.109 | 790 | +0.097 | +11% | 0.0000 |
| H4 | RVOL>=5 + RSI>50 + GapUp | 134 | +0.198 | 266 | +0.160 | +19% | 0.0000 |

### 11.6 Replizierbare Strategie (D13 Ergebnis)

```
Strategie:      Gap 3.0+ ADR Opening Drive
Qualifikation:  od_strength > 0.5 AND gap_size_in_adr >= 3.0
Entry:          close_935 um 09:35 ET
Trade-Richtung: Long wenn od_long=True, Short wenn od_long=False
Stop-Loss:      0.25 x ADR_10 (fix)
Trail:          TRAIL_D: +1R -> Breakeven, dann Trail = Peak - 0.5R
Timeout:        15:55 ET

IS:  N=297, WR=50.8%, EV=+0.147 ADR
OOS: N=506, WR=54.3%, EV=+0.135 ADR, P=0.0000
Gesamt: 803 Trades, Shrinkage 8%, Equity +273.73R, MaxDD -10.26R
```

**Optional (explorativ bestaetigt):**
- GapUp-Only Filter erhoeht EV auf +0.198 ADR (OOS)
- RVOL>=5 + RSI>50 + GapUp Filter erhoeht EV auf +0.160 ADR (OOS)

### 11.7 Projektstruktur D13

```
d13_team_analysis/
  quant/
    d13_quant_exploration.py         Quant IS-Analyse (5 Hypothesen, 23 Kombinationen)
  trader/
    d13_trader_analysis.py           Trader IS-Analyse (5 Analysen)
  devils_advocate/
    d13_devils_advocate.py           Kritische Pruefung (5 Checks)
  results/
    d13_quant_results.txt            Quant-Ergebnisse
    d13_trader_results.txt           Trader-Ergebnisse
    d13_devils_advocate_results.txt  Devils-Advocate-Ergebnisse
    d13_oos_validation.py            OOS-Validierungs-Script
    d13_oos_results.txt              OOS-Ergebnisse
  memory/
    TEAM_MEMORY.md                   Team-Wissensbasis
  TEAM_BRIEFING.md                   Team-Briefing
```
