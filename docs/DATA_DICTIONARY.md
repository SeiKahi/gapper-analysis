# DATA DICTIONARY

## WICHTIGSTE DATENQUELLE: raw_1min (OHLC!)
Pfad: `data/raw_1min/{ticker}/{date}.parquet`
1-Minuten OHLCV mit VWAP. **Das ist die reichste Datenquelle.**

| Spalte | Typ | Beschreibung |
|--------|-----|--------------|
| timestamp_utc | datetime | UTC Timestamp |
| datetime_et | datetime | Eastern Time |
| time_et | str | Uhrzeit ET (HH:MM) |
| open | float | Open der 1min Kerze |
| high | float | High der 1min Kerze |
| low | float | Low der 1min Kerze |
| close | float | Close der 1min Kerze |
| volume | int | Volumen der 1min Kerze |
| vwap | float | VWAP (von Polygon, kumulativ) |
| transactions | int | Anzahl Transaktionen |
| ticker | str | Ticker-Symbol |
| date | str | Datum |
| session | str | Session-Info |

## baseline_1min (OHLCV ohne VWAP)
Pfad: `data/baseline_1min/{ticker}/{date}.parquet`
Gleiche Struktur wie raw_1min aber OHNE vwap/transactions. 
Enthält auch Nicht-Gapper-Tage (Baseline-Daten).

## vwap/ (nur Close + berechnete VWAP-Bänder)
Pfad: `data/vwap/{ticker}/{date}.parquet`
NUR Close + Volume + berechnete VWAP + StdDev-Bänder. KEIN Open/High/Low!
Enthält: close, volume, vwap, std_dev, upper/lower 1/2/3 std, z_score, 
diverse band_upper/band_lower Spalten.

**Empfehlung:** Nutze `raw_1min` als Hauptdatenquelle und berechne 
VWAP-Bänder selbst, ODER merge `raw_1min` OHLC mit `vwap/` Bändern 
über ticker+date+time_et.

## metadata_master.parquet
Pfad: `data/metadata/metadata_master.parquet`
Ein Eintrag pro Gapper-Tag. **63 Spalten** (mehr als ursprünglich dokumentiert).

Wichtigste Spalten:
| Spalte | Typ | Beschreibung |
|--------|-----|--------------|
| ticker | str | Ticker-Symbol |
| date | str | Datum (YYYY-MM-DD) |
| gap_direction | str | 'up' oder 'down' |
| gap_pct | float | Gap-Größe in Prozent |
| gap_size_in_adr | float | Gap-Größe / ADR_10 |
| prev_close | float | Vortags-Schlusskurs |
| open_price | float | Eröffnungskurs |
| adr_10 | float | Average Daily Range (10 Tage, VORTAGE) |
| rvol_at_time_30min | float | Relative Volume nach 30min |
| sector | str | Sektor |
| catalyst_type | str | Catalyst-Typ |
| spy_return_day | float | SPY Tagesreturn |
| vix_level | float | VIX Level |
| rsi_14_prev | float | RSI 14 Vortag |
| dist_from_52w_high | float | Distanz zum 52W-Hoch |
| float_shares | float | Free Float |
| market_cap | float | Marktkapitalisierung |

Für vollständige Spaltenliste: 
`python -c "import pandas as pd; m=pd.read_parquet('data/metadata/metadata_master.parquet'); print(m.columns.tolist())"`

## daily/ und normalized/ und volume_profile/
Weitere Datenordner vorhanden, Inhalt unbekannt. Bei Bedarf inspizieren.
