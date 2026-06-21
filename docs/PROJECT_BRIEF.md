# PROJECT BRIEF — Bisherige Erkenntnisse

## Datensatz
- 10,142 unique Ticker-Tage, 421,817 Zone-Swings erkannt
- Zeitraum: ~2020-2024 (US-Markt)
- Gapper-Definition: Aktien mit >4% Gap at Open
- Datenquelle: 1-Minuten OHLCV + berechnete VWAP + ±1/2/3 StdDev Bänder

## ADR% Verteilung der Gapper
- P10: 2.4%, P25: 3.5%, P50: 5.6%, P75: 8.4%, P90: 11.8%
- 94.5% haben ADR >= 2%, 81.7% haben ADR >= 3%
- Erkenntnis: ADR% als Filter bringt fast nichts, weil Gapper sowieso volatil sind

## Preis-Verteilung
- P10: $13.18, P25: $18.97, P50: $35.32, P75: $78.42, P90: $168.43

## Szenarien (12 Stück, 6 relevante)
Getestet wurden Zone-Crosses an VWAP-Bändern. Die 6 Kern-Szenarien:

| Szenario | Baseline →1Lvl | Mit Morgen-Filter |
|----------|----------------|-------------------|
| GapUp +2σ Fade (down) | 35.9% | **58.1%** (Open30) |
| GapUp +1σ Fade (down) | 19.4% | **37.8%** (Open30) |
| GapUp VWAP Break (down) | 19.7% | **34.9%** (Open30) |
| GapDn -2σ Fade (up) | 34.4% | **56.3%** (Open30) |
| GapDn -1σ Fade (up) | 18.4% | **33.8%** (Open30) |
| GapDn VWAP Break (up) | 19.1% | **33.1%** (Open30) |

## Stärkste Filter (nach Effekt-Größe)
1. **Tageszeit** — KILLER-Filter. Open 30min verdoppelt/verdreifacht Hit Rates
2. **%ADR Used** — Frische Aktien (<50% ADR verbraucht) deutlich besser
3. **ADR%** — Fast kein Effekt (Gapper sind sowieso volatil)
4. **Gap-ADR / RVOL** — Schwache Einzelfilter, beeinflussen Move-Größe, nicht Wahrscheinlichkeit
5. **Early Range** — INVERS! Hohe ER = breitere Bänder = weniger Crosses (aber größere Moves)

## Zeitfenster-Analyse (Neue Erkenntnisse)
Die Moves sind SCHNELL:

### GapUp +2σ Fade, Open 30min (n=769):
- 67% der Gewinner erreichen Level 1 in 5min, 90% in 15min
- Median Time-to-Level-1: 4min (P25=2min, P75=7min)
- Median Time-to-Level-2: 17min
- Zeitfenster-Win-Rate: 51.6% innerhalb 15min, 55.3% innerhalb 30min

### MAE (Max Adverse Excursion):
- ±2σ Gewinner: Median MAE = 0.131 ADR, P75 = 0.283 ADR, P90 = 0.539 ADR
- ±1σ Gewinner: Median MAE = 0.032 ADR, P75 = 0.074 ADR — viel cleaner
- Verlierer sterben in Median 2 Minuten, Max Move = 0.000 ADR (gehen NIE in richtige Richtung)

### Gewinner-Profil (±2σ Morgen):
- 55% Win Rate, Median 5min bis Level 1
- 73% conditional weiter zu Level 2 (Median +23min)
- Pfad-Effizienz nur 0.074 — viel Hin-und-Her trotz Treffer

### Gewinner-Profil (±1σ Morgen):
- 33% Win Rate, Median 4min bis Level 1
- 62% conditional weiter zu Level 2 (Median +13min)
- Viel niedrigere MAE (0.032 ADR Median)

## Conditional Cascading
Wenn Level 1 erreicht → 59-73% weiter zu Level 2
Wenn Level 2 erreicht → 47-58% weiter zu Level 3
Stark konsistent über alle Szenarien.

## Offene Fragen für ML
1. Gibt es Feature-Interaktionen die in den Matrizen unsichtbar sind?
2. Kann ML non-lineare Kombinationen finden? (z.B. "RVOL>3 + Gap>1.5 + Morning + <50%ADR + ER>75% = 65%")
3. Validierung auf Out-of-Sample Daten (zeitbasierter Split)
4. Welche konkreten Entry/Exit-Regeln maximieren den Erwartungswert?
