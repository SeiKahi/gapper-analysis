# ENTRY & EXIT DEFINITIONEN

## Ziel
Jede Kombination aus Entry-Trigger × Stop-Loss × Target muss exakt definiert sein.
Kein Interpretationsspielraum. Jede Regel muss auf 1-Minuten OHLCV-Daten umsetzbar sein.

---

## 1. LEVEL-BERÜHRUNG: Wann gilt ein Level als "erreicht"?

### Definition A: Close-Cross (bisherige Methode)
- Level gilt als erreicht wenn: `close` einer 1min-Kerze auf der anderen Seite des Levels liegt
- Problem: Ignoriert Dochte. Aktie kann Level mit Docht berühren, Close bleibt drüber

### Definition B: Wick-Touch
- Level gilt als berührt wenn: `low <= level` (für obere Bänder) oder `high >= level` (für untere)
- Problem: Ein einzelner Tick reicht aus. Sehr rauschig.

### Definition C: Close-Beyond (strenger)
- Level gilt als durchbrochen wenn: `close` einer 1min-Kerze mindestens 1 Tick JENSEITS des Levels liegt
- Für Fade down bei +2σ: `close < upper_2std`
- Strikter als A, weil Close nicht auf dem Level sein darf, sondern darüber hinaus

**→ Für ML: Alle drei testen. "Definition" als Feature kodieren.**

---

## 2. ENTRY-TRIGGER: Wann genau steigst du ein?

### Entry E1: Sofort bei Zone-Cross
- Sobald close einer 1min Kerze das Level kreuzt → Entry am close dieser Kerze
- Entry-Preis = close der Signal-Kerze
- Einfachste Methode, kein Bestätigungs-Filter

### Entry E2: Nächste Kerze bricht Extremum der Signal-Kerze
- Signal: 1min Kerze berührt oder kreuzt das Level
- Bestätigung: NÄCHSTE 1min Kerze bricht das Low (für Short) / High (für Long) der Signal-Kerze
- Entry-Preis = Low der Signal-Kerze (Short) / High der Signal-Kerze (Long)
- Logik: Bestätigt dass Momentum in die richtige Richtung geht

### Entry E3: Doppelter Kerzen-Break (2 Bestätigungen)
- Signal: 1min Kerze berührt/kreuzt Level
- Bestätigung 1: Nächste Kerze bricht Low/High der Signal-Kerze
- Bestätigung 2: ÜBERNÄCHSTE Kerze bricht Low/High der Bestätigungs-Kerze
- Entry-Preis = Low/High der Bestätigungs-1-Kerze
- Logik: Stärkere Momentum-Bestätigung, weniger Fakeouts, aber späterer Entry

### Entry E4: Close + Retest
- Signal: 1min Kerze schließt jenseits des Levels
- Warte: Bis Preis nochmal zum Level zurückkehrt (Retest) — high >= level (Short) oder low <= level (Long)
- Entry: Bei der Kerze die das Level retestet, am close
- Timeout: Wenn kein Retest innerhalb von 5 Kerzen → kein Trade
- Logik: Besserer Preis, Level wird als Widerstand/Unterstützung bestätigt

### Entry E5: Aggro Wick-Entry
- Signal: 1min Kerze hat Docht der das Level berührt, aber Close bleibt auf der "alten" Seite
- Entry: Sofort am Close dieser Kerze (antizipiert den Cross)
- Logik: Frühester Entry, größtes R:R, aber höchste False-Positive-Rate

**→ Für ML: E1-E5 jeweils als separater Backtest. Danach vergleichen.**

---

## 3. STOP LOSS: Wo liegt der Stopp?

### SL1: Pivot-High/Low (Signal-Kerze)
- Short: SL = High der Signal-Kerze (die Kerze die das Level berührt hat)
- Long: SL = Low der Signal-Kerze
- Dynamisch, passt sich dem Kontext an
- Problem: Kann sehr eng sein (1-2 Ticks) wenn Signal-Kerze klein ist

### SL2: Pivot + Buffer
- Wie SL1, aber mit Buffer: SL = Pivot ± 0.05 ADR
- Gibt etwas mehr Raum für Noise

### SL3: Fixe ADR-Größe (klein)
- SL = Entry ± 0.10 ADR
- Sehr eng, wird oft ausgestoppt
- ADR = ADR_10 (basierend auf den letzten 10 Handelstagen VOR dem aktuellen Tag)

### SL4: Fixe ADR-Größe (mittel)
- SL = Entry ± 0.20 ADR
- Basierend auf MAE-Analyse: Median MAE der Gewinner bei ±1σ = 0.032, P75 = 0.074
- 0.20 ADR deckt >90% der Gewinner-MAE ab bei ±1σ

### SL5: Fixe ADR-Größe (groß)
- SL = Entry ± 0.35 ADR
- Basierend auf MAE-Analyse: Median MAE der Gewinner bei ±2σ = 0.131, P75 = 0.283
- 0.35 ADR deckt ~80% der Gewinner-MAE ab bei ±2σ

### SL6: Level-basiert (zurück über das gekreuzte Level)
- Short bei +2σ: SL = upper_2std der aktuellen Kerze + 0.02 ADR Buffer
- Dynamisch — bewegt sich mit den Bändern mit
- Vorteil: Genau die Logik von "Trade ist ungültig wenn Level zurückerobert"

### SL7: ATR-basiert (5-Perioden ATR auf 1min)
- Berechne ATR(5) auf den letzten 5 1min-Kerzen
- SL = Entry ± 2 × ATR(5)
- Passt sich an aktuelle Mikro-Volatilität an

**→ Für ML: SL-Größe als Feature, NICHT als fixen Parameter. Das Modell lernt welche SL-Größe bei welchen Bedingungen optimal ist.**

---

## 4. TARGET: Wo liegt das Gewinnziel?

### T1: Nächstes VWAP-Level
- Von +2σ → Target = +1σ
- Von +1σ → Target = VWAP
- Von VWAP (break down) → Target = -1σ
- Dynamisch — Level bewegt sich mit den Bändern

### T2: Zwei VWAP-Levels
- Von +2σ → Target = VWAP (2 Levels)
- Vorteil: Conditional Cascading zeigt 59-73% Weiter-Rate

### T3: Fixe ADR-Größe
- Target = Entry ± 0.25 ADR (konservativ)
- Target = Entry ± 0.50 ADR (aggressiv)
- Unabhängig von Bändern

### T4: Risk-Multiple (R:R basiert)
- Target = Entry ± (SL-Distanz × 1.5) → 1.5R
- Target = Entry ± (SL-Distanz × 2.0) → 2R
- Target = Entry ± (SL-Distanz × 3.0) → 3R
- Hängt von der gewählten SL-Methode ab

### T5: Trailing Stop
- Kein fixes Target
- Trailing Stop = aktueller Preis ± 0.15 ADR (eng)
- Oder: Trailing Stop = aktueller Preis ± 0.25 ADR (weit)
- Startet erst nach Erreichen von 0.10 ADR Profit (Breakeven-Trigger)

### T6: Partieller Exit
- 50% Position bei T1 (nächstes Level)
- Restliche 50% mit Trailing Stop oder T2
- Sichert Profit, lässt Runner laufen

**→ Für ML: Target-Typ als Dimension. Berechne für jeden Swing ALLE Target-Varianten und vergleiche Expectancy.**

---

## 5. ZEIT-FILTER

### Z1: Nur Open 30min (09:30-10:00 ET)
### Z2: Nur Morning (09:30-10:30 ET)
### Z3: Keine Trades nach 11:30 ET
### Z4: Keine Trades in letzten 30min (15:30-16:00 ET)
### Z5: Kein Filter

**→ Stärkster bekannter Filter. MUSS in ML-Features enthalten sein.**

---

## 6. ZUSÄTZLICHE KREATIVE IDEEN

### Volumen-Bestätigung
- Entry nur wenn die Signal-Kerze überdurchschnittliches Volumen hat
  (Volume > 1.5 × Average der letzten 10 Kerzen)
- Logik: Volumen bestätigt dass das Level "beachtet" wird

### Band-Breite als Filter
- Entry nur wenn BandWidth < 0.50 ADR (enge Bänder = klar definierte Levels)
- Oder nur wenn BandWidth > 0.30 ADR (zu enge Bänder = Rauschen)

### Konsekutive Kerzen in Zone
- Entry nur wenn Preis mindestens 3 Kerzen in der neuen Zone war
- Logik: Bestätigt dass der Cross nicht nur ein Spike war

### Momentum-Divergenz
- RSI(14) auf 1min: Preis macht neues High bei +2σ, aber RSI macht tieferes High
- Logik: Klassisches Divergenz-Signal, deutet auf Erschöpfung hin

### VWAP-Slope
- Berechne Anstieg/Abfall des VWAP über die letzten 15 Minuten
- Flat VWAP = Range-Day → Fade wahrscheinlicher
- Stark steigender VWAP = Trend-Day → Continuation wahrscheinlicher

### Orderbuch-Proxy: Bid/Ask-Spread (falls verfügbar)
- Nicht in unseren Daten, aber für Live-Trading relevant
- Breiter Spread bei Level-Berührung = weniger Liquidität = mehr Slippage

### Multi-Timeframe Bestätigung
- 5min Kerze muss ebenfalls unter dem Level schließen
- Logik: Höherer Timeframe bestätigt den Break

---

## 7. KOMBINATIONEN FÜR ML

### Primäre Dimension: Szenario (6 Stück)
GapUp: +2σ Fade, +1σ Fade, VWAP Break
GapDn: -2σ Fade, -1σ Fade, VWAP Break

### Sekundäre Dimensionen (Features):
- Entry-Typ (E1-E5)
- SL-Methode/Größe (SL1-SL7)
- Target-Methode (T1-T6)
- Zeit-Filter (Z1-Z5)
- ADR% (kontinuierlich)
- Gap-ADR (kontinuierlich)
- RVOL (kontinuierlich)
- Early Range 15/30 (kontinuierlich)
- %ADR Used (kontinuierlich)
- BandWidth (kontinuierlich)
- Volume-Ratio der Signal-Kerze
- VWAP-Slope
- Kerzen seit Market Open (kontinuierlich)
- Preis relativ zum VWAP (in ADR)

### Label (was wird vorhergesagt):
- Binär: Erreicht Trade das Target bevor SL? (1/0)
- Regression: Max Favorable Excursion in ADR
- Multi-Class: Outcome (SL hit, Target hit, Timeout, Breakeven)

### Train/Test Split:
- Train: Erste ~60% der Daten (zeitbasiert, z.B. bis Ende 2022)
- Validation: Nächste ~20% (z.B. 2023)
- Test: Letzte ~20% (z.B. 2024) — wird NICHT angeschaut bis zum Schluss
