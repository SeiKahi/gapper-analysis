# D13 TEAM BRIEFING - Agent-Teams fuer Systematische Trading-Strategien

## Mission
Entwicklung replizierbarer, systematischer Trading-Strategien auf Basis der Gapper-Daten.
Alles NUR auf IS-Daten (H1: 2021-02-21 bis 2023-12-31). OOS erst nach Hypothesen-Aufstellung.

## Team-Rollen

### QUANT (Agent 1)
**Aufgabe:** Statistische Analyse, Backtesting, Feature Engineering
- Systematische Exploration der 98 Metadata-Features auf Trennschaerfe
- Neue Strategie-Hypothesen formulieren und IS-backtesten
- Monte-Carlo / Bootstrap Signifikanz-Tests
- Fokus: Welche Filter-Kombinationen erzeugen robusten, positiven EV?

### DISCRETIONARY TRADER (Agent 2)
**Aufgabe:** Marktintuition, Pattern Recognition, kreative Setup-Ideen
- Analyse von Intraday-Price-Action-Mustern in den 1-Min-Daten
- Timing-basierte Strategien (wann einsteigen, wann raus?)
- Microstructure: Wie verhalten sich Gaps in den ersten 30 Min?
- Kreative Ideen: Reversal-Setups, Continuation-Setups, Time-of-Day-Effekte

### DEVILS ADVOCATE (Agent 3)
**Aufgabe:** Kritische Pruefung, Overfitting-Checks, Bias-Erkennung
- Prueft alle IS-Ergebnisse auf Robustheit
- Sucht nach Data-Mining-Bias, Look-Ahead-Bias, Survivorship-Bias
- Prueft ob Ergebnisse durch strukturelle Artefakte erklaerbar sind
- Stellt sicher, dass N genuegend gross ist und Effekte nicht durch Zufall erklaerbar

## Arbeitsregeln

1. **Nur IS-Daten verwenden** fuer Hypothesenbildung
2. **Scripts in d13_team_analysis/{agent}/ ablegen**
3. **Ergebnisse als .txt + .parquet in d13_team_analysis/results/**
4. **Memory in d13_team_analysis/memory/ aktualisieren**
5. **EV immer in ADR messen**
6. **N < 30 = keine Aussage moeglich**
7. **Referenz-Trail: TRAIL_D** (simulate_trail_d aus d10_trailing.py)
8. **Python: .\gapper_env\Scripts\python.exe**

## Bekannte Edges (zu bestaetigen/erweitern):
- OD > 0.5 + Breakout Entry + OD-SL + TRAIL_D (D12)
- RVOL_30 > 5x verbessert EV
- Earnings-Gaps haben andere Mikrostruktur
- Big Runner Filter: RVOL_30>5 + drift>0.25 + OD>1.0 (~50% BR-Rate)

## Noch nicht erforscht (Ideen):
- Premarket-Momentum als Praediktor
- Gap-Fill-Wahrscheinlichkeit nach Kontext
- Sector-Rotation-Effekte
- VIX-Regime-Effekte (Daten muessen geholt werden)
- Multi-Day-Follow-Through (Tag 2-5 nach Gap)
- Intraday-Volume-Distribution (wann kommt das Volumen?)
- Opening Range Breakout Varianten
- Gap-Size-Optimierung (3% vs 5% vs 8%+)
- Kombinierte Strategien (OD + VWAP + Volume Profile)
