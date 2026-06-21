# FINDINGS LOG
Agent schreibt hier fortlaufend Erkenntnisse rein.

---

## 2026-02-13 — Session 1: ML Pipeline v2

### Daten-Realitaet (WICHTIG!)
- Die VWAP-Parquet-Dateien haben **NUR close + volume** — kein open/high/low!
- 29 Spalten pro Datei: datetime_et, time_et, close, volume, vwap, std_dev, upper/lower 1/2/3 std, z_score, plus band_upper/band_lower Paare fuer jedes Level
- Die `_band_upper`/`_band_lower` Spalten sind Konfidenz-Baender um die VWAP-Levels, NICHT Preis-OHLC
- Konsequenz: Alle Entry/Exit/SL-Logik muss auf Close-Crosses basieren (kein Wick-Touch moeglich)
- Entry-Typen E3 (Doppelter Break), E4 (Retest), E5 (Wick-Touch) aus ENTRY_DEFINITIONS.md sind nicht umsetzbar

### Datensatz-Ueberblick
- 10,161 Gapper-Tage, 10,161 VWAP-Dateien (1:1 Zuordnung)
- Zeitraum: 2021-02-10 bis 2026-02-06
- Metadata hat 63 Spalten (viel mehr als im DATA_DICTIONARY beschrieben)
- adr_10 hat 10,142 non-null Werte (99.8% Coverage)
- Zusaetzliche nuetzliche Metadata-Spalten: sector, catalyst_type, spy_return_day, vix_level, rsi_14_prev, dist_from_52w_high, etc.

### Split-Aufteilung (korrigiert per CLAUDE.md v2)
- Nur Daten 2021-02-21 bis 2023-12-31 (4,254 Gapper)
- Train: 2021-02-21 bis 2022-12-31 (~3,054 Gapper)
- Validation: 2023-01-01 bis 2023-12-31 (~1,200 Gapper)
- 2024+ Daten: GESPERRT (nicht geladen)

### Pipeline v3 Design
- Entry E1: Sofort bei Close-Cross (close[i-1] auf einer Seite, close[i] auf der anderen)
- Entry E2: E1 + naechste Kerze bestaetigt Richtung (close continues in trade direction)
- SL-Varianten: SL3 (0.10 ADR), SL4 (0.20 ADR), SL5 (0.35 ADR), SL6 (Level-basiert + 0.05 ADR Buffer)
- Target-Varianten: T1 (naechstes VWAP Level, dynamisch), T3_025 (0.25 ADR), T3_050 (0.50 ADR), T4_15R/2R/3R (R-Multiple)
- 6 Szenarien: GapUp 2s/1s/VWAP Fade + GapDn 2s/1s/VWAP Fade
- Features: 23 numerische + 4 kategorische (scenario/entry/sl/target als IDs)
- Modelle: XGBoost + LightGBM mit Early Stopping auf Validation
- Timeout: 240 Bars (4h), Cooldown zwischen Signalen: 5 Bars

---

## 2026-02-13 — Session 1: ML Pipeline v3 RESULTATE

### Datensatz
- 4,254 Gapper verarbeitet (5,907 per Datumsfilter uebersprungen, 0 Fehler)
- 1,570,872 Trades generiert (jeder Gapper erzeugt viele Kombis aus Entry x SL x Target)
- Train: 1,121,592 Trades (label_mean=0.273), Val: 449,280 Trades (label_mean=0.286)

### Szenario-Ueberblick (alle Trades)
| Szenario | N | WinRate | AvgPnL (ADR) |
|----------|---|---------|--------------|
| GapDn_2s_Fade | 111,504 | 29.7% | +0.0054 |
| GapDn_VWAP_Brk | 306,552 | 28.6% | +0.0170 |
| GapUp_2s_Fade | 132,816 | 30.1% | +0.0042 |
| GapUp_VWAP_Brk | 348,120 | 27.6% | +0.0079 |
| GapUp_1s_Fade | 352,488 | 26.9% | +0.0028 |
| GapDn_1s_Fade | 319,392 | 25.9% | -0.0001 |

**Erkenntnis:** +-2s Fades und VWAP Breaks haben positive Expectancy. +-1s Fades sind marginal/negativ.

### ML Modell-Performance
- **XGBoost**: Train AUC=0.7966, Val AUC=0.7407 (Overfit-Gap ~0.06)
- **LightGBM**: Train AUC=0.8719, Val AUC=0.7336 (Overfit-Gap ~0.14, staerker overfitted)
- XGBoost generalisiert besser als LightGBM

### Feature Importance (XGBoost, Top 10)
1. **tgt_id** (0.1484) — Target-Methode ist wichtigster Praediktor
2. **minutes_since_open** (0.0953) — Tageszeit entscheidend
3. **e2_confirmed** (0.0792) — Ob naechste Kerze bestaetigt
4. **sl_dist_adr** (0.0692) — SL-Groesse
5. **entry_id** (0.0647) — Entry-Typ
6. **time_bucket** (0.0631) — Grobe Tageszeit
7. **sl_id** (0.0464) — SL-Methode
8. **early_range_15** (0.0385) — Fruehe Preisbewegung
9. **early_range_30** (0.0368) — Fruehe Preisbewegung
10. **pct_adr_used** (0.0344) — Wieviel ADR bereits verbraucht

### TOP 10 Profitable Setups (Validation)
| # | Szenario | Entry | SL | Target | N | WR | AvgPnL |
|---|----------|-------|----|--------|---|-----|--------|
| 1 | GapDn_2s_Fade | E2 | SL5_035 | T4_3R | 345 | 8.7% | +0.0739 |
| 2 | GapDn_VWAP_Brk | E2 | SL5_035 | T4_3R | 1120 | 6.5% | +0.0563 |
| 3 | GapDn_VWAP_Brk | E1 | SL5_035 | T4_3R | 2380 | 6.0% | +0.0546 |
| 4 | GapDn_2s_Fade | E2 | SL5_035 | T4_2R | 345 | 15.4% | +0.0521 |
| 5 | GapDn_VWAP_Brk | E1 | SL5_035 | T4_2R | 2380 | 14.5% | +0.0519 |
| 6 | GapDn_VWAP_Brk | E2 | SL5_035 | T4_2R | 1120 | 14.7% | +0.0512 |
| 7 | GapDn_2s_Fade | E2 | SL4_020 | T4_3R | 345 | 16.8% | +0.0507 |
| 8 | GapDn_2s_Fade | E2 | SL6_level | T4_3R | 345 | 25.5% | +0.0497 |
| 9 | GapUp_2s_Fade | E2 | SL6_level | T4_3R | 557 | 22.4% | +0.0484 |
| 10 | GapDn_VWAP_Brk | E2 | SL5_035 | T4_15R | 1120 | 23.7% | +0.0481 |

**Kern-Erkenntnisse aus den Top-Setups:**
- **GapDn dominiert**: 9 von 10 Top-Setups sind GapDown-Szenarien (Fade/Break nach oben)
- **R-Multiple Targets (T4)**: Beste Expectancy durch hohe R:R trotz niedriger WR
- **SL5 (0.35 ADR)**: Weiter Stop = weniger Ausstoppung = hoehere Expectancy bei R-Multiple Targets
- **E2 (Bestaetigungs-Entry)** erscheint haeufiger als E1 in den Top-Setups

### WORST Setups (Validation)
- **T1_next + SL5_035** sind fast alle unter den Verlierern
- Kombination "weiter SL + kleines Target" = schlechteste Expectancy
- GapUp-Szenarien erscheinen haeufiger bei den Verlierern

### High-Confidence Trades (Ensemble > 0.6, Validation)
- **N=26,307 Trades, WinRate=69.4%, AvgPnL=+0.0192 ADR**
- GapDn_2s_Fade hat hoechste WR (76.3%) und bestes PnL (+0.0294)
- SL6_level hat bestes PnL (+0.0409) unter den HC-Trades
- T3_050 (0.50 ADR Target) hat hoechste Expectancy (+0.1800) bei HC-Trades
- **E1 (+0.0347 PnL) deutlich besser als E2 (-0.0166 PnL)** bei HC-Trades (Gegensatz zu Top-Setup-Ranking!)

### Morning Filter Effekt (Validation)
| Zeitfenster | N | WR | AvgPnL |
|-------------|---|-----|--------|
| Morning (0-30m) | 41,544 | 36.7% | +0.0029 |
| Midday (30-120m) | 167,256 | 33.2% | +0.0115 |
| Afternoon (>120m) | 240,480 | 24.1% | +0.0092 |

**Erkenntnis:** Morning hat hoechste WR (36.7%), aber Midday hat die beste Expectancy (+0.0115).
Afternoon hat deutlich schlechtere WR (24.1%) — TIME-OF-DAY ist klarer Edge.

### Szenario x Zeit Interaktion (Validation)
- **GapUp_2s_Fade Morning: WR=40.2%** (bestes Morning-Szenario)
- **GapDn_VWAP_Brk Morning: WR=38.7%** (zweitbestes)
- Alle Szenarien verlieren nachmittags deutlich
- ABER: PnL bei "Rest" (nicht-Morning) ist teilweise hoeher → weiter SL laeuft nachmittags besser

### %ADR Used Filter (Validation)
- **<30% ADR verbraucht: WR=19.2%, PnL=-0.0124** — NEGATIVER Erwartungswert!
- 30-50% ADR: WR=28.6%, PnL=+0.0119 — gut
- 50-75% ADR: WR=30.3%, PnL=+0.0103 — ebenfalls gut
- >75% ADR: WR=28.6%, PnL=+0.0076 — leicht schlechter
- **Erkenntnis:** Trades bei <30% ADR used vermeiden. Sweet Spot: 30-75% ADR verbraucht.

### Overfitting-Check: Train vs Val
- Overall: Train WR=27.3%, Val WR=28.6% — Val leicht BESSER (kein Overfitting-Signal)
- GapDn_VWAP_Brk: Train +0.0125 → Val +0.0288 (verbessert sich OOS, robustes Signal)
- GapUp_VWAP_Brk: Train +0.0115 → Val -0.0004 (zerfaellt, NICHT robust)
- GapDn_2s_Fade: Train +0.0009 → Val +0.0202 (verbessert sich OOS)

### Zusammenfassung der Edges
1. **Staerkster Edge: GapDn + VWAP Break + weiter SL + R-Multiple Target** → robust, positiv in Train UND Val
2. **GapDn + 2s Fade + E2 + SL5/SL6 + T4_2R/3R** → hoechste Expectancy, aber kleine N
3. **Morning-Trades (0-30min)** → hoechste WR, sollte als Filter verwendet werden
4. **%ADR Used 30-75%** → Sweet Spot, <30% vermeiden
5. **GapUp VWAP Break ist NICHT robust** — zerfaellt auf Validation
6. **High-Confidence Ensemble** (>0.6) → 69.4% WR, stabiler Edge

### Dateien
- Script: `scripts/ml_pipeline_v3.py`
- Ergebnisse: `results/ml_pipeline_v3_results.txt`
- Trades: `results/all_trades_v3.parquet` (1.57M Trades)
- Modelle: `models/xgb_v3.pkl`, `models/lgb_v3.pkl`
- Feature Importance: `results/feature_importance_xgboost_v3.csv`, `results/feature_importance_lightgbm_v3.csv`
- Setup Rankings: `results/setup_rankings_v3_val.csv`

---

## 2026-02-13 — Session 2 (QUANT Round 2): Kritische Test-Bewertung

### 4 Kritische Tests durchgefuehrt und bewertet

**Test 1 — Random Entry Baseline:**
- VWAP-Cross Entry hat NULL Mehrwert gegenueber Random Entry
- Random SL5+T4_3R: +0.0249 ADR vs Pipeline: +0.0132 ADR
- Der Edge liegt NICHT im Entry-Timing, sondern in der Struktur (Gap-Richtung + SL/Target)

**Test 2 — Slippage Stress:**
- Top-6 GapDn-Setups mit SL5_035 ueberleben 0.10 ADR Slippage
- GapDn_VWAP_Brk (groesstes N=21000): +0.0395 bei slip=0, +0.0050 bei slip=0.10
- GapUp-Setups sind bei slip=0 bereits NEGATIV (-0.0088)
- Slippage-Robustheit beweist NICHT Existenz eines Edges

**Test 3 — e2_confirmed Leakage:**
- ML-Modell nach Leak-Fix WERTLOS fuer Trade-Selektion
- High-Confidence (>0.6) PnL: +0.0192 (mit Leak) -> -0.0159 (ohne Leak)
- WR steigt sogar (69.4% -> 73.0%), aber PnL wird negativ
- ML selektiert nicht die Tail-Gewinner, nur die haeufigeren kleinen Gewinner
- KONSEQUENZ: Kein ML-Modell fuer Trade-Selektion noetig oder nuetzlich

**Test 4 — Bootstrap CI:**
- Mean PnL aller Top-5: Signifikant positiv (P < 0.005)
- Median PnL: ~0 oder negativ (nicht signifikant)
- Edge ist rechtsschief — 94% Trades flat/negativ, 6% treffen 3R
- Setups #2, #3, #5 bestehen auch Bonferroni (288 Kombinationen)

### Zentrale Erkenntnis: GapDn vs GapUp Asymmetrie
- GapDn SL5 (alle Targets, N=21000): +0.0395 ADR
- GapUp SL5 (alle Targets, N=26472): -0.0088 ADR
- Differenz: +0.0483 ADR — deutlicher Richtungseffekt
- ABER: Bei Random Entries (N=675 GapDn, N=825 GapUp) sind beide positiv
  -> Kleine Samples, nicht belastbar. Groesserer Test noetig.

### Definitive OOS-Hypothese
- H1: "GapDn + Long + SL5_035 + T4_3R hat AvgPnL > 0 auf 2024-2026"
- Szenarien: GapDn_2s_Fade, GapDn_1s_Fade, GapDn_VWAP_Brk
- Signifikanz: Bootstrap P < 0.005, Bonferroni-korrigiert P < 0.0017

### 4 Tests VOR OOS-Validierung (Prioritaet):
1. **Richtungs-Experiment**: Nackte Long/Short am Open, kein SL/Target -> isoliert Richtungseffekt
2. **Shuffled-Returns**: Zeitliche Reihenfolge der 1-min Returns shufflen -> isoliert Autokorrelation
3. **GapUp Symmetrie (grosses N)**: Random GapUp-Short vs Random GapDn-Long, N>=1000
4. **Jahres-Stabilitaet**: 2021 vs 2022 vs 2023 separat -> Regime-Check

### Dateien
- Analyse: `results/quant_round2.txt`

---

## 2026-02-13 — Session 2: Devil's Advocate Runde 2

### SL/Target Artefakt-Hypothese: STARK VERDAECHTIG
- Random Entry mit SL5+T4_3R ist in BEIDE Richtungen positiv (GapDn: +0.0243, GapUp: +0.0253)
- Pipeline-Entry hat NULL Mehrwert (GapDn Pipeline=+0.0245 vs Random=+0.0243)
- Positive Expectancy kommt vermutlich aus der STRUKTUR von weitem SL + R-Multiple Target + Timeout
- ~50% der Trades sind Timeouts, deren PnL den Mean nach oben treibt
- KRITISCHER TEST: Random Walk Simulation mit SL5+T4_3R muss zeigen ob Edge rein mathematisch ist

### Median PnL = 0: Kein Trendfollowing-Edge
- Setups #2 und #3 (groesstes N) haben Median PnL = 0 oder negativ
- System ist ein "Tail-Harvester" — verliert meistens, gewinnt selten gross
- Unterschied zu echtem Trendfollowing: Kein klarer Mechanismus fuer positiven Mean
- Vermutung: Timeout-Trades auf Gappern haben positiven Drift (Mean Reversion nach Gap Down)

### Random-Entry Test: Implementierungsprobleme
- Random Entry Window nur Bar 20-200 (favorisiert Random-Arm wegen Morning-Bias)
- Pipeline-Arm verduennt durch alle SL/Target-Kombis (inkl. schlechte)
- N-Vergleich unfair: Random N=675 vs Pipeline N=3386 fuer GapDn
- Aber der FAIRE Subset-Vergleich zeigt trotzdem: Delta = +0.0002 ADR = KEIN Unterschied

### ML-Modell nach Leak-Fix: WERTLOS
- HC-Trades (>0.6 Ensemble) haben NEGATIVE PnL (-0.0150) nach e2_confirmed Fix
- WR steigt (69.4% -> 73.0%), aber ML selektiert die falschen Trades
- ML lernt NICHTS Nuetzliches ueber die Setup-Struktur hinaus

### Pflicht-Tests VOR Haelfte 2:
1. **Random Walk Simulation** — definitivster Nullhypothese-Test (SL5+T4_3R auf drift=0 Random Walks)
2. **Timeout-Zerlegung** — AvgPnL getrennt fuer Target-Hit, SL-Hit, Timeout
3. **Ausreisser-Sensitivitaet** — Mean PnL nach Entfernung der Top-5% Gewinner
4. **Gegen-Richtung** — Short auf GapDn Aktien (wenn auch positiv -> kein Richtungs-Edge)

### Brutale Einschaetzung
Stand jetzt gibt es KEINEN nachgewiesenen handelbaren Edge. Die positive Expectancy ist
sehr wahrscheinlich ein Artefakt von: weiter SL + R-Multiple Target + 4h Timeout +
natuerlicher Intraday-Drift volatiler Gapper-Aktien.

### Dateien
- Analyse: `results/devils_advocate_round2.txt`

---

## 2026-02-13 — Session 3 (Runde 3): OOS Validation Half 2

### ENTSCHEIDENDER OOS-TEST: Ergebnis

**Hypothese:** GapDn + Long + SL5_035 + T4_3R hat AvgPnL > 0 auf Half 2 (2024-2026)

**Daten:** 5,812 Gapper (2024: 1,922, 2025: 3,451, 2026: 439)
- GapUp: 3,224 (55.5%), GapDn: 2,588 (44.5%)
- 52,305 Trades simuliert (3 Entries x 3 Targets x alle Gapper)

### KERN-ERGEBNIS: EDGE NICHT BESTAETIGT

| Setup | Half-1 PnL | Half-2 PnL | Delta | Half-2 P(<=0) |
|-------|-----------|-----------|-------|---------------|
| GapDn + SL5 + T4_3R | +0.0228 | +0.0034 | -0.0194 | 0.2385 |
| GapDn + SL5 + T4_2R | +0.0158 | +0.0024 | -0.0134 | 0.2893 |
| GapDn + SL5 + T3_050 | +0.0080 | +0.0005 | -0.0075 | 0.4485 |
| **GapUp + SL5 + T4_3R** | **+0.0086** | **+0.0331** | **+0.0245** | **0.0000** |

- GapDn Long ist NICHT signifikant positiv (P=0.24, CI umschliesst 0)
- GapDn Long AvgPnL (+0.0034) liegt UNTER dem RW-Artefakt (+0.01)
- **INVERSION:** GapUp Short ist jetzt der staerkste Edge (+0.0331, P<0.001)!
- In Half 1 war GapUp Short bei -0.0006, jetzt bei +0.0331

### Richtungs-Asymmetrie HAT SICH UMGEKEHRT
- Half 1: GapDn Long (+0.0228) > GapUp Short (+0.0086)
- Half 2: GapUp Short (+0.0331) > GapDn Long (+0.0034)
- Die "GapDn ist besser als GapUp" These gilt NICHT auf neuen Daten

### Jahres-Instabilitaet (GapDn + SL5 + T4_3R)
- 2024: +0.0138 (schwach positiv, P=0.05)
- 2025: +0.0044 (marginal, nicht signifikant)
- 2026: -0.0582 (stark negativ!)
- Edge schrumpft ueber Zeit und wird 2026 negativ

### Naked Direction (Open-to-Close) auf Half 2
- GapDn Long: +0.0257 ADR (P=0.12, nicht signifikant)
- GapUp Short: +0.0117 ADR (P=0.38, nicht signifikant)
- Beide Richtungen zeigen schwachen positiven Drift, aber keiner signifikant

### GapUp Short: Der unerwartete Edge
- GapUp_1s_Fade Short: +0.0589 ADR (P<0.0001) — staerkster Szenario-Edge!
- GapUp_2s_Fade Short: +0.0382 ADR (P=0.0003)
- GapUp_below_VWAP Short: +0.0256 ADR (P<0.0001)
- Stabil ueber 2024 und 2025

### Entry-Zeitpunkt (GapDn)
- Bar 30 (Morning): +0.0217, P=0.01 — einziger signifikanter GapDn-Entry
- Bar 90 (Midday): -0.0001, P=0.51
- Bar 180 (Afternoon): -0.0115, P=0.94
- Nur Morning-Entry hat ueberhaupt noch positive Tendenz

### Timeout-Zerlegung
- GapDn SL%=41.7% (deutlich hoeher als Half-1 ~30-35%)
- Mehr SL-Hits = weniger Timeout-Gewinne = niedrigere Expectancy
- Timeout-PnL immer noch positiv (+0.1648), aber SL-Verluste fressen Edge auf

### FAZIT
1. Der GapDn-Long-Edge aus Half 1 repliziert NICHT auf Half 2
2. AvgPnL schrumpft von +0.0228 auf +0.0034 (85% Rueckgang)
3. Nicht signifikant, CI umschliesst 0, liegt unter RW-Baseline
4. Richtungs-Asymmetrie hat sich umgekehrt (GapUp jetzt besser)
5. Der Edge war vermutlich eine Kombination aus:
   - Strukturellem RW-Artefakt (+0.01)
   - Marktregime 2021-2023 (Post-COVID Retail-Mania, Bear Market Recovery)
   - Zufall/Noise bei moderatem N
6. GapUp Short zeigt auf Half 2 Staerke — aber das war NICHT die Hypothese
   und darf NICHT als "neuer Edge" deklariert werden (Hypothese-nach-Daten)

### Dateien
- Script: `scripts/oos_validation_half2.py`
- Ergebnisse: `results/oos_validation_half2.txt`

---

## 2026-02-13 — Durchlauf 2.0: Behavioral Analysis (KEIN SL/Target)

### Paradigmenwechsel
Runde 1 zeigte, dass ~26% des Edges ein SL/Target-Artefakt war und der GapDn-Edge OOS nicht replizierte.
Runde 2 eliminiert SL/Target komplett und analysiert NUR beobachtetes Verhalten nach VWAP-Level-Crosses.

### Neue Regeln
- Keine Signale vor 9:45 ET (erste 15min = Noise)
- Nur Haelfte 1 (2021-02-21 bis 2023-12-31)
- Slippage-Awareness: 0.005-0.01 ADR
- Metriken: MFE, MAE, Drift, Time-to-Move, Conditional Cascading, Reversal Rate

### Agent Team
- **QUANT**: Statistisches Framework, Sample Sizes, Bootstrap CIs
- **DISCRETIONARY TRADER**: Praktische Hypothesen (H1-H7)
- **DEVIL'S ADVOCATE**: Permutationstest, Multiples-Testen-Korrektur
- **CODER**: 3 Scripts geschrieben und ausgefuehrt
- **DOKUMENTATION**: Dieses Log

---

### Datensatz
- 9,590 Cross-Events aus 4,254 Gapper-Tagen
- Train: 6,811 Events (2021-02-21 bis 2022-12-31)
- Val: 2,779 Events (2023-01-01 bis 2023-12-31)
- 6 Szenarien: GapUp/GapDn x 2s_Fade / 1s_Fade / VWAP_Break
- Nur erster Cross pro Szenario pro Tag (kein Oversampling)

---

### KERN-ERGEBNIS 1: Drift nach Level-Cross (Val)

| Szenario | N | Drift30 (ADR) | P(<=0) | Signifikant? |
|----------|---|---------------|--------|-------------|
| GapDn_VWAP_Brk | 497 | **+0.0373** | 0.026 | Marginal |
| GapDn_2s_Fade | 289 | +0.0325 | 0.162 | Nein |
| GapDn_1s_Fade | 441 | -0.0193 | 0.824 | Nein (falsch) |
| GapUp_1s_Fade | 556 | -0.0044 | 0.586 | Nein |
| GapUp_2s_Fade | 382 | **-0.1196** | 0.986 | **Ja, NEGATIV** |
| GapUp_VWAP_Brk | 614 | **-0.0560** | 0.986 | **Ja, NEGATIV** |

**Interpretation:**
- GapDn Mean-Reversion existiert als schwaches Basisphänomen (Drift ~+0.03)
- GapUp Fading ist FALSCH — GapUp-Aktien setzen Momentum nach Level-Cross fort
- Richtungsasymmetrie bestätigt sich OHNE SL/Target-Artefakt
- Aber: GapDn overall ist NICHT signifikant (P=0.118 im Bootstrap)

---

### KERN-ERGEBNIS 2: Conditional Cascading (Val)

| Szenario | P(Next Level) | Median Time |
|----------|--------------|-------------|
| GapDn_2s_Fade | **91.0%** | 9 min |
| GapUp_2s_Fade | **92.4%** | 8 min |
| GapUp_1s_Fade | 76.6% | 11 min |
| GapDn_1s_Fade | 70.5% | 13 min |
| GapDn_VWAP_Brk | 69.8% | 11 min |
| GapUp_VWAP_Brk | 66.8% | 11 min |

**Interpretation:**
- 2s-Fades erreichen fast immer das naechste Level (91-92%)
- Aber das heisst nur, dass der Preis nach Verlassen von 2σ zurueck zu 1σ kommt — trivial
- VWAP-Breaks haben niedrigere Rate (~67-70%) — schwieriger, Level 1 zu durchbrechen

---

### KERN-ERGEBNIS 3: Reversal-Rate = ~80% überall

| Szenario | Reversal in 10 Bars | MFE15 wenn Reversal | MFE15 wenn Clean |
|----------|--------------------|--------------------|-----------------|
| GapDn_1s_Fade | 79.8% | 0.10 ADR | **0.38 ADR** |
| GapDn_2s_Fade | 80.6% | 0.12 ADR | **0.37 ADR** |
| GapDn_VWAP_Brk | 78.1% | 0.12 ADR | **0.35 ADR** |
| GapUp_2s_Fade | 79.8% | 0.12 ADR | **0.36 ADR** |

**Interpretation:**
- 80% aller Crosses erleben sofortige Gegenbewegung (>0.5% ADR)
- Die 20% "cleanen" Trades haben 3x hoehere MFE
- ABER: Kein Feature kann Clean von Reversal unterscheiden (ML R²=0.025)
- Einziger schwacher Hinweis: SPY-Return (+0.06 clean vs -0.08 reversal)

---

### KERN-ERGEBNIS 4: Trader-Hypothesen

| Hypothese | Ergebnis | Bewertung |
|-----------|----------|-----------|
| H1: Morning vs Late | Morning MFE=0.31, Midday Drift=+0.05, Afternoon schwach | **BESTÄTIGT** |
| H3: Volume bestaetigt Cross | High Vol SCHLECHTER (NextLvl 62% vs 79%) | **WIDERLEGT** |
| H4: Gap Size Sweet Spot | Small Gaps bester Drift, Extreme Gaps hoechstes MFE+MAE | **TEILWEISE** |
| H5: VWAP Slope | Flat VWAP: Drift +0.008, 80% NextLvl. Trending: -0.053 | **BESTÄTIGT** |
| H6: Sector Effects | Finance best (83% NextLvl), Services schwach | **SCHWACH** |
| H7: RVOL Threshold | Mid RVOL (1.5-3) bester Drift, Extreme RVOL chaotisch | **TEILWEISE** |

---

### KERN-ERGEBNIS 5: ML Interaktionsmodell

- XGBoost Regressor auf MFE30: Train R²=0.77, Val R²=0.07 → massives Overfitting
- XGBoost Regressor auf Drift30: Train R²=0.63, **Val R²=0.025** → WERTLOS
- XGBoost auf Drift60: Val R²=**-0.085** → SCHLECHTER als Mittelwert-Schaetzer
- **Die 18 Features koennen Drift NICHT vorhersagen**
- Top Feature Importance (MFE30): ADR (0.15), pct_adr_at_cross (0.13), early_range (0.11)
- Diese Features erklären Varianz im MFE (=wie volatil die Aktie ist), nicht Drift-Richtung

---

### KERN-ERGEBNIS 6: Interaktionseffekte (Conditional Analysis)

**Beste GapDn-Subgruppe (Val):**
- GapDn_2s_Fade + Early (9:45-10:15): Drift30 = +0.060, N=154
- GapDn_VWAP_Brk + Early: Drift30 = +0.044, N=325, NextLvl=82.5%
- LargeGap (>3 ADR) + Early: Drift30 = +0.099, MFE30 = 0.53 (!), N=211

**Starke Interaktion: >100% ADR used + Early = Drift +0.084**
- Aktien die schon viel bewegt haben und dann ein Level kreuzen = starke Reversion
- Intuition: Exhaustion-Signal

**SPY-Return Interaktion:**
- GapDn Long + SPY_Up: Drift = +0.031 (doppelt so gut wie SPY_Down = +0.002)
- GapUp Short + SPY_Down: Drift = +0.006 (vs SPY_Up = -0.085)
- Markt-Umfeld spielt eine Rolle

---

### KERN-ERGEBNIS 7: DEVIL'S ADVOCATE — Permutationstest

**Subgruppen-Selektion ist WERTLOS:**
- Permutationstest: P = 0.976 → beste Combo ist NICHT besser als zufaellige Cherry-Picking
- Perm-Mean = +0.143, real = +0.060 → zufaellig findet man BESSERE Combos!
- Train beste Combo ≠ Val beste Combo → INSTABIL

**GapDn Overall Drift: NICHT signifikant**
- Mean = +0.016 ADR, CI [-0.009, +0.042], P = 0.118
- Enthalt Null → kein gesicherter Richtungseffekt

**Einziger marginal signifikanter Fund:**
- GapDn_VWAP_Brk + Early: Drift = +0.044, P = 0.050 (an der Grenze)
- Ueberlebt 0.01 ADR Slippage: Drift_adj = +0.034

---

### SLIPPAGE-IMPACT (0.005 / 0.010 ADR)

| Szenario | Drift30_raw | Drift30_adj (0.010) | Überlebt? |
|----------|-------------|-------------------|-----------|
| GapDn_VWAP_Brk | +0.0373 | +0.0273 | JA |
| GapDn_2s_Fade | +0.0325 | +0.0225 | JA |
| GapDn_1s_Fade | -0.0193 | -0.0293 | NEIN |
| GapUp_2s_Fade | -0.1196 | -0.1296 | NEIN |

---

### ZUSAMMENFASSUNG DURCHLAUF 2.0

**Was funktioniert (schwach):**
1. GapDn_VWAP_Brk + Early Morning (9:45-10:15): Drift30 = +0.044, marginal signifikant
2. GapDn nach grossen Gaps (>3 ADR) + Early: starke Mean Reversion (MFE 0.53 ADR)
3. Flat VWAP = bessere Fade-Bedingungen als Trending VWAP
4. SPY_Up Tage = bessere GapDn Long Bedingungen

**Was NICHT funktioniert:**
1. ML kann Drift NICHT vorhersagen (R²=0.025)
2. Subgruppen-Selektion hat keinen Wert (Permutationstest P=0.98)
3. GapUp Fading ist NEGATIV (Drift -0.06 bis -0.12)
4. Hohes Volumen am Cross = SCHLECHTER, nicht besser
5. Reversal in 80% der Faelle → SL-Design ist Hauptproblem, nicht Entry
6. GapDn Overall Drift ist NICHT signifikant (P=0.118)

**Fazit Runde 2.0:**
Der Paradigmenwechsel von Trade-Simulation zu Verhaltens-Analyse bestaetigt die Runde-1-Ergebnisse:
Es gibt KEINEN robusten, statistisch gesicherten Edge in VWAP-Level-Crosses.
Es gibt schwache Hinweise auf GapDn Mean Reversion in den ersten 30min nach 9:45,
aber diese sind marginal signifikant und wuerdein bei Bonferroni-Korrektur durchfallen.
Die beste Erkenntnis: **Der Hauptgrund fuer Verluste ist die 80% Reversal-Rate.**
Jede zukuenftige Strategie muss primär das Reversal-Problem loesen.

### Dateien
- Scripts: `scripts/behavioral_analysis_v1.py`, `scripts/behavioral_ml_interactions_v1.py`, `scripts/behavioral_permutation_test.py`
- Ergebnisse: `results/behavioral_analysis_v1.txt`, `results/behavioral_ml_interactions_v1.txt`, `results/behavioral_permutation_test.txt`
- Events-Daten: `results/behavioral_events_v1.parquet` (9,590 Cross-Events)

---

## 2026-02-13 — Durchlauf 3.0: Offene Edge Discovery (OHLC-Daten)

### Paradigmenwechsel
Runde 1+2 zeigten: VWAP-Level-Crosses haben KEINEN Edge. Jetzt nutzen wir raw_1min OHLC-Daten
(data/raw_1min/) und suchen NEUE Anomalien: Opening Drive, ORB, Gap Fill, HOD/LOD Timing.

### Daten
- raw_1min hat FULL OHLC + pre-market (session='rth' fuer Regular Trading Hours)
- 4,254 Gapper-Tage verarbeitet (Train: 3,006, Val: 1,248)
- Metadata hat 63 Spalten inkl. pre-computed `opening_drive_direction`

---

### KERN-ERGEBNIS 1: Opening Drive Direction (Staerkster Praediktor)

**Metadata-Exploration (v3) entdeckte:** `opening_drive_direction` hat 0.548 Korrelation mit CloseVsOpen.

**ORB + Opening Drive v1 Analyse (OHLC-basiert):**

| Gruppe | N (Val) | Drift60 | Drift240 | P(<=0) | MFE60 |
|--------|---------|---------|----------|--------|-------|
| GapUp+OD_WITH | 331 | +0.1440 | +0.2073 | **0.0006** | 0.694 |
| GapUp+OD_AGAINST | 357 | -0.0292 | -0.0399 | 0.7996 | 0.376 |
| GapDn+OD_AGAINST | 277 | +0.0273 | +0.0603 | 0.1452 | 0.442 |
| GapDn+OD_WITH | 283 | -0.0027 | +0.0326 | 0.3238 | 0.458 |

**Spread GapUp:** WITH vs AGAINST = 0.2073 - (-0.0399) = **+0.247 ADR** (Val)

---

### KERN-ERGEBNIS 2: ORB Alignment mit Opening Drive

| ORB aligns OD | N (Val) | ORB_MFE | ORB_Drift60 |
|---------------|---------|---------|------------|
| YES | 769 | 0.827 | +0.042 |
| NO | 246 | 0.592 | -0.077 |

ORB aligned mit OD = 40% hoehere MFE und positive Drift.

---

### KERN-ERGEBNIS 3: Combined Filters (beste Combos, Val)

| Gap | OD | SPY | ORB | N | Drift60 | Drift240 | MFE60 |
|-----|----|-----|-----|---|---------|----------|-------|
| up | WITH | SPY_Up | ORB_align | 152 | +0.330 | +0.472 | 1.008 |
| up | WITH | AnySPY | ORB_align | 224 | +0.317 | +0.343 | 0.941 |
| down | WITH | SPY_Dn | ORB_align | 98 | +0.239 | +0.354 | 0.700 |

**OD Staerke:** VStrong (>0.6 ADR) GapUp+WITH: Drift60=+0.332, MFE60=1.239

---

### DEVIL'S ADVOCATE REVIEW (10 Tests)

**Tests die BESTANDEN:**
1. **Permutation Test GapUp**: P=0.001 — OD Labels tragen echte Information
2. **Bonferroni-Korrektur**: P=0.023 (38 Tests) — ueberlebt
3. **Random Direction Baseline**: OD-guided > random (Z=2.30, P=0.02)
4. **OD Magnitude**: Korrelation r=0.30 (staerker OD = staerkere Continuation)
5. **Sign Convention**: Korrekt (LONG bei OD_up, SHORT bei OD_down)

**KRITISCHE BEFUNDE:**

1. **JAHRES-INSTABILITAET (GROESSTER CONCERN):**
   - GapUp+WITH 2021: Drift240 = **-0.010** (NEGATIV!)
   - GapUp+WITH 2022: Drift240 = **-0.023** (NEGATIV!)
   - GapUp+WITH 2023: Drift240 = **+0.207** (POSITIV!)
   - Das Signal war IN TRAINING NEGATIV und wurde NUR 2023 positiv.
   - Train-D240 = -0.0166, Val-D240 = +0.2073 → **Signal hat sich umgekehrt!**

2. **OUTLIER-GETRIEBEN:**
   - drift_60m: Mean=+0.14, **Median=-0.005** (Coin Flip!)
   - Win Rate 60m: **49.5%**
   - Mean ohne Top 5%: +0.004 (near zero)
   - Mean ohne Top 10%: **-0.051** (NEGATIV!)
   - Edge existiert NUR dank weniger grosser Gewinner

3. **Gap Direction ADDIERT Info:**
   - OD=up + GapUp: Drift240=+0.207 vs OD=up + GapDn: Drift240=+0.060
   - Es ist NICHT nur Momentum — Gap-Richtung verstaerkt/daempft den Effekt

---

### KERN-ERGEBNIS 4: Gap Fill Analyse

- Nur **20-25%** der Gaps werden intraday gefuellt
- Kleinere Gaps (4-6%): ~25-30% Fill Rate, Median 31-60min
- Grosse Gaps (>20%): <5% Fill Rate
- **Post-Fill Drift: NICHT signifikant** (P=0.845/0.629)
- Gap Fill ist KEIN tradbares Event

---

### KERN-ERGEBNIS 5: HOD/LOD Timing

- **38-43% der HOD/LOD werden in den ersten 15 Minuten gesetzt!**
- Das ist die haeufigste Periode fuer Tageshoch/-tief
- GapUp+OD_up: Wenn LOD waehrend OD → RoD=+0.666 (stark bullish)
- GapUp+OD_up: Wenn HOD waehrend OD → RoD=-0.516 (stark bearish)
- **Close Location:** OD_up Aktien schliessen bei ~0.57 (leicht ueber Mitte), OD_down bei ~0.43
- Stabil Train→Val

---

### ZUSAMMENFASSUNG DURCHLAUF 3.0

**Was funktioniert (mit Vorbehalten):**
1. Opening Drive Direction hat echte Vorhersagekraft (Permutation P=0.001)
2. GapUp + OD_WITH zeigt staerkste Continuation (Drift240=+0.21, P=0.0006)
3. ORB Alignment mit OD verstaerkt das Signal (MFE +40%)
4. OD Staerke korreliert mit Drift (r=0.30)
5. HOD/LOD Timing: 38-43% der Extreme in ersten 15min

**Was NICHT funktioniert:**
1. Gap Fill ist nicht tradebar (Post-Fill Drift = 0)
2. GapDn OD-Effekte sind NICHT signifikant
3. Jahres-Stabilisierung fehlt — Signal war 2021/2022 NEGATIV

**KRITISCHER VORBEHALT:**
Das GapUp+WITH Signal ist **temporaer instabil**. Es war in den Trainingsdaten (2021-2022)
NEGATIV und wurde erst 2023 stark positiv. Dies koennte ein 2023-spezifisches Regime sein
(Post-Bear-Market Recovery, AI-Boom). Die OOS-Validierung auf Haelfte 2 (2024-2026) wird
zeigen ob es ein dauerhafter Effekt oder Zufall ist.

**EMPFEHLUNG:** Trotz Vorbehalten OOS testen, da:
- Permutation Test signifikant
- Bonferroni ueberlebt
- Die ALLGEMEINE These "OD predicts rest of day" ist fundamental plausibel
- Aber mit KLARER Erwartung: Signal koennte in 2024-2026 verschwinden

### Dateien
- Scripts: `scripts/metadata_exploration_v3.py`, `scripts/orb_opening_drive_v1.py`, `scripts/devils_advocate_od_v1.py`, `scripts/gap_fill_hod_lod_v1.py`
- Ergebnisse: `results/metadata_exploration_v3.txt`, `results/orb_opening_drive_v1.txt`, `results/devils_advocate_od_v1.txt`, `results/gap_fill_hod_lod_v1.txt`
- Events-Daten: `results/orb_opening_drive_events_v1.parquet`, `results/gap_fill_hod_lod_events_v1.parquet`

---

## 2026-02-13 — Durchlauf 3.0: OOS VALIDATION auf Haelfte 2 (2024-2026)

### Datensatz
- 5,812 Gapper auf Half 2 (2024: 1,922, 2025: 3,451, 2026: 439)
- GapUp: 3,224, GapDn: 2,588

### ERGEBNIS: GEMISCHT — Hauptsignal schrumpft, ORB-Alignment repliziert

---

### H1: GapUp+OD_WITH Drift240 > 0 → SCHWACH, NICHT SIGNIFIKANT

| Gruppe | N | Drift60 | Drift240 | MFE60 | P(<=0) |
|--------|---|---------|----------|-------|--------|
| GapUp+OD_WITH | 1461 | +0.022 | **+0.105** | 0.444 | **0.098** |
| GapUp+OD_AGAINST | 1763 | +0.019 | +0.042 | 0.373 | 0.014 |
| GapDn+OD_AGAINST | 1299 | +0.048 | +0.034 | 0.384 | 0.047 |
| GapDn+OD_WITH | 1289 | -0.031 | **-0.037** | 0.337 | 0.952 |

- GapUp+OD_WITH: IS_Val=+0.207 → OOS=+0.105 (**~50% Shrinkage**), P=0.098
- Bootstrap CI: [-0.022, +0.317] → **enthalt Null**
- WR=48.1%, Median=-0.032 → noch staerker outlier-getrieben als IS

---

### H2: OD Spread (WITH vs AGAINST) → GESCHEITERT

- GapUp Spread: IS=+0.247 → OOS=**+0.063**, Perm P=**0.359** (NOT significant)
- **Overall WITH vs AGAINST: +0.038 vs +0.038 = Spread 0.0001 (KEIN Unterschied!)**
- Die Opening Drive Richtung (WITH/AGAINST Gap) ist OOS WERTLOS als Filter

---

### H3: ORB Alignment → **REPLIZIERT! STAERKSTES SIGNAL!**

| ORB Alignment | N | Drift60 | MFE60 |
|--------------|---|---------|-------|
| **Aligned mit OD** | 3607 | **+0.199** | **0.534** |
| **Opposing OD** | 1108 | **-0.405** | **0.090** |

- **Spread: +0.604 ADR** zwischen aligned und opposing
- Dies ist das staerkste und stabilste Signal im gesamten Datensatz
- ORB aligned mit OD = positive Continuation
- ORB opposing OD = starke Gegenbewegung (MFE nur 0.09!)
- **ABER:** Dies ist ein NACH-9:45-Signal (ORB Break muss erst passieren)

---

### UEBERRASCHUNGEN:

1. **GapUp+OD_AGAINST ist jetzt POSITIV** (OOS +0.042, IS -0.040) → Sign Reversal!
2. **GapDn+OD_WITH VERLIERT** (OOS -0.037) → Continuation nach GapDn-Opening-Drive = Verlust
3. **GapDn+OD_AGAINST = signifikant positiv** (Drift60=+0.048, P=0.0004)
   Reversal gegen OD-Richtung bei GapDn funktioniert OOS!

---

### Jahres-Stabilitaet (GapUp+OD_WITH):
- 2024: Drift240=+0.039 (schwach)
- 2025: Drift240=+0.139 (moderat)
- 2026: Drift240=+0.119 (moderat, aber nur N=118)
- Signal ist 2024-2026 konsistenter als 2021-2023 (wo es negativ war!)

---

### VERGLEICH IN-SAMPLE vs OOS

| Gruppe | IS_Val_D240 | OOS_D240 | Repliziert? |
|--------|-------------|----------|-------------|
| GapUp+OD_WITH | +0.207 | +0.105 | TEILWEISE (geschrumpft) |
| GapUp+OD_AGAINST | -0.040 | +0.042 | NEIN (Vorzeichen gewechselt!) |
| GapDn+OD_WITH | +0.033 | -0.037 | NEIN (Vorzeichen gewechselt!) |
| GapDn+OD_AGAINST | +0.060 | +0.034 | JA (geschrumpft aber gleiche Richtung) |

---

### GESAMTFAZIT DURCHLAUF 3.0

**Was repliziert OOS:**
1. **ORB Alignment mit Opening Drive** → Staerkstes Signal (+0.60 ADR Spread)
2. GapUp+OD_WITH zeigt positive Tendenz (+0.10 ADR) aber nicht signifikant
3. GapDn+OD_AGAINST (Reversal) zeigt signifikanten Drift60 (+0.048, P=0.0004)

**Was NICHT repliziert:**
1. Die WITH/AGAINST Spread als Filter → OOS = 0 (komplett wertlos)
2. GapDn+OD_WITH → Vorzeichen gewechselt (jetzt negativ!)
3. GapUp+OD_AGAINST → Vorzeichen gewechselt (jetzt positiv!)
4. Die 0.2+ ADR Drifts aus 2023 → massiv geschrumpft

**Muster ueber alle Durchlaeufe:**
In JEDEM Durchlauf (1, 2, 3) wiederholt sich dasselbe Muster:
- In-Sample zeigt vielversprechende Edges
- OOS schrumpft der Effekt um 50-80% oder wechselt das Vorzeichen
- VWAP-Crosses (Runde 1+2), GapDn Mean Reversion (Runde 1+2), OD Direction (Runde 3) — alle fallen OOS
- **Einziges konsistentes Signal: ORB Alignment mit OD** (aber erst nach 9:45 bekannt)

**Fuer eine handelbare Strategie:**
- ORB Break in OD-Richtung (Entry nach 9:45, wenn ORB ausbricht)
- Massive Asymmetrie: Aligned MFE=0.53 vs Opposing MFE=0.09
- Problem: Wie fruh erkennt man ob der Break "aligned" ist? Der OD ist nach 15min bekannt,
  der ORB-Break passiert danach → realtime umsetzbar!
- Slippage-robust: Drift60=+0.20 ueberlebt 0.01 ADR Slippage locker

### Dateien (OOS)
- Script: `scripts/oos_opening_drive_half2.py`
- Ergebnisse: `results/oos_opening_drive_half2.txt`
- Events: `results/oos_opening_drive_half2_events.parquet`

---

## 2026-02-13 — Durchlauf 3.5: ORB Alignment Deep Dive

### Hypothese
User's These: "Der ORB-Break in seine Richtung IST der Trade. Nicht der OD."
Opposing Trades (OD vs ORB disagree) hatten im Durchlauf 3.0 OOS -0.405 ADR in OD-Richtung.
Die Erwartung war: Das sind +0.405 in ORB-Richtung.

### KERN-ERGEBNIS: Hypothese WIDERLEGT (teilweise)

**Die Opposing-Trades VERLIEREN auch in ORB-Richtung!**

| Gruppe | N | d60 (ORB-dir) | d240 (ORB-dir) | P(<=0) d240 |
|--------|---|---------------|----------------|-------------|
| Aligned IS | 2730 | +0.023 | **+0.047** | **0.001** |
| Opposing IS | 885 | **-0.028** | **-0.046** | 0.963 |
| Aligned OOS | 3607 | **+0.022** | **+0.062** | **0.008** |
| Opposing OOS | 1108 | +0.017 | +0.025 | 0.124 |

**Warum die Diskrepanz zum Durchlauf 3.0 (+0.40 ADR "Edge")?**
Der +0.40 ADR Spread war gemessen vom **OD Close** (Bar 15), nicht vom ORB Break-Preis.
Wenn der ORB gegen den OD ausbricht, hat der Preis sich bereits erheblich bewegt.
Ab dem tatsaechlichen ORB-Break-Preis gibt es KEINE positive Drift mehr fuer Opposing.
→ Das +0.40 Signal war ein Mess-Artefakt (Einstieg nach grosser Bewegung = positive Drift trivial).

**ABER: Aligned Trades replizieren!**
- IS d240 = +0.047 (P=0.001) → OOS d240 = **+0.062 (P=0.008)** — sogar STAERKER OOS!
- Dies ist das ERSTE Signal im gesamten Projekt, das OOS nicht schrumpft.
- CI OOS: [+0.006, +0.147] → schliesst Null AUS

---

### TASK 1: Parameter Breakdowns (Aligned, IS)

**Stärkste Buckets für Aligned ORB Breaks:**

| Parameter | Bester Bucket | N | d60 | d240 | Besonderheit |
|-----------|---------------|---|-----|------|-------------|
| ADR% | 3-5% | 735 | +0.061 | +0.059 | Höchster Mean |
| RVOL | 3-5x | 450 | +0.057 | +0.063 | Sweet Spot |
| Gap Size | 2-3x ADR | 280 | +0.051 | **+0.114** | Stärkster Drift |
| Gap Dir | GapUp | 1450 | +0.036 | +0.066 | Besser als GapDn |
| VWAP Pos | <-1σ | 1414 | +0.016 | +0.037 | Bessere WR (54%) |
| BandWidth | >0.4 | 979 | +0.044 | +0.077 | Aber volatil |
| %ADR Used | >75% | 1352 | +0.034 | +0.062 | "Exhaustion" |
| Sector | Retail | 205 | **+0.089** | **+0.122** | Bester Sektor! |
| OD Strength | >0.6 ADR | 734 | +0.049 | +0.084 | Aber volatil |

**Opposing: Starker OD = Katastrophe!**
- OD Strength >0.6: d60 = **-0.313**, WR = **37.9%** → Niemals ORB gegen starken OD traden!
- BandWidth >0.4: d60 = -0.123, d240 = -0.215

---

### TASK 2: MAE Distribution (Aligned, ORB-Richtung)

| Window | Losers P50 | Losers P75 | Winners P50 | Winners P75 |
|--------|-----------|-----------|-------------|-------------|
| 5min | 0.130 | 0.232 | 0.109 | 0.175 |
| 15min | 0.231 | 0.402 | 0.139 | 0.231 |
| 30min | 0.333 | 0.553 | 0.152 | 0.256 |
| 60min | 0.466 | 0.718 | 0.163 | 0.267 |

**Erkenntnis für SL-Design:**
- Winners ziehen maximal ~0.27 ADR gegen sich (P75 bei 60min)
- Losers ziehen 0.72 ADR gegen sich (P75 bei 60min)
- Ein SL bei 0.30 ADR wuerde ~75% der Winner-Drawdowns ueberleben
- Aber selbst bei 0.30 SL werden 50% der Loser erst bei 0.47+ gestoppt

---

### TASK 3: SL x Target Matrix

**IS Aligned (N=2730) — Beste Kombos:**

| SL | TGT | WR | AvgPnL | Bemerkung |
|----|-----|------|--------|-----------|
| 0.50 | 1.0 | 45% | **+0.038** | Bester Mean |
| 0.30 | 1.0 | 35% | +0.031 | Engerer SL |
| 0.50 | 0.5 | 53% | +0.027 | Hoehere WR |
| 0.50 | 0.3 | 64% | +0.024 | Noch hoehere WR |

**OOS Validation (SL=0.50/TGT=1.0):**

| Gruppe | IS | OOS | P(<=0) | vs Random |
|--------|-----|-----|--------|-----------|
| Aligned | +0.038 | **+0.018** | 0.027 | +0.033 vs -0.015 |
| Opposing | -0.013 | **+0.031** | 0.029 | — |
| ALL | +0.025 | **+0.021** | **0.006** | +0.033 vs -0.014 |

- ALL ORB SL=0.50/TGT=1.0: **OOS Avg = +0.021, P = 0.006**
- Beats random by +0.033 ADR
- Ueberraschend: Opposing ist OOS BESSER als Aligned im SL/Target Framework!

**OOS Full Matrix (Aligned):**
- Positive Expectancy ab SL >= 0.30
- SL=0.30/TGT=1.0: +0.008 (knapp positiv)
- SL=0.50/TGT=1.0: +0.018 (robustester Punkt)

---

### TASK 5: Random Baseline

**DELTA (ORB-directed minus Random-directed):**

| SL | T0.1 | T0.2 | T0.3 | T0.5 | T1.0 |
|----|------|------|------|------|------|
| 0.10 | -0.012 | -0.006 | -0.001 | +0.002 | +0.003 |
| 0.20 | -0.007 | +0.005 | +0.014 | +0.017 | +0.019 |
| 0.30 | -0.003 | +0.013 | +0.023 | +0.026 | +0.028 |
| 0.50 | +0.002 | +0.018 | +0.027 | **+0.030** | **+0.033** |

**Erkenntnis:** ORB-Richtung schlaegt Random erst ab groesseren SL+TGT Combos.
Bei engem SL (<0.15) ist die Richtung wertlos — die Reversal-Rate frisst den Edge.
Bei weitem SL (0.30-0.50) + weitem Target (0.50-1.0) schlaegt ORB Random um +0.03 ADR.

---

### OOS Jahres-Stabilitaet

**Aligned (d60 ORB-Richtung):**
- 2024: N=1244, d60=+0.005, d240=+0.013
- 2025: N=2117, d60=+0.028, d240=+0.093
- 2026: N=246, d60=+0.057, d240=+0.043

**Opposing (d60 ORB-Richtung):**
- 2024: N=353, d60=+0.051, d240=+0.101
- 2025: N=653, d60=+0.006, d240=-0.006
- 2026: N=102, d60=-0.035, d240=-0.038

**Aligned ist stabil (alle Jahre positiv), Opposing ist instabil (2026 negativ).**

---

### ZUSAMMENFASSUNG DURCHLAUF 3.5

**Bestaetigte Erkenntnisse:**
1. **Aligned ORB Breaks haben einen echten, kleinen Edge** — IS +0.047 d240 → OOS +0.062 d240 (P=0.008)
2. **Dieser Edge ueberlebt Slippage**: +0.062 ADR >> 0.01 ADR Slippage
3. **Random Baseline geschlagen**: ORB Richtung > Random um +0.033 bei SL=0.50/TGT=1.0
4. **Jahres-stabil fuer Aligned**: Positiv in 2024, 2025, 2026
5. **SL/Target repliziert**: OOS +0.021 AvgPnL bei SL=0.50/TGT=1.0 (P=0.006)

**Widerlegte Hypothesen:**
1. **Opposing = +0.40 in ORB direction** → FALSCH. Opposing verliert auch in ORB-Richtung (-0.046 IS)
2. **ORB Break IST der Trade (unabhaengig von OD)** → FALSCH. Nur ALIGNED Breaks funktionieren
3. **OD ist nur ein Filter** → FALSCH. OD-Alignment ist NOTWENDIG, nicht optional

**Handel bare Strategie-Parameter:**
- Entry: ORB Break in OD-Richtung (nach 9:45, wenn 15min-Range bricht in OD direction)
- SL: 0.50 ADR (weiter Stop noetig wegen 80% Reversal-Rate)
- Target: 1.0 ADR (maximiert Expectancy)
- WR: ~45% (kein High-WR System)
- Avg PnL: +0.02 ADR pro Trade (nach Slippage: +0.01 ADR)
- Erwartete Trades/Tag: ~3-5 pro Gapper-Tag

**Edge-Groesse realistisch einschaetzen:**
- +0.02 ADR pro Trade klingt klein, IST klein
- Bei $50 Aktie mit 5% ADR ($2.50): +0.02 × $2.50 = **$0.05 pro Aktie pro Trade**
- Bei 100 Shares: **$5 pro Trade** vor Kommissionen
- Das reicht NICHT fuer profitable Einzeltrades — nur im Aggregat ueber viele Trades
- Commission + Slippage koennten den Edge komplett auffressen

### Dateien
- Script: `scripts/orb_deep_dive_v1.py`
- Ergebnisse: `results/orb_deep_dive_v1.txt`

---

## 2026-02-13 -- Durchlauf 4.0: Conditional Probability Cheat Sheet

### Auftrag
Statistisches Nachschlagewerk fuer diskretionaeren Gapper-Trader.
KEINE SL/Target-Optimierung, KEIN ML. Bedingte Wahrscheinlichkeiten.
7 Kategorien: VWAP-Regime, Level-Respekt, Gap-Kontext, Zeitstruktur,
Volume-Verhalten, Multi-Day, Markt-Kontext.

### TOP-ERKENNTNISSE (sortiert nach Staerke)

**1. SPY-Kontext ist das staerkste Signal**
- GapUp + SPY <-1%: Drift -0.63 ADR, CL=0.21 (FADING GOLDGRUBE)
- GapDown + SPY >+1%: Drift +0.53 ADR, CL=0.79 (BOUNCE GOLDGRUBE)
- Markt-Kontext uebertrumpft alle anderen Signale!

**2. RVOL bestimmt den Tagestyp**
- RVOL >5x: Momentum-Tag (GapUp bleibt oben, GapDown bleibt unten, Fill <11%)
- RVOL <1x: Fade-Tag (Fill 36-45%, "nobody cares")
- DayRange steigt linear: 0.9 ADR bei <1x -> 2.0 ADR bei >5x RVOL

**3. Opening Drive setzt Tagesrichtung in ~68%**
- OD mit Gap: 68% schliessen mit Gap
- OD gegen Gap: 70% schliessen gegen Gap
- Erste 5 Minuten bereits ~62% praediktiv

**4. Vorherige Performance = Fade/Bounce Signal**
- GapUp nach >20% 10d-Run: Fill 43%, Drift -0.21 ADR (STARKES Fading)
- GapDown bei RSI <30: Fill 42%, Drift +0.42 ADR (STAERKSTES Bounce)

**5. Gap-Groesse bestimmt Fill-Chance**
- <1 ADR: ~40% Fill (Fade plausibel)
- >2 ADR: <5% Fill (fast NIE gefuellt)

**6. VWAP-Levels: Richtung ja, Stop nein**
- Rejections halten nur ~25% fuer 30min (NICHT als Trade-Level geeignet)
- Sigma-1 wird in ~92-96% erreicht wenn VWAP als Barriere steht
- VWAP bricht fast immer sofort (Median 4min), Retest ist 50/50

**7. Kein Multi-Day Follow-Through**
- Next-Day Return ~50/50 unabhaengig von Gap-Tag Close
- Gapper-Tage sind Einzel-Events

**8. Multi-Touch staerkt Levels kaum**
- Break-Rate sinkt von 61% (Touch #1) auf 57% (Touch #5)
- Effekt ist real aber klein (~4pp ueber 5 Touches)

### Was NICHT funktioniert
- VWAP-Level als Trade-Entry/Stop (nur 25% Hold-Rate)
- Multi-Day Follow-Through (50/50)
- Lunch-Zone Breakouts (50/50 Follow-Through)
- Level-Rejections als alleiniges Signal

### Dateien
- Scripts: scripts/cheat_cat*.py
- Ergebnisse: results/cheat_cat*.txt
- Master Cheat Sheet: results/cheat_sheet_v1.txt
- Rohdaten: results/cheat_cat*_raw.parquet, results/cheat_cat2_touches.parquet

---

## 2026-02-13 — Durchlauf 4.5: Cheat Sheet v2 (Trader-Vertiefung)

### Auftrag
Vertiefung des Cheat Sheet v1 mit 6 trader-spezifischen Aufgaben.
Trading-Fenster: 9:30-11:00 ET. Nur Haelfte 1 (4254 Gapper).
Fokus auf SPY-Interaktionen, Faktor-Matrix, Opening Drive MFE, VWAP Rejection, Premarket-Volumen.

### Bug-Fixes waehrend Analyse
1. **close_location ist STRING**: Metadata-Spalte `close_location` enthaelt Strings (`above_vwap`, `below_vwap`), nicht Zahlen. Fix: CL = (rth_close - rth_low) / (rth_high - rth_low). Betraf Aufgaben 1, 2, 5, 6.
2. **VWAP-Spalten-Kollision in Aufgabe 4**: raw_1min UND vwap-Dateien haben beide eine `vwap`-Spalte. Fix: VWAP-Datei's `vwap` zu `running_vwap` umbenannt vor Merge.
3. **prev_close_location fehlt**: Metadata hat kein `prev_close_location`, daher ist Aufgabe 6.3 leer.

---

### AUFGABE 1: SPY-Kontext mit Parameter-Filtern

**Datenproblem:** SPY-Daten sind bimodal — fast keine Gapper bei SPY neutral (leicht_rot/leicht_gruen: N<10). Nur stark_rot (<-1%) und stark_gruen (>+1%) haben genug Daten.

**Kern-Erkenntnisse:**
- Staerkstes Signal: GapDown + stark_gruen + RVOL<1x = **+0.240 Drift** (N=190)
  → Starker Markt + wenig Aufregung = bester Bounce fuer GapDown
- SPY-Spread waechst mit Gap-Groesse bei GapDown:
  - <1 ADR: Spread=+0.088
  - 1-2 ADR: Spread=+0.290
  - 2-3 ADR: Spread=+0.337
- GapUp + stark_rot + RVOL<1x = **-0.260 Drift** (N=129) → schwaechster Wert
- RVOL 3-5x neutralisiert SPY-Effekt bei GapUp (Spread=-0.010)
- Bei GapDown + stark_rot + RVOL 3-5x: **-0.306 Drift** (staerkste Kapitulation)

**Trader-Takeaway:** SPY-Kontext ist umso wichtiger je groesser der Gap. Bei RVOL 3-5x ueberwiegt Volumen-Effekt den SPY-Kontext.

---

### AUFGABE 2: Interaktions-Matrix (Gap x RVOL x Prior 10d Return)

**54 Zellen (3 Gap-Groessen x 3 RVOL x 3 Prior-Return x 2 Richtungen)**

**Top Drift-Zellen (mit ausreichend N):**
- GapDown + 1-2 ADR + RVOL 2-5x + Prior<-10%: N=130, Drift=-0.266 → **Kapitulation bounced NICHT**
- GapDown + >2 ADR + RVOL 2-5x + Prior<-10%: N=87, Drift=-0.268 → **Doppelte Kapitulation noch schlimmer**
- GapDown + >2 ADR + RVOL >5x + Prior<-10%: N=163, Drift=-0.262 → **Massive Kapitulation setzt sich fort**
- GapUp + 1-2 ADR + RVOL >5x: N=65, Drift=+0.32 → Starkes Momentum (aber LOW N pro Prior-Bucket)

**Marginale Effekte (ueber alle Dimensionen):**
- GapDown RVOL<2x: **+0.065 Drift** (signifikant, CI: +0.033 to +0.100) → Einziger signifikanter Faktor
- GapDown RVOL 2-5x: **-0.093 Drift** → Mittleres RVOL ist das schlechteste
- Gap-Groesse hat bei GapUp KEINEN marginalen Effekt (alles um 0)

**Hypothesen-Check:**
- H1 (Kleiner Gap + hohes RVOL = Momentum): GapUp +0.125 Drift → TEILWEISE BESTAETIGT
- H2 (Grosser Gap + hohes RVOL nach Rally = Fade): -0.061 → SCHWACH
- H3 (Kapitulation bounced): -0.264 → **WIDERLEGT** (Kapitulation setzt sich fort!)

**Trader-Takeaway:** Kapitulation (GapDown + Prior<-10% + RVOL 2-5x/5x) bounced NICHT — die Aktie faellt weiter. Einziger zuverlaessiger Faktor: Niedriges RVOL bei GapDown = leichte Mean Reversion.

---

### AUFGABE 3: Opening Drive MFE vor SL-Hit

**SL = 5min High/Low (OD-Range). Trade in OD-Richtung ab 9:35.**

**Gesamt: SL-Hit=50.3%, Median SL=0.37 ADR, MFE=0.50 ADR, R:R=1.04**

**Nach OD-Richtung:**
- GapDown + OD with_gap: WR@0.50=**36.8%** (bester Wert), R:R=1.05
- GapUp + OD with_gap: WR@0.50=35.4%, R:R=1.07

**OD-Staerke ist entscheidend:**
- Strength >0.5 ADR: SL-Hit **39-47%** (deutlich besser)
  - GapDown + OD with + Strong: SL-Hit=39.5%, MFE=0.677, WR@0.50=51.0%, WR@1.00=20.4%
  - GapDown + OD against + Strong: SL-Hit=36.9%, MFE=0.625, WR@0.50=45.7%
- Strength <0.2 ADR: SL-Hit **77-81%** → Fast immer gestoppt!

**SL-Hit Timing:** 17.8% in ersten 5 Minuten, 50% innerhalb 15 Minuten. Wer ueberlebt, hat gute Chancen.

**3-Bar Alternative:** Alt 3-Bar gegen Gap hat niedrigere SL-Hit (36-39%) als Standard-OD.

**Trader-Takeaway:**
1. OD-Staerke >0.5 ADR = Signal (SL-Hit sinkt auf ~40%, WR@0.50 steigt auf ~50%)
2. Schwacher OD (<0.2 ADR) = kein Trade (80% SL-Hit)
3. Die ersten 5 Minuten sind kritisch — wenn SL nicht fruh getroffen, gute Chancen
4. GapDown + OD with_gap + Strong = bester Trade: WR@0.50=51%, MFE=0.68 ADR

---

### AUFGABE 4: VWAP-Ablehnung nach 9:50

**ERGEBNIS: METRIK STRUKTURELL FEHLERHAFT — KEINE BRAUCHBAREN ERKENNTNISSE**

**Problem:** Die Metrik vergleicht asymmetrische Schwellen:
- "Sigma-1 erreicht" = Preis bewegt sich 1 StdDev in Gap-Richtung ab VWAP → LEICHT (oft schon bei Rejection-Close erfuellt)
- "Rejection gebrochen" = Preis uebersteigt das High/Low des Rejection-Bars → SCHWER (erfordert Rueckkehr ueber volle Bar-Range)

**Resultat:** 0% gebrochen / 62% Sigma-1 / 1 min Median Zeit → Daten sind korrekt, aber die Metrik misst nichts Nuetzliches.

**Was die Daten trotzdem zeigen (deskriptiv):**
- 1404 Rejections in 4254 Tagen (33% der Gapper haben mindestens eine Rejection nach 9:50)
- Fruehere Rejections (9:50-10:00): hoehere Sigma-1-Rate (69%)
- Spaetere Rejections (10:30-11:00): niedrigere Rate (33-40%)
- VWAP nie gebrochen + Rejection: 71-74% Sigma-1 vs vorher gebrochen: 55-57%
- Enge Bandbreite (<0.2): 73-78% Sigma-1 vs Weite (>0.4): 48-51%
- GapUp > GapDown (>2 ADR): hoehere Sigma-1-Rate und breitere Bandbreite

**Trader-Takeaway:** Fuer brauchbare VWAP-Rejection-Analyse muesste man "gebrochen" als Close-Cross ueber VWAP definieren (nicht als Extreme-ueber-Bar). Die jetzigen Daten zeigen nur: je frueher und enger die Baender, desto eher bewegt sich der Preis in Gap-Richtung nach Rejection.

---

### AUFGABE 5: Premarket-Volumen als Signal

**PM/RTH30 Ratio = Staerkstes neues Signal im Durchlauf 4.5!**

| PM/RTH Ratio | GapUp Drift | GapDown Drift | Interpretation |
|--------------|------------|---------------|----------------|
| <10% | **+0.312** | **-0.309** | Continuation (wenig PM = wenig Preisdiscovery vor Open) |
| 10-30% | -0.030 | +0.006 | Neutral |
| >30% | **-0.244** | **+0.180** | Reversal (viel PM = Preisdiscovery bereits passiert) |

**Spread GapUp:** +0.312 - (-0.244) = **+0.556 ADR** (groesster Spread aller Aufgaben!)
**Spread GapDown:** -0.309 - (+0.180) = **-0.489 ADR**

**Interpretation:**
- Niedriges PM/RTH Ratio (<10%) → Markt hat Gap noch NICHT "verdaut" → Continuation in Gap-Richtung
- Hohes PM/RTH Ratio (>30%) → Preisdiscovery bereits abgeschlossen → Fade/Reversal

**PM-Volumen Quartile (absolut):** Schwaecher als Ratio. GapUp Top25% hat -0.117 Drift (PM-Vol korreliert mit grossem Gap/Chaos). GapDown: kein klarer Effekt.

**PM-Range:** Eng (<0.3 ADR) hat hoehere Fill-Rate (30-41%), Weit (>0.7 ADR) niedrigere (10-12%).

**PM x Gap-Groesse:** Kein klarer Interaktionseffekt bei >2 ADR Gaps (N zu klein, hohe Varianz).

**Trader-Takeaway:** PM/RTH30 Ratio ist einfach zu berechnen und zeigt den groessten Drift-Spread. Bei PM/RTH<10% mit Gap traden; bei PM/RTH>30% Reversal erwarten.

---

### AUFGABE 6: Bonus-Fragen

**6.1: VWAP-Hold nach 30 Minuten**
- In ~91% der Faelle wird VWAP innerhalb von 30min gebrochen (3+ aufeinanderfolgende Closes)
- Wenn VWAP 30min NICHT gebrochen: haelt er NUR 1-1.6% bis 11:00
- Wenn VWAP dann doch bricht (30-90min): starke Continuation
  - GapUp: Drift=+0.595 (N=197) → Price explodiert in Gap-Richtung
  - GapDown: Drift=-0.444 (N=179) → Starker Selloff
- **Trader-Takeaway:** Spaeter VWAP-Break (nach 30min Hold) = starkes Signal!

**6.2: Erste Kerze (1min)**
- Kerzen-RICHTUNG ist wichtiger als Kerzen-GROESSE:
  - GapUp + erste Kerze up: Drift=+0.175 vs down: Drift=-0.166 (Spread=0.341!)
  - GapDown + erste Kerze up: Drift=+0.164 vs down: Drift=-0.189 (Spread=0.353!)
- Kerzen-Groesse hat keinen konsistenten Effekt (Drift um 0 fuer alle Groessen)
- Grosse Kerze + passende Richtung = staerkstes Signal:
  - GapUp + Grosse Kerze + up: Drift=+0.275 (N=460)
  - GapDown + Grosse Kerze + down: Drift=-0.345 (N=402)
- **Trader-Takeaway:** Erste Kerze bestimmt Bias. Grosse Gap-konforme erste Kerze = staerkstes Morning-Signal.

**6.3: Vorheriger Tag Close-Location**
- **KEINE DATEN** (prev_close_location nicht in Metadata vorhanden)

---

### GESAMTFAZIT DURCHLAUF 4.5

**Neue Signale (nach Staerke sortiert):**
1. **PM/RTH30 Ratio** → Spread 0.49-0.56 ADR (NEU, staerkstes neues Signal)
2. **OD-Staerke >0.5 ADR** → SL-Hit sinkt auf ~40%, WR@0.50 ~51%
3. **Erste Kerze Richtung** → Spread 0.34-0.35 ADR
4. **Spaeter VWAP-Break** → Starke Continuation (Drift ±0.44-0.60)
5. **SPY x Gap-Groesse Interaktion** → Spread waechst mit Gap-Groesse
6. **Kapitulation bounced NICHT** → GapDown + Prior<-10% + RVOL 2-5x = weiter runter

**Was NICHT funktioniert:**
1. VWAP-Rejection Metrik (strukturell fehlerhaft)
2. PM-Volumen absolut (kein klarer Effekt)
3. Gap-Groesse marginaler Effekt (nicht signifikant)
4. Prev-Close-Location (keine Daten)

**Vergleich mit Cheat Sheet v1:**
- SPY-Kontext bleibt staerkstes Signal (v1: Drift ±0.53-0.63, v2 SPY-Interaktionen vertiefen das)
- RVOL-Effekt differenziert: RVOL<2x bei GapDown = einziger signifikanter marginaler Faktor
- PM/RTH Ratio ist NEUES Top-Signal, war in v1 nicht untersucht
- OD-Staerke quantifiziert den Opening Drive besser als v1 (Threshold: 0.5 ADR)

### Dateien
- Scripts: scripts/cheat_v2_spy_filtered.py, scripts/cheat_v2_interaction_matrix.py, scripts/cheat_v2_opening_drive_mfe.py, scripts/cheat_v2_vwap_rejection.py, scripts/cheat_v2_premarket.py, scripts/cheat_v2_bonus.py
- Ergebnisse: results/cheat_sheet_v2_*.txt
- Rohdaten: results/cheat_v2_*_raw.parquet

---

## 2026-02-13 — Durchlauf 5.0: RL Agent (Data Engineering)

### Aufgabe 0: Earnings-Daten von Polygon

**API:** /vX/reference/financials (v3/reference/earnings nicht verfuegbar im Starter Plan)
- 952/1050 Ticker mit Earnings-Daten gefunden
- 32,922 Earnings-Records heruntergeladen
- 1,201 von 10,161 Gapper-Tagen mit Earnings verknuepft (11.8%)

**EINSCHRAENKUNG:** Polygon's /vX/reference/financials liefert nur actual EPS/Revenue,
KEINE Estimates. Daher ist `earnings_score` fuer alle = 0 (Meet).
Feature `has_earnings` (binary) ist trotzdem nuetzlich.

### Feature-Vektor

**19 Features gebaut:**
1. gap_direction_num: +1/-1
2. gap_size_adr: Mean=1.94, Std=2.53 (clipped auf 5.0)
3. adr_pct: Mean=0.069 (6.9% ADR)
4. prior_10d_return: Normalisiert auf Dezimal (z.B. 0.034)
5. prior_30d_return: Normalisiert auf Dezimal
6. rsi_14: Mean=0.50 (normalisiert 0-1)
7. pm_volume_ratio: Median-normalisiert, Mean=2.01
8. pm_range_adr: Mean=1.88
9. has_earnings: 12.95% der Gapper
10. earnings_score: Alle 0 (Polygon-Limitierung)
11-19: Post-Open Features (berechnet im Environment zur Laufzeit)

**Episode Dataset:**
- Train: 3,006 Episoden (2021-02-21 bis 2022-12-31)
- Val: 1,248 Episoden (2023-01-01 bis 2023-12-31)
- Total: 4,254 Episoden, 0 uebersprungen

### Environment + Training Pipeline

**GapperEnv (Gym):**
- 17 Entscheidungspunkte pro Tag (alle 5 min, 9:35-10:55)
- 8 Aktionen (Nothing, Long x3 SL, Short x3 SL, Close)
- SL-Pruefung auf 1-min Bars (nicht nur Close)
- Max 3 Trades pro Tag, Timeout um 11:00
- Costs: 0.015 ADR Slippage + 0.005 ADR Commission = 0.02 ADR/Trade

**PPO Training:**
- 256x256 Hidden Units (2 Layers)
- Feature Dropout 20%, Noise Std 0.05
- Val Check alle 10k Steps, Early Stopping nach 3 Degradations
- Smoke Test bestanden (1000 Steps in 2.6 Minuten)

### Dateien
- scripts/rl_data_engineer.py → Earnings + Features
- scripts/rl_environment.py → Gym Environment
- scripts/rl_train.py → Training Loop
- scripts/rl_evaluate.py → Baselines + Evaluation
- scripts/rl_sensitivity.py → Feature Importance
- scripts/rl_risk_analysis.py → Risk Manager
- data/metadata/metadata_with_earnings.parquet
- data/metadata/rl_episodes.parquet

### Baseline-Ergebnisse (Val, 1,248 Episoden, VOR RL-Training)

| Agent | Mean Reward (ADR) | Sharpe | CI 95% | Trade Days% | WR% |
|-------|------------------|--------|--------|-------------|-----|
| B1 Random | -0.0352 | -0.049 | [-0.074, +0.003] | 100% | 42.6% |
| B2 Always Long | **+0.0136** | 0.022 | [-0.021, +0.049] | 100% | 33.6% |
| B3 With Gap | -0.0107 | -0.019 | [-0.038, +0.023] | 100% | 32.4% |
| B4 OD Follower | -0.0043 | -0.008 | [-0.034, +0.026] | 74.3% | 32.6% |
| B5 PM/RTH Ratio | -0.0041 | -0.021 | [-0.015, +0.007] | 16.8% | 44.3% |

**Erkenntnisse:**
- B2 Always Long ist die staerkste Baseline, aber CI umschliesst 0 → NICHT signifikant
- B3 With Gap (Long bei GapUp, Short bei GapDown) ist NEGATIV → Gap-Richtung allein KEIN Edge
- B5 PM/RTH Ratio tradet sehr selektiv (nur 17% der Tage), hat aber beste WR
- Alle Baselines haben Sharpe < 0.05 → KEIN robuster Edge mit fixen Regeln
- Quarterly instabil: B2 ist Q1/Q4 positiv, Q2/Q3 negativ

---

### RL-Training Ergebnis

**Training:** 60k Timesteps, PPO, Early Stopping nach 6 Checks (18 Minuten)
- Bestes Modell bei Step 20k: Val Reward = +0.028, Sharpe = 0.068
- Danach kontinuierliche Val-Verschlechterung → Early Stopping

### RL Agent vs Baselines (Val, 1,248 Episoden)

| Agent | Mean Reward | Sharpe | CI 95% | Trade Days | WR% |
|-------|------------|--------|--------|------------|-----|
| B1 Random | -0.0878 | -0.144 | [-0.122, -0.055] | 100% | 40.9% |
| **B2 Always Long** | **+0.0136** | **0.022** | [-0.020, +0.052] | 100% | 33.6% |
| B3 With Gap | -0.0107 | -0.019 | [-0.042, +0.021] | 100% | 32.4% |
| B4 OD Follower | -0.0043 | -0.008 | [-0.033, +0.025] | 74.3% | 32.6% |
| B5 PM/RTH Ratio | -0.0041 | -0.021 | [-0.015, +0.007] | 16.8% | 44.3% |
| **RL Agent** | **-0.0262** | **-0.083** | **[-0.044, -0.008]** | 63.2% | 40.4% |

**ERGEBNIS: RL Agent VERLIERT gegen ALLE Baselines ausser Random.**
- RL CI schliesst Null AUS (signifikant negativ!)
- RL schlaegt nur Random (+0.062 ADR)
- RL verliert gegen Always Long um -0.040 ADR
- RL verliert gegen OD Follower um -0.022 ADR
- RL verliert gegen PM/RTH Ratio um -0.022 ADR

### Sensitivity Analysis — Was der Agent "gelernt" hat

**Permutation Importance (INVERTIERTE Ergebnisse!):**
Negative "Drop" = Feature shufflen VERBESSERT Performance → Agent nutzt Feature SCHAEDLICH

| Feature | Drop | Interpretation |
|---------|------|---------------|
| rsi_14 | +0.019 | **Einziger positiver Beitrag** |
| pm_range_adr | -0.023 | Schaedlich |
| od_strength_adr | -0.094 | **Am schaedlichsten — shufflen gewinnt +0.09 ADR!** |
| gap_direction | -0.071 | Schaedlich |
| prior_30d_return | -0.078 | Schaedlich |
| pct_adr_used | -0.076 | Schaedlich |

→ Der Agent hat fuer 18 von 19 Features **FALSCHE Assoziationen gelernt**.
  Nur RSI hat marginalen Wert.

**Action Distribution:**
- 65.5% CLOSE (schliesst zu frueh)
- 13.8% SHORT_TIGHT (favorisiert Short mit engem SL)
- 7.5% LONG total (stark Short-biased)
- Agent ist stark SHORT-biased — fatal in einem ueberwiegend bullishen Markt

**Partial Dependence (Top Feature: RSI):**
- Niedriges RSI → mehr Short (30.2%), weniger Close (66.8%)
- Hohes RSI → weniger Short (24.2%), mehr Close (73.8%)
- Agent shorted oversold Aktien — GENAU FALSCH (RSI<30 = Bounce-Signal aus Durchlauf 4.0)

**Trade vs No-Trade Unterschiede:**
- Agent tradet bevorzugt bei: niedrigem PM-Volumen, OD mit Gap, mehr VWAP-Tests, Earnings-Tagen
- Agent vermeidet: starken OD, positive VWAP-Position

### Risk Analysis

| Metrik | Wert |
|--------|------|
| Max Drawdown | 32.5 ADR ueber 1204 Episoden |
| SPY Korrelation | -0.006 (markt-neutral) |
| Win/Loss Ratio | 1.12 |
| Win Rate (Trade Days) | 40.4% |
| Worst 1%ile | -0.81 ADR |
| Worst Losing Streak | 6 Tage |

Quartals-Stabilitaet:
- Q1 2023: -0.005 (near zero)
- Q2 2023: -0.027
- Q3 2023: -0.029
- Q4 2023: -0.048
- **TREND: Verschlechtert sich ueber die Zeit**

### GESAMTFAZIT DURCHLAUF 5.0

**1. RL hat KEINEN Edge gefunden.**
Der PPO-Agent mit 19 Features, 8 Aktionen und 60k Training-Steps konnte keine profitable
Strategie lernen. Er verliert signifikant gegen alle nicht-zufaelligen Baselines.

**2. Overfitting war das Hauptproblem.**
Trotz Anti-Overfitting-Massnahmen (Feature Dropout 20%, Noise, Early Stopping, kleine
Architektur) hat der Agent Noise im Trainingsset memorisiert statt echte Muster zu finden.
Die Permutation Importance zeigt, dass 18/19 Features SCHAEDLICH genutzt werden.

**3. 4,254 Samples sind zu wenig fuer RL.**
Bei 3,006 Train-Episoden, 17 Steps pro Episode und 8 Aktionen reicht die Datenmenge nicht,
um sinnvolle Policies zu lernen. Das Verhaeltnis von State-Space-Groesse zu Datenmenge ist
ungünstig.

**4. Die Baselines bestaetigen die Erkenntnisse aus Durchlaeufen 1-4.5:**
- Always Long (+0.014) ist die einzige positive Baseline → Gapper haben leicht bullischen Drift
- With Gap (-0.011) ist negativ → Gap-Richtung allein kein Edge
- OD Follower (-0.004) ist near-zero → OD allein kein robuster Edge
- PM/RTH Ratio (-0.004) tradet selektiv (17% der Tage), beste WR (44.3%), aber negativ

**5. Die Sensitivity-Analyse ist der WERTVOLLSTE Output:**
Sie bestaetigt, dass KEINE Feature-Kombination aus den 19 Variablen einen ausreichend
starken, konsistenten Edge generiert, der die Transaktionskosten (0.02 ADR/Trade) uebersteigt.
Dies ist konsistent mit den Ergebnissen aller frueheren Durchlaeufe.

**6. Erwartungsmanagement war korrekt:**
Die vorab ehrliche Einschaetzung "Wahrscheinlich kein Edge ueber einfache Regeln" hat sich
bestaetigt. Das ist ein VALIDES ERGEBNIS — nicht jede Analyse muss einen handelbaren Edge finden.

### Dateien
- Training Log: results/rl_training_log.txt
- Evaluation: results/rl_evaluation.txt
- Sensitivity: results/rl_sensitivity.txt
- Risk: results/rl_risk_analysis.txt
- Modell: models/rl_agent_v1.zip (Step 20k, bestes Val)
- Scripts: scripts/rl_*.py (6 Scripts)

---

## Durchlauf 6.0 — OOS-Validierung Cheat Sheet auf Haelfte 2

**Datum:** 2026-02-13
**Ziel:** Einmalige OOS-Validierung aller 20 Findings aus Durchlauf 4.0/4.5 auf Half 2 (2024-01-01 bis 2026-02-06)
**Datenbasis:** Half 2 N=5,812 Gapper vs. Half 1 (IS) N=4,254 Gapper

### Gesamtergebnis

| Status | Anzahl | Anteil |
|--------|--------|--------|
| REPLIZIERT (≥50% IS-Effekt, gleiche Richtung) | 28/39 | 72% |
| TEILWEISE (<50% IS-Effekt, gleiche Richtung) | 3/39 | 8% |
| NICHT REPLIZIERT (Effekt ~0) | 1/39 | 3% |
| UMGEKEHRT (Vorzeichen-Flip) | 5/39 | 13% |
| N/A (nicht testbar) | 2/39 | 5% |

**72% Replikationsrate** — die Mehrheit der Findings sind echte Marktmuster, kein In-Sample-Zufall.

### Validierungstabelle

| # | Finding | IS-Effekt | OOS-Effekt | Status |
|---|---------|-----------|------------|--------|
| F1 | SPY-Kontext (GapUp+rot) | Drift -0.630 | Drift -0.337 | REPLIZIERT |
| F1 | SPY-Kontext (GapDn+gruen) | Drift +0.534 | Drift +0.136 | TEILWEISE |
| F2 | RVOL >5x Fill GapUp | Fill 11% | Fill 11% | REPLIZIERT |
| F2 | RVOL <1x Fill GapUp | Fill 45% | Fill 48% | REPLIZIERT |
| F2 | RVOL >5x Fill GapDn | Fill 9% | Fill 7% | REPLIZIERT |
| F2 | RVOL <1x Fill GapDn | Fill 36% | Fill 35% | REPLIZIERT |
| F3 | OD mit Gap = Tagesrichtung | 67-68% | 65-66% | REPLIZIERT |
| F3 | OD gegen Gap | 28-30% | 26-31% | REPLIZIERT |
| F4 | GapUp nach >20% Run | Drift -0.207 | Drift +0.184 | UMGEKEHRT |
| F4 | GapDn + RSI<30 | Drift +0.415 | Drift +0.098 | TEILWEISE |
| F4 | GapUp + RSI>70 | Drift -0.201 | Drift +0.102 | UMGEKEHRT |
| F5 | Gap-Fill <1 ADR | Fill ~40% | Fill 41-46% | REPLIZIERT |
| F5 | Gap-Fill 1-2 ADR | Fill ~17% | Fill 17-21% | REPLIZIERT |
| F5 | Gap-Fill >2 ADR | Fill <5% | Fill 0-10% | REPLIZIERT |
| F6 | PM/RTH<10% GapUp | Drift +0.312 | Drift +0.512 | REPLIZIERT |
| F6 | PM/RTH>30% GapUp | Drift -0.244 | Drift -0.132 | REPLIZIERT |
| F6 | PM/RTH<10% GapDn | Drift -0.309 | Drift -0.215 | REPLIZIERT |
| F6 | PM/RTH>30% GapDn | Drift +0.180 | Drift +0.208 | REPLIZIERT |
| F7 | OD-Staerke >0.5 SL-Hit | ~40% | 39-40% | REPLIZIERT |
| F7 | OD-Staerke <0.2 SL-Hit | ~80% | 65-78% | REPLIZIERT |
| F7 | GapDn+ODwith+Stark WR@50 | 51% | 49% | REPLIZIERT |
| F8 | Spaeter VWAP-Break GapUp | Drift +0.595 | Drift +0.514 | REPLIZIERT |
| F8 | Spaeter VWAP-Break GapDn | Drift -0.444 | Drift -0.505 | REPLIZIERT |
| F9 | Erste Kerze GapUp+up gross | Drift +0.275 | Drift +0.338 | REPLIZIERT |
| F9 | Erste Kerze GapDn+dn gross | Drift -0.345 | Drift -0.317 | REPLIZIERT |
| F10 | Vol>30% GapUp (Fade) | Drift -0.238 | Drift +0.062 | UMGEKEHRT |
| F10 | Vol>30% GapDn (Bounce) | Drift +0.203 | Drift +0.534 | REPLIZIERT |
| F11 | Kapitulation bounced nicht | Drift -0.264 | Drift +0.239 | UMGEKEHRT |
| F12 | VWAP Regime (held 30min) | VWAP Widerstand 17% | VWAP held 5-6% | NICHT REPLIZIERT |
| F13 | Level-Rejections halten | 25% hold 30min | 18% hold 30min | REPLIZIERT |
| F14 | Multi-Touch | 61%->57% Break | nicht getestet | N/A |
| F15 | HOD/LOD in 30min | 45-48% | 47-48% | REPLIZIERT |
| F16 | Follow-Through Tag 2 | ~50/50 (kein Edge) | ~50/50 bestaetigt | REPLIZIERT |
| F17 | VWAP Slope -> CL | Rising CL=0.61 | Daten nicht parsebar | N/A |
| F18 | SPY x Gap-Groesse Spread | Waechst bei GapDn | GapDn: +0.19 -> +0.39 | REPLIZIERT |
| F19 | Sektor: Retail GapUp | Drift +0.159 | Drift -0.220 | UMGEKEHRT |
| F19 | Sektor: Finance GapDn | Drift +0.087 | Drift +0.038 | TEILWEISE |
| F20 | Kleine Orders GapUp (Fade) | Drift -0.134 | Drift -0.203 | REPLIZIERT |
| F20 | Grosse Orders GapUp | Drift +0.020 | Drift +0.026 | REPLIZIERT |

### Kern-Erkenntnisse

**1. Staerkstes robustes Signal: PM/RTH30 Ratio (F6)**
- Alle 4 Richtungs-Hypothesen replizieren OOS
- GapUp PM/RTH<10%: IS=+0.312, OOS=+0.512 (STAERKER OOS!)
- Spread GapUp (<10% vs >30%): IS=+0.556, OOS=+0.645
- Einziges Signal das OOS staerker wird

**2. Strukturelle Wahrheiten (exakte Replikation):**
- RVOL bestimmt Fill-Raten (F2): Identische Werte IS/OOS
- Gap-Fill sinkt mit Gap-Groesse (F5): Exakte Replikation
- Opening Drive setzt Tagesrichtung in ~66% (F3)
- HOD/LOD in ersten 30min: 47-48% IS und OOS (F15)
- Kein Follow-Through Tag 2 (F16): Bestaetigt

**3. Starke Replikation:**
- Spaeter VWAP-Break (F8): IS +0.60/-0.44, OOS +0.51/-0.51
- Erste Kerze Richtung (F9): IS +0.28/-0.35, OOS +0.34/-0.32
- OD-Staerke SL-Hit (F7): ~40% bei Stark repliziert exakt

**4. Gescheiterte Hypothesen (UMGEKEHRT OOS):**
- F4 Prior Performance / RSI: GapUp nach Hot Run oder RSI>70 NICHT fade-bar
- F11 Kapitulation: OOS bounced es DOCH (+0.24 statt -0.26)
- F10 Vol-Konzentration GapUp: Kein Fade-Signal
- F19 Sektor Retail: Momentum-Hypothese umgekehrt

**5. SPY-Kontext (F1) geschrumpft:**
- GapUp + SPY rot: IS=-0.63, OOS=-0.34 (46% Shrinkage)
- GapDown + SPY gruen: IS=+0.53, OOS=+0.14 (74% Shrinkage)
- Richtung stimmt, Effekt massiv reduziert

### Finales Trader-Cheat-Sheet (OOS-validiert)

**NUTZEN:**
- PM/RTH<10% → Trade MIT dem Gap (Continuation)
- PM/RTH>30% → Fade erwarten
- OD-Staerke >0.5 ADR → Niedrigere SL-Hit-Rate (~40%)
- Erste grosse Kerze in Gap-Richtung → Staerkstes Morning-Signal
- Spaeter VWAP-Break (nach 30min Hold) → Starke Continuation
- SPY-Kontext bei GapUp → Zuverlaessiger als bei GapDown
- RVOL >5x = Momentum (Fill <11%), RVOL <1x = Fade (Fill ~45%)
- HOD/LOD in ersten 30min in ~48% → Frueher Entry hat Bedeutung

**NICHT NUTZEN (instabil OOS):**
- Prior Performance / RSI als Fade-Signal
- Kapitulation = Bounce (instabil)
- Sektor-basierte Trades
- Volumen-Konzentration bei GapUp
- VWAP als alleiniger Regime-Filter

### Dateien
- Konsolidierter Report: results/oos_cheat_sheet_validation.txt
- Metadata-Validierung: results/oos_cheat_validation_meta.txt
- 1min-Validierung: results/oos_cheat_validation_1min.txt
- Rohdaten: results/oos_cheat_validation_1min_raw.parquet
- Scripts: scripts/oos_cheat_validation_meta.py, scripts/oos_cheat_validation_1min.py, scripts/oos_cheat_validation_final.py

## Durchlauf 6.5 -- Kreuzungsanalyse Top-Findings mit Parametern

**Datum:** 2026-02-13
**Ziel:** Top-Findings aus D6.0 mit Parametern kreuzen (Gap-Groesse, RVOL, OD-Staerke, OD-Richtung, PM/RTH). Phase 1 auf H1 (IS), Phase 2 auf H2 (OOS).
**Datenbasis:** H1=4,254 Gapper, H2=5,812 Gapper

### Gesamtergebnis OOS-Validierung

| Status | Anzahl | Anteil |
|--------|--------|--------|
| REPLIZIERT | 34/35 | 97% |
| TEILWEISE | 1/35 | 3% |
| NICHT REPLIZIERT | 0/35 | 0% |
| UMGEKEHRT | 0/35 | 0% |

**97% Replikationsrate** -- fast alle Kreuzungen sind robust.

### Aufgabe 1: F6 PM/RTH x Parameter

**Kernerkenntnisse:**

1. **PM/RTH<10% ist bei GROSSEN Gaps STAERKER als bei kleinen:**
   - GapUp: Gap<1 Drift=+0.13 (IS) / +0.68 (OOS!), Gap>2 Drift=+0.59 / +0.48
   - GapDown: Gap<1 Drift=+0.03 / +0.20, Gap>2 Drift=+0.49 / +0.18 (teilweise)
   - OOS zeigt: Gap<1 bei GapUp massiv staerker als IS! Gap>2 GapDown schwaecht sich ab.

2. **PM/RTH<10% bei RVOL>5x = Staerkstes 2er-Signal:**
   - GapUp: IS=+0.689, OOS=+0.525 (repliziert)
   - GapDown: IS=+0.605, OOS=+0.339 (repliziert)

3. **PM/RTH<10% + OD>0.5 with_gap = Extremstes Signal:**
   - GapUp: IS=+1.172, OOS=+1.030 (N=82, repliziert)
   - GapDown: IS=+1.199, OOS=+0.989 (N=40, repliziert)
   - Das ist >1.0 ADR Drift in Gap-Richtung!

4. **PM/RTH bei verschiedenem RVOL:**
   - PM/RTH<10% wirkt bei JEDEM RVOL-Level (repliziert ueberall)
   - Aber RVOL>5x verstaerkt den Effekt deutlich

### Aufgabe 2: F8 Spaeter VWAP-Break

**Kernerkenntnisse:**

1. **Post-Break Move ist KLEIN:**
   - GapUp: PB_15min=+0.005, PB_30min=+0.029, PB_11:00=+0.043 ADR (OOS)
   - GapDown: PB_15min=+0.013, PB_30min=+0.018, PB_11:00=+0.030 ADR
   - Der VWAP-Break selbst generiert keinen grossen zusaetzlichen Move!

2. **Beste Post-Break Combos:**
   - F8-A (Gap>2 + RVOL>5x): PB_11:00=+0.170/+0.134 ADR (IS/OOS repliziert)
   - F8-G (Gap>2 + OD with + RVOL>2x): PB_11:00=+0.164/+0.124 (repliziert)
   - F8-I (Grosse 1.Kerze in Gap): PB_11:00=+0.121 (OOS GapUp)

3. **VWAP-Break + PM/RTH<10% (Doppelsignal):**
   - GapUp: IS=+0.678, OOS=+0.937 (STAERKER OOS!)
   - GapDown: IS=+0.550, OOS=+0.792 (STAERKER OOS!)
   - Aber: N ist klein (46/27), Post-Break Drift selbst bescheiden

4. **F8-F (Break + PM/RTH>30%):**
   - GapDown Post-Break: -0.095 ADR (OOS) -- Break GEGEN den Fade = schlecht
   - Verstaerkt NICHT den Move

### Aufgabe 3: F9 Erste Kerze

**Kernerkenntnisse:**

1. **Schwellen-Test (Sweet Spot):**
   - >0.10 ADR: N=793/728, Drift=+0.34/+0.32 (gut, hohes N)
   - >0.20 ADR: N=469/438, Drift=+0.54/+0.44 (besser, noch gutes N)
   - >0.30 ADR: N=304/262, Drift=+0.70/+0.58 (stark, N akzeptabel)
   - >0.50 ADR: N=136/90, Drift=+1.13/+0.81 (extrem, N wird duenn)
   - Sweet Spot: >0.20 ADR bietet bestes Verhaeltnis Drift/N

2. **Erste Kerze + PM/RTH<10%:**
   - GapUp: IS=+0.590, OOS=+0.880 (STAERKER OOS!, N=112)
   - GapDown: IS=+0.395, OOS=+0.672 (STAERKER OOS!, N=72)

### Aufgabe 4: F7 OD-Staerke

**Kernerkenntnisse:**

1. **WR@0.50 ueber 55% bei OD>0.5 -- ALLE REPLIZIERT:**
   - GapDown+ODwith+PM<10%: IS=73%, OOS=78% (N=40, LOW N aber STAERKER)
   - GapUp+ODwith+RVOL>5x: IS=68%, OOS=73% (N=186, STAERKER!)
   - GapUp+ODwith+Gap>2: IS=65%, OOS=70% (N=198, STAERKER!)
   - GapDown+ODwith+RVOL>5x: IS=67%, OOS=69% (N=162)
   - GapDown+ODagainst+RVOL>5x: IS=64%, OOS=66% (N=118)

2. **Vollstaendige OD>0.5 Tabelle (OOS):**
   - GapUp with_gap: WR@0.50=62%, SL-Hit=26% (N=380)
   - GapDown with_gap: WR@0.50=60%, SL-Hit=26% (N=287)
   - GapUp against_gap: WR@0.50=56%, SL-Hit=22% (N=395)
   - GapDown against_gap: WR@0.50=54%, SL-Hit=21% (N=286)
   - OD with_gap hat HOEHERE WR aber auch hoeheren SL-Hit

### Aufgabe 5: Multi-Factor Signalkombinationen

**Alle Combos repliziert OOS! Die besten:**

| Combo | Beschreibung | GapUp IS/OOS | GapDown IS/OOS | N (GU/GD) |
|-------|-------------|-------------|---------------|-----------|
| F | PM/RTH<10% + Kerze + OD>0.5w (TRIPLE) | +1.28/+1.11 | +1.17/+1.13 | 64/27 |
| E | Grosse Kerze + OD>0.5 with Gap | +1.05/+1.08 | +1.06/+1.07 | 263/186 |
| B | PM/RTH<10% + OD>0.5 with Gap | +1.17/+1.03 | +1.20/+0.99 | 82/40 |
| D | VWAP-Break + PM/RTH<10% | +0.68/+0.94 | +0.55/+0.79 | 46/27 |
| A | PM/RTH<10% + Grosse Kerze | +0.59/+0.88 | +0.40/+0.67 | 112/72 |
| H | PM/RTH>30% + OD against (FADE) | -0.39/-0.32 | -0.41/-0.43 | 601/491 |

**Combo E** hat das beste Verhaeltnis aus Drift (>1.0 ADR) und Stichprobengroesse (N=263/186).
**Combo F** (Triple) hat den hoechsten Drift, aber N wird duenn.
**Combo H** ist das staerkste FADE-Signal mit grossem N.

### Kern-Schlussfolgerungen Durchlauf 6.5

1. **PM/RTH<10% ist NICHT gleich stark ueberall** -- es interagiert stark mit OD-Staerke und Richtung. PM/RTH<10% + OD>0.5 with_gap = >1.0 ADR Drift.

2. **Der VWAP-Break selbst erzeugt wenig zusaetzlichen Move** (0.03-0.04 ADR). Der Wert liegt darin, dass er ein Filter-Signal ist, nicht ein Entry-Signal. Die Tages-Drift-Wirkung kommt primaer aus den anderen Faktoren.

3. **OD>0.5 mit SL am 5min H/L hat WR@0.50 von 60-73%** je nach Parameterkonstellation. Die 49% aus D6.0 war der GESAMTDURCHSCHNITT -- die gefilterten Subgruppen liegen deutlich hoeher.

4. **Erste Kerze >0.20 ADR in Gap-Richtung ist der Sweet Spot** fuer den Schwellenwert (Drift 0.44-0.54 ADR bei N>400).

5. **Combo E (Grosse Kerze + OD>0.5 with Gap) ist der "Goldstandard"**: >1.0 ADR Drift, N>180, repliziert OOS sogar STAERKER.

6. **Fade-Signal: PM/RTH>30% + OD against Gap** liefert -0.32 bis -0.43 ADR Drift (Fade) mit sehr grossem N (>490). Robustes Gegenrichtungs-Signal.

### Dateien
- Phase 1 (IS): results/cross_analysis_h1.txt
- Phase 2 (OOS): results/cross_analysis_h2_validation.txt
- Scripts: scripts/cross_analysis_h1.py, scripts/cross_analysis_h2.py

---

## 2026-02-14 — Durchlauf 7.0: Finaler Backtest mit neuen Features + Komplett-Revalidierung

### Ziel
Drei neue Features evaluieren (RVOL_5, PM/RTH5, rest_drift), alle bei 9:35 verfuegbar —
25 Minuten frueher als bisherige Signale. Komplette Revalidierung aller Strategien.

### Neue Features (metadata_v7.parquet, 82 Spalten)
- **RVOL_5**: Relative Volume nach 5min RTH (= existing rvol_at_time_5min, r=1.000)
- **PM/RTH5**: Premarket-Volumen / erste 5 RTH-Minuten Volumen
- **rest_drift**: Drift von 9:35 bis Close, normiert auf ADR, in Gap-Richtung
- **rest_drift_1000/1100**: Gleich, aber nur bis 10:00 bzw. 11:00
- **close_935**: Preis um 9:35 (Entry-Level)
- **od_strength/od_direction**: Opening Drive Staerke und Richtung
- **first_candle_dir/size**: Erste Kerze Richtung und Groesse
- **full_drift, cl**: Tages-Drift und Close Location

### Kern-Ergebnisse IS (H1: 2021-2023, N=4,254)

**Task 1: RVOL Granular**
- Log-Trend passt besser als Linear (AIC) fuer alle 6 Vergleiche
- Aber: R²<0.01 — RVOL erklaert <1% der Drift-Varianz
- RVOL_30 gewinnt Head-to-Head vs RVOL_5 (Spearman +0.078 vs +0.053)
- Korrelation RVOL_5/RVOL_30: r=0.87 (Pearson), r=0.96 (Spearman)

**Task 2: PM/RTH5 vs PM/RTH30**
- PM/RTH30 ist ca. 2x praediktiver (Spearman -0.122 vs -0.077 fuer full_drift)
- PM/RTH5<30% hat 89% Precision fuer PM/RTH30<10% (aber nur 59% Recall)
- PM/RTH5 taugt als Fruehwarnung, nicht als Ersatz

**Task 3: Combo E Rest-Drift — DAS KERNRESULTAT**
- GapUp: rest_drift = -0.022 (NULL!) — Gesamter Drift kommt aus OD
- GapDown: rest_drift = +0.252 (positiv!) — Echte Post-OD Continuation
- Implikation: Bei GapUp ist der Zug um 9:35 abgefahren

**Task 4: Full Cross Matrix**
- 597 Zellen berechnet (9 Parameter × Buckets × 2 Gap-Richtungen)
- Top GapUp: FxG (OD with_gap + >0.5) = +1.07 ADR
- Top GapDown: FxG = +1.00, DxG (PM30<10% + OD>0.5) = +0.85

**Task 5: Gescheiterte Strategien**
- KEINE wird durch neue Features gerettet
- Kein RVOL_5 oder PM/RTH5 Bucket stabilisiert die IS-Effekte

**Task 6: WR bei Targets**
- Bester EV: PM30<10%+OD>0.5w GapDown at T=0.50: EV=+0.171 ADR
- GapDown schlaegt GapUp systematisch
- Optimaler Target: 0.25-0.50 ADR (groessere Targets haben neg. EV)

### OOS-Validierung (H2: 2024-2026, N=5,812)

**Replikationsrate:** Hervorragend!

| Finding | IS | OOS | Status |
|---------|-----|-----|--------|
| Combo E GapUp full_drift | +1.06 | +1.18 | REPLIZIERT |
| Combo E GapDown full_drift | +1.18 | +1.11 | REPLIZIERT |
| Combo E GapUp rest_drift~0 | -0.02 | +0.09 | REPLIZIERT |
| Combo E GapDown rest_drift>0 | +0.25 | +0.18 | TEILWEISE |
| PM30<10%+OD>0.5w GapUp fd | +0.60 | +1.03 | REPLIZIERT |
| PM30<10%+OD>0.5w GapDown fd | +0.47 | +0.99 | REPLIZIERT |
| OD>0.5 with_gap GapUp fd | +0.48 | +1.09 | REPLIZIERT |
| OD>0.5 with_gap GapDown fd | +0.45 | +0.92 | REPLIZIERT |
| RVOL_30 > RVOL_5 | Spear. wins | Spear. wins | REPLIZIERT |
| PM/RTH30 > PM/RTH5 | 2x staerker | 2x staerker | REPLIZIERT |
| Kapitulation failed | -0.23 | -0.19 | STILL NEGATIVE |
| Prior/RSI failed | -0.12/-0.24 | +0.18/+0.10 | INSTABIL |

**Cross-Combos:** 8/12 Top-Combos REPLIZIERT, 4/12 TEILWEISE, 0 gescheitert

**WR/EV OOS:**
- PM30<10%+OD>0.5w GapDown: WR@0.50=77.5%, EV=+0.156 (bester, aber N=40)
- Combo E GapDown: WR@0.25=81.2%, EV=+0.032
- OD>0.5 with_gap GapDown: WR@0.25=80.5%, EV=+0.017

### Finale Top-5 Setups

1. **OD>0.5 with_gap GapDown** (9:35) — N=287 OOS, fd=+0.92, WR@0.25=80.5%
2. **OD>0.5 with_gap GapUp** (9:35) — N=380 OOS, fd=+1.09, WR@0.25=80.3%
3. **PM30<10%+OD>0.5w GapDown** (10:00) — N=40 OOS, EV@0.50=+0.156
4. **RVOL30>5x+OD>0.5w GapUp** (10:00) — N=186 OOS, fd=+1.39, rd=+0.23
5. **Combo E GapDown** (9:35) — N=154 OOS, rd=+0.18 (best rest-drift)

### Schlussfolgerungen D7.0

1. **Neue Features bringen KEINEN Durchbruch** — RVOL_5/PM/RTH5 sind schwaecher als RVOL_30/PM/RTH30
2. **Die OD-Phase (0-5min) bestimmt ALLES** — rest_drift ist ~0 fuer GapUp, schwach positiv fuer GapDown
3. **GapDown > GapUp systematisch** — Besserer EV, besserer rest_drift
4. **Gescheiterte Strategien bleiben gescheitert** — Keine Rettung durch neue Features
5. **OD>0.5 with_gap ist das robusteste Signal** — Groesstes N, konsistent IS/OOS
6. **Optimaler Target: 0.25-0.50 ADR** — Groessere Targets haben negativen EV
7. **PM/RTH5<30% taugt als Fruehwarnung** fuer PM/RTH30<10% (77% Precision OOS)

### Dateien
- Feature Engineering: scripts/d7_feature_engineer.py → data/metadata/metadata_v7.parquet
- RVOL Granular: scripts/d7_rvol_granular.py → results/d7_rvol_granular.txt
- PM/RTH5: scripts/d7_pm_rth5.py → results/d7_pm_rth5.txt
- Combo E: scripts/d7_combo_e_new.py → results/d7_combo_e_new.txt
- Full Cross: scripts/d7_full_cross.py → results/d7_full_cross.txt
- Failed Strategies: scripts/d7_failed_strategies.py → results/d7_failed_strategies.txt
- WR Targets: scripts/d7_wr_targets.py → results/d7_wr_targets.txt
- OOS Validation: scripts/d7_oos_validation.py → results/d7_oos_validation.txt
- Synthese: results/durchlauf_7_synthese.txt

---

## Durchlauf 7.5 — PM/RTH5 x RVOL_5 x Gap_in_ADR: Das 9:35-Dreieck

### Mission
Komplette Verhaltens-Landkarte fuer ALLE Kombinationen von PM/RTH5, RVOL_5 und Gap_in_ADR — drei Parameter, die bereits um 9:35 ET bekannt sind. Ziel: Continuation, Fade, Gap-Fill, MFE, MAE, Close-Location fuer jede Zelle.

### Bucket-Definitionen
- **PM/RTH5**: 3 Buckets (<50%, 50-100%, >100%) — auch 4er getestet aber zu duenn
- **RVOL_5**: 3 Buckets (<3x, 3-7x, >7x) — auch 4er getestet (<2, 2-5, 5-10, >10)
- **Gap_in_ADR**: 3 Buckets (<1, 1-2, >2 ADR)
- Ergibt 3×3×3 = 27 Zellen × 2 Gap-Richtungen = 54 Zellen total

### Kern-Ergebnisse

#### 1. Verteilung & Robustheit
- IS (H1): 4,254 Gapper, OOS (H2): 5,812 Gapper
- **4×4×3 Matrix**: Nur 27/96 robuste Zellen (N≥50) — zu duenn fuer reliable Analyse
- **3×3×3 Matrix**: 29/54 robuste Zellen — akzeptabel, als Primaer-Matrix verwendet
- **RVOL_5 und Gap_in_ADR stark korreliert**: rho=+0.645 → tragen redundante Information
- PM/RTH5 vs. RVOL_5: rho=+0.086 (fast unkorreliert)
- PM/RTH5 vs. Gap_in_ADR: rho=+0.124 (schwach)

#### 2. Marginale Effekte
- **PM/RTH5 ist der staerkste Einzelparameter im Dreieck**
  - PM5_LO vs PM5_HI Spread: ~0.35 ADR (konsistent GapUp + GapDown)
  - PM5_LO: Continuation-Bias, PM5_HI: Fade/Flat
  - t-Wert: -3.71 (GapUp), -2.78 (GapDown)
- RVOL_5 allein erklaert fast nichts (R² < 0.001)
- Gap_in_ADR allein hat KEINEN marginalen Effekt
- **Alle drei zusammen: R² = 0.5-0.7% der Drift-Varianz** → schwaches Signal

#### 3. Interaktionen
- PM5 LO-HI Spread WAECHST bei hohem RVOL (RV5_HI: Spread +0.801 GapUp)
- PM5 LO-HI Spread WAECHST bei grossem Gap (GAP_LG: Spread +0.570 GapUp, +0.878 GapDown)
- → PM/RTH5 wird praediktiver wenn die Aktie "heiss" ist (hohe Vol + grosse Gaps)

#### 4. OOS-Validierung: ERNUECHTERND
- **Spearman IS vs OOS full_drift: rho = 0.04 (p=0.78)** → KEINE Replikation
- Pearson: r = -0.083 (p=0.57)
- Status-Verteilung: 27% FLAT, 22% UMGEKEHRT, 20% GESCHRUMPFT, 18% REPLIZIERT, 12% SHIFT
- Die staerksten IS-Zellen (>+0.50 fd) sind OOS INSTABIL
  - z.B. GapDown PM5_LO RV5_HI GAP_LG: IS +0.57 → OOS -0.15 (umgekehrt!)

#### 5. Stabile Zellen (repliziert IS → OOS)
- **GapUp PM5_LO RV5_HI GAP_LG**: IS +0.38, OOS +0.41, N_OOS=148 — BESTES Cont-Signal
- GapUp PM5_MID RV5_LO GAP_LG: IS -0.30, OOS -0.38, N_OOS=52 — Stabiler Fade
- GapDown PM5_MID RV5_LO GAP_LG: IS -0.24, OOS -0.27, N_OOS=43 — Stabiler Fade
- GapUp PM5_LO RV5_MID GAP_MD: IS +0.10, OOS +0.28, N_OOS=90 — Schwache Cont

#### 6. 9:35 vs 10:00 Vergleich
- Missed drift (9:35→10:00): nur ~0.02 ADR → Warten kostet fast nichts
- PM/RTH30 ist ~2x praediktiver als PM/RTH5 (Spearman -0.122 vs -0.077)
- RVOL_30 leicht besser als RVOL_5 (Spearman +0.078 vs +0.053)
- **Fazit: Warten bis 10:00 lohnt sich** — man verliert kaum Drift, gewinnt aber bessere Parameter

#### 7. Trade-Simulation
- Bestes IS-Setup (Cont): GapUp PM5_HI RV5_HI GAP_MD, SL=0.25, TGT=1.0 → EV=+0.524 (N=28)
  - OOS: WR=36%, EV=+0.241 (N=25) — repliziert, aber N winzig
- Bestes IS-Setup (Fade): GapUp PM5_MID RV5_LO GAP_LG, SL=0.25, TGT=1.0 → EV=+0.458 (N=22)
  - OOS: geschrumpft
- **Problem**: Alle besten Zellen haben N < 50 im IS

### Haupt-Erkenntnisse

1. **Das 9:35-Dreieck ist ein SCHWACHES Signal** — R² < 1%, OOS rho = 0.04
2. **PM/RTH5 ist der einzige nuetzliche Parameter** — als schnelles Screening, nicht als Trade-Signal
3. **RVOL_5 und Gap_in_ADR sind stark korreliert** (rho=0.65) → eines genuegt
4. **Das Dreieck ist SCHWAECHER als alle bisherigen Top-Signale**:
   - OD>0.5 with_gap: +0.9-1.1 ADR OOS (D7.0)
   - PM/RTH30<10% + OD>0.5: +1.0 ADR OOS (D6.5)
   - Combo E: +1.1 ADR OOS (D6.5)
   - Erste Kerze > 0.20 ADR: +0.34 ADR OOS (D6.0)
5. **Warten auf 10:00 ist besser als 9:35 zu traden** — PM/RTH30 ist doppelt so praediktiv
6. **Ausnahme**: Bei extremer OD (>0.5 ADR) + grosse Kerze um 9:35 SOFORT handeln

### Dateien
- Distribution: scripts/d7_5_distribution.py → results/d7_5_distribution.txt
- 2er-Combos: scripts/d7_5_2er_combos.py → results/d7_5_2er_combos.txt
- 3er-Matrix: scripts/d7_5_3er_matrix.py → results/d7_5_3er_matrix.txt
- Pfad-Analyse: scripts/d7_5_pfad_analyse.py → results/d7_5_pfad_analyse.txt
- Trade-Sim: scripts/d7_5_trade_sim.py → results/d7_5_trade_sim.txt
- Marginal: scripts/d7_5_marginal.py → results/d7_5_marginal.txt
- 10-Uhr-Vergleich: scripts/d7_5_10uhr_vergleich.py → results/d7_5_10uhr_vergleich.txt
- OOS-Validierung: scripts/d7_5_oos.py → results/d7_5_oos.txt
- Synthese: scripts/d7_5_synthese.py → results/d7_5_synthese.txt
- Rohdaten: results/d7_5_3er_matrix_3x3x3.parquet, d7_5_3er_matrix_4x4x3.parquet, d7_5_2er_combos_raw.parquet, d7_5_trade_sim_raw.parquet, d7_5_oos_comparison.parquet, d7_5_pfad_raw.parquet

---

## D8.0 — Earnings-Split + 10:00-Reversal-Prediction

**Kernfragen:**
A) Verhalten sich Earnings-Gapper anders als Non-Earnings-Gapper?
B) Kann man um 9:35 vorhersagen, ob der Tag um 10:00 reverst?

**Datenbasis:** metadata_v8.parquet (10,161 Zeilen + is_earnings Flag)
- IS: 2021-02-21 bis 2023-12-31 (4,349 Zeilen)
- OOS: 2024-01-01 bis 2026-02-06 (5,812 Zeilen)
- Earnings-Daten via Polygon vx.list_stock_financials (filing_date)

### Aufgabe 0: Earnings-Daten & Plausibilitaet

- 953/1050 Tickers mit Earnings-Daten gefetcht
- **Verteilung:** Earnings 15.6% (1,582), Non-Earnings 78.2% (7,945), Unknown 6.2% (634)
- **Plausibilitaet bestaetigt:**
  - Earnings-Gaps 2.24x groesser (median 2.04 ADR vs 0.91 ADR)
  - Earnings-RVOL 2.14x hoeher (4.27 vs 2.43)
  - Earnings-PM/RTH5 niedriger (0.55 vs 0.61) — Nachts weniger PM-Handel

### Aufgabe 1: Basis-Statistik (IS)

- Earnings OD with Gap: GapUp 40.3%, GapDn 53.2%
- Non-E OD with Gap: GapUp 46.7%, GapDn 47.0%
- **Kern-Finding:** OD-against+stark GapUp rest_drift: Earnings +0.34 vs Non-E -0.21 (Delta +0.55 ADR)
  - Earnings OD-against "reverst" oefter zurueck → Flush-Hypothese!
- Earnings Fill-Rate niedriger (19.4%) vs Non-E (25.6%)

### Aufgabe 2: Alle Top-Setups getrennt (IS)

- OD>0.5 with GapUp: Earnings fd=+1.01 vs Non-E fd=+1.06 → aehnlich stark
- OD>0.5 with GapDn: Earnings fd=-1.20 vs Non-E fd=-0.75 → Earnings overshoots!
- **Fade-Zellen (OD-against):** Earnings N zu gering fuer reliable Fade; Non-E zeigt funktionierenden Fade
- OD>0.7 with HiRV GapUp: stark bei beiden (+1.37 E, +1.53 NE)

### Aufgabe 3: Flush vs Fade Analyse (IS) — KERN-HYPOTHESE

**GapUp Flush-Rate: Earnings 30.0% vs Non-E 15.8% (2x!) — Hypothese bestaetigt!**
- GapDn: Kein Unterschied (15.7% vs 16.3%) — Effekt ist GapUp-spezifisch
- Post-Flush Verhalten (Earnings):
  - MFE: 1.33 ADR (vs Non-E 0.95) — Flush geht weiter
  - Close: 0.70 ADR ueber Open (vs Non-E 0.43)
- Flush-Praediktoren: Hoeheres RVOL (7.9 vs 5.3), groesserer Gap (4.08 vs 1.95), hoeheres Wick Ratio
- **Fazit:** Bei Earnings GapUp OD-against → NICHT faden! Flush wahrscheinlich.

### Aufgabe 4: 10:00-Reversal Prediction (IS)

- Gesamt-Reversal-Rate: ~26% (Full), ~36% (Partial)
- **Top-3 Praediktoren:** od_body_pct (r=-0.30), od_strength_abs (r=-0.24), wick_ratio (r=+0.16)
- AUC: 0.69 (Logistic Regression), 0.74 (Random Forest)
- Starke OD (>0.5 ADR): nur 11.4% Full-Reversal
- Schwache OD (<0.2 ADR): 37% Full-Reversal
- Earnings OD-against GapUp: 32.0% Reversal vs Non-E 26.2%
- Beste Combo: OD_LO + WICK_HI + Earnings = 40% Reversal
- is_earnings Effekt ist MODEST (Koeffizient 0.07, Feature Importance 0.006)

### Aufgabe 5: Post-Reversal Trade-Simulation (IS)

- **Post-Reversal Drift (Full Rev):**
  - Earnings: Drift->Gap +0.013, Drift->OD +0.129 (leicht Richtung OD)
  - Non-E: Drift->Gap -0.065, Drift->OD +0.034 (neutral)
- **Trade-Simulation (Reversal-Trade, Entry 10:00, gegen OD):**
  - SL_025/Tgt_025: WR=48.2%, EV=+0.001 (ca. Break-Even)
  - SL_050/Tgt_025: WR=60.7%, EV=+0.004 (leicht positiv)
  - Kein SL/Tgt-Kombi liefert statistisch signifikanten positiven EV
  - Earnings schlechter als Non-E bei Reversal-Trades
- **5c Vergleich (auf Full-Reversal-Tagen):**
  - Continuation (9:35, in OD): EV=-0.204 (schrecklich — OD reversed ja)
  - Fade (9:35, gegen Gap): EV=+0.005 (WR=31.4%)
  - Reversal (10:00, gegen OD): EV=0.000 (WR=27.9%)
  - Fade @ 9:35 minimal besser als Reversal @ 10:00

### Aufgabe 6: Non-Earnings Top-Setups (IS)

- **Non-E 3x3x3 Dreieck (PM x RV x Gap):**
  - GapUp Best Fade: PM_HI RV_MID GAP_LG fd=-0.52 N=38
  - GapUp Best Cont: PM_LO RV_HI GAP_LG fd=+0.42 N=89
- **Earnings Continuation:**
  - GapUp OD>0.5 with: fd=+1.01 N=48
  - GapDn OD>0.5 with: fd=-1.20 N=61

### Aufgabe 7: OOS-Validierung (2024-2026)

- OOS: 5,812 Zeilen (852 Earnings, 4,571 Non-E, 389 Unknown)
- **Q1 Flush-Rate OOS:** GapUp Earnings 12.5% vs Non-E 12.4% → **NICHT REPLIZIERT!**
  - IS hatte 30% vs 15.8% — OOS zeigt keinen Unterschied mehr
  - GapDn dagegen: Earnings 26.3% vs Non-E 11.9% → **Effekt verschoben auf GapDn?**
- **Q2 Fade-Drift OOS:** Earnings OD-ag fd stärker negativ (-1.34 vs -0.99 Non-E) → Earnings faden STAERKER
- **Q3 Continuation OOS:** Earnings GapUp +0.93 vs Non-E +0.87 → aehnlich (bestaetigt)
- **Q4 Reversal-Praediktoren OOS:**
  - od_strength_abs: r=-0.30*** → **BESTAETIGT**
  - od_body_pct: r=-0.29*** → **BESTAETIGT**
  - wick_ratio: r=+0.16*** → **BESTAETIGT**
  - vol_profile: r=+0.06*** → **BESTAETIGT**
  - is_earnings: r=-0.01, p=0.42 → NICHT signifikant (konsistent mit IS)
  - **OOS AUC: 0.71** (IS: 0.69) → **Reversal-Modell repliziert gut!**

### Haupt-Erkenntnisse

1. **Earnings-Flush-Hypothese (GapUp) hat sich OOS NICHT repliziert** — IS 30% vs 15.8%, OOS 12.5% vs 12.4%. Moeglicherweise Sample-spezifisch oder zeitabhaengig.
2. **Reversal-Praediktoren sind ROBUST OOS** — AUC 0.71, Top-Praediktoren alle signifikant. od_body_pct und od_strength sind die besten Signale.
3. **is_earnings ist KEIN nützlicher Reversal-Praediktor** — weder IS (modest) noch OOS (nicht signifikant).
4. **Continuation-Setups (OD>0.5 with) replizieren gut** — Earnings +0.93, Non-E +0.87 ADR OOS.
5. **Reversal-Trade (10:00) hat keinen positiven EV** — bestenfalls Break-Even. Fade @ 9:35 ist leicht besser.
6. **Die ERSTE Frage vor jedem Trade bleibt:** Wie stark ist die OD? (nicht: Ist es Earnings?)
7. **Entscheidungsbaum-Update:**
   - OD stark (>0.5 ADR) → Continuation, egal ob Earnings
   - OD schwach (<0.2 ADR) → Fade/Reversal wahrscheinlich (~37%), aber kein positiver Trade-EV
   - Wick Ratio + Volume Profile als zusaetzliche Reversal-Indikatoren

### Dateien
- Earnings-Fetch: scripts/d8_earnings_fetch.py → data/metadata/earnings_dates.parquet, metadata_v8.parquet
- Basis: scripts/d8_earnings_basis.py → results/d8_1_earnings_basis.txt
- Setup-Split: scripts/d8_setup_split.py → results/d8_2_setup_split.txt
- Flush-Analyse: scripts/d8_flush_analyse.py → results/d8_3_flush_analyse.txt, d8_3_flush_raw.parquet
- Reversal-Predict: scripts/d8_reversal_predict.py → results/d8_4_reversal_predict.txt, d8_4_reversal_raw.parquet
- Reversal-Trade: scripts/d8_reversal_trade.py → results/d8_5_reversal_trade.txt, d8_5_trade_sim.parquet
- Non-E Setups: scripts/d8_non_earnings_setups.py → results/d8_6_non_earnings_setups.txt
- OOS: scripts/d8_oos.py → results/d8_7_oos.txt
- Synthese: scripts/d8_synthese.py -> results/d8_synthese.txt

---

## D8.5 -- Reversal-Filter + Kontext-Parameter: Redundanz oder Synergie?

**Kernfrage:** Wenn OD > 0.5 ADR UND der Reversal-Filter bestanden ist (Body% > 50%, Wick < 0.35) -- bringen RVOL, Gap-Groesse und PM/RTH dann noch ZUSAETZLICHEN Wert?

**Datenbasis:** metadata_v8_5.parquet (88 Spalten, mit od_body_pct, od_wick_ratio, quality_high)
- IS: 4,254 Gapper, OOS: 5,811 Gapper
- QUALITY_HIGH: 43% aller Gapper, 85% der OD>0.5 Gapper
- QH + OD>0.5: N=650 IS, N=1,117 OOS

### Aufgabe 1: Feature-Berechnung

- od_body_pct: Median=0.472, Q25=0.249, Q75=0.681
- od_wick_ratio: Median=0.220, Q25=0.088, Q75=0.396
- Bei OD>0.5: Median Body%=0.729, Median Wick=0.093 (fast alle sind QH!)
- QUALITY_HIGH Definition: od_body_pct >= 0.50 AND od_wick_ratio < 0.35

### Aufgabe 2: Marginale Effekte (IS)

- RVOL_30 5-10x bei QH+OD>0.5 with GapUp: WR@0.50 = 51.1% (Bestes Bucket)
- PM/RTH30 < 10%: Konsistent bestes Bucket (43-49% WR@0.50)
- **Q_HIGH vs Q_LOW bei OD>0.5: KEIN klarer Vorteil fuer Q_HIGH** (IS Delta ~0%)

### Aufgabe 3: Kreuzungen (IS)

- **R2 bei QH+OD>0.5: 4.6%** (vs 0.7% fuer alle Gapper in D7.5) -- 6x mehr Varianz erklaert
- Einzelne Richtungen: with GapUp R2=23.4%, against GapDn R2=41.8%
- PM/RTH30 staerkster Koeffizient in der Regression
- Spread-Analyse: RVOL_30 und PM/RTH30 Spreads bleiben AEHNLICH nach QH-Filter
- **Kontext-Parameter sind NICHT redundant nach Reversal-Filter**

### Aufgabe 4: Trade-Simulation (IS)

- Entry 9:35, SL=0.25 ADR, Target 0.50 ADR
- L0 (OD>0.5 with): EV ~ 0 (marginal)
- L1 (+ QH): EV kaum besser (+0.004 bis +0.010)
- **L2c (+ PM/RTH30<10%): EV = +0.079 GapUp, +0.091 GapDn** (Bester Layer!)
- L2a (+ RVOL_30>5x): EV = +0.054 GapUp, +0.022 GapDn
- Against-Gap: Durchgehend NEGATIV (kein Setup)
- L3 (Doppelfilter): Kein Zusatz-EV, nur duenneres N

### Aufgabe 5: 9:35 vs 10:00 (IS) -- WICHTIGSTES ERGEBNIS

- **9:35 Entry ist KLAR besser als 10:00 Entry nach Reversal-Filter**
- PM/RTH30<10% GapUp: Entry@9:35 EV=+0.079, Entry@10:00 EV=-0.024 (Delta -0.103!)
- PM/RTH30<10% GapDn: Entry@9:35 EV=+0.091, Entry@10:00 EV=-0.017 (Delta -0.108!)
- Missed drift (9:35 bis 10:00): +0.09-0.12 ADR (signifikant!)
- **Schlussfolgerung: Bei OD>0.5 with_gap SOFORT einsteigen, nicht auf 10:00 warten**
- Das widerspricht D7.5 ("Warten lohnt sich") -- aber D7.5 hatte keinen QH-Filter

### Aufgabe 6: OOS-Validierung

**KERN-ERGEBNIS:**
- **Reversal-Filter: NICHT repliziert!**
  - IS: Q_HIGH WR@0.50=37.4% vs Q_LOW=38.6% (Delta -1.2%)
  - OOS: Q_HIGH WR@0.50=33.5% vs Q_LOW=39.1% (Delta -5.6%)
  - Q_LOW schlaegt Q_HIGH in ALLEN 4 Richtungs-Buckets OOS

- **Kontext-Parameter: REPLIZIERT!**
  - RVOL_30>5x: +13.1pp WR OOS (Staerkstes Signal!)
  - PM/RTH30<10%: +9.5pp WR OOS
  - Gap>2ADR: +8.4pp WR OOS
  - RVOL_5>5x: +5.4pp WR OOS (schwach)
  - PM/RTH5<50%: -4.1pp WR OOS (NEGATIV, nicht verwenden)

- **Trade-Sim OOS:**
  - GapDn + QH + PM/RTH30<10%: WR@0.50=46.9%, EV=+0.055 (N=32)
  - GapDn + QH + RVOL_30>5x: WR@0.50=45.9%, EV=+0.047 (N=74)
  - GapUp + QH + RVOL_30>5x: WR@0.50=46.7%, EV=+0.011 (N=90)
  - OHNE QH + RVOL_30>5x: EV OOS GapUp=+0.018, GapDn=+0.057 (AEHNLICH wie mit QH!)

- **Top IS-Zellen OOS:**
  - RVOL_30 5-10x with GapUp: IS WR=51.1% -> OOS WR=47.9% (TEILWEISE REPLIZIERT)
  - PM/RTH30<10% with GapUp: IS WR=38.1% -> OOS WR=40.0% (REPLIZIERT)
  - PM/RTH30<10% with GapDn: IS WR=43.2% -> OOS WR=43.8% (REPLIZIERT)

### Aufgabe 7: Synthese

**REDUNDANZ-MATRIX:**

| Parameter | WR-Lift OOS | Bewertung |
|-----------|-------------|-----------|
| RVOL_30>5x | +13.1pp | EXTRA-INFO (staerkstes!) |
| PM/RTH30<10% | +9.5pp | EXTRA-INFO |
| Gap>2ADR | +8.4pp | SCHWACH EXTRA |
| RVOL_5>5x | +5.4pp | SCHWACH EXTRA |
| PM/RTH5<50% | -4.1pp | REDUNDANT |
| REVERSAL-FILTER | -5.6pp | NICHT NUETZLICH |

**FINALE EMPFEHLUNG:**
1. Reversal-Filter (Body%, Wick) bei OD>0.5 **streichen** -- er ist tautologisch (85% der OD>0.5 sind bereits QH)
2. RVOL_30>5x und PM/RTH30<10% **beibehalten** als Qualitaetsmarker (+9-13pp WR OOS)
3. Aber: Diese sind erst um 10:00 bekannt. Da Warten -0.10 EV kostet -> **retrospektive Marker, NICHT Entry-Filter**
4. Entry um 9:35 bei OD>0.5 with_gap, SL=0.25 ADR
5. EV bleibt oekonomisch marginal (+0.01-0.09 ADR pro Trade)

### Dateien
- Features: scripts/d8_5_features.py -> data/metadata/metadata_v8_5.parquet, results/d8_5_features.txt
- Marginale Effekte: scripts/d8_5_marginal.py -> results/d8_5_marginal.txt
- Kreuzungen: scripts/d8_5_kreuzung.py -> results/d8_5_kreuzung.txt
- Trade-Sim: scripts/d8_5_trade_sim.py -> results/d8_5_trade_sim.txt
- Timing: scripts/d8_5_timing.py -> results/d8_5_timing.txt
- OOS: scripts/d8_5_oos.py -> results/d8_5_oos.txt
- Synthese: scripts/d8_5_synthese.py -> results/d8_5_synthese.txt
