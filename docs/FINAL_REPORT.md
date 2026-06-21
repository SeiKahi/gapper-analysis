# Arc 1 Report (Superseded): VWAP Gapper ML Edge Discovery

> ⚠️ **Status: Arc 1, superseded.** This report covers the project's first approach, ML on VWAP-cross entries, and its honest null result. A later bias-controlled re-evaluation (runs `d14_bias_free` and `d15`) reached different conclusions on a different strategy family. See the [README](../README.md) under "Key results" for the current status. This file is kept on purpose, as part of the honest research record.

## Executive Summary

Wir haben 10,161 Gapper-Tage (Aktien mit >4% Gap at Open) mit 1-Minuten VWAP-Daten systematisch auf tradeable Edges untersucht. **Kernergebnis: KEIN robuster, tradeabler Edge gefunden.** Der auf Haelfte 1 (2021-2023) identifizierte GapDn Mean-Reversion Edge (+0.029 ADR genuine) **repliziert NICHT auf Out-of-Sample Daten (2024-2026)**. OOS AvgPnL fiel um 85% auf +0.003 ADR (P=0.24, nicht signifikant, CI enthalt Null). Die Richtungsasymmetrie (GapDn > GapUp) hat sich komplett umgekehrt. Das VWAP-Cross-Signal und das ML-Modell sind ebenfalls wertlos. **Fazit: Die in-sample Ergebnisse waren eine Kombination aus strukturellem SL/Target-Artefakt, guenstigem Marktregime und Sampling Noise.**

---

## 1. Datensatz & Methodik

**Datensatz:**
- 10,161 Gapper-Tage, Zeitraum 2021-02 bis 2026-02
- VWAP-Dateien: 1-min Close + Volume + VWAP + StdDev-Baender (kein OHLC)
- Metadata: 63 Spalten inkl. gap_size, adr_10, sector, rvol_30 etc.

**Split (nur Haelfte 1 verwendet):**
- Train: 2021-02 bis 2022-12 (~3,054 Gapper)
- Validation: 2023-01 bis 2023-12 (~1,200 Gapper)
- Test/OOS: 2024-2026 -- GESPERRT

**Pipeline-Design:**
- 6 Szenarien: GapUp/GapDn x 2s_Fade / 1s_Fade / VWAP_Break
- 2 Entry-Typen: E1 (sofort bei Close-Cross), E2 (naechste Kerze bestaetigt)
- 4 SL-Varianten: SL3 (0.10 ADR), SL4 (0.20 ADR), SL5 (0.35 ADR), SL6 (Level-basiert)
- 6 Target-Varianten: T1_next, T3_025, T3_050, T4_15R, T4_2R, T4_3R
- 288 Kombinationen total, 1,570,872 Trades generiert
- Timeout: 240 Bars (4h), Cooldown: 5 Bars

**ML-Modelle:**
- XGBoost: Train AUC=0.7966, Val AUC=0.7407 (Gap 0.056, akzeptabel)
- LightGBM: Train AUC=0.8719, Val AUC=0.7336 (Gap 0.138, overfitted)

---

## 2. ML Pipeline Ergebnisse (Haelfte 1: 2021-2023)

### Top-5 Setups (Validation 2023)

| # | Szenario | Entry | SL | Target | N | WR | AvgPnL (ADR) |
|---|----------|-------|----|--------|---|-----|-------------|
| 1 | GapDn_2s_Fade | E2 | SL5_035 | T4_3R | 345 | 8.7% | +0.0739 |
| 2 | GapDn_VWAP_Brk | E2 | SL5_035 | T4_3R | 1120 | 6.5% | +0.0563 |
| 3 | GapDn_VWAP_Brk | E1 | SL5_035 | T4_3R | 2380 | 6.0% | +0.0546 |
| 4 | GapDn_2s_Fade | E2 | SL5_035 | T4_2R | 345 | 15.4% | +0.0521 |
| 5 | GapDn_VWAP_Brk | E1 | SL5_035 | T4_2R | 2380 | 14.5% | +0.0519 |

**Muster:** 9 von 10 Top-Setups sind GapDn (Long). SL5 (0.35 ADR) dominiert. R-Multiple Targets (T4) liefern beste Expectancy.

### Feature Importance (XGBoost Top 5)
1. tgt_id (0.148) -- trivial, kodiert Target-Methode
2. minutes_since_open (0.095) -- echter Markt-Feature
3. e2_confirmed (0.079) -- DATENLECK fuer E1-Trades
4. sl_dist_adr (0.069) -- Setup-Parameter
5. entry_id (0.065) -- Setup-Parameter

~34% der Importance stammt aus Setup-Parametern, nicht aus Markt-Features.

### Szenario-Ueberblick (Train vs Val)
| Szenario | Train PnL | Val PnL | Bewertung |
|----------|-----------|---------|-----------|
| GapDn_VWAP_Brk | +0.0125 | +0.0288 | ROBUST |
| GapDn_2s_Fade | +0.0009 | +0.0202 | Positiv |
| GapUp_2s_Fade | +0.0031 | +0.0067 | Schwach |
| GapUp_VWAP_Brk | +0.0115 | -0.0004 | ZERFALL |

---

## 3. Kritische Validierung

### 3.1 Random Entry Baseline
- Random Entry SL5+T4_3R: +0.0249 ADR vs Pipeline: +0.0132 ADR
- GapDn Pipeline (+0.0245) vs GapDn Random (+0.0243): Delta = +0.0002 = KEIN Unterschied
- **Fazit: VWAP-Cross Entry hat NULL Mehrwert.** Edge liegt in Gap-Richtung + SL/Target-Struktur, nicht im Timing.

### 3.2 e2_confirmed Datenleck
- e2_confirmed ist Look-Ahead fuer E1-Trades (69% aller Trades)
- AUC-Drop nach Fix: 0.7407 -> 0.7124 (-0.028)
- HC-Trades (>0.6) PnL: +0.0192 (mit Leak) -> -0.0159 (ohne Leak)
- **Fazit: ML-Modell ist WERTLOS fuer Trade-Selektion.** WR steigt sogar (73%), aber PnL wird negativ -- Modell selektiert nicht die Tail-Gewinner.

### 3.3 Slippage Stress Test
- Top-6 GapDn-Setups (SL5_035) ueberleben 0.10 ADR Slippage
- #3 GapDn_VWAP_Brk E1 SL5 T4_3R: +0.0546 -> +0.0214 bei 0.10 ADR Slippage
- GapDn_VWAP_Brk Szenario (N=21000): +0.0395 bei slip=0, +0.0050 bei slip=0.10
- Setups mit engerem SL (SL4, SL6) sterben bei 0.07-0.10 ADR
- **Fazit: SL5-basierte GapDn-Setups sind slippage-robust.**

### 3.4 Bootstrap Confidence Intervals
- Alle Top-5 Mean PnL: 95% CI schliesst Null NICHT ein (signifikant positiv)
- #3 (N=2380): CI [+0.0383, +0.0707], P<0.001 -- besteht auch Bonferroni
- Median PnL aller Setups: CI enthaelt Null (nicht signifikant)
- **Fazit: Edge ist rechtsschief. ~94% der Trades flat/negativ, ~6% treffen 3R. Mean ist signifikant, Median nicht.**

### 3.5 Random Walk Simulation (Killer-Test)
- Kalibrierter Random Walk (drift=0, sigma=0.001039): SL5+T4_3R AvgPnL = +0.0101
- Reale GapDn-Daten (Val 2023): AvgPnL = +0.0396
- **Strukturelles Artefakt: +0.0101 ADR (26% des Edges)**
- **Genuiner Residual: +0.0294 ADR (74% des Edges)**
- Kritisch: Real WR=6.1% vs RW WR=0.6% -- GapDn-Aktien erreichen 3R-Target 10x oefter als Random Walk
- GapUp Short: -0.0006 ADR (unter RW-Artefakt) -- Richtungsasymmetrie kann NICHT durch Artefakt erklaert werden

### 3.6 Timeout PnL Decomposition
- GapDn SL5 T4_3R (N=7920): Target +0.0636, SL -0.1218, Timeout +0.0977
- Ohne Timeouts: AvgPnL = -0.1423 (stark negativ)
- Timeout-Trades: 59.1%, davon 69% positiv, Mean +0.1652 ADR, Median +0.1147 ADR
- **Fazit: Edge ist primaer timeout-getrieben.** GapDn-Aktien driften intraday zurueck (Mean Reversion), was Timeout-Trades positiv macht.

### 3.7 Naked Direction Test
- GapDn Long (Open-to-Close, kein SL/Target): Mean +0.0346 ADR, CI [-0.081, +0.158] -- NICHT signifikant
- GapUp Short (Open-to-Close): Mean -0.1035 ADR -- signifikant NEGATIV
- **Fazit: Kein nackter GapDn-Vorteil ohne SL/Target-Framework.** Der Edge entsteht erst durch die Kombination von Richtungsfilter + weitem SL + R-Multiple Target + Timeout.

### 3.8 Jahres-Stabilitaet
- GapDn SL5 T4_3R: 2021 +0.0292, 2022 +0.0070, 2023 +0.0396 -- alle 3 Jahre positiv
- GapDn_VWAP_Brk SL5 T4_3R: 2021 +0.0578, 2022 +0.0152, 2023 +0.0551 -- stabil
- 2022 ist das schwaechste Jahr (Baermarkt), aber immer noch positiv
- GapUp SL5 T4_3R: 2023 bereits negativ (-0.0006) -- nicht stabil
- Monatlich: 60% der Monate positiv (GapDn gesamt), 66% (GapDn_VWAP_Brk)

---

## 4. OOS Validierung (Haelfte 2: 2024-2026)

### Daten
- 5,812 Gapper-Tage (GapUp: 3,224 / GapDn: 2,588)
- 52,305 Trades simuliert, 5,812 Naked Direction Trades
- Jahre: 2024 (1,922), 2025 (3,451), 2026 (439)

### Kern-Hypothese: NICHT BESTAETIGT

| Metrik | Haelfte 1 (2021-2023) | Haelfte 2 (2024-2026) |
|--------|---------------------|---------------------|
| AvgPnL | +0.0228 ADR | **+0.0034 ADR** |
| 95% CI | [+0.019, +0.027] | **[-0.006, +0.013]** |
| P(<=0) | 0.0000 | **0.2385** |
| WR | 5.0% | 6.0% |
| SL-Hit Rate | ~30% | **41.7%** |

- P-Wert 0.24 — weit entfernt von Bonferroni-Schwelle (0.0017)
- CI enthaelt Null — kein signifikanter Edge
- AvgPnL (+0.0034) liegt UNTER dem Random-Walk-Artefakt (+0.01)
- SL-Hit Rate von ~30% auf 41.7% gestiegen — Edge wird aufgefressen

### Richtungsasymmetrie: KOMPLETT UMGEKEHRT

| Richtung | Haelfte 1 PnL | Haelfte 2 PnL |
|----------|-------------|-------------|
| GapDn Long | +0.0228 | +0.0034 (nicht sig.) |
| GapUp Short | +0.0086 | **+0.0331** (P<0.001) |

GapUp Short ist auf Haelfte 2 signifikant positiv — aber dies war NICHT die vorregistrierte Hypothese und kann daher nicht als bestaetigter Edge gelten (Post-hoc Finding).

### Jahres-Degradation (GapDn SL5 T4_3R)

| Jahr | AvgPnL | P(<=0) |
|------|--------|--------|
| 2021 | +0.0292 | 0.0000 |
| 2022 | +0.0070 | 0.0169 |
| 2023 | +0.0396 | 0.0000 |
| 2024 | +0.0138 | 0.0505 |
| 2025 | +0.0044 | 0.2429 |
| **2026** | **-0.0582** | **0.9996** |

Der Edge schrumpft monoton und wird 2026 stark negativ.

### OOS-Verdikt
**Der GapDn Mean-Reversion Edge existiert NICHT als stabiles, tradeables Phaenomen.** Die In-Sample Ergebnisse (2021-2023) waren eine Kombination aus:
1. Strukturellem SL/Target-Artefakt (~26% = +0.01 ADR)
2. Guenstigem Marktregime (2021 + 2023 besonders stark)
3. Sampling Noise bei rechtsschiefem Payoff-Profil

---

## 5. Konkrete Trading-Strategie

**KEINE STRATEGIE EMPFOHLEN.**

Der Edge hat die OOS-Validierung nicht bestanden. Die auf Haelfte 1 gefundenen Muster sind nicht stabil genug fuer Live-Trading.

**Moegliche Weiterentwicklung (falls gewuenscht):**
- GapUp Short auf 2024-2026 zeigt Potential (+0.0331 ADR, P<0.001) — muesste aber auf NEUEN OOS-Daten validiert werden
- Sektor-spezifische Analyse (z.B. nur Biotech oder nur Small Caps)
- Volumen- und Catalyst-Filter koennten die Population einschraenken
- Kuerzere Timeout-Fenster (60min statt 240min) testen
- Kombination mit Makro-Filtern (VIX-Level, SPY-Return)

---

## 6. Was NICHT funktioniert

1. **VWAP-Cross Entry**: Kein Mehrwert gegenueber Random Entry (Delta = +0.0002 ADR)
2. **ML Trade-Selektion**: Nach e2_confirmed Leak-Fix hat HC-Filter NEGATIVE PnL (-0.0159)
3. **GapUp Short**: Negativ oder flat auf Validation (-0.0006 ADR), nicht stabil ueber Jahre
4. **Enge SL (SL3, SL4) mit R-Multiple Targets**: Sterben bei 0.07-0.10 ADR Slippage
5. **T1_next mit weitem SL**: Schlechteste Expectancy (weiter SL + kleines Target)
6. **GapDn_1s_Fade**: Marginal/instabil, 2022 negativ
7. **LightGBM**: Stark overfitted (Gap 0.138), kein Mehrwert gegenueber XGBoost

---

## 7. Offene Fragen & Moegliche Naechste Schritte

1. **GapUp Short Hypothese**: Auf 2024-2026 signifikant positiv (+0.033), aber post-hoc — braucht eigene OOS-Validierung
2. **Sektor-Analyse**: Biotech-GapDn vs Tech-GapDn — unterschiedliche Bounce-Qualitaet?
3. **Volumen-Filter**: RVOL > 2x oder Volume-Spike als Entry-Filter nicht getestet
4. **Kuerzere Timeouts**: 60min oder 120min statt 240min — aendert das die Dynamik?
5. **Makro-Filter**: VIX-Level oder SPY-Tagesreturn als Regime-Filter
6. **Survivorship Bias**: Unklar ob delistete Aktien im Datensatz enthalten sind
7. **Fat-Tail-Korrektur**: Random Walk mit Student-t statt Gaussian wuerde RW-Artefakt genauer schaetzen
8. **Intrabar-Slippage**: Close-Only SL-Pruefung ueberschaetzt PnL systematisch (kein High/Low)
9. **Warum die Richtungsumkehr?**: 2021-2023 favorisierte GapDn Long, 2024-2026 GapUp Short — Regime-Wechsel?

---

## Appendix: Alle Dateien und Scripts

### Ergebnisse (`results/`)
| Datei | Beschreibung |
|-------|-------------|
| `all_trades_v3.parquet` | 1.57M Trades, alle Kombinationen |
| `setup_rankings_v3_val.csv` | Ranking aller 288 Setups nach Val-PnL |
| `feature_importance_xgboost_v3.csv` | XGBoost Feature Importance |
| `feature_importance_lightgbm_v3.csv` | LightGBM Feature Importance |
| `ml_pipeline_v3_results.txt` | Volle Pipeline-Ausgabe |
| `ml_summary_v3.json` | Pipeline-Zusammenfassung |
| `quant_analysis_round1.txt` | Statistische Analyse Runde 1 |
| `trader_analysis_round1.txt` | Trader-Perspektive Runde 1 |
| `devils_advocate_round1.txt` | Kritische Pruefung Runde 1 |
| `quant_round2.txt` | Statistische Bewertung Runde 2 |
| `devils_advocate_round2.txt` | Kritische Pruefung Runde 2 |
| `test_random_baseline.txt` | Random Entry vs Pipeline |
| `test_slippage_stress.txt` | Slippage-Robustheit |
| `test_e2_leakage.txt` | e2_confirmed Leak-Fix |
| `test_bootstrap_ci.txt` | Bootstrap Konfidenzintervalle |
| `test_random_walk.txt` | Random Walk Simulation |
| `test_timeout_decomp.txt` | Timeout PnL Zerlegung |
| `test_naked_direction.txt` | Nackter Richtungstest |
| `test_yearly_stability.txt` | Jahres-Stabilitaet |
| `oos_validation_half2.txt` | OOS Validierung Haelfte 2 (2024-2026) |

### Scripts (`scripts/`)
| Datei | Beschreibung |
|-------|-------------|
| `ml_pipeline_v3.py` | Haupt-Pipeline (Trade-Generierung + ML) |
| `test_random_entry_baseline.py` | Random Entry Baseline Test |
| `test_slippage_stress.py` | Slippage Stress Test |
| `test_e2_leakage_fix.py` | e2_confirmed Leak-Fix |
| `test_bootstrap_ci.py` | Bootstrap CI Berechnung |
| `test_random_walk_v3.py` | Kalibrierte Random Walk Simulation |
| `test_timeout_decomposition.py` | Timeout PnL Decomposition |
| `test_naked_direction.py` | Nackter Richtungstest |
| `test_yearly_stability.py` | Jahres-Stabilitaet |
| `oos_validation_half2.py` | OOS Validierung auf Haelfte 2 |

### Modelle (`models/`)
| Datei | Beschreibung |
|-------|-------------|
| `xgb_v3.pkl` | XGBoost v3 (Val AUC 0.7407) |
| `lgb_v3.pkl` | LightGBM v3 (Val AUC 0.7336) |

### Dokumentation (`docs/`)
| Datei | Beschreibung |
|-------|-------------|
| `FINDINGS_LOG.md` | Fortlaufendes Journal aller Erkenntnisse |
| `PROJECT_BRIEF.md` | Projekt-Briefing und bisherige Erkenntnisse |
| `ENTRY_DEFINITIONS.md` | Entry/SL/Target Definitionen |
| `DATA_DICTIONARY.md` | Datenstruktur der Parquet-Dateien |
