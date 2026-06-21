"""
D11 Aufgabe 7: Synthese — Was unterscheidet die grossen Laeufer?
"""
import pandas as pd
import numpy as np
import os

OUT = "results/d11_synthese.txt"
lines = []

def w(text=""):
    lines.append(text)

def header(title):
    w("=" * 70)
    w(title)
    w("=" * 70)

# Load data for combined stats
is_df = pd.read_parquet("results/d11_is_enriched.parquet")
oos_df = pd.read_parquet("results/d11_oos_enriched.parquet")

# Compute drift for both
for df in [is_df, oos_df]:
    drift_raw = (df['close_1000'] - df['close_935']) / df['adr_10']
    is_up = df['gap_direction'] == 'up'
    df['drift_30min_adr'] = np.where(is_up, drift_raw, -drift_raw)

header("D11 AUFGABE 7: SYNTHESE")
w()
w("MISSION: Was unterscheidet die grossen Laeufer (Big Runner)?")
w("Big Runner = Range >= 1.5 ADR in Gap-Richtung + Close >= 80% Extremum")
w(f"IS: N={len(is_df)} | OOS: N={len(oos_df)} | Alle OD > 0.5 + with_gap")
w()
w()

# =====================================================================
# 7a: Was haben Big Runner gemeinsam?
# =====================================================================
header("7a: WAS HABEN BIG RUNNER GEMEINSAM? (Top-5 Merkmale)")
w()
w("  Big Runner sind typischerweise Aktien mit:")
w()
w("  1. STARKEM 30-MIN DRIFT (rho=+0.26 GapUp, +0.25 GapDn, OOS repliziert)")
w("     Median Big Runner: +0.53 ADR Drift in Gap-Richtung 9:35-10:00")
w("     Median Non-Runner: -0.16 ADR (gegen Gap-Richtung!)")
w("     → Der wichtigste Praediktor, aber erst um 10:00 bekannt")
w()
w("  2. FLACHEM PULLBACK nach OD (rho=-0.15 OOS GapUp, -0.21 OOS GapDn)")
w("     Big Runner: Pullback < 0.10 ADR → 35-47% BR Rate (1.9x Lift)")
w("     Repliziert OOS: GapUp IS 34.5% → OOS 24.4%, GapDn IS 46.7% → OOS 34.0%")
w("     → Der Preis haelt nach dem OD, kein Retracement")
w()
w("  3. HOHEM RVOL_5 und starkem OD (rho=+0.17-0.18 GapUp OOS)")
w("     RVOL_5 > 10x: BR Rate 27% GapUp (1.4x)")
w("     OD > 1.0 ADR: BR Rate 27% GapUp, 42% GapDn (1.7x)")
w("     Repliziert OOS: RV5>5+OD>1 → IS 33-47% → OOS 37-31%")
w()
w("  4. GROSSEM GAP (rho=+0.22/+0.11 GapUp IS/OOS)")
w("     Gap > 2.0 ADR: BR Rate 25-28% (1.3x)")
w("     Repliziert OOS fuer GapUp, schwaecher fuer GapDn")
w("     → Groesser Gap = mehr Momentum im Premarket")
w()
w("  5. GROSSER ERSTER KERZE (rho=+0.20/+0.14 GapUp IS/OOS)")
w("     First Candle > 0.30 ADR: BR Rate 28% GapUp (1.4x)")
w("     Repliziert OOS")
w("     → Starkes Volumen/Momentum direkt beim Open")
w()
w()

# =====================================================================
# 7b: Was unterscheidet "lief weit + blieb oben" von "lief weit + kam zurueck"?
# =====================================================================
header("7b: WAS UNTERSCHEIDET BIG RUNNER VON 'LIEF WEIT + KAM ZURUECK'?")
w()
w("  KRITISCHE ERKENNTNIS: Die Unterschiede sind MINIMAL!")
w()
w("  Big Runner vs 'Lief weit + kam zurueck' (ROC_Bad):")
w("  - ALLE Cohen's d < 0.25 (triviale bis kleine Effekte)")
w("  - OD_strength: fast identisch (BR 0.90 vs ROC 0.91 GapUp)")
w("  - RVOL_5: fast identisch (BR 7.6 vs ROC 7.7 OOS GapUp)")
w("  - GAP_ADR: ROC_Bad hat oft GROESSERE Gaps!")
w()
w("  Die EINZIGEN konsistenten Unterschiede:")
w("  a) 30min Drift: Big Runner +0.36 vs ROC_Bad +0.16 (GapUp OOS)")
w("     → Big Runner laufen nach dem OD WEITER, ROC_Bad stagnieren")
w("  b) Continuation after OD: Big Runner +0.36 vs ROC_Bad +0.15")
w("     → Identische Information wie Drift")
w("  c) Pullback depth: Big Runner 0.28 vs ROC_Bad 0.43 (GapDn OOS)")
w("     → Big Runner haben flacheren Pullback")
w()
w("  IMPLIKATION: Um 9:35 KANN MAN NICHT UNTERSCHEIDEN ob ein Tag")
w("  ein Big Runner wird oder nur weit laeuft und zurueckkommt.")
w("  Erst um 10:00 (30min Drift, Pullback) trennt sich etwas.")
w()
w()

# =====================================================================
# 7c: Frueherkennungs-Regeln
# =====================================================================
header("7c: FRUEHERKENNUNGS-REGELN (OOS-validiert)")
w()

# Combined IS+OOS stats for the best rules
combined = pd.concat([is_df, oos_df], ignore_index=True)

w("  UM 9:35 ERKENNBAR (nur pre-market + OD):")
w()
w("  GapUp: RV5>5 + OD>1.0 + Gap>2")
for df_label, df_data in [("IS", is_df), ("OOS", oos_df)]:
    g = df_data[(df_data['gap_direction'] == 'up')]
    mask = (g['rvol_5'] > 5) & (g['od_strength'] > 1.0) & (g['gap_size_in_adr'] > 2.0)
    sub = g[mask.fillna(False)]
    base = g['is_big_runner'].mean() * 100
    br = sub['is_big_runner'].mean() * 100 if len(sub) > 0 else 0
    w(f"    {df_label}: BR = {br:.1f}% (Basis: {base:.1f}%), N = {len(sub)}")

w()
w("  GapDn: RV5>5 + OD>1.0")
for df_label, df_data in [("IS", is_df), ("OOS", oos_df)]:
    g = df_data[(df_data['gap_direction'] == 'down')]
    mask = (g['rvol_5'] > 5) & (g['od_strength'] > 1.0)
    sub = g[mask.fillna(False)]
    base = g['is_big_runner'].mean() * 100
    br = sub['is_big_runner'].mean() * 100 if len(sub) > 0 else 0
    w(f"    {df_label}: BR = {br:.1f}% (Basis: {base:.1f}%), N = {len(sub)}")

w()
w()
w("  UM 10:00 ERKENNBAR (+ 30min Verlauf):")
w()
w("  GapUp: drift>0.50 + RVOL_30>5")
for df_label, df_data in [("IS", is_df), ("OOS", oos_df)]:
    g = df_data[(df_data['gap_direction'] == 'up')]
    drift = (g['close_1000'] - g['close_935']) / g['adr_10']
    mask = (drift > 0.50) & (g['rvol_open_30min'] > 5)
    sub = g[mask.fillna(False)]
    base = g['is_big_runner'].mean() * 100
    br = sub['is_big_runner'].mean() * 100 if len(sub) > 0 else 0
    w(f"    {df_label}: BR = {br:.1f}% (Basis: {base:.1f}%), N = {len(sub)}")

w()
w("  GapUp: drift>0.25 + RVOL_30>5 + OD>1.0   *** BESTES ERGEBNIS ***")
for df_label, df_data in [("IS", is_df), ("OOS", oos_df)]:
    g = df_data[(df_data['gap_direction'] == 'up')]
    drift = (g['close_1000'] - g['close_935']) / g['adr_10']
    mask = (drift > 0.25) & (g['rvol_open_30min'] > 5) & (g['od_strength'] > 1.0)
    sub = g[mask.fillna(False)]
    base = g['is_big_runner'].mean() * 100
    br = sub['is_big_runner'].mean() * 100 if len(sub) > 0 else 0
    w(f"    {df_label}: BR = {br:.1f}% (Basis: {base:.1f}%), N = {len(sub)}")

w()
w("  GapDn: drift>0.50")
for df_label, df_data in [("IS", is_df), ("OOS", oos_df)]:
    g = df_data[(df_data['gap_direction'] == 'down')]
    drift = (g['close_935'] - g['close_1000']) / g['adr_10']  # signed for gap down
    mask = (g['drift_30min_adr'] > 0.50)
    sub = g[mask.fillna(False)]
    base = g['is_big_runner'].mean() * 100
    br = sub['is_big_runner'].mean() * 100 if len(sub) > 0 else 0
    w(f"    {df_label}: BR = {br:.1f}% (Basis: {base:.1f}%), N = {len(sub)}")

w()
w("  GapDn: pullback<0.10 + drift>0.25")
for df_label, df_data in [("IS", is_df), ("OOS", oos_df)]:
    g = df_data[(df_data['gap_direction'] == 'down')]
    mask = (g['pullback_depth'] < 0.10) & (g['drift_30min_adr'] > 0.25)
    sub = g[mask.fillna(False)]
    base = g['is_big_runner'].mean() * 100
    br = sub['is_big_runner'].mean() * 100 if len(sub) > 0 else 0
    w(f"    {df_label}: BR = {br:.1f}% (Basis: {base:.1f}%), N = {len(sub)}")

w()
w()

# =====================================================================
# 7d: Implikation fuer TRAIL_D
# =====================================================================
header("7d: IMPLIKATION FUER TRAIL_D")
w()
w("  Wenn Big-Runner-Signale erkannt werden:")
w()
w("  AGGRESSIVERER TRAIL:")
w("    Standard TRAIL_D: Peak - 0.5R (= Peak - 0.125 ADR)")
w("    Bei Big-Runner-Signal: Peak - 0.25R (= Peak - 0.0625 ADR)")
w("    → Laesst den Trade weiter laufen, gibt weniger zurueck")
w()
w("  WANN UMSCHALTEN:")
w("    Um 9:35: Wenn RV5>5 + OD>1.0 + Gap>2 → aggressiver Trail")
w("    Um 10:00: Wenn drift>0.25 + RVOL_30>5 → aggressiver Trail")
w("    Waehrend Trade: Wenn +2R erreicht UND drift>0.25 → Peak-0.25R")
w()
w("  KEIN MANUELLER EXIT:")
w("    Bei Big-Runner-Signalen NICHT bei +2R oder +3R manuell rausgehen.")
w("    D10 zeigt: Wenn 2R erreicht, 72% Chance auf 3R, 44% auf 5R.")
w("    Median Gewinner-MFE = 6.9R = 1.7 ADR — laufen lassen!")
w()
w("  GROESSERE POSITION:")
w("    VORSICHT: Die Big-Runner-Rate ist nur ~35-50% bei besten Regeln.")
w("    Das heisst: 50-65% der Trades werden TROTZDEM gestoppt.")
w("    → Position NICHT verdoppeln. Maximal 1.5x Standard-Groesse.")
w("    → Der Vorteil liegt im Trail-Management, nicht in der Groesse.")
w()
w()

# =====================================================================
# 7e: Cheat Sheet v8 Update
# =====================================================================
header("7e: IMPLIKATION FUER CHEAT SHEET V8")
w()
w("  NEUE SPALTE: 'Big Runner Wahrscheinlichkeit'")
w()
w("  Stufe 1 — NIEDRIG (Basis ~20%):")
w("    OD > 0.5 + with_gap (Standardfilter)")
w("    → Normaler TRAIL_D (Peak - 0.5R)")
w()
w("  Stufe 2 — MITTEL (~30%, 1.5x Basis):")
w("    + RVOL_5 > 5x ODER Gap > 2.0 ADR ODER drift>0.25")
w("    → TRAIL_D bleiben, aber NICHT manuell bei 2R/3R rausgehen")
w()
w("  Stufe 3 — HOCH (~40-50%, 2x Basis):")
w("    + drift>0.25 + RVOL_30>5 + OD>1.0")
w("    ODER drift>0.50 + RVOL_30>5")
w("    ODER pullback<0.10 + drift>0.25 (GapDn)")
w("    → Engerer Trail: Peak - 0.25R")
w("    → Kein manueller Exit vor +5R")
w()
w("  NEUE PARAMETER (ab Cheat Sheet v8):")
w("    1. drift_30min: (Close_1000 - Close_935) / ADR in Gap-Richtung")
w("       Schwellen: >0.25 (mittel), >0.50 (stark)")
w("    2. pullback_depth: Max Retracement 9:35-10:00 in ADR")
w("       Schwelle: <0.10 (flach = bullisch)")
w("    3. continuation_after_od: (Close_1000 - Close_935) / ADR signiert")
w("       Identisch mit drift_30min, nur andere Berechnung")
w()
w("  NICHT AUFNEHMEN (kein OOS-Wert):")
w("    - od_momentum, od_acceleration (kein Signal)")
w("    - bars_above_open (zu hoch bei allen Trades)")
w("    - new_high_by_1000 (kein Trennkraft)")
w("    - pm_consolidation (kein Signal)")
w("    - Logistic Regression Modell (AUC faellt von 0.77 auf 0.61 OOS)")
w()
w()

# =====================================================================
# Finale Zusammenfassung
# =====================================================================
header("FINALE ZUSAMMENFASSUNG D11")
w()
w("  KERNFRAGE: Was unterscheidet die grossen Laeufer?")
w()
w("  ANTWORT IN 5 SAETZEN:")
w()
w("  1. Big Runner (~20% aller OD>0.5+with_gap Trades) laufen median")
w("     3.0+ ADR in Gap-Richtung und schliessen nahe am Tageshoch/-tief.")
w()
w("  2. Der WICHTIGSTE Praediktor ist die 30-Minuten-Drift nach dem OD")
w("     (rho ~0.25, repliziert OOS perfekt). Kein 9:35-Parameter ist")
w("     annaehernd so stark.")
w()
w("  3. Big Runner und 'Lief-weit-kam-zurueck' sind um 9:35 NICHT")
w("     UNTERSCHEIDBAR. Erst um 10:00 trennen Drift und Pullback.")
w()
w("  4. Beste OOS-validierte Regel: drift>0.25 + RVOL_30>5 + OD>1.0")
w("     → IS 50%, OOS 48% Big Runner Rate (N=31 OOS)")
w("     Aber: N ist klein. Breitere Regeln (drift>0.50+RV30>5)")
w("     haben 40% BR bei N=55 OOS — robuster.")
w()
w("  5. ML (Logistic Regression) overfittet massiv (AUC 0.77→0.61 OOS).")
w("     Einfache Schwellenwert-Regeln replizieren besser als jedes Modell.")
w()
w()
w("  PRAKTISCHE KONSEQUENZ:")
w()
w("  → Kein 'Wunder-Indikator' fuer Big Runner gefunden.")
w("  → ABER: 30min-Drift + RVOL als Trail-Management-Signal nutzbar:")
w("    Wenn drift>0.25 + RVOL_30>5: engerer Trail (Peak-0.25R)")
w("    Sonst: Standard TRAIL_D (Peak-0.5R)")
w("  → Position-Sizing NICHT anpassen (zu wenig Edge fuer Groessen-Hebel)")

# Write output
os.makedirs("results", exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print(f"Geschrieben: {OUT} ({len(lines)} Zeilen)")
