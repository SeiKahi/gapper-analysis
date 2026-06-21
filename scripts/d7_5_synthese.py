"""
Durchlauf 7.5 -- Aufgabe 9: Synthese-Bericht
Liest alle Ergebnisse und erstellt den finalen Report.
"""
import pandas as pd
import numpy as np
import sys, warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_parquet('data/metadata/metadata_v7.parquet')
h1 = df[(df['date'] >= '2021-02-21') & (df['date'] <= '2023-12-31')].copy()
h2 = df[(df['date'] >= '2024-01-01') & (df['date'] <= '2026-02-06')].copy()
for d in [h1, h2]:
    d = d.dropna(subset=['pm_rth5', 'rvol_5', 'gap_size_in_adr', 'full_drift'])

# Load pre-computed results
matrix_3x3 = pd.read_parquet('results/d7_5_3er_matrix_3x3x3.parquet')
oos_comp = pd.read_parquet('results/d7_5_oos_comparison.parquet')
trades = pd.read_parquet('results/d7_5_trade_sim_raw.parquet')

out = open('results/d7_5_synthese.txt', 'w', encoding='utf-8')
def p(text=''):
    out.write(text + '\n')

p("=" * 110)
p("DURCHLAUF 7.5 -- SYNTHESE: DAS 9:35-DREIECK")
p("PM/RTH5 x RVOL_5 x Gap_in_ADR -- Komplette Verhaltens-Landkarte")
p("=" * 110)
p()
p("Datum: 2026-02-14")
p("IS (H1): 4,254 Gapper (2021-02-21 bis 2023-12-31)")
p("OOS (H2): 5,812 Gapper (2024-01-01 bis 2026-02-06)")
p()

# ============================================================
# 9a: DIE LANDKARTE
# ============================================================
p("=" * 110)
p("9a: DIE LANDKARTE -- Was passiert bei jeder Kombi?")
p("=" * 110)
p()
p("Leserichtung: Bei PM/RTH5=X, RVOL_5=Y, Gap=Z passiert im Schnitt:")
p()

# Merge IS and OOS
merged = oos_comp.copy()

def action(fd_is, fd_oos, n_oos):
    """Recommended action based on IS drift and OOS confirmation."""
    if n_oos < 10:
        return "SKIP (N<10)"
    if np.isnan(fd_is) or np.isnan(fd_oos):
        return "SKIP (n/a)"
    # Both agree on direction
    if fd_is > 0.15 and fd_oos > 0.10:
        return "CONTINUATION"
    elif fd_is < -0.15 and fd_oos < -0.10:
        return "FADE"
    elif abs(fd_is) < 0.10 and abs(fd_oos) < 0.15:
        return "FLAT/SKIP"
    elif (fd_is > 0 and fd_oos < -0.10) or (fd_is < 0 and fd_oos > 0.10):
        return "UNSTABLE"
    else:
        return "WEAK/SKIP"

p(f"{'Dir':>7s} {'PM5':>7s} {'RV5':>7s} {'GAP':>7s} | {'N_IS':>5s} {'N_OOS':>5s} | {'fd_IS':>6s} {'fd_OOS':>6s} | {'rd_IS':>6s} {'rd_OOS':>6s} | {'Action':>13s}")
p("-" * 100)

for _, r in merged.iterrows():
    if r['N_IS'] < 10 and r['N_OOS'] < 10:
        continue
    act = action(r['fd_IS'], r['fd_OOS'], r['N_OOS'])
    fd_is_str = f"{r['fd_IS']:>+6.2f}" if not np.isnan(r['fd_IS']) else "   n/a"
    fd_oos_str = f"{r['fd_OOS']:>+6.2f}" if not np.isnan(r['fd_OOS']) else "   n/a"
    rd_is_str = f"{r['rd_IS']:>+6.2f}" if not np.isnan(r['rd_IS']) else "   n/a"
    rd_oos_str = f"{r['rd_OOS']:>+6.2f}" if not np.isnan(r['rd_OOS']) else "   n/a"

    p(f"{r['dir']:>7s} {r['pm5']:>7s} {r['rv5']:>7s} {r['gap']:>7s} | "
      f"{r['N_IS']:>5.0f} {r['N_OOS']:>5.0f} | {fd_is_str} {fd_oos_str} | {rd_is_str} {rd_oos_str} | {act:>13s}")

# ============================================================
# 9b: TOP CONTINUATION / FADE / SKIP
# ============================================================
p(f"\n\n{'='*110}")
p("9b: TOP-5 CONTINUATION / FADE / SKIP ZELLEN (OOS-validiert)")
p(f"{'='*110}")

# IS-OOS-validated continuation
valid = merged.dropna(subset=['fd_IS', 'fd_OOS'])
valid = valid[valid['N_OOS'] >= 20]

p("\n--- Top-5 CONTINUATION (beide IS+OOS positiv, sortiert nach OOS fd) ---")
cont_valid = valid[(valid['fd_IS'] > 0.05) & (valid['fd_OOS'] > 0.10)]
cont_top = cont_valid.nlargest(5, 'fd_OOS')
p(f"{'Dir':>7s} {'PM5':>7s} {'RV5':>7s} {'GAP':>7s} | {'N_IS':>5s} {'fd_IS':>7s} | {'N_OOS':>5s} {'fd_OOS':>7s} | {'Status':>12s}")
for _, r in cont_top.iterrows():
    p(f"{r['dir']:>7s} {r['pm5']:>7s} {r['rv5']:>7s} {r['gap']:>7s} | {r['N_IS']:>5.0f} {r['fd_IS']:>+7.3f} | {r['N_OOS']:>5.0f} {r['fd_OOS']:>+7.3f} | {r['status']:>12s}")

p("\n--- Top-5 FADE (beide IS+OOS negativ, sortiert nach OOS fd) ---")
fade_valid = valid[(valid['fd_IS'] < -0.05) & (valid['fd_OOS'] < -0.10)]
fade_top = fade_valid.nsmallest(5, 'fd_OOS')
p(f"{'Dir':>7s} {'PM5':>7s} {'RV5':>7s} {'GAP':>7s} | {'N_IS':>5s} {'fd_IS':>7s} | {'N_OOS':>5s} {'fd_OOS':>7s} | {'Status':>12s}")
for _, r in fade_top.iterrows():
    p(f"{r['dir']:>7s} {r['pm5']:>7s} {r['rv5']:>7s} {r['gap']:>7s} | {r['N_IS']:>5.0f} {r['fd_IS']:>+7.3f} | {r['N_OOS']:>5.0f} {r['fd_OOS']:>+7.3f} | {r['status']:>12s}")

p("\n--- Top-5 SKIP (IS zeigt etwas, OOS widerspricht = UNSTABLE) ---")
unstable = valid[
    ((valid['fd_IS'] > 0.15) & (valid['fd_OOS'] < -0.10)) |
    ((valid['fd_IS'] < -0.15) & (valid['fd_OOS'] > 0.10))
]
unstable_top = unstable.reindex(unstable['fd_IS'].abs().sort_values(ascending=False).index).head(5)
p(f"{'Dir':>7s} {'PM5':>7s} {'RV5':>7s} {'GAP':>7s} | {'N_IS':>5s} {'fd_IS':>7s} | {'N_OOS':>5s} {'fd_OOS':>7s} | {'Status':>12s}")
for _, r in unstable_top.iterrows():
    p(f"{r['dir']:>7s} {r['pm5']:>7s} {r['rv5']:>7s} {r['gap']:>7s} | {r['N_IS']:>5.0f} {r['fd_IS']:>+7.3f} | {r['N_OOS']:>5.0f} {r['fd_OOS']:>+7.3f} | {r['status']:>12s}")

# ============================================================
# 9c: EV-TABELLE
# ============================================================
p(f"\n\n{'='*110}")
p("9c: EV-TABELLE fuer die besten Trades (IS + OOS)")
p(f"{'='*110}")
p()
p("Trade-Simulation: Entry 9:35, Day H/L basiert")
p()

# Best IS trades
top_is = trades.nlargest(15, 'EV')
p("--- Top-15 Trades nach IS EV (N>=20) ---")
p(f"{'Type':>5s} {'Dir':>7s} {'PM5':>7s} {'RV5':>7s} {'GAP':>7s} {'SL':>4s} {'TGT':>4s} | {'N':>5s} {'WR%':>5s} {'SL%':>5s} {'EV':>7s}")
for _, r in top_is.iterrows():
    p(f"{r['type']:>5s} {r['dir']:>7s} {r['pm5']:>7s} {r['rv5']:>7s} {r['gap']:>7s} {r['SL']:>4.2f} {r['TGT']:>4.2f} | "
      f"{r['N']:>5.0f} {r['WR']:>5.1f} {r['SL_rate']:>5.1f} {r['EV']:>+7.3f}")

# ============================================================
# 9d: VERGLEICH 9:35 vs 10:00
# ============================================================
p(f"\n\n{'='*110}")
p("9d: VERGLEICH 9:35 vs 10:00")
p(f"{'='*110}")
p()
p("Kernfrage: Bringt Warten auf PM/RTH30 + RVOL_30 mehr als PM/RTH5 + RVOL_5 um 9:35?")
p()
p("ERGEBNIS:")
p("  1. PM/RTH30 ist ~2x praediktiver als PM/RTH5 (Spearman -0.122 vs -0.077)")
p("  2. RVOL_30 ist leicht besser als RVOL_5 (Spearman +0.078 vs +0.053)")
p("  3. Missed drift (9:35 -> 10:00) ist KLEIN:")
p("     GapUp IS: +0.028 ADR, OOS: +0.005 ADR")
p("     GapDown IS: +0.005 ADR, OOS: -0.031 ADR")
p("  4. PM/RTH30<10% + OD>0.5 hat >1.0 ADR Drift (aus D6.5/D7.0)")
p("     PM/RTH5<50% + hohe RVOL hat max ~0.4 ADR Drift")
p()
p("FAZIT: Warten bis 10:00 LOHNT SICH, weil:")
p("  - Man verliert nur ~0.02 ADR an Drift (vernachlaessigbar)")
p("  - PM/RTH30 ist doppelt so praediktiv")
p("  - PM/RTH30<10% + OD>0.5 ist das staerkste Setup (>1.0 ADR)")
p("  - AUSNAHME: Bei extremer OD (>0.5 ADR) + grosse Kerze SOFORT traden,")
p("    da dieser Move NICHT auf 10:00 wartet")

# ============================================================
# 9e: IMPLIKATIONEN FUER DEN TRADER
# ============================================================
p(f"\n\n{'='*110}")
p("9e: IMPLIKATIONEN FUER DEN TRADER")
p(f"{'='*110}")
p()

p("=== DAS 9:35-DREIECK: ZUSAMMENFASSUNG ===")
p()
p("1. PM/RTH5 ist der STAERKSTE Einzelparameter im Dreieck")
p("   - PM5_LO (<50%): Continuation-Bias (+0.19 GapUp, +0.16 GapDown IS)")
p("   - PM5_HI (>100%): Fade-Bias oder Flat")
p("   - Spread PM5_LO vs PM5_HI: ~0.35 ADR (konsistent)")
p()
p("2. RVOL_5 ist schwaecher und mit Gap-Groesse korreliert (rho=0.65)")
p("   - Allein erklaert RVOL_5 fast nichts (R2 < 0.001)")
p("   - Interaktion mit PM5: Bei hohem RVOL + niedrigem PM5 = staerkste Cont")
p("   - RVOL_5 > 7x = volatiler Tag (DayRange > 2.0 ADR)")
p()
p("3. Gap_in_ADR allein hat KEINEN marginalen Effekt (R2 ~ 0)")
p("   - Aber: Korreliert stark mit RVOL (groessere Gaps = hoheres RVOL)")
p("   - Gap > 2 ADR: Fast NIE gefuellt (<5%)")
p("   - Gap < 1 ADR: ~40% Fill-Rate")
p()
p("4. Alle drei Parameter zusammen erklaeren nur 0.5-0.7% der Drift-Varianz")
p("   -> Das 9:35-Dreieck ist ein SCHWACHES Signal")
p("   -> Es taugt fuer Orientierung, NICHT fuer mechanisches Trading")
p()

p("=== OOS-VALIDIERUNG: ERNUECHTERND ===")
p()
p("- Spearman IS vs OOS full_drift: rho = 0.04 (p = 0.78)")
p("  -> KEINE systematische Replikation der IS-Muster!")
p("- 18% repliziert, 22% umgekehrt, 20% geschrumpft, 27% flat")
p("- Die staerksten IS-Zellen (>+0.50 fd) sind OOS INSTABIL")
p("  z.B. GapDown PM5_LO RV5_HI GAP_LG: IS +0.57 -> OOS -0.15 (umgekehrt)")
p()
p("AUSNAHMEN (stabil IS -> OOS):")
p("  - GapUp PM5_LO RV5_HI GAP_LG: IS +0.38, OOS +0.41 (N=148 OOS)")
p("  - GapUp PM5_MID RV5_LO GAP_LG: IS -0.30, OOS -0.38 (Fade, N=52)")
p("  - GapDown PM5_MID RV5_LO GAP_LG: IS -0.24, OOS -0.27 (Fade, N=43)")
p()

p("=== TRADE-SIMULATION: Wo gibt es EV? ===")
p()
p("Bestes IS-Setup (Cont): GapUp PM5_HI RV5_HI GAP_MD, SL=0.25, TGT=1.0")
p("  IS: WR=60.7%, EV=+0.524 (N=28) -> OOS: WR=36%, EV=+0.241 (N=25)")
p("  REPLIZIERT, aber N sehr klein!")
p()
p("Bestes IS-Setup (Fade): GapUp PM5_MID RV5_LO GAP_LG, SL=0.25, TGT=1.0")
p("  IS: WR=54.5%, EV=+0.458 (N=22) -> OOS: WR=32.7%, EV=+0.159 (N=52)")
p("  REPLIZIERT, geschrumpft")
p()
p("PROBLEM: Die besten Zellen haben ALLE N < 50 im IS.")
p("Bei N=28 hat jeder einzelne Ausreisser massiven Einfluss auf den Mean.")
p()

p("=== VERGLEICH MIT FRUEHEREN DURCHLAEUFEN ===")
p()
p("Das 9:35-Dreieck (PM/RTH5 x RVOL_5 x Gap_in_ADR) ist SCHWAECHER als:")
p("  1. OD>0.5 with_gap: fd +0.9-1.1 ADR OOS, N~300 (D7.0)")
p("  2. PM/RTH30<10% + OD>0.5: fd +1.0 ADR OOS, N~40-80 (D6.5)")
p("  3. Combo E (grosse Kerze + OD>0.5): fd +1.1 ADR OOS, N~150-260 (D6.5)")
p("  4. Erste Kerze > 0.20 ADR: fd +0.34 ADR OOS, N>700 (D6.0)")
p()
p("Das 9:35-Dreieck bringt KEINEN signifikanten Zusatzwert ueber diese Signale.")
p("  - R2 der drei Parameter zusammen: 0.5-0.7% -> fast nichts")
p("  - OOS-Korrelation der Zellen: rho=0.04 -> kein Pattern")
p("  - Die Parameter-Interaktionen sind IS interessant, aber OOS instabil")
p()

p("=== HANDLUNGSANWEISUNGEN ===")
p()
p("UM 9:35 (nach 5 Min RTH):")
p("  1. CHECK: Hatte die erste Kerze eine starke Richtung (>0.20 ADR)?")
p("     -> Wenn ja + in Gap-Richtung: Continuation-Bias, +0.34 ADR Drift erwartet")
p("  2. CHECK: War der Opening Drive stark (>0.5 ADR)?")
p("     -> Wenn ja + in Gap-Richtung: Staerkstes Signal, +0.9 ADR Drift erwartet")
p("  3. OPTIONAL: PM/RTH5 < 50%?")
p("     -> Leichter Continuation-Bias, aber NICHT allein tradebar")
p("  4. OPTIONAL: Gap > 2 ADR + RVOL > 7x?")
p("     -> Hohe Volatilitaet, grosse Moves moeglich, Fill-Rate < 5%")
p()
p("UM 10:00 (nach 30 Min RTH):")
p("  5. CHECK: PM/RTH30 < 10%?")
p("     -> STARKES Continuation-Signal, besser als PM/RTH5")
p("     -> In Kombination mit OD>0.5: +1.0 ADR Drift erwartet")
p("  6. CHECK: PM/RTH30 > 30% + OD gegen Gap?")
p("     -> Fade-Signal: -0.32 bis -0.43 ADR Drift erwartet")
p()
p("WANN NICHT TRADEN:")
p("  7. PM/RTH5 = mittel (50-100%) + RVOL < 3 + Gap < 1 ADR")
p("     -> FLAT-Zone, kein Trend, kein Fade, Day Range < 1.2 ADR")
p("  8. Zellen mit IS-OOS Divergenz (oben als UNSTABLE markiert)")
p("     -> Kein zuverlaessiges Muster")
p()

p("=== SCHLUSSFOLGERUNG ===")
p()
p("Das 9:35-Dreieck (PM/RTH5 x RVOL_5 x Gap_in_ADR) ist eine nuetzliche")
p("ORIENTIERUNGSHILFE, aber KEIN mechanisch handelbares Signal.")
p()
p("Die drei Parameter zusammen erklaeren <1% der Drift-Varianz.")
p("Die OOS-Replikationsrate der Zellen-Drifts ist praktisch null (rho=0.04).")
p()
p("Fuer den Trader bedeutet das:")
p("  - PM/RTH5 als SCHNELLES Screening (niedrig = Continuation wahrscheinlicher)")
p("  - RVOL_5 als Volatilitaets-Indikator (nicht als Richtungs-Signal)")
p("  - Gap-Groesse als Fill-Rate-Indikator (klein = fuellt oft, gross = nie)")
p("  - Fuer Richtungs-Entscheidungen: OD-Staerke + erste Kerze >> 9:35-Dreieck")
p("  - Fuer maximales Signal: Warten auf 10:00, PM/RTH30 + OD nutzen")

out.close()
print("Done!", file=sys.stderr)
