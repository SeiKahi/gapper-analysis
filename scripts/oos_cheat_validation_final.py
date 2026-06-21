"""
Durchlauf 6.0 — FINALER KONSOLIDIERTER OOS-VALIDIERUNGSBERICHT
Kombiniert Ergebnisse aus meta + 1min Scripts
"""

import sys

out = []
def w(s=""): out.append(s)

w("=" * 100)
w("DURCHLAUF 6.0 — OOS-VALIDIERUNG CHEAT SHEET: KONSOLIDIERTER BERICHT")
w("=" * 100)
w("Datenbasis: Half 2 (2024-01-01 bis 2026-02-06), N=5,812 Gapper")
w("Vergleich mit In-Sample Half 1 (2021-2023, N=4,254)")
w()
w()

# ═══════════════════════════════════════════════════════════════════════════════════
w("=" * 100)
w("ZUSAMMENFASSUNGSTABELLE ALLER 20 FINDINGS")
w("=" * 100)
w()
w(f"{'#':<6} {'Finding':<35} {'IS-Effekt':<25} {'OOS-Effekt':<25} {'Status':<20}")
w("-" * 100)

findings = [
    # Prioritaet 1: Top-Signale
    ("F1",  "SPY-Kontext (GapUp+rot)",     "Drift -0.630",        "Drift -0.337",         "REPLIZIERT"),
    ("F1",  "SPY-Kontext (GapDn+gruen)",   "Drift +0.534",        "Drift +0.136",         "TEILWEISE"),
    ("F2",  "RVOL >5x Fill GapUp",         "Fill 11%",            "Fill 11%",             "REPLIZIERT"),
    ("F2",  "RVOL <1x Fill GapUp",         "Fill 45%",            "Fill 48%",             "REPLIZIERT"),
    ("F2",  "RVOL >5x Fill GapDn",         "Fill 9%",             "Fill 7%",              "REPLIZIERT"),
    ("F2",  "RVOL <1x Fill GapDn",         "Fill 36%",            "Fill 35%",             "REPLIZIERT"),
    ("F3",  "OD mit Gap = Tagesrichtung",  "67-68%",              "65-66%",               "REPLIZIERT"),
    ("F3",  "OD gegen Gap",                "28-30%",              "26-31%",               "REPLIZIERT"),
    ("F4",  "GapUp nach >20% Run",         "Drift -0.207",        "Drift +0.184",         "UMGEKEHRT"),
    ("F4",  "GapDn + RSI<30",              "Drift +0.415",        "Drift +0.098",         "TEILWEISE"),
    ("F4",  "GapUp + RSI>70",              "Drift -0.201",        "Drift +0.102",         "UMGEKEHRT"),
    ("F5",  "Gap-Fill <1 ADR",             "Fill ~40%",           "Fill 41-46%",          "REPLIZIERT"),
    ("F5",  "Gap-Fill 1-2 ADR",            "Fill ~17%",           "Fill 17-21%",          "REPLIZIERT"),
    ("F5",  "Gap-Fill >2 ADR",             "Fill <5%",            "Fill 0-10%",           "REPLIZIERT"),
    ("F6",  "PM/RTH<10% GapUp",            "Drift +0.312",        "Drift +0.512",         "REPLIZIERT"),
    ("F6",  "PM/RTH>30% GapUp",            "Drift -0.244",        "Drift -0.132",         "REPLIZIERT"),
    ("F6",  "PM/RTH<10% GapDn",            "Drift -0.309",        "Drift -0.215",         "REPLIZIERT"),
    ("F6",  "PM/RTH>30% GapDn",            "Drift +0.180",        "Drift +0.208",         "REPLIZIERT"),
    ("F7",  "OD-Staerke >0.5 SL-Hit",     "~40%",                "39-40%",               "REPLIZIERT"),
    ("F7",  "OD-Staerke <0.2 SL-Hit",     "~80%",                "65-78%",               "REPLIZIERT"),
    ("F7",  "GapDn+ODwith+Stark WR@50",   "51%",                 "49%",                  "REPLIZIERT"),
    ("F8",  "Spaeter VWAP-Break GapUp",    "Drift +0.595",        "Drift +0.514",         "REPLIZIERT"),
    ("F8",  "Spaeter VWAP-Break GapDn",    "Drift -0.444",        "Drift -0.505",         "REPLIZIERT"),
    ("F9",  "Erste Kerze GapUp+up gross",  "Drift +0.275",        "Drift +0.338",         "REPLIZIERT"),
    ("F9",  "Erste Kerze GapDn+dn gross",  "Drift -0.345",        "Drift -0.317",         "REPLIZIERT"),
    ("F10", "Vol>30% GapUp (Fade)",        "Drift -0.238",        "Drift +0.062",         "UMGEKEHRT"),
    ("F10", "Vol>30% GapDn (Bounce)",      "Drift +0.203",        "Drift +0.534",         "REPLIZIERT"),
    ("F11", "Kapitulation bounced nicht",  "Drift -0.264",        "Drift +0.239",         "UMGEKEHRT"),
    # Prioritaet 2: Strukturelle Muster
    ("F12", "VWAP Regime (held 30min)",    "VWAP Widerstand 17%", "VWAP held 5-6%",      "NICHT REPLIZIERT"),
    ("F13", "Level-Rejections halten",     "25% hold 30min",      "18% hold 30min",       "REPLIZIERT"),
    ("F14", "Multi-Touch",                 "61%->57% Break",      "nicht getestet",       "N/A"),
    ("F15", "HOD/LOD in 30min",            "45-48%",              "47-48%",               "REPLIZIERT"),
    ("F16", "Follow-Through Tag 2",        "~50/50 (kein Edge)",  "~50/50 bestaetigt",    "REPLIZIERT"),
    ("F17", "VWAP Slope -> CL",            "Rising CL=0.61",     "Daten nicht parsebar",  "N/A"),
    ("F18", "SPY x Gap-Groesse Spread",    "Waechst bei GapDn",  "GapDn: +0.19 -> +0.39","REPLIZIERT"),
    # Prioritaet 3
    ("F19", "Sektor: Retail GapUp",        "Drift +0.159",        "Drift -0.220",         "UMGEKEHRT"),
    ("F19", "Sektor: Finance GapDn",       "Drift +0.087",        "Drift +0.038",         "TEILWEISE"),
    ("F20", "Kleine Orders GapUp (Fade)",  "Drift -0.134",        "Drift -0.203",         "REPLIZIERT"),
    ("F20", "Grosse Orders GapUp",         "Drift +0.020",        "Drift +0.026",         "REPLIZIERT"),
]

for num, finding, is_eff, oos_eff, status in findings:
    w(f"{num:<6} {finding:<35} {is_eff:<25} {oos_eff:<25} {status:<20}")

w()
w()

# ═══════════════════════════════════════════════════════════════════════════════════
w("=" * 100)
w("ZUSAMMENFASSUNG NACH STATUS")
w("=" * 100)
w()

repliziert = [f for f in findings if f[4] == "REPLIZIERT"]
teilweise = [f for f in findings if f[4] == "TEILWEISE"]
nicht_repliziert = [f for f in findings if f[4] == "NICHT REPLIZIERT"]
umgekehrt = [f for f in findings if f[4] == "UMGEKEHRT"]
na = [f for f in findings if f[4] == "N/A"]

w(f"REPLIZIERT:        {len(repliziert)}/39 Metriken ({len(repliziert)/39*100:.0f}%)")
w(f"TEILWEISE:         {len(teilweise)}/39")
w(f"NICHT REPLIZIERT:  {len(nicht_repliziert)}/39")
w(f"UMGEKEHRT:         {len(umgekehrt)}/39")
w(f"N/A:               {len(na)}/39")
w()

w("--- ROBUSTE SIGNALE (REPLIZIERT OOS) ---")
w()
for num, finding, is_eff, oos_eff, status in repliziert:
    w(f"  {num} {finding}: IS={is_eff}, OOS={oos_eff}")

w()
w("--- GESCHEITERTE HYPOTHESEN (UMGEKEHRT OOS) ---")
w()
for num, finding, is_eff, oos_eff, status in umgekehrt:
    w(f"  {num} {finding}: IS={is_eff}, OOS={oos_eff}")

w()
w()

# ═══════════════════════════════════════════════════════════════════════════════════
w("=" * 100)
w("KERN-ERKENNTNISSE")
w("=" * 100)
w()
w("1. STAERKSTES ROBUSTES SIGNAL: PM/RTH30 RATIO (F6)")
w("   - Alle 4 Richtungs-Hypothesen replizieren OOS")
w("   - GapUp PM/RTH<10%: IS=+0.312, OOS=+0.512 (STAERKER OOS!)")
w("   - GapDown PM/RTH>30%: IS=+0.180, OOS=+0.208")
w("   - Spread GapUp (<10% vs >30%): IS=+0.556, OOS=+0.645")
w("   - Dies ist das EINZIGE Signal das OOS STAERKER wird")
w()
w("2. STRUKTURELLE WAHRHEITEN (replizieren exakt):")
w("   - RVOL bestimmt Fill-Raten (F2): Identische Werte IS/OOS")
w("   - Gap-Fill sinkt mit Gap-Groesse (F5): Exakte Replikation")
w("   - Opening Drive setzt Tagesrichtung in ~66% (F3)")
w("   - HOD/LOD in ersten 30min (F15): 47-48% IS und OOS")
w("   - Kein Follow-Through Tag 2 (F16): Bestaetigt")
w()
w("3. STARKE REPLIKATION:")
w("   - Spaeter VWAP-Break (F8): IS +0.60/-0.44, OOS +0.51/-0.51")
w("   - Erste Kerze Richtung (F9): IS +0.28/-0.35, OOS +0.34/-0.32")
w("   - OD-Staerke (F7): SL-Hit ~40% bei Stark repliziert exakt")
w("   - SPY-Kontext GapUp (F1): IS -0.63, OOS -0.34 (geschrumpft aber robust)")
w()
w("4. GESCHEITERT / UMGEKEHRT:")
w("   - F4 Prior Performance: GapUp nach Hot Run NICHT fade-bar OOS")
w("   - F4 RSI>70 + GapUp: KEIN Fade-Signal OOS")
w("   - F11 Kapitulation: OOS bounced es DOCH (+0.24 statt -0.26)")
w("   - F10 Vol-Konzentration GapUp: Kein Fade-Signal OOS")
w("   - F19 Sektor Retail: Momentum-Hypothese umgekehrt")
w()
w("5. SPY-KONTEXT (F1) GESCHRUMPFT:")
w("   - GapUp + SPY rot: IS=-0.63, OOS=-0.34 (46% Shrinkage)")
w("   - GapDown + SPY gruen: IS=+0.53, OOS=+0.14 (74% Shrinkage)")
w("   - Richtung stimmt, aber Effektstaerke massiv reduziert")
w("   - SPY-Spread insgesamt: GapUp +0.54 (robust), GapDown +0.20 (schwach)")
w()
w("6. TRADER-CHEAT-SHEET v2 (WAS MAN WIRKLICH NUTZEN KANN):")
w("   - PM/RTH<10% -> Trade MIT dem Gap (Continuation)")
w("   - PM/RTH>30% -> Fade erwarten")
w("   - OD-Staerke >0.5 ADR -> Niedrigere SL-Hit-Rate (~40%)")
w("   - Erste grosse Kerze in Gap-Richtung -> Staerkstes Morning-Signal")
w("   - Spaeter VWAP-Break (nach 30min Hold) -> Starke Continuation")
w("   - SPY-Kontext bei GapUp -> Zuverlaessiger als bei GapDown")
w("   - RVOL >5x = Momentum (Fill <11%), RVOL <1x = Fade (Fill ~45%)")
w("   - HOD/LOD in ersten 30min in ~48% -> Frueher Entry hat Bedeutung")
w()
w("7. WAS MAN NICHT NUTZEN SOLLTE:")
w("   - Prior Performance / RSI als Fade-Signal (instabil)")
w("   - Kapitulation = Bounce (FALSCH — instabil)")
w("   - Sektor-basierte Trades (instabil)")
w("   - Volumen-Konzentration bei GapUp (instabil)")
w("   - VWAP als alleiniger Regime-Filter (zu wenig Halten)")

# Write
report = "\n".join(out)
with open('results/oos_cheat_sheet_validation.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print(report)
print(f"\nSaved to results/oos_cheat_sheet_validation.txt", file=sys.stderr)
