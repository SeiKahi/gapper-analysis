"""
D8.0 Aufgabe 8: Synthese
=========================
Combine all findings into cheat sheets, decision trees, and final setup table.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

BASE_DIR = Path(__file__).parent.parent
META_DIR = BASE_DIR / "data" / "metadata"
RESULTS_DIR = BASE_DIR / "results"


def read_result(filename):
    """Read a result file if it exists."""
    path = RESULTS_DIR / filename
    if path.exists():
        with open(path, encoding='utf-8') as f:
            return f.read()
    return None


def main():
    lines = []
    lines.append("=" * 90)
    lines.append("D8.0 SYNTHESE: EARNINGS-SPLIT + 10:00-REVERSAL")
    lines.append("=" * 90)

    # Read all result files for reference
    r1 = read_result("d8_1_earnings_basis.txt")
    r2 = read_result("d8_2_setup_split.txt")
    r3 = read_result("d8_3_flush_analyse.txt")
    r4 = read_result("d8_4_reversal_predict.txt")
    r5 = read_result("d8_5_reversal_trade.txt")
    r6 = read_result("d8_6_non_earnings_setups.txt")
    r7 = read_result("d8_7_oos.txt")

    # 8a: Earnings vs Non-Earnings Cheat Sheet
    lines.append("\n" + "=" * 60)
    lines.append("8a: EARNINGS vs NON-EARNINGS CHEAT SHEET")
    lines.append("=" * 60)

    lines.append("""
Bei EARNINGS-Gappern:
  - OD-against Gap ist oft ein FLUSH (temporaer), kein echter Fade
    → NICHT sofort shorten/longen gegen den Gap
    → WARTEN auf 10:00 Reversal
  - OD-with Gap ist Confirmation → Continuation-Trade moeglich
    → Entry 9:35 in Gap-Richtung
  - Groessere Gaps und hoeheres RVOL als Non-Earnings (erwartbar)
  - PM/RTH5 Interpretation:
    → Hohes PM bei Earnings = Institutionelle reagieren auf News
    → NICHT gleich zu interpretieren wie hohes PM bei Non-Earnings

Bei NON-EARNINGS-Gappern:
  - OD-against Gap ist oefter ein echter Fade
    → Fade-Trade bei PM_LO + RV_LO moeglich
  - Das 9:35-Dreieck (PM x RVOL x Gap) funktioniert BESSER
    → Weniger Noise durch Earnings-Flush
  - Continuation-Setups schwaecher als bei Earnings
    → Non-Earnings Gaps haben weniger 'Ueberzeugung'

KERNREGEL:
  "Ist es ein Earnings-Tag?" → ERSTE Frage vor jedem Trade!
""")

    # 8b: 10:00-Reversal Decision Tree
    lines.append("\n" + "=" * 60)
    lines.append("8b: 10:00-REVERSAL DECISION TREE")
    lines.append("=" * 60)

    lines.append("""
Um 9:35 beobachten:
  1. OD-Richtung: with Gap oder against Gap?
  2. OD-Staerke: > 0.5 ADR (stark) oder < 0.2 ADR (schwach)?
  3. ist es Earnings?
  4. Wick Ratio der OD (gross = Ablehnung)
  5. Volume Profile: K1/K2-5 Ratio (hoch = Flush-Muster)

Entscheidungsbaum:

  WENN OD against Gap + Earnings:
    → Hohe Flush-Wahrscheinlichkeit
    → NICHT faden! Warten auf 10:00
    → Bei Reversal um 10:00: Entry gegen OD, in Gap-Richtung

  WENN OD against Gap + Non-Earnings:
    → Niedrigere Flush-Wahrscheinlichkeit
    → Fade-Trade moeglich (PM_LO + RV_LO bevorzugt)
    → SL am OD-Extremum

  WENN OD with Gap + Earnings:
    → Continuation-Signal staerker
    → Entry 9:35 in Gap-Richtung

  WENN OD with Gap + Non-Earnings:
    → Continuation schwaecher, mehr Vorsicht
    → Nur bei starkem OD (> 0.5 ADR) + HiRV

  REVERSAL-INDIKATOREN (erhoehen Rev-Wahrscheinlichkeit):
    ✓ Earnings-Tag
    ✓ Hohe Wick Ratio in der OD
    ✓ Hohes K1/K2-5 Volumen-Ratio (Flush-Muster)
    ✓ Candle 5 dreht gegen Candle 1
""")

    # 8c: Finale Setup-Tabelle
    lines.append("\n" + "=" * 60)
    lines.append("8c: FINALE SETUP-TABELLE")
    lines.append("=" * 60)

    # Load OOS data if available
    meta = pd.read_parquet(META_DIR / "metadata_v8.parquet")
    meta['date_dt'] = pd.to_datetime(meta['date'])

    lines.append("""
| Situation                | Earnings?  | Aktion     | Entry | Richtung      | Kommentar                        |
|--------------------------|------------|------------|-------|---------------|----------------------------------|
| OD with + stark (>0.5)   | Earnings   | CONT       | 9:35  | In Gap-Dir    | Staerkstes Earnings-Signal       |
| OD with + stark (>0.5)   | Non-E      | CONT       | 9:35  | In Gap-Dir    | Schwaecher als Earnings          |
| OD against + stark       | Earnings   | WAIT       | -     | -             | Flush wahrscheinlich, 10:00 ab-  |
|                          |            |            |       |               | warten                           |
| OD against + stark       | Non-E      | FADE       | 9:35  | Gegen Gap     | Besonders bei PM_LO + RV_LO     |
| OD against → Reversal    | Earnings   | REV-TRADE  | 10:00 | In Gap-Dir    | Nach Flush, zurueck in Gap       |
| OD against → Reversal    | Non-E      | REV-TRADE  | 10:00 | In Gap-Dir    | Selten, aber wenn dann stark     |
| OD schwach (<0.2)        | Any        | SKIP       | -     | -             | Kein klares Signal               |
""")

    lines.append("\nDatenquellen: IS=2021-2024, OOS=2024-2026")
    lines.append("Alle Werte in ADR normalisiert, signiert in Gap-Richtung")

    # Append summary stats from results
    lines.append("\n\n" + "=" * 60)
    lines.append("ERGEBNISSE REFERENZ")
    lines.append("=" * 60)

    for fname, label in [
        ("d8_0_earnings_check.txt", "Aufgabe 0: Earnings Check"),
        ("d8_1_earnings_basis.txt", "Aufgabe 1: Basis-Statistik"),
        ("d8_2_setup_split.txt", "Aufgabe 2: Setup-Split"),
        ("d8_3_flush_analyse.txt", "Aufgabe 3: Flush-Analyse"),
        ("d8_4_reversal_predict.txt", "Aufgabe 4: Reversal-Prediction"),
        ("d8_5_reversal_trade.txt", "Aufgabe 5: Reversal-Trade"),
        ("d8_6_non_earnings_setups.txt", "Aufgabe 6: Non-E Setups"),
        ("d8_7_oos.txt", "Aufgabe 7: OOS-Validierung"),
    ]:
        content = read_result(fname)
        if content:
            # Just include first 30 lines as reference
            snippet = "\n".join(content.split("\n")[:30])
            lines.append(f"\n--- {label} (Auszug) ---")
            lines.append(snippet)
            lines.append("  [...]")

    text = "\n".join(lines)
    out_file = RESULTS_DIR / "d8_synthese.txt"
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(text)
    safe = text.replace('\u2192', '->').replace('\u2713', '[x]')
    print(safe)


if __name__ == "__main__":
    main()
