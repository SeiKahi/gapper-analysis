"""
D9 Aufgabe 7: Synthese — R:R-basierte Empfehlungen
Fasst alle Ergebnisse aus Aufgaben 1-6 zusammen.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

RAW_DIR = Path('data/raw_1min')


def simulate_rr_trade(bars_rth, entry_price, sl_level, trade_dir, adr,
                      entry_time='09:36', timeout='15:55', rr_targets=[1, 2, 3]):
    """Bar-by-bar R:R trade simulation."""
    sl_dist = abs(entry_price - sl_level)
    if sl_dist <= 0 or pd.isna(sl_dist):
        return None
    targets = {}
    for rr in rr_targets:
        if trade_dir == 'long':
            targets[rr] = entry_price + rr * sl_dist
        else:
            targets[rr] = entry_price - rr * sl_dist
    results = {f'hit_{rr}R': False for rr in rr_targets}
    results['sl_hit'] = False
    results['sl_time'] = None
    results['exit_price'] = np.nan
    results['exit_type'] = 'timeout'
    results['sl_dist_adr'] = sl_dist / adr if adr > 0 else np.nan
    post_entry = bars_rth[bars_rth['time_et'] >= entry_time]
    for _, bar in post_entry.iterrows():
        if bar['time_et'] > timeout:
            break
        if trade_dir == 'long':
            if bar['low'] <= sl_level:
                results['sl_hit'] = True
                results['exit_price'] = sl_level
                results['exit_type'] = 'sl'
                results['sl_time'] = bar['time_et']
                break
            for rr in sorted(rr_targets):
                if bar['high'] >= targets[rr] and not results[f'hit_{rr}R']:
                    results[f'hit_{rr}R'] = True
        else:
            if bar['high'] >= sl_level:
                results['sl_hit'] = True
                results['exit_price'] = sl_level
                results['exit_type'] = 'sl'
                results['sl_time'] = bar['time_et']
                break
            for rr in sorted(rr_targets):
                if bar['low'] <= targets[rr] and not results[f'hit_{rr}R']:
                    results[f'hit_{rr}R'] = True
    if not results['sl_hit']:
        timeout_bars = post_entry[post_entry['time_et'] <= timeout]
        if len(timeout_bars) > 0:
            results['exit_price'] = timeout_bars.iloc[-1]['close']
    if pd.notna(results['exit_price']) and adr > 0:
        if trade_dir == 'long':
            results['pnl_adr'] = (results['exit_price'] - entry_price) / adr
        else:
            results['pnl_adr'] = (entry_price - results['exit_price']) / adr
    if pd.notna(results['exit_price']) and sl_dist > 0:
        if trade_dir == 'long':
            results['pnl_r'] = (results['exit_price'] - entry_price) / sl_dist
        else:
            results['pnl_r'] = (entry_price - results['exit_price']) / sl_dist
    return results


def run_simulation(meta_subset, sl_type='sl_half'):
    """Run trade simulation for a metadata subset."""
    all_results = []
    for _, row in meta_subset.iterrows():
        ticker = row['ticker']
        date = str(row['date'])
        fpath = RAW_DIR / ticker / f"{date}.parquet"
        if not fpath.exists():
            continue
        bars = pd.read_parquet(fpath)
        rth = bars[bars['session'] == 'rth']
        if len(rth) < 10:
            continue
        entry_price = row['close_935']
        adr = row.get('adr_10', np.nan)
        trade_dir = row.get('trade_direction', None)
        if pd.isna(entry_price) or pd.isna(adr) or adr <= 0 or trade_dir is None:
            continue
        if sl_type == 'sl_half':
            sl_level = row.get('sl_half_level', np.nan)
        elif sl_type == 'sl_fix025':
            if trade_dir == 'long':
                sl_level = entry_price - 0.25 * adr
            else:
                sl_level = entry_price + 0.25 * adr
        else:
            continue
        if pd.isna(sl_level):
            continue
        result = simulate_rr_trade(rth, entry_price, sl_level, trade_dir, adr)
        if result is None:
            continue
        result['ticker'] = ticker
        result['date'] = date
        result['sl_type'] = sl_type
        result['od_direction'] = row['od_direction']
        result['gap_direction'] = row['gap_direction']
        all_results.append(result)
    return pd.DataFrame(all_results)


def fmt(trades, label=""):
    """Compact summary."""
    n = len(trades)
    if n < 10:
        return f"  {label:<40} | N={n:>4} [N<10]"
    med_sl = trades['sl_dist_adr'].median()
    wr1 = trades['hit_1R'].mean()
    wr2 = trades['hit_2R'].mean()
    wr3 = trades['hit_3R'].mean()
    sl_pct = trades['sl_hit'].mean()
    ev_2r = (wr2 * 2 - (1 - wr2) * 1) * med_sl if pd.notna(med_sl) else np.nan
    low_n = " [LOW N]" if n < 20 else ""
    return (f"  {label:<40} | N={n:>4} | SL={med_sl:.3f} | "
            f"WR@2R={wr2:>5.1%} | EV@2R={ev_2r:>+.3f}{low_n}")


def main():
    lines = []
    lines.append("=" * 70)
    lines.append("D9 AUFGABE 7: SYNTHESE -- R:R-BASIERTE EMPFEHLUNGEN")
    lines.append("=" * 70)

    meta = pd.read_parquet('data/metadata/metadata_v9.parquet')
    h1 = meta[(meta['date'] >= '2021-02-21') & (meta['date'] <= '2023-12-31')].copy()
    h1 = h1[h1['od_body_pct'].notna() & h1['sl_full_adr'].notna()].copy()
    h2 = meta[(meta['date'] >= '2024-01-01') & (meta['date'] <= '2026-02-06')].copy()
    h2 = h2[h2['od_body_pct'].notna() & h2['sl_full_adr'].notna()].copy()

    # =====================================================================
    # SECTION 1: SL-TYP EMPFEHLUNG
    # =====================================================================
    lines.append(f"\n\n{'='*70}")
    lines.append("1. SL-TYP EMPFEHLUNG")
    lines.append(f"{'='*70}")
    lines.append("""
  ERGEBNIS: SL_FIX025 (0.25 ADR) ist der EINZIGE SL-Typ mit positivem
  OOS-EV bei WR@2R. SL_HALF ist IS-positiv aber kollabiert OOS.

  SL-Typ    | Median SL | IS WR@2R | OOS WR@2R | IS EV@2R | OOS EV@2R | Urteil
  -------------------------------------------------------------------------------
  SL_FULL   | 0.87 ADR  |  ~16%    |    ---     | -0.44    |    ---    | UNBRAUCHBAR
  SL_HALF   | 0.53 ADR  |  ~35%    |  ~26%      | +0.03    |  -0.14   | KOLLABIERT OOS
  SL_FIX025 | 0.25 ADR  |  ~39%    |  ~41%      | +0.04    |  +0.06   | REPLIZIERT

  WARNUNG SL_FIX025: 69% SL-Hits innerhalb 5 Minuten!
  -> Erfordert disziplinierten Entry GENAU bei 9:35 close.

  EMPFEHLUNG: SL_FIX025 (0.25 ADR) als Standard-SL.
  SL_HALF nur bei PM30<10% GapDown (einzige OOS-replizierende Ausnahme).
""")

    # =====================================================================
    # SECTION 2: FILTER-HIERARCHIE
    # =====================================================================
    lines.append(f"{'='*70}")
    lines.append("2. FILTER-HIERARCHIE (OOS-validiert)")
    lines.append(f"{'='*70}")
    lines.append("""
  Filter-Schicht     | Info-Typ | IS Spread | OOS Spread | Repliziert?
  -----------------------------------------------------------------------
  OD > 0.5 ADR       | 9:35     | Basis     | Basis      | JA (Basis)
  QUALITY_HIGH        | 9:35     |  ~0 pp    |  ~0 pp     | NEUTRAL (tautologisch)
  RVOL_30 > 5x        | 10:00    | +7-10 pp  |  +6-2 pp   | TEILWEISE
  PM/RTH30 < 10%      | 10:00    | +8-1 pp   |  +7-20 pp  | JA (stark GapDn!)
  Gap < 1 ADR         | 9:35     | +21 pp GU | +11 pp GU  | JA GapUp
  RV30>5 + PM30<10    | 10:00    | +15 pp    | +18 pp (N<20) | UNKLAR (LOW N)

  WICHTIG: QUALITY_HIGH (Body%/Wick-Filter) ist bei OD>0.5 nahezu TAUTOLOGISCH
  (85% qualifizieren sich). Kein signifikanter marginaler Wert.
""")

    # =====================================================================
    # SECTION 3: REDUNDANZ-MATRIX
    # =====================================================================
    lines.append(f"{'='*70}")
    lines.append("3. REDUNDANZ-MATRIX")
    lines.append(f"{'='*70}")
    lines.append("""
  Filter A            | Filter B         | Redundant? | Grund
  -----------------------------------------------------------------------
  QUALITY_HIGH        | OD > 0.5         | JA         | 85% Overlap
  RVOL_5              | RVOL_30          | TEILWEISE  | Korreliert, 30 besser
  PM/RTH5             | PM/RTH30         | TEILWEISE  | 30 besser (mehr Info)
  RVOL_30 > 5x        | PM/RTH30 < 10%   | NEIN       | Unabhaengig, Synergie!
  Gap < 1 ADR         | RVOL_30 > 5x     | NEIN       | Gegensaetzlich (kleine Gaps = wenig Vol)
  Gap < 1 ADR         | PM/RTH30 < 10%   | TEILWEISE  | Kleine Gaps haben oft niedrige PM

  KERN-ERGEBNIS: Die einzige NICHT-REDUNDANTE Kombination mit Synergie ist
  RVOL_30 > 5x + PM/RTH30 < 10%. Aber: N wird zu klein (IS=28, OOS=16).
""")

    # =====================================================================
    # SECTION 4: TIMING-EMPFEHLUNG
    # =====================================================================
    lines.append(f"{'='*70}")
    lines.append("4. TIMING-EMPFEHLUNG")
    lines.append(f"{'='*70}")
    lines.append("""
  Vergleich           | Entry  | WR@2R GU IS | WR@2R GU OOS | Urteil
  -----------------------------------------------------------------------
  9:35 Entry          | 9:35   |    35%      |     25%      | Standard
  9:35 + 30min Filter | 9:35   |    42%      |     33%      | Wartekosten: -25min
  10:00 Entry SL_30MIN| 10:00  |    10%      |     ---      | KATASTROPHAL
  10:00 Entry SL_FIX  | 10:00  |    34%      |     ---      | Vergleichbar

  ERGEBNIS: 9:35-Entry ist BESSER als 10:00-Entry.
  - SL_30MIN (30min-Extremum) ist viel zu weit (Median 1.13 ADR!)
  - 10:00-Filter (RVOL_30, PM30) wertvoller als spaeterer Entry
  - Optimaler Ansatz: 9:35 Entry + 10:00 Filter-Bestaetigung
    (Entry sofort, aber Position nur halten wenn 30min-Daten bestaetigen)
""")

    # =====================================================================
    # SECTION 5: FINAL TRADE SETUP VORSCHLAG
    # =====================================================================
    lines.append(f"{'='*70}")
    lines.append("5. FINAL TRADE SETUP VORSCHLAG")
    lines.append(f"{'='*70}")

    lines.append("""
  ==========================================
  SETUP A: "SOFORT" (9:35, ohne Warten)
  ==========================================
  Eintritt:    9:35 (close_935)
  Richtung:    OD-Richtung (with Gap)
  Bedingung:   OD > 0.5 ADR + Gap < 1 ADR (nur GapUp!)
  SL:          0.25 ADR unter Entry (Long) / ueber Entry (Short)
  Target:      2R = 0.50 ADR
  Exit:        Target ODER SL ODER 15:55 Timeout

  IS Performance:  WR@2R = 55.9%, EV = +0.268 ADR (N=34)
  OOS Performance: WR@2R = 36.7%, EV = +0.046 ADR (N=79)
  Shrinkage:       34%
  Urteil:          REPLIZIERT (OOS > 33.3% BE)


  ==========================================
  SETUP B: "GEFILTERT" (9:35 Entry, 10:00 Filter)
  ==========================================
  Eintritt:    9:35 (close_935), ggf. kleine Position
  Richtung:    OD-Richtung (with Gap)
  Bedingung:   OD > 0.5 ADR
  10:00 Filter: RVOL_30 > 5x (bei 10:00 pruefen, Position aufstocken)
  SL:          0.25 ADR
  Target:      2R = 0.50 ADR

  IS Performance:  WR@2R = 46.4%, EV = +0.098 ADR (N=69)
  OOS Performance: WR@2R = 47.8%, EV = +0.108 ADR (N=90)
  Shrinkage:       NEGATIV (-3%)
  Urteil:          STARK REPLIZIERT — Bestes OOS-Signal!


  ==========================================
  SETUP C: "PM30-GAPDOWN" (SL_HALF Sonderfall)
  ==========================================
  Eintritt:    9:35 (close_935)
  Richtung:    Short (with GapDown)
  Bedingung:   OD > 0.5 ADR + QH + PM/RTH30 < 10%
  SL:          SL_HALF (halbe OD-Range)
  Target:      1R = SL_HALF Distanz (optimal 1R fuer GapDn)

  IS Performance:  WR@1R = 70.5% (N=44)
  OOS Performance: WR@1R = 62.5% (N=32)
  Urteil:          REPLIZIERT — Einziger Fall wo SL_HALF OOS funktioniert
""")

    # =====================================================================
    # SECTION 6: VALIDIERUNGS-TABELLE (Kompakt)
    # =====================================================================
    lines.append(f"{'='*70}")
    lines.append("6. VALIDIERUNGS-UEBERSICHT (Alle R:R Setups)")
    lines.append(f"{'='*70}\n")

    # Run the key setups for IS and OOS
    setups = [
        ('GapUp L0 OD>0.5w', lambda df: df[(df['od_strength'] > 0.5) & (df['od_direction'] == 'with_gap') & (df['gap_direction'] == 'up')]),
        ('GapUp L0+QH', lambda df: df[(df['od_strength'] > 0.5) & (df['od_direction'] == 'with_gap') & (df['gap_direction'] == 'up') & (df['quality_high'] == True)]),
        ('GapUp QH+Gap<1', lambda df: df[(df['od_strength'] > 0.5) & (df['od_direction'] == 'with_gap') & (df['gap_direction'] == 'up') & (df['quality_high'] == True) & (df['gap_size_in_adr'] < 1)]),
        ('GapUp QH+RV30>5', lambda df: df[(df['od_strength'] > 0.5) & (df['od_direction'] == 'with_gap') & (df['gap_direction'] == 'up') & (df['quality_high'] == True) & (df['rvol_open_30min'] >= 5)]),
        ('GapUp QH+PM30<10', lambda df: df[(df['od_strength'] > 0.5) & (df['od_direction'] == 'with_gap') & (df['gap_direction'] == 'up') & (df['quality_high'] == True) & (df['pm_rth30_computed'] < 0.10)]),
        ('GapDn L0 OD>0.5w', lambda df: df[(df['od_strength'] > 0.5) & (df['od_direction'] == 'with_gap') & (df['gap_direction'] == 'down')]),
        ('GapDn L0+QH', lambda df: df[(df['od_strength'] > 0.5) & (df['od_direction'] == 'with_gap') & (df['gap_direction'] == 'down') & (df['quality_high'] == True)]),
        ('GapDn QH+RV30>5', lambda df: df[(df['od_strength'] > 0.5) & (df['od_direction'] == 'with_gap') & (df['gap_direction'] == 'down') & (df['quality_high'] == True) & (df['rvol_open_30min'] >= 5)]),
        ('GapDn QH+PM30<10', lambda df: df[(df['od_strength'] > 0.5) & (df['od_direction'] == 'with_gap') & (df['gap_direction'] == 'down') & (df['quality_high'] == True) & (df['pm_rth30_computed'] < 0.10)]),
        ('Against GapUp', lambda df: df[(df['od_strength'] > 0.5) & (df['od_direction'] == 'against_gap') & (df['gap_direction'] == 'up')]),
        ('Against GapDn', lambda df: df[(df['od_strength'] > 0.5) & (df['od_direction'] == 'against_gap') & (df['gap_direction'] == 'down')]),
    ]

    lines.append(f"  {'Setup':<28} | {'SL':>7} | {'IS N':>5} | {'IS WR@2R':>8} | {'IS EV':>7} | {'OOS N':>5} | {'OOS WR@2R':>9} | {'OOS EV':>7} | Repl?")
    lines.append(f"  {'-'*115}")

    for sl_type, sl_label in [('sl_fix025', 'FIX025'), ('sl_half', 'HALF')]:
        for slabel, sfn in setups:
            is_sub = sfn(h1)
            oos_sub = sfn(h2)
            is_trades = run_simulation(is_sub, sl_type)
            oos_trades = run_simulation(oos_sub, sl_type)

            is_n = len(is_trades)
            oos_n = len(oos_trades)
            if is_n >= 10:
                is_wr2 = is_trades['hit_2R'].mean()
                is_med_sl = is_trades['sl_dist_adr'].median()
                is_ev = (is_wr2 * 2 - (1 - is_wr2)) * is_med_sl
            else:
                is_wr2 = np.nan
                is_ev = np.nan
            if oos_n >= 10:
                oos_wr2 = oos_trades['hit_2R'].mean()
                oos_med_sl = oos_trades['sl_dist_adr'].median()
                oos_ev = (oos_wr2 * 2 - (1 - oos_wr2)) * oos_med_sl
            else:
                oos_wr2 = np.nan
                oos_ev = np.nan

            repl = "---"
            if pd.notna(oos_wr2):
                repl = "JA" if oos_wr2 > 0.333 else "NEIN"

            is_wr_str = f"{is_wr2:>7.1%}" if pd.notna(is_wr2) else " N<10  "
            is_ev_str = f"{is_ev:>+7.3f}" if pd.notna(is_ev) else "  ---  "
            oos_wr_str = f"{oos_wr2:>8.1%}" if pd.notna(oos_wr2) else "  N<10   "
            oos_ev_str = f"{oos_ev:>+7.3f}" if pd.notna(oos_ev) else "  ---  "

            lines.append(f"  {slabel:<28} | {sl_label:>7} | {is_n:>5} | {is_wr_str} | {is_ev_str} | {oos_n:>5} | {oos_wr_str} | {oos_ev_str} | {repl}")

    # =====================================================================
    # SECTION 7: KERN-ERKENNTNISSE D9
    # =====================================================================
    lines.append(f"\n\n{'='*70}")
    lines.append("7. KERN-ERKENNTNISSE D9")
    lines.append(f"{'='*70}")
    lines.append("""
  1. SL-Distanz ist ENTSCHEIDEND:
     - SL_FULL (OD-Extremum, ~0.87 ADR): Viel zu weit -> WR@2R ~16%
     - SL_HALF (~0.53 ADR): IS marginal, OOS kollabiert
     - SL_FIX025 (0.25 ADR): Einziger OOS-replizierender SL-Typ
     -> PARADOX: Engster SL = bestes R:R, aber 69% sofortiger SL-Hit

  2. QUALITY_HIGH Filter ist REDUNDANT bei OD > 0.5:
     - 85% der OD > 0.5 Trades sind bereits QUALITY_HIGH
     - Kein marginaler Wert, weder IS noch OOS

  3. Beste OOS-validierten Filter (mit SL_FIX025):
     a) Gap < 1 ADR (GapUp): IS 55.9% -> OOS 36.7% WR@2R
     b) RVOL_30 > 5x: IS 46.4% -> OOS 47.8% WR@2R (KEIN SHRINKAGE!)
     c) PM/RTH30 < 10%: IS 47.6% -> OOS 38.6% WR@2R

  4. Against-Gap Trades bleiben NEGATIV:
     - IS WR@2R = 25-29%, OOS = 22-28% -> Konsistent unter Break-Even
     -> KEIN handelbarer Edge gegen den OD

  5. Timing: 9:35 Entry > 10:00 Entry:
     - SL_30MIN (30min-Extremum) ist katastrophal (Median 1.13 ADR)
     - 10:00-Filter ohne 10:00-Entry = optimaler Ansatz

  6. GapUp vs GapDown unterschiedlich:
     - GapUp: Optimal bei 2R Target (Momentum-Potential)
     - GapDown: Optimal bei 1R Target (schneller Exit besser)

  7. Der handlebste Trade:
     Setup B: OD>0.5 + with_gap + RVOL_30>5x + SL_FIX025
     -> WR@2R = 47.8% OOS (N=90), EV = +0.108 ADR pro Trade
     -> Kein Shrinkage, ausreichend N
     -> ABER: 80% SL-Hit Rate, nur 20% erreichen Timeout
        -> Erwartung: ~48% der Trades treffen 2R,
           ~32% werden gestoppt, ~20% Timeout
""")

    # =====================================================================
    # SECTION 8: OFFENE FRAGEN / NAECHSTE SCHRITTE
    # =====================================================================
    lines.append(f"{'='*70}")
    lines.append("8. OFFENE FRAGEN / NAECHSTE SCHRITTE")
    lines.append(f"{'='*70}")
    lines.append("""
  1. SL_FIX025 Paradox weiter untersuchen:
     - 69% SL-Hit in <5min = Preis laeuft zuerst gegen OD-Richtung
     - Ist dies ein normales OD-Retracement das man aushalten muss?
     - Idee: Entry etwas spaeter (9:36-9:37) um Retracement abzuwarten?

  2. Dynamisches SL testen:
     - Start mit SL_FIX025, nach +1R auf Break-Even ziehen?
     - Trail-Stop ab +1.5R?

  3. Setup B (RVOL_30>5x) OOS-Stabilität ueber Zeit:
     - Ist der OOS-Edge gleichmaessig verteilt oder geclustert?
     - Monats-/Quartals-Breakdown

  4. Kosten einrechnen:
     - Slippage: 0.01-0.05 ADR realistisch?
     - Spread: Entry/Exit Kosten
     - SL_FIX025 bei 0.25 ADR = SL_dist ~$0.50-5.00 je nach Aktienpreis
       -> Bei $100 Aktie: SL = $2.50, Target 2R = $5.00

  5. Earnings vs Non-Earnings Split bei Setup B:
     - D8 zeigte unterschiedliche Mikrostruktur
     - Gilt Setup B fuer beide Gruppen?
""")

    output = "\n".join(lines)
    with open('results/d9_synthese.txt', 'w', encoding='utf-8') as f:
        f.write(output)
    print(output)
    print(f"\nSaved to results/d9_synthese.txt", file=sys.stderr)


if __name__ == '__main__':
    main()
