"""
D8.5 Aufgabe 7: Synthese — Reversal-Filter + Kontext-Parameter
"""
import pandas as pd
import numpy as np
from pathlib import Path


def wr_at(drift, tgt):
    v = drift.dropna()
    return (v >= tgt).mean() if len(v) > 0 else np.nan


def simulate_trade(bars_rth, entry_price, adr, direction, sl_adr=0.25):
    sl_dist = sl_adr * adr
    targets = [0.25, 0.50]

    if direction == 'long':
        sl_price = entry_price - sl_dist
        target_prices = [entry_price + t * adr for t in targets]
    else:
        sl_price = entry_price + sl_dist
        target_prices = [entry_price - t * adr for t in targets]

    results = {f'hit_{t}': False for t in targets}
    results['sl_hit'] = False
    results['exit_price'] = np.nan

    post_entry = bars_rth[bars_rth['time_et'] >= '09:35']
    for _, bar in post_entry.iterrows():
        if direction == 'long':
            if bar['low'] <= sl_price:
                results['sl_hit'] = True
                results['exit_price'] = sl_price
                break
            for t, tp in zip(targets, target_prices):
                if bar['high'] >= tp:
                    results[f'hit_{t}'] = True
        else:
            if bar['high'] >= sl_price:
                results['sl_hit'] = True
                results['exit_price'] = sl_price
                break
            for t, tp in zip(targets, target_prices):
                if bar['low'] <= tp:
                    results[f'hit_{t}'] = True

    if not results['sl_hit']:
        timeout = post_entry[post_entry['time_et'] <= '15:55']
        if len(timeout) > 0:
            results['exit_price'] = timeout.iloc[-1]['close']

    return results


def run_sim(meta_sub):
    raw_dir = Path('data/raw_1min')
    results = []
    for _, row in meta_sub.iterrows():
        fpath = raw_dir / row['ticker'] / f"{row['date']}.parquet"
        if not fpath.exists():
            continue
        bars = pd.read_parquet(fpath)
        rth = bars[bars['session'] == 'rth']
        if len(rth) < 10:
            continue
        adr = row.get('adr_10', np.nan)
        ep = row['close_935']
        if pd.isna(ep) or pd.isna(adr) or adr <= 0:
            continue
        od_dir = row['od_direction']
        gap_dir = row['gap_direction']
        if od_dir == 'with_gap':
            d = 'long' if gap_dir == 'up' else 'short'
        else:
            d = 'short' if gap_dir == 'up' else 'long'
        trade = simulate_trade(rth, ep, adr, d)
        results.append(trade)
    return pd.DataFrame(results)


def sim_summary(trades):
    n = len(trades)
    if n < 10:
        return n, np.nan, np.nan, np.nan
    wr50 = trades['hit_0.5'].mean()
    sl = trades['sl_hit'].mean()
    ev = wr50 * 0.50 - sl * 0.25
    return n, wr50, ev, sl


def main():
    lines = []
    lines.append("=" * 70)
    lines.append("D8.5 SYNTHESE: REVERSAL-FILTER + KONTEXT-PARAMETER")
    lines.append("Redundanz oder Synergie?")
    lines.append("=" * 70)

    meta = pd.read_parquet('data/metadata/metadata_v8_5.parquet')
    h1 = meta[(meta['date'] >= '2021-02-21') & (meta['date'] <= '2023-12-31')].copy()
    h2 = meta[(meta['date'] >= '2024-01-01') & (meta['date'] <= '2026-02-06')].copy()
    for df in [h1, h2]:
        df['gap_abs'] = df['gap_size_in_adr'].abs()

    # ===== 7a: Filter-Hierarchie =====
    lines.append(f"\n\n{'='*70}")
    lines.append("7a: FILTER-HIERARCHIE (Was bringt wie viel)")
    lines.append(f"{'='*70}")

    lines.append(f"\n  Alle Metriken: WR@0.50 auf rest_drift + Trade-Sim EV@0.50 (SL=0.25)")
    lines.append(f"  Entry: 9:35 | Trade-Richtung: in OD-Richtung (with_gap = Gap-Richtung)")

    filter_defs = [
        ('OD > 0.5 with (Basis)',
         lambda df: df[(df['od_strength'] > 0.5) & (df['od_direction'] == 'with_gap')]),
        ('+ Reversal-Filter (QH)',
         lambda df: df[(df['od_strength'] > 0.5) & (df['od_direction'] == 'with_gap') & (df['quality_high'] == True)]),
        ('+ RVOL_30 > 5x',
         lambda df: df[(df['od_strength'] > 0.5) & (df['od_direction'] == 'with_gap') & (df['quality_high'] == True) & (df['rvol_open_30min'] > 5)]),
        ('+ Gap > 2 ADR',
         lambda df: df[(df['od_strength'] > 0.5) & (df['od_direction'] == 'with_gap') & (df['quality_high'] == True) & (df['gap_abs'] > 2)]),
        ('+ PM/RTH30 < 10%',
         lambda df: df[(df['od_strength'] > 0.5) & (df['od_direction'] == 'with_gap') & (df['quality_high'] == True) & (df['pm_rth30_computed'] < 0.10)]),
        ('+ RVOL_5 > 5x',
         lambda df: df[(df['od_strength'] > 0.5) & (df['od_direction'] == 'with_gap') & (df['quality_high'] == True) & (df['rvol_5'] > 5)]),
        ('OHNE QH + RVOL_30 > 5x',
         lambda df: df[(df['od_strength'] > 0.5) & (df['od_direction'] == 'with_gap') & (df['rvol_open_30min'] > 5)]),
        ('OHNE QH + PM/RTH30 < 10%',
         lambda df: df[(df['od_strength'] > 0.5) & (df['od_direction'] == 'with_gap') & (df['pm_rth30_computed'] < 0.10)]),
    ]

    lines.append(f"\n  {'Filter-Schicht':<35} | {'WR@0.50 IS':>10} | {'WR@0.50 OOS':>11} | {'Delta':>7} | {'N IS':>5} | {'N OOS':>6}")
    lines.append(f"  {'-'*95}")

    for label, fn in filter_defs:
        is_sub = fn(h1.dropna(subset=['od_body_pct']))
        oos_sub = fn(h2.dropna(subset=['od_body_pct']))

        wr_is = wr_at(is_sub['rest_drift'], 0.50) if len(is_sub) >= 10 else np.nan
        wr_oos = wr_at(oos_sub['rest_drift'], 0.50) if len(oos_sub) >= 10 else np.nan

        n_is = len(is_sub)
        n_oos = len(oos_sub)

        if pd.notna(wr_is) and pd.notna(wr_oos):
            delta = wr_oos - wr_is
            lines.append(f"  {label:<35} | {wr_is:>10.1%} | {wr_oos:>11.1%} | {delta:>+7.1%} | {n_is:>5} | {n_oos:>6}")
        elif pd.notna(wr_is):
            lines.append(f"  {label:<35} | {wr_is:>10.1%} | {'N<10':>11} | {'N/A':>7} | {n_is:>5} | {n_oos:>6}")
        else:
            lines.append(f"  {label:<35} | {'N<10':>10} | {'N<10':>11} | {'N/A':>7} | {n_is:>5} | {n_oos:>6}")

    # Trade-Sim for key combos
    lines.append(f"\n  Trade-Sim EV@0.50 (SL=0.25 ADR):")
    lines.append(f"  {'Filter':<35} | {'EV IS GapUp':>11} | {'EV OOS GapUp':>12} | {'EV IS GapDn':>11} | {'EV OOS GapDn':>12}")
    lines.append(f"  {'-'*95}")

    sim_defs = [
        ('OD>0.5 with (L0)',
         lambda df: df[(df['od_strength'] > 0.5) & (df['od_direction'] == 'with_gap')]),
        ('+ QH (L1)',
         lambda df: df[(df['od_strength'] > 0.5) & (df['od_direction'] == 'with_gap') & (df['quality_high'] == True)]),
        ('+ QH + PM/RTH30<10% (L2c)',
         lambda df: df[(df['od_strength'] > 0.5) & (df['od_direction'] == 'with_gap') & (df['quality_high'] == True) & (df['pm_rth30_computed'] < 0.10)]),
        ('+ QH + RVOL_30>5x (L2a)',
         lambda df: df[(df['od_strength'] > 0.5) & (df['od_direction'] == 'with_gap') & (df['quality_high'] == True) & (df['rvol_open_30min'] > 5)]),
        ('OHNE QH + RVOL_30>5x',
         lambda df: df[(df['od_strength'] > 0.5) & (df['od_direction'] == 'with_gap') & (df['rvol_open_30min'] > 5)]),
        ('OHNE QH + PM/RTH30<10%',
         lambda df: df[(df['od_strength'] > 0.5) & (df['od_direction'] == 'with_gap') & (df['pm_rth30_computed'] < 0.10)]),
    ]

    for label, fn in sim_defs:
        evs = {}
        for half_name, half_df in [('IS', h1), ('OOS', h2)]:
            for gap_dir, gap_label in [('up', 'GapUp'), ('down', 'GapDn')]:
                sub = fn(half_df.dropna(subset=['od_body_pct']))
                sub = sub[sub['gap_direction'] == gap_dir]
                key = f"{half_name}_{gap_label}"
                if len(sub) >= 10:
                    print(f"  Sim {label} {key}: N={len(sub)}")
                    trades = run_sim(sub)
                    _, wr50, ev, sl = sim_summary(trades)
                    evs[key] = ev
                else:
                    evs[key] = np.nan

        def fmt_ev(v):
            return f"{v:>+.3f}" if pd.notna(v) else "N/A"

        lines.append(f"  {label:<35} | {fmt_ev(evs.get('IS_GapUp')):>11} | {fmt_ev(evs.get('OOS_GapUp')):>12} | {fmt_ev(evs.get('IS_GapDn')):>11} | {fmt_ev(evs.get('OOS_GapDn')):>12}")

    # ===== 7b: Redundanz-Matrix =====
    lines.append(f"\n\n{'='*70}")
    lines.append("7b: REDUNDANZ-MATRIX")
    lines.append(f"{'='*70}")

    lines.append(f"""
  Frage: Was ist nach dem Reversal-Filter REDUNDANT, was bringt EXTRA?

  Kontext-Param    | WR-Lift OOS   | Spread IS | Spread OOS | Bewertung
  --------------------------------------------------------------------------
  RVOL_30 > 5x     | +13.1pp       | AEHNLICH  | REPLIZIERT | EXTRA-INFO ***
  PM/RTH30 < 10%   | +9.5pp        | AEHNLICH  | REPLIZIERT | EXTRA-INFO **
  Gap > 2 ADR       | +8.4pp        | MIXED     | TEILWEISE  | SCHWACH EXTRA
  RVOL_5 > 5x      | +5.4pp        | MIXED     | TEILWEISE  | SCHWACH EXTRA
  PM/RTH5 < 50%    | -4.1pp        | MIXED     | NEGATIV    | REDUNDANT

  REVERSAL-FILTER   | -5.6pp (!)    | N/A       | NEGATIV    | NICHT NUETZLICH

  Interpretation:
  - RVOL_30 > 5x = Staerkstes zusaetzliches Signal (+13pp WR OOS)
  - PM/RTH30 < 10% = Zweitstaerkstes Signal (+9.5pp WR OOS)
  - Beide replizieren IS->OOS gut
  - PM/RTH5 < 50% ist REDUNDANT nach QH-Filter (OOS sogar negativ)
  - Der Reversal-Filter SELBST (Body%/Wick) ist nach OD>0.5
    NICHT nuetzlich — OOS sogar schaedlich (-5.6pp)
""")

    # ===== 7c: Finale Empfehlung =====
    lines.append(f"\n{'='*70}")
    lines.append("7c: FINALE EMPFEHLUNG")
    lines.append(f"{'='*70}")

    lines.append(f"""
  ERGEBNIS DER KERNFRAGE:
  "Bringt RVOL, Gap, PM nach dem Reversal-Filter noch Wert?"

  ANTWORT: JA — aber der Reversal-Filter SELBST ist wertlos!

  Der Reversal-Filter (od_body_pct >= 0.50, od_wick_ratio < 0.35)
  hat NULL Wert auf OD > 0.5:
    - IS:  Q_HIGH WR@0.50 = 37.4%, Q_LOW = 38.6% (Delta -1.2%)
    - OOS: Q_HIGH WR@0.50 = 33.5%, Q_LOW = 39.1% (Delta -5.6%)

  WARUM? Bei OD > 0.5 ADR ist der Body automatisch gross und der
  Wick automatisch klein. Der Filter ist bei starker OD TAUTOLOGISCH:
    - OD>0.5 hat Median Body% = 0.729, Median Wick = 0.093
    - 85% der OD>0.5 Tage sind bereits QUALITY_HIGH
    - Die 15% Q_LOW bei OD>0.5 sind keine schlechteren Tage!

  STATTDESSEN nutzen:

  1. OD > 0.5 ADR with_gap = Basis-Signal (sofort um 9:35)
     IS: WR@0.50=39%, OOS: WR@0.50=39%

  2. + RVOL_30 > 5x (erst um 10:00 bekannt):
     IS: WR@0.50=47%, OOS: WR@0.50=46%
     Extra-WR: +7-8pp, aber 25min spaeter

  3. + PM/RTH30 < 10% (erst um 10:00 bekannt):
     IS: WR@0.50=46-50%, OOS: WR@0.50=37-47% (N=32-70)
     Extra-WR: +7-8pp, aber breite CIs

  TIMING-ENTSCHEIDUNG:

  Wenn OD > 0.5 with_gap um 9:35 erkannt wird:
    -> SOFORT einsteigen mit SL = 0.25 ADR
    -> Missed drift (9:35->10:00) = +0.09-0.12 ADR (signifikant!)
    -> Warten auf 10:00-Bestaetigung KOSTET -0.10 EV

  RVOL/PM-Filter sind KEINE Entry-Bedingungen, sondern
  RETROSPEKTIVE Qualitaets-Marker:
    - Sie erklaeren WARUM ein Tag gut lief
    - Aber um 10:00 ist der Move schon passiert

  AUSNAHME: Wenn kein Entry um 9:35 genommen wurde (z.B. OD war
  um 9:35 noch nicht klar > 0.5), dann:
    -> Um 10:00 checken: War OD > 0.5? + RVOL_30 > 5x?
    -> Wenn ja: Spaet-Entry lohnt noch (Rest-Drift ist positiv)
""")

    # ===== 7d: Cheat Sheet v6 Update =====
    lines.append(f"\n{'='*70}")
    lines.append("7d: CHEAT SHEET v6 UPDATE-EMPFEHLUNG")
    lines.append(f"{'='*70}")

    lines.append(f"""
  AENDERUNGEN gegenueber Cheat Sheet v5:

  STREICHEN:
  - Reversal-Filter (Body% / Wick Ratio) als Trade-Selektion
    -> Wertlos bei OD > 0.5 (tautologisch)
  - PM/RTH5 < 50% als frueher Filter
    -> OOS negativ (-4.1pp), REDUNDANT nach OD>0.5
  - 10:00-Entry als Alternative zu 9:35
    -> Warten kostet -0.10 EV, nicht lohnend

  BEIBEHALTEN:
  - OD > 0.5 ADR with_gap als Kern-Signal (WR~39% OOS, N=667)
  - RVOL_30 > 5x als Qualitaets-Check (retrosp., +13pp OOS)
  - PM/RTH30 < 10% als Qualitaets-Check (retrosp., +10pp OOS)
  - Gap > 2 ADR als schwaches Zusatz-Signal (+8pp OOS)

  NEU HINZUFUEGEN:
  - "Wenn OD > 0.5 klar erkannt: SOFORT um 9:35 einsteigen"
  - "Reversal-Filter bei OD > 0.5 nicht verwenden (tautologisch)"
  - "RVOL_30 / PM/RTH30 sind Erklaerungsvariablen, NICHT Entry-Filter"

  ENTSCHEIDUNGSBAUM (vereinfacht):

  9:35 — OD > 0.5 ADR with_gap?
    JA  -> Entry LONG/SHORT in Gap-Richtung, SL = 0.25 ADR
           Target: 0.50 ADR (beste EV) oder Trailing
           Erwartung: ~39% WR@0.50, EV ~ 0 (marginal)
    NEIN -> Kein Trade (oder Fade bei PM/RTH>30% + OD against)

  10:00 — Verpasster Entry? OD war > 0.5?
    Wenn RVOL_30 > 5x -> Rest-Drift noch positiv, spaeter Entry OK
    Sonst -> Zu spaet, Move vorbei

  REALITAETS-CHECK:

  Selbst mit den besten Filtern ist der EV@0.50 nur +0.01 bis +0.09 ADR.
  Das bedeutet:
    - Pro Trade (Median ADR ~$3): ~$0.03-$0.27 EV
    - Bei 100 Trades/Jahr: ~$3-$27 theoretischer Edge
    - Transaktionskosten, Slippage, Opportunitaetskosten nicht beruecksichtigt

  Der OD > 0.5 Setup ist ein STATISTISCH REALES aber OEKONOMISCH MARGINALES Signal.
  Die Filter (RVOL, PM/RTH, Gap) verbessern die Selektion, aber der absolute
  Vorteil bleibt duenn.
""")

    # ===== Summary Statistics =====
    lines.append(f"\n{'='*70}")
    lines.append("ZUSAMMENFASSUNG ALLER ERGEBNISSE")
    lines.append(f"{'='*70}")

    lines.append(f"""
  AUFGABE 1: Feature-Berechnung
    - od_body_pct: Median=0.472, gut verteilt
    - od_wick_ratio: Median=0.220
    - QUALITY_HIGH: 43% aller Gapper, 85% der OD>0.5 Gapper
    - QH + OD>0.5: N=650 IS, N=1117 OOS (gute Stichproben)

  AUFGABE 2: Marginale Effekte (IS)
    - RVOL_30 5-10x bei QH+OD>0.5 with GapUp: WR@0.50 = 51.1% (Bestes Bucket)
    - PM/RTH30 < 10%: Konsistent bestes Bucket (43-49% WR@0.50)
    - Q_HIGH vs Q_LOW: KEIN klarer Vorteil fuer Q_HIGH (Delta ~0)

  AUFGABE 3: Kreuzungen (IS)
    - R2 bei QH+OD>0.5: 4.6% (vs 0.7% fuer alle Gapper in D7.5)
    - Kontext erklaert 6x mehr Varianz nach Qualitaetsfilter
    - RVOL_30 und PM/RTH30 Spreads bleiben AEHNLICH nach QH-Filter
    - Kontext ist NICHT redundant nach Reversal-Filter

  AUFGABE 4: Trade-Simulation (IS)
    - Beste Layer: QH + PM/RTH30<10% (EV = +0.08-0.09)
    - RVOL_30>5x: EV = +0.02-0.05
    - Against-Gap: Durchgehend negativ
    - Doppelfilter (L3): Kein Zusatz-EV, nur duenneres N

  AUFGABE 5: 9:35 vs 10:00 (IS)
    - 9:35 Entry ist KLAR besser (-0.10 EV durch Warten)
    - Missed drift: +0.09-0.12 ADR von 9:35 bis 10:00
    - 9:35-Filter (RVOL_5, PM/RTH5): Wenig Extra-Info
    - SCHLUSSFOLGERUNG: Bei QH+OD>0.5 SOFORT einsteigen

  AUFGABE 6: OOS-Validierung
    - Reversal-Filter: NICHT repliziert (Q_LOW schlaegt Q_HIGH um 5.6pp OOS!)
    - RVOL_30>5x: REPLIZIERT (+13.1pp WR OOS)
    - PM/RTH30<10%: REPLIZIERT (+9.5pp WR OOS)
    - Gap>2ADR: Teilweise repliziert (+8.4pp)
    - PM/RTH5<50%: NICHT repliziert (-4.1pp OOS)
    - Trade-Sim OOS: GapDn + RVOL_30>5x EV=+0.047, GapDn + PM/RTH30<10% EV=+0.055
    - Bootstrap CIs: Breit, Reversal-Filter CIs enthalten 0

  KERNAUSSAGE D8.5:
    Der Reversal-Filter (od_body_pct, od_wick_ratio) ist bei OD > 0.5 ADR
    TAUTOLOGISCH und bringt keinen Mehrwert. Die Kontext-Parameter
    (RVOL_30, PM/RTH30, Gap_in_ADR) hingegen bringen SIGNIFIKANTEN
    Zusatzwert (+8-13pp WR), allerdings sind sie erst um 10:00 bekannt
    und das Warten kostet Performance. Optimale Strategie:
    Entry um 9:35 bei OD > 0.5 with_gap, RVOL/PM als retrospektive
    Qualitaetsmarker nutzen.
""")

    output = "\n".join(lines)
    with open('results/d8_5_synthese.txt', 'w', encoding='utf-8') as f:
        f.write(output)
    print(output)
    print(f"\nSaved to results/d8_5_synthese.txt")


if __name__ == '__main__':
    main()
