"""
D12 Ergebnis: Saubere Zusammenfassung aller Variationen.
Alle 4 Quadranten (GapUp/Dn x OD_Long/Short) mit beiden SL-Methoden.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

RAW_DIR = Path('data/raw_1min')
IS_START = '2021-02-21'
IS_END = '2023-12-31'


def simulate_trail_d(bars_post_entry, entry_price, sl_dist, trade_dir, adr):
    """TRAIL_D: +1R -> BE, trail = peak - 0.5R. Timeout 15:55."""
    if sl_dist <= 0 or pd.isna(sl_dist) or adr <= 0 or pd.isna(adr):
        return None
    if len(bars_post_entry) == 0:
        return None

    sl_level = entry_price - sl_dist if trade_dir == 'long' else entry_price + sl_dist
    highest_r = 0.0
    trail_active = False
    exit_price = None
    exit_type = 'timeout'

    for _, bar in bars_post_entry.iterrows():
        if trade_dir == 'long':
            current_r = max(0, (bar['high'] - entry_price) / sl_dist)
        else:
            current_r = max(0, (entry_price - bar['low']) / sl_dist)
        highest_r = max(highest_r, current_r)

        if highest_r >= 1.0 and not trail_active:
            trail_active = True
            sl_level = entry_price
        if trail_active:
            if trade_dir == 'long':
                sl_level = max(sl_level, entry_price + (highest_r - 0.5) * sl_dist)
            else:
                sl_level = min(sl_level, entry_price - (highest_r - 0.5) * sl_dist)

        if trade_dir == 'long' and bar['low'] <= sl_level:
            exit_price = sl_level
            exit_type = 'trail_sl' if trail_active else 'initial_sl'
            break
        elif trade_dir == 'short' and bar['high'] >= sl_level:
            exit_price = sl_level
            exit_type = 'trail_sl' if trail_active else 'initial_sl'
            break

    if exit_price is None:
        exit_price = bars_post_entry.iloc[-1]['close']
        exit_type = 'timeout'

    pnl = (exit_price - entry_price) if trade_dir == 'long' else (entry_price - exit_price)
    return {
        'pnl': pnl, 'pnl_r': pnl / sl_dist, 'pnl_adr': pnl / adr,
        'exit_type': exit_type, 'highest_r': highest_r, 'winner': pnl > 0,
    }


def run_quadrant(meta_sub, method, label=""):
    """
    Run one quadrant with a given method.
    method: 'baseline' (fix 0.25 ADR SL, entry 09:35) or 'breakout_od' (OD-based SL, breakout entry)
    Trade direction always follows od_long.
    """
    results = []
    skip_no_bo = 0
    skip_entry_sl = 0
    skip_sl_inv = 0
    skip_data = 0

    for _, row in meta_sub.iterrows():
        ticker = row['ticker']
        date = str(row['date'])
        close_935 = row['close_935']
        adr = row['adr_10']
        od_long = row['od_long']

        if pd.isna(close_935) or pd.isna(adr) or adr <= 0 or pd.isna(od_long):
            skip_data += 1
            continue

        trade_dir = 'long' if od_long else 'short'

        fpath = RAW_DIR / ticker / f"{date}.parquet"
        if not fpath.exists():
            skip_data += 1
            continue
        bars = pd.read_parquet(fpath)
        rth = bars[bars['session'] == 'rth'].copy()
        if len(rth) < 10:
            skip_data += 1
            continue

        if method == 'baseline':
            entry_price = close_935
            sl_dist = 0.25 * adr
            post_entry = rth[(rth['time_et'] >= '09:36') & (rth['time_et'] <= '15:55')]

        elif method == 'breakout_od':
            od_low5 = row['od_low5']
            od_high5 = row['od_high5']
            if pd.isna(od_low5) or pd.isna(od_high5):
                skip_data += 1
                continue

            # Find breakout 09:36-10:00
            scan = rth[(rth['time_et'] >= '09:36') & (rth['time_et'] <= '10:00')]
            bo_idx = None
            bo_bar = None
            for idx, bar in scan.iterrows():
                if trade_dir == 'long' and bar['high'] > close_935:
                    bo_idx, bo_bar = idx, bar
                    break
                elif trade_dir == 'short' and bar['low'] < close_935:
                    bo_idx, bo_bar = idx, bar
                    break

            if bo_idx is None:
                skip_no_bo += 1
                continue

            entry_price = close_935
            if trade_dir == 'long':
                sl_level = od_low5 - 0.01
                sl_dist = entry_price - sl_level
            else:
                sl_level = od_high5 + 0.01
                sl_dist = sl_level - entry_price

            if sl_dist <= 0:
                skip_sl_inv += 1
                continue

            # Check if breakout bar itself hits SL
            if trade_dir == 'long' and bo_bar['low'] <= sl_level:
                skip_entry_sl += 1
                results.append({
                    'pnl': -sl_dist, 'pnl_r': -1.0, 'pnl_adr': -sl_dist / adr,
                    'exit_type': 'entry_bar_sl', 'highest_r': 0.0, 'winner': False,
                    'sl_dist_adr': sl_dist / adr,
                })
                continue
            elif trade_dir == 'short' and bo_bar['high'] >= sl_level:
                skip_entry_sl += 1
                results.append({
                    'pnl': -sl_dist, 'pnl_r': -1.0, 'pnl_adr': -sl_dist / adr,
                    'exit_type': 'entry_bar_sl', 'highest_r': 0.0, 'winner': False,
                    'sl_dist_adr': sl_dist / adr,
                })
                continue

            bo_pos = rth.index.get_loc(bo_idx)
            post_entry = rth.iloc[bo_pos + 1:]
            post_entry = post_entry[post_entry['time_et'] <= '15:55']
        else:
            continue

        if len(post_entry) == 0:
            skip_data += 1
            continue

        res = simulate_trail_d(post_entry, entry_price, sl_dist, trade_dir, adr)
        if res is None:
            skip_data += 1
            continue

        if method == 'breakout_od':
            res['sl_dist_adr'] = sl_dist / adr
        else:
            res['sl_dist_adr'] = 0.25

        results.append(res)

    df = pd.DataFrame(results) if results else pd.DataFrame()
    return df, {
        'skip_no_bo': skip_no_bo,
        'skip_entry_sl': skip_entry_sl,
        'skip_sl_inv': skip_sl_inv,
        'skip_data': skip_data,
    }


def stats_line(df):
    """Return compact stats string."""
    n = len(df)
    if n == 0:
        return "  ---  keine Trades  ---"
    if n < 10:
        return f"  N={n}  [zu wenig Daten]"

    med_r = df['pnl_r'].median()
    wr = df['winner'].mean() * 100
    ev = df['pnl_adr'].mean()
    n_init = (df['exit_type'] == 'initial_sl').sum()
    n_entry_sl = (df['exit_type'] == 'entry_bar_sl').sum()
    n_trail = (df['exit_type'] == 'trail_sl').sum()
    n_to = (df['exit_type'] == 'timeout').sum()
    sl_pct = (n_init + n_entry_sl) / n * 100

    med_sl = df['sl_dist_adr'].median() if 'sl_dist_adr' in df.columns else np.nan
    sl_info = f"  SL-Dist: {med_sl:.3f} ADR" if pd.notna(med_sl) else ""

    return (f"  N={n:>4d}  |  MedPnL={med_r:>+.2f}R  |  WR={wr:>5.1f}%  |  EV={ev:>+.3f} ADR  |  "
            f"InitSL={n_init}  TrailSL={n_trail}  Timeout={n_to}"
            f"{f'  EntryBarSL={n_entry_sl}' if n_entry_sl > 0 else ''}{sl_info}")


def main():
    L = []

    def w(t=""):
        L.append(t)

    meta = pd.read_parquet('data/metadata/metadata_v9.parquet')
    h1 = meta[(meta['date'] >= IS_START) & (meta['date'] <= IS_END)].copy()
    od = h1[h1['od_strength'] > 0.5].copy()

    # 4 Quadranten definieren
    quadrants = {
        'GapUp + OD_Long':  od[(od['gap_direction'] == 'up')   & (od['od_long'] == True)],
        'GapUp + OD_Short': od[(od['gap_direction'] == 'up')   & (od['od_long'] == False)],
        'GapDn + OD_Short': od[(od['gap_direction'] == 'down') & (od['od_long'] == False)],
        'GapDn + OD_Long':  od[(od['gap_direction'] == 'down') & (od['od_long'] == True)],
    }

    # Header
    w("=" * 72)
    w("D12 ERGEBNIS: TRADE SETUP VARIATIONEN (IS ONLY)")
    w("=" * 72)
    w()
    w(f"IS-Zeitraum: {IS_START} bis {IS_END}")
    w(f"Gesamt IS: {len(h1)} Tage  |  OD > 0.5: {len(od)} Tage")
    w()
    w("-" * 72)
    w("WAS WURDE GETESTET?")
    w("-" * 72)
    w()
    w("Alle Trades folgen der OD-Richtung:")
    w("  - OD_Long (od_long=True)  -> Long-Trade")
    w("  - OD_Short (od_long=False) -> Short-Trade")
    w()
    w("Zwei SL-Methoden werden verglichen:")
    w()
    w("  METHODE A: BASELINE (bisheriges Setup)")
    w("    Entry:  close_935 direkt um 09:35")
    w("    SL:     0.25 x ADR_10 (fix)")
    w()
    w("  METHODE B: OD-BASIERTER SL + BREAKOUT-ENTRY")
    w("    Entry:  Breakout ueber/unter close_935 (09:36-10:00)")
    w("            Kein Breakout bis 10:00 -> kein Trade")
    w("    SL:     Long: od_low5 - 0.01 (1 Tick unter 5-Min-Tief)")
    w("            Short: od_high5 + 0.01 (1 Tick ueber 5-Min-Hoch)")
    w()
    w("  Exit bei beiden: TRAIL_D (+1R -> Breakeven, dann Trail = Peak - 0.5R)")
    w("  Timeout: 15:55")
    w()
    w("4 Quadranten werden einzeln ausgewertet:")
    w("  1. GapUp + OD_Long   (= with_gap)    -> Long-Trade in Gap-Richtung")
    w("  2. GapUp + OD_Short  (= against_gap)  -> Short-Trade gegen Gap")
    w("  3. GapDn + OD_Short  (= with_gap)    -> Short-Trade in Gap-Richtung")
    w("  4. GapDn + OD_Long   (= against_gap)  -> Long-Trade gegen Gap")
    w()

    # Run all combinations
    all_results = {}
    for qname, qdata in quadrants.items():
        print(f"  {qname} (N={len(qdata)})...", file=sys.stderr)
        for method, mlabel in [('baseline', 'Baseline'), ('breakout_od', 'Breakout+OD-SL')]:
            print(f"    {mlabel}...", file=sys.stderr)
            df, skips = run_quadrant(qdata, method)
            all_results[(qname, method)] = (df, skips, len(qdata))

    # ===================================================================
    # ERGEBNIS 1: Uebersichtstabelle
    # ===================================================================
    w("=" * 72)
    w("ERGEBNIS 1: UEBERSICHTSTABELLE")
    w("=" * 72)
    w()

    # Compact comparison table
    hdr = f"  {'Quadrant':<22s} | {'Methode':<18s} | {'N':>4s} | {'MedR':>5s} | {'WR%':>5s} | {'EV_ADR':>7s} | {'SL%':>5s} | {'MedSL':>7s}"
    sep = "  " + "-" * 90
    w(hdr)
    w(sep)

    for qname in quadrants:
        for method, mlabel in [('baseline', 'Baseline (fixSL)'), ('breakout_od', 'Breakout+OD-SL')]:
            df, skips, n_input = all_results[(qname, method)]
            n = len(df)
            if n < 10:
                low = " [LOW N]" if 0 < n < 10 else ""
                w(f"  {qname:<22s} | {mlabel:<18s} | {n:>4d} | {'---':>5s} | {'---':>5s} | {'---':>7s} | {'---':>5s} | {'---':>7s}{low}")
            else:
                med_r = df['pnl_r'].median()
                wr = df['winner'].mean() * 100
                ev = df['pnl_adr'].mean()
                n_init = (df['exit_type'] == 'initial_sl').sum()
                n_entry_sl = (df['exit_type'] == 'entry_bar_sl').sum()
                sl_pct = (n_init + n_entry_sl) / n * 100
                med_sl = df['sl_dist_adr'].median()
                w(f"  {qname:<22s} | {mlabel:<18s} | {n:>4d} | {med_r:>+5.2f} | {wr:>5.1f} | {ev:>+7.3f} | {sl_pct:>5.1f} | {med_sl:>7.3f}")
        w(sep)

    w()

    # ===================================================================
    # ERGEBNIS 2: Detail pro Quadrant
    # ===================================================================
    w("=" * 72)
    w("ERGEBNIS 2: DETAIL PRO QUADRANT")
    w("=" * 72)

    for qname in quadrants:
        w()
        w(f"--- {qname} ---")
        n_input = len(quadrants[qname])
        w(f"  Input: {n_input} Tage (OD>0.5)")
        w()

        for method, mlabel in [('baseline', 'Baseline (fix 0.25 ADR SL, Entry 09:35)'),
                                ('breakout_od', 'Breakout-Entry + OD-basierter SL')]:
            df, skips, _ = all_results[(qname, method)]
            w(f"  {mlabel}:")
            w(f"  {stats_line(df)}")
            if method == 'breakout_od':
                w(f"    Skips: kein Breakout={skips['skip_no_bo']}, "
                  f"EntryBar-SL={skips['skip_entry_sl']}, "
                  f"SL ungueltig={skips['skip_sl_inv']}")
            w()

    # ===================================================================
    # ERGEBNIS 3: WITH_GAP vs AGAINST_GAP Vergleich
    # ===================================================================
    w("=" * 72)
    w("ERGEBNIS 3: WITH_GAP vs AGAINST_GAP")
    w("=" * 72)
    w()
    w("with_gap  = OD geht in Gap-Richtung (GapUp+OD_Long, GapDn+OD_Short)")
    w("against_gap = OD geht gegen Gap (GapUp+OD_Short, GapDn+OD_Long)")
    w()

    for method, mlabel in [('baseline', 'Baseline'), ('breakout_od', 'Breakout+OD-SL')]:
        w(f"  --- {mlabel} ---")

        # with_gap = GapUp+OD_Long + GapDn+OD_Short
        df_wg1, _, _ = all_results[('GapUp + OD_Long', method)]
        df_wg2, _, _ = all_results[('GapDn + OD_Short', method)]
        df_wg = pd.concat([df_wg1, df_wg2], ignore_index=True) if len(df_wg1) + len(df_wg2) > 0 else pd.DataFrame()

        # against_gap = GapUp+OD_Short + GapDn+OD_Long
        df_ag1, _, _ = all_results[('GapUp + OD_Short', method)]
        df_ag2, _, _ = all_results[('GapDn + OD_Long', method)]
        df_ag = pd.concat([df_ag1, df_ag2], ignore_index=True) if len(df_ag1) + len(df_ag2) > 0 else pd.DataFrame()

        w(f"  with_gap:    {stats_line(df_wg)}")
        w(f"  against_gap: {stats_line(df_ag)}")

        # Combined
        df_all = pd.concat([df_wg, df_ag], ignore_index=True) if len(df_wg) + len(df_ag) > 0 else pd.DataFrame()
        w(f"  alle OD>0.5: {stats_line(df_all)}")
        w()

    # ===================================================================
    # ERGEBNIS 4: SL-Distanz Analyse (nur Breakout-Methode)
    # ===================================================================
    w("=" * 72)
    w("ERGEBNIS 4: SL-DISTANZ ANALYSE (Breakout+OD-SL)")
    w("=" * 72)
    w()
    w("Zum Vergleich: Baseline nutzt immer 0.250 ADR als SL-Distanz.")
    w("OD-basierter SL haengt von der Groesse der 5-Min-Kerze ab.")
    w()

    for qname in quadrants:
        df, _, _ = all_results[(qname, 'breakout_od')]
        if len(df) > 0 and 'sl_dist_adr' in df.columns:
            sl = df['sl_dist_adr']
            w(f"  {qname:<22s}: Median={sl.median():.3f} ADR  Mean={sl.mean():.3f} ADR  "
              f"Min={sl.min():.3f}  Max={sl.max():.3f}  (N={len(df)})")
        else:
            w(f"  {qname:<22s}: keine Daten")
    w()

    # ===================================================================
    # FAZIT
    # ===================================================================
    w("=" * 72)
    w("FAZIT / INTERPRETATION")
    w("=" * 72)
    w()

    # Auto-generate some key findings
    # Best EV per method
    w("Beste Quadranten nach EV (ADR):")
    w()
    for method, mlabel in [('baseline', 'Baseline'), ('breakout_od', 'Breakout+OD-SL')]:
        best_ev = -999
        best_q = ""
        for qname in quadrants:
            df, _, _ = all_results[(qname, method)]
            if len(df) >= 10:
                ev = df['pnl_adr'].mean()
                if ev > best_ev:
                    best_ev = ev
                    best_q = qname
        if best_ev > -999:
            df_best, _, _ = all_results[(best_q, method)]
            wr = df_best['winner'].mean() * 100
            n = len(df_best)
            w(f"  {mlabel:<18s}: {best_q} -> EV={best_ev:+.3f} ADR, WR={wr:.1f}%, N={n}")
    w()

    # Key comparisons
    w("Kernfragen:")
    w()

    # Q1: OD against gap?
    df_a1, _, _ = all_results[('GapUp + OD_Long', 'baseline')]
    df_a2u, _, _ = all_results[('GapUp + OD_Short', 'baseline')]
    if len(df_a1) >= 10 and len(df_a2u) >= 10:
        w(f"  1. OD against gap bei GapUp (Baseline):")
        w(f"     with_gap (OD_Long):    EV={df_a1['pnl_adr'].mean():+.3f} ADR, WR={df_a1['winner'].mean()*100:.1f}%, N={len(df_a1)}")
        w(f"     against_gap (OD_Short): EV={df_a2u['pnl_adr'].mean():+.3f} ADR, WR={df_a2u['winner'].mean()*100:.1f}%, N={len(df_a2u)}")
        delta = df_a1['pnl_adr'].mean() - df_a2u['pnl_adr'].mean()
        w(f"     -> Unterschied: {delta:+.3f} ADR zugunsten {'with_gap' if delta > 0 else 'against_gap'}")
    w()

    df_a1d, _, _ = all_results[('GapDn + OD_Short', 'baseline')]
    df_a2d, _, _ = all_results[('GapDn + OD_Long', 'baseline')]
    if len(df_a1d) >= 10 and len(df_a2d) >= 10:
        w(f"  2. OD against gap bei GapDn (Baseline):")
        w(f"     with_gap (OD_Short):   EV={df_a1d['pnl_adr'].mean():+.3f} ADR, WR={df_a1d['winner'].mean()*100:.1f}%, N={len(df_a1d)}")
        w(f"     against_gap (OD_Long):  EV={df_a2d['pnl_adr'].mean():+.3f} ADR, WR={df_a2d['winner'].mean()*100:.1f}%, N={len(df_a2d)}")
        delta = df_a1d['pnl_adr'].mean() - df_a2d['pnl_adr'].mean()
        w(f"     -> Unterschied: {delta:+.3f} ADR zugunsten {'with_gap' if delta > 0 else 'against_gap'}")
    w()

    # Q2: Breakout besser als direkt?
    w(f"  3. Breakout-Entry vs Direkt-Entry:")
    for qname in quadrants:
        df_base, _, _ = all_results[(qname, 'baseline')]
        df_bo, _, _ = all_results[(qname, 'breakout_od')]
        if len(df_base) >= 10 and len(df_bo) >= 10:
            ev_base = df_base['pnl_adr'].mean()
            ev_bo = df_bo['pnl_adr'].mean()
            delta = ev_bo - ev_base
            better = "Breakout" if delta > 0 else "Baseline"
            w(f"     {qname:<22s}: Baseline={ev_base:+.3f}, Breakout={ev_bo:+.3f}, Delta={delta:+.3f} -> {better}")
        elif len(df_base) >= 10:
            w(f"     {qname:<22s}: Baseline={df_base['pnl_adr'].mean():+.3f}, Breakout=N<10")
        elif len(df_bo) >= 10:
            w(f"     {qname:<22s}: Baseline=N<10, Breakout={df_bo['pnl_adr'].mean():+.3f}")
    w()

    # Q3: OD-SL vs fix SL
    w(f"  4. SL-Distanz Effekt:")
    w(f"     Baseline nutzt 0.250 ADR fix.")
    w(f"     OD-basierter SL ist typischerweise weiter (groessere 5-Min-Kerze).")
    w(f"     Weiterer SL -> weniger SL-Hits, aber kleinerer PnL pro R.")
    w(f"     ACHTUNG: EV in ADR ist vergleichbar, aber PnL in R nicht direkt,")
    w(f"     weil 1R bei OD-SL ein groesserer Betrag ist.")
    w()

    # ===================================================================
    # Baseline-Verifikation
    # ===================================================================
    w("=" * 72)
    w("VERIFIKATION")
    w("=" * 72)
    w()
    w("A1 Baseline (with_gap) muss d10_trailing.py TRAIL_D reproduzieren:")
    w("  Erwartet: GapUp N=205, EV=+0.149, WR=55.6%")
    w("            GapDn N=181, EV=+0.026, WR=54.1%")
    df_vu, _, _ = all_results[('GapUp + OD_Long', 'baseline')]
    df_vd, _, _ = all_results[('GapDn + OD_Short', 'baseline')]
    if len(df_vu) >= 10:
        w(f"  Ergebnis:  GapUp N={len(df_vu)}, EV={df_vu['pnl_adr'].mean():+.3f}, WR={df_vu['winner'].mean()*100:.1f}%")
    if len(df_vd) >= 10:
        w(f"             GapDn N={len(df_vd)}, EV={df_vd['pnl_adr'].mean():+.3f}, WR={df_vd['winner'].mean()*100:.1f}%")
    w()

    n_wg = len(df_vu) + len(df_vd)
    df_ag_u, _, _ = all_results[('GapUp + OD_Short', 'baseline')]
    df_ag_d, _, _ = all_results[('GapDn + OD_Long', 'baseline')]
    n_ag = len(df_ag_u) + len(df_ag_d)
    n_all = n_wg + n_ag
    w(f"N-Check: with_gap={n_wg} + against_gap={n_ag} = {n_all} (soll: {len(od)})")
    w()

    # Save
    output = "\n".join(L)
    outpath = 'D12_Ergebnis.txt'
    with open(outpath, 'w', encoding='utf-8') as f:
        f.write(output + "\n")

    sys.stdout.buffer.write((output + "\n").encode('utf-8'))
    print(f"\nGespeichert: {outpath}", file=sys.stderr)


if __name__ == '__main__':
    main()
