"""
Durchlauf 7.5 -- Aufgabe 7: Vergleich 9:35 vs 10:00 Parameter (IS)
"""
import pandas as pd
import numpy as np
from scipy import stats
import sys, warnings
warnings.filterwarnings('ignore')

# === LOAD ===
df = pd.read_parquet('data/metadata/metadata_v7.parquet')
h1 = df[(df['date'] >= '2021-02-21') & (df['date'] <= '2023-12-31')].copy()
h1 = h1.dropna(subset=['pm_rth5', 'rvol_5', 'gap_size_in_adr', 'full_drift'])
h1['gap_dir'] = h1['gap_direction'].map({'up': 'GapUp', 'down': 'GapDown'})
h1['cl_num'] = ((h1['rth_close'] - h1['rth_low']) / (h1['rth_high'] - h1['rth_low'])).clip(0, 1)

print(f"H1: {len(h1)}", file=sys.stderr)

# Check if pm_rth30 and rvol_30 exist
has_pm30 = 'pm_rth30' in h1.columns
has_rvol30 = 'rvol_30' in h1.columns or 'rvol_at_time_30min' in h1.columns

# Find the 30-min columns
rvol30_col = None
pm30_col = None
for c in h1.columns:
    if 'pm_rth30' in c.lower() or 'pm_rth_30' in c.lower():
        pm30_col = c
    if 'rvol_30' in c.lower() or 'rvol_at_time_30' in c.lower():
        rvol30_col = c

# Also check for close_1000 (close at 10:00)
close_1000_col = None
for c in h1.columns:
    if c in ['close_1000', 'close_10am', 'price_at_1000']:
        close_1000_col = c

print(f"pm30_col={pm30_col}, rvol30_col={rvol30_col}, close_1000_col={close_1000_col}", file=sys.stderr)
print(f"Available columns with 30: {[c for c in h1.columns if '30' in c.lower()]}", file=sys.stderr)
print(f"Available columns with 1000: {[c for c in h1.columns if '1000' in c.lower()]}", file=sys.stderr)

out = open('results/d7_5_10uhr_vergleich.txt', 'w', encoding='utf-8')
def p(text=''):
    out.write(text + '\n')

p("=" * 100)
p("DURCHLAUF 7.5 -- AUFGABE 7: VERGLEICH 9:35 vs 10:00 PARAMETER")
p("=" * 100)

# === 7a: 10:00-Dreieck mit PM/RTH30 x RVOL_30 x Gap_in_ADR ===
p(f"\n{'='*100}")
p("7a: 10:00-DREIECK (PM/RTH30 x RVOL_30 x Gap_in_ADR)")
p(f"{'='*100}")

# Try to find or compute PM/RTH30 and RVOL_30
# They might be in the original metadata with different names
vol_cols = [c for c in h1.columns if 'vol' in c.lower() or 'rvol' in c.lower()]
pm_cols = [c for c in h1.columns if 'pm' in c.lower() or 'premarket' in c.lower()]
print(f"Volume-related cols: {vol_cols}", file=sys.stderr)
print(f"PM-related cols: {pm_cols}", file=sys.stderr)

if pm30_col is None:
    # Check if we have pm_volume_total and rth_volume_30min
    for c in h1.columns:
        if 'pm_volume' in c.lower():
            print(f"  Found pm_volume col: {c}", file=sys.stderr)
        if 'rth_volume' in c.lower() or 'volume_30' in c.lower():
            print(f"  Found rth_volume col: {c}", file=sys.stderr)

# If columns don't exist, we can approximate from 5-min data
# PM/RTH30 ~ PM/RTH5 * (RTH5_vol / RTH30_vol)
# But we don't have RTH30 vol directly. Let's check what we have.

if pm30_col and rvol30_col:
    h1_30 = h1.dropna(subset=[pm30_col, rvol30_col]).copy()
    p(f"\nPM/RTH30 column: {pm30_col}")
    p(f"RVOL_30 column: {rvol30_col}")
    p(f"N with both: {len(h1_30)}")

    # Define 10:00 buckets
    def pm30_bucket(x):
        if x < 0.10: return 'PM30_LO'
        elif x < 0.30: return 'PM30_MID'
        else: return 'PM30_HI'

    def rv30_bucket(x):
        if x < 2: return 'RV30_LO'
        elif x < 5: return 'RV30_MID'
        elif x < 10: return 'RV30_HI'
        else: return 'RV30_EX'

    h1_30['pm30'] = h1_30[pm30_col].apply(pm30_bucket)
    h1_30['rv30'] = h1_30[rvol30_col].apply(rv30_bucket)
    h1_30['gap'] = h1_30['gap_size_in_adr'].apply(lambda x: 'GAP_SM' if x < 1 else ('GAP_MD' if x < 2 else 'GAP_LG'))

    pm30_order = ['PM30_LO', 'PM30_MID', 'PM30_HI']
    rv30_order = ['RV30_LO', 'RV30_MID', 'RV30_HI', 'RV30_EX']
    gap_order = ['GAP_SM', 'GAP_MD', 'GAP_LG']

    for direction in ['GapUp', 'GapDown']:
        sub = h1_30[h1_30['gap_dir'] == direction]
        p(f"\n--- {direction} (N={len(sub)}) ---")

        hdr = f"  {'PM30':>7s} {'RV30':>7s} {'GAP':>7s} | {'N':>5s} | {'fd':>7s} {'rd':>7s} {'r1100':>7s} | {'CL':>4s}"
        p(hdr)
        p("  " + "-" * (len(hdr)-2))

        for pm in pm30_order:
            for rv in rv30_order:
                for g in gap_order:
                    mask = (sub['pm30'] == pm) & (sub['rv30'] == rv) & (sub['gap'] == g)
                    cell = sub[mask]
                    if len(cell) < 10:
                        continue
                    ntag = '' if len(cell) >= 50 else (' [LN]' if len(cell) >= 20 else ' ***')
                    p(f"  {pm:>7s} {rv:>7s} {g:>7s} | {len(cell):>5d} | "
                      f"{cell['full_drift'].mean():>+7.3f} {cell['rest_drift'].mean():>+7.3f} "
                      f"{cell['rest_drift_1100'].mean():>+7.3f} | {cell['cl_num'].mean():>.2f}{ntag}")
else:
    p("\n  PM/RTH30 und/oder RVOL_30 Spalten NICHT in metadata_v7 vorhanden.")
    p("  Verwende pm_rth5 und rvol_5 als Proxy-Vergleich.")
    p("  Der Vergleich 9:35 vs 10:00 ist nur approximativ moeglich.")

# === 7b: 9:35 vs 10:00 Spread-Vergleich ===
p(f"\n\n{'='*100}")
p("7b: VERGLEICH 9:35-DREIECK vs 10:00-DREIECK")
p(f"{'='*100}")

# Use 3x3x3 results from Aufgabe 3
matrix_935 = pd.read_parquet('results/d7_5_3er_matrix_3x3x3.parquet')

p("\n9:35-Dreieck: Drift-Spread (Max minus Min full_drift) pro Gap-Richtung:")
for direction in ['GapUp', 'GapDown']:
    cells = matrix_935[(matrix_935['dir'] == direction) & (matrix_935['N'] >= 20)]
    if len(cells) > 0:
        spread = cells['fd'].max() - cells['fd'].min()
        p(f"  {direction}: Spread = {spread:.3f} ADR ({cells['fd'].max():+.3f} bis {cells['fd'].min():+.3f})")

# === 7c: Rest-Drift ab 10:00 ===
p(f"\n\n{'='*100}")
p("7c: REST-DRIFT AB 10:00 (rest_drift_1000 bis Close)")
p(f"{'='*100}")

# rest_drift = close - close_935 / adr (in gap dir)
# rest_drift_1000 = close_1000 - close_935 / adr (in gap dir)
# rest_from_1000 = rest_drift - rest_drift_1000

if 'rest_drift_1000' in h1.columns:
    h1['rest_from_1000'] = h1['rest_drift'] - h1['rest_drift_1000']

    # 3x3x3 buckets
    h1['pm5'] = h1['pm_rth5'].apply(lambda x: 'PM5_LO' if x < 0.50 else ('PM5_MID' if x < 1.00 else 'PM5_HI'))
    h1['rv5'] = h1['rvol_5'].apply(lambda x: 'RV5_LO' if x < 3 else ('RV5_MID' if x < 7 else 'RV5_HI'))
    h1['gap_b'] = h1['gap_size_in_adr'].apply(lambda x: 'GAP_SM' if x < 1 else ('GAP_MD' if x < 2 else 'GAP_LG'))

    for direction in ['GapUp', 'GapDown']:
        sub = h1[h1['gap_dir'] == direction]
        p(f"\n--- {direction} ---")
        p(f"  {'PM5':>7s} {'RV5':>7s} {'GAP':>7s} | {'N':>5s} | {'rd_full':>7s} {'rd_1000':>7s} {'rd_from1000':>11s} | {'pct_done_1000':>13s}")
        p(f"  {'-'*80}")

        for pm in ['PM5_LO', 'PM5_MID', 'PM5_HI']:
            for rv in ['RV5_LO', 'RV5_MID', 'RV5_HI']:
                for g in ['GAP_SM', 'GAP_MD', 'GAP_LG']:
                    mask = (sub['pm5'] == pm) & (sub['rv5'] == rv) & (sub['gap_b'] == g)
                    cell = sub[mask]
                    if len(cell) < 20:
                        continue
                    rd = cell['rest_drift'].mean()
                    rd1000 = cell['rest_drift_1000'].mean()
                    rd_from = cell['rest_from_1000'].mean()
                    pct = (rd1000 / rd * 100) if abs(rd) > 0.01 else np.nan
                    pct_str = f"{pct:>+12.0f}%" if not np.isnan(pct) else "         n/a"
                    ntag = '' if len(cell) >= 50 else ' [LN]'
                    p(f"  {pm:>7s} {rv:>7s} {g:>7s} | {len(cell):>5d} | "
                      f"{rd:>+7.3f} {rd1000:>+7.3f} {rd_from:>+11.3f} | {pct_str}{ntag}")

# === 7d: Zeitwert-Analyse ===
p(f"\n\n{'='*100}")
p("7d: ZEITWERT-ANALYSE (Was verpasst man wenn man um 10:00 statt 9:35 tradet?)")
p(f"{'='*100}")
p()

if 'rest_drift_1000' in h1.columns:
    # missed_drift = Drift von 9:35 bis 10:00 = rest_drift_1000
    # Positive = Drift WAR in Gap-Richtung (man hat Continuation verpasst)
    # Negative = Drift WAR gegen Gap (man hat besseren Fade-Entry bekommen)

    p("missed_drift = rest_drift_1000 = (Close_1000 - Close_935) / ADR, in Gap-Richtung")
    p("  Positiv: Continuation verpasst (Warten kostet)")
    p("  Negativ: Fade verpasst, aber besserer Entry (Warten lohnt fuer Fader)")
    p()

    for direction in ['GapUp', 'GapDown']:
        sub = h1[h1['gap_dir'] == direction]
        p(f"--- {direction} ---")

        for pm in ['PM5_LO', 'PM5_MID', 'PM5_HI']:
            for rv in ['RV5_LO', 'RV5_MID', 'RV5_HI']:
                for g in ['GAP_SM', 'GAP_MD', 'GAP_LG']:
                    mask = (sub['pm5'] == pm) & (sub['rv5'] == rv) & (sub['gap_b'] == g)
                    cell = sub[mask]
                    if len(cell) < 20:
                        continue
                    missed = cell['rest_drift_1000'].median()
                    missed_mean = cell['rest_drift_1000'].mean()
                    if abs(missed_mean) > 0.05:
                        interp = "WARTEN KOSTET" if missed_mean > 0 else "Warten lohnt (besserer Fade-Entry)"
                    else:
                        interp = "Kein Unterschied"
                    ntag = '' if len(cell) >= 50 else ' [LN]'
                    p(f"  {pm:>7s} {rv:>7s} {g:>7s} | N={len(cell):>4d} | "
                      f"missed_mean={missed_mean:>+.3f} missed_med={missed:>+.3f} | {interp}{ntag}")
        p()

# Summary
p(f"\n{'='*100}")
p("ZUSAMMENFASSUNG: 9:35 vs 10:00")
p(f"{'='*100}")

if 'rest_drift_1000' in h1.columns:
    for direction in ['GapUp', 'GapDown']:
        sub = h1[h1['gap_dir'] == direction]
        p(f"\n{direction}:")
        p(f"  Gesamt missed_drift (median): {sub['rest_drift_1000'].median():+.3f} ADR")
        p(f"  Gesamt missed_drift (mean): {sub['rest_drift_1000'].mean():+.3f} ADR")

        # By PM5 bucket only
        for pm in ['PM5_LO', 'PM5_MID', 'PM5_HI']:
            s = sub[sub['pm5'] == pm]
            p(f"  {pm}: missed_mean={s['rest_drift_1000'].mean():+.3f}, N={len(s)}")

p("\nFAZIT:")
p("  Wenn missed_drift ~0: Warten kostet nichts -> 10:00-Infos nutzen!")
p("  Wenn missed_drift >>0: Continuation bereits gelaufen -> Entry um 9:35 besser")

out.close()
print("Done!", file=sys.stderr)
