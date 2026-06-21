"""
Durchlauf 7.5 -- Aufgabe 1: Verteilung der 96 Zellen verstehen
3D-Verteilungstabelle, marginale Verteilungen, Korrelationen
"""
import pandas as pd
import numpy as np
from scipy import stats
import sys, os

os.environ['PYTHONIOENCODING'] = 'utf-8'

# === LOAD DATA ===
df = pd.read_parquet('data/metadata/metadata_v7.parquet')

# H1 (IS): 2021-02-21 bis 2023-12-31
h1 = df[(df['date'] >= '2021-02-21') & (df['date'] <= '2023-12-31')].copy()
h1 = h1.dropna(subset=['pm_rth5', 'rvol_5', 'gap_size_in_adr', 'full_drift'])

# Normalize gap_direction
h1['gap_dir'] = h1['gap_direction'].map({'up': 'GapUp', 'down': 'GapDown', 'GapUp': 'GapUp', 'GapDown': 'GapDown'})
print(f"H1 (IS) after dropping NaN: {len(h1)} Gapper", file=sys.stderr)
print(f"gap_dir values: {h1['gap_dir'].value_counts().to_dict()}", file=sys.stderr)

# === BUCKET DEFINITIONS ===
def pm5_bucket(x):
    if x < 0.30: return 'PM5_LO'
    elif x < 0.70: return 'PM5_MID'
    elif x < 1.50: return 'PM5_HI'
    else: return 'PM5_EX'

def rv5_bucket(x):
    if x < 2: return 'RV5_LO'
    elif x < 5: return 'RV5_MID'
    elif x < 10: return 'RV5_HI'
    else: return 'RV5_EX'

def gap_bucket(x):
    if x < 1: return 'GAP_SM'
    elif x < 2: return 'GAP_MD'
    else: return 'GAP_LG'

h1['pm5_bkt'] = h1['pm_rth5'].apply(pm5_bucket)
h1['rv5_bkt'] = h1['rvol_5'].apply(rv5_bucket)
h1['gap_bkt'] = h1['gap_size_in_adr'].apply(gap_bucket)

pm5_order = ['PM5_LO', 'PM5_MID', 'PM5_HI', 'PM5_EX']
rv5_order = ['RV5_LO', 'RV5_MID', 'RV5_HI', 'RV5_EX']
gap_order = ['GAP_SM', 'GAP_MD', 'GAP_LG']
dir_order = ['GapUp', 'GapDown']

out = open('results/d7_5_distribution.txt', 'w', encoding='utf-8')

def p(text=''):
    print(text)
    out.write(text + '\n')

p("=" * 90)
p("DURCHLAUF 7.5 -- AUFGABE 1: VERTEILUNG DER BUCKET-ZELLEN")
p("=" * 90)
p(f"\nH1 (IS): {len(h1)} Gapper (GapUp={h1['gap_dir'].value_counts().get('GapUp',0)}, GapDown={h1['gap_dir'].value_counts().get('GapDown',0)})")
p()

# === 1a: 3D-Verteilungstabelle ===
p("=" * 90)
p("1a: 3D-VERTEILUNGSTABELLE (N pro Zelle)")
p("    Zeilen = PM5 x RV5 (16), Spalten = GAP x Direction (6)")
p("    N>=50 = robust, N 20-49 = [D], N 10-19 = [*], N<10 = [**]")
p("=" * 90)
p()

# Build header
cols = []
for gap in gap_order:
    for d in dir_order:
        short_d = 'Up' if d == 'GapUp' else 'Dn'
        cols.append(f"{gap}_{short_d}")

header = f"{'PM5':>8s} {'RV5':>8s} |"
for c in cols:
    header += f" {c:>9s} |"
header += " TOTAL"
p(header)
p("-" * len(header))

total_robust = 0
total_thin = 0
total_star = 0
total_unusable = 0

for pm in pm5_order:
    for rv in rv5_order:
        row = f"{pm:>8s} {rv:>8s} |"
        row_total = 0
        for gap in gap_order:
            for d in dir_order:
                mask = (h1['pm5_bkt'] == pm) & (h1['rv5_bkt'] == rv) & (h1['gap_bkt'] == gap) & (h1['gap_dir'] == d)
                n = mask.sum()
                row_total += n
                if n >= 50:
                    tag = f"{n:>9d}"
                    total_robust += 1
                elif n >= 20:
                    tag = f"{n:>6d}[D]"
                    total_thin += 1
                elif n >= 10:
                    tag = f"{n:>5d} [*]"
                    total_star += 1
                else:
                    tag = f"{n:>4d} [**]"
                    total_unusable += 1
                row += f" {tag} |"
        row += f" {row_total:>5d}"
        p(row)
    p("-" * len(header))

# Totals row
row = f"{'TOTAL':>8s} {'':>8s} |"
grand_total = 0
for gap in gap_order:
    for d in dir_order:
        mask = (h1['gap_bkt'] == gap) & (h1['gap_dir'] == d)
        n = mask.sum()
        grand_total += n
        row += f" {n:>9d} |"
row += f" {grand_total:>5d}"
p(row)

p(f"\nRobuste Zellen (N>=50):  {total_robust}/96")
p(f"Duenne Zellen (N 20-49): {total_thin}/96")
p(f"Duenn* (N 10-19):        {total_star}/96")
p(f"Unbrauchbar (N<10):      {total_unusable}/96")

# === 1b: Marginale Verteilungen ===
p()
p("=" * 90)
p("1b: MARGINALE VERTEILUNGEN")
p("=" * 90)
p()

total = len(h1)
total_up = (h1['gap_dir'] == 'GapUp').sum()
total_dn = (h1['gap_dir'] == 'GapDown').sum()

for bkt_name, bkt_col, bkt_order_list in [
    ('PM/RTH5', 'pm5_bkt', pm5_order),
    ('RVOL_5', 'rv5_bkt', rv5_order),
    ('Gap_in_ADR', 'gap_bkt', gap_order),
]:
    p(f"--- {bkt_name} ---")
    p(f"{'Bucket':<12s} | {'All':>6s} {'%':>6s} | {'GapUp':>6s} {'%':>6s} | {'GapDn':>6s} {'%':>6s}")
    for bkt in bkt_order_list:
        n_all = (h1[bkt_col] == bkt).sum()
        n_up = ((h1[bkt_col] == bkt) & (h1['gap_dir'] == 'GapUp')).sum()
        n_dn = ((h1[bkt_col] == bkt) & (h1['gap_dir'] == 'GapDown')).sum()
        pct_up = n_up / total_up * 100 if total_up > 0 else 0
        pct_dn = n_dn / total_dn * 100 if total_dn > 0 else 0
        p(f"{bkt:<12s} | {n_all:>6d} {n_all/total*100:>5.1f}% | {n_up:>6d} {pct_up:>5.1f}% | {n_dn:>6d} {pct_dn:>5.1f}%")
    p(f"{'TOTAL':<12s} | {total:>6d} 100.0% | {total_up:>6d} 100.0% | {total_dn:>6d} 100.0%")
    p()

# === 1c: Korrelationen ===
p("=" * 90)
p("1c: KORRELATIONEN ZWISCHEN DEN DREI PARAMETERN (Spearman)")
p("=" * 90)
p()

for label, col_a, col_b in [
    ('PM/RTH5 vs RVOL_5', 'pm_rth5', 'rvol_5'),
    ('PM/RTH5 vs Gap_in_ADR', 'pm_rth5', 'gap_size_in_adr'),
    ('RVOL_5 vs Gap_in_ADR', 'rvol_5', 'gap_size_in_adr'),
]:
    rho_all, p_all = stats.spearmanr(h1[col_a], h1[col_b])
    mask_up = h1['gap_dir'] == 'GapUp'
    mask_dn = h1['gap_dir'] == 'GapDown'
    rho_up, p_up = stats.spearmanr(h1.loc[mask_up, col_a], h1.loc[mask_up, col_b])
    rho_dn, p_dn = stats.spearmanr(h1.loc[mask_dn, col_a], h1.loc[mask_dn, col_b])

    interp = "stark" if abs(rho_all) > 0.5 else ("moderat" if abs(rho_all) > 0.3 else ("schwach" if abs(rho_all) > 0.1 else "sehr schwach"))

    p(f"{label}:")
    p(f"  All:     rho = {rho_all:+.4f} (p = {p_all:.2e})")
    p(f"  GapUp:   rho = {rho_up:+.4f} (p = {p_up:.2e})")
    p(f"  GapDown: rho = {rho_dn:+.4f} (p = {p_dn:.2e})")
    p(f"  -> {interp} korreliert")
    p()

p("INTERPRETATION:")
p("  Wenn alle drei Parameter schwach korreliert sind (<0.3),")
p("  bringt die 3er-Kreuzung maximale Zusatzinformation.")
p("  Bei starker Korrelation (>0.5) redundante Information.")

# === ALTERNATIVE BUCKETS (3x3x3) ===
p()
p("=" * 90)
p("ALTERNATIVE BUCKETS (3x3x3 = 54 Zellen) fuer robustere Analyse")
p("=" * 90)
p()

def pm5_alt(x):
    if x < 0.50: return 'PM5_LO'
    elif x < 1.00: return 'PM5_MID'
    else: return 'PM5_HI'

def rv5_alt(x):
    if x < 3: return 'RV5_LO'
    elif x < 7: return 'RV5_MID'
    else: return 'RV5_HI'

h1['pm5_alt'] = h1['pm_rth5'].apply(pm5_alt)
h1['rv5_alt'] = h1['rvol_5'].apply(rv5_alt)

pm5_alt_order = ['PM5_LO', 'PM5_MID', 'PM5_HI']
rv5_alt_order = ['RV5_LO', 'RV5_MID', 'RV5_HI']

header_alt = f"{'PM5':>8s} {'RV5':>8s} |"
for c in cols:
    header_alt += f" {c:>9s} |"
header_alt += " TOTAL"
p(header_alt)
p("-" * len(header_alt))

alt_robust = 0
alt_thin = 0
alt_star = 0
alt_unusable = 0

for pm in pm5_alt_order:
    for rv in rv5_alt_order:
        row = f"{pm:>8s} {rv:>8s} |"
        row_total = 0
        for gap in gap_order:
            for d in dir_order:
                mask = (h1['pm5_alt'] == pm) & (h1['rv5_alt'] == rv) & (h1['gap_bkt'] == gap) & (h1['gap_dir'] == d)
                n = mask.sum()
                row_total += n
                if n >= 50:
                    tag = f"{n:>9d}"
                    alt_robust += 1
                elif n >= 20:
                    tag = f"{n:>6d}[D]"
                    alt_thin += 1
                elif n >= 10:
                    tag = f"{n:>5d} [*]"
                    alt_star += 1
                else:
                    tag = f"{n:>4d} [**]"
                    alt_unusable += 1
                row += f" {tag} |"
        row += f" {row_total:>5d}"
        p(row)
    p("-" * len(header_alt))

p(f"\nAlternative 3x3x3: Robuste Zellen (N>=50): {alt_robust}/54")
p(f"Duenne Zellen (N 20-49): {alt_thin}/54")
p(f"Duenn* (N 10-19): {alt_star}/54")
p(f"Unbrauchbar (N<10): {alt_unusable}/54")

# Decision
if total_robust > 40:
    p(f"\nEMPFEHLUNG: 4x4x3 Buckets nutzbar ({total_robust} robuste Zellen)")
else:
    p(f"\nEMPFEHLUNG: Beide Bucket-Varianten parallel analysieren.")
    p(f"  4x4x3 fuer Detailblick (manche Zellen duenn)")
    p(f"  3x3x3 fuer robuste Aussagen")

out.close()
print("Done! Results in results/d7_5_distribution.txt", file=sys.stderr)
