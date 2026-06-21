"""
Durchlauf 7.5 -- Aufgabe 2: 2er-Kombinations-Landkarte (IS)
PM/RTH5 x Gap, RVOL_5 x Gap, PM/RTH5 x RVOL_5
Alle Verhaltensmetriken pro Zelle
"""
import pandas as pd
import numpy as np
from scipy import stats
import sys, os, warnings
warnings.filterwarnings('ignore')

# === LOAD DATA ===
df = pd.read_parquet('data/metadata/metadata_v7.parquet')
h1 = df[(df['date'] >= '2021-02-21') & (df['date'] <= '2023-12-31')].copy()
h1 = h1.dropna(subset=['pm_rth5', 'rvol_5', 'gap_size_in_adr', 'full_drift'])
h1['gap_dir'] = h1['gap_direction'].map({'up': 'GapUp', 'down': 'GapDown'})

# Compute CL numerically
h1['cl_num'] = (h1['rth_close'] - h1['rth_low']) / (h1['rth_high'] - h1['rth_low'])
h1['cl_num'] = h1['cl_num'].clip(0, 1)

# Compute prev_close for gap fill
h1['prev_close_calc'] = h1['rth_open'] / (1 + h1['gap_pct'] / 100)

# Gap fill check
h1['gap_filled'] = False
mask_up = h1['gap_dir'] == 'GapUp'
mask_dn = h1['gap_dir'] == 'GapDown'
h1.loc[mask_up, 'gap_filled'] = h1.loc[mask_up, 'rth_low'] <= h1.loc[mask_up, 'prev_close_calc']
h1.loc[mask_dn, 'gap_filled'] = h1.loc[mask_dn, 'rth_high'] >= h1.loc[mask_dn, 'prev_close_calc']

# Day range in ADR
h1['day_range_adr'] = (h1['rth_high'] - h1['rth_low']) / h1['adr_10']

# OD with gap
h1['od_with_gap'] = False
h1.loc[(h1['gap_dir'] == 'GapUp') & (h1['od_direction'] == 'up'), 'od_with_gap'] = True
h1.loc[(h1['gap_dir'] == 'GapDown') & (h1['od_direction'] == 'down'), 'od_with_gap'] = True

# ADR pct
h1['adr_pct'] = h1['adr_10'] / h1['rth_open'] * 100

print(f"H1 (IS): {len(h1)} Gapper", file=sys.stderr)

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

out = open('results/d7_5_2er_combos.txt', 'w', encoding='utf-8')

def p(text=''):
    out.write(text + '\n')

def compute_cell_metrics(subset):
    """Compute all metrics for a cell subset."""
    n = len(subset)
    if n < 5:
        return None

    m = {}
    m['N'] = n
    m['full_drift_mean'] = subset['full_drift'].mean()
    m['full_drift_med'] = subset['full_drift'].median()
    m['rest_drift_mean'] = subset['rest_drift'].mean()
    m['rest_drift_med'] = subset['rest_drift'].median()
    m['rest_d1000_mean'] = subset['rest_drift_1000'].mean()
    m['rest_d1100_mean'] = subset['rest_drift_1100'].mean()
    m['cl_mean'] = subset['cl_num'].mean()
    m['fill_rate'] = subset['gap_filled'].mean() * 100

    filled = subset[subset['gap_filled'] == True]
    if len(filled) > 0 and 'gap_fill_time_minutes' in filled.columns:
        fill_times = filled['gap_fill_time_minutes'].dropna()
        m['fill_time_med'] = fill_times.median() if len(fill_times) > 0 else np.nan
    else:
        m['fill_time_med'] = np.nan

    m['day_range_adr'] = subset['day_range_adr'].mean()
    m['adr_pct_med'] = subset['adr_pct'].median()
    m['od_with_pct'] = subset['od_with_gap'].mean() * 100
    m['od_str_med'] = subset['od_strength'].median()

    return m

def drift_tag(d):
    if d > 0.30: return '[CONT+]'
    elif d > 0.10: return '[CONT] '
    elif d > -0.10: return '[FLAT] '
    elif d > -0.30: return '[FADE] '
    else: return '[FADE+]'

def n_tag(n):
    if n >= 50: return ''
    elif n >= 20: return ' [LOW N]'
    elif n >= 10: return ' ***'
    else: return ' [N<10]'

# === PRINT COMBO TABLE ===
def print_combo_table(combo_name, row_buckets, row_order, col_buckets, col_order, row_bkt_col, col_bkt_col):
    p(f"\n{'='*90}")
    p(f"2: {combo_name}")
    p(f"{'='*90}")

    for direction in dir_order:
        sub = h1[h1['gap_dir'] == direction]
        p(f"\n--- {direction} ---")

        # Header
        hdr = f"{'Row':>10s} {'Col':>10s} | {'N':>5s} | {'fd_m':>6s} {'fd_md':>6s} {'tag':>7s} | {'rd_m':>6s} {'r1000':>6s} {'r1100':>6s} | {'CL':>4s} | {'Fill%':>5s} {'FillT':>5s} | {'DayR':>5s} {'ADR%':>5s} | {'OD%':>5s} {'ODst':>5s}"
        p(hdr)
        p("-" * len(hdr))

        results = []
        for r in row_order:
            for c in col_order:
                mask = (sub[row_bkt_col] == r) & (sub[col_bkt_col] == c)
                cell = sub[mask]
                m = compute_cell_metrics(cell)
                if m is None:
                    continue

                tag = drift_tag(m['full_drift_mean'])
                ntag = n_tag(m['N'])

                line = (f"{r:>10s} {c:>10s} | {m['N']:>5d} | "
                       f"{m['full_drift_mean']:>+6.3f} {m['full_drift_med']:>+6.3f} {tag} | "
                       f"{m['rest_drift_mean']:>+6.3f} {m['rest_d1000_mean']:>+6.3f} {m['rest_d1100_mean']:>+6.3f} | "
                       f"{m['cl_mean']:>.2f} | "
                       f"{m['fill_rate']:>5.1f} {m['fill_time_med']:>5.0f} | " if not np.isnan(m['fill_time_med']) else
                       f"{r:>10s} {c:>10s} | {m['N']:>5d} | "
                       f"{m['full_drift_mean']:>+6.3f} {m['full_drift_med']:>+6.3f} {tag} | "
                       f"{m['rest_drift_mean']:>+6.3f} {m['rest_d1000_mean']:>+6.3f} {m['rest_d1100_mean']:>+6.3f} | "
                       f"{m['cl_mean']:>.2f} | "
                       f"{m['fill_rate']:>5.1f}   n/a | ")
                line += (f"{m['day_range_adr']:>5.2f} {m['adr_pct_med']:>5.1f} | "
                        f"{m['od_with_pct']:>5.1f} {m['od_str_med']:>5.3f}{ntag}")
                p(line)

                results.append(m)

        p(f"\n  Legende: fd_m=full_drift mean, fd_md=median, rd_m=rest_drift mean")
        p(f"  r1000/r1100=rest_drift bis 10:00/11:00, CL=Close Location (0=Low,1=High)")
        p(f"  Fill%=Gap-Fill-Rate, FillT=Median Fill-Zeit(min), DayR=DayRange/ADR")
        p(f"  ADR%=Median ADR%, OD%=OD mit Gap%, ODst=OD Strength median")

# === 2a: PM/RTH5 x Gap_in_ADR ===
print_combo_table("PM/RTH5 x Gap_in_ADR (12 Zellen x 2 Richtungen)",
                  pm5_order, pm5_order, gap_order, gap_order, 'pm5_bkt', 'gap_bkt')

# === 2b: RVOL_5 x Gap_in_ADR ===
print_combo_table("RVOL_5 x Gap_in_ADR (12 Zellen x 2 Richtungen)",
                  rv5_order, rv5_order, gap_order, gap_order, 'rv5_bkt', 'gap_bkt')

# === 2c: PM/RTH5 x RVOL_5 ===
print_combo_table("PM/RTH5 x RVOL_5 (16 Zellen x 2 Richtungen)",
                  pm5_order, pm5_order, rv5_order, rv5_order, 'pm5_bkt', 'rv5_bkt')

# === SUMMARY: Top/Bottom cells ===
p(f"\n{'='*90}")
p("ZUSAMMENFASSUNG: Top/Bottom Zellen nach full_drift")
p(f"{'='*90}")

all_cells = []
for direction in dir_order:
    sub = h1[h1['gap_dir'] == direction]

    # All 2er combos
    for combo_name, bkt1_col, bkt1_order, bkt2_col, bkt2_order in [
        ('PM5xGAP', 'pm5_bkt', pm5_order, 'gap_bkt', gap_order),
        ('RV5xGAP', 'rv5_bkt', rv5_order, 'gap_bkt', gap_order),
        ('PM5xRV5', 'pm5_bkt', pm5_order, 'rv5_bkt', rv5_order),
    ]:
        for b1 in bkt1_order:
            for b2 in bkt2_order:
                mask = (sub[bkt1_col] == b1) & (sub[bkt2_col] == b2)
                cell = sub[mask]
                m = compute_cell_metrics(cell)
                if m and m['N'] >= 10:
                    all_cells.append({
                        'dir': direction, 'combo': combo_name,
                        'b1': b1, 'b2': b2,
                        'N': m['N'], 'full_drift': m['full_drift_mean'],
                        'rest_drift': m['rest_drift_mean'], 'cl': m['cl_mean'],
                        'fill_rate': m['fill_rate']
                    })

cells_df = pd.DataFrame(all_cells)

p("\nTop-10 Continuation (hoechster full_drift, N>=20):")
top_cont = cells_df[cells_df['N'] >= 20].nlargest(10, 'full_drift')
p(f"{'Dir':>7s} {'Combo':>8s} {'B1':>10s} {'B2':>10s} | {'N':>5s} {'fd':>7s} {'rd':>7s} {'CL':>5s} {'Fill%':>6s}")
for _, r in top_cont.iterrows():
    p(f"{r['dir']:>7s} {r['combo']:>8s} {r['b1']:>10s} {r['b2']:>10s} | {r['N']:>5.0f} {r['full_drift']:>+7.3f} {r['rest_drift']:>+7.3f} {r['cl']:>5.2f} {r['fill_rate']:>6.1f}")

p("\nTop-10 Fade (niedrigster full_drift, N>=20):")
top_fade = cells_df[cells_df['N'] >= 20].nsmallest(10, 'full_drift')
p(f"{'Dir':>7s} {'Combo':>8s} {'B1':>10s} {'B2':>10s} | {'N':>5s} {'fd':>7s} {'rd':>7s} {'CL':>5s} {'Fill%':>6s}")
for _, r in top_fade.iterrows():
    p(f"{r['dir']:>7s} {r['combo']:>8s} {r['b1']:>10s} {r['b2']:>10s} | {r['N']:>5.0f} {r['full_drift']:>+7.3f} {r['rest_drift']:>+7.3f} {r['cl']:>5.2f} {r['fill_rate']:>6.1f}")

# Save raw data
cells_df.to_parquet('results/d7_5_2er_combos_raw.parquet', index=False)

out.close()
print("Done! Results in results/d7_5_2er_combos.txt", file=sys.stderr)
