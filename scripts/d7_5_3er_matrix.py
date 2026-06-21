"""
Durchlauf 7.5 -- Aufgabe 3: Volle 3er-Matrix PM/RTH5 x RVOL_5 x Gap_in_ADR (IS)
96 Zellen (48 pro Gap-Richtung), plus alternative 3x3x3 = 54 Zellen
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
h1['cl_num'] = (h1['rth_close'] - h1['rth_low']) / (h1['rth_high'] - h1['rth_low'])
h1['cl_num'] = h1['cl_num'].clip(0, 1)
h1['prev_close_calc'] = h1['rth_open'] / (1 + h1['gap_pct'] / 100)
h1['gap_filled'] = False
h1.loc[h1['gap_dir'] == 'GapUp', 'gap_filled'] = h1.loc[h1['gap_dir'] == 'GapUp', 'rth_low'] <= h1.loc[h1['gap_dir'] == 'GapUp', 'prev_close_calc']
h1.loc[h1['gap_dir'] == 'GapDown', 'gap_filled'] = h1.loc[h1['gap_dir'] == 'GapDown', 'rth_high'] >= h1.loc[h1['gap_dir'] == 'GapDown', 'prev_close_calc']
h1['day_range_adr'] = (h1['rth_high'] - h1['rth_low']) / h1['adr_10']
h1['od_with_gap'] = h1['od_direction'] == 'with_gap'

print(f"H1 (IS): {len(h1)} Gapper", file=sys.stderr)

# === BUCKETS ===
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

h1['pm5'] = h1['pm_rth5'].apply(pm5_bucket)
h1['rv5'] = h1['rvol_5'].apply(rv5_bucket)
h1['gap'] = h1['gap_size_in_adr'].apply(gap_bucket)

# Alt buckets
def pm5_alt(x):
    if x < 0.50: return 'PM5_LO'
    elif x < 1.00: return 'PM5_MID'
    else: return 'PM5_HI'
def rv5_alt(x):
    if x < 3: return 'RV5_LO'
    elif x < 7: return 'RV5_MID'
    else: return 'RV5_HI'

h1['pm5a'] = h1['pm_rth5'].apply(pm5_alt)
h1['rv5a'] = h1['rvol_5'].apply(rv5_alt)

pm5_order = ['PM5_LO', 'PM5_MID', 'PM5_HI', 'PM5_EX']
rv5_order = ['RV5_LO', 'RV5_MID', 'RV5_HI', 'RV5_EX']
gap_order = ['GAP_SM', 'GAP_MD', 'GAP_LG']
pm5a_order = ['PM5_LO', 'PM5_MID', 'PM5_HI']
rv5a_order = ['RV5_LO', 'RV5_MID', 'RV5_HI']
dir_order = ['GapUp', 'GapDown']

out = open('results/d7_5_3er_matrix.txt', 'w', encoding='utf-8')

def p(text=''):
    out.write(text + '\n')

def drift_tag(d):
    if d > 0.30: return '[CONT+]'
    elif d > 0.10: return '[CONT] '
    elif d > -0.10: return '[FLAT] '
    elif d > -0.30: return '[FADE] '
    else: return '[FADE+]'

def bootstrap_ci(data, n_boot=1000, ci=0.95):
    """Bootstrap confidence interval for mean."""
    if len(data) < 5:
        return np.nan, np.nan
    means = []
    for _ in range(n_boot):
        sample = np.random.choice(data, size=len(data), replace=True)
        means.append(np.mean(sample))
    lo = np.percentile(means, (1-ci)/2*100)
    hi = np.percentile(means, (1+ci)/2*100)
    return lo, hi

def compute_cell(subset, do_bootstrap=False):
    n = len(subset)
    if n < 10:
        return None
    m = {
        'N': n,
        'fd': subset['full_drift'].mean(),
        'fd_med': subset['full_drift'].median(),
        'rd': subset['rest_drift'].mean(),
        'rd_1100': subset['rest_drift_1100'].mean(),
        'fill_rate': subset['gap_filled'].mean() * 100,
        'day_range': subset['day_range_adr'].mean(),
        'cl': subset['cl_num'].mean(),
        'od_with': subset['od_with_gap'].mean() * 100,
    }
    if do_bootstrap and n >= 20:
        lo, hi = bootstrap_ci(subset['full_drift'].values)
        m['fd_ci_lo'] = lo
        m['fd_ci_hi'] = hi
    return m

# ============================================================
# AUFGABE 3: VOLLE 3er-MATRIX (4x4x3)
# ============================================================
p("=" * 100)
p("DURCHLAUF 7.5 -- AUFGABE 3: VOLLE 3er-MATRIX PM/RTH5 x RVOL_5 x Gap_in_ADR")
p("=" * 100)
p()

all_rows = []
for direction in dir_order:
    sub = h1[h1['gap_dir'] == direction]
    p(f"\n{'='*100}")
    p(f"--- {direction} (N={len(sub)}) ---")
    p(f"{'='*100}")

    hdr = f"{'PM5':>7s} {'RV5':>7s} {'GAP':>7s} | {'N':>5s} | {'fd_m':>7s} {'tag':>7s} | {'rd_m':>7s} {'r1100':>7s} | {'Fill%':>5s} {'DayR':>5s} {'CL':>4s} {'OD%':>5s}"
    p(hdr)
    p("-" * len(hdr))

    for pm in pm5_order:
        for rv in rv5_order:
            for g in gap_order:
                mask = (sub['pm5'] == pm) & (sub['rv5'] == rv) & (sub['gap'] == g)
                cell = sub[mask]
                m = compute_cell(cell, do_bootstrap=True)
                if m is None:
                    continue

                tag = drift_tag(m['fd'])
                ntag = '' if m['N'] >= 50 else (' [LOW N]' if m['N'] >= 20 else ' ***')

                ci_str = ''
                if 'fd_ci_lo' in m:
                    ci_str = f" CI[{m['fd_ci_lo']:+.2f},{m['fd_ci_hi']:+.2f}]"

                line = (f"{pm:>7s} {rv:>7s} {g:>7s} | {m['N']:>5d} | "
                       f"{m['fd']:>+7.3f} {tag} | "
                       f"{m['rd']:>+7.3f} {m['rd_1100']:>+7.3f} | "
                       f"{m['fill_rate']:>5.1f} {m['day_range']:>5.2f} {m['cl']:>.2f} {m['od_with']:>5.1f}"
                       f"{ntag}{ci_str}")
                p(line)

                all_rows.append({
                    'dir': direction, 'pm5': pm, 'rv5': rv, 'gap': g,
                    'N': m['N'], 'fd': m['fd'], 'fd_med': m['fd_med'],
                    'rd': m['rd'], 'rd_1100': m['rd_1100'],
                    'fill_rate': m['fill_rate'], 'day_range': m['day_range'],
                    'cl': m['cl'], 'od_with': m['od_with'],
                    'fd_ci_lo': m.get('fd_ci_lo', np.nan),
                    'fd_ci_hi': m.get('fd_ci_hi', np.nan),
                })

# ============================================================
# ALTERNATIVE 3x3x3 MATRIX
# ============================================================
p(f"\n\n{'='*100}")
p("ALTERNATIVE 3x3x3 MATRIX (robustere Zellen)")
p(f"{'='*100}")

alt_rows = []
for direction in dir_order:
    sub = h1[h1['gap_dir'] == direction]
    p(f"\n--- {direction} (N={len(sub)}) ---")

    hdr = f"{'PM5':>7s} {'RV5':>7s} {'GAP':>7s} | {'N':>5s} | {'fd_m':>7s} {'tag':>7s} | {'rd_m':>7s} {'r1100':>7s} | {'Fill%':>5s} {'DayR':>5s} {'CL':>4s} {'OD%':>5s}"
    p(hdr)
    p("-" * len(hdr))

    for pm in pm5a_order:
        for rv in rv5a_order:
            for g in gap_order:
                mask = (sub['pm5a'] == pm) & (sub['rv5a'] == rv) & (sub['gap'] == g)
                cell = sub[mask]
                m = compute_cell(cell, do_bootstrap=True)
                if m is None:
                    continue

                tag = drift_tag(m['fd'])
                ntag = '' if m['N'] >= 50 else (' [LOW N]' if m['N'] >= 20 else ' ***')

                ci_str = ''
                if 'fd_ci_lo' in m:
                    ci_str = f" CI[{m['fd_ci_lo']:+.2f},{m['fd_ci_hi']:+.2f}]"

                line = (f"{pm:>7s} {rv:>7s} {g:>7s} | {m['N']:>5d} | "
                       f"{m['fd']:>+7.3f} {tag} | "
                       f"{m['rd']:>+7.3f} {m['rd_1100']:>+7.3f} | "
                       f"{m['fill_rate']:>5.1f} {m['day_range']:>5.2f} {m['cl']:>.2f} {m['od_with']:>5.1f}"
                       f"{ntag}{ci_str}")
                p(line)

                alt_rows.append({
                    'dir': direction, 'pm5': pm, 'rv5': rv, 'gap': g,
                    'N': m['N'], 'fd': m['fd'], 'fd_med': m['fd_med'],
                    'rd': m['rd'], 'rd_1100': m['rd_1100'],
                    'fill_rate': m['fill_rate'], 'day_range': m['day_range'],
                    'cl': m['cl'], 'od_with': m['od_with'],
                    'fd_ci_lo': m.get('fd_ci_lo', np.nan),
                    'fd_ci_hi': m.get('fd_ci_hi', np.nan),
                })

# ============================================================
# ZUSAMMENFASSUNG
# ============================================================
cells_df = pd.DataFrame(all_rows)
alt_df = pd.DataFrame(alt_rows)

p(f"\n\n{'='*100}")
p("ZUSAMMENFASSUNG: Top/Bottom Zellen (4x4x3 Matrix)")
p(f"{'='*100}")

p(f"\nTotal Zellen mit N>=10: {len(cells_df)}")
p(f"  davon N>=50: {(cells_df['N']>=50).sum()}")
p(f"  davon N>=20: {(cells_df['N']>=20).sum()}")

p("\n--- Top-10 CONTINUATION (hoechster full_drift, N>=20) ---")
top = cells_df[cells_df['N'] >= 20].nlargest(10, 'fd')
p(f"{'Dir':>7s} {'PM5':>7s} {'RV5':>7s} {'GAP':>7s} | {'N':>5s} | {'fd':>7s} {'rd':>7s} {'CL':>4s} {'Fill%':>5s} | CI_95%")
for _, r in top.iterrows():
    ci = f"[{r['fd_ci_lo']:+.2f},{r['fd_ci_hi']:+.2f}]" if not np.isnan(r['fd_ci_lo']) else "n/a"
    p(f"{r['dir']:>7s} {r['pm5']:>7s} {r['rv5']:>7s} {r['gap']:>7s} | {r['N']:>5.0f} | {r['fd']:>+7.3f} {r['rd']:>+7.3f} {r['cl']:>.2f} {r['fill_rate']:>5.1f} | {ci}")

p("\n--- Top-10 FADE (niedrigster full_drift, N>=20) ---")
bot = cells_df[cells_df['N'] >= 20].nsmallest(10, 'fd')
p(f"{'Dir':>7s} {'PM5':>7s} {'RV5':>7s} {'GAP':>7s} | {'N':>5s} | {'fd':>7s} {'rd':>7s} {'CL':>4s} {'Fill%':>5s} | CI_95%")
for _, r in bot.iterrows():
    ci = f"[{r['fd_ci_lo']:+.2f},{r['fd_ci_hi']:+.2f}]" if not np.isnan(r['fd_ci_lo']) else "n/a"
    p(f"{r['dir']:>7s} {r['pm5']:>7s} {r['rv5']:>7s} {r['gap']:>7s} | {r['N']:>5.0f} | {r['fd']:>+7.3f} {r['rd']:>+7.3f} {r['cl']:>.2f} {r['fill_rate']:>5.1f} | {ci}")

p("\n--- Top-10 FLAT (|full_drift| < 0.10, hoechstes N) ---")
flat = cells_df[(cells_df['fd'].abs() < 0.10) & (cells_df['N'] >= 20)].nlargest(10, 'N')
p(f"{'Dir':>7s} {'PM5':>7s} {'RV5':>7s} {'GAP':>7s} | {'N':>5s} | {'fd':>7s} {'rd':>7s} {'CL':>4s} {'Fill%':>5s} | {'DayR':>5s}")
for _, r in flat.iterrows():
    p(f"{r['dir']:>7s} {r['pm5']:>7s} {r['rv5']:>7s} {r['gap']:>7s} | {r['N']:>5.0f} | {r['fd']:>+7.3f} {r['rd']:>+7.3f} {r['cl']:>.2f} {r['fill_rate']:>5.1f} | {r['day_range']:>5.2f}")

# Same for alt
p(f"\n\n{'='*100}")
p("ZUSAMMENFASSUNG: Top/Bottom Zellen (3x3x3 Matrix)")
p(f"{'='*100}")

p(f"\nTotal Zellen mit N>=10: {len(alt_df)}")

p("\n--- Top-10 CONTINUATION (3x3x3, N>=20) ---")
top_a = alt_df[alt_df['N'] >= 20].nlargest(10, 'fd')
p(f"{'Dir':>7s} {'PM5':>7s} {'RV5':>7s} {'GAP':>7s} | {'N':>5s} | {'fd':>7s} {'rd':>7s} {'CL':>4s} {'Fill%':>5s} | CI_95%")
for _, r in top_a.iterrows():
    ci = f"[{r['fd_ci_lo']:+.2f},{r['fd_ci_hi']:+.2f}]" if not np.isnan(r['fd_ci_lo']) else "n/a"
    p(f"{r['dir']:>7s} {r['pm5']:>7s} {r['rv5']:>7s} {r['gap']:>7s} | {r['N']:>5.0f} | {r['fd']:>+7.3f} {r['rd']:>+7.3f} {r['cl']:>.2f} {r['fill_rate']:>5.1f} | {ci}")

p("\n--- Top-10 FADE (3x3x3, N>=20) ---")
bot_a = alt_df[alt_df['N'] >= 20].nsmallest(10, 'fd')
p(f"{'Dir':>7s} {'PM5':>7s} {'RV5':>7s} {'GAP':>7s} | {'N':>5s} | {'fd':>7s} {'rd':>7s} {'CL':>4s} {'Fill%':>5s} | CI_95%")
for _, r in bot_a.iterrows():
    ci = f"[{r['fd_ci_lo']:+.2f},{r['fd_ci_hi']:+.2f}]" if not np.isnan(r['fd_ci_lo']) else "n/a"
    p(f"{r['dir']:>7s} {r['pm5']:>7s} {r['rv5']:>7s} {r['gap']:>7s} | {r['N']:>5.0f} | {r['fd']:>+7.3f} {r['rd']:>+7.3f} {r['cl']:>.2f} {r['fill_rate']:>5.1f} | {ci}")

# Save raw
cells_df.to_parquet('results/d7_5_3er_matrix_4x4x3.parquet', index=False)
alt_df.to_parquet('results/d7_5_3er_matrix_3x3x3.parquet', index=False)

out.close()
print("Done!", file=sys.stderr)
