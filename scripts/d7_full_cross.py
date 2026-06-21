"""
Durchlauf 7.0 — Aufgabe 4: Vollstaendige Kreuzungsmatrix
Alle 2er-Kombinationen + Top-10 3er-Kombinationen.
IS only (H1: 2021-02-21 bis 2023-12-31).
"""
import pandas as pd
import numpy as np
from itertools import combinations
import sys
import warnings
warnings.filterwarnings('ignore')

def bootstrap_ci(data, n_boot=500, ci=0.95):
    data = np.array(data, dtype=float)
    data = data[~np.isnan(data)]
    if len(data) < 5: return np.nan, np.nan
    rng = np.random.default_rng(42)
    means = [np.mean(rng.choice(data, len(data), True)) for _ in range(n_boot)]
    return np.percentile(means, (1-ci)/2*100), np.percentile(means, (1+ci)/2*100)

def low_n(n):
    if n < 20: return " *** VERY LOW N ***"
    if n < 50: return " [LOW N]"
    return ""

out = []
def w(s=""): out.append(s)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════════════════════════════════════════
print("Loading metadata_v7...", file=sys.stderr)
meta = pd.read_parquet('data/metadata/metadata_v7.parquet')
meta['date'] = pd.to_datetime(meta['date'])
h1 = meta[(meta['date'] >= '2021-02-21') & (meta['date'] <= '2023-12-31')].copy()
h1 = h1[h1['adr_10'].notna() & (h1['adr_10'] > 0)].copy()
print(f"H1: {len(h1)} gappers", file=sys.stderr)

# ══════════════════════════════════════════════════════════════════════════════
# DEFINE PARAMETER BUCKETS
# ══════════════════════════════════════════════════════════════════════════════

def assign_buckets(df):
    d = df.copy()

    # A. Gap-Groesse
    gs = d['gap_size_in_adr'].fillna(0)
    d['A_gap'] = pd.cut(gs, bins=[0, 1, 2, 999], labels=['<1', '1-2', '>2'], right=False)

    # B. RVOL_30
    rv30 = d['rvol_at_time_30min'].fillna(-1)
    d['B_rvol30'] = pd.cut(rv30, bins=[-2, 2, 5, 999], labels=['<2x', '2-5x', '>5x'], right=False)

    # C. RVOL_5
    rv5 = d['rvol_5'].fillna(-1)
    d['C_rvol5'] = pd.cut(rv5, bins=[-2, 2, 5, 999], labels=['<2x', '2-5x', '>5x'], right=False)

    # D. PM/RTH30
    pmr30 = d['pm_rth30_computed'].fillna(-1)
    d['D_pmrth30'] = pd.cut(pmr30, bins=[-2, 0.10, 0.30, 999], labels=['<10%', '10-30%', '>30%'], right=False)

    # E. PM/RTH5 (using 0.40/1.50 as preliminary thresholds)
    pmr5 = d['pm_rth5'].fillna(-1)
    d['E_pmrth5'] = pd.cut(pmr5, bins=[-2, 0.40, 1.50, 999], labels=['<40%', '40-150%', '>150%'], right=False)

    # F. OD-Richtung
    d['F_od_dir'] = d['od_direction'].fillna('unknown')

    # G. OD-Staerke
    ods = d['od_strength'].fillna(-1)
    d['G_od_str'] = pd.cut(ods, bins=[-2, 0.2, 0.5, 999], labels=['<0.2', '0.2-0.5', '>0.5'], right=False)

    # H. Erste Kerze
    conditions = []
    labels_h = []
    for _, r in d.iterrows():
        fc_dir = r.get('first_candle_dir', '')
        fc_size = r.get('first_candle_size', 0)
        if fc_dir == 'with_gap' and fc_size > 0.20:
            labels_h.append('with+gross')
        elif fc_dir == 'with_gap':
            labels_h.append('with+klein')
        else:
            labels_h.append('against')
    d['H_candle'] = labels_h

    # I. SPY
    spy = d['spy_return_day'].fillna(0)
    d['I_spy'] = pd.cut(spy, bins=[-999, -1, 1, 999], labels=['rot', 'neutral', 'gruen'], right=False)

    return d

print("Assigning buckets...", file=sys.stderr)
h1 = assign_buckets(h1)

# ══════════════════════════════════════════════════════════════════════════════
# COMPUTE METRICS FOR EACH GROUP
# ══════════════════════════════════════════════════════════════════════════════
param_cols = {
    'A': 'A_gap', 'B': 'B_rvol30', 'C': 'C_rvol5', 'D': 'D_pmrth30',
    'E': 'E_pmrth5', 'F': 'F_od_dir', 'G': 'G_od_str', 'H': 'H_candle', 'I': 'I_spy'
}

def compute_group_metrics(grp):
    n = len(grp)
    return {
        'N': n,
        'full_drift': grp['full_drift'].mean() if n > 0 else np.nan,
        'rest_drift': grp['rest_drift'].mean() if n > 0 else np.nan,
        'fill_rate': grp['gap_filled'].mean() * 100 if n > 0 and 'gap_filled' in grp.columns else np.nan,
        'cl': grp['cl'].mean() if n > 0 else np.nan,
    }

w("=" * 80)
w("AUFGABE 4a: ALLE 2er-KOMBINATIONEN")
w("=" * 80)

all_results = []
pair_count = 0

param_keys = list(param_cols.keys())

for gap_dir in ['up', 'down']:
    data = h1[h1['gap_direction'] == gap_dir].copy()
    w(f"\n{'='*60}")
    w(f"Gap{gap_dir.title()}: N={len(data)}")
    w(f"{'='*60}")

    for p1, p2 in combinations(param_keys, 2):
        col1, col2 = param_cols[p1], param_cols[p2]
        pair_label = f"{p1}x{p2}"

        w(f"\n--- {pair_label}: {col1} x {col2} ---")
        w(f"{'Bucket1':<15} {'Bucket2':<15} {'N':>6} {'full_drift':>12} {'rest_drift':>12} {'Fill%':>8} {'CL':>8}")
        w("-" * 80)

        for b1 in data[col1].dropna().unique():
            for b2 in data[col2].dropna().unique():
                grp = data[(data[col1] == b1) & (data[col2] == b2)]
                if len(grp) == 0:
                    continue
                m = compute_group_metrics(grp)
                tag = low_n(m['N'])
                w(f"{str(b1):<15} {str(b2):<15} {m['N']:>6}{tag} {m['full_drift']:>+12.4f} {m['rest_drift']:>+12.4f} {m['fill_rate']:>8.1f} {m['cl']:>8.3f}")

                all_results.append({
                    'gap_dir': gap_dir, 'pair': pair_label,
                    'bucket1': str(b1), 'bucket2': str(b2),
                    **m
                })
                pair_count += 1

w(f"\nTotal cells computed: {pair_count}")

# ══════════════════════════════════════════════════════════════════════════════
# TOP-10 2er-Combinations by drift spread
# ══════════════════════════════════════════════════════════════════════════════
w("\n" + "=" * 80)
w("TOP-10 2er-COMBINATIONS (by full_drift, N>=20)")
w("=" * 80)

res_df = pd.DataFrame(all_results)
res_valid = res_df[res_df['N'] >= 20].copy()

for gap_dir in ['up', 'down']:
    gd = res_valid[res_valid['gap_dir'] == gap_dir].nlargest(10, 'full_drift')
    w(f"\nGap{gap_dir.title()} Top-10 (highest full_drift):")
    w(f"{'Pair':<8} {'Bucket1':<15} {'Bucket2':<15} {'N':>6} {'full_drift':>12} {'rest_drift':>12}")
    w("-" * 75)
    for _, r in gd.iterrows():
        w(f"{r['pair']:<8} {r['bucket1']:<15} {r['bucket2']:<15} {r['N']:>6} {r['full_drift']:>+12.4f} {r['rest_drift']:>+12.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 4b: 3er-Kombinationen (vielversprechendste)
# ══════════════════════════════════════════════════════════════════════════════
w("\n" + "=" * 80)
w("AUFGABE 4b: 3er-KOMBINATIONEN (SPEZIFISCHE)")
w("=" * 80)

triple_combos = [
    # (label, filters_dict per gap_dir)
    ("ComboE+RVOL5>5+PM5<40%", {
        'H_candle': 'with+gross', 'G_od_str': '>0.5', 'F_od_dir': 'with_gap',
        'C_rvol5': '>5x', 'E_pmrth5': '<40%'
    }),
    ("ComboE+RVOL5>5", {
        'H_candle': 'with+gross', 'G_od_str': '>0.5', 'F_od_dir': 'with_gap',
        'C_rvol5': '>5x'
    }),
    ("ComboE+Gap>2", {
        'H_candle': 'with+gross', 'G_od_str': '>0.5', 'F_od_dir': 'with_gap',
        'A_gap': '>2'
    }),
    ("PM5<40%+OD>0.5with+Gap>2", {
        'E_pmrth5': '<40%', 'G_od_str': '>0.5', 'F_od_dir': 'with_gap',
        'A_gap': '>2'
    }),
    ("PM5<40%+Kerze+RVOL5>5", {
        'E_pmrth5': '<40%', 'H_candle': 'with+gross', 'C_rvol5': '>5x'
    }),
    ("PM30<10%+OD>0.5with+RVOL30>5", {
        'D_pmrth30': '<10%', 'G_od_str': '>0.5', 'F_od_dir': 'with_gap',
        'B_rvol30': '>5x'
    }),
    ("RVOL5>5+OD>0.5with+Gap>2", {
        'C_rvol5': '>5x', 'G_od_str': '>0.5', 'F_od_dir': 'with_gap',
        'A_gap': '>2'
    }),
]

for gap_dir in ['up', 'down']:
    data = h1[h1['gap_direction'] == gap_dir].copy()
    w(f"\n--- Gap{gap_dir.title()} ---")
    w(f"{'Combo':<35} {'N':>6} {'full_drift':>12} {'rest_drift':>12} {'Fill%':>8} {'CL':>8}")
    w("-" * 90)

    for label, filters in triple_combos:
        mask = pd.Series(True, index=data.index)
        for col, val in filters.items():
            mask &= (data[col] == val)
        grp = data[mask]
        m = compute_group_metrics(grp)
        tag = low_n(m['N'])
        if m['N'] > 0:
            w(f"{label:<35} {m['N']:>6}{tag} {m['full_drift']:>+12.4f} {m['rest_drift']:>+12.4f} {m['fill_rate']:>8.1f} {m['cl']:>8.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════
with open('results/d7_full_cross.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))
print(f"Saved results/d7_full_cross.txt", file=sys.stderr)

res_df.to_parquet('results/d7_full_cross_raw.parquet', index=False)
print('\n'.join(out))
