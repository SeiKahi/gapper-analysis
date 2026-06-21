"""
Durchlauf 7.0 — Aufgabe 2: PM/RTH5 vs PM/RTH30 Vergleich
NUR auf IS (H1: 2021-02-21 bis 2023-12-31).
"""
import pandas as pd
import numpy as np
from scipy import stats
import sys
import warnings
warnings.filterwarnings('ignore')

def bootstrap_ci(data, n_boot=1000, ci=0.95):
    data = np.array(data, dtype=float)
    data = data[~np.isnan(data)]
    if len(data) < 5:
        return np.nan, np.nan
    rng = np.random.default_rng(42)
    means = [np.mean(rng.choice(data, len(data), replace=True)) for _ in range(n_boot)]
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
# 2a: PM/RTH5 Buckets
# ══════════════════════════════════════════════════════════════════════════════
w("=" * 80)
w("AUFGABE 2a: PM/RTH5 BUCKETS")
w("=" * 80)

pm5_edges = [0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.00, 999]
pm5_labels = ['<5%', '5-10%', '10-20%', '20-30%', '30-50%', '50-100%', '>100%']

for gap_dir in ['up', 'down']:
    data = h1[(h1['gap_direction'] == gap_dir) & h1['pm_rth5'].notna()].copy()
    data['pm5_bucket'] = pd.cut(data['pm_rth5'], bins=pm5_edges, labels=pm5_labels, right=False)

    w(f"\nGap{gap_dir.title()} — PM/RTH5 Buckets:")
    w(f"{'Bucket':<12} {'N':>6} {'full_drift':>12} {'rest_drift':>12} {'Fill%':>8} {'CL':>8}")
    w("-" * 66)

    for bkt in pm5_labels:
        grp = data[data['pm5_bucket'] == bkt]
        n = len(grp)
        if n == 0:
            continue
        fd = grp['full_drift'].mean()
        rd = grp['rest_drift'].mean()
        fill = grp['gap_filled'].mean() * 100 if 'gap_filled' in grp.columns else np.nan
        cl = grp['cl'].mean()
        w(f"{bkt:<12} {n:>6}{low_n(n)} {fd:>+12.4f} {rd:>+12.4f} {fill:>8.1f} {cl:>8.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# 2b: Optimal PM/RTH5 3er-Buckets (matching PM/RTH30)
# ══════════════════════════════════════════════════════════════════════════════
w("\n" + "=" * 80)
w("AUFGABE 2b: OPTIMALE PM/RTH5 SCHWELLENWERTE")
w("=" * 80)

for gap_dir in ['up', 'down']:
    data = h1[(h1['gap_direction'] == gap_dir) & h1['pm_rth5'].notna() & h1['pm_rth30_computed'].notna()].copy()

    w(f"\nGap{gap_dir.title()}:")

    # Reference: PM/RTH30 <10% / 10-30% / >30%
    ref_low = data[data['pm_rth30_computed'] < 0.10]['full_drift'].mean()
    ref_high = data[data['pm_rth30_computed'] >= 0.30]['full_drift'].mean()
    ref_spread = ref_low - ref_high if gap_dir == 'up' else ref_high - ref_low
    w(f"  PM/RTH30 Reference: <10% drift={ref_low:+.4f}, >30% drift={ref_high:+.4f}, spread={ref_spread:+.4f}")

    # Test different thresholds for PM/RTH5
    best_spread = 0
    best_lo = 0
    best_hi = 0

    w(f"\n  {'Lo_thresh':<10} {'Hi_thresh':<10} {'N_low':>6} {'Drift_low':>12} {'N_high':>6} {'Drift_high':>12} {'Spread':>10}")
    w("  " + "-" * 72)

    for lo_thresh in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
        for hi_thresh in [1.0, 1.5, 2.0, 2.5, 3.0]:
            if hi_thresh <= lo_thresh:
                continue
            low_grp = data[data['pm_rth5'] < lo_thresh]
            high_grp = data[data['pm_rth5'] >= hi_thresh]
            if len(low_grp) < 20 or len(high_grp) < 20:
                continue
            d_low = low_grp['full_drift'].mean()
            d_high = high_grp['full_drift'].mean()
            spread = d_low - d_high if gap_dir == 'up' else d_high - d_low
            w(f"  {lo_thresh:<10.2f} {hi_thresh:<10.2f} {len(low_grp):>6} {d_low:>+12.4f} {len(high_grp):>6} {d_high:>+12.4f} {spread:>+10.4f}")
            if spread > best_spread:
                best_spread = spread
                best_lo = lo_thresh
                best_hi = hi_thresh

    w(f"\n  BEST PM/RTH5: Lo<{best_lo:.2f}, Hi>={best_hi:.2f} (spread={best_spread:+.4f})")
    w(f"  PM/RTH30 Reference spread: {ref_spread:+.4f}")
    w(f"  PM/RTH5 {'BETTER' if best_spread > ref_spread else 'WORSE'} than PM/RTH30 ({best_spread:+.4f} vs {ref_spread:+.4f})")

# ══════════════════════════════════════════════════════════════════════════════
# 2c: PM/RTH5 Zeitvorteil
# ══════════════════════════════════════════════════════════════════════════════
w("\n" + "=" * 80)
w("AUFGABE 2c: PM/RTH5 ZEITVORTEIL (ab 9:35 vs PM/RTH30 ab 10:00)")
w("=" * 80)

data = h1[h1['pm_rth5'].notna() & h1['pm_rth30_computed'].notna()].copy()

# Classification overlap: PM/RTH5 < optimal threshold vs PM/RTH30 < 10%
for lo5 in [0.30, 0.40, 0.50]:
    pm30_low = data['pm_rth30_computed'] < 0.10
    pm5_low = data['pm_rth5'] < lo5
    overlap = (pm30_low & pm5_low).sum()
    pm30_n = pm30_low.sum()
    pm5_n = pm5_low.sum()
    precision = overlap / pm5_n if pm5_n > 0 else 0
    recall = overlap / pm30_n if pm30_n > 0 else 0

    w(f"\n  PM/RTH5 < {lo5:.2f} vs PM/RTH30 < 0.10:")
    w(f"    PM/RTH30<10%: N={pm30_n}")
    w(f"    PM/RTH5<{lo5:.0%}: N={pm5_n}")
    w(f"    Both: N={overlap}")
    w(f"    Precision (PM5 correctly predicts PM30<10%): {precision:.1%}")
    w(f"    Recall (PM30<10% caught by PM5): {recall:.1%}")

# Unique captures
w("\n  UNIQUE CAPTURES:")
for lo5 in [0.30, 0.40, 0.50]:
    pm5_low = data['pm_rth5'] < lo5
    pm30_low = data['pm_rth30_computed'] < 0.10
    only_pm5 = pm5_low & ~pm30_low
    only_pm30 = pm30_low & ~pm5_low
    w(f"    PM/RTH5<{lo5:.0%} captures {only_pm5.sum()} that PM/RTH30 misses")
    w(f"    PM/RTH30<10% captures {only_pm30.sum()} that PM/RTH5<{lo5:.0%} misses")
    if only_pm5.sum() > 10:
        w(f"      PM5-only drift: {data.loc[only_pm5, 'full_drift'].mean():+.4f}")
    if only_pm30.sum() > 10:
        w(f"      PM30-only drift: {data.loc[only_pm30, 'full_drift'].mean():+.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 2d: Korrelation PM/RTH5 vs PM/RTH30
# ══════════════════════════════════════════════════════════════════════════════
w("\n" + "=" * 80)
w("AUFGABE 2d: KORRELATION PM/RTH5 vs PM/RTH30")
w("=" * 80)

valid = h1[h1['pm_rth5'].notna() & h1['pm_rth30_computed'].notna()].copy()
w(f"N: {len(valid)}")

r_pear, p_pear = stats.pearsonr(valid['pm_rth5'], valid['pm_rth30_computed'])
r_spear, p_spear = stats.spearmanr(valid['pm_rth5'], valid['pm_rth30_computed'])
w(f"Pearson:  r={r_pear:.4f}, p={p_pear:.6f}")
w(f"Spearman: r={r_spear:.4f}, p={p_spear:.6f}")

# Stability comparison (variance within buckets)
w("\n--- Stability (within-bucket variance) ---")
for name, col in [('PM/RTH5', 'pm_rth5'), ('PM/RTH30', 'pm_rth30_computed')]:
    edges = [0, 0.10, 0.30, 999]
    labels_3 = ['low', 'mid', 'high']
    valid[f'{col}_bkt'] = pd.cut(valid[col], bins=edges, labels=labels_3, right=False)
    for bkt in labels_3:
        grp = valid[valid[f'{col}_bkt'] == bkt]
        if len(grp) > 10:
            w(f"  {name} {bkt}: N={len(grp)}, drift std={grp['full_drift'].std():.4f}")

# Head-to-head predictive power
w("\n--- Head-to-Head: Predictive Power ---")
for m in ['full_drift', 'rest_drift']:
    v = h1[[m, 'pm_rth5', 'pm_rth30_computed']].dropna()
    if len(v) < 50:
        continue
    r5, _ = stats.pearsonr(v['pm_rth5'], v[m])
    r30, _ = stats.pearsonr(v['pm_rth30_computed'], v[m])
    rs5, _ = stats.spearmanr(v['pm_rth5'], v[m])
    rs30, _ = stats.spearmanr(v['pm_rth30_computed'], v[m])
    w(f"  {m}:")
    w(f"    Pearson:  PM/RTH5 r={r5:+.4f} vs PM/RTH30 r={r30:+.4f} => {'PM5' if abs(r5) > abs(r30) else 'PM30'} wins")
    w(f"    Spearman: PM/RTH5 r={rs5:+.4f} vs PM/RTH30 r={rs30:+.4f} => {'PM5' if abs(rs5) > abs(rs30) else 'PM30'} wins")

# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════
with open('results/d7_pm_rth5.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))
print(f"Saved results/d7_pm_rth5.txt", file=sys.stderr)
print('\n'.join(out))
