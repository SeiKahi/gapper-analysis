"""
Durchlauf 7.0 — Aufgabe 1: RVOL Granular-Analyse (1x-Stufen)
RVOL_30 und RVOL_5 granular, lineare/log Regression, Korrelation, Head-to-Head.
NUR auf IS (H1: 2021-02-21 bis 2023-12-31).
"""
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from tqdm import tqdm
import sys
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
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
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("Loading metadata_v7...", file=sys.stderr)
meta = pd.read_parquet('data/metadata/metadata_v7.parquet')
meta['date'] = pd.to_datetime(meta['date'])
h1 = meta[(meta['date'] >= '2021-02-21') & (meta['date'] <= '2023-12-31')].copy()
h1 = h1[h1['adr_10'].notna() & (h1['adr_10'] > 0)].copy()
print(f"H1: {len(h1)} gappers", file=sys.stderr)

# ══════════════════════════════════════════════════════════════════════════════
# COMMON ANALYSIS FUNCTION
# ══════════════════════════════════════════════════════════════════════════════
def analyze_rvol_buckets(df, rvol_col, label, gap_dir_filter=None):
    """Granular RVOL analysis with 1x-step buckets."""
    w(f"\n{'='*80}")
    w(f"RVOL GRANULAR: {label} ({rvol_col})")
    if gap_dir_filter:
        w(f"Gap Direction: {gap_dir_filter}")
    w(f"{'='*80}")

    if gap_dir_filter:
        data = df[df['gap_direction'] == gap_dir_filter].copy()
    else:
        data = df.copy()

    data = data[data[rvol_col].notna()].copy()
    w(f"N total: {len(data)}")

    # Define buckets
    bucket_edges = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 999]
    bucket_labels = ['<1x', '1-2x', '2-3x', '3-4x', '4-5x', '5-6x', '6-7x', '7-8x', '8-9x', '9-10x', '>10x']
    data['rvol_bucket'] = pd.cut(data[rvol_col], bins=bucket_edges, labels=bucket_labels, right=False)

    w(f"\n{'Bucket':<10} {'N':>6} {'full_drift':>12} {'rest_drift':>12} {'Fill%':>8} {'CL':>8} {'CI_lo':>8} {'CI_hi':>8}")
    w("-" * 80)

    bucket_centers = []
    bucket_drifts = []
    bucket_rest_drifts = []

    for bkt in bucket_labels:
        grp = data[data['rvol_bucket'] == bkt]
        n = len(grp)
        if n == 0:
            continue

        fd = grp['full_drift'].mean() if 'full_drift' in grp.columns else np.nan
        rd = grp['rest_drift'].mean() if 'rest_drift' in grp.columns else np.nan
        fill_rate = grp['gap_filled'].mean() * 100 if 'gap_filled' in grp.columns else np.nan
        cl_val = grp['cl'].mean() if 'cl' in grp.columns else np.nan

        ci_lo, ci_hi = bootstrap_ci(grp['full_drift'].values) if n >= 5 else (np.nan, np.nan)

        w(f"{bkt:<10} {n:>6}{low_n(n)} {fd:>+12.4f} {rd:>+12.4f} {fill_rate:>8.1f} {cl_val:>8.3f} {ci_lo:>+8.3f} {ci_hi:>+8.3f}")

        # For regression: use bucket midpoint
        mid = bucket_edges[bucket_labels.index(bkt)] + 0.5
        if bkt == '>10x':
            mid = data[data['rvol_bucket'] == bkt][rvol_col].median()
        if n >= 20:
            bucket_centers.append(mid)
            bucket_drifts.append(fd)
            bucket_rest_drifts.append(rd)

    # ── Correlations ──
    w(f"\n--- Correlations ({label}) ---")
    valid = data[[rvol_col, 'full_drift', 'rest_drift']].dropna()
    if len(valid) > 50:
        r_pear_fd, p_pear_fd = stats.pearsonr(valid[rvol_col], valid['full_drift'])
        r_spear_fd, p_spear_fd = stats.spearmanr(valid[rvol_col], valid['full_drift'])
        r_pear_rd, p_pear_rd = stats.pearsonr(valid[rvol_col], valid['rest_drift'])
        r_spear_rd, p_spear_rd = stats.spearmanr(valid[rvol_col], valid['rest_drift'])

        w(f"  Pearson  {rvol_col} vs full_drift:  r={r_pear_fd:+.4f}, p={p_pear_fd:.4f}")
        w(f"  Spearman {rvol_col} vs full_drift:  r={r_spear_fd:+.4f}, p={p_spear_fd:.4f}")
        w(f"  Pearson  {rvol_col} vs rest_drift:  r={r_pear_rd:+.4f}, p={p_pear_rd:.4f}")
        w(f"  Spearman {rvol_col} vs rest_drift:  r={r_spear_rd:+.4f}, p={p_spear_rd:.4f}")

    # ── Linear Regression ──
    w(f"\n--- Linear Regression: drift = a + b * {rvol_col} ---")
    if len(valid) > 50:
        # Clip extreme outliers for regression (RVOL > 50x)
        reg_data = valid[valid[rvol_col] <= 50].copy()
        slope_fd, intercept_fd, r_fd, p_fd, se_fd = stats.linregress(reg_data[rvol_col], reg_data['full_drift'])
        slope_rd, intercept_rd, r_rd, p_rd, se_rd = stats.linregress(reg_data[rvol_col], reg_data['rest_drift'])
        w(f"  full_drift = {intercept_fd:+.4f} + {slope_fd:+.4f} * {rvol_col}  (R²={r_fd**2:.4f}, p={p_fd:.4f})")
        w(f"  rest_drift = {intercept_rd:+.4f} + {slope_rd:+.4f} * {rvol_col}  (R²={r_rd**2:.4f}, p={p_rd:.4f})")
        w(f"  => Jede 1x RVOL mehr = {slope_fd:+.4f} ADR full_drift, {slope_rd:+.4f} ADR rest_drift")

    # ── Log Regression ──
    w(f"\n--- Log Regression: drift = a + b * log({rvol_col}) ---")
    if len(valid) > 50:
        reg_data = valid[valid[rvol_col] > 0].copy()
        reg_data['log_rvol'] = np.log(reg_data[rvol_col])
        slope_log_fd, intercept_log_fd, r_log_fd, p_log_fd, _ = stats.linregress(reg_data['log_rvol'], reg_data['full_drift'])
        slope_log_rd, intercept_log_rd, r_log_rd, p_log_rd, _ = stats.linregress(reg_data['log_rvol'], reg_data['rest_drift'])
        w(f"  full_drift = {intercept_log_fd:+.4f} + {slope_log_fd:+.4f} * log({rvol_col})  (R²={r_log_fd**2:.4f})")
        w(f"  rest_drift = {intercept_log_rd:+.4f} + {slope_log_rd:+.4f} * log({rvol_col})  (R²={r_log_rd**2:.4f})")

        # AIC/BIC comparison (simplified)
        n_reg = len(reg_data)
        ss_lin_fd = np.sum((reg_data['full_drift'] - (intercept_fd + slope_fd * reg_data[rvol_col].clip(upper=50)))**2)
        ss_log_fd = np.sum((reg_data['full_drift'] - (intercept_log_fd + slope_log_fd * reg_data['log_rvol']))**2)
        aic_lin = n_reg * np.log(ss_lin_fd / n_reg) + 2 * 2
        aic_log = n_reg * np.log(ss_log_fd / n_reg) + 2 * 2
        w(f"\n  AIC Linear: {aic_lin:.1f} vs AIC Log: {aic_log:.1f}")
        w(f"  => {'LOG' if aic_log < aic_lin else 'LINEAR'} fits better (lower AIC)")

    return bucket_centers, bucket_drifts

# ══════════════════════════════════════════════════════════════════════════════
# 1a: RVOL_30 Granular
# ══════════════════════════════════════════════════════════════════════════════
w("=" * 80)
w("AUFGABE 1a: RVOL_30 GRANULAR (rvol_at_time_30min)")
w("=" * 80)

analyze_rvol_buckets(h1, 'rvol_at_time_30min', 'RVOL_30 GapUp', 'up')
analyze_rvol_buckets(h1, 'rvol_at_time_30min', 'RVOL_30 GapDown', 'down')
analyze_rvol_buckets(h1, 'rvol_at_time_30min', 'RVOL_30 ALL')

# ══════════════════════════════════════════════════════════════════════════════
# 1b: RVOL_5 Granular
# ══════════════════════════════════════════════════════════════════════════════
w("\n" + "=" * 80)
w("AUFGABE 1b: RVOL_5 GRANULAR (rvol_5)")
w("=" * 80)

analyze_rvol_buckets(h1, 'rvol_5', 'RVOL_5 GapUp', 'up')
analyze_rvol_buckets(h1, 'rvol_5', 'RVOL_5 GapDown', 'down')
analyze_rvol_buckets(h1, 'rvol_5', 'RVOL_5 ALL')

# ══════════════════════════════════════════════════════════════════════════════
# 1b extra: RVOL_5 vs RVOL_30 Relationship
# ══════════════════════════════════════════════════════════════════════════════
w("\n" + "=" * 80)
w("RVOL_5 vs RVOL_30 RELATIONSHIP")
w("=" * 80)

both = h1[h1['rvol_5'].notna() & h1['rvol_at_time_30min'].notna()].copy()
w(f"N with both: {len(both)}")

r_pear, p_pear = stats.pearsonr(both['rvol_5'], both['rvol_at_time_30min'])
r_spear, p_spear = stats.spearmanr(both['rvol_5'], both['rvol_at_time_30min'])
w(f"Pearson:  r={r_pear:.4f}, p={p_pear:.6f}")
w(f"Spearman: r={r_spear:.4f}, p={p_spear:.6f}")

# Discrepancy analysis
both['rvol_5_high_30_low'] = (both['rvol_5'] > 5) & (both['rvol_at_time_30min'] < 2)
both['rvol_5_low_30_high'] = (both['rvol_5'] < 2) & (both['rvol_at_time_30min'] > 5)

n_spike = both['rvol_5_high_30_low'].sum()
n_late = both['rvol_5_low_30_high'].sum()
w(f"\nDiscrepancy cases:")
w(f"  RVOL_5 >5x but RVOL_30 <2x (early spike): {n_spike} ({n_spike/len(both)*100:.1f}%)")
w(f"  RVOL_5 <2x but RVOL_30 >5x (late surge):  {n_late} ({n_late/len(both)*100:.1f}%)")

if n_spike >= 10:
    spike = both[both['rvol_5_high_30_low']]
    w(f"  Early spike drift: full={spike['full_drift'].mean():+.4f}, rest={spike['rest_drift'].mean():+.4f}")
if n_late >= 10:
    late = both[both['rvol_5_low_30_high']]
    w(f"  Late surge drift:  full={late['full_drift'].mean():+.4f}, rest={late['rest_drift'].mean():+.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 1c: RVOL_5 vs RVOL_30 Head-to-Head
# ══════════════════════════════════════════════════════════════════════════════
w("\n" + "=" * 80)
w("AUFGABE 1c: RVOL_5 vs RVOL_30 HEAD-TO-HEAD")
w("=" * 80)

metrics = ['full_drift', 'rest_drift']
for m in metrics:
    valid = h1[[m, 'rvol_5', 'rvol_at_time_30min']].dropna()
    if len(valid) < 50:
        continue

    # Pearson
    r5, p5 = stats.pearsonr(valid['rvol_5'], valid[m])
    r30, p30 = stats.pearsonr(valid['rvol_at_time_30min'], valid[m])

    # Spearman
    rs5, ps5 = stats.spearmanr(valid['rvol_5'], valid[m])
    rs30, ps30 = stats.spearmanr(valid['rvol_at_time_30min'], valid[m])

    # R-squared (linear, clipped at 50x)
    v_clip = valid[(valid['rvol_5'] <= 50) & (valid['rvol_at_time_30min'] <= 50)]
    _, _, r_lin5, _, _ = stats.linregress(v_clip['rvol_5'], v_clip[m])
    _, _, r_lin30, _, _ = stats.linregress(v_clip['rvol_at_time_30min'], v_clip[m])

    w(f"\n--- {m} ---")
    w(f"  {'Metric':<20} {'RVOL_5':>12} {'RVOL_30':>12} {'Winner':>10}")
    w(f"  {'Pearson r':<20} {r5:>+12.4f} {r30:>+12.4f} {'RVOL_5' if abs(r5) > abs(r30) else 'RVOL_30':>10}")
    w(f"  {'Spearman r':<20} {rs5:>+12.4f} {rs30:>+12.4f} {'RVOL_5' if abs(rs5) > abs(rs30) else 'RVOL_30':>10}")
    w(f"  {'R² (linear)':<20} {r_lin5**2:>12.4f} {r_lin30**2:>12.4f} {'RVOL_5' if r_lin5**2 > r_lin30**2 else 'RVOL_30':>10}")

w(f"\n--- FAZIT ---")
w("Wenn NUR ein RVOL gewählt werden kann:")
w("  RVOL_5  Vorteil: Ab 9:35 verfuegbar (25min frueher)")
w("  RVOL_30 Vorteil: Stabilere Schaetzung (mehr Datenpunkte)")
w("  => Vergleich der R²-Werte und Korrelationen entscheidet")

# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════
with open('results/d7_rvol_granular.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))
print(f"Saved results/d7_rvol_granular.txt", file=sys.stderr)

# Save raw data
h1_export = h1[['ticker', 'date', 'gap_direction', 'adr_10', 'rvol_5', 'rvol_at_time_30min',
                 'full_drift', 'rest_drift', 'gap_filled', 'cl']].copy()
h1_export.to_parquet('results/d7_rvol_granular_raw.parquet', index=False)

print('\n'.join(out))
