"""
D11 Aufgabe 2: Bestehende Parameter — Wer trennt Big Runner? (IS)
"""
import pandas as pd
import numpy as np
from scipy import stats
import os

OUT = "results/d11_existing_params.txt"
lines = []

def w(text=""):
    lines.append(text)

def header(title):
    w("=" * 70)
    w(title)
    w("=" * 70)

# Load IS base (OD>0.5 + with_gap)
df = pd.read_parquet("results/d11_is_base.parquet")
print(f"Loaded: {len(df)} rows")

# Also load 1min data for 30min features
# We need: 30min_range_ADR, 30min_drift_ADR
# Compute from close_935, close_1000, rth_open, adr_10

def compute_30min_features(df):
    """Compute 30min features from metadata columns."""
    df = df.copy()

    # 30min drift in gap direction (Close_1000 - Close_935) / ADR
    drift_raw = (df['close_1000'] - df['close_935']) / df['adr_10']
    is_up = df['gap_direction'] == 'up'
    df['drift_30min_adr'] = np.where(is_up, drift_raw, -drift_raw)

    # We need 30min range from 1min bars — but we don't have it in metadata
    # We'll compute it in Aufgabe 3 from 1min bars
    # For now, use rest_drift_1000 as a proxy: (Close_1000 - Close_935) in gap dir
    # rest_drift_1000 is already in ADR and signed for gap direction
    if 'rest_drift_1000' in df.columns:
        df['rest_drift_1000_abs'] = df['rest_drift_1000'].abs()

    return df

df = compute_30min_features(df)

header("D11 AUFGABE 2: BESTEHENDE PARAMETER — WER TRENNT? (IS)")
w()
w(f"Basis: OD > 0.5 ADR + with_gap, N = {len(df)}")
w()
w()

# === Helper: bucket analysis ===
def bucket_analysis(df, col, buckets, gap_dir, label):
    """Analyze Big Runner rate by buckets."""
    g = df[df['gap_direction'] == gap_dir].dropna(subset=[col])
    base_rate = g['is_big_runner'].mean() * 100

    results = []
    for bname, lo, hi in buckets:
        mask = (g[col] >= lo) & (g[col] < hi)
        sub = g[mask]
        n = len(sub)
        if n < 10:
            continue
        br_n = sub['is_big_runner'].sum()
        br_pct = br_n / n * 100 if n > 0 else 0
        lift = br_pct / base_rate if base_rate > 0 else 0
        low_n = " [LOW N]" if n < 20 else ""
        results.append((bname, n, br_n, br_pct, lift, low_n))

    return results, base_rate

def print_bucket_table(results, base_rate, gap_label):
    w(f"  --- {gap_label} (Base Rate: {base_rate:.1f}%) ---")
    w(f"  {'Bucket':<18s} | {'N':>5s} | {'BR':>4s} | {'BR%':>7s} | {'Lift':>5s}")
    w(f"  {'-'*48}")
    for bname, n, br_n, br_pct, lift, low_n in results:
        w(f"  {bname:<18s} | {n:>5d} | {br_n:>4d} | {br_pct:>6.1f}% | {lift:>4.1f}x{low_n}")
    w()


# =====================================================================
# 2a: OPENING DRIVE PARAMETER (9:35 bekannt)
# =====================================================================
header("2a: OPENING DRIVE PARAMETER (ab 9:35 bekannt)")
w()

# --- OD_strength_ADR ---
w("  OD_strength_ADR:")
od_buckets = [
    ("< 0.3 ADR", 0, 0.3),
    ("0.3 - 0.5", 0.3, 0.5),
    ("0.5 - 0.7", 0.5, 0.7),
    ("0.7 - 1.0", 0.7, 1.0),
    ("1.0 - 1.5", 1.0, 1.5),
    ("> 1.5 ADR", 1.5, 999),
]
for gap_dir, glabel in [('up', 'GapUp'), ('down', 'GapDn')]:
    results, base = bucket_analysis(df, 'od_strength', od_buckets, gap_dir, glabel)
    print_bucket_table(results, base, glabel)

# --- RVOL_5 ---
w("  RVOL_5:")
rvol5_buckets = [
    ("< 1x", 0, 1),
    ("1 - 2x", 1, 2),
    ("2 - 3x", 2, 3),
    ("3 - 5x", 3, 5),
    ("5 - 10x", 5, 10),
    ("> 10x", 10, 9999),
]
for gap_dir, glabel in [('up', 'GapUp'), ('down', 'GapDn')]:
    results, base = bucket_analysis(df, 'rvol_5', rvol5_buckets, gap_dir, glabel)
    print_bucket_table(results, base, glabel)

# --- Gap_in_ADR ---
w("  Gap_in_ADR:")
gap_buckets = [
    ("< 0.5 ADR", 0, 0.5),
    ("0.5 - 1.0", 0.5, 1.0),
    ("1.0 - 2.0", 1.0, 2.0),
    ("2.0 - 4.0", 2.0, 4.0),
    ("> 4.0 ADR", 4.0, 999),
]
for gap_dir, glabel in [('up', 'GapUp'), ('down', 'GapDn')]:
    results, base = bucket_analysis(df, 'gap_size_in_adr', gap_buckets, gap_dir, glabel)
    print_bucket_table(results, base, glabel)

# --- PM/RTH5 ---
w("  PM/RTH5:")
pm5_buckets = [
    ("< 25%", 0, 0.25),
    ("25 - 50%", 0.25, 0.50),
    ("50 - 100%", 0.50, 1.00),
    ("> 100%", 1.00, 999),
]
for gap_dir, glabel in [('up', 'GapUp'), ('down', 'GapDn')]:
    results, base = bucket_analysis(df, 'pm_rth5', pm5_buckets, gap_dir, glabel)
    print_bucket_table(results, base, glabel)

# --- od_body_pct ---
w("  od_body_pct:")
body_buckets = [
    ("< 0.30", 0, 0.30),
    ("0.30 - 0.50", 0.30, 0.50),
    ("0.50 - 0.70", 0.50, 0.70),
    ("> 0.70", 0.70, 1.01),
]
for gap_dir, glabel in [('up', 'GapUp'), ('down', 'GapDn')]:
    results, base = bucket_analysis(df, 'od_body_pct', body_buckets, gap_dir, glabel)
    print_bucket_table(results, base, glabel)

# --- od_wick_ratio ---
w("  od_wick_ratio:")
wick_buckets = [
    ("< 0.10", 0, 0.10),
    ("0.10 - 0.20", 0.10, 0.20),
    ("0.20 - 0.35", 0.20, 0.35),
    ("> 0.35", 0.35, 999),
]
for gap_dir, glabel in [('up', 'GapUp'), ('down', 'GapDn')]:
    results, base = bucket_analysis(df, 'od_wick_ratio', wick_buckets, gap_dir, glabel)
    print_bucket_table(results, base, glabel)

# --- first_candle_dir ---
w("  first_candle_dir:")
for gap_dir, glabel in [('up', 'GapUp'), ('down', 'GapDn')]:
    g = df[df['gap_direction'] == gap_dir]
    base_rate = g['is_big_runner'].mean() * 100
    w(f"  --- {glabel} (Base Rate: {base_rate:.1f}%) ---")
    w(f"  {'Direction':<18s} | {'N':>5s} | {'BR':>4s} | {'BR%':>7s} | {'Lift':>5s}")
    w(f"  {'-'*48}")
    for val in ['with_gap', 'against_gap']:
        sub = g[g['first_candle_dir'] == val]
        n = len(sub)
        if n < 10:
            continue
        br_n = sub['is_big_runner'].sum()
        br_pct = br_n / n * 100 if n > 0 else 0
        lift = br_pct / base_rate if base_rate > 0 else 0
        w(f"  {val:<18s} | {n:>5d} | {br_n:>4d} | {br_pct:>6.1f}% | {lift:>4.1f}x")
    w()

# --- first_candle_size ---
w("  first_candle_size (ADR):")
fc_buckets = [
    ("< 0.05", 0, 0.05),
    ("0.05 - 0.15", 0.05, 0.15),
    ("0.15 - 0.30", 0.15, 0.30),
    ("> 0.30", 0.30, 999),
]
for gap_dir, glabel in [('up', 'GapUp'), ('down', 'GapDn')]:
    results, base = bucket_analysis(df, 'first_candle_size', fc_buckets, gap_dir, glabel)
    print_bucket_table(results, base, glabel)

w()

# =====================================================================
# 2b: 30-MINUTEN PARAMETER (ab 10:00 bekannt)
# =====================================================================
header("2b: 30-MINUTEN PARAMETER (ab 10:00 bekannt)")
w()

# --- RVOL_30 ---
w("  RVOL_30 (rvol_open_30min):")
rvol30_buckets = [
    ("< 2x", 0, 2),
    ("2 - 3x", 2, 3),
    ("3 - 5x", 3, 5),
    ("5 - 10x", 5, 10),
    ("> 10x", 10, 9999),
]
for gap_dir, glabel in [('up', 'GapUp'), ('down', 'GapDn')]:
    results, base = bucket_analysis(df, 'rvol_open_30min', rvol30_buckets, gap_dir, glabel)
    print_bucket_table(results, base, glabel)

# --- PM/RTH30 ---
w("  PM/RTH30:")
pm30_buckets = [
    ("< 5%", 0, 0.05),
    ("5 - 10%", 0.05, 0.10),
    ("10 - 20%", 0.10, 0.20),
    ("20 - 30%", 0.20, 0.30),
    ("> 30%", 0.30, 999),
]
for gap_dir, glabel in [('up', 'GapUp'), ('down', 'GapDn')]:
    results, base = bucket_analysis(df, 'pm_rth30_computed', pm30_buckets, gap_dir, glabel)
    print_bucket_table(results, base, glabel)

# --- 30min drift in gap direction ---
w("  30min Drift (Gap-Dir, close_1000 - close_935):")
drift_buckets = [
    ("< 0 (gegen)", -999, 0),
    ("0 - 0.25", 0, 0.25),
    ("0.25 - 0.50", 0.25, 0.50),
    ("0.50 - 1.00", 0.50, 1.00),
    ("> 1.00 ADR", 1.00, 999),
]
for gap_dir, glabel in [('up', 'GapUp'), ('down', 'GapDn')]:
    results, base = bucket_analysis(df, 'drift_30min_adr', drift_buckets, gap_dir, glabel)
    print_bucket_table(results, base, glabel)

# --- rest_drift_1000 ---
w("  rest_drift_1000 (Drift Open->10:00 in ADR, signed for gap):")
rd_buckets = [
    ("< -0.25", -999, -0.25),
    ("-0.25 - 0", -0.25, 0),
    ("0 - 0.25", 0, 0.25),
    ("0.25 - 0.50", 0.25, 0.50),
    ("> 0.50", 0.50, 999),
]
for gap_dir, glabel in [('up', 'GapUp'), ('down', 'GapDn')]:
    results, base = bucket_analysis(df, 'rest_drift_1000', rd_buckets, gap_dir, glabel)
    print_bucket_table(results, base, glabel)

w()

# =====================================================================
# 2c: RANKING NACH TRENNSCHAERFE (Spearman)
# =====================================================================
header("2c: RANKING NACH TRENNSCHAERFE (Spearman rho)")
w()

params_to_test = [
    ('od_strength', 'OD_strength'),
    ('rvol_5', 'RVOL_5'),
    ('rvol_open_30min', 'RVOL_30'),
    ('gap_size_in_adr', 'Gap_in_ADR'),
    ('pm_rth5', 'PM/RTH5'),
    ('pm_rth30_computed', 'PM/RTH30'),
    ('od_body_pct', 'od_body_pct'),
    ('od_wick_ratio', 'od_wick_ratio'),
    ('first_candle_size', 'first_candle_size'),
    ('prev_close', 'Preis'),
    ('market_cap', 'MarketCap'),
    ('adr_10', 'ADR_10'),
    ('drift_30min_adr', '30min_drift'),
    ('rest_drift_1000', 'rest_drift_1000'),
    ('spy_return_day', 'SPY_return'),
    ('rsi_14_prev', 'RSI_14_prev'),
    ('prior_return_10d', 'Prior_10d'),
    ('dist_from_52w_high', 'Dist_52wHigh'),
]

all_rankings = []

for gap_dir, glabel in [('up', 'GapUp'), ('down', 'GapDn')]:
    g = df[df['gap_direction'] == gap_dir].copy()
    y = g['is_big_runner'].astype(int)

    rankings = []
    for col, label in params_to_test:
        if col not in g.columns:
            continue
        valid = g[col].notna()
        if valid.sum() < 30:
            continue
        rho, pval = stats.spearmanr(g.loc[valid, col], y[valid])
        rankings.append((label, rho, pval, valid.sum()))

    rankings.sort(key=lambda x: abs(x[1]), reverse=True)
    all_rankings.append((glabel, rankings))

    w(f"  --- {glabel} ---")
    w(f"  {'Rank':>4s} | {'Parameter':<18s} | {'rho':>8s} | {'p-value':>10s} | {'N':>5s}")
    w(f"  {'-'*55}")
    for i, (label, rho, pval, n) in enumerate(rankings):
        sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else ""))
        w(f"  {i+1:>4d} | {label:<18s} | {rho:>+8.3f} | {pval:>10.4f} | {n:>5d} {sig}")
    w()

w()
w("  Signifikanz: *** p<0.001, ** p<0.01, * p<0.05")

# Write output
os.makedirs("results", exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print(f"Geschrieben: {OUT} ({len(lines)} Zeilen)")
