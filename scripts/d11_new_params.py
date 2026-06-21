"""
D11 Aufgabe 3: Neue Parameter berechnen + testen (IS)
Reads 1min bars for each gapper, computes new features, tests for Big Runner separation.
"""
import pandas as pd
import numpy as np
from scipy import stats
import os, sys
from tqdm import tqdm

OUT = "results/d11_new_params.txt"
lines = []

def w(text=""):
    lines.append(text)

def header(title):
    w("=" * 70)
    w(title)
    w("=" * 70)

# Load IS base data
base = pd.read_parquet("results/d11_is_base.parquet")
print(f"Base: {len(base)} trades")

# =====================================================================
# Compute new features from 1min bars
# =====================================================================

new_features = []
errors = 0

for idx, row in tqdm(base.iterrows(), total=len(base), desc="Computing 1min features", file=sys.stderr):
    ticker = row['ticker']
    date = row['date']
    gap_dir = row['gap_direction']
    adr = row['adr_10']
    rth_open = row['rth_open']

    feat = {'ticker': ticker, 'date': date}

    if pd.isna(adr) or adr <= 0:
        new_features.append(feat)
        continue

    path = f"data/raw_1min/{ticker}/{date}.parquet"
    if not os.path.exists(path):
        errors += 1
        new_features.append(feat)
        continue

    try:
        bars = pd.read_parquet(path)
    except:
        errors += 1
        new_features.append(feat)
        continue

    # Filter RTH
    rth = bars[bars['session'] == 'rth'].copy()
    if len(rth) < 30:
        new_features.append(feat)
        continue

    rth['time_et'] = rth['time_et'].astype(str)

    # PM data
    pm = bars[bars['session'] == 'premarket'].copy()

    # === 3a: Premarket Features ===
    if len(pm) > 0:
        pm_high = pm['high'].max()
        pm_low = pm['low'].min()
        pm_open_price = pm.iloc[0]['open']
        pm_close_price = pm.iloc[-1]['close']

        feat['pm_range_adr'] = (pm_high - pm_low) / adr
        pm_drift_raw = (pm_close_price - pm_open_price) / adr
        feat['pm_drift_adr'] = pm_drift_raw if gap_dir == 'up' else -pm_drift_raw
        feat['pm_high_vs_open'] = (pm_high - rth_open) / adr if gap_dir == 'up' else (rth_open - pm_low) / adr
        gap_adr = row['gap_size_in_adr'] if pd.notna(row['gap_size_in_adr']) and row['gap_size_in_adr'] > 0 else np.nan
        feat['pm_consolidation'] = feat['pm_range_adr'] / gap_adr if pd.notna(gap_adr) and gap_adr > 0 else np.nan
        feat['pm_vol_total'] = pm['volume'].sum()

    # === 3b: Stock characteristics (already in metadata, skip) ===
    feat['price_bucket'] = (
        '<$10' if row['prev_close'] < 10 else
        '$10-30' if row['prev_close'] < 30 else
        '$30-100' if row['prev_close'] < 100 else
        '>$100'
    ) if pd.notna(row['prev_close']) else np.nan

    feat['adr_pct'] = (adr / row['prev_close'] * 100) if pd.notna(row['prev_close']) and row['prev_close'] > 0 else np.nan

    # === 3c: OD Microstructure (9:30-9:34, 5 bars) ===
    od_bars = rth[rth['time_et'].between('09:30', '09:34')]
    if len(od_bars) >= 5:
        candle1 = od_bars.iloc[0]
        candle5 = od_bars.iloc[4]

        body1 = candle1['close'] - candle1['open']
        body5 = candle5['close'] - candle5['open']

        # Sign for gap direction
        if gap_dir == 'up':
            body1_dir = body1
            body5_dir = body5
        else:
            body1_dir = -body1
            body5_dir = -body5

        feat['od_momentum'] = 1 if (body1_dir > 0 and body5_dir > 0) else 0
        feat['od_acceleration'] = abs(body5) / abs(body1) if abs(body1) > 0 else np.nan
        feat['od_vol_profile'] = candle5['volume'] / candle1['volume'] if candle1['volume'] > 0 else np.nan

        # OD close vs extremum
        od_range = od_bars['high'].max() - od_bars['low'].min()
        close_935 = od_bars.iloc[-1]['close']
        if gap_dir == 'up':
            od_extremum = od_bars['high'].max()
            feat['od_close_vs_high'] = abs(close_935 - od_extremum) / od_range if od_range > 0 else np.nan
        else:
            od_extremum = od_bars['low'].min()
            feat['od_close_vs_high'] = abs(close_935 - od_extremum) / od_range if od_range > 0 else np.nan

        # First candle body pct
        c1_range = candle1['high'] - candle1['low']
        feat['first_candle_body_pct'] = abs(body1) / c1_range if c1_range > 0 else np.nan

    # === 3d: 10:00 Features (9:35-10:00, bars 6-30 approximately) ===
    post_od = rth[rth['time_et'].between('09:35', '09:59')]
    if len(post_od) >= 10:
        close_935_val = row['close_935'] if pd.notna(row['close_935']) else rth[rth['time_et'] == '09:35'].iloc[0]['close'] if len(rth[rth['time_et'] == '09:35']) > 0 else np.nan

        if pd.notna(close_935_val):
            # pullback_depth
            if gap_dir == 'up':
                pullback = max(close_935_val - post_od['low'].min(), 0) / adr
            else:
                pullback = max(post_od['high'].max() - close_935_val, 0) / adr
            feat['pullback_depth'] = pullback

            # continuation_after_od
            close_1000_val = row['close_1000'] if pd.notna(row['close_1000']) else np.nan
            if pd.notna(close_1000_val):
                if gap_dir == 'up':
                    feat['continuation_after_od'] = (close_1000_val - close_935_val) / adr
                else:
                    feat['continuation_after_od'] = (close_935_val - close_1000_val) / adr

        # new_high_by_1000
        od_5min = rth[rth['time_et'].between('09:30', '09:34')]
        if len(od_5min) > 0:
            if gap_dir == 'up':
                od_high = od_5min['high'].max()
                new_high = post_od['high'].max() > od_high
            else:
                od_low = od_5min['low'].min()
                new_high = post_od['low'].min() < od_low
            feat['new_high_by_1000'] = 1 if new_high else 0

        # bars_above_open (9:36-10:00)
        post_od_bars = rth[rth['time_et'].between('09:36', '09:59')]
        if len(post_od_bars) > 0:
            if gap_dir == 'up':
                above = (post_od_bars['close'] > rth_open).sum()
            else:
                above = (post_od_bars['close'] < rth_open).sum()
            feat['bars_above_open'] = above / len(post_od_bars)

        # vol_30_vs_5
        od_5min_bars = rth[rth['time_et'].between('09:30', '09:34')]
        vol_5 = od_5min_bars['volume'].sum() if len(od_5min_bars) > 0 else 0
        vol_6_30 = post_od['volume'].sum()
        feat['vol_30_vs_5'] = vol_6_30 / vol_5 if vol_5 > 0 else np.nan

    # === 3e: Day context (already in base, skip) ===

    # 30min range from bars
    first_30 = rth[rth['time_et'].between('09:30', '09:59')]
    if len(first_30) > 0:
        feat['range_30min_adr'] = (first_30['high'].max() - first_30['low'].min()) / adr

    new_features.append(feat)

print(f"Computed features for {len(new_features)} trades, errors: {errors}")

# Merge with base
feat_df = pd.DataFrame(new_features)
merged = base.merge(feat_df, on=['ticker', 'date'], how='left', suffixes=('', '_new'))
# Drop duplicate columns from feat_df
drop_cols = [c for c in merged.columns if c.endswith('_new')]
merged = merged.drop(columns=drop_cols)
print(f"Merged: {len(merged)} rows, new columns: {[c for c in feat_df.columns if c not in base.columns]}")

# Save enriched dataset
merged.to_parquet("results/d11_is_enriched.parquet", index=False)

# =====================================================================
# Report and test new features
# =====================================================================

header("D11 AUFGABE 3: NEUE PARAMETER BERECHNEN + TESTEN (IS)")
w()
w(f"Basis: OD > 0.5 ADR + with_gap, N = {len(merged)}")
w()
w()

# --- 3a-3e: Feature distributions ---
header("3a-3e: NEUE FEATURE-VERTEILUNGEN")
w()

new_cols = ['pm_range_adr', 'pm_drift_adr', 'pm_high_vs_open', 'pm_consolidation',
            'pm_vol_total', 'adr_pct',
            'od_momentum', 'od_acceleration', 'od_vol_profile', 'od_close_vs_high',
            'first_candle_body_pct',
            'pullback_depth', 'continuation_after_od', 'new_high_by_1000',
            'bars_above_open', 'vol_30_vs_5', 'range_30min_adr']

w(f"  {'Feature':<25s} | {'Median':>8s} | {'Q25':>8s} | {'Q75':>8s} | {'NaN%':>6s}")
w(f"  {'-'*62}")
for col in new_cols:
    if col not in merged.columns:
        continue
    vals = merged[col].dropna()
    nan_pct = merged[col].isna().mean() * 100
    if len(vals) > 0:
        med = vals.median()
        q25 = vals.quantile(0.25)
        q75 = vals.quantile(0.75)
        w(f"  {col:<25s} | {med:>8.3f} | {q25:>8.3f} | {q75:>8.3f} | {nan_pct:>5.1f}%")
    else:
        w(f"  {col:<25s} | {'N/A':>8s} | {'N/A':>8s} | {'N/A':>8s} | {nan_pct:>5.1f}%")

w()
w()

# --- 3f: Test all new params ---
header("3f: NEUE PARAMETER — TRENNSCHAERFE (Buckets + Spearman)")
w()

def bucket_analysis(df, col, buckets, gap_dir):
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

def print_bucket(results, base_rate, gap_label, param_label):
    w(f"  {param_label}:")
    w(f"  --- {gap_label} (Base Rate: {base_rate:.1f}%) ---")
    w(f"  {'Bucket':<18s} | {'N':>5s} | {'BR':>4s} | {'BR%':>7s} | {'Lift':>5s}")
    w(f"  {'-'*48}")
    for bname, n, br_n, br_pct, lift, low_n in results:
        w(f"  {bname:<18s} | {n:>5d} | {br_n:>4d} | {br_pct:>6.1f}% | {lift:>4.1f}x{low_n}")
    w()

# pm_range_adr
pm_range_buckets = [("< 0.3", 0, 0.3), ("0.3 - 0.5", 0.3, 0.5), ("0.5 - 1.0", 0.5, 1.0), ("> 1.0", 1.0, 999)]
for gd, gl in [('up', 'GapUp'), ('down', 'GapDn')]:
    r, b = bucket_analysis(merged, 'pm_range_adr', pm_range_buckets, gd)
    print_bucket(r, b, gl, "pm_range_adr")

# pm_consolidation
pm_cons_buckets = [("< 0.10", 0, 0.10), ("0.10 - 0.25", 0.10, 0.25), ("0.25 - 0.50", 0.25, 0.50), ("> 0.50", 0.50, 999)]
for gd, gl in [('up', 'GapUp'), ('down', 'GapDn')]:
    r, b = bucket_analysis(merged, 'pm_consolidation', pm_cons_buckets, gd)
    print_bucket(r, b, gl, "pm_consolidation")

# pullback_depth
pb_buckets = [("< 0.10", 0, 0.10), ("0.10 - 0.25", 0.10, 0.25), ("0.25 - 0.50", 0.25, 0.50), ("> 0.50", 0.50, 999)]
for gd, gl in [('up', 'GapUp'), ('down', 'GapDn')]:
    r, b = bucket_analysis(merged, 'pullback_depth', pb_buckets, gd)
    print_bucket(r, b, gl, "pullback_depth")

# continuation_after_od
cont_buckets = [("< -0.25", -999, -0.25), ("-0.25 - 0", -0.25, 0), ("0 - 0.25", 0, 0.25), ("0.25 - 0.50", 0.25, 0.50), ("> 0.50", 0.50, 999)]
for gd, gl in [('up', 'GapUp'), ('down', 'GapDn')]:
    r, b = bucket_analysis(merged, 'continuation_after_od', cont_buckets, gd)
    print_bucket(r, b, gl, "continuation_after_od")

# new_high_by_1000
w("  new_high_by_1000:")
for gd, gl in [('up', 'GapUp'), ('down', 'GapDn')]:
    g = merged[(merged['gap_direction'] == gd) & merged['new_high_by_1000'].notna()]
    base_rate = g['is_big_runner'].mean() * 100
    w(f"  --- {gl} (Base Rate: {base_rate:.1f}%) ---")
    w(f"  {'Value':<18s} | {'N':>5s} | {'BR':>4s} | {'BR%':>7s} | {'Lift':>5s}")
    w(f"  {'-'*48}")
    for val in [0, 1]:
        sub = g[g['new_high_by_1000'] == val]
        n = len(sub)
        if n < 10:
            continue
        br_n = sub['is_big_runner'].sum()
        br_pct = br_n / n * 100
        lift = br_pct / base_rate if base_rate > 0 else 0
        w(f"  {'No' if val == 0 else 'Yes':<18s} | {n:>5d} | {br_n:>4d} | {br_pct:>6.1f}% | {lift:>4.1f}x")
    w()

# bars_above_open
bao_buckets = [("< 0.40", 0, 0.40), ("0.40 - 0.60", 0.40, 0.60), ("0.60 - 0.80", 0.60, 0.80), ("> 0.80", 0.80, 1.01)]
for gd, gl in [('up', 'GapUp'), ('down', 'GapDn')]:
    r, b = bucket_analysis(merged, 'bars_above_open', bao_buckets, gd)
    print_bucket(r, b, gl, "bars_above_open")

# range_30min_adr
r30_buckets = [("< 0.5", 0, 0.5), ("0.5 - 1.0", 0.5, 1.0), ("1.0 - 1.5", 1.0, 1.5), ("> 1.5", 1.5, 999)]
for gd, gl in [('up', 'GapUp'), ('down', 'GapDn')]:
    r, b = bucket_analysis(merged, 'range_30min_adr', r30_buckets, gd)
    print_bucket(r, b, gl, "range_30min_adr")

# od_momentum
w("  od_momentum (Kerze1+5 beide in Gap-Richtung):")
for gd, gl in [('up', 'GapUp'), ('down', 'GapDn')]:
    g = merged[(merged['gap_direction'] == gd) & merged['od_momentum'].notna()]
    base_rate = g['is_big_runner'].mean() * 100
    w(f"  --- {gl} (Base Rate: {base_rate:.1f}%) ---")
    w(f"  {'Value':<18s} | {'N':>5s} | {'BR':>4s} | {'BR%':>7s} | {'Lift':>5s}")
    w(f"  {'-'*48}")
    for val in [0, 1]:
        sub = g[g['od_momentum'] == val]
        n = len(sub)
        if n < 10:
            continue
        br_n = sub['is_big_runner'].sum()
        br_pct = br_n / n * 100
        lift = br_pct / base_rate if base_rate > 0 else 0
        w(f"  {'No' if val == 0 else 'Yes':<18s} | {n:>5d} | {br_n:>4d} | {br_pct:>6.1f}% | {lift:>4.1f}x")
    w()

# od_acceleration
oda_buckets = [("< 0.5", 0, 0.5), ("0.5 - 1.0", 0.5, 1.0), ("1.0 - 2.0", 1.0, 2.0), ("> 2.0", 2.0, 999)]
for gd, gl in [('up', 'GapUp'), ('down', 'GapDn')]:
    r, b = bucket_analysis(merged, 'od_acceleration', oda_buckets, gd)
    print_bucket(r, b, gl, "od_acceleration")

# od_close_vs_high (distance from OD extremum)
odch_buckets = [("< 0.10 (near)", 0, 0.10), ("0.10 - 0.25", 0.10, 0.25), ("0.25 - 0.50", 0.25, 0.50), ("> 0.50 (far)", 0.50, 999)]
for gd, gl in [('up', 'GapUp'), ('down', 'GapDn')]:
    r, b = bucket_analysis(merged, 'od_close_vs_high', odch_buckets, gd)
    print_bucket(r, b, gl, "od_close_vs_high")

# adr_pct
adrp_buckets = [("< 3%", 0, 3), ("3 - 5%", 3, 5), ("5 - 10%", 5, 10), ("> 10%", 10, 999)]
for gd, gl in [('up', 'GapUp'), ('down', 'GapDn')]:
    r, b = bucket_analysis(merged, 'adr_pct', adrp_buckets, gd)
    print_bucket(r, b, gl, "adr_pct")

# is_earnings
w("  is_earnings:")
for gd, gl in [('up', 'GapUp'), ('down', 'GapDn')]:
    g = merged[(merged['gap_direction'] == gd) & merged['is_earnings'].notna()]
    base_rate = g['is_big_runner'].mean() * 100
    w(f"  --- {gl} (Base Rate: {base_rate:.1f}%) ---")
    w(f"  {'Value':<18s} | {'N':>5s} | {'BR':>4s} | {'BR%':>7s} | {'Lift':>5s}")
    w(f"  {'-'*48}")
    for val in [False, True]:
        sub = g[g['is_earnings'] == val]
        n = len(sub)
        if n < 10:
            continue
        br_n = sub['is_big_runner'].sum()
        br_pct = br_n / n * 100
        lift = br_pct / base_rate if base_rate > 0 else 0
        w(f"  {'No' if not val else 'Yes':<18s} | {n:>5d} | {br_n:>4d} | {br_pct:>6.1f}% | {lift:>4.1f}x")
    w()

# vol_30_vs_5
v35_buckets = [("< 3", 0, 3), ("3 - 5", 3, 5), ("5 - 8", 5, 8), ("> 8", 8, 999)]
for gd, gl in [('up', 'GapUp'), ('down', 'GapDn')]:
    r, b = bucket_analysis(merged, 'vol_30_vs_5', v35_buckets, gd)
    print_bucket(r, b, gl, "vol_30_vs_5")

w()

# =====================================================================
# Combined Spearman ranking (old + new)
# =====================================================================
header("3f: INTEGRIERTES RANKING (alt + neu)")
w()

all_params = [
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
    ('spy_return_day', 'SPY_return'),
    ('rsi_14_prev', 'RSI_14_prev'),
    ('prior_return_10d', 'Prior_10d'),
    ('dist_from_52w_high', 'Dist_52wHigh'),
    # New features
    ('pm_range_adr', 'pm_range_adr'),
    ('pm_drift_adr', 'pm_drift_adr'),
    ('pm_high_vs_open', 'pm_high_vs_open'),
    ('pm_consolidation', 'pm_consolidation'),
    ('adr_pct', 'adr_pct'),
    ('od_momentum', 'od_momentum'),
    ('od_acceleration', 'od_acceleration'),
    ('od_vol_profile', 'od_vol_profile'),
    ('od_close_vs_high', 'od_close_vs_high'),
    ('first_candle_body_pct', 'first_candle_body_pct'),
    ('pullback_depth', 'pullback_depth'),
    ('continuation_after_od', 'continuation_after_od'),
    ('new_high_by_1000', 'new_high_by_1000'),
    ('bars_above_open', 'bars_above_open'),
    ('vol_30_vs_5', 'vol_30_vs_5'),
    ('range_30min_adr', 'range_30min_adr'),
]

# Compute drift_30min_adr for merged df
drift_raw = (merged['close_1000'] - merged['close_935']) / merged['adr_10']
is_up = merged['gap_direction'] == 'up'
merged['drift_30min_adr'] = np.where(is_up, drift_raw, -drift_raw)
all_params.append(('drift_30min_adr', '30min_drift'))

for gap_dir, glabel in [('up', 'GapUp'), ('down', 'GapDn')]:
    g = merged[merged['gap_direction'] == gap_dir].copy()
    y = g['is_big_runner'].astype(int)

    rankings = []
    for col, label in all_params:
        if col not in g.columns:
            continue
        valid = g[col].notna()
        if valid.sum() < 30:
            continue
        rho, pval = stats.spearmanr(g.loc[valid, col], y[valid])
        rankings.append((label, rho, pval, valid.sum(), col))

    rankings.sort(key=lambda x: abs(x[1]), reverse=True)

    w(f"  --- {glabel} ---")
    w(f"  {'Rank':>4s} | {'Parameter':<25s} | {'rho':>8s} | {'p-value':>10s} | {'N':>5s} | {'Avail':>6s}")
    w(f"  {'-'*68}")
    for i, (label, rho, pval, n, col) in enumerate(rankings[:25]):
        sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else ""))
        avail = "9:35" if col in ['od_strength', 'rvol_5', 'gap_size_in_adr', 'pm_rth5',
                                   'od_body_pct', 'od_wick_ratio', 'first_candle_size',
                                   'prev_close', 'market_cap', 'adr_10', 'spy_return_day',
                                   'rsi_14_prev', 'prior_return_10d', 'dist_from_52w_high',
                                   'pm_range_adr', 'pm_drift_adr', 'pm_high_vs_open',
                                   'pm_consolidation', 'adr_pct', 'pm_vol_total',
                                   'od_momentum', 'od_acceleration', 'od_vol_profile',
                                   'od_close_vs_high', 'first_candle_body_pct'] else "10:00"
        w(f"  {i+1:>4d} | {label:<25s} | {rho:>+8.3f} | {pval:>10.4f} | {n:>5d} | {avail:>6s} {sig}")
    w()

w()
w("  Signifikanz: *** p<0.001, ** p<0.01, * p<0.05")
w("  Avail: Zeitpunkt ab dem der Parameter bekannt ist")

# Write output
os.makedirs("results", exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print(f"Geschrieben: {OUT} ({len(lines)} Zeilen)")
