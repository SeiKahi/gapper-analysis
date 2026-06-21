"""
D11 Aufgabe 6: OOS-Validierung (H2)
Validate all IS findings on OOS data (2024-01-01 to 2026-02-06).
"""
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import os, sys
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

OUT = "results/d11_oos.txt"
lines = []

def w(text=""):
    lines.append(text)

def header(title):
    w("=" * 70)
    w(title)
    w("=" * 70)

# =====================================================================
# Load and prepare OOS data
# =====================================================================
meta = pd.read_parquet("data/metadata/metadata_v8_5.parquet")
oos_all = meta[(meta['date'] >= '2024-01-01') & (meta['date'] <= '2026-02-06')].copy()
print(f"OOS all gapper: {len(oos_all)}")

# Compute big runner
def compute_big_runner(df):
    df = df.copy()
    adr = df['adr_10']
    range_up = (df['rth_high'] - df['rth_open']) / adr
    denom_up = df['rth_high'] - df['rth_open']
    close_ret_up = np.where(denom_up > 0,
                            (df['rth_close'] - df['rth_open']) / denom_up, np.nan)
    range_dn = (df['rth_open'] - df['rth_low']) / adr
    denom_dn = df['rth_open'] - df['rth_low']
    close_ret_dn = np.where(denom_dn > 0,
                            (df['rth_open'] - df['rth_close']) / denom_dn, np.nan)
    is_up = df['gap_direction'] == 'up'
    is_dn = df['gap_direction'] == 'down'
    range_in_gap_dir = np.where(is_up, range_up, np.where(is_dn, range_dn, np.nan))
    close_retention = np.where(is_up, close_ret_up, np.where(is_dn, close_ret_dn, np.nan))
    df['range_in_gap_dir_adr'] = range_in_gap_dir
    df['close_retention'] = close_retention
    range_ok = range_in_gap_dir >= 1.5
    close_ok = close_retention >= 0.80
    df['is_big_runner'] = range_ok & close_ok
    df['range_ok_close_bad'] = range_ok & ~close_ok
    df['range_bad'] = ~range_ok
    nan_mask = df['adr_10'].isna()
    df.loc[nan_mask, 'is_big_runner'] = False
    df.loc[nan_mask, 'range_ok_close_bad'] = False
    df.loc[nan_mask, 'range_bad'] = True
    return df

oos_all = compute_big_runner(oos_all)

# Filter to OD > 0.5 + with_gap
od_mask = (oos_all['od_strength'] > 0.5) & (oos_all['od_direction'] == 'with_gap')
oos = oos_all[od_mask].copy()
print(f"OOS OD>0.5+with_gap: {len(oos)}")

# Compute new features from 1min data
new_features = []
errors = 0

for idx, row in tqdm(oos.iterrows(), total=len(oos), desc="OOS 1min features", file=sys.stderr):
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

    rth = bars[bars['session'] == 'rth'].copy()
    if len(rth) < 30:
        new_features.append(feat)
        continue
    rth['time_et'] = rth['time_et'].astype(str)
    pm = bars[bars['session'] == 'premarket'].copy()

    # PM features
    if len(pm) > 0:
        pm_open_price = pm.iloc[0]['open']
        pm_close_price = pm.iloc[-1]['close']
        pm_drift_raw = (pm_close_price - pm_open_price) / adr
        feat['pm_drift_adr'] = pm_drift_raw if gap_dir == 'up' else -pm_drift_raw
        feat['pm_range_adr'] = (pm['high'].max() - pm['low'].min()) / adr

    # OD micro
    od_bars = rth[rth['time_et'].between('09:30', '09:34')]
    if len(od_bars) >= 5:
        c1 = od_bars.iloc[0]
        c5 = od_bars.iloc[4]
        body1 = c1['close'] - c1['open']
        body5 = c5['close'] - c5['open']
        feat['od_momentum'] = 1 if ((body1 > 0 and body5 > 0) if gap_dir == 'up' else (body1 < 0 and body5 < 0)) else 0
        c1_range = c1['high'] - c1['low']
        feat['first_candle_body_pct'] = abs(body1) / c1_range if c1_range > 0 else np.nan
        od_range_val = od_bars['high'].max() - od_bars['low'].min()
        close_935 = od_bars.iloc[-1]['close']
        if gap_dir == 'up':
            feat['od_close_vs_high'] = abs(close_935 - od_bars['high'].max()) / od_range_val if od_range_val > 0 else np.nan
        else:
            feat['od_close_vs_high'] = abs(close_935 - od_bars['low'].min()) / od_range_val if od_range_val > 0 else np.nan

    # 10:00 features
    post_od = rth[rth['time_et'].between('09:35', '09:59')]
    close_935_val = row['close_935'] if pd.notna(row['close_935']) else np.nan
    close_1000_val = row['close_1000'] if pd.notna(row['close_1000']) else np.nan

    if len(post_od) >= 10 and pd.notna(close_935_val):
        if gap_dir == 'up':
            feat['pullback_depth'] = max(close_935_val - post_od['low'].min(), 0) / adr
        else:
            feat['pullback_depth'] = max(post_od['high'].max() - close_935_val, 0) / adr

        if pd.notna(close_1000_val):
            if gap_dir == 'up':
                feat['continuation_after_od'] = (close_1000_val - close_935_val) / adr
            else:
                feat['continuation_after_od'] = (close_935_val - close_1000_val) / adr

        od_5min = rth[rth['time_et'].between('09:30', '09:34')]
        if len(od_5min) > 0:
            if gap_dir == 'up':
                feat['new_high_by_1000'] = 1 if post_od['high'].max() > od_5min['high'].max() else 0
            else:
                feat['new_high_by_1000'] = 1 if post_od['low'].min() < od_5min['low'].min() else 0

        post_od_b = rth[rth['time_et'].between('09:36', '09:59')]
        if len(post_od_b) > 0:
            if gap_dir == 'up':
                feat['bars_above_open'] = (post_od_b['close'] > rth_open).sum() / len(post_od_b)
            else:
                feat['bars_above_open'] = (post_od_b['close'] < rth_open).sum() / len(post_od_b)

        od_5v = rth[rth['time_et'].between('09:30', '09:34')]
        vol_5 = od_5v['volume'].sum() if len(od_5v) > 0 else 0
        feat['vol_30_vs_5'] = post_od['volume'].sum() / vol_5 if vol_5 > 0 else np.nan

    # 30min range
    first_30 = rth[rth['time_et'].between('09:30', '09:59')]
    if len(first_30) > 0:
        feat['range_30min_adr'] = (first_30['high'].max() - first_30['low'].min()) / adr

    feat['adr_pct'] = (adr / row['prev_close'] * 100) if pd.notna(row['prev_close']) and row['prev_close'] > 0 else np.nan

    new_features.append(feat)

print(f"OOS features computed: {len(new_features)}, errors: {errors}")

feat_df = pd.DataFrame(new_features)
oos = oos.merge(feat_df, on=['ticker', 'date'], how='left', suffixes=('', '_new'))
drop_cols = [c for c in oos.columns if c.endswith('_new')]
oos = oos.drop(columns=drop_cols)

# Compute drift
drift_raw = (oos['close_1000'] - oos['close_935']) / oos['adr_10']
is_up = oos['gap_direction'] == 'up'
oos['drift_30min_adr'] = np.where(is_up, drift_raw, -drift_raw)

# Also load IS data for comparison
is_df = pd.read_parquet("results/d11_is_enriched.parquet")
drift_raw_is = (is_df['close_1000'] - is_df['close_935']) / is_df['adr_10']
is_up_is = is_df['gap_direction'] == 'up'
is_df['drift_30min_adr'] = np.where(is_up_is, drift_raw_is, -drift_raw_is)

# =====================================================================
# Write results
# =====================================================================
header("D11 AUFGABE 6: OOS-VALIDIERUNG (H2)")
w()
w(f"OOS: 2024-01-01 bis 2026-02-06")
w(f"OOS alle Gapper: {len(oos_all)}, OOS OD>0.5+with_gap: {len(oos)}")
w(f"1min Fehler: {errors}")
w()
w()

# --- 6a: Big-Runner-Rate ---
header("6a: BIG-RUNNER-RATE IS vs OOS")
w()

for gap_dir, glabel in [('up', 'GapUp'), ('down', 'GapDn')]:
    is_g = is_df[is_df['gap_direction'] == gap_dir]
    oos_g = oos[oos['gap_direction'] == gap_dir]

    is_rate = is_g['is_big_runner'].mean() * 100
    oos_rate = oos_g['is_big_runner'].mean() * 100

    w(f"  {glabel}: IS = {is_rate:.1f}% (N={len(is_g)}) | OOS = {oos_rate:.1f}% (N={len(oos_g)}) | Delta = {oos_rate - is_rate:+.1f}%")

w()
w("  → Stabil?" )
w()
w()

# --- 6b: Top-5 Parameter Trennschaerfe ---
header("6b: TOP-5 PARAMETER IS vs OOS")
w()

top5_params = [
    ('drift_30min_adr', '30min_drift'),
    ('continuation_after_od', 'cont_after_od'),
    ('range_30min_adr', 'range_30min'),
    ('od_strength', 'OD_strength'),
    ('gap_size_in_adr', 'Gap_in_ADR'),
    ('rvol_5', 'RVOL_5'),
    ('rvol_open_30min', 'RVOL_30'),
    ('pullback_depth', 'pullback_depth'),
    ('first_candle_size', 'first_candle_size'),
]

for gap_dir, glabel in [('up', 'GapUp'), ('down', 'GapDn')]:
    is_g = is_df[is_df['gap_direction'] == gap_dir]
    oos_g = oos[oos['gap_direction'] == gap_dir]
    is_y = is_g['is_big_runner'].astype(int)
    oos_y = oos_g['is_big_runner'].astype(int)

    w(f"  --- {glabel} ---")
    w(f"  {'Parameter':<20s} | {'rho IS':>8s} | {'rho OOS':>8s} | {'Delta':>8s} | {'Repliz.':>8s}")
    w(f"  {'-'*60}")

    for col, label in top5_params:
        is_valid = is_g[col].notna() if col in is_g.columns else pd.Series([False]*len(is_g))
        oos_valid = oos_g[col].notna() if col in oos_g.columns else pd.Series([False]*len(oos_g))

        if is_valid.sum() < 30 or oos_valid.sum() < 30:
            continue

        rho_is, p_is = stats.spearmanr(is_g.loc[is_valid, col], is_y[is_valid])
        rho_oos, p_oos = stats.spearmanr(oos_g.loc[oos_valid, col], oos_y[oos_valid])

        delta = rho_oos - rho_is
        # Replicated if same sign and OOS is significant or at least half the IS effect
        replicated = "JA" if (np.sign(rho_is) == np.sign(rho_oos) and abs(rho_oos) >= abs(rho_is) * 0.5) else "NEIN"
        sig_oos = "***" if p_oos < 0.001 else ("**" if p_oos < 0.01 else ("*" if p_oos < 0.05 else ""))

        w(f"  {label:<20s} | {rho_is:>+8.3f} | {rho_oos:>+8.3f} | {delta:>+8.3f} | {replicated:>6s} {sig_oos}")

    w()

w()

# --- 6c: Combination rules OOS ---
header("6c: KOMBINATIONS-REGELN IS vs OOS")
w()

rules_to_validate = [
    # GapUp rules
    ("drift>0.25+RV5>5+Gap>2", lambda d: (d['drift_30min_adr'] > 0.25) & (d['rvol_5'] > 5) & (d['gap_size_in_adr'] > 2.0)),
    ("drift>0.25+RV30>5+Gap>2", lambda d: (d['drift_30min_adr'] > 0.25) & (d['rvol_open_30min'] > 5) & (d['gap_size_in_adr'] > 2.0)),
    ("drift>0.25+RV30>5+OD>1", lambda d: (d['drift_30min_adr'] > 0.25) & (d['rvol_open_30min'] > 5) & (d['od_strength'] > 1.0)),
    ("drift>0.50+RV30>5", lambda d: (d['drift_30min_adr'] > 0.50) & (d['rvol_open_30min'] > 5)),
    ("drift>0.25+RV30>5", lambda d: (d['drift_30min_adr'] > 0.25) & (d['rvol_open_30min'] > 5)),
    ("drift>0.25", lambda d: d['drift_30min_adr'] > 0.25),
    ("drift>0.50", lambda d: d['drift_30min_adr'] > 0.50),
    # GapDn rules
    ("pullback<0.10+drift>0.25", lambda d: (d['pullback_depth'] < 0.10) & (d['drift_30min_adr'] > 0.25)),
    ("pullback<0.10", lambda d: d['pullback_depth'] < 0.10),
    ("RV5>5+OD>1.0", lambda d: (d['rvol_5'] > 5) & (d['od_strength'] > 1.0)),
    ("RV30>5+OD>1.0", lambda d: (d['rvol_open_30min'] > 5) & (d['od_strength'] > 1.0)),
    # 9:35 rules
    ("fc>0.30+RV5>5+Gap>2", lambda d: (d['first_candle_size'] > 0.30) & (d['rvol_5'] > 5) & (d['gap_size_in_adr'] > 2.0)),
    ("RV5>5+OD>1+Gap>2", lambda d: (d['rvol_5'] > 5) & (d['od_strength'] > 1.0) & (d['gap_size_in_adr'] > 2.0)),
]

for gap_dir, glabel in [('up', 'GapUp'), ('down', 'GapDn')]:
    is_g = is_df[is_df['gap_direction'] == gap_dir]
    oos_g = oos[oos['gap_direction'] == gap_dir]
    is_base = is_g['is_big_runner'].mean() * 100
    oos_base = oos_g['is_big_runner'].mean() * 100

    w(f"  --- {glabel} (IS Base: {is_base:.1f}%, OOS Base: {oos_base:.1f}%) ---")
    w(f"  {'Regel':<28s} | {'BR% IS':>7s} | {'N IS':>5s} | {'BR% OOS':>8s} | {'N OOS':>6s} | {'Repliz.':>8s}")
    w(f"  {'-'*72}")

    for rname, rfunc in rules_to_validate:
        try:
            is_mask = rfunc(is_g).fillna(False)
            oos_mask = rfunc(oos_g).fillna(False)
            is_sub = is_g[is_mask]
            oos_sub = oos_g[oos_mask]
            n_is = len(is_sub)
            n_oos = len(oos_sub)
            if n_is < 10 and n_oos < 10:
                continue
            is_br = is_sub['is_big_runner'].mean() * 100 if n_is > 0 else 0
            oos_br = oos_sub['is_big_runner'].mean() * 100 if n_oos > 0 else 0
            replicated = "JA" if oos_br >= is_base * 1.3 and n_oos >= 10 else "NEIN"
            w(f"  {rname:<28s} | {is_br:>6.1f}% | {n_is:>5d} | {oos_br:>7.1f}% | {n_oos:>6d} | {replicated:>8s}")
        except:
            pass
    w()

w()

# --- 6d: Profile comparison OOS ---
header("6d: PROFIL-VERGLEICH OOS (BR vs ROC_Bad)")
w()

profile_cols = [
    ('od_strength', 'OD_strength'),
    ('gap_size_in_adr', 'Gap_in_ADR'),
    ('rvol_5', 'RVOL_5'),
    ('rvol_open_30min', 'RVOL_30'),
    ('drift_30min_adr', '30min_drift'),
    ('pullback_depth', 'pullback_depth'),
    ('continuation_after_od', 'cont_after_od'),
    ('prev_close', 'Preis'),
]

for gap_dir, glabel in [('up', 'GapUp'), ('down', 'GapDn')]:
    g = oos[oos['gap_direction'] == gap_dir]
    br = g[g['is_big_runner']]
    roc = g[g['range_ok_close_bad']]
    rb = g[g['range_bad']]

    w(f"  --- {glabel} OOS (BR N={len(br)}, ROC N={len(roc)}, RB N={len(rb)}) ---")
    w(f"  {'Metrik':<18s} | {'Big Runner':>10s} | {'Lief+Kam':>10s} | {'LiefNicht':>10s} | {'BR-ROC':>8s}")
    w(f"  {'-'*64}")

    for col, label in profile_cols:
        if col not in g.columns:
            continue
        br_med = br[col].median() if len(br) > 0 else np.nan
        roc_med = roc[col].median() if len(roc) > 0 else np.nan
        rb_med = rb[col].median() if len(rb) > 0 else np.nan
        delta = br_med - roc_med if pd.notna(br_med) and pd.notna(roc_med) else np.nan
        w(f"  {label:<18s} | {br_med:>10.3f} | {roc_med:>10.3f} | {rb_med:>10.3f} | {delta:>+8.3f}")

    w()

w()

# --- 6e: AUC OOS ---
header("6e: AUC IS vs OOS")
w()

features_935 = ['od_strength', 'rvol_5', 'gap_size_in_adr', 'first_candle_size',
                'pm_drift_adr', 'od_body_pct', 'od_wick_ratio', 'prev_close',
                'adr_10', 'pm_rth5', 'od_momentum', 'first_candle_body_pct',
                'od_close_vs_high', 'pm_range_adr']

features_1000 = features_935 + ['drift_30min_adr', 'rvol_open_30min', 'range_30min_adr',
                                  'pullback_depth', 'continuation_after_od',
                                  'new_high_by_1000', 'bars_above_open',
                                  'pm_rth30_computed', 'vol_30_vs_5']

for gap_dir, glabel in [('up', 'GapUp'), ('down', 'GapDn')]:
    is_g = is_df[is_df['gap_direction'] == gap_dir]
    oos_g = oos[oos['gap_direction'] == gap_dir]

    w(f"  --- {glabel} ---")

    for feat_label, feat_cols in [("9:35", features_935), ("+ 10:00", features_1000)]:
        # Get common columns
        avail_is = [c for c in feat_cols if c in is_g.columns]
        avail_oos = [c for c in feat_cols if c in oos_g.columns]
        common = [c for c in avail_is if c in avail_oos]

        is_valid = is_g[common].notna().all(axis=1)
        oos_valid = oos_g[common].notna().all(axis=1)

        X_is = is_g.loc[is_valid, common]
        y_is = is_g.loc[is_valid, 'is_big_runner'].astype(int)
        X_oos = oos_g.loc[oos_valid, common]
        y_oos = oos_g.loc[oos_valid, 'is_big_runner'].astype(int)

        if len(y_is) < 30 or len(y_oos) < 30 or y_is.sum() < 5 or y_oos.sum() < 5:
            continue

        scaler = StandardScaler()
        X_is_s = scaler.fit_transform(X_is)
        X_oos_s = scaler.transform(X_oos)

        lr = LogisticRegression(max_iter=1000, penalty='l2', C=1.0)
        lr.fit(X_is_s, y_is)

        auc_is = roc_auc_score(y_is, lr.predict_proba(X_is_s)[:, 1])
        auc_oos = roc_auc_score(y_oos, lr.predict_proba(X_oos_s)[:, 1])

        w(f"  {feat_label:<10s}: IS AUC = {auc_is:.3f} | OOS AUC = {auc_oos:.3f} | Delta = {auc_oos - auc_is:+.3f}")

    w()

w()
w("  HINWEIS: Modell trainiert auf IS, getestet auf OOS (echte Generalisierung).")

# Save OOS enriched data
oos.to_parquet("results/d11_oos_enriched.parquet", index=False)

# Write output
os.makedirs("results", exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print(f"Geschrieben: {OUT} ({len(lines)} Zeilen)")
