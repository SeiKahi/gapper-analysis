"""
D11 Aufgabe 4: Multivariater Vergleich (IS)
Logistic Regression feature importance, combination rules, profile comparison.
"""
import pandas as pd
import numpy as np
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

OUT = "results/d11_multivariate.txt"
lines = []

def w(text=""):
    lines.append(text)

def header(title):
    w("=" * 70)
    w(title)
    w("=" * 70)

# Load enriched IS data
df = pd.read_parquet("results/d11_is_enriched.parquet")
print(f"Loaded: {len(df)} rows")

# Recompute drift_30min_adr
drift_raw = (df['close_1000'] - df['close_935']) / df['adr_10']
is_up = df['gap_direction'] == 'up'
df['drift_30min_adr'] = np.where(is_up, drift_raw, -drift_raw)

header("D11 AUFGABE 4: MULTIVARIATER VERGLEICH (IS)")
w()
w(f"Basis: OD > 0.5 ADR + with_gap, N = {len(df)}")
w()
w()

# =====================================================================
# 4a: Top-10 Parameters per gap direction
# =====================================================================
header("4a: TOP-10 PARAMETER NACH TRENNSCHAERFE")
w()

# Top features from integrated ranking (Aufgabe 2+3)
top_features_up = [
    ('drift_30min_adr', '30min_drift'),
    ('continuation_after_od', 'continuation_after_od'),
    ('range_30min_adr', 'range_30min_adr'),
    ('gap_size_in_adr', 'Gap_in_ADR'),
    ('bars_above_open', 'bars_above_open'),
    ('first_candle_size', 'first_candle_size'),
    ('rvol_open_30min', 'RVOL_30'),
    ('rvol_5', 'RVOL_5'),
    ('od_strength', 'OD_strength'),
    ('spy_return_day', 'SPY_return'),
]

top_features_dn = [
    ('drift_30min_adr', '30min_drift'),
    ('continuation_after_od', 'continuation_after_od'),
    ('range_30min_adr', 'range_30min_adr'),
    ('od_strength', 'OD_strength'),
    ('pullback_depth', 'pullback_depth'),
    ('adr_10', 'ADR_10'),
    ('rsi_14_prev', 'RSI_14_prev'),
    ('prev_close', 'Preis'),
    ('pm_drift_adr', 'pm_drift_adr'),
    ('spy_return_day', 'SPY_return'),
]

for glabel, top_feats in [('GapUp', top_features_up), ('GapDn', top_features_dn)]:
    w(f"  --- {glabel} Top-10 ---")
    for i, (col, label) in enumerate(top_feats):
        w(f"    {i+1}. {label} ({col})")
    w()

# =====================================================================
# 4b: Logistic Regression
# =====================================================================
header("4b: LOGISTIC REGRESSION (Feature Importance)")
w()

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

for gap_dir, glabel, top_feats in [('up', 'GapUp', top_features_up), ('down', 'GapDn', top_features_dn)]:
    g = df[df['gap_direction'] == gap_dir].copy()
    y = g['is_big_runner'].astype(int)

    feat_cols = [c for c, _ in top_feats]
    X = g[feat_cols].copy()

    # Drop rows with NaN
    valid = X.notna().all(axis=1)
    X_valid = X[valid]
    y_valid = y[valid]

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_valid)

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, penalty='l2', C=1.0)
    lr.fit(X_scaled, y_valid)

    # Get coefficients and pseudo p-values via permutation
    coefs = lr.coef_[0]
    odds_ratios = np.exp(coefs)

    # Approximate p-values using Wald test
    # SE from Hessian approximation
    probs = lr.predict_proba(X_scaled)[:, 1]
    W = np.diag(probs * (1 - probs))
    try:
        cov_matrix = np.linalg.inv(X_scaled.T @ W @ X_scaled)
        se = np.sqrt(np.diag(cov_matrix))
        z_scores = coefs / se
        p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))
    except:
        p_values = np.full(len(coefs), np.nan)

    # Sort by abs coefficient
    order = np.argsort(np.abs(coefs))[::-1]

    w(f"  --- {glabel} (N={len(y_valid)}, BR={y_valid.sum()}) ---")
    w(f"  {'Rank':>4s} | {'Feature':<25s} | {'Coef':>8s} | {'p-value':>10s} | {'OR':>6s}")
    w(f"  {'-'*62}")
    for rank, i in enumerate(order):
        col_name = [lab for _, lab in top_feats][i]
        sig = "***" if p_values[i] < 0.001 else ("**" if p_values[i] < 0.01 else ("*" if p_values[i] < 0.05 else ""))
        w(f"  {rank+1:>4d} | {col_name:<25s} | {coefs[i]:>+8.3f} | {p_values[i]:>10.4f} | {odds_ratios[i]:>6.2f} {sig}")
    w()

    # Training accuracy
    preds = lr.predict(X_scaled)
    acc = (preds == y_valid).mean()
    auc_score = None
    try:
        from sklearn.metrics import roc_auc_score
        auc_score = roc_auc_score(y_valid, probs)
    except:
        pass
    w(f"  Accuracy: {acc:.1%}, AUC: {auc_score:.3f}" if auc_score else f"  Accuracy: {acc:.1%}")
    w()

w("  HINWEIS: Logistic Regression NUR fuer Feature-Importance, NICHT als Prediktor!")
w()
w()

# =====================================================================
# 4c: Combination Rules
# =====================================================================
header("4c: EINFACHE KOMBINATIONS-REGELN")
w()

# Define rules to test — using predefined thresholds (no optimization)
rules = [
    # Single parameter rules
    ("drift_30min > 0.50", lambda d: d['drift_30min_adr'] > 0.50),
    ("drift_30min > 0.25", lambda d: d['drift_30min_adr'] > 0.25),
    ("RVOL_30 > 5x", lambda d: d['rvol_open_30min'] > 5),
    ("RVOL_5 > 5x", lambda d: d['rvol_5'] > 5),
    ("OD > 1.0 ADR", lambda d: d['od_strength'] > 1.0),
    ("Gap > 2.0 ADR", lambda d: d['gap_size_in_adr'] > 2.0),
    ("range_30 > 1.5", lambda d: d['range_30min_adr'] > 1.5),
    ("pullback < 0.10", lambda d: d['pullback_depth'] < 0.10),
    ("cont_od > 0.50", lambda d: d['continuation_after_od'] > 0.50),

    # 2er combinations
    ("drift>0.25 + RVOL30>5", lambda d: (d['drift_30min_adr'] > 0.25) & (d['rvol_open_30min'] > 5)),
    ("drift>0.50 + RVOL30>5", lambda d: (d['drift_30min_adr'] > 0.50) & (d['rvol_open_30min'] > 5)),
    ("drift>0.25 + OD>1.0", lambda d: (d['drift_30min_adr'] > 0.25) & (d['od_strength'] > 1.0)),
    ("drift>0.25 + Gap>2.0", lambda d: (d['drift_30min_adr'] > 0.25) & (d['gap_size_in_adr'] > 2.0)),
    ("RVOL30>5 + Gap>2.0", lambda d: (d['rvol_open_30min'] > 5) & (d['gap_size_in_adr'] > 2.0)),
    ("RVOL30>5 + OD>1.0", lambda d: (d['rvol_open_30min'] > 5) & (d['od_strength'] > 1.0)),
    ("range30>1.5 + RVOL30>5", lambda d: (d['range_30min_adr'] > 1.5) & (d['rvol_open_30min'] > 5)),
    ("pullback<0.10 + drift>0.25", lambda d: (d['pullback_depth'] < 0.10) & (d['drift_30min_adr'] > 0.25)),

    # 3er combinations
    ("drift>0.25+RV30>5+OD>1.0", lambda d: (d['drift_30min_adr'] > 0.25) & (d['rvol_open_30min'] > 5) & (d['od_strength'] > 1.0)),
    ("drift>0.25+RV30>5+Gap>2", lambda d: (d['drift_30min_adr'] > 0.25) & (d['rvol_open_30min'] > 5) & (d['gap_size_in_adr'] > 2.0)),
    ("drift>0.50+RV30>5+Gap>2", lambda d: (d['drift_30min_adr'] > 0.50) & (d['rvol_open_30min'] > 5) & (d['gap_size_in_adr'] > 2.0)),

    # 9:35-only rules
    ("RVOL5>5 + OD>1.0 + Gap>2", lambda d: (d['rvol_5'] > 5) & (d['od_strength'] > 1.0) & (d['gap_size_in_adr'] > 2.0)),
    ("RVOL5>10 + Gap>2", lambda d: (d['rvol_5'] > 10) & (d['gap_size_in_adr'] > 2.0)),
    ("RVOL5>5 + Gap>2", lambda d: (d['rvol_5'] > 5) & (d['gap_size_in_adr'] > 2.0)),
]

for gap_dir, glabel in [('up', 'GapUp'), ('down', 'GapDn')]:
    g = df[df['gap_direction'] == gap_dir].copy()
    base_rate = g['is_big_runner'].mean() * 100

    w(f"  --- {glabel} (Base Rate: {base_rate:.1f}%, N={len(g)}) ---")
    w(f"  {'Regel':<30s} | {'N':>5s} | {'BR':>4s} | {'BR%':>7s} | {'Lift':>5s}")
    w(f"  {'-'*60}")

    results_list = []
    for rname, rfunc in rules:
        try:
            mask = rfunc(g)
            sub = g[mask.fillna(False)]
            n = len(sub)
            if n < 10:
                continue
            br_n = sub['is_big_runner'].sum()
            br_pct = br_n / n * 100 if n > 0 else 0
            lift = br_pct / base_rate if base_rate > 0 else 0
            results_list.append((rname, n, br_n, br_pct, lift))
        except:
            pass

    # Sort by lift
    results_list.sort(key=lambda x: x[3], reverse=True)
    for rname, n, br_n, br_pct, lift in results_list:
        low_n = " [LOW N]" if n < 20 else ""
        w(f"  {rname:<30s} | {n:>5d} | {br_n:>4d} | {br_pct:>6.1f}% | {lift:>4.1f}x{low_n}")
    w()

w()

# =====================================================================
# 4d: Profile Comparison (3 groups)
# =====================================================================
header("4d: PROFIL-VERGLEICH (3 Gruppen)")
w()

profile_cols = [
    ('od_strength', 'OD_strength'),
    ('gap_size_in_adr', 'Gap_in_ADR'),
    ('rvol_5', 'RVOL_5'),
    ('rvol_open_30min', 'RVOL_30'),
    ('drift_30min_adr', '30min_drift'),
    ('range_30min_adr', 'range_30min'),
    ('pullback_depth', 'pullback_depth'),
    ('continuation_after_od', 'cont_after_od'),
    ('pm_drift_adr', 'pm_drift'),
    ('prev_close', 'Preis'),
    ('adr_10', 'ADR_10'),
    ('first_candle_size', 'first_candle_size'),
    ('pm_rth30_computed', 'PM/RTH30'),
]

for gap_dir, glabel in [('up', 'GapUp'), ('down', 'GapDn')]:
    g = df[df['gap_direction'] == gap_dir].copy()

    br = g[g['is_big_runner']]
    roc_bad = g[g['range_ok_close_bad']]
    range_bad = g[g['range_bad']]

    w(f"  --- {glabel} (BR N={len(br)}, ROC_Bad N={len(roc_bad)}, Range_Bad N={len(range_bad)}) ---")
    w(f"  {'Metrik':<18s} | {'Big Runner':>10s} | {'Lief+Kam':>10s} | {'LiefNicht':>10s} | {'BR-ROC':>8s} | {'d(BR-ROC)':>8s}")
    w(f"  {'-'*78}")

    for col, label in profile_cols:
        if col not in g.columns:
            continue
        br_med = br[col].median()
        roc_med = roc_bad[col].median()
        rb_med = range_bad[col].median()

        delta = br_med - roc_med if pd.notna(br_med) and pd.notna(roc_med) else np.nan

        # Cohen's d (BR vs ROC_BAD)
        br_vals = br[col].dropna()
        roc_vals = roc_bad[col].dropna()
        if len(br_vals) > 2 and len(roc_vals) > 2:
            pooled_std = np.sqrt((br_vals.std()**2 + roc_vals.std()**2) / 2)
            cohens_d = (br_vals.mean() - roc_vals.mean()) / pooled_std if pooled_std > 0 else 0
            d_s = f"{cohens_d:>+7.2f}"
        else:
            d_s = "    N/A"

        fmt = ".3f" if abs(br_med) < 100 else ".0f"
        w(f"  {label:<18s} | {br_med:>10{fmt}} | {roc_med:>10{fmt}} | {rb_med:>10{fmt}} | {delta:>+8{fmt}} | {d_s}")

    w()
    w()

w("  Legende:")
w("    Big Runner = Range >= 1.5 ADR + Close >= 80% Extremum")
w("    Lief+Kam (ROC_Bad) = Range >= 1.5 ADR + Close < 80% (kam zurueck)")
w("    LiefNicht (Range_Bad) = Range < 1.5 ADR")
w("    BR-ROC = Delta Big Runner vs Lief+Kam")
w("    d(BR-ROC) = Cohen's d (Effektstaerke)")

# Write output
os.makedirs("results", exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print(f"Geschrieben: {OUT} ({len(lines)} Zeilen)")
