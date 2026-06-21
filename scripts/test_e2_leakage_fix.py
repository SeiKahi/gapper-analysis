###############################################################################
# TEST 3: e2_confirmed Leakage Fix
#
# e2_confirmed is a 1-bar look-ahead feature for E1 trades.
# At E1 entry time, the next bar's close is NOT yet known.
# Fix: Set e2_confirmed = 0 for all E1 trades
# Re-score with saved models, compare AUC before vs after
###############################################################################

import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import roc_auc_score

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE_DIR = str(PROJECT_ROOT)
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# ============================================================
# Load data
# ============================================================
print("Loading trades...", file=sys.stderr)
trades_all = pd.read_parquet(os.path.join(RESULTS_DIR, 'all_trades_v3.parquet'))

# Need to reconstruct the encoded features (scenario_id, entry_id, sl_id, tgt_id)
SCENARIOS = [
    ('GapUp_2s_Fade', 'up', 'upper_2std', 'short', 'upper_1std', 'vwap'),
    ('GapUp_1s_Fade', 'up', 'upper_1std', 'short', 'vwap', 'lower_1std'),
    ('GapUp_VWAP_Brk', 'up', 'vwap', 'short', 'lower_1std', 'lower_2std'),
    ('GapDn_2s_Fade', 'down', 'lower_2std', 'long', 'lower_1std', 'vwap'),
    ('GapDn_1s_Fade', 'down', 'lower_1std', 'long', 'vwap', 'upper_1std'),
    ('GapDn_VWAP_Brk', 'down', 'vwap', 'long', 'upper_1std', 'upper_2std'),
]

scenario_map = {s[0]: i for i, s in enumerate(SCENARIOS)}
trades_all['scenario_id'] = trades_all['scenario'].map(scenario_map)

entry_map = {'E1': 0, 'E2': 1}
trades_all['entry_id'] = trades_all['entry_type'].map(entry_map)

sl_methods_unique = sorted(trades_all['sl_method'].unique())
sl_map = {s: i for i, s in enumerate(sl_methods_unique)}
trades_all['sl_id'] = trades_all['sl_method'].map(sl_map)

tgt_methods_unique = sorted(trades_all['target_method'].unique())
tgt_map = {t: i for i, t in enumerate(tgt_methods_unique)}
trades_all['tgt_id'] = trades_all['target_method'].map(tgt_map)

# Label
trades_all['label'] = (trades_all['outcome'] == 'target_hit').astype(int)

# Feature columns (must match pipeline v3)
feature_cols = [
    'minutes_since_open', 'time_bucket',
    'dist_to_level_adr', 'dist_to_vwap_adr', 'pct_adr_used',
    'band_width_adr', 'z_score', 'close_vol_10',
    'momentum_5', 'momentum_15', 'volume_ratio', 'vwap_slope',
    'e2_confirmed', 'consec_closes_in_zone',
    'gap_pct', 'gap_adr', 'rvol_30', 'adr_pct', 'adr_10',
    'early_range_15', 'early_range_30', 'log_market_cap',
    'sl_dist_adr',
]
all_features = feature_cols + ['scenario_id', 'entry_id', 'sl_id', 'tgt_id']

# NaN handling
for col in all_features:
    if col in trades_all.columns:
        trades_all[col] = trades_all[col].fillna(0).astype(float)

# Split
train = trades_all[trades_all['split'] == 'train'].copy()
val = trades_all[trades_all['split'] == 'val'].copy()

print(f"Train: {len(train)}, Val: {len(val)}", file=sys.stderr)
print(f"E1 in val: {(val['entry_type']=='E1').sum()}, E2 in val: {(val['entry_type']=='E2').sum()}", file=sys.stderr)

# ============================================================
# Load saved models
# ============================================================
print("Loading models...", file=sys.stderr)
with open(os.path.join(MODELS_DIR, 'xgb_v3.pkl'), 'rb') as f:
    xgb_model = pickle.load(f)
with open(os.path.join(MODELS_DIR, 'lgb_v3.pkl'), 'rb') as f:
    lgb_model = pickle.load(f)

# ============================================================
# A) ORIGINAL predictions (with leak)
# ============================================================
print("Computing original predictions...", file=sys.stderr)
xgb_probs_orig = xgb_model.predict_proba(val[all_features])[:, 1]
lgb_probs_orig = lgb_model.predict_proba(val[all_features])[:, 1]
ensemble_orig = (xgb_probs_orig + lgb_probs_orig) / 2

xgb_auc_orig = roc_auc_score(val['label'], xgb_probs_orig)
lgb_auc_orig = roc_auc_score(val['label'], lgb_probs_orig)
ens_auc_orig = roc_auc_score(val['label'], ensemble_orig)

# E1-only AUC
e1_mask = val['entry_type'] == 'E1'
e2_mask = val['entry_type'] == 'E2'

xgb_auc_e1_orig = roc_auc_score(val.loc[e1_mask, 'label'], xgb_probs_orig[e1_mask.values])
xgb_auc_e2_orig = roc_auc_score(val.loc[e2_mask, 'label'], xgb_probs_orig[e2_mask.values])

# ============================================================
# B) FIXED predictions (e2_confirmed = 0 for E1 trades)
# ============================================================
print("Computing fixed predictions...", file=sys.stderr)
val_fixed = val.copy()
val_fixed.loc[val_fixed['entry_type'] == 'E1', 'e2_confirmed'] = 0

xgb_probs_fixed = xgb_model.predict_proba(val_fixed[all_features])[:, 1]
lgb_probs_fixed = lgb_model.predict_proba(val_fixed[all_features])[:, 1]
ensemble_fixed = (xgb_probs_fixed + lgb_probs_fixed) / 2

xgb_auc_fixed = roc_auc_score(val['label'], xgb_probs_fixed)
lgb_auc_fixed = roc_auc_score(val['label'], lgb_probs_fixed)
ens_auc_fixed = roc_auc_score(val['label'], ensemble_fixed)

xgb_auc_e1_fixed = roc_auc_score(val.loc[e1_mask, 'label'], xgb_probs_fixed[e1_mask.values])

# ============================================================
# C) High-confidence trade analysis (before and after)
# ============================================================
print("Analyzing HC trades...", file=sys.stderr)

hc_thresholds = [0.5, 0.6, 0.7]
hc_results = {}

for thresh in hc_thresholds:
    # Original
    hc_orig_mask = ensemble_orig >= thresh
    hc_orig = val[hc_orig_mask]
    hc_orig_n = len(hc_orig)
    hc_orig_wr = hc_orig['label'].mean() if hc_orig_n > 0 else 0
    hc_orig_pnl = hc_orig['pnl_adr'].mean() if hc_orig_n > 0 else 0

    # Fixed
    hc_fixed_mask = ensemble_fixed >= thresh
    hc_fixed = val[hc_fixed_mask]
    hc_fixed_n = len(hc_fixed)
    hc_fixed_wr = hc_fixed['label'].mean() if hc_fixed_n > 0 else 0
    hc_fixed_pnl = hc_fixed['pnl_adr'].mean() if hc_fixed_n > 0 else 0

    hc_results[thresh] = {
        'orig': (hc_orig_n, hc_orig_wr, hc_orig_pnl),
        'fixed': (hc_fixed_n, hc_fixed_wr, hc_fixed_pnl),
    }

# ============================================================
# D) Retrain model without e2_confirmed entirely
# ============================================================
print("Retraining without e2_confirmed...", file=sys.stderr)
from xgboost import XGBClassifier

features_no_e2 = [f for f in all_features if f != 'e2_confirmed']

xgb_no_e2 = XGBClassifier(
    n_estimators=800,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=50,
    reg_alpha=0.1,
    reg_lambda=1.0,
    eval_metric='auc',
    early_stopping_rounds=50,
    random_state=42,
    n_jobs=-1,
)
xgb_no_e2.fit(
    train[features_no_e2], train['label'],
    eval_set=[(val[features_no_e2], val['label'])],
    verbose=0,
)

xgb_probs_no_e2 = xgb_no_e2.predict_proba(val[features_no_e2])[:, 1]
xgb_auc_no_e2 = roc_auc_score(val['label'], xgb_probs_no_e2)
xgb_auc_e1_no_e2 = roc_auc_score(val.loc[e1_mask, 'label'], xgb_probs_no_e2[e1_mask.values])

# Also retrain with corrected e2_confirmed (=0 for E1 in train too)
print("Retraining with corrected e2_confirmed...", file=sys.stderr)
train_corrected = train.copy()
train_corrected.loc[train_corrected['entry_type'] == 'E1', 'e2_confirmed'] = 0

xgb_corrected = XGBClassifier(
    n_estimators=800,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=50,
    reg_alpha=0.1,
    reg_lambda=1.0,
    eval_metric='auc',
    early_stopping_rounds=50,
    random_state=42,
    n_jobs=-1,
)
xgb_corrected.fit(
    train_corrected[all_features], train_corrected['label'],
    eval_set=[(val_fixed[all_features], val_fixed['label'])],
    verbose=0,
)

xgb_probs_corrected = xgb_corrected.predict_proba(val_fixed[all_features])[:, 1]
xgb_auc_corrected = roc_auc_score(val['label'], xgb_probs_corrected)
xgb_auc_e1_corrected = roc_auc_score(val.loc[e1_mask, 'label'], xgb_probs_corrected[e1_mask.values])

# HC trades with corrected model
ens_corrected = xgb_probs_corrected  # just XGB for simplicity
hc_corrected_results = {}
for thresh in hc_thresholds:
    hc_mask = ens_corrected >= thresh
    hc = val[hc_mask]
    hc_n = len(hc)
    hc_wr = hc['label'].mean() if hc_n > 0 else 0
    hc_pnl = hc['pnl_adr'].mean() if hc_n > 0 else 0
    hc_corrected_results[thresh] = (hc_n, hc_wr, hc_pnl)

# ============================================================
# Write results
# ============================================================
output_lines = []
output_lines.append("=" * 80)
output_lines.append("TEST 3: e2_confirmed LEAKAGE FIX")
output_lines.append("=" * 80)
output_lines.append(f"")
output_lines.append(f"ISSUE: e2_confirmed checks if the NEXT bar closes in trade direction.")
output_lines.append(f"       For E2 trades this is the entry condition (known at entry time).")
output_lines.append(f"       For E1 trades this is FUTURE information (1-bar look-ahead leak).")
output_lines.append(f"       69% of trades are E1 -> majority affected by leak.")
output_lines.append(f"")
output_lines.append(f"Val size: {len(val)}, E1: {e1_mask.sum()}, E2: {e2_mask.sum()}")
output_lines.append(f"")

output_lines.append("=" * 80)
output_lines.append("A) AUC COMPARISON: Original vs Fixed (zero e2_confirmed for E1)")
output_lines.append("=" * 80)
output_lines.append(f"")
output_lines.append(f"{'Model':>15s} | {'Original AUC':>12s} | {'Fixed AUC':>12s} | {'Delta':>8s}")
output_lines.append("-" * 60)
output_lines.append(f"{'XGBoost':>15s} | {xgb_auc_orig:>12.4f} | {xgb_auc_fixed:>12.4f} | {xgb_auc_fixed - xgb_auc_orig:>+7.4f}")
output_lines.append(f"{'LightGBM':>15s} | {lgb_auc_orig:>12.4f} | {lgb_auc_fixed:>12.4f} | {lgb_auc_fixed - lgb_auc_orig:>+7.4f}")
output_lines.append(f"{'Ensemble':>15s} | {ens_auc_orig:>12.4f} | {ens_auc_fixed:>12.4f} | {ens_auc_fixed - ens_auc_orig:>+7.4f}")
output_lines.append(f"")
output_lines.append(f"NOTE: 'Fixed' uses the SAME models but sets e2_confirmed=0 for E1 at inference time.")
output_lines.append(f"      This measures how much the model RELIED on leaked e2_confirmed for E1 predictions.")
output_lines.append(f"")

output_lines.append("=" * 80)
output_lines.append("B) E1-ONLY AUC (most affected by leak)")
output_lines.append("=" * 80)
output_lines.append(f"")
output_lines.append(f"  XGBoost E1 AUC original:  {xgb_auc_e1_orig:.4f}")
output_lines.append(f"  XGBoost E1 AUC fixed:     {xgb_auc_e1_fixed:.4f}")
output_lines.append(f"  Delta:                    {xgb_auc_e1_fixed - xgb_auc_e1_orig:+.4f}")
output_lines.append(f"")
output_lines.append(f"  XGBoost E2 AUC original:  {xgb_auc_e2_orig:.4f} (should be unchanged)")
output_lines.append(f"")

output_lines.append("=" * 80)
output_lines.append("C) HIGH-CONFIDENCE TRADES: Before vs After Fix")
output_lines.append("=" * 80)
output_lines.append(f"")

for thresh in hc_thresholds:
    orig = hc_results[thresh]['orig']
    fixed = hc_results[thresh]['fixed']
    output_lines.append(f"  Threshold >= {thresh}:")
    output_lines.append(f"    Original:  N={orig[0]:6d}, WR={orig[1]:.1%}, AvgPnL={orig[2]:+.4f}")
    output_lines.append(f"    Fixed:     N={fixed[0]:6d}, WR={fixed[1]:.1%}, AvgPnL={fixed[2]:+.4f}")
    output_lines.append(f"    N change:  {fixed[0] - orig[0]:+d}")
    output_lines.append(f"")

output_lines.append("=" * 80)
output_lines.append("D) RETRAINED MODELS (proper fix)")
output_lines.append("=" * 80)
output_lines.append(f"")
output_lines.append(f"Method 1: Remove e2_confirmed entirely")
output_lines.append(f"  XGBoost AUC (all val):  {xgb_auc_no_e2:.4f}  (original: {xgb_auc_orig:.4f}, delta: {xgb_auc_no_e2 - xgb_auc_orig:+.4f})")
output_lines.append(f"  XGBoost AUC (E1 only):  {xgb_auc_e1_no_e2:.4f}  (original: {xgb_auc_e1_orig:.4f}, delta: {xgb_auc_e1_no_e2 - xgb_auc_e1_orig:+.4f})")
output_lines.append(f"")
output_lines.append(f"Method 2: Retrain with e2_confirmed=0 for E1 (keep feature, fix in train+val)")
output_lines.append(f"  XGBoost AUC (all val):  {xgb_auc_corrected:.4f}  (original: {xgb_auc_orig:.4f}, delta: {xgb_auc_corrected - xgb_auc_orig:+.4f})")
output_lines.append(f"  XGBoost AUC (E1 only):  {xgb_auc_e1_corrected:.4f}  (original: {xgb_auc_e1_orig:.4f}, delta: {xgb_auc_e1_corrected - xgb_auc_e1_orig:+.4f})")
output_lines.append(f"")

output_lines.append(f"HC Trades with corrected model (XGB only):")
for thresh in hc_thresholds:
    n, wr, pnl = hc_corrected_results[thresh]
    orig = hc_results[thresh]['orig']
    output_lines.append(f"  >= {thresh}: N={n:6d} (was {orig[0]}), WR={wr:.1%} (was {orig[1]:.1%}), PnL={pnl:+.4f} (was {orig[2]:+.4f})")
output_lines.append(f"")

output_lines.append("=" * 80)
output_lines.append("E) VERDICT")
output_lines.append("=" * 80)
output_lines.append(f"")

auc_drop = xgb_auc_orig - xgb_auc_corrected
if auc_drop > 0.03:
    severity = "SEVERE LEAK — AUC drops significantly when fixed"
elif auc_drop > 0.01:
    severity = "MODERATE LEAK — noticeable AUC impact"
elif auc_drop > 0.005:
    severity = "MILD LEAK — small but measurable impact"
else:
    severity = "NEGLIGIBLE — e2_confirmed was not a meaningful leak"

output_lines.append(f"  AUC drop from original to corrected retrain: {auc_drop:+.4f}")
output_lines.append(f"  Severity: {severity}")
output_lines.append(f"")
output_lines.append(f"  e2_confirmed had importance rank 3 (0.079) but the actual AUC impact")
output_lines.append(f"  from fixing the leak tells us how much of that importance was")
output_lines.append(f"  genuinely predictive vs just exploiting future information.")

# Write to file
output_path = os.path.join(RESULTS_DIR, 'test_e2_leakage.txt')
with open(output_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(output_lines))

print(f"\nResults written to {output_path}", file=sys.stderr)
print("Done!", file=sys.stderr)
