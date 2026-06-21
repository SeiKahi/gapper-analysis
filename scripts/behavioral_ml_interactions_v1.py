###############################################################################
# ML INTERACTION MODEL v1 — Durchlauf 2.0
#
# Verwendet behavioral_events_v1.parquet als Input.
# XGBoost Regressor auf MFE_30 (kein binaeres Label!).
# SHAP Values fuer Interaktions-Effekte.
# Conditional Analysis nach ML-identifizierten Subgruppen.
#
# Run:
#   .\gapper_env\Scripts\python.exe scripts\behavioral_ml_interactions_v1.py
###############################################################################

import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE_DIR = str(PROJECT_ROOT)
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
EVENTS_PATH = os.path.join(RESULTS_DIR, 'behavioral_events_v1.parquet')

# ============================================================
# LOAD EVENTS
# ============================================================
print("Loading behavioral events...", file=sys.stderr)
df = pd.read_parquet(EVENTS_PATH)
print(f"  {len(df)} events loaded", file=sys.stderr)

train = df[df['split'] == 'train'].copy()
val = df[df['split'] == 'val'].copy()
print(f"  Train: {len(train)}, Val: {len(val)}", file=sys.stderr)

# ============================================================
# FEATURE ENGINEERING
# ============================================================
# Encode scenario as separate features
df['is_gapdn'] = (df['gap_direction'] == 'down').astype(int)
df['is_2s'] = df['scenario'].str.contains('2s').astype(int)
df['is_1s'] = df['scenario'].str.contains('1s').astype(int)
df['is_vwap_brk'] = df['scenario'].str.contains('VWAP').astype(int)

train['is_gapdn'] = (train['gap_direction'] == 'down').astype(int)
train['is_2s'] = train['scenario'].str.contains('2s').astype(int)
train['is_1s'] = train['scenario'].str.contains('1s').astype(int)
train['is_vwap_brk'] = train['scenario'].str.contains('VWAP').astype(int)

val['is_gapdn'] = (val['gap_direction'] == 'down').astype(int)
val['is_2s'] = val['scenario'].str.contains('2s').astype(int)
val['is_1s'] = val['scenario'].str.contains('1s').astype(int)
val['is_vwap_brk'] = val['scenario'].str.contains('VWAP').astype(int)

# Sector encoding (top sectors get their own flag)
for split_df in [df, train, val]:
    split_df['sector_finance'] = (split_df['sector'] == 'Finance').astype(int)
    split_df['sector_manufacturing'] = (split_df['sector'] == 'Manufacturing').astype(int)
    split_df['sector_services'] = (split_df['sector'] == 'Services').astype(int)

FEATURES = [
    'gap_size_adr', 'rvol', 'cross_bar', 'vix_level', 'spy_return',
    'early_range_15', 'vwap_slope_early', 'pct_adr_used_945', 'pct_adr_at_cross',
    'volume_ratio_at_cross', 'is_gapdn', 'is_2s', 'is_1s', 'is_vwap_brk',
    'sector_finance', 'sector_manufacturing', 'sector_services', 'adr'
]

# Targets: MFE and Drift at different windows
TARGETS = {
    'mfe_30': 'MFE 30min',
    'drift_30': 'Drift 30min',
    'mfe_60': 'MFE 60min',
    'drift_60': 'Drift 60min',
}

# ============================================================
# ML MODEL: XGBoost Regressor
# ============================================================
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: xgboost not installed, using sklearn GradientBoosting", file=sys.stderr)
    from sklearn.ensemble import GradientBoostingRegressor

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("WARNING: shap not installed, skipping SHAP analysis", file=sys.stderr)

# Prepare data
X_train = train[FEATURES].copy()
X_val = val[FEATURES].copy()

# Fill NaN with median
for col in FEATURES:
    median_val = X_train[col].median()
    X_train[col] = X_train[col].fillna(median_val)
    X_val[col] = X_val[col].fillna(median_val)

output_path = os.path.join(RESULTS_DIR, 'behavioral_ml_interactions_v1.txt')
with open(output_path, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("ML INTERACTION MODEL v1 — Durchlauf 2.0\n")
    f.write("XGBoost Regressor auf MFE/Drift (KEINE binaeren Labels)\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"Features ({len(FEATURES)}): {', '.join(FEATURES)}\n")
    f.write(f"Train: {len(X_train)}, Val: {len(X_val)}\n\n")

    # ---- Train models for each target ----
    for target_col, target_name in TARGETS.items():
        f.write("=" * 80 + "\n")
        f.write(f"TARGET: {target_name} ({target_col})\n")
        f.write("=" * 80 + "\n\n")

        y_train = train[target_col].values
        y_val = val[target_col].values

        # Drop NaN targets
        mask_train = ~np.isnan(y_train)
        mask_val = ~np.isnan(y_val)

        X_tr = X_train[mask_train]
        y_tr = y_train[mask_train]
        X_vl = X_val[mask_val]
        y_vl = y_val[mask_val]

        if HAS_XGB:
            model = XGBRegressor(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbosity=0,
            )
            model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)],
                     verbose=False)
        else:
            model = GradientBoostingRegressor(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, random_state=42
            )
            model.fit(X_tr, y_tr)

        # Predictions
        pred_train = model.predict(X_tr)
        pred_val = model.predict(X_vl)

        # R² scores
        from sklearn.metrics import r2_score, mean_absolute_error
        r2_train = r2_score(y_tr, pred_train)
        r2_val = r2_score(y_vl, pred_val)
        mae_train = mean_absolute_error(y_tr, pred_train)
        mae_val = mean_absolute_error(y_vl, pred_val)

        f.write(f"  R² Train: {r2_train:.4f}, R² Val: {r2_val:.4f}\n")
        f.write(f"  MAE Train: {mae_train:.4f}, MAE Val: {mae_val:.4f}\n")
        f.write(f"  Mean Target Train: {np.mean(y_tr):.4f}, Val: {np.mean(y_vl):.4f}\n\n")

        # Feature Importance
        if HAS_XGB:
            importances = model.feature_importances_
        else:
            importances = model.feature_importances_

        imp_df = pd.DataFrame({'feature': FEATURES, 'importance': importances})
        imp_df = imp_df.sort_values('importance', ascending=False)

        f.write("  Feature Importance:\n")
        for _, row in imp_df.iterrows():
            bar = '#' * int(row['importance'] * 100)
            f.write(f"    {row['feature']:<28} {row['importance']:.4f} {bar}\n")
        f.write("\n")

        # SHAP Analysis
        if HAS_SHAP and target_col == 'drift_30':
            print(f"  Computing SHAP values for {target_col}...", file=sys.stderr)
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_vl)

                f.write("  SHAP Mean |Values| (Val):\n")
                shap_mean = np.abs(shap_values).mean(axis=0)
                shap_df = pd.DataFrame({'feature': FEATURES, 'mean_shap': shap_mean})
                shap_df = shap_df.sort_values('mean_shap', ascending=False)
                for _, row in shap_df.iterrows():
                    f.write(f"    {row['feature']:<28} {row['mean_shap']:.4f}\n")
                f.write("\n")

                # Top SHAP interactions
                f.write("  SHAP Interaction Effects (Top Pairs):\n")
                # Approximate interactions via correlation of SHAP values
                shap_corr = np.corrcoef(shap_values.T)
                pairs = []
                for i in range(len(FEATURES)):
                    for j in range(i+1, len(FEATURES)):
                        pairs.append((FEATURES[i], FEATURES[j], abs(shap_corr[i, j])))
                pairs.sort(key=lambda x: x[2], reverse=True)
                for feat1, feat2, corr in pairs[:15]:
                    f.write(f"    {feat1} x {feat2}: corr={corr:.3f}\n")
                f.write("\n")
            except Exception as e:
                f.write(f"  SHAP analysis failed: {str(e)}\n\n")

    # ============================================================
    # CONDITIONAL ANALYSIS: ML-IDENTIFIED SUBGROUPS
    # ============================================================
    f.write("=" * 80 + "\n")
    f.write("CONDITIONAL SUBGROUP ANALYSIS (Val)\n")
    f.write("Kombinationen mit positivem Drift identifizieren\n")
    f.write("=" * 80 + "\n\n")

    # Use val data for conditional analysis
    v = val.copy()

    # --- Best conditions for GapDn (the promising direction) ---
    f.write("--- GapDn Conditional Analysis ---\n\n")
    gapdn_val = v[v['gap_direction'] == 'down']

    # Time of day bins
    f.write("  BY TIME OF DAY:\n")
    time_bins = [
        (15, 30, '9:45-10:00'),
        (30, 60, '10:00-10:30'),
        (60, 90, '10:30-11:00'),
        (90, 120, '11:00-11:30'),
        (120, 240, '11:30-13:30'),
    ]
    for lo, hi, label in time_bins:
        subset = gapdn_val[(gapdn_val['cross_bar'] >= lo) & (gapdn_val['cross_bar'] < hi)]
        if len(subset) < 20:
            continue
        f.write(f"    {label}: N={len(subset)}, MFE30={subset['mfe_30'].mean():.4f}, "
                f"MAE30={subset['mae_30'].mean():.4f}, Drift30={subset['drift_30'].mean():.4f}, "
                f"MFE/MAE={abs(subset['mfe_30'].mean()/subset['mae_30'].mean()):.2f}, "
                f"NextLvl={subset['next_level_reached'].mean()*100:.1f}%\n")
    f.write("\n")

    # Gap Size x Time interaction
    f.write("  GAP SIZE x TIME (GapDn):\n")
    for gap_lo, gap_hi, gap_label in [(0, 1.5, 'SmallGap'), (1.5, 3.0, 'MedGap'), (3.0, 99, 'LargeGap')]:
        for t_lo, t_hi, t_label in [(15, 45, 'Early'), (45, 120, 'Mid'), (120, 390, 'Late')]:
            subset = gapdn_val[
                (gapdn_val['gap_size_adr'] >= gap_lo) & (gapdn_val['gap_size_adr'] < gap_hi) &
                (gapdn_val['cross_bar'] >= t_lo) & (gapdn_val['cross_bar'] < t_hi)
            ]
            if len(subset) < 15:
                continue
            f.write(f"    {gap_label} + {t_label}: N={len(subset)}, Drift30={subset['drift_30'].mean():.4f}, "
                    f"MFE30={subset['mfe_30'].mean():.4f}, NextLvl={subset['next_level_reached'].mean()*100:.1f}%\n")
    f.write("\n")

    # VWAP Slope x Scenario interaction
    f.write("  VWAP SLOPE x SCENARIO (Val, all directions):\n")
    slope_med = v['vwap_slope_early'].abs().median()
    for scen in sorted(v['scenario'].unique()):
        for slope_label, slope_mask in [('Flat', v['vwap_slope_early'].abs() <= slope_med),
                                         ('Trending', v['vwap_slope_early'].abs() > slope_med)]:
            subset = v[(v['scenario'] == scen) & slope_mask]
            if len(subset) < 20:
                continue
            f.write(f"    {scen} + {slope_label}: N={len(subset)}, Drift30={subset['drift_30'].mean():.4f}, "
                    f"MFE30={subset['mfe_30'].mean():.4f}\n")
    f.write("\n")

    # %ADR Used x Time interaction (GapDn)
    f.write("  %ADR USED x TIME (GapDn):\n")
    for adr_lo, adr_hi, adr_label in [(0, 0.5, '<50%ADR'), (0.5, 1.0, '50-100%ADR'), (1.0, 99, '>100%ADR')]:
        for t_lo, t_hi, t_label in [(15, 45, 'Early'), (45, 120, 'Mid'), (120, 390, 'Late')]:
            subset = gapdn_val[
                (gapdn_val['pct_adr_at_cross'] >= adr_lo) & (gapdn_val['pct_adr_at_cross'] < adr_hi) &
                (gapdn_val['cross_bar'] >= t_lo) & (gapdn_val['cross_bar'] < t_hi)
            ]
            if len(subset) < 15:
                continue
            f.write(f"    {adr_label} + {t_label}: N={len(subset)}, Drift30={subset['drift_30'].mean():.4f}, "
                    f"MFE30={subset['mfe_30'].mean():.4f}\n")
    f.write("\n")

    # SPY Return interaction
    f.write("  SPY RETURN x GAP DIRECTION (Val):\n")
    spy_med = v['spy_return'].median()
    for gap_dir in ['down', 'up']:
        for spy_label, spy_mask in [('SPY_Up', v['spy_return'] > 0), ('SPY_Down', v['spy_return'] <= 0)]:
            subset = v[(v['gap_direction'] == gap_dir) & spy_mask]
            if len(subset) < 20:
                continue
            trade = 'Long' if gap_dir == 'down' else 'Short'
            f.write(f"    Gap{gap_dir.capitalize()} ({trade}) + {spy_label}: N={len(subset)}, "
                    f"Drift30={subset['drift_30'].mean():.4f}, MFE30={subset['mfe_30'].mean():.4f}\n")
    f.write("\n")

    # VIX Level interaction
    f.write("  VIX LEVEL x GAP DIRECTION (Val):\n")
    vix_bins = [(0, 15, 'LowVIX<15'), (15, 20, 'MidVIX15-20'), (20, 30, 'HighVIX20-30'), (30, 999, 'ExtremeVIX>30')]
    for gap_dir in ['down', 'up']:
        for vix_lo, vix_hi, vix_label in vix_bins:
            subset = v[(v['gap_direction'] == gap_dir) &
                      (v['vix_level'] >= vix_lo) & (v['vix_level'] < vix_hi)]
            if len(subset) < 15:
                continue
            trade = 'Long' if gap_dir == 'down' else 'Short'
            f.write(f"    Gap{gap_dir.capitalize()} ({trade}) + {vix_label}: N={len(subset)}, "
                    f"Drift30={subset['drift_30'].mean():.4f}, MFE30={subset['mfe_30'].mean():.4f}\n")
    f.write("\n")

    # ============================================================
    # BEST COMBINED FILTERS (GapDn only)
    # ============================================================
    f.write("=" * 80 + "\n")
    f.write("BEST COMBINED FILTERS — GapDn (Val)\n")
    f.write("=" * 80 + "\n\n")

    # Systematically test combinations
    results = []
    for scen in ['GapDn_2s_Fade', 'GapDn_VWAP_Brk', 'GapDn_1s_Fade']:
        base = gapdn_val[gapdn_val['scenario'] == scen]
        if len(base) < 30:
            continue

        for t_lo, t_hi, t_label in [(15, 45, 'Early'), (45, 120, 'Mid')]:
            for slope_label, slope_cond in [('Flat', base['vwap_slope_early'].abs() <= slope_med),
                                             ('Any', pd.Series(True, index=base.index))]:
                for adr_label, adr_lo, adr_hi in [('<75%ADR', 0, 0.75), ('Any', 0, 99)]:
                    for gap_label, g_lo, g_hi in [('SmGap', 0, 2.0), ('AnyGap', 0, 99)]:
                        mask = (
                            (base['cross_bar'] >= t_lo) & (base['cross_bar'] < t_hi) &
                            slope_cond &
                            (base['pct_adr_at_cross'] >= adr_lo) & (base['pct_adr_at_cross'] < adr_hi) &
                            (base['gap_size_adr'] >= g_lo) & (base['gap_size_adr'] < g_hi)
                        )
                        subset = base[mask]
                        if len(subset) < 25:
                            continue

                        drift = subset['drift_30'].mean()
                        mfe = subset['mfe_30'].mean()
                        mae = subset['mae_30'].mean()
                        next_lvl = subset['next_level_reached'].mean()
                        mfe_mae = abs(mfe / mae) if mae != 0 else 0

                        results.append({
                            'scenario': scen,
                            'time': t_label,
                            'slope': slope_label,
                            'adr_used': adr_label,
                            'gap_size': gap_label,
                            'n': len(subset),
                            'drift_30': drift,
                            'mfe_30': mfe,
                            'mae_30': mae,
                            'mfe_mae_ratio': mfe_mae,
                            'next_lvl': next_lvl,
                        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('drift_30', ascending=False)

    f.write(f"{'Scenario':<20} {'Time':>6} {'Slope':>8} {'ADR%':>9} {'Gap':>7} {'N':>5} "
            f"{'Drift30':>8} {'MFE30':>7} {'MAE30':>8} {'MFE/MAE':>7} {'NextLvl':>7}\n")
    f.write("-" * 105 + "\n")

    for _, r in results_df.head(25).iterrows():
        f.write(f"{r['scenario']:<20} {r['time']:>6} {r['slope']:>8} {r['adr_used']:>9} {r['gap_size']:>7} "
                f"{r['n']:>5} {r['drift_30']:>8.4f} {r['mfe_30']:>7.4f} {r['mae_30']:>8.4f} "
                f"{r['mfe_mae_ratio']:>7.2f} {r['next_lvl']*100:>6.1f}%\n")
    f.write("\n")

    # Also show worst combos
    f.write("--- WORST Combined Filters (Bottom 10) ---\n")
    for _, r in results_df.tail(10).iterrows():
        f.write(f"{r['scenario']:<20} {r['time']:>6} {r['slope']:>8} {r['adr_used']:>9} {r['gap_size']:>7} "
                f"{r['n']:>5} {r['drift_30']:>8.4f} {r['mfe_30']:>7.4f} {r['mae_30']:>8.4f} "
                f"{r['mfe_mae_ratio']:>7.2f} {r['next_lvl']*100:>6.1f}%\n")
    f.write("\n")

    # ============================================================
    # SAME FOR GapUp (Short direction)
    # ============================================================
    f.write("=" * 80 + "\n")
    f.write("BEST COMBINED FILTERS — GapUp Short (Val)\n")
    f.write("Negative Drift = gute Short-Seite\n")
    f.write("=" * 80 + "\n\n")

    gapup_val = v[v['gap_direction'] == 'up']

    # For GapUp shorts, we want NEGATIVE drift (price drops = short profits)
    # But our drift is from the "trade direction" perspective (short)
    # So positive drift = good for shorts, negative drift = bad for shorts
    # Wait - in compute_cross_metrics, sign = -1 for shorts
    # So drift_30 POSITIVE = price went DOWN = good for shorts
    # And drift_30 NEGATIVE = price went UP = bad for shorts

    results_up = []
    for scen in ['GapUp_2s_Fade', 'GapUp_VWAP_Brk', 'GapUp_1s_Fade']:
        base = gapup_val[gapup_val['scenario'] == scen]
        if len(base) < 30:
            continue

        for t_lo, t_hi, t_label in [(15, 45, 'Early'), (45, 120, 'Mid')]:
            for slope_label, slope_cond in [('Flat', base['vwap_slope_early'].abs() <= slope_med),
                                             ('Any', pd.Series(True, index=base.index))]:
                for gap_label, g_lo, g_hi in [('SmGap', 0, 2.0), ('AnyGap', 0, 99)]:
                    mask = (
                        (base['cross_bar'] >= t_lo) & (base['cross_bar'] < t_hi) &
                        slope_cond &
                        (base['gap_size_adr'] >= g_lo) & (base['gap_size_adr'] < g_hi)
                    )
                    subset = base[mask]
                    if len(subset) < 25:
                        continue

                    drift = subset['drift_30'].mean()
                    mfe = subset['mfe_30'].mean()
                    mae = subset['mae_30'].mean()
                    next_lvl = subset['next_level_reached'].mean()

                    results_up.append({
                        'scenario': scen,
                        'time': t_label,
                        'slope': slope_label,
                        'gap_size': gap_label,
                        'n': len(subset),
                        'drift_30': drift,
                        'mfe_30': mfe,
                        'mae_30': mae,
                        'next_lvl': next_lvl,
                    })

    if results_up:
        results_up_df = pd.DataFrame(results_up)
        results_up_df = results_up_df.sort_values('drift_30', ascending=False)

        f.write(f"{'Scenario':<20} {'Time':>6} {'Slope':>8} {'Gap':>7} {'N':>5} "
                f"{'Drift30':>8} {'MFE30':>7} {'MAE30':>8} {'NextLvl':>7}\n")
        f.write("-" * 80 + "\n")

        for _, r in results_up_df.head(15).iterrows():
            f.write(f"{r['scenario']:<20} {r['time']:>6} {r['slope']:>8} {r['gap_size']:>7} "
                    f"{r['n']:>5} {r['drift_30']:>8.4f} {r['mfe_30']:>7.4f} {r['mae_30']:>8.4f} "
                    f"{r['next_lvl']*100:>6.1f}%\n")
        f.write("\n")

    # ============================================================
    # REVERSAL PREDICTION: What predicts no-reversal (clean trades)?
    # ============================================================
    f.write("=" * 80 + "\n")
    f.write("REVERSAL PREDICTION: Was unterscheidet cleane von reversal Trades?\n")
    f.write("=" * 80 + "\n\n")

    # For GapDn only
    gapdn_v = val[val['gap_direction'] == 'down']
    clean = gapdn_v[gapdn_v['reversal_10'] == False]
    reversal = gapdn_v[gapdn_v['reversal_10'] == True]

    f.write(f"GapDn Clean (no reversal in 10 bars): N={len(clean)}\n")
    f.write(f"GapDn Reversal: N={len(reversal)}\n\n")

    compare_cols = ['gap_size_adr', 'rvol', 'cross_bar', 'early_range_15',
                    'vwap_slope_early', 'pct_adr_at_cross', 'volume_ratio_at_cross',
                    'vix_level', 'spy_return']

    f.write(f"{'Feature':<28} {'Clean_mean':>10} {'Rev_mean':>10} {'Delta':>8}\n")
    f.write("-" * 60 + "\n")
    for col in compare_cols:
        c_mean = clean[col].mean()
        r_mean = reversal[col].mean()
        delta = c_mean - r_mean
        f.write(f"{col:<28} {c_mean:>10.4f} {r_mean:>10.4f} {delta:>8.4f}\n")
    f.write("\n")

    # Clean trades behavioral profile
    f.write("Clean trades behavioral profile:\n")
    f.write(f"  MFE30: {clean['mfe_30'].mean():.4f} (vs reversal: {reversal['mfe_30'].mean():.4f})\n")
    f.write(f"  Drift30: {clean['drift_30'].mean():.4f} (vs reversal: {reversal['drift_30'].mean():.4f})\n")
    f.write(f"  NextLvl: {clean['next_level_reached'].mean()*100:.1f}% (vs reversal: {reversal['next_level_reached'].mean()*100:.1f}%)\n")
    f.write(f"  MFE60: {clean['mfe_60'].mean():.4f} (vs reversal: {reversal['mfe_60'].mean():.4f})\n")
    f.write(f"  Drift120: {clean['drift_120'].mean():.4f} (vs reversal: {reversal['drift_120'].mean():.4f})\n\n")

    f.write("=" * 80 + "\n")
    f.write("ENDE ML INTERACTION MODEL v1\n")
    f.write("=" * 80 + "\n")

print(f"Ergebnisse geschrieben: {output_path}", file=sys.stderr)
print("Done.", file=sys.stderr)
