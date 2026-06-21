###############################################################################
# ML PIPELINE v2 — VWAP Gapper Analysis (Close-Only Data)
#
# Adapted for actual data format: only close + volume per 1-min bar
# No open/high/low available — all logic uses close prices only
#
# Phase 1: Feature Engineering + Trade Detection on close-cross signals
# Phase 2: XGBoost + LightGBM Training (time-based split)
# Phase 3: Feature Importance + Edge Discovery
# Phase 4: Concrete Setup Ranking (Scenario x Entry x SL x Target)
#
# Run:
#   cd gapper-analysis
#   .\gapper_env\Scripts\python.exe scripts\ml_pipeline.py
###############################################################################

import pandas as pd
import numpy as np
import glob
import json
import os
import sys
import io
import pickle
import warnings
from tqdm import tqdm
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE_DIR = str(PROJECT_ROOT)
VWAP_DIR = os.path.join(BASE_DIR, 'data', 'vwap')
METADATA_PATH = os.path.join(BASE_DIR, 'data', 'metadata', 'metadata_master.parquet')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DOCS_DIR = os.path.join(BASE_DIR, 'docs')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Time-based split (data goes 2021-02 to 2026-02)
TRAIN_END = '2023-12-31'
VAL_END = '2024-12-31'
# Test = 2025+

# Scenarios
SCENARIOS = [
    # (name, gap_dir, level_col, trade_dir, next_level, second_level)
    ('GapUp_2s_Fade', 'up', 'upper_2std', 'short', 'upper_1std', 'vwap'),
    ('GapUp_1s_Fade', 'up', 'upper_1std', 'short', 'vwap', 'lower_1std'),
    ('GapUp_VWAP_Brk', 'up', 'vwap', 'short', 'lower_1std', 'lower_2std'),
    ('GapDn_2s_Fade', 'down', 'lower_2std', 'long', 'lower_1std', 'vwap'),
    ('GapDn_1s_Fade', 'down', 'lower_1std', 'long', 'vwap', 'upper_1std'),
    ('GapDn_VWAP_Brk', 'down', 'vwap', 'long', 'upper_1std', 'upper_2std'),
]

# ============================================================
# LOAD METADATA
# ============================================================
print("=" * 80)
print("VWAP GAPPER ML PIPELINE v2 (Close-Only)")
print("=" * 80)

print("\nLoading metadata...")
meta = pd.read_parquet(METADATA_PATH)
meta['key'] = meta['ticker'] + '_' + meta['date']
meta_lookup = meta.set_index('key').to_dict('index')
print(f"Metadata: {len(meta)} gappers")

dates = sorted(meta['date'].unique())
print(f"Date range: {dates[0]} to {dates[-1]}")
train_n = sum(1 for d in dates if d <= TRAIN_END)
val_n = sum(1 for d in dates if TRAIN_END < d <= VAL_END)
test_n = sum(1 for d in dates if d > VAL_END)
print(f"Train: {train_n} dates (<=  {TRAIN_END})")
print(f"Val:   {val_n} dates ({TRAIN_END} - {VAL_END})")
print(f"Test:  {test_n} dates (> {VAL_END})")


# ============================================================
# HELPER FUNCTIONS (close-only)
# ============================================================

def close_volatility(closes, period=10):
    """Rolling std of 1-min close returns, normalized"""
    n = len(closes)
    vol = np.full(n, np.nan)
    for i in range(period, n):
        rets = np.diff(closes[i-period:i+1]) / closes[i-period:i]
        vol[i] = np.std(rets)
    return vol


def close_momentum(closes, period=5):
    """Close return over last `period` bars"""
    n = len(closes)
    mom = np.full(n, np.nan)
    for i in range(period, n):
        if closes[i - period] != 0:
            mom[i] = (closes[i] - closes[i - period]) / closes[i - period]
    return mom


def volume_ratio(volumes, period=10):
    """Volume of current bar vs rolling average"""
    n = len(volumes)
    ratio = np.full(n, np.nan)
    for i in range(period, n):
        avg = np.mean(volumes[i-period:i])
        ratio[i] = volumes[i] / avg if avg > 0 else 1.0
    return ratio


def vwap_slope(vwaps, period=15):
    """Pct change of VWAP over last `period` bars"""
    n = len(vwaps)
    slope = np.full(n, np.nan)
    for i in range(period, n):
        if vwaps[i - period] != 0:
            slope[i] = (vwaps[i] - vwaps[i - period]) / vwaps[i - period]
    return slope


def consecutive_closes_in_zone(closes, levels, direction, bar_idx):
    """Count how many consecutive bars closed beyond the level before bar_idx"""
    count = 0
    is_short = (direction == 'short')
    for j in range(bar_idx - 1, max(bar_idx - 10, -1), -1):
        if is_short and closes[j] < levels[j]:
            count += 1
        elif not is_short and closes[j] > levels[j]:
            count += 1
        else:
            break
    return count


# ============================================================
# TRADE DETECTION + OUTCOME (close-only)
# ============================================================

def detect_trades(df, level_col, trade_dir, next_level_col, second_level_col, adr_10):
    """
    Detect close-cross signals and simulate trade outcomes.
    Uses only close prices (no high/low available).
    """
    trades = []

    if len(df) < 50 or adr_10 <= 0:
        return trades

    n = len(df)
    closes = df['close'].values
    volumes = df['volume'].values
    times = df['time_et'].values
    levels = df[level_col].values
    vwaps = df['vwap'].values
    stds = df['std_dev'].values
    zscores = df['z_score'].values if 'z_score' in df.columns else np.zeros(n)
    next_levels = df[next_level_col].values
    second_levels = df[second_level_col].values

    # Pre-compute indicators
    vol_10 = close_volatility(closes, 10)
    mom_5 = close_momentum(closes, 5)
    mom_15 = close_momentum(closes, 15)
    vol_rat = volume_ratio(volumes, 10)
    vslope = vwap_slope(vwaps, 15)

    # Running high/low from closes for %ADR
    running_high = np.maximum.accumulate(closes)
    running_low = np.minimum.accumulate(closes)
    pct_adr = np.where(adr_10 > 0, (running_high - running_low) / adr_10, 0)

    # BandWidth (2 * std / adr)
    band_width = np.where(
        (stds > 0) & ~np.isnan(stds),
        2 * stds / adr_10,
        np.nan
    )

    is_short = (trade_dir == 'short')

    # Find market start
    market_start_idx = 0
    for i in range(n):
        if times[i] >= '09:30':
            market_start_idx = i
            break

    # Need at least 20 bars after open for indicators to warm up
    min_bar = market_start_idx + 20
    last_signal_bar = -10

    for i in range(min_bar, n - 5):
        # Cooldown: 5 bars between signals
        if i - last_signal_bar < 5:
            continue

        # Skip if level data is NaN
        if np.isnan(levels[i]) or np.isnan(levels[i-1]):
            continue

        # === SIGNAL: Close crosses level ===
        if is_short:
            # Was above, now below
            signal = (closes[i-1] >= levels[i-1]) and (closes[i] < levels[i])
        else:
            # Was below, now above
            signal = (closes[i-1] <= levels[i-1]) and (closes[i] > levels[i])

        if not signal:
            continue

        last_signal_bar = i
        entry_price = closes[i]
        entry_bar = i
        minutes_since_open = i - market_start_idx

        # === FEATURES at signal time ===
        dist_to_level = abs(closes[i] - levels[i]) / adr_10
        dist_to_vwap = (closes[i] - vwaps[i]) / adr_10 if not np.isnan(vwaps[i]) else 0

        # Check E2: next bar continues in trade direction
        e2_valid = False
        if i + 1 < n:
            if is_short and closes[i+1] < closes[i]:
                e2_valid = True
            elif not is_short and closes[i+1] > closes[i]:
                e2_valid = True

        # Consecutive closes in zone before signal
        consec = consecutive_closes_in_zone(closes, levels, trade_dir, i)

        features = {
            'minutes_since_open': minutes_since_open,
            'time_bucket': 0 if minutes_since_open <= 30 else (1 if minutes_since_open <= 60 else 2),
            'dist_to_level_adr': dist_to_level,
            'dist_to_vwap_adr': dist_to_vwap,
            'pct_adr_used': pct_adr[i],
            'band_width_adr': band_width[i] if not np.isnan(band_width[i]) else 0,
            'z_score': zscores[i] if not np.isnan(zscores[i]) else 0,
            'close_vol_10': vol_10[i] if not np.isnan(vol_10[i]) else 0,
            'momentum_5': mom_5[i] if not np.isnan(mom_5[i]) else 0,
            'momentum_15': mom_15[i] if not np.isnan(mom_15[i]) else 0,
            'volume_ratio': vol_rat[i] if not np.isnan(vol_rat[i]) else 1.0,
            'vwap_slope': vslope[i] if not np.isnan(vslope[i]) else 0,
            'e2_confirmed': int(e2_valid),
            'consec_closes_in_zone': consec,
        }

        # === ENTRY TYPES ===
        # E1: immediate at signal close
        # E2: next bar must confirm (close continues in direction)
        entry_configs = [('E1', entry_bar, entry_price)]
        if e2_valid and i + 1 < n:
            entry_configs.append(('E2', i + 1, closes[i + 1]))

        for entry_type, e_bar, e_price in entry_configs:

            # === SL VARIANTS (all fixed-ADR since we only have close) ===
            sl_configs = {
                'SL3_010': 0.10 * adr_10,
                'SL4_020': 0.20 * adr_10,
                'SL5_035': 0.35 * adr_10,
                'SL6_level': abs(e_price - levels[e_bar]) + 0.05 * adr_10 if not np.isnan(levels[e_bar]) else 0.20 * adr_10,
            }

            for sl_name, sl_dist in sl_configs.items():
                if sl_dist <= 0:
                    continue

                if is_short:
                    sl_price = e_price + sl_dist
                else:
                    sl_price = e_price - sl_dist

                sl_dist_adr = sl_dist / adr_10

                # === TARGET VARIANTS ===
                tgt_configs = {}

                # T1: next VWAP level (dynamic)
                tgt_configs['T1_next'] = 'dynamic_t1'

                # T3: fixed ADR
                if is_short:
                    tgt_configs['T3_025'] = e_price - 0.25 * adr_10
                    tgt_configs['T3_050'] = e_price - 0.50 * adr_10
                else:
                    tgt_configs['T3_025'] = e_price + 0.25 * adr_10
                    tgt_configs['T3_050'] = e_price + 0.50 * adr_10

                # T4: R-multiple targets
                if sl_dist > 0:
                    for mult, label in [(1.5, 'T4_15R'), (2.0, 'T4_2R'), (3.0, 'T4_3R')]:
                        if is_short:
                            tgt_configs[label] = e_price - mult * sl_dist
                        else:
                            tgt_configs[label] = e_price + mult * sl_dist

                for tgt_name, tgt_price in tgt_configs.items():

                    # === SIMULATE TRADE BAR-BY-BAR ===
                    outcome = 'timeout'
                    exit_price = closes[-1]
                    max_favorable = 0.0
                    max_adverse = 0.0
                    bars_held = 0

                    for j in range(e_bar + 1, min(n, e_bar + 241)):
                        bars_held = j - e_bar

                        # Dynamic target
                        curr_tgt = tgt_price
                        if tgt_price == 'dynamic_t1':
                            curr_tgt = next_levels[j] if not np.isnan(next_levels[j]) else next_levels[e_bar]

                        # Track excursion (close-based)
                        if is_short:
                            fav = (e_price - closes[j]) / adr_10
                            adv = (closes[j] - e_price) / adr_10
                        else:
                            fav = (closes[j] - e_price) / adr_10
                            adv = (e_price - closes[j]) / adr_10

                        max_favorable = max(max_favorable, fav)
                        max_adverse = max(max_adverse, adv)

                        # Check SL (close-based)
                        sl_hit = False
                        if is_short and closes[j] >= sl_price:
                            sl_hit = True
                        elif not is_short and closes[j] <= sl_price:
                            sl_hit = True

                        # Check Target (close-based)
                        tgt_hit = False
                        if is_short and closes[j] <= curr_tgt:
                            tgt_hit = True
                        elif not is_short and closes[j] >= curr_tgt:
                            tgt_hit = True

                        if sl_hit and tgt_hit:
                            # Both in same bar — conservative: SL wins
                            outcome = 'sl_hit'
                            exit_price = sl_price
                            break
                        elif tgt_hit:
                            outcome = 'target_hit'
                            exit_price = curr_tgt if curr_tgt != 'dynamic_t1' else next_levels[j]
                            break
                        elif sl_hit:
                            outcome = 'sl_hit'
                            exit_price = sl_price
                            break

                    # If timeout, use last close
                    if outcome == 'timeout':
                        exit_price = closes[min(e_bar + 240, n - 1)]

                    # PnL
                    if is_short:
                        pnl_adr = (e_price - exit_price) / adr_10
                    else:
                        pnl_adr = (exit_price - e_price) / adr_10

                    trade = {**features}
                    trade.update({
                        'entry_type': entry_type,
                        'entry_bar': e_bar,
                        'entry_price': e_price,
                        'sl_method': sl_name,
                        'sl_dist_adr': round(sl_dist_adr, 4),
                        'target_method': tgt_name,
                        'outcome': outcome,
                        'pnl_adr': round(pnl_adr, 4),
                        'max_favorable_adr': round(max_favorable, 4),
                        'max_adverse_adr': round(max_adverse, 4),
                        'bars_held': bars_held,
                    })
                    trades.append(trade)

    return trades


# ============================================================
# PHASE 1: PROCESS ALL FILES
# ============================================================
print("\n" + "=" * 80)
print("PHASE 1: FEATURE ENGINEERING + TRADE DETECTION")
print("=" * 80)

files = glob.glob(f'{VWAP_DIR}/**/*.parquet', recursive=True)
print(f"\n{len(files)} VWAP files found")

all_trades = []
processed = 0
errors = 0
skipped = 0

for filepath in tqdm(files, desc="Processing"):
    try:
        df = pd.read_parquet(filepath)
        if len(df) < 50:
            skipped += 1
            continue

        ticker = df['ticker'].iloc[0]
        date = df['date'].iloc[0]
        key = f"{ticker}_{date}"

        info = meta_lookup.get(key)
        if info is None:
            skipped += 1
            continue

        adr_10 = info.get('adr_10', np.nan)
        if pd.isna(adr_10) or adr_10 <= 0:
            # Fallback: use adr_5 or adr_20
            adr_10 = info.get('adr_5', np.nan)
            if pd.isna(adr_10) or adr_10 <= 0:
                adr_10 = info.get('adr_20', np.nan)
            if pd.isna(adr_10) or adr_10 <= 0:
                skipped += 1
                continue

        gap_dir = info.get('gap_direction', '')
        gap_pct = abs(info.get('gap_pct', 0)) if not pd.isna(info.get('gap_pct', 0)) else 0
        gap_adr = abs(info.get('gap_size_in_adr', 0)) if not pd.isna(info.get('gap_size_in_adr', 0)) else 0
        rvol_30 = info.get('rvol_at_time_30min', np.nan)
        rvol_30 = rvol_30 if not pd.isna(rvol_30) else 1.0
        market_cap = info.get('market_cap', np.nan)
        market_cap = market_cap if not pd.isna(market_cap) else 0

        # Compute ADR%
        first_close = df[df['time_et'] >= '09:30']['close'].iloc[0] if len(df[df['time_et'] >= '09:30']) > 0 else df['close'].iloc[0]
        adr_pct = (adr_10 / first_close * 100) if first_close > 0 else 0

        # Early range from close prices
        market = df[(df['time_et'] >= '09:30') & (df['time_et'] <= '16:00')]
        if len(market) < 30:
            skipped += 1
            continue

        t15 = market[market['time_et'] < '09:45']
        t30 = market[market['time_et'] < '10:00']
        er15 = (t15['close'].max() - t15['close'].min()) / adr_10 if len(t15) > 1 else 0
        er30 = (t30['close'].max() - t30['close'].min()) / adr_10 if len(t30) > 1 else 0

        for s_name, s_gap_dir, level_col, trade_dir, next_lvl, second_lvl in SCENARIOS:
            if gap_dir != s_gap_dir:
                continue

            # Check that level columns exist and have data
            if level_col not in df.columns or next_lvl not in df.columns:
                continue

            trades = detect_trades(df, level_col, trade_dir, next_lvl, second_lvl, adr_10)

            for t in trades:
                t['scenario'] = s_name
                t['ticker'] = ticker
                t['date'] = date
                t['gap_direction'] = gap_dir
                t['gap_pct'] = round(gap_pct, 2)
                t['gap_adr'] = round(gap_adr, 3)
                t['rvol_30'] = round(rvol_30, 2)
                t['adr_pct'] = round(adr_pct, 2)
                t['adr_10'] = round(adr_10, 4)
                t['early_range_15'] = round(er15, 3)
                t['early_range_30'] = round(er30, 3)
                t['log_market_cap'] = round(np.log10(market_cap + 1), 2) if market_cap > 0 else 0

            all_trades.extend(trades)

        processed += 1
    except Exception as e:
        errors += 1
        continue

print(f"\nProcessed: {processed}, Skipped: {skipped}, Errors: {errors}")
print(f"Total trades detected: {len(all_trades):,}")

if len(all_trades) == 0:
    print("ERROR: No trades generated!")
    sys.exit(1)

# ============================================================
# BUILD DATAFRAME + SAVE
# ============================================================
df_trades = pd.DataFrame(all_trades)

# Assign split
df_trades['split'] = 'test'
df_trades.loc[df_trades['date'] <= TRAIN_END, 'split'] = 'train'
df_trades.loc[(df_trades['date'] > TRAIN_END) & (df_trades['date'] <= VAL_END), 'split'] = 'val'

print(f"\nSplit distribution:")
print(df_trades['split'].value_counts().to_string())
print(f"\nScenarios:")
print(df_trades['scenario'].value_counts().to_string())
print(f"\nEntry types:")
print(df_trades['entry_type'].value_counts().to_string())
print(f"\nOutcomes (all):")
print(df_trades['outcome'].value_counts(normalize=True).round(3).to_string())
print(f"\nOutcome by scenario:")
for sc in df_trades['scenario'].unique():
    sub = df_trades[df_trades['scenario'] == sc]
    wr = (sub['outcome'] == 'target_hit').mean()
    avg_pnl = sub['pnl_adr'].mean()
    print(f"  {sc:20s}: N={len(sub):6d}, WinRate={wr:.1%}, AvgPnL={avg_pnl:+.4f} ADR")

# Save intermediate
trades_path = os.path.join(RESULTS_DIR, 'all_trades_v2.parquet')
df_trades.to_parquet(trades_path, index=False)
print(f"\nTrades saved: {trades_path}")


# ============================================================
# PHASE 2: ML TRAINING
# ============================================================
print("\n" + "=" * 80)
print("PHASE 2: ML TRAINING")
print("=" * 80)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, roc_auc_score

# === Feature columns ===
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

# Encode categoricals as integers
scenario_map = {s[0]: i for i, s in enumerate(SCENARIOS)}
df_trades['scenario_id'] = df_trades['scenario'].map(scenario_map)

entry_map = {'E1': 0, 'E2': 1}
df_trades['entry_id'] = df_trades['entry_type'].map(entry_map)

sl_methods_unique = sorted(df_trades['sl_method'].unique())
sl_map = {s: i for i, s in enumerate(sl_methods_unique)}
df_trades['sl_id'] = df_trades['sl_method'].map(sl_map)

tgt_methods_unique = sorted(df_trades['target_method'].unique())
tgt_map = {t: i for i, t in enumerate(tgt_methods_unique)}
df_trades['tgt_id'] = df_trades['target_method'].map(tgt_map)

# Label: target_hit = 1
df_trades['label'] = (df_trades['outcome'] == 'target_hit').astype(int)

all_features = feature_cols + ['scenario_id', 'entry_id', 'sl_id', 'tgt_id']

# NaN handling
for col in all_features:
    if col in df_trades.columns:
        df_trades[col] = df_trades[col].fillna(0).astype(float)

# Split
train = df_trades[df_trades['split'] == 'train'].copy()
val = df_trades[df_trades['split'] == 'val'].copy()
test = df_trades[df_trades['split'] == 'test'].copy()

print(f"\nTrain: {len(train):,} trades (label_mean={train['label'].mean():.3f})")
print(f"Val:   {len(val):,} trades (label_mean={val['label'].mean():.3f})")
print(f"Test:  {len(test):,} trades (label_mean={test['label'].mean():.3f})")

# --- XGBoost ---
print("\nTraining XGBoost...")
xgb_model = XGBClassifier(
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
xgb_model.fit(
    train[all_features], train['label'],
    eval_set=[(val[all_features], val['label'])],
    verbose=100,
)

# --- LightGBM ---
print("\nTraining LightGBM...")
lgb_model = LGBMClassifier(
    n_estimators=800,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=50,
    reg_alpha=0.1,
    reg_lambda=1.0,
    n_jobs=-1,
    random_state=42,
    verbose=-1,
)
lgb_model.fit(
    train[all_features], train['label'],
    eval_set=[(val[all_features], val['label'])],
)

# Save models
with open(os.path.join(MODELS_DIR, 'xgb_v2.pkl'), 'wb') as f:
    pickle.dump(xgb_model, f)
with open(os.path.join(MODELS_DIR, 'lgb_v2.pkl'), 'wb') as f:
    pickle.dump(lgb_model, f)
print("Models saved.")


# ============================================================
# PHASE 3: EVALUATION
# ============================================================
print("\n" + "=" * 80)
print("PHASE 3: EVALUATION")
print("=" * 80)

results_summary = {}

for model_name, model in [('XGBoost', xgb_model), ('LightGBM', lgb_model)]:
    print(f"\n{'='*40}")
    print(f"  {model_name}")
    print(f"{'='*40}")

    for split_name, split_df in [('Validation', val), ('Test', test)]:
        if len(split_df) == 0:
            continue

        probs = model.predict_proba(split_df[all_features])[:, 1]
        preds = (probs >= 0.5).astype(int)

        print(f"\n--- {split_name} ---")
        print(classification_report(split_df['label'], preds, digits=3, zero_division=0))

        auc = roc_auc_score(split_df['label'], probs)
        print(f"AUC: {auc:.4f}")

        results_summary[f'{model_name}_{split_name}_AUC'] = round(auc, 4)

        # Win rate by confidence bucket
        tmp = split_df.copy()
        tmp['prob'] = probs

        print(f"\nWin Rate by Model Confidence ({model_name}, {split_name}):")
        print(f"  {'Conf':>10s}  {'N':>7s}  {'WinRate':>8s}  {'AvgPnL':>8s}  {'MedPnL':>8s}  {'Expectancy':>10s}")
        print(f"  {'-'*10}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}")

        for lo, hi in [(0, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.0)]:
            bucket = tmp[(tmp['prob'] >= lo) & (tmp['prob'] < hi)]
            if len(bucket) < 20:
                continue
            wr = bucket['label'].mean()
            avg_pnl = bucket['pnl_adr'].mean()
            med_pnl = bucket['pnl_adr'].median()
            # Expectancy: avg pnl per trade
            print(f"  {lo:.1f}-{hi:.1f}     {len(bucket):7d}  {wr:>7.1%}  {avg_pnl:>+7.4f}  {med_pnl:>+7.4f}  {avg_pnl:>+9.4f}")


# ============================================================
# PHASE 4: FEATURE IMPORTANCE
# ============================================================
print("\n" + "=" * 80)
print("PHASE 4: FEATURE IMPORTANCE")
print("=" * 80)

for model_name, model in [('XGBoost', xgb_model), ('LightGBM', lgb_model)]:
    importances = model.feature_importances_
    imp_df = pd.DataFrame({
        'feature': all_features,
        'importance': importances
    }).sort_values('importance', ascending=False)

    print(f"\n{model_name} Top 15:")
    for _, row in imp_df.head(15).iterrows():
        bar = '#' * int(row['importance'] * 100)
        print(f"  {row['feature']:25s}  {row['importance']:.4f}  {bar}")

    imp_df.to_csv(os.path.join(RESULTS_DIR, f'feature_importance_{model_name.lower()}_v2.csv'), index=False)


# ============================================================
# PHASE 5: EDGE DISCOVERY — Concrete Setups
# ============================================================
print("\n" + "=" * 80)
print("PHASE 5: EDGE DISCOVERY")
print("=" * 80)

# Use ensemble: average of XGB and LGB probabilities on VALIDATION
val_probs_xgb = xgb_model.predict_proba(val[all_features])[:, 1]
val_probs_lgb = lgb_model.predict_proba(val[all_features])[:, 1]
val['prob_ensemble'] = (val_probs_xgb + val_probs_lgb) / 2

# Also compute for test
test_probs_xgb = xgb_model.predict_proba(test[all_features])[:, 1]
test_probs_lgb = lgb_model.predict_proba(test[all_features])[:, 1]
test['prob_ensemble'] = (test_probs_xgb + test_probs_lgb) / 2

# === 5a: Best setups by Scenario x SL x Target ===
print("\n--- Best Setup Combinations (Validation) ---")
print(f"{'Scenario':>20s} | {'Entry':>5s} | {'SL':>10s} | {'Target':>8s} | {'N':>6s} | {'WR':>6s} | {'AvgPnL':>8s} | {'Expectancy':>10s}")
print("-" * 95)

setup_results = []
for scenario in sorted(df_trades['scenario'].unique()):
    for entry in sorted(df_trades['entry_type'].unique()):
        for sl in sorted(df_trades['sl_method'].unique()):
            for tgt in sorted(df_trades['target_method'].unique()):
                mask = (
                    (val['scenario'] == scenario) &
                    (val['entry_type'] == entry) &
                    (val['sl_method'] == sl) &
                    (val['target_method'] == tgt)
                )
                sub = val[mask]
                if len(sub) < 30:
                    continue

                wr = sub['label'].mean()
                avg_pnl = sub['pnl_adr'].mean()
                n = len(sub)

                setup_results.append({
                    'scenario': scenario,
                    'entry': entry,
                    'sl': sl,
                    'target': tgt,
                    'n_val': n,
                    'win_rate_val': round(wr, 4),
                    'avg_pnl_val': round(avg_pnl, 4),
                })

setup_df = pd.DataFrame(setup_results)
if len(setup_df) > 0:
    # Sort by expectancy (avg_pnl)
    setup_df = setup_df.sort_values('avg_pnl_val', ascending=False)

    # Print top 20 profitable
    print("\nTOP 20 PROFITABLE SETUPS (Validation):")
    for _, row in setup_df.head(20).iterrows():
        print(f"  {row['scenario']:>20s} | {row['entry']:>5s} | {row['sl']:>10s} | {row['target']:>8s} | {row['n_val']:>6d} | {row['win_rate_val']:>5.1%} | {row['avg_pnl_val']:>+7.4f}")

    # Print bottom 10 (worst)
    print("\nBOTTOM 10 WORST SETUPS (Validation):")
    for _, row in setup_df.tail(10).iterrows():
        print(f"  {row['scenario']:>20s} | {row['entry']:>5s} | {row['sl']:>10s} | {row['target']:>8s} | {row['n_val']:>6d} | {row['win_rate_val']:>5.1%} | {row['avg_pnl_val']:>+7.4f}")

    setup_df.to_csv(os.path.join(RESULTS_DIR, 'setup_rankings_val.csv'), index=False)

# === 5b: High-confidence trades (ensemble > 0.6) ===
print("\n--- High-Confidence Trades (Ensemble > 0.6) ---")
for split_name, split_df in [('Validation', val), ('Test', test)]:
    hc = split_df[split_df['prob_ensemble'] >= 0.6]
    if len(hc) < 10:
        print(f"  {split_name}: Only {len(hc)} high-conf trades, skipping")
        continue

    wr = hc['label'].mean()
    avg_pnl = hc['pnl_adr'].mean()
    med_pnl = hc['pnl_adr'].median()
    print(f"  {split_name}: N={len(hc)}, WinRate={wr:.1%}, AvgPnL={avg_pnl:+.4f}, MedPnL={med_pnl:+.4f}")

    # Breakdown by scenario
    for sc in sorted(hc['scenario'].unique()):
        sc_sub = hc[hc['scenario'] == sc]
        if len(sc_sub) < 5:
            continue
        sc_wr = sc_sub['label'].mean()
        sc_pnl = sc_sub['pnl_adr'].mean()
        print(f"    {sc:20s}: N={len(sc_sub):4d}, WR={sc_wr:.1%}, PnL={sc_pnl:+.4f}")

# === 5c: Morning filter analysis ===
print("\n--- Morning Filter Effect ---")
for split_name, split_df in [('Validation', val), ('Test', test)]:
    morning = split_df[split_df['minutes_since_open'] <= 30]
    midday = split_df[(split_df['minutes_since_open'] > 30) & (split_df['minutes_since_open'] <= 120)]
    afternoon = split_df[split_df['minutes_since_open'] > 120]

    print(f"  {split_name}:")
    for label, sub in [('Morning (0-30m)', morning), ('Midday (30-120m)', midday), ('Afternoon (>120m)', afternoon)]:
        if len(sub) < 20:
            continue
        wr = sub['label'].mean()
        pnl = sub['pnl_adr'].mean()
        print(f"    {label:20s}: N={len(sub):6d}, WR={wr:.1%}, PnL={pnl:+.4f}")

# === 5d: Validate top setups on TEST ===
print("\n--- TEST VALIDATION of Top Val Setups ---")
if len(setup_df) > 0:
    top_setups = setup_df.head(10)
    print(f"{'Scenario':>20s} | {'Entry':>5s} | {'SL':>10s} | {'Target':>8s} | {'N_val':>6s} | {'WR_val':>6s} | {'N_test':>6s} | {'WR_test':>7s} | {'PnL_test':>8s}")
    print("-" * 105)

    for _, row in top_setups.iterrows():
        mask_test = (
            (test['scenario'] == row['scenario']) &
            (test['entry_type'] == row['entry']) &
            (test['sl_method'] == row['sl']) &
            (test['target_method'] == row['target'])
        )
        t_sub = test[mask_test]
        if len(t_sub) < 5:
            t_wr = float('nan')
            t_pnl = float('nan')
        else:
            t_wr = t_sub['label'].mean()
            t_pnl = t_sub['pnl_adr'].mean()

        print(f"  {row['scenario']:>20s} | {row['entry']:>5s} | {row['sl']:>10s} | {row['target']:>8s} | {row['n_val']:>6d} | {row['win_rate_val']:>5.1%} | {len(t_sub):>6d} | {t_wr:>6.1%} | {t_pnl:>+7.4f}")


# ============================================================
# SAVE FULL SUMMARY
# ============================================================
summary = {
    'timestamp': datetime.now().isoformat(),
    'pipeline_version': 'v2_close_only',
    'total_trades': len(df_trades),
    'train_size': len(train),
    'val_size': len(val),
    'test_size': len(test),
    'train_win_rate': float(train['label'].mean()),
    'val_win_rate': float(val['label'].mean()),
    'test_win_rate': float(test['label'].mean()),
    'results': results_summary,
}

with open(os.path.join(RESULTS_DIR, 'ml_summary_v2.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 80)
print("PIPELINE COMPLETE")
print(f"Results saved to {RESULTS_DIR}")
print("=" * 80)
