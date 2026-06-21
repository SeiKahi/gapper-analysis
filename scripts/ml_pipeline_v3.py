###############################################################################
# ML PIPELINE v3 — VWAP Gapper Analysis (Close-Only Data)
#
# Corrected data split per CLAUDE.md:
#   - Only data 2021-02-21 to 2023-12-31
#   - Train: 2021-02-21 to 2022-12-31 (~3,100 Gapper)
#   - Validation: 2023-01-01 to 2023-12-31 (~1,200 Gapper)
#   - 2024+ data is LOCKED — not loaded
#
# Close-only: VWAP files have only close + volume (no open/high/low)
#
# Run:
#   .\gapper_env\Scripts\python.exe scripts\ml_pipeline_v3.py
###############################################################################

import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import glob
import json
import pickle
import warnings
from tqdm import tqdm
from datetime import datetime

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

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Data boundaries (CLAUDE.md rules)
DATA_START = '2021-02-21'
DATA_END = '2023-12-31'
TRAIN_END = '2022-12-31'
# Val = 2023-01-01 to 2023-12-31

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
# HELPER FUNCTIONS (close-only)
# ============================================================

def close_volatility(closes, period=10):
    """Rolling std of 1-min close returns"""
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
    """Count consecutive bars closed beyond the level before bar_idx"""
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
            signal = (closes[i-1] >= levels[i-1]) and (closes[i] < levels[i])
        else:
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
        entry_configs = [('E1', entry_bar, entry_price)]
        if e2_valid and i + 1 < n:
            entry_configs.append(('E2', i + 1, closes[i + 1]))

        for entry_type, e_bar, e_price in entry_configs:

            # === SL VARIANTS ===
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
# LOAD METADATA (filtered per CLAUDE.md)
# ============================================================
print("Loading metadata...", file=sys.stderr)
meta = pd.read_parquet(METADATA_PATH)
# STRICT: only use data from 2021-02-21 to 2023-12-31
meta = meta[(meta['date'] >= DATA_START) & (meta['date'] <= DATA_END)]
meta['key'] = meta['ticker'] + '_' + meta['date']
meta_lookup = meta.set_index('key').to_dict('index')

# Build allowed keys set for fast filtering of VWAP files
allowed_keys = set(meta['key'].values)
print(f"Metadata after filter: {len(meta)} gappers ({DATA_START} to {DATA_END})", file=sys.stderr)

dates = sorted(meta['date'].unique())
train_dates = [d for d in dates if d <= TRAIN_END]
val_dates = [d for d in dates if d > TRAIN_END]
print(f"Train dates: {len(train_dates)} ({train_dates[0]} to {train_dates[-1]})", file=sys.stderr)
print(f"Val dates: {len(val_dates)} ({val_dates[0]} to {val_dates[-1]})", file=sys.stderr)


# ============================================================
# PHASE 1: PROCESS ALL FILES
# ============================================================
print("\nPHASE 1: Trade Detection...", file=sys.stderr)

files = glob.glob(f'{VWAP_DIR}/**/*.parquet', recursive=True)
print(f"{len(files)} total VWAP files found (will filter to allowed dates)", file=sys.stderr)

all_trades = []
processed = 0
errors = 0
skipped = 0
skipped_date = 0

for filepath in tqdm(files, desc="Processing", file=sys.stderr):
    try:
        df = pd.read_parquet(filepath)
        if len(df) < 50:
            skipped += 1
            continue

        ticker = df['ticker'].iloc[0] if 'ticker' in df.columns else ''
        date = df['date'].iloc[0] if 'date' in df.columns else ''
        key = f"{ticker}_{date}"

        # Skip if not in allowed date range
        if key not in allowed_keys:
            skipped_date += 1
            continue

        info = meta_lookup.get(key)
        if info is None:
            skipped += 1
            continue

        adr_10 = info.get('adr_10', np.nan)
        if pd.isna(adr_10) or adr_10 <= 0:
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
        market_df = df[df['time_et'] >= '09:30']
        first_close = market_df['close'].iloc[0] if len(market_df) > 0 else df['close'].iloc[0]
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

print(f"\nProcessed: {processed}, Skipped (date filter): {skipped_date}, Skipped (other): {skipped}, Errors: {errors}", file=sys.stderr)
print(f"Total trades detected: {len(all_trades):,}", file=sys.stderr)

if len(all_trades) == 0:
    print("ERROR: No trades generated!", file=sys.stderr)
    sys.exit(1)

# ============================================================
# BUILD DATAFRAME + SAVE
# ============================================================
df_trades = pd.DataFrame(all_trades)

# Assign split
df_trades['split'] = 'val'  # default
df_trades.loc[df_trades['date'] <= TRAIN_END, 'split'] = 'train'

print(f"\nSplit distribution:", file=sys.stderr)
print(df_trades['split'].value_counts().to_string(), file=sys.stderr)

# Save intermediate trades
trades_path = os.path.join(RESULTS_DIR, 'all_trades_v3.parquet')
df_trades.to_parquet(trades_path, index=False)
print(f"Trades saved: {trades_path}", file=sys.stderr)


# ============================================================
# PHASE 2: ML TRAINING
# ============================================================
print("\nPHASE 2: ML Training...", file=sys.stderr)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Feature columns
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

# Encode categoricals
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

# Label
df_trades['label'] = (df_trades['outcome'] == 'target_hit').astype(int)

all_features = feature_cols + ['scenario_id', 'entry_id', 'sl_id', 'tgt_id']

# NaN handling
for col in all_features:
    if col in df_trades.columns:
        df_trades[col] = df_trades[col].fillna(0).astype(float)

# Split
train = df_trades[df_trades['split'] == 'train'].copy()
val = df_trades[df_trades['split'] == 'val'].copy()

print(f"Train: {len(train):,} trades (label_mean={train['label'].mean():.3f})", file=sys.stderr)
print(f"Val:   {len(val):,} trades (label_mean={val['label'].mean():.3f})", file=sys.stderr)

# --- XGBoost ---
print("Training XGBoost...", file=sys.stderr)
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
print("Training LightGBM...", file=sys.stderr)
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
with open(os.path.join(MODELS_DIR, 'xgb_v3.pkl'), 'wb') as f:
    pickle.dump(xgb_model, f)
with open(os.path.join(MODELS_DIR, 'lgb_v3.pkl'), 'wb') as f:
    pickle.dump(lgb_model, f)
print("Models saved.", file=sys.stderr)


# ============================================================
# PHASE 3-5: EVALUATION + FEATURE IMPORTANCE + EDGE DISCOVERY
# ============================================================
# All results written to .txt file (per CLAUDE.md)

output_path = os.path.join(RESULTS_DIR, 'ml_pipeline_v3_results.txt')

with open(output_path, 'w', encoding='utf-8') as out:
    out.write("=" * 80 + "\n")
    out.write("ML PIPELINE v3 — VWAP Gapper Analysis (Close-Only)\n")
    out.write(f"Run: {datetime.now().isoformat()}\n")
    out.write(f"Data: {DATA_START} to {DATA_END}\n")
    out.write(f"Train: <= {TRAIN_END}, Val: > {TRAIN_END}\n")
    out.write("=" * 80 + "\n\n")

    # --- Dataset Stats ---
    out.write("=" * 60 + "\n")
    out.write("DATASET OVERVIEW\n")
    out.write("=" * 60 + "\n")
    out.write(f"Metadata gappers (after filter): {len(meta)}\n")
    out.write(f"Files processed: {processed}\n")
    out.write(f"Files skipped (date filter): {skipped_date}\n")
    out.write(f"Files skipped (other): {skipped}\n")
    out.write(f"Files with errors: {errors}\n")
    out.write(f"Total trades: {len(df_trades):,}\n")
    out.write(f"Train trades: {len(train):,} (label_mean={train['label'].mean():.3f})\n")
    out.write(f"Val trades:   {len(val):,} (label_mean={val['label'].mean():.3f})\n\n")

    out.write("Split distribution:\n")
    out.write(df_trades['split'].value_counts().to_string() + "\n\n")

    out.write("Scenarios:\n")
    out.write(df_trades['scenario'].value_counts().to_string() + "\n\n")

    out.write("Entry types:\n")
    out.write(df_trades['entry_type'].value_counts().to_string() + "\n\n")

    out.write("Outcomes (all):\n")
    out.write(df_trades['outcome'].value_counts(normalize=True).round(3).to_string() + "\n\n")

    out.write("Outcome by scenario:\n")
    for sc in sorted(df_trades['scenario'].unique()):
        sub = df_trades[df_trades['scenario'] == sc]
        wr = (sub['outcome'] == 'target_hit').mean()
        avg_pnl = sub['pnl_adr'].mean()
        out.write(f"  {sc:20s}: N={len(sub):6d}, WinRate={wr:.1%}, AvgPnL={avg_pnl:+.4f} ADR\n")
    out.write("\n")

    # --- PHASE 3: EVALUATION ---
    out.write("=" * 60 + "\n")
    out.write("PHASE 3: MODEL EVALUATION\n")
    out.write("=" * 60 + "\n\n")

    results_summary = {}

    for model_name, model in [('XGBoost', xgb_model), ('LightGBM', lgb_model)]:
        out.write(f"--- {model_name} ---\n")

        for split_name, split_df in [('Train', train), ('Validation', val)]:
            if len(split_df) == 0:
                continue

            probs = model.predict_proba(split_df[all_features])[:, 1]
            preds = (probs >= 0.5).astype(int)

            out.write(f"\n{split_name}:\n")
            report = classification_report(split_df['label'], preds, digits=3, zero_division=0)
            out.write(report + "\n")

            auc = roc_auc_score(split_df['label'], probs)
            out.write(f"AUC: {auc:.4f}\n")
            results_summary[f'{model_name}_{split_name}_AUC'] = round(auc, 4)

            # Win rate by confidence bucket
            tmp = split_df.copy()
            tmp['prob'] = probs

            out.write(f"\nWin Rate by Model Confidence ({model_name}, {split_name}):\n")
            out.write(f"  {'Conf':>10s}  {'N':>7s}  {'WinRate':>8s}  {'AvgPnL':>8s}  {'MedPnL':>8s}\n")
            out.write(f"  {'-'*10}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*8}\n")

            for lo, hi in [(0, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
                           (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.0)]:
                bucket = tmp[(tmp['prob'] >= lo) & (tmp['prob'] < hi)]
                if len(bucket) < 20:
                    continue
                wr = bucket['label'].mean()
                avg_pnl = bucket['pnl_adr'].mean()
                med_pnl = bucket['pnl_adr'].median()
                out.write(f"  {lo:.1f}-{hi:.1f}     {len(bucket):7d}  {wr:>7.1%}  {avg_pnl:>+7.4f}  {med_pnl:>+7.4f}\n")

        out.write("\n")

    # --- PHASE 4: FEATURE IMPORTANCE ---
    out.write("=" * 60 + "\n")
    out.write("PHASE 4: FEATURE IMPORTANCE\n")
    out.write("=" * 60 + "\n\n")

    for model_name, model in [('XGBoost', xgb_model), ('LightGBM', lgb_model)]:
        importances = model.feature_importances_
        imp_df = pd.DataFrame({
            'feature': all_features,
            'importance': importances
        }).sort_values('importance', ascending=False)

        out.write(f"{model_name} Feature Importance (Top 15):\n")
        for _, row in imp_df.head(15).iterrows():
            bar = '#' * int(row['importance'] * 100)
            out.write(f"  {row['feature']:25s}  {row['importance']:.4f}  {bar}\n")
        out.write("\n")

        imp_df.to_csv(os.path.join(RESULTS_DIR, f'feature_importance_{model_name.lower()}_v3.csv'), index=False)

    # --- PHASE 5: EDGE DISCOVERY ---
    out.write("=" * 60 + "\n")
    out.write("PHASE 5: EDGE DISCOVERY\n")
    out.write("=" * 60 + "\n\n")

    # Ensemble on validation
    val_probs_xgb = xgb_model.predict_proba(val[all_features])[:, 1]
    val_probs_lgb = lgb_model.predict_proba(val[all_features])[:, 1]
    val['prob_ensemble'] = (val_probs_xgb + val_probs_lgb) / 2

    # 5a: Best setups by Scenario x Entry x SL x Target (VALIDATION)
    out.write("--- Best Setup Combinations (Validation) ---\n\n")

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
                    if len(sub) < 20:
                        continue

                    wr = sub['label'].mean()
                    avg_pnl = sub['pnl_adr'].mean()
                    med_pnl = sub['pnl_adr'].median()
                    n = len(sub)

                    setup_results.append({
                        'scenario': scenario,
                        'entry': entry,
                        'sl': sl,
                        'target': tgt,
                        'n_val': n,
                        'win_rate_val': round(wr, 4),
                        'avg_pnl_val': round(avg_pnl, 4),
                        'med_pnl_val': round(med_pnl, 4),
                    })

    setup_df = pd.DataFrame(setup_results)
    if len(setup_df) > 0:
        setup_df = setup_df.sort_values('avg_pnl_val', ascending=False)

        out.write("TOP 25 PROFITABLE SETUPS (Validation, sorted by AvgPnL):\n")
        out.write(f"  {'Scenario':>20s} | {'Entry':>5s} | {'SL':>10s} | {'Target':>8s} | {'N':>5s} | {'WR':>6s} | {'AvgPnL':>8s} | {'MedPnL':>8s}\n")
        out.write("  " + "-" * 90 + "\n")
        for _, row in setup_df.head(25).iterrows():
            out.write(f"  {row['scenario']:>20s} | {row['entry']:>5s} | {row['sl']:>10s} | {row['target']:>8s} | {row['n_val']:>5d} | {row['win_rate_val']:>5.1%} | {row['avg_pnl_val']:>+7.4f} | {row['med_pnl_val']:>+7.4f}\n")

        out.write("\nBOTTOM 10 WORST SETUPS (Validation):\n")
        for _, row in setup_df.tail(10).iterrows():
            out.write(f"  {row['scenario']:>20s} | {row['entry']:>5s} | {row['sl']:>10s} | {row['target']:>8s} | {row['n_val']:>5d} | {row['win_rate_val']:>5.1%} | {row['avg_pnl_val']:>+7.4f} | {row['med_pnl_val']:>+7.4f}\n")

        setup_df.to_csv(os.path.join(RESULTS_DIR, 'setup_rankings_v3_val.csv'), index=False)
        out.write("\n")

    # 5b: High-confidence trades (ensemble > 0.6)
    out.write("--- High-Confidence Trades (Ensemble > 0.6) on Validation ---\n")
    hc = val[val['prob_ensemble'] >= 0.6]
    if len(hc) >= 10:
        wr = hc['label'].mean()
        avg_pnl = hc['pnl_adr'].mean()
        med_pnl = hc['pnl_adr'].median()
        out.write(f"  N={len(hc)}, WinRate={wr:.1%}, AvgPnL={avg_pnl:+.4f}, MedPnL={med_pnl:+.4f}\n\n")

        out.write("  Breakdown by scenario:\n")
        for sc in sorted(hc['scenario'].unique()):
            sc_sub = hc[hc['scenario'] == sc]
            if len(sc_sub) < 5:
                continue
            sc_wr = sc_sub['label'].mean()
            sc_pnl = sc_sub['pnl_adr'].mean()
            out.write(f"    {sc:20s}: N={len(sc_sub):4d}, WR={sc_wr:.1%}, PnL={sc_pnl:+.4f}\n")

        out.write("\n  Breakdown by entry:\n")
        for et in sorted(hc['entry_type'].unique()):
            et_sub = hc[hc['entry_type'] == et]
            if len(et_sub) < 5:
                continue
            et_wr = et_sub['label'].mean()
            et_pnl = et_sub['pnl_adr'].mean()
            out.write(f"    {et:10s}: N={len(et_sub):4d}, WR={et_wr:.1%}, PnL={et_pnl:+.4f}\n")

        out.write("\n  Breakdown by SL method:\n")
        for sl in sorted(hc['sl_method'].unique()):
            sl_sub = hc[hc['sl_method'] == sl]
            if len(sl_sub) < 5:
                continue
            sl_wr = sl_sub['label'].mean()
            sl_pnl = sl_sub['pnl_adr'].mean()
            out.write(f"    {sl:12s}: N={len(sl_sub):4d}, WR={sl_wr:.1%}, PnL={sl_pnl:+.4f}\n")

        out.write("\n  Breakdown by target method:\n")
        for tg in sorted(hc['target_method'].unique()):
            tg_sub = hc[hc['target_method'] == tg]
            if len(tg_sub) < 5:
                continue
            tg_wr = tg_sub['label'].mean()
            tg_pnl = tg_sub['pnl_adr'].mean()
            out.write(f"    {tg:10s}: N={len(tg_sub):4d}, WR={tg_wr:.1%}, PnL={tg_pnl:+.4f}\n")
    else:
        out.write(f"  Only {len(hc)} high-confidence trades found (threshold may be too high)\n")
    out.write("\n")

    # 5c: Morning filter analysis
    out.write("--- Morning Filter Effect (Validation) ---\n")
    morning = val[val['minutes_since_open'] <= 30]
    midday = val[(val['minutes_since_open'] > 30) & (val['minutes_since_open'] <= 120)]
    afternoon = val[val['minutes_since_open'] > 120]

    for label, sub in [('Morning (0-30m)', morning), ('Midday (30-120m)', midday), ('Afternoon (>120m)', afternoon)]:
        if len(sub) < 20:
            continue
        wr = sub['label'].mean()
        pnl = sub['pnl_adr'].mean()
        med = sub['pnl_adr'].median()
        out.write(f"  {label:20s}: N={len(sub):6d}, WR={wr:.1%}, AvgPnL={pnl:+.4f}, MedPnL={med:+.4f}\n")
    out.write("\n")

    # 5d: Scenario x Time interaction
    out.write("--- Scenario x Time Interaction (Validation) ---\n")
    for sc in sorted(val['scenario'].unique()):
        sc_sub = val[val['scenario'] == sc]
        morn = sc_sub[sc_sub['minutes_since_open'] <= 30]
        rest = sc_sub[sc_sub['minutes_since_open'] > 30]
        if len(morn) >= 20 and len(rest) >= 20:
            morn_wr = morn['label'].mean()
            morn_pnl = morn['pnl_adr'].mean()
            rest_wr = rest['label'].mean()
            rest_pnl = rest['pnl_adr'].mean()
            out.write(f"  {sc:20s}: Morning N={len(morn):4d} WR={morn_wr:.1%} PnL={morn_pnl:+.4f} | Rest N={len(rest):4d} WR={rest_wr:.1%} PnL={rest_pnl:+.4f}\n")
    out.write("\n")

    # 5e: %ADR Used filter analysis
    out.write("--- %ADR Used Filter (Validation) ---\n")
    for lo, hi, label in [(0, 0.3, '<30%'), (0.3, 0.5, '30-50%'), (0.5, 0.75, '50-75%'), (0.75, 2.0, '>75%')]:
        sub = val[(val['pct_adr_used'] >= lo) & (val['pct_adr_used'] < hi)]
        if len(sub) < 20:
            continue
        wr = sub['label'].mean()
        pnl = sub['pnl_adr'].mean()
        out.write(f"  {label:10s}: N={len(sub):6d}, WR={wr:.1%}, AvgPnL={pnl:+.4f}\n")
    out.write("\n")

    # 5f: Train vs Val consistency check
    out.write("=" * 60 + "\n")
    out.write("OVERFITTING CHECK: Train vs Val\n")
    out.write("=" * 60 + "\n")
    train_wr = train['label'].mean()
    val_wr = val['label'].mean()
    out.write(f"  Overall WR: Train={train_wr:.1%}, Val={val_wr:.1%}\n")

    for sc in sorted(df_trades['scenario'].unique()):
        tr_sub = train[train['scenario'] == sc]
        va_sub = val[val['scenario'] == sc]
        if len(tr_sub) >= 20 and len(va_sub) >= 20:
            tr_wr = tr_sub['label'].mean()
            va_wr = va_sub['label'].mean()
            tr_pnl = tr_sub['pnl_adr'].mean()
            va_pnl = va_sub['pnl_adr'].mean()
            out.write(f"  {sc:20s}: Train WR={tr_wr:.1%} PnL={tr_pnl:+.4f} | Val WR={va_wr:.1%} PnL={va_pnl:+.4f}\n")
    out.write("\n")

    # Summary JSON
    summary = {
        'timestamp': datetime.now().isoformat(),
        'pipeline_version': 'v3_close_only',
        'data_range': f'{DATA_START} to {DATA_END}',
        'train_end': TRAIN_END,
        'total_trades': len(df_trades),
        'train_size': len(train),
        'val_size': len(val),
        'train_win_rate': float(train['label'].mean()),
        'val_win_rate': float(val['label'].mean()),
        'results': results_summary,
    }

    with open(os.path.join(RESULTS_DIR, 'ml_summary_v3.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    out.write("=" * 60 + "\n")
    out.write("PIPELINE COMPLETE\n")
    out.write(f"Results file: {output_path}\n")
    out.write(f"Trades parquet: {trades_path}\n")
    out.write(f"Models: {MODELS_DIR}/xgb_v3.pkl, lgb_v3.pkl\n")
    out.write("=" * 60 + "\n")

print(f"\nDone! Results written to: {output_path}", file=sys.stderr)
