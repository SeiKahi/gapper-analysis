"""
D8.0 Aufgabe 4: 10:00-Reversal Prediction (IS)
================================================
Kann man um 9:35 vorhersagen, ob der Tag um 10:00 dreht?
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
from scipy import stats as scipy_stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
META_DIR = DATA_DIR / "metadata"
RAW_DIR = DATA_DIR / "raw_1min"
RESULTS_DIR = BASE_DIR / "results"
IS_END = "2023-12-31"


def load_1min_data(ticker, date_str):
    path = RAW_DIR / ticker / f"{date_str}.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        if 'time_et' in df.columns:
            df = df.sort_values('time_et').reset_index(drop=True)
        elif 'datetime_et' in df.columns:
            df['time_et'] = pd.to_datetime(df['datetime_et']).dt.strftime('%H:%M')
            df = df.sort_values('time_et').reset_index(drop=True)
        return df
    except Exception:
        return None


def get_price_at_time(bars, target_time):
    row = bars[bars['time_et'] == target_time]
    if len(row) > 0:
        return row.iloc[0]['close']
    return np.nan


def compute_od_with_gap(row):
    od = row.get('od_direction', None)
    if pd.isna(od) or od is None:
        return np.nan
    return od == 'with_gap'


def main():
    meta = pd.read_parquet(META_DIR / "metadata_v8.parquet")
    meta['date_dt'] = pd.to_datetime(meta['date'])
    meta = meta[meta['date_dt'] <= IS_END].copy()

    meta['od_with_gap'] = meta.apply(compute_od_with_gap, axis=1)

    earn = meta['is_earnings'] == True
    non_earn = (~meta['is_earnings']) & (~meta['earnings_unknown'])

    lines = []
    lines.append("=" * 80)
    lines.append("AUFGABE 4: 10:00-REVERSAL PREDICTION (IS)")
    lines.append("=" * 80)

    # Load 1-min data for all gappers and compute reversal features
    print(f"Loading 1-min data for {len(meta)} gappers...", file=sys.stderr)

    results = []
    for idx, row in tqdm(meta.iterrows(), total=len(meta),
                         desc="Computing reversals", file=sys.stderr):
        ticker = row['ticker']
        date_str = str(row['date'])[:10]
        bars = load_1min_data(ticker, date_str)
        if bars is None:
            continue

        open_price = row.get('rth_open', row.get('today_open', np.nan))
        if pd.isna(open_price):
            bar930 = bars[bars['time_et'] == '09:30']
            if len(bar930) > 0:
                open_price = bar930.iloc[0]['open']
            else:
                continue

        close_935 = get_price_at_time(bars, '09:35')
        close_1000 = get_price_at_time(bars, '10:00')

        if pd.isna(close_935) or pd.isna(close_1000) or pd.isna(open_price):
            continue

        adr = row.get('adr_5', row.get('adr_10', np.nan))
        if pd.isna(adr) or adr == 0:
            adr = abs(open_price * 0.05)

        # OD direction (based on close_935 vs open)
        od_bullish = close_935 > open_price
        od_bearish = close_935 < open_price
        if close_935 == open_price:
            continue  # no OD

        od_strength_abs = abs(close_935 - open_price) / adr

        # Reversal definitions
        if od_bullish:
            reversal_full = close_1000 < open_price
            reversal_partial = close_1000 < open_price + 0.5 * (close_935 - open_price)
        else:  # bearish
            reversal_full = close_1000 > open_price
            reversal_partial = close_1000 > open_price - 0.5 * (open_price - close_935)

        # Volume profile (Kerze 1 / mean Kerze 2-5)
        first5 = bars[(bars['time_et'] >= '09:30') & (bars['time_et'] <= '09:34')]
        vol_profile = np.nan
        if len(first5) >= 5:
            v1 = first5.iloc[0]['volume']
            v2to5 = first5.iloc[1:5]['volume'].mean()
            if v2to5 > 0:
                vol_profile = v1 / v2to5

        # Wick ratio
        od_bars = bars[(bars['time_et'] >= '09:30') & (bars['time_et'] <= '09:34')]
        wick_ratio = np.nan
        if len(od_bars) > 0:
            od_high = od_bars['high'].max()
            od_low = od_bars['low'].min()
            od_range = od_high - od_low
            if od_range > 0:
                if od_bullish:
                    wick_ratio = (od_high - close_935) / od_range
                else:
                    wick_ratio = (close_935 - od_low) / od_range

        # Candle 5 vs Candle 1
        c5_vs_c1 = np.nan
        if len(first5) >= 5:
            c5_vs_c1_raw = first5.iloc[4]['close'] - first5.iloc[0]['close']
            # Sign: positive = same direction as OD
            if od_bullish:
                c5_vs_c1 = c5_vs_c1_raw / adr
            else:
                c5_vs_c1 = -c5_vs_c1_raw / adr

        # OD range vs body
        od_body_pct = np.nan
        if len(od_bars) > 0:
            od_high = od_bars['high'].max()
            od_low = od_bars['low'].min()
            od_range = od_high - od_low
            if od_range > 0:
                od_body_pct = abs(close_935 - open_price) / od_range

        # Rest drift (10:00 to close, in OD direction and gap direction)
        close_bars = bars[bars['time_et'] == '15:59']
        rth_close = close_bars.iloc[0]['close'] if len(close_bars) > 0 else np.nan

        rest_drift_od = np.nan
        rest_drift_gap = np.nan
        if not pd.isna(rth_close):
            if od_bullish:
                rest_drift_od = (rth_close - close_1000) / adr
            else:
                rest_drift_od = (close_1000 - rth_close) / adr

            gap_dir = row['gap_direction']
            if gap_dir == 'up':
                rest_drift_gap = (rth_close - close_1000) / adr
            else:
                rest_drift_gap = (close_1000 - rth_close) / adr

        results.append({
            'idx': idx,
            'ticker': ticker,
            'date': date_str,
            'gap_direction': row['gap_direction'],
            'is_earnings': row['is_earnings'],
            'earnings_unknown': row['earnings_unknown'],
            'od_with_gap': row['od_with_gap'],
            'od_bullish': od_bullish,
            'od_strength_abs': od_strength_abs,
            'reversal_full': reversal_full,
            'reversal_partial': reversal_partial,
            'rvol_5': row.get('rvol_5', np.nan),
            'pm_rth5': row.get('pm_rth5', np.nan),
            'gap_size_in_adr': abs(row.get('gap_size_in_adr', np.nan)),
            'vol_profile': vol_profile,
            'wick_ratio': wick_ratio,
            'c5_vs_c1': c5_vs_c1,
            'od_body_pct': od_body_pct,
            'rest_drift_od': rest_drift_od,
            'rest_drift_gap': rest_drift_gap,
            'open_price': open_price,
            'close_935': close_935,
            'close_1000': close_1000,
            'adr': adr,
        })

    res_df = pd.DataFrame(results)
    res_df.to_parquet(RESULTS_DIR / "d8_4_reversal_raw.parquet", index=False)
    print(f"\nComputed reversals for {len(res_df)} gappers", file=sys.stderr)

    # 4b: Basis-Reversal-Rate
    lines.append(f"\n--- 4b: Basis-Reversal-Rate ---")
    lines.append(f"  Total rows with valid data: {len(res_df)}")

    gap_up = res_df['gap_direction'] == 'up'
    gap_dn = res_df['gap_direction'] == 'down'
    od_with = res_df['od_with_gap'] == True
    od_against = res_df['od_with_gap'] == False

    def rev_stats(sub, label):
        n = len(sub)
        if n < 10:
            return f"  {label:<30} | N={n} [INSUFFICIENT]"
        rf = sub['reversal_full'].mean() * 100
        rp = sub['reversal_partial'].mean() * 100
        # Bootstrap CI for reversal_full
        rng = np.random.RandomState(42)
        boots = [rng.choice(sub['reversal_full'].values, n, replace=True).mean()
                 for _ in range(1000)]
        ci_lo = np.percentile(boots, 2.5) * 100
        ci_hi = np.percentile(boots, 97.5) * 100
        tag = " [LOW N]" if n < 20 else ""
        return f"  {label:<30} | Full={rf:5.1f}% [{ci_lo:.1f}-{ci_hi:.1f}] Partial={rp:5.1f}% | N={n}{tag}"

    categories = [
        ("OD with, GapUp", od_with & gap_up),
        ("OD with, GapDn", od_with & gap_dn),
        ("OD against, GapUp", od_against & gap_up),
        ("OD against, GapDn", od_against & gap_dn),
        ("OD stark (>0.5)", res_df['od_strength_abs'] > 0.5),
        ("OD schwach (<0.2)", res_df['od_strength_abs'] < 0.2),
    ]

    # All gappers
    lines.append("\n  ALL GAPPERS:")
    for label, mask in categories:
        lines.append(rev_stats(res_df[mask], label))

    # By earnings
    earn_mask = res_df['is_earnings'] == True
    non_mask = (~res_df['is_earnings']) & (~res_df['earnings_unknown'])

    lines.append("\n  EARNINGS:")
    for label, mask in categories:
        lines.append(rev_stats(res_df[mask & earn_mask], f"E: {label}"))

    lines.append("\n  NON-EARNINGS:")
    for label, mask in categories:
        lines.append(rev_stats(res_df[mask & non_mask], f"NE: {label}"))

    # 4c: Reversal-Praediktoren
    lines.append(f"\n\n--- 4c: Reversal-Praediktoren ---")
    lines.append("  Spearman-Korrelation mit reversal_full:")

    predictors = [
        ('od_strength_abs', 'OD-Staerke'),
        ('rvol_5', 'RVOL_5'),
        ('pm_rth5', 'PM/RTH5'),
        ('gap_size_in_adr', 'Gap in ADR'),
        ('is_earnings', 'is_earnings'),
        ('vol_profile', 'Vol Profile (K1/K2-5)'),
        ('wick_ratio', 'Wick Ratio'),
        ('c5_vs_c1', 'Candle5 vs Candle1'),
        ('od_body_pct', 'OD Body%'),
    ]

    header = f"  {'Prediktor':<25} | {'Spearman r':>12} | {'p-value':>12} | {'Direction':>12}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for pcol, pname in predictors:
        if pcol not in res_df.columns:
            continue
        valid = res_df[[pcol, 'reversal_full']].dropna()
        if len(valid) < 30:
            lines.append(f"  {pname:<25} | {'N/A':>12} | {'N/A':>12} | N<30")
            continue

        if pcol == 'is_earnings':
            x = valid[pcol].astype(float)
        else:
            x = valid[pcol]

        r, p = scipy_stats.spearmanr(x, valid['reversal_full'])
        direction = "more reversal" if r > 0 else "less reversal"
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        lines.append(f"  {pname:<25} | {r:>+12.4f} | {p:>12.4f}{sig:>3} | {direction}")

    # Bucket analysis for top predictors
    lines.append(f"\n\n  Bucket-Analyse fuer Schluessel-Praediktoren:")

    for pcol, pname in [('od_strength_abs', 'OD Staerke'),
                        ('rvol_5', 'RVOL_5'),
                        ('vol_profile', 'Vol Profile'),
                        ('wick_ratio', 'Wick Ratio')]:
        if pcol not in res_df.columns:
            continue
        valid = res_df[res_df[pcol].notna()].copy()
        if len(valid) < 50:
            continue

        lines.append(f"\n  {pname} Buckets:")
        try:
            valid['bucket'] = pd.qcut(valid[pcol], 5, labels=['Q1(low)', 'Q2', 'Q3', 'Q4', 'Q5(high)'],
                                       duplicates='drop')
        except ValueError:
            valid['bucket'] = pd.cut(valid[pcol], 5, labels=False, duplicates='drop')

        for bucket in valid['bucket'].unique():
            if pd.isna(bucket):
                continue
            bsub = valid[valid['bucket'] == bucket]
            if len(bsub) < 10:
                continue
            rf = bsub['reversal_full'].mean() * 100
            rp = bsub['reversal_partial'].mean() * 100
            med_val = bsub[pcol].median()
            lines.append(f"    {str(bucket):<12} (med={med_val:.3f}) | Full={rf:5.1f}% Partial={rp:5.1f}% | N={len(bsub)}")

    # 4d: Multivariate Analyse
    lines.append(f"\n\n--- 4d: Multivariate Analyse ---")

    feature_cols = ['od_strength_abs', 'rvol_5', 'pm_rth5', 'gap_size_in_adr',
                    'vol_profile', 'wick_ratio', 'od_body_pct']
    # Add is_earnings as numeric
    res_df['is_earnings_num'] = res_df['is_earnings'].astype(float)
    feature_cols_with_earn = feature_cols + ['is_earnings_num']

    valid = res_df[feature_cols_with_earn + ['reversal_full']].dropna()
    lines.append(f"  Valid rows for regression: {len(valid)}")

    if len(valid) >= 100:
        X = valid[feature_cols_with_earn].values
        y = valid['reversal_full'].astype(int).values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Logistic Regression
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_scaled, y)
        y_prob = lr.predict_proba(X_scaled)[:, 1]
        auc = roc_auc_score(y, y_prob)

        lines.append(f"\n  Logistische Regression:")
        lines.append(f"    AUC: {auc:.4f}")
        lines.append(f"    Pseudo-R2 (McFadden): {1 - lr.score(X_scaled, y) / np.log(y.mean()) * np.log(1-y.mean()):.4f}" if y.mean() > 0 and y.mean() < 1 else "    Pseudo-R2: N/A")

        lines.append(f"\n    Koeffizienten (standardisiert):")
        header = f"    {'Feature':<25} | {'Coef':>10} | {'Importance':>12}"
        lines.append(header)
        for fname, coef in sorted(zip(feature_cols_with_earn, lr.coef_[0]),
                                   key=lambda x: abs(x[1]), reverse=True):
            lines.append(f"    {fname:<25} | {coef:>+10.4f} | {'***' if abs(coef) > 0.1 else ''}")

        # Random Forest
        rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42,
                                    min_samples_leaf=20)
        rf.fit(X_scaled, y)
        y_prob_rf = rf.predict_proba(X_scaled)[:, 1]
        auc_rf = roc_auc_score(y, y_prob_rf)

        lines.append(f"\n  Random Forest (Validation only):")
        lines.append(f"    AUC: {auc_rf:.4f}")
        lines.append(f"\n    Feature Importance:")
        for fname, imp in sorted(zip(feature_cols_with_earn, rf.feature_importances_),
                                  key=lambda x: x[1], reverse=True):
            lines.append(f"    {fname:<25} | {imp:.4f}")

        # Compare top features
        lr_top = sorted(zip(feature_cols_with_earn, np.abs(lr.coef_[0])),
                        key=lambda x: x[1], reverse=True)[:3]
        rf_top = sorted(zip(feature_cols_with_earn, rf.feature_importances_),
                        key=lambda x: x[1], reverse=True)[:3]
        lines.append(f"\n    LR Top-3: {[x[0] for x in lr_top]}")
        lines.append(f"    RF Top-3: {[x[0] for x in rf_top]}")

    # 4e: Reversal-Buckets basierend auf Top-Praediktoren
    lines.append(f"\n\n--- 4e: Reversal-Buckets (Top Praediktoren) ---")

    # Use od_strength_abs x is_earnings x wick_ratio
    if 'wick_ratio' in res_df.columns:
        valid = res_df[res_df['wick_ratio'].notna() & res_df['od_strength_abs'].notna()].copy()

        # OD strength: low/high
        od_med = valid['od_strength_abs'].median()
        valid['od_bucket'] = np.where(valid['od_strength_abs'] > od_med, 'OD_HI', 'OD_LO')

        # Wick ratio: low/high
        wick_med = valid['wick_ratio'].median()
        valid['wick_bucket'] = np.where(valid['wick_ratio'] > wick_med, 'WICK_HI', 'WICK_LO')

        # Earnings
        valid['earn_bucket'] = np.where(valid['is_earnings'], 'EARN', 'NON_E')

        header = f"  {'OD':>6} {'Wick':>8} {'Earn':>6} | {'Rev_Full%':>10} {'Rev_Part%':>10} | {'N':>5}"
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))

        for od_b in ['OD_LO', 'OD_HI']:
            for wick_b in ['WICK_LO', 'WICK_HI']:
                for earn_b in ['EARN', 'NON_E']:
                    sub = valid[(valid['od_bucket'] == od_b) &
                                (valid['wick_bucket'] == wick_b) &
                                (valid['earn_bucket'] == earn_b)]
                    n = len(sub)
                    if n < 10:
                        lines.append(f"  {od_b:>6} {wick_b:>8} {earn_b:>6} | {'N/A':>10} {'N/A':>10} | {n:>5} [LOW]")
                        continue
                    rf = sub['reversal_full'].mean() * 100
                    rp = sub['reversal_partial'].mean() * 100
                    lines.append(f"  {od_b:>6} {wick_b:>8} {earn_b:>6} | {rf:>9.1f}% {rp:>9.1f}% | {n:>5}")

    text = "\n".join(lines)
    out_file = RESULTS_DIR / "d8_4_reversal_predict.txt"
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
