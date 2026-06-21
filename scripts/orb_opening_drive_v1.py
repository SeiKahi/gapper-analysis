###############################################################################
# ORB + OPENING DRIVE v1 — Durchlauf 3.0
#
# Testet: Opening Range Breakout (5min/15min) + Opening Drive Direction
# Nutzt raw_1min OHLC-Daten. Keine Signale vor 9:45 ET.
#
# Hypothese: Opening Drive Richtung (erste 15min) sagt Rest-of-Day vorher.
# Wenn OD AGAINST Gap → Reversal. Wenn OD WITH Gap → Continuation.
# CloseVsOpen Spread: ~1.0-1.3 ADR zwischen Gruppen (aus Metadata).
#
# Strategie: Entry bei Break der 15min-Range NACH 9:45 in OD-Richtung.
# Kein festes SL/Target — beobachte Verhalten (behavioral analysis).
#
# Run: .\gapper_env\Scripts\python.exe scripts\orb_opening_drive_v1.py
###############################################################################

import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import glob
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE_DIR = str(PROJECT_ROOT)
RAW_1MIN_DIR = os.path.join(BASE_DIR, 'data', 'raw_1min')
METADATA_PATH = os.path.join(BASE_DIR, 'data', 'metadata', 'metadata_master.parquet')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

DATA_START = '2021-02-21'
DATA_END = '2023-12-31'
TRAIN_END = '2022-12-31'


def load_raw_1min(ticker, date_str):
    """Load raw 1min OHLCV for a ticker/date."""
    path = os.path.join(RAW_1MIN_DIR, ticker, f"{date_str}.parquet")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_parquet(path)
        # Filter to regular trading hours
        if 'session' in df.columns:
            df = df[df['session'] == 'rth'].reset_index(drop=True)
        elif 'time_et' in df.columns:
            df = df[(df['time_et'] >= '09:30') & (df['time_et'] <= '15:59')].reset_index(drop=True)
        return df
    except:
        return None


def compute_opening_range(df, period_bars):
    """Compute opening range (high/low) over first N bars."""
    if len(df) < period_bars:
        return None, None
    subset = df.iloc[:period_bars]
    return subset['high'].max(), subset['low'].min()


def compute_opening_drive(df, period_bars=15):
    """
    Compute opening drive direction and strength.
    Uses first 15 bars (9:30-9:45).
    Returns: direction ('up'/'down'), strength (move in ADR), od_close
    """
    if len(df) < period_bars:
        return None, None, None

    first_open = df.iloc[0]['open']
    od_close = df.iloc[period_bars - 1]['close']

    if first_open == 0:
        return None, None, None

    move = (od_close - first_open) / first_open
    direction = 'up' if od_close > first_open else 'down'
    return direction, move, od_close


def main():
    # Load metadata
    print("Loading metadata...", file=sys.stderr)
    meta = pd.read_parquet(METADATA_PATH)
    meta['date'] = pd.to_datetime(meta['date']).dt.strftime('%Y-%m-%d')
    meta = meta[(meta['date'] >= DATA_START) & (meta['date'] <= DATA_END)]
    meta = meta.dropna(subset=['adr_10'])
    meta = meta[meta['adr_10'] > 0]
    meta['split'] = np.where(meta['date'] <= TRAIN_END, 'train', 'val')
    print(f"  {len(meta)} gappers (Train: {(meta['split']=='train').sum()}, Val: {(meta['split']=='val').sum()})", file=sys.stderr)

    all_results = []

    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Processing", file=sys.stderr):
        ticker = row['ticker']
        date_str = row['date']
        adr = row['adr_10']
        gap_dir = row['gap_direction']
        gap_size = row.get('gap_size_in_adr', np.nan)

        df = load_raw_1min(ticker, date_str)
        if df is None or len(df) < 60:
            continue

        # Check columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(c in df.columns for c in required_cols):
            continue

        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        volumes = df['volume'].values
        rth_open = opens[0]

        if rth_open <= 0 or adr <= 0:
            continue

        # ============================================
        # 1. OPENING DRIVE (first 15 bars = 9:30-9:44)
        # ============================================
        od_dir, od_move, od_close_price = compute_opening_drive(df, 15)
        if od_dir is None:
            continue

        od_move_adr = (od_close_price - rth_open) / adr
        od_against_gap = (gap_dir == 'up' and od_dir == 'down') or (gap_dir == 'down' and od_dir == 'up')

        # ============================================
        # 2. OPENING RANGE (5min and 15min)
        # ============================================
        or5_hi, or5_lo = compute_opening_range(df, 5)
        or15_hi, or15_lo = compute_opening_range(df, 15)

        if or15_hi is None:
            continue

        or5_range = (or5_hi - or5_lo) / adr if or5_lo > 0 else np.nan
        or15_range = (or15_hi - or15_lo) / adr if or15_lo > 0 else np.nan

        # ============================================
        # 3. ORB BREAKOUT DETECTION (after bar 15 = 9:45)
        # ============================================
        # Look for break of 15min range
        orb_break_dir = None
        orb_break_bar = None
        orb_break_price = None

        for i in range(15, min(len(df), 60)):  # bars 15-59 (9:45-10:30)
            if highs[i] > or15_hi and orb_break_dir is None:
                orb_break_dir = 'up'
                orb_break_bar = i
                orb_break_price = or15_hi  # assume fill at breakout level
                break
            elif lows[i] < or15_lo and orb_break_dir is None:
                orb_break_dir = 'down'
                orb_break_bar = i
                orb_break_price = or15_lo
                break

        # ============================================
        # 4. BEHAVIORAL METRICS (no SL/Target)
        # ============================================
        # Rest-of-day from 9:45 (bar 15)
        if len(df) > 15:
            rod_close = closes[-1]  # close of day
            rod_return = (rod_close - od_close_price) / adr  # return from 9:45 onward

            # Max favorable/adverse from 9:45
            future_prices = closes[15:]
            if len(future_prices) > 0:
                if od_against_gap:
                    # Against gap = expect reversal toward gap close
                    # For GapUp+OD_down: expect further downside (short)
                    # For GapDn+OD_up: expect further upside (long)
                    if gap_dir == 'up':  # OD was down, continue down
                        sign = -1
                    else:  # OD was up, continue up
                        sign = 1
                else:
                    # With gap = continuation
                    if gap_dir == 'up':  # OD was up, continue up
                        sign = 1
                    else:  # OD was down, continue down
                        sign = -1

                moves = sign * (future_prices - od_close_price) / adr
                mfe_rod = np.max(moves) if len(moves) > 0 else np.nan
                mae_rod = np.min(moves) if len(moves) > 0 else np.nan

                # MFE at specific windows from 9:45
                windows = {'30m': 30, '60m': 60, '120m': 120, '240m': 240}
                mfe_windows = {}
                drift_windows = {}
                for w_name, w_bars in windows.items():
                    end = min(15 + w_bars, len(closes))
                    future = closes[15:end]
                    if len(future) > 0:
                        w_moves = sign * (future - od_close_price) / adr
                        mfe_windows[f'mfe_{w_name}'] = np.max(w_moves)
                        drift_windows[f'drift_{w_name}'] = w_moves[-1]
                    else:
                        mfe_windows[f'mfe_{w_name}'] = np.nan
                        drift_windows[f'drift_{w_name}'] = np.nan
            else:
                mfe_rod = mae_rod = np.nan
                mfe_windows = {f'mfe_{w}': np.nan for w in ['30m', '60m', '120m', '240m']}
                drift_windows = {f'drift_{w}': np.nan for w in ['30m', '60m', '120m', '240m']}
        else:
            rod_return = np.nan
            mfe_rod = mae_rod = np.nan
            mfe_windows = {f'mfe_{w}': np.nan for w in ['30m', '60m', '120m', '240m']}
            drift_windows = {f'drift_{w}': np.nan for w in ['30m', '60m', '120m', '240m']}

        # ============================================
        # 5. ORB BREAKOUT METRICS
        # ============================================
        orb_mfe = np.nan
        orb_mae = np.nan
        orb_drift_60 = np.nan

        if orb_break_bar is not None and orb_break_price > 0:
            orb_sign = 1 if orb_break_dir == 'up' else -1
            future = closes[orb_break_bar:]
            if len(future) > 0:
                orb_moves = orb_sign * (future - orb_break_price) / adr
                orb_mfe = np.max(orb_moves)
                orb_mae = np.min(orb_moves)

                end_60 = min(60, len(future))
                orb_drift_60 = orb_sign * (future[end_60-1] - orb_break_price) / adr

        # ============================================
        # 6. VOLUME ANALYSIS
        # ============================================
        vol_first_15 = np.sum(volumes[:15]) if len(volumes) >= 15 else np.nan
        vol_15_to_30 = np.sum(volumes[15:30]) if len(volumes) >= 30 else np.nan
        vol_ratio_2nd_vs_1st = vol_15_to_30 / vol_first_15 if vol_first_15 > 0 else np.nan

        # ============================================
        # COLLECT RESULTS
        # ============================================
        result = {
            'ticker': ticker,
            'date': date_str,
            'split': row['split'],
            'gap_direction': gap_dir,
            'gap_size_adr': gap_size,
            'adr': adr,
            'rth_open': rth_open,
            # Opening Drive
            'od_direction': od_dir,
            'od_move_adr': od_move_adr,
            'od_against_gap': od_against_gap,
            'od_close_price': od_close_price,
            # Opening Range
            'or5_hi': or5_hi, 'or5_lo': or5_lo, 'or5_range_adr': or5_range,
            'or15_hi': or15_hi, 'or15_lo': or15_lo, 'or15_range_adr': or15_range,
            # ORB breakout
            'orb_break_dir': orb_break_dir,
            'orb_break_bar': orb_break_bar,
            'orb_break_price': orb_break_price,
            'orb_aligns_od': orb_break_dir == od_dir if orb_break_dir else None,
            'orb_mfe': orb_mfe,
            'orb_mae': orb_mae,
            'orb_drift_60': orb_drift_60,
            # Rest-of-day behavioral
            'mfe_rod': mfe_rod,
            'mae_rod': mae_rod,
            # Volume
            'vol_first_15': vol_first_15,
            'vol_ratio_2nd_vs_1st': vol_ratio_2nd_vs_1st,
            # Metadata features
            'rvol': row.get('rvol_at_time_30min', np.nan),
            'sector': row.get('sector', 'Unknown'),
            'spy_return': row.get('spy_return_day', np.nan),
            'rsi_14': row.get('rsi_14_prev', np.nan),
            'prior_ret_5d': row.get('prior_return_5d', np.nan),
        }
        result.update(mfe_windows)
        result.update(drift_windows)
        all_results.append(result)

    # ============================================
    # ANALYSIS
    # ============================================
    df_all = pd.DataFrame(all_results)
    print(f"\nTotal results: {len(df_all)}", file=sys.stderr)
    if len(df_all) == 0:
        print("ERROR: No results! Check data loading.", file=sys.stderr)
        return

    # Save raw data
    df_all.to_parquet(os.path.join(RESULTS_DIR, 'orb_opening_drive_events_v1.parquet'), index=False)

    output_path = os.path.join(RESULTS_DIR, 'orb_opening_drive_v1.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 90 + "\n")
        f.write("ORB + OPENING DRIVE v1 — Durchlauf 3.0\n")
        f.write("OHLC-Daten, keine Signale vor 9:45, nur Haelfte 1\n")
        f.write("=" * 90 + "\n\n")
        f.write(f"Total: {len(df_all)}\n")
        f.write(f"Train: {(df_all['split']=='train').sum()}, Val: {(df_all['split']=='val').sum()}\n\n")

        # === A. OPENING DRIVE DIRECTION EFFECT ===
        f.write("=" * 90 + "\n")
        f.write("A. OPENING DRIVE DIRECTION → REST OF DAY\n")
        f.write("   (Drift = return from 9:45 onward in OD direction)\n")
        f.write("=" * 90 + "\n\n")

        for split in ['train', 'val']:
            s = df_all[df_all['split'] == split]
            f.write(f"--- {split.upper()} ---\n")
            f.write(f"{'GapDir':<8} {'OD_vs_Gap':<12} {'N':>5} {'Drift_30m':>9} {'Drift_60m':>9} "
                    f"{'Drift_120m':>10} {'Drift_240m':>10} {'MFE_60m':>8} {'MAE_60m':>8}\n")
            f.write("-" * 85 + "\n")

            for gap_dir in ['up', 'down']:
                for against in [True, False]:
                    sub = s[(s['gap_direction'] == gap_dir) & (s['od_against_gap'] == against)]
                    label = 'AGAINST' if against else 'WITH'
                    if len(sub) < 20:
                        continue
                    f.write(f"{gap_dir:<8} {label:<12} {len(sub):>5} "
                            f"{sub['drift_30m'].mean():>9.4f} {sub['drift_60m'].mean():>9.4f} "
                            f"{sub['drift_120m'].mean():>10.4f} {sub['drift_240m'].mean():>10.4f} "
                            f"{sub['mfe_60m'].mean():>8.4f} ")
                    # MAE as column
                    # For "against gap" trades, we compute MFE/MAE in the direction of the OD
                    mae_vals = sub['drift_60m'].dropna()
                    mae_min = mae_vals.min() if len(mae_vals) > 0 else np.nan
                    f.write(f"\n")
            f.write("\n")

        # === B. OPENING DRIVE STRENGTH INTERACTION ===
        f.write("=" * 90 + "\n")
        f.write("B. OPENING DRIVE STRENGTH (|od_move_adr|) × DIRECTION\n")
        f.write("=" * 90 + "\n\n")

        val = df_all[df_all['split'] == 'val']
        strength_bins = [(0, 0.1, 'Weak<0.1'), (0.1, 0.3, 'Med0.1-0.3'),
                         (0.3, 0.6, 'Strong0.3-0.6'), (0.6, 99, 'VStrong>0.6')]

        for gap_dir in ['up', 'down']:
            for against in [True, False]:
                sub = val[(val['gap_direction'] == gap_dir) & (val['od_against_gap'] == against)]
                label = f"Gap{gap_dir.capitalize()} {'AGAINST' if against else 'WITH'}"
                f.write(f"--- {label} ---\n")
                for lo, hi, s_label in strength_bins:
                    ss = sub[(sub['od_move_adr'].abs() >= lo) & (sub['od_move_adr'].abs() < hi)]
                    if len(ss) < 15:
                        continue
                    f.write(f"  {s_label}: N={len(ss)}, Drift60={ss['drift_60m'].mean():.4f}, "
                            f"Drift240={ss['drift_240m'].mean():.4f}, MFE60={ss['mfe_60m'].mean():.4f}\n")
                f.write("\n")

        # === C. ORB BREAKOUT ANALYSIS ===
        f.write("=" * 90 + "\n")
        f.write("C. ORB (15min) BREAKOUT ANALYSIS\n")
        f.write("=" * 90 + "\n\n")

        for split in ['train', 'val']:
            s = df_all[df_all['split'] == split]
            f.write(f"--- {split.upper()} ---\n")

            orb_events = s[s['orb_break_dir'].notna()]
            no_orb = s[s['orb_break_dir'].isna()]
            f.write(f"  ORB Break within 60min of 9:45: {len(orb_events)} ({len(orb_events)/len(s)*100:.1f}%)\n")
            f.write(f"  No Break: {len(no_orb)}\n\n")

            # ORB aligned with OD
            f.write(f"  {'ORB aligns OD':<16} {'N':>5} {'ORB_MFE':>8} {'ORB_Drift60':>11}\n")
            f.write("  " + "-" * 45 + "\n")

            for aligns in [True, False]:
                sub = orb_events[orb_events['orb_aligns_od'] == aligns]
                if len(sub) < 20:
                    continue
                f.write(f"  {'YES' if aligns else 'NO':<16} {len(sub):>5} {sub['orb_mfe'].mean():>8.4f} "
                        f"{sub['orb_drift_60'].mean():>11.4f}\n")

            f.write("\n  ORB by Gap Direction + Break Direction:\n")
            for gap_dir in ['up', 'down']:
                for brk_dir in ['up', 'down']:
                    sub = orb_events[(orb_events['gap_direction'] == gap_dir) &
                                     (orb_events['orb_break_dir'] == brk_dir)]
                    if len(sub) < 20:
                        continue
                    f.write(f"    Gap{gap_dir.capitalize()} + Break{brk_dir.capitalize()}: N={len(sub)}, "
                            f"ORB_MFE={sub['orb_mfe'].mean():.4f}, ORB_Drift60={sub['orb_drift_60'].mean():.4f}\n")
            f.write("\n")

        # === D. COMBINED: OD + ORB + SPY ===
        f.write("=" * 90 + "\n")
        f.write("D. COMBINED FILTERS (Val): OD + ORB + SPY\n")
        f.write("=" * 90 + "\n\n")

        val = df_all[df_all['split'] == 'val']
        combos = []

        for gap_dir in ['up', 'down']:
            for against in [True, False]:
                for spy_label, spy_lo, spy_hi in [('SPY_Up', 0.0, 99), ('SPY_Dn', -99, 0.0), ('AnySPY', -99, 99)]:
                    for orb_label, orb_cond in [('ORB_align', True), ('ORB_oppose', False), ('AnyORB', None)]:
                        mask = (
                            (val['gap_direction'] == gap_dir) &
                            (val['od_against_gap'] == against) &
                            (val['spy_return'] >= spy_lo) & (val['spy_return'] < spy_hi)
                        )
                        if orb_cond is not None:
                            mask = mask & (val['orb_aligns_od'] == orb_cond)

                        sub = val[mask]
                        if len(sub) < 25:
                            continue

                        od_label = 'AGAINST' if against else 'WITH'
                        combos.append({
                            'gap': gap_dir, 'od': od_label, 'spy': spy_label, 'orb': orb_label,
                            'n': len(sub),
                            'drift_60': sub['drift_60m'].mean(),
                            'drift_240': sub['drift_240m'].mean(),
                            'mfe_60': sub['mfe_60m'].mean(),
                        })

        combo_df = pd.DataFrame(combos).sort_values('drift_60', ascending=False)
        f.write(f"{'Gap':<6} {'OD':<8} {'SPY':<8} {'ORB':<12} {'N':>5} {'Drift60':>8} {'Drift240':>9} {'MFE60':>7}\n")
        f.write("-" * 70 + "\n")
        for _, r in combo_df.head(20).iterrows():
            f.write(f"{r['gap']:<6} {r['od']:<8} {r['spy']:<8} {r['orb']:<12} {r['n']:>5} "
                    f"{r['drift_60']:>8.4f} {r['drift_240']:>9.4f} {r['mfe_60']:>7.4f}\n")
        f.write("\n--- WORST ---\n")
        for _, r in combo_df.tail(10).iterrows():
            f.write(f"{r['gap']:<6} {r['od']:<8} {r['spy']:<8} {r['orb']:<12} {r['n']:>5} "
                    f"{r['drift_60']:>8.4f} {r['drift_240']:>9.4f} {r['mfe_60']:>7.4f}\n")

        # === E. BOOTSTRAP CIs ===
        f.write("\n" + "=" * 90 + "\n")
        f.write("E. BOOTSTRAP CIs — Key Setups (Val)\n")
        f.write("=" * 90 + "\n\n")

        np.random.seed(42)
        key_tests = [
            ('GapUp+OD_AGAINST', val[(val['gap_direction']=='up') & (val['od_against_gap']==True)]),
            ('GapUp+OD_WITH', val[(val['gap_direction']=='up') & (val['od_against_gap']==False)]),
            ('GapDn+OD_AGAINST', val[(val['gap_direction']=='down') & (val['od_against_gap']==True)]),
            ('GapDn+OD_WITH', val[(val['gap_direction']=='down') & (val['od_against_gap']==False)]),
        ]

        for label, sub in key_tests:
            for target in ['drift_60m', 'drift_240m']:
                vals = sub[target].dropna().values
                if len(vals) < 30:
                    continue
                boots = [np.mean(np.random.choice(vals, size=len(vals), replace=True)) for _ in range(5000)]
                ci = np.percentile(boots, [2.5, 97.5])
                p = np.mean([b <= 0 for b in boots])
                f.write(f"  {label} {target}: mean={np.mean(vals):.4f}, "
                        f"CI [{ci[0]:.4f}, {ci[1]:.4f}], P(<=0)={p:.4f}\n")
            f.write("\n")

        # === F. SLIPPAGE-ADJUSTED DRIFT ===
        f.write("=" * 90 + "\n")
        f.write("F. SLIPPAGE-ADJUSTED DRIFT (0.005 / 0.010 ADR)\n")
        f.write("=" * 90 + "\n\n")

        for label, sub in key_tests:
            d60 = sub['drift_60m'].mean()
            d240 = sub['drift_240m'].mean()
            f.write(f"  {label}:\n")
            f.write(f"    Raw:        Drift60={d60:.4f}, Drift240={d240:.4f}\n")
            f.write(f"    Slip 0.005: Drift60={d60-0.005:.4f}, Drift240={d240-0.005:.4f}\n")
            f.write(f"    Slip 0.010: Drift60={d60-0.010:.4f}, Drift240={d240-0.010:.4f}\n")
        f.write("\n")

        f.write("=" * 90 + "\n")
        f.write("ENDE ORB + OPENING DRIVE v1\n")
        f.write("=" * 90 + "\n")

    print(f"Ergebnisse: {output_path}", file=sys.stderr)


if __name__ == '__main__':
    main()
