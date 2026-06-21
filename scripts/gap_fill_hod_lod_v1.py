###############################################################################
# GAP FILL + HOD/LOD TIMING — Durchlauf 3.0
# Tasks #10 and #11
#
# Hypotheses:
# A. Gap Fill: When does the price fill the gap (return to prev close)?
#    - Time-to-fill, fill rate by gap size, what happens AFTER fill?
# B. HOD/LOD Timing: When are the High/Low of the day set?
#    - Can we predict HOD/LOD timing from Opening Drive?
#    - If HOD/LOD is set early + OD aligns → stronger continuation?
#
# Uses raw_1min OHLC data. Only Half 1. No signals before 9:45.
###############################################################################

import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE_DIR = str(PROJECT_ROOT)
RAW_1MIN_DIR = os.path.join(BASE_DIR, 'data', 'raw_1min')
METADATA_PATH = os.path.join(BASE_DIR, 'data', 'metadata', 'metadata_master.parquet')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

DATA_START = '2021-02-21'
DATA_END = '2023-12-31'
TRAIN_END = '2022-12-31'


def load_raw_1min(ticker, date_str):
    path = os.path.join(RAW_1MIN_DIR, ticker, f"{date_str}.parquet")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_parquet(path)
        if 'session' in df.columns:
            df = df[df['session'] == 'rth'].reset_index(drop=True)
        elif 'time_et' in df.columns:
            df = df[(df['time_et'] >= '09:30') & (df['time_et'] <= '15:59')].reset_index(drop=True)
        return df
    except:
        return None


def main():
    print("Loading metadata...", file=sys.stderr)
    meta = pd.read_parquet(METADATA_PATH)
    meta['date'] = pd.to_datetime(meta['date']).dt.strftime('%Y-%m-%d')
    meta = meta[(meta['date'] >= DATA_START) & (meta['date'] <= DATA_END)]
    meta = meta.dropna(subset=['adr_10'])
    meta = meta[meta['adr_10'] > 0]
    meta['split'] = np.where(meta['date'] <= TRAIN_END, 'train', 'val')
    print(f"  {len(meta)} gappers", file=sys.stderr)

    all_results = []

    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Processing", file=sys.stderr):
        ticker = row['ticker']
        date_str = row['date']
        adr = row['adr_10']
        gap_dir = row['gap_direction']
        gap_pct = row.get('gap_pct', np.nan)
        prev_close = row.get('prev_close', np.nan)

        df = load_raw_1min(ticker, date_str)
        if df is None or len(df) < 60:
            continue

        required = ['open', 'high', 'low', 'close', 'volume']
        if not all(c in df.columns for c in required):
            continue

        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        rth_open = opens[0]

        if rth_open <= 0 or adr <= 0:
            continue

        n_bars = len(df)

        # =============================================
        # Opening Drive (first 15 bars)
        # =============================================
        if n_bars < 15:
            continue
        od_close = closes[14]
        od_dir = 'up' if od_close > rth_open else 'down'
        od_move_adr = (od_close - rth_open) / adr

        # =============================================
        # A. GAP FILL ANALYSIS
        # =============================================
        # Gap fill = price returns to prev_close
        gap_filled = False
        gap_fill_bar = np.nan
        gap_fill_time_min = np.nan

        if not np.isnan(prev_close) and prev_close > 0:
            for i in range(n_bars):
                if gap_dir == 'up':
                    # Gap up: fill = price drops to prev_close
                    if lows[i] <= prev_close:
                        gap_filled = True
                        gap_fill_bar = i
                        gap_fill_time_min = i  # each bar is 1 min
                        break
                else:
                    # Gap down: fill = price rises to prev_close
                    if highs[i] >= prev_close:
                        gap_filled = True
                        gap_fill_bar = i
                        gap_fill_time_min = i
                        break

        # After fill: what happens?
        post_fill_drift_60 = np.nan
        post_fill_drift_240 = np.nan
        post_fill_mfe_60 = np.nan

        if gap_filled and not np.isnan(gap_fill_bar):
            fill_bar = int(gap_fill_bar)
            fill_price = prev_close
            # Direction after fill: mean reversion continues or bounce?
            # For GapUp, after fill (price dropped to prev_close), does it bounce back up or keep falling?
            # Sign: +1 = long (bounce), -1 = short (continuation down)
            if gap_dir == 'up':
                sign = 1  # measure bounce after gap fill (long from prev_close)
            else:
                sign = -1  # measure bounce after gap fill (short from prev_close)

            end_60 = min(fill_bar + 60, n_bars)
            end_240 = min(fill_bar + 240, n_bars)

            if end_60 > fill_bar:
                future_60 = closes[fill_bar:end_60]
                moves_60 = sign * (future_60 - fill_price) / adr
                post_fill_mfe_60 = np.max(moves_60)
                post_fill_drift_60 = moves_60[-1]

            if end_240 > fill_bar:
                future_240 = closes[fill_bar:end_240]
                moves_240 = sign * (future_240 - fill_price) / adr
                post_fill_drift_240 = moves_240[-1]

        # =============================================
        # B. HOD/LOD TIMING
        # =============================================
        hod_bar = np.argmax(highs)
        lod_bar = np.argmin(lows)
        hod_price = highs[hod_bar]
        lod_price = lows[lod_bar]

        # HOD/LOD in ADR
        day_range = (hod_price - lod_price)
        day_range_adr = day_range / adr if adr > 0 else np.nan

        # Timing buckets
        def bar_to_bucket(bar):
            if bar < 15: return 'OD_0-15'
            elif bar < 30: return 'Early_15-30'
            elif bar < 60: return 'Morning_30-60'
            elif bar < 120: return 'Midday_60-120'
            elif bar < 240: return 'Afternoon_120-240'
            else: return 'Late_240+'

        hod_bucket = bar_to_bucket(hod_bar)
        lod_bucket = bar_to_bucket(lod_bar)

        # Key question: if HOD set during OD (first 15 bars), does price trend down rest of day?
        hod_during_od = hod_bar < 15
        lod_during_od = lod_bar < 15

        # Rest-of-day return from bar 15
        rod_from_945 = (closes[-1] - od_close) / adr if n_bars > 15 else np.nan
        # Unsigned direction
        rod_up = rod_from_945 > 0

        # Close location relative to HOD/LOD
        if day_range > 0:
            close_location = (closes[-1] - lod_price) / day_range  # 0 = at LOD, 1 = at HOD
        else:
            close_location = 0.5

        result = {
            'ticker': ticker,
            'date': date_str,
            'split': row['split'],
            'gap_direction': gap_dir,
            'gap_pct': gap_pct,
            'adr': adr,
            'prev_close': prev_close,
            'rth_open': rth_open,
            'od_direction': od_dir,
            'od_move_adr': od_move_adr,
            # Gap Fill
            'gap_filled': gap_filled,
            'gap_fill_bar': gap_fill_bar,
            'gap_fill_time_min': gap_fill_time_min,
            'gap_fill_in_od': gap_fill_bar < 15 if gap_filled else False,
            'gap_fill_after_945': gap_fill_bar >= 15 if gap_filled else False,
            'post_fill_drift_60': post_fill_drift_60,
            'post_fill_drift_240': post_fill_drift_240,
            'post_fill_mfe_60': post_fill_mfe_60,
            # HOD/LOD
            'hod_bar': hod_bar,
            'lod_bar': lod_bar,
            'hod_bucket': hod_bucket,
            'lod_bucket': lod_bucket,
            'hod_during_od': hod_during_od,
            'lod_during_od': lod_during_od,
            'day_range_adr': day_range_adr,
            'close_location': close_location,
            'rod_from_945': rod_from_945,
        }
        all_results.append(result)

    df_all = pd.DataFrame(all_results)
    print(f"\nTotal: {len(df_all)}", file=sys.stderr)
    if len(df_all) == 0:
        print("ERROR: No results!", file=sys.stderr)
        return

    df_all.to_parquet(os.path.join(RESULTS_DIR, 'gap_fill_hod_lod_events_v1.parquet'), index=False)

    # ====================================================================
    # ANALYSIS
    # ====================================================================
    output_path = os.path.join(RESULTS_DIR, 'gap_fill_hod_lod_v1.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 90 + "\n")
        f.write("GAP FILL + HOD/LOD TIMING — Durchlauf 3.0\n")
        f.write("OHLC-Daten, nur Haelfte 1\n")
        f.write("=" * 90 + "\n\n")
        f.write(f"Total: {len(df_all)}\n")
        f.write(f"Train: {(df_all['split']=='train').sum()}, Val: {(df_all['split']=='val').sum()}\n\n")

        # ==== A. GAP FILL RATES ====
        f.write("=" * 90 + "\n")
        f.write("A. GAP FILL RATES\n")
        f.write("=" * 90 + "\n\n")

        for split in ['train', 'val']:
            s = df_all[df_all['split'] == split]
            f.write(f"--- {split.upper()} ---\n")
            for gd in ['up', 'down']:
                sub = s[s['gap_direction'] == gd]
                filled = sub['gap_filled'].sum()
                total = len(sub)
                fill_rate = filled / total if total > 0 else 0
                f.write(f"  Gap{gd.capitalize()}: {filled}/{total} ({fill_rate:.1%}) filled during day\n")

                # Fill by gap size buckets
                if 'gap_pct' in sub.columns:
                    sub_valid = sub.dropna(subset=['gap_pct'])
                    for lo, hi, label in [(4, 6, '4-6%'), (6, 10, '6-10%'), (10, 20, '10-20%'), (20, 100, '>20%')]:
                        bucket = sub_valid[(sub_valid['gap_pct'].abs() >= lo) & (sub_valid['gap_pct'].abs() < hi)]
                        if len(bucket) < 10:
                            continue
                        br = bucket['gap_filled'].mean()
                        med_time = bucket.loc[bucket['gap_filled'], 'gap_fill_time_min'].median()
                        f.write(f"    {label}: {len(bucket)} gappers, fill rate={br:.1%}, "
                                f"median time={med_time:.0f}min\n")
            f.write("\n")

        # ==== B. GAP FILL TIMING ====
        f.write("=" * 90 + "\n")
        f.write("B. GAP FILL TIMING (when does fill happen?)\n")
        f.write("=" * 90 + "\n\n")

        for split in ['train', 'val']:
            s = df_all[(df_all['split'] == split) & (df_all['gap_filled'])]
            f.write(f"--- {split.upper()} (only filled gaps) ---\n")

            for gd in ['up', 'down']:
                sub = s[s['gap_direction'] == gd]
                if len(sub) < 20:
                    continue
                times = sub['gap_fill_time_min'].dropna()
                f.write(f"  Gap{gd.capitalize()}: N={len(sub)}\n")
                f.write(f"    Mean fill time: {times.mean():.0f} min\n")
                f.write(f"    Median fill time: {times.median():.0f} min\n")
                pcts = np.percentile(times, [10, 25, 50, 75, 90])
                f.write(f"    P10={pcts[0]:.0f}, P25={pcts[1]:.0f}, P50={pcts[2]:.0f}, "
                        f"P75={pcts[3]:.0f}, P90={pcts[4]:.0f}\n")

                # Fill during OD vs after
                in_od = (sub['gap_fill_bar'] < 15).sum()
                after_od = (sub['gap_fill_bar'] >= 15).sum()
                f.write(f"    Filled during OD (first 15min): {in_od} ({in_od/len(sub):.1%})\n")
                f.write(f"    Filled after 9:45: {after_od} ({after_od/len(sub):.1%})\n")
            f.write("\n")

        # ==== C. POST-FILL BEHAVIOR ====
        f.write("=" * 90 + "\n")
        f.write("C. POST-FILL BEHAVIOR (what happens after gap fills?)\n")
        f.write("  Positive drift = bounce back toward gap direction\n")
        f.write("=" * 90 + "\n\n")

        for split in ['train', 'val']:
            s = df_all[(df_all['split'] == split) & (df_all['gap_filled'])]
            f.write(f"--- {split.upper()} ---\n")

            for gd in ['up', 'down']:
                sub = s[s['gap_direction'] == gd]
                if len(sub) < 20:
                    continue
                d60 = sub['post_fill_drift_60'].dropna()
                d240 = sub['post_fill_drift_240'].dropna()
                mfe60 = sub['post_fill_mfe_60'].dropna()
                f.write(f"  Gap{gd.capitalize()}: N={len(sub)}\n")
                f.write(f"    Post-fill Drift60: {d60.mean():.4f} (bounce back)\n")
                f.write(f"    Post-fill Drift240: {d240.mean():.4f}\n")
                f.write(f"    Post-fill MFE60: {mfe60.mean():.4f}\n")

                # Split by fill timing
                early = sub[sub['gap_fill_bar'] < 30]
                late = sub[sub['gap_fill_bar'] >= 30]
                if len(early) >= 15:
                    f.write(f"    Early fill (<30min): N={len(early)}, "
                            f"Drift60={early['post_fill_drift_60'].mean():.4f}, "
                            f"Drift240={early['post_fill_drift_240'].mean():.4f}\n")
                if len(late) >= 15:
                    f.write(f"    Late fill (>=30min): N={len(late)}, "
                            f"Drift60={late['post_fill_drift_60'].mean():.4f}, "
                            f"Drift240={late['post_fill_drift_240'].mean():.4f}\n")
            f.write("\n")

        # ==== D. HOD/LOD TIMING ====
        f.write("=" * 90 + "\n")
        f.write("D. HOD/LOD TIMING — When is the High/Low of Day set?\n")
        f.write("=" * 90 + "\n\n")

        for split in ['train', 'val']:
            s = df_all[df_all['split'] == split]
            f.write(f"--- {split.upper()} ---\n")

            f.write("  HOD timing:\n")
            hod_dist = s['hod_bucket'].value_counts(normalize=True).sort_index()
            for bucket, pct in hod_dist.items():
                f.write(f"    {bucket}: {pct:.1%}\n")

            f.write("  LOD timing:\n")
            lod_dist = s['lod_bucket'].value_counts(normalize=True).sort_index()
            for bucket, pct in lod_dist.items():
                f.write(f"    {bucket}: {pct:.1%}\n")

            f.write("\n  HOD during OD + Gap Direction:\n")
            for gd in ['up', 'down']:
                sub = s[s['gap_direction'] == gd]
                hod_od = sub['hod_during_od'].mean()
                lod_od = sub['lod_during_od'].mean()
                f.write(f"    Gap{gd.capitalize()}: HOD in OD={hod_od:.1%}, LOD in OD={lod_od:.1%}\n")
            f.write("\n")

        # ==== E. HOD/LOD + OPENING DRIVE INTERACTION ====
        f.write("=" * 90 + "\n")
        f.write("E. HOD/LOD + OPENING DRIVE INTERACTION\n")
        f.write("  If OD=up and HOD is set DURING OD → price reversed (bearish rest of day)\n")
        f.write("  If OD=up and LOD is set DURING OD → price continued up (bullish rest of day)\n")
        f.write("=" * 90 + "\n\n")

        val = df_all[df_all['split'] == 'val']

        for gd in ['up', 'down']:
            for od in ['up', 'down']:
                sub = val[(val['gap_direction'] == gd) & (val['od_direction'] == od)]
                if len(sub) < 30:
                    continue
                label = f"Gap{gd.capitalize()}+OD_{od}"
                f.write(f"  {label} (N={len(sub)}):\n")

                # HOD during OD
                hod_od = sub[sub['hod_during_od']]
                hod_later = sub[~sub['hod_during_od']]
                if len(hod_od) >= 10 and len(hod_later) >= 10:
                    f.write(f"    HOD in OD: N={len(hod_od)}, CloseLocation={hod_od['close_location'].mean():.3f}, "
                            f"RoD={hod_od['rod_from_945'].mean():.4f}\n")
                    f.write(f"    HOD later: N={len(hod_later)}, CloseLocation={hod_later['close_location'].mean():.3f}, "
                            f"RoD={hod_later['rod_from_945'].mean():.4f}\n")

                # LOD during OD
                lod_od = sub[sub['lod_during_od']]
                lod_later = sub[~sub['lod_during_od']]
                if len(lod_od) >= 10 and len(lod_later) >= 10:
                    f.write(f"    LOD in OD: N={len(lod_od)}, CloseLocation={lod_od['close_location'].mean():.3f}, "
                            f"RoD={lod_od['rod_from_945'].mean():.4f}\n")
                    f.write(f"    LOD later: N={len(lod_later)}, CloseLocation={lod_later['close_location'].mean():.3f}, "
                            f"RoD={lod_later['rod_from_945'].mean():.4f}\n")
                f.write("\n")

        # ==== F. CLOSE LOCATION ANALYSIS ====
        f.write("=" * 90 + "\n")
        f.write("F. CLOSE LOCATION (where does price close relative to day range?)\n")
        f.write("  0 = at LOD, 1 = at HOD\n")
        f.write("=" * 90 + "\n\n")

        for split in ['train', 'val']:
            s = df_all[df_all['split'] == split]
            f.write(f"--- {split.upper()} ---\n")

            for gd in ['up', 'down']:
                for od in ['up', 'down']:
                    sub = s[(s['gap_direction'] == gd) & (s['od_direction'] == od)]
                    if len(sub) < 20:
                        continue
                    cl = sub['close_location']
                    label = f"Gap{gd.capitalize()}+OD_{od}"
                    f.write(f"  {label}: N={len(sub)}, Mean={cl.mean():.3f}, Median={cl.median():.3f}\n")
            f.write("\n")

        # ==== G. BOOTSTRAP CIs for key findings ====
        f.write("=" * 90 + "\n")
        f.write("G. BOOTSTRAP CIs — Key Gap Fill Findings (Val)\n")
        f.write("=" * 90 + "\n\n")

        np.random.seed(42)
        val_filled = df_all[(df_all['split'] == 'val') & (df_all['gap_filled'])]

        for gd in ['up', 'down']:
            sub = val_filled[val_filled['gap_direction'] == gd]
            vals = sub['post_fill_drift_60'].dropna().values
            if len(vals) < 30:
                continue
            boots = [np.mean(np.random.choice(vals, len(vals), replace=True)) for _ in range(5000)]
            ci = np.percentile(boots, [2.5, 97.5])
            p = np.mean([b <= 0 for b in boots])
            f.write(f"  Gap{gd.capitalize()} post-fill drift_60: mean={np.mean(vals):.4f}, "
                    f"CI [{ci[0]:.4f}, {ci[1]:.4f}], P(<=0)={p:.4f}\n")

        f.write("\n")
        f.write("=" * 90 + "\n")
        f.write("ENDE GAP FILL + HOD/LOD\n")
        f.write("=" * 90 + "\n")

    print(f"Ergebnisse: {output_path}", file=sys.stderr)


if __name__ == '__main__':
    main()
