###############################################################################
# ORB ALIGNMENT DEEP DIVE — Durchlauf 3.5
#
# Tasks:
# 1. Parameter breakdowns (a-j) for Aligned vs Opposing
# 2. MAE distribution for ORB-direction trades
# 3. SL/Target simulation using OHLC High/Low
# 4. Reversed perspective (ORB direction instead of OD direction)
# 5. Random baseline (50/50 direction at break bar)
#
# Uses existing parquets + raw_1min for bar-by-bar metrics.
# Half 1 for analysis, Half 2 for one-shot OOS validation.
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
VWAP_DIR = os.path.join(BASE_DIR, 'data', 'vwap')
METADATA_PATH = os.path.join(BASE_DIR, 'data', 'metadata', 'metadata_master.parquet')
IS_EVENTS = os.path.join(BASE_DIR, 'results', 'orb_opening_drive_events_v1.parquet')
OOS_EVENTS = os.path.join(BASE_DIR, 'results', 'oos_opening_drive_half2_events.parquet')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

SL_LEVELS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
TGT_LEVELS = [0.10, 0.20, 0.30, 0.50, 1.0]
WINDOWS = [5, 10, 15, 30, 60, 120, 240]

np.random.seed(42)


def load_raw_1min(ticker, date_str):
    path = os.path.join(RAW_1MIN_DIR, ticker, f"{date_str}.parquet")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_parquet(path)
        if 'session' in df.columns:
            df = df[df['session'] == 'rth'].reset_index(drop=True)
        return df
    except:
        return None


def load_vwap(ticker, date_str):
    path = os.path.join(VWAP_DIR, ticker, f"{date_str}.parquet")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_parquet(path)
        if 'time_et' in df.columns:
            df = df[(df['time_et'] >= '09:30') & (df['time_et'] <= '15:59')].reset_index(drop=True)
        return df
    except:
        return None


def process_orb_trade(row, df_1min, vwap_df, random_long):
    """Process one ORB event. Returns dict with all computed features."""
    break_bar = int(row['orb_break_bar'])
    break_price = row['orb_break_price']
    adr = row['adr']
    is_long = row['orb_break_dir'] == 'up'

    highs = df_1min['high'].values
    lows = df_1min['low'].values
    closes = df_1min['close'].values
    n_bars = len(df_1min)

    if break_bar >= n_bars - 1 or adr <= 0 or break_price <= 0:
        return None

    # Future bars from break
    fh = highs[break_bar:]
    fl = lows[break_bar:]
    fc = closes[break_bar:]
    n_future = len(fc)

    if n_future < 5:
        return None

    result = {}

    # --- ORB-direction drift/MFE/MAE at time windows ---
    for w in WINDOWS:
        end = min(w, n_future)
        if end <= 0:
            result[f'd{w}'] = result[f'mfe{w}'] = result[f'mae{w}'] = np.nan
            continue
        if is_long:
            result[f'mfe{w}'] = (np.max(fh[:end]) - break_price) / adr
            result[f'mae{w}'] = (break_price - np.min(fl[:end])) / adr
            result[f'd{w}'] = (fc[end - 1] - break_price) / adr
        else:
            result[f'mfe{w}'] = (break_price - np.min(fl[:end])) / adr
            result[f'mae{w}'] = (np.max(fh[:end]) - break_price) / adr
            result[f'd{w}'] = (break_price - fc[end - 1]) / adr

    # --- %ADR used at break bar ---
    if break_bar > 0:
        range_at_break = np.max(highs[:break_bar + 1]) - np.min(lows[:break_bar + 1])
        result['pct_adr_used'] = range_at_break / adr
    else:
        result['pct_adr_used'] = 0.0

    # --- ADR% (as pct of price) ---
    rth_open = row.get('rth_open', np.nan)
    result['adr_pct'] = (adr / rth_open * 100) if rth_open and rth_open > 0 else np.nan

    # --- VWAP features ---
    result['price_vs_vwap'] = np.nan
    result['bandwidth'] = np.nan
    if vwap_df is not None and break_bar < len(vwap_df):
        vr = vwap_df.iloc[break_bar]
        vwap_val = vr.get('vwap', np.nan)
        std_val = vr.get('std_dev', np.nan)
        if pd.notna(vwap_val) and pd.notna(std_val) and std_val > 0:
            result['price_vs_vwap'] = (break_price - vwap_val) / std_val
            result['bandwidth'] = 2 * std_val / adr

    # --- SL/Target simulation ---
    # Precompute first hit bars for each level
    sl_bars = {}
    for sl in SL_LEVELS:
        if is_long:
            hits = np.where(fl <= break_price - sl * adr)[0]
        else:
            hits = np.where(fh >= break_price + sl * adr)[0]
        sl_bars[sl] = hits[0] if len(hits) > 0 else n_future + 1

    tgt_bars = {}
    for tgt in TGT_LEVELS:
        if is_long:
            hits = np.where(fh >= break_price + tgt * adr)[0]
        else:
            hits = np.where(fl <= break_price - tgt * adr)[0]
        tgt_bars[tgt] = hits[0] if len(hits) > 0 else n_future + 1

    # Timeout PnL
    if is_long:
        timeout_pnl = (fc[-1] - break_price) / adr
    else:
        timeout_pnl = (break_price - fc[-1]) / adr

    for sl in SL_LEVELS:
        for tgt in TGT_LEVELS:
            sb = sl_bars[sl]
            tb = tgt_bars[tgt]
            if tb < sb:
                pnl = tgt
            elif sb <= n_future:
                pnl = -sl
            else:
                pnl = timeout_pnl
            result[f's{sl}_t{tgt}'] = pnl

    # --- Random baseline (same bar, random direction) ---
    r_long = random_long
    r_sl_bars = {}
    for sl in SL_LEVELS:
        if r_long:
            hits = np.where(fl <= break_price - sl * adr)[0]
        else:
            hits = np.where(fh >= break_price + sl * adr)[0]
        r_sl_bars[sl] = hits[0] if len(hits) > 0 else n_future + 1

    r_tgt_bars = {}
    for tgt in TGT_LEVELS:
        if r_long:
            hits = np.where(fh >= break_price + tgt * adr)[0]
        else:
            hits = np.where(fl <= break_price - tgt * adr)[0]
        r_tgt_bars[tgt] = hits[0] if len(hits) > 0 else n_future + 1

    if r_long:
        r_timeout = (fc[-1] - break_price) / adr
    else:
        r_timeout = (break_price - fc[-1]) / adr

    for sl in SL_LEVELS:
        for tgt in TGT_LEVELS:
            sb = r_sl_bars[sl]
            tb = r_tgt_bars[tgt]
            if tb < sb:
                pnl = tgt
            elif sb <= n_future:
                pnl = -sl
            else:
                pnl = r_timeout
            result[f'r_s{sl}_t{tgt}'] = pnl

    return result


def bucket_analysis(df, bucket_col, bucket_labels, out, metrics_cols, prefix=""):
    """Write breakdown table for a given bucket column."""
    hdr = ""
    for m in metrics_cols:
        hdr += f" {m:>8}"
    out.append(f"{'Bucket':<18} {'N':>5}{hdr}  {'MedD60':>7} {'WR60%':>6}")
    out.append("-" * (35 + 8 * len(metrics_cols) + 15))

    for label in bucket_labels:
        sub = df[df[bucket_col] == label]
        if len(sub) < 15:
            continue
        line = f"{prefix}{label:<18} {len(sub):>5}"
        for m in metrics_cols:
            line += f" {sub[m].mean():>8.4f}"
        med = sub['d60'].median()
        wr = (sub['d60'] > 0).mean() * 100
        line += f"  {med:>7.4f} {wr:>5.1f}%"
        out.append(line)


def write_breakdown(out, df, col, bins, labels, title, metrics):
    """Generic breakdown writer."""
    out.append(f"\n--- {title} ---")
    df_tmp = df.copy()
    df_tmp['_bucket'] = pd.cut(df_tmp[col], bins=bins, labels=labels, include_lowest=True)
    hdr = f"  {'Bucket':<20} {'N':>5}"
    for m in metrics:
        hdr += f" {m:>8}"
    hdr += f" {'MedD60':>8} {'WR60%':>6}"
    out.append(hdr)

    for lbl in labels:
        sub = df_tmp[df_tmp['_bucket'] == lbl]
        if len(sub) < 10:
            continue
        line = f"  {lbl:<20} {len(sub):>5}"
        for m in metrics:
            vals = sub[m].dropna()
            line += f" {vals.mean():>8.4f}" if len(vals) > 0 else f" {'N/A':>8}"
        d60 = sub['d60'].dropna()
        line += f" {d60.median():>8.4f}" if len(d60) > 0 else f" {'N/A':>8}"
        wr = (d60 > 0).mean() * 100 if len(d60) > 0 else 0
        line += f" {wr:>5.1f}%"
        out.append(line)


def bootstrap_ci(vals, n_boot=5000):
    boots = [np.mean(np.random.choice(vals, len(vals), replace=True)) for _ in range(n_boot)]
    ci = np.percentile(boots, [2.5, 97.5])
    p = np.mean([b <= 0 for b in boots])
    return np.mean(vals), ci, p


def main():
    # ================================================================
    # LOAD DATA
    # ================================================================
    print("Loading IS events...", file=sys.stderr)
    is_ev = pd.read_parquet(IS_EVENTS)
    is_orb = is_ev[is_ev['orb_break_dir'].notna()].copy()
    print(f"  IS ORB events: {len(is_orb)}", file=sys.stderr)

    # Merge metadata for VIX (skip if all NaN)
    meta = pd.read_parquet(METADATA_PATH)
    meta['date'] = pd.to_datetime(meta['date']).dt.strftime('%Y-%m-%d')

    # Pre-generate random directions for all IS events
    random_dirs = np.random.random(len(is_orb)) > 0.5  # True = long

    # ================================================================
    # PROCESS IS EVENTS
    # ================================================================
    print("Processing IS events (raw_1min + VWAP)...", file=sys.stderr)
    all_trades = []
    vwap_loaded = 0

    for idx, (_, row) in enumerate(tqdm(is_orb.iterrows(), total=len(is_orb),
                                         desc="IS", file=sys.stderr)):
        df_1min = load_raw_1min(row['ticker'], row['date'])
        if df_1min is None or len(df_1min) < 60:
            continue

        vwap_df = load_vwap(row['ticker'], row['date'])
        if vwap_df is not None:
            vwap_loaded += 1

        result = process_orb_trade(row, df_1min, vwap_df, random_dirs[idx])
        if result is None:
            continue

        # Carry forward event-level features
        result['ticker'] = row['ticker']
        result['date'] = row['date']
        result['split'] = row['split']
        result['gap_direction'] = row['gap_direction']
        result['gap_size_adr'] = row.get('gap_size_adr', np.nan)
        result['orb_break_dir'] = row['orb_break_dir']
        result['orb_aligns_od'] = row['orb_aligns_od']
        result['od_move_adr'] = row['od_move_adr']
        result['od_direction'] = row['od_direction']
        result['orb_break_bar'] = row['orb_break_bar']
        result['rvol'] = row.get('rvol', np.nan)
        result['sector'] = row.get('sector', 'Unknown')
        result['spy_return'] = row.get('spy_return', np.nan)
        all_trades.append(result)

    print(f"  Processed: {len(all_trades)}, VWAP loaded: {vwap_loaded}", file=sys.stderr)
    tr = pd.DataFrame(all_trades)
    tr.to_parquet(os.path.join(RESULTS_DIR, 'orb_deep_dive_is_trades.parquet'), index=False)

    # ================================================================
    # ANALYSIS OUTPUT
    # ================================================================
    out = []
    M = ['d30', 'd60', 'd240', 'mfe30', 'mfe60', 'mae30', 'mae60']

    out.append("=" * 90)
    out.append("ORB ALIGNMENT DEEP DIVE — Durchlauf 3.5")
    out.append("Half 1 (Train+Val), ORB-Direction Metrics")
    out.append("=" * 90)
    out.append(f"\nTotal IS ORB trades: {len(tr)}")
    out.append(f"  Aligned (OD=ORB): {(tr['orb_aligns_od']==True).sum()}")
    out.append(f"  Opposing (OD!=ORB): {(tr['orb_aligns_od']==False).sum()}")
    out.append(f"  VWAP data available: {tr['price_vs_vwap'].notna().sum()}")
    out.append("")

    # ================================================================
    # TASK 4: REVERSED PERSPECTIVE (do first — changes framework)
    # ================================================================
    out.append("=" * 90)
    out.append("TASK 4: ORB DIRECTION PERSPECTIVE (Reversed from OD)")
    out.append("  All metrics in ORB break direction. +drift = price moves in ORB dir.")
    out.append("=" * 90)
    out.append("")

    for label, mask in [('ALIGNED', tr['orb_aligns_od'] == True),
                         ('OPPOSING', tr['orb_aligns_od'] == False),
                         ('ALL ORB', tr['orb_aligns_od'].notna())]:
        sub = tr[mask]
        if len(sub) < 20:
            continue
        out.append(f"  {label}: N={len(sub)}")
        for w in WINDOWS:
            d = sub[f'd{w}'].dropna()
            mfe = sub[f'mfe{w}'].dropna()
            mae = sub[f'mae{w}'].dropna()
            if len(d) > 0:
                out.append(f"    {w:>3}min: Drift={d.mean():>+.4f} Median={d.median():>+.4f} "
                           f"MFE={mfe.mean():.4f} MAE={mae.mean():.4f} "
                           f"WR={100*(d>0).mean():.1f}%")
        out.append("")

    # KEY: Does opposing = +0.40 in ORB direction?
    aligned = tr[tr['orb_aligns_od'] == True]
    opposing = tr[tr['orb_aligns_od'] == False]
    out.append("  KEY HYPOTHESIS: Opposing should show ~+0.40 ADR in ORB direction")
    out.append(f"  Opposing d60: {opposing['d60'].mean():+.4f} (was -0.405 in OD dir)")
    out.append(f"  Aligned d60:  {aligned['d60'].mean():+.4f} (was +0.199 in OD dir?)")
    out.append("")

    # Bootstrap CIs
    for label, sub in [('Aligned', aligned), ('Opposing', opposing)]:
        for col in ['d60', 'd240']:
            vals = sub[col].dropna().values
            if len(vals) < 30:
                continue
            m, ci, p = bootstrap_ci(vals)
            out.append(f"  {label} {col}: mean={m:+.4f}, CI [{ci[0]:+.4f}, {ci[1]:+.4f}], P(<=0)={p:.4f}")
    out.append("")

    # ================================================================
    # TASK 1: PARAMETER BREAKDOWNS
    # ================================================================
    out.append("=" * 90)
    out.append("TASK 1: PARAMETER BREAKDOWNS")
    out.append("  All in ORB direction. Separate for Aligned vs Opposing.")
    out.append("=" * 90)

    for grp_label, grp_mask in [('ALIGNED', tr['orb_aligns_od'] == True),
                                 ('OPPOSING', tr['orb_aligns_od'] == False)]:
        grp = tr[grp_mask].copy()
        out.append(f"\n{'='*40} {grp_label} (N={len(grp)}) {'='*40}")

        # a) ADR%
        write_breakdown(out, grp, 'adr_pct',
                        bins=[0, 3, 5, 10, 100], labels=['<3%', '3-5%', '5-10%', '>10%'],
                        title='a) ADR%', metrics=M)

        # b) RVOL
        write_breakdown(out, grp, 'rvol',
                        bins=[0, 1.5, 3, 5, 100], labels=['<1.5x', '1.5-3x', '3-5x', '>5x'],
                        title='b) RVOL', metrics=M)

        # c) Gap Size in ADR
        write_breakdown(out, grp, 'gap_size_adr',
                        bins=[0, 1, 2, 3, 100], labels=['<1x', '1-2x', '2-3x', '>3x'],
                        title='c) Gap Size (ADR)', metrics=M)

        # d) Gap Direction
        out.append(f"\n--- d) Gap Direction ---")
        hdr = f"  {'GapDir':<20} {'N':>5}"
        for m in M:
            hdr += f" {m:>8}"
        hdr += f" {'MedD60':>8} {'WR60%':>6}"
        out.append(hdr)
        for gd in ['up', 'down']:
            sub = grp[grp['gap_direction'] == gd]
            if len(sub) < 15:
                continue
            line = f"  {'Gap' + gd.capitalize():<20} {len(sub):>5}"
            for m in M:
                line += f" {sub[m].mean():>8.4f}"
            line += f" {sub['d60'].median():>8.4f} {100*(sub['d60']>0).mean():>5.1f}%"
            out.append(line)

        # e) Price vs VWAP at break
        vwap_valid = grp[grp['price_vs_vwap'].notna()]
        if len(vwap_valid) > 50:
            write_breakdown(out, vwap_valid, 'price_vs_vwap',
                            bins=[-100, -1, 0, 1, 100],
                            labels=['<-1σ', '-1σ to 0', '0 to +1σ', '>+1σ'],
                            title='e) Price vs VWAP (StdDevs)', metrics=M)
        else:
            out.append(f"\n--- e) Price vs VWAP --- SKIPPED (N={len(vwap_valid)} too small)")

        # f) BandWidth
        bw_valid = grp[grp['bandwidth'].notna()]
        if len(bw_valid) > 50:
            write_breakdown(out, bw_valid, 'bandwidth',
                            bins=[0, 0.1, 0.2, 0.4, 10],
                            labels=['<0.1', '0.1-0.2', '0.2-0.4', '>0.4'],
                            title='f) BandWidth (2σ/ADR)', metrics=M)
        else:
            out.append(f"\n--- f) BandWidth --- SKIPPED (N={len(bw_valid)} too small)")

        # g) %ADR Used at break
        write_breakdown(out, grp, 'pct_adr_used',
                        bins=[0, 0.3, 0.5, 0.75, 10],
                        labels=['<30%', '30-50%', '50-75%', '>75%'],
                        title='g) %ADR Used at Break', metrics=M)

        # h) Sector (Top 5)
        top_sectors = grp['sector'].value_counts().head(5).index.tolist()
        out.append(f"\n--- h) Sector (Top 5) ---")
        hdr = f"  {'Sector':<20} {'N':>5}"
        for m in M:
            hdr += f" {m:>8}"
        hdr += f" {'MedD60':>8} {'WR60%':>6}"
        out.append(hdr)
        for sect in top_sectors:
            sub = grp[grp['sector'] == sect]
            if len(sub) < 10:
                continue
            line = f"  {sect[:20]:<20} {len(sub):>5}"
            for m in M:
                line += f" {sub[m].mean():>8.4f}"
            line += f" {sub['d60'].median():>8.4f} {100*(sub['d60']>0).mean():>5.1f}%"
            out.append(line)

        # i) VIX — skipped (all NaN in metadata)
        out.append(f"\n--- i) VIX --- SKIPPED (all NaN in metadata)")

        # j) OD Strength
        grp['od_strength'] = grp['od_move_adr'].abs()
        write_breakdown(out, grp, 'od_strength',
                        bins=[0, 0.3, 0.6, 100],
                        labels=['<0.3', '0.3-0.6', '>0.6'],
                        title='j) OD Strength (|OD move| in ADR)', metrics=M)

    # ================================================================
    # TASK 2: MAE DISTRIBUTION
    # ================================================================
    out.append("\n" + "=" * 90)
    out.append("TASK 2: MAE DISTRIBUTION (ORB Direction)")
    out.append("  Adverse excursion = price moves AGAINST ORB break direction")
    out.append("=" * 90)

    for label, mask in [('ALIGNED', tr['orb_aligns_od'] == True),
                         ('OPPOSING', tr['orb_aligns_od'] == False)]:
        sub = tr[mask]
        out.append(f"\n--- {label} (N={len(sub)}) ---")

        # MAE percentiles at each time window
        out.append(f"  MAE Percentiles:")
        out.append(f"  {'Window':<8} {'P25':>7} {'P50':>7} {'P75':>7} {'P90':>7} {'P95':>7} {'Mean':>7}")
        for w in [5, 10, 15, 30, 60]:
            vals = sub[f'mae{w}'].dropna().values
            if len(vals) < 20:
                continue
            pcts = np.percentile(vals, [25, 50, 75, 90, 95])
            out.append(f"  {w:>3}min   {pcts[0]:>7.4f} {pcts[1]:>7.4f} {pcts[2]:>7.4f} "
                       f"{pcts[3]:>7.4f} {pcts[4]:>7.4f} {np.mean(vals):>7.4f}")

        # Split by winners vs losers
        for wl_label, wl_mask in [('Winners (d60>0)', sub['d60'] > 0),
                                   ('Losers (d60<=0)', sub['d60'] <= 0)]:
            wl = sub[wl_mask]
            out.append(f"\n  {wl_label} (N={len(wl)}):")
            out.append(f"  {'Window':<8} {'P25':>7} {'P50':>7} {'P75':>7} {'P90':>7} {'Mean':>7}")
            for w in [5, 10, 15, 30, 60]:
                vals = wl[f'mae{w}'].dropna().values
                if len(vals) < 10:
                    continue
                pcts = np.percentile(vals, [25, 50, 75, 90])
                out.append(f"  {w:>3}min   {pcts[0]:>7.4f} {pcts[1]:>7.4f} {pcts[2]:>7.4f} "
                           f"{pcts[3]:>7.4f} {np.mean(vals):>7.4f}")

    # ================================================================
    # TASK 3: SL/TARGET MATRIX
    # ================================================================
    out.append("\n" + "=" * 90)
    out.append("TASK 3: SL x TARGET MATRIX (ORB Direction, OHLC-based)")
    out.append("  Entry at ORB break price. SL/Target checked vs High/Low.")
    out.append("=" * 90)

    for label, mask in [('ALIGNED', tr['orb_aligns_od'] == True),
                         ('OPPOSING', tr['orb_aligns_od'] == False),
                         ('ALL ORB', tr['orb_aligns_od'].notna())]:
        sub = tr[mask]
        out.append(f"\n--- {label} (N={len(sub)}) ---")
        hdr = f"  {'SL\\TGT':<8}"
        for tgt in TGT_LEVELS:
            hdr += f" {'T'+str(tgt):>12}"
        out.append(hdr)

        for sl in SL_LEVELS:
            line = f"  SL{sl:<5}"
            for tgt in TGT_LEVELS:
                col = f's{sl}_t{tgt}'
                vals = sub[col].dropna().values
                if len(vals) == 0:
                    line += f" {'N/A':>12}"
                    continue
                wr = np.mean(vals > 0) * 100
                avg = np.mean(vals)
                exp = avg  # expectancy = avg PnL per trade
                line += f" {wr:>4.0f}%/{avg:>+.3f}"
            out.append(line)

        # Show best combos
        out.append(f"\n  Top 5 by Expectancy:")
        combos = []
        for sl in SL_LEVELS:
            for tgt in TGT_LEVELS:
                col = f's{sl}_t{tgt}'
                vals = sub[col].dropna().values
                if len(vals) > 0:
                    combos.append({
                        'sl': sl, 'tgt': tgt, 'n': len(vals),
                        'wr': np.mean(vals > 0), 'avg': np.mean(vals),
                        'med': np.median(vals)
                    })
        combos_df = pd.DataFrame(combos).sort_values('avg', ascending=False)
        for _, c in combos_df.head(5).iterrows():
            out.append(f"    SL={c['sl']:.2f} TGT={c['tgt']:.1f}: WR={c['wr']:.1%}, "
                       f"Avg={c['avg']:+.4f}, Med={c['med']:+.4f}")

    # ================================================================
    # TASK 5: RANDOM BASELINE
    # ================================================================
    out.append("\n" + "=" * 90)
    out.append("TASK 5: RANDOM BASELINE (50/50 direction at break bar)")
    out.append("  Same SL/Target, same break bar, random Long/Short")
    out.append("=" * 90)

    out.append(f"\n--- ALL ORB EVENTS (N={len(tr)}) ---")
    out.append(f"  Comparing ORB-direction vs Random-direction:")
    out.append("")

    hdr = f"  {'SL\\TGT':<8}"
    for tgt in TGT_LEVELS:
        hdr += f" {'T'+str(tgt):>12}"
    out.append("  ORB-directed:")
    out.append(hdr)
    for sl in SL_LEVELS:
        line = f"  SL{sl:<5}"
        for tgt in TGT_LEVELS:
            vals = tr[f's{sl}_t{tgt}'].dropna().values
            if len(vals) > 0:
                line += f" {np.mean(vals):>+11.4f}"
            else:
                line += f" {'N/A':>12}"
        out.append(line)

    out.append("\n  Random-directed:")
    out.append(hdr)
    for sl in SL_LEVELS:
        line = f"  SL{sl:<5}"
        for tgt in TGT_LEVELS:
            vals = tr[f'r_s{sl}_t{tgt}'].dropna().values
            if len(vals) > 0:
                line += f" {np.mean(vals):>+11.4f}"
            else:
                line += f" {'N/A':>12}"
        out.append(line)

    out.append("\n  DELTA (ORB - Random):")
    out.append(hdr)
    for sl in SL_LEVELS:
        line = f"  SL{sl:<5}"
        for tgt in TGT_LEVELS:
            orb_vals = tr[f's{sl}_t{tgt}'].dropna().values
            rnd_vals = tr[f'r_s{sl}_t{tgt}'].dropna().values
            if len(orb_vals) > 0 and len(rnd_vals) > 0:
                delta = np.mean(orb_vals) - np.mean(rnd_vals)
                line += f" {delta:>+11.4f}"
            else:
                line += f" {'N/A':>12}"
        out.append(line)

    # Aligned vs Opposing vs Random for best combo
    out.append("\n  Best combo comparison:")
    best_combo = combos_df.iloc[0] if len(combos_df) > 0 else None
    if best_combo is not None:
        bsl, btgt = best_combo['sl'], best_combo['tgt']
        col = f's{bsl}_t{btgt}'
        rcol = f'r_s{bsl}_t{btgt}'
        for lbl, mask in [('Aligned', tr['orb_aligns_od'] == True),
                           ('Opposing', tr['orb_aligns_od'] == False),
                           ('All ORB', tr['orb_aligns_od'].notna())]:
            sub = tr[mask]
            orb_pnl = sub[col].mean()
            rnd_pnl = sub[rcol].mean()
            out.append(f"    {lbl}: ORB={orb_pnl:+.4f}, Random={rnd_pnl:+.4f}, "
                       f"Delta={orb_pnl - rnd_pnl:+.4f}")

    # ================================================================
    # BOOTSTRAP CIs FOR KEY FINDINGS
    # ================================================================
    out.append("\n" + "=" * 90)
    out.append("BOOTSTRAP CIs — Key Findings")
    out.append("=" * 90)
    out.append("")

    np.random.seed(42)
    for label, mask in [('Aligned', tr['orb_aligns_od'] == True),
                         ('Opposing', tr['orb_aligns_od'] == False),
                         ('All ORB', tr['orb_aligns_od'].notna())]:
        sub = tr[mask]
        for col_name, col in [('d30', 'd30'), ('d60', 'd60'), ('d240', 'd240')]:
            vals = sub[col].dropna().values
            if len(vals) < 30:
                continue
            m, ci, p = bootstrap_ci(vals)
            out.append(f"  {label} {col_name}: mean={m:+.4f}, "
                       f"CI [{ci[0]:+.4f}, {ci[1]:+.4f}], P(<=0)={p:.4f}")
        out.append("")

    # ================================================================
    # OOS VALIDATION
    # ================================================================
    out.append("=" * 90)
    out.append("OOS VALIDATION — Half 2 (2024-2026)")
    out.append("  One-shot test of key findings from IS analysis")
    out.append("=" * 90)
    out.append("")

    print("\nLoading OOS events...", file=sys.stderr)
    oos_ev = pd.read_parquet(OOS_EVENTS)
    oos_orb = oos_ev[oos_ev['orb_break_dir'].notna()].copy()
    # Merge metadata for sector/rvol
    oos_orb = oos_orb.merge(
        meta[['ticker', 'date', 'sector', 'rvol_at_time_30min', 'gap_size_in_adr']].rename(
            columns={'rvol_at_time_30min': 'rvol', 'gap_size_in_adr': 'gap_size_adr'}),
        on=['ticker', 'date'], how='left'
    )
    print(f"  OOS ORB events: {len(oos_orb)}", file=sys.stderr)

    # Need to recompute orb_break_bar/price for OOS
    random_dirs_oos = np.random.random(len(oos_orb)) > 0.5
    oos_trades = []

    for idx, (_, row) in enumerate(tqdm(oos_orb.iterrows(), total=len(oos_orb),
                                         desc="OOS", file=sys.stderr)):
        df_1min = load_raw_1min(row['ticker'], row['date'])
        if df_1min is None or len(df_1min) < 60:
            continue

        highs_all = df_1min['high'].values
        lows_all = df_1min['low'].values
        closes_all = df_1min['close'].values
        adr = row['adr']
        rth_open = df_1min['open'].values[0]

        # Recompute ORB
        if len(df_1min) < 15:
            continue
        or15_hi = np.max(highs_all[:15])
        or15_lo = np.min(lows_all[:15])

        orb_break_bar = None
        orb_break_dir = None
        orb_break_price = None
        for i in range(15, min(len(df_1min), 60)):
            if highs_all[i] > or15_hi:
                orb_break_bar = i
                orb_break_dir = 'up'
                orb_break_price = or15_hi
                break
            elif lows_all[i] < or15_lo:
                orb_break_bar = i
                orb_break_dir = 'down'
                orb_break_price = or15_lo
                break

        if orb_break_bar is None:
            continue

        # Check consistency with existing data
        if orb_break_dir != row['orb_break_dir']:
            continue  # Skip inconsistent

        od_dir = row['od_direction']
        orb_aligns = orb_break_dir == od_dir

        # Build row-like object for process_orb_trade
        fake_row = {
            'orb_break_bar': orb_break_bar,
            'orb_break_price': orb_break_price,
            'adr': adr,
            'orb_break_dir': orb_break_dir,
            'rth_open': rth_open,
        }
        fake_row_obj = pd.Series(fake_row)

        result = process_orb_trade(fake_row_obj, df_1min, None, random_dirs_oos[idx])
        if result is None:
            continue

        result['ticker'] = row['ticker']
        result['date'] = row['date']
        result['gap_direction'] = row['gap_direction']
        result['orb_break_dir'] = orb_break_dir
        result['orb_aligns_od'] = orb_aligns
        result['od_move_adr'] = row['od_move_adr']
        result['od_direction'] = od_dir
        result['gap_size_adr'] = row.get('gap_size_adr', np.nan)
        result['rvol'] = row.get('rvol', np.nan)
        result['sector'] = row.get('sector', 'Unknown')
        result['year'] = row.get('year', row['date'][:4])
        oos_trades.append(result)

    print(f"  OOS processed: {len(oos_trades)}", file=sys.stderr)
    oos = pd.DataFrame(oos_trades)
    oos.to_parquet(os.path.join(RESULTS_DIR, 'orb_deep_dive_oos_trades.parquet'), index=False)

    # --- OOS Task 4: Reversed Perspective ---
    out.append("\n--- OOS Task 4: ORB Direction Perspective ---")
    for label, mask in [('ALIGNED', oos['orb_aligns_od'] == True),
                         ('OPPOSING', oos['orb_aligns_od'] == False),
                         ('ALL', oos['orb_aligns_od'].notna())]:
        sub = oos[mask]
        if len(sub) < 20:
            continue
        out.append(f"  {label}: N={len(sub)}")
        for w in [30, 60, 240]:
            d = sub[f'd{w}'].dropna()
            mfe = sub[f'mfe{w}'].dropna()
            mae = sub[f'mae{w}'].dropna()
            if len(d) > 0:
                out.append(f"    {w:>3}min: Drift={d.mean():>+.4f} MFE={mfe.mean():.4f} "
                           f"MAE={mae.mean():.4f} WR={100*(d>0).mean():.1f}%")
        out.append("")

    # --- OOS Bootstrap CIs ---
    out.append("--- OOS Bootstrap CIs ---")
    np.random.seed(42)
    for label, mask in [('Aligned', oos['orb_aligns_od'] == True),
                         ('Opposing', oos['orb_aligns_od'] == False),
                         ('All', oos['orb_aligns_od'].notna())]:
        sub = oos[mask]
        for col in ['d60', 'd240']:
            vals = sub[col].dropna().values
            if len(vals) < 30:
                continue
            m, ci, p = bootstrap_ci(vals)
            out.append(f"  {label} {col}: mean={m:+.4f}, CI [{ci[0]:+.4f}, {ci[1]:+.4f}], P(<=0)={p:.4f}")
    out.append("")

    # --- OOS SL/Target for best IS combo ---
    out.append("--- OOS SL/Target (best IS combos) ---")
    if best_combo is not None:
        bsl, btgt = best_combo['sl'], best_combo['tgt']
        col = f's{bsl}_t{btgt}'
        rcol = f'r_s{bsl}_t{btgt}'
        for lbl, mask in [('Aligned', oos['orb_aligns_od'] == True),
                           ('Opposing', oos['orb_aligns_od'] == False),
                           ('All', oos['orb_aligns_od'].notna())]:
            sub = oos[mask]
            orb_vals = sub[col].dropna().values
            rnd_vals = sub[rcol].dropna().values
            if len(orb_vals) > 0:
                m, ci, p = bootstrap_ci(orb_vals)
                rnd_m = np.mean(rnd_vals) if len(rnd_vals) > 0 else np.nan
                out.append(f"  {lbl} SL={bsl}/T={btgt}: ORB={m:+.4f} CI[{ci[0]:+.4f},{ci[1]:+.4f}] "
                           f"P(<=0)={p:.4f} | Random={rnd_m:+.4f}")
    out.append("")

    # --- OOS Year-by-Year ---
    out.append("--- OOS Year-by-Year (d60 ORB direction) ---")
    for label, mask in [('Aligned', oos['orb_aligns_od'] == True),
                         ('Opposing', oos['orb_aligns_od'] == False)]:
        sub = oos[mask]
        out.append(f"  {label}:")
        for yr in sorted(oos['year'].unique()):
            ys = sub[sub['year'] == yr]
            if len(ys) < 15:
                continue
            out.append(f"    {yr}: N={len(ys)}, d60={ys['d60'].mean():+.4f}, "
                       f"d240={ys['d240'].mean():+.4f}, WR={100*(ys['d60']>0).mean():.1f}%")
    out.append("")

    # --- OOS Full SL/Target Matrix for top combos ---
    out.append("--- OOS Full SL/Target (Aligned) ---")
    oos_aligned = oos[oos['orb_aligns_od'] == True]
    hdr = f"  {'SL\\TGT':<8}"
    for tgt in TGT_LEVELS:
        hdr += f" {'T'+str(tgt):>12}"
    out.append(hdr)
    for sl in SL_LEVELS:
        line = f"  SL{sl:<5}"
        for tgt in TGT_LEVELS:
            vals = oos_aligned[f's{sl}_t{tgt}'].dropna().values
            if len(vals) > 0:
                wr = np.mean(vals > 0) * 100
                avg = np.mean(vals)
                line += f" {wr:>4.0f}%/{avg:>+.3f}"
            else:
                line += f" {'N/A':>12}"
        out.append(line)

    out.append("\n--- OOS Full SL/Target (Opposing) ---")
    oos_opposing = oos[oos['orb_aligns_od'] == False]
    out.append(hdr)
    for sl in SL_LEVELS:
        line = f"  SL{sl:<5}"
        for tgt in TGT_LEVELS:
            vals = oos_opposing[f's{sl}_t{tgt}'].dropna().values
            if len(vals) > 0:
                wr = np.mean(vals > 0) * 100
                avg = np.mean(vals)
                line += f" {wr:>4.0f}%/{avg:>+.3f}"
            else:
                line += f" {'N/A':>12}"
        out.append(line)

    out.append("\n" + "=" * 90)
    out.append("ENDE ORB DEEP DIVE")
    out.append("=" * 90)

    # Write output
    output_path = os.path.join(RESULTS_DIR, 'orb_deep_dive_v1.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out))
    print(f"\nErgebnisse: {output_path}", file=sys.stderr)


if __name__ == '__main__':
    main()
