"""
Durchlauf 6.0 — OOS-Validierung Cheat Sheet: 1-Min Daten Findings
Haelfte 2: 2024-01-01 bis 2026-02-06

Validates: F6 (PM/RTH30 Ratio), F7 (OD-Staerke), F8 (Spaeter VWAP-Break),
           F9 (Erste Kerze), F10 (Volumen-Konzentration), F12 (VWAP Regime),
           F13 (Level-Rejections), F14 (Multi-Touch), F17 (VWAP Slope),
           F20 (Order-Groesse)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import warnings
warnings.filterwarnings('ignore')

# ── helpers ──────────────────────────────────────────────────────────────────
def bootstrap_ci(data, n_boot=1000, ci=0.95):
    if len(data) < 5:
        return np.nan, np.nan
    rng = np.random.default_rng(42)
    stats = np.array([np.mean(rng.choice(data, size=len(data), replace=True)) for _ in range(n_boot)])
    return np.percentile(stats, (1-ci)/2*100), np.percentile(stats, (1+ci)/2*100)

def replication_status(is_val, oos_val):
    if abs(is_val) < 0.001:
        return "N/A"
    if is_val * oos_val < 0:
        return "UMGEKEHRT"
    ratio = abs(oos_val / is_val) if abs(is_val) > 0.001 else 0
    if ratio >= 0.5:
        return "REPLIZIERT"
    elif ratio > 0.1:
        return "TEILWEISE"
    else:
        return "NICHT REPLIZIERT"

# ── Load metadata ────────────────────────────────────────────────────────────
print("Loading metadata...", file=sys.stderr)
meta = pd.read_parquet('data/metadata/metadata_master.parquet')
meta['date'] = pd.to_datetime(meta['date'])
h2 = meta[(meta['date'] >= '2024-01-01') & (meta['date'] <= '2026-02-06')].copy()
h2 = h2[h2['adr_10'].notna()].copy()
h2['gap_size_in_adr'] = h2['gap_size_in_adr'].fillna(
    (h2['today_open'] - h2['prev_close']).abs() / h2['adr_10']
)

print(f"Half 2: {len(h2)} gapper days", file=sys.stderr)

# ── Process each gapper day ──────────────────────────────────────────────────
results = []
skipped = 0
errors = 0

for _, row in tqdm(h2.iterrows(), total=len(h2), desc="Processing", file=sys.stderr):
    ticker = row['ticker']
    date_str = row['date'].strftime('%Y-%m-%d')
    adr = row['adr_10']
    gap_dir = row['gap_direction']
    prev_close = row.get('prev_close', np.nan)

    if pd.isna(adr) or adr <= 0:
        skipped += 1
        continue

    raw_path = Path(f'data/raw_1min/{ticker}/{date_str}.parquet')
    vwap_path = Path(f'data/vwap/{ticker}/{date_str}.parquet')

    if not raw_path.exists():
        skipped += 1
        continue

    try:
        raw = pd.read_parquet(raw_path)
    except Exception:
        errors += 1
        continue

    # Separate PM and RTH
    pm = raw[raw['session'] == 'premarket'].copy() if 'session' in raw.columns else pd.DataFrame()
    rth = raw[raw['session'] == 'rth'].copy() if 'session' in raw.columns else raw.copy()

    if len(rth) < 10:
        skipped += 1
        continue

    rth = rth.sort_values('time_et').reset_index(drop=True)
    rth['min_idx'] = range(len(rth))

    open_price = rth.iloc[0]['open']

    rec = {
        'ticker': ticker, 'date': date_str, 'gap_dir': gap_dir, 'adr': adr,
        'gap_size_adr': row['gap_size_in_adr'],
        'rvol_30': row.get('rvol_at_time_30min', np.nan),
        'spy_ret': row.get('spy_return_day', np.nan),
    }

    # ── F6: PM/RTH30 Ratio ──────────────────────────────────────────────
    pm_vol = pm['volume'].sum() if len(pm) > 0 else 0
    rth_30 = rth[rth['min_idx'] < 30]
    rth_30_vol = rth_30['volume'].sum() if len(rth_30) > 0 else 0
    rec['pm_rth_ratio'] = pm_vol / rth_30_vol if rth_30_vol > 0 else np.nan

    # PM/RTH bucket
    r = rec['pm_rth_ratio']
    if pd.notna(r):
        if r < 0.10:
            rec['pm_rth_bucket'] = '<10%'
        elif r < 0.30:
            rec['pm_rth_bucket'] = '10-30%'
        else:
            rec['pm_rth_bucket'] = '>30%'
    else:
        rec['pm_rth_bucket'] = 'unknown'

    # ── F7/F3: Opening Drive (first 5min) ────────────────────────────────
    od_bars = rth[rth['min_idx'] < 5]
    if len(od_bars) >= 5:
        od_open = od_bars.iloc[0]['open']
        od_close = od_bars.iloc[-1]['close']
        od_high = od_bars['high'].max()
        od_low = od_bars['low'].min()
        od_range = (od_high - od_low) / adr
        od_long = od_close > od_open

        rec['od_range_adr'] = od_range
        rec['od_long'] = od_long

        if od_range < 0.2:
            rec['od_str_bucket'] = '<0.2'
        elif od_range < 0.5:
            rec['od_str_bucket'] = '0.2-0.5'
        else:
            rec['od_str_bucket'] = '>0.5'

        # OD direction relative to gap
        if gap_dir == 'up':
            rec['od_with_gap'] = od_long
        else:
            rec['od_with_gap'] = not od_long

        # SL = 5min high/low
        sl_price = od_low if od_long else od_high
        entry_price = od_close
        rec['sl_size_adr'] = abs(entry_price - sl_price) / adr

        # Track MFE/MAE and SL hit from bar 5 to bar 90 (11:00)
        post_od = rth[(rth['min_idx'] >= 5) & (rth['min_idx'] <= 90)]
        if len(post_od) > 0:
            sl_hit = False
            mfe_before_sl = 0.0
            targets = {0.25: False, 0.50: False, 0.75: False, 1.00: False}
            tgt_before_sl = {t: False for t in targets}

            for _, bar in post_od.iterrows():
                if od_long:
                    bar_mfe = (bar['high'] - entry_price) / adr
                    bar_sl = bar['low'] <= sl_price
                    for t in targets:
                        if not targets[t] and bar['high'] >= entry_price + t * adr:
                            targets[t] = True
                            if not sl_hit:
                                tgt_before_sl[t] = True
                else:
                    bar_mfe = (entry_price - bar['low']) / adr
                    bar_sl = bar['high'] >= sl_price
                    for t in targets:
                        if not targets[t] and bar['low'] <= entry_price - t * adr:
                            targets[t] = True
                            if not sl_hit:
                                tgt_before_sl[t] = True

                if not sl_hit and bar_mfe > mfe_before_sl:
                    mfe_before_sl = bar_mfe
                if not sl_hit and bar_sl:
                    sl_hit = True

            rec['sl_hit'] = sl_hit
            rec['mfe_before_sl'] = mfe_before_sl
            rec['wr_050'] = tgt_before_sl[0.50]
    else:
        rec['od_range_adr'] = np.nan
        rec['od_str_bucket'] = 'unknown'

    # ── F9: Erste Kerze ──────────────────────────────────────────────────
    first_bar = rth.iloc[0]
    candle_dir = 'up' if first_bar['close'] > first_bar['open'] else 'down'
    candle_size = abs(first_bar['close'] - first_bar['open']) / adr
    rec['first_candle_dir'] = candle_dir
    rec['first_candle_size'] = candle_size
    rec['first_candle_big'] = candle_size > 0.10  # "grosse" Kerze threshold

    # ── F10: Volumen-Konzentration ───────────────────────────────────────
    total_vol = rth['volume'].sum()
    if total_vol > 0:
        first_30_vol = rth[rth['min_idx'] < 30]['volume'].sum()
        rec['vol_pct_30'] = first_30_vol / total_vol
    else:
        rec['vol_pct_30'] = np.nan

    # ── F20: Order-Groesse ───────────────────────────────────────────────
    if 'transactions' in rth.columns:
        first_hour = rth[rth['min_idx'] < 60]
        fh_vol = first_hour['volume'].sum()
        fh_txn = first_hour['transactions'].sum()
        rec['avg_order_size'] = fh_vol / fh_txn if fh_txn > 0 else np.nan
    else:
        rec['avg_order_size'] = np.nan

    # ── F8: Spaeter VWAP-Break ──────────────────────────────────────────
    # Load VWAP data
    has_vwap = False
    if vwap_path.exists():
        try:
            vwap_df = pd.read_parquet(vwap_path)
            has_vwap = True
        except:
            pass

    if has_vwap and len(vwap_df) >= 30:
        vwap_df = vwap_df.sort_values('time_et').reset_index(drop=True)

        # Merge VWAP data with RTH
        # Match by index position (both should be sorted by time)
        min_len = min(len(rth), len(vwap_df))
        rth_v = rth.iloc[:min_len].copy()
        rth_v['vwap'] = vwap_df.iloc[:min_len]['vwap'].values if 'vwap' in vwap_df.columns else np.nan

        # Check if VWAP broken in first 30 min (3 consecutive closes beyond)
        first30 = rth_v[rth_v['min_idx'] < 30]
        vwap_broken_30 = False
        consec = 0
        for _, bar in first30.iterrows():
            if pd.isna(bar.get('vwap', np.nan)):
                consec = 0
                continue
            if gap_dir == 'down':
                if bar['close'] > bar['vwap']:
                    consec += 1
                else:
                    consec = 0
            else:
                if bar['close'] < bar['vwap']:
                    consec += 1
                else:
                    consec = 0
            if consec >= 3:
                vwap_broken_30 = True
                break

        rec['vwap_broken_30'] = vwap_broken_30

        # If NOT broken in 30min, check if it breaks later (30-90min)
        if not vwap_broken_30:
            bars_30_90 = rth_v[(rth_v['min_idx'] >= 30) & (rth_v['min_idx'] <= 90)]
            consec = 0
            late_break = False
            for _, bar in bars_30_90.iterrows():
                if pd.isna(bar.get('vwap', np.nan)):
                    consec = 0
                    continue
                if gap_dir == 'down':
                    if bar['close'] > bar['vwap']:
                        consec += 1
                    else:
                        consec = 0
                else:
                    if bar['close'] < bar['vwap']:
                        consec += 1
                    else:
                        consec = 0
                if consec >= 3:
                    late_break = True
                    break
            rec['late_vwap_break'] = late_break
        else:
            rec['late_vwap_break'] = np.nan  # Not applicable

        # ── F12: VWAP als Regime-Filter ──────────────────────────────────
        # Check VWAP not broken by 10:00 for GapDown
        first30_v = rth_v[rth_v['min_idx'] < 30]
        if len(first30_v) > 0 and first30_v['vwap'].notna().any():
            # VWAP resistance for GapDown: never close above VWAP in first 30min
            if gap_dir == 'down':
                vwap_held_30 = not any(first30_v['close'] > first30_v['vwap'])
            else:
                vwap_held_30 = not any(first30_v['close'] < first30_v['vwap'])
            rec['vwap_held_30'] = vwap_held_30
        else:
            rec['vwap_held_30'] = np.nan

        # Sigma levels reached
        if 'upper_1_std' in vwap_df.columns and 'lower_1_std' in vwap_df.columns:
            vwap_bands = vwap_df.iloc[:min_len]
            if gap_dir == 'down':
                # For gap down: -1sigma = lower_1_std
                sigma1_reached = any(rth_v['close'] <= vwap_bands['lower_1_std'].values[:len(rth_v)])
                if 'lower_2_std' in vwap_df.columns:
                    sigma2_reached = any(rth_v['close'] <= vwap_bands['lower_2_std'].values[:len(rth_v)])
                else:
                    sigma2_reached = np.nan
            else:
                sigma1_reached = any(rth_v['close'] >= vwap_bands['upper_1_std'].values[:len(rth_v)])
                if 'upper_2_std' in vwap_df.columns:
                    sigma2_reached = any(rth_v['close'] >= vwap_bands['upper_2_std'].values[:len(rth_v)])
                else:
                    sigma2_reached = np.nan
            rec['sigma1_reached'] = sigma1_reached
            rec['sigma2_reached'] = sigma2_reached
        else:
            rec['sigma1_reached'] = np.nan
            rec['sigma2_reached'] = np.nan

        # ── F17: VWAP Slope ──────────────────────────────────────────────
        if 'vwap' in vwap_df.columns:
            # VWAP slope over first 60 min
            vwap_60 = vwap_df.iloc[:min(60, len(vwap_df))]
            if len(vwap_60) >= 10 and vwap_60['vwap'].notna().sum() >= 5:
                vwap_first = vwap_60['vwap'].iloc[0]
                vwap_last = vwap_60['vwap'].iloc[-1]
                if pd.notna(vwap_first) and pd.notna(vwap_last) and adr > 0:
                    slope = (vwap_last - vwap_first) / adr
                    rec['vwap_slope'] = slope
                    rec['vwap_slope_dir'] = 'rising' if slope > 0.01 else ('falling' if slope < -0.01 else 'flat')
                else:
                    rec['vwap_slope'] = np.nan
                    rec['vwap_slope_dir'] = 'unknown'
            else:
                rec['vwap_slope'] = np.nan
                rec['vwap_slope_dir'] = 'unknown'
        else:
            rec['vwap_slope'] = np.nan
            rec['vwap_slope_dir'] = 'unknown'

        # ── F13: Level-Rejections ────────────────────────────────────────
        # Simplified: check VWAP rejections (touch + bounce) after 9:50 (bar 20+)
        post_950 = rth_v[(rth_v['min_idx'] >= 20) & (rth_v['min_idx'] <= 90)]
        rejection_count = 0
        rejection_held_30 = 0

        if len(post_950) > 0 and 'vwap' in rth_v.columns:
            for i in range(1, len(post_950)):
                bar = post_950.iloc[i]
                prev_bar = post_950.iloc[i-1]
                if pd.isna(bar.get('vwap')) or pd.isna(prev_bar.get('vwap')):
                    continue

                # Touch VWAP (close crosses VWAP direction)
                if gap_dir == 'down':
                    # Approaching VWAP from below (close near or above VWAP), then rejected
                    touched = prev_bar['close'] < prev_bar['vwap'] and bar['high'] >= bar['vwap']
                    rejected = touched and bar['close'] < bar['vwap']
                else:
                    touched = prev_bar['close'] > prev_bar['vwap'] and bar['low'] <= bar['vwap']
                    rejected = touched and bar['close'] > bar['vwap']

                if rejected:
                    rejection_count += 1
                    # Check if rejection holds for 30 bars
                    bar_idx = bar.name if isinstance(bar.name, int) else post_950.index.get_loc(bar.name)
                    future = rth_v.iloc[rth_v.index.get_loc(bar.name):rth_v.index.get_loc(bar.name)+30] if bar.name in rth_v.index else pd.DataFrame()
                    if len(future) >= 5:
                        if gap_dir == 'down':
                            held = not any(future['close'] > future.get('vwap', pd.Series([np.inf]*len(future))).values)
                        else:
                            held = not any(future['close'] < future.get('vwap', pd.Series([-np.inf]*len(future))).values)
                        if held:
                            rejection_held_30 += 1

        rec['rejection_count'] = rejection_count
        rec['rejection_held_30'] = rejection_held_30

    else:
        rec['vwap_broken_30'] = np.nan
        rec['late_vwap_break'] = np.nan
        rec['vwap_held_30'] = np.nan
        rec['sigma1_reached'] = np.nan
        rec['sigma2_reached'] = np.nan
        rec['vwap_slope'] = np.nan
        rec['vwap_slope_dir'] = 'unknown'
        rec['rejection_count'] = np.nan
        rec['rejection_held_30'] = np.nan

    # Day drift (full day)
    rth_close = row.get('rth_close', np.nan)
    drift_day = (rth_close - open_price) / adr if pd.notna(rth_close) else np.nan
    rec['drift_day'] = drift_day

    # Close location
    rth_high = row.get('rth_high', np.nan)
    rth_low = row.get('rth_low', np.nan)
    rec['cl'] = (rth_close - rth_low) / (rth_high - rth_low) if pd.notna(rth_high) and pd.notna(rth_low) and (rth_high - rth_low) > 0 else np.nan

    # Fill
    if pd.notna(prev_close):
        if gap_dir == 'up':
            rec['filled'] = rth['low'].min() <= prev_close
        else:
            rec['filled'] = rth['high'].max() >= prev_close
    else:
        rec['filled'] = np.nan

    results.append(rec)

print(f"\nProcessed {len(results)} days, skipped {skipped}, errors {errors}", file=sys.stderr)
df = pd.DataFrame(results)
df.to_parquet('results/oos_cheat_validation_1min_raw.parquet', index=False)

# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
out = []
def w(s=""): out.append(s)

w("=" * 90)
w("DURCHLAUF 6.0 — OOS-VALIDIERUNG CHEAT SHEET (1-MIN DATEN)")
w("=" * 90)
w(f"Datenbasis: Half 2 (2024-2026), N={len(df)} verarbeitet")
w()

# ═══════════════════════════════════════════════════════════════════════════════
# F6: PM/RTH30 RATIO
# ═══════════════════════════════════════════════════════════════════════════════
w("=" * 90)
w("=== OOS-VALIDIERUNG: F6 PM/RTH30 RATIO ===")
w("=" * 90)
w("IS-Werte (Haelfte 1):")
w("  PM/RTH<10% GapUp Drift=+0.312, GapDown Drift=-0.309")
w("  PM/RTH>30% GapUp Drift=-0.244, GapDown Drift=+0.180")
w()

for gd in ['up', 'down']:
    label = f'Gap{"Up" if gd=="up" else "Down"}'
    gsub = df[(df['gap_dir'] == gd) & (df['pm_rth_bucket'] != 'unknown')]

    w(f"  --- {label} ---")
    for bucket in ['<10%', '10-30%', '>30%']:
        sub = gsub[gsub['pm_rth_bucket'] == bucket]
        if len(sub) < 10:
            w(f"  PM/RTH {bucket}: N={len(sub)} (TOO FEW)")
            continue
        drift = sub['drift_day'].mean()
        drift_ci = bootstrap_ci(sub['drift_day'].dropna().values)
        fill = sub['filled'].astype(float).mean() if sub['filled'].notna().any() else np.nan
        cl = sub['cl'].mean() if sub['cl'].notna().any() else np.nan
        w(f"  PM/RTH {bucket}: N={len(sub)}, Drift={drift:+.3f} (CI: {drift_ci[0]:+.3f} to {drift_ci[1]:+.3f}), "
          f"Fill={fill*100:.0f}%, CL={cl:.2f}")

# Explicit comparison
for gd, bucket, is_val in [('up', '<10%', +0.312), ('up', '>30%', -0.244),
                            ('down', '<10%', -0.309), ('down', '>30%', +0.180)]:
    sub = df[(df['gap_dir']==gd) & (df['pm_rth_bucket']==bucket)]
    if len(sub) >= 10:
        oos = sub['drift_day'].mean()
        oos_ci = bootstrap_ci(sub['drift_day'].dropna().values)
        label = f'Gap{"Up" if gd=="up" else "Down"}+PM/RTH{bucket}'
        w(f"\n  {label}: IS={is_val:+.3f}, OOS={oos:+.3f} (CI: {oos_ci[0]:+.3f} to {oos_ci[1]:+.3f}) -> {replication_status(is_val, oos)}")

# Spread
for gd in ['up', 'down']:
    lo = df[(df['gap_dir']==gd) & (df['pm_rth_bucket']=='<10%')]['drift_day']
    hi = df[(df['gap_dir']==gd) & (df['pm_rth_bucket']=='>30%')]['drift_day']
    if len(lo) >= 10 and len(hi) >= 10:
        spread = lo.mean() - hi.mean()
        label = f'Gap{"Up" if gd=="up" else "Down"}'
        w(f"  {label} Spread (<10% minus >30%): {spread:+.3f}")

w()

# ═══════════════════════════════════════════════════════════════════════════════
# F7: OD-STAERKE
# ═══════════════════════════════════════════════════════════════════════════════
w("=" * 90)
w("=== OOS-VALIDIERUNG: F7 OD-STAERKE >0.5 ADR ===")
w("=" * 90)
w("IS: SL-Hit sinkt auf ~40% (vs ~80% bei <0.2 ADR)")
w("IS: WR@0.50 = 51% bei GapDown + OD with + Stark")
w()

for gd in ['up', 'down']:
    label = f'Gap{"Up" if gd=="up" else "Down"}'
    gsub = df[(df['gap_dir'] == gd) & (df['od_str_bucket'].notna()) & (df['od_str_bucket'] != 'unknown')]

    for od in [True, False]:
        od_label = 'with_gap' if od else 'against_gap'
        osub = gsub[gsub['od_with_gap'] == od] if 'od_with_gap' in gsub.columns else pd.DataFrame()

        for strength in ['<0.2', '0.2-0.5', '>0.5']:
            sub = osub[osub['od_str_bucket'] == strength] if len(osub) > 0 else pd.DataFrame()
            if len(sub) < 10:
                continue
            sl_rate = sub['sl_hit'].mean() * 100 if 'sl_hit' in sub.columns and sub['sl_hit'].notna().any() else np.nan
            wr_050 = sub['wr_050'].mean() * 100 if 'wr_050' in sub.columns and sub['wr_050'].notna().any() else np.nan
            mfe = sub['mfe_before_sl'].mean() if 'mfe_before_sl' in sub.columns else np.nan
            sl_ci = bootstrap_ci(sub['sl_hit'].astype(float).values) if 'sl_hit' in sub.columns else (np.nan, np.nan)
            w(f"  {label} + OD {od_label} + Str {strength}: N={len(sub)}, SL-Hit={sl_rate:.0f}% "
              f"(CI: {sl_ci[0]*100:.0f}-{sl_ci[1]*100:.0f}%), WR@0.50={wr_050:.0f}%, MFE={mfe:.3f}")

w()

# ═══════════════════════════════════════════════════════════════════════════════
# F8: SPAETER VWAP-BREAK
# ═══════════════════════════════════════════════════════════════════════════════
w("=" * 90)
w("=== OOS-VALIDIERUNG: F8 SPAETER VWAP-BREAK ===")
w("=" * 90)
w("IS: GapUp nach 30min VWAP-Hold -> spaeter Break Drift=+0.595 (N=197)")
w("IS: GapDown nach 30min VWAP-Hold -> spaeter Break Drift=-0.444 (N=179)")
w()

for gd in ['up', 'down']:
    label = f'Gap{"Up" if gd=="up" else "Down"}'
    # Days where VWAP was NOT broken in first 30min
    not_broken = df[(df['gap_dir'] == gd) & (df['vwap_broken_30'] == False)]
    late_break = not_broken[not_broken['late_vwap_break'] == True]

    w(f"  {label}:")
    w(f"    VWAP nicht gebrochen in 30min: N={len(not_broken)}")
    w(f"    Davon spaeter gebrochen (30-90min): N={len(late_break)}")

    if len(late_break) >= 10:
        drift = late_break['drift_day'].mean()
        drift_ci = bootstrap_ci(late_break['drift_day'].dropna().values)
        is_val = +0.595 if gd == 'up' else -0.444
        w(f"    Drift: IS={is_val:+.3f}, OOS={drift:+.3f} (CI: {drift_ci[0]:+.3f} to {drift_ci[1]:+.3f})")
        w(f"    -> {replication_status(is_val, drift)}")
    else:
        w(f"    N zu klein fuer Analyse")

w()

# ═══════════════════════════════════════════════════════════════════════════════
# F9: ERSTE KERZE RICHTUNG
# ═══════════════════════════════════════════════════════════════════════════════
w("=" * 90)
w("=== OOS-VALIDIERUNG: F9 ERSTE KERZE ===")
w("=" * 90)
w("IS: GapUp + grosse erste Kerze up = Drift +0.275")
w("IS: GapDown + grosse erste Kerze down = Drift -0.345")
w()

for gd in ['up', 'down']:
    label = f'Gap{"Up" if gd=="up" else "Down"}'
    gsub = df[df['gap_dir'] == gd]

    # Direction effect
    for cdir, clbl in [('up', 'up'), ('down', 'down')]:
        sub = gsub[gsub['first_candle_dir'] == cdir]
        if len(sub) >= 10:
            drift = sub['drift_day'].mean()
            drift_ci = bootstrap_ci(sub['drift_day'].dropna().values)
            w(f"  {label} + erste Kerze {clbl}: N={len(sub)}, Drift={drift:+.3f} (CI: {drift_ci[0]:+.3f} to {drift_ci[1]:+.3f})")

    # Big candle + direction
    for cdir, clbl in [('up', 'up'), ('down', 'down')]:
        sub = gsub[(gsub['first_candle_dir'] == cdir) & (gsub['first_candle_big'] == True)]
        if len(sub) >= 10:
            drift = sub['drift_day'].mean()
            drift_ci = bootstrap_ci(sub['drift_day'].dropna().values)
            w(f"  {label} + GROSSE Kerze {clbl}: N={len(sub)}, Drift={drift:+.3f} (CI: {drift_ci[0]:+.3f} to {drift_ci[1]:+.3f})")

# Explicit comparison
gu_big_up = df[(df['gap_dir']=='up') & (df['first_candle_dir']=='up') & (df['first_candle_big']==True)]
gd_big_dn = df[(df['gap_dir']=='down') & (df['first_candle_dir']=='down') & (df['first_candle_big']==True)]
if len(gu_big_up) >= 10:
    oos = gu_big_up['drift_day'].mean()
    w(f"\n  GapUp+grosse Kerze up: IS=+0.275, OOS={oos:+.3f} -> {replication_status(+0.275, oos)}")
if len(gd_big_dn) >= 10:
    oos = gd_big_dn['drift_day'].mean()
    w(f"  GapDown+grosse Kerze down: IS=-0.345, OOS={oos:+.3f} -> {replication_status(-0.345, oos)}")
w()

# ═══════════════════════════════════════════════════════════════════════════════
# F10: VOLUMEN-KONZENTRATION
# ═══════════════════════════════════════════════════════════════════════════════
w("=" * 90)
w("=== OOS-VALIDIERUNG: F10 VOLUMEN-KONZENTRATION ===")
w("=" * 90)
w("IS: >30% Vol in 30min bei GapUp = Drift -0.238 (Fade)")
w("IS: >30% Vol in 30min bei GapDown = Drift +0.203 (Bounce)")
w()

for gd in ['up', 'down']:
    label = f'Gap{"Up" if gd=="up" else "Down"}'
    gsub = df[(df['gap_dir'] == gd) & (df['vol_pct_30'].notna())]

    for blabel, lo_v, hi_v in [('<20%', 0, 0.20), ('20-30%', 0.20, 0.30), ('>30%', 0.30, 1.01)]:
        sub = gsub[(gsub['vol_pct_30'] >= lo_v) & (gsub['vol_pct_30'] < hi_v)]
        if len(sub) < 10:
            w(f"  {label} Vol {blabel}: N={len(sub)} (TOO FEW)")
            continue
        drift = sub['drift_day'].mean()
        drift_ci = bootstrap_ci(sub['drift_day'].dropna().values)
        w(f"  {label} Vol {blabel}: N={len(sub)}, Drift={drift:+.3f} (CI: {drift_ci[0]:+.3f} to {drift_ci[1]:+.3f})")

# Explicit comparison
gu_30 = df[(df['gap_dir']=='up') & (df['vol_pct_30'] > 0.30)]
gd_30 = df[(df['gap_dir']=='down') & (df['vol_pct_30'] > 0.30)]
if len(gu_30) >= 10:
    oos = gu_30['drift_day'].mean()
    w(f"\n  GapUp >30% Vol: IS=-0.238, OOS={oos:+.3f} -> {replication_status(-0.238, oos)}")
if len(gd_30) >= 10:
    oos = gd_30['drift_day'].mean()
    w(f"  GapDown >30% Vol: IS=+0.203, OOS={oos:+.3f} -> {replication_status(+0.203, oos)}")
w()

# ═══════════════════════════════════════════════════════════════════════════════
# F12: VWAP ALS REGIME-FILTER
# ═══════════════════════════════════════════════════════════════════════════════
w("=" * 90)
w("=== OOS-VALIDIERUNG: F12 VWAP ALS REGIME-FILTER ===")
w("=" * 90)
w("IS: VWAP bis 10:00 nicht gebrochen (DownGap) = VWAP Widerstand 17%")
w("IS: -1sigma erreicht 92%, -2sigma 54%")
w()

# VWAP held 30min
for gd in ['up', 'down']:
    label = f'Gap{"Up" if gd=="up" else "Down"}'
    gsub = df[(df['gap_dir'] == gd) & (df['vwap_held_30'].notna())]
    if len(gsub) < 20:
        w(f"  {label}: N={len(gsub)} (TOO FEW)")
        continue
    held = gsub['vwap_held_30'].mean()
    w(f"  {label} VWAP held 30min: {held*100:.1f}% (N={len(gsub)})")

# Sigma reached
for gd in ['up', 'down']:
    label = f'Gap{"Up" if gd=="up" else "Down"}'
    gsub = df[(df['gap_dir'] == gd) & (df['sigma1_reached'].notna())]
    if len(gsub) < 20:
        continue
    s1 = gsub['sigma1_reached'].astype(float).mean()
    s2_sub = gsub[gsub['sigma2_reached'].notna()]
    s2 = s2_sub['sigma2_reached'].astype(float).mean() if len(s2_sub) >= 10 else np.nan
    s2_str = f"{s2*100:.0f}%" if pd.notna(s2) else "N/A"
    w(f"  {label} -1sigma reached: {s1*100:.0f}%, -2sigma: {s2_str} (N={len(gsub)})")
w()

# ═══════════════════════════════════════════════════════════════════════════════
# F13: LEVEL-REJECTIONS
# ═══════════════════════════════════════════════════════════════════════════════
w("=" * 90)
w("=== OOS-VALIDIERUNG: F13 LEVEL-REJECTIONS ===")
w("=" * 90)
w("IS: Nur 25% halten 30min")
w()

rej_df = df[df['rejection_count'].notna() & (df['rejection_count'] > 0)]
if len(rej_df) >= 20:
    total_rej = rej_df['rejection_count'].sum()
    total_held = rej_df['rejection_held_30'].sum()
    hold_rate = total_held / total_rej if total_rej > 0 else np.nan
    w(f"  Total Rejections: {int(total_rej)}, davon 30min gehalten: {int(total_held)}")
    w(f"  Hold-Rate: IS=25%, OOS={hold_rate*100:.0f}%")
    w(f"  -> {replication_status(0.25, hold_rate)}")
else:
    w(f"  N mit Rejections: {len(rej_df)} (TOO FEW)")
w()

# ═══════════════════════════════════════════════════════════════════════════════
# F17: VWAP SLOPE
# ═══════════════════════════════════════════════════════════════════════════════
w("=" * 90)
w("=== OOS-VALIDIERUNG: F17 VWAP SLOPE ===")
w("=" * 90)
w("IS: Steigend = CL 0.60-0.62, Fallend = CL 0.32-0.42")
w()

for slope_dir in ['rising', 'falling', 'flat']:
    sub = df[df['vwap_slope_dir'] == slope_dir]
    if len(sub) < 20:
        w(f"  VWAP {slope_dir}: N={len(sub)} (TOO FEW)")
        continue
    cl = sub['cl'].mean() if sub['cl'].notna().any() else np.nan
    cl_ci = bootstrap_ci(sub['cl'].dropna().values)
    drift = sub['drift_day'].mean()
    w(f"  VWAP {slope_dir}: N={len(sub)}, CL={cl:.2f} (CI: {cl_ci[0]:.2f} to {cl_ci[1]:.2f}), Drift={drift:+.3f}")
w()

# ═══════════════════════════════════════════════════════════════════════════════
# F20: ORDER-GROESSE
# ═══════════════════════════════════════════════════════════════════════════════
w("=" * 90)
w("=== OOS-VALIDIERUNG: F20 ORDER-GROESSE ===")
w("=" * 90)
w("IS: Grosse Orders bei GapUp = Drift +0.020 (leicht bullish)")
w("IS: Kleine Orders = Drift -0.134 (Fade)")
w()

valid_orders = df[df['avg_order_size'].notna()]
if len(valid_orders) >= 50:
    # Quartile buckets
    q25, q50, q75 = valid_orders['avg_order_size'].quantile([0.25, 0.50, 0.75])
    for gd in ['up', 'down']:
        label = f'Gap{"Up" if gd=="up" else "Down"}'
        gsub = valid_orders[valid_orders['gap_dir'] == gd]

        for blabel, lo_v, hi_v in [('Klein (<P25)', 0, q25), ('Mittel (P25-P75)', q25, q75), ('Gross (>P75)', q75, 1e12)]:
            sub = gsub[(gsub['avg_order_size'] >= lo_v) & (gsub['avg_order_size'] < hi_v)]
            if len(sub) < 10:
                continue
            drift = sub['drift_day'].mean()
            drift_ci = bootstrap_ci(sub['drift_day'].dropna().values)
            w(f"  {label} {blabel}: N={len(sub)}, Drift={drift:+.3f} (CI: {drift_ci[0]:+.3f} to {drift_ci[1]:+.3f})")
else:
    w(f"  N mit Order-Daten: {len(valid_orders)} (TOO FEW)")
w()

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════════
w("\n" + "=" * 90)
w("ZUSAMMENFASSUNG 1-MIN FINDINGS")
w("=" * 90)

# F6 summary
f6_results = {}
for gd, bucket, is_val in [('up', '<10%', +0.312), ('up', '>30%', -0.244),
                            ('down', '<10%', -0.309), ('down', '>30%', +0.180)]:
    sub = df[(df['gap_dir']==gd) & (df['pm_rth_bucket']==bucket)]
    if len(sub) >= 10:
        f6_results[f'{gd}_{bucket}'] = (sub['drift_day'].mean(), is_val)

# F7 summary
f7_strong_gd = df[(df['gap_dir']=='down') & (df.get('od_with_gap', pd.Series([False]*len(df)))==True) & (df['od_str_bucket']=='>0.5')]
f7_sl = f7_strong_gd['sl_hit'].mean() if len(f7_strong_gd) >= 10 and 'sl_hit' in f7_strong_gd.columns else np.nan

w(f"\n{'#':<5} {'Finding':<30} {'IS-Effekt':<20} {'OOS-Effekt':<20} {'Status':<20}")
w("-" * 90)

findings = []

# F6
for gd, bucket, is_val in [('up', '<10%', +0.312), ('up', '>30%', -0.244),
                            ('down', '<10%', -0.309), ('down', '>30%', +0.180)]:
    sub = df[(df['gap_dir']==gd) & (df['pm_rth_bucket']==bucket)]
    if len(sub) >= 10:
        oos = sub['drift_day'].mean()
        status = replication_status(is_val, oos)
        findings.append(('F6', f'PM/RTH {gd}{bucket}', f'{is_val:+.3f}', f'{oos:+.3f}', status))

# F7
if not np.isnan(f7_sl):
    findings.append(('F7', 'OD-Staerke SL-Hit', '~40%', f'{f7_sl*100:.0f}%', replication_status(0.40, f7_sl)))

# F8
for gd, is_val in [('up', +0.595), ('down', -0.444)]:
    sub = df[(df['gap_dir']==gd) & (df['vwap_broken_30']==False) & (df['late_vwap_break']==True)]
    if len(sub) >= 10:
        oos = sub['drift_day'].mean()
        findings.append(('F8', f'Spaeter VWAP-Break {gd}', f'{is_val:+.3f}', f'{oos:+.3f}', replication_status(is_val, oos)))

# F9
for sub_df, is_val, lbl in [(gu_big_up, +0.275, 'Erste Kerze GU+up'), (gd_big_dn, -0.345, 'Erste Kerze GD+dn')]:
    if len(sub_df) >= 10:
        oos = sub_df['drift_day'].mean()
        findings.append(('F9', lbl, f'{is_val:+.3f}', f'{oos:+.3f}', replication_status(is_val, oos)))

# F10
for sub_df, is_val, lbl in [(gu_30, -0.238, 'Vol>30% GapUp'), (gd_30, +0.203, 'Vol>30% GapDown')]:
    if len(sub_df) >= 10:
        oos = sub_df['drift_day'].mean()
        findings.append(('F10', lbl, f'{is_val:+.3f}', f'{oos:+.3f}', replication_status(is_val, oos)))

# F12
for gd in ['down']:
    gsub = df[(df['gap_dir']==gd) & (df['sigma1_reached'].notna())]
    if len(gsub) >= 20:
        s1 = gsub['sigma1_reached'].astype(float).mean()
        findings.append(('F12', f'-1sigma {gd}', '92%', f'{s1*100:.0f}%', replication_status(0.92, s1)))

# F13
if len(rej_df) >= 20 and total_rej > 0:
    findings.append(('F13', 'Rejection Hold', '25%', f'{hold_rate*100:.0f}%', replication_status(0.25, hold_rate)))

# F17
for slope_dir, is_cl in [('rising', 0.61), ('falling', 0.37)]:
    sub = df[df['vwap_slope_dir'] == slope_dir]
    if len(sub) >= 20:
        oos_cl = sub['cl'].mean()
        findings.append(('F17', f'VWAP Slope {slope_dir}', f'CL={is_cl:.2f}', f'CL={oos_cl:.2f}', replication_status(is_cl, oos_cl)))

for num, finding, is_eff, oos_eff, status in findings:
    w(f"{num:<5} {finding:<30} {is_eff:<20} {oos_eff:<20} {status:<20}")

# Write output
report = "\n".join(out)
with open('results/oos_cheat_validation_1min.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print(report)
print(f"\nSaved to results/oos_cheat_validation_1min.txt", file=sys.stderr)
