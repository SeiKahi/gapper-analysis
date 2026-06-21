"""
Durchlauf 6.5 — Kreuzungsanalyse Phase 2: OOS-Validierung (Half 2)
Half 2: 2024-01-01 bis 2026-02-06

Validiert die TOP-10+ interessantesten Ergebnisse aus Phase 1 auf Half 2.
PLUS: Alle Aufgaben komplett auf H2 berechnen zum Vergleich.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def bootstrap_ci(data, n_boot=1000, ci=0.95):
    data = np.array(data, dtype=float)
    data = data[~np.isnan(data)]
    if len(data) < 5:
        return np.nan, np.nan
    rng = np.random.default_rng(42)
    stats = np.array([np.mean(rng.choice(data, size=len(data), replace=True)) for _ in range(n_boot)])
    return np.percentile(stats, (1-ci)/2*100), np.percentile(stats, (1+ci)/2*100)

def fmt_ci(lo, hi, fmt="+.3f"):
    if np.isnan(lo) or np.isnan(hi):
        return "N/A"
    return f"({lo:{fmt}} to {hi:{fmt}})"

def low_n_tag(n, threshold=50):
    return " [LOW N]" if n < threshold else ""

def repl_status(is_val, oos_val):
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

out = []
def w(s=""): out.append(s)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD AND PREPARE DATA (Half 2)
# ══════════════════════════════════════════════════════════════════════════════
print("Loading metadata...", file=sys.stderr)
meta = pd.read_parquet('data/metadata/metadata_master.parquet')
meta['date'] = pd.to_datetime(meta['date'])

h2 = meta[(meta['date'] >= '2024-01-01') & (meta['date'] <= '2026-02-06')].copy()
h2 = h2[h2['adr_10'].notna()].copy()
print(f"Half 2: {len(h2)} gapper days", file=sys.stderr)

# ── Process each gapper day ──────────────────────────────────────────────────
records = []
skipped = 0
errors = 0

for _, row in tqdm(h2.iterrows(), total=len(h2), desc="Processing H2", file=sys.stderr):
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
        'gap_size_adr': row['gap_size_in_adr'] if pd.notna(row['gap_size_in_adr']) else abs(row['today_open'] - row['prev_close']) / adr,
        'rvol_30': row.get('rvol_at_time_30min', np.nan),
        'prev_close': prev_close,
        'open_price': open_price,
    }

    gs = rec['gap_size_adr']
    rec['gap_bucket'] = '<1' if gs < 1 else ('1-2' if gs < 2 else '>2')

    rv = rec['rvol_30']
    if pd.notna(rv):
        rec['rvol_bucket'] = '<2x' if rv < 2 else ('2-5x' if rv < 5 else '>5x')
    else:
        rec['rvol_bucket'] = 'unknown'

    rec['gap_up'] = gap_dir == 'up'

    # PM/RTH30
    pm_vol = pm['volume'].sum() if len(pm) > 0 else 0
    rth_30 = rth[rth['min_idx'] < 30]
    rth_30_vol = rth_30['volume'].sum() if len(rth_30) > 0 else 0
    rec['pm_rth_ratio'] = pm_vol / rth_30_vol if rth_30_vol > 0 else np.nan

    r = rec['pm_rth_ratio']
    if pd.notna(r):
        rec['pm_rth_bucket'] = '<10%' if r < 0.10 else ('10-30%' if r < 0.30 else '>30%')
    else:
        rec['pm_rth_bucket'] = 'unknown'

    # Opening Drive
    od_bars = rth[rth['min_idx'] < 5]
    if len(od_bars) >= 5:
        od_open = od_bars.iloc[0]['open']
        od_close = od_bars.iloc[-1]['close']
        od_high = od_bars['high'].max()
        od_low = od_bars['low'].min()
        od_move = abs(od_close - od_open) / adr
        od_long = od_close > od_open

        rec['od_strength'] = od_move
        rec['od_str_bucket'] = '<0.2' if od_move < 0.2 else ('0.2-0.5' if od_move < 0.5 else '>0.5')
        rec['od_long'] = od_long

        if gap_dir == 'up':
            rec['od_with_gap'] = od_long
        else:
            rec['od_with_gap'] = not od_long
        rec['od_dir_label'] = 'with_gap' if rec['od_with_gap'] else 'against_gap'

        sl_price = od_low if od_long else od_high
        entry_price = od_close
        rec['sl_size_adr'] = abs(entry_price - sl_price) / adr
        rec['od_entry'] = entry_price
        rec['od_sl'] = sl_price

        post_od = rth[(rth['min_idx'] >= 5) & (rth['min_idx'] <= 90)]
        if len(post_od) > 0:
            sl_hit = False
            mfe_val = 0.0
            mae_val = 0.0
            targets = {0.25: False, 0.50: False, 0.75: False, 1.00: False}
            tgt_before_sl = {t: False for t in targets}

            for _, bar in post_od.iterrows():
                if od_long:
                    bar_mfe = (bar['high'] - entry_price) / adr
                    bar_mae = (entry_price - bar['low']) / adr
                    bar_sl = bar['low'] <= sl_price
                    for t in targets:
                        if not targets[t] and bar['high'] >= entry_price + t * adr:
                            targets[t] = True
                            if not sl_hit:
                                tgt_before_sl[t] = True
                else:
                    bar_mfe = (entry_price - bar['low']) / adr
                    bar_mae = (bar['high'] - entry_price) / adr
                    bar_sl = bar['high'] >= sl_price
                    for t in targets:
                        if not targets[t] and bar['low'] <= entry_price - t * adr:
                            targets[t] = True
                            if not sl_hit:
                                tgt_before_sl[t] = True

                if bar_mfe > mfe_val:
                    mfe_val = bar_mfe
                if bar_mae > mae_val:
                    mae_val = bar_mae
                if not sl_hit and bar_sl:
                    sl_hit = True

            rec['sl_hit'] = sl_hit
            rec['mfe'] = mfe_val
            rec['mae'] = mae_val
            for t in targets:
                rec[f'wr_{t:.2f}'] = tgt_before_sl[t]
    else:
        rec['od_strength'] = np.nan
        rec['od_str_bucket'] = 'unknown'
        rec['od_with_gap'] = np.nan
        rec['od_dir_label'] = 'unknown'

    # Erste Kerze
    first_bar = rth.iloc[0]
    candle_body = first_bar['close'] - first_bar['open']
    candle_size = abs(candle_body) / adr
    candle_dir = 'up' if candle_body > 0 else 'down'
    in_gap_dir = (gap_dir == 'up' and candle_dir == 'up') or (gap_dir == 'down' and candle_dir == 'down')

    rec['first_candle_size'] = candle_size
    rec['first_candle_dir'] = candle_dir
    rec['first_candle_in_gap'] = in_gap_dir

    # Day metrics
    rth_close_price = rth.iloc[-1]['close']
    drift_raw = rth_close_price - open_price
    if gap_dir == 'up':
        rec['drift_day'] = drift_raw / adr
    else:
        rec['drift_day'] = -drift_raw / adr

    rec['drift_day_signed'] = drift_raw / adr

    if pd.notna(prev_close):
        if gap_dir == 'up':
            rec['filled'] = rth['low'].min() <= prev_close
        else:
            rec['filled'] = rth['high'].max() >= prev_close
    else:
        rec['filled'] = np.nan

    day_high = rth['high'].max()
    day_low = rth['low'].min()
    rec['cl'] = (rth_close_price - day_low) / (day_high - day_low) if (day_high - day_low) > 0 else np.nan

    if gap_dir == 'up':
        rec['mfe_day'] = (rth['high'].max() - open_price) / adr
        rec['mae_day'] = (open_price - rth['low'].min()) / adr
    else:
        rec['mfe_day'] = (open_price - rth['low'].min()) / adr
        rec['mae_day'] = (rth['high'].max() - open_price) / adr

    first_60 = rth[rth['min_idx'] < 60]
    if len(first_60) > 0:
        if gap_dir == 'up':
            rec['mfe_60'] = (first_60['high'].max() - open_price) / adr
            rec['mae_60'] = (open_price - first_60['low'].min()) / adr
        else:
            rec['mfe_60'] = (open_price - first_60['low'].min()) / adr
            rec['mae_60'] = (first_60['high'].max() - open_price) / adr

    # VWAP-Break
    has_vwap = False
    if vwap_path.exists():
        try:
            vwap_df = pd.read_parquet(vwap_path)
            has_vwap = True
        except:
            pass

    rec['late_vwap_break'] = False
    rec['vwap_break_bar'] = np.nan
    rec['vwap_break_price'] = np.nan

    if has_vwap and len(vwap_df) >= 30:
        vwap_df = vwap_df.sort_values('time_et').reset_index(drop=True)
        min_len = min(len(rth), len(vwap_df))
        rth_v = rth.iloc[:min_len].copy()
        rth_v['vwap'] = vwap_df.iloc[:min_len]['vwap'].values if 'vwap' in vwap_df.columns else np.nan

        first30 = rth_v[rth_v['min_idx'] < 30]
        vwap_broken_30 = False
        consec = 0
        for _, bar in first30.iterrows():
            if pd.isna(bar.get('vwap', np.nan)):
                consec = 0
                continue
            if gap_dir == 'down':
                beyond = bar['close'] > bar['vwap']
            else:
                beyond = bar['close'] < bar['vwap']
            consec = consec + 1 if beyond else 0
            if consec >= 3:
                vwap_broken_30 = True
                break

        rec['vwap_held_30'] = not vwap_broken_30

        if not vwap_broken_30:
            bars_30_90 = rth_v[(rth_v['min_idx'] >= 30) & (rth_v['min_idx'] <= 90)]
            consec = 0
            for _, bar in bars_30_90.iterrows():
                if pd.isna(bar.get('vwap', np.nan)):
                    consec = 0
                    continue
                if gap_dir == 'up':
                    beyond = bar['close'] < bar['vwap']
                else:
                    beyond = bar['close'] > bar['vwap']
                consec = consec + 1 if beyond else 0
                if consec >= 3:
                    rec['late_vwap_break'] = True
                    rec['vwap_break_bar'] = bar['min_idx']
                    rec['vwap_break_price'] = bar['close']
                    break

        if rec['late_vwap_break'] and pd.notna(rec['vwap_break_bar']):
            break_bar = int(rec['vwap_break_bar'])
            break_price = rec['vwap_break_price']
            post_break = rth[rth['min_idx'] > break_bar]

            if len(post_break) > 0:
                pb_15 = rth[(rth['min_idx'] > break_bar) & (rth['min_idx'] <= break_bar + 15)]
                if len(pb_15) > 0:
                    if gap_dir == 'up':
                        rec['pb_drift_15'] = (pb_15.iloc[-1]['close'] - break_price) / adr
                    else:
                        rec['pb_drift_15'] = (break_price - pb_15.iloc[-1]['close']) / adr

                pb_30 = rth[(rth['min_idx'] > break_bar) & (rth['min_idx'] <= break_bar + 30)]
                if len(pb_30) > 0:
                    if gap_dir == 'up':
                        rec['pb_drift_30'] = (pb_30.iloc[-1]['close'] - break_price) / adr
                    else:
                        rec['pb_drift_30'] = (break_price - pb_30.iloc[-1]['close']) / adr

                pb_1100 = rth[(rth['min_idx'] > break_bar) & (rth['min_idx'] <= 90)]
                if len(pb_1100) > 0:
                    if gap_dir == 'up':
                        rec['pb_drift_1100'] = (pb_1100.iloc[-1]['close'] - break_price) / adr
                    else:
                        rec['pb_drift_1100'] = (break_price - pb_1100.iloc[-1]['close']) / adr

                if gap_dir == 'up':
                    rec['pb_drift_close'] = (rth_close_price - break_price) / adr
                else:
                    rec['pb_drift_close'] = (break_price - rth_close_price) / adr

                if gap_dir == 'up':
                    rec['pb_mfe'] = (post_break['high'].max() - break_price) / adr
                    rec['pb_mae'] = (break_price - post_break['low'].min()) / adr
                else:
                    rec['pb_mfe'] = (break_price - post_break['low'].min()) / adr
                    rec['pb_mae'] = (post_break['high'].max() - break_price) / adr
    else:
        rec['vwap_held_30'] = np.nan

    records.append(rec)

print(f"\nProcessed {len(records)} days, skipped {skipped}, errors {errors}", file=sys.stderr)
df = pd.DataFrame(records)

# ══════════════════════════════════════════════════════════════════════════════
# REPORT HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def report_group(sub, label):
    n = len(sub)
    tag = low_n_tag(n)
    if n < 5:
        w(f"  {label}: N={n} (TOO FEW)")
        return
    drift = sub['drift_day'].mean()
    ci_lo, ci_hi = bootstrap_ci(sub['drift_day'].values)
    fill = sub['filled'].astype(float).mean() * 100 if sub['filled'].notna().any() else np.nan
    cl = sub['cl'].mean() if sub['cl'].notna().any() else np.nan
    mfe60 = sub['mfe_60'].mean() if 'mfe_60' in sub.columns and sub['mfe_60'].notna().any() else np.nan
    mae60 = sub['mae_60'].mean() if 'mae_60' in sub.columns and sub['mae_60'].notna().any() else np.nan
    line = f"  {label}: N={n}{tag}, Drift={drift:+.3f} {fmt_ci(ci_lo, ci_hi)}"
    if pd.notna(fill):
        line += f", Fill={fill:.0f}%"
    if pd.notna(cl):
        line += f", CL={cl:.2f}"
    if pd.notna(mfe60):
        line += f", MFE60={mfe60:.3f}"
    if pd.notna(mae60):
        line += f", MAE60={mae60:.3f}"
    w(line)

def report_od_group(sub, label):
    n = len(sub)
    tag = low_n_tag(n)
    if n < 5:
        w(f"  {label}: N={n} (TOO FEW)")
        return
    sl_rate = sub['sl_hit'].astype(float).mean() * 100 if 'sl_hit' in sub.columns and sub['sl_hit'].notna().any() else np.nan
    mfe = sub['mfe'].mean() if 'mfe' in sub.columns and sub['mfe'].notna().any() else np.nan
    mae = sub['mae'].mean() if 'mae' in sub.columns and sub['mae'].notna().any() else np.nan
    wr_parts = []
    for t in [0.25, 0.50, 0.75, 1.00]:
        col = f'wr_{t:.2f}'
        if col in sub.columns and sub[col].notna().any():
            wr = sub[col].astype(float).mean() * 100
            wr_parts.append(f"WR@{t:.2f}={wr:.0f}%")
    rr = np.nan
    if pd.notna(mfe) and 'sl_size_adr' in sub.columns:
        avg_sl = sub['sl_size_adr'].mean()
        if pd.notna(avg_sl) and avg_sl > 0:
            rr = mfe / avg_sl
    line = f"  {label}: N={n}{tag}"
    if pd.notna(sl_rate):
        line += f", SL-Hit={sl_rate:.0f}%"
    if wr_parts:
        line += f", {', '.join(wr_parts)}"
    if pd.notna(mfe):
        line += f", MFE={mfe:.3f}"
    if pd.notna(mae):
        line += f", MAE={mae:.3f}"
    if pd.notna(rr):
        line += f", R:R={rr:.2f}"
    w(line)

def report_pb_group(sub, label):
    n = len(sub)
    tag = low_n_tag(n)
    if n < 5:
        w(f"  {label}: N={n} (TOO FEW)")
        return
    parts = [f"  {label}: N={n}{tag}"]
    for col, clabel in [('pb_drift_15', 'PB15'), ('pb_drift_30', 'PB30'),
                         ('pb_drift_1100', 'PB11'), ('pb_drift_close', 'PBCl')]:
        if col in sub.columns and sub[col].notna().any():
            val = sub[col].mean()
            ci_lo, ci_hi = bootstrap_ci(sub[col].values)
            parts.append(f"{clabel}={val:+.3f} {fmt_ci(ci_lo, ci_hi)}")
    for col, clabel in [('pb_mfe', 'MFE'), ('pb_mae', 'MAE')]:
        if col in sub.columns and sub[col].notna().any():
            val = sub[col].mean()
            parts.append(f"{clabel}={val:.3f}")
    w(", ".join(parts))


# ══════════════════════════════════════════════════════════════════════════════
# OOS VALIDATION REPORT
# ══════════════════════════════════════════════════════════════════════════════
w("=" * 120)
w("DURCHLAUF 6.5 — KREUZUNGSANALYSE PHASE 2: OOS-VALIDIERUNG (HALF 2: 2024-2026)")
w("=" * 120)
w(f"Datenbasis: N={len(df)} verarbeitet")
w()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: TOP-10 IS->OOS DIRECT COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
w("=" * 120)
w("SECTION 1: TOP-10 IS->OOS DIREKTE VALIDIERUNG")
w("=" * 120)
w()

pm_valid = df[df['pm_rth_bucket'] != 'unknown']
pm_buckets = ['<10%', '10-30%', '>30%']
od_strong = df[df['od_str_bucket'] == '>0.5']
vb = df[df['late_vwap_break'] == True]
big_in_gap = df[(df['first_candle_in_gap'] == True) & (df['first_candle_size'] > 0.10)]

# Top IS findings and their OOS validation
top10 = [
    ("ComboF GapUp: PM/RTH<10% + Grosse 1.Kerze + OD>0.5 with Gap",
     +1.278,
     lambda d: d[(d['pm_rth_bucket']=='<10%') & (d['first_candle_in_gap']==True) & (d['first_candle_size']>0.10) & (d['od_str_bucket']=='>0.5') & (d['od_dir_label']=='with_gap') & (d['gap_dir']=='up')]),

    ("ComboB GapDown: PM/RTH<10% + OD>0.5 with Gap",
     +1.199,
     lambda d: d[(d['pm_rth_bucket']=='<10%') & (d['od_str_bucket']=='>0.5') & (d['od_dir_label']=='with_gap') & (d['gap_dir']=='down')]),

    ("ComboB GapUp: PM/RTH<10% + OD>0.5 with Gap",
     +1.172,
     lambda d: d[(d['pm_rth_bucket']=='<10%') & (d['od_str_bucket']=='>0.5') & (d['od_dir_label']=='with_gap') & (d['gap_dir']=='up')]),

    ("ComboF GapDown: PM/RTH<10% + Grosse 1.Kerze + OD>0.5 with Gap",
     +1.172,
     lambda d: d[(d['pm_rth_bucket']=='<10%') & (d['first_candle_in_gap']==True) & (d['first_candle_size']>0.10) & (d['od_str_bucket']=='>0.5') & (d['od_dir_label']=='with_gap') & (d['gap_dir']=='down')]),

    ("ComboE GapDown: Grosse 1.Kerze + OD>0.5 with Gap",
     +1.056,
     lambda d: d[(d['first_candle_in_gap']==True) & (d['first_candle_size']>0.10) & (d['od_str_bucket']=='>0.5') & (d['od_dir_label']=='with_gap') & (d['gap_dir']=='down')]),

    ("ComboE GapUp: Grosse 1.Kerze + OD>0.5 with Gap",
     +1.053,
     lambda d: d[(d['first_candle_in_gap']==True) & (d['first_candle_size']>0.10) & (d['od_str_bucket']=='>0.5') & (d['od_dir_label']=='with_gap') & (d['gap_dir']=='up')]),

    ("OD>0.5 with_gap + PM/RTH<10% GapDown WR@0.50",
     0.73,
     lambda d: d[(d['od_str_bucket']=='>0.5') & (d['od_dir_label']=='with_gap') & (d['pm_rth_bucket']=='<10%') & (d['gap_dir']=='down')]),

    ("OD>0.5 with_gap + PM/RTH<10% GapUp WR@0.50",
     0.70,
     lambda d: d[(d['od_str_bucket']=='>0.5') & (d['od_dir_label']=='with_gap') & (d['pm_rth_bucket']=='<10%') & (d['gap_dir']=='up')]),

    ("PM/RTH<10% + RVOL>5x GapUp",
     +0.689,
     lambda d: d[(d['pm_rth_bucket']=='<10%') & (d['rvol_bucket']=='>5x') & (d['gap_dir']=='up')]),

    ("PM/RTH<10% + RVOL>5x GapDown",
     +0.605,
     lambda d: d[(d['pm_rth_bucket']=='<10%') & (d['rvol_bucket']=='>5x') & (d['gap_dir']=='down')]),

    ("OD>0.5 with_gap + Gap>2 GapDown WR@0.50",
     0.68,
     lambda d: d[(d['od_str_bucket']=='>0.5') & (d['od_dir_label']=='with_gap') & (d['gap_bucket']=='>2') & (d['gap_dir']=='down')]),

    ("OD>0.5 with_gap + RVOL>5x GapUp WR@0.50",
     0.68,
     lambda d: d[(d['od_str_bucket']=='>0.5') & (d['od_dir_label']=='with_gap') & (d['rvol_bucket']=='>5x') & (d['gap_dir']=='up')]),

    ("ComboD GapUp: VWAP-Break + PM/RTH<10%",
     +0.678,
     lambda d: d[(d['late_vwap_break']==True) & (d['pm_rth_bucket']=='<10%') & (d['gap_dir']=='up')]),

    ("ComboA GapUp: PM/RTH<10% + Grosse 1.Kerze",
     +0.590,
     lambda d: d[(d['pm_rth_bucket']=='<10%') & (d['first_candle_in_gap']==True) & (d['first_candle_size']>0.10) & (d['gap_dir']=='up')]),

    ("ComboD GapDown: VWAP-Break + PM/RTH<10%",
     +0.550,
     lambda d: d[(d['late_vwap_break']==True) & (d['pm_rth_bucket']=='<10%') & (d['gap_dir']=='down')]),

    ("PM/RTH<10% + Gap>2 GapUp",
     +0.587,
     lambda d: d[(d['pm_rth_bucket']=='<10%') & (d['gap_bucket']=='>2') & (d['gap_dir']=='up')]),

    ("PM/RTH<10% + Gap>2 GapDown",
     +0.490,
     lambda d: d[(d['pm_rth_bucket']=='<10%') & (d['gap_bucket']=='>2') & (d['gap_dir']=='down')]),

    ("ComboH GapUp: PM/RTH>30% + OD against Gap (FADE)",
     -0.394,
     lambda d: d[(d['pm_rth_bucket']=='>30%') & (d['od_dir_label']=='against_gap') & (d['gap_dir']=='up')]),

    ("ComboH GapDown: PM/RTH>30% + OD against Gap (FADE)",
     -0.408,
     lambda d: d[(d['pm_rth_bucket']=='>30%') & (d['od_dir_label']=='against_gap') & (d['gap_dir']=='down')]),
]

w(f"{'#':>3} {'Finding':<60} {'IS':>8} {'OOS':>8} {'N':>5} {'Status':<15}")
w("-" * 110)

for i, (label, is_val, func) in enumerate(top10):
    sub = func(df)
    n = len(sub)
    tag = low_n_tag(n)

    # Determine metric type
    if 'WR@0.50' in label:
        # WR metric
        if n >= 5 and 'wr_0.50' in sub.columns:
            oos_val = sub['wr_0.50'].astype(float).mean()
            status = repl_status(is_val, oos_val)
            w(f"{i+1:3d}. {label:<60} {is_val:.0%} {oos_val:.0%} {n:5d}{tag} {status}")
        else:
            w(f"{i+1:3d}. {label:<60} {is_val:.0%} {'N/A':>8} {n:5d}{tag} N/A")
    else:
        # Drift metric
        if n >= 5:
            oos_val = sub['drift_day'].mean()
            ci_lo, ci_hi = bootstrap_ci(sub['drift_day'].values)
            status = repl_status(is_val, oos_val)
            w(f"{i+1:3d}. {label:<60} {is_val:+.3f} {oos_val:+.3f} {n:5d}{tag} {status} CI:{fmt_ci(ci_lo, ci_hi)}")
        else:
            w(f"{i+1:3d}. {label:<60} {is_val:+.3f} {'N/A':>8} {n:5d}{tag} N/A")

w()
w()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: AUFGABE 1 — PM/RTH x PARAMETER (vollstaendig)
# ══════════════════════════════════════════════════════════════════════════════
w("=" * 120)
w("AUFGABE 1 OOS: F6 PM/RTH30 RATIO x PARAMETER")
w("=" * 120)
w()

# 1a) PM/RTH x Gap-Groesse x Gap-Richtung
w("--- 1a) PM/RTH x Gap-Groesse x Gap-Richtung ---")
w()
for gd_label, gd_val in [('GapUp', 'up'), ('GapDown', 'down')]:
    w(f"  {gd_label}:")
    for pmb in pm_buckets:
        for gb in ['<1', '1-2', '>2']:
            sub = pm_valid[(pm_valid['gap_dir'] == gd_val) &
                           (pm_valid['pm_rth_bucket'] == pmb) &
                           (pm_valid['gap_bucket'] == gb)]
            report_group(sub, f"PM/RTH {pmb} + Gap {gb} ADR")
    w()

# 1b) PM/RTH x RVOL x Gap-Richtung
w("--- 1b) PM/RTH x RVOL x Gap-Richtung ---")
w()
for gd_label, gd_val in [('GapUp', 'up'), ('GapDown', 'down')]:
    w(f"  {gd_label}:")
    for pmb in pm_buckets:
        for rvb in ['<2x', '2-5x', '>5x']:
            sub = pm_valid[(pm_valid['gap_dir'] == gd_val) &
                           (pm_valid['pm_rth_bucket'] == pmb) &
                           (pm_valid['rvol_bucket'] == rvb)]
            report_group(sub, f"PM/RTH {pmb} + RVOL {rvb}")
    w()

# Kernfragen
w("--- KERNFRAGEN A1 (IS vs OOS) ---")
w()

is_vals_a1 = {
    'PM/RTH<10% GapUp Gap<1': +0.132, 'PM/RTH<10% GapUp Gap>2': +0.587,
    'PM/RTH<10% GapDown Gap<1': +0.033, 'PM/RTH<10% GapDown Gap>2': +0.490,
    'PM/RTH<10% GapUp RVOL<2x': +0.117, 'PM/RTH<10% GapUp RVOL>5x': +0.689,
    'PM/RTH<10% GapDown RVOL<2x': +0.085, 'PM/RTH<10% GapDown RVOL>5x': +0.605,
    'PM/RTH<10%+OD>0.5w GapUp': +1.172, 'PM/RTH<10%+OD>0.5w GapDown': +1.199,
}

oos_queries = {
    'PM/RTH<10% GapUp Gap<1': lambda d: d[(d['gap_dir']=='up') & (d['pm_rth_bucket']=='<10%') & (d['gap_bucket']=='<1')],
    'PM/RTH<10% GapUp Gap>2': lambda d: d[(d['gap_dir']=='up') & (d['pm_rth_bucket']=='<10%') & (d['gap_bucket']=='>2')],
    'PM/RTH<10% GapDown Gap<1': lambda d: d[(d['gap_dir']=='down') & (d['pm_rth_bucket']=='<10%') & (d['gap_bucket']=='<1')],
    'PM/RTH<10% GapDown Gap>2': lambda d: d[(d['gap_dir']=='down') & (d['pm_rth_bucket']=='<10%') & (d['gap_bucket']=='>2')],
    'PM/RTH<10% GapUp RVOL<2x': lambda d: d[(d['gap_dir']=='up') & (d['pm_rth_bucket']=='<10%') & (d['rvol_bucket']=='<2x')],
    'PM/RTH<10% GapUp RVOL>5x': lambda d: d[(d['gap_dir']=='up') & (d['pm_rth_bucket']=='<10%') & (d['rvol_bucket']=='>5x')],
    'PM/RTH<10% GapDown RVOL<2x': lambda d: d[(d['gap_dir']=='down') & (d['pm_rth_bucket']=='<10%') & (d['rvol_bucket']=='<2x')],
    'PM/RTH<10% GapDown RVOL>5x': lambda d: d[(d['gap_dir']=='down') & (d['pm_rth_bucket']=='<10%') & (d['rvol_bucket']=='>5x')],
    'PM/RTH<10%+OD>0.5w GapUp': lambda d: d[(d['gap_dir']=='up') & (d['pm_rth_bucket']=='<10%') & (d['od_str_bucket']=='>0.5') & (d['od_dir_label']=='with_gap')],
    'PM/RTH<10%+OD>0.5w GapDown': lambda d: d[(d['gap_dir']=='down') & (d['pm_rth_bucket']=='<10%') & (d['od_str_bucket']=='>0.5') & (d['od_dir_label']=='with_gap')],
}

for key in is_vals_a1:
    is_v = is_vals_a1[key]
    sub = oos_queries[key](pm_valid)
    n = len(sub)
    if n >= 5:
        oos_v = sub['drift_day'].mean()
        ci_lo, ci_hi = bootstrap_ci(sub['drift_day'].values)
        status = repl_status(is_v, oos_v)
        w(f"  {key}: IS={is_v:+.3f}, OOS={oos_v:+.3f} {fmt_ci(ci_lo, ci_hi)}, N={n}{low_n_tag(n)} -> {status}")
    else:
        w(f"  {key}: N={n} (TOO FEW)")

w()
w()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: AUFGABE 2 — F8 VWAP-BREAK
# ══════════════════════════════════════════════════════════════════════════════
w("=" * 120)
w("AUFGABE 2 OOS: F8 SPAETER VWAP-BREAK x PARAMETER")
w("=" * 120)
w()

no_vb = df[(df['vwap_held_30'] == True) & (df['late_vwap_break'] == False)]
w(f"Spaeter VWAP-Break (ja): N={len(vb)}")
w(f"VWAP held 30min + kein Break: N={len(no_vb)}")
w()

# Post-Break Metrics
w("--- POST-BREAK DRIFT ---")
for gd_label, gd_val in [('GapUp', 'up'), ('GapDown', 'down')]:
    sub = vb[vb['gap_dir'] == gd_val]
    report_pb_group(sub, f"{gd_label} ALL breaks")
w()

# F8 Combos
w("--- F8 PARAMETER-KOMBINATIONEN ---")
w()

combos_f8 = {
    'F8-A': ('Break + Gap>2 + RVOL>5x',
             lambda d: d[(d['late_vwap_break']==True) & (d['gap_bucket']=='>2') & (d['rvol_bucket']=='>5x')]),
    'F8-C': ('Break + OD with + OD>0.5',
             lambda d: d[(d['late_vwap_break']==True) & (d['od_dir_label']=='with_gap') & (d['od_str_bucket']=='>0.5')]),
    'F8-E': ('Break + PM/RTH<10%',
             lambda d: d[(d['late_vwap_break']==True) & (d['pm_rth_bucket']=='<10%')]),
    'F8-F': ('Break + PM/RTH>30%',
             lambda d: d[(d['late_vwap_break']==True) & (d['pm_rth_bucket']=='>30%')]),
    'F8-G': ('Break + Gap>2 + OD with + RVOL>2x',
             lambda d: d[(d['late_vwap_break']==True) & (d['gap_bucket']=='>2') & (d['od_dir_label']=='with_gap') & (d['rvol_bucket'].isin(['2-5x', '>5x']))]),
    'F8-I': ('Break + Grosse 1.Kerze in Gap',
             lambda d: d[(d['late_vwap_break']==True) & (d['first_candle_in_gap']==True) & (d['first_candle_size'] > 0.10)]),
    'F8-J': ('Break + Kleine 1.Kerze gegen Gap',
             lambda d: d[(d['late_vwap_break']==True) & (d['first_candle_in_gap']==False) & (d['first_candle_size'] < 0.10)]),
}

for combo_id in sorted(combos_f8.keys()):
    clabel, func = combos_f8[combo_id]
    w(f"  {combo_id}: {clabel}")
    for gd_label, gd_val in [('GapUp', 'up'), ('GapDown', 'down')]:
        sub = func(df)
        sub = sub[sub['gap_dir'] == gd_val]
        report_pb_group(sub, f"    {gd_label}")
    w()

w()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: AUFGABE 3 — F9 ERSTE KERZE
# ══════════════════════════════════════════════════════════════════════════════
w("=" * 120)
w("AUFGABE 3 OOS: F9 ERSTE KERZE x PARAMETER")
w("=" * 120)
w()

w(f"Grosse erste Kerze in Gap-Richtung (>0.10 ADR): N={len(big_in_gap)}")
w()

# Schwellen-Test
w("--- Schwellen-Test OOS ---")
w()
for threshold in [0.10, 0.20, 0.30, 0.50]:
    w(f"  Schwelle > {threshold:.2f} ADR:")
    for gd_label, gd_val in [('GapUp', 'up'), ('GapDown', 'down')]:
        sub = df[(df['first_candle_in_gap'] == True) &
                 (df['first_candle_size'] > threshold) &
                 (df['gap_dir'] == gd_val)]
        n = len(sub)
        if n >= 5:
            drift = sub['drift_day'].mean()
            fill = sub['filled'].astype(float).mean() * 100 if sub['filled'].notna().any() else np.nan
            ci_lo, ci_hi = bootstrap_ci(sub['drift_day'].values)
            w(f"    {gd_label}: N={n}{low_n_tag(n)}, Drift={drift:+.3f} {fmt_ci(ci_lo, ci_hi)}, Fill={fill:.0f}%")
        else:
            w(f"    {gd_label}: N={n} (TOO FEW)")
    w()

# x PM/RTH30
w("--- Grosse 1.Kerze x PM/RTH ---")
for gd_label, gd_val in [('GapUp', 'up'), ('GapDown', 'down')]:
    w(f"  {gd_label}:")
    for pmb in pm_buckets:
        sub = big_in_gap[(big_in_gap['gap_dir'] == gd_val) & (big_in_gap['pm_rth_bucket'] == pmb)]
        report_group(sub, f"Grosse 1.Kerze + PM/RTH {pmb}")
    w()

w()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: AUFGABE 4 — F7 OD-STAERKE
# ══════════════════════════════════════════════════════════════════════════════
w("=" * 120)
w("AUFGABE 4 OOS: F7 OD-STAERKE x PARAMETER")
w("=" * 120)
w()

w(f"OD>0.5 ADR: N={len(od_strong)}")
w()

# Key WR combos from IS
w("--- WR@0.50 Kernkombinationen ---")
w()

wr_combos = [
    ("GapDown+ODwith+PM<10%", 0.73,
     lambda d: d[(d['gap_dir']=='down') & (d['od_dir_label']=='with_gap') & (d['pm_rth_bucket']=='<10%')]),
    ("GapUp+ODwith+PM<10%", 0.70,
     lambda d: d[(d['gap_dir']=='up') & (d['od_dir_label']=='with_gap') & (d['pm_rth_bucket']=='<10%')]),
    ("GapDown+ODwith+Gap>2", 0.68,
     lambda d: d[(d['gap_dir']=='down') & (d['od_dir_label']=='with_gap') & (d['gap_bucket']=='>2')]),
    ("GapUp+ODwith+RVOL>5x", 0.68,
     lambda d: d[(d['gap_dir']=='up') & (d['od_dir_label']=='with_gap') & (d['rvol_bucket']=='>5x')]),
    ("GapDown+ODwith+RVOL>5x", 0.67,
     lambda d: d[(d['gap_dir']=='down') & (d['od_dir_label']=='with_gap') & (d['rvol_bucket']=='>5x')]),
    ("GapDown+ODwith+10-30%", 0.65,
     lambda d: d[(d['gap_dir']=='down') & (d['od_dir_label']=='with_gap') & (d['pm_rth_bucket']=='10-30%')]),
    ("GapUp+ODwith+Gap>2", 0.65,
     lambda d: d[(d['gap_dir']=='up') & (d['od_dir_label']=='with_gap') & (d['gap_bucket']=='>2')]),
    ("GapDown+ODagainst+RVOL>5x", 0.64,
     lambda d: d[(d['gap_dir']=='down') & (d['od_dir_label']=='against_gap') & (d['rvol_bucket']=='>5x')]),
    ("GapUp+ODagainst+RVOL>5x", 0.64,
     lambda d: d[(d['gap_dir']=='up') & (d['od_dir_label']=='against_gap') & (d['rvol_bucket']=='>5x')]),
]

w(f"  {'Combo':<35} {'IS':>8} {'OOS':>8} {'N':>5} {'Status':<15}")
w(f"  {'-'*80}")
for label, is_wr, func in wr_combos:
    sub = func(od_strong)
    n = len(sub)
    if n >= 5 and 'wr_0.50' in sub.columns:
        oos_wr = sub['wr_0.50'].astype(float).mean()
        status = repl_status(is_wr, oos_wr)
        w(f"  {label:<35} {is_wr:.0%} {oos_wr:.0%} {n:5d}{low_n_tag(n)} {status}")
    else:
        w(f"  {label:<35} {is_wr:.0%} {'N/A':>8} {n:5d}{low_n_tag(n)} N/A")

w()

# Full OD table
w("--- Vollstaendige WR@0.50 Tabelle OD>0.5 ---")
for gd_label, gd_val in [('GapUp', 'up'), ('GapDown', 'down')]:
    for od_dir in ['with_gap', 'against_gap']:
        sub = od_strong[(od_strong['gap_dir'] == gd_val) & (od_strong['od_dir_label'] == od_dir)]
        report_od_group(sub, f"{gd_label} + OD {od_dir}")
w()
w()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: AUFGABE 5 — MULTI-FACTOR
# ══════════════════════════════════════════════════════════════════════════════
w("=" * 120)
w("AUFGABE 5 OOS: SIGNALKOMBINATIONEN (MULTI-FACTOR)")
w("=" * 120)
w()

combos = {
    'A': ('PM/RTH<10% + Grosse 1.Kerze in Gap',
          lambda d: d[(d['pm_rth_bucket']=='<10%') & (d['first_candle_in_gap']==True) & (d['first_candle_size']>0.10)]),
    'B': ('PM/RTH<10% + OD>0.5 with Gap',
          lambda d: d[(d['pm_rth_bucket']=='<10%') & (d['od_str_bucket']=='>0.5') & (d['od_dir_label']=='with_gap')]),
    'C': ('PM/RTH<10% + RVOL>5x',
          lambda d: d[(d['pm_rth_bucket']=='<10%') & (d['rvol_bucket']=='>5x')]),
    'D': ('Spaeter VWAP-Break + PM/RTH<10%',
          lambda d: d[(d['late_vwap_break']==True) & (d['pm_rth_bucket']=='<10%')]),
    'E': ('Grosse 1.Kerze in Gap + OD>0.5 with Gap',
          lambda d: d[(d['first_candle_in_gap']==True) & (d['first_candle_size']>0.10) & (d['od_str_bucket']=='>0.5') & (d['od_dir_label']=='with_gap')]),
    'F': ('PM/RTH<10% + Grosse 1.Kerze + OD>0.5 with Gap (TRIPLE)',
          lambda d: d[(d['pm_rth_bucket']=='<10%') & (d['first_candle_in_gap']==True) & (d['first_candle_size']>0.10) & (d['od_str_bucket']=='>0.5') & (d['od_dir_label']=='with_gap')]),
    'G': ('PM/RTH<10% + RVOL<2x',
          lambda d: d[(d['pm_rth_bucket']=='<10%') & (d['rvol_bucket']=='<2x')]),
    'H': ('PM/RTH>30% + OD against Gap (FADE)',
          lambda d: d[(d['pm_rth_bucket']=='>30%') & (d['od_dir_label']=='against_gap')]),
}

# IS values for comparison
is_combos = {
    'A': {'up': +0.590, 'down': +0.395},
    'B': {'up': +1.172, 'down': +1.199},
    'C': {'up': +0.689, 'down': +0.605},
    'D': {'up': +0.678, 'down': +0.550},
    'E': {'up': +1.053, 'down': +1.056},
    'F': {'up': +1.278, 'down': +1.172},
    'G': {'up': +0.117, 'down': +0.085},
    'H': {'up': -0.394, 'down': -0.408},
}

w(f"  {'Combo':<5} {'Label':<55} {'Dir':<8} {'IS':>8} {'OOS':>8} {'N':>5} {'Status':<15}")
w(f"  {'-'*110}")

for combo_id in sorted(combos.keys()):
    label, func = combos[combo_id]
    for gd_label, gd_val in [('GapUp', 'up'), ('GapDown', 'down')]:
        sub = func(df)
        sub = sub[sub['gap_dir'] == gd_val]
        n = len(sub)
        is_v = is_combos.get(combo_id, {}).get(gd_val, np.nan)
        tag = low_n_tag(n)

        if n >= 5:
            oos_v = sub['drift_day'].mean()
            ci_lo, ci_hi = bootstrap_ci(sub['drift_day'].values)
            status = repl_status(is_v, oos_v) if not np.isnan(is_v) else "N/A"
            w(f"  {combo_id:<5} {label[:55]:<55} {gd_label:<8} {is_v:+.3f} {oos_v:+.3f} {n:5d}{tag} {status}")
        else:
            w(f"  {combo_id:<5} {label[:55]:<55} {gd_label:<8} {is_v:+.3f} {'N/A':>8} {n:5d}{tag} N/A")

w()
w()

# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
w("=" * 120)
w("ZUSAMMENFASSUNG: WAS REPLIZIERT OOS?")
w("=" * 120)
w()

w("LEGENDE: ++ = OOS staerker, + = repliziert (>=50%), ~ = teilweise, - = nicht repliziert, X = umgekehrt")
w()

# Collect all validated results
all_validated = []

# Multi-Factor Combos
for combo_id in sorted(combos.keys()):
    label, func = combos[combo_id]
    for gd_label, gd_val in [('GapUp', 'up'), ('GapDown', 'down')]:
        sub = func(df)
        sub = sub[sub['gap_dir'] == gd_val]
        n = len(sub)
        is_v = is_combos.get(combo_id, {}).get(gd_val, np.nan)
        if n >= 10 and not np.isnan(is_v):
            oos_v = sub['drift_day'].mean()
            status = repl_status(is_v, oos_v)
            ratio = oos_v / is_v if abs(is_v) > 0.001 else 0
            symbol = "++" if ratio > 1.0 else ("+" if status == "REPLIZIERT" else ("~" if status == "TEILWEISE" else ("-" if status == "NICHT REPLIZIERT" else "X")))
            all_validated.append((f"Combo{combo_id} {gd_label}", is_v, oos_v, n, status, symbol))

# Kernfragen A1
for key in is_vals_a1:
    is_v = is_vals_a1[key]
    sub = oos_queries[key](pm_valid)
    n = len(sub)
    if n >= 10:
        oos_v = sub['drift_day'].mean()
        status = repl_status(is_v, oos_v)
        ratio = oos_v / is_v if abs(is_v) > 0.001 else 0
        symbol = "++" if ratio > 1.0 else ("+" if status == "REPLIZIERT" else ("~" if status == "TEILWEISE" else ("-" if status == "NICHT REPLIZIERT" else "X")))
        all_validated.append((f"A1: {key}", is_v, oos_v, n, status, symbol))

# WR combos
for label, is_wr, func in wr_combos:
    sub = func(od_strong)
    n = len(sub)
    if n >= 10 and 'wr_0.50' in sub.columns:
        oos_wr = sub['wr_0.50'].astype(float).mean()
        status = repl_status(is_wr, oos_wr)
        ratio = oos_wr / is_wr if abs(is_wr) > 0.001 else 0
        symbol = "++" if ratio > 1.0 else ("+" if status == "REPLIZIERT" else ("~" if status == "TEILWEISE" else ("-" if status == "NICHT REPLIZIERT" else "X")))
        all_validated.append((f"A4: {label} WR@50", is_wr, oos_wr, n, status, symbol))

# Sort by OOS effect size
all_validated.sort(key=lambda x: abs(x[2]), reverse=True)

w(f"  {'S':>2} {'Finding':<55} {'IS':>8} {'OOS':>8} {'N':>5} {'Status':<15}")
w(f"  {'-'*100}")
for label, is_v, oos_v, n, status, symbol in all_validated:
    tag = low_n_tag(n)
    if 'WR@50' in label:
        w(f"  {symbol:>2} {label:<55} {is_v:.0%} {oos_v:.0%} {n:5d}{tag} {status}")
    else:
        w(f"  {symbol:>2} {label:<55} {is_v:+.3f} {oos_v:+.3f} {n:5d}{tag} {status}")

w()

# Replicated count
repl = [x for x in all_validated if x[4] in ('REPLIZIERT',)]
teil = [x for x in all_validated if x[4] == 'TEILWEISE']
nicht = [x for x in all_validated if x[4] == 'NICHT REPLIZIERT']
umgek = [x for x in all_validated if x[4] == 'UMGEKEHRT']
w(f"REPLIZIERT: {len(repl)}/{len(all_validated)} ({len(repl)/len(all_validated)*100:.0f}%)")
w(f"TEILWEISE: {len(teil)}")
w(f"NICHT REPLIZIERT: {len(nicht)}")
w(f"UMGEKEHRT: {len(umgek)}")
w()

# Key trading takeaways
w("=" * 120)
w("TRADING-RELEVANTE ERKENNTNISSE")
w("=" * 120)
w()

# Best combos that replicated
repl_strong = [x for x in all_validated if x[4] == 'REPLIZIERT' and abs(x[2]) >= 0.3]
if repl_strong:
    w("STAERKSTE REPLIZIERTE SIGNALE (OOS Drift >= 0.3 ADR):")
    for label, is_v, oos_v, n, status, symbol in repl_strong:
        tag = low_n_tag(n)
        w(f"  {label}: IS={is_v:+.3f}, OOS={oos_v:+.3f}, N={n}{tag}")
    w()

# Best WR combos that replicated
wr_repl = [x for x in all_validated if 'WR@50' in x[0] and x[4] == 'REPLIZIERT']
if wr_repl:
    w("REPLIZIERTE WR@0.50 COMBOS:")
    for label, is_v, oos_v, n, status, symbol in wr_repl:
        tag = low_n_tag(n)
        w(f"  {label}: IS={is_v:.0%}, OOS={oos_v:.0%}, N={n}{tag}")
    w()

# Write output
report = "\n".join(out)
with open('results/cross_analysis_h2_validation.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print(report)
print(f"\nSaved to results/cross_analysis_h2_validation.txt", file=sys.stderr)
