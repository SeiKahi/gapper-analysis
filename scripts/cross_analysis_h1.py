"""
Durchlauf 6.5 — Kreuzungsanalyse Top-Findings mit Parametern (Phase 1: Half 1)
Half 1: 2021-02-21 bis 2023-12-31

Aufgabe 1: F6 PM/RTH30 x Parameter
Aufgabe 2: F8 Spaeter VWAP-Break x Parameter + Post-Break Metrics + Combos
Aufgabe 3: F9 Erste Kerze x Parameter + Schwellen-Test
Aufgabe 4: F7 OD-Staerke x Parameter
Aufgabe 5: Multi-Factor Signalkombinationen
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

out = []
def w(s=""): out.append(s)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD AND PREPARE DATA
# ══════════════════════════════════════════════════════════════════════════════
print("Loading metadata...", file=sys.stderr)
meta = pd.read_parquet('data/metadata/metadata_master.parquet')
meta['date'] = pd.to_datetime(meta['date'])

h1 = meta[(meta['date'] >= '2021-02-21') & (meta['date'] <= '2023-12-31')].copy()
h1 = h1[h1['adr_10'].notna()].copy()
print(f"Half 1: {len(h1)} gapper days", file=sys.stderr)

# ── Process each gapper day to get 1min-derived features ──────────────────
records = []
skipped = 0
errors = 0

for _, row in tqdm(h1.iterrows(), total=len(h1), desc="Processing H1", file=sys.stderr):
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

    # ── Gap-Groesse Bucket (P1) ──
    gs = rec['gap_size_adr']
    rec['gap_bucket'] = '<1' if gs < 1 else ('1-2' if gs < 2 else '>2')

    # ── RVOL Bucket (P2) ──
    rv = rec['rvol_30']
    if pd.notna(rv):
        rec['rvol_bucket'] = '<2x' if rv < 2 else ('2-5x' if rv < 5 else '>5x')
    else:
        rec['rvol_bucket'] = 'unknown'

    # ── Gap-Richtung (P3) ──
    rec['gap_up'] = gap_dir == 'up'

    # ── PM/RTH30 Ratio ──
    pm_vol = pm['volume'].sum() if len(pm) > 0 else 0
    rth_30 = rth[rth['min_idx'] < 30]
    rth_30_vol = rth_30['volume'].sum() if len(rth_30) > 0 else 0
    rec['pm_rth_ratio'] = pm_vol / rth_30_vol if rth_30_vol > 0 else np.nan

    r = rec['pm_rth_ratio']
    if pd.notna(r):
        rec['pm_rth_bucket'] = '<10%' if r < 0.10 else ('10-30%' if r < 0.30 else '>30%')
    else:
        rec['pm_rth_bucket'] = 'unknown'

    # ── Opening Drive (first 5min) → P4 OD-Staerke, P5 OD-Richtung ──
    od_bars = rth[rth['min_idx'] < 5]
    if len(od_bars) >= 5:
        od_open = od_bars.iloc[0]['open']
        od_close = od_bars.iloc[-1]['close']
        od_high = od_bars['high'].max()
        od_low = od_bars['low'].min()
        od_move = abs(od_close - od_open) / adr  # OD-Staerke
        od_long = od_close > od_open

        rec['od_strength'] = od_move
        rec['od_str_bucket'] = '<0.2' if od_move < 0.2 else ('0.2-0.5' if od_move < 0.5 else '>0.5')
        rec['od_long'] = od_long

        # OD direction relative to gap
        if gap_dir == 'up':
            rec['od_with_gap'] = od_long
        else:
            rec['od_with_gap'] = not od_long
        rec['od_dir_label'] = 'with_gap' if rec['od_with_gap'] else 'against_gap'

        # SL from 5min H/L
        sl_price = od_low if od_long else od_high
        entry_price = od_close
        rec['sl_size_adr'] = abs(entry_price - sl_price) / adr
        rec['od_entry'] = entry_price
        rec['od_sl'] = sl_price

        # Track SL hit, targets, MFE/MAE from bar 5 to bar 90
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

    # ── Erste Kerze (F9) ──
    first_bar = rth.iloc[0]
    candle_body = first_bar['close'] - first_bar['open']
    candle_range = (first_bar['high'] - first_bar['low']) / adr
    candle_size = abs(candle_body) / adr
    candle_dir = 'up' if candle_body > 0 else 'down'
    in_gap_dir = (gap_dir == 'up' and candle_dir == 'up') or (gap_dir == 'down' and candle_dir == 'down')

    rec['first_candle_size'] = candle_size
    rec['first_candle_dir'] = candle_dir
    rec['first_candle_in_gap'] = in_gap_dir

    # ── Day drift and metrics ──
    rth_close_price = rth.iloc[-1]['close']
    drift_raw = rth_close_price - open_price
    if gap_dir == 'up':
        rec['drift_day'] = drift_raw / adr
    else:
        rec['drift_day'] = -drift_raw / adr  # Normalize: positive = with gap

    rec['drift_day_signed'] = drift_raw / adr  # Keep signed version too

    # Fill
    if pd.notna(prev_close):
        if gap_dir == 'up':
            rec['filled'] = rth['low'].min() <= prev_close
        else:
            rec['filled'] = rth['high'].max() >= prev_close
    else:
        rec['filled'] = np.nan

    # Close Location
    day_high = rth['high'].max()
    day_low = rth['low'].min()
    rec['cl'] = (rth_close_price - day_low) / (day_high - day_low) if (day_high - day_low) > 0 else np.nan

    # MFE/MAE from open (full day, in gap direction)
    if gap_dir == 'up':
        rec['mfe_day'] = (rth['high'].max() - open_price) / adr
        rec['mae_day'] = (open_price - rth['low'].min()) / adr
    else:
        rec['mfe_day'] = (open_price - rth['low'].min()) / adr
        rec['mae_day'] = (rth['high'].max() - open_price) / adr

    # ── MFE/MAE at 60min from open ──
    first_60 = rth[rth['min_idx'] < 60]
    if len(first_60) > 0:
        if gap_dir == 'up':
            rec['mfe_60'] = (first_60['high'].max() - open_price) / adr
            rec['mae_60'] = (open_price - first_60['low'].min()) / adr
        else:
            rec['mfe_60'] = (open_price - first_60['low'].min()) / adr
            rec['mae_60'] = (first_60['high'].max() - open_price) / adr

    # ── F8: VWAP-Break Analysis ──
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

        # Check VWAP NOT broken in first 30min
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

        # If VWAP held 30min, check for late break (30-90min)
        if not vwap_broken_30:
            bars_30_90 = rth_v[(rth_v['min_idx'] >= 30) & (rth_v['min_idx'] <= 90)]
            consec = 0
            for _, bar in bars_30_90.iterrows():
                if pd.isna(bar.get('vwap', np.nan)):
                    consec = 0
                    continue
                # Break = price crosses VWAP in gap direction (continuation)
                if gap_dir == 'up':
                    # For GapUp: VWAP was above price (held), break = close above VWAP
                    # Wait... need to clarify: "spaeter VWAP-Break" means
                    # VWAP held as resistance/support for 30min, then breaks THROUGH
                    # For GapUp: price was BELOW VWAP for 30min, then breaks ABOVE
                    # Actually from the D6.0 code: for GapUp, break = close < vwap
                    # That means for GapUp, VWAP is resistance FROM ABOVE and break = price goes below
                    # No wait - re-reading the D6.0 code:
                    # For gap_dir == 'down': break = close > vwap (price goes above VWAP = against gap)
                    # For gap_dir == 'up': break = close < vwap (price goes below VWAP = against gap)
                    # So "VWAP Break" in D6.0 means AGAINST gap break
                    # But D6.0 Finding says: "Spaeter VWAP-Break GapUp → Drift +0.595"
                    # That's POSITIVE drift for GapUp = continuation
                    # Hmm, that's confusing. Let me re-read...
                    # Actually the D6.0 late break definition checks if price crosses
                    # to the OTHER side of VWAP. For GapUp, if VWAP holds (price stays
                    # above VWAP), then "break" means price drops below VWAP.
                    # But the drift is positive??? That doesn't make sense.
                    #
                    # Let me re-think: For GapUp stocks, VWAP starts near/below open.
                    # If price stays ABOVE VWAP for 30min = VWAP holds as SUPPORT.
                    # "VWAP not broken" in 30min = support held.
                    # Then "late break" = price drops below VWAP = SUPPORT breaks.
                    # But that should be bearish...
                    #
                    # Actually from D6.0 code line 241: gap_dir=='down', close > vwap → consec++
                    # That means for GapDown: "break" = close crosses ABOVE VWAP
                    # And line 248: gap_dir=='up' (else), close < vwap → consec++
                    # For GapUp: "break" = close crosses BELOW VWAP
                    #
                    # BUT the result shows GapUp late break = +0.595 drift.
                    # That means even though VWAP was broken (price went below), the
                    # DAY DRIFT was still strongly positive.
                    #
                    # Hmm wait - OR the "vwap_broken_30=False" check for GapUp means:
                    # In first 30min, price never had 3 consecutive closes below VWAP.
                    # So for GapUp: VWAP held as support. Price stayed above.
                    # Then late_vwap_break (30-90min): price DOES break below VWAP.
                    # But drift is +0.595??? That means they recover and close higher???
                    #
                    # Actually, I think "spaeter VWAP-Break" in the TRADING context means:
                    # The VWAP acts as a barrier, holds for 30min, then price breaks
                    # THROUGH the barrier IN THE GAP DIRECTION (continuation).
                    #
                    # For GapUp: VWAP could be BELOW price. If VWAP is support and
                    # price stays above, that's "held". Then later the stock goes even
                    # higher = continuation. But that's not a "break".
                    #
                    # OR: For GapUp, sometimes price is below VWAP at open (didn't hold above).
                    # VWAP acts as resistance. "Not broken in 30min" = price couldn't get above VWAP.
                    # Then later (30-90min) it breaks ABOVE VWAP = continuation breakout.
                    # THAT would explain +0.595 drift!
                    #
                    # Let me re-read the D6.0 code more carefully:
                    # Line 233-252: checks if VWAP IS broken in first 30min.
                    # For GapDown: break = close > vwap (price above VWAP = against gap)
                    # For GapUp: break = close < vwap (price below VWAP = against gap)
                    #
                    # So "vwap_broken_30 = True" means price crossed to AGAINST-gap side in 30min
                    # "vwap_broken_30 = False" means price STAYED on gap side for 30min
                    #
                    # Then "late_vwap_break" (lines 257-278) checks 30-90min for same condition.
                    # For GapDown: close > vwap (price goes against gap = above VWAP)
                    # For GapUp: close < vwap (price goes against gap = below VWAP)
                    #
                    # So late_vwap_break means: price was on gap side for 30min, then crossed against.
                    # For GapUp: price was above VWAP for 30min, then dropped below.
                    # Drift +0.595 = DESPITE dropping below VWAP, day ends strongly UP.
                    # That's a "shakeout" pattern - fake breakdown then continuation.
                    #
                    # For GapDown: price was below VWAP for 30min, then popped above.
                    # Drift -0.444 = DESPITE popping above VWAP, day ends strongly DOWN.
                    # Same pattern - fake breakout against gap, then continuation.
                    #
                    # This actually makes sense! VWAP holds as barrier (support for GapUp,
                    # resistance for GapDown). Late break against gap = shakeout.
                    # Price recovers and continues in gap direction.
                    #
                    # For Durchlauf 6.5, the user says:
                    # "VWAP haelt 30min als Barriere, dann Break → Continuation ~0.5 ADR"
                    # This confirms: VWAP holds as barrier, then break = continuation.
                    # The "break" here seems to be IN GAP direction (continuation break).
                    #
                    # Let me reconsider: Maybe the code is checking for break IN gap direction:
                    # For GapDown: close > vwap means price is ABOVE VWAP = AGAINST gap.
                    # Hmm that's against gap, not with gap.
                    #
                    # I think the simplest interpretation consistent with results is:
                    # "VWAP held 30min" = VWAP acted as barrier (price stayed on gap side)
                    # "Late break" = the barrier was eventually broken (price crossed to other side)
                    # Despite this break against gap, the DAY drift is still strongly with gap.
                    # Meaning: the late VWAP break against gap is a FALSE SIGNAL - you should
                    # still expect gap-direction continuation.
                    #
                    # OR more likely: I'm overcomplicating this. Let me just use the exact
                    # same logic as D6.0 code and match the exact same definition.
                    beyond = bar['close'] < bar['vwap']
                else:
                    beyond = bar['close'] > bar['vwap']
                consec = consec + 1 if beyond else 0
                if consec >= 3:
                    rec['late_vwap_break'] = True
                    rec['vwap_break_bar'] = bar['min_idx']
                    rec['vwap_break_price'] = bar['close']
                    break

        # Post-break metrics (if late break happened)
        if rec['late_vwap_break'] and pd.notna(rec['vwap_break_bar']):
            break_bar = int(rec['vwap_break_bar'])
            break_price = rec['vwap_break_price']

            # Post-break bars
            post_break = rth[rth['min_idx'] > break_bar]

            if len(post_break) > 0:
                # Drift at various horizons after break
                # +15min
                pb_15 = rth[(rth['min_idx'] > break_bar) & (rth['min_idx'] <= break_bar + 15)]
                if len(pb_15) > 0:
                    # Use signed drift (in gap direction)
                    if gap_dir == 'up':
                        rec['pb_drift_15'] = (pb_15.iloc[-1]['close'] - break_price) / adr
                    else:
                        rec['pb_drift_15'] = (break_price - pb_15.iloc[-1]['close']) / adr

                # +30min
                pb_30 = rth[(rth['min_idx'] > break_bar) & (rth['min_idx'] <= break_bar + 30)]
                if len(pb_30) > 0:
                    if gap_dir == 'up':
                        rec['pb_drift_30'] = (pb_30.iloc[-1]['close'] - break_price) / adr
                    else:
                        rec['pb_drift_30'] = (break_price - pb_30.iloc[-1]['close']) / adr

                # Until 11:00 (bar 90)
                pb_1100 = rth[(rth['min_idx'] > break_bar) & (rth['min_idx'] <= 90)]
                if len(pb_1100) > 0:
                    if gap_dir == 'up':
                        rec['pb_drift_1100'] = (pb_1100.iloc[-1]['close'] - break_price) / adr
                    else:
                        rec['pb_drift_1100'] = (break_price - pb_1100.iloc[-1]['close']) / adr

                # Until close
                if gap_dir == 'up':
                    rec['pb_drift_close'] = (rth_close_price - break_price) / adr
                else:
                    rec['pb_drift_close'] = (break_price - rth_close_price) / adr

                # Post-break MFE/MAE (in gap direction)
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
def report_group(sub, label, metrics='standard'):
    """Report standard metrics for a subgroup."""
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
    """Report OD-specific metrics for a subgroup."""
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
    """Report post-break metrics for F8 subgroup."""
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
# AUFGABE 1: F6 PM/RTH30 RATIO x PARAMETER
# ══════════════════════════════════════════════════════════════════════════════
w("=" * 110)
w("DURCHLAUF 6.5 — KREUZUNGSANALYSE PHASE 1 (HALF 1: 2021-2023)")
w("=" * 110)
w(f"Datenbasis: N={len(df)} verarbeitet")
w()

w("=" * 110)
w("AUFGABE 1: F6 PM/RTH30 RATIO x PARAMETER")
w("=" * 110)
w()

pm_valid = df[df['pm_rth_bucket'] != 'unknown']
pm_buckets = ['<10%', '10-30%', '>30%']

# a) PM/RTH x Gap-Groesse (P1) x Gap-Richtung (P3)
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

# b) PM/RTH x RVOL (P2) x Gap-Richtung (P3)
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

# c) PM/RTH x OD-Staerke (P4) x Gap-Richtung (P3)
w("--- 1c) PM/RTH x OD-Staerke x Gap-Richtung ---")
w()
od_valid = pm_valid[pm_valid['od_str_bucket'] != 'unknown']
for gd_label, gd_val in [('GapUp', 'up'), ('GapDown', 'down')]:
    w(f"  {gd_label}:")
    for pmb in pm_buckets:
        for odb in ['<0.2', '0.2-0.5', '>0.5']:
            sub = od_valid[(od_valid['gap_dir'] == gd_val) &
                           (od_valid['pm_rth_bucket'] == pmb) &
                           (od_valid['od_str_bucket'] == odb)]
            report_group(sub, f"PM/RTH {pmb} + OD {odb}")
    w()

# d) PM/RTH x OD-Richtung (P5) x Gap-Richtung (P3)
w("--- 1d) PM/RTH x OD-Richtung x Gap-Richtung ---")
w()
for gd_label, gd_val in [('GapUp', 'up'), ('GapDown', 'down')]:
    w(f"  {gd_label}:")
    for pmb in pm_buckets:
        for od_dir in ['with_gap', 'against_gap']:
            sub = pm_valid[(pm_valid['gap_dir'] == gd_val) &
                           (pm_valid['pm_rth_bucket'] == pmb) &
                           (pm_valid['od_dir_label'] == od_dir)]
            report_group(sub, f"PM/RTH {pmb} + OD {od_dir}")
    w()

# Kernfragen
w("--- KERNFRAGEN AUFGABE 1 ---")
w()
# PM/RTH<10% bei grossen Gaps (>2 ADR)
for gd_label, gd_val in [('GapUp', 'up'), ('GapDown', 'down')]:
    sub_big = pm_valid[(pm_valid['gap_dir'] == gd_val) &
                        (pm_valid['pm_rth_bucket'] == '<10%') &
                        (pm_valid['gap_bucket'] == '>2')]
    sub_small = pm_valid[(pm_valid['gap_dir'] == gd_val) &
                          (pm_valid['pm_rth_bucket'] == '<10%') &
                          (pm_valid['gap_bucket'] == '<1')]
    d_big = sub_big['drift_day'].mean() if len(sub_big) >= 5 else np.nan
    d_small = sub_small['drift_day'].mean() if len(sub_small) >= 5 else np.nan
    w(f"  PM/RTH<10% {gd_label}: Gap<1={d_small:+.3f}(N={len(sub_small)}), Gap>2={d_big:+.3f}(N={len(sub_big)})")

w()
# PM/RTH bei RVOL>5x vs RVOL<2x
for gd_label, gd_val in [('GapUp', 'up'), ('GapDown', 'down')]:
    sub_hi = pm_valid[(pm_valid['gap_dir'] == gd_val) &
                       (pm_valid['pm_rth_bucket'] == '<10%') &
                       (pm_valid['rvol_bucket'] == '>5x')]
    sub_lo = pm_valid[(pm_valid['gap_dir'] == gd_val) &
                       (pm_valid['pm_rth_bucket'] == '<10%') &
                       (pm_valid['rvol_bucket'] == '<2x')]
    d_hi = sub_hi['drift_day'].mean() if len(sub_hi) >= 5 else np.nan
    d_lo = sub_lo['drift_day'].mean() if len(sub_lo) >= 5 else np.nan
    w(f"  PM/RTH<10% {gd_label}: RVOL<2x={d_lo:+.3f}(N={len(sub_lo)}), RVOL>5x={d_hi:+.3f}(N={len(sub_hi)})")

w()
# Verstaerkt starker OD with Gap den PM/RTH Effekt?
for gd_label, gd_val in [('GapUp', 'up'), ('GapDown', 'down')]:
    sub_od_strong = pm_valid[(pm_valid['gap_dir'] == gd_val) &
                              (pm_valid['pm_rth_bucket'] == '<10%') &
                              (pm_valid['od_with_gap'] == True) &
                              (pm_valid['od_str_bucket'] == '>0.5')]
    sub_base = pm_valid[(pm_valid['gap_dir'] == gd_val) &
                         (pm_valid['pm_rth_bucket'] == '<10%')]
    d_combo = sub_od_strong['drift_day'].mean() if len(sub_od_strong) >= 5 else np.nan
    d_base = sub_base['drift_day'].mean() if len(sub_base) >= 5 else np.nan
    w(f"  {gd_label}: PM/RTH<10% alone={d_base:+.3f}(N={len(sub_base)}), +OD>0.5 with_gap={d_combo:+.3f}(N={len(sub_od_strong)})")

w()
w()

# ══════════════════════════════════════════════════════════════════════════════
# AUFGABE 2: F8 SPAETER VWAP-BREAK x PARAMETER
# ══════════════════════════════════════════════════════════════════════════════
w("=" * 110)
w("AUFGABE 2: F8 SPAETER VWAP-BREAK x PARAMETER")
w("=" * 110)
w()

vb = df[df['late_vwap_break'] == True]
no_vb = df[(df['vwap_held_30'] == True) & (df['late_vwap_break'] == False)]

w(f"Spaeter VWAP-Break (ja): N={len(vb)}")
w(f"VWAP held 30min + kein Break: N={len(no_vb)}")
w()

# a) VWAP-Break x Gap-Groesse x Gap-Richtung
w("--- 2a) VWAP-Break x Gap-Groesse x Gap-Richtung ---")
w()
for gd_label, gd_val in [('GapUp', 'up'), ('GapDown', 'down')]:
    w(f"  {gd_label}:")
    for brk_label, brk_df in [('Break=JA', vb), ('Break=NEIN', no_vb)]:
        for gb in ['<1', '1-2', '>2']:
            sub = brk_df[(brk_df['gap_dir'] == gd_val) & (brk_df['gap_bucket'] == gb)]
            report_group(sub, f"{brk_label} + Gap {gb} ADR")
    w()

# b) VWAP-Break x RVOL x Gap-Richtung
w("--- 2b) VWAP-Break x RVOL x Gap-Richtung ---")
w()
for gd_label, gd_val in [('GapUp', 'up'), ('GapDown', 'down')]:
    w(f"  {gd_label}:")
    for brk_label, brk_df in [('Break=JA', vb), ('Break=NEIN', no_vb)]:
        for rvb in ['<2x', '2-5x', '>5x']:
            sub = brk_df[(brk_df['gap_dir'] == gd_val) & (brk_df['rvol_bucket'] == rvb)]
            report_group(sub, f"{brk_label} + RVOL {rvb}")
    w()

# c) VWAP-Break x OD-Staerke x Gap-Richtung
w("--- 2c) VWAP-Break x OD-Staerke x Gap-Richtung ---")
w()
for gd_label, gd_val in [('GapUp', 'up'), ('GapDown', 'down')]:
    w(f"  {gd_label}:")
    for brk_label, brk_df in [('Break=JA', vb), ('Break=NEIN', no_vb)]:
        for odb in ['<0.2', '0.2-0.5', '>0.5']:
            sub = brk_df[(brk_df['gap_dir'] == gd_val) & (brk_df['od_str_bucket'] == odb)]
            report_group(sub, f"{brk_label} + OD {odb}")
    w()

# d) VWAP-Break x PM/RTH30 x Gap-Richtung
w("--- 2d) VWAP-Break x PM/RTH30 x Gap-Richtung ---")
w()
for gd_label, gd_val in [('GapUp', 'up'), ('GapDown', 'down')]:
    w(f"  {gd_label}:")
    for brk_label, brk_df in [('Break=JA', vb), ('Break=NEIN', no_vb)]:
        for pmb in pm_buckets:
            sub = brk_df[(brk_df['gap_dir'] == gd_val) & (brk_df['pm_rth_bucket'] == pmb)]
            report_group(sub, f"{brk_label} + PM/RTH {pmb}")
    w()

# Post-Break Metrics
w("--- F8 POST-BREAK DRIFT (ab Zeitpunkt des Breaks) ---")
w()
for gd_label, gd_val in [('GapUp', 'up'), ('GapDown', 'down')]:
    sub = vb[vb['gap_dir'] == gd_val]
    report_pb_group(sub, f"{gd_label} ALL breaks")
w()

# Parameter-Kombinationen F8
w("--- F8 PARAMETER-KOMBINATIONEN ---")
w()

combos_f8 = {
    'F8-A': lambda d: d[(d['late_vwap_break']==True) & (d['gap_bucket']=='>2') & (d['rvol_bucket']=='>5x')],
    'F8-B': lambda d: d[(d['late_vwap_break']==True) & (d['gap_bucket']=='>2') & (d['rvol_bucket']=='<2x')],
    'F8-C': lambda d: d[(d['late_vwap_break']==True) & (d['od_dir_label']=='with_gap') & (d['od_str_bucket']=='>0.5')],
    'F8-D': lambda d: d[(d['late_vwap_break']==True) & (d['od_dir_label']=='against_gap') & (d['od_str_bucket']=='>0.5')],
    'F8-E': lambda d: d[(d['late_vwap_break']==True) & (d['pm_rth_bucket']=='<10%')],
    'F8-F': lambda d: d[(d['late_vwap_break']==True) & (d['pm_rth_bucket']=='>30%')],
    'F8-G': lambda d: d[(d['late_vwap_break']==True) & (d['gap_bucket']=='>2') & (d['od_dir_label']=='with_gap') & (d['rvol_bucket'].isin(['2-5x', '>5x']))],
    'F8-H': lambda d: d[(d['late_vwap_break']==True) & (d['gap_bucket']=='<1') & (d['rvol_bucket']=='<2x')],
    'F8-I': lambda d: d[(d['late_vwap_break']==True) & (d['first_candle_in_gap']==True) & (d['first_candle_size'] > 0.10)],
    'F8-J': lambda d: d[(d['late_vwap_break']==True) & (d['first_candle_in_gap']==False) & (d['first_candle_size'] < 0.10)],
}

combo_labels = {
    'F8-A': 'Break + Gap>2 + RVOL>5x',
    'F8-B': 'Break + Gap>2 + RVOL<2x',
    'F8-C': 'Break + OD with + OD>0.5',
    'F8-D': 'Break + OD against + OD>0.5',
    'F8-E': 'Break + PM/RTH<10%',
    'F8-F': 'Break + PM/RTH>30%',
    'F8-G': 'Break + Gap>2 + OD with + RVOL>2x',
    'F8-H': 'Break + Gap<1 + RVOL<2x',
    'F8-I': 'Break + Grosse 1.Kerze in Gap',
    'F8-J': 'Break + Kleine 1.Kerze gegen Gap',
}

for combo_id in sorted(combos_f8.keys()):
    combo_func = combos_f8[combo_id]
    w(f"  {combo_id}: {combo_labels[combo_id]}")
    for gd_label, gd_val in [('GapUp', 'up'), ('GapDown', 'down')]:
        sub = combo_func(df)
        sub = sub[sub['gap_dir'] == gd_val]
        report_pb_group(sub, f"    {gd_label}")
    w()

# Kernfragen F8
w("--- KERNFRAGEN AUFGABE 2 ---")
w()
for gd_label, gd_val in [('GapUp', 'up'), ('GapDown', 'down')]:
    sub = vb[vb['gap_dir'] == gd_val]
    if len(sub) >= 5:
        d1100 = sub['pb_drift_1100'].mean() if 'pb_drift_1100' in sub.columns and sub['pb_drift_1100'].notna().any() else np.nan
        w(f"  {gd_label} Post-Break bis 11:00: {d1100:+.3f} (N={len(sub)})")

# Doppelsignal
w()
doppel = df[(df['late_vwap_break']==True) & (df['pm_rth_bucket']=='<10%')]
for gd_label, gd_val in [('GapUp', 'up'), ('GapDown', 'down')]:
    sub = doppel[doppel['gap_dir'] == gd_val]
    d = sub['drift_day'].mean() if len(sub) >= 5 else np.nan
    w(f"  Doppelsignal PM/RTH<10% + Break {gd_label}: Drift={d:+.3f} (N={len(sub)})")

w()
w()

# ══════════════════════════════════════════════════════════════════════════════
# AUFGABE 3: F9 ERSTE KERZE x PARAMETER
# ══════════════════════════════════════════════════════════════════════════════
w("=" * 110)
w("AUFGABE 3: F9 ERSTE KERZE x PARAMETER")
w("=" * 110)
w()

big_in_gap = df[(df['first_candle_in_gap'] == True) & (df['first_candle_size'] > 0.10)]
w(f"Grosse erste Kerze in Gap-Richtung (>0.10 ADR): N={len(big_in_gap)}")
w()

# a) x Gap-Groesse
w("--- 3a) Grosse 1.Kerze in Gap x Gap-Groesse ---")
w()
for gd_label, gd_val in [('GapUp', 'up'), ('GapDown', 'down')]:
    w(f"  {gd_label}:")
    for gb in ['<1', '1-2', '>2']:
        sub = big_in_gap[(big_in_gap['gap_dir'] == gd_val) & (big_in_gap['gap_bucket'] == gb)]
        report_group(sub, f"Grosse 1.Kerze + Gap {gb} ADR")
    w()

# b) x RVOL
w("--- 3b) Grosse 1.Kerze in Gap x RVOL ---")
w()
for gd_label, gd_val in [('GapUp', 'up'), ('GapDown', 'down')]:
    w(f"  {gd_label}:")
    for rvb in ['<2x', '2-5x', '>5x']:
        sub = big_in_gap[(big_in_gap['gap_dir'] == gd_val) & (big_in_gap['rvol_bucket'] == rvb)]
        report_group(sub, f"Grosse 1.Kerze + RVOL {rvb}")
    w()

# c) x PM/RTH30
w("--- 3c) Grosse 1.Kerze in Gap x PM/RTH30 ---")
w()
for gd_label, gd_val in [('GapUp', 'up'), ('GapDown', 'down')]:
    w(f"  {gd_label}:")
    for pmb in pm_buckets:
        sub = big_in_gap[(big_in_gap['gap_dir'] == gd_val) & (big_in_gap['pm_rth_bucket'] == pmb)]
        report_group(sub, f"Grosse 1.Kerze + PM/RTH {pmb}")
    w()

# Schwellen-Test
w("--- 3d) Schwellen-Test: Was ist 'gross'? ---")
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

w()

# ══════════════════════════════════════════════════════════════════════════════
# AUFGABE 4: F7 OD-STAERKE x PARAMETER
# ══════════════════════════════════════════════════════════════════════════════
w("=" * 110)
w("AUFGABE 4: F7 OD-STAERKE x PARAMETER (OD>0.5 mit SL am 5min H/L)")
w("=" * 110)
w()

od_strong = df[df['od_str_bucket'] == '>0.5']
w(f"OD>0.5 ADR: N={len(od_strong)}")
w()

# a) x Gap-Groesse x Gap-Richtung
w("--- 4a) OD>0.5 x Gap-Groesse x Gap-Richtung ---")
w()
for gd_label, gd_val in [('GapUp', 'up'), ('GapDown', 'down')]:
    w(f"  {gd_label}:")
    for gb in ['<1', '1-2', '>2']:
        sub = od_strong[(od_strong['gap_dir'] == gd_val) & (od_strong['gap_bucket'] == gb)]
        report_od_group(sub, f"OD>0.5 + Gap {gb} ADR")
    w()

# b) x RVOL x Gap-Richtung
w("--- 4b) OD>0.5 x RVOL x Gap-Richtung ---")
w()
for gd_label, gd_val in [('GapUp', 'up'), ('GapDown', 'down')]:
    w(f"  {gd_label}:")
    for rvb in ['<2x', '2-5x', '>5x']:
        sub = od_strong[(od_strong['gap_dir'] == gd_val) & (od_strong['rvol_bucket'] == rvb)]
        report_od_group(sub, f"OD>0.5 + RVOL {rvb}")
    w()

# c) x PM/RTH30 x Gap-Richtung (retrospektiv)
w("--- 4c) OD>0.5 x PM/RTH30 x Gap-Richtung (retrospektiv) ---")
w()
for gd_label, gd_val in [('GapUp', 'up'), ('GapDown', 'down')]:
    w(f"  {gd_label}:")
    for pmb in pm_buckets:
        sub = od_strong[(od_strong['gap_dir'] == gd_val) & (od_strong['pm_rth_bucket'] == pmb)]
        report_od_group(sub, f"OD>0.5 + PM/RTH {pmb}")
    w()

# d) x OD with_gap vs against_gap x Gap-Richtung
w("--- 4d) OD>0.5 x OD-Richtung x Gap-Richtung ---")
w()
for gd_label, gd_val in [('GapUp', 'up'), ('GapDown', 'down')]:
    w(f"  {gd_label}:")
    for od_dir in ['with_gap', 'against_gap']:
        sub = od_strong[(od_strong['gap_dir'] == gd_val) & (od_strong['od_dir_label'] == od_dir)]
        report_od_group(sub, f"OD>0.5 + OD {od_dir}")
    w()

# Kernfrage: Wo wird aus 49% WR@0.50 eine 55%+ WR?
w("--- KERNFRAGE: Wo wird WR@0.50 > 55%? ---")
w()
for gd_label, gd_val in [('GapUp', 'up'), ('GapDown', 'down')]:
    for od_dir in ['with_gap', 'against_gap']:
        for gb in ['<1', '1-2', '>2']:
            sub = od_strong[(od_strong['gap_dir'] == gd_val) &
                            (od_strong['od_dir_label'] == od_dir) &
                            (od_strong['gap_bucket'] == gb)]
            if len(sub) >= 10 and 'wr_0.50' in sub.columns:
                wr = sub['wr_0.50'].astype(float).mean() * 100
                if wr >= 55:
                    w(f"  *** {gd_label} + OD {od_dir} + Gap {gb}: WR@0.50={wr:.0f}% (N={len(sub)}) ***")

for gd_label, gd_val in [('GapUp', 'up'), ('GapDown', 'down')]:
    for od_dir in ['with_gap', 'against_gap']:
        for rvb in ['<2x', '2-5x', '>5x']:
            sub = od_strong[(od_strong['gap_dir'] == gd_val) &
                            (od_strong['od_dir_label'] == od_dir) &
                            (od_strong['rvol_bucket'] == rvb)]
            if len(sub) >= 10 and 'wr_0.50' in sub.columns:
                wr = sub['wr_0.50'].astype(float).mean() * 100
                if wr >= 55:
                    w(f"  *** {gd_label} + OD {od_dir} + RVOL {rvb}: WR@0.50={wr:.0f}% (N={len(sub)}) ***")

for gd_label, gd_val in [('GapUp', 'up'), ('GapDown', 'down')]:
    for od_dir in ['with_gap', 'against_gap']:
        for pmb in pm_buckets:
            sub = od_strong[(od_strong['gap_dir'] == gd_val) &
                            (od_strong['od_dir_label'] == od_dir) &
                            (od_strong['pm_rth_bucket'] == pmb)]
            if len(sub) >= 10 and 'wr_0.50' in sub.columns:
                wr = sub['wr_0.50'].astype(float).mean() * 100
                if wr >= 55:
                    w(f"  *** {gd_label} + OD {od_dir} + PM/RTH {pmb}: WR@0.50={wr:.0f}% (N={len(sub)}) ***")

# Also report ALL combos with WR@0.50 for reference
w()
w("  Vollstaendige WR@0.50 Tabelle OD>0.5:")
for gd_label, gd_val in [('GapUp', 'up'), ('GapDown', 'down')]:
    for od_dir in ['with_gap', 'against_gap']:
        sub = od_strong[(od_strong['gap_dir'] == gd_val) & (od_strong['od_dir_label'] == od_dir)]
        if len(sub) >= 10 and 'wr_0.50' in sub.columns:
            wr = sub['wr_0.50'].astype(float).mean() * 100
            sl = sub['sl_hit'].astype(float).mean() * 100 if 'sl_hit' in sub.columns else np.nan
            w(f"    {gd_label} + OD {od_dir}: WR@0.50={wr:.0f}%, SL-Hit={sl:.0f}% (N={len(sub)})")

w()
w()

# ══════════════════════════════════════════════════════════════════════════════
# AUFGABE 5: SIGNALKOMBINATIONEN (MULTI-FACTOR)
# ══════════════════════════════════════════════════════════════════════════════
w("=" * 110)
w("AUFGABE 5: SIGNALKOMBINATIONEN (MULTI-FACTOR)")
w("=" * 110)
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
    'H': ('PM/RTH>30% + OD against Gap',
          lambda d: d[(d['pm_rth_bucket']=='>30%') & (d['od_dir_label']=='against_gap')]),
}

for combo_id in sorted(combos.keys()):
    label, func = combos[combo_id]
    w(f"  Combo {combo_id}: {label}")
    for gd_label, gd_val in [('GapUp', 'up'), ('GapDown', 'down')]:
        sub = func(df)
        sub = sub[sub['gap_dir'] == gd_val]
        report_group(sub, f"    {gd_label}")
    w()

# ══════════════════════════════════════════════════════════════════════════════
# TOP-10 SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
w("=" * 110)
w("TOP-10 INTERESSANTESTE ERGEBNISSE (fuer OOS-Validierung)")
w("=" * 110)
w()
w("Kriterien: Hoher Drift ODER hohe WR UND N >= 30")
w()

# Collect all interesting results
interesting = []

# Scan Aufgabe 1 — PM/RTH cross combos
for gd_label, gd_val in [('GapUp', 'up'), ('GapDown', 'down')]:
    for pmb in pm_buckets:
        for gb in ['<1', '1-2', '>2']:
            sub = pm_valid[(pm_valid['gap_dir'] == gd_val) &
                           (pm_valid['pm_rth_bucket'] == pmb) &
                           (pm_valid['gap_bucket'] == gb)]
            if len(sub) >= 15:
                d = sub['drift_day'].mean()
                interesting.append(('A1', f'{gd_label}+PM/RTH{pmb}+Gap{gb}', len(sub), d, abs(d)))

        for rvb in ['<2x', '2-5x', '>5x']:
            sub = pm_valid[(pm_valid['gap_dir'] == gd_val) &
                           (pm_valid['pm_rth_bucket'] == pmb) &
                           (pm_valid['rvol_bucket'] == rvb)]
            if len(sub) >= 15:
                d = sub['drift_day'].mean()
                interesting.append(('A1', f'{gd_label}+PM/RTH{pmb}+RVOL{rvb}', len(sub), d, abs(d)))

# Scan F8 combos
for combo_id in sorted(combos_f8.keys()):
    combo_func = combos_f8[combo_id]
    for gd_label, gd_val in [('GapUp', 'up'), ('GapDown', 'down')]:
        sub = combo_func(df)
        sub = sub[sub['gap_dir'] == gd_val]
        if len(sub) >= 10 and 'pb_drift_1100' in sub.columns and sub['pb_drift_1100'].notna().any():
            d = sub['pb_drift_1100'].mean()
            interesting.append(('A2', f'{gd_label}+{combo_id}', len(sub), d, abs(d)))

# Scan Multi-Factor
for combo_id in sorted(combos.keys()):
    label, func = combos[combo_id]
    for gd_label, gd_val in [('GapUp', 'up'), ('GapDown', 'down')]:
        sub = func(df)
        sub = sub[sub['gap_dir'] == gd_val]
        if len(sub) >= 10:
            d = sub['drift_day'].mean()
            interesting.append(('A5', f'{gd_label}+Combo{combo_id}', len(sub), d, abs(d)))

# OD combos with high WR
for gd_label, gd_val in [('GapUp', 'up'), ('GapDown', 'down')]:
    for od_dir in ['with_gap', 'against_gap']:
        for extra_label, extra_filter in [
            ('Gap<1', lambda d: d[d['gap_bucket']=='<1']),
            ('Gap>2', lambda d: d[d['gap_bucket']=='>2']),
            ('RVOL>5x', lambda d: d[d['rvol_bucket']=='>5x']),
            ('RVOL<2x', lambda d: d[d['rvol_bucket']=='<2x']),
            ('PM<10%', lambda d: d[d['pm_rth_bucket']=='<10%']),
        ]:
            sub = extra_filter(od_strong)
            sub = sub[(sub['gap_dir'] == gd_val) & (sub['od_dir_label'] == od_dir)]
            if len(sub) >= 10 and 'wr_0.50' in sub.columns:
                wr = sub['wr_0.50'].astype(float).mean()
                if wr >= 0.53:
                    interesting.append(('A4', f'{gd_label}+OD{od_dir}+{extra_label} WR@50={wr*100:.0f}%', len(sub), wr, wr))

# Sort by absolute effect size
interesting.sort(key=lambda x: x[4], reverse=True)

for i, (aufgabe, desc, n, val, _) in enumerate(interesting[:20]):
    tag = " [LOW N]" if n < 50 else ""
    w(f"  {i+1:2d}. [{aufgabe}] {desc}: N={n}{tag}, Value={val:+.3f}")

w()

# Write output
report = "\n".join(out)
with open('results/cross_analysis_h1.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print(report)
print(f"\nSaved to results/cross_analysis_h1.txt", file=sys.stderr)
