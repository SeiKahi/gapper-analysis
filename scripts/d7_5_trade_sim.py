"""
Durchlauf 7.5 -- Aufgabe 5: Trade-Simulation Continuation + Fade (IS)
Entry: 9:35 (close_935), SL-Varianten, Timeout 10:59
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys, os, warnings
warnings.filterwarnings('ignore')

# === LOAD ===
df = pd.read_parquet('data/metadata/metadata_v7.parquet')
h1 = df[(df['date'] >= '2021-02-21') & (df['date'] <= '2023-12-31')].copy()
h1 = h1.dropna(subset=['pm_rth5', 'rvol_5', 'gap_size_in_adr', 'full_drift', 'close_935', 'adr_10', 'rth_high', 'rth_low'])
h1['gap_dir'] = h1['gap_direction'].map({'up': 'GapUp', 'down': 'GapDown'})

# 3x3x3 buckets
def pm5_alt(x):
    if x < 0.50: return 'PM5_LO'
    elif x < 1.00: return 'PM5_MID'
    else: return 'PM5_HI'
def rv5_alt(x):
    if x < 3: return 'RV5_LO'
    elif x < 7: return 'RV5_MID'
    else: return 'RV5_HI'
def gap_bucket(x):
    if x < 1: return 'GAP_SM'
    elif x < 2: return 'GAP_MD'
    else: return 'GAP_LG'

h1['pm5'] = h1['pm_rth5'].apply(pm5_alt)
h1['rv5'] = h1['rvol_5'].apply(rv5_alt)
h1['gap_b'] = h1['gap_size_in_adr'].apply(gap_bucket)

print(f"H1: {len(h1)} Gapper", file=sys.stderr)

# === COMPUTE 5-MIN HIGH/LOW FOR SL ===
# We need to load 1-min data for the first 5 minutes to get the OD range
# But we can approximate from metadata: od_strength * adr_10 for the range
# Actually, let's use a simpler approach: compute from rth_open and close_935
# The 5-min H/L is approximately: rth_open + max/min(od_strength * adr_10)
# Better: We have od_strength = max excursion in first 5 min in ADR units

# For SL_5min:
# GapUp: SL = 5-min Low. Approx: close_935 - od_strength * adr (if OD down) or rth_open (if OD up)
# This is imprecise. Let's use pre-computed data if available, or fall back to fixed SL.

# Actually, the simplest accurate approach: since we need 1-min data for MFE/MAE anyway,
# let's compute SL levels from metadata approximations and do the trade sim purely from
# the summary stats (rth_high, rth_low, close_935, adr_10).

# For the continuation trade FROM 9:35:
# - Entry = close_935
# - Direction: Gap direction (Long for GapUp, Short for GapDown)
# - SL_5min: We approximate the 5min range as |rth_open - close_935| + buffer
#   Better: od_strength * adr_10 gives the OD magnitude, so:
#   5min_high ~= max(rth_open, close_935) + something
#   Let's just use fixed SLs and the 5min SL from loading bar data
#
# SIMPLIFICATION: Use only the day's high/low to determine if target/SL was hit.
# This overestimates WR slightly (can't distinguish order of hits within day).
# For more accurate results, we'd need bar-by-bar simulation.
#
# For this analysis: Use rth_high/rth_low to compute MFE/MAE from entry at close_935.

def compute_trade_stats(gappers, trade_dir, sl_adr, target_adr):
    """
    Compute trade stats for a set of gappers.
    trade_dir: 'long' or 'short'
    sl_adr: SL distance in ADR (positive)
    target_adr: target distance in ADR (positive)

    Returns dict with WR, SL-hit rate, median MFE, median MAE, EV
    Uses day high/low (approximation - can't determine intraday order).
    """
    entries = gappers['close_935'].values
    highs = gappers['rth_high'].values
    lows = gappers['rth_low'].values
    adrs = gappers['adr_10'].values

    n = len(gappers)
    if n == 0:
        return None

    wins = 0
    sl_hits = 0
    mfe_list = []
    mae_list = []
    pnl_list = []

    for i in range(n):
        entry = entries[i]
        high = highs[i]
        low = lows[i]
        adr = adrs[i]

        if adr <= 0 or np.isnan(entry):
            continue

        sl_dist = sl_adr * adr
        tgt_dist = target_adr * adr

        if trade_dir == 'long':
            mfe = (high - entry) / adr  # favorable = up
            mae = (entry - low) / adr   # adverse = down
            target_hit = (high - entry) >= tgt_dist
            sl_hit = (entry - low) >= sl_dist
        else:  # short
            mfe = (entry - low) / adr   # favorable = down
            mae = (high - entry) / adr  # adverse = up
            target_hit = (entry - low) >= tgt_dist
            sl_hit = (high - entry) >= sl_dist

        mfe_list.append(mfe)
        mae_list.append(mae)

        if target_hit:
            wins += 1
            pnl_list.append(target_adr)
        elif sl_hit:
            sl_hits += 1
            pnl_list.append(-sl_adr)
        else:
            # Timeout: use close
            close = gappers.iloc[i]['rth_close']
            if trade_dir == 'long':
                timeout_pnl = (close - entry) / adr
            else:
                timeout_pnl = (entry - close) / adr
            pnl_list.append(timeout_pnl)

    valid_n = len(mfe_list)
    if valid_n == 0:
        return None

    wr = wins / valid_n
    sl_rate = sl_hits / valid_n

    return {
        'N': valid_n,
        'WR': wr,
        'SL_rate': sl_rate,
        'MFE_med': np.median(mfe_list),
        'MAE_med': np.median(mae_list),
        'EV': np.mean(pnl_list),
        'EV_med': np.median(pnl_list),
    }

# === OUTPUT ===
out = open('results/d7_5_trade_sim.txt', 'w', encoding='utf-8')
def p(text=''):
    out.write(text + '\n')

p("=" * 120)
p("DURCHLAUF 7.5 -- AUFGABE 5: TRADE-SIMULATION CONTINUATION + FADE (IS)")
p("  Entry: 9:35 (close_935), Timeout: End of Day (using day High/Low)")
p("  NOTE: Day H/L approximation - can't determine intraday order of SL vs TGT hit")
p("=" * 120)

sl_options = [0.25, 0.50]
tgt_options = [0.25, 0.50, 0.75, 1.00]

# ============================================================
# 5a: CONTINUATION TRADES
# ============================================================
p(f"\n{'='*120}")
p("5a: CONTINUATION TRADES (in Gap-Richtung)")
p(f"{'='*120}")

# By 3x3x3 cells
pm5a_order = ['PM5_LO', 'PM5_MID', 'PM5_HI']
rv5a_order = ['RV5_LO', 'RV5_MID', 'RV5_HI']
gap_order = ['GAP_SM', 'GAP_MD', 'GAP_LG']

for direction in ['GapUp', 'GapDown']:
    sub = h1[h1['gap_dir'] == direction]
    trade_dir = 'long' if direction == 'GapUp' else 'short'

    p(f"\n--- {direction} CONTINUATION ({trade_dir}) ---")

    for sl in sl_options:
        p(f"\n  SL = {sl:.2f} ADR:")
        hdr = f"  {'PM5':>7s} {'RV5':>7s} {'GAP':>7s} | {'N':>5s} | {'WR25':>5s} {'WR50':>5s} {'WR75':>5s} {'WR100':>5s} | {'SL%':>5s} | {'MFE':>5s} {'MAE':>5s} | {'EV25':>7s} {'EV50':>7s} {'EV100':>7s}"
        p(hdr)
        p("  " + "-" * (len(hdr) - 2))

        for pm in pm5a_order:
            for rv in rv5a_order:
                for g in gap_order:
                    mask = (sub['pm5'] == pm) & (sub['rv5'] == rv) & (sub['gap_b'] == g)
                    cell = sub[mask]
                    if len(cell) < 10:
                        continue

                    # Compute for each target
                    results = {}
                    for tgt in tgt_options:
                        r = compute_trade_stats(cell, trade_dir, sl, tgt)
                        if r:
                            results[tgt] = r

                    if not results:
                        continue

                    n = results[tgt_options[0]]['N'] if tgt_options[0] in results else 0
                    wr25 = results.get(0.25, {}).get('WR', 0) * 100
                    wr50 = results.get(0.50, {}).get('WR', 0) * 100
                    wr75 = results.get(0.75, {}).get('WR', 0) * 100
                    wr100 = results.get(1.00, {}).get('WR', 0) * 100
                    sl_rate = results.get(0.50, {}).get('SL_rate', 0) * 100
                    mfe = results.get(0.50, {}).get('MFE_med', 0)
                    mae = results.get(0.50, {}).get('MAE_med', 0)
                    ev25 = results.get(0.25, {}).get('EV', 0)
                    ev50 = results.get(0.50, {}).get('EV', 0)
                    ev100 = results.get(1.00, {}).get('EV', 0)

                    ntag = '' if n >= 50 else (' [LN]' if n >= 20 else ' ***')

                    p(f"  {pm:>7s} {rv:>7s} {g:>7s} | {n:>5d} | "
                      f"{wr25:>5.1f} {wr50:>5.1f} {wr75:>5.1f} {wr100:>5.1f} | "
                      f"{sl_rate:>5.1f} | {mfe:>5.2f} {mae:>5.2f} | "
                      f"{ev25:>+7.3f} {ev50:>+7.3f} {ev100:>+7.3f}{ntag}")

# ============================================================
# 5b: FADE TRADES
# ============================================================
p(f"\n\n{'='*120}")
p("5b: FADE TRADES (GEGEN Gap-Richtung)")
p(f"{'='*120}")

for direction in ['GapUp', 'GapDown']:
    sub = h1[h1['gap_dir'] == direction]
    # Fade = gegen Gap: GapUp -> Short, GapDown -> Long
    trade_dir = 'short' if direction == 'GapUp' else 'long'

    p(f"\n--- {direction} FADE ({trade_dir}) ---")

    for sl in sl_options:
        p(f"\n  SL = {sl:.2f} ADR:")
        hdr = f"  {'PM5':>7s} {'RV5':>7s} {'GAP':>7s} | {'N':>5s} | {'WR25':>5s} {'WR50':>5s} {'WR75':>5s} {'WR100':>5s} | {'SL%':>5s} | {'MFE':>5s} {'MAE':>5s} | {'EV25':>7s} {'EV50':>7s} {'EV100':>7s}"
        p(hdr)
        p("  " + "-" * (len(hdr) - 2))

        for pm in pm5a_order:
            for rv in rv5a_order:
                for g in gap_order:
                    mask = (sub['pm5'] == pm) & (sub['rv5'] == rv) & (sub['gap_b'] == g)
                    cell = sub[mask]
                    if len(cell) < 10:
                        continue

                    results = {}
                    for tgt in tgt_options:
                        r = compute_trade_stats(cell, trade_dir, sl, tgt)
                        if r:
                            results[tgt] = r

                    if not results:
                        continue

                    n = results[tgt_options[0]]['N'] if tgt_options[0] in results else 0
                    wr25 = results.get(0.25, {}).get('WR', 0) * 100
                    wr50 = results.get(0.50, {}).get('WR', 0) * 100
                    wr75 = results.get(0.75, {}).get('WR', 0) * 100
                    wr100 = results.get(1.00, {}).get('WR', 0) * 100
                    sl_rate = results.get(0.50, {}).get('SL_rate', 0) * 100
                    mfe = results.get(0.50, {}).get('MFE_med', 0)
                    mae = results.get(0.50, {}).get('MAE_med', 0)
                    ev25 = results.get(0.25, {}).get('EV', 0)
                    ev50 = results.get(0.50, {}).get('EV', 0)
                    ev100 = results.get(1.00, {}).get('EV', 0)

                    ntag = '' if n >= 50 else (' [LN]' if n >= 20 else ' ***')

                    p(f"  {pm:>7s} {rv:>7s} {g:>7s} | {n:>5d} | "
                      f"{wr25:>5.1f} {wr50:>5.1f} {wr75:>5.1f} {wr100:>5.1f} | "
                      f"{sl_rate:>5.1f} | {mfe:>5.2f} {mae:>5.2f} | "
                      f"{ev25:>+7.3f} {ev50:>+7.3f} {ev100:>+7.3f}{ntag}")

# ============================================================
# 5c: FLAT CELLS
# ============================================================
p(f"\n\n{'='*120}")
p("5c: FLAT ZELLEN (|full_drift| < 0.10) - Lohnt sich Scalping?")
p(f"{'='*120}")

# Load 3x3x3 matrix for drift info
matrix = pd.read_parquet('results/d7_5_3er_matrix_3x3x3.parquet')
flat_cells = matrix[(matrix['fd'].abs() < 0.10) & (matrix['N'] >= 20)].nlargest(20, 'N')

p(f"\n{'Dir':>7s} {'PM5':>7s} {'RV5':>7s} {'GAP':>7s} | {'N':>5s} | {'fd':>7s} | {'DayR':>5s} | {'CL':>4s} | Interpretation")
p("-" * 80)

for _, r in flat_cells.iterrows():
    dr = r['day_range']
    if dr > 1.5:
        interp = "Volatil genug fuer Scalps"
    elif dr > 1.0:
        interp = "Moderate Range, Scalps moeglich"
    else:
        interp = "Zu ruhig, kein Trade"
    p(f"{r['dir']:>7s} {r['pm5']:>7s} {r['rv5']:>7s} {r['gap']:>7s} | {r['N']:>5.0f} | {r['fd']:>+7.3f} | {dr:>5.2f} | {r['cl']:>.2f} | {interp}")

# ============================================================
# ZUSAMMENFASSUNG: Beste Trades
# ============================================================
p(f"\n\n{'='*120}")
p("ZUSAMMENFASSUNG: Top Continuation + Fade Trades")
p(f"{'='*120}")

# Collect all trade results
all_trades = []
for direction in ['GapUp', 'GapDown']:
    sub = h1[h1['gap_dir'] == direction]

    for trade_type, trade_dir in [('CONT', 'long' if direction == 'GapUp' else 'short'),
                                   ('FADE', 'short' if direction == 'GapUp' else 'long')]:
        for pm in pm5a_order:
            for rv in rv5a_order:
                for g in gap_order:
                    mask = (sub['pm5'] == pm) & (sub['rv5'] == rv) & (sub['gap_b'] == g)
                    cell = sub[mask]
                    if len(cell) < 20:
                        continue

                    for sl in sl_options:
                        for tgt in [0.50, 1.00]:
                            r = compute_trade_stats(cell, trade_dir, sl, tgt)
                            if r and r['N'] >= 20:
                                all_trades.append({
                                    'dir': direction, 'type': trade_type,
                                    'pm5': pm, 'rv5': rv, 'gap': g,
                                    'SL': sl, 'TGT': tgt,
                                    'N': r['N'], 'WR': r['WR']*100,
                                    'SL_rate': r['SL_rate']*100,
                                    'MFE': r['MFE_med'], 'MAE': r['MAE_med'],
                                    'EV': r['EV'],
                                })

trades_df = pd.DataFrame(all_trades)

# Top Continuation by EV
p("\n--- Top-10 CONTINUATION Trades (by EV, N>=20) ---")
top_cont = trades_df[trades_df['type'] == 'CONT'].nlargest(10, 'EV')
p(f"{'Dir':>7s} {'PM5':>7s} {'RV5':>7s} {'GAP':>7s} {'SL':>4s} {'TGT':>4s} | {'N':>5s} {'WR%':>5s} {'SL%':>5s} {'MFE':>5s} {'MAE':>5s} | {'EV':>7s}")
for _, r in top_cont.iterrows():
    p(f"{r['dir']:>7s} {r['pm5']:>7s} {r['rv5']:>7s} {r['gap']:>7s} {r['SL']:>4.2f} {r['TGT']:>4.2f} | "
      f"{r['N']:>5.0f} {r['WR']:>5.1f} {r['SL_rate']:>5.1f} {r['MFE']:>5.2f} {r['MAE']:>5.2f} | {r['EV']:>+7.3f}")

# Top Fade by EV
p("\n--- Top-10 FADE Trades (by EV, N>=20) ---")
top_fade = trades_df[trades_df['type'] == 'FADE'].nlargest(10, 'EV')
p(f"{'Dir':>7s} {'PM5':>7s} {'RV5':>7s} {'GAP':>7s} {'SL':>4s} {'TGT':>4s} | {'N':>5s} {'WR%':>5s} {'SL%':>5s} {'MFE':>5s} {'MAE':>5s} | {'EV':>7s}")
for _, r in top_fade.iterrows():
    p(f"{r['dir']:>7s} {r['pm5']:>7s} {r['rv5']:>7s} {r['gap']:>7s} {r['SL']:>4.2f} {r['TGT']:>4.2f} | "
      f"{r['N']:>5.0f} {r['WR']:>5.1f} {r['SL_rate']:>5.1f} {r['MFE']:>5.2f} {r['MAE']:>5.2f} | {r['EV']:>+7.3f}")

# Worst trades (most negative EV)
p("\n--- Bottom-10 Trades (worst EV, N>=20) ---")
worst = trades_df.nsmallest(10, 'EV')
p(f"{'Dir':>7s} {'Type':>5s} {'PM5':>7s} {'RV5':>7s} {'GAP':>7s} {'SL':>4s} {'TGT':>4s} | {'N':>5s} {'WR%':>5s} | {'EV':>7s}")
for _, r in worst.iterrows():
    p(f"{r['dir']:>7s} {r['type']:>5s} {r['pm5']:>7s} {r['rv5']:>7s} {r['gap']:>7s} {r['SL']:>4.2f} {r['TGT']:>4.2f} | "
      f"{r['N']:>5.0f} {r['WR']:>5.1f} | {r['EV']:>+7.3f}")

trades_df.to_parquet('results/d7_5_trade_sim_raw.parquet', index=False)
out.close()
print("Done!", file=sys.stderr)
