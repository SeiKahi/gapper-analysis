"""
Durchlauf 7.5 -- Aufgabe 8: OOS-Validierung auf H2 (2024-2026)
EINMALIG. Testet die besten IS-Setups.
"""
import pandas as pd
import numpy as np
from scipy import stats
import sys, warnings
warnings.filterwarnings('ignore')

# === LOAD ===
df = pd.read_parquet('data/metadata/metadata_v7.parquet')

# H1 (IS)
h1 = df[(df['date'] >= '2021-02-21') & (df['date'] <= '2023-12-31')].copy()
h1 = h1.dropna(subset=['pm_rth5', 'rvol_5', 'gap_size_in_adr', 'full_drift'])
h1['gap_dir'] = h1['gap_direction'].map({'up': 'GapUp', 'down': 'GapDown'})

# H2 (OOS)
h2 = df[(df['date'] >= '2024-01-01') & (df['date'] <= '2026-02-06')].copy()
h2 = h2.dropna(subset=['pm_rth5', 'rvol_5', 'gap_size_in_adr', 'full_drift'])
h2['gap_dir'] = h2['gap_direction'].map({'up': 'GapUp', 'down': 'GapDown'})

print(f"H1 (IS): {len(h1)}, H2 (OOS): {len(h2)}", file=sys.stderr)

# Buckets (3x3x3)
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

for d in [h1, h2]:
    d['pm5'] = d['pm_rth5'].apply(pm5_alt)
    d['rv5'] = d['rvol_5'].apply(rv5_alt)
    d['gap_b'] = d['gap_size_in_adr'].apply(gap_bucket)
    d['cl_num'] = ((d['rth_close'] - d['rth_low']) / (d['rth_high'] - d['rth_low'])).clip(0, 1)
    d['prev_close_calc'] = d['rth_open'] / (1 + d['gap_pct'] / 100)
    d['gap_filled'] = False
    d.loc[d['gap_dir'] == 'GapUp', 'gap_filled'] = d.loc[d['gap_dir'] == 'GapUp', 'rth_low'] <= d.loc[d['gap_dir'] == 'GapUp', 'prev_close_calc']
    d.loc[d['gap_dir'] == 'GapDown', 'gap_filled'] = d.loc[d['gap_dir'] == 'GapDown', 'rth_high'] >= d.loc[d['gap_dir'] == 'GapDown', 'prev_close_calc']
    d['day_range_adr'] = (d['rth_high'] - d['rth_low']) / d['adr_10']
    d['od_with_gap'] = d['od_direction'] == 'with_gap'

def compute_cell(subset):
    n = len(subset)
    if n < 5:
        return None
    return {
        'N': n,
        'fd': subset['full_drift'].mean(),
        'fd_med': subset['full_drift'].median(),
        'rd': subset['rest_drift'].mean(),
        'rd_1100': subset['rest_drift_1100'].mean(),
        'fill_rate': subset['gap_filled'].mean() * 100,
        'day_range': subset['day_range_adr'].mean(),
        'cl': subset['cl_num'].mean(),
        'od_with': subset['od_with_gap'].mean() * 100,
    }

def drift_tag(d):
    if d > 0.30: return '[CONT+]'
    elif d > 0.10: return '[CONT] '
    elif d > -0.10: return '[FLAT] '
    elif d > -0.30: return '[FADE] '
    else: return '[FADE+]'

def trade_stats(gappers, trade_dir, sl, tgt):
    entries = gappers['close_935'].values
    highs = gappers['rth_high'].values
    lows = gappers['rth_low'].values
    adrs = gappers['adr_10'].values
    closes = gappers['rth_close'].values
    n = len(gappers)
    if n == 0: return None
    wins = 0; sl_hits = 0; pnl_list = []
    for i in range(n):
        entry = entries[i]; high = highs[i]; low = lows[i]; adr = adrs[i]; close = closes[i]
        if adr <= 0 or np.isnan(entry): continue
        sd = sl * adr; td = tgt * adr
        if trade_dir == 'long':
            th = (high - entry) >= td; sh = (entry - low) >= sd
            tp = (close - entry) / adr
        else:
            th = (entry - low) >= td; sh = (high - entry) >= sd
            tp = (entry - close) / adr
        if th: wins += 1; pnl_list.append(tgt)
        elif sh: sl_hits += 1; pnl_list.append(-sl)
        else: pnl_list.append(tp)
    vn = len(pnl_list)
    if vn == 0: return None
    return {'N': vn, 'WR': wins/vn*100, 'SL%': sl_hits/vn*100, 'EV': np.mean(pnl_list)}

out = open('results/d7_5_oos.txt', 'w', encoding='utf-8')
def p(text=''):
    out.write(text + '\n')

p("=" * 120)
p("DURCHLAUF 7.5 -- AUFGABE 8: OOS-VALIDIERUNG AUF H2 (2024-2026)")
p(f"  H1 (IS): {len(h1)} Gapper, H2 (OOS): {len(h2)} Gapper")
p("=" * 120)

# ============================================================
# 8a: Volle 3x3x3 Matrix auf OOS
# ============================================================
p(f"\n{'='*120}")
p("8a: VOLLE 3x3x3 MATRIX AUF OOS (Vergleich mit IS)")
p(f"{'='*120}")

comparison_rows = []
pm5a_order = ['PM5_LO', 'PM5_MID', 'PM5_HI']
rv5a_order = ['RV5_LO', 'RV5_MID', 'RV5_HI']
gap_order = ['GAP_SM', 'GAP_MD', 'GAP_LG']

for direction in ['GapUp', 'GapDown']:
    p(f"\n--- {direction} ---")
    hdr = f"  {'PM5':>7s} {'RV5':>7s} {'GAP':>7s} | {'N_IS':>5s} {'fd_IS':>7s} {'tag_IS':>7s} | {'N_OOS':>5s} {'fd_OOS':>7s} {'tag_OOS':>7s} | {'Delta':>7s} | {'Status':>12s}"
    p(hdr)
    p("  " + "-" * (len(hdr)-2))

    for pm in pm5a_order:
        for rv in rv5a_order:
            for g in gap_order:
                is_mask = (h1['gap_dir']==direction) & (h1['pm5']==pm) & (h1['rv5']==rv) & (h1['gap_b']==g)
                oos_mask = (h2['gap_dir']==direction) & (h2['pm5']==pm) & (h2['rv5']==rv) & (h2['gap_b']==g)

                is_cell = compute_cell(h1[is_mask])
                oos_cell = compute_cell(h2[oos_mask])

                if is_cell is None and oos_cell is None:
                    continue

                is_n = is_cell['N'] if is_cell else 0
                is_fd = is_cell['fd'] if is_cell else np.nan
                oos_n = oos_cell['N'] if oos_cell else 0
                oos_fd = oos_cell['fd'] if oos_cell else np.nan

                is_tag = drift_tag(is_fd) if not np.isnan(is_fd) else '  n/a  '
                oos_tag = drift_tag(oos_fd) if not np.isnan(oos_fd) else '  n/a  '

                delta = oos_fd - is_fd if not (np.isnan(is_fd) or np.isnan(oos_fd)) else np.nan
                delta_str = f"{delta:>+7.3f}" if not np.isnan(delta) else "    n/a"

                # Status
                if np.isnan(is_fd) or np.isnan(oos_fd):
                    status = "N/A"
                elif is_fd > 0.10 and oos_fd > 0.10:
                    status = "REPLIZIERT" if oos_fd >= is_fd * 0.5 else "TEILWEISE"
                elif is_fd < -0.10 and oos_fd < -0.10:
                    status = "REPLIZIERT" if oos_fd <= is_fd * 0.5 else "TEILWEISE"
                elif abs(is_fd) < 0.10:
                    status = "FLAT (ok)" if abs(oos_fd) < 0.20 else "SHIFT"
                elif (is_fd > 0.10 and oos_fd < -0.10) or (is_fd < -0.10 and oos_fd > 0.10):
                    status = "UMGEKEHRT"
                else:
                    status = "GESCHRUMPFT"

                is_fd_str = f"{is_fd:>+7.3f}" if not np.isnan(is_fd) else "    n/a"
                oos_fd_str = f"{oos_fd:>+7.3f}" if not np.isnan(oos_fd) else "    n/a"

                p(f"  {pm:>7s} {rv:>7s} {g:>7s} | {is_n:>5d} {is_fd_str} {is_tag} | "
                  f"{oos_n:>5d} {oos_fd_str} {oos_tag} | {delta_str} | {status:>12s}")

                comparison_rows.append({
                    'dir': direction, 'pm5': pm, 'rv5': rv, 'gap': g,
                    'N_IS': is_n, 'fd_IS': is_fd, 'N_OOS': oos_n, 'fd_OOS': oos_fd,
                    'rd_IS': is_cell['rd'] if is_cell else np.nan,
                    'rd_OOS': oos_cell['rd'] if oos_cell else np.nan,
                    'status': status
                })

comp_df = pd.DataFrame(comparison_rows)

# ============================================================
# 8b: Trade-Simulation auf OOS
# ============================================================
p(f"\n\n{'='*120}")
p("8b: TRADE-SIMULATION AUF OOS (Top IS-Setups)")
p(f"{'='*120}")

# Load IS trade results
is_trades = pd.read_parquet('results/d7_5_trade_sim_raw.parquet')

# Top-5 Cont and Top-5 Fade by IS EV
top_cont_is = is_trades[is_trades['type'] == 'CONT'].nlargest(5, 'EV')
top_fade_is = is_trades[is_trades['type'] == 'FADE'].nlargest(5, 'EV')
top_setups = pd.concat([top_cont_is, top_fade_is])

p(f"\n--- Top IS-Setups: OOS-Validierung ---")
p(f"  {'Type':>5s} {'Dir':>7s} {'PM5':>7s} {'RV5':>7s} {'GAP':>7s} {'SL':>4s} {'TGT':>4s} | "
  f"{'WR_IS':>6s} {'EV_IS':>7s} {'N_IS':>5s} | {'WR_OOS':>6s} {'EV_OOS':>7s} {'N_OOS':>5s} | {'Status':>10s}")
p("  " + "-" * 100)

for _, setup in top_setups.iterrows():
    direction = setup['dir']
    trade_type = setup['type']
    pm = setup['pm5']; rv = setup['rv5']; g = setup['gap']
    sl = setup['SL']; tgt = setup['TGT']

    # OOS trade
    oos_sub = h2[(h2['gap_dir']==direction) & (h2['pm5']==pm) & (h2['rv5']==rv) & (h2['gap_b']==g)]
    if trade_type == 'CONT':
        trade_dir = 'long' if direction == 'GapUp' else 'short'
    else:
        trade_dir = 'short' if direction == 'GapUp' else 'long'

    oos_r = trade_stats(oos_sub.dropna(subset=['close_935', 'adr_10']), trade_dir, sl, tgt)

    if oos_r:
        oos_wr = oos_r['WR']
        oos_ev = oos_r['EV']
        oos_n = oos_r['N']
        if setup['EV'] > 0 and oos_ev > 0:
            status = "REPLIZIERT"
        elif setup['EV'] > 0 and oos_ev <= 0:
            status = "GESCHEITERT"
        else:
            status = "OK" if abs(oos_ev - setup['EV']) < 0.1 else "SHIFT"
    else:
        oos_wr = 0; oos_ev = 0; oos_n = 0; status = "N/A"

    p(f"  {trade_type:>5s} {direction:>7s} {pm:>7s} {rv:>7s} {g:>7s} {sl:>4.2f} {tgt:>4.2f} | "
      f"{setup['WR']:>6.1f} {setup['EV']:>+7.3f} {setup['N']:>5.0f} | "
      f"{oos_wr:>6.1f} {oos_ev:>+7.3f} {oos_n:>5d} | {status:>10s}")

# ============================================================
# 8c: Replikationstabelle
# ============================================================
p(f"\n\n{'='*120}")
p("8c: REPLIKATIONSTABELLE (alle 3x3x3 Zellen)")
p(f"{'='*120}")

valid = comp_df[(comp_df['N_IS'] >= 10) & (comp_df['N_OOS'] >= 10)]
p(f"\nZellen mit N>=10 in IS UND OOS: {len(valid)}")

status_counts = valid['status'].value_counts()
for status, count in status_counts.items():
    p(f"  {status}: {count} ({count/len(valid)*100:.0f}%)")

# Correlation IS vs OOS full_drift
valid_both = valid.dropna(subset=['fd_IS', 'fd_OOS'])
if len(valid_both) > 5:
    rho, pval = stats.spearmanr(valid_both['fd_IS'], valid_both['fd_OOS'])
    pearson_r, pearson_p = stats.pearsonr(valid_both['fd_IS'], valid_both['fd_OOS'])
    p(f"\nKorrelation IS vs OOS full_drift:")
    p(f"  Spearman rho = {rho:.3f} (p = {pval:.4f})")
    p(f"  Pearson r    = {pearson_r:.3f} (p = {pearson_p:.4f})")
    p(f"  -> {'Hohe Replikation' if rho > 0.5 else ('Moderate Replikation' if rho > 0.3 else 'Schwache Replikation')}")

# ============================================================
# 8d: Kernfragen
# ============================================================
p(f"\n\n{'='*120}")
p("8d: FINALE ANTWORTEN AUF KERNFRAGEN")
p(f"{'='*120}")

# Q1: Gibt es 9:35-Fade-Setups mit positivem EV im OOS?
p("\n1. Gibt es 9:35-Fade-Setups mit positivem EV im OOS?")
fade_oos = comp_df[(comp_df['fd_OOS'] < -0.10) & (comp_df['N_OOS'] >= 20)]
if len(fade_oos) > 0:
    best_fade = fade_oos.nsmallest(5, 'fd_OOS')
    for _, r in best_fade.iterrows():
        p(f"   {r['dir']} {r['pm5']} {r['rv5']} {r['gap']}: IS fd={r['fd_IS']:+.3f}, OOS fd={r['fd_OOS']:+.3f}, N_OOS={r['N_OOS']:.0f}")
    p("   -> JA, es gibt Fade-Zellen mit negativem OOS-Drift (Fade funktioniert)")
else:
    p("   -> NEIN, keine robusten Fade-Zellen im OOS")

# Q2: Continuation-Setups besser als OD>0.5?
p("\n2. Gibt es 9:35-Continuation-Setups besser als reines OD>0.5?")
cont_oos = comp_df[(comp_df['fd_OOS'] > 0.10) & (comp_df['N_OOS'] >= 20)]
if len(cont_oos) > 0:
    best_cont = cont_oos.nlargest(5, 'fd_OOS')
    for _, r in best_cont.iterrows():
        p(f"   {r['dir']} {r['pm5']} {r['rv5']} {r['gap']}: IS fd={r['fd_IS']:+.3f}, OOS fd={r['fd_OOS']:+.3f}, N_OOS={r['N_OOS']:.0f}")
    p("   (Vergleich: OD>0.5 with_gap: ~+0.9-1.1 ADR full_drift OOS, aber N~300)")

# Q3: Stabilste 3er-Kombos
p("\n3. Welche 3er-Kombos sind die stabilsten (IS -> OOS)?")
stable = valid_both.copy()
stable['abs_delta'] = (stable['fd_OOS'] - stable['fd_IS']).abs()
stable_top = stable.nsmallest(10, 'abs_delta')
for _, r in stable_top.iterrows():
    p(f"   {r['dir']} {r['pm5']} {r['rv5']} {r['gap']}: IS={r['fd_IS']:+.3f}, OOS={r['fd_OOS']:+.3f}, Delta={r['fd_OOS']-r['fd_IS']:+.3f}")

# Q4: Lohnt sich Warten bis 10:00?
p("\n4. Lohnt sich Warten bis 10:00?")
if 'rest_drift_1000' in h1.columns:
    for direction in ['GapUp', 'GapDown']:
        for dataset, label in [(h1, 'IS'), (h2, 'OOS')]:
            sub = dataset[dataset['gap_dir'] == direction]
            rd1000_mean = sub['rest_drift_1000'].mean()
            p(f"   {direction} {label}: missed_drift(mean)={rd1000_mean:+.3f} ADR")
    p("   -> Wenn missed_drift klein (~0): Warten kostet wenig")
    p("   -> Wenn missed_drift gross (>0.05): Warten kostet Geld bei Continuation")

# Save
comp_df.to_parquet('results/d7_5_oos_comparison.parquet', index=False)

out.close()
print("Done!", file=sys.stderr)
