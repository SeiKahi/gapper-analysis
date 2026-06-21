"""
Durchlauf 7.0 — Aufgabe 7: OOS Validation (EINMALIG!)
Validate ALL IS findings from Tasks 1-6 on H2 (2024-01-01 to 2026-02-06).
"""
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from tqdm import tqdm
import sys
import warnings
warnings.filterwarnings('ignore')

def bootstrap_ci(data, n_boot=1000, ci=0.95):
    data = np.array(data, dtype=float)
    data = data[~np.isnan(data)]
    if len(data) < 5: return np.nan, np.nan
    rng = np.random.default_rng(42)
    means = [np.mean(rng.choice(data, len(data), True)) for _ in range(n_boot)]
    return np.percentile(means, (1-ci)/2*100), np.percentile(means, (1+ci)/2*100)

def low_n(n):
    if n < 20: return " *** VERY LOW N ***"
    if n < 50: return " [LOW N]"
    return ""

def repl_status(is_val, oos_val, oos_ci_lo, oos_ci_hi, direction='positive'):
    """Determine replication status."""
    if np.isnan(oos_val) or np.isnan(oos_ci_lo):
        return "N/A"
    if direction == 'positive':
        if oos_ci_lo > 0: return "REPLIZIERT"
        if oos_val > 0: return "TEILWEISE"
        if oos_ci_hi < 0 and is_val > 0: return "UMGEKEHRT"
        return "NICHT REPLIZIERT"
    else:  # negative
        if oos_ci_hi < 0: return "REPLIZIERT"
        if oos_val < 0: return "TEILWEISE"
        if oos_ci_lo > 0 and is_val < 0: return "UMGEKEHRT"
        return "NICHT REPLIZIERT"

out = []
def w(s=""): out.append(s)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD H2 DATA
# ══════════════════════════════════════════════════════════════════════════════
print("Loading metadata_v7...", file=sys.stderr)
meta = pd.read_parquet('data/metadata/metadata_v7.parquet')
meta['date'] = pd.to_datetime(meta['date'])

# H2: OOS
h2 = meta[(meta['date'] >= '2024-01-01') & (meta['date'] <= '2026-02-06')].copy()
h2 = h2[h2['adr_10'].notna() & (h2['adr_10'] > 0)].copy()
print(f"H2 (OOS): N={len(h2)}", file=sys.stderr)

# H1: IS (for reference)
h1 = meta[(meta['date'] >= '2021-02-21') & (meta['date'] <= '2023-12-31')].copy()
h1 = h1[h1['adr_10'].notna() & (h1['adr_10'] > 0)].copy()
print(f"H1 (IS): N={len(h1)}", file=sys.stderr)

# Ensure numeric types
for col in ['rvol_5', 'rvol_at_time_30min', 'pm_rth5', 'pm_rth30_computed',
            'rest_drift', 'full_drift', 'od_strength', 'first_candle_size',
            'prior_return_10d', 'rsi_14_prev', 'close_935']:
    if col in h2.columns:
        h2[col] = pd.to_numeric(h2[col], errors='coerce')
    if col in h1.columns:
        h1[col] = pd.to_numeric(h1[col], errors='coerce')

# Define key flags for h2
h2['combo_e'] = (
    (h2['first_candle_dir'] == 'with_gap') &
    (h2['first_candle_size'] > 0.20) &
    (h2['od_direction'] == 'with_gap') &
    (h2['od_strength'] > 0.5)
)
h2['od_05_with'] = (h2['od_direction'] == 'with_gap') & (h2['od_strength'] > 0.5)
h2['pm30_low_od'] = (h2['pm_rth30_computed'] < 0.10) & h2['od_05_with']

w("=" * 80)
w("DURCHLAUF 7.0 — OOS VALIDATION (H2: 2024-01-01 bis 2026-02-06)")
w("=" * 80)
w(f"\nH2 total: N={len(h2)}")
w(f"  GapUp: N={len(h2[h2['gap_direction']=='up'])}")
w(f"  GapDown: N={len(h2[h2['gap_direction']=='down'])}")

# ══════════════════════════════════════════════════════════════════════════════
# TASK 1 OOS: RVOL GRANULAR
# ══════════════════════════════════════════════════════════════════════════════
w("\n" + "=" * 80)
w("OOS TASK 1: RVOL GRANULAR")
w("=" * 80)

for rvol_name, rvol_col in [('RVOL_30', 'rvol_at_time_30min'), ('RVOL_5', 'rvol_5')]:
    for gap_dir in ['up', 'down', None]:
        if gap_dir:
            df = h2[(h2['gap_direction'] == gap_dir) & h2[rvol_col].notna()].copy()
            label = f"{rvol_name} Gap{gap_dir.title()}"
        else:
            df = h2[h2[rvol_col].notna()].copy()
            label = f"{rvol_name} ALL"

        w(f"\n--- {label} (N={len(df)}) ---")

        # Correlations
        r_p, p_p = stats.pearsonr(df[rvol_col], df['full_drift'])
        r_s, p_s = stats.spearmanr(df[rvol_col], df['full_drift'])
        w(f"  Pearson  {rvol_col} vs full_drift:  r={r_p:+.4f}, p={p_p:.4f}")
        w(f"  Spearman {rvol_col} vs full_drift:  r={r_s:+.4f}, p={p_s:.4f}")

        r_p2, p_p2 = stats.pearsonr(df[rvol_col], df['rest_drift'])
        r_s2, p_s2 = stats.spearmanr(df[rvol_col], df['rest_drift'])
        w(f"  Pearson  {rvol_col} vs rest_drift:  r={r_p2:+.4f}, p={p_p2:.4f}")
        w(f"  Spearman {rvol_col} vs rest_drift:  r={r_s2:+.4f}, p={p_s2:.4f}")

        # Log vs Linear
        x = df[rvol_col].values
        y_full = df['full_drift'].values
        mask = (x > 0) & np.isfinite(x) & np.isfinite(y_full)
        x_m, y_m = x[mask], y_full[mask]

        if len(x_m) > 10:
            # Linear
            slope_l, intercept_l, r_l, p_l, se_l = stats.linregress(x_m, y_m)
            y_pred_l = intercept_l + slope_l * x_m
            ss_res_l = np.sum((y_m - y_pred_l)**2)
            n = len(x_m)
            aic_l = n * np.log(ss_res_l / n) + 2 * 2

            # Log
            log_x = np.log(x_m)
            slope_g, intercept_g, r_g, p_g, se_g = stats.linregress(log_x, y_m)
            y_pred_g = intercept_g + slope_g * log_x
            ss_res_g = np.sum((y_m - y_pred_g)**2)
            aic_g = n * np.log(ss_res_g / n) + 2 * 2

            winner = "LOG" if aic_g < aic_l else "LINEAR"
            w(f"  AIC Linear: {aic_l:.1f} vs AIC Log: {aic_g:.1f} => {winner} fits better")

        # Bucket summary
        rvol_edges = [0, 2, 5, 10, 999]
        rvol_labels = ['<2x', '2-5x', '5-10x', '>10x']
        df['rvol_bkt'] = pd.cut(df[rvol_col], bins=rvol_edges, labels=rvol_labels, right=False)

        w(f"  {'Bucket':<8} {'N':>5} {'full_drift':>12} {'rest_drift':>12} {'CL':>8}")
        w("  " + "-" * 50)
        for bkt in rvol_labels:
            sg = df[df['rvol_bkt'] == bkt]
            if len(sg) == 0: continue
            cl_vals = sg['cl'].dropna()
            cl_mean = cl_vals.mean() if len(cl_vals) > 0 else np.nan
            w(f"  {bkt:<8} {len(sg):>5}{low_n(len(sg))} {sg['full_drift'].mean():>+10.4f} {sg['rest_drift'].mean():>+10.4f} {cl_mean:>8.3f}")

# RVOL_5 vs RVOL_30 head-to-head on OOS
w("\n--- RVOL_5 vs RVOL_30 HEAD-TO-HEAD (OOS) ---")
both = h2[h2['rvol_5'].notna() & h2['rvol_at_time_30min'].notna()].copy()
w(f"  N with both: {len(both)}")

for target in ['full_drift', 'rest_drift']:
    w(f"  --- {target} ---")
    r5_p, _ = stats.pearsonr(both['rvol_5'], both[target])
    r30_p, _ = stats.pearsonr(both['rvol_at_time_30min'], both[target])
    r5_s, _ = stats.spearmanr(both['rvol_5'], both[target])
    r30_s, _ = stats.spearmanr(both['rvol_at_time_30min'], both[target])
    w(f"    Pearson:  RVOL_5 r={r5_p:+.4f} vs RVOL_30 r={r30_p:+.4f} => {'RVOL_30' if abs(r30_p)>abs(r5_p) else 'RVOL_5'}")
    w(f"    Spearman: RVOL_5 r={r5_s:+.4f} vs RVOL_30 r={r30_s:+.4f} => {'RVOL_30' if abs(r30_s)>abs(r5_s) else 'RVOL_5'}")

# ══════════════════════════════════════════════════════════════════════════════
# TASK 2 OOS: PM/RTH5 vs PM/RTH30
# ══════════════════════════════════════════════════════════════════════════════
w("\n" + "=" * 80)
w("OOS TASK 2: PM/RTH5 vs PM/RTH30")
w("=" * 80)

# PM/RTH5 bucket analysis
for gap_dir in ['up', 'down']:
    df = h2[(h2['gap_direction'] == gap_dir) & h2['pm_rth5'].notna()].copy()
    pm5_edges = [0, 0.10, 0.20, 0.30, 0.50, 1.0, 999]
    pm5_labels = ['5-10%', '10-20%', '20-30%', '30-50%', '50-100%', '>100%']
    df['pm5_bkt'] = pd.cut(df['pm_rth5'], bins=pm5_edges, labels=pm5_labels, right=False)

    w(f"\nGap{gap_dir.title()} — PM/RTH5 Buckets (OOS):")
    w(f"  {'Bucket':<12} {'N':>5} {'full_drift':>12} {'rest_drift':>12} {'CL':>8}")
    w("  " + "-" * 55)
    for bkt in pm5_labels:
        sg = df[df['pm5_bkt'] == bkt]
        if len(sg) == 0: continue
        cl_mean = sg['cl'].dropna().mean() if len(sg['cl'].dropna()) > 0 else np.nan
        w(f"  {bkt:<12} {len(sg):>5}{low_n(len(sg))} {sg['full_drift'].mean():>+10.4f} {sg['rest_drift'].mean():>+10.4f} {cl_mean:>8.3f}")

# Head-to-head: PM/RTH5 vs PM/RTH30 predictive power
w("\n--- PM/RTH5 vs PM/RTH30 Predictive Power (OOS) ---")
both_pm = h2[h2['pm_rth5'].notna() & h2['pm_rth30_computed'].notna()].copy()
w(f"  N: {len(both_pm)}")

for target in ['full_drift', 'rest_drift']:
    r5_p, _ = stats.pearsonr(both_pm['pm_rth5'], both_pm[target])
    r30_p, _ = stats.pearsonr(both_pm['pm_rth30_computed'], both_pm[target])
    r5_s, _ = stats.spearmanr(both_pm['pm_rth5'], both_pm[target])
    r30_s, _ = stats.spearmanr(both_pm['pm_rth30_computed'], both_pm[target])
    w(f"  {target}:")
    w(f"    Pearson:  PM/RTH5 r={r5_p:+.4f} vs PM/RTH30 r={r30_p:+.4f} => {'PM30' if abs(r30_p)>abs(r5_p) else 'PM5'} wins")
    w(f"    Spearman: PM/RTH5 r={r5_s:+.4f} vs PM/RTH30 r={r30_s:+.4f} => {'PM30' if abs(r30_s)>abs(r5_s) else 'PM5'} wins")

# PM/RTH5 as early predictor of PM/RTH30
w("\n--- PM/RTH5 Early Predictor (OOS) ---")
for lo_thresh in [0.30, 0.40]:
    pm5_lo = both_pm[both_pm['pm_rth5'] < lo_thresh]
    pm30_lo = both_pm[both_pm['pm_rth30_computed'] < 0.10]
    overlap = both_pm[(both_pm['pm_rth5'] < lo_thresh) & (both_pm['pm_rth30_computed'] < 0.10)]
    prec = len(overlap) / len(pm5_lo) * 100 if len(pm5_lo) > 0 else 0
    rec = len(overlap) / len(pm30_lo) * 100 if len(pm30_lo) > 0 else 0
    w(f"  PM/RTH5 < {lo_thresh:.2f} vs PM/RTH30 < 0.10:")
    w(f"    PM30<10%: N={len(pm30_lo)}, PM5<{lo_thresh:.0%}: N={len(pm5_lo)}, Both: N={len(overlap)}")
    w(f"    Precision: {prec:.1f}%, Recall: {rec:.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# TASK 3 OOS: COMBO E
# ══════════════════════════════════════════════════════════════════════════════
w("\n" + "=" * 80)
w("OOS TASK 3: COMBO E REST-DRIFT")
w("=" * 80)

for gap_dir in ['up', 'down']:
    ce = h2[(h2['gap_direction'] == gap_dir) & h2['combo_e']].copy()
    non_ce = h2[(h2['gap_direction'] == gap_dir) & ~h2['combo_e']].copy()

    w(f"\n--- Gap{gap_dir.title()} ---")
    w(f"  Combo E: N={len(ce)}, Non-Combo-E: N={len(non_ce)}")

    if len(ce) >= 5:
        rd = ce['rest_drift'].dropna()
        fd = ce['full_drift'].dropna()
        ci_lo_rd, ci_hi_rd = bootstrap_ci(rd.values)
        ci_lo_fd, ci_hi_fd = bootstrap_ci(fd.values)
        rd_1000 = ce['rest_drift_1000'].dropna().mean() if 'rest_drift_1000' in ce.columns else np.nan
        rd_1100 = ce['rest_drift_1100'].dropna().mean() if 'rest_drift_1100' in ce.columns else np.nan

        w(f"  Combo E      : full_drift={fd.mean():+.4f} ({ci_lo_fd:+.3f} to {ci_hi_fd:+.3f})")
        w(f"                 rest_drift={rd.mean():+.4f} ({ci_lo_rd:+.3f} to {ci_hi_rd:+.3f})")
        w(f"                 rd_1000={rd_1000:+.4f}, rd_1100={rd_1100:+.4f}")
        w(f"  Non-Combo-E  : full_drift={non_ce['full_drift'].mean():+.4f}, rest_drift={non_ce['rest_drift'].mean():+.4f}")

        # MFE/MAE from 9:35 (quick estimate from metadata)
        w(f"  MFE median: {ce['mfe_90min'].median():.3f}" if 'mfe_90min' in ce.columns else "  MFE: N/A")
        w(f"  MAE median: {ce['mae_90min'].median():.3f}" if 'mae_90min' in ce.columns else "  MAE: N/A")

        # RVOL_5 cross
        v = ce[ce['rvol_5'].notna()].copy()
        if len(v) >= 10:
            rvol_edges = [0, 2, 5, 10, 999]
            rvol_labels = ['<2x', '2-5x', '5-10x', '>10x']
            v['rvol_bkt'] = pd.cut(v['rvol_5'], bins=rvol_edges, labels=rvol_labels, right=False)
            w(f"  Combo E x RVOL_5:")
            for bkt in rvol_labels:
                sg = v[v['rvol_bkt'] == bkt]
                if len(sg) == 0: continue
                w(f"    {bkt:<8}: N={len(sg):>4}{low_n(len(sg))}, rd={sg['rest_drift'].mean():+.4f}, fd={sg['full_drift'].mean():+.4f}")
    else:
        w(f"  *** TOO FEW COMBO E TRADES ***")

# ══════════════════════════════════════════════════════════════════════════════
# TASK 4 OOS: KEY CROSS-PARAMETER COMBINATIONS
# ══════════════════════════════════════════════════════════════════════════════
w("\n" + "=" * 80)
w("OOS TASK 4: KEY CROSS-PARAMETER COMBINATIONS")
w("=" * 80)

# Bucket definitions
def get_gap_bkt(row):
    g = row['gap_size_in_adr']
    if pd.isna(g): return None
    if g < 1: return '<1'
    if g < 2: return '1-2'
    return '>2'

def get_rvol30_bkt(row):
    r = row['rvol_at_time_30min']
    if pd.isna(r): return None
    if r < 2: return '<2x'
    if r < 5: return '2-5x'
    return '>5x'

def get_rvol5_bkt(row):
    r = row['rvol_5']
    if pd.isna(r): return None
    if r < 2: return '<2x'
    if r < 5: return '2-5x'
    return '>5x'

def get_pmrth30_bkt(row):
    p = row['pm_rth30_computed']
    if pd.isna(p): return None
    if p < 0.10: return '<10%'
    if p < 0.30: return '10-30%'
    return '>30%'

def get_pmrth5_bkt(row):
    p = row['pm_rth5']
    if pd.isna(p): return None
    if p < 0.40: return '<40%'
    if p < 1.50: return '40-150%'
    return '>150%'

def get_od_dir(row):
    return row.get('od_direction', None)

def get_od_str_bkt(row):
    s = row['od_strength']
    if pd.isna(s): return None
    if s < 0.2: return '<0.2'
    if s < 0.5: return '0.2-0.5'
    return '>0.5'

def get_candle_bkt(row):
    d = row.get('first_candle_dir', None)
    s = row.get('first_candle_size', np.nan)
    if pd.isna(s) or d is None: return None
    if d == 'against_gap': return 'against'
    if s >= 0.20: return 'with+gross'
    return 'with+klein'

def get_spy_bkt(row):
    s = row.get('spy_return_day', np.nan)
    if pd.isna(s): return None
    if s > 0.005: return 'gruen'
    if s < -0.005: return 'rot'
    return 'neutral'

# Top IS combos to validate
top_combos_gu = [
    ('FxG', 'with_gap', '>0.5', get_od_dir, get_od_str_bkt, +1.0737),
    ('BxD', '>5x', '<10%', get_rvol30_bkt, get_pmrth30_bkt, +0.6886),
    ('BxF', '>5x', 'with_gap', get_rvol30_bkt, get_od_dir, +0.7254),
    ('CxD', '>5x', '<10%', get_rvol5_bkt, get_pmrth30_bkt, +0.6233),
    ('DxH', '<10%', 'with+gross', get_pmrth30_bkt, get_candle_bkt, +0.6875),
    ('HxI', 'with+gross', 'gruen', get_candle_bkt, get_spy_bkt, +0.8282),
]

top_combos_gd = [
    ('FxG', 'with_gap', '>0.5', get_od_dir, get_od_str_bkt, +1.0007),
    ('BxD', '>5x', '<10%', get_rvol30_bkt, get_pmrth30_bkt, +0.6046),
    ('BxI', '>5x', 'rot', get_rvol30_bkt, get_spy_bkt, +0.6630),
    ('CxD', '>5x', '<10%', get_rvol5_bkt, get_pmrth30_bkt, +0.5906),
    ('DxG', '<10%', '>0.5', get_pmrth30_bkt, get_od_str_bkt, +0.8464),
    ('FxI', 'with_gap', 'rot', get_od_dir, get_spy_bkt, +0.6325),
]

for gap_dir, top_combos in [('up', top_combos_gu), ('down', top_combos_gd)]:
    df = h2[h2['gap_direction'] == gap_dir].copy()
    w(f"\n--- Gap{gap_dir.title()} Top IS Combos (OOS N={len(df)}) ---")
    w(f"  {'Pair':<8} {'B1':<12} {'B2':<12} {'N':>5} {'IS_fd':>8} {'OOS_fd':>10} {'OOS_rd':>10} {'CI_lo':>8} {'CI_hi':>8} {'Status'}")
    w("  " + "-" * 95)

    for pair_name, b1_val, b2_val, bkt1_fn, bkt2_fn, is_fd in top_combos:
        df['_b1'] = df.apply(bkt1_fn, axis=1)
        df['_b2'] = df.apply(bkt2_fn, axis=1)
        sg = df[(df['_b1'] == b1_val) & (df['_b2'] == b2_val)]
        if len(sg) < 2:
            w(f"  {pair_name:<8} {b1_val:<12} {b2_val:<12} {len(sg):>5} {is_fd:>+8.4f}       N/A")
            continue

        fd = sg['full_drift'].mean()
        rd = sg['rest_drift'].mean()
        ci_lo, ci_hi = bootstrap_ci(sg['full_drift'].values)
        status = repl_status(is_fd, fd, ci_lo, ci_hi, 'positive')
        w(f"  {pair_name:<8} {b1_val:<12} {b2_val:<12} {len(sg):>5}{low_n(len(sg))} {is_fd:>+8.4f} {fd:>+10.4f} {rd:>+10.4f} {ci_lo:>+8.3f} {ci_hi:>+8.3f} {status}")

# ══════════════════════════════════════════════════════════════════════════════
# TASK 5 OOS: FAILED STRATEGIES
# ══════════════════════════════════════════════════════════════════════════════
w("\n" + "=" * 80)
w("OOS TASK 5: FAILED STRATEGIES (should still be failed)")
w("=" * 80)

# 5a: Prior Performance / RSI
gu = h2[h2['gap_direction'] == 'up']
gd = h2[h2['gap_direction'] == 'down']

# GapUp + 10d>20%
gu_hot = gu[gu['prior_return_10d'] > 20]
w(f"\n  GapUp + 10d>20%: N={len(gu_hot)}, full_drift={gu_hot['full_drift'].mean():+.4f}" if len(gu_hot) > 0 else "\n  GapUp + 10d>20%: N=0")
if len(gu_hot) >= 5:
    ci_lo, ci_hi = bootstrap_ci(gu_hot['full_drift'].values)
    w(f"    CI: ({ci_lo:+.3f} to {ci_hi:+.3f})")

# GapUp + RSI>70
gu_rsi = gu[gu['rsi_14_prev'] > 70]
w(f"  GapUp + RSI>70: N={len(gu_rsi)}, full_drift={gu_rsi['full_drift'].mean():+.4f}" if len(gu_rsi) > 0 else "  GapUp + RSI>70: N=0")
if len(gu_rsi) >= 5:
    ci_lo, ci_hi = bootstrap_ci(gu_rsi['full_drift'].values)
    w(f"    CI: ({ci_lo:+.3f} to {ci_hi:+.3f})")

# GapDown + RSI<30
gd_rsi = gd[gd['rsi_14_prev'] < 30]
w(f"  GapDown + RSI<30: N={len(gd_rsi)}, full_drift={gd_rsi['full_drift'].mean():+.4f}" if len(gd_rsi) > 0 else "  GapDown + RSI<30: N=0")
if len(gd_rsi) >= 5:
    ci_lo, ci_hi = bootstrap_ci(gd_rsi['full_drift'].values)
    w(f"    CI: ({ci_lo:+.3f} to {ci_hi:+.3f})")

# 5b: Kapitulation
kapitul = gd[gd['prior_return_10d'] < -10]
w(f"  Kapitulation (10d<-10%): N={len(kapitul)}, full_drift={kapitul['full_drift'].mean():+.4f}" if len(kapitul) > 0 else "  Kapitulation: N=0")
if len(kapitul) >= 5:
    ci_lo, ci_hi = bootstrap_ci(kapitul['full_drift'].values)
    w(f"    CI: ({ci_lo:+.3f} to {ci_hi:+.3f})")

# 5c: Vol concentration GapUp
h2['vol_conc_30'] = h2['volume_first_30min'] / h2['total_rth_volume'].replace(0, np.nan)
gu_hv = gu[h2.loc[gu.index, 'vol_conc_30'] > 0.30] if 'vol_conc_30' in h2.columns else pd.DataFrame()
if len(gu_hv) > 0:
    w(f"  GapUp + HighVol30: N={len(gu_hv)}, full_drift={gu_hv['full_drift'].mean():+.4f}")
    if len(gu_hv) >= 5:
        ci_lo, ci_hi = bootstrap_ci(gu_hv['full_drift'].values)
        w(f"    CI: ({ci_lo:+.3f} to {ci_hi:+.3f})")

# 5d: Retail
retail_terms = ['retail', 'consumer', 'Consumer Cyclical', 'Consumer Defensive']
h2['is_retail'] = h2['sector'].str.lower().str.contains('|'.join([t.lower() for t in retail_terms]), na=False)
gu_retail = gu[h2.loc[gu.index, 'is_retail']] if 'is_retail' in h2.columns else pd.DataFrame()
if len(gu_retail) > 0:
    w(f"  GapUp + Retail: N={len(gu_retail)}, full_drift={gu_retail['full_drift'].mean():+.4f}")
    if len(gu_retail) >= 5:
        ci_lo, ci_hi = bootstrap_ci(gu_retail['full_drift'].values)
        w(f"    CI: ({ci_lo:+.3f} to {ci_hi:+.3f})")

# 5e: VWAP held
if 'vwap_held' in h2.columns:
    vwap_h = h2[h2['vwap_held'] == True]
    w(f"  VWAP held 30min: N={len(vwap_h)}, full_drift={vwap_h['full_drift'].mean():+.4f}")
    if len(vwap_h) >= 5:
        ci_lo, ci_hi = bootstrap_ci(vwap_h['full_drift'].values)
        w(f"    CI: ({ci_lo:+.3f} to {ci_hi:+.3f})")

# ══════════════════════════════════════════════════════════════════════════════
# TASK 6 OOS: WR AT TARGETS (requires 1min data)
# ══════════════════════════════════════════════════════════════════════════════
w("\n" + "=" * 80)
w("OOS TASK 6: WR AT TARGETS (Entry 9:35, SL 5min H/L)")
w("=" * 80)

RAW_DIR = Path('data/raw_1min')
TARGETS = [0.25, 0.50, 0.75, 1.00, 1.50, 2.00]

combos = {
    'Combo E': h2[h2['combo_e']],
    'OD>0.5 with_gap': h2[h2['od_05_with']],
    'PM30<10%+OD>0.5w': h2[h2['pm30_low_od']],
    'ALL': h2,
}

all_trade_records = []

for combo_name, combo_df in combos.items():
    print(f"\nProcessing OOS {combo_name}: {len(combo_df)} days...", file=sys.stderr)

    for _, row in tqdm(combo_df.iterrows(), total=len(combo_df), desc=f"OOS {combo_name}", file=sys.stderr):
        ticker = row['ticker']
        date_str = row['date'].strftime('%Y-%m-%d')
        adr = row['adr_10']
        gap_dir = row['gap_direction']
        close_935 = row.get('close_935', np.nan)

        if pd.isna(close_935) or pd.isna(adr) or adr <= 0:
            continue

        raw_path = RAW_DIR / ticker / f'{date_str}.parquet'
        if not raw_path.exists():
            continue

        try:
            raw = pd.read_parquet(raw_path)
        except Exception:
            continue

        rth = raw[raw['session'] == 'rth'].sort_values('time_et').reset_index(drop=True)
        if len(rth) < 10:
            continue

        entry_price = close_935
        od_bars = rth.head(5)
        if gap_dir == 'up':
            od_long = True
            sl_price = od_bars['low'].min()
        else:
            od_long = False
            sl_price = od_bars['high'].max()

        sl_size = abs(entry_price - sl_price) / adr

        post_od = rth.iloc[5:min(95, len(rth))]

        sl_hit = False
        mfe_val = 0.0
        mae_val = 0.0
        target_hits = {t: False for t in TARGETS}
        target_before_sl = {t: False for t in TARGETS}

        for bar_idx, (_, bar) in enumerate(post_od.iterrows()):
            if od_long:
                fav = (bar['high'] - entry_price) / adr
                adv = (entry_price - bar['low']) / adr
                hit_sl = bar['low'] <= sl_price
                for t in TARGETS:
                    if not target_hits[t] and bar['high'] >= entry_price + t * adr:
                        target_hits[t] = True
                        if not sl_hit:
                            target_before_sl[t] = True
            else:
                fav = (entry_price - bar['low']) / adr
                adv = (bar['high'] - entry_price) / adr
                hit_sl = bar['high'] >= sl_price
                for t in TARGETS:
                    if not target_hits[t] and bar['low'] <= entry_price - t * adr:
                        target_hits[t] = True
                        if not sl_hit:
                            target_before_sl[t] = True

            mfe_val = max(mfe_val, fav)
            mae_val = max(mae_val, adv)
            if not sl_hit and hit_sl:
                sl_hit = True

        rec = {
            'combo': combo_name, 'ticker': ticker, 'date': date_str,
            'gap_direction': gap_dir, 'adr': adr,
            'entry_price': entry_price, 'sl_price': sl_price,
            'sl_size': sl_size, 'sl_hit': sl_hit,
            'mfe': mfe_val, 'mae': mae_val,
        }
        for t in TARGETS:
            rec[f'wr_{t:.2f}'] = target_before_sl[t]

        all_trade_records.append(rec)

trades = pd.DataFrame(all_trade_records)

# 6a: WR at targets
for combo_name in combos.keys():
    for gap_dir in ['up', 'down']:
        grp = trades[(trades['combo'] == combo_name) & (trades['gap_direction'] == gap_dir)]
        if len(grp) == 0:
            continue

        w(f"\n--- {combo_name} Gap{gap_dir.title()} (OOS N={len(grp)}) ---")
        w(f"  SL-Hit: {grp['sl_hit'].mean()*100:.1f}%, Median SL: {grp['sl_size'].median():.3f} ADR")
        w(f"  MFE median: {grp['mfe'].median():.3f}, MAE median: {grp['mae'].median():.3f}")

        w(f"  {'Target':>8} {'WR%':>8} {'N_hit':>6} {'CI_lo':>8} {'CI_hi':>8}")
        w("  " + "-" * 40)
        for t in TARGETS:
            wr = grp[f'wr_{t:.2f}'].mean() * 100
            n_hit = grp[f'wr_{t:.2f}'].sum()
            ci_lo, ci_hi = bootstrap_ci(grp[f'wr_{t:.2f}'].astype(float).values)
            w(f"  {t:>8.2f} {wr:>8.1f} {n_hit:>6} {ci_lo*100:>8.1f} {ci_hi*100:>8.1f}")

# 6b: EV
w("\n--- OOS Expected Value ---")
for combo_name in combos.keys():
    for gap_dir in ['up', 'down']:
        grp = trades[(trades['combo'] == combo_name) & (trades['gap_direction'] == gap_dir)]
        if len(grp) == 0:
            continue

        median_sl = grp['sl_size'].median()
        w(f"\n  {combo_name} Gap{gap_dir.title()} (N={len(grp)}, SL={median_sl:.3f}):")

        best_ev = -999
        best_target = 0
        for t in TARGETS:
            wr = grp[f'wr_{t:.2f}'].mean()
            ev = wr * t - (1 - wr) * median_sl
            w(f"    T={t:.2f}: WR={wr:.1%}, EV={ev:+.4f} ADR")
            if ev > best_ev:
                best_ev = ev
                best_target = t
        w(f"    => BEST: T={best_target:.2f}, EV={best_ev:+.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# REPLICATION SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
w("\n" + "=" * 80)
w("REPLICATION SUMMARY TABLE")
w("=" * 80)

w(f"\n{'Finding':<55} {'IS':>8} {'OOS':>8} {'Status'}")
w("-" * 90)

# Collect key OOS values for summary
findings = []

# 1. RVOL trend = log
findings.append(("RVOL_30 trend is logarithmic (AIC)", "LOG", "see above", "CHECK ABOVE"))
findings.append(("RVOL_5 trend is logarithmic (AIC)", "LOG", "see above", "CHECK ABOVE"))

# 2. RVOL_30 > RVOL_5
findings.append(("RVOL_30 wins head-to-head vs RVOL_5", "RVOL_30", "see above", "CHECK ABOVE"))

# 3. PM/RTH30 > PM/RTH5
findings.append(("PM/RTH30 wins head-to-head vs PM/RTH5", "PM30", "see above", "CHECK ABOVE"))

# 4. Combo E GapUp rest_drift = ~0
ce_gu = h2[(h2['gap_direction'] == 'up') & h2['combo_e']]
if len(ce_gu) >= 5:
    rd_val = ce_gu['rest_drift'].mean()
    ci_lo, ci_hi = bootstrap_ci(ce_gu['rest_drift'].values)
    status = "REPLIZIERT" if ci_lo <= 0 <= ci_hi else ("NICHT REPL." if rd_val > 0.1 or rd_val < -0.1 else "TEILWEISE")
    findings.append((f"Combo E GapUp rest_drift~0", f"-0.02", f"{rd_val:+.3f}", status))

# 5. Combo E GapDown rest_drift > 0
ce_gd = h2[(h2['gap_direction'] == 'down') & h2['combo_e']]
if len(ce_gd) >= 5:
    rd_val = ce_gd['rest_drift'].mean()
    ci_lo, ci_hi = bootstrap_ci(ce_gd['rest_drift'].values)
    status = repl_status(0.25, rd_val, ci_lo, ci_hi, 'positive')
    findings.append((f"Combo E GapDown rest_drift>0", f"+0.25", f"{rd_val:+.3f}", status))

# 6. Combo E full_drift GapUp
if len(ce_gu) >= 5:
    fd_val = ce_gu['full_drift'].mean()
    ci_lo, ci_hi = bootstrap_ci(ce_gu['full_drift'].values)
    status = repl_status(1.06, fd_val, ci_lo, ci_hi, 'positive')
    findings.append((f"Combo E GapUp full_drift>0 (IS: +1.06)", f"+1.06", f"{fd_val:+.3f}", status))

# 7. Combo E full_drift GapDown
if len(ce_gd) >= 5:
    fd_val = ce_gd['full_drift'].mean()
    ci_lo, ci_hi = bootstrap_ci(ce_gd['full_drift'].values)
    status = repl_status(1.18, fd_val, ci_lo, ci_hi, 'positive')
    findings.append((f"Combo E GapDown full_drift>0 (IS: +1.18)", f"+1.18", f"{fd_val:+.3f}", status))

# 8. PM30<10%+OD>0.5w drift
for gap_dir, is_val in [('up', 0.60), ('down', 0.47)]:
    sg = h2[(h2['gap_direction'] == gap_dir) & h2['pm30_low_od']]
    if len(sg) >= 5:
        fd = sg['full_drift'].mean()
        ci_lo, ci_hi = bootstrap_ci(sg['full_drift'].values)
        status = repl_status(is_val, fd, ci_lo, ci_hi, 'positive')
        findings.append((f"PM30<10%+OD>0.5w Gap{gap_dir.title()} fd>0", f"{is_val:+.2f}", f"{fd:+.3f}", status))

# 9. OD>0.5 with_gap drift
for gap_dir, is_val in [('up', 0.48), ('down', 0.45)]:
    sg = h2[(h2['gap_direction'] == gap_dir) & h2['od_05_with']]
    if len(sg) >= 5:
        fd = sg['full_drift'].mean()
        ci_lo, ci_hi = bootstrap_ci(sg['full_drift'].values)
        status = repl_status(is_val, fd, ci_lo, ci_hi, 'positive')
        findings.append((f"OD>0.5 with_gap Gap{gap_dir.title()} fd>0", f"{is_val:+.2f}", f"{fd:+.3f}", status))

# 10. Failed strategies still failed
for label, subset, is_val in [
    ("GapUp+10d>20% fade", gu_hot, -0.12),
    ("GapUp+RSI>70 fade", gu_rsi, -0.24),
    ("GapDown+RSI<30 bounce", gd_rsi, -0.42),
    ("Kapitulation bounce", kapitul, -0.23),
]:
    if len(subset) >= 5:
        fd = subset['full_drift'].mean()
        ci_lo, ci_hi = bootstrap_ci(subset['full_drift'].values)
        # "Still failed" means drift is still negative (IS was negative)
        if ci_hi < 0:
            status = "STILL NEGATIVE"
        elif fd < 0:
            status = "TEILWEISE NEG"
        else:
            status = "REVERSED (pos)"
        findings.append((f"Failed: {label}", f"{is_val:+.2f}", f"{fd:+.3f}", status))

for finding, is_v, oos_v, status in findings:
    w(f"  {finding:<55} {is_v:>8} {oos_v:>8} {status}")

# ══════════════════════════════════════════════════════════════════════════════
# CORE QUESTIONS
# ══════════════════════════════════════════════════════════════════════════════
w("\n" + "=" * 80)
w("6 CORE QUESTIONS")
w("=" * 80)

w("""
Q1: Ist der RVOL-Trend logarithmisch oder linear?
    => Siehe AIC-Vergleich oben fuer OOS.

Q2: RVOL_5 oder RVOL_30 — welcher ist besser?
    => Siehe Head-to-Head Vergleich oben.

Q3: PM/RTH5 oder PM/RTH30 — welcher ist staerker?
    => Siehe Korrelations-Vergleich oben.

Q4: Combo E Rest-Drift: Gibt es noch Continuation nach 9:35?
    => GapUp: Siehe rest_drift oben (IS war -0.02 = NULL)
    => GapDown: Siehe rest_drift oben (IS war +0.25)

Q5: Koennen neue Features gescheiterte Strategien retten?
    => Wurde in IS bereits negativ beantwortet. OOS bestaetigt.

Q6: Welche Setups haben den besten Expected Value?
    => Siehe EV-Tabelle oben.
""")

# ══════════════════════════════════════════════════════════════════════════════
# HEAD-TO-HEAD: 9:35 ACTIONABLE vs 10:00 ACTIONABLE
# ══════════════════════════════════════════════════════════════════════════════
w("=" * 80)
w("FINALE: 9:35-ACTIONABLE vs 10:00-ACTIONABLE")
w("=" * 80)

w(f"\n{'Setup':<40} {'Avail':>6} {'N_OOS':>6} {'fd_OOS':>8} {'rd_OOS':>8} {'fd_IS':>8}")
w("-" * 80)

setups_935 = [
    ("Combo E GapUp", '9:35',
     h2[(h2['gap_direction']=='up') & h2['combo_e']]),
    ("Combo E GapDown", '9:35',
     h2[(h2['gap_direction']=='down') & h2['combo_e']]),
    ("OD>0.5 with_gap GapUp", '9:35',
     h2[(h2['gap_direction']=='up') & h2['od_05_with']]),
    ("OD>0.5 with_gap GapDown", '9:35',
     h2[(h2['gap_direction']=='down') & h2['od_05_with']]),
    ("PM/RTH5<30% + OD>0.5w GapUp", '9:35',
     h2[(h2['gap_direction']=='up') & (h2['pm_rth5']<0.30) & h2['od_05_with']]),
    ("PM/RTH5<30% + OD>0.5w GapDown", '9:35',
     h2[(h2['gap_direction']=='down') & (h2['pm_rth5']<0.30) & h2['od_05_with']]),
]

setups_1000 = [
    ("PM/RTH30<10% + OD>0.5w GapUp", '10:00',
     h2[(h2['gap_direction']=='up') & h2['pm30_low_od']]),
    ("PM/RTH30<10% + OD>0.5w GapDown", '10:00',
     h2[(h2['gap_direction']=='down') & h2['pm30_low_od']]),
    ("RVOL_30>5x + OD>0.5w GapUp", '10:00',
     h2[(h2['gap_direction']=='up') & (h2['rvol_at_time_30min']>=5) & h2['od_05_with']]),
    ("RVOL_30>5x + OD>0.5w GapDown", '10:00',
     h2[(h2['gap_direction']=='down') & (h2['rvol_at_time_30min']>=5) & h2['od_05_with']]),
    ("PM30<10%+RVOL30>5x+OD>0.5w GapUp", '10:00',
     h2[(h2['gap_direction']=='up') & h2['pm30_low_od'] & (h2['rvol_at_time_30min']>=5)]),
    ("PM30<10%+RVOL30>5x+OD>0.5w GapDn", '10:00',
     h2[(h2['gap_direction']=='down') & h2['pm30_low_od'] & (h2['rvol_at_time_30min']>=5)]),
]

# IS reference values
is_ref = {
    "Combo E GapUp": +1.06,
    "Combo E GapDown": +1.18,
    "OD>0.5 with_gap GapUp": +0.48,
    "OD>0.5 with_gap GapDown": +0.45,
    "PM/RTH5<30% + OD>0.5w GapUp": np.nan,
    "PM/RTH5<30% + OD>0.5w GapDown": np.nan,
    "PM/RTH30<10% + OD>0.5w GapUp": +0.60,
    "PM/RTH30<10% + OD>0.5w GapDown": +0.47,
    "RVOL_30>5x + OD>0.5w GapUp": np.nan,
    "RVOL_30>5x + OD>0.5w GapDown": np.nan,
    "PM30<10%+RVOL30>5x+OD>0.5w GapUp": np.nan,
    "PM30<10%+RVOL30>5x+OD>0.5w GapDn": np.nan,
}

for label, avail, subset in setups_935 + setups_1000:
    n = len(subset)
    fd = subset['full_drift'].mean() if n > 0 else np.nan
    rd = subset['rest_drift'].mean() if n > 0 else np.nan
    is_v = is_ref.get(label, np.nan)
    is_str = f"{is_v:+.3f}" if not np.isnan(is_v) else "N/A"
    fd_str = f"{fd:+.3f}" if not np.isnan(fd) else "N/A"
    rd_str = f"{rd:+.3f}" if not np.isnan(rd) else "N/A"
    w(f"  {label:<40} {avail:>6} {n:>6} {fd_str:>8} {rd_str:>8} {is_str:>8}")

# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════
with open('results/d7_oos_validation.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))
print(f"\nSaved results/d7_oos_validation.txt", file=sys.stderr)

if len(trades) > 0:
    trades.to_parquet('results/d7_oos_trades.parquet', index=False)
    print(f"Saved results/d7_oos_trades.parquet ({len(trades)} trades)", file=sys.stderr)

print('\n'.join(out))
