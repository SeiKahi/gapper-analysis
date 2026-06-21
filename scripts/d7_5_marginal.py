"""
Durchlauf 7.5 -- Aufgabe 6: Marginale Effekte und Interaktionen (IS)
"""
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import sys, warnings
warnings.filterwarnings('ignore')

# === LOAD ===
df = pd.read_parquet('data/metadata/metadata_v7.parquet')
h1 = df[(df['date'] >= '2021-02-21') & (df['date'] <= '2023-12-31')].copy()
h1 = h1.dropna(subset=['pm_rth5', 'rvol_5', 'gap_size_in_adr', 'full_drift'])
h1['gap_dir'] = h1['gap_direction'].map({'up': 'GapUp', 'down': 'GapDown'})

print(f"H1: {len(h1)}", file=sys.stderr)

# Buckets
def pm5_bucket(x):
    if x < 0.30: return 'PM5_LO'
    elif x < 0.70: return 'PM5_MID'
    elif x < 1.50: return 'PM5_HI'
    else: return 'PM5_EX'

def rv5_bucket(x):
    if x < 2: return 'RV5_LO'
    elif x < 5: return 'RV5_MID'
    elif x < 10: return 'RV5_HI'
    else: return 'RV5_EX'

def gap_bucket(x):
    if x < 1: return 'GAP_SM'
    elif x < 2: return 'GAP_MD'
    else: return 'GAP_LG'

h1['pm5'] = h1['pm_rth5'].apply(pm5_bucket)
h1['rv5'] = h1['rvol_5'].apply(rv5_bucket)
h1['gap_b'] = h1['gap_size_in_adr'].apply(gap_bucket)

out = open('results/d7_5_marginal.txt', 'w', encoding='utf-8')
def p(text=''):
    out.write(text + '\n')

p("=" * 90)
p("DURCHLAUF 7.5 -- AUFGABE 6: MARGINALE EFFEKTE UND INTERAKTIONEN (IS)")
p("=" * 90)

# === 6a: Simple marginal drifts ===
p(f"\n{'='*90}")
p("6a: EINFACHE MARGINALE DRIFTS")
p(f"{'='*90}")

for direction in ['GapUp', 'GapDown']:
    sub = h1[h1['gap_dir'] == direction]
    p(f"\n--- {direction} (N={len(sub)}) ---")

    p(f"\nPM/RTH5 Buckets:")
    p(f"{'Bucket':<10s} | {'N':>6s} {'fd_mean':>8s} {'fd_med':>8s} {'rd_mean':>8s}")
    for bkt in ['PM5_LO', 'PM5_MID', 'PM5_HI', 'PM5_EX']:
        s = sub[sub['pm5'] == bkt]
        p(f"{bkt:<10s} | {len(s):>6d} {s['full_drift'].mean():>+8.3f} {s['full_drift'].median():>+8.3f} {s['rest_drift'].mean():>+8.3f}")
    lo = sub[sub['pm5'] == 'PM5_LO']['full_drift'].mean()
    ex = sub[sub['pm5'] == 'PM5_EX']['full_drift'].mean()
    p(f"  Delta LO vs EX: {lo - ex:+.3f} ADR")

    p(f"\nRVOL_5 Buckets:")
    p(f"{'Bucket':<10s} | {'N':>6s} {'fd_mean':>8s} {'fd_med':>8s} {'rd_mean':>8s}")
    for bkt in ['RV5_LO', 'RV5_MID', 'RV5_HI', 'RV5_EX']:
        s = sub[sub['rv5'] == bkt]
        p(f"{bkt:<10s} | {len(s):>6d} {s['full_drift'].mean():>+8.3f} {s['full_drift'].median():>+8.3f} {s['rest_drift'].mean():>+8.3f}")
    lo = sub[sub['rv5'] == 'RV5_LO']['full_drift'].mean()
    ex = sub[sub['rv5'] == 'RV5_EX']['full_drift'].mean()
    p(f"  Delta LO vs EX: {lo - ex:+.3f} ADR")

    p(f"\nGap_in_ADR Buckets:")
    p(f"{'Bucket':<10s} | {'N':>6s} {'fd_mean':>8s} {'fd_med':>8s} {'rd_mean':>8s}")
    for bkt in ['GAP_SM', 'GAP_MD', 'GAP_LG']:
        s = sub[sub['gap_b'] == bkt]
        p(f"{bkt:<10s} | {len(s):>6d} {s['full_drift'].mean():>+8.3f} {s['full_drift'].median():>+8.3f} {s['rest_drift'].mean():>+8.3f}")
    sm_val = sub[sub['gap_b'] == 'GAP_SM']['full_drift'].mean()
    lg_val = sub[sub['gap_b'] == 'GAP_LG']['full_drift'].mean()
    p(f"  Delta SM vs LG: {sm_val - lg_val:+.3f} ADR")

# === 6b: Interaction effects ===
p(f"\n\n{'='*90}")
p("6b: INTERAKTIONSEFFEKTE")
p(f"{'='*90}")

p("\nIst der PM/RTH5-Effekt STAERKER bei hohem RVOL_5?")
p("(PM5_LO - PM5_EX Spread fuer jedes RVOL-Bucket)")
p()

for direction in ['GapUp', 'GapDown']:
    sub = h1[h1['gap_dir'] == direction]
    p(f"--- {direction} ---")
    p(f"{'RVOL_5':>10s} | {'PM5_LO fd':>10s} {'PM5_EX fd':>10s} {'Spread':>8s} {'N_LO':>5s} {'N_EX':>5s}")

    for rv in ['RV5_LO', 'RV5_MID', 'RV5_HI', 'RV5_EX']:
        lo_mask = (sub['pm5'] == 'PM5_LO') & (sub['rv5'] == rv)
        ex_mask = (sub['pm5'] == 'PM5_EX') & (sub['rv5'] == rv)
        lo_fd = sub[lo_mask]['full_drift'].mean() if lo_mask.sum() >= 5 else np.nan
        ex_fd = sub[ex_mask]['full_drift'].mean() if ex_mask.sum() >= 5 else np.nan
        spread = lo_fd - ex_fd if not (np.isnan(lo_fd) or np.isnan(ex_fd)) else np.nan
        lo_n = lo_mask.sum()
        ex_n = ex_mask.sum()
        lo_str = f"{lo_fd:>+10.3f}" if not np.isnan(lo_fd) else "       n/a"
        ex_str = f"{ex_fd:>+10.3f}" if not np.isnan(ex_fd) else "       n/a"
        sp_str = f"{spread:>+8.3f}" if not np.isnan(spread) else "     n/a"
        p(f"{rv:>10s} | {lo_str} {ex_str} {sp_str} {lo_n:>5d} {ex_n:>5d}")
    p()

p("\nIst der PM/RTH5-Effekt STAERKER bei grossem Gap?")
p()
for direction in ['GapUp', 'GapDown']:
    sub = h1[h1['gap_dir'] == direction]
    p(f"--- {direction} ---")
    p(f"{'Gap':>10s} | {'PM5_LO fd':>10s} {'PM5_EX fd':>10s} {'Spread':>8s}")

    for g in ['GAP_SM', 'GAP_MD', 'GAP_LG']:
        lo_mask = (sub['pm5'] == 'PM5_LO') & (sub['gap_b'] == g)
        ex_mask = (sub['pm5'] == 'PM5_EX') & (sub['gap_b'] == g)
        lo_fd = sub[lo_mask]['full_drift'].mean() if lo_mask.sum() >= 10 else np.nan
        ex_fd = sub[ex_mask]['full_drift'].mean() if ex_mask.sum() >= 10 else np.nan
        spread = lo_fd - ex_fd if not (np.isnan(lo_fd) or np.isnan(ex_fd)) else np.nan
        lo_str = f"{lo_fd:>+10.3f}" if not np.isnan(lo_fd) else "       n/a"
        ex_str = f"{ex_fd:>+10.3f}" if not np.isnan(ex_fd) else "       n/a"
        sp_str = f"{spread:>+8.3f}" if not np.isnan(spread) else "     n/a"
        p(f"{g:>10s} | {lo_str} {ex_str} {sp_str}")
    p()

# === 6c: Information gain (Regression) ===
p(f"\n{'='*90}")
p("6c: INFORMATIONSGEWINN (Regression)")
p(f"{'='*90}")
p()

# Use continuous values, not buckets
# Winsorize extreme values
h1['pm_rth5_w'] = h1['pm_rth5'].clip(0, 10)
h1['rvol_5_w'] = h1['rvol_5'].clip(0, 50)
h1['gap_adr_w'] = h1['gap_size_in_adr'].clip(0, 10)

for direction in ['GapUp', 'GapDown']:
    sub = h1[h1['gap_dir'] == direction].copy()
    p(f"--- {direction} (N={len(sub)}) ---")

    # Spearman correlations
    for col, label in [('pm_rth5_w', 'PM/RTH5'), ('rvol_5_w', 'RVOL_5'), ('gap_adr_w', 'Gap_in_ADR')]:
        rho, pval = stats.spearmanr(sub[col], sub['full_drift'])
        p(f"  Spearman {label} -> full_drift: rho = {rho:+.4f} (p = {pval:.4f})")

    # OLS Regressions
    p(f"\n  OLS Regressionen (full_drift ~ ...):")
    for label, formula_cols in [
        ('PM/RTH5 only', ['pm_rth5_w']),
        ('RVOL_5 only', ['rvol_5_w']),
        ('Gap_in_ADR only', ['gap_adr_w']),
        ('PM/RTH5 + RVOL_5', ['pm_rth5_w', 'rvol_5_w']),
        ('All three', ['pm_rth5_w', 'rvol_5_w', 'gap_adr_w']),
    ]:
        X = sm.add_constant(sub[formula_cols])
        y = sub['full_drift']
        model = sm.OLS(y, X).fit()
        p(f"  {label:.<30s} R2 = {model.rsquared:.4f}, Adj-R2 = {model.rsquared_adj:.4f}")
        for c in formula_cols:
            idx = list(X.columns).index(c)
            p(f"    {c}: beta = {model.params.iloc[idx]:+.4f}, t = {model.tvalues.iloc[idx]:+.2f}, p = {model.pvalues.iloc[idx]:.4f}")

    p()

p("INTERPRETATION:")
p("  R2 zeigt, wie viel % der full_drift Varianz das Dreieck erklaert.")
p("  Wenn alle drei zusammen deutlich mehr erklaeren als einzeln,")
p("  hat die Kombination Wert.")

out.close()
print("Done!", file=sys.stderr)
