"""
Durchlauf 7.0 — Aufgabe 5: Gescheiterte Strategien Re-Test mit neuen Features
IS only (H1: 2021-02-21 bis 2023-12-31).
"""
import pandas as pd
import numpy as np
from scipy import stats
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

out = []
def w(s=""): out.append(s)

print("Loading metadata_v7...", file=sys.stderr)
meta = pd.read_parquet('data/metadata/metadata_v7.parquet')
meta['date'] = pd.to_datetime(meta['date'])
h1 = meta[(meta['date'] >= '2021-02-21') & (meta['date'] <= '2023-12-31')].copy()
h1 = h1[h1['adr_10'].notna() & (h1['adr_10'] > 0)].copy()

# Ensure numeric types
for col in ['prior_return_10d', 'rsi_14_prev', 'rvol_5', 'pm_rth5', 'rvol_at_time_30min']:
    if col in h1.columns:
        h1[col] = pd.to_numeric(h1[col], errors='coerce')

def analyze_subgroup(data, label, rvol_col='rvol_5', pm_col='pm_rth5'):
    """Analyze a subgroup with RVOL and PM/RTH buckets."""
    w(f"\n  {label}: N={len(data)}, full_drift={data['full_drift'].mean():+.4f}, rest_drift={data['rest_drift'].mean():+.4f}")

    # RVOL buckets
    v = data[data[rvol_col].notna()].copy()
    if len(v) > 20:
        rvol_edges = [0, 2, 5, 10, 999]
        rvol_labels = ['<2x', '2-5x', '5-10x', '>10x']
        v['rvol_bkt'] = pd.cut(v[rvol_col], bins=rvol_edges, labels=rvol_labels, right=False)

        w(f"    + {rvol_col} buckets:")
        for bkt in rvol_labels:
            grp = v[v['rvol_bkt'] == bkt]
            if len(grp) == 0: continue
            ci_lo, ci_hi = bootstrap_ci(grp['full_drift'].values)
            w(f"      {bkt:<8}: N={len(grp):>4}{low_n(len(grp))}, full_drift={grp['full_drift'].mean():+.4f} ({ci_lo:+.3f} to {ci_hi:+.3f})")

    # PM/RTH5 buckets
    p = data[data[pm_col].notna()].copy()
    if len(p) > 20:
        pm_edges = [0, 0.30, 1.0, 999]
        pm_labels = ['<30%', '30-100%', '>100%']
        p['pm_bkt'] = pd.cut(p[pm_col], bins=pm_edges, labels=pm_labels, right=False)

        w(f"    + {pm_col} buckets:")
        for bkt in pm_labels:
            grp = p[p['pm_bkt'] == bkt]
            if len(grp) == 0: continue
            w(f"      {bkt:<10}: N={len(grp):>4}{low_n(len(grp))}, full_drift={grp['full_drift'].mean():+.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 5a: Prior Performance / RSI
# ══════════════════════════════════════════════════════════════════════════════
w("=" * 80)
w("AUFGABE 5a: PRIOR PERFORMANCE / RSI (F4 — gescheitert OOS)")
w("=" * 80)

# GapUp nach >20% 10d-Run
w("\n--- GapUp + Prior 10d Return > 20% (IS: fade, OOS: continuation) ---")
gu_hot = h1[(h1['gap_direction'] == 'up') & (h1['prior_return_10d'] > 20)].copy()
analyze_subgroup(gu_hot, "GapUp + 10d>20%")

# RSI>70 + GapUp
w("\n--- GapUp + RSI>70 (IS: fade, OOS: continuation) ---")
gu_rsi70 = h1[(h1['gap_direction'] == 'up') & (h1['rsi_14_prev'] > 70)].copy()
analyze_subgroup(gu_rsi70, "GapUp + RSI>70")

# RSI<30 + GapDown
w("\n--- GapDown + RSI<30 (IS: bounce, OOS: only +0.098) ---")
gd_rsi30 = h1[(h1['gap_direction'] == 'down') & (h1['rsi_14_prev'] < 30)].copy()
analyze_subgroup(gd_rsi30, "GapDown + RSI<30")

# ══════════════════════════════════════════════════════════════════════════════
# 5b: Kapitulation
# ══════════════════════════════════════════════════════════════════════════════
w("\n" + "=" * 80)
w("AUFGABE 5b: KAPITULATION (F11 — IS: kein Bounce, OOS: bounced DOCH)")
w("=" * 80)

# GapDown + Prior<-10% + RVOL 2-5x
w("\n--- GapDown + Prior 10d < -10% + RVOL_30 2-5x ---")
kapitul_30 = h1[
    (h1['gap_direction'] == 'down') &
    (h1['prior_return_10d'] < -10) &
    (h1['rvol_at_time_30min'] >= 2) &
    (h1['rvol_at_time_30min'] < 5)
].copy()
analyze_subgroup(kapitul_30, "Kapitulation (RVOL_30 2-5x)")

# With RVOL_5 granular
w("\n--- Kapitulation + RVOL_5 granular ---")
kapitul_all = h1[
    (h1['gap_direction'] == 'down') & (h1['prior_return_10d'] < -10)
].copy()
analyze_subgroup(kapitul_all, "Kapitulation (any RVOL)")

# With PM/RTH5
w("\n--- Kapitulation + PM/RTH5 ---")
analyze_subgroup(kapitul_all, "Kapitulation PM", pm_col='pm_rth5')

# ══════════════════════════════════════════════════════════════════════════════
# 5c: Volumen-Konzentration GapUp
# ══════════════════════════════════════════════════════════════════════════════
w("\n" + "=" * 80)
w("AUFGABE 5c: VOLUMEN-KONZENTRATION GAPUP (F10 — IS: fade, OOS: kein Effekt)")
w("=" * 80)

# Compute vol concentration: volume_first_30min / total_rth_volume
h1['vol_conc_30'] = h1['volume_first_30min'] / h1['total_rth_volume'].replace(0, np.nan)

gu_highvol = h1[
    (h1['gap_direction'] == 'up') & (h1['vol_conc_30'] > 0.30)
].copy()
w(f"\nGapUp + Vol>30% in first 30min:")
analyze_subgroup(gu_highvol, "GapUp + HighVol")

# ══════════════════════════════════════════════════════════════════════════════
# 5d: Sektor Retail
# ══════════════════════════════════════════════════════════════════════════════
w("\n" + "=" * 80)
w("AUFGABE 5d: SEKTOR RETAIL (F19 — IS: momentum, OOS: fade)")
w("=" * 80)

retail_terms = ['retail', 'consumer', 'Consumer Cyclical', 'Consumer Defensive']
h1['is_retail'] = h1['sector'].str.lower().str.contains('|'.join([t.lower() for t in retail_terms]), na=False)

gu_retail = h1[(h1['gap_direction'] == 'up') & h1['is_retail']].copy()
w(f"\nGapUp + Retail sector:")
analyze_subgroup(gu_retail, "GapUp + Retail")

# ══════════════════════════════════════════════════════════════════════════════
# 5e: VWAP-Regime
# ══════════════════════════════════════════════════════════════════════════════
w("\n" + "=" * 80)
w("AUFGABE 5e: VWAP REGIME (F12 — IS: 17% RoD hold, OOS: 5-6%)")
w("=" * 80)

# VWAP held 30min → based on vwap_held column
if 'vwap_held' in h1.columns:
    vwap_held = h1[h1['vwap_held'] == True].copy()
    w(f"\nVWAP held 30min: N={len(vwap_held)}")
    analyze_subgroup(vwap_held, "VWAP held 30min")
else:
    w("\n  vwap_held column not available.")

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
w("\n" + "=" * 80)
w("ZUSAMMENFASSUNG: Koennen neue Features gescheiterte Strategien retten?")
w("=" * 80)
w()
w("Fuer jede gescheiterte Strategie: Gibt es einen RVOL_5 oder PM/RTH5 Bucket,")
w("der den IS-Effekt stabilisiert oder verstaerkt?")
w()
w("Die Antwort steht in den Bucket-Tabellen oben. Kriterien:")
w("  GERETTET: Ein Bucket hat N>=30, gleiche Richtung wie IS, und CI schliesst 0 aus")
w("  NICHT GERETTET: Kein Bucket erfuellt diese Kriterien")

# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════
with open('results/d7_failed_strategies.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))
print(f"Saved results/d7_failed_strategies.txt", file=sys.stderr)
print('\n'.join(out))
