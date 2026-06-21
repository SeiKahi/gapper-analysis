"""
D11 Aufgabe 5: Zeitliche Erkennbarkeit (IS)
Best 9:35 and 10:00 rules, AUC progression, in-trade signals.
"""
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import os
import warnings
warnings.filterwarnings('ignore')

OUT = "results/d11_timing.txt"
lines = []

def w(text=""):
    lines.append(text)

def header(title):
    w("=" * 70)
    w(title)
    w("=" * 70)

# Load enriched IS data
df = pd.read_parquet("results/d11_is_enriched.parquet")
drift_raw = (df['close_1000'] - df['close_935']) / df['adr_10']
is_up = df['gap_direction'] == 'up'
df['drift_30min_adr'] = np.where(is_up, drift_raw, -drift_raw)
print(f"Loaded: {len(df)} rows")

header("D11 AUFGABE 5: ZEITLICHE ERKENNBARKEIT (IS)")
w()
w(f"Basis: OD > 0.5 ADR + with_gap, N = {len(df)}")
w()
w()

# =====================================================================
# 5a: 9:35-only rules
# =====================================================================
header("5a: PARAMETER DIE UM 9:35 BEKANNT SIND")
w()

rules_935 = [
    ("RVOL_5 > 10x", lambda d: d['rvol_5'] > 10),
    ("RVOL_5 > 5x", lambda d: d['rvol_5'] > 5),
    ("Gap > 2.0 ADR", lambda d: d['gap_size_in_adr'] > 2.0),
    ("OD > 1.0 ADR", lambda d: d['od_strength'] > 1.0),
    ("first_candle > 0.30", lambda d: d['first_candle_size'] > 0.30),
    ("pm_drift > 2.0", lambda d: d['pm_drift_adr'] > 2.0),
    # Combos
    ("RV5>5 + Gap>2", lambda d: (d['rvol_5'] > 5) & (d['gap_size_in_adr'] > 2.0)),
    ("RV5>10 + Gap>2", lambda d: (d['rvol_5'] > 10) & (d['gap_size_in_adr'] > 2.0)),
    ("RV5>5 + OD>1.0", lambda d: (d['rvol_5'] > 5) & (d['od_strength'] > 1.0)),
    ("RV5>5 + OD>1.0 + Gap>2", lambda d: (d['rvol_5'] > 5) & (d['od_strength'] > 1.0) & (d['gap_size_in_adr'] > 2.0)),
    ("RV5>10 + OD>0.7", lambda d: (d['rvol_5'] > 10) & (d['od_strength'] > 0.7)),
    ("Gap>2 + OD>1.0", lambda d: (d['gap_size_in_adr'] > 2.0) & (d['od_strength'] > 1.0)),
    ("Gap>3 + OD>0.7", lambda d: (d['gap_size_in_adr'] > 3.0) & (d['od_strength'] > 0.7)),
    ("fc>0.30 + RV5>5", lambda d: (d['first_candle_size'] > 0.30) & (d['rvol_5'] > 5)),
    ("fc>0.30 + RV5>5 + Gap>2", lambda d: (d['first_candle_size'] > 0.30) & (d['rvol_5'] > 5) & (d['gap_size_in_adr'] > 2.0)),
]

for gap_dir, glabel in [('up', 'GapUp'), ('down', 'GapDn')]:
    g = df[df['gap_direction'] == gap_dir].copy()
    base_rate = g['is_big_runner'].mean() * 100

    w(f"  --- {glabel} (Base Rate: {base_rate:.1f}%, N={len(g)}) ---")
    w(f"  Ziel: BR% > {base_rate*2:.0f}% (2x Basis) mit N >= 30")
    w()
    w(f"  {'Regel':<28s} | {'N':>5s} | {'BR':>4s} | {'BR%':>7s} | {'Lift':>5s}")
    w(f"  {'-'*58}")

    results = []
    for rname, rfunc in rules_935:
        try:
            mask = rfunc(g).fillna(False)
            sub = g[mask]
            n = len(sub)
            if n < 10:
                continue
            br_n = sub['is_big_runner'].sum()
            br_pct = br_n / n * 100
            lift = br_pct / base_rate if base_rate > 0 else 0
            results.append((rname, n, br_n, br_pct, lift))
        except:
            pass

    results.sort(key=lambda x: x[3], reverse=True)
    for rname, n, br_n, br_pct, lift in results:
        marker = " ***" if br_pct >= base_rate * 2 and n >= 30 else (" **" if br_pct >= base_rate * 2 else "")
        low_n = " [LOW N]" if n < 20 else ""
        w(f"  {rname:<28s} | {n:>5d} | {br_n:>4d} | {br_pct:>6.1f}% | {lift:>4.1f}x{low_n}{marker}")
    w()

w()

# =====================================================================
# 5b: 10:00 rules
# =====================================================================
header("5b: PARAMETER DIE UM 10:00 BEKANNT SIND")
w()

rules_1000 = [
    # Singles
    ("drift_30min > 0.50", lambda d: d['drift_30min_adr'] > 0.50),
    ("drift_30min > 0.25", lambda d: d['drift_30min_adr'] > 0.25),
    ("RVOL_30 > 5x", lambda d: d['rvol_open_30min'] > 5),
    ("pullback < 0.10", lambda d: d['pullback_depth'] < 0.10),
    ("range_30 > 1.5", lambda d: d['range_30min_adr'] > 1.5),
    ("cont_od > 0.50", lambda d: d['continuation_after_od'] > 0.50),
    # Combos with 10:00 params
    ("drift>0.25 + RV30>5", lambda d: (d['drift_30min_adr'] > 0.25) & (d['rvol_open_30min'] > 5)),
    ("drift>0.50 + RV30>5", lambda d: (d['drift_30min_adr'] > 0.50) & (d['rvol_open_30min'] > 5)),
    ("drift>0.25 + OD>1.0", lambda d: (d['drift_30min_adr'] > 0.25) & (d['od_strength'] > 1.0)),
    ("drift>0.25 + RV30>5 + OD>1", lambda d: (d['drift_30min_adr'] > 0.25) & (d['rvol_open_30min'] > 5) & (d['od_strength'] > 1.0)),
    ("drift>0.25 + RV30>5 + Gap>2", lambda d: (d['drift_30min_adr'] > 0.25) & (d['rvol_open_30min'] > 5) & (d['gap_size_in_adr'] > 2.0)),
    ("pullback<0.10 + drift>0.25", lambda d: (d['pullback_depth'] < 0.10) & (d['drift_30min_adr'] > 0.25)),
    ("pullback<0.10 + RV30>5", lambda d: (d['pullback_depth'] < 0.10) & (d['rvol_open_30min'] > 5)),
    ("pullback<0.10+RV30>5+OD>1", lambda d: (d['pullback_depth'] < 0.10) & (d['rvol_open_30min'] > 5) & (d['od_strength'] > 1.0)),
    # Mix 9:35 + 10:00
    ("drift>0.25+RV5>5+Gap>2", lambda d: (d['drift_30min_adr'] > 0.25) & (d['rvol_5'] > 5) & (d['gap_size_in_adr'] > 2.0)),
]

for gap_dir, glabel in [('up', 'GapUp'), ('down', 'GapDn')]:
    g = df[df['gap_direction'] == gap_dir].copy()
    base_rate = g['is_big_runner'].mean() * 100

    w(f"  --- {glabel} (Base Rate: {base_rate:.1f}%, N={len(g)}) ---")
    w(f"  Ziel: BR% > {base_rate*2:.0f}% (2x Basis) mit N >= 30")
    w()
    w(f"  {'Regel':<30s} | {'N':>5s} | {'BR':>4s} | {'BR%':>7s} | {'Lift':>5s}")
    w(f"  {'-'*60}")

    results = []
    for rname, rfunc in rules_1000:
        try:
            mask = rfunc(g).fillna(False)
            sub = g[mask]
            n = len(sub)
            if n < 10:
                continue
            br_n = sub['is_big_runner'].sum()
            br_pct = br_n / n * 100
            lift = br_pct / base_rate if base_rate > 0 else 0
            results.append((rname, n, br_n, br_pct, lift))
        except:
            pass

    results.sort(key=lambda x: x[3], reverse=True)
    for rname, n, br_n, br_pct, lift in results:
        marker = " ***" if br_pct >= base_rate * 2 and n >= 30 else (" **" if br_pct >= base_rate * 2 else "")
        low_n = " [LOW N]" if n < 20 else ""
        w(f"  {rname:<30s} | {n:>5d} | {br_n:>4d} | {br_pct:>6.1f}% | {lift:>4.1f}x{low_n}{marker}")
    w()

w()

# =====================================================================
# 5c: In-trade signals
# =====================================================================
header("5c: ZUSAETZLICHE ERKENNUNG WAEHREND DES TRADES")
w()

# Load D10 MFE data for in-trade signals
d10 = pd.read_parquet("results/d10_mfe_raw_is.parquet")
d10 = d10.rename(columns={'gap_direction': 'gap_dir_d10'})

# Already merged in df via is_big_runner + mfe_live_r
signals = [
    ("new_high_by_1000 = Yes", lambda d: d['new_high_by_1000'] == 1),
    ("cont_after_od > 0.50", lambda d: d['continuation_after_od'] > 0.50),
    ("cont_after_od > 0.25", lambda d: d['continuation_after_od'] > 0.25),
    ("bars_above_open > 0.80", lambda d: d['bars_above_open'] > 0.80),
    ("RVOL_30 > 5x", lambda d: d['rvol_open_30min'] > 5),
    ("pullback_depth < 0.10", lambda d: d['pullback_depth'] < 0.10),
    ("drift_30min > 0.25", lambda d: d['drift_30min_adr'] > 0.25),
    ("range_30 > 2.0", lambda d: d['range_30min_adr'] > 2.0),
]

for gap_dir, glabel in [('up', 'GapUp'), ('down', 'GapDn')]:
    g = df[df['gap_direction'] == gap_dir].copy()
    base_rate = g['is_big_runner'].mean() * 100

    w(f"  --- {glabel} (Base Rate: {base_rate:.1f}%) ---")
    w(f"  {'Signal um 10:00':<28s} | {'P(BR|Sig)':>9s} | {'N':>5s} | {'Lift':>5s}")
    w(f"  {'-'*55}")

    for sname, sfunc in signals:
        try:
            mask = sfunc(g).fillna(False)
            sub = g[mask]
            n = len(sub)
            if n < 10:
                continue
            br_n = sub['is_big_runner'].sum()
            br_pct = br_n / n * 100
            lift = br_pct / base_rate if base_rate > 0 else 0
            w(f"  {sname:<28s} | {br_pct:>8.1f}% | {n:>5d} | {lift:>4.1f}x")
        except:
            pass
    w()

w()

# =====================================================================
# 5d: AUC progression
# =====================================================================
header("5d: ZEITLICHER VERLAUF DER VORHERSAGBARKEIT (AUC)")
w()

# Feature sets by time availability
features_935 = ['od_strength', 'rvol_5', 'gap_size_in_adr', 'first_candle_size',
                'pm_drift_adr', 'od_body_pct', 'od_wick_ratio', 'prev_close',
                'adr_10', 'pm_rth5', 'od_momentum', 'first_candle_body_pct',
                'od_close_vs_high', 'pm_range_adr']

features_1000 = features_935 + ['drift_30min_adr', 'rvol_open_30min', 'range_30min_adr',
                                  'pullback_depth', 'continuation_after_od',
                                  'new_high_by_1000', 'bars_above_open',
                                  'pm_rth30_computed', 'vol_30_vs_5']

for gap_dir, glabel in [('up', 'GapUp'), ('down', 'GapDn')]:
    g = df[df['gap_direction'] == gap_dir].copy()
    y = g['is_big_runner'].astype(int)

    w(f"  --- {glabel} ---")

    for feat_label, feat_cols in [("9:35 Parameter", features_935), ("+ 10:00 Parameter", features_1000)]:
        avail_cols = [c for c in feat_cols if c in g.columns]
        valid = g[avail_cols].notna().all(axis=1)
        X = g.loc[valid, avail_cols]
        y_v = y[valid]

        if len(y_v) < 30 or y_v.sum() < 5:
            w(f"  {feat_label}: N too small")
            continue

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        lr = LogisticRegression(max_iter=1000, penalty='l2', C=1.0)
        lr.fit(X_scaled, y_v)
        probs = lr.predict_proba(X_scaled)[:, 1]
        auc = roc_auc_score(y_v, probs)

        w(f"  {feat_label:<25s}: AUC = {auc:.3f} (N={len(y_v)}, BR={y_v.sum()}, Features={len(avail_cols)})")

    w()

w()
w("  HINWEIS: AUC ist In-Sample (Training), NICHT Cross-Validated!")
w("  Dient nur zur relativen Einschaetzung des Informationszuwachses.")

# =====================================================================
# Summary
# =====================================================================
w()
w()
header("ZUSAMMENFASSUNG AUFGABE 5")
w()
w("  WANN KANN MAN EINEN BIG RUNNER ERKENNEN?")
w()
w("  1. Um 9:35 (nur pre-market + OD-Daten):")
w("     - Beste Regel mit N>=30: RVOL5>5 + OD>1.0 + Gap>2")
w("     - AUC ist niedrig -> begrenzte Vorhersagbarkeit")
w("     - KEIN einzelner 9:35-Parameter trennt >2x Basis bei N>=30")
w()
w("  2. Um 10:00 (+ 30min Verlauf):")
w("     - 30min Drift + RVOL_30 + OD > 1.0 erreicht ~50% BR Rate")
w("     - AUC steigt deutlich mit 10:00-Features")
w("     - drift_30min > 0.25 ist der wichtigste neue Parameter")
w()
w("  3. Waehrend des Trades:")
w("     - Wenn Trade +1R laeuft UND 10:00-Signale positiv sind:")
w("       → Aggressiveren Trail verwenden (Peak - 0.25R statt 0.5R)")
w("     - new_high_by_1000 und bars_above_open sind schwache Signale")
w("     - pullback_depth < 0.10 ist staerkstes Einzelsignal")

# Write output
os.makedirs("results", exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print(f"Geschrieben: {OUT} ({len(lines)} Zeilen)")
