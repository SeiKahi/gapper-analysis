"""
Durchlauf 6.0 — OOS-Validierung Cheat Sheet: Metadata-basierte Findings
Haelfte 2: 2024-01-01 bis 2026-02-06

Validates: F1 (SPY), F2 (RVOL), F3 (Opening Drive partial), F4 (Prior Perf),
           F5 (Gap Fill), F11 (Kapitulation), F15 (HOD/LOD), F16 (Follow-Through),
           F18 (SPY x Gap Size), F19 (Sektor)
"""

import pandas as pd
import numpy as np
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

def ci_overlap(is_lo, is_hi, oos_lo, oos_hi):
    """Check if two CIs overlap."""
    if any(np.isnan([is_lo, is_hi, oos_lo, oos_hi])):
        return "N/A"
    if is_lo <= oos_hi and oos_lo <= is_hi:
        return "JA"
    return "NEIN"

def same_direction(is_val, oos_val):
    """Check if both values have the same sign."""
    if is_val * oos_val > 0:
        return "JA"
    elif abs(oos_val) < 0.01:
        return "NEUTRAL"
    return "NEIN"

def replication_status(is_val, oos_val, is_ci=None, oos_ci=None):
    """Determine replication status."""
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

# Half 2 only
h2 = meta[(meta['date'] >= '2024-01-01') & (meta['date'] <= '2026-02-06')].copy()
h2 = h2[h2['adr_10'].notna()].copy()

# Derived columns
h2['gap_size_in_adr'] = h2['gap_size_in_adr'].fillna(
    (h2['today_open'] - h2['prev_close']).abs() / h2['adr_10']
)
h2['close_loc'] = (h2['rth_close'] - h2['rth_low']) / (h2['rth_high'] - h2['rth_low'])
h2['day_range_adr'] = (h2['rth_high'] - h2['rth_low']) / h2['adr_10']
h2['drift_day'] = h2['close_vs_open_adr']

def gap_bucket(v):
    if v < 1: return '<1 ADR'
    if v < 2: return '1-2 ADR'
    if v < 3: return '2-3 ADR'
    return '>3 ADR'
h2['gap_bucket'] = h2['gap_size_in_adr'].apply(gap_bucket)

def spy_bucket(v):
    if pd.isna(v): return 'unknown'
    if v < -0.01: return 'stark_rot'
    if v < 0: return 'leicht_rot'
    if v < 0.01: return 'leicht_gruen'
    return 'stark_gruen'
h2['spy_bucket'] = h2['spy_return_day'].apply(spy_bucket)

def rvol_bucket(v):
    if pd.isna(v): return 'unknown'
    if v < 1: return '<1x'
    if v < 2: return '1-2x'
    if v < 3: return '2-3x'
    if v < 5: return '3-5x'
    return '>5x'
h2['rvol_bucket'] = h2['rvol_at_time_30min'].apply(rvol_bucket)

print(f"Half 2: {len(h2)} gapper days", file=sys.stderr)

# ── Output ───────────────────────────────────────────────────────────────────
out = []
def w(s=""): out.append(s)

w("=" * 90)
w("DURCHLAUF 6.0 — OOS-VALIDIERUNG CHEAT SHEET (METADATA-FINDINGS)")
w("=" * 90)
w(f"Datenbasis: Half 2 (2024-01-01 bis 2026-02-06), N={len(h2)}")
w(f"GapUp: {len(h2[h2['gap_direction']=='up'])}, GapDown: {len(h2[h2['gap_direction']=='down'])}")
w()

# ═══════════════════════════════════════════════════════════════════════════════
# F1: SPY-KONTEXT
# ═══════════════════════════════════════════════════════════════════════════════
w("=" * 90)
w("=== OOS-VALIDIERUNG: F1 SPY-KONTEXT ===")
w("=" * 90)

for gd, spy_cat, is_drift, is_cl, is_fill in [
    ('up', 'stark_rot', -0.630, 0.21, 0.37),
    ('up', 'stark_gruen', None, None, None),
    ('down', 'stark_gruen', +0.534, 0.79, None),
    ('down', 'stark_rot', None, None, None),
]:
    sub = h2[(h2['gap_direction'] == gd) & (h2['spy_bucket'] == spy_cat)]
    label = f"Gap{'Up' if gd=='up' else 'Down'} + SPY {spy_cat}"
    n = len(sub)
    if n < 5:
        w(f"\n{label}: N={n} (TOO FEW)")
        continue

    drift = sub['drift_day'].mean()
    drift_ci = bootstrap_ci(sub['drift_day'].dropna().values)
    cl = sub['close_loc'].mean()
    cl_ci = bootstrap_ci(sub['close_loc'].dropna().values)
    fill = sub['gap_filled'].astype(float).mean()
    fill_ci = bootstrap_ci(sub['gap_filled'].astype(float).values)

    w(f"\n{label}:")
    w(f"  N:     IS=???, OOS={n}")
    if is_drift is not None:
        w(f"  Drift: IS={is_drift:+.3f}, OOS={drift:+.3f} (CI: {drift_ci[0]:+.3f} to {drift_ci[1]:+.3f})")
        w(f"         Diff={drift-is_drift:+.3f}, Gleiche Richtung={same_direction(is_drift, drift)}")
    else:
        w(f"  Drift: OOS={drift:+.3f} (CI: {drift_ci[0]:+.3f} to {drift_ci[1]:+.3f})")
    if is_cl is not None:
        w(f"  CL:    IS={is_cl:.2f}, OOS={cl:.2f} (CI: {cl_ci[0]:.2f} to {cl_ci[1]:.2f})")
    else:
        w(f"  CL:    OOS={cl:.2f} (CI: {cl_ci[0]:.2f} to {cl_ci[1]:.2f})")
    if is_fill is not None:
        w(f"  Fill:  IS={is_fill*100:.0f}%, OOS={fill*100:.0f}% (CI: {fill_ci[0]*100:.0f}-{fill_ci[1]*100:.0f}%)")
    else:
        w(f"  Fill:  OOS={fill*100:.0f}% (CI: {fill_ci[0]*100:.0f}-{fill_ci[1]*100:.0f}%)")

# Overall SPY effect
for gd in ['up', 'down']:
    gsub = h2[h2['gap_direction'] == gd]
    stark_rot = gsub[gsub['spy_bucket'] == 'stark_rot']['drift_day']
    stark_gruen = gsub[gsub['spy_bucket'] == 'stark_gruen']['drift_day']
    if len(stark_rot) >= 20 and len(stark_gruen) >= 20:
        spread = stark_gruen.mean() - stark_rot.mean()
        w(f"\n  Gap{'Up' if gd=='up' else 'Down'} SPY-Spread (gruen-rot): {spread:+.3f}")

# F1 Status
gu_rot = h2[(h2['gap_direction']=='up') & (h2['spy_bucket']=='stark_rot')]
gd_gruen = h2[(h2['gap_direction']=='down') & (h2['spy_bucket']=='stark_gruen')]
f1_gu = gu_rot['drift_day'].mean() if len(gu_rot) >= 5 else np.nan
f1_gd = gd_gruen['drift_day'].mean() if len(gd_gruen) >= 5 else np.nan
f1_status = replication_status(-0.630, f1_gu) if not np.isnan(f1_gu) else "N/A"
f1_status2 = replication_status(+0.534, f1_gd) if not np.isnan(f1_gd) else "N/A"
w(f"\nBEWERTUNG F1: GapUp+SPY<-1% = {f1_status} | GapDown+SPY>+1% = {f1_status2}")

# ═══════════════════════════════════════════════════════════════════════════════
# F2: RVOL BESTIMMT TAG
# ═══════════════════════════════════════════════════════════════════════════════
w("\n" + "=" * 90)
w("=== OOS-VALIDIERUNG: F2 RVOL BESTIMMT TAG ===")
w("=" * 90)

for gd in ['up', 'down']:
    label = 'GapUp' if gd == 'up' else 'GapDown'
    gsub = h2[(h2['gap_direction'] == gd) & (h2['rvol_bucket'] != 'unknown')]

    w(f"\n--- {label} ---")
    for rb in ['<1x', '1-2x', '2-3x', '3-5x', '>5x']:
        sub = gsub[gsub['rvol_bucket'] == rb]
        if len(sub) < 10:
            w(f"  RVOL {rb}: N={len(sub)} (TOO FEW)")
            continue
        fill = sub['gap_filled'].astype(float).mean()
        fill_ci = bootstrap_ci(sub['gap_filled'].astype(float).values)
        drift = sub['drift_day'].mean()
        drift_ci = bootstrap_ci(sub['drift_day'].dropna().values)
        dr = sub['day_range_adr'].mean()
        w(f"  RVOL {rb}: N={len(sub)}, Fill={fill*100:.0f}% (CI: {fill_ci[0]*100:.0f}-{fill_ci[1]*100:.0f}%), "
          f"Drift={drift:+.3f} (CI: {drift_ci[0]:+.3f} to {drift_ci[1]:+.3f}), DayRange={dr:.2f}")

# IS comparison
w("\n  IS-Vergleich:")
w("  IS: RVOL>5x GapUp Fill=11%, GapDown Fill=9%")
w("  IS: RVOL<1x GapUp Fill=45%, GapDown Fill=36%")
gu_hi = h2[(h2['gap_direction']=='up') & (h2['rvol_bucket']=='>5x')]
gu_lo = h2[(h2['gap_direction']=='up') & (h2['rvol_bucket']=='<1x')]
gd_hi = h2[(h2['gap_direction']=='down') & (h2['rvol_bucket']=='>5x')]
gd_lo = h2[(h2['gap_direction']=='down') & (h2['rvol_bucket']=='<1x')]
for lbl, sub, is_val in [
    ('GapUp RVOL>5x Fill', gu_hi, 0.11), ('GapUp RVOL<1x Fill', gu_lo, 0.45),
    ('GapDown RVOL>5x Fill', gd_hi, 0.09), ('GapDown RVOL<1x Fill', gd_lo, 0.36)]:
    if len(sub) >= 5:
        oos = sub['gap_filled'].astype(float).mean()
        w(f"  OOS: {lbl} = {oos*100:.0f}% (IS={is_val*100:.0f}%) -> {replication_status(is_val, oos)}")

w(f"\nBEWERTUNG F2: Siehe oben")

# ═══════════════════════════════════════════════════════════════════════════════
# F3: OPENING DRIVE SETZT RICHTUNG (from metadata)
# ═══════════════════════════════════════════════════════════════════════════════
w("\n" + "=" * 90)
w("=== OOS-VALIDIERUNG: F3 OPENING DRIVE SETZT RICHTUNG ===")
w("=" * 90)

for gd in ['up', 'down']:
    label = 'GapUp' if gd == 'up' else 'GapDown'
    gsub = h2[(h2['gap_direction'] == gd) & (h2['opening_drive_direction'].notna())]

    w(f"\n--- {label} (N={len(gsub)}) ---")

    # OD values are 'with_gap' and 'against_gap'
    od_with = gsub[gsub['opening_drive_direction'] == 'with_gap']
    od_against = gsub[gsub['opening_drive_direction'] == 'against_gap']

    # "Tagesrichtung mit Gap" = close in gap direction
    if gd == 'up':
        od_with_correct = (od_with['close_vs_open_adr'] > 0).mean() if len(od_with) > 0 else np.nan
        od_against_correct = (od_against['close_vs_open_adr'] > 0).mean() if len(od_against) > 0 else np.nan
    else:
        od_with_correct = (od_with['close_vs_open_adr'] < 0).mean() if len(od_with) > 0 else np.nan
        od_against_correct = (od_against['close_vs_open_adr'] < 0).mean() if len(od_against) > 0 else np.nan

    if len(od_with) >= 10:
        ci = bootstrap_ci((od_with['close_vs_open_adr'] > 0 if gd=='up' else od_with['close_vs_open_adr'] < 0).astype(float).values)
        w(f"  OD mit Gap: {od_with_correct*100:.1f}% Tagesrichtung mit Gap (N={len(od_with)}) (CI: {ci[0]*100:.0f}-{ci[1]*100:.0f}%)")
    if len(od_against) >= 10:
        ci = bootstrap_ci((od_against['close_vs_open_adr'] > 0 if gd=='up' else od_against['close_vs_open_adr'] < 0).astype(float).values)
        w(f"  OD gegen Gap: {od_against_correct*100:.1f}% Tagesrichtung mit Gap (N={len(od_against)}) (CI: {ci[0]*100:.0f}-{ci[1]*100:.0f}%)")

    # Drift by OD direction
    if len(od_with) >= 10:
        drift_with = od_with['drift_day'].mean()
        drift_with_ci = bootstrap_ci(od_with['drift_day'].dropna().values)
        w(f"  OD mit Gap Drift: {drift_with:+.3f} (CI: {drift_with_ci[0]:+.3f} to {drift_with_ci[1]:+.3f})")
    if len(od_against) >= 10:
        drift_against = od_against['drift_day'].mean()
        drift_against_ci = bootstrap_ci(od_against['drift_day'].dropna().values)
        w(f"  OD gegen Gap Drift: {drift_against:+.3f} (CI: {drift_against_ci[0]:+.3f} to {drift_against_ci[1]:+.3f})")

w("\n  IS-Vergleich: OD mit Gap = 67-68%, OD gegen Gap = 28-30%")
w(f"\nBEWERTUNG F3: Berechnung oben")

# ═══════════════════════════════════════════════════════════════════════════════
# F4: VORHERIGE PERFORMANCE = FADE-SIGNAL
# ═══════════════════════════════════════════════════════════════════════════════
w("\n" + "=" * 90)
w("=== OOS-VALIDIERUNG: F4 VORHERIGE PERFORMANCE ===")
w("=" * 90)

# GapUp nach >20% 10d-Run
gu_hot = h2[(h2['gap_direction'] == 'up') & (h2['prior_return_10d'] > 20)]
if len(gu_hot) >= 10:
    fill = gu_hot['gap_filled'].astype(float).mean()
    fill_ci = bootstrap_ci(gu_hot['gap_filled'].astype(float).values)
    drift = gu_hot['drift_day'].mean()
    drift_ci = bootstrap_ci(gu_hot['drift_day'].dropna().values)
    w(f"GapUp + Prior10d>+20%: N={len(gu_hot)}")
    w(f"  Fill:  IS=43%, OOS={fill*100:.0f}% (CI: {fill_ci[0]*100:.0f}-{fill_ci[1]*100:.0f}%)")
    w(f"  Drift: IS=-0.207, OOS={drift:+.3f} (CI: {drift_ci[0]:+.3f} to {drift_ci[1]:+.3f})")
    w(f"  -> {replication_status(-0.207, drift)}")
else:
    w(f"GapUp + Prior10d>+20%: N={len(gu_hot)} (TOO FEW)")

# GapDown + RSI<30
gd_os = h2[(h2['gap_direction'] == 'down') & (h2['rsi_14_prev'] < 30)]
if len(gd_os) >= 10:
    drift = gd_os['drift_day'].mean()
    drift_ci = bootstrap_ci(gd_os['drift_day'].dropna().values)
    w(f"\nGapDown + RSI<30: N={len(gd_os)}")
    w(f"  Drift: IS=+0.415, OOS={drift:+.3f} (CI: {drift_ci[0]:+.3f} to {drift_ci[1]:+.3f})")
    w(f"  -> {replication_status(+0.415, drift)}")
else:
    w(f"\nGapDown + RSI<30: N={len(gd_os)} (TOO FEW)")

# RSI>70 + GapUp
gu_ob = h2[(h2['gap_direction'] == 'up') & (h2['rsi_14_prev'] > 70)]
if len(gu_ob) >= 10:
    drift = gu_ob['drift_day'].mean()
    drift_ci = bootstrap_ci(gu_ob['drift_day'].dropna().values)
    w(f"\nGapUp + RSI>70: N={len(gu_ob)}")
    w(f"  Drift: IS=-0.201, OOS={drift:+.3f} (CI: {drift_ci[0]:+.3f} to {drift_ci[1]:+.3f})")
    w(f"  -> {replication_status(-0.201, drift)}")
else:
    w(f"\nGapUp + RSI>70: N={len(gu_ob)} (TOO FEW)")

w(f"\nBEWERTUNG F4: Siehe oben")

# ═══════════════════════════════════════════════════════════════════════════════
# F5: GAP-FILL RATEN
# ═══════════════════════════════════════════════════════════════════════════════
w("\n" + "=" * 90)
w("=== OOS-VALIDIERUNG: F5 GAP-FILL RATEN ===")
w("=" * 90)

w("\n                    IS             OOS")
for gd in ['up', 'down']:
    label = 'GapUp' if gd == 'up' else 'GapDown'
    gsub = h2[h2['gap_direction'] == gd]
    w(f"\n  --- {label} ---")
    is_vals = {'<1 ADR': 0.40, '1-2 ADR': 0.17, '2-3 ADR': None, '>3 ADR': 0.05}
    for gb in ['<1 ADR', '1-2 ADR', '2-3 ADR', '>3 ADR']:
        sub = gsub[gsub['gap_bucket'] == gb]
        if len(sub) < 10:
            w(f"  {gb}: N={len(sub)} (TOO FEW)")
            continue
        fill = sub['gap_filled'].astype(float).mean()
        fill_ci = bootstrap_ci(sub['gap_filled'].astype(float).values)
        is_v = is_vals.get(gb)
        is_str = f"IS={is_v*100:.0f}%" if is_v else "IS=???"
        status = replication_status(is_v, fill) if is_v else "---"
        w(f"  {gb}: {is_str}, OOS={fill*100:.0f}% (CI: {fill_ci[0]*100:.0f}-{fill_ci[1]*100:.0f}%), N={len(sub)} -> {status}")

w(f"\nBEWERTUNG F5: Siehe oben")

# ═══════════════════════════════════════════════════════════════════════════════
# F11: KAPITULATION BOUNCED NICHT
# ═══════════════════════════════════════════════════════════════════════════════
w("\n" + "=" * 90)
w("=== OOS-VALIDIERUNG: F11 KAPITULATION ===")
w("=" * 90)

kap = h2[
    (h2['gap_direction'] == 'down') &
    (h2['prior_return_10d'] < -10) &
    (h2['rvol_at_time_30min'].between(2, 5))
]
if len(kap) >= 10:
    drift = kap['drift_day'].mean()
    drift_ci = bootstrap_ci(kap['drift_day'].dropna().values)
    fill = kap['gap_filled'].astype(float).mean()
    w(f"GapDown + Prior<-10% + RVOL 2-5x: N={len(kap)}")
    w(f"  Drift: IS=-0.264, OOS={drift:+.3f} (CI: {drift_ci[0]:+.3f} to {drift_ci[1]:+.3f})")
    w(f"  Fill:  OOS={fill*100:.0f}%")
    w(f"  -> {replication_status(-0.264, drift)}")
else:
    w(f"GapDown + Prior<-10% + RVOL 2-5x: N={len(kap)} (TOO FEW)")

# Also check broader kapitulation (any high RVOL)
kap2 = h2[
    (h2['gap_direction'] == 'down') &
    (h2['prior_return_10d'] < -10) &
    (h2['rvol_at_time_30min'] >= 2)
]
if len(kap2) >= 10:
    drift2 = kap2['drift_day'].mean()
    drift2_ci = bootstrap_ci(kap2['drift_day'].dropna().values)
    w(f"\nBroader: GapDown + Prior<-10% + RVOL>=2x: N={len(kap2)}")
    w(f"  Drift: OOS={drift2:+.3f} (CI: {drift2_ci[0]:+.3f} to {drift2_ci[1]:+.3f})")

w(f"\nBEWERTUNG F11: Siehe oben")

# ═══════════════════════════════════════════════════════════════════════════════
# F15: HOD/LOD TIMING
# ═══════════════════════════════════════════════════════════════════════════════
w("\n" + "=" * 90)
w("=== OOS-VALIDIERUNG: F15 HOD/LOD TIMING ===")
w("=" * 90)

# hod_time and lod_time are in metadata as integers (minutes from open)
h2['hod_min'] = pd.to_numeric(h2['hod_time'], errors='coerce')
h2['lod_min'] = pd.to_numeric(h2['lod_time'], errors='coerce')

hod_valid = h2[h2['hod_min'].notna()]
lod_valid = h2[h2['lod_min'].notna()]

hod_first30 = (hod_valid['hod_min'] <= 30).mean() if len(hod_valid) > 0 else np.nan
lod_first30 = (lod_valid['lod_min'] <= 30).mean() if len(lod_valid) > 0 else np.nan

w(f"HOD in ersten 30min: IS=45-48%, OOS={hod_first30*100:.1f}% (N={len(hod_valid)})")
w(f"LOD in ersten 30min: IS=45-48%, OOS={lod_first30*100:.1f}% (N={len(lod_valid)})")

# By gap direction
for gd in ['up', 'down']:
    label = 'GapUp' if gd == 'up' else 'GapDown'
    gsub_hod = hod_valid[hod_valid['gap_direction'] == gd]
    gsub_lod = lod_valid[lod_valid['gap_direction'] == gd]
    hod_30 = (gsub_hod['hod_min'] <= 30).mean() if len(gsub_hod) > 0 else np.nan
    lod_30 = (gsub_lod['lod_min'] <= 30).mean() if len(gsub_lod) > 0 else np.nan
    w(f"  {label}: HOD_30m={hod_30*100:.1f}%, LOD_30m={lod_30*100:.1f}%")

w(f"\nBEWERTUNG F15: {replication_status(0.46, (hod_first30+lod_first30)/2 if not np.isnan(hod_first30) and not np.isnan(lod_first30) else np.nan)}")

# ═══════════════════════════════════════════════════════════════════════════════
# F16: FOLLOW-THROUGH TAG 2
# ═══════════════════════════════════════════════════════════════════════════════
w("\n" + "=" * 90)
w("=== OOS-VALIDIERUNG: F16 FOLLOW-THROUGH TAG 2 ===")
w("=" * 90)

# We need next-day return. Sort by ticker+date and look at next day.
h2_sorted = h2.sort_values(['ticker', 'date']).copy()
h2_sorted['next_day_return'] = np.nan

# For each ticker, get next day return
for ticker in h2_sorted['ticker'].unique():
    mask = h2_sorted['ticker'] == ticker
    idx = h2_sorted[mask].index
    if len(idx) < 2:
        continue
    # Next day close vs this day close (using rth_close)
    closes = h2_sorted.loc[idx, 'rth_close'].values
    dates = h2_sorted.loc[idx, 'date'].values
    for i in range(len(idx) - 1):
        # Only if consecutive trading days (within 5 calendar days)
        delta = (dates[i+1] - dates[i]) / np.timedelta64(1, 'D')
        if delta <= 5 and closes[i] > 0:
            h2_sorted.loc[idx[i], 'next_day_return'] = (closes[i+1] - closes[i]) / closes[i]

# Actually, a simpler approach: check if the NEXT day return correlates with gap-day close
# For "no follow-through", we expect ~50/50
for gd in ['up', 'down']:
    label = 'GapUp' if gd == 'up' else 'GapDown'
    gsub = h2_sorted[(h2_sorted['gap_direction'] == gd) & (h2_sorted['next_day_return'].notna())]
    if len(gsub) < 20:
        w(f"  {label}: N={len(gsub)} (TOO FEW for follow-through)")
        continue

    # Split by gap-day close
    bullish_day = gsub[gsub['close_vs_open_adr'] > 0]
    bearish_day = gsub[gsub['close_vs_open_adr'] < 0]

    if len(bullish_day) >= 10:
        next_pos = (bullish_day['next_day_return'] > 0).mean()
        w(f"  {label} bullish close -> next day positive: {next_pos*100:.1f}% (N={len(bullish_day)})")
    if len(bearish_day) >= 10:
        next_pos = (bearish_day['next_day_return'] > 0).mean()
        w(f"  {label} bearish close -> next day positive: {next_pos*100:.1f}% (N={len(bearish_day)})")

w("\n  IS: Kein Edge (~50/50)")
w(f"\nBEWERTUNG F16: Bestaetigung erwartet (kein Signal)")

# ═══════════════════════════════════════════════════════════════════════════════
# F18: SPY x GAP-GROESSE INTERAKTION
# ═══════════════════════════════════════════════════════════════════════════════
w("\n" + "=" * 90)
w("=== OOS-VALIDIERUNG: F18 SPY x GAP-GROESSE ===")
w("=" * 90)

w("  IS: SPY-Spread waechst mit Gap-Groesse bei GapDown")
w("  IS: <1 ADR Spread=+0.088, 1-2 ADR Spread=+0.290, 2-3 ADR Spread=+0.337\n")

for gd in ['up', 'down']:
    label = 'GapUp' if gd == 'up' else 'GapDown'
    gsub = h2[h2['gap_direction'] == gd]
    w(f"  {label}:")
    for gb in ['<1 ADR', '1-2 ADR', '2-3 ADR', '>3 ADR']:
        gruen = gsub[(gsub['spy_bucket'] == 'stark_gruen') & (gsub['gap_bucket'] == gb)]
        rot = gsub[(gsub['spy_bucket'] == 'stark_rot') & (gsub['gap_bucket'] == gb)]
        if len(gruen) >= 10 and len(rot) >= 10:
            spread = gruen['drift_day'].mean() - rot['drift_day'].mean()
            w(f"    {gb}: Spread(gruen-rot)={spread:+.3f} (N_gruen={len(gruen)}, N_rot={len(rot)})")
        else:
            w(f"    {gb}: N too small (gruen={len(gruen)}, rot={len(rot)})")

w(f"\nBEWERTUNG F18: Siehe oben")

# ═══════════════════════════════════════════════════════════════════════════════
# F19: SEKTOR-UNTERSCHIEDE
# ═══════════════════════════════════════════════════════════════════════════════
w("\n" + "=" * 90)
w("=== OOS-VALIDIERUNG: F19 SEKTOR ===")
w("=" * 90)

w("  IS: Retail GapUp Drift=+0.159 (staerkstes Momentum)")
w("  IS: Finance GapDown Drift=+0.087 (bounced am meisten)\n")

for gd in ['up', 'down']:
    label = 'GapUp' if gd == 'up' else 'GapDown'
    gsub = h2[h2['gap_direction'] == gd]
    w(f"\n  {label}:")
    sector_counts = gsub['sector'].value_counts()
    for sec in sector_counts.index[:10]:
        sub = gsub[gsub['sector'] == sec]
        if len(sub) < 20:
            continue
        drift = sub['drift_day'].mean()
        drift_ci = bootstrap_ci(sub['drift_day'].dropna().values)
        fill = sub['gap_filled'].astype(float).mean()
        w(f"    {sec}: N={len(sub)}, Drift={drift:+.3f} (CI: {drift_ci[0]:+.3f} to {drift_ci[1]:+.3f}), Fill={fill*100:.0f}%")

w(f"\nBEWERTUNG F19: Siehe oben")

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════════
w("\n\n" + "=" * 90)
w("ZUSAMMENFASSUNG METADATA-FINDINGS")
w("=" * 90)
w(f"{'#':<5} {'Finding':<30} {'IS-Effekt':<20} {'OOS-Effekt':<20} {'Status':<20}")
w("-" * 90)

# Collect summary
summary = []

# F1
f1a = gu_rot['drift_day'].mean() if len(gu_rot) >= 5 else np.nan
f1b = gd_gruen['drift_day'].mean() if len(gd_gruen) >= 5 else np.nan
summary.append(('F1', 'SPY-Kontext', f'-0.63/+0.53', f'{f1a:+.3f}/{f1b:+.3f}' if not (np.isnan(f1a) or np.isnan(f1b)) else 'N/A',
                replication_status(-0.630, f1a) if not np.isnan(f1a) else 'N/A'))

# F2
gu_hi_fill = gu_hi['gap_filled'].astype(float).mean() if len(gu_hi) >= 5 else np.nan
gu_lo_fill = gu_lo['gap_filled'].astype(float).mean() if len(gu_lo) >= 5 else np.nan
summary.append(('F2', 'RVOL', f'Fill 11%/45%', f'Fill {gu_hi_fill*100:.0f}%/{gu_lo_fill*100:.0f}%' if not (np.isnan(gu_hi_fill) or np.isnan(gu_lo_fill)) else 'N/A',
                'TBD'))

# F3 - will be updated by 1min script
summary.append(('F3', 'Opening Drive', '67-68%', 'see above', 'TBD'))

# F4
gu_hot_drift = gu_hot['drift_day'].mean() if len(gu_hot) >= 10 else np.nan
gd_os_drift = gd_os['drift_day'].mean() if len(gd_os) >= 10 else np.nan
gu_ob_drift = gu_ob['drift_day'].mean() if len(gu_ob) >= 10 else np.nan
summary.append(('F4', 'Prior Performance', f'-0.21/+0.42/-0.20',
                f'{gu_hot_drift:+.3f}/{gd_os_drift:+.3f}/{gu_ob_drift:+.3f}' if not any(np.isnan([gu_hot_drift, gd_os_drift, gu_ob_drift])) else 'PARTIAL',
                'TBD'))

# F5
gu_small = h2[(h2['gap_direction']=='up') & (h2['gap_bucket']=='<1 ADR')]
f5_oos = gu_small['gap_filled'].astype(float).mean() if len(gu_small) >= 10 else np.nan
summary.append(('F5', 'Gap-Fill Raten', '40%/17%/<5%', 'see above', 'TBD'))

# F11
kap_drift = kap['drift_day'].mean() if len(kap) >= 10 else np.nan
summary.append(('F11', 'Kapitulation', '-0.264', f'{kap_drift:+.3f}' if not np.isnan(kap_drift) else 'N/A',
                replication_status(-0.264, kap_drift) if not np.isnan(kap_drift) else 'N/A'))

# F15
avg_30 = (hod_first30 + lod_first30) / 2 if not (np.isnan(hod_first30) or np.isnan(lod_first30)) else np.nan
summary.append(('F15', 'HOD/LOD Timing', '45-48%', f'{avg_30*100:.0f}%' if not np.isnan(avg_30) else 'N/A', 'TBD'))

# F16
summary.append(('F16', 'Follow-Through', '~50/50', 'see above', 'TBD'))

# F18
summary.append(('F18', 'SPY x Gap-Groesse', 'Spread waechst', 'see above', 'TBD'))

# F19
summary.append(('F19', 'Sektor', 'Retail+0.16', 'see above', 'TBD'))

for num, finding, is_eff, oos_eff, status in summary:
    w(f"{num:<5} {finding:<30} {is_eff:<20} {oos_eff:<20} {status:<20}")

# Write output
report = "\n".join(out)
with open('results/oos_cheat_validation_meta.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print(report)
print(f"\nSaved to results/oos_cheat_validation_meta.txt", file=sys.stderr)
