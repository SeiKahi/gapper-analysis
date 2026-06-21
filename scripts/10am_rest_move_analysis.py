"""
10:00 AM REST-MOVE ANALYSIS
============================
Explorative Analyse: Gibt es einen Edge für Trades NACH dem Opening Drive?
Methodik: IS zuerst (vor 2024), dann OOS-Validation, Shrinkage-Check.
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG ──────────────────────────────────────────────────────────
DATA_PATH = 'data/metadata/metadata_v8_5.parquet'
OUT_DIR = 'results/10am_analysis'
IS_CUTOFF = '2024-01-01'
DPI = 300

STYLE_BG = '#1a1a2e'
STYLE_FG = '#e0e0e0'
STYLE_GRID = '#333355'
STYLE_GREEN = '#00e676'
STYLE_RED = '#ff5252'
STYLE_BLUE = '#448aff'
STYLE_CYAN = '#00e5ff'
STYLE_GREY = '#666688'
STYLE_YELLOW = '#ffd740'

# ── LOAD AND PREPARE ───────────────────────────────────────────────
print("=" * 70)
print("LOADING DATA")
print("=" * 70)

df = pd.read_parquet(DATA_PATH)
print(f"Total rows: {len(df)}")

# Filter: od_strength > 0.5 AND od_direction == 'with_gap'
df = df[(df['od_strength'] > 0.5) & (df['od_direction'] == 'with_gap')].copy()
print(f"After OD filter (od>0.5, with_gap): {len(df)}")

# Drop rows with essential NaNs
essential = ['close_935', 'close_1000', 'rth_close', 'adr_10',
             'rth_high', 'rth_low', 'rest_drift_1000', 'rest_drift']
df = df.dropna(subset=essential)
print(f"After dropping NaN in essential cols: {len(df)}")

# ── COMPUTED COLUMNS ───────────────────────────────────────────────
# rest_drift_1000 = drift from 9:35 to 10:00 (gap-adjusted) = "drift_30"
# rest_drift = drift from 9:35 to close (gap-adjusted) = "total_move"

df['drift_30'] = df['rest_drift_1000']
df['total_move'] = df['rest_drift']
df['move_after_1000'] = df['total_move'] - df['drift_30']  # close-to-1000, gap-adjusted

# move_before_1000 = drift_30 (same thing, different name for clarity)
df['move_before_1000'] = df['drift_30']

# peak_after_1000: approximation using day high/low
# LIMITATION: high/low may have been reached before 10:00
df['peak_after_1000'] = np.where(
    df['gap_direction'] == 'up',
    (df['rth_high'] - df['close_1000']) / df['adr_10'],
    (df['close_1000'] - df['rth_low']) / df['adr_10']
)

# Max pullback after 1000 (approximation)
# For GapUp: how far below close_1000 did it go?
# For GapDn: how far above close_1000 did it go?
df['max_pullback_after_1000'] = np.where(
    df['gap_direction'] == 'up',
    (df['close_1000'] - df['rth_low']) / df['adr_10'],
    (df['rth_high'] - df['close_1000']) / df['adr_10']
)

# Distance: close_1000 from close_935
df['dist_1000_from_935'] = np.where(
    df['gap_direction'] == 'up',
    (df['close_1000'] - df['close_935']) / df['adr_10'],
    (df['close_935'] - df['close_1000']) / df['adr_10']
)

# RVOL_30
df['RVOL_30'] = df['rvol_open_30min']

# IS / OOS split
df['date_dt'] = pd.to_datetime(df['date'])
df_is = df[df['date_dt'] < IS_CUTOFF].copy()
df_oos = df[df['date_dt'] >= IS_CUTOFF].copy()

print(f"\nIS  (before {IS_CUTOFF}): {len(df_is)}")
print(f"OOS (from   {IS_CUTOFF}): {len(df_oos)}")
print(f"GapUp IS: {(df_is['gap_direction']=='up').sum()}, GapDn IS: {(df_is['gap_direction']=='down').sum()}")
print(f"GapUp OOS: {(df_oos['gap_direction']=='up').sum()}, GapDn OOS: {(df_oos['gap_direction']=='down').sum()}")


# ── SEGMENT DEFINITIONS ────────────────────────────────────────────
def get_segments(data):
    """Return dict of segment_name -> filtered dataframe."""
    segs = {}
    segs['A: ALL'] = data
    segs['B: drift>0.25 + RVOL>5'] = data[(data['drift_30'] > 0.25) & (data['RVOL_30'] > 5)]
    segs['C: drift>0.50 + RVOL>5'] = data[(data['drift_30'] > 0.50) & (data['RVOL_30'] > 5)]
    segs['D: drift>0.50'] = data[data['drift_30'] > 0.50]
    segs['E: RVOL>5'] = data[data['RVOL_30'] > 5]
    segs['F: drift<0 (fader)'] = data[data['drift_30'] < 0]
    segs['G: -0.25<drift<0'] = data[(data['drift_30'] < 0) & (data['drift_30'] > -0.25)]
    return segs


# ── ANALYSIS FUNCTIONS ─────────────────────────────────────────────
def compute_segment_stats(segs, label=""):
    """Compute stats for each segment. Returns DataFrame."""
    rows = []
    for name, seg in segs.items():
        n = len(seg)
        if n == 0:
            rows.append({'Segment': name, 'N': 0})
            continue
        ma = seg['move_after_1000']
        pa = seg['peak_after_1000']
        row = {
            'Segment': name,
            'N': n,
            'Median_MoveAfter': ma.median(),
            'Mean_MoveAfter': ma.mean(),
            'Std_MoveAfter': ma.std(),
            'Pct_Positive': (ma > 0).mean() * 100,
            'P25_MoveAfter': ma.quantile(0.25),
            'P75_MoveAfter': ma.quantile(0.75),
            'Sharpe': ma.mean() / ma.std() if ma.std() > 0 else 0,
            'Median_PeakAfter': pa.median(),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def compute_timing_stats(data, label=""):
    """Compute timing: pct of total move before/after 10:00."""
    # Only where total_move > 0 (in gap direction)
    sub = data[data['total_move'] > 0.01].copy()  # avoid division by ~0
    if len(sub) == 0:
        return None
    sub['pct_before'] = sub['move_before_1000'] / sub['total_move'] * 100
    sub['pct_after'] = sub['move_after_1000'] / sub['total_move'] * 100
    # Clip extreme values for stable stats
    sub['pct_before_c'] = sub['pct_before'].clip(-500, 500)
    sub['pct_after_c'] = sub['pct_after'].clip(-500, 500)
    return {
        'N': len(sub),
        'Median_Pct_Before': sub['pct_before_c'].median(),
        'Mean_Pct_Before': sub['pct_before_c'].mean(),
        'Median_Pct_After': sub['pct_after_c'].median(),
        'Mean_Pct_After': sub['pct_after_c'].mean(),
    }


def compute_fading_stats(data):
    """Analyse 3: fading stocks segmented by drift_30 magnitude."""
    bins = [
        ('drift < -0.25 (strong fade)', data[data['drift_30'] < -0.25]),
        ('-0.25 < drift < 0 (light fade)', data[(data['drift_30'] >= -0.25) & (data['drift_30'] < 0)]),
        ('0 < drift < 0.25 (weak drift)', data[(data['drift_30'] >= 0) & (data['drift_30'] < 0.25)]),
        ('drift > 0.25 (strong drift)', data[data['drift_30'] >= 0.25]),
    ]
    rows = []
    for name, seg in bins:
        n = len(seg)
        if n == 0:
            rows.append({'Segment': name, 'N': 0})
            continue
        ma = seg['move_after_1000']
        rows.append({
            'Segment': name,
            'N': n,
            'Median_MoveAfter': ma.median(),
            'Mean_MoveAfter': ma.mean(),
            'Pct_Positive': (ma > 0).mean() * 100,
            'Pct_Wende': (ma > 0).mean() * 100 if name.startswith('-') or name.startswith('drift <') else None,
            'Median_PeakAfter': seg['peak_after_1000'].median(),
        })
    return pd.DataFrame(rows)


def compute_entry_level_stats(data, gap_dir=None):
    """Analyse 4: Entry level problem for segment C."""
    seg = data[(data['drift_30'] > 0.50) & (data['RVOL_30'] > 5)]
    if gap_dir:
        seg = seg[seg['gap_direction'] == gap_dir]
    if len(seg) == 0:
        return None
    stats_dict = {
        'N': len(seg),
        'Median_Dist_1000_from_935_ADR': seg['dist_1000_from_935'].median(),
        'Mean_Dist_1000_from_935_ADR': seg['dist_1000_from_935'].mean(),
        'Median_PeakAfter_ADR': seg['peak_after_1000'].median(),
        'Median_MaxPullback_ADR': seg['max_pullback_after_1000'].median(),
        'Mean_MaxPullback_ADR': seg['max_pullback_after_1000'].mean(),
        'P75_MaxPullback_ADR': seg['max_pullback_after_1000'].quantile(0.75),
        'P90_MaxPullback_ADR': seg['max_pullback_after_1000'].quantile(0.90),
        'SL025_HitRate': (seg['max_pullback_after_1000'] > 0.25).mean() * 100,
        'SL050_HitRate': (seg['max_pullback_after_1000'] > 0.50).mean() * 100,
        'Median_MoveAfter_ADR': seg['move_after_1000'].median(),
        'Mean_MoveAfter_ADR': seg['move_after_1000'].mean(),
    }
    return stats_dict


def compute_correlations(data, label=""):
    """Analyse 5: correlations and regression."""
    d = data.dropna(subset=['drift_30', 'RVOL_30', 'move_after_1000'])
    if len(d) < 10:
        return None
    r_drift, p_drift = stats.pearsonr(d['drift_30'], d['move_after_1000'])
    r_rvol, p_rvol = stats.pearsonr(d['RVOL_30'], d['move_after_1000'])
    slope, intercept, r_value, p_value, std_err = stats.linregress(d['drift_30'], d['move_after_1000'])
    return {
        'N': len(d),
        'Pearson_drift_vs_moveAfter': r_drift,
        'p_value_drift': p_drift,
        'Pearson_RVOL_vs_moveAfter': r_rvol,
        'p_value_RVOL': p_rvol,
        'Regression_slope': slope,
        'Regression_intercept': intercept,
        'R_squared': r_value ** 2,
    }


def compute_shrinkage(is_df, oos_df):
    """Compute shrinkage between IS and OOS stats."""
    metrics = ['Median_MoveAfter', 'Mean_MoveAfter', 'Pct_Positive', 'Sharpe', 'Median_PeakAfter']
    rows = []
    for seg_name in is_df['Segment']:
        is_row = is_df[is_df['Segment'] == seg_name].iloc[0] if seg_name in is_df['Segment'].values else None
        oos_row = oos_df[oos_df['Segment'] == seg_name].iloc[0] if seg_name in oos_df['Segment'].values else None
        if is_row is None or oos_row is None:
            continue
        row = {'Segment': seg_name, 'N_IS': is_row['N'], 'N_OOS': oos_row['N']}
        for m in metrics:
            is_val = is_row.get(m, np.nan)
            oos_val = oos_row.get(m, np.nan)
            if pd.notna(is_val) and abs(is_val) > 1e-6:
                row[f'{m}_IS'] = is_val
                row[f'{m}_OOS'] = oos_val
                row[f'{m}_Shrinkage%'] = (oos_val - is_val) / abs(is_val) * 100
            else:
                row[f'{m}_IS'] = is_val
                row[f'{m}_OOS'] = oos_val
                row[f'{m}_Shrinkage%'] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════
# MAIN: RUN ALL ANALYSES — IS FIRST, THEN OOS
# ══════════════════════════════════════════════════════════════════════

output_lines = []

def out(text=""):
    print(text)
    output_lines.append(text)


def out_df(df_out, float_fmt='.4f'):
    text = df_out.to_string(index=False, float_format=lambda x: f'{x:{float_fmt}}')
    for line in text.split('\n'):
        out(line)


def section(title):
    out("\n" + "=" * 70)
    out(title)
    out("=" * 70)


# ──────────────────────────────────────────────────────────────────────
# PART 1: IS RESULTS
# ──────────────────────────────────────────────────────────────────────

section("=== IS RESULTS (before 2024-01-01) ===")

# --- ANALYSE 1: Segment Stats (ALL gap directions combined) ---
section("ANALYSE 1: REST-MOVE NACH 10:00 — IS (ALL)")
is_segs = get_segments(df_is)
is_stats_all = compute_segment_stats(is_segs, "IS_ALL")
out_df(is_stats_all)

# By GapUp
section("ANALYSE 1: REST-MOVE NACH 10:00 — IS (GapUp only)")
is_segs_up = get_segments(df_is[df_is['gap_direction'] == 'up'])
is_stats_up = compute_segment_stats(is_segs_up, "IS_UP")
out_df(is_stats_up)

# By GapDn
section("ANALYSE 1: REST-MOVE NACH 10:00 — IS (GapDn only)")
is_segs_dn = get_segments(df_is[df_is['gap_direction'] == 'down'])
is_stats_dn = compute_segment_stats(is_segs_dn, "IS_DN")
out_df(is_stats_dn)

# --- ANALYSE 2: Timing ---
section("ANALYSE 2: TIMING DES MOVES — IS")
for seg_name in ['A: ALL', 'C: drift>0.50 + RVOL>5', 'F: drift<0 (fader)']:
    timing = compute_timing_stats(is_segs[seg_name], f"IS_{seg_name}")
    if timing:
        out(f"\n  {seg_name}:")
        for k, v in timing.items():
            out(f"    {k}: {v:.2f}" if isinstance(v, float) else f"    {k}: {v}")
    else:
        out(f"\n  {seg_name}: N=0 or insufficient data")

# --- ANALYSE 3: Fading ---
section("ANALYSE 3: FADING-AKTIEN — IS (ALL)")
is_fading_all = compute_fading_stats(df_is)
out_df(is_fading_all)

section("ANALYSE 3: FADING-AKTIEN — IS (GapUp)")
is_fading_up = compute_fading_stats(df_is[df_is['gap_direction'] == 'up'])
out_df(is_fading_up)

section("ANALYSE 3: FADING-AKTIEN — IS (GapDn)")
is_fading_dn = compute_fading_stats(df_is[df_is['gap_direction'] == 'down'])
out_df(is_fading_dn)

# --- Fading detail: how many faders reverse? ---
section("ANALYSE 3 DETAIL: FADER-WENDE — IS")
faders_is = df_is[df_is['drift_30'] < 0]
n_faders = len(faders_is)
n_wende = (faders_is['move_after_1000'] > 0).sum()
out(f"  Total faders (drift<0): {n_faders}")
out(f"  Davon wenden (move_after>0): {n_wende} ({n_wende/n_faders*100:.1f}%)" if n_faders > 0 else "  N/A")
if n_wende > 0:
    out(f"  Median move_after bei Wendern: {faders_is[faders_is['move_after_1000']>0]['move_after_1000'].median():.4f} ADR")
    out(f"  Median move_after bei Nicht-Wendern: {faders_is[faders_is['move_after_1000']<=0]['move_after_1000'].median():.4f} ADR")

# --- ANALYSE 4: Entry Level ---
section("ANALYSE 4: ENTRY-LEVEL PROBLEM — IS")
for gd_label, gd_val in [("ALL", None), ("GapUp", "up"), ("GapDn", "down")]:
    entry = compute_entry_level_stats(df_is, gd_val)
    if entry:
        out(f"\n  Segment C — {gd_label}:")
        for k, v in entry.items():
            out(f"    {k}: {v:.2f}%" if 'Rate' in k else (f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}"))
    else:
        out(f"\n  Segment C — {gd_label}: No data")

# --- ANALYSE 5: Correlations ---
section("ANALYSE 5: KORRELATIONEN — IS")
for gd_label, gd_filter in [("ALL", df_is), ("GapUp", df_is[df_is['gap_direction'] == 'up'])]:
    corr = compute_correlations(gd_filter, f"IS_{gd_label}")
    if corr:
        out(f"\n  {gd_label}:")
        for k, v in corr.items():
            out(f"    {k}: {v:.6f}" if isinstance(v, float) else f"    {k}: {v}")


# ──────────────────────────────────────────────────────────────────────
# IS FAZIT
# ──────────────────────────────────────────────────────────────────────

section("=== IS FAZIT ===")

# Segment C IS stats
seg_c_is = is_stats_all[is_stats_all['Segment'] == 'C: drift>0.50 + RVOL>5']
if len(seg_c_is) > 0:
    c = seg_c_is.iloc[0]
    out(f"  Segment C (drift>0.50 + RVOL>5):")
    out(f"    N = {int(c['N'])}")
    out(f"    Median move_after_1000 = {c['Median_MoveAfter']:.4f} ADR")
    out(f"    Mean move_after_1000   = {c['Mean_MoveAfter']:.4f} ADR")
    out(f"    % positiv              = {c['Pct_Positive']:.1f}%")
    out(f"    Sharpe-like            = {c['Sharpe']:.4f}")
    if c['Median_MoveAfter'] > 0 and c['Pct_Positive'] > 50:
        out(f"  -> IS: POSITIVER EDGE VORHANDEN")
    elif c['Median_MoveAfter'] > 0:
        out(f"  -> IS: Schwach positiver Edge (Median pos, aber <50% WR)")
    else:
        out(f"  -> IS: KEIN EDGE (Median <= 0)")

seg_a_is = is_stats_all[is_stats_all['Segment'] == 'A: ALL']
if len(seg_a_is) > 0:
    a = seg_a_is.iloc[0]
    out(f"\n  Baseline (ALL):")
    out(f"    N = {int(a['N'])}, Median = {a['Median_MoveAfter']:.4f}, % pos = {a['Pct_Positive']:.1f}%")


# ──────────────────────────────────────────────────────────────────────
# PART 2: OOS VALIDATION
# ──────────────────────────────────────────────────────────────────────

section("=== OOS VALIDATION (from 2024-01-01) ===")

section("ANALYSE 1: REST-MOVE NACH 10:00 — OOS (ALL)")
oos_segs = get_segments(df_oos)
oos_stats_all = compute_segment_stats(oos_segs, "OOS_ALL")
out_df(oos_stats_all)

section("ANALYSE 1: REST-MOVE NACH 10:00 — OOS (GapUp only)")
oos_segs_up = get_segments(df_oos[df_oos['gap_direction'] == 'up'])
oos_stats_up = compute_segment_stats(oos_segs_up, "OOS_UP")
out_df(oos_stats_up)

section("ANALYSE 1: REST-MOVE NACH 10:00 — OOS (GapDn only)")
oos_segs_dn = get_segments(df_oos[df_oos['gap_direction'] == 'down'])
oos_stats_dn = compute_segment_stats(oos_segs_dn, "OOS_DN")
out_df(oos_stats_dn)

# Timing OOS
section("ANALYSE 2: TIMING DES MOVES — OOS")
for seg_name in ['A: ALL', 'C: drift>0.50 + RVOL>5', 'F: drift<0 (fader)']:
    timing = compute_timing_stats(oos_segs[seg_name], f"OOS_{seg_name}")
    if timing:
        out(f"\n  {seg_name}:")
        for k, v in timing.items():
            out(f"    {k}: {v:.2f}" if isinstance(v, float) else f"    {k}: {v}")
    else:
        out(f"\n  {seg_name}: N=0 or insufficient data")

# Fading OOS
section("ANALYSE 3: FADING-AKTIEN — OOS (ALL)")
oos_fading_all = compute_fading_stats(df_oos)
out_df(oos_fading_all)

section("ANALYSE 3: FADING-AKTIEN — OOS (GapUp)")
oos_fading_up = compute_fading_stats(df_oos[df_oos['gap_direction'] == 'up'])
out_df(oos_fading_up)

section("ANALYSE 3: FADING-AKTIEN — OOS (GapDn)")
oos_fading_dn = compute_fading_stats(df_oos[df_oos['gap_direction'] == 'down'])
out_df(oos_fading_dn)

section("ANALYSE 3 DETAIL: FADER-WENDE — OOS")
faders_oos = df_oos[df_oos['drift_30'] < 0]
n_faders_oos = len(faders_oos)
n_wende_oos = (faders_oos['move_after_1000'] > 0).sum()
out(f"  Total faders (drift<0): {n_faders_oos}")
out(f"  Davon wenden (move_after>0): {n_wende_oos} ({n_wende_oos/n_faders_oos*100:.1f}%)" if n_faders_oos > 0 else "  N/A")
if n_wende_oos > 0:
    out(f"  Median move_after bei Wendern: {faders_oos[faders_oos['move_after_1000']>0]['move_after_1000'].median():.4f} ADR")
    out(f"  Median move_after bei Nicht-Wendern: {faders_oos[faders_oos['move_after_1000']<=0]['move_after_1000'].median():.4f} ADR")

# Entry Level OOS
section("ANALYSE 4: ENTRY-LEVEL PROBLEM — OOS")
for gd_label, gd_val in [("ALL", None), ("GapUp", "up"), ("GapDn", "down")]:
    entry = compute_entry_level_stats(df_oos, gd_val)
    if entry:
        out(f"\n  Segment C — {gd_label}:")
        for k, v in entry.items():
            out(f"    {k}: {v:.2f}%" if 'Rate' in k else (f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}"))
    else:
        out(f"\n  Segment C — {gd_label}: No data")

# Correlations OOS
section("ANALYSE 5: KORRELATIONEN — OOS")
for gd_label, gd_filter in [("ALL", df_oos), ("GapUp", df_oos[df_oos['gap_direction'] == 'up'])]:
    corr = compute_correlations(gd_filter, f"OOS_{gd_label}")
    if corr:
        out(f"\n  {gd_label}:")
        for k, v in corr.items():
            out(f"    {k}: {v:.6f}" if isinstance(v, float) else f"    {k}: {v}")


# ──────────────────────────────────────────────────────────────────────
# PART 3: SHRINKAGE
# ──────────────────────────────────────────────────────────────────────

section("=== SHRINKAGE (OOS vs IS) ===")

shrinkage_all = compute_shrinkage(is_stats_all, oos_stats_all)
out("\nALL gap directions:")
out_df(shrinkage_all, float_fmt='.2f')

shrinkage_up = compute_shrinkage(is_stats_up, oos_stats_up)
out("\n\nGapUp only:")
out_df(shrinkage_up, float_fmt='.2f')

shrinkage_dn = compute_shrinkage(is_stats_dn, oos_stats_dn)
out("\n\nGapDn only:")
out_df(shrinkage_dn, float_fmt='.2f')


# ──────────────────────────────────────────────────────────────────────
# PART 4: GESAMTFAZIT
# ──────────────────────────────────────────────────────────────────────

section("=== GESAMTFAZIT ===")

# Check segment C IS vs OOS
seg_c_oos = oos_stats_all[oos_stats_all['Segment'] == 'C: drift>0.50 + RVOL>5']
if len(seg_c_is) > 0 and len(seg_c_oos) > 0:
    c_is = seg_c_is.iloc[0]
    c_oos = seg_c_oos.iloc[0]
    out(f"\n  SEGMENT C (drift>0.50 + RVOL>5):")
    out(f"    IS:  N={int(c_is['N']):4d}  Median={c_is['Median_MoveAfter']:.4f}  Mean={c_is['Mean_MoveAfter']:.4f}  %pos={c_is['Pct_Positive']:.1f}%  Sharpe={c_is['Sharpe']:.4f}")
    out(f"    OOS: N={int(c_oos['N']):4d}  Median={c_oos['Median_MoveAfter']:.4f}  Mean={c_oos['Mean_MoveAfter']:.4f}  %pos={c_oos['Pct_Positive']:.1f}%  Sharpe={c_oos['Sharpe']:.4f}")

    median_shrink = (c_oos['Median_MoveAfter'] - c_is['Median_MoveAfter']) / abs(c_is['Median_MoveAfter']) * 100 if abs(c_is['Median_MoveAfter']) > 1e-6 else np.nan
    out(f"    Median Shrinkage: {median_shrink:.1f}%")

    is_edge = c_is['Median_MoveAfter'] > 0 and c_is['Pct_Positive'] > 50
    oos_edge = c_oos['Median_MoveAfter'] > 0 and c_oos['Pct_Positive'] > 50

    if is_edge and oos_edge:
        out(f"\n  -> IS-Edge BESTÄTIGT auf OOS: JA")
        out(f"  -> Empfehlung: BACKTEST LOHNT SICH")
        out(f"  -> Vielversprechendes Segment: drift_30 > 0.50 + RVOL_30 > 5")
    elif is_edge and not oos_edge:
        out(f"\n  -> IS-Edge bricht auf OOS zusammen: OVERFITTING")
        out(f"  -> Empfehlung: Backtest lohnt sich NICHT für dieses Segment")
    elif not is_edge:
        out(f"\n  -> Kein IS-Edge: KEIN EDGE")
        out(f"  -> Empfehlung: Kein weiterer Test nötig")

# Check all segments
out(f"\n  ÜBERSICHT ALLER SEGMENTE:")
for _, row_is in is_stats_all.iterrows():
    seg_name = row_is['Segment']
    row_oos = oos_stats_all[oos_stats_all['Segment'] == seg_name]
    if len(row_oos) == 0:
        continue
    row_oos = row_oos.iloc[0]
    is_pos = row_is['Median_MoveAfter'] > 0 and row_is['Pct_Positive'] > 50
    oos_pos = row_oos['Median_MoveAfter'] > 0 and row_oos['Pct_Positive'] > 50
    status = "IS+OOS OK" if is_pos and oos_pos else ("IS only (OOS broken)" if is_pos else "No IS edge")
    out(f"    {seg_name:30s}  IS_med={row_is['Median_MoveAfter']:+.4f}  OOS_med={row_oos['Median_MoveAfter']:+.4f}  Status={status}")


# ── SAVE TEXT OUTPUT ────────────────────────────────────────────────
with open(f'{OUT_DIR}/rest_move_analysis.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(output_lines))
print(f"\n\nText output saved to {OUT_DIR}/rest_move_analysis.txt")


# ══════════════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════════════

def setup_style():
    plt.style.use('dark_background')
    plt.rcParams.update({
        'figure.facecolor': STYLE_BG,
        'axes.facecolor': STYLE_BG,
        'axes.edgecolor': STYLE_GRID,
        'axes.labelcolor': STYLE_FG,
        'text.color': STYLE_FG,
        'xtick.color': STYLE_FG,
        'ytick.color': STYLE_FG,
        'grid.color': STYLE_GRID,
        'grid.alpha': 0.3,
        'font.size': 10,
    })

setup_style()

seg_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
seg_names = list(get_segments(df_is).keys())


# ── CHART 1: Boxplot move_after_1000 per segment (IS | OOS) ──────
print("Generating Chart 1: Boxplots...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
fig.suptitle('Move After 10:00 by Segment (ADR)', fontsize=14, fontweight='bold', color=STYLE_FG)

for ax, data_split, title in [(ax1, df_is, 'IS (before 2024)'), (ax2, df_oos, 'OOS (from 2024)')]:
    segs = get_segments(data_split)
    box_data = []
    labels = []
    medians = []
    ns = []
    for i, (name, seg) in enumerate(segs.items()):
        if len(seg) > 0:
            box_data.append(seg['move_after_1000'].clip(-3, 5).values)
            medians.append(seg['move_after_1000'].median())
            ns.append(len(seg))
        else:
            box_data.append([0])
            medians.append(0)
            ns.append(0)
        labels.append(seg_labels[i])

    bp = ax.boxplot(box_data, labels=labels, patch_artist=True, showfliers=False,
                    medianprops=dict(color=STYLE_YELLOW, linewidth=2),
                    whiskerprops=dict(color=STYLE_FG),
                    capprops=dict(color=STYLE_FG))
    for patch in bp['boxes']:
        patch.set_facecolor(STYLE_BLUE + '66')
        patch.set_edgecolor(STYLE_BLUE)

    ax.axhline(0, color=STYLE_RED, linewidth=1, linestyle='--', alpha=0.7)
    ax.set_title(title, fontsize=12, color=STYLE_FG)
    ax.set_xlabel('Segment')
    ax.set_ylabel('move_after_1000 (ADR)' if ax == ax1 else '')
    ax.grid(True, alpha=0.2)

    for i, (med, n) in enumerate(zip(medians, ns)):
        ax.annotate(f'N={n}\nMed={med:.3f}', xy=(i + 1, ax.get_ylim()[1] * 0.92),
                    ha='center', fontsize=7, color=STYLE_CYAN)

plt.tight_layout()
fig.savefig(f'{OUT_DIR}/chart1_boxplot_segments.png', dpi=DPI, bbox_inches='tight',
            facecolor=STYLE_BG, edgecolor='none')
plt.close()


# ── CHART 2: Scatter drift_30 vs move_after_1000 (IS | OOS) ──────
print("Generating Chart 2: Scatter drift vs move...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('drift_30 vs move_after_1000 (GapUp only)', fontsize=14, fontweight='bold')

for ax, data_split, title in [(ax1, df_is, 'IS'), (ax2, df_oos, 'OOS')]:
    d = data_split[data_split['gap_direction'] == 'up'].dropna(subset=['drift_30', 'move_after_1000', 'RVOL_30'])
    # Clip for visibility
    d_plot = d.copy()
    d_plot['drift_30_c'] = d_plot['drift_30'].clip(-2, 5)
    d_plot['move_after_c'] = d_plot['move_after_1000'].clip(-3, 5)

    hi_rvol = d_plot[d_plot['RVOL_30'] > 5]
    lo_rvol = d_plot[d_plot['RVOL_30'] <= 5]

    ax.scatter(lo_rvol['drift_30_c'], lo_rvol['move_after_c'], s=12, alpha=0.25, c=STYLE_GREY, label='RVOL≤5')
    ax.scatter(hi_rvol['drift_30_c'], hi_rvol['move_after_c'], s=18, alpha=0.5, c=STYLE_GREEN, label='RVOL>5')

    # Regression
    slope, intercept, r_value, p_value, _ = stats.linregress(d['drift_30'], d['move_after_1000'])
    x_line = np.linspace(d_plot['drift_30_c'].min(), d_plot['drift_30_c'].max(), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color=STYLE_RED, linewidth=2, linestyle='--')

    ax.axhline(0, color=STYLE_FG, linewidth=0.5, alpha=0.3)
    ax.axvline(0, color=STYLE_FG, linewidth=0.5, alpha=0.3)
    ax.set_title(f'{title}  (N={len(d)}, R²={r_value**2:.4f}, p={p_value:.4f})', fontsize=11)
    ax.set_xlabel('drift_30 (ADR)')
    ax.set_ylabel('move_after_1000 (ADR)')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.2)

plt.tight_layout()
fig.savefig(f'{OUT_DIR}/chart2_scatter_drift_vs_move.png', dpi=DPI, bbox_inches='tight',
            facecolor=STYLE_BG, edgecolor='none')
plt.close()


# ── CHART 3: Scatter RVOL_30 vs move_after_1000 (IS | OOS) ──────
print("Generating Chart 3: Scatter RVOL vs move...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('RVOL_30 vs move_after_1000', fontsize=14, fontweight='bold')

for ax, data_split, title in [(ax1, df_is, 'IS'), (ax2, df_oos, 'OOS')]:
    d = data_split.dropna(subset=['drift_30', 'move_after_1000', 'RVOL_30'])
    d_plot = d.copy()
    d_plot['RVOL_c'] = d_plot['RVOL_30'].clip(0, 30)
    d_plot['move_after_c'] = d_plot['move_after_1000'].clip(-3, 5)

    hi_drift = d_plot[d_plot['drift_30'] > 0.50]
    neg_drift = d_plot[d_plot['drift_30'] < 0]
    mid_drift = d_plot[(d_plot['drift_30'] >= 0) & (d_plot['drift_30'] <= 0.50)]

    ax.scatter(mid_drift['RVOL_c'], mid_drift['move_after_c'], s=12, alpha=0.25, c=STYLE_GREY, label='0≤drift≤0.50')
    ax.scatter(neg_drift['RVOL_c'], neg_drift['move_after_c'], s=12, alpha=0.35, c=STYLE_RED, label='drift<0')
    ax.scatter(hi_drift['RVOL_c'], hi_drift['move_after_c'], s=18, alpha=0.5, c=STYLE_GREEN, label='drift>0.50')

    # Regression
    slope, intercept, r_value, p_value, _ = stats.linregress(d['RVOL_30'], d['move_after_1000'])
    x_line = np.linspace(0, 30, 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color=STYLE_CYAN, linewidth=2, linestyle='--')

    ax.axhline(0, color=STYLE_FG, linewidth=0.5, alpha=0.3)
    ax.set_title(f'{title}  (N={len(d)}, R²={r_value**2:.4f}, p={p_value:.4f})', fontsize=11)
    ax.set_xlabel('RVOL_30')
    ax.set_ylabel('move_after_1000 (ADR)')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.2)

plt.tight_layout()
fig.savefig(f'{OUT_DIR}/chart3_scatter_rvol_vs_move.png', dpi=DPI, bbox_inches='tight',
            facecolor=STYLE_BG, edgecolor='none')
plt.close()


# ── CHART 4: Stacked Bar — Anteil Move vor/nach 10:00 ────────────
print("Generating Chart 4: Timing stacked bars...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Anteil Move vor / nach 10:00 (nur positive total_move)', fontsize=14, fontweight='bold')

timing_seg_names = ['A: ALL', 'C: drift>0.50 + RVOL>5', 'F: drift<0 (fader)']
timing_labels = ['A: ALL', 'C: drift>0.5\n+RVOL>5', 'F: fader']

for ax, data_split, title in [(ax1, df_is, 'IS'), (ax2, df_oos, 'OOS')]:
    segs = get_segments(data_split)
    befores = []
    afters = []
    for sn in timing_seg_names:
        t = compute_timing_stats(segs[sn])
        if t:
            befores.append(t['Median_Pct_Before'])
            afters.append(t['Median_Pct_After'])
        else:
            befores.append(0)
            afters.append(0)

    x = np.arange(len(timing_labels))
    width = 0.5
    # Normalize: some pct_before can be >100 if move_after is negative
    # Just show raw medians clipped
    befores_c = np.clip(befores, 0, 200)
    afters_c = np.clip(afters, -100, 200)

    ax.bar(x, befores_c, width, color=STYLE_GREEN, alpha=0.7, label='Before 10:00')
    ax.bar(x, afters_c, width, bottom=befores_c, color=STYLE_BLUE, alpha=0.7, label='After 10:00')

    ax.set_xticks(x)
    ax.set_xticklabels(timing_labels, fontsize=9)
    ax.set_ylabel('Median % of total move')
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    ax.axhline(100, color=STYLE_FG, linewidth=0.5, linestyle=':', alpha=0.5)
    ax.grid(True, alpha=0.2, axis='y')

    for i, (b, a) in enumerate(zip(befores_c, afters_c)):
        ax.annotate(f'{b:.0f}%', xy=(i, b / 2), ha='center', fontsize=9, color='white', fontweight='bold')
        if a > 5:
            ax.annotate(f'{a:.0f}%', xy=(i, b + a / 2), ha='center', fontsize=9, color='white', fontweight='bold')

plt.tight_layout()
fig.savefig(f'{OUT_DIR}/chart4_timing_stacked.png', dpi=DPI, bbox_inches='tight',
            facecolor=STYLE_BG, edgecolor='none')
plt.close()


# ── CHART 5: Histogram move_after_1000 for Segment C ─────────────
print("Generating Chart 5: Histogram Segment C...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Distribution of move_after_1000 — Segment C (drift>0.50 + RVOL>5)', fontsize=13, fontweight='bold')

bins_range = np.arange(-1.0, 3.25, 0.25)

for ax, data_split, title in [(ax1, df_is, 'IS'), (ax2, df_oos, 'OOS')]:
    seg_c = data_split[(data_split['drift_30'] > 0.50) & (data_split['RVOL_30'] > 5)]
    vals = seg_c['move_after_1000'].clip(-1.0, 3.0)

    ax.hist(vals, bins=bins_range, color=STYLE_BLUE, edgecolor=STYLE_CYAN, alpha=0.7)
    ax.axvline(0, color=STYLE_RED, linewidth=1.5, linestyle='--')
    if len(seg_c) > 0:
        med = seg_c['move_after_1000'].median()
        ax.axvline(med, color=STYLE_YELLOW, linewidth=2, linestyle='-', label=f'Median={med:.3f}')
        pct_pos = (seg_c['move_after_1000'] > 0).mean() * 100
        ax.set_title(f'{title}  (N={len(seg_c)}, Med={med:.3f}, %pos={pct_pos:.1f}%)', fontsize=11)
        ax.legend(fontsize=9)
    else:
        ax.set_title(f'{title}  (N=0)', fontsize=11)

    ax.set_xlabel('move_after_1000 (ADR)')
    ax.set_ylabel('Count')
    ax.grid(True, alpha=0.2)

plt.tight_layout()
fig.savefig(f'{OUT_DIR}/chart5_histogram_segC.png', dpi=DPI, bbox_inches='tight',
            facecolor=STYLE_BG, edgecolor='none')
plt.close()


# ── CHART 6: Shrinkage Overview Bar Chart ────────────────────────
print("Generating Chart 6: Shrinkage overview...")
fig, ax = plt.subplots(figsize=(14, 7))
fig.suptitle('Median move_after_1000 — IS vs OOS by Segment', fontsize=14, fontweight='bold')

x = np.arange(len(seg_labels))
width = 0.35

is_medians = []
oos_medians = []
for sn in seg_names:
    is_row = is_stats_all[is_stats_all['Segment'] == sn]
    oos_row = oos_stats_all[oos_stats_all['Segment'] == sn]
    is_medians.append(is_row.iloc[0]['Median_MoveAfter'] if len(is_row) > 0 else 0)
    oos_medians.append(oos_row.iloc[0]['Median_MoveAfter'] if len(oos_row) > 0 else 0)

bars_is = ax.bar(x - width/2, is_medians, width, color=STYLE_BLUE, alpha=0.8, label='IS')
bars_oos = ax.bar(x + width/2, oos_medians, width, color=STYLE_CYAN, alpha=0.8, label='OOS',
                   edgecolor=STYLE_CYAN, linewidth=1.5)

ax.axhline(0, color=STYLE_RED, linewidth=1, linestyle='--', alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels([f'{l}\n{n.split(":")[1][:20]}' for l, n in zip(seg_labels, seg_names)], fontsize=8)
ax.set_ylabel('Median move_after_1000 (ADR)')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.2, axis='y')

# Annotate with shrinkage %
for i, (is_m, oos_m) in enumerate(zip(is_medians, oos_medians)):
    if abs(is_m) > 1e-4:
        shrink = (oos_m - is_m) / abs(is_m) * 100
        color = STYLE_GREEN if shrink > -20 else (STYLE_YELLOW if shrink > -50 else STYLE_RED)
        y_pos = max(is_m, oos_m) + 0.02
        ax.annotate(f'{shrink:+.0f}%', xy=(i, y_pos), ha='center', fontsize=9,
                    color=color, fontweight='bold')

plt.tight_layout()
fig.savefig(f'{OUT_DIR}/chart6_shrinkage_overview.png', dpi=DPI, bbox_inches='tight',
            facecolor=STYLE_BG, edgecolor='none')
plt.close()


print(f"\nAll charts saved to {OUT_DIR}/")
print("DONE.")
