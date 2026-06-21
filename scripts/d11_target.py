"""
D11 Aufgabe 1: Target-Variable (is_big_runner) + Basisstatistik (IS)
"""
import pandas as pd
import numpy as np
import os, sys

OUT = "results/d11_target.txt"
lines = []

def w(text=""):
    lines.append(text)

def header(title):
    w("=" * 70)
    w(title)
    w("=" * 70)

# =====================================================================
# Load metadata
# =====================================================================
meta = pd.read_parquet("data/metadata/metadata_v8_5.parquet")
is_all = meta[(meta['date'] >= '2021-02-21') & (meta['date'] <= '2023-12-31')].copy()
print(f"IS all gapper: {len(is_all)}")

# =====================================================================
# 1a: Compute is_big_runner for ALL gappers
# =====================================================================
# RTH values already in metadata: rth_open, rth_high, rth_low, rth_close, adr_10

def compute_big_runner(df):
    """Compute big runner classification for all rows."""
    df = df.copy()
    adr = df['adr_10']

    # GapUp: range_up and close_retention
    range_up = (df['rth_high'] - df['rth_open']) / adr
    denom_up = df['rth_high'] - df['rth_open']
    close_ret_up = np.where(denom_up > 0,
                            (df['rth_close'] - df['rth_open']) / denom_up,
                            np.nan)

    # GapDown: range_down and close_retention
    range_dn = (df['rth_open'] - df['rth_low']) / adr
    denom_dn = df['rth_open'] - df['rth_low']
    close_ret_dn = np.where(denom_dn > 0,
                            (df['rth_open'] - df['rth_close']) / denom_dn,
                            np.nan)

    is_up = df['gap_direction'] == 'up'
    is_dn = df['gap_direction'] == 'down'

    # Range condition
    range_in_gap_dir = np.where(is_up, range_up, np.where(is_dn, range_dn, np.nan))
    close_retention = np.where(is_up, close_ret_up, np.where(is_dn, close_ret_dn, np.nan))

    df['range_in_gap_dir_adr'] = range_in_gap_dir
    df['close_retention'] = close_retention

    range_ok = range_in_gap_dir >= 1.5
    close_ok = close_retention >= 0.80

    df['is_big_runner'] = range_ok & close_ok
    df['range_ok_close_bad'] = range_ok & ~close_ok
    df['range_bad'] = ~range_ok

    # Handle NaN adr
    nan_mask = df['adr_10'].isna()
    df.loc[nan_mask, 'is_big_runner'] = False
    df.loc[nan_mask, 'range_ok_close_bad'] = False
    df.loc[nan_mask, 'range_bad'] = True

    return df

is_all = compute_big_runner(is_all)

header("D11 AUFGABE 1: TARGET-VARIABLE + BASISSTATISTIK (IS)")
w()
w("Big Runner: Range >= 1.5 ADR in Gap-Richtung AND Close >= 80% des Extremums")
w(f"IS: 2021-02-21 bis 2023-12-31, N = {len(is_all)}")
w()
w()

# --- 1a: ALL gappers ---
header("1a: BIG RUNNER VERTEILUNG — ALLE GAPPER (IS)")
w()

for label, mask in [("Big Runner", is_all['is_big_runner']),
                     ("Range OK Close Bad", is_all['range_ok_close_bad']),
                     ("Range Bad", is_all['range_bad'])]:
    n_up = mask[is_all['gap_direction'] == 'up'].sum()
    n_dn = mask[is_all['gap_direction'] == 'down'].sum()
    tot_up = (is_all['gap_direction'] == 'up').sum()
    tot_dn = (is_all['gap_direction'] == 'down').sum()
    pct_up = n_up / tot_up * 100 if tot_up > 0 else 0
    pct_dn = n_dn / tot_dn * 100 if tot_dn > 0 else 0
    w(f"  {label:<22s} | N GapUp={n_up:>5d} ({pct_up:>5.1f}%) | N GapDn={n_dn:>5d} ({pct_dn:>5.1f}%)")

tot_up = (is_all['gap_direction'] == 'up').sum()
tot_dn = (is_all['gap_direction'] == 'down').sum()
w(f"  {'Total':<22s} | N GapUp={tot_up:>5d}         | N GapDn={tot_dn:>5d}")
w()
w()

# --- 1b: Only OD > 0.5 + with_gap ---
header("1b: BIG RUNNER INNERHALB OD > 0.5 + WITH_GAP (IS)")
w()

od_mask = (is_all['od_strength'] > 0.5) & (is_all['od_direction'] == 'with_gap')
filtered = is_all[od_mask].copy()
w(f"  Filter: OD > 0.5 ADR + with_gap → N = {len(filtered)}")
w()

for gap_dir, label in [('up', 'GapUp'), ('down', 'GapDn')]:
    g = filtered[filtered['gap_direction'] == gap_dir]
    n_br = g['is_big_runner'].sum()
    n_total = len(g)
    pct = n_br / n_total * 100 if n_total > 0 else 0
    w(f"  {label}: Big Runner = {n_br}/{n_total} = {pct:.1f}%")
    n_roc = g['range_ok_close_bad'].sum()
    n_rb = g['range_bad'].sum()
    w(f"         Range OK Close Bad = {n_roc} ({n_roc/n_total*100:.1f}%)")
    w(f"         Range Bad = {n_rb} ({n_rb/n_total*100:.1f}%)")
    w()

w("  → Das ist die 'Target-Rate' die wir vorhersagen wollen.")
w()
w()

# --- 1c: Big Runner Profil ---
header("1c: BIG RUNNER PROFIL (Median-Werte, OD > 0.5 + with_gap)")
w()

# Load D10 MFE data and merge
d10_is = pd.read_parquet("results/d10_mfe_raw_is.parquet")
d10_is = d10_is.rename(columns={'gap_direction': 'gap_dir_d10'})
# Merge on ticker+date
filtered = filtered.merge(
    d10_is[['ticker', 'date', 'mfe_live_r', 'sl_hit']],
    on=['ticker', 'date'], how='left'
)

profile_cols = [
    ('gap_size_in_adr', 'Gap_in_ADR'),
    ('od_strength', 'OD_strength_ADR'),
    ('rvol_5', 'RVOL_5'),
    ('rvol_open_30min', 'RVOL_30'),
    ('pm_rth5', 'PM/RTH5'),
    ('pm_rth30_computed', 'PM/RTH30'),
    ('adr_10', 'ADR_10 (Dollar)'),
    ('prev_close', 'Preis (Dollar)'),
    ('market_cap', 'Marktkapital.'),
    ('od_body_pct', 'od_body_pct'),
    ('od_wick_ratio', 'od_wick_ratio'),
    ('full_drift', 'full_drift (ADR)'),
    ('rest_drift', 'rest_drift (ADR)'),
    ('mfe_live_r', 'MFE_R (D10)'),
    ('range_in_gap_dir_adr', 'Range in Gap Dir'),
    ('close_retention', 'Close Retention'),
]

for gap_dir, label in [('up', 'GapUp'), ('down', 'GapDn')]:
    g = filtered[filtered['gap_direction'] == gap_dir]
    br = g[g['is_big_runner']]
    nr = g[~g['is_big_runner']]

    w(f"  --- {label} (Big Runner N={len(br)}, Non-Runner N={len(nr)}) ---")
    w(f"  {'Metrik':<22s} | {'Big Runner':>12s} | {'Non-Runner':>12s} | {'Delta':>10s}")
    w(f"  {'-'*64}")

    for col, col_label in profile_cols:
        if col not in g.columns:
            continue
        br_val = br[col].median()
        nr_val = nr[col].median()
        delta = br_val - nr_val if pd.notna(br_val) and pd.notna(nr_val) else np.nan

        # Format based on magnitude
        if col == 'market_cap':
            br_s = f"{br_val/1e9:.2f}B" if pd.notna(br_val) else "NaN"
            nr_s = f"{nr_val/1e9:.2f}B" if pd.notna(nr_val) else "NaN"
            d_s = f"{delta/1e9:+.2f}B" if pd.notna(delta) else "NaN"
        elif col in ('prev_close', 'adr_10'):
            br_s = f"${br_val:.1f}" if pd.notna(br_val) else "NaN"
            nr_s = f"${nr_val:.1f}" if pd.notna(nr_val) else "NaN"
            d_s = f"${delta:+.1f}" if pd.notna(delta) else "NaN"
        else:
            br_s = f"{br_val:.3f}" if pd.notna(br_val) else "NaN"
            nr_s = f"{nr_val:.3f}" if pd.notna(nr_val) else "NaN"
            d_s = f"{delta:+.3f}" if pd.notna(delta) else "NaN"

        w(f"  {col_label:<22s} | {br_s:>12s} | {nr_s:>12s} | {d_s:>10s}")

    w()
    w()

# Save intermediate data for later scripts
save_cols = ['ticker', 'date', 'gap_direction', 'is_big_runner', 'range_ok_close_bad',
             'range_bad', 'range_in_gap_dir_adr', 'close_retention',
             'od_strength', 'od_direction', 'od_body_pct', 'od_wick_ratio',
             'rvol_5', 'rvol_open_30min', 'pm_rth5', 'pm_rth30_computed',
             'gap_size_in_adr', 'adr_10', 'prev_close', 'market_cap',
             'market_cap_bucket', 'sector', 'is_earnings', 'day_of_week',
             'full_drift', 'rest_drift', 'rest_drift_1000',
             'close_935', 'close_1000', 'rth_open', 'rth_high', 'rth_low', 'rth_close',
             'spy_return_day', 'rsi_14_prev', 'prior_return_10d',
             'dist_from_52w_high', 'first_candle_dir', 'first_candle_size',
             'candle5_vs_candle1', 'close_vs_open_adr',
             'rth5_vol', 'rth30_vol', 'pm_vol', 'premarket_volume',
             'hod_time', 'lod_time', 'mfe_live_r', 'sl_hit']

# Keep only cols that exist
save_cols = [c for c in save_cols if c in filtered.columns]
filtered[save_cols].to_parquet("results/d11_is_base.parquet", index=False)

# Also save ALL gapper IS data for broader analysis (Aufgabe 2 needs it for all gappers)
all_save_cols = [c for c in save_cols if c in is_all.columns]
is_all[all_save_cols].to_parquet("results/d11_is_all.parquet", index=False)

w(f"  Gespeichert: results/d11_is_base.parquet (OD>0.5+with_gap, N={len(filtered)})")
w(f"  Gespeichert: results/d11_is_all.parquet (alle Gapper IS, N={len(is_all)})")

# Write output
os.makedirs("results", exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print(f"Geschrieben: {OUT} ({len(lines)} Zeilen)")
