"""
Durchlauf 7.0 — Aufgabe 3: Combo E mit neuen Features
Combo E = Grosse erste Kerze (>0.20 ADR in Gap-Richtung) + OD >0.5 ADR with Gap
IS only (H1: 2021-02-21 bis 2023-12-31).
"""
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
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

# ══════════════════════════════════════════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════════════════════════════════════════
print("Loading metadata_v7...", file=sys.stderr)
meta = pd.read_parquet('data/metadata/metadata_v7.parquet')
meta['date'] = pd.to_datetime(meta['date'])
h1 = meta[(meta['date'] >= '2021-02-21') & (meta['date'] <= '2023-12-31')].copy()
h1 = h1[h1['adr_10'].notna() & (h1['adr_10'] > 0)].copy()

# Combo E definition
h1['combo_e'] = (
    (h1['first_candle_dir'] == 'with_gap') &
    (h1['first_candle_size'] > 0.20) &
    (h1['od_direction'] == 'with_gap') &
    (h1['od_strength'] > 0.5)
)

w("=" * 80)
w("AUFGABE 3: COMBO E MIT NEUEN FEATURES")
w("=" * 80)

for gap_dir in ['up', 'down']:
    ce = h1[(h1['gap_direction'] == gap_dir) & h1['combo_e']].copy()
    non_ce = h1[(h1['gap_direction'] == gap_dir) & ~h1['combo_e']].copy()
    w(f"\n--- Gap{gap_dir.title()} ---")
    w(f"Combo E: N={len(ce)}, Non-Combo-E: N={len(non_ce)}")

    # ══════════════════════════════════════════════════════════════════════
    # 3a: Combo E Rest-Drift
    # ══════════════════════════════════════════════════════════════════════
    w(f"\n  3a: Combo E Rest-Drift")
    for label, grp in [('Combo E', ce), ('Non-Combo-E', non_ce)]:
        fd = grp['full_drift'].mean()
        rd = grp['rest_drift'].mean()
        rd1000 = grp['rest_drift_1000'].mean()
        rd1100 = grp['rest_drift_1100'].mean()
        ci_lo, ci_hi = bootstrap_ci(grp['rest_drift'].values)
        w(f"    {label:15s}: N={len(grp):>5}, full_drift={fd:+.4f}, rest_drift={rd:+.4f} ({ci_lo:+.3f} to {ci_hi:+.3f}), "
          f"rd_1000={rd1000:+.4f}, rd_1100={rd1100:+.4f}")

    w(f"\n  Rest-Drift Decomposition for Combo E:")
    w(f"    full_drift (open to close):  {ce['full_drift'].mean():+.4f}")
    w(f"    OD contribution (open to 935): {ce['full_drift'].mean() - ce['rest_drift'].mean():+.4f}")
    w(f"    rest_drift (935 to close):   {ce['rest_drift'].mean():+.4f}")
    w(f"    rest_drift_1000 (935-1000):  {ce['rest_drift_1000'].mean():+.4f}")
    w(f"    rest_drift_1100 (935-1100):  {ce['rest_drift_1100'].mean():+.4f}")

    # ══════════════════════════════════════════════════════════════════════
    # 3b: Combo E x RVOL_5
    # ══════════════════════════════════════════════════════════════════════
    w(f"\n  3b: Combo E x RVOL_5")
    rvol5_edges = [0, 1, 2, 3, 4, 5, 7, 10, 999]
    rvol5_labels = ['<1x', '1-2x', '2-3x', '3-4x', '4-5x', '5-7x', '7-10x', '>10x']
    ce_v = ce[ce['rvol_5'].notna()].copy()
    ce_v['rvol5_bkt'] = pd.cut(ce_v['rvol_5'], bins=rvol5_edges, labels=rvol5_labels, right=False)

    w(f"    {'RVOL_5':<10} {'N':>5} {'rest_drift':>12} {'full_drift':>12} {'CL':>8}")
    w("    " + "-" * 55)
    for bkt in rvol5_labels:
        grp = ce_v[ce_v['rvol5_bkt'] == bkt]
        if len(grp) == 0: continue
        w(f"    {bkt:<10} {len(grp):>5}{low_n(len(grp))} {grp['rest_drift'].mean():>+12.4f} {grp['full_drift'].mean():>+12.4f} {grp['cl'].mean():>8.3f}")

    # ══════════════════════════════════════════════════════════════════════
    # 3c: Combo E x PM/RTH5
    # ══════════════════════════════════════════════════════════════════════
    w(f"\n  3c: Combo E x PM/RTH5")
    pm5_edges = [0, 0.30, 0.50, 1.0, 2.0, 999]
    pm5_labels = ['<30%', '30-50%', '50-100%', '100-200%', '>200%']
    ce_p = ce[ce['pm_rth5'].notna()].copy()
    ce_p['pm5_bkt'] = pd.cut(ce_p['pm_rth5'], bins=pm5_edges, labels=pm5_labels, right=False)

    w(f"    {'PM/RTH5':<12} {'N':>5} {'rest_drift':>12} {'full_drift':>12} {'CL':>8}")
    w("    " + "-" * 55)
    for bkt in pm5_labels:
        grp = ce_p[ce_p['pm5_bkt'] == bkt]
        if len(grp) == 0: continue
        w(f"    {bkt:<12} {len(grp):>5}{low_n(len(grp))} {grp['rest_drift'].mean():>+12.4f} {grp['full_drift'].mean():>+12.4f} {grp['cl'].mean():>8.3f}")

    # ══════════════════════════════════════════════════════════════════════
    # 3d: Combo E x RVOL_30 (retrospektiv)
    # ══════════════════════════════════════════════════════════════════════
    w(f"\n  3d: Combo E x RVOL_30 (retrospektiv, ab 10:00)")
    ce_r30 = ce[ce['rvol_at_time_30min'].notna()].copy()
    ce_r30['rvol30_bkt'] = pd.cut(ce_r30['rvol_at_time_30min'], bins=rvol5_edges, labels=rvol5_labels, right=False)

    w(f"    {'RVOL_30':<10} {'N':>5} {'rest_drift':>12} {'full_drift':>12} {'CL':>8}")
    w("    " + "-" * 55)
    for bkt in rvol5_labels:
        grp = ce_r30[ce_r30['rvol30_bkt'] == bkt]
        if len(grp) == 0: continue
        w(f"    {bkt:<10} {len(grp):>5}{low_n(len(grp))} {grp['rest_drift'].mean():>+12.4f} {grp['full_drift'].mean():>+12.4f} {grp['cl'].mean():>8.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# MFE/MAE from 9:35 (requires 1min data)
# ══════════════════════════════════════════════════════════════════════════════
w("\n" + "=" * 80)
w("3a EXTENDED: MFE/MAE FROM 9:35 FOR COMBO E")
w("=" * 80)

RAW_DIR = Path('data/raw_1min')
mfe_records = []

combo_e_rows = h1[h1['combo_e']].copy()
print(f"Computing MFE/MAE from 9:35 for {len(combo_e_rows)} Combo E days...", file=sys.stderr)

for _, row in tqdm(combo_e_rows.iterrows(), total=len(combo_e_rows), desc="MFE/MAE", file=sys.stderr):
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
    if len(rth) < 60:
        continue

    # Bars from index 5 (09:35) onward
    post_od = rth.iloc[5:]

    # MFE/MAE at 60min from 9:35 (bars 5-65)
    for window, window_end in [(60, 65), (30, 35), (90, 95)]:
        win_bars = rth.iloc[5:min(5+window, len(rth))]
        if len(win_bars) == 0:
            continue

        if gap_dir == 'up':
            mfe = (win_bars['high'].max() - close_935) / adr
            mae = (close_935 - win_bars['low'].min()) / adr
        else:
            mfe = (close_935 - win_bars['low'].min()) / adr
            mae = (win_bars['high'].max() - close_935) / adr

        mfe_records.append({
            'ticker': ticker, 'date': date_str, 'gap_direction': gap_dir,
            'window': window, 'mfe': mfe, 'mae': mae,
            'rvol_5': row.get('rvol_5', np.nan),
            'pm_rth5': row.get('pm_rth5', np.nan),
        })

mfe_df = pd.DataFrame(mfe_records)

if len(mfe_df) > 0:
    for gap_dir in ['up', 'down']:
        w(f"\n--- Gap{gap_dir.title()} Combo E MFE/MAE from 9:35 ---")
        for window in [30, 60, 90]:
            grp = mfe_df[(mfe_df['gap_direction'] == gap_dir) & (mfe_df['window'] == window)]
            if len(grp) == 0: continue
            w(f"  {window}min: N={len(grp)}, MFE median={grp['mfe'].median():.4f}, mean={grp['mfe'].mean():.4f}, "
              f"MAE median={grp['mae'].median():.4f}, mean={grp['mae'].mean():.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════
with open('results/d7_combo_e_new.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))
print(f"Saved results/d7_combo_e_new.txt", file=sys.stderr)

if len(mfe_df) > 0:
    mfe_df.to_parquet('results/d7_combo_e_mfe.parquet', index=False)

print('\n'.join(out))
