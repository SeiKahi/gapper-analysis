###############################################################################
# METADATA EXPLORATION v3 — Durchlauf 3.0
#
# Analysiert die 63 Metadata-Spalten auf Patterns und potentielle Edges.
# KEINE Datei-Ladung noetig — alles in metadata_master.parquet.
# Nur Haelfte 1 (2021-02-21 bis 2023-12-31).
#
# Run: .\gapper_env\Scripts\python.exe scripts\metadata_exploration_v3.py
###############################################################################

import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

METADATA_PATH = r'data\metadata\metadata_master.parquet'
RESULTS_DIR = r'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load and filter
meta = pd.read_parquet(METADATA_PATH)
meta['date'] = pd.to_datetime(meta['date'])
meta = meta[(meta['date'] >= '2021-02-21') & (meta['date'] <= '2023-12-31')].copy()
meta['year'] = meta['date'].dt.year

# Train/Val split
meta['split'] = np.where(meta['date'] <= '2022-12-31', 'train', 'val')

print(f"Loaded: {len(meta)} gappers, Train: {(meta['split']=='train').sum()}, Val: {(meta['split']=='val').sum()}", file=sys.stderr)

output_path = os.path.join(RESULTS_DIR, 'metadata_exploration_v3.txt')
with open(output_path, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("METADATA EXPLORATION v3 — Durchlauf 3.0\n")
    f.write("63 Spalten, nur Haelfte 1 (2021-2023), N=" + str(len(meta)) + "\n")
    f.write("=" * 80 + "\n\n")

    # ================================================================
    # 1. BASIC STATS
    # ================================================================
    f.write("=" * 80 + "\n")
    f.write("1. BASIC STATS\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Total: {len(meta)}\n")
    f.write(f"GapUp: {(meta['gap_direction']=='up').sum()}, GapDn: {(meta['gap_direction']=='down').sum()}\n")
    f.write(f"Train (2021-2022): {(meta['split']=='train').sum()}\n")
    f.write(f"Val (2023): {(meta['split']=='val').sum()}\n\n")

    # Gap fill stats
    if 'gap_filled' in meta.columns:
        f.write(f"Gap Filled: {meta['gap_filled'].mean()*100:.1f}%\n")
        for d in ['up', 'down']:
            sub = meta[meta['gap_direction'] == d]
            f.write(f"  Gap{d.capitalize()} Fill Rate: {sub['gap_filled'].mean()*100:.1f}% (N={len(sub)})\n")
    f.write("\n")

    # ================================================================
    # 2. OPENING DRIVE ANALYSIS
    # ================================================================
    f.write("=" * 80 + "\n")
    f.write("2. OPENING DRIVE ANALYSIS\n")
    f.write("=" * 80 + "\n\n")

    if 'opening_drive_direction' in meta.columns and 'opening_drive_atr' in meta.columns:
        f.write("Opening Drive = first N minutes candle direction & strength\n\n")

        # Opening Drive direction vs gap direction
        for split in ['train', 'val']:
            s = meta[meta['split'] == split]
            f.write(f"--- {split.upper()} ---\n")

            for gap_dir in ['up', 'down']:
                g = s[s['gap_direction'] == gap_dir]
                od_vals = g['opening_drive_direction'].value_counts()
                f.write(f"  Gap{gap_dir.capitalize()} Opening Drive:\n")
                for od, cnt in od_vals.items():
                    sub = g[g['opening_drive_direction'] == od]
                    mean_ret = sub['close_vs_open_adr'].mean() if 'close_vs_open_adr' in sub.columns else np.nan
                    gap_fill = sub['gap_filled'].mean() * 100 if 'gap_filled' in sub.columns else np.nan
                    mean_ext = sub['max_extension_adr'].mean() if 'max_extension_adr' in sub.columns else np.nan
                    f.write(f"    {od}: N={cnt} ({cnt/len(g)*100:.0f}%), CloseVsOpen={mean_ret:.3f} ADR, "
                            f"GapFill={gap_fill:.1f}%, MaxExt={mean_ext:.3f} ADR\n")
            f.write("\n")

        # Opening drive strength
        f.write("Opening Drive Strength (opening_drive_atr):\n")
        od_atr = meta['opening_drive_atr'].dropna()
        f.write(f"  P10={od_atr.quantile(0.1):.3f}, P25={od_atr.quantile(0.25):.3f}, "
                f"P50={od_atr.median():.3f}, P75={od_atr.quantile(0.75):.3f}, "
                f"P90={od_atr.quantile(0.9):.3f}\n\n")

    # ================================================================
    # 3. GAP FILL ANALYSIS
    # ================================================================
    f.write("=" * 80 + "\n")
    f.write("3. GAP FILL ANALYSIS\n")
    f.write("=" * 80 + "\n\n")

    if 'gap_filled' in meta.columns:
        for split in ['train', 'val']:
            s = meta[meta['split'] == split]
            f.write(f"--- {split.upper()} ---\n")

            # By gap direction
            for gap_dir in ['up', 'down']:
                g = s[s['gap_direction'] == gap_dir]
                fill_rate = g['gap_filled'].mean() * 100
                fill_times = g[g['gap_filled'] == True]['gap_fill_time_minutes']
                f.write(f"  Gap{gap_dir.capitalize()}: Fill Rate={fill_rate:.1f}% (N={len(g)})\n")
                if len(fill_times) > 0:
                    f.write(f"    Fill Time: P25={fill_times.quantile(0.25):.0f}m, "
                            f"Med={fill_times.median():.0f}m, P75={fill_times.quantile(0.75):.0f}m\n")

            # By gap size
            f.write("\n  By Gap Size (in ADR):\n")
            for lo, hi, label in [(0, 0.5, '<0.5'), (0.5, 1.0, '0.5-1.0'), (1.0, 1.5, '1.0-1.5'),
                                   (1.5, 2.0, '1.5-2.0'), (2.0, 3.0, '2.0-3.0'), (3.0, 99, '>3.0')]:
                sub = s[(s['gap_size_in_adr'] >= lo) & (s['gap_size_in_adr'] < hi)]
                if len(sub) < 20:
                    continue
                fill = sub['gap_filled'].mean() * 100
                ret = sub['close_vs_open_adr'].mean() if 'close_vs_open_adr' in sub.columns else np.nan
                f.write(f"    {label}: N={len(sub)}, FillRate={fill:.1f}%, CloseVsOpen={ret:.3f} ADR\n")
            f.write("\n")

    # ================================================================
    # 4. HOD/LOD TIMING
    # ================================================================
    f.write("=" * 80 + "\n")
    f.write("4. HOD/LOD TIMING (minutes since open)\n")
    f.write("=" * 80 + "\n\n")

    if 'hod_time' in meta.columns and 'lod_time' in meta.columns:
        for split in ['train', 'val']:
            s = meta[meta['split'] == split]
            f.write(f"--- {split.upper()} ---\n")

            for gap_dir in ['up', 'down']:
                g = s[s['gap_direction'] == gap_dir]
                hod = g['hod_time'].dropna()
                lod = g['lod_time'].dropna()
                f.write(f"  Gap{gap_dir.capitalize()} (N={len(g)}):\n")
                f.write(f"    HOD: P25={hod.quantile(0.25):.0f}m, Med={hod.median():.0f}m, "
                        f"P75={hod.quantile(0.75):.0f}m, Mean={hod.mean():.0f}m\n")
                f.write(f"    LOD: P25={lod.quantile(0.25):.0f}m, Med={lod.median():.0f}m, "
                        f"P75={lod.quantile(0.75):.0f}m, Mean={lod.mean():.0f}m\n")

                # HOD in first 15min?
                hod_early = (hod <= 15).mean() * 100
                lod_early = (lod <= 15).mean() * 100
                hod_first_hour = (hod <= 60).mean() * 100
                lod_first_hour = (lod <= 60).mean() * 100
                f.write(f"    HOD in first 15m: {hod_early:.1f}%, first 60m: {hod_first_hour:.1f}%\n")
                f.write(f"    LOD in first 15m: {lod_early:.1f}%, first 60m: {lod_first_hour:.1f}%\n")
            f.write("\n")

    # ================================================================
    # 5. REVERSAL ANALYSIS
    # ================================================================
    f.write("=" * 80 + "\n")
    f.write("5. REVERSAL ANALYSIS\n")
    f.write("=" * 80 + "\n\n")

    if 'reversal_occurred' in meta.columns:
        for split in ['train', 'val']:
            s = meta[meta['split'] == split]
            f.write(f"--- {split.upper()} ---\n")

            for gap_dir in ['up', 'down']:
                g = s[s['gap_direction'] == gap_dir]
                rev_rate = g['reversal_occurred'].mean() * 100
                rev_times = g[g['reversal_occurred'] == True]['reversal_time_minutes']
                f.write(f"  Gap{gap_dir.capitalize()}: Reversal Rate={rev_rate:.1f}% (N={len(g)})\n")
                if len(rev_times) > 0:
                    f.write(f"    Reversal Time: Med={rev_times.median():.0f}m, Mean={rev_times.mean():.0f}m\n")

                # Daily return for reversal vs non-reversal
                rev = g[g['reversal_occurred'] == True]
                norev = g[g['reversal_occurred'] == False]
                if 'close_vs_open_adr' in g.columns:
                    f.write(f"    Reversal CloseVsOpen: {rev['close_vs_open_adr'].mean():.3f} ADR\n")
                    f.write(f"    No Reversal CloseVsOpen: {norev['close_vs_open_adr'].mean():.3f} ADR\n")
            f.write("\n")

    # ================================================================
    # 6. MAX EXTENSION & ADVERSE (MFE/MAE from metadata)
    # ================================================================
    f.write("=" * 80 + "\n")
    f.write("6. MAX EXTENSION & ADVERSE (already in metadata)\n")
    f.write("=" * 80 + "\n\n")

    if 'max_extension_adr' in meta.columns:
        for split in ['train', 'val']:
            s = meta[meta['split'] == split]
            f.write(f"--- {split.upper()} ---\n")

            for gap_dir in ['up', 'down']:
                g = s[s['gap_direction'] == gap_dir]
                ext = g['max_extension_adr'].dropna()
                adv = g['max_adverse_adr'].dropna()
                f.write(f"  Gap{gap_dir.capitalize()} (N={len(g)}):\n")
                f.write(f"    MaxExtension: Med={ext.median():.3f}, Mean={ext.mean():.3f}, "
                        f"P75={ext.quantile(0.75):.3f}, P90={ext.quantile(0.9):.3f}\n")
                f.write(f"    MaxAdverse:   Med={adv.median():.3f}, Mean={adv.mean():.3f}, "
                        f"P75={adv.quantile(0.75):.3f}, P90={adv.quantile(0.9):.3f}\n")
                # Close vs open
                cvo = g['close_vs_open_adr'].dropna()
                f.write(f"    CloseVsOpen:  Med={cvo.median():.3f}, Mean={cvo.mean():.3f}\n")
            f.write("\n")

    # ================================================================
    # 7. PREDICTIVE FEATURES: What predicts daily return?
    # ================================================================
    f.write("=" * 80 + "\n")
    f.write("7. FEATURE CORRELATIONS with close_vs_open_adr\n")
    f.write("=" * 80 + "\n\n")

    target = 'close_vs_open_adr'
    if target in meta.columns:
        features = ['gap_size_in_adr', 'adr_10', 'rvol_at_time_30min', 'prior_return_5d',
                     'prior_return_10d', 'prior_return_20d', 'dist_from_20sma',
                     'dist_from_52w_high', 'rsi_14_prev', 'premarket_volume',
                     'gap_vs_prior_range', 'opening_drive_atr',
                     'volume_first_30min', 'spy_return_day']

        for split in ['train', 'val']:
            s = meta[meta['split'] == split]
            f.write(f"--- {split.upper()} ---\n")

            for gap_dir in ['up', 'down']:
                g = s[s['gap_direction'] == gap_dir]
                f.write(f"  Gap{gap_dir.capitalize()} (N={len(g)}):\n")
                f.write(f"  {'Feature':<28} {'Corr':>8} {'P25_ret':>8} {'P75_ret':>8}\n")
                f.write("  " + "-" * 50 + "\n")

                for feat in features:
                    if feat not in g.columns:
                        continue
                    valid = g[[feat, target]].dropna()
                    if len(valid) < 50:
                        continue
                    # Skip non-numeric
                    if not pd.api.types.is_numeric_dtype(valid[feat]):
                        continue
                    corr = valid[feat].corr(valid[target])

                    # Quartile split
                    q25 = valid[feat].quantile(0.25)
                    q75 = valid[feat].quantile(0.75)
                    low_q = valid[valid[feat] <= q25][target].mean()
                    high_q = valid[valid[feat] >= q75][target].mean()

                    f.write(f"  {feat:<28} {corr:>8.3f} {low_q:>8.3f} {high_q:>8.3f}\n")
                f.write("\n")

    # ================================================================
    # 8. OPENING DRIVE → REST OF DAY (KEY HYPOTHESIS)
    # ================================================================
    f.write("=" * 80 + "\n")
    f.write("8. OPENING DRIVE → REST OF DAY\n")
    f.write("=" * 80 + "\n\n")

    if 'opening_drive_direction' in meta.columns and 'close_vs_open_adr' in meta.columns:
        # Does opening drive predict rest of day (from 9:45 onward)?
        for split in ['train', 'val']:
            s = meta[meta['split'] == split]
            f.write(f"--- {split.upper()} ---\n")

            for gap_dir in ['up', 'down']:
                g = s[s['gap_direction'] == gap_dir]
                f.write(f"\n  Gap{gap_dir.capitalize()}:\n")

                # Opening drive aligns with gap = continuation
                # Opening drive opposes gap = reversal
                for od in g['opening_drive_direction'].unique():
                    sub = g[g['opening_drive_direction'] == od]
                    if len(sub) < 20:
                        continue

                    cvo = sub['close_vs_open_adr'].mean()
                    ext = sub['max_extension_adr'].mean()
                    adv = sub['max_adverse_adr'].mean()
                    fill = sub['gap_filled'].mean() * 100

                    # Classify: does OD align with gap?
                    if gap_dir == 'up':
                        aligns = 'CONTINUATION' if od == 'up' else 'REVERSAL' if od == 'down' else 'FLAT'
                    else:
                        aligns = 'CONTINUATION' if od == 'down' else 'REVERSAL' if od == 'up' else 'FLAT'

                    f.write(f"    OD={od} ({aligns}): N={len(sub)}, CloseVsOpen={cvo:.3f}, "
                            f"MaxExt={ext:.3f}, MaxAdv={adv:.3f}, GapFill={fill:.1f}%\n")
            f.write("\n")

    # ================================================================
    # 9. PRE-MARKET VOLUME AS PREDICTOR
    # ================================================================
    f.write("=" * 80 + "\n")
    f.write("9. PRE-MARKET VOLUME AS PREDICTOR\n")
    f.write("=" * 80 + "\n\n")

    if 'premarket_volume' in meta.columns:
        for split in ['train', 'val']:
            s = meta[meta['split'] == split]
            f.write(f"--- {split.upper()} ---\n")

            pm_vol = s['premarket_volume'].dropna()
            f.write(f"  PM Volume: P25={pm_vol.quantile(0.25):.0f}, Med={pm_vol.median():.0f}, "
                    f"P75={pm_vol.quantile(0.75):.0f}, P90={pm_vol.quantile(0.9):.0f}\n")

            # Quartile buckets
            for gap_dir in ['up', 'down']:
                g = s[s['gap_direction'] == gap_dir].dropna(subset=['premarket_volume'])
                if len(g) < 50:
                    continue

                q_labels = ['Q1_low', 'Q2', 'Q3', 'Q4_high']
                try:
                    g['pm_q'] = pd.qcut(g['premarket_volume'], 4, labels=q_labels, duplicates='drop')
                except:
                    continue

                f.write(f"  Gap{gap_dir.capitalize()} by PM Volume Quartile:\n")
                for q in q_labels:
                    sub = g[g['pm_q'] == q]
                    if len(sub) < 10:
                        continue
                    cvo = sub['close_vs_open_adr'].mean()
                    fill = sub['gap_filled'].mean() * 100
                    ext = sub['max_extension_adr'].mean()
                    f.write(f"    {q}: N={len(sub)}, CloseVsOpen={cvo:.3f}, GapFill={fill:.1f}%, MaxExt={ext:.3f}\n")
            f.write("\n")

    # ================================================================
    # 10. DAY OF WEEK EFFECT
    # ================================================================
    f.write("=" * 80 + "\n")
    f.write("10. DAY OF WEEK EFFECT\n")
    f.write("=" * 80 + "\n\n")

    if 'day_of_week' in meta.columns:
        for split in ['train', 'val']:
            s = meta[meta['split'] == split]
            f.write(f"--- {split.upper()} ---\n")
            f.write(f"  {'Day':<12} {'N':>6} {'CloseVsOpen':>12} {'GapFill%':>9} {'MaxExt':>8}\n")
            f.write("  " + "-" * 50 + "\n")

            for dow in sorted(s['day_of_week'].dropna().unique()):
                sub = s[s['day_of_week'] == dow]
                cvo = sub['close_vs_open_adr'].mean()
                fill = sub['gap_filled'].mean() * 100
                ext = sub['max_extension_adr'].mean()
                f.write(f"  {dow:<12} {len(sub):>6} {cvo:>12.3f} {fill:>8.1f}% {ext:>8.3f}\n")
            f.write("\n")

    # ================================================================
    # 11. RSI + PRIOR MOMENTUM → GAP DAY RETURN
    # ================================================================
    f.write("=" * 80 + "\n")
    f.write("11. RSI + PRIOR MOMENTUM\n")
    f.write("=" * 80 + "\n\n")

    if 'rsi_14_prev' in meta.columns:
        for split in ['train', 'val']:
            s = meta[meta['split'] == split]
            f.write(f"--- {split.upper()} ---\n")

            for gap_dir in ['up', 'down']:
                g = s[s['gap_direction'] == gap_dir].dropna(subset=['rsi_14_prev'])
                if len(g) < 50:
                    continue

                bins = [(0, 30, 'Oversold<30'), (30, 50, 'Weak30-50'),
                        (50, 70, 'Neutral50-70'), (70, 100, 'Overbought>70')]
                f.write(f"  Gap{gap_dir.capitalize()} by RSI_14:\n")
                for lo, hi, label in bins:
                    sub = g[(g['rsi_14_prev'] >= lo) & (g['rsi_14_prev'] < hi)]
                    if len(sub) < 15:
                        continue
                    cvo = sub['close_vs_open_adr'].mean()
                    fill = sub['gap_filled'].mean() * 100
                    f.write(f"    {label}: N={len(sub)}, CloseVsOpen={cvo:.3f}, GapFill={fill:.1f}%\n")
            f.write("\n")

    # ================================================================
    # 12. CLOSE LOCATION (where does price close relative to day range)
    # ================================================================
    f.write("=" * 80 + "\n")
    f.write("12. CLOSE LOCATION + VWAP HELD\n")
    f.write("=" * 80 + "\n\n")

    if 'close_location' in meta.columns:
        for split in ['train', 'val']:
            s = meta[meta['split'] == split]
            f.write(f"--- {split.upper()} ---\n")

            cl = s['close_location'].dropna()
            if pd.api.types.is_numeric_dtype(cl):
                f.write(f"  Close Location: P10={cl.quantile(0.1):.2f}, P25={cl.quantile(0.25):.2f}, "
                        f"Med={cl.median():.2f}, P75={cl.quantile(0.75):.2f}, P90={cl.quantile(0.9):.2f}\n")
            else:
                f.write(f"  Close Location (categorical):\n")
                for val_name, cnt in cl.value_counts().head(10).items():
                    sub = s[s['close_location'] == val_name]
                    cvo = sub['close_vs_open_adr'].mean() if 'close_vs_open_adr' in sub.columns else np.nan
                    f.write(f"    {val_name}: N={cnt}, CloseVsOpen={cvo:.3f}\n")

            for gap_dir in ['up', 'down']:
                g = s[s['gap_direction'] == gap_dir]
                cl_g = g['close_location'].dropna()
                if pd.api.types.is_numeric_dtype(cl_g):
                    f.write(f"  Gap{gap_dir.capitalize()}: Med CloseLocation={cl_g.median():.2f}\n")
                else:
                    f.write(f"  Gap{gap_dir.capitalize()} CloseLocation dist: {cl_g.value_counts().head(5).to_dict()}\n")
            f.write("\n")

    if 'vwap_held' in meta.columns:
        for split in ['train', 'val']:
            s = meta[meta['split'] == split]
            f.write(f"--- {split.upper()} VWAP Held ---\n")
            for gap_dir in ['up', 'down']:
                g = s[s['gap_direction'] == gap_dir]
                held = g['vwap_held'].mean() * 100 if g['vwap_held'].notna().sum() > 0 else np.nan
                # VWAP held = closed on same side of VWAP as gap direction
                f.write(f"  Gap{gap_dir.capitalize()}: VWAP Held={held:.1f}%\n")
            f.write("\n")

    # ================================================================
    # 13. SPY RETURN DAY INTERACTION
    # ================================================================
    f.write("=" * 80 + "\n")
    f.write("13. SPY RETURN INTERACTION\n")
    f.write("=" * 80 + "\n\n")

    if 'spy_return_day' in meta.columns:
        for split in ['train', 'val']:
            s = meta[meta['split'] == split]
            f.write(f"--- {split.upper()} ---\n")

            for gap_dir in ['up', 'down']:
                g = s[s['gap_direction'] == gap_dir].dropna(subset=['spy_return_day'])
                spy_up = g[g['spy_return_day'] > 0]
                spy_dn = g[g['spy_return_day'] <= 0]

                f.write(f"  Gap{gap_dir.capitalize()}:\n")
                f.write(f"    SPY Up:   N={len(spy_up)}, CloseVsOpen={spy_up['close_vs_open_adr'].mean():.3f}, "
                        f"GapFill={spy_up['gap_filled'].mean()*100:.1f}%\n")
                f.write(f"    SPY Down: N={len(spy_dn)}, CloseVsOpen={spy_dn['close_vs_open_adr'].mean():.3f}, "
                        f"GapFill={spy_dn['gap_filled'].mean()*100:.1f}%\n")
            f.write("\n")

    # ================================================================
    # 14. BOOTSTRAP CI FOR KEY FINDINGS
    # ================================================================
    f.write("=" * 80 + "\n")
    f.write("14. BOOTSTRAP CIs fuer zentrale Metriken (Val)\n")
    f.write("=" * 80 + "\n\n")

    np.random.seed(42)
    val = meta[meta['split'] == 'val']

    tests = {
        'GapUp CloseVsOpen': val[val['gap_direction']=='up']['close_vs_open_adr'].dropna().values,
        'GapDn CloseVsOpen': val[val['gap_direction']=='down']['close_vs_open_adr'].dropna().values,
        'All CloseVsOpen': val['close_vs_open_adr'].dropna().values,
    }

    for label, vals in tests.items():
        if len(vals) < 30:
            continue
        boots = [np.mean(np.random.choice(vals, size=len(vals), replace=True)) for _ in range(5000)]
        ci = np.percentile(boots, [2.5, 97.5])
        p = np.mean([b <= 0 for b in boots])
        f.write(f"  {label}: mean={np.mean(vals):.4f}, 95% CI [{ci[0]:.4f}, {ci[1]:.4f}], P(<=0)={p:.4f}\n")
    f.write("\n")

    # ================================================================
    # 15. COMBINED FILTER SEARCH
    # ================================================================
    f.write("=" * 80 + "\n")
    f.write("15. COMBINED FILTER SEARCH — Staerkste Subgruppen (Val)\n")
    f.write("=" * 80 + "\n\n")

    val = meta[meta['split'] == 'val']
    results = []

    for gap_dir in ['up', 'down']:
        g = val[val['gap_direction'] == gap_dir]

        for gap_lo, gap_hi, gap_label in [(0, 1.0, 'SmGap'), (1.0, 2.0, 'MedGap'), (2.0, 99, 'LgGap'), (0, 99, 'AnyGap')]:
            for rsi_lo, rsi_hi, rsi_label in [(0, 40, 'RSI<40'), (40, 60, 'RSI40-60'), (60, 100, 'RSI>60'), (0, 100, 'AnyRSI')]:
                for od in list(g['opening_drive_direction'].unique()) + ['Any']:
                    mask = (
                        (g['gap_size_in_adr'] >= gap_lo) & (g['gap_size_in_adr'] < gap_hi) &
                        (g['rsi_14_prev'] >= rsi_lo) & (g['rsi_14_prev'] < rsi_hi)
                    )
                    if od != 'Any':
                        mask = mask & (g['opening_drive_direction'] == od)

                    sub = g[mask]
                    if len(sub) < 30:
                        continue

                    cvo = sub['close_vs_open_adr'].mean()
                    fill = sub['gap_filled'].mean() * 100
                    ext = sub['max_extension_adr'].mean()

                    results.append({
                        'gap_dir': gap_dir,
                        'gap_size': gap_label,
                        'rsi': rsi_label,
                        'od': od,
                        'n': len(sub),
                        'close_vs_open': cvo,
                        'gap_fill': fill,
                        'max_ext': ext,
                    })

    res_df = pd.DataFrame(results)

    # Best positive (for potential longs on GapDn or shorts on GapUp)
    f.write("--- GapDn: Best positive CloseVsOpen (= bullish for longs) ---\n")
    gd = res_df[res_df['gap_dir'] == 'down'].sort_values('close_vs_open', ascending=False).head(15)
    f.write(f"{'GapSize':<10} {'RSI':<10} {'OD':<8} {'N':>5} {'CloseVsOpen':>11} {'GapFill':>8} {'MaxExt':>7}\n")
    f.write("-" * 65 + "\n")
    for _, r in gd.iterrows():
        f.write(f"{r['gap_size']:<10} {r['rsi']:<10} {r['od']:<8} {r['n']:>5} {r['close_vs_open']:>11.3f} "
                f"{r['gap_fill']:>7.1f}% {r['max_ext']:>7.3f}\n")
    f.write("\n")

    f.write("--- GapUp: Best negative CloseVsOpen (= bearish for shorts) ---\n")
    gu = res_df[res_df['gap_dir'] == 'up'].sort_values('close_vs_open', ascending=True).head(15)
    for _, r in gu.iterrows():
        f.write(f"{r['gap_size']:<10} {r['rsi']:<10} {r['od']:<8} {r['n']:>5} {r['close_vs_open']:>11.3f} "
                f"{r['gap_fill']:>7.1f}% {r['max_ext']:>7.3f}\n")
    f.write("\n")

    f.write("=" * 80 + "\n")
    f.write("ENDE METADATA EXPLORATION v3\n")
    f.write("=" * 80 + "\n")

print(f"Ergebnisse: {output_path}", file=sys.stderr)
