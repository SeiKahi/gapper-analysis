"""
DIP & GO Analysis — LOD/HOD-Timing als Setup-Definition
=========================================================
Finde GapUp-Aktien deren LOD zwischen 9:45-10:20 liegt ("Dip & Go"),
vergleiche mit allen anderen, und analysiere welche Parameter die
Gruppen trennen. Dann OOS-Validation + Shrinkage.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from sklearn.metrics import roc_auc_score
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
META_PATH = os.path.join(DATA_DIR, 'metadata', 'metadata_v8_5.parquet')
RAW_1MIN_DIR = os.path.join(DATA_DIR, 'raw_1min')
OUT_DIR = os.path.join(BASE_DIR, 'results', 'dipgo_analysis')
os.makedirs(OUT_DIR, exist_ok=True)

IS_CUTOFF = '2024-01-01'

# Time windows to test (minutes after 9:30)
WINDOWS = {
    'very_tight': (20, 45),   # 9:50-10:15
    'tight':      (10, 40),   # 9:40-10:10
    'standard':   (15, 50),   # 9:45-10:20
    'wide':       (15, 60),   # 9:45-10:30
}
PRIMARY_WINDOW = 'standard'

# Chart style
BG_COLOR = '#1a1a2e'
GRID_COLOR = '#2a2a4e'
TEXT_COLOR = '#e0e0e0'
ACCENT1 = '#00d4ff'  # cyan
ACCENT2 = '#ff6b6b'  # red
ACCENT3 = '#51cf66'  # green
ACCENT4 = '#ffd43b'  # yellow
ACCENT5 = '#cc5de8'  # purple

plt.style.use('dark_background')
plt.rcParams.update({
    'figure.facecolor': BG_COLOR,
    'axes.facecolor': BG_COLOR,
    'axes.edgecolor': GRID_COLOR,
    'axes.grid': True,
    'grid.color': GRID_COLOR,
    'grid.alpha': 0.3,
    'text.color': TEXT_COLOR,
    'axes.labelcolor': TEXT_COLOR,
    'xtick.color': TEXT_COLOR,
    'ytick.color': TEXT_COLOR,
    'font.size': 10,
})

# ============================================================
# REPORT
# ============================================================
report_lines = []

def rpt(text=''):
    report_lines.append(text)
    print(text)

def rpt_header(title):
    rpt('')
    rpt('=' * 70)
    rpt(f'  {title}')
    rpt('=' * 70)

def rpt_subheader(title):
    rpt('')
    rpt(f'--- {title} ---')

def save_report():
    path = os.path.join(OUT_DIR, 'dipgo_report.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f'\nReport saved to {path}')


# ============================================================
# LOAD & PREPARE DATA
# ============================================================
def load_data():
    rpt_header('DATA LOADING')
    meta = pd.read_parquet(META_PATH)
    meta['date'] = pd.to_datetime(meta['date'])

    # Filter: od_strength > 0.5, od_direction == with_gap
    df = meta[(meta['od_strength'] > 0.5) & (meta['od_direction'] == 'with_gap')].copy()
    rpt(f'Qualifying trades (OD>0.5, with_gap): {len(df)}')
    rpt(f'  GapUp: {(df["gap_direction"]=="up").sum()}')
    rpt(f'  GapDn: {(df["gap_direction"]=="down").sum()}')

    # IS/OOS split
    df['is_IS'] = df['date'] < IS_CUTOFF
    rpt(f'  IS (< {IS_CUTOFF}): {df["is_IS"].sum()}')
    rpt(f'  OOS (>= {IS_CUTOFF}): {(~df["is_IS"]).sum()}')

    # Compute derived columns
    # lod_time and hod_time are already minutes after 9:30
    # For GapUp: LOD is the key timing event
    # For GapDn: HOD is the key timing event
    df['key_time'] = np.where(df['gap_direction'] == 'up', df['lod_time'], df['hod_time'])
    df['key_price'] = np.where(df['gap_direction'] == 'up', df['rth_low'], df['rth_high'])

    # rest_move: move from LOD/HOD to close (in ADR)
    # GapUp: (close - LOD) / adr_10  (positive = good, stock bounced from LOD)
    # GapDn: (HOD - close) / adr_10  (positive = good, stock dropped from HOD)
    df['rest_move'] = np.where(
        df['gap_direction'] == 'up',
        (df['rth_close'] - df['rth_low']) / df['adr_10'],
        (df['rth_high'] - df['rth_close']) / df['adr_10']
    )

    # lod_depth: how deep the dip was from entry (9:35 close) in ADR
    # GapUp: (close_935 - LOD) / adr_10
    # GapDn: (HOD - close_935) / adr_10
    df['lod_depth'] = np.where(
        df['gap_direction'] == 'up',
        (df['close_935'] - df['rth_low']) / df['adr_10'],
        (df['rth_high'] - df['close_935']) / df['adr_10']
    )

    # move_before_1000: direction of price from 9:35 to 10:00
    # For GapUp: negative = pullback (good for dip&go)
    # For GapDn: positive = bounce (good for dip&go in short direction, meaning we reverse sign)
    df['move_935_to_1000'] = np.where(
        df['gap_direction'] == 'up',
        (df['close_1000'] - df['close_935']) / df['adr_10'],
        (df['close_935'] - df['close_1000']) / df['adr_10']
    )

    # lod_to_1000: recovery from LOD to 10:00
    # GapUp: (close_1000 - LOD) / adr_10
    # GapDn: (HOD - close_1000) / adr_10
    df['lod_to_1000'] = np.where(
        df['gap_direction'] == 'up',
        (df['close_1000'] - df['rth_low']) / df['adr_10'],
        (df['rth_high'] - df['close_1000']) / df['adr_10']
    )

    # move_after_1000: from 10:00 to close in ADR
    df['move_after_1000'] = np.where(
        df['gap_direction'] == 'up',
        (df['rth_close'] - df['close_1000']) / df['adr_10'],
        (df['close_1000'] - df['rth_close']) / df['adr_10']
    )

    # drift_30 = move from 9:35 to 10:00 (already available as rest_drift_1000 adjusted)
    # Actually let's use the raw move
    df['drift_30'] = (df['close_1000'] - df['close_935']) / df['adr_10']

    return df


# ============================================================
# LOAD 1-MIN BARS FOR PEAK-AFTER-LOD & SL ANALYSIS
# ============================================================
def load_1min_bars(ticker, date_str):
    path = os.path.join(RAW_1MIN_DIR, ticker, f'{date_str}.parquet')
    if not os.path.exists(path):
        return None
    bars = pd.read_parquet(path)
    # Filter RTH only (9:30-16:00)
    rth = bars[bars['session'] == 'rth'].copy()
    if len(rth) == 0:
        # Try filtering by time
        rth = bars[(bars['time_et'] >= '09:30') & (bars['time_et'] < '16:00')].copy()
    return rth


def compute_peak_after_key(df):
    """For each trade, compute the peak price AFTER the key_time (LOD for GapUp, HOD for GapDn)."""
    rpt_subheader('Computing peak after LOD/HOD from 1-min bars...')
    peaks = []
    sl_025_survive = []  # SL = 0.25 ADR below entry at 10:00
    sl_015_lod_survive = []  # SL = 0.15 ADR below LOD
    sl_lod_retrigger = []  # SL = LOD price itself

    count = 0
    for idx, row in df.iterrows():
        ticker = row['ticker']
        date_str = str(row['date'].date())
        gap_dir = row['gap_direction']
        key_time_min = int(row['key_time'])
        adr = row['adr_10']
        close_1000 = row['close_1000']

        bars = load_1min_bars(ticker, date_str)
        if bars is None or len(bars) == 0:
            peaks.append(np.nan)
            sl_025_survive.append(np.nan)
            sl_015_lod_survive.append(np.nan)
            sl_lod_retrigger.append(np.nan)
            continue

        # Parse time to minutes after 9:30
        try:
            bars = bars.copy()
            time_parts = bars['time_et'].str.split(':')
            bars['minutes_after_930'] = time_parts.str[0].astype(int) * 60 + time_parts.str[1].astype(int) - 570  # 9*60+30=570
            bars = bars[bars['minutes_after_930'] >= 0].copy()
        except Exception:
            peaks.append(np.nan)
            sl_025_survive.append(np.nan)
            sl_015_lod_survive.append(np.nan)
            sl_lod_retrigger.append(np.nan)
            continue

        # Bars after key_time
        after_key = bars[bars['minutes_after_930'] > key_time_min]

        if gap_dir == 'up':
            # Peak = highest high after LOD
            if len(after_key) > 0:
                peak_price = after_key['high'].max()
                peak_adr = (peak_price - row['rth_low']) / adr if adr > 0 else np.nan
            else:
                peak_adr = 0
            peaks.append(peak_adr)

            # SL analysis (entry at 10:00, long)
            bars_after_1000 = bars[bars['minutes_after_930'] >= 30]  # 30 min = 10:00
            if len(bars_after_1000) > 0 and adr > 0:
                sl_025_level = close_1000 - 0.25 * adr
                sl_015_level = row['rth_low'] - 0.15 * adr
                sl_lod_level = row['rth_low']
                sl_025_survive.append(int(bars_after_1000['low'].min() >= sl_025_level))
                sl_015_lod_survive.append(int(bars_after_1000['low'].min() >= sl_015_level))
                sl_lod_retrigger.append(int(bars_after_1000['low'].min() < sl_lod_level))
            else:
                sl_025_survive.append(np.nan)
                sl_015_lod_survive.append(np.nan)
                sl_lod_retrigger.append(np.nan)
        else:
            # GapDn: Peak = lowest low after HOD (in short direction)
            if len(after_key) > 0:
                peak_price = after_key['low'].min()
                peak_adr = (row['rth_high'] - peak_price) / adr if adr > 0 else np.nan
            else:
                peak_adr = 0
            peaks.append(peak_adr)

            # SL analysis (entry at 10:00, short)
            bars_after_1000 = bars[bars['minutes_after_930'] >= 30]
            if len(bars_after_1000) > 0 and adr > 0:
                sl_025_level = close_1000 + 0.25 * adr
                sl_015_level = row['rth_high'] + 0.15 * adr
                sl_lod_level = row['rth_high']
                sl_025_survive.append(int(bars_after_1000['high'].max() <= sl_025_level))
                sl_015_lod_survive.append(int(bars_after_1000['high'].max() <= sl_015_level))
                sl_lod_retrigger.append(int(bars_after_1000['high'].max() > sl_lod_level))
            else:
                sl_025_survive.append(np.nan)
                sl_015_lod_survive.append(np.nan)
                sl_lod_retrigger.append(np.nan)

        count += 1
        if count % 200 == 0:
            print(f'  Processed {count}/{len(df)} trades...')

    df['peak_after_key_adr'] = peaks
    df['sl_025_survive'] = sl_025_survive
    df['sl_015_lod_survive'] = sl_015_lod_survive
    df['sl_lod_retrigger'] = sl_lod_retrigger

    rpt(f'  Peak after key computed for {sum(~np.isnan(p) for p in peaks)}/{len(df)} trades')
    return df


# ============================================================
# STEP 1: LOD/HOD TIMING HISTOGRAM
# ============================================================
def step1_timing_histogram(df):
    rpt_header('STEP 1: LOD/HOD TIMING DISTRIBUTION')

    for gap_dir in ['up', 'down']:
        subset = df[df['gap_direction'] == gap_dir]
        label = 'GapUp (LOD time)' if gap_dir == 'up' else 'GapDn (HOD time)'
        col = 'lod_time' if gap_dir == 'up' else 'hod_time'

        rpt_subheader(f'{label}')

        for split_name, mask in [('IS', subset['is_IS']), ('OOS', ~subset['is_IS'])]:
            data = subset.loc[mask, col]
            rpt(f'  {split_name}: N={len(data)}')
            rpt(f'    Mean: {data.mean():.1f} min ({minutes_to_time(data.mean())})')
            rpt(f'    Median: {data.median():.1f} min ({minutes_to_time(data.median())})')

            # Count in windows
            for wname, (lo, hi) in WINDOWS.items():
                in_window = ((data >= lo) & (data <= hi)).sum()
                pct = in_window / len(data) * 100 if len(data) > 0 else 0
                rpt(f'    {wname} ({minutes_to_time(lo)}-{minutes_to_time(hi)}): {in_window} ({pct:.1f}%)')


def minutes_to_time(minutes):
    """Convert minutes after 9:30 to HH:MM string."""
    h = int(9 + (30 + minutes) // 60)
    m = int((30 + minutes) % 60)
    return f'{h:02d}:{m:02d}'


def chart1_timing_histogram(df):
    """Chart 1: LOD/HOD timing histogram."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('CHART 1: LOD/HOD Timing Distribution', fontsize=14, fontweight='bold', color=TEXT_COLOR)

    bins = np.arange(0, 395, 5)  # 5-min bins from 9:30 to 16:00
    bin_labels = [minutes_to_time(b) for b in bins]

    for row, gap_dir in enumerate(['up', 'down']):
        subset = df[df['gap_direction'] == gap_dir]
        col = 'lod_time' if gap_dir == 'up' else 'hod_time'
        label = 'GapUp LOD' if gap_dir == 'up' else 'GapDn HOD'

        for col_idx, (split_name, mask) in enumerate([('IS', subset['is_IS']), ('OOS', ~subset['is_IS'])]):
            ax = axes[row, col_idx]
            data = subset.loc[mask, col].dropna()

            ax.hist(data, bins=bins, color=ACCENT1, alpha=0.7, edgecolor='none')

            # Highlight standard window
            lo, hi = WINDOWS[PRIMARY_WINDOW]
            ax.axvspan(lo, hi, alpha=0.2, color=ACCENT4, label=f'{minutes_to_time(lo)}-{minutes_to_time(hi)}')

            ax.set_title(f'{label} — {split_name} (N={len(data)})', fontsize=11, color=TEXT_COLOR)
            ax.set_xlabel('Time (minutes after 9:30)')
            ax.set_ylabel('Count')

            # X-axis: show time labels every 30 min
            tick_positions = np.arange(0, 391, 30)
            tick_labels = [minutes_to_time(t) for t in tick_positions]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=45, fontsize=8)
            ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'chart1_timing_histogram.png'), dpi=300, bbox_inches='tight')
    plt.close()
    rpt('  Chart 1 saved.')


# ============================================================
# STEP 2: GROUP DEFINITION
# ============================================================
def step2_define_groups(df):
    rpt_header('STEP 2: GROUP DEFINITIONS')

    for wname, (lo, hi) in WINDOWS.items():
        # DIP_GO: key_time in window AND rest_move > 0
        mask_dipgo = (df['key_time'] >= lo) & (df['key_time'] <= hi) & (df['rest_move'] > 0)
        df[f'dipgo_{wname}'] = mask_dipgo

        n_total = len(df)
        n_dipgo = mask_dipgo.sum()
        pct = n_dipgo / n_total * 100

        rpt(f'  Window "{wname}" ({minutes_to_time(lo)}-{minutes_to_time(hi)}):')
        rpt(f'    DIP_GO: {n_dipgo} ({pct:.1f}% of all)')

        for gap_dir in ['up', 'down']:
            sub = df[df['gap_direction'] == gap_dir]
            n_sub = mask_dipgo[sub.index].sum()
            pct_sub = n_sub / len(sub) * 100 if len(sub) > 0 else 0
            rpt(f'      {gap_dir}: {n_sub} ({pct_sub:.1f}%)')

        for split_name, split_mask in [('IS', df['is_IS']), ('OOS', ~df['is_IS'])]:
            n_split = mask_dipgo[split_mask].sum()
            pct_split = n_split / split_mask.sum() * 100 if split_mask.sum() > 0 else 0
            rpt(f'      {split_name}: {n_split} ({pct_split:.1f}%)')

    # Primary label
    lo, hi = WINDOWS[PRIMARY_WINDOW]
    df['is_dipgo'] = (df['key_time'] >= lo) & (df['key_time'] <= hi) & (df['rest_move'] > 0)
    rpt(f'\n  PRIMARY group (standard window): DIP_GO={df["is_dipgo"].sum()}, OTHER={((~df["is_dipgo"])).sum()}')

    return df


# ============================================================
# STEP 3: PARAMETER COMPARISON
# ============================================================
def compute_stats_table(dipgo_data, other_data, param_name):
    """Compute comparison statistics for a single parameter."""
    d = dipgo_data.dropna()
    o = other_data.dropna()

    if len(d) < 5 or len(o) < 5:
        return None

    mean_d = d.mean()
    mean_o = o.mean()
    median_d = d.median()
    median_o = o.median()
    diff = mean_d - mean_o

    # Welch t-test
    try:
        t_stat, p_val = stats.ttest_ind(d, o, equal_var=False)
    except Exception:
        t_stat, p_val = np.nan, np.nan

    # Cohen's d
    pooled_std = np.sqrt((d.var() * (len(d) - 1) + o.var() * (len(o) - 1)) / (len(d) + len(o) - 2))
    cohens_d = diff / pooled_std if pooled_std > 0 else np.nan

    # AUC
    try:
        labels = np.concatenate([np.ones(len(d)), np.zeros(len(o))])
        scores = np.concatenate([d.values, o.values])
        auc = roc_auc_score(labels, scores)
    except Exception:
        auc = np.nan

    return {
        'parameter': param_name,
        'n_dipgo': len(d),
        'n_other': len(o),
        'mean_dipgo': mean_d,
        'mean_other': mean_o,
        'median_dipgo': median_d,
        'median_other': median_o,
        'diff': diff,
        'p_value': p_val,
        'cohens_d': cohens_d,
        'auc': auc,
    }


def step3_parameter_comparison(df, split_name='IS', split_mask=None):
    rpt_header(f'{split_name}: PARAMETER COMPARISON (DIP_GO vs OTHER)')

    if split_mask is None:
        split_mask = df['is_IS'] if split_name == 'IS' else ~df['is_IS']

    sub = df[split_mask].copy()
    dipgo = sub[sub['is_dipgo']]
    other = sub[~sub['is_dipgo']]

    rpt(f'  DIP_GO: {len(dipgo)}, OTHER: {len(other)}')

    # Parameters to compare
    params = {
        # Entry params (known at 9:35)
        'od_strength': 'od_strength',
        'gap_pct': 'gap_pct',
        'adr_10': 'adr_10',
        'rvol_5': 'rvol_5',
        'close_935': 'close_935',
        'gap_size_in_adr': 'gap_size_in_adr',
        'pm_rth5': 'pm_rth5',
        'od_body_pct': 'od_body_pct',
        'od_wick_ratio': 'od_wick_ratio',
        'first_candle_size': 'first_candle_size',
        'prior_return_5d': 'prior_return_5d',
        'prior_return_20d': 'prior_return_20d',
        'dist_from_20sma': 'dist_from_20sma',
        'dist_from_52w_high': 'dist_from_52w_high',
        'rsi_14_prev': 'rsi_14_prev',
        'vix_level': 'vix_level',
        # 10:00 params
        'rvol_30min': 'rvol_at_time_30min',
        'drift_30': 'drift_30',
        'move_935_to_1000': 'move_935_to_1000',
        # Derived
        'lod_depth': 'lod_depth',
        'lod_to_1000': 'lod_to_1000',
    }

    results = []
    for label, col in params.items():
        if col not in sub.columns:
            continue
        row = compute_stats_table(dipgo[col], other[col], label)
        if row is not None:
            results.append(row)

    if not results:
        rpt('  No valid parameters to compare.')
        return pd.DataFrame()

    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values('auc', ascending=False, key=lambda x: abs(x - 0.5))
    res_df = res_df.reset_index(drop=True)

    # Print table
    rpt(f'\n  {"Parameter":<22} {"Mean DG":>8} {"Mean OT":>8} {"Diff":>8} {"p-val":>8} {"Cohen d":>8} {"AUC":>6}')
    rpt(f'  {"-"*22} {"-"*8} {"-"*8} {"-"*8} {"-"*8} {"-"*8} {"-"*6}')
    for _, r in res_df.iterrows():
        p_str = f'{r["p_value"]:.4f}' if r['p_value'] >= 0.0001 else '<.0001'
        rpt(f'  {r["parameter"]:<22} {r["mean_dipgo"]:>8.3f} {r["mean_other"]:>8.3f} {r["diff"]:>+8.3f} {p_str:>8} {r["cohens_d"]:>+8.3f} {r["auc"]:>6.3f}')

    return res_df


# ============================================================
# STEP 4: PROFITABILITY
# ============================================================
def step4_profitability(df, split_name='IS', split_mask=None):
    rpt_header(f'{split_name}: PROFITABILITY OF DIP_GO GROUP')

    if split_mask is None:
        split_mask = df['is_IS'] if split_name == 'IS' else ~df['is_IS']

    sub = df[split_mask].copy()
    dipgo = sub[sub['is_dipgo']]
    other = sub[~sub['is_dipgo']]

    for group_name, group in [('DIP_GO', dipgo), ('OTHER', other), ('ALL', sub)]:
        rpt_subheader(f'{group_name} (N={len(group)})')

        if len(group) == 0:
            rpt('  (empty)')
            continue

        # Rest move from LOD/HOD to Close
        rm = group['rest_move']
        rpt(f'  rest_move (key->Close):')
        rpt(f'    Median: {rm.median():+.3f} ADR')
        rpt(f'    Mean:   {rm.mean():+.3f} ADR')
        rpt(f'    % positive: {(rm > 0).mean()*100:.1f}%')

        # Peak after key
        if 'peak_after_key_adr' in group.columns:
            pk = group['peak_after_key_adr'].dropna()
            if len(pk) > 0:
                rpt(f'  Peak after key (in ADR):')
                rpt(f'    Median: {pk.median():.3f}')
                rpt(f'    Mean:   {pk.mean():.3f}')

        # Move after 10:00
        ma = group['move_after_1000']
        rpt(f'  move_after_1000 (10:00→Close):')
        rpt(f'    Median: {ma.median():+.3f} ADR')
        rpt(f'    Mean:   {ma.mean():+.3f} ADR')
        rpt(f'    % positive: {(ma > 0).mean()*100:.1f}%')

    # SL Analysis for DIP_GO
    rpt_subheader('SL SURVIVAL ANALYSIS (DIP_GO only, entry at 10:00)')
    if len(dipgo) > 0:
        for sl_col, sl_label in [
            ('sl_025_survive', 'SL = 0.25 ADR below entry'),
            ('sl_015_lod_survive', 'SL = 0.15 ADR below LOD'),
        ]:
            if sl_col in dipgo.columns:
                vals = dipgo[sl_col].dropna()
                if len(vals) > 0:
                    survive_pct = vals.mean() * 100
                    rpt(f'  {sl_label}: {survive_pct:.1f}% survive (N={len(vals)})')

        if 'sl_lod_retrigger' in dipgo.columns:
            vals = dipgo['sl_lod_retrigger'].dropna()
            if len(vals) > 0:
                retrig_pct = vals.mean() * 100
                rpt(f'  SL = LOD price itself: {retrig_pct:.1f}% re-triggered (N={len(vals)})')


# ============================================================
# STEP 5: SENSITIVITY ANALYSIS (DECILE)
# ============================================================
def step5_sensitivity(df, res_df_is, split_name='IS', split_mask=None):
    rpt_header(f'{split_name}: SENSITIVITY ANALYSIS (Top-3 Parameters)')

    if split_mask is None:
        split_mask = df['is_IS'] if split_name == 'IS' else ~df['is_IS']

    sub = df[split_mask].copy()

    if res_df_is is None or len(res_df_is) == 0:
        rpt('  No parameters to analyze.')
        return {}

    # Top 3 by AUC distance from 0.5
    res_df_is['auc_dist'] = abs(res_df_is['auc'] - 0.5)
    top3 = res_df_is.nlargest(3, 'auc_dist')['parameter'].tolist()

    # Parameter name -> column mapping
    param_map = {
        'od_strength': 'od_strength', 'gap_pct': 'gap_pct', 'adr_10': 'adr_10',
        'rvol_5': 'rvol_5', 'close_935': 'close_935', 'gap_size_in_adr': 'gap_size_in_adr',
        'pm_rth5': 'pm_rth5', 'od_body_pct': 'od_body_pct', 'od_wick_ratio': 'od_wick_ratio',
        'first_candle_size': 'first_candle_size', 'prior_return_5d': 'prior_return_5d',
        'prior_return_20d': 'prior_return_20d', 'dist_from_20sma': 'dist_from_20sma',
        'dist_from_52w_high': 'dist_from_52w_high', 'rsi_14_prev': 'rsi_14_prev',
        'vix_level': 'vix_level', 'rvol_30min': 'rvol_at_time_30min',
        'drift_30': 'drift_30', 'move_935_to_1000': 'move_935_to_1000',
        'lod_depth': 'lod_depth', 'lod_to_1000': 'lod_to_1000',
    }

    decile_data = {}
    for param in top3:
        col = param_map.get(param, param)
        if col not in sub.columns:
            continue

        vals = sub[col].dropna()
        if len(vals) < 30:
            continue

        rpt_subheader(f'Parameter: {param}')

        # Create deciles
        try:
            sub_clean = sub.dropna(subset=[col])
            sub_clean['decile'] = pd.qcut(sub_clean[col], 10, labels=False, duplicates='drop')
        except Exception:
            continue

        rpt(f'  {"Decile":<8} {"Range":<24} {"N":>5} {"N_DG":>5} {"%DG":>7} {"Med rest_move":>13}')
        rpt(f'  {"-"*8} {"-"*24} {"-"*5} {"-"*5} {"-"*7} {"-"*13}')

        decile_results = []
        for d in sorted(sub_clean['decile'].unique()):
            dec = sub_clean[sub_clean['decile'] == d]
            n = len(dec)
            n_dg = dec['is_dipgo'].sum()
            pct_dg = n_dg / n * 100 if n > 0 else 0
            med_rm = dec['rest_move'].median()
            val_min = dec[col].min()
            val_max = dec[col].max()
            rpt(f'  {d:<8} [{val_min:>9.3f}, {val_max:>9.3f}] {n:>5} {n_dg:>5} {pct_dg:>6.1f}% {med_rm:>+12.3f}')
            decile_results.append({
                'decile': d, 'n': n, 'n_dipgo': n_dg, 'pct_dipgo': pct_dg,
                'median_rest_move': med_rm, 'val_min': val_min, 'val_max': val_max
            })

        decile_data[param] = pd.DataFrame(decile_results)

    return decile_data


# ============================================================
# STEP 6: COMBINATION FILTER
# ============================================================
def step6_combination(df, res_df_is, split_name='IS', split_mask=None):
    rpt_header(f'{split_name}: COMBINATION FILTER (Top-2 Parameters)')

    if split_mask is None:
        split_mask = df['is_IS'] if split_name == 'IS' else ~df['is_IS']

    sub = df[split_mask].copy()

    if res_df_is is None or len(res_df_is) < 2:
        rpt('  Not enough parameters.')
        return None

    res_df_is['auc_dist'] = abs(res_df_is['auc'] - 0.5)
    top2 = res_df_is.nlargest(2, 'auc_dist')['parameter'].tolist()

    param_map = {
        'od_strength': 'od_strength', 'gap_pct': 'gap_pct', 'adr_10': 'adr_10',
        'rvol_5': 'rvol_5', 'close_935': 'close_935', 'gap_size_in_adr': 'gap_size_in_adr',
        'pm_rth5': 'pm_rth5', 'od_body_pct': 'od_body_pct', 'od_wick_ratio': 'od_wick_ratio',
        'first_candle_size': 'first_candle_size', 'prior_return_5d': 'prior_return_5d',
        'prior_return_20d': 'prior_return_20d', 'dist_from_20sma': 'dist_from_20sma',
        'dist_from_52w_high': 'dist_from_52w_high', 'rsi_14_prev': 'rsi_14_prev',
        'vix_level': 'vix_level', 'rvol_30min': 'rvol_at_time_30min',
        'drift_30': 'drift_30', 'move_935_to_1000': 'move_935_to_1000',
        'lod_depth': 'lod_depth', 'lod_to_1000': 'lod_to_1000',
    }

    col1 = param_map.get(top2[0], top2[0])
    col2 = param_map.get(top2[1], top2[1])

    if col1 not in sub.columns or col2 not in sub.columns:
        rpt(f'  Columns {col1} or {col2} not found.')
        return None

    sub_clean = sub.dropna(subset=[col1, col2]).copy()
    try:
        sub_clean['q1'] = pd.qcut(sub_clean[col1], 4, labels=False, duplicates='drop')
        sub_clean['q2'] = pd.qcut(sub_clean[col2], 4, labels=False, duplicates='drop')
    except Exception as e:
        rpt(f'  Could not create quartiles: {e}')
        return None

    rpt(f'  Parameters: {top2[0]} x {top2[1]}')
    rpt(f'  N: {len(sub_clean)}')
    rpt('')

    # 2D heatmap data
    heatmap_pct = np.zeros((4, 4))
    heatmap_n = np.zeros((4, 4), dtype=int)

    rpt(f'  {"":>12} | {"Q0":>12} {"Q1":>12} {"Q2":>12} {"Q3":>12}  <- {top2[1]}')
    rpt(f'  {"-"*12}-+-{"-"*12}-{"-"*12}-{"-"*12}-{"-"*12}')

    for q1 in range(4):
        row_str = f'  {top2[0]} Q{q1:<5} |'
        for q2 in range(4):
            cell = sub_clean[(sub_clean['q1'] == q1) & (sub_clean['q2'] == q2)]
            n = len(cell)
            n_dg = cell['is_dipgo'].sum()
            pct = n_dg / n * 100 if n > 0 else 0
            heatmap_pct[q1, q2] = pct
            heatmap_n[q1, q2] = n
            row_str += f' {pct:>5.1f}%({n:>3})'
        rpt(row_str)

    max_pct = heatmap_pct.max()
    max_pos = np.unravel_index(heatmap_pct.argmax(), heatmap_pct.shape)
    rpt(f'\n  Max DIP_GO rate: {max_pct:.1f}% at ({top2[0]} Q{max_pos[0]}, {top2[1]} Q{max_pos[1]})')
    if max_pct > 40:
        rpt(f'  >> DIP_GO rate > 40% found! Potential filter candidate.')

    return {
        'param1': top2[0], 'param2': top2[1],
        'col1': col1, 'col2': col2,
        'heatmap_pct': heatmap_pct, 'heatmap_n': heatmap_n,
        'data': sub_clean
    }


# ============================================================
# CHARTS
# ============================================================
def chart2_auc_ranking(res_is, res_oos):
    """Chart 2: Parameter AUC ranking."""
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle('CHART 2: Parameter AUC Ranking (DIP_GO vs OTHER)', fontsize=14, fontweight='bold')

    # Merge IS and OOS
    merged = res_is[['parameter', 'auc']].copy()
    merged.columns = ['parameter', 'auc_is']
    if res_oos is not None and len(res_oos) > 0:
        oos_auc = res_oos[['parameter', 'auc']].copy()
        oos_auc.columns = ['parameter', 'auc_oos']
        merged = merged.merge(oos_auc, on='parameter', how='left')
    else:
        merged['auc_oos'] = np.nan

    merged = merged.sort_values('auc_is', ascending=True)

    y_pos = np.arange(len(merged))
    ax.barh(y_pos - 0.15, merged['auc_is'], height=0.3, color=ACCENT1, label='IS', alpha=0.8)
    if merged['auc_oos'].notna().any():
        ax.barh(y_pos + 0.15, merged['auc_oos'], height=0.3, color=ACCENT3, label='OOS', alpha=0.8)

    ax.axvline(x=0.5, color=ACCENT2, linestyle='--', linewidth=1, label='No separation (0.5)')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(merged['parameter'], fontsize=9)
    ax.set_xlabel('AUC')
    ax.legend(loc='lower right')
    ax.set_xlim(0.3, 0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'chart2_auc_ranking.png'), dpi=300, bbox_inches='tight')
    plt.close()
    rpt('  Chart 2 saved.')


def chart3_decile_analysis(decile_is, decile_oos):
    """Chart 3: Top-3 parameter decile analysis."""
    params = list(decile_is.keys())[:3]
    if len(params) == 0:
        return

    fig, axes = plt.subplots(1, len(params), figsize=(6 * len(params), 5))
    if len(params) == 1:
        axes = [axes]
    fig.suptitle('CHART 3: Top-3 Parameter Decile Analysis (% DIP_GO)', fontsize=14, fontweight='bold')

    for i, param in enumerate(params):
        ax = axes[i]
        is_data = decile_is[param]
        ax.plot(is_data['decile'], is_data['pct_dipgo'], 'o-', color=ACCENT1, linewidth=2, label='IS')

        if param in decile_oos and decile_oos[param] is not None:
            oos_data = decile_oos[param]
            ax.plot(oos_data['decile'], oos_data['pct_dipgo'], 's--', color=ACCENT3, linewidth=2, label='OOS')

        ax.set_title(param, fontsize=11)
        ax.set_xlabel('Decile')
        ax.set_ylabel('% DIP_GO')
        ax.legend()
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'chart3_decile_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    rpt('  Chart 3 saved.')


def chart4_heatmap(combo_is, combo_oos):
    """Chart 4: 2D heatmap of top-2 parameter combination."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('CHART 4: 2D Combination Filter (% DIP_GO)', fontsize=14, fontweight='bold')

    cmap = LinearSegmentedColormap.from_list('custom', ['#1a1a2e', ACCENT4, ACCENT2])

    for col_idx, (data, title) in enumerate([(combo_is, 'IS'), (combo_oos, 'OOS')]):
        ax = axes[col_idx]
        if data is None:
            ax.set_title(f'{title}: No data')
            continue

        im = ax.imshow(data['heatmap_pct'], cmap=cmap, aspect='auto', vmin=0, vmax=50)

        # Annotate cells
        for i in range(4):
            for j in range(4):
                pct = data['heatmap_pct'][i, j]
                n = data['heatmap_n'][i, j]
                color = 'black' if pct > 25 else TEXT_COLOR
                ax.text(j, i, f'{pct:.0f}%\n(n={n})', ha='center', va='center', fontsize=9, color=color)

        ax.set_title(f'{title}', fontsize=11)
        ax.set_xlabel(data['param2'])
        ax.set_ylabel(data['param1'])
        ax.set_xticks(range(4))
        ax.set_xticklabels([f'Q{i}' for i in range(4)])
        ax.set_yticks(range(4))
        ax.set_yticklabels([f'Q{i}' for i in range(4)])

        plt.colorbar(im, ax=ax, label='% DIP_GO')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'chart4_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    rpt('  Chart 4 saved.')


def chart5_restmove_boxplot(df):
    """Chart 5: rest_move boxplot DIP_GO vs OTHER vs ALL."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('CHART 5: rest_move Distribution (key→Close, in ADR)', fontsize=14, fontweight='bold')

    for col_idx, (split_name, mask) in enumerate([('IS', df['is_IS']), ('OOS', ~df['is_IS'])]):
        ax = axes[col_idx]
        sub = df[mask]
        dipgo = sub[sub['is_dipgo']]['rest_move'].dropna()
        other = sub[~sub['is_dipgo']]['rest_move'].dropna()
        all_data = sub['rest_move'].dropna()

        data = [dipgo, other, all_data]
        labels = [f'DIP_GO\n(N={len(dipgo)})', f'OTHER\n(N={len(other)})', f'ALL\n(N={len(all_data)})']

        bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False,
                       medianprops=dict(color=ACCENT4, linewidth=2))
        colors = [ACCENT3, ACCENT2, ACCENT1]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.axhline(y=0, color=TEXT_COLOR, linestyle='--', alpha=0.5)
        ax.set_title(f'{split_name}', fontsize=11)
        ax.set_ylabel('rest_move (ADR)')

        # Add median annotations
        for i, d in enumerate(data):
            ax.text(i + 1, d.median() + 0.02, f'{d.median():.3f}', ha='center', fontsize=9, color=ACCENT4)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'chart5_restmove_boxplot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    rpt('  Chart 5 saved.')


def chart6_restmove_lod_vs_1000(df):
    """Chart 6: rest_move from LOD vs from 10:00 (DIP_GO only)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('CHART 6: DIP_GO — Move from LOD vs from 10:00 (in ADR)', fontsize=14, fontweight='bold')

    for col_idx, (split_name, mask) in enumerate([('IS', df['is_IS']), ('OOS', ~df['is_IS'])]):
        ax = axes[col_idx]
        dipgo = df[mask & df['is_dipgo']]

        from_lod = dipgo['rest_move'].dropna()
        from_1000 = dipgo['move_after_1000'].dropna()

        data = [from_lod, from_1000]
        labels = [f'From LOD\n(N={len(from_lod)})', f'From 10:00\n(N={len(from_1000)})']

        bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False,
                       medianprops=dict(color=ACCENT4, linewidth=2))
        colors = [ACCENT3, ACCENT1]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.axhline(y=0, color=TEXT_COLOR, linestyle='--', alpha=0.5)
        ax.set_title(f'{split_name} — DIP_GO only (N={len(dipgo)})', fontsize=11)
        ax.set_ylabel('Move (ADR)')

        for i, d in enumerate(data):
            ax.text(i + 1, d.median() + 0.02, f'{d.median():.3f}', ha='center', fontsize=9, color=ACCENT4)

        # Show how much is "missed" by entering at 10:00
        missed = from_lod.median() - from_1000.median()
        ax.text(1.5, ax.get_ylim()[1] * 0.9, f'Missed: {missed:.3f} ADR',
                ha='center', fontsize=10, color=ACCENT2,
                bbox=dict(boxstyle='round', facecolor=BG_COLOR, edgecolor=ACCENT2))

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'chart6_lod_vs_1000.png'), dpi=300, bbox_inches='tight')
    plt.close()
    rpt('  Chart 6 saved.')


# ============================================================
# STEP 7: OOS VALIDATION + SHRINKAGE
# ============================================================
def step7_shrinkage(res_is, res_oos, decile_is, decile_oos, combo_is, combo_oos, df):
    rpt_header('SHRINKAGE: IS vs OOS')

    if res_oos is None or len(res_oos) == 0:
        rpt('  No OOS results to compare.')
        return

    # Merge AUC
    merged = res_is[['parameter', 'auc']].copy()
    merged.columns = ['parameter', 'auc_is']
    oos_auc = res_oos[['parameter', 'auc']].copy()
    oos_auc.columns = ['parameter', 'auc_oos']
    merged = merged.merge(oos_auc, on='parameter', how='left')
    merged['shrinkage'] = merged['auc_is'] - merged['auc_oos']
    merged['auc_is_dist'] = abs(merged['auc_is'] - 0.5)
    merged['auc_oos_dist'] = abs(merged['auc_oos'] - 0.5)
    merged['shrinkage_pct'] = np.where(
        merged['auc_is_dist'] > 0,
        (1 - merged['auc_oos_dist'] / merged['auc_is_dist']) * 100,
        0
    )

    rpt(f'\n  {"Parameter":<22} {"AUC IS":>8} {"AUC OOS":>8} {"Shrink":>8} {"Shrink%":>8}')
    rpt(f'  {"-"*22} {"-"*8} {"-"*8} {"-"*8} {"-"*8}')
    for _, r in merged.sort_values('auc_is_dist', ascending=False).iterrows():
        rpt(f'  {r["parameter"]:<22} {r["auc_is"]:>8.3f} {r["auc_oos"]:>8.3f} {r["shrinkage"]:>+8.3f} {r["shrinkage_pct"]:>7.1f}%')

    # DIP_GO rate comparison
    rpt_subheader('DIP_GO Rate Comparison')
    is_sub = df[df['is_IS']]
    oos_sub = df[~df['is_IS']]
    is_rate = is_sub['is_dipgo'].mean() * 100
    oos_rate = oos_sub['is_dipgo'].mean() * 100
    rpt(f'  IS DIP_GO rate:  {is_rate:.1f}%')
    rpt(f'  OOS DIP_GO rate: {oos_rate:.1f}%')
    rpt(f'  Difference: {oos_rate - is_rate:+.1f}pp')

    # rest_move comparison
    rpt_subheader('rest_move Comparison (DIP_GO only)')
    is_dg = is_sub[is_sub['is_dipgo']]['rest_move']
    oos_dg = oos_sub[oos_sub['is_dipgo']]['rest_move']
    rpt(f'  IS  median rest_move: {is_dg.median():+.3f} ADR (N={len(is_dg)})')
    rpt(f'  OOS median rest_move: {oos_dg.median():+.3f} ADR (N={len(oos_dg)})')
    rpt(f'  Shrinkage: {is_dg.median() - oos_dg.median():+.3f} ADR')

    # move_after_1000 comparison
    rpt_subheader('move_after_1000 Comparison (DIP_GO only)')
    is_ma = is_sub[is_sub['is_dipgo']]['move_after_1000']
    oos_ma = oos_sub[oos_sub['is_dipgo']]['move_after_1000']
    rpt(f'  IS  median move_after_1000: {is_ma.median():+.3f} ADR (N={len(is_ma)})')
    rpt(f'  OOS median move_after_1000: {oos_ma.median():+.3f} ADR (N={len(oos_ma)})')
    rpt(f'  Shrinkage: {is_ma.median() - oos_ma.median():+.3f} ADR')


# ============================================================
# FAZIT
# ============================================================
def write_fazit(res_is, res_oos, df):
    rpt_header('GESAMTFAZIT')

    is_sub = df[df['is_IS']]
    oos_sub = df[~df['is_IS']]

    # IS stats
    is_dg = is_sub[is_sub['is_dipgo']]
    oos_dg = oos_sub[oos_sub['is_dipgo']]

    is_rate = is_sub['is_dipgo'].mean() * 100
    oos_rate = oos_sub['is_dipgo'].mean() * 100

    is_rm = is_dg['rest_move'].median()
    oos_rm = oos_dg['rest_move'].median() if len(oos_dg) > 0 else np.nan

    is_ma = is_dg['move_after_1000'].median()
    oos_ma = oos_dg['move_after_1000'].median() if len(oos_dg) > 0 else np.nan

    # Best parameters
    if res_is is not None and len(res_is) > 0:
        res_is['auc_dist'] = abs(res_is['auc'] - 0.5)
        top_params = res_is.nlargest(3, 'auc_dist')
        rpt(f'\n  Best IS parameters (by AUC distance from 0.5):')
        for _, r in top_params.iterrows():
            rpt(f'    {r["parameter"]}: AUC={r["auc"]:.3f}, Cohen d={r["cohens_d"]:+.3f}')

    rpt(f'\n  DIP_GO Setup Statistics:')
    rpt(f'    IS DIP_GO rate:     {is_rate:.1f}% (N={len(is_dg)})')
    rpt(f'    OOS DIP_GO rate:    {oos_rate:.1f}% (N={len(oos_dg)})')
    rpt(f'    IS rest_move:       {is_rm:+.3f} ADR')
    rpt(f'    OOS rest_move:      {oos_rm:+.3f} ADR')
    rpt(f'    IS move_after_1000: {is_ma:+.3f} ADR')
    rpt(f'    OOS move_after_1000:{oos_ma:+.3f} ADR')

    # Check if OOS validates
    # Criteria: OOS AUC of top parameters still meaningfully above 0.5
    validated = False
    if res_oos is not None and len(res_oos) > 0:
        res_oos['auc_dist'] = abs(res_oos['auc'] - 0.5)
        top_oos = res_oos.nlargest(3, 'auc_dist')
        max_oos_auc_dist = top_oos['auc_dist'].max()
        if max_oos_auc_dist > 0.05:
            validated = True

    # Check rest_move
    rm_positive = oos_rm > 0 if not np.isnan(oos_rm) else False
    ma_positive = oos_ma > 0 if not np.isnan(oos_ma) else False

    rpt(f'\n  VERDICT:')
    rpt(f'    DIP_GO is a valid setup: {"JA" if (validated and rm_positive) else "NEIN (marginal)" if rm_positive else "NEIN"}')
    rpt(f'    OOS parameter separation holds: {"JA" if validated else "NEIN"}')
    rpt(f'    OOS rest_move positive: {"JA" if rm_positive else "NEIN"}')
    rpt(f'    OOS move_after_1000 positive: {"JA" if ma_positive else "NEIN"}')
    rpt(f'    Empfehlung: {"Backtest lohnt sich" if (validated and ma_positive) else "Backtest lohnt sich bedingt" if ma_positive else "Backtest lohnt sich NICHT"}')


# ============================================================
# SEPARATE BY GAP DIRECTION
# ============================================================
def run_analysis_for_direction(df, direction, label):
    """Run full analysis for a specific gap direction."""
    sub = df[df['gap_direction'] == direction].copy()
    rpt_header(f'======  {label}  (N={len(sub)})  ======')
    rpt(f'  IS: {sub["is_IS"].sum()}, OOS: {(~sub["is_IS"]).sum()}')

    if len(sub) < 30:
        rpt('  Too few trades for meaningful analysis. Skipping.')
        return None, None, None, None, None, None

    # Step 3: IS
    res_is = step3_parameter_comparison(sub, 'IS', sub['is_IS'])

    # Step 4: IS
    step4_profitability(sub, 'IS', sub['is_IS'])

    # Step 5: IS
    decile_is = step5_sensitivity(sub, res_is, 'IS', sub['is_IS'])

    # Step 6: IS
    combo_is = step6_combination(sub, res_is, 'IS', sub['is_IS'])

    # IS FAZIT
    rpt_header(f'{label} — IS FAZIT')
    if res_is is not None and len(res_is) > 0:
        res_is['auc_dist'] = abs(res_is['auc'] - 0.5)
        top = res_is.nlargest(3, 'auc_dist')
        rpt('  Top-3 IS Parameters:')
        for _, r in top.iterrows():
            rpt(f'    {r["parameter"]}: AUC={r["auc"]:.3f}')
    is_dg = sub[(sub['is_IS']) & (sub['is_dipgo'])]
    rpt(f'  IS DIP_GO rate: {sub[sub["is_IS"]]["is_dipgo"].mean()*100:.1f}%')
    rpt(f'  IS DIP_GO rest_move median: {is_dg["rest_move"].median():+.3f} ADR' if len(is_dg) > 0 else '  (no DIP_GO trades)')

    # OOS
    res_oos = step3_parameter_comparison(sub, 'OOS', ~sub['is_IS'])
    step4_profitability(sub, 'OOS', ~sub['is_IS'])
    decile_oos = step5_sensitivity(sub, res_is, 'OOS', ~sub['is_IS'])
    combo_oos = step6_combination(sub, res_is, 'OOS', ~sub['is_IS'])

    # Shrinkage
    step7_shrinkage(res_is, res_oos, decile_is, decile_oos, combo_is, combo_oos, sub)

    return res_is, res_oos, decile_is, decile_oos, combo_is, combo_oos


# ============================================================
# MAIN
# ============================================================
def main():
    rpt('DIP & GO ANALYSIS')
    rpt(f'Date: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}')
    rpt(f'IS cutoff: {IS_CUTOFF}')
    rpt(f'Primary window: {PRIMARY_WINDOW} ({minutes_to_time(WINDOWS[PRIMARY_WINDOW][0])}-{minutes_to_time(WINDOWS[PRIMARY_WINDOW][1])})')

    # Load data
    df = load_data()

    # Step 1: Timing histogram
    step1_timing_histogram(df)

    # Step 2: Define groups
    df = step2_define_groups(df)

    # Step 1 chart (needs group labels)
    chart1_timing_histogram(df)

    # Compute peak-after-LOD from 1-min bars
    df = compute_peak_after_key(df)

    # ============================================================
    # COMBINED ANALYSIS (ALL DIRECTIONS)
    # ============================================================
    rpt_header('====== COMBINED (GapUp + GapDn) ======')

    # IS Analysis
    res_is_all = step3_parameter_comparison(df, 'IS', df['is_IS'])
    step4_profitability(df, 'IS', df['is_IS'])
    decile_is_all = step5_sensitivity(df, res_is_all, 'IS', df['is_IS'])
    combo_is_all = step6_combination(df, res_is_all, 'IS', df['is_IS'])

    # OOS Analysis
    res_oos_all = step3_parameter_comparison(df, 'OOS', ~df['is_IS'])
    step4_profitability(df, 'OOS', ~df['is_IS'])
    decile_oos_all = step5_sensitivity(df, res_is_all, 'OOS', ~df['is_IS'])
    combo_oos_all = step6_combination(df, res_is_all, 'OOS', ~df['is_IS'])

    step7_shrinkage(res_is_all, res_oos_all, decile_is_all, decile_oos_all, combo_is_all, combo_oos_all, df)

    # ============================================================
    # SEPARATE BY GAP DIRECTION
    # ============================================================
    res_is_up, res_oos_up, dec_is_up, dec_oos_up, combo_is_up, combo_oos_up = \
        run_analysis_for_direction(df, 'up', 'GAP UP')

    res_is_dn, res_oos_dn, dec_is_dn, dec_oos_dn, combo_is_dn, combo_oos_dn = \
        run_analysis_for_direction(df, 'down', 'GAP DOWN')

    # ============================================================
    # GESAMTFAZIT
    # ============================================================
    write_fazit(res_is_all, res_oos_all, df)

    # ============================================================
    # CHARTS
    # ============================================================
    rpt_header('GENERATING CHARTS')

    # Chart 2: AUC ranking
    if res_is_all is not None and len(res_is_all) > 0:
        chart2_auc_ranking(res_is_all, res_oos_all)

    # Chart 3: Decile analysis
    if decile_is_all:
        chart3_decile_analysis(decile_is_all, decile_oos_all)

    # Chart 4: 2D heatmap
    chart4_heatmap(combo_is_all, combo_oos_all)

    # Chart 5: rest_move boxplot
    chart5_restmove_boxplot(df)

    # Chart 6: LOD vs 10:00
    chart6_restmove_lod_vs_1000(df)

    # Save report
    save_report()

    rpt('\n=== DONE ===')


if __name__ == '__main__':
    main()
