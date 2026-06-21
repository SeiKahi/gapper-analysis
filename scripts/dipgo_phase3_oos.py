"""
DIP & GO — Phase 3: OOS-Validierung der Phase 2 Erkenntnisse
==============================================================
Repliziert alle IS-Metriken aus Phase 2 auf OOS (>= 2024-01-01)
und berechnet Shrinkage.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
OUT_DIR = os.path.join(BASE_DIR, 'results', 'dipgo_phase3')
os.makedirs(OUT_DIR, exist_ok=True)

IS_CUTOFF = '2024-01-01'
WINDOW_LO = 15   # 9:45
WINDOW_HI = 50   # 10:20

BG_COLOR = '#1a1a2e'
GRID_COLOR = '#2a2a4e'
TEXT_COLOR = '#e0e0e0'
C_CYAN = '#00d4ff'
C_RED = '#ff6b6b'
C_GREEN = '#51cf66'
C_YELLOW = '#ffd43b'
C_ORANGE = '#ff922b'
C_PURPLE = '#cc5de8'

plt.style.use('dark_background')
plt.rcParams.update({
    'figure.facecolor': BG_COLOR, 'axes.facecolor': BG_COLOR,
    'axes.edgecolor': GRID_COLOR, 'axes.grid': True,
    'grid.color': GRID_COLOR, 'grid.alpha': 0.3,
    'text.color': TEXT_COLOR, 'axes.labelcolor': TEXT_COLOR,
    'xtick.color': TEXT_COLOR, 'ytick.color': TEXT_COLOR, 'font.size': 10,
})

# ============================================================
# HELPERS
# ============================================================
R = []
def rpt(t=''):
    R.append(t); print(t)
def rpt_h(t):
    rpt(''); rpt('=' * 70); rpt(f'  {t}'); rpt('=' * 70)
def rpt_s(t):
    rpt(''); rpt(f'--- {t} ---')
def save_report():
    p = os.path.join(OUT_DIR, 'dipgo_phase3_oos.txt')
    with open(p, 'w', encoding='utf-8') as f:
        f.write('\n'.join(R))
    print(f'\nReport saved: {p}')
def mt(m):
    h = int(9 + (30 + m) // 60); mi = int((30 + m) % 60)
    return f'{h:02d}:{mi:02d}'
def pct(n, t):
    return n / t * 100 if t > 0 else 0.0

def load_rth_bars(ticker, date_str):
    path = os.path.join(RAW_1MIN_DIR, ticker, f'{date_str}.parquet')
    if not os.path.exists(path):
        return None
    bars = pd.read_parquet(path)
    rth = bars[bars['session'] == 'rth'].copy()
    if len(rth) == 0:
        rth = bars[(bars['time_et'] >= '09:30') & (bars['time_et'] < '16:00')].copy()
    if len(rth) == 0:
        return None
    parts = rth['time_et'].str.split(':')
    rth['min930'] = parts.str[0].astype(int) * 60 + parts.str[1].astype(int) - 570
    rth = rth[rth['min930'] >= 0].copy()
    return rth


# ============================================================
# ENRICH SPLIT FROM 1-MIN BARS
# ============================================================
def enrich_split(df):
    """Add 1-min-bar-derived columns for DIP_GO trades."""
    cols = ['dip_depth', 'dip_vs_open', 'would_trigger_sl025',
            'bounce_5min', 'bar_range_at_lod', 'lower_wick_pct',
            'recovery_time', 'time_to_new_high', 'rest_move_from_lod',
            'peak_after_lod', 'retention', 'local_low_is_true_lod',
            'mae_after_window']
    for c in cols:
        df[c] = np.nan

    dipgo_idx = df[df['is_dipgo']].index
    count = 0
    for idx in dipgo_idx:
        row = df.loc[idx]
        ticker = row['ticker']
        date_str = str(row['date'].date())
        gap_dir = row['gap_direction']
        adr = row['adr_10']
        c935 = row['close_935']
        bars = load_rth_bars(ticker, date_str)
        if bars is None or len(bars) < 10 or adr <= 0:
            continue

        if gap_dir == 'up':
            lod_idx = bars['low'].idxmin()
            lod_bar = bars.loc[lod_idx]
            lod_price = lod_bar['low']
            lod_min = lod_bar['min930']

            df.at[idx, 'dip_depth'] = (c935 - lod_price) / adr
            df.at[idx, 'dip_vs_open'] = (row['rth_open'] - lod_price) / adr

            sl_level = c935 - 0.25 * adr
            after_entry = bars[(bars['min930'] >= 5) & (bars['min930'] <= lod_min)]
            if len(after_entry) > 0:
                df.at[idx, 'would_trigger_sl025'] = 1.0 if after_entry['low'].min() <= sl_level else 0.0

            bh, bl, bc, bo = lod_bar['high'], lod_bar['low'], lod_bar['close'], lod_bar['open']
            br = bh - bl
            df.at[idx, 'bar_range_at_lod'] = br / adr if adr > 0 else np.nan
            if br > 0:
                df.at[idx, 'lower_wick_pct'] = (min(bo, bc) - bl) / br

            after_lod = bars[bars['min930'] > lod_min]
            if len(after_lod) >= 5:
                df.at[idx, 'bounce_5min'] = (after_lod.iloc[4]['close'] - lod_price) / adr
            elif len(after_lod) > 0:
                df.at[idx, 'bounce_5min'] = (after_lod.iloc[-1]['close'] - lod_price) / adr

            if len(after_lod) > 0:
                rec = after_lod[after_lod['close'] >= c935]
                if len(rec) > 0:
                    df.at[idx, 'recovery_time'] = rec.iloc[0]['min930'] - lod_min

            bars_935_1000 = bars[(bars['min930'] >= 5) & (bars['min930'] <= 30)]
            if len(bars_935_1000) > 0 and len(after_lod) > 0:
                pre_high = bars_935_1000['high'].max()
                nh = after_lod[after_lod['high'] > pre_high]
                if len(nh) > 0:
                    df.at[idx, 'time_to_new_high'] = nh.iloc[0]['min930'] - lod_min

            df.at[idx, 'rest_move_from_lod'] = (row['rth_close'] - lod_price) / adr
            if len(after_lod) > 0:
                pk = after_lod['high'].max()
                pk_m = (pk - lod_price) / adr
                df.at[idx, 'peak_after_lod'] = pk_m
                rm = (row['rth_close'] - lod_price) / adr
                if pk_m > 0:
                    df.at[idx, 'retention'] = rm / pk_m

            wb = bars[(bars['min930'] >= WINDOW_LO) & (bars['min930'] <= WINDOW_HI)]
            if len(wb) > 0:
                ll = wb['low'].min()
                tl = bars['low'].min()
                df.at[idx, 'local_low_is_true_lod'] = 1.0 if abs(ll - tl) < 0.005 else 0.0
                aw = bars[bars['min930'] > WINDOW_HI]
                if len(aw) > 0:
                    df.at[idx, 'mae_after_window'] = (ll - aw['low'].min()) / adr
                else:
                    df.at[idx, 'mae_after_window'] = 0.0

        else:  # GapDn
            hod_idx = bars['high'].idxmax()
            hod_bar = bars.loc[hod_idx]
            hod_price = hod_bar['high']
            hod_min = hod_bar['min930']

            df.at[idx, 'dip_depth'] = (hod_price - c935) / adr
            df.at[idx, 'dip_vs_open'] = (hod_price - row['rth_open']) / adr

            sl_level = c935 + 0.25 * adr
            after_entry = bars[(bars['min930'] >= 5) & (bars['min930'] <= hod_min)]
            if len(after_entry) > 0:
                df.at[idx, 'would_trigger_sl025'] = 1.0 if after_entry['high'].max() >= sl_level else 0.0

            bh, bl, bc, bo = hod_bar['high'], hod_bar['low'], hod_bar['close'], hod_bar['open']
            br = bh - bl
            df.at[idx, 'bar_range_at_lod'] = br / adr if adr > 0 else np.nan
            if br > 0:
                df.at[idx, 'lower_wick_pct'] = (bh - max(bo, bc)) / br

            after_hod = bars[bars['min930'] > hod_min]
            if len(after_hod) >= 5:
                df.at[idx, 'bounce_5min'] = (hod_price - after_hod.iloc[4]['close']) / adr
            elif len(after_hod) > 0:
                df.at[idx, 'bounce_5min'] = (hod_price - after_hod.iloc[-1]['close']) / adr

            if len(after_hod) > 0:
                rec = after_hod[after_hod['close'] <= c935]
                if len(rec) > 0:
                    df.at[idx, 'recovery_time'] = rec.iloc[0]['min930'] - hod_min

            bars_935_1000 = bars[(bars['min930'] >= 5) & (bars['min930'] <= 30)]
            if len(bars_935_1000) > 0 and len(after_hod) > 0:
                pre_low = bars_935_1000['low'].min()
                nl = after_hod[after_hod['low'] < pre_low]
                if len(nl) > 0:
                    df.at[idx, 'time_to_new_high'] = nl.iloc[0]['min930'] - hod_min

            df.at[idx, 'rest_move_from_lod'] = (hod_price - row['rth_close']) / adr
            if len(after_hod) > 0:
                tr = after_hod['low'].min()
                pk_m = (hod_price - tr) / adr
                df.at[idx, 'peak_after_lod'] = pk_m
                rm = (hod_price - row['rth_close']) / adr
                if pk_m > 0:
                    df.at[idx, 'retention'] = rm / pk_m

            wb = bars[(bars['min930'] >= WINDOW_LO) & (bars['min930'] <= WINDOW_HI)]
            if len(wb) > 0:
                lh = wb['high'].max()
                th = bars['high'].max()
                df.at[idx, 'local_low_is_true_lod'] = 1.0 if abs(lh - th) < 0.005 else 0.0
                aw = bars[bars['min930'] > WINDOW_HI]
                if len(aw) > 0:
                    df.at[idx, 'mae_after_window'] = (aw['high'].max() - lh) / adr
                else:
                    df.at[idx, 'mae_after_window'] = 0.0

        count += 1
        if count % 20 == 0:
            print(f'  Enriched {count}/{len(dipgo_idx)}...')

    rpt(f'  Enriched {count} DIP_GO trades.')
    return df


# ============================================================
# COMPUTE ALL METRICS FOR A SPLIT
# ============================================================
def compute_metrics(df, label):
    """Compute all Phase-2 metrics for a df that is already filtered to a split."""
    m = {}
    m['label'] = label
    n = len(df)
    m['n_total'] = n

    dipgo = df[df['is_dipgo']]
    n_dg = len(dipgo)
    m['n_dipgo'] = n_dg
    m['dipgo_rate'] = pct(n_dg, n)

    win = dipgo[dipgo['is_dipwin']]
    lose = dipgo[~dipgo['is_dipwin']]
    m['n_win'] = len(win)
    m['n_lose'] = len(lose)
    m['win_rate'] = pct(len(win), n_dg)

    # By direction
    for gd in ['up', 'down']:
        sub = dipgo[dipgo['gap_direction'] == gd]
        w = sub['is_dipwin'].sum()
        m[f'n_dipgo_{gd}'] = len(sub)
        m[f'win_rate_{gd}'] = pct(w, len(sub))

    # Timing
    kt = dipgo['key_time']
    if n_dg > 0:
        m['timing_mean'] = kt.mean()
        m['timing_median'] = kt.median()
        m['timing_std'] = kt.std()
        m['timing_p25'] = kt.quantile(0.25)
        m['timing_p75'] = kt.quantile(0.75)
    else:
        for k in ['timing_mean','timing_median','timing_std','timing_p25','timing_p75']:
            m[k] = np.nan

    # Sub-windows
    sub_windows = [('9:45-9:50', 15, 20), ('9:50-10:00', 20, 30),
                   ('10:00-10:10', 30, 40), ('10:10-10:20', 40, 50)]
    for lab, lo, hi in sub_windows:
        mask = (kt >= lo) & (kt <= hi)
        n_sw = mask.sum()
        n_sw_win = win['key_time'].between(lo, hi).sum() if len(win) > 0 else 0
        m[f'sw_{lab}_n'] = n_sw
        m[f'sw_{lab}_pct'] = pct(n_sw, n_dg)
        m[f'sw_{lab}_wr'] = pct(n_sw_win, n_sw)

    # WIN vs LOSE timing
    if len(win) > 0:
        m['timing_win_median'] = win['key_time'].median()
    else:
        m['timing_win_median'] = np.nan
    if len(lose) > 0:
        m['timing_lose_median'] = lose['key_time'].median()
    else:
        m['timing_lose_median'] = np.nan

    # Depth
    dd = dipgo['dip_depth'].dropna()
    if len(dd) > 0:
        m['depth_mean'] = dd.mean()
        m['depth_median'] = dd.median()
        m['depth_std'] = dd.std()
        for q in [10, 25, 50, 75, 90]:
            m[f'depth_p{q}'] = dd.quantile(q / 100)
    else:
        for k in ['depth_mean','depth_median','depth_std'] + [f'depth_p{q}' for q in [10,25,50,75,90]]:
            m[k] = np.nan

    # WIN vs LOSE depth
    dd_w = win['dip_depth'].dropna()
    dd_l = lose['dip_depth'].dropna()
    m['depth_win_median'] = dd_w.median() if len(dd_w) > 0 else np.nan
    m['depth_lose_median'] = dd_l.median() if len(dd_l) > 0 else np.nan

    # Dip vs open
    dvo = dipgo['dip_vs_open'].dropna()
    if len(dvo) > 0:
        m['pct_under_open'] = pct((dvo > 0).sum(), len(dvo))
    else:
        m['pct_under_open'] = np.nan

    # SL trigger
    sl = dipgo['would_trigger_sl025'].dropna()
    m['sl025_trigger_pct'] = pct(sl.sum(), len(sl)) if len(sl) > 0 else np.nan

    # Recovery
    rt = win['recovery_time'].dropna()
    m['recovery_median'] = rt.median() if len(rt) > 0 else np.nan
    m['recovery_mean'] = rt.mean() if len(rt) > 0 else np.nan
    m['recovery_never_pct'] = pct(win['recovery_time'].isna().sum(), len(win)) if len(win) > 0 else np.nan

    tnh = win['time_to_new_high'].dropna()
    m['new_high_median'] = tnh.median() if len(tnh) > 0 else np.nan
    m['new_high_never_pct'] = pct(win['time_to_new_high'].isna().sum(), len(win)) if len(win) > 0 else np.nan

    # Bounce
    b5 = dipgo['bounce_5min'].dropna()
    m['bounce_5min_median'] = b5.median() if len(b5) > 0 else np.nan
    b5w = win['bounce_5min'].dropna()
    b5l = lose['bounce_5min'].dropna()
    m['bounce_5min_win'] = b5w.median() if len(b5w) > 0 else np.nan
    m['bounce_5min_lose'] = b5l.median() if len(b5l) > 0 else np.nan

    # Wick
    lw_w = win['lower_wick_pct'].dropna()
    lw_l = lose['lower_wick_pct'].dropna()
    m['wick_win'] = lw_w.median() if len(lw_w) > 0 else np.nan
    m['wick_lose'] = lw_l.median() if len(lw_l) > 0 else np.nan

    # Rest move
    rm = dipgo['rest_move_from_lod'].dropna()
    if len(rm) > 0:
        m['restmove_mean'] = rm.mean()
        m['restmove_median'] = rm.median()
        m['restmove_std'] = rm.std()
        m['restmove_pct_pos'] = pct((rm > 0).sum(), len(rm))
        for q in [10, 25, 50, 75, 90]:
            m[f'restmove_p{q}'] = rm.quantile(q / 100)
    else:
        for k in ['restmove_mean','restmove_median','restmove_std','restmove_pct_pos']:
            m[k] = np.nan

    # Peak & retention
    pk = dipgo['peak_after_lod'].dropna()
    ret = dipgo['retention'].dropna()
    m['peak_median'] = pk.median() if len(pk) > 0 else np.nan
    m['retention_median'] = ret.median() if len(ret) > 0 else np.nan

    # Fenster vs Tages
    ll = dipgo['local_low_is_true_lod'].dropna()
    m['fenster_eq_tages_pct'] = pct(ll.sum(), len(ll)) if len(ll) > 0 else np.nan
    mae = dipgo['mae_after_window'].dropna()
    mae_pos = mae[mae > 0]
    m['mae_n'] = len(mae_pos)
    m['mae_median'] = mae_pos.median() if len(mae_pos) > 0 else np.nan
    m['mae_p75'] = mae_pos.quantile(0.75) if len(mae_pos) > 0 else np.nan

    return m


# ============================================================
# SHRINKAGE TABLE
# ============================================================
def print_shrinkage(is_m, oos_m):
    rpt_h('SHRINKAGE: IS vs OOS')

    rows = [
        ('DIP_GO Rate (%)', 'dipgo_rate', '%', False),
        ('Win-Rate (%)', 'win_rate', '%', False),
        ('Win-Rate GapUp (%)', 'win_rate_up', '%', False),
        ('Win-Rate GapDn (%)', 'win_rate_down', '%', False),
        ('Timing Median (min)', 'timing_median', 'min', False),
        ('Timing StdDev (min)', 'timing_std', 'min', False),
        ('Sub 9:45-10:00 WR (%)', 'sw_9:50-10:00_wr', '%', False),
        ('Sub 10:10-10:20 WR (%)', 'sw_10:10-10:20_wr', '%', False),
        ('Depth Median (ADR)', 'depth_median', 'ADR', False),
        ('Depth P90 (ADR)', 'depth_p90', 'ADR', False),
        ('% under Open', 'pct_under_open', '%', False),
        ('SL-0.25 Trigger (%)', 'sl025_trigger_pct', '%', False),
        ('Recovery Median (min)', 'recovery_median', 'min', False),
        ('Recovery Never (%)', 'recovery_never_pct', '%', False),
        ('Bounce 5min Median', 'bounce_5min_median', 'ADR', False),
        ('Rest Move Median', 'restmove_median', 'ADR', False),
        ('Rest Move % pos', 'restmove_pct_pos', '%', False),
        ('Peak Median', 'peak_median', 'ADR', False),
        ('Retention Median', 'retention_median', 'ratio', False),
        ('Fenster=Tages (%)', 'fenster_eq_tages_pct', '%', False),
    ]

    rpt(f'  {"Metric":<28} {"IS":>10} {"OOS":>10} {"Diff":>10} {"Shrink%":>10}')
    rpt(f'  {"-"*28} {"-"*10} {"-"*10} {"-"*10} {"-"*10}')

    for label, key, unit, invert in rows:
        iv = is_m.get(key, np.nan)
        ov = oos_m.get(key, np.nan)
        if pd.isna(iv) or pd.isna(ov):
            rpt(f'  {label:<28} {_fmt(iv, unit):>10} {_fmt(ov, unit):>10} {"N/A":>10} {"N/A":>10}')
            continue
        diff = ov - iv
        # Shrinkage % relative to IS value for rates, absolute for others
        if abs(iv) > 0.01:
            shrink = (1 - ov / iv) * 100
        else:
            shrink = np.nan
        rpt(f'  {label:<28} {_fmt(iv, unit):>10} {_fmt(ov, unit):>10} {diff:>+10.2f} {_fmt(shrink, "%"):>10}')


def _fmt(v, unit):
    if pd.isna(v):
        return 'N/A'
    if unit == '%':
        return f'{v:.1f}%'
    elif unit == 'min':
        return f'{v:.1f}'
    elif unit == 'ADR':
        return f'{v:.3f}'
    elif unit == 'ratio':
        return f'{v:.1%}'
    return f'{v:.3f}'


# ============================================================
# DETAILED OOS REPORT (mirror Phase 2)
# ============================================================
def detailed_report(df, label, m):
    rpt_h(f'{label}: DETAILLIERTE ERGEBNISSE')

    dipgo = df[df['is_dipgo']]
    win = dipgo[dipgo['is_dipwin']]
    lose = dipgo[~dipgo['is_dipwin']]

    # Definition
    rpt_s('DEFINITION')
    rpt(f'  Qualifying trades: {m["n_total"]}')
    rpt(f'  DIP_GO: {m["n_dipgo"]} ({m["dipgo_rate"]:.1f}%)')
    rpt(f'  DIP_WIN: {m["n_win"]} ({m["win_rate"]:.1f}%)')
    rpt(f'  DIP_LOSE: {m["n_lose"]}')
    for gd in ['up', 'down']:
        rpt(f'  Gap{gd.title()}: N={m[f"n_dipgo_{gd}"]}, WR={m[f"win_rate_{gd}"]:.1f}%')

    # Timing
    rpt_s('TIMING')
    rpt(f'  Mean:   {m["timing_mean"]:.1f} min = {mt(m["timing_mean"])}')
    rpt(f'  Median: {m["timing_median"]:.1f} min = {mt(m["timing_median"])}')
    rpt(f'  StdDev: {m["timing_std"]:.1f} min')
    rpt(f'  P25-P75: {mt(m["timing_p25"])} - {mt(m["timing_p75"])}')
    rpt(f'  WIN median: {mt(m["timing_win_median"])}' if not pd.isna(m['timing_win_median']) else '  WIN median: N/A')
    rpt(f'  LOSE median: {mt(m["timing_lose_median"])}' if not pd.isna(m['timing_lose_median']) else '  LOSE median: N/A')

    sub_windows = [('9:45-9:50', 15, 20), ('9:50-10:00', 20, 30),
                   ('10:00-10:10', 30, 40), ('10:10-10:20', 40, 50)]
    for lab, lo, hi in sub_windows:
        n_sw = m[f'sw_{lab}_n']
        pct_sw = m[f'sw_{lab}_pct']
        wr_sw = m[f'sw_{lab}_wr']
        rpt(f'  {lab}: {n_sw:>3} ({pct_sw:.1f}%), WR={wr_sw:.0f}%')

    # Depth
    rpt_s('TIEFE')
    rpt(f'  Median: {m["depth_median"]:.3f} ADR')
    rpt(f'  Mean:   {m["depth_mean"]:.3f} ADR')
    rpt(f'  P10-P90: {m["depth_p10"]:.3f} - {m["depth_p90"]:.3f} ADR')
    rpt(f'  WIN median: {m["depth_win_median"]:.3f}' if not pd.isna(m['depth_win_median']) else '  WIN: N/A')
    rpt(f'  LOSE median: {m["depth_lose_median"]:.3f}' if not pd.isna(m['depth_lose_median']) else '  LOSE: N/A')
    rpt(f'  % unter Open: {m["pct_under_open"]:.1f}%')
    rpt(f'  SL-0.25 Trigger: {m["sl025_trigger_pct"]:.1f}%')

    # Recovery
    rpt_s('RECOVERY')
    rpt(f'  Recovery median: {m["recovery_median"]:.0f} min' if not pd.isna(m['recovery_median']) else '  Recovery: N/A')
    rpt(f'  Never recovered: {m["recovery_never_pct"]:.1f}%' if not pd.isna(m['recovery_never_pct']) else '')
    rpt(f'  New High median: {m["new_high_median"]:.0f} min' if not pd.isna(m['new_high_median']) else '  New High: N/A')
    rpt(f'  Bounce 5min: {m["bounce_5min_median"]:.3f} ADR')
    rpt(f'    WIN: {m["bounce_5min_win"]:.3f}, LOSE: {_fmt(m["bounce_5min_lose"], "ADR")}')
    rpt(f'  Wick WIN: {_fmt(m["wick_win"], "ratio")}, LOSE: {_fmt(m["wick_lose"], "ratio")}')

    # Rest-move
    rpt_s('REST-MOVE')
    rpt(f'  Median: {m["restmove_median"]:.3f} ADR')
    rpt(f'  Mean:   {m["restmove_mean"]:.3f} ADR')
    rpt(f'  % positiv: {m["restmove_pct_pos"]:.1f}%')
    rpt(f'  Peak median: {m["peak_median"]:.3f} ADR')
    rpt(f'  Retention: {m["retention_median"]:.1%}' if not pd.isna(m['retention_median']) else '')

    # Fenster vs Tages
    rpt_s('FENSTER vs TAGES-LOW')
    rpt(f'  Fenster=Tages: {m["fenster_eq_tages_pct"]:.1f}%')
    rpt(f'  MAE after window: N={m["mae_n"]}, median={_fmt(m["mae_median"], "ADR")}, P75={_fmt(m["mae_p75"], "ADR")}')


# ============================================================
# CHARTS
# ============================================================
def chart1_timing_is_vs_oos(df_is, df_oos):
    """Side-by-side timing histogram."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('CHART 1: LOD/HOD Timing — IS vs OOS', fontsize=14, fontweight='bold')
    bins = np.arange(WINDOW_LO, WINDOW_HI + 2, 1)

    for col_i, (dfs, label) in enumerate([(df_is, 'IS'), (df_oos, 'OOS')]):
        ax = axes[col_i]
        dipgo = dfs[dfs['is_dipgo']]
        win = dipgo[dipgo['is_dipwin']]
        lose = dipgo[~dipgo['is_dipwin']]

        h_w, _ = np.histogram(win['key_time'].dropna(), bins=bins)
        h_l, _ = np.histogram(lose['key_time'].dropna(), bins=bins)
        ax.bar(bins[:-1], h_w, width=1, color=C_GREEN, alpha=0.7, label=f'WIN (N={len(win)})', align='edge')
        ax.bar(bins[:-1], h_l, width=1, color=C_RED, alpha=0.7, label=f'LOSE (N={len(lose)})',
               align='edge', bottom=h_w)

        kt = dipgo['key_time']
        if len(kt) > 0:
            ax.axvline(kt.median(), color=C_YELLOW, linewidth=2, label=f'Median {mt(kt.median())}')

        ticks = np.arange(WINDOW_LO, WINDOW_HI + 1, 5)
        ax.set_xticks(ticks)
        ax.set_xticklabels([mt(t) for t in ticks], rotation=45)
        ax.set_title(f'{label} (N={len(dipgo)})', fontsize=12)
        ax.set_xlabel('LOD/HOD Time')
        ax.set_ylabel('Count')
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'chart1_timing_is_oos.png'), dpi=300, bbox_inches='tight')
    plt.close()


def chart2_depth_is_vs_oos(df_is, df_oos):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('CHART 2: Dip Depth — IS vs OOS', fontsize=14, fontweight='bold')

    for col_i, (dfs, label) in enumerate([(df_is, 'IS'), (df_oos, 'OOS')]):
        ax = axes[col_i]
        dipgo = dfs[dfs['is_dipgo']]
        win = dipgo[dipgo['is_dipwin']]
        lose = dipgo[~dipgo['is_dipwin']]
        dd_w = win['dip_depth'].dropna()
        dd_l = lose['dip_depth'].dropna()
        all_dd = pd.concat([dd_w, dd_l])
        if len(all_dd) == 0:
            continue
        mx = min(all_dd.quantile(0.98) if len(all_dd) > 2 else 3.0, 4.0)
        b = np.arange(0, mx + 0.1, 0.1)
        hw, _ = np.histogram(dd_w, bins=b)
        hl, _ = np.histogram(dd_l, bins=b)
        ax.bar(b[:-1], hw, width=0.1, color=C_GREEN, alpha=0.7, label=f'WIN (N={len(dd_w)})', align='edge')
        ax.bar(b[:-1], hl, width=0.1, color=C_RED, alpha=0.7, label=f'LOSE (N={len(dd_l)})',
               align='edge', bottom=hw)
        ax.axvline(0.25, color=C_YELLOW, linestyle='--', linewidth=2, label='0.25 ADR SL')
        ax.set_title(f'{label}', fontsize=12)
        ax.set_xlabel('Dip Depth (ADR)')
        ax.set_ylabel('Count')
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'chart2_depth_is_oos.png'), dpi=300, bbox_inches='tight')
    plt.close()


def chart3_shrinkage_bars(is_m, oos_m):
    """Key metrics comparison bar chart."""
    metrics = [
        ('Win-Rate', 'win_rate'),
        ('DIP_GO %', 'dipgo_rate'),
        ('Depth (ADR)', 'depth_median'),
        ('Rest Move (ADR)', 'restmove_median'),
        ('Bounce 5m (ADR)', 'bounce_5min_median'),
        ('Recovery (min)', 'recovery_median'),
        ('Fenster=Tages %', 'fenster_eq_tages_pct'),
        ('SL Trigger %', 'sl025_trigger_pct'),
        ('% Positiv', 'restmove_pct_pos'),
    ]

    labels = [m[0] for m in metrics]
    is_vals = [is_m.get(m[1], np.nan) for m in metrics]
    oos_vals = [oos_m.get(m[1], np.nan) for m in metrics]

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.suptitle('CHART 3: IS vs OOS — Key Metrics', fontsize=14, fontweight='bold')

    y = np.arange(len(labels))
    ax.barh(y - 0.17, is_vals, height=0.34, color=C_CYAN, alpha=0.8, label='IS')
    ax.barh(y + 0.17, oos_vals, height=0.34, color=C_GREEN, alpha=0.8, label='OOS')

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.legend(fontsize=11)

    # Annotate values
    for i, (iv, ov) in enumerate(zip(is_vals, oos_vals)):
        if not pd.isna(iv):
            ax.text(iv + 0.5, i - 0.17, f'{iv:.1f}', va='center', fontsize=8, color=C_CYAN)
        if not pd.isna(ov):
            ax.text(ov + 0.5, i + 0.17, f'{ov:.1f}', va='center', fontsize=8, color=C_GREEN)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'chart3_shrinkage_bars.png'), dpi=300, bbox_inches='tight')
    plt.close()


def chart4_subwindow_wr(is_m, oos_m):
    """Sub-window win rates IS vs OOS."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('CHART 4: Sub-Window Win-Rate — IS vs OOS', fontsize=14, fontweight='bold')

    sw_labels = ['9:45-9:50', '9:50-10:00', '10:00-10:10', '10:10-10:20']
    is_wr = [is_m.get(f'sw_{l}_wr', np.nan) for l in sw_labels]
    oos_wr = [oos_m.get(f'sw_{l}_wr', np.nan) for l in sw_labels]
    is_n = [is_m.get(f'sw_{l}_n', 0) for l in sw_labels]
    oos_n = [oos_m.get(f'sw_{l}_n', 0) for l in sw_labels]

    x = np.arange(len(sw_labels))
    w = 0.35
    bars1 = ax.bar(x - w/2, is_wr, w, color=C_CYAN, alpha=0.8, label='IS')
    bars2 = ax.bar(x + w/2, oos_wr, w, color=C_GREEN, alpha=0.8, label='OOS')

    # Annotate N
    for i, (b1, b2) in enumerate(zip(bars1, bars2)):
        ax.text(b1.get_x() + b1.get_width()/2, b1.get_height() + 1,
                f'N={is_n[i]}', ha='center', fontsize=9, color=C_CYAN)
        ax.text(b2.get_x() + b2.get_width()/2, b2.get_height() + 1,
                f'N={oos_n[i]}', ha='center', fontsize=9, color=C_GREEN)

    ax.set_xticks(x)
    ax.set_xticklabels(sw_labels)
    ax.set_ylabel('Win-Rate (%)')
    ax.axhline(50, color=C_RED, linestyle='--', alpha=0.5, label='50% (coin flip)')
    ax.set_ylim(0, 110)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'chart4_subwindow_wr.png'), dpi=300, bbox_inches='tight')
    plt.close()


def chart5_restmove_boxplot(df_is, df_oos):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('CHART 5: Rest Move from LOD — IS vs OOS', fontsize=14, fontweight='bold')

    for col_i, (dfs, label) in enumerate([(df_is, 'IS'), (df_oos, 'OOS')]):
        ax = axes[col_i]
        dipgo = dfs[dfs['is_dipgo']]
        win = dipgo[dipgo['is_dipwin']]['rest_move_from_lod'].dropna()
        lose = dipgo[~dipgo['is_dipwin']]['rest_move_from_lod'].dropna()
        data = [d for d in [win, lose] if len(d) > 0]
        labs = []
        if len(win) > 0:
            labs.append(f'WIN\n(N={len(win)})')
        if len(lose) > 0:
            labs.append(f'LOSE\n(N={len(lose)})')
        if len(data) > 0:
            bp = ax.boxplot(data, labels=labs, patch_artist=True, showfliers=False,
                           medianprops=dict(color=C_YELLOW, linewidth=2))
            colors = [C_GREEN, C_RED][:len(data)]
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color); patch.set_alpha(0.6)
            for i, d in enumerate(data):
                ax.text(i + 1, d.median() + 0.02, f'{d.median():.3f}', ha='center', fontsize=10, color=C_YELLOW)
        ax.axhline(0, color=TEXT_COLOR, linestyle='--', alpha=0.3)
        ax.set_title(f'{label}', fontsize=12)
        ax.set_ylabel('Rest Move from LOD (ADR)')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'chart5_restmove_boxplot.png'), dpi=300, bbox_inches='tight')
    plt.close()


def chart6_fenster_vs_tages(df_is, df_oos):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('CHART 6: Fenster-Low vs Tages-Low — IS vs OOS', fontsize=14, fontweight='bold')

    for col_i, (dfs, label) in enumerate([(df_is, 'IS'), (df_oos, 'OOS')]):
        ax = axes[col_i]
        dipgo = dfs[dfs['is_dipgo']]
        ll = dipgo['local_low_is_true_lod'].dropna()
        n_true = int(ll.sum())
        n_not = len(ll) - n_true
        if len(ll) > 0:
            sizes = [n_true, n_not]
            labs = [f'Fenster=Tages\n{n_true} ({pct(n_true, len(ll)):.0f}%)',
                    f'Tieferes Tief\n{n_not} ({pct(n_not, len(ll)):.0f}%)']
            cols = [C_GREEN, C_RED]
            wedges, texts = ax.pie(sizes, labels=labs, colors=cols,
                                    startangle=90, textprops={'color': TEXT_COLOR, 'fontsize': 11})
            for w in wedges:
                w.set_edgecolor(BG_COLOR); w.set_linewidth(2)
        ax.set_title(f'{label} (N={len(ll)})', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'chart6_fenster_vs_tages.png'), dpi=300, bbox_inches='tight')
    plt.close()


def chart7_scatter_is_oos(df_is, df_oos):
    """Depth vs rest move scatter for both splits."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('CHART 7: Dip Depth vs Rest Move — IS vs OOS', fontsize=14, fontweight='bold')

    for col_i, (dfs, label) in enumerate([(df_is, 'IS'), (df_oos, 'OOS')]):
        ax = axes[col_i]
        dipgo = dfs[dfs['is_dipgo']]
        win = dipgo[dipgo['is_dipwin']]
        lose = dipgo[~dipgo['is_dipwin']]

        for sub, c, lab in [(win, C_GREEN, 'WIN'), (lose, C_RED, 'LOSE')]:
            x = sub['dip_depth'].dropna()
            y = sub.loc[x.index, 'rest_move_from_lod'].dropna()
            valid = x.index.intersection(y.index)
            if len(valid) > 0:
                ax.scatter(x[valid], y[valid], color=c, alpha=0.7, s=50, label=f'{lab} (N={len(valid)})',
                          edgecolors='white', linewidth=0.5)

        ax.axhline(0, color=TEXT_COLOR, linestyle='--', alpha=0.3)
        ax.set_xlabel('Dip Depth (ADR)')
        ax.set_ylabel('Rest Move from LOD (ADR)')
        ax.set_title(f'{label}', fontsize=12)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'chart7_scatter_is_oos.png'), dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================
# GESAMTFAZIT
# ============================================================
def write_fazit(is_m, oos_m):
    rpt_h('GESAMTFAZIT')

    checks = []
    def check(name, is_val, oos_val, tolerance_pct=30, direction='same'):
        """Check if OOS is within tolerance of IS."""
        if pd.isna(is_val) or pd.isna(oos_val):
            checks.append((name, 'N/A', 'N/A', 'SKIP'))
            return
        if abs(is_val) < 0.01:
            checks.append((name, f'{is_val:.2f}', f'{oos_val:.2f}', 'OK (baseline ~0)'))
            return
        ratio = oos_val / is_val
        shrink = (1 - ratio) * 100
        if abs(shrink) <= tolerance_pct:
            verdict = 'REPLIZIERT'
        elif abs(shrink) <= 50:
            verdict = 'TEILWEISE'
        else:
            verdict = 'NICHT REPLIZIERT'
        checks.append((name, f'{is_val:.2f}', f'{oos_val:.2f}', f'{verdict} ({shrink:+.0f}%)'))

    check('DIP_GO Rate', is_m['dipgo_rate'], oos_m['dipgo_rate'], 25)
    check('Win-Rate', is_m['win_rate'], oos_m['win_rate'], 20)
    check('Depth Median', is_m['depth_median'], oos_m['depth_median'], 30)
    check('Rest Move Median', is_m['restmove_median'], oos_m['restmove_median'], 30)
    check('Bounce 5min', is_m['bounce_5min_median'], oos_m['bounce_5min_median'], 40)
    check('Fenster=Tages %', is_m['fenster_eq_tages_pct'], oos_m['fenster_eq_tages_pct'], 15)
    check('SL-025 Trigger', is_m['sl025_trigger_pct'], oos_m['sl025_trigger_pct'], 10)
    check('Recovery Median', is_m['recovery_median'], oos_m['recovery_median'], 40)

    rpt(f'\n  {"Metric":<24} {"IS":>10} {"OOS":>10} {"Verdict":>28}')
    rpt(f'  {"-"*24} {"-"*10} {"-"*10} {"-"*28}')
    for name, iv, ov, verdict in checks:
        rpt(f'  {name:<24} {iv:>10} {ov:>10} {verdict:>28}')

    n_rep = sum(1 for _, _, _, v in checks if 'REPLIZIERT' in v and 'NICHT' not in v)
    n_part = sum(1 for _, _, _, v in checks if 'TEILWEISE' in v)
    n_fail = sum(1 for _, _, _, v in checks if 'NICHT REPLIZIERT' in v)
    n_total = n_rep + n_part + n_fail

    rpt(f'\n  Repliziert: {n_rep}/{n_total}')
    rpt(f'  Teilweise:  {n_part}/{n_total}')
    rpt(f'  Gescheitert: {n_fail}/{n_total}')

    rpt_s('KERN-ERKENNTNISSE')

    # Key IS findings and their OOS status
    rpt(f'  1. DIP_GO Rate: IS={is_m["dipgo_rate"]:.1f}% -> OOS={oos_m["dipgo_rate"]:.1f}%')
    rpt(f'  2. Win-Rate: IS={is_m["win_rate"]:.1f}% -> OOS={oos_m["win_rate"]:.1f}%')
    rpt(f'  3. Timing Median: IS={mt(is_m["timing_median"])} -> OOS={mt(oos_m["timing_median"])}')
    rpt(f'  4. Dip Depth: IS={is_m["depth_median"]:.2f} ADR -> OOS={oos_m["depth_median"]:.2f} ADR')
    rpt(f'  5. Rest Move: IS={is_m["restmove_median"]:.3f} ADR -> OOS={oos_m["restmove_median"]:.3f} ADR')
    rpt(f'  6. Fenster=Tages: IS={is_m["fenster_eq_tages_pct"]:.0f}% -> OOS={oos_m["fenster_eq_tages_pct"]:.0f}%')
    rpt(f'  7. SL-025 Trigger: IS={is_m["sl025_trigger_pct"]:.0f}% -> OOS={oos_m["sl025_trigger_pct"]:.0f}%')

    rpt_s('EMPFEHLUNG')
    wr_ok = oos_m['win_rate'] > 60
    rm_ok = oos_m['restmove_median'] > 0.5
    fenster_ok = oos_m['fenster_eq_tages_pct'] > 70
    sl_reentry = oos_m['sl025_trigger_pct'] > 80

    if wr_ok and rm_ok and fenster_ok:
        rpt('  SETUP VALIDIERT.')
        rpt(f'  -> OOS Win-Rate {oos_m["win_rate"]:.0f}% (>{60}%)')
        rpt(f'  -> OOS Rest-Move {oos_m["restmove_median"]:.3f} ADR (>{0.5} ADR)')
        rpt(f'  -> SL unter Fenster-Low sicher in {oos_m["fenster_eq_tages_pct"]:.0f}% der Faelle')
        if sl_reentry:
            rpt(f'  -> ACHTUNG: {oos_m["sl025_trigger_pct"]:.0f}% waeren vorher ausgestoppt')
            rpt('     -> DIP_GO bleibt ein REENTRY-Setup nach SL-Hit!')
        rpt('  -> Backtest mit konkreten Entry/Exit-Regeln empfohlen')
    elif wr_ok and rm_ok:
        rpt('  SETUP TEILWEISE VALIDIERT.')
        rpt('  -> Win-Rate und Rest-Move OK, aber SL-Risiko beachten')
    else:
        rpt('  SETUP NICHT VALIDIERT.')
        rpt(f'  -> OOS Win-Rate: {oos_m["win_rate"]:.0f}%, Rest-Move: {oos_m["restmove_median"]:.3f} ADR')


# ============================================================
# MAIN
# ============================================================
def main():
    rpt('DIP & GO — PHASE 3: OOS-VALIDIERUNG')
    rpt(f'Date: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}')
    rpt(f'IS: date < {IS_CUTOFF}, OOS: date >= {IS_CUTOFF}')
    rpt(f'Window: {mt(WINDOW_LO)}-{mt(WINDOW_HI)}')

    # Load metadata
    meta = pd.read_parquet(META_PATH)
    meta['date'] = pd.to_datetime(meta['date'])
    df = meta[(meta['od_strength'] > 0.5) & (meta['od_direction'] == 'with_gap')].copy()

    # key_time, is_dipgo, is_dipwin
    df['key_time'] = np.where(df['gap_direction'] == 'up', df['lod_time'], df['hod_time'])
    df['key_price'] = np.where(df['gap_direction'] == 'up', df['rth_low'], df['rth_high'])
    df['is_dipgo'] = (df['key_time'] >= WINDOW_LO) & (df['key_time'] <= WINDOW_HI)
    df['is_dipwin'] = False
    mask_up = (df['gap_direction'] == 'up') & df['is_dipgo']
    mask_dn = (df['gap_direction'] == 'down') & df['is_dipgo']
    df.loc[mask_up, 'is_dipwin'] = df.loc[mask_up, 'rth_close'] > df.loc[mask_up, 'close_1000']
    df.loc[mask_dn, 'is_dipwin'] = df.loc[mask_dn, 'rth_close'] < df.loc[mask_dn, 'close_1000']
    df['is_IS'] = df['date'] < IS_CUTOFF

    rpt(f'\nTotal qualifying: {len(df)}')
    rpt(f'IS: {df["is_IS"].sum()}, OOS: {(~df["is_IS"]).sum()}')
    rpt(f'DIP_GO IS: {df[df["is_IS"]]["is_dipgo"].sum()}, OOS: {df[~df["is_IS"]]["is_dipgo"].sum()}')

    # Enrich IS
    rpt_h('ENRICHING IS')
    df_is = df[df['is_IS']].copy()
    df_is = enrich_split(df_is)

    # Enrich OOS
    rpt_h('ENRICHING OOS')
    df_oos = df[~df['is_IS']].copy()
    df_oos = enrich_split(df_oos)

    # Compute metrics
    is_m = compute_metrics(df_is, 'IS')
    oos_m = compute_metrics(df_oos, 'OOS')

    # Detailed reports
    detailed_report(df_is, 'IS', is_m)
    detailed_report(df_oos, 'OOS', oos_m)

    # Shrinkage
    print_shrinkage(is_m, oos_m)

    # Fazit
    write_fazit(is_m, oos_m)

    # Charts
    rpt_h('GENERATING CHARTS')
    chart1_timing_is_vs_oos(df_is, df_oos)
    rpt('  Chart 1 saved.')
    chart2_depth_is_vs_oos(df_is, df_oos)
    rpt('  Chart 2 saved.')
    chart3_shrinkage_bars(is_m, oos_m)
    rpt('  Chart 3 saved.')
    chart4_subwindow_wr(is_m, oos_m)
    rpt('  Chart 4 saved.')
    chart5_restmove_boxplot(df_is, df_oos)
    rpt('  Chart 5 saved.')
    chart6_fenster_vs_tages(df_is, df_oos)
    rpt('  Chart 6 saved.')
    chart7_scatter_is_oos(df_is, df_oos)
    rpt('  Chart 7 saved.')

    save_report()
    rpt('\n=== DONE ===')


if __name__ == '__main__':
    main()
