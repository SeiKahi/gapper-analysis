"""
DIP & GO — Phase 2: Timing & Tiefe des Pullbacks
==================================================
IS-only Analyse. Detaillierte Untersuchung:
  - WANN genau passiert der Dip?
  - WIE TIEF ist der Dip?
  - WIE SCHNELL dreht die Aktie?
  - Fenster-Low vs Tages-Low (echte SL-Frage)
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
OUT_DIR = os.path.join(BASE_DIR, 'results', 'dipgo_phase2')
os.makedirs(OUT_DIR, exist_ok=True)

IS_CUTOFF = '2024-01-01'
WINDOW_LO = 15   # 9:45 = 15 min after 9:30
WINDOW_HI = 50   # 10:20 = 50 min after 9:30

# Chart style
BG_COLOR = '#1a1a2e'
GRID_COLOR = '#2a2a4e'
TEXT_COLOR = '#e0e0e0'
C_CYAN = '#00d4ff'
C_RED = '#ff6b6b'
C_GREEN = '#51cf66'
C_YELLOW = '#ffd43b'
C_PURPLE = '#cc5de8'
C_ORANGE = '#ff922b'

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
# REPORT HELPERS
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

def rpt_sub(title):
    rpt('')
    rpt(f'--- {title} ---')

def save_report():
    path = os.path.join(OUT_DIR, 'dipgo_timing_depth.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f'\nReport saved: {path}')

def mt(minutes):
    """Minutes after 9:30 -> HH:MM."""
    h = int(9 + (30 + minutes) // 60)
    m = int((30 + minutes) % 60)
    return f'{h:02d}:{m:02d}'

def pct(n, total):
    return n / total * 100 if total > 0 else 0.0


# ============================================================
# LOAD 1-MIN BARS
# ============================================================
def load_rth_bars(ticker, date_str):
    """Load RTH 1-min bars for a ticker+date, return with minutes_after_930."""
    path = os.path.join(RAW_1MIN_DIR, ticker, f'{date_str}.parquet')
    if not os.path.exists(path):
        return None
    bars = pd.read_parquet(path)
    # Filter RTH
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
# MAIN DATA LOAD + ENRICHMENT
# ============================================================
def load_and_enrich():
    rpt_header('DEFINITION')

    meta = pd.read_parquet(META_PATH)
    meta['date'] = pd.to_datetime(meta['date'])

    # Filter
    df = meta[(meta['od_strength'] > 0.5) & (meta['od_direction'] == 'with_gap')].copy()
    df = df[df['date'] < IS_CUTOFF].copy()
    rpt(f'IS qualifying trades: {len(df)}')
    rpt(f'  GapUp: {(df["gap_direction"]=="up").sum()}')
    rpt(f'  GapDn: {(df["gap_direction"]=="down").sum()}')

    # key_time: LOD for GapUp, HOD for GapDn
    df['key_time'] = np.where(df['gap_direction'] == 'up', df['lod_time'], df['hod_time'])
    df['key_price'] = np.where(df['gap_direction'] == 'up', df['rth_low'], df['rth_high'])

    # DIP_GO: key_time in window (NO rest_move filter!)
    df['is_dipgo'] = (df['key_time'] >= WINDOW_LO) & (df['key_time'] <= WINDOW_HI)

    n_dipgo = df['is_dipgo'].sum()
    rpt(f'\nDIP_GO (LOD/HOD in {mt(WINDOW_LO)}-{mt(WINDOW_HI)}): {n_dipgo} ({pct(n_dipgo, len(df)):.1f}%)')

    # DIP_WIN vs DIP_LOSE
    # GapUp win: close > close_1000 (continued in gap direction after 10:00)
    # GapDn win: close < close_1000
    df['is_dipwin'] = False
    mask_up = (df['gap_direction'] == 'up') & df['is_dipgo']
    mask_dn = (df['gap_direction'] == 'down') & df['is_dipgo']
    df.loc[mask_up, 'is_dipwin'] = df.loc[mask_up, 'rth_close'] > df.loc[mask_up, 'close_1000']
    df.loc[mask_dn, 'is_dipwin'] = df.loc[mask_dn, 'rth_close'] < df.loc[mask_dn, 'close_1000']

    dipgo = df[df['is_dipgo']]
    n_win = dipgo['is_dipwin'].sum()
    n_lose = len(dipgo) - n_win
    rpt(f'  DIP_WIN:  {n_win} ({pct(n_win, n_dipgo):.1f}%)')
    rpt(f'  DIP_LOSE: {n_lose} ({pct(n_lose, n_dipgo):.1f}%)')

    for gd in ['up', 'down']:
        sub = dipgo[dipgo['gap_direction'] == gd]
        w = sub['is_dipwin'].sum()
        rpt(f'  Gap{gd.title()}: DIP_GO={len(sub)}, WIN={w} ({pct(w, len(sub)):.1f}%), LOSE={len(sub)-w}')

    # === Enrich from 1-min bars ===
    rpt_sub('Loading 1-min bars for DIP_GO trades...')

    # Pre-compute derived columns
    df['dip_depth'] = np.nan
    df['dip_vs_open'] = np.nan
    df['would_trigger_sl025'] = np.nan
    df['bounce_5min'] = np.nan
    df['bar_range_at_lod'] = np.nan
    df['lower_wick_pct'] = np.nan
    df['recovery_time'] = np.nan
    df['time_to_new_high'] = np.nan
    df['rest_move_from_lod'] = np.nan
    df['peak_after_lod'] = np.nan
    df['retention'] = np.nan
    df['local_low_is_true_lod'] = np.nan
    df['mae_after_window'] = np.nan

    count = 0
    for idx in df[df['is_dipgo']].index:
        row = df.loc[idx]
        ticker = row['ticker']
        date_str = str(row['date'].date())
        gap_dir = row['gap_direction']
        adr = row['adr_10']
        c935 = row['close_935']
        key_t = int(row['key_time'])

        bars = load_rth_bars(ticker, date_str)
        if bars is None or len(bars) < 10 or adr <= 0:
            continue

        if gap_dir == 'up':
            # LOD bar
            lod_bar_idx = bars['low'].idxmin()
            lod_bar = bars.loc[lod_bar_idx]
            lod_price = lod_bar['low']
            lod_min = lod_bar['min930']

            # dip_depth: how far from 9:35 entry to LOD
            df.at[idx, 'dip_depth'] = (c935 - lod_price) / adr

            # dip vs open
            df.at[idx, 'dip_vs_open'] = (row['rth_open'] - lod_price) / adr

            # would 0.25 ADR SL from 9:35 entry trigger before LOD?
            sl_level = c935 - 0.25 * adr
            bars_before_lod = bars[bars['min930'] <= lod_min]
            if len(bars_before_lod) > 0:
                # Check if any bar between entry and LOD would have triggered SL
                bars_after_entry = bars_before_lod[bars_before_lod['min930'] >= 5]  # after 9:35
                if len(bars_after_entry) > 0:
                    min_low_before = bars_after_entry['low'].min()
                    df.at[idx, 'would_trigger_sl025'] = 1.0 if min_low_before <= sl_level else 0.0

            # LOD bar candle analysis
            bar_h = lod_bar['high']
            bar_l = lod_bar['low']
            bar_c = lod_bar['close']
            bar_o = lod_bar['open']
            bar_range = bar_h - bar_l
            df.at[idx, 'bar_range_at_lod'] = bar_range / adr if adr > 0 else np.nan
            if bar_range > 0:
                # Lower wick for GapUp: how much of the bar is below the body
                body_low = min(bar_o, bar_c)
                df.at[idx, 'lower_wick_pct'] = (body_low - bar_l) / bar_range

            # Bounce 5 min after LOD
            bars_after_lod = bars[bars['min930'] > lod_min]
            if len(bars_after_lod) >= 5:
                close_5_after = bars_after_lod.iloc[4]['close']
                df.at[idx, 'bounce_5min'] = (close_5_after - lod_price) / adr
            elif len(bars_after_lod) > 0:
                close_last = bars_after_lod.iloc[-1]['close']
                df.at[idx, 'bounce_5min'] = (close_last - lod_price) / adr

            # Recovery time: first bar after LOD where close >= close_935
            if len(bars_after_lod) > 0:
                recovered = bars_after_lod[bars_after_lod['close'] >= c935]
                if len(recovered) > 0:
                    df.at[idx, 'recovery_time'] = recovered.iloc[0]['min930'] - lod_min

            # Time to new high: first bar after LOD where high > max(high 9:35-10:00)
            bars_935_to_1000 = bars[(bars['min930'] >= 5) & (bars['min930'] <= 30)]
            if len(bars_935_to_1000) > 0 and len(bars_after_lod) > 0:
                pre_dip_high = bars_935_to_1000['high'].max()
                new_high_bars = bars_after_lod[bars_after_lod['high'] > pre_dip_high]
                if len(new_high_bars) > 0:
                    df.at[idx, 'time_to_new_high'] = new_high_bars.iloc[0]['min930'] - lod_min

            # Rest move from LOD to close
            df.at[idx, 'rest_move_from_lod'] = (row['rth_close'] - lod_price) / adr

            # Peak after LOD
            if len(bars_after_lod) > 0:
                peak = bars_after_lod['high'].max()
                peak_move = (peak - lod_price) / adr
                df.at[idx, 'peak_after_lod'] = peak_move
                rest = (row['rth_close'] - lod_price) / adr
                if peak_move > 0:
                    df.at[idx, 'retention'] = rest / peak_move

            # Local low (window 9:45-10:20) vs true LOD (full day)
            window_bars = bars[(bars['min930'] >= WINDOW_LO) & (bars['min930'] <= WINDOW_HI)]
            if len(window_bars) > 0:
                local_low = window_bars['low'].min()
                true_lod = bars['low'].min()
                df.at[idx, 'local_low_is_true_lod'] = 1.0 if abs(local_low - true_lod) < 0.005 else 0.0

                # MAE after window: how much lower does it go after 10:20?
                bars_after_window = bars[bars['min930'] > WINDOW_HI]
                if len(bars_after_window) > 0:
                    min_after = bars_after_window['low'].min()
                    df.at[idx, 'mae_after_window'] = (local_low - min_after) / adr
                else:
                    df.at[idx, 'mae_after_window'] = 0.0

        else:  # GapDn — mirror everything
            # HOD bar
            hod_bar_idx = bars['high'].idxmax()
            hod_bar = bars.loc[hod_bar_idx]
            hod_price = hod_bar['high']
            hod_min = hod_bar['min930']

            # dip_depth: how far from 9:35 entry to HOD (for short, this is adverse)
            df.at[idx, 'dip_depth'] = (hod_price - c935) / adr

            # dip vs open
            df.at[idx, 'dip_vs_open'] = (hod_price - row['rth_open']) / adr

            # would SL trigger? For short: SL at entry + 0.25 ADR
            sl_level = c935 + 0.25 * adr
            bars_before_hod = bars[bars['min930'] <= hod_min]
            if len(bars_before_hod) > 0:
                bars_after_entry = bars_before_hod[bars_before_hod['min930'] >= 5]
                if len(bars_after_entry) > 0:
                    max_high_before = bars_after_entry['high'].max()
                    df.at[idx, 'would_trigger_sl025'] = 1.0 if max_high_before >= sl_level else 0.0

            # HOD bar candle analysis
            bar_h = hod_bar['high']
            bar_l = hod_bar['low']
            bar_c = hod_bar['close']
            bar_o = hod_bar['open']
            bar_range = bar_h - bar_l
            df.at[idx, 'bar_range_at_lod'] = bar_range / adr if adr > 0 else np.nan
            if bar_range > 0:
                # Upper wick for GapDn: selling pressure at high
                body_high = max(bar_o, bar_c)
                df.at[idx, 'lower_wick_pct'] = (bar_h - body_high) / bar_range

            # Bounce 5 min after HOD (bounce down = good for short)
            bars_after_hod = bars[bars['min930'] > hod_min]
            if len(bars_after_hod) >= 5:
                close_5_after = bars_after_hod.iloc[4]['close']
                df.at[idx, 'bounce_5min'] = (hod_price - close_5_after) / adr
            elif len(bars_after_hod) > 0:
                close_last = bars_after_hod.iloc[-1]['close']
                df.at[idx, 'bounce_5min'] = (hod_price - close_last) / adr

            # Recovery time: first bar where close <= close_935
            if len(bars_after_hod) > 0:
                recovered = bars_after_hod[bars_after_hod['close'] <= c935]
                if len(recovered) > 0:
                    df.at[idx, 'recovery_time'] = recovered.iloc[0]['min930'] - hod_min

            # Time to new low
            bars_935_to_1000 = bars[(bars['min930'] >= 5) & (bars['min930'] <= 30)]
            if len(bars_935_to_1000) > 0 and len(bars_after_hod) > 0:
                pre_dip_low = bars_935_to_1000['low'].min()
                new_low_bars = bars_after_hod[bars_after_hod['low'] < pre_dip_low]
                if len(new_low_bars) > 0:
                    df.at[idx, 'time_to_new_high'] = new_low_bars.iloc[0]['min930'] - hod_min

            # Rest move from HOD to close (positive = price dropped = good for short)
            df.at[idx, 'rest_move_from_lod'] = (hod_price - row['rth_close']) / adr

            # Peak after HOD (lowest low after = best for short)
            if len(bars_after_hod) > 0:
                trough = bars_after_hod['low'].min()
                peak_move = (hod_price - trough) / adr
                df.at[idx, 'peak_after_lod'] = peak_move
                rest = (hod_price - row['rth_close']) / adr
                if peak_move > 0:
                    df.at[idx, 'retention'] = rest / peak_move

            # Local high (window) vs true HOD
            window_bars = bars[(bars['min930'] >= WINDOW_LO) & (bars['min930'] <= WINDOW_HI)]
            if len(window_bars) > 0:
                local_high = window_bars['high'].max()
                true_hod = bars['high'].max()
                df.at[idx, 'local_low_is_true_lod'] = 1.0 if abs(local_high - true_hod) < 0.005 else 0.0

                # MAE after window: how much higher does it go after 10:20?
                bars_after_window = bars[bars['min930'] > WINDOW_HI]
                if len(bars_after_window) > 0:
                    max_after = bars_after_window['high'].max()
                    df.at[idx, 'mae_after_window'] = (max_after - local_high) / adr
                else:
                    df.at[idx, 'mae_after_window'] = 0.0

        count += 1
        if count % 10 == 0:
            print(f'  Processed {count}/{n_dipgo} DIP_GO trades...')

    rpt(f'  Enriched {count} DIP_GO trades from 1-min bars.')
    return df


# ============================================================
# SCHRITT 2: TIMING
# ============================================================
def step2_timing(df):
    rpt_header('TIMING: WANN PASSIERT DER DIP?')

    dipgo = df[df['is_dipgo']].copy()
    kt = dipgo['key_time']

    rpt_sub('A) LOD/HOD-Uhrzeit Verteilung (alle DIP_GO)')
    rpt(f'  N: {len(dipgo)}')
    rpt(f'  Mean:   {kt.mean():.1f} min = {mt(kt.mean())}')
    rpt(f'  Median: {kt.median():.1f} min = {mt(kt.median())}')
    rpt(f'  StdDev: {kt.std():.1f} min')
    rpt(f'  P25:    {kt.quantile(0.25):.0f} min = {mt(kt.quantile(0.25))}')
    rpt(f'  P75:    {kt.quantile(0.75):.0f} min = {mt(kt.quantile(0.75))}')
    rpt(f'  -> Der typische Dip passiert um {mt(kt.median())} +/- {kt.std():.0f} Min')

    rpt_sub('B) DIP_WIN vs DIP_LOSE Timing')
    win = dipgo[dipgo['is_dipwin']]
    lose = dipgo[~dipgo['is_dipwin']]
    rpt(f'  DIP_WIN  median LOD: {mt(win["key_time"].median())} (N={len(win)})')
    rpt(f'  DIP_LOSE median LOD: {mt(lose["key_time"].median())} (N={len(lose)})')
    if len(win) > 2 and len(lose) > 2:
        t, p = stats.ttest_ind(win['key_time'], lose['key_time'], equal_var=False)
        rpt(f'  Welch t-test: t={t:.2f}, p={p:.3f}')

    rpt_sub('C) Sub-Fenster Verteilung')
    sub_windows = [
        ('9:45-9:50', 15, 20),
        ('9:50-10:00', 20, 30),
        ('10:00-10:10', 30, 40),
        ('10:10-10:20', 40, 50),
    ]
    for label, lo, hi in sub_windows:
        n = ((kt >= lo) & (kt <= hi)).sum()
        n_w = ((win['key_time'] >= lo) & (win['key_time'] <= hi)).sum()
        n_l = n - n_w
        wr = pct(n_w, n) if n > 0 else 0
        rpt(f'  {label}: {n:>3} trades ({pct(n, len(dipgo)):>5.1f}%), WIN={n_w}, LOSE={n_l}, WinRate={wr:.0f}%')

    # By gap direction
    for gd in ['up', 'down']:
        sub = dipgo[dipgo['gap_direction'] == gd]
        if len(sub) > 0:
            rpt(f'  Gap{gd.title()}: median={mt(sub["key_time"].median())}, N={len(sub)}')


# ============================================================
# SCHRITT 3: TIEFE
# ============================================================
def step3_depth(df):
    rpt_header('TIEFE: WIE TIEF IST DER DIP?')

    dipgo = df[df['is_dipgo']].copy()
    dd = dipgo['dip_depth'].dropna()

    rpt_sub('A) Dip-Depth Verteilung (alle DIP_GO)')
    rpt(f'  N: {len(dd)}')
    rpt(f'  Mean:   {dd.mean():.3f} ADR')
    rpt(f'  Median: {dd.median():.3f} ADR')
    rpt(f'  StdDev: {dd.std():.3f} ADR')
    for q in [0.10, 0.25, 0.50, 0.75, 0.90]:
        rpt(f'  P{int(q*100):>2}:    {dd.quantile(q):.3f} ADR')
    rpt(f'  -> Der typische Dip geht {dd.median():.2f} +/- {dd.std():.2f} ADR unter den Entry')

    rpt_sub('B) DIP_WIN vs DIP_LOSE Depth')
    win = dipgo[dipgo['is_dipwin']]['dip_depth'].dropna()
    lose = dipgo[~dipgo['is_dipwin']]['dip_depth'].dropna()
    rpt(f'  DIP_WIN  median depth: {win.median():.3f} ADR (N={len(win)})')
    rpt(f'  DIP_LOSE median depth: {lose.median():.3f} ADR (N={len(lose)})')
    if len(win) > 2 and len(lose) > 2:
        t, p = stats.ttest_ind(win, lose, equal_var=False)
        rpt(f'  Welch t-test: t={t:.2f}, p={p:.3f}')
        rpt(f'  -> Verlierer-Dips {"tiefer" if lose.median() > win.median() else "flacher"} als Gewinner')

    rpt_sub('C) Dip relativ zum Open')
    dvo = dipgo['dip_vs_open'].dropna()
    rpt(f'  Median dip_vs_open: {dvo.median():.3f} ADR')
    n_under_open = (dvo > 0).sum()
    rpt(f'  % wo Dip UNTER Open liegt: {pct(n_under_open, len(dvo)):.1f}% ({n_under_open}/{len(dvo)})')
    n_above_open = (dvo <= 0).sum()
    rpt(f'  % wo Dip UEBER Open bleibt: {pct(n_above_open, len(dvo)):.1f}% ({n_above_open}/{len(dvo)})')

    rpt_sub('D) Dip vs 9:35 SL-Level (Entry - 0.25 ADR)')
    sl = dipgo['would_trigger_sl025'].dropna()
    n_triggered = int(sl.sum())
    rpt(f'  % die 0.25-ADR SL VOR dem Dip getriggert haetten: {pct(n_triggered, len(sl)):.1f}% ({n_triggered}/{len(sl)})')
    rpt(f'  -> {"ACHTUNG: Grosser Teil waere schon ausgestoppt!" if pct(n_triggered, len(sl)) > 50 else "Die meisten waeren noch im Trade."}')

    # SL trigger by depth bucket
    rpt_sub('D2) SL-Trigger Rate nach Dip-Depth Bucket')
    dipgo_sl = dipgo.dropna(subset=['dip_depth', 'would_trigger_sl025']).copy()
    if len(dipgo_sl) > 0:
        bins = [0, 0.15, 0.25, 0.40, 0.60, 1.0, 999]
        labels = ['0-0.15', '0.15-0.25', '0.25-0.40', '0.40-0.60', '0.60-1.0', '1.0+']
        dipgo_sl['depth_bucket'] = pd.cut(dipgo_sl['dip_depth'], bins=bins, labels=labels)
        rpt(f'  {"Depth Bucket":<14} {"N":>4} {"SL Hit":>6} {"%SL":>6}')
        for lab in labels:
            bucket = dipgo_sl[dipgo_sl['depth_bucket'] == lab]
            n = len(bucket)
            if n > 0:
                n_sl = int(bucket['would_trigger_sl025'].sum())
                rpt(f'  {lab:<14} {n:>4} {n_sl:>6} {pct(n_sl, n):>5.0f}%')


# ============================================================
# SCHRITT 4: RECOVERY SPEED
# ============================================================
def step4_recovery(df):
    rpt_header('RECOVERY: WIE SCHNELL DREHT DIE AKTIE?')

    dipgo = df[df['is_dipgo']].copy()
    win = dipgo[dipgo['is_dipwin']]
    lose = dipgo[~dipgo['is_dipwin']]

    rpt_sub('A) Recovery-Time (Minuten bis Entry-Level, nur DIP_WIN)')
    rt = win['recovery_time'].dropna()
    n_never = win['recovery_time'].isna().sum()
    rpt(f'  N mit Recovery: {len(rt)}')
    rpt(f'  N nie recovered: {n_never} ({pct(n_never, len(win)):.1f}%)')
    if len(rt) > 0:
        rpt(f'  Median: {rt.median():.0f} min')
        rpt(f'  Mean:   {rt.mean():.0f} min')
        rpt(f'  StdDev: {rt.std():.0f} min')
        rpt(f'  P25:    {rt.quantile(0.25):.0f} min')
        rpt(f'  P75:    {rt.quantile(0.75):.0f} min')
        rpt(f'  -> Nach dem Dip dauert es {rt.median():.0f} +/- {rt.std():.0f} Min bis Entry-Level')

    rpt_sub('A2) Time to New High (DIP_WIN)')
    tnh = win['time_to_new_high'].dropna()
    n_never_nh = win['time_to_new_high'].isna().sum()
    rpt(f'  N mit New High: {len(tnh)}')
    rpt(f'  N nie New High: {n_never_nh} ({pct(n_never_nh, len(win)):.1f}%)')
    if len(tnh) > 0:
        rpt(f'  Median: {tnh.median():.0f} min')
        rpt(f'  P75:    {tnh.quantile(0.75):.0f} min')

    rpt_sub('B) Bounce Speed (erste 5 Min nach LOD/HOD)')
    for label, sub in [('DIP_WIN', win), ('DIP_LOSE', lose), ('ALL DIP_GO', dipgo)]:
        b5 = sub['bounce_5min'].dropna()
        if len(b5) > 0:
            rpt(f'  {label:<12}: median bounce_5min = {b5.median():.3f} ADR (N={len(b5)})')
    # Test
    b_win = win['bounce_5min'].dropna()
    b_lose = lose['bounce_5min'].dropna()
    if len(b_win) > 2 and len(b_lose) > 2:
        t, p = stats.ttest_ind(b_win, b_lose, equal_var=False)
        rpt(f'  Welch t-test (WIN vs LOSE): t={t:.2f}, p={p:.3f}')
        rpt(f'  -> {"Gewinner bouncen SCHNELLER" if b_win.median() > b_lose.median() else "Kein klarer Unterschied"}')

    rpt_sub('C) LOD/HOD Bar Candle Analysis')
    for label, sub in [('DIP_WIN', win), ('DIP_LOSE', lose)]:
        br = sub['bar_range_at_lod'].dropna()
        lw = sub['lower_wick_pct'].dropna()
        rpt(f'  {label}:')
        if len(br) > 0:
            rpt(f'    Bar range at LOD/HOD: median {br.median():.3f} ADR')
        if len(lw) > 0:
            rpt(f'    Wick %: median {lw.median():.1%}')
    lw_win = win['lower_wick_pct'].dropna()
    lw_lose = lose['lower_wick_pct'].dropna()
    if len(lw_win) > 2 and len(lw_lose) > 2:
        t, p = stats.ttest_ind(lw_win, lw_lose, equal_var=False)
        rpt(f'  Wick Welch t-test: t={t:.2f}, p={p:.3f}')


# ============================================================
# SCHRITT 5: REST-MOVE
# ============================================================
def step5_rest_move(df):
    rpt_header('REST-MOVE NACH DEM DIP')

    dipgo = df[df['is_dipgo']].copy()

    rpt_sub('A) rest_move_from_lod Verteilung (alle DIP_GO)')
    rm = dipgo['rest_move_from_lod'].dropna()
    rpt(f'  N: {len(rm)}')
    rpt(f'  Mean:     {rm.mean():.3f} ADR')
    rpt(f'  Median:   {rm.median():.3f} ADR')
    rpt(f'  StdDev:   {rm.std():.3f} ADR')
    rpt(f'  Skewness: {rm.skew():.2f}')
    for q in [0.10, 0.25, 0.50, 0.75, 0.90]:
        rpt(f'  P{int(q*100):>2}:      {rm.quantile(q):.3f} ADR')
    n_pos = (rm > 0).sum()
    rpt(f'  % positiv (close > LOD): {pct(n_pos, len(rm)):.1f}%')

    rpt_sub('B) Peak nach LOD/HOD')
    pk = dipgo['peak_after_lod'].dropna()
    ret = dipgo['retention'].dropna()
    rpt(f'  Median peak_after_lod: {pk.median():.3f} ADR')
    rpt(f'  Mean peak_after_lod:   {pk.mean():.3f} ADR')
    rpt(f'  Median retention (close/peak): {ret.median():.1%}')

    rpt_sub('C) KRITISCH: Fenster-Low vs Tages-Low')
    ll = dipgo['local_low_is_true_lod'].dropna()
    n_true = int(ll.sum())
    n_not = len(ll) - n_true
    rpt(f'  Fenster-Low == Tages-Low:  {n_true} ({pct(n_true, len(ll)):.1f}%)')
    rpt(f'  Fenster-Low != Tages-Low:  {n_not} ({pct(n_not, len(ll)):.1f}%)')
    rpt(f'  -> In {pct(n_not, len(ll)):.0f}% der Faelle kommt ein tieferes Tief NACH dem Fenster!')

    mae = dipgo['mae_after_window'].dropna()
    mae_pos = mae[mae > 0]  # positive = price went below window low
    if len(mae_pos) > 0:
        rpt(f'\n  Bei Trades wo Fenster-Low != Tages-Low:')
        rpt(f'    N: {len(mae_pos)}')
        rpt(f'    Median weitere Tiefe: {mae_pos.median():.3f} ADR')
        rpt(f'    Mean weitere Tiefe:   {mae_pos.mean():.3f} ADR')
        rpt(f'    P75:                  {mae_pos.quantile(0.75):.3f} ADR')
        rpt(f'    P90:                  {mae_pos.quantile(0.90):.3f} ADR')
        rpt(f'    -> SL unter Fenster-Low muesste mind. {mae_pos.quantile(0.75):.2f} ADR Puffer haben')

    # Broken down by win/lose
    rpt_sub('C2) Fenster-Low vs Tages-Low: DIP_WIN vs DIP_LOSE')
    for label, sub in [('DIP_WIN', dipgo[dipgo['is_dipwin']]), ('DIP_LOSE', dipgo[~dipgo['is_dipwin']])]:
        ll_sub = sub['local_low_is_true_lod'].dropna()
        n_t = int(ll_sub.sum())
        rpt(f'  {label}: Fenster=Tages-Low in {pct(n_t, len(ll_sub)):.0f}% ({n_t}/{len(ll_sub)})')


# ============================================================
# SCHRITT 6: PARAMETER PROFILE
# ============================================================
def step6_parameters(df):
    rpt_header('PARAMETER: DIP_GO vs OTHER')

    dipgo = df[df['is_dipgo']].copy()
    other = df[~df['is_dipgo']].copy()

    # Parameters to compare
    params = [
        ('od_strength', 'od_strength'),
        ('gap_pct', 'gap_pct'),
        ('rvol_5', 'rvol_5'),
        ('adr_10', 'adr_10'),
        ('close_935', 'close_935'),
        ('gap_size_in_adr', 'gap_size_in_adr'),
        ('pm_rth5', 'pm_rth5'),
        ('od_body_pct', 'od_body_pct'),
        ('od_wick_ratio', 'od_wick_ratio'),
        ('first_candle_size', 'first_candle_size'),
        ('prior_return_5d', 'prior_return_5d'),
        ('prior_return_20d', 'prior_return_20d'),
        ('dist_from_20sma', 'dist_from_20sma'),
        ('rsi_14_prev', 'rsi_14_prev'),
        ('vix_level', 'vix_level'),
        ('rvol_30min', 'rvol_at_time_30min'),
        ('drift_30', None),  # computed
        ('dip_depth', 'dip_depth'),
        ('bounce_5min', 'bounce_5min'),
        ('lower_wick_pct', 'lower_wick_pct'),
        ('bar_range_at_lod', 'bar_range_at_lod'),
    ]

    # Compute drift_30 if needed
    if 'drift_30' not in df.columns:
        df['drift_30'] = (df['close_1000'] - df['close_935']) / df['adr_10']
        dipgo = df[df['is_dipgo']].copy()
        other = df[~df['is_dipgo']].copy()

    rpt_sub('A) DIP_GO vs OTHER')
    rpt(f'  DIP_GO: {len(dipgo)}, OTHER: {len(other)}')

    results_go_vs_other = []
    for label, col in params:
        col = col if col else label
        if col not in df.columns:
            continue
        d = dipgo[col].dropna()
        o = other[col].dropna()
        if len(d) < 3 or len(o) < 3:
            continue
        row = _compute_stats(d, o, label)
        if row:
            results_go_vs_other.append(row)

    res_df = pd.DataFrame(results_go_vs_other)
    res_df = res_df.sort_values('auc', key=lambda x: abs(x - 0.5), ascending=False)
    _print_stats_table(res_df)

    # DIP_WIN vs DIP_LOSE
    rpt_sub('B) DIP_WIN vs DIP_LOSE (innerhalb DIP_GO)')
    win = dipgo[dipgo['is_dipwin']]
    lose = dipgo[~dipgo['is_dipwin']]
    rpt(f'  DIP_WIN: {len(win)}, DIP_LOSE: {len(lose)}')

    results_wl = []
    for label, col in params:
        col = col if col else label
        if col not in df.columns:
            continue
        w = win[col].dropna()
        l = lose[col].dropna()
        if len(w) < 3 or len(l) < 3:
            continue
        row = _compute_stats(w, l, label)
        if row:
            results_wl.append(row)

    if results_wl:
        res_wl = pd.DataFrame(results_wl)
        res_wl = res_wl.sort_values('auc', key=lambda x: abs(x - 0.5), ascending=False)
        _print_stats_table(res_wl)
    else:
        rpt('  Too few trades for WIN vs LOSE comparison.')
        res_wl = pd.DataFrame()

    return res_df, res_wl


def _compute_stats(group1, group2, label):
    if len(group1) < 3 or len(group2) < 3:
        return None
    m1, m2 = group1.mean(), group2.mean()
    diff = m1 - m2
    try:
        t, p = stats.ttest_ind(group1, group2, equal_var=False)
    except:
        p = np.nan
    pooled = np.sqrt((group1.var() * (len(group1)-1) + group2.var() * (len(group2)-1)) / (len(group1)+len(group2)-2))
    cd = diff / pooled if pooled > 0 else np.nan
    try:
        labels = np.concatenate([np.ones(len(group1)), np.zeros(len(group2))])
        scores = np.concatenate([group1.values, group2.values])
        auc = roc_auc_score(labels, scores)
    except:
        auc = np.nan
    return {'parameter': label, 'mean_g1': m1, 'mean_g2': m2, 'diff': diff,
            'p_value': p, 'cohens_d': cd, 'auc': auc,
            'n1': len(group1), 'n2': len(group2)}


def _print_stats_table(res_df):
    rpt(f'\n  {"Parameter":<20} {"Mean G1":>8} {"Mean G2":>8} {"Diff":>8} {"p-val":>8} {"Cohen d":>8} {"AUC":>6}')
    rpt(f'  {"-"*20} {"-"*8} {"-"*8} {"-"*8} {"-"*8} {"-"*8} {"-"*6}')
    for _, r in res_df.iterrows():
        ps = f'{r["p_value"]:.4f}' if r['p_value'] >= 0.0001 else '<.0001'
        rpt(f'  {r["parameter"]:<20} {r["mean_g1"]:>8.3f} {r["mean_g2"]:>8.3f} {r["diff"]:>+8.3f} {ps:>8} {r["cohens_d"]:>+8.3f} {r["auc"]:>6.3f}')


# ============================================================
# FAZIT
# ============================================================
def write_fazit(df):
    rpt_header('IS FAZIT')

    dipgo = df[df['is_dipgo']]
    win = dipgo[dipgo['is_dipwin']]
    lose = dipgo[~dipgo['is_dipwin']]

    rpt(f'  DIP_GO Trades: {len(dipgo)} von {len(df)} ({pct(len(dipgo), len(df)):.1f}%)')
    rpt(f'  Win-Rate (close > close_1000 in Gap-Richtung): {pct(len(win), len(dipgo)):.1f}%')
    rpt('')

    kt = dipgo['key_time']
    dd = dipgo['dip_depth'].dropna()
    rpt(f'  Timing: Median LOD/HOD bei {mt(kt.median())} +/- {kt.std():.0f} Min')
    rpt(f'  Tiefe: Median Dip = {dd.median():.2f} ADR unter Entry')
    rpt('')

    sl = dipgo['would_trigger_sl025'].dropna()
    rpt(f'  SL-025 Trigger vor Dip: {pct(sl.sum(), len(sl)):.0f}%')
    ll = dipgo['local_low_is_true_lod'].dropna()
    rpt(f'  Fenster-Low == Tages-Low: {pct(ll.sum(), len(ll)):.0f}%')
    rpt(f'  -> In {100 - pct(ll.sum(), len(ll)):.0f}% kommt tieferes Tief nach Fenster')
    rpt('')

    rm = dipgo['rest_move_from_lod'].dropna()
    rpt(f'  Rest-Move ab LOD: median {rm.median():.3f} ADR, {pct((rm > 0).sum(), len(rm)):.0f}% positiv')
    b5 = dipgo['bounce_5min'].dropna()
    rpt(f'  Bounce 5min: median {b5.median():.3f} ADR')
    rt = win['recovery_time'].dropna()
    if len(rt) > 0:
        rpt(f'  Recovery-Time (DIP_WIN): median {rt.median():.0f} Min')
    rpt('')

    rpt('  Vorlaeufige Einschaetzung (NOCH OHNE OOS!):')
    win_rate = pct(len(win), len(dipgo))
    local_true = pct(ll.sum(), len(ll))
    if win_rate > 65 and local_true > 60:
        rpt('  -> DIP_GO ist vielversprechend: Hohe Win-Rate, Fenster-Low oft = Tages-Low')
    elif win_rate > 50:
        rpt('  -> DIP_GO hat Potenzial, aber SL-Management ist kritisch')
    else:
        rpt('  -> DIP_GO Win-Rate unter 50% — Setup fragwuerdig')

    sl_rate = pct(sl.sum(), len(sl))
    if sl_rate > 60:
        rpt(f'  -> PROBLEM: {sl_rate:.0f}% waeren mit 0.25-ADR SL schon ausgestoppt')
        rpt('     -> DIP_GO ist primaer ein REENTRY nach SL-Hit!')
    rpt('')
    rpt('  NAECHSTER SCHRITT: OOS-Validierung in Phase 3.')


# ============================================================
# CHARTS
# ============================================================
def chart1_lod_timing(df):
    dipgo = df[df['is_dipgo']]
    win = dipgo[dipgo['is_dipwin']]
    lose = dipgo[~dipgo['is_dipwin']]

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle('CHART 1: LOD/HOD Timing (DIP_GO, IS only)', fontsize=14, fontweight='bold')

    bins = np.arange(WINDOW_LO, WINDOW_HI + 2, 1)
    ax.hist(win['key_time'], bins=bins, color=C_GREEN, alpha=0.7, label=f'DIP_WIN (N={len(win)})', edgecolor='none')
    ax.hist(lose['key_time'], bins=bins, color=C_RED, alpha=0.7, label=f'DIP_LOSE (N={len(lose)})', edgecolor='none',
            bottom=np.histogram(win['key_time'], bins=bins)[0])

    kt = dipgo['key_time']
    med = kt.median()
    std = kt.std()
    ax.axvline(med, color=C_YELLOW, linestyle='-', linewidth=2, label=f'Median {mt(med)}')
    ax.axvline(med - std, color=C_YELLOW, linestyle='--', linewidth=1, alpha=0.6)
    ax.axvline(med + std, color=C_YELLOW, linestyle='--', linewidth=1, alpha=0.6, label=f'+/- 1 StdDev ({std:.0f} min)')

    tick_pos = np.arange(WINDOW_LO, WINDOW_HI + 1, 5)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([mt(t) for t in tick_pos], rotation=45)
    ax.set_xlabel('LOD/HOD Time')
    ax.set_ylabel('Count')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'chart1_lod_timing.png'), dpi=300, bbox_inches='tight')
    plt.close()


def chart2_dip_depth(df):
    dipgo = df[df['is_dipgo']]
    win = dipgo[dipgo['is_dipwin']]
    lose = dipgo[~dipgo['is_dipwin']]

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle('CHART 2: Dip Depth Distribution (DIP_GO, IS only)', fontsize=14, fontweight='bold')

    dd_win = win['dip_depth'].dropna()
    dd_lose = lose['dip_depth'].dropna()

    all_dd = pd.concat([dd_win, dd_lose])
    max_d = min(all_dd.quantile(0.98), 3.0)
    bins = np.arange(0, max_d + 0.05, 0.05)

    h_win, _ = np.histogram(dd_win, bins=bins)
    h_lose, _ = np.histogram(dd_lose, bins=bins)

    ax.bar(bins[:-1], h_win, width=0.05, color=C_GREEN, alpha=0.7, label=f'DIP_WIN (N={len(dd_win)})', align='edge')
    ax.bar(bins[:-1], h_lose, width=0.05, color=C_RED, alpha=0.7, label=f'DIP_LOSE (N={len(dd_lose)})', align='edge',
           bottom=h_win)

    ax.axvline(0.25, color=C_YELLOW, linestyle='--', linewidth=2, label='0.25 ADR (normal SL)')
    ax.set_xlabel('Dip Depth (ADR)')
    ax.set_ylabel('Count')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'chart2_dip_depth.png'), dpi=300, bbox_inches='tight')
    plt.close()


def chart3_scatter_depth_vs_rest(df):
    dipgo = df[df['is_dipgo']].copy()
    win = dipgo[dipgo['is_dipwin']]
    lose = dipgo[~dipgo['is_dipwin']]

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle('CHART 3: Dip Depth vs Rest Move (IS only)', fontsize=14, fontweight='bold')

    x_w = win['dip_depth'].dropna()
    y_w = win.loc[x_w.index, 'rest_move_from_lod']
    valid_w = x_w.notna() & y_w.notna()
    x_w, y_w = x_w[valid_w], y_w[valid_w]

    x_l = lose['dip_depth'].dropna()
    y_l = lose.loc[x_l.index, 'rest_move_from_lod']
    valid_l = x_l.notna() & y_l.notna()
    x_l, y_l = x_l[valid_l], y_l[valid_l]

    ax.scatter(x_w, y_w, color=C_GREEN, alpha=0.7, s=50, label=f'DIP_WIN (N={len(x_w)})', edgecolors='white', linewidth=0.5)
    ax.scatter(x_l, y_l, color=C_RED, alpha=0.7, s=50, label=f'DIP_LOSE (N={len(x_l)})', edgecolors='white', linewidth=0.5)

    # Regression on all
    x_all = pd.concat([x_w, x_l])
    y_all = pd.concat([y_w, y_l])
    if len(x_all) > 5:
        slope, intercept, r, p, se = stats.linregress(x_all, y_all)
        x_line = np.linspace(x_all.min(), x_all.max(), 100)
        ax.plot(x_line, slope * x_line + intercept, color=C_YELLOW, linewidth=2,
                label=f'Regression R2={r**2:.3f}')

    ax.axhline(0, color=TEXT_COLOR, linestyle='--', alpha=0.3)
    ax.set_xlabel('Dip Depth (ADR from Entry)')
    ax.set_ylabel('Rest Move from LOD to Close (ADR)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'chart3_depth_vs_restmove.png'), dpi=300, bbox_inches='tight')
    plt.close()


def chart4_recovery_time(df):
    win = df[(df['is_dipgo']) & (df['is_dipwin'])]
    rt = win['recovery_time'].dropna()

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle('CHART 4: Recovery Time (DIP_WIN, IS only)', fontsize=14, fontweight='bold')

    if len(rt) > 0:
        max_rt = min(rt.quantile(0.95), 200)
        bins = np.arange(0, max_rt + 5, 5)
        ax.hist(rt, bins=bins, color=C_CYAN, alpha=0.7, edgecolor='none')
        ax.axvline(rt.median(), color=C_YELLOW, linestyle='-', linewidth=2, label=f'Median: {rt.median():.0f} min')
        ax.axvline(rt.quantile(0.75), color=C_ORANGE, linestyle='--', linewidth=1.5, label=f'P75: {rt.quantile(0.75):.0f} min')
        ax.set_xlabel('Minutes after LOD/HOD to recover Entry level')
        ax.set_ylabel('Count')
        n_never = win['recovery_time'].isna().sum()
        ax.text(0.95, 0.95, f'Never recovered: {n_never} ({pct(n_never, len(win)):.0f}%)',
                transform=ax.transAxes, ha='right', va='top', fontsize=11, color=C_RED,
                bbox=dict(boxstyle='round', facecolor=BG_COLOR, edgecolor=C_RED))
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'chart4_recovery_time.png'), dpi=300, bbox_inches='tight')
    plt.close()


def chart5_bounce_speed(df):
    dipgo = df[df['is_dipgo']]
    win = dipgo[dipgo['is_dipwin']]['bounce_5min'].dropna()
    lose = dipgo[~dipgo['is_dipwin']]['bounce_5min'].dropna()

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('CHART 5: Bounce Speed (5min after LOD/HOD, IS only)', fontsize=14, fontweight='bold')

    data = [win, lose]
    labels = [f'DIP_WIN\n(N={len(win)})', f'DIP_LOSE\n(N={len(lose)})']
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False,
                   medianprops=dict(color=C_YELLOW, linewidth=2))
    for patch, color in zip(bp['boxes'], [C_GREEN, C_RED]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.axhline(0, color=TEXT_COLOR, linestyle='--', alpha=0.3)
    ax.set_ylabel('Bounce in 5min (ADR)')
    for i, d in enumerate(data):
        if len(d) > 0:
            ax.text(i + 1, d.median() + 0.01, f'{d.median():.3f}', ha='center', fontsize=11, color=C_YELLOW)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'chart5_bounce_speed.png'), dpi=300, bbox_inches='tight')
    plt.close()


def chart6_auc_ranking(res_go_other, res_wl):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('CHART 6: Parameter AUC Ranking (IS only)', fontsize=14, fontweight='bold')

    # DIP_GO vs OTHER
    ax = axes[0]
    if len(res_go_other) > 0:
        data = res_go_other.sort_values('auc', ascending=True)
        y = np.arange(len(data))
        ax.barh(y, data['auc'], color=C_CYAN, alpha=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(data['parameter'], fontsize=9)
        ax.axvline(0.5, color=C_RED, linestyle='--', linewidth=1)
        ax.set_xlabel('AUC')
        ax.set_title('DIP_GO vs OTHER', fontsize=11)
        ax.set_xlim(0.2, 0.8)

    # DIP_WIN vs DIP_LOSE
    ax = axes[1]
    if len(res_wl) > 0:
        data = res_wl.sort_values('auc', ascending=True)
        y = np.arange(len(data))
        ax.barh(y, data['auc'], color=C_GREEN, alpha=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(data['parameter'], fontsize=9)
        ax.axvline(0.5, color=C_RED, linestyle='--', linewidth=1)
        ax.set_xlabel('AUC')
        ax.set_title('DIP_WIN vs DIP_LOSE', fontsize=11)
        ax.set_xlim(0.2, 0.8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'chart6_auc_ranking.png'), dpi=300, bbox_inches='tight')
    plt.close()


def chart7_fenster_vs_tages(df):
    dipgo = df[df['is_dipgo']]
    ll = dipgo['local_low_is_true_lod'].dropna()
    n_true = int(ll.sum())
    n_not = len(ll) - n_true

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('CHART 7: Fenster-Low vs Tages-Low (IS only)', fontsize=14, fontweight='bold')

    # Pie chart
    ax = axes[0]
    sizes = [n_true, n_not]
    labels_pie = [f'Fenster = Tages-Low\n{n_true} ({pct(n_true, len(ll)):.0f}%)',
                  f'Tieferes Tief danach\n{n_not} ({pct(n_not, len(ll)):.0f}%)']
    colors_pie = [C_GREEN, C_RED]
    wedges, texts = ax.pie(sizes, labels=labels_pie, colors=colors_pie,
                            startangle=90, textprops={'color': TEXT_COLOR, 'fontsize': 11})
    for w in wedges:
        w.set_edgecolor(BG_COLOR)
        w.set_linewidth(2)
    ax.set_title('Wie oft ist Fenster-Low = Tages-Low?', fontsize=11)

    # MAE histogram for those where fenster != tages
    ax = axes[1]
    mae = dipgo['mae_after_window'].dropna()
    mae_pos = mae[mae > 0]
    if len(mae_pos) > 0:
        max_mae = min(mae_pos.quantile(0.95), 2.0)
        bins_mae = np.arange(0, max_mae + 0.05, 0.05)
        ax.hist(mae_pos, bins=bins_mae, color=C_RED, alpha=0.7, edgecolor='none')
        ax.axvline(mae_pos.median(), color=C_YELLOW, linestyle='-', linewidth=2,
                   label=f'Median: {mae_pos.median():.3f} ADR')
        ax.axvline(mae_pos.quantile(0.75), color=C_ORANGE, linestyle='--', linewidth=1.5,
                   label=f'P75: {mae_pos.quantile(0.75):.3f} ADR')
        ax.set_xlabel('Further Depth below Fenster-Low (ADR)')
        ax.set_ylabel('Count')
        ax.set_title(f'MAE after Window (N={len(mae_pos)} trades)', fontsize=11)
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'chart7_fenster_vs_tages.png'), dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================
# MAIN
# ============================================================
def main():
    rpt('DIP & GO — PHASE 2: TIMING & TIEFE')
    rpt(f'Date: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}')
    rpt(f'IS only (date < {IS_CUTOFF})')
    rpt(f'Window: {mt(WINDOW_LO)}-{mt(WINDOW_HI)}')

    df = load_and_enrich()
    step2_timing(df)
    step3_depth(df)
    step4_recovery(df)
    step5_rest_move(df)
    res_go_other, res_wl = step6_parameters(df)
    write_fazit(df)

    rpt_header('GENERATING CHARTS')
    chart1_lod_timing(df)
    rpt('  Chart 1 saved.')
    chart2_dip_depth(df)
    rpt('  Chart 2 saved.')
    chart3_scatter_depth_vs_rest(df)
    rpt('  Chart 3 saved.')
    chart4_recovery_time(df)
    rpt('  Chart 4 saved.')
    chart5_bounce_speed(df)
    rpt('  Chart 5 saved.')
    chart6_auc_ranking(res_go_other, res_wl)
    rpt('  Chart 6 saved.')
    chart7_fenster_vs_tages(df)
    rpt('  Chart 7 saved.')

    save_report()
    rpt('\n=== DONE ===')


if __name__ == '__main__':
    main()
