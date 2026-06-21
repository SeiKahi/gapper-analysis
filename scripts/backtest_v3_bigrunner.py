"""
Backtest v3: Big Runner Capture — 6 Approaches + Control
=========================================================
7 exit management strategies tested against identical trade entries.
All share the same qualification: od_strength > 0.5 AND od_direction == 'with_gap'.

Approaches:
  CTRL — Original TRAIL_D (Peak - 0.5R, SL 0.25 ADR)
  A    — Split: Half TRAIL_D + Half Hold-to-Close
  B    — Wider SL (0.5 ADR) + Trail Peak-0.5R + 3-Tier at 10:00
  C    — Time-Stop (BE + Hold to 15:55)
  D    — Split: Half TRAIL_D + Half wide trail (Peak-2R) + 10:00 management
  E    — Scaled: 50% at +1R, 25% at +2R, 25% Hold-to-Close
  F    — Wider Trail (Peak - 1R = Peak - 0.25 ADR) + 10:00 management

Output: results/backtest_v3/
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats as sp_stats
import sys
import os
import time
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors

# ============================================================
# CONSTANTS
# ============================================================
RAW_1MIN_DIR = Path('data/raw_1min')
RAW_1SEC_DIR = Path('data/raw_1sec')
RESULTS_DIR = Path('results/backtest_v3')
META_PATH = Path('data/metadata/metadata_v8_5.parquet')
V2_TRADES_PATH = Path('results/backtest_v2/all_trades_v2.parquet')

RISK = 1000.0
SL_FRAC_025 = 0.25
SL_FRAC_050 = 0.50
MC_ITERS = 1000
SLIPPAGE = 0.00

IS_START = '2021-02-21'
IS_END = '2023-12-31'
OOS_START = '2024-01-01'
OOS_END = '2026-02-13'

APPROACHES = ['CTRL', 'A', 'B', 'C', 'D', 'E', 'F']

# Colors
C_BG = '#1a1a2e'
ACOLS = {
    'CTRL': 'cyan', 'A': '#06d6a0', 'B': '#ffd166',
    'C': '#ff9f1c', 'D': '#e040fb', 'E': 'white', 'F': '#ff6b6b',
}

# Walk-forward windows
WF_WINDOWS = [
    ('2021-H1', '2021-02-21', '2021-06-30'),
    ('2021-H2', '2021-07-01', '2021-12-31'),
    ('2022-H1', '2022-01-01', '2022-06-30'),
    ('2022-H2', '2022-07-01', '2022-12-31'),
    ('2023-H1', '2023-01-01', '2023-06-30'),
    ('2023-H2', '2023-07-01', '2023-12-31'),
    ('2024-H1', '2024-01-01', '2024-06-30'),
    ('2024-H2', '2024-07-01', '2024-12-31'),
    ('2025-H1', '2025-01-01', '2025-06-30'),
    ('2025-H2', '2025-07-01', '2026-02-13'),
]

# Big Runner: MFE >= 5R (peak_r from CTRL sim)
BR_MFE_THRESHOLD = 5.0
# Big Runner entry: hold-to-close PnL >= 3R (approach C)
BR_ENTRY_THRESHOLD = 3.0

# ============================================================
# 1-SEC CACHE
# ============================================================
_sec_cache = {}


def load_1sec(ticker, date_str):
    key = (ticker, date_str)
    if key not in _sec_cache:
        path = RAW_1SEC_DIR / ticker / f"{date_str}.parquet"
        _sec_cache[key] = pd.read_parquet(path) if path.exists() else None
    return _sec_cache[key]


# ============================================================
# HELPERS
# ============================================================
def compute_drift_30(meta_row, entry, adr):
    """Compute direction-aware 30min drift (no lookahead)."""
    c1000 = meta_row.get('close_1000', np.nan)
    gap_dir = meta_row.get('gap_direction', 'up')
    if pd.isna(c1000) or pd.isna(entry) or adr <= 0:
        return np.nan
    if gap_dir == 'up':
        return (c1000 - entry) / adr
    else:
        return (entry - c1000) / adr


def compute_trail_tier(drift_30, rvol_30, od_str):
    """Compute 10:00 trail tier. Returns 1 or 3."""
    if pd.isna(drift_30) or pd.isna(rvol_30):
        return 1
    cond_a = (drift_30 > 0.25 and rvol_30 > 5 and
              pd.notna(od_str) and od_str > 1.0)
    cond_b = drift_30 > 0.50 and rvol_30 > 5
    if cond_a or cond_b:
        return 3
    return 1


def _stop_hit(trade_dir, low, high, sl):
    """Check if stop is hit on a bar."""
    if trade_dir == 'long':
        return low <= sl
    else:
        return high >= sl


def _update_peak(trade_dir, peak, high, low):
    """Update peak favorable price. Returns new peak."""
    if trade_dir == 'long':
        return max(peak, high)
    else:
        return min(peak, low)


def _compute_mfe(trade_dir, entry, high, low, prev_mfe):
    """Update maximum favorable excursion (absolute $)."""
    if trade_dir == 'long':
        return max(prev_mfe, high - entry)
    else:
        return max(prev_mfe, entry - low)


def _trail_level(trade_dir, peak, trail_dist):
    """Compute trail stop level from peak and distance."""
    if trade_dir == 'long':
        return peak - trail_dist
    else:
        return peak + trail_dist


def _ratchet_sl(trade_dir, current_sl, new_sl):
    """Trail only moves forward (tighter). Returns new SL."""
    if trade_dir == 'long':
        return max(current_sl, new_sl)
    else:
        return min(current_sl, new_sl)


# ============================================================
# PHASE 1: Before +1R (shared for same-SL approaches)
# ============================================================
def phase1_simulate(bars_post, entry, sl_level, trade_dir, one_r,
                    one_r_level, sec_bars):
    """
    Simulate until +1R reached, initial SL hit, or timeout.
    Shared for all approaches with the same SL.

    Returns dict:
        outcome: 'sl_hit' | 'timeout' | '1r_reached' | '1r_in_zoom'
        peak: peak favorable price
        mfe: max favorable excursion ($)
        next_bar_idx: index in bars_post for Phase 2 start
        exit_price, exit_time: if sl_hit or timeout
        zoom_used, zoom_changed: bools
        zoom_remaining_secs: DataFrame (if 1r_in_zoom)
        zoom_bar_low, zoom_bar_high: (if 1r_in_zoom)
        zoom_bar_time: str (if 1r_in_zoom)
    """
    peak = entry
    mfe = 0.0
    zoom_used = False
    zoom_changed = False

    post = bars_post[
        (bars_post['time_et'] >= '09:36') &
        (bars_post['time_et'] <= '15:55')
    ]
    if len(post) == 0:
        return {'outcome': 'skip'}

    post_idx = post.index.tolist()

    for pos, idx in enumerate(post_idx):
        bar = post.loc[idx]
        bt = str(bar['time_et'])[:5]
        bh = bar['high']
        bl = bar['low']

        stop_possible = _stop_hit(trade_dir, bl, bh, sl_level)

        if trade_dir == 'long':
            can_reach_1r = bh >= one_r_level
            fav_new = bh > peak
        else:
            can_reach_1r = bl <= one_r_level
            fav_new = bl < peak

        ambig = stop_possible and (fav_new or can_reach_1r)

        if ambig:
            resolved = False
            can_zoom = (sec_bars is not None and bt >= '09:30' and bt < '09:45')
            if can_zoom:
                mins = sec_bars[sec_bars['time_et'].str[:5] == bt]
                if len(mins) > 0:
                    zoom_used = True
                    min_indices = mins.index.tolist()
                    for si, sec_idx in enumerate(min_indices):
                        s = mins.loc[sec_idx]
                        # Check stop first
                        if _stop_hit(trade_dir, s['low'], s['high'], sl_level):
                            return {
                                'outcome': 'sl_hit',
                                'peak': peak, 'mfe': mfe,
                                'exit_price': sl_level,
                                'exit_time': s['time_et'],
                                'zoom_used': True,
                                'zoom_changed': True,
                            }
                        # Update peak
                        peak = _update_peak(trade_dir, peak, s['high'], s['low'])
                        mfe = _compute_mfe(trade_dir, entry, s['high'], s['low'], mfe)
                        # Check +1R
                        if trade_dir == 'long' and peak >= one_r_level:
                            remaining_secs = mins.iloc[si + 1:]
                            return {
                                'outcome': '1r_in_zoom',
                                'peak': peak, 'mfe': mfe,
                                'next_bar_idx': pos + 1,
                                'zoom_remaining_secs': remaining_secs,
                                'zoom_bar_low': bl,
                                'zoom_bar_high': bh,
                                'zoom_bar_time': bt,
                                'zoom_used': True,
                                'zoom_changed': True,
                            }
                        if trade_dir == 'short' and peak <= one_r_level:
                            remaining_secs = mins.iloc[si + 1:]
                            return {
                                'outcome': '1r_in_zoom',
                                'peak': peak, 'mfe': mfe,
                                'next_bar_idx': pos + 1,
                                'zoom_remaining_secs': remaining_secs,
                                'zoom_bar_low': bl,
                                'zoom_bar_high': bh,
                                'zoom_bar_time': bt,
                                'zoom_used': True,
                                'zoom_changed': True,
                            }

                    # After zoom: re-check if SL still hit
                    if _stop_hit(trade_dir, bl, bh, sl_level):
                        return {
                            'outcome': 'sl_hit',
                            'peak': peak, 'mfe': mfe,
                            'exit_price': sl_level,
                            'exit_time': bt,
                            'zoom_used': True,
                            'zoom_changed': False,
                        }
                    # Check if +1R reached (bar data after zoom)
                    if trade_dir == 'long' and peak >= one_r_level:
                        return {
                            'outcome': '1r_reached',
                            'peak': peak, 'mfe': mfe,
                            'next_bar_idx': pos + 1,
                            'zoom_used': True,
                            'zoom_changed': True,
                        }
                    if trade_dir == 'short' and peak <= one_r_level:
                        return {
                            'outcome': '1r_reached',
                            'peak': peak, 'mfe': mfe,
                            'next_bar_idx': pos + 1,
                            'zoom_used': True,
                            'zoom_changed': True,
                        }
                    resolved = True  # zoom handled but neither SL nor +1R

            if not resolved:
                # Conservative: SL first
                return {
                    'outcome': 'sl_hit',
                    'peak': peak, 'mfe': mfe,
                    'exit_price': sl_level,
                    'exit_time': bt,
                    'zoom_used': zoom_used,
                    'zoom_changed': zoom_changed,
                }

        elif stop_possible:
            return {
                'outcome': 'sl_hit',
                'peak': peak, 'mfe': mfe,
                'exit_price': sl_level,
                'exit_time': bt,
                'zoom_used': zoom_used,
                'zoom_changed': zoom_changed,
            }
        else:
            # No stop — update peak and check +1R
            peak = _update_peak(trade_dir, peak, bh, bl)
            mfe = _compute_mfe(trade_dir, entry, bh, bl, mfe)

            if trade_dir == 'long' and peak >= one_r_level:
                return {
                    'outcome': '1r_reached',
                    'peak': peak, 'mfe': mfe,
                    'next_bar_idx': pos + 1,
                    'zoom_used': zoom_used,
                    'zoom_changed': zoom_changed,
                }
            if trade_dir == 'short' and peak <= one_r_level:
                return {
                    'outcome': '1r_reached',
                    'peak': peak, 'mfe': mfe,
                    'next_bar_idx': pos + 1,
                    'zoom_used': zoom_used,
                    'zoom_changed': zoom_changed,
                }

    # Timeout: +1R never reached, SL never hit
    last = post.iloc[-1]
    return {
        'outcome': 'timeout',
        'peak': peak, 'mfe': mfe,
        'exit_price': last['close'],
        'exit_time': str(last['time_et'])[:5],
        'zoom_used': zoom_used,
        'zoom_changed': zoom_changed,
    }


# ============================================================
# PHASE 2a: Trailing Stop (for CTRL, A_T1, B, D_T1, D_T2, F)
# ============================================================
def phase2_trail(bars_remaining, entry, trade_dir, one_r, init_peak,
                 shares, trail_width_r, sec_bars, adr, meta_row,
                 ten_oclock_enabled=False, stufe3_trail_r=None,
                 zoom_start=None, slippage=0.0):
    """
    Trail stop after +1R. SL starts at BE (entry), trail at peak - width.

    trail_width_r: trail distance in R multiples (0.5 for CTRL, 1.0 for F, etc.)
    stufe3_trail_r: trail width for Stufe 3 (if ten_oclock_enabled)
    zoom_start: dict with remaining_secs, bar_low, bar_high, bar_time (from Phase 1)

    Returns dict with exit details.
    """
    trail_dist = trail_width_r * one_r
    peak = init_peak
    current_sl = _ratchet_sl(trade_dir, entry,
                             _trail_level(trade_dir, peak, trail_dist))
    mfe_from_entry = abs(peak - entry) if trade_dir == 'long' else abs(entry - peak)
    trail_tier = np.nan
    survived_to_1000 = False
    ten_checked = False
    active_trail_r = trail_width_r

    # Drift for 10:00 management
    drift_30 = compute_drift_30(meta_row, entry, adr) if ten_oclock_enabled else np.nan
    rvol_30 = meta_row.get('rvol_open_30min', np.nan) if ten_oclock_enabled else np.nan
    od_str = meta_row.get('od_strength', np.nan) if ten_oclock_enabled else np.nan

    def _do_trail_update():
        nonlocal current_sl, active_trail_r
        tl = _trail_level(trade_dir, peak, active_trail_r * one_r)
        current_sl = _ratchet_sl(trade_dir, current_sl, tl)

    # --- Process zoom continuation (if +1R reached during zoom) ---
    if zoom_start is not None:
        rem_secs = zoom_start.get('zoom_remaining_secs')
        if rem_secs is not None and len(rem_secs) > 0:
            for _, s in rem_secs.iterrows():
                if _stop_hit(trade_dir, s['low'], s['high'], current_sl):
                    pnl = _pnl(trade_dir, entry, current_sl, shares, slippage)
                    return _p2_result(current_sl, s['time_et'], 'trail', peak,
                                      mfe_from_entry, one_r, trail_tier,
                                      survived_to_1000, shares, pnl)
                peak = _update_peak(trade_dir, peak, s['high'], s['low'])
                mfe_from_entry = _compute_mfe(trade_dir, entry, s['high'], s['low'],
                                              mfe_from_entry)
                _do_trail_update()

        # Post-zoom bar check
        zbl = zoom_start.get('zoom_bar_low', np.nan)
        zbh = zoom_start.get('zoom_bar_high', np.nan)
        zbt = zoom_start.get('zoom_bar_time', '')
        if pd.notna(zbl) and _stop_hit(trade_dir, zbl, zbh, current_sl):
            pnl = _pnl(trade_dir, entry, current_sl, shares, slippage)
            return _p2_result(current_sl, zbt, 'trail', peak, mfe_from_entry,
                              one_r, trail_tier, survived_to_1000, shares, pnl)

    # --- Process remaining 1-min bars ---
    if len(bars_remaining) == 0:
        pnl = 0.0
        return _p2_result(entry, '15:55', 'timeout', peak, mfe_from_entry,
                          one_r, trail_tier, survived_to_1000, shares, pnl)

    for _, bar in bars_remaining.iterrows():
        bt = str(bar['time_et'])[:5]
        bh, bl = bar['high'], bar['low']

        # 10:00 management
        if ten_oclock_enabled and not ten_checked and bt >= '10:00':
            ten_checked = True
            survived_to_1000 = True
            tier = compute_trail_tier(drift_30, rvol_30, od_str)
            trail_tier = tier
            if tier == 3 and stufe3_trail_r is not None:
                if stufe3_trail_r < active_trail_r:
                    active_trail_r = stufe3_trail_r
                    _do_trail_update()

        # Check 10:00 survival (non-ten_oclock approaches too)
        if not ten_checked and bt >= '10:00':
            ten_checked = True
            survived_to_1000 = True

        stop_possible = _stop_hit(trade_dir, bl, bh, current_sl)
        fav_new = (bh > peak) if trade_dir == 'long' else (bl < peak)
        ambig = stop_possible and fav_new

        if ambig:
            resolved = False
            can_zoom = (sec_bars is not None and bt >= '09:30' and bt < '09:45')
            if can_zoom:
                mins = sec_bars[sec_bars['time_et'].str[:5] == bt]
                if len(mins) > 0:
                    for _, s in mins.iterrows():
                        if _stop_hit(trade_dir, s['low'], s['high'], current_sl):
                            pnl = _pnl(trade_dir, entry, current_sl, shares, slippage)
                            return _p2_result(current_sl, s['time_et'], 'trail',
                                              peak, mfe_from_entry, one_r,
                                              trail_tier, survived_to_1000, shares, pnl)
                        peak = _update_peak(trade_dir, peak, s['high'], s['low'])
                        mfe_from_entry = _compute_mfe(trade_dir, entry, s['high'],
                                                      s['low'], mfe_from_entry)
                        _do_trail_update()

                    # Post-zoom bar check
                    if _stop_hit(trade_dir, bl, bh, current_sl):
                        pnl = _pnl(trade_dir, entry, current_sl, shares, slippage)
                        return _p2_result(current_sl, bt, 'trail', peak,
                                          mfe_from_entry, one_r, trail_tier,
                                          survived_to_1000, shares, pnl)
                    resolved = True

            if not resolved:
                # Conservative: trail hit
                pnl = _pnl(trade_dir, entry, current_sl, shares, slippage)
                return _p2_result(current_sl, bt, 'trail', peak, mfe_from_entry,
                                  one_r, trail_tier, survived_to_1000, shares, pnl)

        elif stop_possible:
            pnl = _pnl(trade_dir, entry, current_sl, shares, slippage)
            return _p2_result(current_sl, bt, 'trail', peak, mfe_from_entry,
                              one_r, trail_tier, survived_to_1000, shares, pnl)
        else:
            # No stop — update peak and trail
            peak = _update_peak(trade_dir, peak, bh, bl)
            mfe_from_entry = _compute_mfe(trade_dir, entry, bh, bl, mfe_from_entry)
            _do_trail_update()

    # Timeout
    last = bars_remaining.iloc[-1]
    exit_p = last['close']
    pnl = _pnl(trade_dir, entry, exit_p, shares, slippage)
    return _p2_result(exit_p, str(last['time_et'])[:5], 'timeout', peak,
                      mfe_from_entry, one_r, trail_tier, survived_to_1000,
                      shares, pnl)


# ============================================================
# PHASE 2b: Hold to Close (for C, A_T2, E_T3)
# ============================================================
def phase2_hold(bars_remaining, entry, trade_dir, one_r, init_peak,
                shares, sec_bars, zoom_start=None, slippage=0.0):
    """
    After +1R: SL at entry (BE), no trail, hold until BE hit or 15:55.
    """
    sl = entry
    peak = init_peak
    mfe = abs(peak - entry) if trade_dir == 'long' else abs(entry - peak)
    survived_to_1000 = False
    ten_checked = False

    # Zoom continuation
    if zoom_start is not None:
        rem_secs = zoom_start.get('zoom_remaining_secs')
        if rem_secs is not None and len(rem_secs) > 0:
            for _, s in rem_secs.iterrows():
                if _stop_hit(trade_dir, s['low'], s['high'], sl):
                    pnl = _pnl(trade_dir, entry, sl, shares, slippage)
                    return _p2_result(sl, s['time_et'], 'be_stop', peak, mfe,
                                      one_r, np.nan, False, shares, pnl)
                peak = _update_peak(trade_dir, peak, s['high'], s['low'])
                mfe = _compute_mfe(trade_dir, entry, s['high'], s['low'], mfe)
        zbl = zoom_start.get('zoom_bar_low', np.nan)
        zbh = zoom_start.get('zoom_bar_high', np.nan)
        zbt = zoom_start.get('zoom_bar_time', '')
        if pd.notna(zbl) and _stop_hit(trade_dir, zbl, zbh, sl):
            pnl = _pnl(trade_dir, entry, sl, shares, slippage)
            return _p2_result(sl, zbt, 'be_stop', peak, mfe, one_r, np.nan,
                              False, shares, pnl)

    if len(bars_remaining) == 0:
        return _p2_result(entry, '15:55', 'timeout', peak, mfe, one_r,
                          np.nan, False, shares, 0.0)

    for _, bar in bars_remaining.iterrows():
        bt = str(bar['time_et'])[:5]
        bh, bl = bar['high'], bar['low']

        if not ten_checked and bt >= '10:00':
            ten_checked = True
            survived_to_1000 = True

        stop_possible = _stop_hit(trade_dir, bl, bh, sl)
        fav_new = (bh > peak) if trade_dir == 'long' else (bl < peak)
        ambig = stop_possible and fav_new

        if ambig:
            resolved = False
            can_zoom = (sec_bars is not None and bt >= '09:30' and bt < '09:45')
            if can_zoom:
                mins = sec_bars[sec_bars['time_et'].str[:5] == bt]
                if len(mins) > 0:
                    for _, s in mins.iterrows():
                        if _stop_hit(trade_dir, s['low'], s['high'], sl):
                            pnl = _pnl(trade_dir, entry, sl, shares, slippage)
                            return _p2_result(sl, s['time_et'], 'be_stop', peak,
                                              mfe, one_r, np.nan,
                                              survived_to_1000, shares, pnl)
                        peak = _update_peak(trade_dir, peak, s['high'], s['low'])
                        mfe = _compute_mfe(trade_dir, entry, s['high'], s['low'], mfe)
                    if _stop_hit(trade_dir, bl, bh, sl):
                        pnl = _pnl(trade_dir, entry, sl, shares, slippage)
                        return _p2_result(sl, bt, 'be_stop', peak, mfe, one_r,
                                          np.nan, survived_to_1000, shares, pnl)
                    resolved = True
            if not resolved:
                pnl = _pnl(trade_dir, entry, sl, shares, slippage)
                return _p2_result(sl, bt, 'be_stop', peak, mfe, one_r, np.nan,
                                  survived_to_1000, shares, pnl)
        elif stop_possible:
            pnl = _pnl(trade_dir, entry, sl, shares, slippage)
            return _p2_result(sl, bt, 'be_stop', peak, mfe, one_r, np.nan,
                              survived_to_1000, shares, pnl)
        else:
            peak = _update_peak(trade_dir, peak, bh, bl)
            mfe = _compute_mfe(trade_dir, entry, bh, bl, mfe)

    # Timeout
    last = bars_remaining.iloc[-1]
    exit_p = last['close']
    pnl = _pnl(trade_dir, entry, exit_p, shares, slippage)
    return _p2_result(exit_p, str(last['time_et'])[:5], 'timeout', peak, mfe,
                      one_r, np.nan, survived_to_1000, shares, pnl)


# ============================================================
# PHASE 2c: Sell at Target (for E_T2 — sell at +2R with BE)
# ============================================================
def phase2_sell_at(bars_remaining, entry, trade_dir, one_r, init_peak,
                   shares, target_r, sec_bars, zoom_start=None, slippage=0.0):
    """
    After +1R: SL at entry (BE), sell at target_r * one_r from entry.
    """
    sl = entry
    if trade_dir == 'long':
        target_price = entry + target_r * one_r
    else:
        target_price = entry - target_r * one_r
    peak = init_peak
    mfe = abs(peak - entry) if trade_dir == 'long' else abs(entry - peak)

    def _check_target(low, high):
        if trade_dir == 'long':
            return high >= target_price
        else:
            return low <= target_price

    # Zoom continuation
    if zoom_start is not None:
        rem_secs = zoom_start.get('zoom_remaining_secs')
        if rem_secs is not None and len(rem_secs) > 0:
            for _, s in rem_secs.iterrows():
                sh, sl_ = s['high'], s['low']
                hit_be = _stop_hit(trade_dir, sl_, sh, sl)
                hit_tgt = _check_target(sl_, sh)
                if hit_be and hit_tgt:
                    # Ambiguous within second — conservative: BE first
                    pnl = _pnl(trade_dir, entry, sl, shares, slippage)
                    return _p2_result(sl, s['time_et'], 'be_stop', peak, mfe,
                                      one_r, np.nan, False, shares, pnl)
                if hit_be:
                    pnl = _pnl(trade_dir, entry, sl, shares, slippage)
                    return _p2_result(sl, s['time_et'], 'be_stop', peak, mfe,
                                      one_r, np.nan, False, shares, pnl)
                if hit_tgt:
                    pnl = _pnl(trade_dir, entry, target_price, shares, slippage)
                    return _p2_result(target_price, s['time_et'], f'partial_{int(target_r)}r',
                                      peak, mfe, one_r, np.nan, False, shares, pnl)
                peak = _update_peak(trade_dir, peak, sh, sl_)
                mfe = _compute_mfe(trade_dir, entry, sh, sl_, mfe)

        zbl = zoom_start.get('zoom_bar_low', np.nan)
        zbh = zoom_start.get('zoom_bar_high', np.nan)
        zbt = zoom_start.get('zoom_bar_time', '')
        if pd.notna(zbl):
            hit_be = _stop_hit(trade_dir, zbl, zbh, sl)
            hit_tgt = _check_target(zbl, zbh)
            if hit_be:
                pnl = _pnl(trade_dir, entry, sl, shares, slippage)
                return _p2_result(sl, zbt, 'be_stop', peak, mfe, one_r,
                                  np.nan, False, shares, pnl)
            if hit_tgt:
                pnl = _pnl(trade_dir, entry, target_price, shares, slippage)
                return _p2_result(target_price, zbt, f'partial_{int(target_r)}r',
                                  peak, mfe, one_r, np.nan, False, shares, pnl)

    if len(bars_remaining) == 0:
        return _p2_result(entry, '15:55', 'timeout', peak, mfe, one_r,
                          np.nan, False, shares, 0.0)

    for _, bar in bars_remaining.iterrows():
        bt = str(bar['time_et'])[:5]
        bh, bl_ = bar['high'], bar['low']

        hit_be = _stop_hit(trade_dir, bl_, bh, sl)
        hit_tgt = _check_target(bl_, bh)

        if hit_be and hit_tgt:
            # Ambiguous — try zoom
            resolved = False
            can_zoom = (sec_bars is not None and bt >= '09:30' and bt < '09:45')
            if can_zoom:
                mins = sec_bars[sec_bars['time_et'].str[:5] == bt]
                if len(mins) > 0:
                    for _, s in mins.iterrows():
                        sh, sl2 = s['high'], s['low']
                        if _stop_hit(trade_dir, sl2, sh, sl):
                            pnl = _pnl(trade_dir, entry, sl, shares, slippage)
                            return _p2_result(sl, s['time_et'], 'be_stop', peak,
                                              mfe, one_r, np.nan, False, shares, pnl)
                        if _check_target(sl2, sh):
                            pnl = _pnl(trade_dir, entry, target_price, shares, slippage)
                            return _p2_result(target_price, s['time_et'],
                                              f'partial_{int(target_r)}r',
                                              peak, mfe, one_r, np.nan, False, shares, pnl)
                        peak = _update_peak(trade_dir, peak, sh, sl2)
                        mfe = _compute_mfe(trade_dir, entry, sh, sl2, mfe)
                    resolved = True
                    # After zoom, re-check
                    if _stop_hit(trade_dir, bl_, bh, sl):
                        pnl = _pnl(trade_dir, entry, sl, shares, slippage)
                        return _p2_result(sl, bt, 'be_stop', peak, mfe, one_r,
                                          np.nan, False, shares, pnl)
                    if _check_target(bl_, bh):
                        pnl = _pnl(trade_dir, entry, target_price, shares, slippage)
                        return _p2_result(target_price, bt,
                                          f'partial_{int(target_r)}r',
                                          peak, mfe, one_r, np.nan, False, shares, pnl)
            if not resolved:
                # Conservative: BE first
                pnl = _pnl(trade_dir, entry, sl, shares, slippage)
                return _p2_result(sl, bt, 'be_stop', peak, mfe, one_r,
                                  np.nan, False, shares, pnl)
        elif hit_be:
            pnl = _pnl(trade_dir, entry, sl, shares, slippage)
            return _p2_result(sl, bt, 'be_stop', peak, mfe, one_r,
                              np.nan, False, shares, pnl)
        elif hit_tgt:
            pnl = _pnl(trade_dir, entry, target_price, shares, slippage)
            return _p2_result(target_price, bt, f'partial_{int(target_r)}r',
                              peak, mfe, one_r, np.nan, False, shares, pnl)
        else:
            peak = _update_peak(trade_dir, peak, bh, bl_)
            mfe = _compute_mfe(trade_dir, entry, bh, bl_, mfe)

    # Timeout
    last = bars_remaining.iloc[-1]
    exit_p = last['close']
    pnl = _pnl(trade_dir, entry, exit_p, shares, slippage)
    return _p2_result(exit_p, str(last['time_et'])[:5], 'timeout', peak, mfe,
                      one_r, np.nan, False, shares, pnl)


# ============================================================
# PnL and Result Helpers
# ============================================================
def _pnl(trade_dir, entry, exit_price, shares, slippage=0.0):
    if trade_dir == 'long':
        return (exit_price - entry) * shares - slippage * shares * 2
    else:
        return (entry - exit_price) * shares - slippage * shares * 2


def _p2_result(exit_price, exit_time, exit_reason, peak, mfe, one_r,
               trail_tier, survived_to_1000, shares, pnl):
    return {
        'exit_price': exit_price,
        'exit_time': exit_time,
        'exit_reason': exit_reason,
        'peak': peak,
        'mfe': mfe,
        'one_r': one_r,
        'trail_tier': trail_tier,
        'survived_to_1000': survived_to_1000,
        'shares': shares,
        'pnl': pnl,
    }


# ============================================================
# APPROACH DISPATCH: Run all 7 approaches for one trade
# ============================================================
def run_all_approaches(bars_rth, entry, trade_dir, adr, meta_row,
                       sec_bars, risk=RISK, slippage=SLIPPAGE):
    """
    Run all 7 approaches for a single qualifying trade.
    Returns list of result dicts (one per approach).
    """
    results = []

    # Precompute shared values
    one_r_025 = SL_FRAC_025 * adr
    one_r_050 = SL_FRAC_050 * adr
    if trade_dir == 'long':
        sl_025 = entry - one_r_025
        sl_050 = entry - one_r_050
        one_r_level_025 = entry + one_r_025
        one_r_level_050 = entry + one_r_050
    else:
        sl_025 = entry + one_r_025
        sl_050 = entry + one_r_050
        one_r_level_025 = entry - one_r_025
        one_r_level_050 = entry - one_r_050
    shares_025 = risk / one_r_025
    shares_050 = risk / one_r_050

    # Filter bars
    post = bars_rth[
        (bars_rth['time_et'] >= '09:36') &
        (bars_rth['time_et'] <= '15:55')
    ]
    if len(post) == 0:
        return []

    # Meta values
    drift_30 = compute_drift_30(meta_row, entry, adr)
    rvol_30 = meta_row.get('rvol_open_30min', np.nan)
    od_str = meta_row.get('od_strength', np.nan)

    # ===========================================
    # Phase 1 for Group 0.25 ADR (CTRL/A/C/D/E/F)
    # ===========================================
    p1 = phase1_simulate(bars_rth, entry, sl_025, trade_dir,
                         one_r_025, one_r_level_025, sec_bars)
    if p1.get('outcome') == 'skip':
        return []

    # ===========================================
    # Phase 1 for B (0.50 ADR)
    # ===========================================
    p1b = phase1_simulate(bars_rth, entry, sl_050, trade_dir,
                          one_r_050, one_r_level_050, sec_bars)

    # Helper: build remaining bars for Phase 2
    def _remaining_bars(p1_res):
        idx = p1_res.get('next_bar_idx', 0)
        return post.iloc[idx:]

    def _zoom_start(p1_res):
        if p1_res.get('outcome') == '1r_in_zoom':
            return {
                'zoom_remaining_secs': p1_res.get('zoom_remaining_secs'),
                'zoom_bar_low': p1_res.get('zoom_bar_low'),
                'zoom_bar_high': p1_res.get('zoom_bar_high'),
                'zoom_bar_time': p1_res.get('zoom_bar_time'),
            }
        return None

    # Helper: build base record
    def _base_rec(approach, shares, one_r, sl_dollar, p1_res, pnl, pnl_r,
                  peak_r, exit_price, exit_time, exit_reason, trail_tier,
                  survived_to_1000, zoom_used, zoom_changed):
        return {
            'approach': approach,
            'shares': shares,
            'sl_dollar': sl_dollar,
            'exit_price': exit_price,
            'exit_time': exit_time,
            'exit_reason': exit_reason,
            'pnl_dollar': pnl,
            'pnl_r': pnl_r,
            'peak_r': peak_r,
            'mfe_r': peak_r,
            'trail_tier': trail_tier,
            'survived_to_1000': survived_to_1000,
            'used_1sec_zoom': zoom_used,
            'zoom_changed': zoom_changed,
            'drift_30min': drift_30,
            'rvol_30': rvol_30,
            'od_strength': od_str,
            # Multi-tranche (NaN for single)
            'shares_t1': np.nan, 'exit_price_t1': np.nan,
            'exit_time_t1': np.nan, 'exit_reason_t1': np.nan, 'pnl_t1': np.nan,
            'shares_t2': np.nan, 'exit_price_t2': np.nan,
            'exit_time_t2': np.nan, 'exit_reason_t2': np.nan, 'pnl_t2': np.nan,
            'shares_t3': np.nan, 'exit_price_t3': np.nan,
            'exit_time_t3': np.nan, 'exit_reason_t3': np.nan, 'pnl_t3': np.nan,
        }

    def _make_sl_loss(approach, shares, one_r, p1_res):
        """Create loss record when initial SL is hit."""
        pnl = -risk  # always -$1000 loss
        pnl_r = -1.0
        peak_r = p1_res['mfe'] / one_r if one_r > 0 else 0
        return _base_rec(approach, shares, one_r, one_r, p1_res,
                         pnl, pnl_r, peak_r,
                         p1_res['exit_price'], p1_res['exit_time'],
                         'init_sl', np.nan, False,
                         p1_res.get('zoom_used', False),
                         p1_res.get('zoom_changed', False))

    def _make_timeout(approach, shares, one_r, p1_res):
        """Create timeout record when +1R never reached."""
        exit_p = p1_res['exit_price']
        pnl = _pnl(trade_dir, entry, exit_p, shares, slippage)
        pnl_r = pnl / risk
        peak_r = p1_res['mfe'] / one_r if one_r > 0 else 0
        return _base_rec(approach, shares, one_r, one_r, p1_res,
                         pnl, pnl_r, peak_r,
                         exit_p, p1_res['exit_time'], 'timeout',
                         np.nan, False,
                         p1_res.get('zoom_used', False),
                         p1_res.get('zoom_changed', False))

    def _make_from_p2(approach, shares, one_r, p1_res, p2_res):
        """Create record from Phase 2 result (single tranche)."""
        pnl = p2_res['pnl']
        pnl_r = pnl / risk
        peak_r = p2_res['mfe'] / one_r if one_r > 0 else 0
        return _base_rec(approach, shares, one_r, one_r, p1_res,
                         pnl, pnl_r, peak_r,
                         p2_res['exit_price'], p2_res['exit_time'],
                         p2_res['exit_reason'],
                         p2_res.get('trail_tier', np.nan),
                         p2_res.get('survived_to_1000', False),
                         p1_res.get('zoom_used', False) or (p1_res.get('outcome') == '1r_in_zoom'),
                         p1_res.get('zoom_changed', False) or (p1_res.get('outcome') == '1r_in_zoom'))

    def _make_multi(approach, shares_total, one_r, p1_res, t_results):
        """Create record from multi-tranche Phase 2 results."""
        pnl_total = sum(t['pnl'] for t in t_results)
        pnl_r = pnl_total / risk
        # Peak is from price data (same for all tranches)
        peak_r = max(t['mfe'] for t in t_results) / one_r if one_r > 0 else 0
        # Last exit time
        exit_times = [t['exit_time'] for t in t_results if t['exit_time']]
        last_exit = max(exit_times) if exit_times else '15:55'
        # Trail tier from the tranche that has it
        tt = np.nan
        s1000 = False
        for t in t_results:
            if pd.notna(t.get('trail_tier', np.nan)):
                tt = t['trail_tier']
            if t.get('survived_to_1000', False):
                s1000 = True

        rec = _base_rec(approach, shares_total, one_r, one_r, p1_res,
                        pnl_total, pnl_r, peak_r,
                        np.nan, last_exit, 'multi',
                        tt, s1000,
                        p1_res.get('zoom_used', False) or (p1_res.get('outcome') == '1r_in_zoom'),
                        p1_res.get('zoom_changed', False))
        # Fill tranche details
        for i, t in enumerate(t_results):
            idx = i + 1
            rec[f'shares_t{idx}'] = t['shares']
            rec[f'exit_price_t{idx}'] = t['exit_price']
            rec[f'exit_time_t{idx}'] = t['exit_time']
            rec[f'exit_reason_t{idx}'] = t['exit_reason']
            rec[f'pnl_t{idx}'] = t['pnl']
        return rec

    # ===========================================
    # CTRL — Original TRAIL_D (Peak - 0.5R)
    # ===========================================
    if p1['outcome'] == 'sl_hit':
        results.append(_make_sl_loss('CTRL', shares_025, one_r_025, p1))
    elif p1['outcome'] == 'timeout':
        results.append(_make_timeout('CTRL', shares_025, one_r_025, p1))
    else:
        rem = _remaining_bars(p1)
        zs = _zoom_start(p1)
        t = phase2_trail(rem, entry, trade_dir, one_r_025, p1['peak'],
                         shares_025, trail_width_r=0.5, sec_bars=sec_bars,
                         adr=adr, meta_row=meta_row, zoom_start=zs,
                         slippage=slippage)
        results.append(_make_from_p2('CTRL', shares_025, one_r_025, p1, t))

    # ===========================================
    # A — Split: Half TRAIL_D + Half Hold
    # ===========================================
    s_a1 = shares_025 / 2
    s_a2 = shares_025 - s_a1

    if p1['outcome'] == 'sl_hit':
        results.append(_make_sl_loss('A', shares_025, one_r_025, p1))
    elif p1['outcome'] == 'timeout':
        results.append(_make_timeout('A', shares_025, one_r_025, p1))
    else:
        rem = _remaining_bars(p1)
        zs = _zoom_start(p1)
        t1 = phase2_trail(rem, entry, trade_dir, one_r_025, p1['peak'],
                          s_a1, trail_width_r=0.5, sec_bars=sec_bars,
                          adr=adr, meta_row=meta_row, zoom_start=zs,
                          slippage=slippage)
        t2 = phase2_hold(rem, entry, trade_dir, one_r_025, p1['peak'],
                         s_a2, sec_bars=sec_bars, zoom_start=zs,
                         slippage=slippage)
        results.append(_make_multi('A', shares_025, one_r_025, p1, [t1, t2]))

    # ===========================================
    # B — Wider SL (0.5 ADR) + Trail Peak-0.5R + 10:00 tier
    # ===========================================
    if p1b.get('outcome') == 'skip':
        pass  # skip B
    elif p1b['outcome'] == 'sl_hit':
        results.append(_make_sl_loss('B', shares_050, one_r_050, p1b))
    elif p1b['outcome'] == 'timeout':
        results.append(_make_timeout('B', shares_050, one_r_050, p1b))
    else:
        rem = _remaining_bars(p1b)
        zs = _zoom_start(p1b)
        t = phase2_trail(rem, entry, trade_dir, one_r_050, p1b['peak'],
                         shares_050, trail_width_r=0.5, sec_bars=sec_bars,
                         adr=adr, meta_row=meta_row,
                         ten_oclock_enabled=True, stufe3_trail_r=0.25,
                         zoom_start=zs, slippage=slippage)
        results.append(_make_from_p2('B', shares_050, one_r_050, p1b, t))

    # ===========================================
    # C — Time-Stop (BE + Hold to 15:55)
    # ===========================================
    if p1['outcome'] == 'sl_hit':
        results.append(_make_sl_loss('C', shares_025, one_r_025, p1))
    elif p1['outcome'] == 'timeout':
        results.append(_make_timeout('C', shares_025, one_r_025, p1))
    else:
        rem = _remaining_bars(p1)
        zs = _zoom_start(p1)
        t = phase2_hold(rem, entry, trade_dir, one_r_025, p1['peak'],
                        shares_025, sec_bars=sec_bars, zoom_start=zs,
                        slippage=slippage)
        results.append(_make_from_p2('C', shares_025, one_r_025, p1, t))

    # ===========================================
    # D — Split: Half TRAIL_D + Half wide trail (Peak-2R) + 10:00
    # ===========================================
    s_d1 = shares_025 / 2
    s_d2 = shares_025 - s_d1

    if p1['outcome'] == 'sl_hit':
        results.append(_make_sl_loss('D', shares_025, one_r_025, p1))
    elif p1['outcome'] == 'timeout':
        results.append(_make_timeout('D', shares_025, one_r_025, p1))
    else:
        rem = _remaining_bars(p1)
        zs = _zoom_start(p1)
        t1 = phase2_trail(rem, entry, trade_dir, one_r_025, p1['peak'],
                          s_d1, trail_width_r=0.5, sec_bars=sec_bars,
                          adr=adr, meta_row=meta_row, zoom_start=zs,
                          slippage=slippage)
        t2 = phase2_trail(rem, entry, trade_dir, one_r_025, p1['peak'],
                          s_d2, trail_width_r=2.0, sec_bars=sec_bars,
                          adr=adr, meta_row=meta_row,
                          ten_oclock_enabled=True, stufe3_trail_r=1.0,
                          zoom_start=zs, slippage=slippage)
        results.append(_make_multi('D', shares_025, one_r_025, p1, [t1, t2]))

    # ===========================================
    # E — Scaled: 50% at +1R, 25% at +2R, 25% Hold
    # ===========================================
    s_e1 = shares_025 * 0.50
    s_e2 = shares_025 * 0.25
    s_e3 = shares_025 - s_e1 - s_e2

    if p1['outcome'] == 'sl_hit':
        results.append(_make_sl_loss('E', shares_025, one_r_025, p1))
    elif p1['outcome'] == 'timeout':
        results.append(_make_timeout('E', shares_025, one_r_025, p1))
    else:
        rem = _remaining_bars(p1)
        zs = _zoom_start(p1)
        # E1: sold at +1R (immediate)
        pnl_e1 = s_e1 * one_r_025 - slippage * s_e1 * 2
        e1_exit_price = one_r_level_025
        e1_time = p1.get('exit_time', p1.get('zoom_bar_time', '09:36'))
        if p1.get('outcome') == '1r_in_zoom':
            e1_time = p1.get('zoom_bar_time', '09:36')
        t1 = {
            'exit_price': e1_exit_price, 'exit_time': e1_time,
            'exit_reason': 'partial_1r', 'peak': p1['peak'],
            'mfe': p1['mfe'], 'one_r': one_r_025,
            'trail_tier': np.nan, 'survived_to_1000': False,
            'shares': s_e1, 'pnl': pnl_e1,
        }
        # E2: sell at +2R or BE or timeout
        t2 = phase2_sell_at(rem, entry, trade_dir, one_r_025, p1['peak'],
                            s_e2, target_r=2.0, sec_bars=sec_bars,
                            zoom_start=zs, slippage=slippage)
        # E3: hold to close
        t3 = phase2_hold(rem, entry, trade_dir, one_r_025, p1['peak'],
                         s_e3, sec_bars=sec_bars, zoom_start=zs,
                         slippage=slippage)
        results.append(_make_multi('E', shares_025, one_r_025, p1, [t1, t2, t3]))

    # ===========================================
    # F — Wider Trail (Peak - 1R) + 10:00 management
    # ===========================================
    if p1['outcome'] == 'sl_hit':
        results.append(_make_sl_loss('F', shares_025, one_r_025, p1))
    elif p1['outcome'] == 'timeout':
        results.append(_make_timeout('F', shares_025, one_r_025, p1))
    else:
        rem = _remaining_bars(p1)
        zs = _zoom_start(p1)
        t = phase2_trail(rem, entry, trade_dir, one_r_025, p1['peak'],
                         shares_025, trail_width_r=1.0, sec_bars=sec_bars,
                         adr=adr, meta_row=meta_row,
                         ten_oclock_enabled=True, stufe3_trail_r=0.5,
                         zoom_start=zs, slippage=slippage)
        results.append(_make_from_p2('F', shares_025, one_r_025, p1, t))

    return results


# ============================================================
# BUILD ALL TRADES
# ============================================================
def build_trades(meta, slippage=SLIPPAGE, risk=RISK):
    """Simulate all qualifying trades with 7 approaches each."""
    base = meta[(meta['od_strength'] > 0.5) &
                (meta['od_direction'] == 'with_gap')].copy()

    all_results = []
    skipped = 0

    for _, row in tqdm(base.iterrows(), total=len(base),
                       desc="Simulating trades", file=sys.stderr):
        ticker = row['ticker']
        date = str(row['date'])
        entry = row.get('close_935', np.nan)
        adr = row.get('adr_10', np.nan)
        gap_dir = row['gap_direction']

        if pd.isna(entry) or pd.isna(adr) or adr <= 0:
            skipped += 1
            continue

        trade_dir = 'long' if gap_dir == 'up' else 'short'

        fpath = RAW_1MIN_DIR / ticker / f"{date}.parquet"
        if not fpath.exists():
            skipped += 1
            continue
        bars = pd.read_parquet(fpath)
        rth = bars[bars['session'] == 'rth']
        if len(rth) < 10:
            skipped += 1
            continue

        if IS_START <= date <= IS_END:
            period = 'IS'
        elif date >= OOS_START:
            period = 'OOS'
        else:
            continue

        sec_bars = load_1sec(ticker, date)
        meta_row = row.to_dict()

        approach_results = run_all_approaches(rth, entry, trade_dir, adr,
                                              meta_row, sec_bars, risk, slippage)

        # Tier classification
        rvol_30 = row.get('rvol_open_30min', np.nan)
        is_tier1 = pd.notna(rvol_30) and rvol_30 > 5

        for rec in approach_results:
            rec['ticker'] = ticker
            rec['date'] = date
            rec['gap_direction'] = gap_dir
            rec['trade_dir'] = trade_dir
            rec['entry_price'] = entry
            rec['adr'] = adr
            rec['period'] = period
            rec['is_tier1'] = is_tier1
            all_results.append(rec)

    print(f"  Skipped: {skipped}, Total records: {len(all_results)}",
          file=sys.stderr)

    df = pd.DataFrame(all_results)

    # Compute Big Runner flags from CTRL and C results
    if len(df) > 0:
        ctrl_peaks = df[df['approach'] == 'CTRL'][['ticker', 'date', 'peak_r']].copy()
        ctrl_peaks = ctrl_peaks.rename(columns={'peak_r': 'ctrl_peak_r'})
        df = df.merge(ctrl_peaks, on=['ticker', 'date'], how='left')
        df['is_br_range'] = df['ctrl_peak_r'] >= BR_MFE_THRESHOLD

        c_pnl = df[df['approach'] == 'C'][['ticker', 'date', 'pnl_r']].copy()
        c_pnl = c_pnl.rename(columns={'pnl_r': 'c_pnl_r'})
        df = df.merge(c_pnl, on=['ticker', 'date'], how='left')
        df['is_br_entry'] = df['c_pnl_r'] >= BR_ENTRY_THRESHOLD

        df = df.drop(columns=['ctrl_peak_r', 'c_pnl_r'], errors='ignore')

    return df


# ============================================================
# KPI COMPUTATION
# ============================================================
def compute_kpis(df, label=''):
    """Compute all KPIs for a trade DataFrame."""
    n = len(df)
    if n == 0:
        return {'label': label, 'N': 0}

    pnl = df['pnl_dollar'].values
    pnl_r = df['pnl_r'].values

    w = pnl > 0.01
    l = pnl < -0.01
    be = ~w & ~l

    kpi = {'label': label, 'N': n}
    kpi['Win%'] = w.mean() * 100
    kpi['Loss%'] = l.mean() * 100
    kpi['BE%'] = be.mean() * 100

    kpi['Total PnL'] = pnl.sum()
    kpi['Mean PnL'] = pnl.mean()
    kpi['Median PnL'] = np.median(pnl)
    kpi['StdDev'] = pnl.std()
    kpi['Skew'] = sp_stats.skew(pnl) if n > 2 else 0
    kpi['Best'] = pnl.max()
    kpi['Worst'] = pnl.min()
    kpi['P10'] = np.percentile(pnl, 10)
    kpi['P25'] = np.percentile(pnl, 25)
    kpi['P75'] = np.percentile(pnl, 75)
    kpi['P90'] = np.percentile(pnl, 90)

    kpi['Mean R'] = pnl_r.mean()
    kpi['Median R'] = np.median(pnl_r)
    kpi['Max R'] = pnl_r.max()
    for x in [1, 2, 3, 5, 10]:
        kpi[f'WR@{x}R'] = (pnl_r >= x).mean() * 100

    gw = pnl[w].sum() if w.any() else 0
    gl = abs(pnl[l].sum()) if l.any() else 0.001
    kpi['PF'] = gw / gl
    kpi['Payoff'] = (pnl[w].mean() / abs(pnl[l].mean())) if (l.any() and w.any()) else np.nan

    mc_l = mc_w = cc_l = cc_w = 0
    for p in pnl:
        if p < -0.01:
            cc_l += 1; cc_w = 0; mc_l = max(mc_l, cc_l)
        elif p > 0.01:
            cc_w += 1; cc_l = 0; mc_w = max(mc_w, cc_w)
        else:
            cc_l = cc_w = 0
    kpi['MaxConsL'] = mc_l

    eq = np.cumsum(pnl)
    pk = np.maximum.accumulate(eq)
    dd = eq - pk
    kpi['MaxDD$'] = dd.min()
    kpi['MaxDD% of 50K'] = dd.min() / 50000 * 100

    in_dd = dd < -0.01
    if in_dd.any():
        durs = []
        cd = 0
        for d in dd:
            if d < -0.01:
                cd += 1
            else:
                if cd > 0:
                    durs.append(cd)
                cd = 0
        if cd > 0:
            durs.append(cd)
        kpi['MaxDDDur'] = max(durs) if durs else 0
    else:
        kpi['MaxDDDur'] = 0

    kpi['equity'] = eq.tolist()

    # Exit reason breakdown
    if 'exit_reason' in df.columns:
        vc = df['exit_reason'].value_counts(normalize=True)
        kpi['%InitSL'] = vc.get('init_sl', 0) * 100
        kpi['%Trail'] = vc.get('trail', 0) * 100
        kpi['%Timeout'] = vc.get('timeout', 0) * 100
        kpi['%BEStop'] = vc.get('be_stop', 0) * 100

    # Pct reaching +1R
    reach_1r = df['pnl_r'] >= 1.0
    kpi['%Reach1R'] = reach_1r.mean() * 100 if n > 0 else 0

    # Pct surviving to 10:00
    if 'survived_to_1000' in df.columns:
        kpi['%Surv1000'] = df['survived_to_1000'].mean() * 100

    # Median trade duration (minutes)
    if 'exit_time' in df.columns:
        try:
            entry_mins = 9 * 60 + 36  # 09:36
            durs = []
            for et in df['exit_time'].dropna():
                ets = str(et)[:5]
                if ':' in ets:
                    parts = ets.split(':')
                    exit_mins = int(parts[0]) * 60 + int(parts[1])
                    durs.append(exit_mins - entry_mins)
            if durs:
                kpi['MedDurMin'] = np.median(durs)
                kpi['MeanDurMin'] = np.mean(durs)
            else:
                kpi['MedDurMin'] = np.nan
                kpi['MeanDurMin'] = np.nan
        except Exception:
            kpi['MedDurMin'] = np.nan
            kpi['MeanDurMin'] = np.nan

    return kpi


# ============================================================
# WALK-FORWARD ANALYSIS
# ============================================================
def walk_forward_analysis(all_df):
    """Compute per-window KPIs for each approach."""
    wf_results = []
    for wname, wstart, wend in WF_WINDOWS:
        window_df = all_df[(all_df['date'] >= wstart) & (all_df['date'] <= wend)]
        for approach in APPROACHES:
            for gd in ['up', 'down']:
                sub = window_df[(window_df['approach'] == approach) &
                                (window_df['gap_direction'] == gd)]
                n = len(sub)
                if n == 0:
                    wf_results.append({
                        'window': wname, 'approach': approach, 'gap_dir': gd,
                        'N': 0, 'Total PnL': 0, 'Mean R': 0, 'PF': 0,
                        'Win%': 0, 'MaxDD$': 0,
                    })
                    continue
                pnl = sub['pnl_dollar'].values
                pnl_r = sub['pnl_r'].values
                w = pnl > 0.01
                l = pnl < -0.01
                gw = pnl[w].sum() if w.any() else 0
                gl = abs(pnl[l].sum()) if l.any() else 0.001
                eq = np.cumsum(pnl)
                dd = (eq - np.maximum.accumulate(eq)).min()
                wf_results.append({
                    'window': wname, 'approach': approach, 'gap_dir': gd,
                    'N': n,
                    'Total PnL': pnl.sum(),
                    'Mean R': pnl_r.mean(),
                    'PF': gw / gl if n >= 10 else np.nan,
                    'Win%': w.mean() * 100,
                    'MaxDD$': dd,
                })
    return pd.DataFrame(wf_results)


# ============================================================
# MONTE CARLO
# ============================================================
def monte_carlo(pnl_array, n_iter=MC_ITERS, seed=42):
    """Permutation-based Monte Carlo simulation."""
    rng = np.random.RandomState(seed)
    pnl = np.array(pnl_array)
    n = len(pnl)
    if n < 5:
        return None

    paths = np.zeros((n_iter, n))
    max_dds = np.zeros(n_iter)
    max_cls = np.zeros(n_iter, dtype=int)

    for i in range(n_iter):
        shuf = pnl[rng.permutation(n)]
        eq = np.cumsum(shuf)
        paths[i] = eq
        pk = np.maximum.accumulate(eq)
        dd = eq - pk
        max_dds[i] = dd.min()

        cl = mcl = 0
        for p in shuf:
            if p < -0.01:
                cl += 1
                mcl = max(mcl, cl)
            else:
                cl = 0
        max_cls[i] = mcl

    median_path = np.median(paths, axis=0)
    p5_path = np.percentile(paths, 5, axis=0)
    p95_path = np.percentile(paths, 95, axis=0)

    fe = paths[:, -1]
    return {
        'paths': paths,
        'median_path': median_path,
        'p5_path': p5_path,
        'p95_path': p95_path,
        'fe_median': np.median(fe),
        'fe_p5': np.percentile(fe, 5),
        'fe_p25': np.percentile(fe, 25),
        'fe_p75': np.percentile(fe, 75),
        'fe_p95': np.percentile(fe, 95),
        'dd_median': np.median(max_dds),
        'dd_p5': np.percentile(max_dds, 5),
        'dd_p95': np.percentile(max_dds, 95),
        'worst_dd': max_dds.min(),
        'max_dds': max_dds,
        'cl_median': np.median(max_cls),
        'cl_p95': np.percentile(max_cls, 95),
        **{f'p_ruin_{t}': (max_dds <= -t).mean() * 100
           for t in [10000, 15000, 20000]},
    }


# ============================================================
# CHARTS
# ============================================================
def setup_dark():
    plt.style.use('dark_background')
    plt.rcParams.update({
        'figure.facecolor': C_BG,
        'axes.facecolor': C_BG,
        'savefig.facecolor': C_BG,
    })


def chart1_equity_tier2_gapup(all_df, out):
    """Chart 1: Equity comparison all 7, OOS Tier2 GapUp."""
    setup_dark()
    fig, ax = plt.subplots(figsize=(16, 8))

    for approach in APPROACHES:
        sub = all_df[(all_df['approach'] == approach) &
                     (all_df['period'] == 'OOS') &
                     (all_df['gap_direction'] == 'up')].sort_values('date')
        if len(sub) == 0:
            continue
        eq = np.cumsum(sub['pnl_dollar'].values)
        ax.plot(range(1, len(eq) + 1), eq, color=ACOLS[approach], lw=2,
                label=f"{approach}: ${eq[-1]:,.0f}")

    ax.axhline(0, color='grey', lw=0.5)
    ax.set_title('Equity Curves — All Approaches (OOS GapUp)', fontsize=14)
    ax.set_xlabel('Trade #')
    ax.set_ylabel('Cumulative PnL ($)')
    ax.legend(fontsize=9, loc='upper left')
    plt.tight_layout()
    fig.savefig(out / 'chart1_equity_tier2_gapup.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  Chart 1: equity_tier2_gapup", file=sys.stderr)


def chart2_equity_tier1_gapup(all_df, out):
    """Chart 2: Equity comparison all 7, OOS Tier1 GapUp."""
    setup_dark()
    fig, ax = plt.subplots(figsize=(16, 8))

    for approach in APPROACHES:
        sub = all_df[(all_df['approach'] == approach) &
                     (all_df['period'] == 'OOS') &
                     (all_df['gap_direction'] == 'up') &
                     (all_df['is_tier1'] == True)].sort_values('date')
        if len(sub) == 0:
            continue
        eq = np.cumsum(sub['pnl_dollar'].values)
        ax.plot(range(1, len(eq) + 1), eq, color=ACOLS[approach], lw=2,
                label=f"{approach}: ${eq[-1]:,.0f}")

    ax.axhline(0, color='grey', lw=0.5)
    ax.set_title('Equity Curves — All Approaches (OOS Tier1 GapUp)', fontsize=14)
    ax.set_xlabel('Trade #')
    ax.set_ylabel('Cumulative PnL ($)')
    ax.legend(fontsize=9, loc='upper left')
    plt.tight_layout()
    fig.savefig(out / 'chart2_equity_tier1_gapup.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  Chart 2: equity_tier1_gapup", file=sys.stderr)


def chart3_drawdown_mc(all_df, mc_results, out):
    """Chart 3: Drawdown histograms from Monte Carlo (7 subplots)."""
    setup_dark()
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    axes = axes.flatten()

    for i, approach in enumerate(APPROACHES):
        ax = axes[i]
        mc = mc_results.get(approach)
        if mc is None:
            ax.set_title(f'{approach}: N/A')
            continue
        dds = mc['max_dds']
        ax.hist(dds, bins=40, color=ACOLS[approach], alpha=0.7,
                edgecolor='white', lw=0.3)
        ax.axvline(mc['dd_median'], color='yellow', lw=2,
                   label=f"Med: ${mc['dd_median']:,.0f}")
        ax.axvline(mc['dd_p95'], color='red', lw=1.5, ls='--',
                   label=f"P95: ${mc['dd_p95']:,.0f}")
        ax.set_title(f'{approach}', fontsize=10)
        ax.legend(fontsize=6)
        ax.set_xlabel('Max DD ($)', fontsize=7)

    if len(APPROACHES) < 8:
        axes[-1].axis('off')
    fig.suptitle('Max Drawdown Distribution (MC, OOS GapUp)', fontsize=14)
    plt.tight_layout()
    fig.savefig(out / 'chart3_drawdown_mc.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  Chart 3: drawdown_mc", file=sys.stderr)


def chart4_br_capture(all_df, out):
    """Chart 4: Big Runner Capture bar chart."""
    setup_dark()
    fig, ax = plt.subplots(figsize=(12, 6))

    br_df = all_df[(all_df['is_br_entry'] == True) & (all_df['period'] == 'OOS')]
    if len(br_df) == 0:
        ax.set_title('No Big Runner trades in OOS')
        fig.savefig(out / 'chart4_br_capture.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        return

    # C is 100% capture (hold-to-close)
    c_mean = br_df[br_df['approach'] == 'C']['pnl_dollar'].mean()
    if c_mean <= 0:
        c_mean = 1  # prevent div/0

    captures = []
    means = []
    for approach in APPROACHES:
        sub = br_df[br_df['approach'] == approach]
        if len(sub) > 0:
            m = sub['pnl_dollar'].mean()
            captures.append(m / c_mean * 100)
            means.append(m)
        else:
            captures.append(0)
            means.append(0)

    x = range(len(APPROACHES))
    bars = ax.bar(x, captures, color=[ACOLS[a] for a in APPROACHES],
                  edgecolor='white', lw=0.5)
    for xi, (cap, mn) in enumerate(zip(captures, means)):
        ax.text(xi, cap + 2, f'${mn:,.0f}', ha='center', fontsize=8, color='white')

    ax.set_xticks(list(x))
    ax.set_xticklabels(APPROACHES)
    ax.set_ylabel('% of BR Potential Captured')
    ax.set_title('Big Runner Capture Rate (OOS)', fontsize=14)
    ax.axhline(100, color='grey', ls='--', lw=0.5)
    plt.tight_layout()
    fig.savefig(out / 'chart4_br_capture.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  Chart 4: br_capture", file=sys.stderr)


def chart5_r_distribution(all_df, out):
    """Chart 5: R-distribution histograms (7 subplots)."""
    setup_dark()
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    axes = axes.flatten()

    edges = [-np.inf, -0.01, 0.01, 1, 2, 3, 5, 10, np.inf]
    labels = ['-1R', '0R', '0-1R', '1-2R', '2-3R', '3-5R', '5-10R', '10R+']

    for i, approach in enumerate(APPROACHES):
        ax = axes[i]
        sub = all_df[(all_df['approach'] == approach) &
                     (all_df['period'] == 'OOS') &
                     (all_df['gap_direction'] == 'up')]
        if len(sub) == 0:
            ax.set_title(f'{approach}: N=0')
            continue
        rv = sub['pnl_r'].values
        counts = [((rv > edges[j]) & (rv <= edges[j + 1])).sum()
                  for j in range(len(labels))]
        ax.bar(range(len(labels)), counts, color=ACOLS[approach],
               edgecolor='white', lw=0.3)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=6, rotation=30)
        ax.set_title(f'{approach} (N={len(sub)})', fontsize=9)
        ax.annotate(f"Mean R: {rv.mean():+.2f}\nMed R: {np.median(rv):+.2f}",
                    xy=(0.55, 0.8), xycoords='axes fraction', fontsize=7,
                    color='yellow')

    if len(APPROACHES) < 8:
        axes[-1].axis('off')
    fig.suptitle('R-Multiple Distribution (OOS GapUp)', fontsize=14)
    plt.tight_layout()
    fig.savefig(out / 'chart5_r_distribution.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  Chart 5: r_distribution", file=sys.stderr)


def chart6_trade_duration(all_df, out):
    """Chart 6: Trade duration histograms (7 subplots)."""
    setup_dark()
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    axes = axes.flatten()

    entry_mins = 9 * 60 + 36

    for i, approach in enumerate(APPROACHES):
        ax = axes[i]
        sub = all_df[(all_df['approach'] == approach) &
                     (all_df['period'] == 'OOS') &
                     (all_df['gap_direction'] == 'up')]
        if len(sub) == 0:
            ax.set_title(f'{approach}: N=0')
            continue
        durs = []
        for et in sub['exit_time'].dropna():
            ets = str(et)[:5]
            if ':' in ets:
                parts = ets.split(':')
                exit_m = int(parts[0]) * 60 + int(parts[1])
                durs.append(exit_m - entry_mins)
        if durs:
            ax.hist(durs, bins=30, color=ACOLS[approach], edgecolor='white', lw=0.3)
            ax.axvline(np.median(durs), color='yellow', lw=2,
                       label=f'Med: {np.median(durs):.0f}min')
            ax.legend(fontsize=7)
        ax.set_title(f'{approach}', fontsize=9)
        ax.set_xlabel('Duration (min)', fontsize=7)

    if len(APPROACHES) < 8:
        axes[-1].axis('off')
    fig.suptitle('Trade Duration (OOS GapUp)', fontsize=14)
    plt.tight_layout()
    fig.savefig(out / 'chart6_trade_duration.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  Chart 6: trade_duration", file=sys.stderr)


def chart7_monthly_pnl(all_df, best_approach, out):
    """Chart 7: Monthly PnL — best approach vs CTRL."""
    setup_dark()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

    for ax, approach in [(ax1, 'CTRL'), (ax2, best_approach)]:
        sub = all_df[(all_df['approach'] == approach) &
                     (all_df['period'] == 'OOS') &
                     (all_df['gap_direction'] == 'up')].copy()
        if len(sub) == 0:
            ax.set_title(f'{approach}: No trades')
            continue
        sub['month'] = pd.to_datetime(sub['date']).dt.to_period('M')
        monthly = sub.groupby('month')['pnl_dollar'].sum()
        clrs = ['#06d6a0' if v > 0 else '#e94560' for v in monthly.values]
        x = range(len(monthly))
        ax.bar(x, monthly.values, color=clrs, edgecolor='white', lw=0.3)

        ax2b = ax.twinx()
        cum = monthly.cumsum()
        ax2b.plot(x, cum.values, color='yellow', lw=2)
        ax2b.set_ylabel('Cumulative ($)', color='yellow', fontsize=8)

        ax.set_xticks(list(x))
        ax.set_xticklabels([str(m) for m in monthly.index], rotation=45, fontsize=6)
        ax.set_title(f'{approach} Monthly PnL (OOS GapUp)', fontsize=10)
        ax.set_ylabel('Monthly PnL ($)')

    plt.tight_layout()
    fig.savefig(out / 'chart7_monthly_pnl.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  Chart 7: monthly_pnl", file=sys.stderr)


def chart8_sensitivity(all_df, best_approach, out):
    """Chart 8: Slippage sensitivity for best approach."""
    setup_dark()
    fig, ax = plt.subplots(figsize=(10, 6))

    sub = all_df[(all_df['approach'] == best_approach) &
                 (all_df['period'] == 'OOS') &
                 (all_df['gap_direction'] == 'up')]
    if len(sub) == 0:
        fig.savefig(out / 'chart8_sensitivity.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        return

    pnl_base = sub['pnl_dollar'].values
    shares = sub['shares'].values

    slippages = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03]
    totals = []
    for slip in slippages:
        adj = pnl_base - slip * shares * 2
        totals.append(adj.sum())

    ax.plot(slippages, totals, color=ACOLS[best_approach], lw=2, marker='o')
    ax.axhline(0, color='grey', lw=0.5, ls='--')
    ax.set_xlabel('Slippage per Share ($)')
    ax.set_ylabel('Total PnL ($)')
    ax.set_title(f'Slippage Sensitivity — {best_approach} (OOS GapUp)', fontsize=12)

    # Break-even slippage
    total_rt_shares = (shares * 2).sum()
    if total_rt_shares > 0:
        be_slip = max(0, pnl_base.sum() / total_rt_shares)
        ax.axvline(be_slip, color='yellow', lw=1.5, ls='--',
                   label=f'BE: ${be_slip:.4f}')
        ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(out / 'chart8_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  Chart 8: sensitivity", file=sys.stderr)


def chart9_summary_table(all_kpis, out):
    """Chart 9: Summary table as image."""
    setup_dark()
    fig, ax = plt.subplots(figsize=(18, 8))
    ax.axis('off')

    row_defs = [
        ('N', 'N', 'd'), ('Win%', 'Win%', '.1f'),
        ('Mean R', 'Mean R', '+.2f'), ('PF', 'PF', '.2f'),
        ('Total PnL', 'Total PnL', '$'), ('MaxDD$', 'MaxDD$', '$'),
        ('Med Duration', 'MedDurMin', '.0f'),
        ('WR@1R', 'WR@1R', '.1f'), ('WR@2R', 'WR@2R', '.1f'),
        ('WR@5R', 'WR@5R', '.1f'), ('WR@10R', 'WR@10R', '.1f'),
    ]

    def fmt(val, f):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return '-'
        if f == 'd':
            return str(int(val))
        if f == '$':
            return f"${val:,.0f}"
        return f"{val:{f}}"

    cell_text = []
    for rname, rkey, rfmt in row_defs:
        row = []
        for approach in APPROACHES:
            kpi = all_kpis.get(approach, {})
            if kpi.get('N', 0) == 0:
                row.append('-')
            else:
                row.append(fmt(kpi.get(rkey), rfmt))
        cell_text.append(row)

    table = ax.table(cellText=cell_text,
                     rowLabels=[r[0] for r in row_defs],
                     colLabels=APPROACHES,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.5)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor('white')
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor('#e94560')
            cell.set_text_props(color='white', fontweight='bold')
        elif c == -1:
            cell.set_facecolor('#2a2a4e')
            cell.set_text_props(color='white')
        else:
            cell.set_facecolor(C_BG)
            cell.set_text_props(color='white')

    ax.set_title('KPI Summary — All Approaches (OOS GapUp)', fontsize=14,
                 color='white', pad=20)
    fig.savefig(out / 'chart9_summary_table.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  Chart 9: summary_table", file=sys.stderr)


def chart10_walk_forward_heatmap(wf_df, out):
    """Chart 10: Walk-forward heatmap (PnL per window per approach)."""
    setup_dark()
    fig, ax = plt.subplots(figsize=(18, 6))

    # Filter GapUp only
    wf_up = wf_df[wf_df['gap_dir'] == 'up']
    if len(wf_up) == 0:
        ax.set_title('No walk-forward data')
        fig.savefig(out / 'chart10_walk_forward.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        return

    windows = [w[0] for w in WF_WINDOWS]
    matrix = np.zeros((len(APPROACHES), len(windows)))
    for i, approach in enumerate(APPROACHES):
        for j, wname in enumerate(windows):
            row = wf_up[(wf_up['approach'] == approach) & (wf_up['window'] == wname)]
            if len(row) > 0:
                matrix[i, j] = row['Total PnL'].values[0]

    vmax = max(abs(matrix.min()), abs(matrix.max())) or 1
    cmap = plt.cm.RdYlGn
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect='auto')

    ax.set_xticks(range(len(windows)))
    ax.set_xticklabels(windows, fontsize=8, rotation=30)
    ax.set_yticks(range(len(APPROACHES)))
    ax.set_yticklabels(APPROACHES, fontsize=10)

    for i in range(len(APPROACHES)):
        for j in range(len(windows)):
            v = matrix[i, j]
            ax.text(j, i, f'${v:,.0f}', ha='center', va='center',
                    fontsize=6, color='black' if abs(v) < vmax * 0.5 else 'white')

    plt.colorbar(im, ax=ax, label='PnL ($)')
    ax.set_title('Walk-Forward Heatmap (GapUp)', fontsize=14)
    plt.tight_layout()
    fig.savefig(out / 'chart10_walk_forward.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  Chart 10: walk_forward heatmap", file=sys.stderr)


def chart11_relative_performance(wf_df, out):
    """Chart 11: Relative performance vs CTRL per window."""
    setup_dark()
    fig, ax = plt.subplots(figsize=(16, 7))

    wf_up = wf_df[wf_df['gap_dir'] == 'up']
    windows = [w[0] for w in WF_WINDOWS]

    ctrl_pnl = {}
    for wname in windows:
        row = wf_up[(wf_up['approach'] == 'CTRL') & (wf_up['window'] == wname)]
        ctrl_pnl[wname] = row['Total PnL'].values[0] if len(row) > 0 else 0

    for approach in APPROACHES:
        if approach == 'CTRL':
            continue
        deltas = []
        for wname in windows:
            row = wf_up[(wf_up['approach'] == approach) & (wf_up['window'] == wname)]
            a_pnl = row['Total PnL'].values[0] if len(row) > 0 else 0
            deltas.append(a_pnl - ctrl_pnl.get(wname, 0))
        n_better = sum(1 for d in deltas if d > 0)
        ax.plot(range(len(windows)), deltas, color=ACOLS[approach], lw=2,
                marker='o', label=f'{approach} ({n_better}/{len(windows)} better)')

    ax.axhline(0, color='cyan', lw=1, ls='--', label='CTRL baseline')
    ax.set_xticks(range(len(windows)))
    ax.set_xticklabels(windows, fontsize=8, rotation=30)
    ax.set_ylabel('Delta PnL vs CTRL ($)')
    ax.set_title('Relative Performance vs CTRL per Window (GapUp)', fontsize=14)
    ax.legend(fontsize=8, loc='best')
    plt.tight_layout()
    fig.savefig(out / 'chart11_relative_perf.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  Chart 11: relative performance", file=sys.stderr)


# ============================================================
# TEXT OUTPUT: KPI Summary
# ============================================================
def format_kpi_block(lines, kpi, label):
    """Format KPI block for one approach/segment."""
    n = kpi.get('N', 0)
    low = ' [LOW N]' if 0 < n < 20 else ''
    lines.append(f"\n  {label} — N={n}{low}")
    if n == 0:
        return

    def r(name, key, fmt='$'):
        v = kpi.get(key, np.nan)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            vs = '-'
        elif fmt == '$':
            vs = f"${v:>10,.0f}"
        elif fmt == 'r':
            vs = f"{v:>+10.3f}"
        elif fmt == '%':
            vs = f"{v:>10.1f}%"
        elif fmt == 'd':
            vs = f"{int(v):>10d}"
        else:
            vs = f"{v:>10.2f}"
        lines.append(f"    {name:<22} {vs}")

    r('Win%', 'Win%', '%')
    r('Loss%', 'Loss%', '%')
    r('BE%', 'BE%', '%')
    r('Total PnL', 'Total PnL', '$')
    r('Mean PnL', 'Mean PnL', '$')
    r('Median PnL', 'Median PnL', '$')
    r('StdDev', 'StdDev', '$')
    r('Skew', 'Skew', 'f')
    r('Best', 'Best', '$')
    r('Worst', 'Worst', '$')
    r('P10', 'P10', '$')
    r('P90', 'P90', '$')
    lines.append('')
    r('Mean R', 'Mean R', 'r')
    r('Median R', 'Median R', 'r')
    r('WR@1R', 'WR@1R', '%')
    r('WR@2R', 'WR@2R', '%')
    r('WR@3R', 'WR@3R', '%')
    r('WR@5R', 'WR@5R', '%')
    r('WR@10R', 'WR@10R', '%')
    lines.append('')
    r('Profit Factor', 'PF', 'f')
    r('Payoff Ratio', 'Payoff', 'f')
    r('Max Cons Losses', 'MaxConsL', 'd')
    r('Max DD $', 'MaxDD$', '$')
    r('Max DD %50K', 'MaxDD% of 50K', '%')
    r('Max DD Dur', 'MaxDDDur', 'd')
    lines.append('')
    r('% Reach +1R', '%Reach1R', '%')
    r('% Survive 10:00', '%Surv1000', '%')
    r('% InitSL', '%InitSL', '%')
    r('% Trail', '%Trail', '%')
    r('% Timeout', '%Timeout', '%')
    r('% BE Stop', '%BEStop', '%')
    r('Med Duration (min)', 'MedDurMin', 'f')


def write_comparison_summary(all_df, all_kpis_oos, wf_df, out):
    """Write comparison_summary.txt."""
    lines = []
    lines.append("=" * 70)
    lines.append("BACKTEST v3: BIG RUNNER CAPTURE — COMPARISON SUMMARY")
    lines.append("=" * 70)
    lines.append(f"Risk: ${RISK:.0f}/Trade, Slippage: ${SLIPPAGE}")
    lines.append(f"IS: {IS_START} to {IS_END}, OOS: {OOS_START} to {OOS_END}")
    lines.append("")

    ctrl_kpi = all_kpis_oos.get('CTRL', {})

    for approach in APPROACHES:
        if approach == 'CTRL':
            continue
        kpi = all_kpis_oos.get(approach, {})
        lines.append(f"\n{'=' * 70}")
        lines.append(f"=== {approach} vs CTRL (OOS GapUp) ===")
        lines.append(f"{'=' * 70}")

        def _v(k, d):
            return d.get(k, 0) if d.get('N', 0) > 0 else 0

        lines.append(f"  CTRL:  Total=${_v('Total PnL', ctrl_kpi):>10,.0f}  "
                     f"PF={_v('PF', ctrl_kpi):.2f}  "
                     f"MaxDD=${_v('MaxDD$', ctrl_kpi):>8,.0f}  "
                     f"MeanR={_v('Mean R', ctrl_kpi):+.3f}")
        lines.append(f"  {approach:5s}:  Total=${_v('Total PnL', kpi):>10,.0f}  "
                     f"PF={_v('PF', kpi):.2f}  "
                     f"MaxDD=${_v('MaxDD$', kpi):>8,.0f}  "
                     f"MeanR={_v('Mean R', kpi):+.3f}")

        dt = _v('Total PnL', kpi) - _v('Total PnL', ctrl_kpi)
        lines.append(f"  Delta: Total=${dt:>+10,.0f}")

        # Top 3 largest PnL differences
        ctrl_trades = all_df[(all_df['approach'] == 'CTRL') &
                             (all_df['period'] == 'OOS') &
                             (all_df['gap_direction'] == 'up')].set_index(['ticker', 'date'])
        a_trades = all_df[(all_df['approach'] == approach) &
                          (all_df['period'] == 'OOS') &
                          (all_df['gap_direction'] == 'up')].set_index(['ticker', 'date'])

        common = ctrl_trades.index.intersection(a_trades.index)
        if len(common) > 0:
            diffs = []
            for key in common:
                cp = ctrl_trades.loc[key, 'pnl_dollar']
                ap = a_trades.loc[key, 'pnl_dollar']
                if isinstance(cp, pd.Series):
                    cp = cp.iloc[0]
                if isinstance(ap, pd.Series):
                    ap = ap.iloc[0]
                diffs.append((key, cp, ap, ap - cp))
            diffs.sort(key=lambda x: abs(x[3]), reverse=True)
            lines.append(f"\n  Top 3 largest differences:")
            for j, (key, cp, ap, d) in enumerate(diffs[:3]):
                lines.append(f"    {j + 1}. {key[0]} {key[1]}: CTRL=${cp:+,.0f} "
                             f"{approach}=${ap:+,.0f} Delta=${d:+,.0f}")

    # Ranking
    lines.append(f"\n\n{'=' * 70}")
    lines.append("=== RANKING (OOS GapUp) ===")
    lines.append(f"{'=' * 70}")

    ranked = sorted(APPROACHES,
                    key=lambda a: all_kpis_oos.get(a, {}).get('Total PnL', 0),
                    reverse=True)
    for i, a in enumerate(ranked):
        kpi = all_kpis_oos.get(a, {})
        lines.append(f"  {i + 1}. {a:5s}: Total=${kpi.get('Total PnL', 0):>10,.0f}  "
                     f"PF={kpi.get('PF', 0):.2f}  "
                     f"MaxDD=${kpi.get('MaxDD$', 0):>8,.0f}")

    # Walk-forward consistency
    lines.append(f"\n\n{'=' * 70}")
    lines.append("=== WALK-FORWARD KONSISTENZ (GapUp) ===")
    lines.append(f"{'=' * 70}")

    wf_up = wf_df[wf_df['gap_dir'] == 'up']
    windows = [w[0] for w in WF_WINDOWS]
    ctrl_wf = {}
    for wname in windows:
        row = wf_up[(wf_up['approach'] == 'CTRL') & (wf_up['window'] == wname)]
        ctrl_wf[wname] = row['Total PnL'].values[0] if len(row) > 0 else 0

    for approach in APPROACHES:
        n_prof = 0
        n_beat = 0
        for wname in windows:
            row = wf_up[(wf_up['approach'] == approach) & (wf_up['window'] == wname)]
            pnl = row['Total PnL'].values[0] if len(row) > 0 else 0
            if pnl > 0:
                n_prof += 1
            if pnl > ctrl_wf.get(wname, 0):
                n_beat += 1
        lines.append(f"  {approach:5s}: {n_prof}/{len(windows)} profitabel  "
                     f"{n_beat}/{len(windows)} besser als CTRL")

    # Recommendation
    lines.append(f"\n\n{'=' * 70}")
    lines.append("=== EMPFEHLUNG ===")
    lines.append(f"{'=' * 70}")

    best = ranked[0]
    best_kpi = all_kpis_oos.get(best, {})
    ctrl_total = ctrl_kpi.get('Total PnL', 0)
    best_total = best_kpi.get('Total PnL', 0)

    if best == 'CTRL' or best_total <= ctrl_total * 1.1:
        lines.append("  Kein Ansatz schlaegt CTRL robust. EMPFEHLUNG: CTRL beibehalten.")
    else:
        lines.append(f"  Bester Ansatz: {best} mit ${best_total:,.0f} Total PnL OOS")
        lines.append(f"  (CTRL: ${ctrl_total:,.0f})")
        lines.append(f"  Differenz: ${best_total - ctrl_total:+,.0f}")

    output = "\n".join(lines)
    with open(out / 'comparison_summary.txt', 'w', encoding='utf-8') as f:
        f.write(output)
    return output


# ============================================================
# MAIN
# ============================================================
def main():
    t_start = time.time()
    np.random.seed(42)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- Load metadata ---
    print("Loading metadata_v8_5...", file=sys.stderr)
    meta = pd.read_parquet(META_PATH)
    meta = meta[meta['date'] >= IS_START].copy()
    meta['date'] = meta['date'].astype(str)
    print(f"  {len(meta)} rows", file=sys.stderr)

    # --- Build all trades ---
    print("\nSimulating trades (7 approaches each)...", file=sys.stderr)
    all_df = build_trades(meta)

    if len(all_df) == 0:
        print("ERROR: No trades generated!", file=sys.stderr)
        return

    all_df.to_parquet(RESULTS_DIR / 'all_trades_v3.parquet', index=False)
    print(f"  Saved {len(all_df)} trade records.", file=sys.stderr)

    # --- CTRL Check vs v2 ---
    print("\n--- CTRL Check vs v2 ---", file=sys.stderr)
    if V2_TRADES_PATH.exists():
        v2 = pd.read_parquet(V2_TRADES_PATH)
        v2_uni = v2[v2['trail_type'] == 'TRAIL_UNIFORM']
        v3_ctrl = all_df[all_df['approach'] == 'CTRL']

        v2_total = v2_uni['pnl_dollar'].sum()
        v3_total = v3_ctrl['pnl_dollar'].sum()
        print(f"  v2 TRAIL_UNIFORM Total: ${v2_total:,.0f}", file=sys.stderr)
        print(f"  v3 CTRL Total:          ${v3_total:,.0f}", file=sys.stderr)
        print(f"  Delta:                  ${v3_total - v2_total:+,.0f}", file=sys.stderr)

        # Per-period comparison
        for per in ['IS', 'OOS']:
            for gd in ['up', 'down']:
                v2_sub = v2_uni[(v2_uni['period'] == per) &
                                (v2_uni['gap_direction'] == gd)]
                v3_sub = v3_ctrl[(v3_ctrl['period'] == per) &
                                 (v3_ctrl['gap_direction'] == gd)]
                print(f"  {per} Gap{'Up' if gd == 'up' else 'Dn'}: "
                      f"v2=${v2_sub['pnl_dollar'].sum():,.0f} "
                      f"v3=${v3_sub['pnl_dollar'].sum():,.0f} "
                      f"N_v2={len(v2_sub)} N_v3={len(v3_sub)}",
                      file=sys.stderr)
    else:
        print("  v2 trades not found — skipping check.", file=sys.stderr)

    # --- Trade counts ---
    print("\nTrade counts:", file=sys.stderr)
    for approach in APPROACHES:
        for gd in ['up', 'down']:
            for per in ['IS', 'OOS']:
                n = len(all_df[(all_df['approach'] == approach) &
                               (all_df['gap_direction'] == gd) &
                               (all_df['period'] == per)])
                if n > 0:
                    gl = 'GapUp' if gd == 'up' else 'GapDn'
                    print(f"  {approach} {gl} {per}: {n}", file=sys.stderr)

    # Zoom stats
    for approach in APPROACHES:
        sub = all_df[all_df['approach'] == approach]
        nz = sub['used_1sec_zoom'].sum() if 'used_1sec_zoom' in sub.columns else 0
        print(f"  {approach}: {int(nz)} trades used 1-sec zoom", file=sys.stderr)

    # --- KPIs ---
    print("\nComputing KPIs...", file=sys.stderr)
    all_kpis = {}
    all_kpis_oos_gapup = {}

    for approach in APPROACHES:
        for gd in ['up', 'down']:
            for per in ['IS', 'OOS']:
                sub = all_df[(all_df['approach'] == approach) &
                             (all_df['gap_direction'] == gd) &
                             (all_df['period'] == per)]
                gl = 'GapUp' if gd == 'up' else 'GapDn'
                label = f"{approach} {gl} {per}"
                kpi = compute_kpis(sub, label)
                all_kpis[(approach, gd, per)] = kpi
                if per == 'OOS' and gd == 'up':
                    all_kpis_oos_gapup[approach] = kpi

    # --- Walk-Forward ---
    print("\nWalk-Forward Analysis...", file=sys.stderr)
    wf_df = walk_forward_analysis(all_df)

    # Write walk-forward table
    wf_lines = []
    wf_lines.append("=" * 70)
    wf_lines.append("WALK-FORWARD ANALYSIS")
    wf_lines.append("=" * 70)
    wf_up = wf_df[wf_df['gap_dir'] == 'up']

    windows = [w[0] for w in WF_WINDOWS]
    header = f"{'Window':<10} "
    for a in APPROACHES:
        header += f"| {a:>10} "
    wf_lines.append(header)
    wf_lines.append("-" * len(header))

    for wname in windows:
        row_str = f"{wname:<10} "
        for a in APPROACHES:
            sub = wf_up[(wf_up['approach'] == a) & (wf_up['window'] == wname)]
            pnl = sub['Total PnL'].values[0] if len(sub) > 0 else 0
            row_str += f"| ${pnl:>9,.0f} "
        wf_lines.append(row_str)

    with open(RESULTS_DIR / 'walk_forward.txt', 'w', encoding='utf-8') as f:
        f.write("\n".join(wf_lines))
    print("  Saved walk_forward.txt", file=sys.stderr)

    # --- Monte Carlo ---
    print("\nMonte Carlo simulations...", file=sys.stderr)
    mc_results = {}
    for approach in tqdm(APPROACHES, desc="MC", file=sys.stderr):
        sub = all_df[(all_df['approach'] == approach) &
                     (all_df['period'] == 'OOS') &
                     (all_df['gap_direction'] == 'up')]
        if len(sub) >= 5:
            mc = monte_carlo(sub['pnl_dollar'].values)
            mc_results[approach] = mc
        else:
            mc_results[approach] = None

    # --- Big Runner Analysis ---
    print("\nBig Runner Analysis...", file=sys.stderr)
    br_lines = []
    br_lines.append("=" * 70)
    br_lines.append("BIG RUNNER ANALYSE (OOS)")
    br_lines.append("=" * 70)

    br_oos = all_df[(all_df['period'] == 'OOS') & (all_df['is_br_entry'] == True)]
    non_br_oos = all_df[(all_df['period'] == 'OOS') & (all_df['is_br_entry'] == False)]

    # C is the reference (100% capture)
    c_br_mean = 0
    c_br = br_oos[br_oos['approach'] == 'C']
    if len(c_br) > 0:
        c_br_mean = c_br['pnl_dollar'].mean()

    for gd_label, gd in [('GapUp', 'up'), ('GapDn', 'down')]:
        br_lines.append(f"\n--- {gd_label} ---")
        br_gd = br_oos[br_oos['gap_direction'] == gd]
        nbr_gd = non_br_oos[non_br_oos['gap_direction'] == gd]
        c_ref = br_gd[br_gd['approach'] == 'C']['pnl_dollar'].mean() if len(
            br_gd[br_gd['approach'] == 'C']) > 0 else 1

        br_lines.append(f"  {'Approach':<8} | {'N BR':>5} | {'Mean BR$':>10} | "
                        f"{'Mean NR$':>10} | {'Spread$':>10} | {'Capture%':>8}")
        br_lines.append(f"  {'-' * 65}")

        for approach in APPROACHES:
            ab = br_gd[br_gd['approach'] == approach]
            anb = nbr_gd[nbr_gd['approach'] == approach]
            n_br = len(ab)
            mean_br = ab['pnl_dollar'].mean() if n_br > 0 else 0
            mean_nr = anb['pnl_dollar'].mean() if len(anb) > 0 else 0
            spread = mean_br - mean_nr
            capture = mean_br / c_ref * 100 if c_ref != 0 and n_br > 0 else 0
            br_lines.append(f"  {approach:<8} | {n_br:>5} | ${mean_br:>9,.0f} | "
                            f"${mean_nr:>9,.0f} | ${spread:>9,.0f} | {capture:>7.1f}%")

    with open(RESULTS_DIR / 'big_runner_analysis.txt', 'w', encoding='utf-8') as f:
        f.write("\n".join(br_lines))
    print("  Saved big_runner_analysis.txt", file=sys.stderr)

    # --- Charts ---
    print("\nGenerating charts...", file=sys.stderr)

    # Determine best approach (highest Total PnL OOS GapUp)
    best_approach = max(APPROACHES,
                        key=lambda a: all_kpis_oos_gapup.get(a, {}).get('Total PnL', 0))
    print(f"  Best approach (OOS GapUp Total PnL): {best_approach}", file=sys.stderr)

    chart1_equity_tier2_gapup(all_df, RESULTS_DIR)
    chart2_equity_tier1_gapup(all_df, RESULTS_DIR)
    chart3_drawdown_mc(all_df, mc_results, RESULTS_DIR)
    chart4_br_capture(all_df, RESULTS_DIR)
    chart5_r_distribution(all_df, RESULTS_DIR)
    chart6_trade_duration(all_df, RESULTS_DIR)
    chart7_monthly_pnl(all_df, best_approach, RESULTS_DIR)
    chart8_sensitivity(all_df, best_approach, RESULTS_DIR)
    chart9_summary_table(all_kpis_oos_gapup, RESULTS_DIR)
    chart10_walk_forward_heatmap(wf_df, RESULTS_DIR)
    chart11_relative_performance(wf_df, RESULTS_DIR)

    # --- Text output ---
    print("\nWriting text output...", file=sys.stderr)

    # Main KPI summary
    kpi_lines = []
    kpi_lines.append("=" * 70)
    kpi_lines.append("BACKTEST v3: BIG RUNNER CAPTURE — KPI SUMMARY")
    kpi_lines.append("=" * 70)
    kpi_lines.append(f"Risk: ${RISK:.0f}/Trade, SL: 0.25 ADR (B: 0.50 ADR), "
                     f"Slippage: ${SLIPPAGE}")
    kpi_lines.append(f"IS: {IS_START} to {IS_END}, OOS: {OOS_START} to {OOS_END}")
    kpi_lines.append(f"Approaches: {', '.join(APPROACHES)}")

    for approach in APPROACHES:
        kpi_lines.append(f"\n\n{'=' * 70}")
        kpi_lines.append(f"=== {approach} ===")
        kpi_lines.append(f"{'=' * 70}")

        for gd_label, gd in [('GapUp', 'up'), ('GapDn', 'down')]:
            for per in ['IS', 'OOS']:
                kpi = all_kpis.get((approach, gd, per), {'N': 0})
                format_kpi_block(kpi_lines, kpi, f"{approach} {gd_label} {per}")

    # Monte Carlo summary
    kpi_lines.append(f"\n\n{'=' * 70}")
    kpi_lines.append("=== MONTE CARLO (OOS GapUp) ===")
    kpi_lines.append(f"{'=' * 70}")

    for approach in APPROACHES:
        mc = mc_results.get(approach)
        if mc is None:
            kpi_lines.append(f"\n  {approach}: N < 5, skipped")
            continue
        kpi_lines.append(f"\n  {approach}:")
        kpi_lines.append(f"    Final Equity: Median ${mc['fe_median']:>10,.0f}  "
                         f"[P5: ${mc['fe_p5']:>10,.0f}  P95: ${mc['fe_p95']:>10,.0f}]")
        kpi_lines.append(f"    Max DD:       Median ${mc['dd_median']:>10,.0f}  "
                         f"[P5: ${mc['dd_p5']:>10,.0f}  P95: ${mc['dd_p95']:>10,.0f}]")
        kpi_lines.append(f"    Worst DD:     ${mc['worst_dd']:>10,.0f}")
        kpi_lines.append(f"    Max Cons L:   Median {mc['cl_median']:.0f}  "
                         f"[P95: {mc['cl_p95']:.0f}]")
        for t in [10000, 15000, 20000]:
            kpi_lines.append(f"    P(DD >= ${t:>6,}): {mc[f'p_ruin_{t}']:>5.1f}%")

    elapsed = time.time() - t_start
    kpi_lines.append(f"\n\nLaufzeit: {elapsed:.1f}s")

    with open(RESULTS_DIR / 'kpis_summary.txt', 'w', encoding='utf-8') as f:
        f.write("\n".join(kpi_lines))

    # Comparison summary
    comp_text = write_comparison_summary(all_df, all_kpis_oos_gapup, wf_df, RESULTS_DIR)

    # Print to stdout
    print("\n".join(kpi_lines))
    print("\n" + comp_text)

    print(f"\n\nTotal time: {elapsed:.1f}s", file=sys.stderr)
    print(f"Results saved to {RESULTS_DIR}/", file=sys.stderr)


if __name__ == '__main__':
    main()
