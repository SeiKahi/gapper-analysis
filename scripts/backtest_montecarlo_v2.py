"""
Backtest + Monte Carlo Simulation v2.0
=======================================
Aufgaben 1-6: Trade-Simulation (3 Exit-Strategien), Setup-Klassifikation,
KPIs, Monte Carlo, Charts, Sensitivitaet.

Kernneuerung vs v1:
  - 4-State Trail-Machine: INIT -> BREAKEVEN -> TRAILING_NORMAL -> TRAILING_TIGHT
  - Trail-Switching um 10:00 basierend auf drift_30min, rvol_open_30min, od_strength
  - 3 Exit-Strategien pro Trade: TRAIL_UNIFORM, TRAIL_3TIER, FIX_2R
  - 12 Kombinationen (Tier x Gap x Exit) fuer KPIs
  - Vergleichscharts TRAIL_3TIER vs TRAIL_UNIFORM

Output: results/backtest_v2/
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

# ============================================================
# CONSTANTS
# ============================================================
RAW_1MIN_DIR = Path('data/raw_1min')
RAW_1SEC_DIR = Path('data/raw_1sec')
RESULTS_DIR = Path('results/backtest_v2')

RISK_PER_TRADE = 1000.0
SL_ADR_FRACTION = 0.25
SLIPPAGE_PER_SHARE = 0.00
MC_ITERATIONS = 1000

C_ACCENT = '#e94560'
C_ACCENT2 = '#00b4d8'
C_GREEN = '#06d6a0'
C_YELLOW = '#ffd166'
C_BG = '#1a1a2e'

IS_START = '2021-02-21'
IS_END = '2023-12-31'
OOS_START = '2024-01-01'
OOS_END = '2026-02-13'

# ============================================================
# 1-SEC CACHE
# ============================================================
_sec_cache = {}

def load_1sec(ticker, date):
    key = (ticker, date)
    if key not in _sec_cache:
        path = RAW_1SEC_DIR / ticker / f"{date}.parquet"
        _sec_cache[key] = pd.read_parquet(path) if path.exists() else None
    return _sec_cache[key]


# ============================================================
# AUFGABE 1: TRAIL SIMULATION — 4-State Machine
# ============================================================
def simulate_trail(bars_rth, entry, sl_level, trade_dir, adr, one_r,
                   ticker, date, meta_row, trail_mode='uniform',
                   slippage=0.0, risk=1000.0):
    """
    Simulate a trade with trailing stop.

    trail_mode:
      'uniform'  — always peak - 0.5R after +1R (TRAIL_UNIFORM)
      '3tier'    — switches to peak - 0.25R at 10:00 if Stufe 3 criteria met (TRAIL_3TIER)

    State machine:
      INIT        — SL at sl_level (0.25 ADR)
      BREAKEVEN   — SL at entry (activated when +1R reached)
      TRAILING_NORMAL — trail = peak - 0.5R
      TRAILING_TIGHT  — trail = peak - 0.25R (only in 3tier mode, Stufe 3)
    """
    sl_dist = abs(entry - sl_level)
    if sl_dist <= 0 or pd.isna(sl_dist) or adr <= 0:
        return None

    shares = risk / sl_dist
    one_r_level = entry + one_r if trade_dir == 'long' else entry - one_r

    state = 'INIT'
    peak_fav = entry
    current_sl = sl_level
    mfe = 0.0
    reached_1r = False
    zoom_used = False
    zoom_changed = False
    exit_price = np.nan
    exit_time = None
    exit_reason = 'timeout'
    done = False
    trail_tier = np.nan  # 1/2/3, set at 10:00
    tier_switched = False  # whether we switched to TIGHT at 10:00

    sec_bars = load_1sec(ticker, date)
    post = bars_rth[(bars_rth['time_et'] >= '09:36') & (bars_rth['time_et'] <= '15:55')]
    if len(post) == 0:
        return None

    # Read metadata fields for 10:00 switch (no lookahead — these are known at 10:00)
    rvol_30 = meta_row.get('rvol_open_30min', np.nan)
    od_str = meta_row.get('od_strength', np.nan)
    c1000 = meta_row.get('close_1000', np.nan)
    rvol_5 = meta_row.get('rvol_5', np.nan)
    gap_size_adr = meta_row.get('gap_size_in_adr', np.nan)
    gap_dir = meta_row.get('gap_direction', 'up')

    # Compute drift_30min (direction-aware)
    drift_30min = np.nan
    if pd.notna(c1000) and pd.notna(entry) and adr > 0:
        if gap_dir == 'up':
            drift_30min = (c1000 - entry) / adr
        else:
            drift_30min = (entry - c1000) / adr

    def update_trail(peak, state_now):
        """Compute trail level based on current state."""
        if state_now == 'TRAILING_TIGHT':
            if trade_dir == 'long':
                return peak - 0.25 * one_r
            else:
                return peak + 0.25 * one_r
        elif state_now in ('TRAILING_NORMAL', 'BREAKEVEN'):
            if trade_dir == 'long':
                return peak - 0.5 * one_r
            else:
                return peak + 0.5 * one_r
        return current_sl  # INIT — keep original SL

    def check_stop(price_low, price_high, sl):
        if trade_dir == 'long':
            return price_low <= sl
        else:
            return price_high >= sl

    ten_oclock_checked = False

    for _, bar in post.iterrows():
        if done:
            break
        bt = str(bar['time_et'])[:5]

        # --- 10:00 Trail Switching (EINMALIG) ---
        if not ten_oclock_checked and bt >= '10:00' and trail_mode == '3tier':
            ten_oclock_checked = True
            # Only switch if we're already trailing (reached +1R)
            # But assign tier regardless for reporting
            if pd.notna(drift_30min) and pd.notna(rvol_30):
                # STUFE 3 — TIGHT TRAIL
                cond_a = (drift_30min > 0.25 and rvol_30 > 5 and
                          pd.notna(od_str) and od_str > 1.0)
                cond_b = (drift_30min > 0.50 and rvol_30 > 5)
                if cond_a or cond_b:
                    trail_tier = 3
                    if state in ('TRAILING_NORMAL', 'BREAKEVEN'):
                        state = 'TRAILING_TIGHT'
                        tier_switched = True
                        current_sl = update_trail(peak_fav, state)
                    elif state == 'TRAILING_TIGHT':
                        pass  # already tight
                # STUFE 2 — NORMAL, NICHT MANUELL RAUS
                elif ((pd.notna(rvol_5) and rvol_5 > 5) or
                      (pd.notna(gap_size_adr) and gap_size_adr > 2) or
                      drift_30min > 0.25):
                    trail_tier = 2
                    # No mechanical difference from Stufe 1
                # STUFE 1 — NORMAL
                else:
                    trail_tier = 1
            else:
                trail_tier = 1

        if not ten_oclock_checked and bt >= '10:00' and trail_mode == 'uniform':
            ten_oclock_checked = True
            # For uniform: still assign tier for analysis, but don't switch
            if pd.notna(drift_30min) and pd.notna(rvol_30):
                cond_a = (drift_30min > 0.25 and rvol_30 > 5 and
                          pd.notna(od_str) and od_str > 1.0)
                cond_b = (drift_30min > 0.50 and rvol_30 > 5)
                if cond_a or cond_b:
                    trail_tier = 3
                elif ((pd.notna(rvol_5) and rvol_5 > 5) or
                      (pd.notna(gap_size_adr) and gap_size_adr > 2) or
                      drift_30min > 0.25):
                    trail_tier = 2
                else:
                    trail_tier = 1
            else:
                trail_tier = 1

        # --- Bar evaluation ---
        if trade_dir == 'long':
            stop_possible = bar['low'] <= current_sl
            fav_new = bar['high'] > peak_fav
            new_peak_val = bar['high']
            bar_mfe = bar['high'] - entry
        else:
            stop_possible = bar['high'] >= current_sl
            fav_new = bar['low'] < peak_fav
            new_peak_val = bar['low']
            bar_mfe = entry - bar['low']

        # Check +1R reachable this bar
        if trade_dir == 'long':
            can_reach_1r = bar['high'] >= one_r_level
        else:
            can_reach_1r = bar['low'] <= one_r_level

        ambig = stop_possible and (fav_new or can_reach_1r)

        if ambig:
            resolved = False
            can_zoom = (sec_bars is not None and bt >= '09:30' and bt < '09:45')
            if can_zoom:
                mins = sec_bars[sec_bars['time_et'].str[:5] == bt]
                if len(mins) > 0:
                    zoom_used = True
                    state_before = state
                    for _, s in mins.iterrows():
                        # Check stop
                        if check_stop(s['low'], s['high'], current_sl):
                            exit_price = current_sl
                            exit_time = s['time_et']
                            if state == 'INIT':
                                exit_reason = 'init_sl'
                            else:
                                exit_reason = 'trail'
                            resolved = done = True
                            break

                        # Update peak
                        if trade_dir == 'long':
                            if s['high'] > peak_fav:
                                peak_fav = s['high']
                            mfe = max(mfe, s['high'] - entry)
                        else:
                            if s['low'] < peak_fav:
                                peak_fav = s['low']
                            mfe = max(mfe, entry - s['low'])

                        # State transition: +1R
                        if state == 'INIT':
                            if ((trade_dir == 'long' and peak_fav >= one_r_level) or
                                    (trade_dir == 'short' and peak_fav <= one_r_level)):
                                state = 'BREAKEVEN'
                                reached_1r = True
                                current_sl = entry  # breakeven
                                # Immediately move to trailing
                                state = 'TRAILING_NORMAL'
                                current_sl = update_trail(peak_fav, state)
                        elif state == 'BREAKEVEN':
                            state = 'TRAILING_NORMAL'
                            current_sl = update_trail(peak_fav, state)
                        elif state in ('TRAILING_NORMAL', 'TRAILING_TIGHT'):
                            new_sl = update_trail(peak_fav, state)
                            # Trail only moves forward (tighter)
                            if trade_dir == 'long':
                                current_sl = max(current_sl, new_sl)
                            else:
                                current_sl = min(current_sl, new_sl)

                    if not resolved:
                        # After zoom: re-check if stop still hit on remaining bar data
                        if check_stop(bar['low'], bar['high'], current_sl):
                            exit_price = current_sl
                            exit_time = bt
                            exit_reason = 'init_sl' if state == 'INIT' else 'trail'
                            resolved = done = True

                    if state_before != state:
                        zoom_changed = True

            if not resolved:
                # Conservative: SL first
                exit_price = current_sl
                exit_time = bt
                exit_reason = 'init_sl' if state == 'INIT' else 'trail'
                done = True

        elif stop_possible:
            exit_price = current_sl
            exit_time = bt
            exit_reason = 'init_sl' if state == 'INIT' else 'trail'
            done = True

        else:
            # No stop — update peak and state
            if trade_dir == 'long':
                if bar['high'] > peak_fav:
                    peak_fav = bar['high']
                mfe = max(mfe, bar['high'] - entry)
            else:
                if bar['low'] < peak_fav:
                    peak_fav = bar['low']
                mfe = max(mfe, entry - bar['low'])

            if state == 'INIT':
                if ((trade_dir == 'long' and peak_fav >= one_r_level) or
                        (trade_dir == 'short' and peak_fav <= one_r_level)):
                    state = 'BREAKEVEN'
                    reached_1r = True
                    current_sl = entry
                    state = 'TRAILING_NORMAL'
                    current_sl = update_trail(peak_fav, state)
            elif state == 'BREAKEVEN':
                state = 'TRAILING_NORMAL'
                current_sl = update_trail(peak_fav, state)
            elif state in ('TRAILING_NORMAL', 'TRAILING_TIGHT'):
                new_sl = update_trail(peak_fav, state)
                if trade_dir == 'long':
                    current_sl = max(current_sl, new_sl)
                else:
                    current_sl = min(current_sl, new_sl)

    # Timeout
    if not done:
        last = post.iloc[-1]
        exit_price = last['close']
        exit_time = str(last['time_et'])[:5]

    if trade_dir == 'long':
        pnl_raw = (exit_price - entry) * shares
        mfe_r = mfe / one_r if one_r > 0 else 0
        peak_r = (peak_fav - entry) / one_r if one_r > 0 else 0
    else:
        pnl_raw = (entry - exit_price) * shares
        mfe_r = mfe / one_r if one_r > 0 else 0
        peak_r = (entry - peak_fav) / one_r if one_r > 0 else 0

    pnl_dollar = pnl_raw - slippage * shares * 2
    pnl_r = pnl_dollar / risk

    return {
        'exit_price': exit_price, 'exit_time': exit_time,
        'exit_reason': exit_reason,
        'pnl_dollar': pnl_dollar, 'pnl_r': pnl_r,
        'peak_r': peak_r, 'mfe_r': mfe_r, 'shares': shares,
        'reached_1r': reached_1r,
        'trail_tier': trail_tier,
        'drift_30min': drift_30min,
        'zoom_used': zoom_used, 'zoom_changed': zoom_changed,
    }


# ============================================================
# AUFGABE 1b: FIX_2R SIMULATION
# ============================================================
def simulate_fix_2r(bars_rth, entry, sl_level, trade_dir, adr, one_r,
                    ticker, date, slippage=0.0, risk=1000.0):
    """FIX 2R: target at +/-2*SL_dist, exit on target/SL/timeout."""
    sl_dist = abs(entry - sl_level)
    if sl_dist <= 0 or pd.isna(sl_dist) or adr <= 0:
        return None

    shares = risk / sl_dist
    target = entry + 2 * sl_dist if trade_dir == 'long' else entry - 2 * sl_dist

    mfe = 0.0
    zoom_used = False
    zoom_changed = False
    exit_price = np.nan
    exit_time = None
    exit_reason = 'timeout'
    done = False

    sec_bars = load_1sec(ticker, date)
    post = bars_rth[(bars_rth['time_et'] >= '09:36') & (bars_rth['time_et'] <= '15:55')]
    if len(post) == 0:
        return None

    for _, bar in post.iterrows():
        if done:
            break
        bt = str(bar['time_et'])[:5]

        if trade_dir == 'long':
            sl_hit = bar['low'] <= sl_level
            tgt_hit = bar['high'] >= target
        else:
            sl_hit = bar['high'] >= sl_level
            tgt_hit = bar['low'] <= target

        if sl_hit and tgt_hit:
            resolved = False
            can_zoom = (sec_bars is not None and bt >= '09:30' and bt < '09:45')
            if can_zoom:
                mins = sec_bars[sec_bars['time_et'].str[:5] == bt]
                if len(mins) > 0:
                    zoom_used = True
                    for _, s in mins.iterrows():
                        if trade_dir == 'long':
                            if s['low'] <= sl_level:
                                exit_price = sl_level
                                exit_time = s['time_et']
                                exit_reason = 'sl_hit'
                                resolved = done = True
                                break
                            if s['high'] >= target:
                                exit_price = target
                                exit_time = s['time_et']
                                exit_reason = 'target'
                                zoom_changed = True
                                resolved = done = True
                                break
                            mfe = max(mfe, s['high'] - entry)
                        else:
                            if s['high'] >= sl_level:
                                exit_price = sl_level
                                exit_time = s['time_et']
                                exit_reason = 'sl_hit'
                                resolved = done = True
                                break
                            if s['low'] <= target:
                                exit_price = target
                                exit_time = s['time_et']
                                exit_reason = 'target'
                                zoom_changed = True
                                resolved = done = True
                                break
                            mfe = max(mfe, entry - s['low'])
                    if not resolved:
                        exit_price = sl_level
                        exit_time = bt
                        exit_reason = 'sl_hit'
                        resolved = done = True

            if not resolved:
                exit_price = sl_level
                exit_time = bt
                exit_reason = 'sl_hit'
                done = True

        elif sl_hit:
            exit_price = sl_level
            exit_time = bt
            exit_reason = 'sl_hit'
            done = True

        elif tgt_hit:
            exit_price = target
            exit_time = bt
            exit_reason = 'target'
            done = True

        else:
            if trade_dir == 'long':
                mfe = max(mfe, bar['high'] - entry)
            else:
                mfe = max(mfe, entry - bar['low'])

    if not done:
        last = post.iloc[-1]
        exit_price = last['close']
        exit_time = str(last['time_et'])[:5]

    if trade_dir == 'long':
        pnl_raw = (exit_price - entry) * shares
    else:
        pnl_raw = (entry - exit_price) * shares

    mfe_r = mfe / one_r if one_r > 0 else 0
    pnl_dollar = pnl_raw - slippage * shares * 2
    pnl_r = pnl_dollar / risk

    return {
        'exit_price': exit_price, 'exit_time': exit_time,
        'exit_reason': exit_reason,
        'pnl_dollar': pnl_dollar, 'pnl_r': pnl_r,
        'peak_r': mfe_r, 'mfe_r': mfe_r, 'shares': shares,
        'zoom_used': zoom_used, 'zoom_changed': zoom_changed,
    }


# ============================================================
# BUILD ALL TRADES
# ============================================================
def build_trades(meta, slippage=SLIPPAGE_PER_SHARE, risk=RISK_PER_TRADE):
    """Simulate all qualifying trades with 3 exit strategies each."""
    base = meta[(meta['od_strength'] > 0.5) &
                (meta['od_direction'] == 'with_gap')].copy()

    results = []
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

        sl_dollar = SL_ADR_FRACTION * adr
        trade_dir = 'long' if gap_dir == 'up' else 'short'
        sl_level = entry - sl_dollar if trade_dir == 'long' else entry + sl_dollar
        one_r = sl_dollar

        fpath = RAW_1MIN_DIR / ticker / f"{date}.parquet"
        if not fpath.exists():
            skipped += 1
            continue
        bars = pd.read_parquet(fpath)
        rth = bars[bars['session'] == 'rth']
        if len(rth) < 10:
            skipped += 1
            continue

        if date >= IS_START and date <= IS_END:
            period = 'IS'
        elif date >= OOS_START:
            period = 'OOS'
        else:
            continue

        # Tier classification
        rvol_30 = row.get('rvol_open_30min', np.nan)
        is_tier2 = True  # all qualifying trades are Tier 2
        is_tier1 = pd.notna(rvol_30) and rvol_30 > 5

        base_rec = {
            'ticker': ticker, 'date': date,
            'gap_direction': gap_dir, 'trade_dir': trade_dir,
            'entry_price': entry, 'sl_level': sl_level,
            'sl_dollar': sl_dollar, 'adr': adr, 'period': period,
            'rvol_30': rvol_30,
            'od_strength': row['od_strength'],
            'is_tier2': is_tier2,
            'is_tier1': is_tier1,
        }

        # a) TRAIL_UNIFORM
        r_uni = simulate_trail(rth, entry, sl_level, trade_dir, adr, one_r,
                               ticker, date, row, trail_mode='uniform',
                               slippage=slippage, risk=risk)
        if r_uni is not None:
            rec = {**base_rec, **r_uni, 'trail_type': 'TRAIL_UNIFORM'}
            results.append(rec)

        # b) TRAIL_3TIER
        r_3t = simulate_trail(rth, entry, sl_level, trade_dir, adr, one_r,
                              ticker, date, row, trail_mode='3tier',
                              slippage=slippage, risk=risk)
        if r_3t is not None:
            rec = {**base_rec, **r_3t, 'trail_type': 'TRAIL_3TIER'}
            results.append(rec)

        # c) FIX_2R
        r_fix = simulate_fix_2r(rth, entry, sl_level, trade_dir, adr, one_r,
                                ticker, date, slippage=slippage, risk=risk)
        if r_fix is not None:
            rec = {**base_rec, **r_fix, 'trail_type': 'FIX_2R',
                   'trail_tier': np.nan, 'drift_30min': np.nan,
                   'reached_1r': np.nan}
            results.append(rec)

    print(f"  Skipped: {skipped}, Total records: {len(results)}", file=sys.stderr)
    df = pd.DataFrame(results)
    return df


# ============================================================
# AUFGABE 3: KPI COMPUTATION
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
    for x in [1, 2, 3, 5]:
        kpi[f'WR@{x}R'] = (pnl_r >= x).mean() * 100

    gw = pnl[w].sum() if w.any() else 0
    gl = abs(pnl[l].sum()) if l.any() else 0.001
    kpi['PF'] = gw / gl
    kpi['Exp $'] = pnl.mean()
    kpi['Exp R'] = pnl_r.mean()
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
    kpi['MaxConsW'] = mc_w

    eq = np.cumsum(pnl)
    pk = np.maximum.accumulate(eq)
    dd = eq - pk
    kpi['MaxDD$'] = dd.min()
    kpi['MaxDD%'] = (dd / np.where(pk > 0, pk, 1)).min() * 100 if pk.max() > 0 else 0

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
        kpi['AvgDD$'] = dd[in_dd].mean()
    else:
        kpi['MaxDDDur'] = 0
        kpi['AvgDD$'] = 0

    kpi['equity'] = eq.tolist()

    # Trail specifics
    if 'exit_reason' in df.columns and 'reached_1r' in df.columns:
        vc = df['exit_reason'].value_counts(normalize=True)
        kpi['%InitSL'] = vc.get('init_sl', 0) * 100
        kpi['%Trail'] = vc.get('trail', 0) * 100
        kpi['%Timeout'] = vc.get('timeout', 0) * 100
        kpi['%Target'] = vc.get('target', 0) * 100

        reached = df[df['reached_1r'] == True]
        if len(reached) > 0:
            kpi['Conv1R'] = (reached['pnl_dollar'] > 0).mean() * 100
        else:
            kpi['Conv1R'] = np.nan

        if 'trail' in vc.index:
            trails = df[df['exit_reason'] == 'trail']
            kpi['MedTrailR'] = trails['pnl_r'].median() if len(trails) > 0 else np.nan
        else:
            kpi['MedTrailR'] = np.nan

    return kpi


# ============================================================
# AUFGABE 4: MONTE CARLO
# ============================================================
def monte_carlo(pnl_array, n_iter=MC_ITERATIONS, seed=42):
    """Permutation-based Monte Carlo simulation."""
    rng = np.random.RandomState(seed)
    pnl = np.array(pnl_array)
    n = len(pnl)
    if n < 5:
        return None

    paths = np.zeros((n_iter, n))
    max_dds = np.zeros(n_iter)
    max_dd_pcts = np.zeros(n_iter)
    max_cls = np.zeros(n_iter, dtype=int)
    sharpe_like = np.zeros(n_iter)

    for i in range(n_iter):
        shuf = pnl[rng.permutation(n)]
        eq = np.cumsum(shuf)
        paths[i] = eq
        pk = np.maximum.accumulate(eq)
        dd = eq - pk
        max_dds[i] = dd.min()
        max_dd_pcts[i] = (dd / np.where(pk > 0, pk, 1)).min() * 100 if pk.max() > 0 else 0

        cl = mcl = 0
        for p in shuf:
            if p < -0.01:
                cl += 1
                mcl = max(mcl, cl)
            else:
                cl = 0
        max_cls[i] = mcl

        std = shuf.std()
        sharpe_like[i] = shuf.mean() / std if std > 0 else 0

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
        'sharpe_median': np.median(sharpe_like),
        **{f'p_ruin_{t}': (max_dds <= -t).mean() * 100
           for t in [5000, 10000, 15000, 20000]},
    }


# ============================================================
# AUFGABE 5: CHARTS
# ============================================================
def setup_dark_style():
    plt.style.use('dark_background')
    plt.rcParams.update({
        'figure.facecolor': C_BG,
        'axes.facecolor': C_BG,
        'savefig.facecolor': C_BG,
    })


def chart_equity(setup_name, kpis_dict, mc_dict, out_dir):
    """5a: Equity curves with MC paths (4 subplots: GapUp IS/OOS, GapDn IS/OOS)."""
    setup_dark_style()
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Equity Curves: {setup_name}', fontsize=14, color='white')

    panels = [('up', 'IS'), ('up', 'OOS'), ('down', 'IS'), ('down', 'OOS')]
    for idx, (gd, per) in enumerate(panels):
        ax = axes[idx // 2, idx % 2]
        key = (gd, per)
        kpi = kpis_dict.get(key)
        mc = mc_dict.get(key)

        if kpi is None or kpi['N'] == 0:
            gl = 'GapUp' if gd == 'up' else 'GapDn'
            ax.set_title(f"{gl} {per}: N=0", fontsize=10)
            continue

        eq = kpi['equity']
        n = len(eq)
        x = np.arange(1, n + 1)

        if mc is not None:
            for path in mc['paths']:
                ax.plot(range(1, n + 1), path, color='white', alpha=0.03, lw=0.5)
            ax.plot(x, mc['median_path'], color='white', lw=2, label='Median')
            ax.plot(x, mc['p5_path'], color=C_YELLOW, lw=1, ls='--', label='P5')
            ax.plot(x, mc['p95_path'], color=C_YELLOW, lw=1, ls='--', label='P95')

        ax.plot(x, eq, color=C_ACCENT2, lw=2, label='Actual')
        ax.axhline(0, color='grey', lw=0.5)

        gl = 'GapUp' if gd == 'up' else 'GapDn'
        low = ' [LOW N]' if n < 20 else ''
        ax.set_title(f"{gl} {per}: N={n}, Final=${eq[-1]:,.0f}{low}", fontsize=10)
        ax.set_xlabel('Trade #')
        ax.set_ylabel('Cumulative PnL ($)')
        ax.legend(fontsize=7, loc='upper left')

    plt.tight_layout()
    fname = f"equity_{setup_name.replace(' ', '_').lower()}.png"
    fig.savefig(out_dir / fname, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Chart: {fname}", file=sys.stderr)


def chart_drawdown(setup_name, mc_dict, out_dir):
    """5b: Drawdown distribution (OOS only)."""
    setup_dark_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Max Drawdown Distribution (OOS): {setup_name}', fontsize=14)

    for idx, gd in enumerate(['up', 'down']):
        ax = axes[idx]
        mc = mc_dict.get((gd, 'OOS'))
        gl = 'GapUp' if gd == 'up' else 'GapDn'

        if mc is None:
            ax.set_title(f"{gl}: N/A")
            continue

        dds = mc['max_dds']
        ax.hist(dds, bins=50, color=C_ACCENT, alpha=0.7, edgecolor='white', lw=0.3)
        ax.axvline(mc['dd_median'], color=C_YELLOW, lw=2,
                   label=f"Median: ${mc['dd_median']:,.0f}")
        ax.axvline(mc['dd_p5'], color=C_ACCENT2, lw=2, ls='--',
                   label=f"P5: ${mc['dd_p5']:,.0f}")
        ax.axvline(mc['worst_dd'], color='red', lw=1.5, ls=':',
                   label=f"Worst: ${mc['worst_dd']:,.0f}")

        ax.set_title(f"{gl} OOS")
        ax.set_xlabel('Max Drawdown ($)')
        ax.set_ylabel('Count')
        ax.legend(fontsize=8)
        ax.annotate(f"95% Konfidenz: DD < ${abs(mc['dd_p5']):,.0f}",
                    xy=(0.5, 0.95), xycoords='axes fraction',
                    fontsize=9, ha='center', color=C_YELLOW)

    plt.tight_layout()
    fname = f"drawdown_{setup_name.replace(' ', '_').lower()}.png"
    fig.savefig(out_dir / fname, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Chart: {fname}", file=sys.stderr)


def chart_r_distribution(all_df, out_dir):
    """5c: R-multiple distribution for all OOS trades."""
    setup_dark_style()

    configs = []
    for tier_name, tier_col, tt in [
        ('Tier2 UNIFORM', 'is_tier2', 'TRAIL_UNIFORM'),
        ('Tier2 3TIER', 'is_tier2', 'TRAIL_3TIER'),
        ('Tier2 FIX_2R', 'is_tier2', 'FIX_2R'),
        ('Tier1 3TIER', 'is_tier1', 'TRAIL_3TIER'),
    ]:
        sub = all_df[(all_df['period'] == 'OOS') & all_df[tier_col] &
                     (all_df['trail_type'] == tt)]
        configs.append((tier_name, sub))

    fig, axes = plt.subplots(1, len(configs), figsize=(5 * len(configs), 5))
    fig.suptitle('R-Multiple Distribution (OOS)', fontsize=14)

    edges = [-np.inf, -0.01, 0.01, 1, 2, 3, 5, 10, np.inf]
    labels = ['-1R', '0R', '0-1R', '1-2R', '2-3R', '3-5R', '5-10R', '10R+']
    colors = [C_ACCENT, '#666666', '#06d6a080', '#06d6a0a0', '#06d6a0c0',
              C_GREEN, '#00b4d8c0', C_ACCENT2]

    for idx, (name, df) in enumerate(configs):
        ax = axes[idx] if len(configs) > 1 else axes
        if len(df) == 0:
            ax.set_title(f"{name}: N=0")
            continue

        rv = df['pnl_r'].values
        counts = [((rv > edges[i]) & (rv <= edges[i + 1])).sum()
                  for i in range(len(labels))]

        ax.bar(range(len(labels)), counts, color=colors, edgecolor='white', lw=0.5)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=7, rotation=30)
        ax.set_title(f"{name} (N={len(df)})")
        ax.set_ylabel('Count')
        ax.annotate(f"Mean R: {rv.mean():+.2f}\nMedian R: {np.median(rv):+.2f}",
                    xy=(0.60, 0.85), xycoords='axes fraction', fontsize=9, color=C_YELLOW)

    plt.tight_layout()
    fig.savefig(out_dir / 'r_distribution.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Chart: r_distribution.png", file=sys.stderr)


def chart_monthly_pnl(all_df, out_dir):
    """5d: Monthly PnL bars (OOS only)."""
    setup_dark_style()

    configs = [
        ('Tier2 GapUp UNIFORM', 'is_tier2', 'up', 'TRAIL_UNIFORM'),
        ('Tier2 GapUp 3TIER', 'is_tier2', 'up', 'TRAIL_3TIER'),
        ('Tier2 GapUp FIX_2R', 'is_tier2', 'up', 'FIX_2R'),
        ('Tier1 GapUp 3TIER', 'is_tier1', 'up', 'TRAIL_3TIER'),
    ]

    for name, tier_col, gd, tt in configs:
        df = all_df[(all_df['period'] == 'OOS') & all_df[tier_col] &
                    (all_df['gap_direction'] == gd) &
                    (all_df['trail_type'] == tt)].copy()
        if len(df) == 0:
            continue

        fig, ax = plt.subplots(figsize=(14, 5))
        df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
        monthly = df.groupby('month')['pnl_dollar'].sum()

        clrs = [C_GREEN if v > 0 else C_ACCENT for v in monthly.values]
        x = range(len(monthly))
        ax.bar(x, monthly.values, color=clrs, edgecolor='white', lw=0.3)

        ax2 = ax.twinx()
        cum = monthly.cumsum()
        ax2.plot(x, cum.values, color=C_YELLOW, lw=2, label='Cumulative')
        ax2.set_ylabel('Cumulative PnL ($)', color=C_YELLOW)

        ax.set_xticks(list(x))
        ax.set_xticklabels([str(m) for m in monthly.index], rotation=45, fontsize=7)
        ax.set_title(f'Monthly PnL (OOS): {name}')
        ax.set_ylabel('Monthly PnL ($)')

        plt.tight_layout()
        fname = f"monthly_{name.replace(' ', '_').lower()}.png"
        fig.savefig(out_dir / fname, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Chart: {fname}", file=sys.stderr)


def chart_summary_table(all_kpis, out_dir):
    """5e: Summary KPI table as image — all 12 combinations."""
    setup_dark_style()

    col_keys = [
        ('T2 Up UNI', 'Tier2', 'up', 'TRAIL_UNIFORM'),
        ('T2 Up 3T', 'Tier2', 'up', 'TRAIL_3TIER'),
        ('T2 Up F2R', 'Tier2', 'up', 'FIX_2R'),
        ('T1 Up UNI', 'Tier1', 'up', 'TRAIL_UNIFORM'),
        ('T1 Up 3T', 'Tier1', 'up', 'TRAIL_3TIER'),
        ('T1 Up F2R', 'Tier1', 'up', 'FIX_2R'),
        ('T2 Dn UNI', 'Tier2', 'down', 'TRAIL_UNIFORM'),
        ('T2 Dn 3T', 'Tier2', 'down', 'TRAIL_3TIER'),
        ('T2 Dn F2R', 'Tier2', 'down', 'FIX_2R'),
        ('T1 Dn UNI', 'Tier1', 'down', 'TRAIL_UNIFORM'),
        ('T1 Dn 3T', 'Tier1', 'down', 'TRAIL_3TIER'),
        ('T1 Dn F2R', 'Tier1', 'down', 'FIX_2R'),
    ]

    row_defs = [
        ('N', 'N', 'd'), ('Win%', 'Win%', '.1f'), ('Loss%', 'Loss%', '.1f'),
        ('Total PnL', 'Total PnL', '$'), ('Mean PnL', 'Mean PnL', '$'),
        ('Mean R', 'Mean R', '+.2f'), ('Median R', 'Median R', '+.2f'),
        ('Max R', 'Max R', '+.1f'),
        ('PF', 'PF', '.2f'), ('Payoff', 'Payoff', '.2f'),
        ('WR@1R', 'WR@1R', '.1f'), ('WR@2R', 'WR@2R', '.1f'),
        ('WR@3R', 'WR@3R', '.1f'),
        ('MaxConsL', 'MaxConsL', 'd'), ('MaxDD$', 'MaxDD$', '$'),
        ('MaxDD%', 'MaxDD%', '.1f'),
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
        for _, tier, gd, tt in col_keys:
            kpi = all_kpis.get((tier, gd, tt, 'OOS'))
            if kpi is None or kpi['N'] == 0:
                row.append('-')
            else:
                row.append(fmt(kpi.get(rkey), rfmt))
        cell_text.append(row)

    fig, ax = plt.subplots(figsize=(22, 9))
    ax.axis('off')
    table = ax.table(cellText=cell_text,
                     rowLabels=[r[0] for r in row_defs],
                     colLabels=[ck[0] for ck in col_keys],
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(6)
    table.scale(1.1, 1.4)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor('white')
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor(C_ACCENT)
            cell.set_text_props(color='white', fontweight='bold', fontsize=6)
        elif c == -1:
            cell.set_facecolor('#2a2a4e')
            cell.set_text_props(color='white')
        else:
            cell.set_facecolor(C_BG)
            cell.set_text_props(color='white')

    ax.set_title('KPI Summary (OOS) — All 12 Combinations', fontsize=14,
                 color='white', pad=20)
    fig.savefig(out_dir / 'summary_table.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Chart: summary_table.png", file=sys.stderr)


def chart_trail_comparison(all_df, all_kpis, out_dir):
    """5f: TRAIL_3TIER vs TRAIL_UNIFORM comparison."""
    setup_dark_style()

    # Get Tier2 GapUp OOS trades for both strategies — reset index for alignment
    mask_base = ((all_df['period'] == 'OOS') &
                 (all_df['gap_direction'] == 'up') &
                 all_df['is_tier2'])
    df_uni = all_df[mask_base & (all_df['trail_type'] == 'TRAIL_UNIFORM')].reset_index(drop=True)
    df_3t = all_df[mask_base & (all_df['trail_type'] == 'TRAIL_3TIER')].reset_index(drop=True)

    if len(df_uni) == 0 or len(df_3t) == 0:
        return

    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # --- Subplot 1: Dual Equity Curves ---
    ax1 = fig.add_subplot(gs[0, :])

    eq_uni = np.cumsum(df_uni['pnl_dollar'].values)
    eq_3t = np.cumsum(df_3t['pnl_dollar'].values)
    x = np.arange(1, len(eq_uni) + 1)

    ax1.plot(x, eq_uni, color=C_ACCENT2, lw=2, label='TRAIL_UNIFORM')
    ax1.plot(x, eq_3t, color=C_GREEN, lw=2, label='TRAIL_3TIER')

    # Mark Stufe 3 trades (use positional mask)
    stufe3_mask = (df_3t['trail_tier'] == 3).values
    if stufe3_mask.any():
        s3_idx = np.where(stufe3_mask)[0]
        ax1.scatter(s3_idx + 1, eq_3t[s3_idx], color=C_YELLOW, s=30,
                    zorder=5, label='Stufe 3 Trades', alpha=0.8)

    ax1.axhline(0, color='grey', lw=0.5)
    ax1.set_title('Tier2 GapUp OOS: TRAIL_3TIER vs TRAIL_UNIFORM', fontsize=12)
    ax1.set_xlabel('Trade #')
    ax1.set_ylabel('Cumulative PnL ($)')

    delta_total = eq_3t[-1] - eq_uni[-1]
    # Stufe 3 delta — use positional indexing
    s3_3t = df_3t.loc[stufe3_mask, 'pnl_dollar'].values if stufe3_mask.any() else np.array([])
    s3_uni = df_uni.loc[stufe3_mask, 'pnl_dollar'].values if stufe3_mask.any() else np.array([])
    delta_s3 = s3_3t.sum() - s3_uni.sum() if len(s3_3t) > 0 else 0

    ax1.annotate(f"Delta gesamt: ${delta_total:+,.0f}\n"
                 f"Delta Stufe 3: ${delta_s3:+,.0f}",
                 xy=(0.02, 0.95), xycoords='axes fraction',
                 fontsize=10, color=C_YELLOW, va='top',
                 bbox=dict(boxstyle='round', facecolor='#2a2a4e', alpha=0.8))
    ax1.legend(fontsize=9, loc='lower right')

    # --- Subplot 2: Stufe-3 PnL distribution comparison ---
    ax2 = fig.add_subplot(gs[1, 0])
    if stufe3_mask.any():
        bins = np.linspace(
            min(s3_uni.min() if len(s3_uni) > 0 else -1000,
                s3_3t.min() if len(s3_3t) > 0 else -1000),
            max(s3_uni.max() if len(s3_uni) > 0 else 1000,
                s3_3t.max() if len(s3_3t) > 0 else 1000),
            25)
        ax2.hist(s3_uni, bins=bins, color=C_ACCENT2, alpha=0.6,
                 label=f'UNIFORM (Mean: ${s3_uni.mean():+,.0f})', edgecolor='white', lw=0.3)
        ax2.hist(s3_3t, bins=bins, color=C_GREEN, alpha=0.6,
                 label=f'3TIER (Mean: ${s3_3t.mean():+,.0f})', edgecolor='white', lw=0.3)
        ax2.set_title(f'Stufe-3 Trades PnL (N={stufe3_mask.sum()})', fontsize=10)
        ax2.legend(fontsize=8)
    else:
        ax2.set_title('Stufe-3: Keine Trades')
    ax2.set_xlabel('PnL ($)')
    ax2.set_ylabel('Count')

    # --- Subplot 3: Per-Stufe EV comparison ---
    ax3 = fig.add_subplot(gs[1, 1])

    stufe_evs = {'UNIFORM': {}, '3TIER': {}}
    for stufe in [1, 2, 3]:
        mask_s = (df_3t['trail_tier'] == stufe).values
        if mask_s.any():
            stufe_evs['3TIER'][stufe] = df_3t.loc[mask_s, 'pnl_dollar'].mean()
            stufe_evs['UNIFORM'][stufe] = df_uni.loc[mask_s, 'pnl_dollar'].mean()

    stufen = sorted(set(stufe_evs['3TIER'].keys()) | set(stufe_evs['UNIFORM'].keys()))
    if stufen:
        x_pos = np.arange(len(stufen))
        w = 0.35
        uni_vals = [stufe_evs['UNIFORM'].get(s, 0) for s in stufen]
        t3_vals = [stufe_evs['3TIER'].get(s, 0) for s in stufen]

        ax3.bar(x_pos - w / 2, uni_vals, w, color=C_ACCENT2, label='UNIFORM')
        ax3.bar(x_pos + w / 2, t3_vals, w, color=C_GREEN, label='3TIER')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([f'Stufe {s}' for s in stufen])
        ax3.axhline(0, color='grey', lw=0.5)
        ax3.set_ylabel('Mean PnL ($)')
        ax3.set_title('EV per Trail-Stufe', fontsize=10)
        ax3.legend(fontsize=8)

    fig.suptitle('TRAIL_3TIER vs TRAIL_UNIFORM — Tier2 GapUp OOS', fontsize=14,
                 color='white')
    fig.savefig(out_dir / 'trail_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Chart: trail_comparison.png", file=sys.stderr)


def chart_sensitivity(all_df, out_dir):
    """Sensitivity chart for slippage + risk scenarios."""
    setup_dark_style()

    df = all_df[(all_df['period'] == 'OOS') &
                (all_df['gap_direction'] == 'up') &
                all_df['is_tier2'] &
                (all_df['trail_type'] == 'TRAIL_3TIER')].copy()
    if len(df) == 0:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Sensitivity: Tier 2 GapUp TRAIL_3TIER OOS', fontsize=14)

    # Slippage
    slippages = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03]
    totals = []
    for slip in slippages:
        adj = df['pnl_dollar'].values - slip * df['shares'].values * 2
        totals.append(adj.sum())
    ax1.plot(slippages, totals, color=C_ACCENT2, lw=2, marker='o')
    ax1.axhline(0, color='grey', lw=0.5, ls='--')
    ax1.set_xlabel('Slippage per Share ($)')
    ax1.set_ylabel('Total PnL ($)')
    ax1.set_title('Slippage Sensitivity')

    # Find break-even slippage
    total_rt_shares = (df['shares'].values * 2).sum()
    be_slip = max(0, df['pnl_dollar'].values.sum() / total_rt_shares)
    ax1.axvline(be_slip, color=C_YELLOW, lw=1.5, ls='--',
                label=f'BE: ${be_slip:.4f}')
    ax1.legend(fontsize=8)

    # Risk
    risks = [500, 1000, 2000, 3000, 5000]
    dds = []
    pnl_base = df['pnl_dollar'].values
    for r in risks:
        scale = r / RISK_PER_TRADE
        adj = pnl_base * scale
        eq = np.cumsum(adj)
        dd = (eq - np.maximum.accumulate(eq)).min()
        dds.append(abs(dd) / r * 100)
    ax2.bar(range(len(risks)), dds, color=C_ACCENT, edgecolor='white', lw=0.5)
    ax2.set_xticks(range(len(risks)))
    ax2.set_xticklabels([f"${r}" for r in risks])
    ax2.set_xlabel('Risk per Trade ($)')
    ax2.set_ylabel('Max DD as % of 1R')
    ax2.set_title('Risk Scaling')

    plt.tight_layout()
    fig.savefig(out_dir / 'sensitivity_chart.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Chart: sensitivity_chart.png", file=sys.stderr)


# ============================================================
# TEXT OUTPUT
# ============================================================
def format_kpi_table(lines, kpis_is, kpis_oos, label):
    """Format side-by-side IS/OOS KPI table."""
    n_is = kpis_is['N']
    n_oos = kpis_oos['N']
    low_is = ' [LOW N]' if 0 < n_is < 20 else ''
    low_oos = ' [LOW N]' if 0 < n_oos < 20 else ''

    lines.append(f"\n  {label} -- IS: N={n_is}{low_is}, OOS: N={n_oos}{low_oos}")
    lines.append(f"    {'Metrik':<20} | {'IS':>12} | {'OOS':>12}")
    lines.append(f"    {'-' * 48}")

    def row(name, key, fmt='$'):
        v_is = kpis_is.get(key, np.nan)
        v_oos = kpis_oos.get(key, np.nan)

        def f(v):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return '-'
            if fmt == '$':
                return f"${v:>10,.0f}"
            if fmt == 'r':
                return f"{v:>+10.3f}"
            if fmt == '%':
                return f"{v:>10.1f}%"
            if fmt == 'd':
                return f"{int(v):>10d}"
            return f"{v:>10.2f}"
        lines.append(f"    {name:<20} | {f(v_is):>12} | {f(v_oos):>12}")

    row('Win Rate', 'Win%', '%')
    row('Loss Rate', 'Loss%', '%')
    row('BE Rate', 'BE%', '%')
    row('Total PnL', 'Total PnL', '$')
    row('Mean PnL', 'Mean PnL', '$')
    row('Median PnL', 'Median PnL', '$')
    row('StdDev PnL', 'StdDev', '$')
    row('Skewness', 'Skew', 'f')
    row('Best Trade', 'Best', '$')
    row('Worst Trade', 'Worst', '$')
    row('P10', 'P10', '$')
    row('P25', 'P25', '$')
    row('P75', 'P75', '$')
    row('P90', 'P90', '$')
    lines.append('')
    row('Mean R', 'Mean R', 'r')
    row('Median R', 'Median R', 'r')
    row('Max R', 'Max R', 'r')
    row('WR@1R', 'WR@1R', '%')
    row('WR@2R', 'WR@2R', '%')
    row('WR@3R', 'WR@3R', '%')
    row('WR@5R', 'WR@5R', '%')
    lines.append('')
    row('Profit Factor', 'PF', 'f')
    row('Expectancy $', 'Exp $', '$')
    row('Expectancy R', 'Exp R', 'r')
    row('Payoff Ratio', 'Payoff', 'f')
    row('Max Cons Losses', 'MaxConsL', 'd')
    row('Max Cons Wins', 'MaxConsW', 'd')
    lines.append('')
    row('Max DD $', 'MaxDD$', '$')
    row('Max DD %', 'MaxDD%', '%')
    row('Max DD Duration', 'MaxDDDur', 'd')
    row('Avg DD $', 'AvgDD$', '$')

    # Trail extras
    if '%InitSL' in kpis_is or '%InitSL' in kpis_oos:
        lines.append('')
        row('% InitSL', '%InitSL', '%')
        row('% Trail', '%Trail', '%')
        row('% Timeout', '%Timeout', '%')
        row('% Target', '%Target', '%')
        row('Conversion 1R', 'Conv1R', '%')
        row('Med Trail R', 'MedTrailR', 'r')


def format_mc_results(lines, mc, label):
    """Format Monte Carlo results."""
    if mc is None:
        lines.append(f"\n  {label}: Monte Carlo nicht moeglich (N < 5)")
        return

    lines.append(f"\n  {label} -- Monte Carlo ({MC_ITERATIONS} Permutationen):")
    lines.append(f"    Finale Equity:  Median ${mc['fe_median']:>10,.0f}  "
                 f"[P5: ${mc['fe_p5']:>10,.0f}  P25: ${mc['fe_p25']:>10,.0f}  "
                 f"P75: ${mc['fe_p75']:>10,.0f}  P95: ${mc['fe_p95']:>10,.0f}]")
    lines.append(f"    Max Drawdown:   Median ${mc['dd_median']:>10,.0f}  "
                 f"[P5: ${mc['dd_p5']:>10,.0f}  P95: ${mc['dd_p95']:>10,.0f}]")
    lines.append(f"    Worst DD:       ${mc['worst_dd']:>10,.0f}")
    lines.append(f"    Max Cons Loss:  Median {mc['cl_median']:.0f}  "
                 f"[P95: {mc['cl_p95']:.0f}]")
    lines.append(f"    Sharpe-like:    Median {mc['sharpe_median']:.3f}")

    lines.append(f"    Probability of Ruin:")
    for t in [5000, 10000, 15000, 20000]:
        lines.append(f"      DD >= ${t:>6,}: {mc[f'p_ruin_{t}']:>5.1f}%")


def format_trail_tier_analysis(lines, all_df):
    """Analyse TRAIL_3TIER vs TRAIL_UNIFORM per Stufe."""
    lines.append(f"\n\n{'=' * 70}")
    lines.append("=== TRAIL_3TIER vs TRAIL_UNIFORM VERGLEICH ===")
    lines.append(f"{'=' * 70}")

    for gd_label, gd in [('GapUp', 'up'), ('GapDn', 'down')]:
        for per in ['IS', 'OOS']:
            lines.append(f"\n--- {gd_label} {per} ---")
            mask_base = ((all_df['period'] == per) &
                         (all_df['gap_direction'] == gd) &
                         all_df['is_tier2'])

            df_3t = all_df[mask_base & (all_df['trail_type'] == 'TRAIL_3TIER')].reset_index(drop=True)
            df_uni = all_df[mask_base & (all_df['trail_type'] == 'TRAIL_UNIFORM')].reset_index(drop=True)

            if len(df_3t) == 0 or len(df_uni) == 0:
                lines.append("  Keine Trades.")
                continue

            # Stufen-Verteilung
            n_total = len(df_3t)
            n_before_10 = int(df_3t['trail_tier'].isna().sum())

            for stufe in [1, 2, 3]:
                n_s = int((df_3t['trail_tier'] == stufe).sum())
                pct = n_s / n_total * 100
                lines.append(f"  Stufe {stufe}: {n_s} ({pct:.1f}%)")
            lines.append(f"  Vor 10:00 raus: {n_before_10} ({n_before_10/n_total*100:.1f}%)")

            # EV Delta
            ev_3t = df_3t['pnl_dollar'].mean()
            ev_uni = df_uni['pnl_dollar'].mean()
            lines.append(f"\n  Gesamt-EV:")
            lines.append(f"    TRAIL_UNIFORM:  ${ev_uni:>+10,.0f} (N={len(df_uni)})")
            lines.append(f"    TRAIL_3TIER:    ${ev_3t:>+10,.0f} (N={len(df_3t)})")
            lines.append(f"    Delta:          ${ev_3t - ev_uni:>+10,.0f}")

            # Per-Stufe Delta (use positional masks)
            for stufe in [1, 2, 3]:
                mask_s = (df_3t['trail_tier'] == stufe).values
                if mask_s.any():
                    ev_s_3t = df_3t.loc[mask_s, 'pnl_dollar'].mean()
                    ev_s_uni = df_uni.loc[mask_s, 'pnl_dollar'].mean()
                    n_s = int(mask_s.sum())
                    lines.append(f"\n  Stufe {stufe} (N={n_s}):")
                    lines.append(f"    UNIFORM EV: ${ev_s_uni:>+10,.0f}")
                    lines.append(f"    3TIER EV:   ${ev_s_3t:>+10,.0f}")
                    lines.append(f"    Delta:      ${ev_s_3t - ev_s_uni:>+10,.0f}")

            # Kernfrage
            mask_s3 = (df_3t['trail_tier'] == 3).values
            if mask_s3.any():
                s3_3t_mean = df_3t.loc[mask_s3, 'pnl_r'].mean()
                s3_uni_mean = df_uni.loc[mask_s3, 'pnl_r'].mean()
                lines.append(f"\n  ** Kernfrage: Lohnt sich enger Trail bei Stufe 3? **")
                lines.append(f"     UNIFORM Mean R: {s3_uni_mean:+.3f}")
                lines.append(f"     3TIER   Mean R: {s3_3t_mean:+.3f}")
                lines.append(f"     Delta R:        {s3_3t_mean - s3_uni_mean:+.3f}")
                if s3_3t_mean > s3_uni_mean:
                    lines.append(f"     -> JA, Stufe 3 profitiert vom engeren Trail.")
                else:
                    lines.append(f"     -> NEIN, engerer Trail bringt keinen Mehrwert.")


def sensitivity_text(all_df):
    """Sensitivity analysis text output."""
    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("SENSITIVITAETS-ANALYSE")
    lines.append("=" * 70)
    lines.append("Basis: Tier 2 GapUp TRAIL_3TIER, OOS\n")

    df = all_df[(all_df['period'] == 'OOS') &
                (all_df['gap_direction'] == 'up') &
                all_df['is_tier2'] &
                (all_df['trail_type'] == 'TRAIL_3TIER')].copy()

    if len(df) == 0:
        lines.append("Keine Trades.")
        return lines

    pnl_base = df['pnl_dollar'].values
    shares = df['shares'].values

    # 6a: Slippage
    lines.append("--- 6a: Slippage-Szenarien ---")
    lines.append(f"  {'Slippage':>10} | {'Total PnL':>12} | {'Mean PnL':>10} | "
                 f"{'Mean R':>8} | {'Win%':>6} | {'PF':>6} | {'MaxDD$':>10}")
    lines.append(f"  {'-' * 78}")

    for slip in [0.0, 0.005, 0.01, 0.02]:
        adj = pnl_base - slip * shares * 2
        adj_r = adj / RISK_PER_TRADE
        w = adj > 0.01
        l = adj < -0.01
        gw = adj[w].sum() if w.any() else 0
        gl = abs(adj[l].sum()) if l.any() else 0.001
        eq = np.cumsum(adj)
        dd = (eq - np.maximum.accumulate(eq)).min()
        lines.append(f"  ${slip:.3f}    | ${adj.sum():>10,.0f} | ${adj.mean():>8,.0f} | "
                     f"{adj_r.mean():>+6.3f} | {w.mean() * 100:>5.1f} | "
                     f"{gw / gl:>5.2f} | ${dd:>8,.0f}")

    # Break-even slippage
    total_roundtrip_shares = (shares * 2).sum()
    be_slip = max(0, pnl_base.sum() / total_roundtrip_shares)
    lines.append(f"\n  Break-Even Slippage: ${be_slip:.4f} pro Share")

    # 6b: Risk
    lines.append(f"\n--- 6b: Risk-per-Trade Szenarien ---")
    lines.append(f"  {'Risk':>8} | {'Total PnL':>12} | {'Mean PnL':>10} | "
                 f"{'MaxDD$':>10} | {'MaxDD/1R':>10}")
    lines.append(f"  {'-' * 60}")

    for r in [500, 1000, 2000, 5000]:
        scale = r / RISK_PER_TRADE
        adj = pnl_base * scale
        eq = np.cumsum(adj)
        dd = (eq - np.maximum.accumulate(eq)).min()
        lines.append(f"  ${r:>6} | ${adj.sum():>10,.0f} | ${adj.mean():>8,.0f} | "
                     f"${dd:>8,.0f} | {abs(dd) / r * 100:>8.1f}%")

    return lines


# ============================================================
# MAIN
# ============================================================
def main():
    t_start = time.time()
    np.random.seed(42)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- Load metadata ---
    print("Loading metadata_v8_5...", file=sys.stderr)
    meta = pd.read_parquet('data/metadata/metadata_v8_5.parquet')
    meta = meta[meta['date'] >= IS_START].copy()
    print(f"  {len(meta)} rows", file=sys.stderr)

    # --- Build all trades ---
    print("\nAufgabe 1+2: Simulating trades (3 strategies each)...", file=sys.stderr)
    all_df = build_trades(meta)

    if len(all_df) == 0:
        print("ERROR: No trades generated!", file=sys.stderr)
        return

    # Save trades
    all_df.to_parquet(RESULTS_DIR / 'all_trades_v2.parquet', index=False)
    print(f"  Saved {len(all_df)} trade records.", file=sys.stderr)

    # Zoom stats
    for tt in ['TRAIL_UNIFORM', 'TRAIL_3TIER', 'FIX_2R']:
        sub = all_df[all_df['trail_type'] == tt]
        nz = sub['zoom_used'].sum()
        nc = sub['zoom_changed'].sum()
        print(f"  {tt}: {nz} zoom used, {nc} changed result", file=sys.stderr)

    # --- Trade counts summary ---
    print("\nTrade counts:", file=sys.stderr)
    for tt in ['TRAIL_UNIFORM', 'TRAIL_3TIER', 'FIX_2R']:
        for tier_name, tier_col in [('Tier2', 'is_tier2'), ('Tier1', 'is_tier1')]:
            for gd in ['up', 'down']:
                for per in ['IS', 'OOS']:
                    n = len(all_df[(all_df['trail_type'] == tt) &
                                   all_df[tier_col] &
                                   (all_df['gap_direction'] == gd) &
                                   (all_df['period'] == per)])
                    if n > 0:
                        gl = 'GapUp' if gd == 'up' else 'GapDn'
                        print(f"  {tier_name} {gl} {tt} {per}: {n}", file=sys.stderr)

    # --- Aufgabe 3: KPIs ---
    print("\nAufgabe 3: Computing KPIs...", file=sys.stderr)
    all_kpis = {}

    for tt in ['TRAIL_UNIFORM', 'TRAIL_3TIER', 'FIX_2R']:
        for tier_name, tier_col in [('Tier2', 'is_tier2'), ('Tier1', 'is_tier1')]:
            for gd in ['up', 'down']:
                for per in ['IS', 'OOS']:
                    sub = all_df[(all_df['trail_type'] == tt) &
                                 all_df[tier_col] &
                                 (all_df['gap_direction'] == gd) &
                                 (all_df['period'] == per)].copy()
                    gl = 'GapUp' if gd == 'up' else 'GapDn'
                    label = f"{tier_name} {gl} {tt} {per}"
                    kpi = compute_kpis(sub, label)
                    all_kpis[(tier_name, gd, tt, per)] = kpi

    print("  Done.", file=sys.stderr)

    # --- Aufgabe 4: Monte Carlo ---
    print("\nAufgabe 4: Monte Carlo simulations...", file=sys.stderr)
    all_mc = {}

    mc_combos = [
        ('Tier2', 'up', 'TRAIL_UNIFORM'),
        ('Tier2', 'up', 'TRAIL_3TIER'),
        ('Tier2', 'up', 'FIX_2R'),
        ('Tier1', 'up', 'TRAIL_3TIER'),
        ('Tier2', 'down', 'TRAIL_3TIER'),
        ('Tier1', 'down', 'TRAIL_3TIER'),
    ]

    for tier_name, gd, tt in tqdm(mc_combos, desc="Monte Carlo", file=sys.stderr):
        tier_col = 'is_tier2' if tier_name == 'Tier2' else 'is_tier1'
        for per in ['IS', 'OOS']:
            kpi = all_kpis.get((tier_name, gd, tt, per))
            if kpi is not None and kpi['N'] >= 5:
                sub = all_df[(all_df['trail_type'] == tt) &
                             all_df[tier_col] &
                             (all_df['gap_direction'] == gd) &
                             (all_df['period'] == per)]
                mc = monte_carlo(sub['pnl_dollar'].values)
                all_mc[(tier_name, gd, tt, per)] = mc
            else:
                all_mc[(tier_name, gd, tt, per)] = None

    print("  Done.", file=sys.stderr)

    # --- Aufgabe 5: Charts ---
    print("\nAufgabe 5: Charts...", file=sys.stderr)

    # 5a: Equity curves
    for tier_name, gd, tt in mc_combos:
        sname = f"{tier_name} {('GapUp' if gd == 'up' else 'GapDn')} {tt}"
        kpis_for = {(gd, per): all_kpis[(tier_name, gd, tt, per)]
                    for per in ['IS', 'OOS']}
        mc_for = {(gd, per): all_mc.get((tier_name, gd, tt, per))
                  for per in ['IS', 'OOS']}
        # For equity chart we need both gap directions on the same setup
        # But MC combos are per gap direction. Build per MC combo.
        setup_dark_style()
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'Equity: {sname}', fontsize=14, color='white')

        for idx, per in enumerate(['IS', 'OOS']):
            ax = axes[idx]
            kpi = all_kpis.get((tier_name, gd, tt, per))
            mc = all_mc.get((tier_name, gd, tt, per))

            if kpi is None or kpi['N'] == 0:
                ax.set_title(f"{per}: N=0")
                continue

            eq = kpi['equity']
            n = len(eq)
            x = np.arange(1, n + 1)

            if mc is not None:
                for path in mc['paths']:
                    ax.plot(range(1, n + 1), path, color='white', alpha=0.03, lw=0.5)
                ax.plot(x, mc['median_path'], color='white', lw=2, label='Median')
                ax.plot(x, mc['p5_path'], color=C_YELLOW, lw=1, ls='--', label='P5')
                ax.plot(x, mc['p95_path'], color=C_YELLOW, lw=1, ls='--', label='P95')

            ax.plot(x, eq, color=C_ACCENT2, lw=2, label='Actual')
            ax.axhline(0, color='grey', lw=0.5)

            low = ' [LOW N]' if n < 20 else ''
            ax.set_title(f"{per}: N={n}, Final=${eq[-1]:,.0f}{low}", fontsize=10)
            ax.set_xlabel('Trade #')
            ax.set_ylabel('Cumulative PnL ($)')
            ax.legend(fontsize=7, loc='upper left')

        plt.tight_layout()
        fname = f"equity_{sname.replace(' ', '_').lower()}.png"
        fig.savefig(RESULTS_DIR / fname, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Chart: {fname}", file=sys.stderr)

    # 5b: Drawdown distributions
    for tier_name, gd, tt in mc_combos:
        sname = f"{tier_name} {('GapUp' if gd == 'up' else 'GapDn')} {tt}"
        mc = all_mc.get((tier_name, gd, tt, 'OOS'))
        if mc is None:
            continue

        setup_dark_style()
        fig, ax = plt.subplots(figsize=(10, 5))
        dds = mc['max_dds']
        ax.hist(dds, bins=50, color=C_ACCENT, alpha=0.7, edgecolor='white', lw=0.3)
        ax.axvline(mc['dd_median'], color=C_YELLOW, lw=2,
                   label=f"Median: ${mc['dd_median']:,.0f}")
        ax.axvline(mc['dd_p5'], color=C_ACCENT2, lw=2, ls='--',
                   label=f"P5: ${mc['dd_p5']:,.0f}")
        ax.axvline(mc['worst_dd'], color='red', lw=1.5, ls=':',
                   label=f"Worst: ${mc['worst_dd']:,.0f}")
        ax.set_title(f'Max DD Distribution OOS: {sname}')
        ax.set_xlabel('Max Drawdown ($)')
        ax.set_ylabel('Count')
        ax.legend(fontsize=8)
        ax.annotate(f"95% Konfidenz: DD < ${abs(mc['dd_p5']):,.0f}",
                    xy=(0.5, 0.95), xycoords='axes fraction',
                    fontsize=9, ha='center', color=C_YELLOW)

        fname = f"drawdown_{sname.replace(' ', '_').lower()}.png"
        fig.savefig(RESULTS_DIR / fname, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Chart: {fname}", file=sys.stderr)

    # 5c: R-distribution
    chart_r_distribution(all_df, RESULTS_DIR)

    # 5d: Monthly PnL
    chart_monthly_pnl(all_df, RESULTS_DIR)

    # 5e: Summary table
    chart_summary_table(all_kpis, RESULTS_DIR)

    # 5f: Trail comparison
    chart_trail_comparison(all_df, all_kpis, RESULTS_DIR)

    # Sensitivity chart
    chart_sensitivity(all_df, RESULTS_DIR)

    # --- Text Output ---
    print("\nWriting text output...", file=sys.stderr)
    lines = []
    lines.append("=" * 70)
    lines.append("BACKTEST + MONTE CARLO v2.0 ERGEBNISSE")
    lines.append("=" * 70)
    lines.append(f"Risk: ${RISK_PER_TRADE:.0f}/Trade, SL: {SL_ADR_FRACTION} ADR, "
                 f"Slippage: ${SLIPPAGE_PER_SHARE}")
    lines.append(f"MC: {MC_ITERATIONS} Permutationen, Seed=42")
    lines.append(f"IS: {IS_START} bis {IS_END}, OOS: {OOS_START} bis {OOS_END}")
    lines.append(f"Exit-Strategien: TRAIL_UNIFORM (peak-0.5R), "
                 f"TRAIL_3TIER (Stufen), FIX_2R (+2R Target)")
    lines.append(f"Trail-States: INIT -> BREAKEVEN -> TRAILING_NORMAL -> TRAILING_TIGHT")
    lines.append("")

    # Zoom stats
    for tt in ['TRAIL_UNIFORM', 'TRAIL_3TIER', 'FIX_2R']:
        sub = all_df[all_df['trail_type'] == tt]
        nz = int(sub['zoom_used'].sum())
        nc = int(sub['zoom_changed'].sum())
        lines.append(f"{tt}: {nz}/{len(sub)} Trades used 1-sec zoom, "
                     f"{nc} changed result")

    # KPIs for all 12 combinations
    for tt in ['TRAIL_UNIFORM', 'TRAIL_3TIER', 'FIX_2R']:
        for tier_name in ['Tier2', 'Tier1']:
            lines.append(f"\n\n{'=' * 70}")
            lines.append(f"=== {tier_name} {tt} ===")
            lines.append(f"{'=' * 70}")

            for gd in ['up', 'down']:
                gl = 'GapUp' if gd == 'up' else 'GapDn'
                k_is = all_kpis[(tier_name, gd, tt, 'IS')]
                k_oos = all_kpis[(tier_name, gd, tt, 'OOS')]
                format_kpi_table(lines, k_is, k_oos, f"{tier_name} {gl} {tt}")

                # MC for OOS
                mc_key = (tier_name, gd, tt, 'OOS')
                if mc_key in all_mc:
                    format_mc_results(lines, all_mc[mc_key],
                                      f"{tier_name} {gl} {tt} OOS")

    # Trail tier analysis
    format_trail_tier_analysis(lines, all_df)

    # Sensitivity
    sens_lines = sensitivity_text(all_df)
    lines.extend(sens_lines)

    elapsed = time.time() - t_start
    lines.append(f"\n\nLaufzeit: {elapsed:.1f}s")

    output = "\n".join(lines)
    with open(RESULTS_DIR / 'kpis_summary.txt', 'w', encoding='utf-8') as f:
        f.write(output)

    # Sensitivity separate file
    sens_output = "\n".join(sens_lines)
    with open(RESULTS_DIR / 'sensitivity.txt', 'w', encoding='utf-8') as f:
        f.write(sens_output)

    print(output)
    print(f"\nSaved to {RESULTS_DIR}/", file=sys.stderr)
    print(f"Total time: {elapsed:.1f}s", file=sys.stderr)


if __name__ == '__main__':
    main()
