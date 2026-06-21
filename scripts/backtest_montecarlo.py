"""
Backtest + Monte Carlo Simulation
Aufgaben 1-6: Trade-Simulation, Setup-Klassifikation, KPIs,
Monte Carlo, Charts, Sensitivitaet.

Output: results/backtest/
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

# ============================================================
# CONSTANTS
# ============================================================
RAW_1MIN_DIR = Path('data/raw_1min')
RAW_1SEC_DIR = Path('data/raw_1sec')
RESULTS_DIR = Path('results/backtest')

RISK_PER_TRADE = 1000.0
SL_ADR_FRACTION = 0.25
SLIPPAGE_PER_SHARE = 0.00
MC_ITERATIONS = 1000

C_ACCENT = '#e94560'
C_ACCENT2 = '#00b4d8'
C_GREEN = '#06d6a0'
C_YELLOW = '#ffd166'
C_BG = '#1a1a2e'

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
# AUFGABE 1a: TRAIL_D SIMULATION
# ============================================================
def simulate_trail_d(bars_rth, entry, sl_level, trade_dir, adr,
                     ticker, date, slippage=0.0, risk=1000.0):
    """TRAIL_D: INIT -> TRAILING at +1R, trail = peak - 0.5R."""
    sl_dist = abs(entry - sl_level)
    if sl_dist <= 0 or pd.isna(sl_dist) or adr <= 0:
        return None

    shares = risk / sl_dist
    one_r = sl_dist
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

    sec_bars = load_1sec(ticker, date)
    post = bars_rth[(bars_rth['time_et'] >= '09:36') & (bars_rth['time_et'] <= '15:55')]
    if len(post) == 0:
        return None

    for _, bar in post.iterrows():
        if done:
            break
        bt = str(bar['time_et'])[:5]

        if trade_dir == 'long':
            stop_possible = bar['low'] <= current_sl
            fav_possible = bar['high'] > peak_fav
        else:
            stop_possible = bar['high'] >= current_sl
            fav_possible = bar['low'] < peak_fav

        ambig = stop_possible and fav_possible

        if ambig:
            resolved = False
            can_zoom = (sec_bars is not None and bt >= '09:30' and bt < '09:45')
            if can_zoom:
                mins = sec_bars[sec_bars['time_et'].str[:5] == bt]
                if len(mins) > 0:
                    zoom_used = True
                    state_before = state
                    for _, s in mins.iterrows():
                        # Check stop first (conservative within 1-sec)
                        if trade_dir == 'long':
                            if s['low'] <= current_sl:
                                exit_price = current_sl
                                exit_time = s['time_et']
                                exit_reason = 'init_sl' if state == 'INIT' else 'trail'
                                resolved = done = True
                                break
                            if s['high'] > peak_fav:
                                peak_fav = s['high']
                            mfe = max(mfe, s['high'] - entry)
                        else:
                            if s['high'] >= current_sl:
                                exit_price = current_sl
                                exit_time = s['time_et']
                                exit_reason = 'init_sl' if state == 'INIT' else 'trail'
                                resolved = done = True
                                break
                            if s['low'] < peak_fav:
                                peak_fav = s['low']
                            mfe = max(mfe, entry - s['low'])

                        # State transition
                        if state == 'INIT':
                            if (trade_dir == 'long' and peak_fav >= one_r_level) or \
                               (trade_dir == 'short' and peak_fav <= one_r_level):
                                state = 'TRAILING'
                                reached_1r = True
                        if state == 'TRAILING':
                            if trade_dir == 'long':
                                current_sl = peak_fav - 0.5 * one_r
                            else:
                                current_sl = peak_fav + 0.5 * one_r

                    if not resolved:
                        # Re-check stop with updated state
                        if trade_dir == 'long':
                            still_hit = bar['low'] <= current_sl
                        else:
                            still_hit = bar['high'] >= current_sl
                        if still_hit:
                            exit_price = current_sl
                            exit_time = bt
                            exit_reason = 'init_sl' if state == 'INIT' else 'trail'
                            resolved = done = True

                    if state_before != state:
                        zoom_changed = True

            if not resolved:
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
            # No stop — update state
            if trade_dir == 'long':
                peak_fav = max(peak_fav, bar['high'])
                mfe = max(mfe, bar['high'] - entry)
            else:
                peak_fav = min(peak_fav, bar['low'])
                mfe = max(mfe, entry - bar['low'])

            if state == 'INIT':
                if (trade_dir == 'long' and peak_fav >= one_r_level) or \
                   (trade_dir == 'short' and peak_fav <= one_r_level):
                    state = 'TRAILING'
                    reached_1r = True
            if state == 'TRAILING':
                if trade_dir == 'long':
                    current_sl = peak_fav - 0.5 * one_r
                else:
                    current_sl = peak_fav + 0.5 * one_r

    # Timeout
    if not done:
        last = post.iloc[-1]
        exit_price = last['close']
        exit_time = str(last['time_et'])[:5]

    if trade_dir == 'long':
        pnl_raw = (exit_price - entry) * shares
        mfe_r = mfe / one_r
        peak_r = (peak_fav - entry) / one_r
    else:
        pnl_raw = (entry - exit_price) * shares
        mfe_r = mfe / one_r
        peak_r = (entry - peak_fav) / one_r

    pnl_dollar = pnl_raw - slippage * shares * 2
    pnl_r = pnl_dollar / risk

    return {
        'exit_price': exit_price, 'exit_time': exit_time,
        'exit_reason': exit_reason,
        'pnl_dollar': pnl_dollar, 'pnl_r': pnl_r,
        'peak_r': peak_r, 'mfe_r': mfe_r, 'shares': shares,
        'reached_1r': reached_1r,
        'zoom_used': zoom_used, 'zoom_changed': zoom_changed,
    }


# ============================================================
# AUFGABE 1b: FIX_2R SIMULATION
# ============================================================
def simulate_fix_2r(bars_rth, entry, sl_level, trade_dir, adr,
                    ticker, date, slippage=0.0, risk=1000.0):
    """FIX 2R: target at +/-2*SL_dist, exit on target/SL/timeout."""
    sl_dist = abs(entry - sl_level)
    if sl_dist <= 0 or pd.isna(sl_dist) or adr <= 0:
        return None

    shares = risk / sl_dist
    one_r = sl_dist
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

    mfe_r = mfe / one_r
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
# BUILD TRADES
# ============================================================
def build_trades(meta, slippage=SLIPPAGE_PER_SHARE, risk=RISK_PER_TRADE):
    """Simulate all qualifying trades."""
    base = meta[(meta['od_strength'] > 0.5) &
                (meta['od_direction'] == 'with_gap')].copy()

    trail_list = []
    fix2r_list = []
    skipped = 0

    for _, row in tqdm(base.iterrows(), total=len(base),
                       desc="Simulating", file=sys.stderr):
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

        fpath = RAW_1MIN_DIR / ticker / f"{date}.parquet"
        if not fpath.exists():
            skipped += 1
            continue
        bars = pd.read_parquet(fpath)
        rth = bars[bars['session'] == 'rth']
        if len(rth) < 10:
            skipped += 1
            continue

        dt = date
        if dt >= '2021-02-21' and dt <= '2023-12-31':
            period = 'IS'
        elif dt >= '2024-01-01':
            period = 'OOS'
        else:
            continue

        base_rec = {
            'ticker': ticker, 'date': date,
            'gap_direction': gap_dir, 'trade_dir': trade_dir,
            'entry_price': entry, 'sl_level': sl_level,
            'sl_dollar': sl_dollar, 'adr': adr, 'period': period,
            'rvol_30': row.get('rvol_open_30min', np.nan),
            'od_strength': row['od_strength'],
        }

        c1000 = row.get('close_1000', np.nan)
        if pd.notna(c1000) and pd.notna(entry) and adr > 0:
            drift = (c1000 - entry) / adr
            base_rec['drift_30min'] = drift if gap_dir == 'up' else -drift
        else:
            base_rec['drift_30min'] = np.nan

        # TRAIL_D
        r_trail = simulate_trail_d(rth, entry, sl_level, trade_dir, adr,
                                   ticker, date, slippage, risk)
        if r_trail is not None:
            rec = {**base_rec, **r_trail, 'strategy': 'TRAIL_D'}
            rec['is_tier1'] = pd.notna(base_rec['rvol_30']) and base_rec['rvol_30'] > 5
            rec['is_tier2'] = True
            trail_list.append(rec)

        # FIX_2R
        r_fix = simulate_fix_2r(rth, entry, sl_level, trade_dir, adr,
                                ticker, date, slippage, risk)
        if r_fix is not None:
            rec = {**base_rec, **r_fix, 'strategy': 'FIX_2R'}
            rec['is_tier3'] = True
            fix2r_list.append(rec)

    print(f"  Skipped: {skipped}, TRAIL_D: {len(trail_list)}, "
          f"FIX_2R: {len(fix2r_list)}", file=sys.stderr)
    return pd.DataFrame(trail_list), pd.DataFrame(fix2r_list)


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

    # TRAIL_D specifics
    if 'exit_reason' in df.columns and 'reached_1r' in df.columns:
        vc = df['exit_reason'].value_counts(normalize=True)
        kpi['%InitSL'] = vc.get('init_sl', 0) * 100
        kpi['%Trail'] = vc.get('trail', 0) * 100
        kpi['%Timeout'] = vc.get('timeout', 0) * 100

        reached = df[df['reached_1r'] == True]
        if len(reached) > 0:
            kpi['Conv1R'] = (reached['pnl_dollar'] > 0).mean() * 100
        else:
            kpi['Conv1R'] = np.nan

        trails = df[df['exit_reason'] == 'trail']
        kpi['MedTrailR'] = trails['pnl_r'].median() if len(trails) > 0 else np.nan

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
    """5a: Equity curves with MC paths (4 subplots)."""
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


def chart_r_distribution(trail_df, fix2r_df, out_dir):
    """5c: R-multiple distribution for all OOS trades."""
    setup_dark_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('R-Multiple Distribution (OOS)', fontsize=14)

    configs = [
        ('Tier 2 TRAIL_D', trail_df[(trail_df['period'] == 'OOS') & trail_df['is_tier2']]),
        ('Tier 1 TRAIL_D', trail_df[(trail_df['period'] == 'OOS') & trail_df['is_tier1']]),
        ('Tier 3 FIX_2R', fix2r_df[(fix2r_df['period'] == 'OOS') & fix2r_df['is_tier3']]),
    ]

    edges = [-np.inf, -0.01, 0.01, 1, 2, 3, 5, 10, np.inf]
    labels = ['-1R', '0R', '0-1R', '1-2R', '2-3R', '3-5R', '5-10R', '10R+']
    colors = [C_ACCENT, '#666666', '#06d6a080', '#06d6a0a0', '#06d6a0c0',
              C_GREEN, '#00b4d8c0', C_ACCENT2]

    for idx, (name, df) in enumerate(configs):
        ax = axes[idx]
        if len(df) == 0:
            ax.set_title(f"{name}: N=0")
            continue

        rv = df['pnl_r'].values
        counts = [((rv > edges[i]) & (rv <= edges[i+1])).sum() for i in range(len(labels))]

        ax.bar(range(len(labels)), counts, color=colors, edgecolor='white', lw=0.5)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8, rotation=30)
        ax.set_title(f"{name} (N={len(df)})")
        ax.set_ylabel('Count')
        ax.annotate(f"Mean R: {rv.mean():+.2f}\nMedian R: {np.median(rv):+.2f}",
                    xy=(0.65, 0.85), xycoords='axes fraction', fontsize=9, color=C_YELLOW)

    plt.tight_layout()
    fig.savefig(out_dir / 'r_distribution.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Chart: r_distribution.png", file=sys.stderr)


def chart_monthly_pnl(trail_df, fix2r_df, out_dir):
    """5d: Monthly PnL bars (OOS only)."""
    setup_dark_style()

    configs = [
        ('Tier 2 TRAIL_D', trail_df[(trail_df['period'] == 'OOS') & trail_df['is_tier2']]),
        ('Tier 1 TRAIL_D', trail_df[(trail_df['period'] == 'OOS') & trail_df['is_tier1']]),
        ('Tier 3 FIX_2R', fix2r_df[(fix2r_df['period'] == 'OOS') & fix2r_df['is_tier3']]),
    ]

    for name, df in configs:
        if len(df) == 0:
            continue
        fig, ax = plt.subplots(figsize=(14, 5))

        df2 = df.copy()
        df2['month'] = pd.to_datetime(df2['date']).dt.to_period('M')
        monthly = df2.groupby('month')['pnl_dollar'].sum()

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


def chart_summary_table(all_oos_kpis, out_dir):
    """5e: Summary KPI table as image."""
    setup_dark_style()

    col_keys = [
        ('Tier 1 GapUp', 'Tier 1 TRAIL_D', 'up'),
        ('Tier 1 GapDn', 'Tier 1 TRAIL_D', 'down'),
        ('Tier 2 GapUp', 'Tier 2 TRAIL_D', 'up'),
        ('Tier 2 GapDn', 'Tier 2 TRAIL_D', 'down'),
        ('Tier 3 GapUp', 'Tier 3 FIX_2R', 'up'),
    ]
    col_labels = [ck[0] for ck in col_keys]

    row_defs = [
        ('N', 'N', 'd'), ('Win%', 'Win%', '.1f'), ('Loss%', 'Loss%', '.1f'),
        ('Total PnL', 'Total PnL', '$'), ('Mean PnL', 'Mean PnL', '$'),
        ('Mean R', 'Mean R', '+.2f'), ('Median R', 'Median R', '+.2f'),
        ('Max R', 'Max R', '+.1f'),
        ('PF', 'PF', '.2f'), ('Payoff', 'Payoff', '.2f'),
        ('WR@1R', 'WR@1R', '.1f'), ('WR@2R', 'WR@2R', '.1f'),
        ('WR@3R', 'WR@3R', '.1f'),
        ('MaxConsL', 'MaxConsL', 'd'), ('MaxDD$', 'MaxDD$', '$'),
        ('MaxDD%', 'MaxDD%', '.1f'), ('MaxDDDur', 'MaxDDDur', 'd'),
        ('%InitSL', '%InitSL', '.1f'), ('%Trail', '%Trail', '.1f'),
        ('%Timeout', '%Timeout', '.1f'),
    ]

    def fmt(val, f):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return '—'
        if f == 'd':
            return str(int(val))
        if f == '$':
            return f"${val:,.0f}"
        return f"{val:{f}}"

    cell_text = []
    for rname, rkey, rfmt in row_defs:
        row = []
        for _, sname, gd in col_keys:
            kpi = all_oos_kpis.get((sname, gd))
            if kpi is None or kpi['N'] == 0:
                row.append('—')
            else:
                row.append(fmt(kpi.get(rkey), rfmt))
        cell_text.append(row)

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.axis('off')
    table = ax.table(cellText=cell_text,
                     rowLabels=[r[0] for r in row_defs],
                     colLabels=col_labels,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.4)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor('white')
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor(C_ACCENT)
            cell.set_text_props(color='white', fontweight='bold')
        elif c == -1:
            cell.set_facecolor('#2a2a4e')
            cell.set_text_props(color='white')
        else:
            cell.set_facecolor(C_BG)
            cell.set_text_props(color='white')

    ax.set_title('KPI Summary (OOS)', fontsize=14, color='white', pad=20)
    fig.savefig(out_dir / 'summary_table.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Chart: summary_table.png", file=sys.stderr)


def chart_sensitivity(trail_df, out_dir):
    """Sensitivity chart for slippage + risk scenarios."""
    setup_dark_style()

    df = trail_df[(trail_df['period'] == 'OOS') &
                  (trail_df['gap_direction'] == 'up') &
                  trail_df['is_tier2']].copy()
    if len(df) == 0:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Sensitivity: Tier 2 GapUp TRAIL_D OOS', fontsize=14)

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

    # Risk
    risks = [500, 1000, 2000, 3000, 5000]
    dds = []
    for r in risks:
        scale = r / RISK_PER_TRADE
        adj = df['pnl_dollar'].values * scale
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

    lines.append(f"\n  {label} — IS: N={n_is}{low_is}, OOS: N={n_oos}{low_oos}")
    lines.append(f"    {'Metrik':<20} | {'IS':>12} | {'OOS':>12}")
    lines.append(f"    {'-'*48}")

    def row(name, key, fmt='$'):
        v_is = kpis_is.get(key, np.nan)
        v_oos = kpis_oos.get(key, np.nan)
        def f(v):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return '—'
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

    # TRAIL_D extras
    if '%InitSL' in kpis_is or '%InitSL' in kpis_oos:
        lines.append('')
        row('% InitSL', '%InitSL', '%')
        row('% Trail', '%Trail', '%')
        row('% Timeout', '%Timeout', '%')
        row('Conversion 1R', 'Conv1R', '%')
        row('Med Trail R', 'MedTrailR', 'r')


def format_mc_results(lines, mc, label):
    """Format Monte Carlo results."""
    if mc is None:
        lines.append(f"\n  {label}: Monte Carlo nicht moeglich (N < 5)")
        return

    lines.append(f"\n  {label} — Monte Carlo ({MC_ITERATIONS} Permutationen):")
    lines.append(f"    Finale Equity:  Median ${mc['fe_median']:>10,.0f}  "
                 f"[P5: ${mc['fe_p5']:>10,.0f}  P95: ${mc['fe_p95']:>10,.0f}]")
    lines.append(f"    Max Drawdown:   Median ${mc['dd_median']:>10,.0f}  "
                 f"[P5: ${mc['dd_p5']:>10,.0f}  P95: ${mc['dd_p95']:>10,.0f}]")
    lines.append(f"    Worst DD:       ${mc['worst_dd']:>10,.0f}")
    lines.append(f"    Max Cons Loss:  Median {mc['cl_median']:.0f}  "
                 f"[P95: {mc['cl_p95']:.0f}]")

    lines.append(f"    Probability of Ruin:")
    for t in [5000, 10000, 15000, 20000]:
        lines.append(f"      DD >= ${t:>6,}: {mc[f'p_ruin_{t}']:>5.1f}%")


# ============================================================
# AUFGABE 6: SENSITIVITY TEXT
# ============================================================
def sensitivity_text(trail_df):
    """Sensitivity analysis text output."""
    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("SENSITIVITAETS-ANALYSE")
    lines.append("=" * 70)
    lines.append("Basis: Tier 2 GapUp TRAIL_D, OOS\n")

    df = trail_df[(trail_df['period'] == 'OOS') &
                  (trail_df['gap_direction'] == 'up') &
                  trail_df['is_tier2']].copy()

    if len(df) == 0:
        lines.append("Keine Trades.")
        return lines

    pnl_base = df['pnl_dollar'].values
    shares = df['shares'].values

    # 6a: Slippage
    lines.append("--- 6a: Slippage-Szenarien ---")
    lines.append(f"  {'Slippage':>10} | {'Total PnL':>12} | {'Mean PnL':>10} | "
                 f"{'Mean R':>8} | {'Win%':>6} | {'PF':>6} | {'MaxDD$':>10}")
    lines.append(f"  {'-'*78}")

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
                     f"{adj_r.mean():>+6.3f} | {w.mean()*100:>5.1f} | "
                     f"{gw/gl:>5.2f} | ${dd:>8,.0f}")

    # Break-even slippage
    total_roundtrip_shares = (shares * 2).sum()
    be_slip = max(0, pnl_base.sum() / total_roundtrip_shares)
    lines.append(f"\n  Break-Even Slippage: ${be_slip:.4f} pro Share")

    # 6b: Risk
    lines.append(f"\n--- 6b: Risk-per-Trade Szenarien ---")
    lines.append(f"  {'Risk':>8} | {'Total PnL':>12} | {'Mean PnL':>10} | "
                 f"{'MaxDD$':>10} | {'MaxDD/1R':>10}")
    lines.append(f"  {'-'*60}")

    for r in [500, 1000, 2000, 5000]:
        scale = r / RISK_PER_TRADE
        adj = pnl_base * scale
        eq = np.cumsum(adj)
        dd = (eq - np.maximum.accumulate(eq)).min()
        lines.append(f"  ${r:>6} | ${adj.sum():>10,.0f} | ${adj.mean():>8,.0f} | "
                     f"${dd:>8,.0f} | {abs(dd)/r*100:>8.1f}%")

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
    meta = meta[meta['date'] >= '2021-02-21'].copy()
    print(f"  {len(meta)} rows", file=sys.stderr)

    # --- Build all trades ---
    print("\nAufgabe 1+2: Simulating trades...", file=sys.stderr)
    trail_df, fix2r_df = build_trades(meta)

    # Zoom stats
    for name, df in [('TRAIL_D', trail_df), ('FIX_2R', fix2r_df)]:
        if len(df) > 0:
            nz = df['zoom_used'].sum()
            nc = df['zoom_changed'].sum()
            print(f"  {name} zoom: {nz} used, {nc} changed", file=sys.stderr)

    # --- Define setup groups ---
    setup_configs = {
        'Tier 1 TRAIL_D': (trail_df, 'is_tier1'),
        'Tier 2 TRAIL_D': (trail_df, 'is_tier2'),
        'Tier 3 FIX_2R':  (fix2r_df, 'is_tier3'),
    }

    # --- Aufgabe 3+4: KPIs and Monte Carlo ---
    print("\nAufgabe 3: KPIs...", file=sys.stderr)
    all_kpis = {}   # (setup_name, gap_dir, period) -> kpi
    all_mc = {}     # (setup_name, gap_dir, period) -> mc

    for sname, (sdf, tier_col) in setup_configs.items():
        for gd in ['up', 'down']:
            for per in ['IS', 'OOS']:
                sub = sdf[(sdf['gap_direction'] == gd) &
                          (sdf['period'] == per) &
                          sdf[tier_col]].copy()
                gl = 'GapUp' if gd == 'up' else 'GapDn'
                label = f"{sname} {gl} {per}"
                kpi = compute_kpis(sub, label)
                all_kpis[(sname, gd, per)] = kpi

                # Monte Carlo
                if kpi['N'] >= 5:
                    mc = monte_carlo(sub['pnl_dollar'].values)
                    all_mc[(sname, gd, per)] = mc
                else:
                    all_mc[(sname, gd, per)] = None

    print("  Done.", file=sys.stderr)

    # --- Aufgabe 5: Charts ---
    print("\nAufgabe 5: Charts...", file=sys.stderr)

    for sname in setup_configs:
        kpis_for_setup = {(gd, per): all_kpis[(sname, gd, per)]
                          for gd in ['up', 'down'] for per in ['IS', 'OOS']}
        mc_for_setup = {(gd, per): all_mc[(sname, gd, per)]
                        for gd in ['up', 'down'] for per in ['IS', 'OOS']}
        chart_equity(sname, kpis_for_setup, mc_for_setup, RESULTS_DIR)
        chart_drawdown(sname, mc_for_setup, RESULTS_DIR)

    chart_r_distribution(trail_df, fix2r_df, RESULTS_DIR)
    chart_monthly_pnl(trail_df, fix2r_df, RESULTS_DIR)

    # Summary table (OOS)
    oos_kpis_for_table = {}
    for sname in setup_configs:
        for gd in ['up', 'down']:
            oos_kpis_for_table[(sname, gd)] = all_kpis[(sname, gd, 'OOS')]
    chart_summary_table(oos_kpis_for_table, RESULTS_DIR)

    # Sensitivity chart
    chart_sensitivity(trail_df, RESULTS_DIR)

    # --- Text Output ---
    print("\nWriting text output...", file=sys.stderr)
    lines = []
    lines.append("=" * 70)
    lines.append("BACKTEST + MONTE CARLO ERGEBNISSE")
    lines.append("=" * 70)
    lines.append(f"Risk: ${RISK_PER_TRADE:.0f}/Trade, SL: {SL_ADR_FRACTION} ADR, "
                 f"Slippage: ${SLIPPAGE_PER_SHARE}")
    lines.append(f"MC: {MC_ITERATIONS} Permutationen, Seed=42")
    lines.append(f"IS: 2021-02-21 bis 2023-12-31, OOS: 2024-01-01+")

    # Zoom stats
    for name, df in [('TRAIL_D', trail_df), ('FIX_2R', fix2r_df)]:
        if len(df) > 0:
            nz = int(df['zoom_used'].sum())
            nc = int(df['zoom_changed'].sum())
            lines.append(f"{name}: {nz}/{len(df)} Trades used 1-sec zoom, "
                         f"{nc} changed result")

    for sname in setup_configs:
        lines.append(f"\n\n{'='*70}")
        lines.append(f"=== {sname} ===")
        lines.append(f"{'='*70}")

        for gd in ['up', 'down']:
            gl = 'GapUp' if gd == 'up' else 'GapDn'
            k_is = all_kpis[(sname, gd, 'IS')]
            k_oos = all_kpis[(sname, gd, 'OOS')]
            format_kpi_table(lines, k_is, k_oos, f"{sname} {gl}")

            # MC results for OOS
            mc_oos = all_mc.get((sname, gd, 'OOS'))
            format_mc_results(lines, mc_oos, f"{sname} {gl} OOS")

    # Sensitivity
    sens_lines = sensitivity_text(trail_df)
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
