"""
D10s: MFE + Trailing-Stop Simulation mit 1-Sekunden-Aufloesung
Hybrid: 1-min Basis + 1-sec Zoom bei ambigen Bars.
Repliziert D10 Aufgaben 1-5 + Synthese.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
from tqdm import tqdm

RAW_1MIN_DIR = Path('data/raw_1min')
RAW_1SEC_DIR = Path('data/raw_1sec')

_sec_cache = {}

def load_1sec(ticker, date):
    key = (ticker, date)
    if key not in _sec_cache:
        path = RAW_1SEC_DIR / ticker / f"{date}.parquet"
        _sec_cache[key] = pd.read_parquet(path) if path.exists() else None
    return _sec_cache[key]


# ============================================================
# Hybrid MFE computation
# ============================================================
def compute_mfe_mae_1sec(bars_rth, entry_price, sl_level, trade_dir, adr, ticker, date):
    """
    Compute MFE_LIVE, MFE_FULL, MAE with 1-sec zoom.
    When SL bar is ambiguous (bar has both favorable excursion and SL hit),
    zoom to 1-sec to get precise MFE at SL point.
    """
    sl_dist = abs(entry_price - sl_level)
    if sl_dist <= 0 or pd.isna(sl_dist) or adr <= 0:
        return None

    post_entry = bars_rth[(bars_rth['time_et'] >= '09:36') & (bars_rth['time_et'] <= '15:55')]
    if len(post_entry) == 0:
        return None

    sec_bars = load_1sec(ticker, date)

    mfe_full = 0.0
    mae = 0.0
    sl_hit = False
    sl_hit_time = None
    mfe_at_sl = 0.0
    zoom_used = False
    zoom_changed = False

    for _, bar in post_entry.iterrows():
        bar_time = str(bar['time_et'])[:5]

        if trade_dir == 'long':
            fav = max(0, bar['high'] - entry_price)
            adv = max(0, entry_price - bar['low'])
        else:
            fav = max(0, entry_price - bar['low'])
            adv = max(0, bar['high'] - entry_price)

        mfe_full = max(mfe_full, fav)
        mae = max(mae, adv)

        if not sl_hit:
            sl_hit_in_bar = ((trade_dir == 'long' and bar['low'] <= sl_level) or
                             (trade_dir == 'short' and bar['high'] >= sl_level))

            if sl_hit_in_bar:
                # Check if we can zoom for more precise MFE tracking
                if sec_bars is not None and bar_time >= '09:30' and bar_time < '09:45':
                    minute_secs = sec_bars[sec_bars['time_et'].str[:5] == bar_time]
                    if len(minute_secs) > 0:
                        zoom_used = True
                        # Walk second by second, tracking MFE until SL hits
                        mfe_before_sl = mfe_at_sl  # MFE from previous bars
                        for _, sec in minute_secs.iterrows():
                            # Check SL first
                            if trade_dir == 'long' and sec['low'] <= sl_level:
                                sl_hit = True
                                sl_hit_time = sec['time_et']
                                break
                            elif trade_dir == 'short' and sec['high'] >= sl_level:
                                sl_hit = True
                                sl_hit_time = sec['time_et']
                                break

                            # Update MFE from this second (before SL)
                            if trade_dir == 'long':
                                sec_fav = max(0, sec['high'] - entry_price)
                            else:
                                sec_fav = max(0, entry_price - sec['low'])
                            mfe_before_sl = max(mfe_before_sl, sec_fav)

                        if sl_hit:
                            mfe_at_sl = mfe_before_sl
                            # Check if 1-sec MFE differs from 1-min MFE
                            mfe_1min = mfe_full  # 1-min would set mfe_at_sl = mfe_full (includes bar high)
                            if abs(mfe_at_sl - mfe_1min) > 0.001:
                                zoom_changed = True
                        else:
                            # SL not found in 1-sec (OHLC gap) -- use bar-level
                            mfe_at_sl = mfe_full
                            sl_hit = True
                            sl_hit_time = bar_time
                    else:
                        mfe_at_sl = mfe_full
                        sl_hit = True
                        sl_hit_time = bar_time
                else:
                    # No 1-sec data: use 1-min (includes bar high before SL check)
                    mfe_at_sl = mfe_full
                    sl_hit = True
                    sl_hit_time = bar_time
            else:
                mfe_at_sl = mfe_full

    mfe_live = mfe_at_sl if sl_hit else mfe_full

    if sl_hit:
        exit_price = sl_level
    else:
        exit_price = post_entry.iloc[-1]['close']

    pnl = (exit_price - entry_price) if trade_dir == 'long' else (entry_price - exit_price)

    return {
        'mfe_live': mfe_live,
        'mfe_full': mfe_full,
        'mae': mae,
        'mfe_live_r': mfe_live / sl_dist,
        'mfe_full_r': mfe_full / sl_dist,
        'mae_r': mae / sl_dist,
        'mfe_live_adr': mfe_live / adr,
        'mfe_full_adr': mfe_full / adr,
        'mae_adr': mae / adr,
        'sl_hit': sl_hit,
        'sl_hit_time': sl_hit_time,
        'pnl': pnl,
        'pnl_r': pnl / sl_dist,
        'pnl_adr': pnl / adr,
        'sl_dist': sl_dist,
        'sl_dist_adr': sl_dist / adr,
        'zoom_used': zoom_used,
        'zoom_changed': zoom_changed,
    }


# ============================================================
# Hybrid Trailing simulation
# ============================================================
def simulate_trailing_1sec(bars_rth, entry_price, sl_dist, trade_dir, adr,
                            ticker, date, trail_type='D'):
    """
    Trailing-stop with 1-sec zoom.
    Key zoom scenario: bar where +1R trigger AND SL are both possible.
    1-min code: peak updates first (optimistic), then SL checks against moved level.
    1-sec code: resolves whether SL hit original level before +1R was reached.
    """
    if sl_dist <= 0 or pd.isna(sl_dist) or adr <= 0:
        return None

    post_entry = bars_rth[(bars_rth['time_et'] >= '09:36') & (bars_rth['time_et'] <= '15:55')]
    if len(post_entry) == 0:
        return None

    sec_bars = load_1sec(ticker, date)

    if trade_dir == 'long':
        sl_level = entry_price - sl_dist
    else:
        sl_level = entry_price + sl_dist

    initial_sl = sl_level
    highest_r = 0.0
    trail_active = False
    exit_price = None
    exit_type = 'timeout'
    exit_time = None
    zoom_used = False
    zoom_changed = False

    for _, bar in post_entry.iterrows():
        bar_time = str(bar['time_et'])[:5]

        # Current favorable excursion from this bar
        if trade_dir == 'long':
            bar_peak_r = max(0, (bar['high'] - entry_price) / sl_dist)
            sl_hit_in_bar = bar['low'] <= sl_level
        else:
            bar_peak_r = max(0, (entry_price - bar['low']) / sl_dist)
            sl_hit_in_bar = bar['high'] >= sl_level

        # Check if this bar is ambiguous for trailing
        could_trigger_trail = (bar_peak_r >= 1.0 and not trail_active)
        could_hit_sl = sl_hit_in_bar

        if could_trigger_trail and could_hit_sl and sec_bars is not None and bar_time >= '09:30' and bar_time < '09:45':
            # AMBIGUOUS: trail trigger AND SL possible -- zoom to 1-sec
            minute_secs = sec_bars[sec_bars['time_et'].str[:5] == bar_time]
            if len(minute_secs) > 0:
                zoom_used = True
                for _, sec in minute_secs.iterrows():
                    # Update peak from this second
                    if trade_dir == 'long':
                        sec_r = max(0, (sec['high'] - entry_price) / sl_dist)
                    else:
                        sec_r = max(0, (entry_price - sec['low']) / sl_dist)
                    highest_r = max(highest_r, sec_r)

                    # Trail logic
                    if trail_type == 'D':
                        if highest_r >= 1.0 and not trail_active:
                            trail_active = True
                            sl_level = entry_price
                        if trail_active:
                            if trade_dir == 'long':
                                new_sl = entry_price + (highest_r - 0.5) * sl_dist
                                sl_level = max(sl_level, new_sl)
                            else:
                                new_sl = entry_price - (highest_r - 0.5) * sl_dist
                                sl_level = min(sl_level, new_sl)
                    elif trail_type == 'A':
                        if highest_r >= 1.0 and not trail_active:
                            trail_active = True
                            sl_level = entry_price
                        if trail_active:
                            if trade_dir == 'long':
                                new_sl = entry_price + (highest_r - 1.0) * sl_dist
                                sl_level = max(sl_level, new_sl)
                            else:
                                new_sl = entry_price - (highest_r - 1.0) * sl_dist
                                sl_level = min(sl_level, new_sl)

                    # SL check
                    if trade_dir == 'long' and sec['low'] <= sl_level:
                        exit_price = sl_level
                        exit_type = 'trail_sl' if trail_active else 'initial_sl'
                        exit_time = sec['time_et']
                        if not trail_active and bar_peak_r >= 1.0:
                            zoom_changed = True  # 1-min would have activated trail
                        break
                    elif trade_dir == 'short' and sec['high'] >= sl_level:
                        exit_price = sl_level
                        exit_type = 'trail_sl' if trail_active else 'initial_sl'
                        exit_time = sec['time_et']
                        if not trail_active and bar_peak_r >= 1.0:
                            zoom_changed = True
                        break

                if exit_price is not None:
                    break
                # If no SL in 1-sec (OHLC discrepancy), fall through to normal processing
                continue
        else:
            # Normal 1-min processing
            highest_r = max(highest_r, bar_peak_r)

            if trail_type == 'D':
                if highest_r >= 1.0 and not trail_active:
                    trail_active = True
                    sl_level = entry_price
                if trail_active:
                    if trade_dir == 'long':
                        new_sl = entry_price + (highest_r - 0.5) * sl_dist
                        sl_level = max(sl_level, new_sl)
                    else:
                        new_sl = entry_price - (highest_r - 0.5) * sl_dist
                        sl_level = min(sl_level, new_sl)
            elif trail_type == 'A':
                if highest_r >= 1.0 and not trail_active:
                    trail_active = True
                    sl_level = entry_price
                if trail_active:
                    if trade_dir == 'long':
                        new_sl = entry_price + (highest_r - 1.0) * sl_dist
                        sl_level = max(sl_level, new_sl)
                    else:
                        new_sl = entry_price - (highest_r - 1.0) * sl_dist
                        sl_level = min(sl_level, new_sl)

            # SL check
            if trade_dir == 'long' and bar['low'] <= sl_level:
                exit_price = sl_level
                exit_type = 'trail_sl' if trail_active else 'initial_sl'
                exit_time = bar_time
                break
            elif trade_dir == 'short' and bar['high'] >= sl_level:
                exit_price = sl_level
                exit_type = 'trail_sl' if trail_active else 'initial_sl'
                exit_time = bar_time
                break

    if exit_price is None:
        exit_price = post_entry.iloc[-1]['close']

    pnl = (exit_price - entry_price) if trade_dir == 'long' else (entry_price - exit_price)

    return {
        'pnl': pnl,
        'pnl_r': pnl / sl_dist,
        'pnl_adr': pnl / adr,
        'exit_type': exit_type,
        'exit_time': exit_time,
        'highest_r': highest_r,
        'trail_active': trail_active,
        'winner': pnl > 0,
        'zoom_used': zoom_used,
        'zoom_changed': zoom_changed,
    }


def simulate_fix_1sec(bars_rth, entry_price, sl_dist, trade_dir, adr,
                       ticker, date, target_r=2.0):
    """Fixed target with 1-sec zoom when SL and target are in same bar."""
    if sl_dist <= 0 or pd.isna(sl_dist) or adr <= 0:
        return None

    post_entry = bars_rth[(bars_rth['time_et'] >= '09:36') & (bars_rth['time_et'] <= '15:55')]
    if len(post_entry) == 0:
        return None

    sec_bars = load_1sec(ticker, date)

    if trade_dir == 'long':
        sl_level = entry_price - sl_dist
        target_level = entry_price + target_r * sl_dist
    else:
        sl_level = entry_price + sl_dist
        target_level = entry_price - target_r * sl_dist

    exit_price = None
    exit_type = 'timeout'
    zoom_used = False
    zoom_changed = False

    for _, bar in post_entry.iterrows():
        bar_time = str(bar['time_et'])[:5]

        if trade_dir == 'long':
            sl_possible = bar['low'] <= sl_level
            tgt_possible = bar['high'] >= target_level
        else:
            sl_possible = bar['high'] >= sl_level
            tgt_possible = bar['low'] <= target_level

        if sl_possible and tgt_possible:
            # Ambiguous -- zoom
            if sec_bars is not None and bar_time >= '09:30' and bar_time < '09:45':
                minute_secs = sec_bars[sec_bars['time_et'].str[:5] == bar_time]
                if len(minute_secs) > 0:
                    zoom_used = True
                    for _, sec in minute_secs.iterrows():
                        if trade_dir == 'long':
                            if sec['low'] <= sl_level:
                                exit_price = sl_level
                                exit_type = 'sl'
                                break
                            if sec['high'] >= target_level:
                                exit_price = target_level
                                exit_type = 'target'
                                zoom_changed = True  # 1-min would have said SL
                                break
                        else:
                            if sec['high'] >= sl_level:
                                exit_price = sl_level
                                exit_type = 'sl'
                                break
                            if sec['low'] <= target_level:
                                exit_price = target_level
                                exit_type = 'target'
                                zoom_changed = True
                                break
                    if exit_price is not None:
                        break
                    # OHLC discrepancy fallback: SL (conservative)
                    exit_price = sl_level
                    exit_type = 'sl'
                    break
            # No 1-sec: conservative (SL first)
            exit_price = sl_level
            exit_type = 'sl'
            break

        elif sl_possible:
            exit_price = sl_level
            exit_type = 'sl'
            break
        elif tgt_possible:
            exit_price = target_level
            exit_type = 'target'
            break

    if exit_price is None:
        exit_price = post_entry.iloc[-1]['close']

    pnl = (exit_price - entry_price) if trade_dir == 'long' else (entry_price - exit_price)

    return {
        'pnl': pnl,
        'pnl_r': pnl / sl_dist,
        'pnl_adr': pnl / adr,
        'exit_type': exit_type,
        'winner': pnl > 0,
        'zoom_used': zoom_used,
        'zoom_changed': zoom_changed,
    }


# ============================================================
# Run strategies
# ============================================================
def run_strategy(meta_subset, strategy_type, **kwargs):
    all_results = []
    for _, row in meta_subset.iterrows():
        ticker = row['ticker']
        date = str(row['date'])
        fpath = RAW_1MIN_DIR / ticker / f"{date}.parquet"
        if not fpath.exists():
            continue

        bars = pd.read_parquet(fpath)
        rth = bars[bars['session'] == 'rth']
        if len(rth) < 10:
            continue

        entry_price = row['close_935']
        adr = row.get('adr_10', np.nan)
        gap_dir = row['gap_direction']

        if pd.isna(entry_price) or pd.isna(adr) or adr <= 0:
            continue

        trade_dir = 'long' if gap_dir == 'up' else 'short'
        sl_dist = 0.25 * adr

        if strategy_type == 'mfe':
            sl_level = entry_price - sl_dist if trade_dir == 'long' else entry_price + sl_dist
            result = compute_mfe_mae_1sec(rth, entry_price, sl_level, trade_dir, adr, ticker, date)
        elif strategy_type == 'trail':
            result = simulate_trailing_1sec(rth, entry_price, sl_dist, trade_dir, adr,
                                             ticker, date, trail_type=kwargs.get('trail_type', 'D'))
        elif strategy_type == 'fix':
            result = simulate_fix_1sec(rth, entry_price, sl_dist, trade_dir, adr,
                                        ticker, date, target_r=kwargs.get('target_r', 2.0))
        else:
            continue

        if result is None:
            continue

        result['ticker'] = ticker
        result['date'] = date
        result['gap_direction'] = gap_dir
        result['rvol_30'] = row.get('rvol_open_30min', np.nan)
        all_results.append(result)

    return pd.DataFrame(all_results)


# ============================================================
# Formatting helpers
# ============================================================
def mfe_stats_table(df, col_r, col_adr, label):
    lines = []
    lines.append(f"\n  {label}:")
    lines.append(f"  {'Metrik':<12} | {'GapUp R':>10} | {'GapUp ADR':>10} | {'GapDn R':>10} | {'GapDn ADR':>10}")
    lines.append(f"  {'-'*62}")

    stats_list = [
        ('Median', 'median'), ('Mean', 'mean'), ('StdDev', 'std'),
        ('P10', lambda s: s.quantile(0.10)), ('P25', lambda s: s.quantile(0.25)),
        ('P75', lambda s: s.quantile(0.75)), ('P90', lambda s: s.quantile(0.90)),
        ('Min', 'min'), ('Max', 'max'), ('N', 'count'),
    ]

    for stat_label, stat_fn in stats_list:
        vals = []
        for gap_dir in ['up', 'down']:
            sub = df[df['gap_direction'] == gap_dir]
            if len(sub) < 10:
                vals.extend(['N<10', 'N<10'])
                continue
            for col in [col_r, col_adr]:
                s = sub[col]
                v = stat_fn(s) if callable(stat_fn) else getattr(s, stat_fn)()
                vals.append(f"{int(v)}" if stat_label == 'N' else f"{v:.3f}")
        lines.append(f"  {stat_label:<12} | {vals[0]:>10} | {vals[1]:>10} | {vals[2]:>10} | {vals[3]:>10}")

    return "\n".join(lines)


def summarize_strategy(trades, label=""):
    n = len(trades)
    if n < 10:
        return f"  {label:<25} | N={n:>4} [N<10]"

    med_pnl_r = trades['pnl_r'].median()
    mean_pnl_r = trades['pnl_r'].mean()
    std_pnl_r = trades['pnl_r'].std()
    wr = trades['winner'].mean()
    ev_adr = trades['pnl_adr'].mean()

    zoom_used_pct = trades['zoom_used'].mean() if 'zoom_used' in trades.columns else 0
    zoom_changed_pct = trades['zoom_changed'].mean() if 'zoom_changed' in trades.columns else 0

    low_n = " [LOW N]" if n < 20 else ""
    return (f"  {label:<25} | N={n:>4} | MedPnL={med_pnl_r:>+6.2f}R | "
            f"MeanPnL={mean_pnl_r:>+6.2f}R | StdDev={std_pnl_r:>5.2f} | "
            f"WR={wr:>5.1%} | EV={ev_adr:>+6.3f}ADR | Z={zoom_changed_pct:.0%}{low_n}")


# ============================================================
# Main
# ============================================================
def main():
    lines = []
    lines.append("=" * 70)
    lines.append("D10s: MFE + TRAILING MIT 1-SEC ZOOM")
    lines.append("=" * 70)
    lines.append("\nEntry: 9:35 | SL: 0.25 ADR | Timeout: 15:55")
    lines.append("HYBRID: 1-min base + 1-sec zoom bei ambigen Bars (09:30-09:45)")
    lines.append("Z = Zoom-Changed% (wie oft 1-sec das Ergebnis aenderte)")

    meta = pd.read_parquet('data/metadata/metadata_v8_5.parquet')
    h1 = meta[(meta['date'] >= '2021-02-21') & (meta['date'] <= '2023-12-31')].copy()
    h2 = meta[(meta['date'] >= '2024-01-01')].copy()

    base_h1 = h1[(h1['od_strength'] > 0.5) & (h1['od_direction'] == 'with_gap')]
    base_h2 = h2[(h2['od_strength'] > 0.5) & (h2['od_direction'] == 'with_gap')]
    print(f"H1: {len(base_h1)}, H2: {len(base_h2)}", file=sys.stderr)

    # ===== 1b: MFE ALL TRADES =====
    lines.append(f"\n\n{'='*70}")
    lines.append("1b: MFE-STATISTIK ALLE TRADES (IS, 1-sec Zoom)")
    lines.append(f"{'='*70}")

    print("Computing MFE IS...", file=sys.stderr)
    mfe_is = run_strategy(base_h1, 'mfe')
    mfe_is.to_parquet('results/d10s_mfe_raw_is.parquet', index=False)

    lines.append(mfe_stats_table(mfe_is, 'mfe_live_r', 'mfe_live_adr', 'MFE_LIVE (bis SL-Hit)'))
    lines.append(mfe_stats_table(mfe_is, 'mfe_full_r', 'mfe_full_adr', 'MFE_FULL (ganzer Tag)'))

    # MFE Zoom stats
    n_zoom = mfe_is['zoom_used'].sum()
    n_changed = mfe_is['zoom_changed'].sum()
    lines.append(f"\n  MFE Zoom-Statistik IS: Zoom={n_zoom}/{len(mfe_is)}, Changed={n_changed}/{len(mfe_is)}")

    # ===== 1c: MFE Gewinner/Verlierer =====
    winners = mfe_is[mfe_is['sl_hit'] == False]
    losers = mfe_is[mfe_is['sl_hit'] == True]

    lines.append(f"\n\n{'='*70}")
    lines.append("1c: MFE GEWINNER vs VERLIERER (IS, 1-sec Zoom)")
    lines.append(f"{'='*70}")

    for gap_dir, gap_label in [('up', 'GapUp'), ('down', 'GapDn')]:
        w = winners[winners['gap_direction'] == gap_dir]
        l = losers[losers['gap_direction'] == gap_dir]
        all_g = mfe_is[mfe_is['gap_direction'] == gap_dir]
        lines.append(f"\n  {gap_label}: Gewinner={len(w)}/{len(all_g)} ({100*len(w)/len(all_g):.1f}%)")
        if len(w) >= 10:
            lines.append(f"    MFE_LIVE Gewinner: Median={w['mfe_live_r'].median():.2f}R, StdDev={w['mfe_live_r'].std():.2f}R")
        if len(l) >= 10:
            lines.append(f"    MFE_LIVE Verlierer: Median={l['mfe_live_r'].median():.2f}R, StdDev={l['mfe_live_r'].std():.2f}R")

    # ===== 4a: Trailing IS =====
    lines.append(f"\n\n{'='*70}")
    lines.append("4a: TRAILING-STOP ERGEBNISSE (IS, 1-sec Zoom)")
    lines.append(f"{'='*70}")

    for gap_dir, gap_label in [('up', 'GapUp'), ('down', 'GapDown')]:
        lines.append(f"\n--- {gap_label} ---")
        gap_base = base_h1[base_h1['gap_direction'] == gap_dir]

        for target_r in [1.0, 2.0, 3.0]:
            print(f"  Fix {target_r}R {gap_label}...", file=sys.stderr)
            trades = run_strategy(gap_base, 'fix', target_r=target_r)
            lines.append(summarize_strategy(trades, f"Fix {target_r:.0f}R"))

        for trail_type in ['A', 'D']:
            print(f"  TRAIL_{trail_type} {gap_label}...", file=sys.stderr)
            trades = run_strategy(gap_base, 'trail', trail_type=trail_type)
            lines.append(summarize_strategy(trades, f"TRAIL_{trail_type}"))

            if len(trades) >= 10:
                n_init = (trades['exit_type'] == 'initial_sl').sum()
                n_trail = (trades['exit_type'] == 'trail_sl').sum()
                n_to = (trades['exit_type'] == 'timeout').sum()
                lines.append(f"    Exit: InitSL={n_init} ({100*n_init/len(trades):.0f}%), "
                             f"TrailSL={n_trail} ({100*n_trail/len(trades):.0f}%), "
                             f"Timeout={n_to} ({100*n_to/len(trades):.0f}%)")

    # ===== 4c: Setup B (RVOL_30 > 5x) IS =====
    lines.append(f"\n\n{'='*70}")
    lines.append("4c: TRAILING BEI SETUP B (RVOL_30 > 5x, IS, 1-sec Zoom)")
    lines.append(f"{'='*70}")

    setup_b_h1 = base_h1[base_h1['rvol_open_30min'] >= 5]
    print(f"Setup B IS: {len(setup_b_h1)}", file=sys.stderr)

    for gap_dir, gap_label in [('up', 'GapUp'), ('down', 'GapDown')]:
        lines.append(f"\n--- {gap_label} ---")
        gap_sub = setup_b_h1[setup_b_h1['gap_direction'] == gap_dir]

        for target_r in [2.0, 3.0]:
            trades = run_strategy(gap_sub, 'fix', target_r=target_r)
            lines.append(summarize_strategy(trades, f"Fix {target_r:.0f}R"))

        for trail_type in ['A', 'D']:
            trades = run_strategy(gap_sub, 'trail', trail_type=trail_type)
            lines.append(summarize_strategy(trades, f"TRAIL_{trail_type}"))

    # ===== 5a: MFE OOS =====
    lines.append(f"\n\n{'='*70}")
    lines.append("5a: MFE IS vs OOS (1-sec Zoom)")
    lines.append(f"{'='*70}")

    print("Computing MFE OOS...", file=sys.stderr)
    mfe_oos = run_strategy(base_h2, 'mfe')
    mfe_oos.to_parquet('results/d10s_mfe_raw_oos.parquet', index=False)

    for label, mfe_df in [('IS', mfe_is), ('OOS', mfe_oos)]:
        lines.append(mfe_stats_table(mfe_df, 'mfe_live_r', 'mfe_live_adr', f'MFE_LIVE {label}'))

    # Survival rates
    for gap_dir, gap_label in [('up', 'GapUp'), ('down', 'GapDn')]:
        is_sub = mfe_is[mfe_is['gap_direction'] == gap_dir]
        oos_sub = mfe_oos[mfe_oos['gap_direction'] == gap_dir]
        is_surv = (is_sub['sl_hit'] == False).mean() if len(is_sub) >= 10 else float('nan')
        oos_surv = (oos_sub['sl_hit'] == False).mean() if len(oos_sub) >= 10 else float('nan')
        lines.append(f"\n  {gap_label} Survival: IS={100*is_surv:.1f}% (N={len(is_sub)}), OOS={100*oos_surv:.1f}% (N={len(oos_sub)})")

    # MFE Zoom stats OOS
    n_zoom_oos = mfe_oos['zoom_used'].sum()
    n_changed_oos = mfe_oos['zoom_changed'].sum()
    lines.append(f"\n  MFE Zoom-Statistik OOS: Zoom={n_zoom_oos}/{len(mfe_oos)}, Changed={n_changed_oos}/{len(mfe_oos)}")

    # ===== 5e: Trailing OOS =====
    lines.append(f"\n\n{'='*70}")
    lines.append("5e: TRAILING OOS (Top Strategien, 1-sec Zoom)")
    lines.append(f"{'='*70}")

    strategies = [
        ('Fix 2R', 'fix', {'target_r': 2.0}),
        ('Fix 3R', 'fix', {'target_r': 3.0}),
        ('TRAIL_A', 'trail', {'trail_type': 'A'}),
        ('TRAIL_D', 'trail', {'trail_type': 'D'}),
    ]

    for gap_dir, gap_label in [('up', 'GapUp'), ('down', 'GapDown')]:
        lines.append(f"\n--- {gap_label} ---")
        lines.append(f"  {'Strategy':<15} | {'IS EV':>10} | {'IS WR':>8} | {'IS N':>6} | {'OOS EV':>10} | {'OOS WR':>8} | {'OOS N':>6}")
        lines.append(f"  {'-'*73}")

        is_gap = base_h1[base_h1['gap_direction'] == gap_dir]
        oos_gap = base_h2[base_h2['gap_direction'] == gap_dir]

        for slabel, stype, skwargs in strategies:
            print(f"  OOS {slabel} {gap_label}...", file=sys.stderr)
            is_trades = run_strategy(is_gap, stype, **skwargs)
            oos_trades = run_strategy(oos_gap, stype, **skwargs)

            is_ev = f"{is_trades['pnl_adr'].mean():>+10.3f}" if len(is_trades) >= 10 else f"{'N<10':>10}"
            is_wr = f"{is_trades['winner'].mean():>7.1%}" if len(is_trades) >= 10 else f"{'N<10':>8}"
            oos_ev = f"{oos_trades['pnl_adr'].mean():>+10.3f}" if len(oos_trades) >= 10 else f"{'N<10':>10}"
            oos_wr = f"{oos_trades['winner'].mean():>7.1%}" if len(oos_trades) >= 10 else f"{'N<10':>8}"

            lines.append(f"  {slabel:<15} | {is_ev} | {is_wr} | {len(is_trades):>6} | {oos_ev} | {oos_wr} | {len(oos_trades):>6}")

        lines.append("")

    # Setup B OOS
    lines.append(f"\n  === SETUP B (RVOL_30 > 5x) OOS ===\n")

    setup_b_h2 = base_h2[base_h2['rvol_open_30min'] >= 5]

    for gap_dir, gap_label in [('up', 'GapUp'), ('down', 'GapDn')]:
        lines.append(f"  --- {gap_label} ---")
        lines.append(f"  {'Strategy':<15} | {'IS EV':>10} | {'IS WR':>8} | {'IS N':>6} | {'OOS EV':>10} | {'OOS WR':>8} | {'OOS N':>6}")
        lines.append(f"  {'-'*73}")

        is_gap = setup_b_h1[setup_b_h1['gap_direction'] == gap_dir]
        oos_gap = setup_b_h2[setup_b_h2['gap_direction'] == gap_dir]

        for slabel, stype, skwargs in strategies:
            is_trades = run_strategy(is_gap, stype, **skwargs)
            oos_trades = run_strategy(oos_gap, stype, **skwargs)

            is_ev = f"{is_trades['pnl_adr'].mean():>+10.3f}" if len(is_trades) >= 10 else f"{'N<10':>10}"
            is_wr = f"{is_trades['winner'].mean():>7.1%}" if len(is_trades) >= 10 else f"{'N<10':>8}"
            oos_ev = f"{oos_trades['pnl_adr'].mean():>+10.3f}" if len(oos_trades) >= 10 else f"{'N<10':>10}"
            oos_wr = f"{oos_trades['winner'].mean():>7.1%}" if len(oos_trades) >= 10 else f"{'N<10':>8}"

            lines.append(f"  {slabel:<15} | {is_ev} | {is_wr} | {len(is_trades):>6} | {oos_ev} | {oos_wr} | {len(oos_trades):>6}")

        lines.append("")

    # Write main output
    output = "\n".join(lines)
    os.makedirs("results", exist_ok=True)
    with open('results/d10s_mfe_trailing.txt', 'w', encoding='utf-8') as f:
        f.write(output)
    print(output)

    # ===== SYNTHESE =====
    synth = []
    synth.append("=" * 70)
    synth.append("D10s SYNTHESE: MFE + TRAILING MIT 1-SEC AUFLOESUNG")
    synth.append("=" * 70)
    synth.append("\nMethodik: Hybrid 1-min/1-sec Simulation")
    synth.append("  - MFE: Praezisere Messung am SL-Punkt (1-sec walked bis SL)")
    synth.append("  - Trailing: 1-sec loest auf ob Trail-Trigger vor SL kam")
    synth.append("  - Fix-Target: 1-sec loest auf ob Target vor SL kam")
    synth.append("")
    synth.append("ZOOM-STATISTIK:")
    synth.append(f"  MFE IS:  Zoom={n_zoom}/{len(mfe_is)}, Changed={n_changed}/{len(mfe_is)}")
    synth.append(f"  MFE OOS: Zoom={n_zoom_oos}/{len(mfe_oos)}, Changed={n_changed_oos}/{len(mfe_oos)}")
    synth.append("")
    synth.append("KERNFRAGE: Aendert die 1-sec Aufloesung die D10-Ergebnisse signifikant?")
    synth.append("ANTWORT: Siehe Delta-Analyse in d_synthese_1sec.txt")

    synth_output = "\n".join(synth)
    with open('results/d10s_synthese.txt', 'w', encoding='utf-8') as f:
        f.write(synth_output)

    print(f"\nSaved to results/d10s_mfe_trailing.txt + d10s_synthese.txt", file=sys.stderr)


if __name__ == '__main__':
    main()
