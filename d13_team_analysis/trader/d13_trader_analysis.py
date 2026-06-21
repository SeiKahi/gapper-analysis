"""
D13 Discretionary Trader Analysis
==================================
Intraday Price-Action Pattern Analysis & Creative Trading Setups
NUR IS-Daten: 2021-02-21 bis 2023-12-31

Analysen:
  1. Erste-30-Minuten Microstructure
  2. Time-of-Day PnL-Kurve
  3. Reversal-Setup (gegen-gap OD -> spaeter reversal)
  4. Gap-Fill-Timing als Signal
  5. "Strong Open" Setup
"""

import sys
import os
import warnings
import random

warnings.filterwarnings('ignore')

# Project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================
METADATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'metadata', 'metadata_v9.parquet')
RAW_1MIN_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw_1min')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'd13_team_analysis', 'results')
RESULTS_FILE = os.path.join(RESULTS_DIR, 'd13_trader_results.txt')

IS_START = '2021-02-21'
IS_END = '2023-12-31'

MAX_EVENTS_1MIN = 500  # Limit for analyses requiring 1-min bar loading

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Output buffer
output_lines = []


def out(text=''):
    """Print and buffer output."""
    print(text)
    output_lines.append(text)


def save_results():
    """Write all buffered output to results file."""
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    out(f'\n[Results saved to {RESULTS_FILE}]')


# ============================================================
# DATA LOADING
# ============================================================
def load_metadata():
    """Load metadata and filter to IS period."""
    df = pd.read_parquet(METADATA_PATH)
    is_mask = (df['date'] >= IS_START) & (df['date'] <= IS_END)
    return df[is_mask].copy()


def load_1min_bars(ticker, date):
    """Load 1-min bars for a given ticker/date. Returns None if not found."""
    path = os.path.join(RAW_1MIN_DIR, ticker, f'{date}.parquet')
    if not os.path.exists(path):
        return None
    try:
        bars = pd.read_parquet(path)
        return bars
    except Exception:
        return None


def get_rth_bars(bars):
    """Filter to RTH bars only (09:30 - 15:59)."""
    if bars is None or len(bars) == 0:
        return pd.DataFrame()
    rth = bars[(bars['time_et'] >= '09:30') & (bars['time_et'] <= '15:59')].copy()
    return rth


# ============================================================
# TRAIL_D IMPLEMENTATION
# ============================================================
def simulate_trail_d(bars_rth, entry_price, sl_dist, trade_dir, adr):
    """
    Simulate TRAIL_D exit strategy.
    Entry assumed at 09:35 close -> post_entry starts 09:36
    """
    if sl_dist <= 0 or pd.isna(sl_dist) or adr <= 0:
        return None

    post_entry = bars_rth[(bars_rth['time_et'] >= '09:36') & (bars_rth['time_et'] <= '15:55')]
    if len(post_entry) == 0:
        return None

    if trade_dir == 'long':
        sl_level = entry_price - sl_dist
    else:
        sl_level = entry_price + sl_dist

    highest_r = 0.0
    trail_active = False
    exit_price = None
    exit_type = 'timeout'

    for _, bar in post_entry.iterrows():
        if trade_dir == 'long':
            current_r = max(0, (bar['high'] - entry_price) / sl_dist)
        else:
            current_r = max(0, (entry_price - bar['low']) / sl_dist)

        highest_r = max(highest_r, current_r)

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

        if trade_dir == 'long' and bar['low'] <= sl_level:
            exit_price = sl_level
            exit_type = 'trail_sl' if trail_active else 'initial_sl'
            break
        elif trade_dir == 'short' and bar['high'] >= sl_level:
            exit_price = sl_level
            exit_type = 'trail_sl' if trail_active else 'initial_sl'
            break

    if exit_price is None:
        exit_price = post_entry.iloc[-1]['close']
        exit_type = 'timeout'

    if trade_dir == 'long':
        pnl = exit_price - entry_price
    else:
        pnl = entry_price - exit_price

    return {
        'pnl': pnl,
        'pnl_r': pnl / sl_dist if sl_dist > 0 else 0,
        'pnl_adr': pnl / adr if adr > 0 else 0,
        'exit_type': exit_type,
        'highest_r': highest_r,
        'trail_active': trail_active,
        'winner': pnl > 0,
    }


# ============================================================
# HELPER: Sample events if too many
# ============================================================
def sample_events(df, max_n, seed=42):
    """Sample at most max_n events, preserving reproducibility."""
    if len(df) <= max_n:
        return df
    return df.sample(n=max_n, random_state=seed)


# ============================================================
# ANALYSE 1: Erste-30-Minuten Microstructure
# ============================================================
def analyse_1_microstructure(meta):
    out('=' * 80)
    out('ANALYSE 1: Erste-30-Minuten Microstructure')
    out('=' * 80)

    qualified = meta[meta['od_strength'] > 0.5].copy()
    out(f'\nQualifizierte Events (od_strength > 0.5): N = {len(qualified)}')

    # Sample for 1-min loading
    sample = sample_events(qualified, MAX_EVENTS_1MIN)
    out(f'Sampled for 1-min analysis: N = {len(sample)}')

    results = []
    for _, row in sample.iterrows():
        bars = load_1min_bars(row['ticker'], row['date'])
        if bars is None:
            continue
        rth = get_rth_bars(bars)
        if len(rth) < 30:
            continue

        gap_dir = row['gap_direction']  # 'up' or 'down'
        adr = row['adr_10']
        if adr <= 0 or pd.isna(adr):
            continue

        # First 5 min bars (09:30 - 09:34)
        first_5 = rth[rth['time_et'] < '09:35']
        # First 15 min bars
        first_15 = rth[rth['time_et'] < '09:45']
        # First 30 min bars
        first_30 = rth[rth['time_et'] < '10:00']

        if len(first_5) == 0:
            continue

        # HOD / LOD of entire day
        day_high = rth['high'].max()
        day_low = rth['low'].min()

        # First 5-min candle high/low
        candle5_high = first_5['high'].max()
        candle5_low = first_5['low'].min()

        # Does first 5-min candle make HOD or LOD?
        hod_in_5 = abs(candle5_high - day_high) < 0.001
        lod_in_5 = abs(candle5_low - day_low) < 0.001

        # Volume distribution
        vol_5 = first_5['volume'].sum() if len(first_5) > 0 else 0
        vol_15 = first_15['volume'].sum() if len(first_15) > 0 else 0
        vol_30 = first_30['volume'].sum() if len(first_30) > 0 else 0
        vol_total = rth['volume'].sum()
        vol_pct_5 = vol_5 / vol_total * 100 if vol_total > 0 else 0
        vol_pct_15 = vol_15 / vol_total * 100 if vol_total > 0 else 0
        vol_pct_30 = vol_30 / vol_total * 100 if vol_total > 0 else 0

        # Pullback after OD: max adverse move in first 30 min
        open_price = rth.iloc[0]['open']
        if gap_dir == 'up':
            # For gap-up, OD with_gap goes higher -> adverse = pullback down
            max_high_30 = first_30['high'].max()
            max_adverse_30 = max_high_30 - first_30['low'].min()
            # Pullback from highest point in first 30 min
            pullback_from_high = max_high_30 - first_30.iloc[-1]['close'] if len(first_30) > 0 else 0
            pullback_depth = (open_price - first_30['low'].min()) / adr if adr > 0 else 0
        else:
            # For gap-down, OD with_gap goes lower -> adverse = pullback up
            min_low_30 = first_30['low'].min()
            max_adverse_30 = first_30['high'].max() - min_low_30
            pullback_from_low = first_30.iloc[-1]['close'] - min_low_30 if len(first_30) > 0 else 0
            pullback_depth = (first_30['high'].max() - open_price) / adr if adr > 0 else 0

        results.append({
            'ticker': row['ticker'],
            'date': row['date'],
            'gap_dir': gap_dir,
            'od_dir': row['od_direction'],
            'hod_in_5': hod_in_5,
            'lod_in_5': lod_in_5,
            'vol_pct_5': vol_pct_5,
            'vol_pct_15': vol_pct_15,
            'vol_pct_30': vol_pct_30,
            'pullback_depth': pullback_depth,
            'adr': adr,
        })

    rdf = pd.DataFrame(results)
    out(f'\nErfolgreich geladen: N = {len(rdf)}')

    if len(rdf) == 0:
        out('KEINE DATEN - Analyse abgebrochen.')
        return

    # --- HOD/LOD in first 5 min ---
    out('\n--- Erste 5-Min-Kerze = HOD oder LOD? ---')
    gap_up = rdf[rdf['gap_dir'] == 'up']
    gap_dn = rdf[rdf['gap_dir'] == 'down']
    with_gap = rdf[rdf['od_dir'] == 'with_gap']
    against_gap = rdf[rdf['od_dir'] == 'against_gap']

    out(f'  Gesamt: HOD in 5min = {rdf["hod_in_5"].mean()*100:.1f}% | LOD in 5min = {rdf["lod_in_5"].mean()*100:.1f}%')
    if len(gap_up) > 0:
        out(f'  GapUp:  HOD in 5min = {gap_up["hod_in_5"].mean()*100:.1f}% | LOD in 5min = {gap_up["lod_in_5"].mean()*100:.1f}%')
    if len(gap_dn) > 0:
        out(f'  GapDn:  HOD in 5min = {gap_dn["hod_in_5"].mean()*100:.1f}% | LOD in 5min = {gap_dn["lod_in_5"].mean()*100:.1f}%')
    if len(with_gap) > 0:
        out(f'  WithGap:    HOD in 5min = {with_gap["hod_in_5"].mean()*100:.1f}% | LOD in 5min = {with_gap["lod_in_5"].mean()*100:.1f}%')
    if len(against_gap) > 0:
        out(f'  AgainstGap: HOD in 5min = {against_gap["hod_in_5"].mean()*100:.1f}% | LOD in 5min = {against_gap["lod_in_5"].mean()*100:.1f}%')

    # --- Volume Distribution ---
    out('\n--- Volume-Distribution (% des Tagesvolumens) ---')
    out(f'  Erste  5 Min: Mean = {rdf["vol_pct_5"].mean():.1f}% | Median = {rdf["vol_pct_5"].median():.1f}%')
    out(f'  Erste 15 Min: Mean = {rdf["vol_pct_15"].mean():.1f}% | Median = {rdf["vol_pct_15"].median():.1f}%')
    out(f'  Erste 30 Min: Mean = {rdf["vol_pct_30"].mean():.1f}% | Median = {rdf["vol_pct_30"].median():.1f}%')

    # Split by OD direction
    for label, sub in [('WithGap', with_gap), ('AgainstGap', against_gap)]:
        if len(sub) > 0:
            out(f'  [{label}] 5min={sub["vol_pct_5"].mean():.1f}% | 15min={sub["vol_pct_15"].mean():.1f}% | 30min={sub["vol_pct_30"].mean():.1f}%')

    # --- Pullback Depth ---
    out('\n--- Pullback-Tiefe nach OD (pullback_depth = max adverse first 30min / ADR) ---')
    out(f'  Gesamt:     Mean = {rdf["pullback_depth"].mean():.3f} ADR | Median = {rdf["pullback_depth"].median():.3f} ADR')
    out(f'  P25 = {rdf["pullback_depth"].quantile(0.25):.3f} | P75 = {rdf["pullback_depth"].quantile(0.75):.3f} | P90 = {rdf["pullback_depth"].quantile(0.90):.3f}')
    if len(with_gap) > 0:
        out(f'  WithGap:    Mean = {with_gap["pullback_depth"].mean():.3f} | Median = {with_gap["pullback_depth"].median():.3f}')
    if len(against_gap) > 0:
        out(f'  AgainstGap: Mean = {against_gap["pullback_depth"].mean():.3f} | Median = {against_gap["pullback_depth"].median():.3f}')

    out()


# ============================================================
# ANALYSE 2: Time-of-Day PnL-Kurve
# ============================================================
def analyse_2_tod_pnl(meta):
    out('=' * 80)
    out('ANALYSE 2: Time-of-Day PnL-Kurve')
    out('=' * 80)

    qualified = meta[(meta['od_strength'] > 0.5)].copy()
    out(f'\nQualifizierte Events (od_strength > 0.5): N = {len(qualified)}')

    sample = sample_events(qualified, MAX_EVENTS_1MIN)
    out(f'Sampled: N = {len(sample)}')

    exit_times = ['10:00', '10:30', '11:00', '12:00', '14:00', '15:55']
    results = []

    for _, row in sample.iterrows():
        bars = load_1min_bars(row['ticker'], row['date'])
        if bars is None:
            continue
        rth = get_rth_bars(bars)
        if len(rth) < 30:
            continue

        gap_dir = row['gap_direction']
        od_dir = row['od_direction']
        adr = row['adr_10']
        if adr <= 0 or pd.isna(adr):
            continue

        # Entry at close of 09:35 bar
        entry_bars = rth[rth['time_et'] == '09:35']
        if len(entry_bars) == 0:
            # Try 09:34 or 09:36 as fallback
            entry_bars = rth[(rth['time_et'] >= '09:34') & (rth['time_et'] <= '09:36')]
            if len(entry_bars) == 0:
                continue

        entry_price = entry_bars.iloc[0]['close']

        # Trade direction: with_gap -> long for gap_up, short for gap_down
        if od_dir == 'with_gap':
            trade_dir = 'long' if gap_dir == 'up' else 'short'
        else:
            trade_dir = 'short' if gap_dir == 'up' else 'long'

        row_result = {
            'ticker': row['ticker'],
            'date': row['date'],
            'gap_dir': gap_dir,
            'od_dir': od_dir,
            'trade_dir': trade_dir,
            'entry_price': entry_price,
            'adr': adr,
        }

        # PnL at each exit time
        for et in exit_times:
            exit_bars = rth[rth['time_et'] == et]
            if len(exit_bars) == 0:
                # Find nearest bar
                exit_bars = rth[rth['time_et'] <= et]
                if len(exit_bars) == 0:
                    row_result[f'pnl_{et}'] = np.nan
                    continue
                exit_bar = exit_bars.iloc[-1]
            else:
                exit_bar = exit_bars.iloc[0]

            exit_price = exit_bar['close']
            if trade_dir == 'long':
                pnl = (exit_price - entry_price) / adr
            else:
                pnl = (entry_price - exit_price) / adr

            row_result[f'pnl_{et}'] = pnl

        results.append(row_result)

    rdf = pd.DataFrame(results)
    out(f'Erfolgreich geladen: N = {len(rdf)}')

    if len(rdf) == 0:
        out('KEINE DATEN.')
        return

    # Overall PnL curve
    out('\n--- PnL-Kurve (Mean PnL in ADR) ---')
    out(f'{"Exit Time":<12} {"N":>5} {"Mean PnL":>10} {"Median PnL":>12} {"WinRate":>8} {"Std":>8}')
    out('-' * 60)
    for et in exit_times:
        col = f'pnl_{et}'
        valid = rdf[col].dropna()
        if len(valid) > 0:
            out(f'{et:<12} {len(valid):>5} {valid.mean():>+10.4f} {valid.median():>+12.4f} {(valid>0).mean()*100:>7.1f}% {valid.std():>8.4f}')

    # Split by GapUp vs GapDn
    for label, sub in [('GapUp', rdf[rdf['gap_dir'] == 'up']),
                       ('GapDn', rdf[rdf['gap_dir'] == 'down'])]:
        out(f'\n--- PnL-Kurve: {label} (N={len(sub)}) ---')
        out(f'{"Exit Time":<12} {"N":>5} {"Mean PnL":>10} {"Median PnL":>12} {"WinRate":>8}')
        out('-' * 52)
        for et in exit_times:
            col = f'pnl_{et}'
            valid = sub[col].dropna()
            if len(valid) > 0:
                out(f'{et:<12} {len(valid):>5} {valid.mean():>+10.4f} {valid.median():>+12.4f} {(valid>0).mean()*100:>7.1f}%')

    # Split by OD direction
    for label, sub in [('WithGap', rdf[rdf['od_dir'] == 'with_gap']),
                       ('AgainstGap', rdf[rdf['od_dir'] == 'against_gap'])]:
        out(f'\n--- PnL-Kurve: {label} (N={len(sub)}) ---')
        out(f'{"Exit Time":<12} {"N":>5} {"Mean PnL":>10} {"Median PnL":>12} {"WinRate":>8}')
        out('-' * 52)
        for et in exit_times:
            col = f'pnl_{et}'
            valid = sub[col].dropna()
            if len(valid) > 0:
                out(f'{et:<12} {len(valid):>5} {valid.mean():>+10.4f} {valid.median():>+12.4f} {(valid>0).mean()*100:>7.1f}%')

    out()


# ============================================================
# ANALYSE 3: Reversal-Setup (kreativ)
# ============================================================
def analyse_3_reversal(meta):
    out('=' * 80)
    out('ANALYSE 3: Reversal-Setup (OD gegen Gap -> spaeter Reversal in Gap-Richtung)')
    out('=' * 80)

    # Filter: against_gap, od_strength > 0.5
    qualified = meta[(meta['od_direction'] == 'against_gap') & (meta['od_strength'] > 0.5)].copy()
    out(f'\nAgainst-Gap OD > 0.5: N = {len(qualified)}')

    sample = sample_events(qualified, MAX_EVENTS_1MIN)
    out(f'Sampled: N = {len(sample)}')

    results = []

    for _, row in sample.iterrows():
        bars = load_1min_bars(row['ticker'], row['date'])
        if bars is None:
            continue
        rth = get_rth_bars(bars)
        if len(rth) < 60:
            continue

        gap_dir = row['gap_direction']
        adr = row['adr_10']
        if adr <= 0 or pd.isna(adr):
            continue

        # Gap direction determines which way we'd want a reversal
        # If gap_up + OD against_gap (down) -> we want a reversal back UP
        # If gap_down + OD against_gap (up) -> we want a reversal back DOWN
        gap_trade_dir = 'long' if gap_dir == 'up' else 'short'

        open_price = rth.iloc[0]['open']

        # After 10:00 bars
        after_10 = rth[rth['time_et'] >= '10:00']
        if len(after_10) == 0:
            continue

        # Price at 10:00
        bar_10 = rth[rth['time_et'] == '10:00']
        price_at_10 = bar_10.iloc[0]['close'] if len(bar_10) > 0 else after_10.iloc[0]['open']

        # Check: Does price reverse back in gap direction after 10:00?
        # For gap_up: did price go above open_price after 10:00?
        # For gap_down: did price go below open_price after 10:00?
        if gap_dir == 'up':
            reversal_back = after_10['high'].max() > open_price
            reversal_above_935 = after_10['high'].max() > row['close_935'] if not pd.isna(row['close_935']) else False
        else:
            reversal_back = after_10['low'].min() < open_price
            reversal_above_935 = after_10['low'].min() < row['close_935'] if not pd.isna(row['close_935']) else False

        # VWAP cross detection after 10:00
        # Find the VWAP column - use the vwap from the bars
        vwap_cross_time = None
        entry_price_vwap = None

        for idx, bar in after_10.iterrows():
            # Check if price crosses VWAP in gap direction
            if 'vwap' not in bar or pd.isna(bar['vwap']):
                continue
            vwap_val = bar['vwap']
            if gap_dir == 'up' and bar['close'] > vwap_val and bar['open'] <= vwap_val:
                vwap_cross_time = bar['time_et']
                entry_price_vwap = bar['close']
                break
            elif gap_dir == 'down' and bar['close'] < vwap_val and bar['open'] >= vwap_val:
                vwap_cross_time = bar['time_et']
                entry_price_vwap = bar['close']
                break

        # Simulate Late Reversal trade if VWAP cross found
        trade_result = None
        if entry_price_vwap is not None and vwap_cross_time is not None:
            sl_dist = 0.25 * adr
            # Get bars after VWAP cross
            post_cross = rth[(rth['time_et'] > vwap_cross_time) & (rth['time_et'] <= '15:55')]
            if len(post_cross) > 0:
                trade_result = simulate_trail_d(post_cross, entry_price_vwap, sl_dist, gap_trade_dir, adr)

        results.append({
            'ticker': row['ticker'],
            'date': row['date'],
            'gap_dir': gap_dir,
            'od_strength': row['od_strength'],
            'vwap_z_10am': row['vwap_z_at_10am'],
            'reversal_back': reversal_back,
            'reversal_above_935': reversal_above_935,
            'vwap_cross_found': vwap_cross_time is not None,
            'vwap_cross_time': vwap_cross_time,
            'trade_pnl_adr': trade_result['pnl_adr'] if trade_result else np.nan,
            'trade_winner': trade_result['winner'] if trade_result else np.nan,
            'trade_exit_type': trade_result['exit_type'] if trade_result else None,
            'adr': adr,
        })

    rdf = pd.DataFrame(results)
    out(f'Erfolgreich geladen: N = {len(rdf)}')

    if len(rdf) == 0:
        out('KEINE DATEN.')
        return

    # Reversal statistics
    out('\n--- Reversal-Statistik (OD gegen Gap, od>0.5) ---')
    out(f'  Reversal zurueck ueber Open nach 10:00: {rdf["reversal_back"].mean()*100:.1f}% ({rdf["reversal_back"].sum()}/{len(rdf)})')
    out(f'  Reversal zurueck ueber close_935:       {rdf["reversal_above_935"].mean()*100:.1f}% ({rdf["reversal_above_935"].sum()}/{len(rdf)})')
    out(f'  VWAP-Cross in Gap-Richtung gefunden:    {rdf["vwap_cross_found"].mean()*100:.1f}% ({rdf["vwap_cross_found"].sum()}/{len(rdf)})')

    # Late Reversal Setup results
    trades = rdf[rdf['vwap_cross_found'] == True].copy()
    if len(trades) > 0:
        valid_trades = trades.dropna(subset=['trade_pnl_adr'])
        out(f'\n--- Late Reversal Setup (VWAP-Cross nach 10:00 in Gap-Richtung) ---')
        out(f'  Trades: N = {len(valid_trades)}')
        if len(valid_trades) > 0:
            out(f'  Mean PnL:   {valid_trades["trade_pnl_adr"].mean():+.4f} ADR')
            out(f'  Median PnL: {valid_trades["trade_pnl_adr"].median():+.4f} ADR')
            out(f'  WinRate:    {valid_trades["trade_winner"].mean()*100:.1f}%')
            out(f'  Std:        {valid_trades["trade_pnl_adr"].std():.4f} ADR')

            # Exit types
            if 'trade_exit_type' in valid_trades.columns:
                exit_types = valid_trades['trade_exit_type'].value_counts()
                out(f'  Exit Types: {exit_types.to_dict()}')

            # Split by overextension (vwap_z_at_10am)
            out('\n  --- Gesplittet nach VWAP-Z at 10am (Ueberextension) ---')
            for z_label, z_min, z_max in [('z < -1.5', -999, -1.5), ('-1.5 <= z < -0.5', -1.5, -0.5),
                                           ('-0.5 <= z < 0.5', -0.5, 0.5), ('0.5 <= z < 1.5', 0.5, 1.5),
                                           ('z >= 1.5', 1.5, 999)]:
                z_sub = valid_trades[(valid_trades['vwap_z_10am'] >= z_min) & (valid_trades['vwap_z_10am'] < z_max)]
                if len(z_sub) >= 5:
                    out(f'    {z_label:20s}: N={len(z_sub):3d} | PnL={z_sub["trade_pnl_adr"].mean():+.4f} | WR={z_sub["trade_winner"].mean()*100:.1f}%')
    else:
        out('\n  Keine VWAP-Cross Trades gefunden.')

    out()


# ============================================================
# ANALYSE 4: Gap-Fill-Timing als Signal
# ============================================================
def analyse_4_gap_fill(meta):
    out('=' * 80)
    out('ANALYSE 4: Gap-Fill-Timing als Signal')
    out('=' * 80)

    filled = meta[meta['gap_filled'] == True].copy()
    out(f'\nGap-Fill Events in IS: N = {len(filled)}')

    sample = sample_events(filled, MAX_EVENTS_1MIN)
    out(f'Sampled: N = {len(sample)}')

    results = []

    for _, row in sample.iterrows():
        bars = load_1min_bars(row['ticker'], row['date'])
        if bars is None:
            continue
        rth = get_rth_bars(bars)
        if len(rth) < 30:
            continue

        gap_dir = row['gap_direction']
        adr = row['adr_10']
        prev_close = row['prev_close']
        fill_time_min = row['gap_fill_time_minutes']

        if adr <= 0 or pd.isna(adr) or pd.isna(prev_close) or pd.isna(fill_time_min):
            continue

        # Find the bar at gap-fill time
        fill_time_idx = int(fill_time_min)
        # Fill time is minutes after open (09:30)
        fill_hour = 9 + (30 + fill_time_idx) // 60
        fill_minute = (30 + fill_time_idx) % 60
        fill_time_str = f'{fill_hour:02d}:{fill_minute:02d}'

        fill_bar = rth[rth['time_et'] == fill_time_str]
        if len(fill_bar) == 0:
            # Find nearest bar
            fill_bar = rth[rth['time_et'] <= fill_time_str]
            if len(fill_bar) == 0:
                continue
            fill_bar = fill_bar.iloc[-1:]

        price_at_fill = fill_bar.iloc[0]['close']

        # After fill bars
        after_fill = rth[rth['time_et'] > fill_time_str]
        if len(after_fill) == 0:
            continue

        close_price = rth.iloc[-1]['close']

        # PnL after fill: does price bounce at fill level?
        # For gap_up: fill = price came back to prev_close -> bounce would be UP
        # For gap_down: fill = price came back up to prev_close -> bounce would be DOWN
        if gap_dir == 'up':
            # Gap-up fill means price pulled back to prev_close
            # Bounce = price goes back up from here
            pnl_after_fill = (close_price - price_at_fill) / adr
            # Max bounce after fill
            max_bounce = (after_fill['high'].max() - price_at_fill) / adr
            # Max adverse after fill
            max_adverse = (price_at_fill - after_fill['low'].min()) / adr
        else:
            # Gap-down fill means price rallied back to prev_close
            # Bounce = price goes back down from here
            pnl_after_fill = (price_at_fill - close_price) / adr
            max_bounce = (price_at_fill - after_fill['low'].min()) / adr
            max_adverse = (after_fill['high'].max() - price_at_fill) / adr

        fill_speed = 'fast' if fill_time_min <= 30 else ('slow' if fill_time_min > 60 else 'medium')

        results.append({
            'ticker': row['ticker'],
            'date': row['date'],
            'gap_dir': gap_dir,
            'fill_time_min': fill_time_min,
            'fill_speed': fill_speed,
            'pnl_after_fill': pnl_after_fill,
            'max_bounce': max_bounce,
            'max_adverse': max_adverse,
            'od_strength': row['od_strength'],
            'adr': adr,
        })

    rdf = pd.DataFrame(results)
    out(f'Erfolgreich geladen: N = {len(rdf)}')

    if len(rdf) == 0:
        out('KEINE DATEN.')
        return

    # Overall
    out(f'\n--- PnL nach Gap-Fill (in Gap-Richtung = positiv) ---')
    out(f'  Gesamt: N={len(rdf)} | Mean PnL={rdf["pnl_after_fill"].mean():+.4f} ADR | Median={rdf["pnl_after_fill"].median():+.4f} | WR={(rdf["pnl_after_fill"]>0).mean()*100:.1f}%')

    # By fill speed
    out('\n--- Nach Fill-Geschwindigkeit ---')
    out(f'{"Speed":<10} {"N":>5} {"Mean PnL":>10} {"Median":>10} {"WinRate":>8} {"MaxBounce":>10} {"MaxAdverse":>10}')
    out('-' * 70)
    for speed in ['fast', 'medium', 'slow']:
        sub = rdf[rdf['fill_speed'] == speed]
        if len(sub) > 0:
            out(f'{speed:<10} {len(sub):>5} {sub["pnl_after_fill"].mean():>+10.4f} {sub["pnl_after_fill"].median():>+10.4f} {(sub["pnl_after_fill"]>0).mean()*100:>7.1f}% {sub["max_bounce"].mean():>+10.4f} {sub["max_adverse"].mean():>10.4f}')

    # By gap direction
    out('\n--- Nach Gap-Richtung ---')
    for label, sub in [('GapUp', rdf[rdf['gap_dir'] == 'up']), ('GapDn', rdf[rdf['gap_dir'] == 'down'])]:
        if len(sub) > 0:
            out(f'  {label}: N={len(sub)} | Mean PnL={sub["pnl_after_fill"].mean():+.4f} | Median={sub["pnl_after_fill"].median():+.4f} | WR={(sub["pnl_after_fill"]>0).mean()*100:.1f}%')

    # Hypothesis test: fast fills bounce better?
    fast = rdf[rdf['fill_speed'] == 'fast']
    slow = rdf[rdf['fill_speed'] == 'slow']
    if len(fast) >= 30 and len(slow) >= 30:
        out(f'\n  Hypothese: Schnelle Fills (<30min) bouncen staerker als langsame (>60min)')
        out(f'    Fast Fill: Mean PnL = {fast["pnl_after_fill"].mean():+.4f} ADR, Max Bounce = {fast["max_bounce"].mean():.4f}')
        out(f'    Slow Fill: Mean PnL = {slow["pnl_after_fill"].mean():+.4f} ADR, Max Bounce = {slow["max_bounce"].mean():.4f}')
        diff = fast['pnl_after_fill'].mean() - slow['pnl_after_fill'].mean()
        out(f'    Differenz: {diff:+.4f} ADR -> {"Fast fills bouncen besser" if diff > 0 else "Slow fills bouncen besser"}')

    out()


# ============================================================
# ANALYSE 5: "Strong Open" Setup
# ============================================================
def analyse_5_strong_open(meta):
    out('=' * 80)
    out('ANALYSE 5: "Strong Open" Setup')
    out('=' * 80)

    # Calculate medians for filtering
    median_candle_size = meta['first_candle_size'].median()
    out(f'\nMedian first_candle_size: {median_candle_size:.4f}')

    # Strong Open criteria
    strong_open = meta[
        (meta['first_candle_size'] > median_candle_size) &
        (meta['od_body_pct'] > 0.7) &
        (meta['od_direction'] == 'with_gap') &
        (meta['rvol_5'] > 3)
    ].copy()
    out(f'Strong Open Events: N = {len(strong_open)}')

    # Reference: all od_strength > 0.5
    reference = meta[meta['od_strength'] > 0.5].copy()
    out(f'Reference (od_strength > 0.5): N = {len(reference)}')

    # Also check overlap
    strong_with_od = strong_open[strong_open['od_strength'] > 0.5]
    out(f'Strong Open + od_strength > 0.5: N = {len(strong_with_od)}')

    # Backtest both with TRAIL_D
    def backtest_group(group_df, label, max_n=MAX_EVENTS_1MIN):
        out(f'\n--- Backtest: {label} ---')
        sample = sample_events(group_df, max_n)
        trades = []

        for _, row in sample.iterrows():
            bars = load_1min_bars(row['ticker'], row['date'])
            if bars is None:
                continue
            rth = get_rth_bars(bars)
            if len(rth) < 30:
                continue

            adr = row['adr_10']
            if adr <= 0 or pd.isna(adr):
                continue

            # Entry at close_935
            entry_bars = rth[rth['time_et'] == '09:35']
            if len(entry_bars) == 0:
                entry_bars = rth[(rth['time_et'] >= '09:34') & (rth['time_et'] <= '09:36')]
                if len(entry_bars) == 0:
                    continue
            entry_price = entry_bars.iloc[0]['close']

            # Trade direction based on gap
            gap_dir = row['gap_direction']
            od_dir = row.get('od_direction', 'with_gap')
            if od_dir == 'with_gap':
                trade_dir = 'long' if gap_dir == 'up' else 'short'
            else:
                trade_dir = 'short' if gap_dir == 'up' else 'long'

            sl_dist = 0.25 * adr
            result = simulate_trail_d(rth, entry_price, sl_dist, trade_dir, adr)
            if result:
                result['ticker'] = row['ticker']
                result['date'] = row['date']
                result['gap_dir'] = gap_dir
                result['od_strength'] = row['od_strength']
                trades.append(result)

        tdf = pd.DataFrame(trades)
        if len(tdf) == 0:
            out(f'  KEINE TRADES geladen.')
            return None

        out(f'  Trades geladen: N = {len(tdf)}')
        out(f'  Mean PnL:     {tdf["pnl_adr"].mean():+.4f} ADR')
        out(f'  Median PnL:   {tdf["pnl_adr"].median():+.4f} ADR')
        out(f'  WinRate:      {tdf["winner"].mean()*100:.1f}%')
        out(f'  Mean PnL (R): {tdf["pnl_r"].mean():+.3f} R')
        out(f'  Std PnL:      {tdf["pnl_adr"].std():.4f} ADR')
        out(f'  Profit Factor: {tdf[tdf["pnl_adr"]>0]["pnl_adr"].sum() / abs(tdf[tdf["pnl_adr"]<0]["pnl_adr"].sum()):.2f}' if tdf[tdf["pnl_adr"]<0]["pnl_adr"].sum() != 0 else '  Profit Factor: inf')

        # Exit types
        out(f'  Exit Types: {tdf["exit_type"].value_counts().to_dict()}')

        # Expectancy
        avg_win = tdf[tdf['winner']]['pnl_adr'].mean() if tdf['winner'].sum() > 0 else 0
        avg_loss = tdf[~tdf['winner']]['pnl_adr'].mean() if (~tdf['winner']).sum() > 0 else 0
        wr = tdf['winner'].mean()
        out(f'  Avg Win:  {avg_win:+.4f} ADR')
        out(f'  Avg Loss: {avg_loss:+.4f} ADR')
        out(f'  Expectancy: {wr * avg_win + (1-wr) * avg_loss:+.4f} ADR')

        return tdf

    # Backtest Strong Open
    strong_trades = backtest_group(strong_open, 'Strong Open (big candle + body>0.7 + with_gap + rvol>3)')

    # Backtest Reference
    ref_trades = backtest_group(reference, 'Reference (od_strength > 0.5)')

    # Compare
    if strong_trades is not None and ref_trades is not None and len(strong_trades) >= 30:
        out('\n--- Vergleich: Strong Open vs Reference ---')
        out(f'  Strong Open: N={len(strong_trades)} | EV={strong_trades["pnl_adr"].mean():+.4f} ADR | WR={strong_trades["winner"].mean()*100:.1f}%')
        out(f'  Reference:   N={len(ref_trades)} | EV={ref_trades["pnl_adr"].mean():+.4f} ADR | WR={ref_trades["winner"].mean()*100:.1f}%')
        diff = strong_trades['pnl_adr'].mean() - ref_trades['pnl_adr'].mean()
        out(f'  Differenz:   {diff:+.4f} ADR -> {"Strong Open besser" if diff > 0 else "Reference besser"}')

    out()


# ============================================================
# MAIN
# ============================================================
def main():
    out('=' * 80)
    out('D13 DISCRETIONARY TRADER ANALYSIS')
    out(f'IS-Periode: {IS_START} bis {IS_END}')
    out(f'Max 1-Min Events pro Analyse: {MAX_EVENTS_1MIN}')
    out('=' * 80)

    meta = load_metadata()
    out(f'\nIS Metadata geladen: {len(meta)} Events')
    out(f'  GapUp: {(meta["gap_direction"]=="up").sum()} | GapDn: {(meta["gap_direction"]=="down").sum()}')
    out(f'  od_strength > 0.5: {(meta["od_strength"]>0.5).sum()}')
    out(f'  gap_filled: {meta["gap_filled"].sum()}')
    out(f'  od_direction with_gap: {(meta["od_direction"]=="with_gap").sum()} | against_gap: {(meta["od_direction"]=="against_gap").sum()}')
    out()

    # Run all analyses
    analyse_1_microstructure(meta)
    analyse_2_tod_pnl(meta)
    analyse_3_reversal(meta)
    analyse_4_gap_fill(meta)
    analyse_5_strong_open(meta)

    # Summary
    out('=' * 80)
    out('ZUSAMMENFASSUNG')
    out('=' * 80)
    out("""
Die 5 Analysen untersuchen verschiedene Aspekte der Intraday-Preisaktion
fuer Gap-Events mit starkem Opening Drive (od_strength > 0.5):

1. MICROSTRUCTURE: Volume-Frontloading und Pullback-Tiefe in den ersten 30 Min
2. TIME-OF-DAY: Wann ist der optimale Ausstiegszeitpunkt?
3. REVERSAL: Kann man gegen-Gap ODs spaeter in Gap-Richtung traden?
4. GAP-FILL: Bounced der Preis nach einem Gap-Fill?
5. STRONG OPEN: Verbessern starke Eroeffnungskerzen den EV?

Siehe oben fuer detaillierte Ergebnisse jeder Analyse.
""")

    save_results()


if __name__ == '__main__':
    main()
