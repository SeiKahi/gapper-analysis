"""
D14 Engine: Bias-Free TRAIL_D Streaming Simulation
===================================================
CHANGES vs D10/D13:
  1. Conservative bar ordering: SL check BEFORE peak update (D13 did peak first = optimistic)
  2. Flexible entry time: Not hardcoded 09:36, accepts any entry_time
  3. Intra-bar resolution: When ambiguous bar hits both SL and new peak -> conservative (SL first)
     or 1-sec zoom if available
"""
import pandas as pd
import numpy as np
from pathlib import Path


SEC_DIR = Path('data/raw_1sec')


def resolve_intrabar_1sec(ticker, date, bar_time, entry_price, sl_level,
                          trade_dir, sl_dist, trail_active, highest_r):
    """
    Resolve ambiguous 1-min bar using 1-sec data.
    Returns: ('sl_hit', exit_price) or ('peak_update', new_highest_r) or ('no_data', None)
    """
    sec_path = SEC_DIR / ticker / f"{date}.parquet"
    if not sec_path.exists():
        return 'no_data', None

    try:
        sec_df = pd.read_parquet(sec_path)
    except Exception:
        return 'no_data', None

    # Filter to the specific minute
    # time_et in 1sec is "HH:MM:SS", bar_time from 1min is "HH:MM"
    sec_bars = sec_df[sec_df['time_et'].str.startswith(bar_time)]
    if len(sec_bars) == 0:
        return 'no_data', None

    # Replay second by second
    for _, sec in sec_bars.iterrows():
        # SL check first (conservative)
        if trade_dir == 'long' and sec['low'] <= sl_level:
            return 'sl_hit', sl_level
        elif trade_dir == 'short' and sec['high'] >= sl_level:
            return 'sl_hit', sl_level

        # Then update peak
        if trade_dir == 'long':
            new_r = max(0, (sec['high'] - entry_price) / sl_dist)
        else:
            new_r = max(0, (entry_price - sec['low']) / sl_dist)
        highest_r = max(highest_r, new_r)

        # Update trail after peak update
        if highest_r >= 1.0 and trail_active:
            if trade_dir == 'long':
                new_sl = entry_price + (highest_r - 0.5) * sl_dist
                sl_level = max(sl_level, new_sl)
            else:
                new_sl = entry_price - (highest_r - 0.5) * sl_dist
                sl_level = min(sl_level, new_sl)

        # Re-check SL after trail update
        if trade_dir == 'long' and sec['low'] <= sl_level:
            return 'sl_hit', sl_level
        elif trade_dir == 'short' and sec['high'] >= sl_level:
            return 'sl_hit', sl_level

    return 'peak_update', highest_r


def is_bar_ambiguous(bar, sl_level, entry_price, sl_dist, trade_dir, highest_r):
    """Check if a bar could hit both SL and a new peak (ambiguous ordering)."""
    if trade_dir == 'long':
        hits_sl = bar['low'] <= sl_level
        new_r = max(0, (bar['high'] - entry_price) / sl_dist)
        makes_new_peak = new_r > highest_r and new_r >= 1.0
    else:
        hits_sl = bar['high'] >= sl_level
        new_r = max(0, (entry_price - bar['low']) / sl_dist)
        makes_new_peak = new_r > highest_r and new_r >= 1.0
    return hits_sl and makes_new_peak


def simulate_trail_d_streaming(bars_rth, entry_price, sl_dist, trade_dir, adr,
                                entry_time='09:36', ticker=None, date=None,
                                use_1sec=True):
    """
    Bias-free TRAIL_D simulation with streaming bar processing.

    CRITICAL DIFFERENCE vs D10:
    - D10: Peak update THEN SL check (optimistic: trail tightens before checking if hit)
    - D14: SL check FIRST, then peak update (conservative: no future knowledge)

    Parameters:
        bars_rth: DataFrame with time_et, open, high, low, close, volume
        entry_price: Entry price
        sl_dist: Stop-loss distance in price
        trade_dir: 'long' or 'short'
        adr: Average daily range
        entry_time: First bar to process (string 'HH:MM')
        ticker: For 1-sec data lookup
        date: For 1-sec data lookup
        use_1sec: Whether to attempt 1-sec resolution for ambiguous bars

    Returns:
        dict with pnl_r, pnl_adr, exit_type, exit_time, highest_r, trail_active, winner
        + stats: ambig_count, zoom_used, zoom_resolved
    """
    if sl_dist <= 0 or pd.isna(sl_dist) or adr <= 0:
        return None

    post_entry = bars_rth[(bars_rth['time_et'] >= entry_time) &
                          (bars_rth['time_et'] <= '15:55')]
    if len(post_entry) == 0:
        return None

    # Initial SL
    if trade_dir == 'long':
        sl_level = entry_price - sl_dist
    else:
        sl_level = entry_price + sl_dist

    highest_r = 0.0
    trail_active = False
    exit_price = None
    exit_type = 'timeout'
    exit_time = None

    # Ambiguity tracking stats
    ambig_count = 0
    zoom_used = 0
    zoom_resolved = 0

    for _, bar in post_entry.iterrows():
        # === STEP 1: CHECK SL FIRST (conservative) ===
        if trade_dir == 'long' and bar['low'] <= sl_level:
            # Check ambiguity: could this bar also make a new peak?
            if is_bar_ambiguous(bar, sl_level, entry_price, sl_dist, trade_dir, highest_r):
                ambig_count += 1
                if use_1sec and ticker and date:
                    resolution, value = resolve_intrabar_1sec(
                        ticker, date, bar['time_et'],
                        entry_price, sl_level, trade_dir, sl_dist,
                        trail_active, highest_r)
                    if resolution == 'sl_hit':
                        zoom_used += 1
                        zoom_resolved += 1
                        exit_price = value
                        exit_type = 'trail_sl' if trail_active else 'initial_sl'
                        exit_time = bar['time_et']
                        break
                    elif resolution == 'peak_update':
                        zoom_used += 1
                        zoom_resolved += 1
                        highest_r = value
                        # Trail may have been updated in 1sec replay, recalculate
                        if highest_r >= 1.0:
                            trail_active = True
                        if trail_active:
                            new_sl = entry_price + (highest_r - 0.5) * sl_dist
                            sl_level = max(sl_level, new_sl)
                        # Re-check SL after update
                        if bar['low'] <= sl_level:
                            exit_price = sl_level
                            exit_type = 'trail_sl' if trail_active else 'initial_sl'
                            exit_time = bar['time_et']
                            break
                        continue
                    else:
                        zoom_used += 0  # no_data
                        # Fall through: conservative = SL first
            # SL hit (no ambiguity or conservative default)
            exit_price = sl_level
            exit_type = 'trail_sl' if trail_active else 'initial_sl'
            exit_time = bar['time_et']
            break

        elif trade_dir == 'short' and bar['high'] >= sl_level:
            if is_bar_ambiguous(bar, sl_level, entry_price, sl_dist, trade_dir, highest_r):
                ambig_count += 1
                if use_1sec and ticker and date:
                    resolution, value = resolve_intrabar_1sec(
                        ticker, date, bar['time_et'],
                        entry_price, sl_level, trade_dir, sl_dist,
                        trail_active, highest_r)
                    if resolution == 'sl_hit':
                        zoom_used += 1
                        zoom_resolved += 1
                        exit_price = value
                        exit_type = 'trail_sl' if trail_active else 'initial_sl'
                        exit_time = bar['time_et']
                        break
                    elif resolution == 'peak_update':
                        zoom_used += 1
                        zoom_resolved += 1
                        highest_r = value
                        if highest_r >= 1.0:
                            trail_active = True
                        if trail_active:
                            new_sl = entry_price - (highest_r - 0.5) * sl_dist
                            sl_level = min(sl_level, new_sl)
                        if bar['high'] >= sl_level:
                            exit_price = sl_level
                            exit_type = 'trail_sl' if trail_active else 'initial_sl'
                            exit_time = bar['time_et']
                            break
                        continue
                    else:
                        pass  # conservative: SL first
            exit_price = sl_level
            exit_type = 'trail_sl' if trail_active else 'initial_sl'
            exit_time = bar['time_et']
            break

        # === STEP 2: UPDATE PEAK (only if SL not hit) ===
        if trade_dir == 'long':
            current_r = max(0, (bar['high'] - entry_price) / sl_dist)
        else:
            current_r = max(0, (entry_price - bar['low']) / sl_dist)
        highest_r = max(highest_r, current_r)

        # === STEP 3: TRAIL ACTIVATION & LEVEL UPDATE ===
        if highest_r >= 1.0 and not trail_active:
            trail_active = True
            if trade_dir == 'long':
                sl_level = entry_price  # BE
            else:
                sl_level = entry_price  # BE

        if trail_active:
            if trade_dir == 'long':
                new_sl = entry_price + (highest_r - 0.5) * sl_dist
                sl_level = max(sl_level, new_sl)
            else:
                new_sl = entry_price - (highest_r - 0.5) * sl_dist
                sl_level = min(sl_level, new_sl)

    # Timeout
    if exit_price is None:
        last_bar = post_entry.iloc[-1]
        exit_price = last_bar['close']
        exit_type = 'timeout'
        exit_time = last_bar['time_et']

    if trade_dir == 'long':
        pnl = exit_price - entry_price
    else:
        pnl = entry_price - exit_price

    return {
        'pnl': pnl,
        'pnl_r': pnl / sl_dist,
        'pnl_adr': pnl / adr,
        'exit_type': exit_type,
        'exit_time': exit_time,
        'highest_r': highest_r,
        'trail_active': trail_active,
        'winner': pnl > 0,
        'ambig_count': ambig_count,
        'zoom_used': zoom_used,
        'zoom_resolved': zoom_resolved,
    }


def simulate_multiday_trail_d_streaming(entry_price, sl_adr_mult, adr, sign,
                                         day0_bars_after_entry, followup_sums,
                                         max_days=5):
    """
    Multi-day TRAIL_D for late-day swing entries.
    CONSERVATIVE: SL check BEFORE peak update on each daily bar.

    Parameters:
        entry_price: Entry price
        sl_adr_mult: SL as multiple of ADR (e.g. 0.25)
        adr: Average daily range
        sign: +1 for long, -1 for short
        day0_bars_after_entry: Remaining 1-min bars on Day 0 after entry (or None)
        followup_sums: List of dicts {rth_open, rth_close, rth_high, rth_low} for D+1..D+N
        max_days: Maximum hold period

    Returns:
        dict with pnl_r, exit_reason, exit_day, max_fav_r, trail_active
    """
    sl_dist = sl_adr_mult * adr
    if sl_dist <= 0:
        return None

    trade_dir = 'long' if sign > 0 else 'short'
    sl_level = entry_price - sign * sl_dist
    trail_active = False
    highest_r = 0.0

    # Day 0 remaining: process bar by bar if available
    if day0_bars_after_entry is not None and len(day0_bars_after_entry) > 0:
        for _, bar in day0_bars_after_entry.iterrows():
            # SL check FIRST
            if sign > 0 and bar['low'] <= sl_level:
                return {'pnl_r': (sl_level - entry_price) / sl_dist,
                        'exit_reason': 'SL_day0', 'exit_day': 0,
                        'max_fav_r': highest_r, 'trail_active': trail_active}
            elif sign < 0 and bar['high'] >= sl_level:
                return {'pnl_r': (entry_price - sl_level) / sl_dist,
                        'exit_reason': 'SL_day0', 'exit_day': 0,
                        'max_fav_r': highest_r, 'trail_active': trail_active}

            # Peak update
            if sign > 0:
                highest_r = max(highest_r, max(0, (bar['high'] - entry_price) / sl_dist))
            else:
                highest_r = max(highest_r, max(0, (entry_price - bar['low']) / sl_dist))

            # Trail activation
            if highest_r >= 1.0 and not trail_active:
                trail_active = True
                sl_level = entry_price
            if trail_active:
                if sign > 0:
                    sl_level = max(sl_level, entry_price + (highest_r - 0.5) * sl_dist)
                else:
                    sl_level = min(sl_level, entry_price - (highest_r - 0.5) * sl_dist)

    # Follow-up days (daily bars)
    last_close, last_day = None, None
    for i, s in enumerate(followup_sums[:max_days]):
        if s is None:
            continue
        offset = i + 1
        dh, dl, dc = s['rth_high'], s['rth_low'], s['rth_close']
        if any(np.isnan(x) for x in [dh, dl, dc]):
            continue
        last_close, last_day = dc, offset

        # SL check FIRST (conservative)
        # Gap-through fix: if RTH-Open already past SL, exit at open (not SL level)
        do = s['rth_open']
        if sign > 0 and dl <= sl_level:
            exit_px = min(do, sl_level) if not np.isnan(do) else sl_level
            return {'pnl_r': (exit_px - entry_price) / sl_dist, 'exit_reason': 'SL_hit',
                    'exit_day': offset, 'max_fav_r': highest_r, 'trail_active': trail_active}
        elif sign < 0 and dh >= sl_level:
            exit_px = max(do, sl_level) if not np.isnan(do) else sl_level
            return {'pnl_r': (entry_price - exit_px) / sl_dist, 'exit_reason': 'SL_hit',
                    'exit_day': offset, 'max_fav_r': highest_r, 'trail_active': trail_active}

        # Peak update
        if sign > 0:
            highest_r = max(highest_r, (dh - entry_price) / sl_dist)
        else:
            highest_r = max(highest_r, (entry_price - dl) / sl_dist)

        # Trail activation & update
        if highest_r >= 1.0 and not trail_active:
            trail_active = True
            sl_level = entry_price
        if trail_active:
            if sign > 0:
                sl_level = max(sl_level, entry_price + (highest_r - 0.5) * sl_dist)
            else:
                sl_level = min(sl_level, entry_price - (highest_r - 0.5) * sl_dist)

        # Trail close check (close below trail)
        if trail_active:
            if sign > 0 and dc < sl_level:
                return {'pnl_r': (dc - entry_price) / sl_dist, 'exit_reason': 'trail_close',
                        'exit_day': offset, 'max_fav_r': highest_r, 'trail_active': True}
            elif sign < 0 and dc > sl_level:
                return {'pnl_r': (entry_price - dc) / sl_dist, 'exit_reason': 'trail_close',
                        'exit_day': offset, 'max_fav_r': highest_r, 'trail_active': True}

    # Max hold timeout
    if last_close is not None:
        pnl = (last_close - entry_price) * sign
        return {'pnl_r': pnl / sl_dist, 'exit_reason': 'max_hold', 'exit_day': last_day,
                'max_fav_r': highest_r, 'trail_active': trail_active}
    return None
