"""
D16v2: Unified Strategy Detection (S1-S4)
==========================================
All 4 strategies share VWAP-band logic. Multiple can fire on the same day.

S1: GapUp Recovery (Long)   — touch -2σ, close@15:58 >= +1σ
S2: GapDown Recovery (Short) — touch +2σ, close@15:58 <= -1σ
S3: GapUp Fade (Short)      — touch +2σ, close@15:58 <= -1σ, gap-close room
S4: GapDown Fade (Long)     — touch -2σ, close@15:58 >= +1σ, gap-close room
"""
from d16v2_swing.engine.vwap_band_scanner import scan_vwap_bands


def detect_strategies(ticker, date, gap_dir, adr, prev_close,
                      rvol30, yesterday_high, yesterday_low):
    """
    Returns list of triggered strategies for this (ticker, date).
    Each dict: {strategy, entry_price, entry_time, trade_dir, strategy_label}
    Multiple strategies can fire on same day.
    """
    if rvol30 < 1.5:
        return []

    scan = scan_vwap_bands(ticker, date)
    if scan is None:
        return []

    close_1558 = scan['close_1558']
    u1 = scan['upper_1std_1558']
    l1 = scan['lower_1std_1558']

    results = []

    # --- S1: GapUp Recovery (Long) ---
    # GapUp + touch -2σ after 11:00 + close@15:58 >= upper_1std
    if gap_dir == 'up' and scan['touch_lower_2std'] and close_1558 >= u1:
        results.append({
            'strategy': 'S1_GapUp_Recovery',
            'entry_price': close_1558,
            'entry_time': '15:58',
            'trade_dir': 'long',
        })

    # --- S2: GapDown Recovery (Short) ---
    # GapDown + touch +2σ after 11:00 + close@15:58 <= lower_1std
    if gap_dir == 'down' and scan['touch_upper_2std'] and close_1558 <= l1:
        results.append({
            'strategy': 'S2_GapDown_Recovery',
            'entry_price': close_1558,
            'entry_time': '15:58',
            'trade_dir': 'short',
        })

    # --- S3: GapUp Fade (Short) ---
    # GapUp + touch +2σ after 11:00 + close@15:58 <= lower_1std
    # + gap-close room: close_1558 - yesterday_high >= 1.0 * adr
    if gap_dir == 'up' and scan['touch_upper_2std'] and close_1558 <= l1:
        gap_room = close_1558 - yesterday_high
        if adr > 0 and gap_room >= 1.0 * adr:
            results.append({
                'strategy': 'S3_GapUp_Fade',
                'entry_price': close_1558,
                'entry_time': '15:58',
                'trade_dir': 'short',
            })

    # --- S4: GapDown Fade (Long) ---
    # GapDown + touch -2σ after 11:00 + close@15:58 >= upper_1std
    # + gap-close room: yesterday_low - close_1558 >= 1.0 * adr
    if gap_dir == 'down' and scan['touch_lower_2std'] and close_1558 >= u1:
        gap_room = yesterday_low - close_1558
        if adr > 0 and gap_room >= 1.0 * adr:
            results.append({
                'strategy': 'S4_GapDown_Fade',
                'entry_price': close_1558,
                'entry_time': '15:58',
                'trade_dir': 'long',
            })

    return results
