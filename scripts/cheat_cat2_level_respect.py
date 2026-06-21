"""
Cheat Sheet Category 2: Level-Respekt und Preis-Gedaechtnis
Questions 2.1-2.4: Level rejections, retests, multi-touch, volume at levels
"""
import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm
from collections import defaultdict

# ── helpers ──────────────────────────────────────────────────────────
def bootstrap_ci(data, n_boot=1000, ci=0.95):
    if len(data) < 5:
        return np.nan, np.nan
    rng = np.random.default_rng(42)
    stats = np.array([np.mean(rng.choice(data, size=len(data), replace=True)) for _ in range(n_boot)])
    return np.percentile(stats, (1-ci)/2*100), np.percentile(stats, (1+ci)/2*100)

def fmt_pct(val, ci_lo, ci_hi):
    return f"{val*100:.1f}% (CI: {ci_lo*100:.0f}-{ci_hi*100:.0f}%)"

# ── load metadata ────────────────────────────────────────────────────
print("Loading metadata...", file=sys.stderr)
meta = pd.read_parquet("data/metadata/metadata_master.parquet")
meta = meta[(meta['date'] <= '2023-12-31') & (meta['adr_10'].notna())].copy()
print(f"Halfte 1: {len(meta)} gapper days", file=sys.stderr)

# ── Event collection ─────────────────────────────────────────────────
# Each event = a level touch/rejection
touch_events = []   # Q2.1/2.2: individual level touches
multi_touch = []    # Q2.3: multi-touch sequences per level per day
vol_events = []     # Q2.4: volume at level touches

LEVELS = {
    'vwap': 'vwap',
    '+1s': 'upper_1std', '-1s': 'lower_1std',
    '+2s': 'upper_2std', '-2s': 'lower_2std',
    '+3s': 'upper_3std', '-3s': 'lower_3std',
}

# Define "next level" mappings
LEVEL_ORDER_UP = ['lower_3std', 'lower_2std', 'lower_1std', 'vwap', 'upper_1std', 'upper_2std', 'upper_3std']
LEVEL_ORDER_DOWN = list(reversed(LEVEL_ORDER_UP))

def get_next_level(current_col, direction):
    """Get the next level column in the given direction."""
    order = LEVEL_ORDER_UP if direction == 'up' else LEVEL_ORDER_DOWN
    try:
        idx = order.index(current_col)
        if idx + 1 < len(order):
            return order[idx + 1]
    except ValueError:
        pass
    return None

for idx, row in tqdm(meta.iterrows(), total=len(meta), desc="Processing", file=sys.stderr):
    ticker = row['ticker']
    date = row['date']
    gap_dir = row['gap_direction']
    adr = row['adr_10']

    vwap_path = f"data/vwap/{ticker}/{date}.parquet"
    raw_path = f"data/raw_1min/{ticker}/{date}.parquet"
    if not os.path.exists(vwap_path) or not os.path.exists(raw_path):
        continue

    try:
        vwap_df = pd.read_parquet(vwap_path)
        raw_df = pd.read_parquet(raw_path)
    except:
        continue

    if 'session' in raw_df.columns:
        raw_df = raw_df[raw_df['session'] == 'rth'].copy()

    if len(vwap_df) < 30 or len(raw_df) < 30 or adr <= 0:
        continue

    merged = pd.merge(
        raw_df[['time_et','open','high','low','close','volume','transactions']],
        vwap_df[['time_et','vwap','upper_1std','lower_1std','upper_2std','lower_2std',
                  'upper_3std','lower_3std','std_dev']],
        on='time_et', how='inner'
    ).sort_values('time_et').reset_index(drop=True)

    if len(merged) < 30:
        continue

    def time_to_min(t):
        if isinstance(t, str):
            parts = t.split(':')
            return (int(parts[0]) - 9) * 60 + (int(parts[1]) - 30)
        return np.nan

    merged['min_from_open'] = merged['time_et'].apply(time_to_min)
    merged = merged[(merged['min_from_open'] >= 0) & merged['vwap'].notna()].reset_index(drop=True)

    if len(merged) < 20:
        continue

    # Compute bar-level volume metrics
    merged['dollar_vol'] = merged['volume'] * (merged['high'] + merged['low']) / 2
    day_avg_vol = merged['volume'].mean()

    # ── Detect level touches and rejections ──
    for level_name, level_col in LEVELS.items():
        if level_col not in merged.columns:
            continue

        # Track touches from above (high touches level from above = potential support)
        # and from below (low touches level from below = potential resistance)
        touches_this_level = []  # (bar_idx, touch_type, rejection_price)

        for i in range(2, len(merged) - 1):
            level_val = merged.iloc[i][level_col]
            if pd.isna(level_val):
                continue

            bar = merged.iloc[i]
            next_bar = merged.iloc[i+1]
            prev_bar = merged.iloc[i-1]

            # Touch from above (testing as support): low touches/crosses level, close above
            touch_from_above = (bar['low'] <= level_val * 1.001 and
                                bar['close'] > level_val and
                                prev_bar['close'] > level_val)

            # Touch from below (testing as resistance): high touches/crosses level, close below
            touch_from_below = (bar['high'] >= level_val * 0.999 and
                                bar['close'] < level_val and
                                prev_bar['close'] < level_val)

            if touch_from_above:
                rejection_price = bar['low']
                touch_dir = 'support'
            elif touch_from_below:
                rejection_price = bar['high']
                touch_dir = 'resistance'
            else:
                continue

            # Check if next approach breaks this fixed price
            # Look forward up to 60 bars
            touch_min = bar['min_from_open']
            approaches = {'15min': 15, '30min': 30, '60min': 60}

            for window_name, window_bars in approaches.items():
                end_idx = min(i + window_bars, len(merged))
                future = merged.iloc[i+1:end_idx]

                if len(future) < 3:
                    continue

                if touch_dir == 'support':
                    # Price was rejected going down -> check if rejection_price (low) holds
                    broken = (future['close'] < rejection_price).any()
                    # Check if next level UP reached before break
                    next_lvl_col = get_next_level(level_col, 'up')
                    next_level_reached = False
                    if next_lvl_col and next_lvl_col in future.columns:
                        next_level_reached = (future['high'] >= future[next_lvl_col]).any()
                else:
                    broken = (future['close'] > rejection_price).any()
                    next_lvl_col = get_next_level(level_col, 'down')
                    next_level_reached = False
                    if next_lvl_col and next_lvl_col in future.columns:
                        next_level_reached = (future['low'] <= future[next_lvl_col]).any()

                touch_events.append({
                    'ticker': ticker, 'date': date, 'gap_dir': gap_dir,
                    'level': level_name, 'touch_dir': touch_dir,
                    'bar_idx': i, 'min_from_open': touch_min,
                    'window': window_name,
                    'rejection_price_held': not broken,
                    'next_level_reached_before_break': next_level_reached and not broken,
                    'vol_ratio': bar['volume'] / max(day_avg_vol, 1),
                    'bar_range_adr': (bar['high'] - bar['low']) / adr,
                })

            touches_this_level.append((i, touch_dir, rejection_price))

        # ── Q2.3: Multi-touch tracking ──
        if len(touches_this_level) >= 2:
            for touch_num in range(1, len(touches_this_level)):
                prev_t = touches_this_level[touch_num - 1]
                curr_t = touches_this_level[touch_num]
                # Same direction touch
                if prev_t[1] == curr_t[1]:
                    # Check if this touch broke the level
                    bar_i = curr_t[0]
                    if bar_i + 3 < len(merged):
                        future_3 = merged.iloc[bar_i+1:bar_i+4]
                        if curr_t[1] == 'support':
                            broke = (future_3['close'] < merged.iloc[bar_i][LEVELS[level_name]]).any()
                        else:
                            broke = (future_3['close'] > merged.iloc[bar_i][LEVELS[level_name]]).any()

                        multi_touch.append({
                            'ticker': ticker, 'date': date, 'gap_dir': gap_dir,
                            'level': level_name, 'touch_dir': curr_t[1],
                            'touch_number': touch_num + 1,  # 2nd, 3rd, etc.
                            'broke': broke,
                            'min_between': merged.iloc[curr_t[0]]['min_from_open'] - merged.iloc[prev_t[0]]['min_from_open'],
                        })

    # ── Q2.4: Volume at level touches ──
    # Already captured vol_ratio and bar_range_adr in touch_events

# ── Convert to DataFrames ──────────────────────────────────────────
print(f"\nTouch events: {len(touch_events)}", file=sys.stderr)
print(f"Multi-touch events: {len(multi_touch)}", file=sys.stderr)

touch_df = pd.DataFrame(touch_events)
multi_df = pd.DataFrame(multi_touch)

touch_df.to_parquet("results/cheat_cat2_touches.parquet", index=False)

# ── Generate Report ─────────────────────────────────────────────────
out = []
out.append("=" * 80)
out.append("CHEAT SHEET — KATEGORIE 2: LEVEL-RESPEKT UND PREIS-GEDAECHTNIS")
out.append("=" * 80)
out.append(f"Datenbasis: {len(touch_df)} Level-Touch Events aus Halfte 1")
out.append("")

# ── Q2.1: Fixed rejection price — how often does it hold on next approach? ──
out.append("=" * 70)
out.append("FRAGE 2.1: Fixierter Ablehnungspreis — Haelt er beim naechsten Anlauf?")
out.append("=" * 70)

for window in ['15min', '30min', '60min']:
    out.append(f"\n--- Zeitfenster: {window} ---")
    wsub = touch_df[touch_df['window'] == window]

    for level in ['vwap', '-1s', '+1s', '-2s', '+2s']:
        for tdir in ['support', 'resistance']:
            lsub = wsub[(wsub['level'] == level) & (wsub['touch_dir'] == tdir)]
            if len(lsub) < 30:
                continue
            held = lsub['rejection_price_held'].astype(float).values
            mean_held = np.mean(held)
            lo, hi = bootstrap_ci(held)
            out.append(f"  {level} als {tdir}: Haelt={fmt_pct(mean_held, lo, hi)}, N={len(lsub)}")

# ── Q2.2: Next level reached before break? ──
out.append("\n" + "=" * 70)
out.append("FRAGE 2.2: Naechstes Level erreicht BEVOR Ablehnungspreis bricht?")
out.append("=" * 70)

wsub = touch_df[touch_df['window'] == '60min']  # 60min window
for level in ['vwap', '-1s', '+1s', '-2s', '+2s']:
    for tdir in ['support', 'resistance']:
        lsub = wsub[(wsub['level'] == level) & (wsub['touch_dir'] == tdir)]
        if len(lsub) < 30:
            continue
        # Of those where rejection held, how many reached next level?
        held_sub = lsub[lsub['rejection_price_held'] == True]
        if len(held_sub) < 10:
            continue
        reached = held_sub['next_level_reached_before_break'].astype(float).values
        mean_r = np.mean(reached)
        lo, hi = bootstrap_ci(reached)
        out.append(f"  {level} {tdir} haelt -> naechstes Level erreicht: {fmt_pct(mean_r, lo, hi)}, N={len(held_sub)}")

# ── Q2.3: Multi-touch — breaks more likely on 2nd, 3rd touch? ──
out.append("\n" + "=" * 70)
out.append("FRAGE 2.3: Multi-Touch — Bricht Level eher beim 2., 3., N-ten Anlauf?")
out.append("=" * 70)

if len(multi_df) > 0:
    for level in ['vwap', '-1s', '+1s', '-2s', '+2s']:
        lsub = multi_df[multi_df['level'] == level]
        if len(lsub) < 20:
            continue
        out.append(f"\n  Level: {level}")
        for touch_n in sorted(lsub['touch_number'].unique()):
            nsub = lsub[lsub['touch_number'] == touch_n]
            if len(nsub) < 10:
                n_str = " (LOW N)" if len(nsub) < 50 else ""
                out.append(f"    Touch #{touch_n}: N={len(nsub)}{n_str}")
                continue
            break_rate = nsub['broke'].astype(float).values
            mean_br = np.mean(break_rate)
            lo, hi = bootstrap_ci(break_rate)
            out.append(f"    Touch #{touch_n}: Break-Rate={fmt_pct(mean_br, lo, hi)}, N={len(nsub)}")

# ── Q2.4: Volume at level touch ──
out.append("\n" + "=" * 70)
out.append("FRAGE 2.4: Volumen bei Level-Touch — Absorption vs Breakout")
out.append("=" * 70)

wsub = touch_df[touch_df['window'] == '30min']

# High volume = vol_ratio > 2x average bar
# Small bar = bar_range < 0.05 ADR (absorption)
# Large bar = bar_range > 0.1 ADR (breakout attempt)

for level in ['vwap', '-1s', '+1s']:
    lsub = wsub[wsub['level'] == level]
    if len(lsub) < 50:
        continue

    out.append(f"\n  Level: {level}")

    # High vol + small bar (absorption)
    absorption = lsub[(lsub['vol_ratio'] > 2.0) & (lsub['bar_range_adr'] < 0.05)]
    if len(absorption) >= 10:
        held = absorption['rejection_price_held'].astype(float).values
        mean_h = np.mean(held)
        lo, hi = bootstrap_ci(held)
        out.append(f"    Absorption (HiVol + kleine Bar): Level haelt={fmt_pct(mean_h, lo, hi)}, N={len(absorption)}")

    # High vol + large bar (breakout attempt)
    breakout = lsub[(lsub['vol_ratio'] > 2.0) & (lsub['bar_range_adr'] > 0.1)]
    if len(breakout) >= 10:
        held = breakout['rejection_price_held'].astype(float).values
        mean_h = np.mean(held)
        lo, hi = bootstrap_ci(held)
        out.append(f"    Breakout-Versuch (HiVol + grosse Bar): Level haelt={fmt_pct(mean_h, lo, hi)}, N={len(breakout)}")

    # Low vol touch
    lowvol = lsub[lsub['vol_ratio'] < 1.0]
    if len(lowvol) >= 10:
        held = lowvol['rejection_price_held'].astype(float).values
        mean_h = np.mean(held)
        lo, hi = bootstrap_ci(held)
        out.append(f"    Low-Vol Touch: Level haelt={fmt_pct(mean_h, lo, hi)}, N={len(lowvol)}")

# ── Write output ────────────────────────────────────────────────────
report = "\n".join(out)
with open("results/cheat_cat2_level_respect.txt", "w", encoding="utf-8") as f:
    f.write(report)

print(report)
print(f"\nResults saved to results/cheat_cat2_level_respect.txt", file=sys.stderr)
