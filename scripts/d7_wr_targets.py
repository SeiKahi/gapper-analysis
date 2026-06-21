"""
Durchlauf 7.0 — Aufgabe 6: WR bei verschiedenen Targets
Entry bei 9:35 (close_935), SL am 5min H/L.
IS only (H1: 2021-02-21 bis 2023-12-31).
"""
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import warnings
warnings.filterwarnings('ignore')

def bootstrap_ci(data, n_boot=1000, ci=0.95):
    data = np.array(data, dtype=float)
    data = data[~np.isnan(data)]
    if len(data) < 5: return np.nan, np.nan
    rng = np.random.default_rng(42)
    means = [np.mean(rng.choice(data, len(data), True)) for _ in range(n_boot)]
    return np.percentile(means, (1-ci)/2*100), np.percentile(means, (1+ci)/2*100)

def low_n(n):
    if n < 20: return " *** VERY LOW N ***"
    if n < 50: return " [LOW N]"
    return ""

out = []
def w(s=""): out.append(s)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════════════════════════════════════════
print("Loading metadata_v7...", file=sys.stderr)
meta = pd.read_parquet('data/metadata/metadata_v7.parquet')
meta['date'] = pd.to_datetime(meta['date'])
h1 = meta[(meta['date'] >= '2021-02-21') & (meta['date'] <= '2023-12-31')].copy()
h1 = h1[h1['adr_10'].notna() & (h1['adr_10'] > 0)].copy()

# Define combos
h1['combo_e'] = (
    (h1['first_candle_dir'] == 'with_gap') &
    (h1['first_candle_size'] > 0.20) &
    (h1['od_direction'] == 'with_gap') &
    (h1['od_strength'] > 0.5)
)
h1['od_05_with'] = (h1['od_direction'] == 'with_gap') & (h1['od_strength'] > 0.5)
h1['pm30_low_od'] = (h1['pm_rth30_computed'] < 0.10) & h1['od_05_with']

RAW_DIR = Path('data/raw_1min')
TARGETS = [0.25, 0.50, 0.75, 1.00, 1.50, 2.00]

# ══════════════════════════════════════════════════════════════════════════════
# COMPUTE WR FOR EACH COMBO
# ══════════════════════════════════════════════════════════════════════════════
combos = {
    'Combo E': h1[h1['combo_e']],
    'OD>0.5 with_gap': h1[h1['od_05_with']],
    'PM30<10%+OD>0.5w': h1[h1['pm30_low_od']],
    'ALL': h1,
}

all_trade_records = []

for combo_name, combo_df in combos.items():
    print(f"\nProcessing {combo_name}: {len(combo_df)} days...", file=sys.stderr)

    for _, row in tqdm(combo_df.iterrows(), total=len(combo_df), desc=combo_name, file=sys.stderr):
        ticker = row['ticker']
        date_str = row['date'].strftime('%Y-%m-%d')
        adr = row['adr_10']
        gap_dir = row['gap_direction']
        close_935 = row.get('close_935', np.nan)

        if pd.isna(close_935) or pd.isna(adr) or adr <= 0:
            continue

        raw_path = RAW_DIR / ticker / f'{date_str}.parquet'
        if not raw_path.exists():
            continue

        try:
            raw = pd.read_parquet(raw_path)
        except Exception:
            continue

        rth = raw[raw['session'] == 'rth'].sort_values('time_et').reset_index(drop=True)
        if len(rth) < 10:
            continue

        # Entry at bar 5 (09:35 close = close_935)
        entry_price = close_935

        # SL from 5min H/L
        od_bars = rth.head(5)
        if gap_dir == 'up':
            od_long = True
            sl_price = od_bars['low'].min()
        else:
            od_long = False
            sl_price = od_bars['high'].max()

        sl_size = abs(entry_price - sl_price) / adr

        # Track through remaining bars (5 to end, max 90 bars = 1.5h)
        post_od = rth.iloc[5:min(95, len(rth))]

        sl_hit = False
        sl_hit_bar = None
        mfe_val = 0.0
        mae_val = 0.0
        target_hits = {t: False for t in TARGETS}
        target_before_sl = {t: False for t in TARGETS}

        for bar_idx, (_, bar) in enumerate(post_od.iterrows()):
            if od_long:
                fav = (bar['high'] - entry_price) / adr
                adv = (entry_price - bar['low']) / adr
                hit_sl = bar['low'] <= sl_price
                for t in TARGETS:
                    if not target_hits[t] and bar['high'] >= entry_price + t * adr:
                        target_hits[t] = True
                        if not sl_hit:
                            target_before_sl[t] = True
            else:
                fav = (entry_price - bar['low']) / adr
                adv = (bar['high'] - entry_price) / adr
                hit_sl = bar['high'] >= sl_price
                for t in TARGETS:
                    if not target_hits[t] and bar['low'] <= entry_price - t * adr:
                        target_hits[t] = True
                        if not sl_hit:
                            target_before_sl[t] = True

            mfe_val = max(mfe_val, fav)
            mae_val = max(mae_val, adv)
            if not sl_hit and hit_sl:
                sl_hit = True
                sl_hit_bar = bar_idx

        rec = {
            'combo': combo_name, 'ticker': ticker, 'date': date_str,
            'gap_direction': gap_dir, 'adr': adr,
            'entry_price': entry_price, 'sl_price': sl_price,
            'sl_size': sl_size, 'sl_hit': sl_hit,
            'mfe': mfe_val, 'mae': mae_val,
            'rvol_5': row.get('rvol_5', np.nan),
            'pm_rth5': row.get('pm_rth5', np.nan),
        }
        for t in TARGETS:
            rec[f'wr_{t:.2f}'] = target_before_sl[t]

        all_trade_records.append(rec)

trades = pd.DataFrame(all_trade_records)

# ══════════════════════════════════════════════════════════════════════════════
# 6a: WR bei allen Targets
# ══════════════════════════════════════════════════════════════════════════════
w("=" * 80)
w("AUFGABE 6a: WR BEI ALLEN TARGETS (Entry 9:35, SL 5min H/L)")
w("=" * 80)

for combo_name in combos.keys():
    for gap_dir in ['up', 'down']:
        grp = trades[(trades['combo'] == combo_name) & (trades['gap_direction'] == gap_dir)]
        if len(grp) == 0:
            continue

        w(f"\n--- {combo_name} Gap{gap_dir.title()} (N={len(grp)}) ---")
        w(f"  SL-Hit: {grp['sl_hit'].mean()*100:.1f}%, Median SL: {grp['sl_size'].median():.3f} ADR")
        w(f"  MFE median: {grp['mfe'].median():.3f}, MAE median: {grp['mae'].median():.3f}")

        w(f"  {'Target':>8} {'WR%':>8} {'N_hit':>6} {'CI_lo':>8} {'CI_hi':>8}")
        w("  " + "-" * 40)
        for t in TARGETS:
            wr = grp[f'wr_{t:.2f}'].mean() * 100
            n_hit = grp[f'wr_{t:.2f}'].sum()
            ci_lo, ci_hi = bootstrap_ci(grp[f'wr_{t:.2f}'].astype(float).values)
            w(f"  {t:>8.2f} {wr:>8.1f} {n_hit:>6} {ci_lo*100:>8.1f} {ci_hi*100:>8.1f}")

# ══════════════════════════════════════════════════════════════════════════════
# 6b: Optimaler Risk:Reward
# ══════════════════════════════════════════════════════════════════════════════
w("\n" + "=" * 80)
w("AUFGABE 6b: OPTIMALER RISK:REWARD (Expected Value)")
w("=" * 80)

for combo_name in combos.keys():
    for gap_dir in ['up', 'down']:
        grp = trades[(trades['combo'] == combo_name) & (trades['gap_direction'] == gap_dir)]
        if len(grp) == 0:
            continue

        median_sl = grp['sl_size'].median()
        w(f"\n--- {combo_name} Gap{gap_dir.title()} (N={len(grp)}, median SL={median_sl:.3f} ADR) ---")
        w(f"  {'Target':>8} {'WR':>8} {'EV (ADR)':>12} {'R:R':>8}")
        w("  " + "-" * 40)

        best_ev = -999
        best_target = 0
        for t in TARGETS:
            wr = grp[f'wr_{t:.2f}'].mean()
            ev = wr * t - (1 - wr) * median_sl
            rr = t / median_sl if median_sl > 0 else np.nan
            w(f"  {t:>8.2f} {wr:>8.1%} {ev:>+12.4f} {rr:>8.2f}")
            if ev > best_ev:
                best_ev = ev
                best_target = t

        w(f"  => BEST: Target={best_target:.2f} ADR, EV={best_ev:+.4f} ADR")

# ══════════════════════════════════════════════════════════════════════════════
# 6c: WR mit RVOL_5 als Filter
# ══════════════════════════════════════════════════════════════════════════════
w("\n" + "=" * 80)
w("AUFGABE 6c: WR MIT RVOL_5 ALS FILTER")
w("=" * 80)

rvol5_edges = [0, 2, 5, 10, 999]
rvol5_labels = ['<2x', '2-5x', '5-10x', '>10x']

for combo_name in ['Combo E', 'OD>0.5 with_gap']:
    for gap_dir in ['up', 'down']:
        grp = trades[(trades['combo'] == combo_name) & (trades['gap_direction'] == gap_dir)].copy()
        if len(grp) < 20:
            continue

        grp['rvol5_bkt'] = pd.cut(grp['rvol_5'], bins=rvol5_edges, labels=rvol5_labels, right=False)

        w(f"\n--- {combo_name} Gap{gap_dir.title()} x RVOL_5 ---")
        w(f"  {'RVOL_5':<8} {'N':>5} {'WR@0.25':>8} {'WR@0.50':>8} {'WR@0.75':>8} {'WR@1.00':>8} {'WR@1.50':>8} {'WR@2.00':>8}")
        w("  " + "-" * 65)

        for bkt in rvol5_labels:
            sg = grp[grp['rvol5_bkt'] == bkt]
            if len(sg) == 0:
                continue
            vals = [sg[f'wr_{t:.2f}'].mean()*100 for t in TARGETS]
            w(f"  {bkt:<8} {len(sg):>5}{low_n(len(sg))} " + " ".join(f"{v:>8.1f}" for v in vals))

# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════
with open('results/d7_wr_targets.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))
print(f"Saved results/d7_wr_targets.txt", file=sys.stderr)

trades.to_parquet('results/d7_wr_targets_raw.parquet', index=False)
print('\n'.join(out))
