"""
TEST C: Naked Direction Test
==============================
Pruefe ob GapDn-Aktien einen systematischen positiven Tagesreturn haben (Mean Reversion),
und GapUp-Aktien einen negativen, OHNE jegliches SL/Target/VWAP.
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
from tqdm import tqdm
import glob

np.random.seed(42)

# ============================================================
# STEP 1: Load metadata for 2023 (Val period)
# ============================================================
print("Step 1: Loading metadata...", file=sys.stderr)
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

meta = pd.read_parquet(str(PROJECT_ROOT / 'data' / 'metadata' / 'metadata_master.parquet'))
meta['date'] = pd.to_datetime(meta['date'])

# Val period: 2023 only
val_meta = meta[(meta['date'] >= '2023-01-01') & (meta['date'] <= '2023-12-31')].copy()
print(f"  Val gappers (2023): {len(val_meta)}", file=sys.stderr)
print(f"  GapDn: {(val_meta['gap_direction']=='down').sum()}", file=sys.stderr)
print(f"  GapUp: {(val_meta['gap_direction']=='up').sum()}", file=sys.stderr)

# ============================================================
# STEP 2: Load VWAP files and compute returns
# ============================================================
print("\nStep 2: Loading VWAP files and computing returns...", file=sys.stderr)

results = []
for _, row in tqdm(val_meta.iterrows(), total=len(val_meta), desc="Processing", file=sys.stderr):
    ticker = row['ticker']
    date_str = row['date'].strftime('%Y-%m-%d')
    gap_dir = row['gap_direction']
    adr_10 = row['adr_10']

    # Find VWAP file
    fpath = f'{PROJECT_ROOT}/data/vwap/{ticker}/{date_str}.parquet'
    try:
        vwap_df = pd.read_parquet(fpath, columns=['close', 'time_et'])
    except:
        # Try alternate path format
        try:
            fpath2 = f'{PROJECT_ROOT}/data/vwap/{ticker}/{ticker}_{date_str}.parquet'
            vwap_df = pd.read_parquet(fpath2, columns=['close', 'time_et'])
        except:
            continue

    if len(vwap_df) < 100:
        continue

    prices = vwap_df['close'].values
    times = vwap_df['time_et'].values

    open_price = prices[0]  # 09:30
    if open_price <= 0 or adr_10 <= 0:
        continue

    # Close price (last bar ~ 15:59)
    close_price = prices[-1]

    # 30 min (bar 30 ~ 10:00)
    price_30 = prices[min(29, len(prices)-1)]

    # 90 min = bar 90 (10:00 to 11:00 = bars 30-89, or 11:00 = bar 90)
    # Let's use: 10:00 = bar 30, 12:00 = bar 150
    price_150 = prices[min(149, len(prices)-1)]

    # Returns in ADR
    ret_open_close = (close_price - open_price) / adr_10
    ret_0_30 = (price_30 - open_price) / adr_10
    ret_30_150 = (price_150 - price_30) / adr_10
    ret_150_close = (close_price - price_150) / adr_10

    results.append({
        'ticker': ticker,
        'date': date_str,
        'gap_direction': gap_dir,
        'gap_pct': row['gap_pct'],
        'adr_10': adr_10,
        'ret_open_close': ret_open_close,
        'ret_0_30': ret_0_30,
        'ret_30_150': ret_30_150,
        'ret_150_close': ret_150_close,
        'open_price': open_price,
        'close_price': close_price,
    })

rdf = pd.DataFrame(results)
print(f"  Successfully loaded: {len(rdf)} gapper-days", file=sys.stderr)

# ============================================================
# STEP 3: Compute direction returns
# ============================================================
print("\nStep 3: Computing directional returns...", file=sys.stderr)

# For GapDn Long: positive return = good (we buy at open, sell at close)
gapdn = rdf[rdf['gap_direction'] == 'down'].copy()
gapup = rdf[rdf['gap_direction'] == 'up'].copy()

# For GapUp Short: we INVERT the return (short = profit when price drops)
# So ret_gapup_short = -ret_open_close

def bootstrap_ci(data, n_boot=10000, ci=0.95):
    data = np.array(data)
    n = len(data)
    rng = np.random.default_rng(42)
    boot_means = np.array([np.mean(rng.choice(data, size=n, replace=True)) for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    lo = np.percentile(boot_means, alpha * 100)
    hi = np.percentile(boot_means, (1 - alpha) * 100)
    p_positive = np.mean(boot_means > 0)
    p_negative = np.mean(boot_means < 0)
    return lo, hi, p_positive, p_negative

# ============================================================
# STEP 4: Write results
# ============================================================
print("\nStep 4: Writing results...", file=sys.stderr)

outpath = str(PROJECT_ROOT / 'results' / 'test_naked_direction.txt')
with open(outpath, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("TEST C: NAKED DIRECTION TEST\n")
    f.write("=" * 80 + "\n")
    f.write("Date: 2026-02-13\n")
    f.write("Period: 2023 (Val)\n")
    f.write("Method: Buy at Open (09:30), hold, sell at close. NO SL, NO Target.\n")
    f.write("Returns normalized by ADR_10.\n\n")

    f.write("=" * 80 + "\n")
    f.write("SECTION 1: FULL DAY RETURNS (Open to Close)\n")
    f.write("=" * 80 + "\n\n")

    for label, data, direction in [
        ('GapDn (LONG at Open)', gapdn['ret_open_close'], 'long'),
        ('GapUp (SHORT at Open)', -gapup['ret_open_close'], 'short'),
        ('GapUp (LONG at Open - for comparison)', gapup['ret_open_close'], 'long'),
    ]:
        vals = data.dropna().values
        n = len(vals)
        if n < 10:
            continue
        mean_r = np.mean(vals)
        med_r = np.median(vals)
        std_r = np.std(vals)
        lo, hi, p_pos, p_neg = bootstrap_ci(vals)

        f.write(f"  {label}:\n")
        f.write(f"    N:         {n}\n")
        f.write(f"    Mean:      {mean_r:+.4f} ADR\n")
        f.write(f"    Median:    {med_r:+.4f} ADR\n")
        f.write(f"    Std:       {std_r:.4f} ADR\n")
        f.write(f"    95% CI:    [{lo:+.4f}, {hi:+.4f}]\n")
        f.write(f"    P(mean>0): {p_pos:.4f}\n")
        f.write(f"    P(mean<0): {p_neg:.4f}\n")

        if mean_r > 0 and p_pos > 0.95:
            f.write(f"    >>> SIGNIFICANTLY POSITIVE (p={p_pos:.4f}) <<<\n")
        elif mean_r < 0 and p_neg > 0.95:
            f.write(f"    >>> SIGNIFICANTLY NEGATIVE (p={p_neg:.4f}) <<<\n")
        else:
            f.write(f"    >>> NOT SIGNIFICANT <<<\n")
        f.write("\n")

    # Asymmetry
    dn_mean = gapdn['ret_open_close'].mean()
    up_mean = gapup['ret_open_close'].mean()
    f.write(f"  Direction Asymmetry:\n")
    f.write(f"    GapDn Long Mean:  {dn_mean:+.4f} ADR\n")
    f.write(f"    GapUp Long Mean:  {up_mean:+.4f} ADR\n")
    f.write(f"    Difference (Dn-Up): {dn_mean - up_mean:+.4f} ADR\n")
    f.write(f"    -> {'GapDn has STRONGER mean reversion' if dn_mean > up_mean else 'GapUp matches or exceeds GapDn'}\n\n")

    f.write("=" * 80 + "\n")
    f.write("SECTION 2: INTRADAY PERIOD RETURNS\n")
    f.write("=" * 80 + "\n\n")

    periods = [
        ('First 30min (09:30-10:00)', 'ret_0_30'),
        ('Mid-day (10:00-12:00)', 'ret_30_150'),
        ('Afternoon (12:00-Close)', 'ret_150_close'),
    ]

    for period_name, col in periods:
        f.write(f"  --- {period_name} ---\n")
        for gap_label, subset in [('GapDn (Long)', gapdn), ('GapUp (Short)', gapup)]:
            vals = subset[col].dropna().values
            if gap_label.startswith('GapUp'):
                vals = -vals  # Short direction
            n = len(vals)
            if n < 10:
                continue
            mean_r = np.mean(vals)
            med_r = np.median(vals)
            lo, hi, p_pos, _ = bootstrap_ci(vals)
            f.write(f"    {gap_label}: N={n}, Mean={mean_r:+.4f}, Med={med_r:+.4f}, CI=[{lo:+.4f},{hi:+.4f}], P(>0)={p_pos:.3f}\n")
        f.write("\n")

    f.write("=" * 80 + "\n")
    f.write("SECTION 3: GAP SIZE BUCKETS\n")
    f.write("=" * 80 + "\n\n")

    # Bucket by gap size
    for gap_dir, subset, direction_label in [('down', gapdn, 'Long'), ('up', gapup, 'Short')]:
        f.write(f"  --- Gap{gap_dir.title()} ({direction_label}) by Gap Size ---\n")
        subset = subset.copy()
        subset['gap_bucket'] = pd.cut(subset['gap_pct'].abs(), bins=[4, 6, 10, 15, 25, 100], labels=['4-6%', '6-10%', '10-15%', '15-25%', '>25%'])

        for bucket in ['4-6%', '6-10%', '10-15%', '15-25%', '>25%']:
            b_data = subset[subset['gap_bucket'] == bucket]
            if len(b_data) < 10:
                f.write(f"    {bucket}: N={len(b_data)} (too few)\n")
                continue
            ret_vals = b_data['ret_open_close'].values
            if direction_label == 'Short':
                ret_vals = -ret_vals
            mean_r = np.mean(ret_vals)
            n = len(ret_vals)
            lo, hi, p_pos, _ = bootstrap_ci(ret_vals)
            f.write(f"    {bucket}: N={n}, Mean={mean_r:+.4f}, CI=[{lo:+.4f},{hi:+.4f}], P(>0)={p_pos:.3f}\n")
        f.write("\n")

    # ---- VERDICT ----
    f.write("=" * 80 + "\n")
    f.write("SECTION 4: VERDICT\n")
    f.write("=" * 80 + "\n\n")

    dn_vals = gapdn['ret_open_close'].dropna().values
    up_vals = -gapup['ret_open_close'].dropna().values  # Short direction
    dn_mean = np.mean(dn_vals)
    up_mean = np.mean(up_vals)
    dn_lo, dn_hi, dn_ppos, _ = bootstrap_ci(dn_vals)
    up_lo, up_hi, up_ppos, _ = bootstrap_ci(up_vals)

    f.write(f"  GapDn Long Open-to-Close: Mean={dn_mean:+.4f} ADR, CI=[{dn_lo:+.4f},{dn_hi:+.4f}], P(>0)={dn_ppos:.4f}\n")
    f.write(f"  GapUp Short Open-to-Close: Mean={up_mean:+.4f} ADR, CI=[{up_lo:+.4f},{up_hi:+.4f}], P(>0)={up_ppos:.4f}\n\n")

    if dn_mean > 0 and dn_ppos > 0.95:
        f.write("  >>> GapDn has a SIGNIFICANT positive mean reversion tendency <<<\n")
        f.write(f"  >>> Raw directional edge: {dn_mean:+.4f} ADR per trade (no SL/Target needed) <<<\n")
    elif dn_mean > 0:
        f.write("  >>> GapDn shows positive tendency but NOT significant <<<\n")
    else:
        f.write("  >>> GapDn shows NO positive mean reversion tendency <<<\n")

    if up_mean > 0 and up_ppos > 0.95:
        f.write("  >>> GapUp Short ALSO has positive mean reversion tendency <<<\n")
    elif up_mean <= 0:
        f.write("  >>> GapUp Short shows NO positive tendency <<<\n")

    f.write("\n" + "=" * 80 + "\n")
    f.write("END OF TEST C\n")
    f.write("=" * 80 + "\n")

print(f"Results written to: {outpath}", file=sys.stderr)
print("TEST C COMPLETE.", file=sys.stderr)
