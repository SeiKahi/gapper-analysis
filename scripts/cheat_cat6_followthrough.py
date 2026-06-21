"""
Cheat Sheet Category 6: Follow-Through Tag 2 + Weekday Effect
Strategy: For each gapper day, find the next trading day's OHLC from baseline_1min/
"""
import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm

def bootstrap_ci(data, n_boot=1000, ci=0.95):
    if len(data) < 5:
        return np.nan, np.nan
    rng = np.random.default_rng(42)
    stats = np.array([np.mean(rng.choice(data, size=len(data), replace=True)) for _ in range(n_boot)])
    return np.percentile(stats, (1-ci)/2*100), np.percentile(stats, (1+ci)/2*100)

def fmt_pct(val, ci_lo, ci_hi):
    return f"{val*100:.1f}% (CI: {ci_lo*100:.0f}-{ci_hi*100:.0f}%)"

# Load metadata
print("Loading metadata...", file=sys.stderr)
meta = pd.read_parquet("data/metadata/metadata_master.parquet")
meta = meta[(meta['date'] <= '2023-12-31') & (meta['adr_10'].notna())].copy()
meta['close_loc'] = (meta['rth_close'] - meta['rth_low']) / (meta['rth_high'] - meta['rth_low'])
meta['gap_size_in_adr'] = meta['gap_size_in_adr'].fillna(
    (meta['today_open'] - meta['prev_close']).abs() / meta['adr_10']
)

def gap_size_bucket(v):
    if v < 1: return '<1x'
    if v < 2: return '1-2x'
    if v < 3: return '2-3x'
    return '>3x'
meta['gap_bucket'] = meta['gap_size_in_adr'].apply(gap_size_bucket)

# For each ticker, build a sorted list of all available dates from baseline_1min + raw_1min
print("Building date index per ticker...", file=sys.stderr)
ticker_dates = {}

# Get all available dates from raw_1min (covers gapper days)
for ticker in os.listdir("data/raw_1min"):
    tp = f"data/raw_1min/{ticker}"
    if not os.path.isdir(tp):
        continue
    dates = sorted([f.replace('.parquet', '') for f in os.listdir(tp) if f.endswith('.parquet')])
    ticker_dates.setdefault(ticker, set()).update(dates)

# Add baseline_1min dates (non-gapper days)
for ticker in os.listdir("data/baseline_1min"):
    tp = f"data/baseline_1min/{ticker}"
    if not os.path.isdir(tp):
        continue
    dates = sorted([f.replace('.parquet', '') for f in os.listdir(tp) if f.endswith('.parquet')])
    ticker_dates.setdefault(ticker, set()).update(dates)

# Convert to sorted lists
for t in ticker_dates:
    ticker_dates[t] = sorted(ticker_dates[t])

print(f"Built date index for {len(ticker_dates)} tickers", file=sys.stderr)

# Find next-day data
next_day_data = []

for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Finding next-day", file=sys.stderr):
    ticker = row['ticker']
    date = str(row['date'])
    gap_dir = row['gap_direction']
    adr = row['adr_10']

    if ticker not in ticker_dates:
        continue

    dates = ticker_dates[ticker]
    try:
        idx = dates.index(date)
    except ValueError:
        continue

    if idx + 1 >= len(dates):
        continue

    next_date = dates[idx + 1]

    # Try to load next day's 1-min data (raw_1min first, then baseline)
    next_path = f"data/raw_1min/{ticker}/{next_date}.parquet"
    if not os.path.exists(next_path):
        next_path = f"data/baseline_1min/{ticker}/{next_date}.parquet"
    if not os.path.exists(next_path):
        continue

    try:
        ndf = pd.read_parquet(next_path)
    except:
        continue

    if 'session' in ndf.columns:
        ndf = ndf[ndf['session'] == 'rth']

    if len(ndf) < 10:
        continue

    next_open = ndf['open'].iloc[0]
    next_close = ndf['close'].iloc[-1]
    next_high = ndf['high'].max()
    next_low = ndf['low'].min()

    gap_day_close = row['rth_close']
    if gap_day_close <= 0 or next_open <= 0:
        continue

    next_day_return = (next_close - next_open) / next_open
    next_day_gap = (next_open - gap_day_close) / gap_day_close
    next_day_range = (next_high - next_low) / adr if adr > 0 else np.nan

    next_day_data.append({
        'ticker': ticker,
        'date': date,
        'gap_dir': gap_dir,
        'close_loc': row['close_loc'],
        'close_vs_open_adr': row['close_vs_open_adr'],
        'gap_bucket': row['gap_bucket'],
        'next_day_return': next_day_return,
        'next_day_gap': next_day_gap,
        'next_day_range': next_day_range,
        'day_of_week': row.get('day_of_week', ''),
    })

nd_df = pd.DataFrame(next_day_data)
print(f"Next-day data: {len(nd_df)} days", file=sys.stderr)

out = []
out.append("=" * 80)
out.append("CHEAT SHEET -- KATEGORIE 6: FOLLOW-THROUGH TAG 2 + WOCHENTAG")
out.append("=" * 80)
out.append(f"Datenbasis: {len(nd_df)} Gapper mit Next-Day-Daten")
out.append("")

if len(nd_df) > 0:
    out.append("=" * 70)
    out.append("FRAGE 6.2: Follow-Through am naechsten Tag")
    out.append("=" * 70)

    for gd in ['up', 'down']:
        label = 'GapUp' if gd == 'up' else 'GapDown'
        gsub = nd_df[nd_df['gap_dir'] == gd]
        out.append(f"\n--- {label} (N={len(gsub)}) ---")

        if len(gsub) < 30:
            out.append("  LOW N")
            continue

        # Overall follow-through
        nd_ret = gsub['next_day_return'].dropna()
        if gd == 'up':
            ft = (nd_ret > 0).mean()
        else:
            ft = (nd_ret < 0).mean()
        lo, hi = bootstrap_ci((nd_ret > 0).astype(float).values if gd == 'up' else (nd_ret < 0).astype(float).values)
        out.append(f"  Gesamt Follow-Through: {fmt_pct(ft, lo, hi)}")
        out.append(f"  NextDay Median Return: {nd_ret.median()*100:.2f}%")

        # By close location
        out.append(f"\n  Breakdown nach Close-Location:")
        for blabel, lo_cl, hi_cl in [
            ('Starker Close (CL>0.7)', 0.7, 1.01),
            ('Mittlerer Close (0.3-0.7)', 0.3, 0.7),
            ('Schwacher Close (CL<0.3)', -0.01, 0.3),
        ]:
            bsub = gsub[(gsub['close_loc'] >= lo_cl) & (gsub['close_loc'] < hi_cl)]
            if len(bsub) < 30:
                out.append(f"    {blabel}: N={len(bsub)} (LOW N)")
                continue

            nd_ret_b = bsub['next_day_return'].dropna()
            if gd == 'up':
                follow = (nd_ret_b > 0).mean()
            else:
                follow = (nd_ret_b < 0).mean()
            lo_b, hi_b = bootstrap_ci(
                (nd_ret_b > 0).astype(float).values if gd == 'up' else (nd_ret_b < 0).astype(float).values
            )
            nd_gap_b = bsub['next_day_gap'].dropna()
            out.append(f"    {blabel}: Follow-Through={fmt_pct(follow, lo_b, hi_b)}, "
                       f"NextDay Median={nd_ret_b.median()*100:.2f}%, "
                       f"Gap Cont={nd_gap_b.median()*100:.2f}%, N={len(bsub)}")

        # By day drift
        out.append(f"\n  Breakdown nach Tages-Drift (Close-Open in ADR):")
        for blabel, lo_d, hi_d in [
            ('Starker Drift mit Gap (>+0.5 ADR)', 0.5, 99),
            ('Moderater Drift (+0.1 bis +0.5)', 0.1, 0.5),
            ('Flach (-0.1 bis +0.1)', -0.1, 0.1),
            ('Gegen Gap (-0.5 bis -0.1)', -0.5, -0.1),
            ('Stark gegen Gap (<-0.5 ADR)', -99, -0.5),
        ]:
            if gd == 'up':
                bsub = gsub[(gsub['close_vs_open_adr'] >= lo_d) & (gsub['close_vs_open_adr'] < hi_d)]
            else:
                bsub = gsub[(gsub['close_vs_open_adr'] >= -hi_d) & (gsub['close_vs_open_adr'] < -lo_d)]

            if len(bsub) < 20:
                out.append(f"    {blabel}: N={len(bsub)} (LOW N)")
                continue

            nd_ret_b = bsub['next_day_return'].dropna()
            if gd == 'up':
                follow = (nd_ret_b > 0).mean()
            else:
                follow = (nd_ret_b < 0).mean()
            out.append(f"    {blabel}: Follow-Through={follow*100:.0f}%, "
                       f"NextDay Median={nd_ret_b.median()*100:.2f}%, N={len(bsub)}")

    # Weekday effect
    out.append("\n\n" + "=" * 70)
    out.append("BONUS: Wochentag-Effekt")
    out.append("=" * 70)

    for gd in ['up', 'down']:
        label = 'GapUp' if gd == 'up' else 'GapDown'
        gsub = nd_df[nd_df['gap_dir'] == gd]
        out.append(f"\n  {label}:")

        for dow in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
            dsub = gsub[gsub['day_of_week'] == dow]
            if len(dsub) < 30:
                out.append(f"    {dow}: N={len(dsub)} (LOW N)")
                continue

            cvo = dsub['close_vs_open_adr'].dropna()
            nd_ret = dsub['next_day_return'].dropna()
            out.append(f"    {dow}: GapDay Drift={cvo.median():.3f} ADR, "
                       f"NextDay Median={nd_ret.median()*100:.2f}%, N={len(dsub)}")
else:
    out.append("KEINE Next-Day-Daten gefunden!")

report = "\n".join(out)
with open("results/cheat_cat6_followthrough.txt", "w", encoding="utf-8") as f:
    f.write(report)

print(report)
print(f"\nResults saved to results/cheat_cat6_followthrough.txt", file=sys.stderr)
