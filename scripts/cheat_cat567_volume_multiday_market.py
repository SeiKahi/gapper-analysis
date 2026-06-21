"""
Cheat Sheet Categories 5-7: Institutional Behavior, Multi-Day Context, Market Context
"""
import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm

# -- helpers --
def bootstrap_ci(data, n_boot=1000, ci=0.95):
    if len(data) < 5:
        return np.nan, np.nan
    rng = np.random.default_rng(42)
    stats = np.array([np.mean(rng.choice(data, size=len(data), replace=True)) for _ in range(n_boot)])
    return np.percentile(stats, (1-ci)/2*100), np.percentile(stats, (1+ci)/2*100)

def fmt_pct(val, ci_lo, ci_hi):
    return f"{val*100:.1f}% (CI: {ci_lo*100:.0f}-{ci_hi*100:.0f}%)"

# -- load metadata --
print("Loading metadata...", file=sys.stderr)
meta = pd.read_parquet("data/metadata/metadata_master.parquet")
meta = meta[(meta['date'] <= '2023-12-31') & (meta['adr_10'].notna())].copy()
meta['adr_pct'] = meta['adr_10'] / meta['today_open'] * 100
meta['gap_size_in_adr'] = meta['gap_size_in_adr'].fillna(
    (meta['today_open'] - meta['prev_close']).abs() / meta['adr_10']
)
meta['day_range_adr'] = (meta['rth_high'] - meta['rth_low']) / meta['adr_10']
meta['close_loc'] = (meta['rth_close'] - meta['rth_low']) / (meta['rth_high'] - meta['rth_low'])
meta['close_vs_open_adr'] = meta['close_vs_open_adr']

def gap_size_bucket(v):
    if v < 1: return '<1x'
    if v < 2: return '1-2x'
    if v < 3: return '2-3x'
    return '>3x'
meta['gap_bucket'] = meta['gap_size_in_adr'].apply(gap_size_bucket)

print(f"Halfte 1: {len(meta)} gapper days", file=sys.stderr)

# -- Process each day for volume-based analysis --
day_results = []

for idx, row in tqdm(meta.iterrows(), total=len(meta), desc="Processing", file=sys.stderr):
    ticker = row['ticker']
    date = row['date']
    gap_dir = row['gap_direction']
    adr = row['adr_10']

    raw_path = f"data/raw_1min/{ticker}/{date}.parquet"
    if not os.path.exists(raw_path):
        continue

    try:
        raw_df = pd.read_parquet(raw_path)
    except:
        continue

    if 'session' in raw_df.columns:
        raw_df = raw_df[raw_df['session'] == 'rth'].copy()

    if len(raw_df) < 30 or adr <= 0:
        continue

    raw_df = raw_df.sort_values('time_et').reset_index(drop=True)

    def time_to_min(t):
        if isinstance(t, str):
            parts = t.split(':')
            return (int(parts[0]) - 9) * 60 + (int(parts[1]) - 30)
        return np.nan

    raw_df['min_from_open'] = raw_df['time_et'].apply(time_to_min)
    raw_df = raw_df[raw_df['min_from_open'] >= 0].reset_index(drop=True)

    if len(raw_df) < 20:
        continue

    total_vol = raw_df['volume'].sum()
    if total_vol == 0:
        continue

    day_result = {
        'ticker': ticker, 'date': date, 'gap_dir': gap_dir,
        'adr': adr, 'gap_bucket': row['gap_bucket'],
    }

    # -- Q5.1: Volume Concentration --
    first_30 = raw_df[raw_df['min_from_open'] <= 30]
    vol_first_30 = first_30['volume'].sum()
    day_result['vol_pct_first_30'] = vol_first_30 / total_vol

    # Volume distribution evenness (using coefficient of variation of 30min blocks)
    block_vols = []
    for start in range(0, 360, 30):
        block = raw_df[(raw_df['min_from_open'] >= start) & (raw_df['min_from_open'] < start + 30)]
        block_vols.append(block['volume'].sum())
    if len(block_vols) > 3 and np.mean(block_vols) > 0:
        cv = np.std(block_vols) / np.mean(block_vols)
        day_result['vol_cv'] = cv  # higher = more concentrated
        day_result['vol_even'] = cv < 0.5  # relatively even

    # -- Q5.3: Transactions vs Volume --
    if 'transactions' in raw_df.columns:
        total_txn = raw_df['transactions'].sum()
        if total_txn > 0 and total_vol > 0:
            day_result['avg_size'] = total_vol / total_txn  # average shares per trade
            first_hour = raw_df[raw_df['min_from_open'] <= 60]
            if len(first_hour) > 10:
                fh_vol = first_hour['volume'].sum()
                fh_txn = first_hour['transactions'].sum()
                if fh_txn > 0:
                    day_result['avg_size_first_hour'] = fh_vol / fh_txn

    # -- Close location and day behavior (needed for breakdowns) --
    day_high = raw_df['high'].max()
    day_low = raw_df['low'].min()
    if day_high > day_low:
        day_result['close_loc'] = (raw_df['close'].iloc[-1] - day_low) / (day_high - day_low)
    day_result['day_range_adr'] = (day_high - day_low) / adr
    day_result['close_vs_open_adr'] = (raw_df['close'].iloc[-1] - raw_df['open'].iloc[0]) / adr

    day_results.append(day_result)

print(f"\nProcessed {len(day_results)} days", file=sys.stderr)
df = pd.DataFrame(day_results)

# -- Generate Report --
out = []
out.append("=" * 80)
out.append("CHEAT SHEET -- KATEGORIEN 5-7: VOLUME, MULTI-DAY, MARKT-KONTEXT")
out.append("=" * 80)
out.append(f"Datenbasis: Halfte 1, N={len(df)}")
out.append("")

# ===== CATEGORY 5: INSTITUTIONAL BEHAVIOR =====
out.append("=" * 70)
out.append("KATEGORIE 5: INSTITUTIONELLES VERHALTEN (PREIS-VOLUMEN)")
out.append("=" * 70)

# Q5.1: Volume Concentration
out.append("\nFRAGE 5.1: Volumen-Konzentration in ersten 30min")
for gd in ['up', 'down']:
    label = 'GapUp' if gd == 'up' else 'GapDown'
    gsub = df[(df['gap_dir'] == gd) & (df['vol_pct_first_30'].notna())]
    out.append(f"\n  {label} (N={len(gsub)}):")

    # Bucket by volume concentration
    for blabel, lo_pct, hi_pct in [
        ('>50% Vol in first 30min', 0.50, 1.0),
        ('40-50%', 0.40, 0.50),
        ('30-40%', 0.30, 0.40),
        ('<30%', 0.0, 0.30),
    ]:
        bsub = gsub[(gsub['vol_pct_first_30'] >= lo_pct) & (gsub['vol_pct_first_30'] < hi_pct)]
        if len(bsub) < 20:
            out.append(f"    {blabel}: N={len(bsub)} (LOW N)")
            continue
        cvo = bsub['close_vs_open_adr'].dropna()
        cl = bsub['close_loc'].dropna()
        dr = bsub['day_range_adr'].dropna()
        out.append(f"    {blabel}: Drift={cvo.median():.3f} ADR, DayRange={dr.median():.2f} ADR, "
                   f"Close-Loc={cl.median():.2f}, N={len(bsub)}")

    # Evenly distributed volume
    even_sub = gsub[gsub['vol_even'] == True] if 'vol_even' in gsub.columns else pd.DataFrame()
    uneven_sub = gsub[gsub['vol_even'] == False] if 'vol_even' in gsub.columns else pd.DataFrame()
    if len(even_sub) > 20 and len(uneven_sub) > 20:
        out.append(f"\n    Gleichmaessig verteilt (CV<0.5): Drift={even_sub['close_vs_open_adr'].median():.3f}, "
                   f"DayRange={even_sub['day_range_adr'].median():.2f}, N={len(even_sub)}")
        out.append(f"    Ungleichmaessig (CV>=0.5): Drift={uneven_sub['close_vs_open_adr'].median():.3f}, "
                   f"DayRange={uneven_sub['day_range_adr'].median():.2f}, N={len(uneven_sub)}")

# Q5.3: Transactions vs Volume (Average trade size)
out.append("\n\nFRAGE 5.3: Durchschnittliche Order-Groesse (Transactions vs Volume)")
avg_size_sub = df[df['avg_size'].notna()].copy()
if len(avg_size_sub) > 100:
    # Categorize by average trade size
    p25 = avg_size_sub['avg_size'].quantile(0.25)
    p75 = avg_size_sub['avg_size'].quantile(0.75)

    for gd in ['up', 'down']:
        label = 'GapUp' if gd == 'up' else 'GapDown'
        gsub = avg_size_sub[avg_size_sub['gap_dir'] == gd]
        out.append(f"\n  {label}:")

        for blabel, mask in [
            (f'Kleine Orders (avg<{p25:.0f} shares)', gsub['avg_size'] < p25),
            (f'Mittlere Orders ({p25:.0f}-{p75:.0f})', (gsub['avg_size'] >= p25) & (gsub['avg_size'] < p75)),
            (f'Grosse Orders (avg>{p75:.0f} shares)', gsub['avg_size'] >= p75),
        ]:
            bsub = gsub[mask]
            if len(bsub) < 20:
                out.append(f"    {blabel}: N={len(bsub)} (LOW N)")
                continue
            cvo = bsub['close_vs_open_adr'].dropna()
            cl = bsub['close_loc'].dropna()
            out.append(f"    {blabel}: Drift={cvo.median():.3f} ADR, Close-Loc={cl.median():.2f}, N={len(bsub)}")

# ===== CATEGORY 6: MULTI-DAY CONTEXT =====
out.append("\n\n" + "=" * 70)
out.append("KATEGORIE 6: MULTI-DAY KONTEXT")
out.append("=" * 70)

# Q6.1: RSI context (already done in cat3 as bonus, but add here for completeness)
out.append("\nFRAGE 6.1: RSI-14 + 52W-High -> schon in Kat 3 beantwortet (siehe dort)")

# Q6.2: Follow-Through Tag 2
out.append("\nFRAGE 6.2: Follow-Through am naechsten Tag")
# Use daily data to check next-day return
print("Loading daily data for next-day analysis...", file=sys.stderr)

# Build a dict of daily data per ticker
daily_cache = {}
daily_dir = "data/daily"
if os.path.isdir(daily_dir):
    for f in os.listdir(daily_dir):
        if f.endswith('.parquet'):
            ticker_name = f.replace('.parquet', '')
            try:
                ddf = pd.read_parquet(os.path.join(daily_dir, f))
                ddf = ddf.sort_values('date').reset_index(drop=True)
                daily_cache[ticker_name] = ddf
            except:
                pass
elif os.path.isdir(daily_dir):
    # Maybe daily is a directory of directories
    for ticker_name in os.listdir(daily_dir):
        tp = os.path.join(daily_dir, ticker_name)
        if os.path.isdir(tp):
            files = [f for f in os.listdir(tp) if f.endswith('.parquet')]
            if files:
                try:
                    frames = [pd.read_parquet(os.path.join(tp, f)) for f in files]
                    ddf = pd.concat(frames).sort_values('date').reset_index(drop=True)
                    daily_cache[ticker_name] = ddf
                except:
                    pass

print(f"Loaded daily data for {len(daily_cache)} tickers", file=sys.stderr)

# Match next-day returns
next_day_data = []
for _, row in meta.iterrows():
    ticker = row['ticker']
    date = row['date']
    if ticker not in daily_cache:
        continue
    ddf = daily_cache[ticker]
    date_mask = ddf['date'] == date
    if not date_mask.any():
        # Try string match
        ddf_dates = ddf['date'].astype(str)
        date_mask = ddf_dates == str(date)
        if not date_mask.any():
            continue

    idx = ddf[date_mask].index[0]
    if idx + 1 >= len(ddf):
        continue

    next_row = ddf.iloc[idx + 1]
    gap_day_close = row['rth_close']
    next_open = next_row['open']
    next_close = next_row['close']

    if gap_day_close > 0:
        next_day_return = (next_close - next_open) / next_open
        next_day_gap = (next_open - gap_day_close) / gap_day_close

        close_loc = row.get('close_loc', np.nan)
        if pd.isna(close_loc) and (row['rth_high'] - row['rth_low']) > 0:
            close_loc = (row['rth_close'] - row['rth_low']) / (row['rth_high'] - row['rth_low'])

        next_day_data.append({
            'gap_dir': row['gap_direction'],
            'close_loc': close_loc,
            'close_vs_open_adr': row['close_vs_open_adr'],
            'next_day_return': next_day_return,
            'next_day_gap': next_day_gap,
            'gap_bucket': row.get('gap_bucket', 'unknown') if 'gap_bucket' in row.index else gap_size_bucket(row['gap_size_in_adr']),
        })

nd_df = pd.DataFrame(next_day_data)
print(f"Next-day data: {len(nd_df)} days", file=sys.stderr)

if len(nd_df) > 100:
    for gd in ['up', 'down']:
        label = 'GapUp' if gd == 'up' else 'GapDown'
        gsub = nd_df[nd_df['gap_dir'] == gd]
        out.append(f"\n  {label} (N={len(gsub)}):")

        # By close location (strong close = >0.7, weak close = <0.3)
        for blabel, lo_cl, hi_cl in [
            ('Starker Close (CL>0.7)', 0.7, 1.01),
            ('Mittlerer Close (0.3-0.7)', 0.3, 0.7),
            ('Schwacher Close (CL<0.3)', -0.01, 0.3),
        ]:
            bsub = gsub[(gsub['close_loc'] >= lo_cl) & (gsub['close_loc'] < hi_cl)]
            if len(bsub) < 30:
                out.append(f"    {blabel}: N={len(bsub)} (LOW N)")
                continue

            nd_ret = bsub['next_day_return'].dropna()
            nd_gap = bsub['next_day_gap'].dropna()
            follow_through = (nd_ret > 0).mean() if gd == 'up' else (nd_ret < 0).mean()
            out.append(f"    {blabel}: NextDay Follow-Through={follow_through*100:.0f}%, "
                       f"NextDay Median Return={nd_ret.median()*100:.2f}%, "
                       f"NextDay Median Gap={nd_gap.median()*100:.2f}%, N={len(bsub)}")

# ===== CATEGORY 7: MARKET CONTEXT =====
out.append("\n\n" + "=" * 70)
out.append("KATEGORIE 7: MARKT-KONTEXT")
out.append("=" * 70)

# Q7.1: SPY Return
out.append("\nFRAGE 7.1: SPY-Tagesreturn -> Gapper-Verhalten")
spy_sub = meta[meta['spy_return_day'].notna()]

for gd in ['up', 'down']:
    label = 'GapUp' if gd == 'up' else 'GapDown'
    gsub = spy_sub[spy_sub['gap_direction'] == gd]
    out.append(f"\n  {label}:")

    spy_buckets = [
        ('SPY stark rot (<-1%)', gsub['spy_return_day'] < -1),
        ('SPY leicht rot (-1% bis 0%)', (gsub['spy_return_day'] >= -1) & (gsub['spy_return_day'] < 0)),
        ('SPY leicht gruen (0% bis +1%)', (gsub['spy_return_day'] >= 0) & (gsub['spy_return_day'] < 1)),
        ('SPY stark gruen (>+1%)', gsub['spy_return_day'] >= 1),
    ]

    for blabel, mask in spy_buckets:
        bsub = gsub[mask]
        if len(bsub) < 30:
            out.append(f"    {blabel}: N={len(bsub)} (LOW N)")
            continue
        cvo = bsub['close_vs_open_adr'].dropna()
        filled = bsub['gap_filled'].astype(float).values
        cl = (bsub['rth_close'] - bsub['rth_low']) / (bsub['rth_high'] - bsub['rth_low'])
        cl = cl.dropna()
        out.append(f"    {blabel}: Fill={np.mean(filled)*100:.0f}%, Drift={cvo.median():.3f} ADR, "
                   f"Close-Loc={cl.median():.2f}, N={len(bsub)}")

# Q7.2: Sector
out.append("\n\nFRAGE 7.2: Sektor-Analyse")
sector_counts = meta['sector'].value_counts()

for gd in ['up', 'down']:
    label = 'GapUp' if gd == 'up' else 'GapDown'
    gsub = meta[meta['gap_direction'] == gd]
    out.append(f"\n  {label}:")

    for sector in sector_counts.index[:15]:  # Top 15 sectors
        ssub = gsub[gsub['sector'] == sector]
        if len(ssub) < 30:
            continue
        cvo = ssub['close_vs_open_adr'].dropna()
        filled = ssub['gap_filled'].astype(float).values
        mae = ssub['max_adverse_adr'].dropna()
        mex = ssub['max_extension_adr'].dropna()
        out.append(f"    {sector}: Fill={np.mean(filled)*100:.0f}%, Drift={cvo.median():.3f} ADR, "
                   f"MaxExt={mex.median():.3f}, MaxAdv={mae.median():.3f}, N={len(ssub)}")

# -- Day of Week bonus --
out.append("\n\nBONUS 7.3: Wochentag-Effekt")
for gd in ['up', 'down']:
    label = 'GapUp' if gd == 'up' else 'GapDown'
    gsub = meta[meta['gap_direction'] == gd]
    out.append(f"\n  {label}:")

    for dow in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
        dsub = gsub[gsub['day_of_week'] == dow]
        if len(dsub) < 30:
            continue
        cvo = dsub['close_vs_open_adr'].dropna()
        filled = dsub['gap_filled'].astype(float).values
        out.append(f"    {dow}: Fill={np.mean(filled)*100:.0f}%, Drift={cvo.median():.3f} ADR, N={len(dsub)}")

# -- Write output --
report = "\n".join(out)
with open("results/cheat_cat567_volume_multiday_market.txt", "w", encoding="utf-8") as f:
    f.write(report)

print(report)
print(f"\nResults saved to results/cheat_cat567_volume_multiday_market.txt", file=sys.stderr)
