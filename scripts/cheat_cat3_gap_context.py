"""
Cheat Sheet Category 3: Gap-Kontext und Vorgeschichte
Questions 3.1-3.4: Gap fill rates, prior performance, RVOL context, catalyst type
"""
import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm

# ── helpers ──────────────────────────────────────────────────────────
def bootstrap_ci(data, n_boot=1000, ci=0.95):
    if len(data) < 5:
        return np.nan, np.nan
    rng = np.random.default_rng(42)
    stats = np.array([np.mean(rng.choice(data, size=len(data), replace=True)) for _ in range(n_boot)])
    return np.percentile(stats, (1-ci)/2*100), np.percentile(stats, (1+ci)/2*100)

def fmt_pct(val, ci_lo, ci_hi):
    return f"{val*100:.1f}% (CI: {ci_lo*100:.0f}-{ci_hi*100:.0f}%)"

def fmt_row(label, arr):
    if len(arr) == 0:
        return f"  {label}: N=0"
    mean = np.mean(arr)
    lo, hi = bootstrap_ci(arr)
    return f"  {label}: {fmt_pct(mean, lo, hi)}, N={len(arr)}"

# ── load metadata ────────────────────────────────────────────────────
print("Loading metadata...", file=sys.stderr)
meta = pd.read_parquet("data/metadata/metadata_master.parquet")
meta = meta[(meta['date'] <= '2023-12-31') & (meta['adr_10'].notna())].copy()

# Derived columns
meta['adr_pct'] = meta['adr_10'] / meta['today_open'] * 100
meta['gap_abs'] = (meta['today_open'] - meta['prev_close']).abs()
meta['gap_size_in_adr'] = meta['gap_size_in_adr'].fillna(meta['gap_abs'] / meta['adr_10'])
meta['gap_pct_abs'] = meta['gap_pct'].abs()

# Day range from metadata
meta['day_range'] = meta['rth_high'] - meta['rth_low']
meta['day_return'] = (meta['rth_close'] - meta['today_open']) / meta['today_open']

# Gap fill: for GapUp, close >= prev_close means NOT filled (actually price > prev close)
# Actually gap_filled is already in metadata
# close_vs_open_adr: how much price moved from open, in ADR units

def gap_size_bucket(v):
    if v < 1: return '<1x'
    if v < 2: return '1-2x'
    if v < 3: return '2-3x'
    return '>3x'

def adr_pct_bucket(v):
    if v < 3: return '<3%'
    if v < 5: return '3-5%'
    if v < 10: return '5-10%'
    return '>10%'

meta['gap_bucket'] = meta['gap_size_in_adr'].apply(gap_size_bucket)
meta['adr_pct_bucket'] = meta['adr_pct'].apply(adr_pct_bucket)

print(f"Halfte 1: {len(meta)} gapper days", file=sys.stderr)

# ── Report ──────────────────────────────────────────────────────────
out = []
out.append("=" * 80)
out.append("CHEAT SHEET — KATEGORIE 3: GAP-KONTEXT UND VORGESCHICHTE")
out.append("=" * 80)
out.append(f"Datenbasis: Halfte 1 (bis 2023-12-31), N={len(meta)}")
out.append("")

# ═══════════════════════════════════════════════════════════════════
# Q3.1: Gap Fill Rates
# ═══════════════════════════════════════════════════════════════════
out.append("=" * 70)
out.append("FRAGE 3.1: Gap-Fill Raten und Fade-Verhalten")
out.append("=" * 70)

for gd in ['up', 'down']:
    label = 'GapUp' if gd == 'up' else 'GapDown'
    gsub = meta[meta['gap_direction'] == gd]
    out.append(f"\n--- {label} (N={len(gsub)}) ---")

    # Overall gap fill rate
    filled = gsub['gap_filled'].astype(float).values
    out.append(fmt_row("Gap komplett gefuellt (Close >= PrevClose)", filled))

    # Median gap fill time
    fill_times = gsub[gsub['gap_filled'] == True]['gap_fill_time_minutes'].dropna()
    if len(fill_times) > 0:
        out.append(f"  Median Gap-Fill-Zeit: {fill_times.median():.0f}min, Mean: {fill_times.mean():.0f}min")

    # By gap size
    out.append(f"\n  Breakdown nach Gap-Groesse:")
    for bucket in ['<1x', '1-2x', '2-3x', '>3x']:
        bsub = gsub[gsub['gap_bucket'] == bucket]
        if len(bsub) < 20:
            out.append(f"    {bucket}: N={len(bsub)} (LOW N)")
            continue
        fill_rate = bsub['gap_filled'].astype(float).values
        mean_f = np.mean(fill_rate)
        lo, hi = bootstrap_ci(fill_rate)
        fill_t = bsub[bsub['gap_filled']==True]['gap_fill_time_minutes'].dropna()
        t_str = f", Median Fill-Zeit: {fill_t.median():.0f}min" if len(fill_t) > 5 else ""
        out.append(f"    {bucket}: Fill-Rate={fmt_pct(mean_f, lo, hi)}, N={len(bsub)}{t_str}")

    # How far does gap fade typically? Use close_vs_open_adr and gap direction
    out.append(f"\n  Wie weit fadet der Gap? (Tages-Close vs Open, in ADR)")
    # For GapUp: negative close_vs_open = fading
    # For GapDown: positive close_vs_open = fading (bouncing)
    cvo = gsub['close_vs_open_adr'].dropna()
    if len(cvo) > 10:
        if gd == 'up':
            out.append(f"    Mean Tages-Return (Close-Open)/ADR: {cvo.mean():.3f}")
            out.append(f"    Median: {cvo.median():.3f}")
            faders = cvo[cvo < 0]
            out.append(f"    Davon negativ (fadet): {len(faders)/len(cvo)*100:.0f}% (N={len(faders)})")
            if len(faders) > 5:
                out.append(f"    Median Fade-Ausmass: {faders.median():.3f} ADR")
        else:
            out.append(f"    Mean Tages-Return (Close-Open)/ADR: {cvo.mean():.3f}")
            out.append(f"    Median: {cvo.median():.3f}")
            faders = cvo[cvo > 0]  # positive = bouncing = fading the gap
            out.append(f"    Davon positiv (bounced/fadet): {len(faders)/len(cvo)*100:.0f}% (N={len(faders)})")
            if len(faders) > 5:
                out.append(f"    Median Bounce-Ausmass: {faders.median():.3f} ADR")

    # Max adverse excursion (max move against gap direction)
    mae = gsub['max_adverse_adr'].dropna()
    if len(mae) > 10:
        out.append(f"    Max Adverse Excursion (gegen Gap): Median={mae.median():.3f} ADR, P75={mae.quantile(0.75):.3f}")

    # Max extension (max move with gap direction)
    mex = gsub['max_extension_adr'].dropna()
    if len(mex) > 10:
        out.append(f"    Max Extension (mit Gap): Median={mex.median():.3f} ADR, P75={mex.quantile(0.75):.3f}")

# ═══════════════════════════════════════════════════════════════════
# Q3.2: Prior Performance + Gap
# ═══════════════════════════════════════════════════════════════════
out.append("\n" + "=" * 70)
out.append("FRAGE 3.2: Vorherige Performance + Gap-Verhalten")
out.append("=" * 70)

# Prior 10d return buckets
for gd in ['up', 'down']:
    label = 'GapUp' if gd == 'up' else 'GapDown'
    gsub = meta[(meta['gap_direction'] == gd) & (meta['prior_return_10d'].notna())]
    out.append(f"\n--- {label}: Prior 10d Return -> Tagesverhalten ---")

    if gd == 'up':
        buckets = [
            ('Prior 10d < -10%', gsub['prior_return_10d'] < -10),
            ('Prior 10d -10% bis 0%', (gsub['prior_return_10d'] >= -10) & (gsub['prior_return_10d'] < 0)),
            ('Prior 10d 0% bis +10%', (gsub['prior_return_10d'] >= 0) & (gsub['prior_return_10d'] < 10)),
            ('Prior 10d +10% bis +20%', (gsub['prior_return_10d'] >= 10) & (gsub['prior_return_10d'] < 20)),
            ('Prior 10d > +20%', gsub['prior_return_10d'] >= 20),
        ]
    else:
        buckets = [
            ('Prior 10d > +20%', gsub['prior_return_10d'] >= 20),
            ('Prior 10d +10% bis +20%', (gsub['prior_return_10d'] >= 10) & (gsub['prior_return_10d'] < 20)),
            ('Prior 10d 0% bis +10%', (gsub['prior_return_10d'] >= 0) & (gsub['prior_return_10d'] < 10)),
            ('Prior 10d -10% bis 0%', (gsub['prior_return_10d'] >= -10) & (gsub['prior_return_10d'] < 0)),
            ('Prior 10d < -10%', gsub['prior_return_10d'] < -10),
        ]

    for blabel, mask in buckets:
        bsub = gsub[mask]
        if len(bsub) < 20:
            out.append(f"  {blabel}: N={len(bsub)} (LOW N)")
            continue
        cvo = bsub['close_vs_open_adr'].dropna()
        filled = bsub['gap_filled'].astype(float).values
        fill_rate = np.mean(filled)
        flo, fhi = bootstrap_ci(filled)
        close_loc_raw = (bsub['rth_close'] - bsub['rth_low']) / (bsub['rth_high'] - bsub['rth_low'])
        close_loc = close_loc_raw.dropna()
        cl_str = f", Close-Location Median={close_loc.median():.2f}" if len(close_loc) > 5 else ""
        out.append(f"  {blabel}: Fill={fmt_pct(fill_rate, flo, fhi)}, Drift={cvo.median():.3f} ADR, N={len(bsub)}{cl_str}")

# Dist from 52w high
out.append(f"\n--- Distanz vom 52W-Hoch -> Tagesverhalten ---")
for gd in ['up', 'down']:
    label = 'GapUp' if gd == 'up' else 'GapDown'
    gsub = meta[(meta['gap_direction'] == gd) & (meta['dist_from_52w_high'].notna())]
    out.append(f"\n  {label}:")
    buckets = [
        ('Nahe ATH (>-5%)', gsub['dist_from_52w_high'] > -5),
        ('-5% bis -20%', (gsub['dist_from_52w_high'] >= -20) & (gsub['dist_from_52w_high'] <= -5)),
        ('-20% bis -50%', (gsub['dist_from_52w_high'] >= -50) & (gsub['dist_from_52w_high'] < -20)),
        ('Weit unter ATH (<-50%)', gsub['dist_from_52w_high'] < -50),
    ]
    for blabel, mask in buckets:
        bsub = gsub[mask]
        if len(bsub) < 20:
            out.append(f"    {blabel}: N={len(bsub)} (LOW N)")
            continue
        cvo = bsub['close_vs_open_adr'].dropna()
        filled = bsub['gap_filled'].astype(float).values
        out.append(f"    {blabel}: Fill={np.mean(filled)*100:.0f}%, Drift={cvo.median():.3f} ADR, N={len(bsub)}")

# ═══════════════════════════════════════════════════════════════════
# Q3.3: RVOL Context
# ═══════════════════════════════════════════════════════════════════
out.append("\n" + "=" * 70)
out.append("FRAGE 3.3: RVOL-Kontext")
out.append("=" * 70)

# Use rvol_at_time_30min from metadata (already computed)
rvol_col = 'rvol_at_time_30min'
for gd in ['up', 'down']:
    label = 'GapUp' if gd == 'up' else 'GapDown'
    gsub = meta[(meta['gap_direction'] == gd) & (meta[rvol_col].notna())]
    out.append(f"\n--- {label}: RVOL at 30min -> Tagesverhalten ---")

    rvol_buckets = [
        ('<1x', gsub[rvol_col] < 1),
        ('1-2x', (gsub[rvol_col] >= 1) & (gsub[rvol_col] < 2)),
        ('2-3x', (gsub[rvol_col] >= 2) & (gsub[rvol_col] < 3)),
        ('3-5x', (gsub[rvol_col] >= 3) & (gsub[rvol_col] < 5)),
        ('>5x', gsub[rvol_col] >= 5),
    ]

    for blabel, mask in rvol_buckets:
        bsub = gsub[mask]
        if len(bsub) < 20:
            out.append(f"  RVOL {blabel}: N={len(bsub)} (LOW N)")
            continue

        cvo = bsub['close_vs_open_adr'].dropna()
        filled = bsub['gap_filled'].astype(float).values
        fill_rate = np.mean(filled)
        day_range = (bsub['rth_high'] - bsub['rth_low']) / bsub['adr_10']
        range_str = f", DayRange={day_range.median():.2f} ADR" if len(day_range.dropna()) > 5 else ""
        out.append(f"  RVOL {blabel}: Fill={fill_rate*100:.0f}%, Drift={cvo.median():.3f} ADR, N={len(bsub)}{range_str}")

# RVOL decay: compare rvol_at_time_5min vs rvol_at_time_60min
out.append(f"\n--- RVOL-Verlauf: Steigend vs Fallend ---")
rvol_sub = meta[(meta['rvol_at_time_5min'].notna()) & (meta['rvol_at_time_60min'].notna())].copy()
rvol_sub['rvol_change'] = rvol_sub['rvol_at_time_60min'] / rvol_sub['rvol_at_time_5min'].clip(lower=0.1)
rvol_sub['rvol_trend'] = np.where(rvol_sub['rvol_change'] > 1.2, 'rising',
                          np.where(rvol_sub['rvol_change'] < 0.8, 'falling', 'stable'))

for trend in ['rising', 'falling', 'stable']:
    tsub = rvol_sub[rvol_sub['rvol_trend'] == trend]
    if len(tsub) < 30:
        out.append(f"  RVOL {trend}: N={len(tsub)} (LOW N)")
        continue
    cvo = tsub['close_vs_open_adr'].dropna()
    filled = tsub['gap_filled'].astype(float).values
    day_range = (tsub['rth_high'] - tsub['rth_low']) / tsub['adr_10']
    out.append(f"  RVOL {trend}: Fill={np.mean(filled)*100:.0f}%, Drift={cvo.median():.3f} ADR, DayRange={day_range.median():.2f} ADR, N={len(tsub)}")

# ═══════════════════════════════════════════════════════════════════
# Q3.4: Catalyst Type
# ═══════════════════════════════════════════════════════════════════
out.append("\n" + "=" * 70)
out.append("FRAGE 3.4: Catalyst-Typ -> Unterschiedliche Muster?")
out.append("=" * 70)

cat_counts = meta['catalyst_type'].value_counts()
out.append(f"\nCatalyst-Verteilung:")
for cat, cnt in cat_counts.items():
    out.append(f"  {cat}: N={cnt}")

out.append(f"\nVerhalten nach Catalyst-Typ:")
for gd in ['up', 'down']:
    label = 'GapUp' if gd == 'up' else 'GapDown'
    gsub = meta[meta['gap_direction'] == gd]
    out.append(f"\n  {label}:")

    for cat in cat_counts.index:
        csub = gsub[gsub['catalyst_type'] == cat]
        if len(csub) < 20:
            continue
        cvo = csub['close_vs_open_adr'].dropna()
        filled = csub['gap_filled'].astype(float).values
        mae = csub['max_adverse_adr'].dropna()
        mex = csub['max_extension_adr'].dropna()
        out.append(f"    {cat}: Fill={np.mean(filled)*100:.0f}%, Drift={cvo.median():.3f} ADR, "
                   f"MaxExtension={mex.median():.3f}, MaxAdverse={mae.median():.3f}, N={len(csub)}")

# ── RSI context ──
out.append("\n" + "=" * 70)
out.append("BONUS 3.5: RSI-14 vor Gap -> Tagesverhalten")
out.append("=" * 70)

for gd in ['up', 'down']:
    label = 'GapUp' if gd == 'up' else 'GapDown'
    gsub = meta[(meta['gap_direction'] == gd) & (meta['rsi_14_prev'].notna())]
    out.append(f"\n  {label}:")

    rsi_buckets = [
        ('RSI < 30 (ueberverkauft)', gsub['rsi_14_prev'] < 30),
        ('RSI 30-50', (gsub['rsi_14_prev'] >= 30) & (gsub['rsi_14_prev'] < 50)),
        ('RSI 50-70', (gsub['rsi_14_prev'] >= 50) & (gsub['rsi_14_prev'] < 70)),
        ('RSI > 70 (ueberkauft)', gsub['rsi_14_prev'] >= 70),
    ]
    for blabel, mask in rsi_buckets:
        bsub = gsub[mask]
        if len(bsub) < 20:
            out.append(f"    {blabel}: N={len(bsub)} (LOW N)")
            continue
        cvo = bsub['close_vs_open_adr'].dropna()
        filled = bsub['gap_filled'].astype(float).values
        out.append(f"    {blabel}: Fill={np.mean(filled)*100:.0f}%, Drift={cvo.median():.3f} ADR, N={len(bsub)}")

# ── Write output ────────────────────────────────────────────────────
report = "\n".join(out)
with open("results/cheat_cat3_gap_context.txt", "w", encoding="utf-8") as f:
    f.write(report)

print(report)
print(f"\nResults saved to results/cheat_cat3_gap_context.txt", file=sys.stderr)
