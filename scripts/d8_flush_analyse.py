"""
D8.0 Aufgabe 3: OD-Against bei Earnings — Flush oder Fade? (IS)
================================================================
Kern-Hypothese: OD against Gap bei Earnings ist oft ein Flush, kein echter Fade.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
META_DIR = DATA_DIR / "metadata"
RAW_DIR = DATA_DIR / "raw_1min"
RESULTS_DIR = BASE_DIR / "results"
IS_END = "2023-12-31"


def load_1min_data(ticker, date_str):
    """Load 1-minute data for a ticker/date."""
    path = RAW_DIR / ticker / f"{date_str}.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        if 'time_et' in df.columns:
            df = df.sort_values('time_et').reset_index(drop=True)
        elif 'datetime_et' in df.columns:
            df['time_et'] = pd.to_datetime(df['datetime_et']).dt.strftime('%H:%M')
            df = df.sort_values('time_et').reset_index(drop=True)
        return df
    except Exception:
        return None


def get_price_at_time(bars, target_time):
    """Get close price at specific time from 1-min bars."""
    row = bars[bars['time_et'] == target_time]
    if len(row) > 0:
        return row.iloc[0]['close']
    return np.nan


def get_bar_at_time(bars, target_time):
    """Get full bar at specific time."""
    row = bars[bars['time_et'] == target_time]
    if len(row) > 0:
        return row.iloc[0]
    return None


def compute_volume_profile(bars):
    """Compute volume profile for first 5 minutes (09:30-09:34)."""
    first5 = bars[(bars['time_et'] >= '09:30') & (bars['time_et'] <= '09:34')]
    if len(first5) < 5:
        return np.nan, np.nan
    vol_1 = first5.iloc[0]['volume']
    vol_2to5 = first5.iloc[1:5]['volume'].mean()
    if vol_2to5 > 0:
        return vol_1 / vol_2to5, vol_1
    return np.nan, vol_1


def compute_wick_ratio(bars):
    """Compute wick ratio for the OD (9:30-9:35 aggregate)."""
    od_bars = bars[(bars['time_et'] >= '09:30') & (bars['time_et'] <= '09:34')]
    if len(od_bars) == 0:
        return np.nan
    od_high = od_bars['high'].max()
    od_low = od_bars['low'].min()
    od_open = od_bars.iloc[0]['open']
    od_close = od_bars.iloc[-1]['close']
    od_range = od_high - od_low
    if od_range == 0:
        return np.nan
    # Bullish OD: wick = (high - close) / range
    # Bearish OD: wick = (close - low) / range
    if od_close > od_open:  # bullish
        return (od_high - od_close) / od_range
    else:  # bearish
        return (od_close - od_low) / od_range


def compute_od_body_pct(bars):
    """Compute body % of OD range."""
    od_bars = bars[(bars['time_et'] >= '09:30') & (bars['time_et'] <= '09:34')]
    if len(od_bars) == 0:
        return np.nan
    od_high = od_bars['high'].max()
    od_low = od_bars['low'].min()
    od_open = od_bars.iloc[0]['open']
    od_close = od_bars.iloc[-1]['close']
    od_range = od_high - od_low
    if od_range == 0:
        return np.nan
    return abs(od_close - od_open) / od_range


def compute_od_with_gap(row):
    od = row.get('od_direction', None)
    if pd.isna(od) or od is None:
        return np.nan
    return od == 'with_gap'


def main():
    meta = pd.read_parquet(META_DIR / "metadata_v8.parquet")
    meta['date_dt'] = pd.to_datetime(meta['date'])
    meta = meta[meta['date_dt'] <= IS_END].copy()

    meta['od_with_gap'] = meta.apply(compute_od_with_gap, axis=1)

    earn = meta['is_earnings'] == True
    non_earn = (~meta['is_earnings']) & (~meta['earnings_unknown'])
    gap_up = meta['gap_direction'] == 'up'
    gap_dn = meta['gap_direction'] == 'down'
    od_against = meta['od_with_gap'] == False
    od_strong = meta['od_strength'].abs() > 0.5

    # Focus: OD against + strong
    focus_mask = od_against & od_strong

    lines = []
    lines.append("=" * 80)
    lines.append("AUFGABE 3: OD-AGAINST BEI EARNINGS — FLUSH ODER FADE? (IS)")
    lines.append("=" * 80)

    # 3a/3b: Compute Flush/Fade/Stall for each gapper
    lines.append("\nLade 1-Minuten-Daten und klassifiziere Flush/Fade/Stall...")

    results = []
    focus_rows = meta[focus_mask].copy()
    print(f"Processing {len(focus_rows)} OD-against+stark rows...", file=sys.stderr)

    for idx, row in tqdm(focus_rows.iterrows(), total=len(focus_rows),
                         desc="Loading 1min data", file=sys.stderr):
        ticker = row['ticker']
        date_str = str(row['date'])[:10]
        bars = load_1min_data(ticker, date_str)
        if bars is None:
            continue

        open_price = row.get('rth_open', row.get('today_open', np.nan))
        if pd.isna(open_price):
            bar_930 = get_bar_at_time(bars, '09:30')
            if bar_930 is not None:
                open_price = bar_930['open']
            else:
                continue

        close_935 = get_price_at_time(bars, '09:35')
        close_1000 = get_price_at_time(bars, '10:00')
        close_1030 = get_price_at_time(bars, '10:30')

        if any(pd.isna(x) for x in [close_935, close_1030]):
            continue

        # Volume profile
        vol_profile, vol_k1 = compute_volume_profile(bars)
        wick_ratio = compute_wick_ratio(bars)
        body_pct = compute_od_body_pct(bars)

        # Candle 5 vs Candle 1 comparison
        bar_930_row = get_bar_at_time(bars, '09:30')
        bar_934_row = get_bar_at_time(bars, '09:34')
        c5_vs_c1 = np.nan
        if bar_930_row is not None and bar_934_row is not None:
            c5_vs_c1 = bar_934_row['close'] - bar_930_row['close']

        # ADR for normalization
        adr = row.get('adr_5', row.get('adr_10', np.nan))
        if pd.isna(adr) or adr == 0:
            adr = abs(row.get('gap_pct', 5)) / 100 * open_price  # rough estimate

        gap_dir = row['gap_direction']

        # Classify: Flush / Fade / Stall at 10:30
        if gap_dir == 'up':
            # OD against = bearish OD (price went down in OD)
            # Flush = price recovers above open by 10:30
            # Fade = price stays below close_935 (or goes further down)
            # Stall = between close_935 and open
            if close_1030 > open_price:
                classification = 'flush'
            elif close_1030 < close_935:
                classification = 'fade'
            else:
                classification = 'stall'
        else:  # gap_down
            # OD against = bullish OD (price went up in OD)
            # Flush = price drops back below open by 10:30
            # Fade = price stays above close_935
            # Stall = between open and close_935
            if close_1030 < open_price:
                classification = 'flush'
            elif close_1030 > close_935:
                classification = 'fade'
            else:
                classification = 'stall'

        # MFE from open in gap direction (after flush)
        rth_bars = bars[(bars['time_et'] >= '10:30') & (bars['time_et'] <= '15:59')]
        if len(rth_bars) > 0 and gap_dir == 'up':
            mfe_gap_dir = (rth_bars['high'].max() - open_price) / adr if adr > 0 else np.nan
        elif len(rth_bars) > 0 and gap_dir == 'down':
            mfe_gap_dir = (open_price - rth_bars['low'].min()) / adr if adr > 0 else np.nan
        else:
            mfe_gap_dir = np.nan

        # Close location
        rth_close = row.get('rth_close', np.nan)
        if pd.isna(rth_close):
            close_bars = bars[bars['time_et'] == '15:59']
            if len(close_bars) > 0:
                rth_close = close_bars.iloc[0]['close']

        close_vs_open = np.nan
        if not pd.isna(rth_close):
            if gap_dir == 'up':
                close_vs_open = (rth_close - open_price) / adr
            else:
                close_vs_open = (open_price - rth_close) / adr

        results.append({
            'idx': idx,
            'ticker': ticker,
            'date': date_str,
            'gap_direction': gap_dir,
            'is_earnings': row['is_earnings'],
            'earnings_unknown': row['earnings_unknown'],
            'open_price': open_price,
            'close_935': close_935,
            'close_1000': close_1000 if not pd.isna(close_1000) else np.nan,
            'close_1030': close_1030,
            'od_strength': row['od_strength'],
            'classification': classification,
            'vol_profile': vol_profile,
            'wick_ratio': wick_ratio,
            'body_pct': body_pct,
            'c5_vs_c1': c5_vs_c1,
            'mfe_gap_dir': mfe_gap_dir,
            'close_vs_open': close_vs_open,
            'rvol_5': row.get('rvol_5', np.nan),
            'pm_rth5': row.get('pm_rth5', np.nan),
            'gap_size_in_adr': row.get('gap_size_in_adr', np.nan),
            'adr': adr,
        })

    res_df = pd.DataFrame(results)
    print(f"\nProcessed {len(res_df)} rows with valid data", file=sys.stderr)

    # Save raw results
    res_df.to_parquet(RESULTS_DIR / "d8_3_flush_raw.parquet", index=False)

    # 3b: Flush-Rate bei Earnings vs Non-Earnings
    lines.append(f"\n--- 3b: Flush-Rate bei Earnings vs Non-Earnings ---")
    lines.append(f"    (OD against + stark > 0.5 ADR, bis 10:30)")

    for gap_label, gdir in [("GapUp", "up"), ("GapDn", "down")]:
        lines.append(f"\n  {gap_label}:")

        for earn_label, emask_val in [("Earnings", True), ("Non-Earnings", False)]:
            if earn_label == "Non-Earnings":
                sub = res_df[(res_df['gap_direction'] == gdir) &
                             (~res_df['is_earnings']) & (~res_df['earnings_unknown'])]
            else:
                sub = res_df[(res_df['gap_direction'] == gdir) &
                             (res_df['is_earnings'] == True)]

            n = len(sub)
            if n < 10:
                lines.append(f"    {earn_label}: N={n} [INSUFFICIENT DATA]")
                continue

            flush_n = (sub['classification'] == 'flush').sum()
            fade_n = (sub['classification'] == 'fade').sum()
            stall_n = (sub['classification'] == 'stall').sum()

            flush_pct = 100 * flush_n / n
            fade_pct = 100 * fade_n / n
            stall_pct = 100 * stall_n / n

            # Bootstrap CI for flush rate
            rng = np.random.RandomState(42)
            flush_boots = []
            for _ in range(1000):
                sample = rng.choice(sub['classification'].values, n, replace=True)
                flush_boots.append((sample == 'flush').mean())
            ci_lo = np.percentile(flush_boots, 2.5) * 100
            ci_hi = np.percentile(flush_boots, 97.5) * 100

            lines.append(f"    {earn_label} (N={n}):")
            lines.append(f"      Flush: {flush_pct:5.1f}% [{ci_lo:.1f}-{ci_hi:.1f}%] (N={flush_n})")
            lines.append(f"      Fade:  {fade_pct:5.1f}% (N={fade_n})")
            lines.append(f"      Stall: {stall_pct:5.1f}% (N={stall_n})")

    # 3c: Was passiert NACH dem Flush?
    lines.append(f"\n\n--- 3c: Was passiert NACH dem Flush? ---")
    lines.append("  MFE in Gap-Richtung ab Open (nach 10:30), Close vs Open")

    flush_rows = res_df[res_df['classification'] == 'flush']
    for earn_label in ["Earnings", "Non-Earnings"]:
        if earn_label == "Non-Earnings":
            sub = flush_rows[(~flush_rows['is_earnings']) & (~flush_rows['earnings_unknown'])]
        else:
            sub = flush_rows[flush_rows['is_earnings'] == True]

        n = len(sub)
        tag = " [LOW N]" if n < 20 else ""
        if n < 5:
            lines.append(f"  {earn_label}: N={n} [INSUFFICIENT]")
            continue

        lines.append(f"\n  {earn_label} Flush (N={n}){tag}:")
        lines.append(f"    MFE in Gap-Dir (ab Open, post-10:30): median={sub['mfe_gap_dir'].median():.4f} ADR")
        lines.append(f"    Close vs Open (in Gap-Dir): median={sub['close_vs_open'].median():.4f} ADR")

    # Compare with OD-with (direct continuation)
    lines.append("\n  Vergleich: Earnings OD-WITH (direkte Continuation):")
    od_with_mask = meta['od_with_gap'] == True
    od_with_earn = meta[od_with_mask & earn & od_strong]
    if len(od_with_earn) >= 10:
        fd = od_with_earn.apply(lambda r: r['full_drift'] if r['gap_direction'] == 'up' else -r['full_drift'], axis=1)
        lines.append(f"    Full Drift in Gap-Dir: median={fd.median():.4f} ADR (N={len(od_with_earn)})")

    # 3d: Kann man den Flush VORHER erkennen?
    lines.append(f"\n\n--- 3d: Flush-Praediktoren (Parameter bei 9:35) ---")
    lines.append("  Vergleiche Earnings-Flushes mit Earnings-Fades:")

    earn_results = res_df[res_df['is_earnings'] == True]
    if len(earn_results) >= 20:
        flushes = earn_results[earn_results['classification'] == 'flush']
        fades = earn_results[earn_results['classification'] == 'fade']
        stalls = earn_results[earn_results['classification'] == 'stall']

        params = [
            ('OD-Staerke (abs)', 'od_strength', True),
            ('RVOL_5', 'rvol_5', False),
            ('PM/RTH5', 'pm_rth5', False),
            ('Gap in ADR (abs)', 'gap_size_in_adr', True),
            ('Vol Profile (K1/K2-5)', 'vol_profile', False),
            ('Wick Ratio', 'wick_ratio', False),
            ('Body %', 'body_pct', False),
        ]

        header = f"  {'Parameter':<25} | {'Flush med':>12} | {'Fade med':>12} | {'Stall med':>12} | {'Flush N':>8} | {'Fade N':>8}"
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))

        for pname, pcol, use_abs in params:
            if pcol not in earn_results.columns:
                continue
            fv = flushes[pcol].abs() if use_abs else flushes[pcol]
            fav = fades[pcol].abs() if use_abs else fades[pcol]
            sv = stalls[pcol].abs() if use_abs else stalls[pcol]

            fm = fv.median() if len(fv.dropna()) >= 5 else np.nan
            fam = fav.median() if len(fav.dropna()) >= 5 else np.nan
            sm = sv.median() if len(sv.dropna()) >= 5 else np.nan

            fm_s = f"{fm:.4f}" if not np.isnan(fm) else "N/A"
            fam_s = f"{fam:.4f}" if not np.isnan(fam) else "N/A"
            sm_s = f"{sm:.4f}" if not np.isnan(sm) else "N/A"

            lines.append(f"  {pname:<25} | {fm_s:>12} | {fam_s:>12} | {sm_s:>12} | {len(fv.dropna()):>8} | {len(fav.dropna()):>8}")
    else:
        lines.append("  Nicht genuegend Earnings-Daten fuer Praediktor-Analyse")

    # Also do the analysis for ALL gappers (not just earnings)
    lines.append(f"\n\n--- Bonus: Flush-Praediktoren ALLE Gapper ---")
    if len(res_df) >= 30:
        flushes = res_df[res_df['classification'] == 'flush']
        fades = res_df[res_df['classification'] == 'fade']

        params = [
            ('OD-Staerke (abs)', 'od_strength', True),
            ('RVOL_5', 'rvol_5', False),
            ('PM/RTH5', 'pm_rth5', False),
            ('Gap in ADR (abs)', 'gap_size_in_adr', True),
            ('Vol Profile (K1/K2-5)', 'vol_profile', False),
            ('Wick Ratio', 'wick_ratio', False),
            ('Body %', 'body_pct', False),
            ('is_earnings', 'is_earnings', False),
        ]

        header = f"  {'Parameter':<25} | {'Flush med':>12} | {'Fade med':>12} | {'Flush N':>8} | {'Fade N':>8}"
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))

        for pname, pcol, use_abs in params:
            if pcol not in res_df.columns:
                continue
            fv = flushes[pcol].abs() if use_abs else flushes[pcol]
            fav = fades[pcol].abs() if use_abs else fades[pcol]

            fm = fv.median() if len(fv.dropna()) >= 5 else np.nan
            fam = fav.median() if len(fav.dropna()) >= 5 else np.nan

            fm_s = f"{fm:.4f}" if not np.isnan(fm) else "N/A"
            fam_s = f"{fam:.4f}" if not np.isnan(fam) else "N/A"

            lines.append(f"  {pname:<25} | {fm_s:>12} | {fam_s:>12} | {len(fv.dropna()):>8} | {len(fav.dropna()):>8}")

    text = "\n".join(lines)
    out_file = RESULTS_DIR / "d8_3_flush_analyse.txt"
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
