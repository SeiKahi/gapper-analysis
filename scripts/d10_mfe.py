"""
D10 Aufgabe 1: MFE-Verteilung Basis (IS)
MFE_LIVE, MFE_FULL, MAE fuer OD>0.5 + with_gap + SL_FIX025.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

RAW_DIR = Path('data/raw_1min')


def compute_mfe_mae(bars_rth, entry_price, sl_level, trade_dir, adr):
    """
    Compute MFE_LIVE, MFE_FULL, MAE from 1min bars.
    Returns dict with all excursion metrics.
    """
    sl_dist = abs(entry_price - sl_level)
    if sl_dist <= 0 or pd.isna(sl_dist) or adr <= 0:
        return None

    post_entry = bars_rth[(bars_rth['time_et'] >= '09:36') & (bars_rth['time_et'] <= '15:55')]
    if len(post_entry) == 0:
        return None

    mfe_full = 0.0
    mae = 0.0
    sl_hit = False
    sl_hit_time = None
    mfe_at_sl = 0.0  # MFE_LIVE at SL hit point

    for _, bar in post_entry.iterrows():
        if trade_dir == 'long':
            fav = bar['high'] - entry_price
            adv = entry_price - bar['low']
        else:
            fav = entry_price - bar['low']
            adv = bar['high'] - entry_price

        fav = max(fav, 0)
        adv = max(adv, 0)

        # MFE_FULL always updates
        mfe_full = max(mfe_full, fav)
        mae = max(mae, adv)

        # SL check
        if not sl_hit:
            mfe_at_sl = mfe_full  # Track MFE up to this point
            if trade_dir == 'long' and bar['low'] <= sl_level:
                sl_hit = True
                sl_hit_time = bar['time_et']
            elif trade_dir == 'short' and bar['high'] >= sl_level:
                sl_hit = True
                sl_hit_time = bar['time_et']

    # MFE_LIVE = MFE up to SL hit (or full day if no SL)
    mfe_live = mfe_at_sl if sl_hit else mfe_full

    # Exit price for PnL
    if sl_hit:
        exit_price = sl_level
    else:
        last_bar = post_entry.iloc[-1]
        exit_price = last_bar['close']

    if trade_dir == 'long':
        pnl = exit_price - entry_price
    else:
        pnl = entry_price - exit_price

    return {
        'mfe_live': mfe_live,
        'mfe_full': mfe_full,
        'mae': mae,
        'mfe_live_r': mfe_live / sl_dist,
        'mfe_full_r': mfe_full / sl_dist,
        'mae_r': mae / sl_dist,
        'mfe_live_adr': mfe_live / adr,
        'mfe_full_adr': mfe_full / adr,
        'mae_adr': mae / adr,
        'sl_hit': sl_hit,
        'sl_hit_time': sl_hit_time,
        'pnl': pnl,
        'pnl_r': pnl / sl_dist,
        'pnl_adr': pnl / adr,
        'sl_dist': sl_dist,
        'sl_dist_adr': sl_dist / adr,
    }


def run_mfe_analysis(meta_subset):
    """Run MFE analysis for a metadata subset."""
    all_results = []
    total = len(meta_subset)
    done = 0

    for _, row in meta_subset.iterrows():
        done += 1
        if done % 100 == 0:
            print(f"  {done}/{total}...", file=sys.stderr)

        ticker = row['ticker']
        date = str(row['date'])
        fpath = RAW_DIR / ticker / f"{date}.parquet"
        if not fpath.exists():
            continue

        bars = pd.read_parquet(fpath)
        rth = bars[bars['session'] == 'rth']
        if len(rth) < 10:
            continue

        entry_price = row['close_935']
        adr = row.get('adr_10', np.nan)
        gap_dir = row['gap_direction']

        if pd.isna(entry_price) or pd.isna(adr) or adr <= 0:
            continue

        # Trade direction from OD with_gap
        if gap_dir == 'up':
            trade_dir = 'long'
            sl_level = entry_price - 0.25 * adr
        else:
            trade_dir = 'short'
            sl_level = entry_price + 0.25 * adr

        result = compute_mfe_mae(rth, entry_price, sl_level, trade_dir, adr)
        if result is None:
            continue

        result['ticker'] = ticker
        result['date'] = date
        result['gap_direction'] = gap_dir
        result['trade_dir'] = trade_dir
        result['od_strength'] = row['od_strength']
        result['rvol_5'] = row.get('rvol_5', np.nan)
        result['rvol_30'] = row.get('rvol_open_30min', np.nan)
        result['gap_adr'] = row.get('gap_size_in_adr', np.nan)
        result['pm_rth30'] = row.get('pm_rth30_computed', np.nan)
        all_results.append(result)

    return pd.DataFrame(all_results)


def mfe_stats_table(df, col_r, col_adr, label):
    """Generate statistics table for MFE/MAE column."""
    lines = []
    lines.append(f"\n  {label}:")
    lines.append(f"  {'Metrik':<12} | {'GapUp R':>10} | {'GapUp ADR':>10} | {'GapDn R':>10} | {'GapDn ADR':>10}")
    lines.append(f"  {'-'*62}")

    for gap_dir, gap_label in [('up', 'GapUp'), ('down', 'GapDn')]:
        sub = df[df['gap_direction'] == gap_dir]
        if len(sub) < 10:
            continue

    # Build stats for both gap directions
    stats_list = [
        ('Median', 'median'),
        ('Mean', 'mean'),
        ('StdDev', 'std'),
        ('P10', lambda s: s.quantile(0.10)),
        ('P25', lambda s: s.quantile(0.25)),
        ('P75', lambda s: s.quantile(0.75)),
        ('P90', lambda s: s.quantile(0.90)),
        ('Min', 'min'),
        ('Max', 'max'),
        ('N', 'count'),
    ]

    for stat_label, stat_fn in stats_list:
        vals = []
        for gap_dir in ['up', 'down']:
            sub = df[df['gap_direction'] == gap_dir]
            if len(sub) < 10:
                vals.extend(['N<10', 'N<10'])
                continue
            for col in [col_r, col_adr]:
                s = sub[col]
                if callable(stat_fn):
                    v = stat_fn(s)
                else:
                    v = getattr(s, stat_fn)()
                if stat_label == 'N':
                    vals.append(f"{int(v)}")
                else:
                    vals.append(f"{v:.3f}")

        lines.append(f"  {stat_label:<12} | {vals[0]:>10} | {vals[1]:>10} | {vals[2]:>10} | {vals[3]:>10}")

    return "\n".join(lines)


def main():
    lines = []
    lines.append("=" * 70)
    lines.append("D10 AUFGABE 1: MFE-VERTEILUNG BASIS (IS)")
    lines.append("=" * 70)
    lines.append("\nSetup: OD > 0.5 ADR + with_gap + Entry 9:35 + SL 0.25 ADR fix")
    lines.append("MFE_LIVE: bis SL-Hit | MFE_FULL: ganzer Tag | MAE: gegen Trade")
    lines.append("1R = 0.25 ADR = SL-Distanz")

    meta = pd.read_parquet('data/metadata/metadata_v8_5.parquet')
    h1 = meta[(meta['date'] >= '2021-02-21') & (meta['date'] <= '2023-12-31')].copy()
    base = h1[(h1['od_strength'] > 0.5) & (h1['od_direction'] == 'with_gap')]
    print(f"Base: {len(base)} (GapUp={len(base[base['gap_direction']=='up'])}, GapDn={len(base[base['gap_direction']=='down'])})", file=sys.stderr)

    # Run MFE analysis
    print("Computing MFE/MAE...", file=sys.stderr)
    results = run_mfe_analysis(base)
    print(f"Results: {len(results)} trades", file=sys.stderr)

    # Save raw results for reuse by later scripts
    results.to_parquet('results/d10_mfe_raw_is.parquet', index=False)

    # ===== 1b: MFE ALL TRADES =====
    lines.append(f"\n\n{'='*70}")
    lines.append("1b: MFE-STATISTIK ALLE TRADES")
    lines.append(f"{'='*70}")

    lines.append(mfe_stats_table(results, 'mfe_live_r', 'mfe_live_adr', 'MFE_LIVE (bis SL-Hit)'))
    lines.append(mfe_stats_table(results, 'mfe_full_r', 'mfe_full_adr', 'MFE_FULL (ganzer Tag)'))

    # ===== 1c: MFE NUR GEWINNER =====
    lines.append(f"\n\n{'='*70}")
    lines.append("1c: MFE-STATISTIK NUR GEWINNER (SL ueberlebt)")
    lines.append(f"{'='*70}")

    winners = results[results['sl_hit'] == False]
    n_win_up = len(winners[winners['gap_direction'] == 'up'])
    n_win_dn = len(winners[winners['gap_direction'] == 'down'])
    lines.append(f"\n  Gewinner: GapUp={n_win_up}/{len(results[results['gap_direction']=='up'])} "
                 f"({100*n_win_up/len(results[results['gap_direction']=='up']):.1f}%), "
                 f"GapDn={n_win_dn}/{len(results[results['gap_direction']=='down'])} "
                 f"({100*n_win_dn/len(results[results['gap_direction']=='down']):.1f}%)")

    lines.append(mfe_stats_table(winners, 'mfe_live_r', 'mfe_live_adr', 'MFE_LIVE Gewinner'))
    lines.append(mfe_stats_table(winners, 'mfe_full_r', 'mfe_full_adr', 'MFE_FULL Gewinner'))

    # ===== 1d: MFE NUR VERLIERER =====
    lines.append(f"\n\n{'='*70}")
    lines.append("1d: MFE-STATISTIK NUR VERLIERER (SL getroffen)")
    lines.append(f"{'='*70}")

    losers = results[results['sl_hit'] == True]
    n_los_up = len(losers[losers['gap_direction'] == 'up'])
    n_los_dn = len(losers[losers['gap_direction'] == 'down'])
    lines.append(f"\n  Verlierer: GapUp={n_los_up}, GapDn={n_los_dn}")

    lines.append(mfe_stats_table(losers, 'mfe_live_r', 'mfe_live_adr', 'MFE_LIVE Verlierer (vor SL-Hit)'))

    # How many losers had MFE >= 1R before getting stopped?
    lines.append("\n\n  Verlierer die erst in Richtung liefen:")
    for gap_dir, gap_label in [('up', 'GapUp'), ('down', 'GapDn')]:
        sub = losers[losers['gap_direction'] == gap_dir]
        if len(sub) == 0:
            continue
        n_total = len(sub)
        for thresh in [0.5, 1.0, 1.5, 2.0]:
            n_above = (sub['mfe_live_r'] >= thresh).sum()
            lines.append(f"    {gap_label}: MFE >= {thresh:.1f}R vor SL: {n_above}/{n_total} ({100*n_above/n_total:.1f}%)")
        lines.append("")

    # ===== 1e: MAE ALL TRADES =====
    lines.append(f"\n{'='*70}")
    lines.append("1e: MAE-STATISTIK ALLE TRADES")
    lines.append(f"{'='*70}")

    lines.append(mfe_stats_table(results, 'mae_r', 'mae_adr', 'MAE (Max Adverse Excursion)'))

    # MAE for winners vs losers
    lines.append("\n  MAE Gewinner vs Verlierer (Median):")
    for gap_dir, gap_label in [('up', 'GapUp'), ('down', 'GapDn')]:
        w = winners[winners['gap_direction'] == gap_dir]
        l = losers[losers['gap_direction'] == gap_dir]
        if len(w) >= 10 and len(l) >= 10:
            lines.append(f"    {gap_label}: Gewinner MAE_R={w['mae_r'].median():.2f}, "
                         f"Verlierer MAE_R={l['mae_r'].median():.2f}")

    output = "\n".join(lines)
    with open('results/d10_mfe.txt', 'w', encoding='utf-8') as f:
        f.write(output)
    print(output)
    print(f"\nSaved to results/d10_mfe.txt + d10_mfe_raw_is.parquet", file=sys.stderr)


if __name__ == '__main__':
    main()
