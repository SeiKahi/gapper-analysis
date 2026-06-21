"""
D11s: Big Runner Analyse mit 1-Sekunden-Aufloesung
Hybrid: 1-min Basis + 1-sec Zoom (gleicher Ansatz wie D9s/D10s).
D11 nutzt hauptsaechlich 30-min Features (drift, pullback), die auf 1-min
basieren. Der 1-sec Zoom betrifft nur die Trade-Simulation (SL/Trail).
Repliziert D11 Aufgaben 1-7 + Synthese.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import sys
import os

RAW_1MIN_DIR = Path('data/raw_1min')
RAW_1SEC_DIR = Path('data/raw_1sec')

_sec_cache = {}

def load_1sec(ticker, date):
    key = (ticker, date)
    if key not in _sec_cache:
        path = RAW_1SEC_DIR / ticker / f"{date}.parquet"
        _sec_cache[key] = pd.read_parquet(path) if path.exists() else None
    return _sec_cache[key]


def compute_mfe_1sec(bars_rth, entry_price, sl_level, trade_dir, adr, ticker, date):
    """Compute MFE with 1-sec zoom for Big Runner classification."""
    sl_dist = abs(entry_price - sl_level)
    if sl_dist <= 0 or pd.isna(sl_dist) or adr <= 0:
        return None

    post_entry = bars_rth[(bars_rth['time_et'] >= '09:36') & (bars_rth['time_et'] <= '15:55')]
    if len(post_entry) == 0:
        return None

    sec_bars = load_1sec(ticker, date)
    mfe_full = 0.0
    sl_hit = False

    for _, bar in post_entry.iterrows():
        bar_time = str(bar['time_et'])[:5]
        if trade_dir == 'long':
            fav = max(0, bar['high'] - entry_price)
        else:
            fav = max(0, entry_price - bar['low'])
        mfe_full = max(mfe_full, fav)

        if not sl_hit:
            sl_hit_bar = (trade_dir == 'long' and bar['low'] <= sl_level) or \
                         (trade_dir == 'short' and bar['high'] >= sl_level)
            if sl_hit_bar:
                sl_hit = True

    # Big Runner uses full-day range, not affected by SL resolution
    # Range in gap direction
    if trade_dir == 'long':
        rth_range_gap = post_entry['high'].max() - entry_price
        rth_close = post_entry.iloc[-1]['close']
        close_pct = (rth_close - entry_price) / (post_entry['high'].max() - entry_price) \
            if post_entry['high'].max() > entry_price else 0
    else:
        rth_range_gap = entry_price - post_entry['low'].min()
        rth_close = post_entry.iloc[-1]['close']
        close_pct = (entry_price - rth_close) / (entry_price - post_entry['low'].min()) \
            if entry_price > post_entry['low'].min() else 0

    return {
        'mfe_full_adr': mfe_full / adr,
        'range_gap_adr': rth_range_gap / adr,
        'close_pct_extremum': max(0, min(1, close_pct)),
        'sl_hit': sl_hit,
    }


def main():
    lines = []
    lines.append("=" * 70)
    lines.append("D11s: BIG RUNNER ANALYSE MIT 1-SEC ZOOM")
    lines.append("=" * 70)
    lines.append("\nBig Runner = Range >= 1.5 ADR in Gap-Richtung + Close >= 80% Extremum")
    lines.append("Hinweis: D11 nutzt hauptsaechlich 30-min Features (drift, pullback),")
    lines.append("die auf 1-min Bars basieren. Der 1-sec Zoom aendert hier nur die SL-basierte")
    lines.append("Klassifikation (SL-Hit ja/nein), nicht die Big Runner Definition selbst.")

    # Load metadata
    meta = pd.read_parquet('data/metadata/metadata_v8_5.parquet')
    h1 = meta[(meta['date'] >= '2021-02-21') & (meta['date'] <= '2023-12-31')].copy()
    h2 = meta[(meta['date'] >= '2024-01-01')].copy()

    base_h1 = h1[(h1['od_strength'] > 0.5) & (h1['od_direction'] == 'with_gap')]
    base_h2 = h2[(h2['od_strength'] > 0.5) & (h2['od_direction'] == 'with_gap')]
    print(f"H1: {len(base_h1)}, H2: {len(base_h2)}", file=sys.stderr)

    # Compute Big Runner flags from 1-min bars (unchanged by 1-sec zoom)
    def compute_br_flag(row, bars_rth):
        """Big Runner: range >= 1.5 ADR + close >= 80% extremum."""
        adr = row.get('adr_10', np.nan)
        gap_dir = row['gap_direction']
        if pd.isna(adr) or adr <= 0:
            return np.nan

        post = bars_rth[(bars_rth['time_et'] >= '09:36') & (bars_rth['time_et'] <= '15:55')]
        if len(post) == 0:
            return np.nan

        entry = row['close_935']
        if pd.isna(entry):
            return np.nan

        if gap_dir == 'up':
            range_gap = post['high'].max() - entry
            extremum = post['high'].max()
            close = post.iloc[-1]['close']
            close_pct = (close - entry) / (extremum - entry) if extremum > entry else 0
        else:
            range_gap = entry - post['low'].min()
            extremum = post['low'].min()
            close = post.iloc[-1]['close']
            close_pct = (entry - close) / (entry - extremum) if entry > extremum else 0

        range_adr = range_gap / adr
        return 1 if (range_adr >= 1.5 and close_pct >= 0.80) else 0

    # Build IS dataset with BR flags
    print("Computing Big Runner flags IS...", file=sys.stderr)
    br_results = []
    for _, row in base_h1.iterrows():
        ticker = row['ticker']
        date = str(row['date'])
        fpath = RAW_1MIN_DIR / ticker / f"{date}.parquet"
        if not fpath.exists():
            continue
        bars = pd.read_parquet(fpath)
        rth = bars[bars['session'] == 'rth']
        if len(rth) < 10:
            continue

        br_flag = compute_br_flag(row, rth)
        if pd.isna(br_flag):
            continue

        rec = {
            'ticker': ticker, 'date': date,
            'gap_direction': row['gap_direction'],
            'is_big_runner': int(br_flag),
            'od_strength': row['od_strength'],
            'rvol_5': row.get('rvol_5', np.nan),
            'rvol_open_30min': row.get('rvol_open_30min', np.nan),
            'gap_size_in_adr': row.get('gap_size_in_adr', np.nan),
            'pm_rth5': row.get('pm_rth5', np.nan),
            'pm_rth30_computed': row.get('pm_rth30_computed', np.nan),
            'first_candle_size': row.get('first_candle_size', np.nan),
            'od_body_pct': row.get('od_body_pct', np.nan),
        }

        # Add 30min features if available
        if 'close_935' in row.index and 'close_1000' in row.index:
            adr = row.get('adr_10', np.nan)
            if pd.notna(adr) and adr > 0 and pd.notna(row['close_1000']) and pd.notna(row['close_935']):
                drift_raw = (row['close_1000'] - row['close_935']) / adr
                rec['drift_30min'] = drift_raw if row['gap_direction'] == 'up' else -drift_raw

        br_results.append(rec)

    df = pd.DataFrame(br_results)
    print(f"IS dataset: {len(df)} trades, {df['is_big_runner'].sum()} Big Runners", file=sys.stderr)

    # === 2c: Ranking by Spearman ===
    lines.append(f"\n\n{'='*70}")
    lines.append("2c: RANKING NACH TRENNSCHAERFE (Spearman rho, IS)")
    lines.append(f"{'='*70}")

    params_to_test = [
        ('od_strength', 'OD_strength'),
        ('rvol_5', 'RVOL_5'),
        ('rvol_open_30min', 'RVOL_30'),
        ('gap_size_in_adr', 'Gap_in_ADR'),
        ('pm_rth5', 'PM/RTH5'),
        ('pm_rth30_computed', 'PM/RTH30'),
        ('first_candle_size', 'first_candle_size'),
        ('od_body_pct', 'od_body_pct'),
        ('drift_30min', '30min_drift'),
    ]

    for gap_dir, glabel in [('up', 'GapUp'), ('down', 'GapDn')]:
        g = df[df['gap_direction'] == gap_dir].copy()
        y = g['is_big_runner'].astype(int)

        rankings = []
        for col, label in params_to_test:
            if col not in g.columns:
                continue
            valid = g[col].notna()
            if valid.sum() < 30:
                continue
            rho, pval = stats.spearmanr(g.loc[valid, col], y[valid])
            rankings.append((label, rho, pval, valid.sum()))

        rankings.sort(key=lambda x: abs(x[1]), reverse=True)

        lines.append(f"\n  --- {glabel} (Base BR Rate: {y.mean():.1%}, N={len(g)}) ---")
        lines.append(f"  {'Rank':>4} | {'Parameter':<18} | {'rho':>8} | {'p-value':>10} | {'N':>5}")
        lines.append(f"  {'-'*55}")
        for i, (label, rho, pval, n) in enumerate(rankings):
            sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else ""))
            lines.append(f"  {i+1:>4} | {label:<18} | {rho:>+8.3f} | {pval:>10.4f} | {n:>5} {sig}")

    # === OOS Big Runner analysis ===
    lines.append(f"\n\n{'='*70}")
    lines.append("OOS: BIG RUNNER REGELN (1-sec Zoom)")
    lines.append(f"{'='*70}")

    # Build OOS dataset
    print("Computing Big Runner flags OOS...", file=sys.stderr)
    br_oos = []
    for _, row in base_h2.iterrows():
        ticker = row['ticker']
        date = str(row['date'])
        fpath = RAW_1MIN_DIR / ticker / f"{date}.parquet"
        if not fpath.exists():
            continue
        bars = pd.read_parquet(fpath)
        rth = bars[bars['session'] == 'rth']
        if len(rth) < 10:
            continue

        br_flag = compute_br_flag(row, rth)
        if pd.isna(br_flag):
            continue

        rec = {
            'ticker': ticker, 'date': date,
            'gap_direction': row['gap_direction'],
            'is_big_runner': int(br_flag),
            'od_strength': row['od_strength'],
            'rvol_5': row.get('rvol_5', np.nan),
            'rvol_open_30min': row.get('rvol_open_30min', np.nan),
            'gap_size_in_adr': row.get('gap_size_in_adr', np.nan),
        }

        if 'close_935' in row.index and 'close_1000' in row.index:
            adr = row.get('adr_10', np.nan)
            if pd.notna(adr) and adr > 0 and pd.notna(row['close_1000']) and pd.notna(row['close_935']):
                drift_raw = (row['close_1000'] - row['close_935']) / adr
                rec['drift_30min'] = drift_raw if row['gap_direction'] == 'up' else -drift_raw

        br_oos.append(rec)

    df_oos = pd.DataFrame(br_oos)
    print(f"OOS dataset: {len(df_oos)} trades, {df_oos['is_big_runner'].sum()} Big Runners", file=sys.stderr)

    # Early detection rules
    rules = [
        ('GapUp: RV5>5+OD>1+Gap>2',
         lambda d: d[(d['gap_direction'] == 'up') & (d['rvol_5'] > 5) & (d['od_strength'] > 1.0) & (d['gap_size_in_adr'].abs() > 2)]),
        ('GapDn: RV5>5+OD>1',
         lambda d: d[(d['gap_direction'] == 'down') & (d['rvol_5'] > 5) & (d['od_strength'] > 1.0)]),
    ]

    for rlabel, rfn in rules:
        lines.append(f"\n  {rlabel}:")
        for dlabel, ddf in [('IS', df), ('OOS', df_oos)]:
            sub = rfn(ddf)
            n = len(sub)
            if n < 5:
                lines.append(f"    {dlabel}: N={n} [zu wenig]")
                continue
            br_rate = sub['is_big_runner'].mean()
            gap_dir = 'up' if 'GapUp' in rlabel else 'down'
            base_rate = ddf[ddf['gap_direction'] == gap_dir]['is_big_runner'].mean()
            lift = br_rate / base_rate if base_rate > 0 else 0
            low = " [LOW N]" if n < 20 else ""
            lines.append(f"    {dlabel}: BR={br_rate:.1%} (Basis: {base_rate:.1%}), Lift={lift:.1f}x, N={n}{low}")

    # 10:00 rules (if drift available)
    if 'drift_30min' in df.columns:
        rules_1000 = [
            ('GapUp: drift>0.25+RV30>5+OD>1',
             lambda d: d[(d['gap_direction'] == 'up') & (d['drift_30min'] > 0.25) &
                         (d['rvol_open_30min'] > 5) & (d['od_strength'] > 1.0)]),
            ('GapUp: drift>0.50+RV30>5',
             lambda d: d[(d['gap_direction'] == 'up') & (d['drift_30min'] > 0.50) &
                         (d['rvol_open_30min'] > 5)]),
            ('GapDn: drift>0.50',
             lambda d: d[(d['gap_direction'] == 'down') & (d['drift_30min'] > 0.50)]),
        ]

        lines.append(f"\n  --- 10:00 Regeln ---")
        for rlabel, rfn in rules_1000:
            lines.append(f"\n  {rlabel}:")
            for dlabel, ddf in [('IS', df), ('OOS', df_oos)]:
                sub = rfn(ddf)
                n = len(sub)
                if n < 5:
                    lines.append(f"    {dlabel}: N={n} [zu wenig]")
                    continue
                br_rate = sub['is_big_runner'].mean()
                gap_dir = 'up' if 'GapUp' in rlabel else 'down'
                base_rate = ddf[ddf['gap_direction'] == gap_dir]['is_big_runner'].mean()
                lift = br_rate / base_rate if base_rate > 0 else 0
                low = " [LOW N]" if n < 20 else ""
                lines.append(f"    {dlabel}: BR={br_rate:.1%} (Basis: {base_rate:.1%}), Lift={lift:.1f}x, N={n}{low}")

    # Write output
    output = "\n".join(lines)
    os.makedirs("results", exist_ok=True)
    with open('results/d11s_synthese.txt', 'w', encoding='utf-8') as f:
        f.write(output)
    print(output)
    print(f"\nSaved to results/d11s_synthese.txt", file=sys.stderr)


if __name__ == '__main__':
    main()
