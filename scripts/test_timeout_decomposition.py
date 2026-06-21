"""
TEST B: Timeout PnL Decomposition
===================================
Zerlege Mean PnL in 3 Komponenten: Target-Hits, SL-Hits, Timeouts.
Pruefe ob der positive Mean PnL hauptsaechlich von Timeouts getrieben ist.
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import numpy as np
import pandas as pd

print("Loading trades...", file=sys.stderr)
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

df = pd.read_parquet(str(PROJECT_ROOT / 'results' / 'all_trades_v3.parquet'))

# Focus on val split only (2023) and pre-OOS data
val = df[df['split'] == 'val'].copy()
print(f"Val trades: {len(val)}", file=sys.stderr)

# Define setups to analyze
setups = [
    ('GapDn + SL5_035 + T4_3R', {'gap_direction': 'down', 'sl_method': 'SL5_035', 'target_method': 'T4_3R'}),
    ('GapDn + SL5_035 + T4_2R', {'gap_direction': 'down', 'sl_method': 'SL5_035', 'target_method': 'T4_2R'}),
    ('GapDn + SL5_035 + T3_050', {'gap_direction': 'down', 'sl_method': 'SL5_035', 'target_method': 'T3_050'}),
    ('GapDn + SL5_035 + T4_15R', {'gap_direction': 'down', 'sl_method': 'SL5_035', 'target_method': 'T4_15R'}),
    ('GapUp + SL5_035 + T4_3R', {'gap_direction': 'up', 'sl_method': 'SL5_035', 'target_method': 'T4_3R'}),
    ('GapUp + SL5_035 + T4_2R', {'gap_direction': 'up', 'sl_method': 'SL5_035', 'target_method': 'T4_2R'}),
]

# Also analyze by scenario
scenarios_focus = [
    ('GapDn_VWAP_Brk + SL5_035 + T4_3R', {'scenario': 'GapDn_VWAP_Brk', 'sl_method': 'SL5_035', 'target_method': 'T4_3R'}),
    ('GapDn_2s_Fade + SL5_035 + T4_3R', {'scenario': 'GapDn_2s_Fade', 'sl_method': 'SL5_035', 'target_method': 'T4_3R'}),
    ('GapDn_1s_Fade + SL5_035 + T4_3R', {'scenario': 'GapDn_1s_Fade', 'sl_method': 'SL5_035', 'target_method': 'T4_3R'}),
]

def analyze_decomposition(subset, name):
    """Decompose PnL into target/sl/timeout components."""
    n = len(subset)
    if n == 0:
        return None

    target_trades = subset[subset['outcome'] == 'target_hit']
    sl_trades = subset[subset['outcome'] == 'sl_hit']
    timeout_trades = subset[subset['outcome'] == 'timeout']

    n_tgt = len(target_trades)
    n_sl = len(sl_trades)
    n_to = len(timeout_trades)

    avg_pnl = subset['pnl_adr'].mean()
    med_pnl = subset['pnl_adr'].median()

    # Component contributions to mean PnL
    tgt_contrib = (n_tgt / n) * target_trades['pnl_adr'].mean() if n_tgt > 0 else 0
    sl_contrib = (n_sl / n) * sl_trades['pnl_adr'].mean() if n_sl > 0 else 0
    to_contrib = (n_to / n) * timeout_trades['pnl_adr'].mean() if n_to > 0 else 0

    # Timeout PnL distribution
    if n_to > 0:
        to_pnls = timeout_trades['pnl_adr']
        to_stats = {
            'min': to_pnls.min(),
            'p5': to_pnls.quantile(0.05),
            'p25': to_pnls.quantile(0.25),
            'median': to_pnls.median(),
            'p75': to_pnls.quantile(0.75),
            'p95': to_pnls.quantile(0.95),
            'max': to_pnls.max(),
            'mean': to_pnls.mean(),
            'std': to_pnls.std(),
            'pct_positive': (to_pnls > 0).mean() * 100,
        }
    else:
        to_stats = None

    # PnL without timeouts
    no_timeout = subset[subset['outcome'] != 'timeout']
    no_to_avg = no_timeout['pnl_adr'].mean() if len(no_timeout) > 0 else 0

    # Outlier sensitivity: remove top 5% and check
    p95 = subset['pnl_adr'].quantile(0.95)
    trimmed = subset[subset['pnl_adr'] <= p95]
    trimmed_avg = trimmed['pnl_adr'].mean()

    # Remove top 3% extreme winners
    p97 = subset['pnl_adr'].quantile(0.97)
    trimmed97 = subset[subset['pnl_adr'] <= p97]
    trimmed97_avg = trimmed97['pnl_adr'].mean()

    return {
        'name': name,
        'n': n,
        'avg_pnl': avg_pnl,
        'med_pnl': med_pnl,
        'n_tgt': n_tgt, 'n_sl': n_sl, 'n_to': n_to,
        'pct_tgt': n_tgt/n*100, 'pct_sl': n_sl/n*100, 'pct_to': n_to/n*100,
        'tgt_contrib': tgt_contrib,
        'sl_contrib': sl_contrib,
        'to_contrib': to_contrib,
        'tgt_avg': target_trades['pnl_adr'].mean() if n_tgt > 0 else 0,
        'sl_avg': sl_trades['pnl_adr'].mean() if n_sl > 0 else 0,
        'to_stats': to_stats,
        'no_to_avg': no_to_avg,
        'trimmed_avg_p95': trimmed_avg,
        'trimmed_avg_p97': trimmed97_avg,
        'p95_threshold': p95,
        'p97_threshold': p97,
    }

# ============================================================
# Run analysis
# ============================================================
print("Analyzing decomposition...", file=sys.stderr)

results = []
for name, filters in setups + scenarios_focus:
    mask = pd.Series(True, index=val.index)
    for col, value in filters.items():
        mask &= val[col] == value
    subset = val[mask]
    r = analyze_decomposition(subset, name)
    if r:
        results.append(r)

# ============================================================
# Write results
# ============================================================
outpath = str(PROJECT_ROOT / 'results' / 'test_timeout_decomp.txt')
with open(outpath, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("TEST B: TIMEOUT PnL DECOMPOSITION\n")
    f.write("=" * 80 + "\n")
    f.write("Date: 2026-02-13\n")
    f.write("Data: Val split (2023 only)\n\n")

    f.write("=" * 80 + "\n")
    f.write("SECTION 1: PnL DECOMPOSITION BY SETUP\n")
    f.write("=" * 80 + "\n\n")

    for r in results:
        f.write(f"--- {r['name']} ---\n")
        f.write(f"  N = {r['n']}\n")
        f.write(f"  Overall: AvgPnL = {r['avg_pnl']:+.4f} ADR, MedPnL = {r['med_pnl']:+.4f} ADR\n\n")

        f.write(f"  Outcome Distribution:\n")
        f.write(f"    Target Hit: {r['n_tgt']:>6d} ({r['pct_tgt']:.1f}%)  AvgPnL = {r['tgt_avg']:+.4f}\n")
        f.write(f"    SL Hit:     {r['n_sl']:>6d} ({r['pct_sl']:.1f}%)  AvgPnL = {r['sl_avg']:+.4f}\n")
        f.write(f"    Timeout:    {r['n_to']:>6d} ({r['pct_to']:.1f}%)\n\n")

        f.write(f"  Mean PnL Decomposition (sum = overall AvgPnL):\n")
        f.write(f"    Target contribution: {r['tgt_contrib']:+.4f} ADR ({r['tgt_contrib']/r['avg_pnl']*100:+.0f}%)\n" if r['avg_pnl'] != 0 else f"    Target contribution: {r['tgt_contrib']:+.4f} ADR\n")
        f.write(f"    SL contribution:     {r['sl_contrib']:+.4f} ADR ({r['sl_contrib']/r['avg_pnl']*100:+.0f}%)\n" if r['avg_pnl'] != 0 else f"    SL contribution:     {r['sl_contrib']:+.4f} ADR\n")
        f.write(f"    Timeout contribution:{r['to_contrib']:+.4f} ADR ({r['to_contrib']/r['avg_pnl']*100:+.0f}%)\n" if r['avg_pnl'] != 0 else f"    Timeout contribution:{r['to_contrib']:+.4f} ADR\n")
        f.write(f"    Sum check:           {r['tgt_contrib']+r['sl_contrib']+r['to_contrib']:+.4f} ADR\n\n")

        if r['to_stats']:
            ts = r['to_stats']
            f.write(f"  Timeout PnL Distribution:\n")
            f.write(f"    Min:    {ts['min']:+.4f}\n")
            f.write(f"    P5:     {ts['p5']:+.4f}\n")
            f.write(f"    P25:    {ts['p25']:+.4f}\n")
            f.write(f"    Median: {ts['median']:+.4f}\n")
            f.write(f"    Mean:   {ts['mean']:+.4f}\n")
            f.write(f"    P75:    {ts['p75']:+.4f}\n")
            f.write(f"    P95:    {ts['p95']:+.4f}\n")
            f.write(f"    Max:    {ts['max']:+.4f}\n")
            f.write(f"    Std:    {ts['std']:.4f}\n")
            f.write(f"    % Positive: {ts['pct_positive']:.1f}%\n\n")

        f.write(f"  Without Timeouts (Target+SL only): AvgPnL = {r['no_to_avg']:+.4f}\n")
        f.write(f"  Outlier Sensitivity:\n")
        f.write(f"    Full:           AvgPnL = {r['avg_pnl']:+.4f}\n")
        f.write(f"    Without top 5%: AvgPnL = {r['trimmed_avg_p95']:+.4f}  (removed > {r['p95_threshold']:+.4f})\n")
        f.write(f"    Without top 3%: AvgPnL = {r['trimmed_avg_p97']:+.4f}  (removed > {r['p97_threshold']:+.4f})\n")

        # Verdict for this setup
        if r['to_contrib'] > 0 and abs(r['to_contrib']) > abs(r['tgt_contrib'] + r['sl_contrib']):
            f.write(f"  >>> TIMEOUT-DRIVEN: Timeout contributes MORE than Target+SL combined <<<\n")
        elif r['tgt_contrib'] > 0 and r['tgt_contrib'] > r['to_contrib']:
            f.write(f"  >>> TARGET-DRIVEN: Target hits are the main PnL driver <<<\n")
        else:
            f.write(f"  >>> MIXED: Both components contribute <<<\n")

        f.write("\n")

    # ---- SECTION 2: Summary table ----
    f.write("=" * 80 + "\n")
    f.write("SECTION 2: SUMMARY TABLE\n")
    f.write("=" * 80 + "\n\n")

    header = f"{'Setup':<45} {'N':>6} {'AvgPnL':>8} {'Tgt%':>6} {'SL%':>6} {'TO%':>6} {'TgtCtb':>8} {'SLCtb':>8} {'TOCtb':>8} {'NoTO':>8} {'Trim95':>8}"
    f.write(header + "\n")
    f.write("-" * len(header) + "\n")

    for r in results:
        line = f"{r['name']:<45} {r['n']:>6d} {r['avg_pnl']:>+8.4f} {r['pct_tgt']:>5.1f}% {r['pct_sl']:>5.1f}% {r['pct_to']:>5.1f}% {r['tgt_contrib']:>+8.4f} {r['sl_contrib']:>+8.4f} {r['to_contrib']:>+8.4f} {r['no_to_avg']:>+8.4f} {r['trimmed_avg_p95']:>+8.4f}"
        f.write(line + "\n")

    # ---- SECTION 3: Verdict ----
    f.write("\n" + "=" * 80 + "\n")
    f.write("SECTION 3: VERDICT\n")
    f.write("=" * 80 + "\n\n")

    # Check main setup
    main = [r for r in results if r['name'] == 'GapDn + SL5_035 + T4_3R'][0]

    f.write(f"KEY SETUP: GapDn + SL5_035 + T4_3R\n")
    f.write(f"  Overall AvgPnL: {main['avg_pnl']:+.4f}\n")
    f.write(f"  Target contribution: {main['tgt_contrib']:+.4f} ({main['tgt_contrib']/main['avg_pnl']*100:.0f}%)\n")
    f.write(f"  SL contribution: {main['sl_contrib']:+.4f} ({main['sl_contrib']/main['avg_pnl']*100:.0f}%)\n")
    f.write(f"  Timeout contribution: {main['to_contrib']:+.4f} ({main['to_contrib']/main['avg_pnl']*100:.0f}%)\n\n")

    if main['to_contrib'] > main['tgt_contrib'] and main['to_contrib'] > 0:
        f.write("  >>> CONCLUSION: The positive expectancy is PRIMARILY TIMEOUT-DRIVEN. <<<\n")
        f.write("  >>> Timeout trades on GapDn stocks tend to end with positive PnL, <<<\n")
        f.write("  >>> which means the GapDn intraday drift (mean reversion) is the real edge. <<<\n")
    else:
        f.write("  >>> CONCLUSION: The positive expectancy is NOT purely timeout-driven. <<<\n")
        f.write("  >>> Target hits contribute meaningfully to the edge. <<<\n")

    # Check if removing top 5% kills the edge
    f.write(f"\n  Outlier test: After removing top 5% of winners:\n")
    f.write(f"    AvgPnL drops from {main['avg_pnl']:+.4f} to {main['trimmed_avg_p95']:+.4f}\n")
    if main['trimmed_avg_p95'] <= 0:
        f.write(f"    >>> EDGE DIES when top 5% winners are removed! <<<\n")
        f.write(f"    >>> The edge depends on a small number of extreme winners. <<<\n")
    else:
        f.write(f"    >>> Edge SURVIVES trimming ({main['trimmed_avg_p95']/main['avg_pnl']*100:.0f}% retained). <<<\n")

    f.write("\n" + "=" * 80 + "\n")
    f.write("END OF TEST B\n")
    f.write("=" * 80 + "\n")

print(f"Results written to: {outpath}", file=sys.stderr)
print("TEST B COMPLETE.", file=sys.stderr)
