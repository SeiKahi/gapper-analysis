"""
TEST D: Year-by-Year Stability
================================
Pruefe ob der GapDn SL5+T4 Edge in jedem Jahr (2021, 2022, 2023) stabil ist.
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import numpy as np
import pandas as pd

np.random.seed(42)

# ============================================================
# Load data
# ============================================================
print("Loading trades...", file=sys.stderr)
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

df = pd.read_parquet(str(PROJECT_ROOT / 'results' / 'all_trades_v3.parquet'))
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['year_month'] = df['date'].dt.to_period('M').astype(str)

# Only use data up to 2023-12-31
df = df[df['date'] <= '2023-12-31'].copy()
print(f"Total trades (up to 2023): {len(df)}", file=sys.stderr)

def bootstrap_ci(data, n_boot=10000, ci=0.95):
    data = np.array(data)
    n = len(data)
    rng = np.random.default_rng(42)
    boot_means = np.array([np.mean(rng.choice(data, size=n, replace=True)) for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    lo = np.percentile(boot_means, alpha * 100)
    hi = np.percentile(boot_means, (1 - alpha) * 100)
    p_val = np.mean(boot_means <= 0)
    return lo, hi, p_val

# ============================================================
# Define setups
# ============================================================
setups = [
    ('GapDn + SL5_035 + T4_3R', {'gap_direction': 'down', 'sl_method': 'SL5_035', 'target_method': 'T4_3R'}),
    ('GapDn + SL5_035 + T4_2R', {'gap_direction': 'down', 'sl_method': 'SL5_035', 'target_method': 'T4_2R'}),
    ('GapDn + SL5_035 + T3_050', {'gap_direction': 'down', 'sl_method': 'SL5_035', 'target_method': 'T3_050'}),
    ('GapUp + SL5_035 + T4_3R', {'gap_direction': 'up', 'sl_method': 'SL5_035', 'target_method': 'T4_3R'}),
]

# Also by scenario
scenario_setups = [
    ('GapDn_VWAP_Brk + SL5_035 + T4_3R', {'scenario': 'GapDn_VWAP_Brk', 'sl_method': 'SL5_035', 'target_method': 'T4_3R'}),
    ('GapDn_2s_Fade + SL5_035 + T4_3R', {'scenario': 'GapDn_2s_Fade', 'sl_method': 'SL5_035', 'target_method': 'T4_3R'}),
    ('GapDn_1s_Fade + SL5_035 + T4_3R', {'scenario': 'GapDn_1s_Fade', 'sl_method': 'SL5_035', 'target_method': 'T4_3R'}),
]

# ============================================================
# Analyze
# ============================================================
print("Analyzing yearly stability...", file=sys.stderr)

outpath = str(PROJECT_ROOT / 'results' / 'test_yearly_stability.txt')
with open(outpath, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("TEST D: YEAR-BY-YEAR STABILITY\n")
    f.write("=" * 80 + "\n")
    f.write("Date: 2026-02-13\n")
    f.write("Data: Train (2021-2022) + Val (2023)\n\n")

    # ---- SECTION 1: Yearly breakdown ----
    f.write("=" * 80 + "\n")
    f.write("SECTION 1: YEARLY BREAKDOWN\n")
    f.write("=" * 80 + "\n\n")

    for name, filters in setups + scenario_setups:
        mask = pd.Series(True, index=df.index)
        for col, value in filters.items():
            mask &= df[col] == value
        subset = df[mask]

        f.write(f"--- {name} ---\n")

        # Overall
        if len(subset) > 0:
            avg = subset['pnl_adr'].mean()
            med = subset['pnl_adr'].median()
            wr = (subset['outcome'] == 'target_hit').mean() * 100
            lo, hi, p_val = bootstrap_ci(subset['pnl_adr'].values)
            f.write(f"  Overall: N={len(subset)}, AvgPnL={avg:+.4f}, MedPnL={med:+.4f}, WR={wr:.1f}%, CI=[{lo:+.4f},{hi:+.4f}], P(<=0)={p_val:.4f}\n")

        for year in [2021, 2022, 2023]:
            yearly = subset[subset['year'] == year]
            if len(yearly) < 10:
                f.write(f"  {year}: N={len(yearly)} (too few)\n")
                continue

            avg = yearly['pnl_adr'].mean()
            med = yearly['pnl_adr'].median()
            wr = (yearly['outcome'] == 'target_hit').mean() * 100
            sl_pct = (yearly['outcome'] == 'sl_hit').mean() * 100
            to_pct = (yearly['outcome'] == 'timeout').mean() * 100
            lo, hi, p_val = bootstrap_ci(yearly['pnl_adr'].values)

            sign = "+" if avg > 0 else ""
            sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.10 else ""

            f.write(f"  {year}: N={len(yearly)}, AvgPnL={avg:+.4f}, MedPnL={med:+.4f}, WR={wr:.1f}%, SL%={sl_pct:.1f}%, TO%={to_pct:.1f}%, CI=[{lo:+.4f},{hi:+.4f}], P(<=0)={p_val:.4f} {sig}\n")

        f.write("\n")

    # ---- SECTION 2: Monthly breakdown for main setup ----
    f.write("=" * 80 + "\n")
    f.write("SECTION 2: MONTHLY BREAKDOWN — GapDn + SL5_035 + T4_3R\n")
    f.write("=" * 80 + "\n\n")

    main_mask = (df['gap_direction'] == 'down') & (df['sl_method'] == 'SL5_035') & (df['target_method'] == 'T4_3R')
    main_df = df[main_mask].copy()

    f.write(f"{'Year-Month':<12} {'N':>6} {'AvgPnL':>8} {'MedPnL':>8} {'WR%':>6} {'SL%':>6} {'TO%':>6}\n")
    f.write("-" * 60 + "\n")

    positive_months = 0
    total_months = 0
    for ym in sorted(main_df['year_month'].unique()):
        monthly = main_df[main_df['year_month'] == ym]
        if len(monthly) < 5:
            continue
        total_months += 1
        avg = monthly['pnl_adr'].mean()
        med = monthly['pnl_adr'].median()
        wr = (monthly['outcome'] == 'target_hit').mean() * 100
        sl_pct = (monthly['outcome'] == 'sl_hit').mean() * 100
        to_pct = (monthly['outcome'] == 'timeout').mean() * 100

        if avg > 0:
            positive_months += 1
        sign = "+" if avg > 0 else ""
        f.write(f"{ym:<12} {len(monthly):>6d} {avg:>+8.4f} {med:>+8.4f} {wr:>5.1f}% {sl_pct:>5.1f}% {to_pct:>5.1f}%\n")

    f.write(f"\nPositive months: {positive_months}/{total_months} ({positive_months/total_months*100:.0f}%)\n")

    # ---- SECTION 3: Monthly breakdown by scenario ----
    f.write("\n" + "=" * 80 + "\n")
    f.write("SECTION 3: MONTHLY BREAKDOWN — GapDn_VWAP_Brk + SL5_035 + T4_3R\n")
    f.write("=" * 80 + "\n\n")

    scen_mask = (df['scenario'] == 'GapDn_VWAP_Brk') & (df['sl_method'] == 'SL5_035') & (df['target_method'] == 'T4_3R')
    scen_df = df[scen_mask].copy()

    f.write(f"{'Year-Month':<12} {'N':>6} {'AvgPnL':>8} {'MedPnL':>8} {'WR%':>6}\n")
    f.write("-" * 50 + "\n")

    pos_m2 = 0
    tot_m2 = 0
    for ym in sorted(scen_df['year_month'].unique()):
        monthly = scen_df[scen_df['year_month'] == ym]
        if len(monthly) < 5:
            continue
        tot_m2 += 1
        avg = monthly['pnl_adr'].mean()
        med = monthly['pnl_adr'].median()
        wr = (monthly['outcome'] == 'target_hit').mean() * 100
        if avg > 0:
            pos_m2 += 1
        f.write(f"{ym:<12} {len(monthly):>6d} {avg:>+8.4f} {med:>+8.4f} {wr:>5.1f}%\n")

    f.write(f"\nPositive months: {pos_m2}/{tot_m2} ({pos_m2/tot_m2*100:.0f}%)\n")

    # ---- SECTION 4: Stability verdict ----
    f.write("\n" + "=" * 80 + "\n")
    f.write("SECTION 4: STABILITY VERDICT\n")
    f.write("=" * 80 + "\n\n")

    for name, filters in [setups[0], scenario_setups[0]]:
        mask = pd.Series(True, index=df.index)
        for col, value in filters.items():
            mask &= df[col] == value
        subset = df[mask]

        yearly_avgs = []
        f.write(f"  {name}:\n")
        for year in [2021, 2022, 2023]:
            yearly = subset[subset['year'] == year]
            if len(yearly) > 0:
                avg = yearly['pnl_adr'].mean()
                yearly_avgs.append(avg)
                f.write(f"    {year}: AvgPnL = {avg:+.4f} (N={len(yearly)})\n")

        if len(yearly_avgs) == 3:
            all_positive = all(a > 0 for a in yearly_avgs)
            spread = max(yearly_avgs) - min(yearly_avgs)
            f.write(f"    All years positive: {all_positive}\n")
            f.write(f"    Spread (max-min): {spread:.4f}\n")
            if all_positive:
                f.write(f"    >>> STABLE: Edge positive in ALL 3 years <<<\n")
            else:
                neg_years = [2021+i for i, a in enumerate(yearly_avgs) if a <= 0]
                f.write(f"    >>> UNSTABLE: Negative in {neg_years} <<<\n")
        f.write("\n")

    # Compare GapDn vs GapUp stability
    f.write("  Comparison GapDn vs GapUp (SL5_035 + T4_3R):\n")
    for gap_dir in ['down', 'up']:
        mask = (df['gap_direction'] == gap_dir) & (df['sl_method'] == 'SL5_035') & (df['target_method'] == 'T4_3R')
        subset = df[mask]
        f.write(f"    Gap{gap_dir.title()}:\n")
        for year in [2021, 2022, 2023]:
            yearly = subset[subset['year'] == year]
            if len(yearly) > 0:
                avg = yearly['pnl_adr'].mean()
                f.write(f"      {year}: N={len(yearly)}, AvgPnL={avg:+.4f}\n")

    f.write("\n" + "=" * 80 + "\n")
    f.write("END OF TEST D\n")
    f.write("=" * 80 + "\n")

print(f"Results written to: {outpath}", file=sys.stderr)
print("TEST D COMPLETE.", file=sys.stderr)
