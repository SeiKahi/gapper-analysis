###############################################################################
# PERMUTATION TEST — Devil's Advocate Counter-Test
#
# Testet ob die "besten" Subgruppen-Ergebnisse aus der Behavioral Analysis
# statistisch signifikant sind oder durch multiples Testen entstanden.
#
# Methode: Shuffle Drift30-Werte, finde "beste" Kombination, wiederhole 1000x.
# Wenn die echte Top-Kombination besser ist als 95% der Permutationen -> echt.
#
# Run:
#   .\gapper_env\Scripts\python.exe scripts\behavioral_permutation_test.py
###############################################################################

import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE_DIR = str(PROJECT_ROOT)
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
EVENTS_PATH = os.path.join(RESULTS_DIR, 'behavioral_events_v1.parquet')

# Load events
df = pd.read_parquet(EVENTS_PATH)
val = df[df['split'] == 'val'].copy()
print(f"Val events: {len(val)}", file=sys.stderr)

# Focus on GapDn (the promising direction)
gapdn = val[val['gap_direction'] == 'down'].copy()
print(f"GapDn Val events: {len(gapdn)}", file=sys.stderr)

# Precompute VWAP slope median for later filtering
slope_med = gapdn['vwap_slope_early'].abs().median()


def best_combo_drift(data, drift_col='drift_30'):
    """Find the best GapDn subgroup combination and return its mean drift."""
    best_drift = -999
    best_n = 0
    best_label = ''

    for scen in ['GapDn_2s_Fade', 'GapDn_VWAP_Brk', 'GapDn_1s_Fade']:
        base = data[data['scenario'] == scen]
        if len(base) < 20:
            continue

        for t_lo, t_hi, t_label in [(15, 45, 'Early'), (45, 120, 'Mid')]:
            for slope_label, slope_cond in [('Flat', base['vwap_slope_early'].abs() <= slope_med),
                                             ('Any', pd.Series(True, index=base.index))]:
                for gap_label, g_lo, g_hi in [('SmGap', 0, 2.0), ('AnyGap', 0, 99)]:
                    for adr_label, a_lo, a_hi in [('<75%', 0, 0.75), ('Any', 0, 99)]:
                        mask = (
                            (base['cross_bar'] >= t_lo) & (base['cross_bar'] < t_hi) &
                            slope_cond &
                            (base['gap_size_adr'] >= g_lo) & (base['gap_size_adr'] < g_hi) &
                            (base['pct_adr_at_cross'] >= a_lo) & (base['pct_adr_at_cross'] < a_hi)
                        )
                        subset = base[mask]
                        if len(subset) < 25:
                            continue

                        mean_drift = subset[drift_col].mean()
                        if mean_drift > best_drift:
                            best_drift = mean_drift
                            best_n = len(subset)
                            best_label = f"{scen}+{t_label}+{slope_label}+{gap_label}+{adr_label}"

    return best_drift, best_n, best_label


# Real best combo
real_drift, real_n, real_label = best_combo_drift(gapdn)
print(f"Real best: {real_label}, Drift30={real_drift:.4f}, N={real_n}", file=sys.stderr)

# Also get real overall GapDn drift as baseline
overall_drift = gapdn['drift_30'].mean()
print(f"Overall GapDn drift30: {overall_drift:.4f}", file=sys.stderr)

# Permutation test
N_PERM = 2000
perm_best_drifts = []
perm_overall_drifts = []

np.random.seed(42)
for i in tqdm(range(N_PERM), desc="Permutations", file=sys.stderr):
    shuffled = gapdn.copy()
    shuffled['drift_30'] = np.random.permutation(shuffled['drift_30'].values)

    best_d, best_n, _ = best_combo_drift(shuffled)
    perm_best_drifts.append(best_d)
    perm_overall_drifts.append(shuffled['drift_30'].mean())  # should be ~same

perm_best_drifts = np.array(perm_best_drifts)

# P-value: fraction of permutations that beat the real value
p_value = np.mean(perm_best_drifts >= real_drift)

# Also test: is GapDn overall drift significant?
# For overall drift, just bootstrap
np.random.seed(123)
boot_drifts = []
drift_vals = gapdn['drift_30'].dropna().values
for _ in range(5000):
    sample = np.random.choice(drift_vals, size=len(drift_vals), replace=True)
    boot_drifts.append(np.mean(sample))
boot_drifts = np.array(boot_drifts)
overall_ci = np.percentile(boot_drifts, [2.5, 97.5])
overall_p = np.mean(boot_drifts <= 0)

# Also test GapDn_VWAP_Brk + Early specifically
gapdn_vb_early = gapdn[(gapdn['scenario'] == 'GapDn_VWAP_Brk') & (gapdn['cross_bar'] >= 15) & (gapdn['cross_bar'] < 45)]
if len(gapdn_vb_early) > 0:
    vb_early_vals = gapdn_vb_early['drift_30'].dropna().values
    vb_boots = [np.mean(np.random.choice(vb_early_vals, size=len(vb_early_vals), replace=True)) for _ in range(5000)]
    vb_ci = np.percentile(vb_boots, [2.5, 97.5])
    vb_p = np.mean([b <= 0 for b in vb_boots])
else:
    vb_ci = [np.nan, np.nan]
    vb_p = np.nan

# Train-Val stability check for top combo
train = df[df['split'] == 'train']
gapdn_train = train[train['gap_direction'] == 'down']
train_drift, train_n, train_label = best_combo_drift(gapdn_train)

# Write results
output_path = os.path.join(RESULTS_DIR, 'behavioral_permutation_test.txt')
with open(output_path, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("PERMUTATION TEST — Devil's Advocate Counter-Test\n")
    f.write("Testet ob beste Subgruppe durch multiples Testen entstand\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"GapDn Val Events: {len(gapdn)}\n")
    f.write(f"Permutationen: {N_PERM}\n\n")

    f.write("--- TEST 1: Beste Subgruppe vs Permutationen ---\n")
    f.write(f"  Reale beste Combo: {real_label}\n")
    f.write(f"  Realer Drift30: {real_drift:.4f} (N={real_n})\n")
    f.write(f"  Permutation Mean: {perm_best_drifts.mean():.4f}\n")
    f.write(f"  Permutation Std: {perm_best_drifts.std():.4f}\n")
    f.write(f"  Permutation 95th Percentile: {np.percentile(perm_best_drifts, 95):.4f}\n")
    f.write(f"  Permutation 99th Percentile: {np.percentile(perm_best_drifts, 99):.4f}\n")
    f.write(f"  P-Value (perm >= real): {p_value:.4f}\n")
    f.write(f"  Signifikant bei 5%: {'JA' if p_value < 0.05 else 'NEIN'}\n")
    f.write(f"  Signifikant bei 1%: {'JA' if p_value < 0.01 else 'NEIN'}\n\n")

    f.write("--- TEST 2: GapDn Overall Drift (Bootstrap) ---\n")
    f.write(f"  Mean Drift30: {overall_drift:.4f}\n")
    f.write(f"  95% CI: [{overall_ci[0]:.4f}, {overall_ci[1]:.4f}]\n")
    f.write(f"  P(<=0): {overall_p:.4f}\n")
    f.write(f"  Signifikant: {'JA' if overall_p < 0.05 else 'NEIN'}\n\n")

    f.write("--- TEST 3: GapDn_VWAP_Brk + Early (Bootstrap) ---\n")
    f.write(f"  N: {len(gapdn_vb_early)}\n")
    f.write(f"  Mean Drift30: {gapdn_vb_early['drift_30'].mean():.4f}\n")
    f.write(f"  95% CI: [{vb_ci[0]:.4f}, {vb_ci[1]:.4f}]\n")
    f.write(f"  P(<=0): {vb_p:.4f}\n")
    f.write(f"  Signifikant: {'JA' if vb_p < 0.05 else 'NEIN'}\n\n")

    f.write("--- TEST 4: Train-Val Stabilitaet der besten Combo ---\n")
    f.write(f"  Train beste Combo: {train_label}\n")
    f.write(f"  Train Drift30: {train_drift:.4f} (N={train_n})\n")
    f.write(f"  Val beste Combo: {real_label}\n")
    f.write(f"  Val Drift30: {real_drift:.4f} (N={real_n})\n")
    f.write(f"  GLEICHE Combo? {'JA' if train_label == real_label else 'NEIN — instabil!'}\n\n")

    f.write("--- INTERPRETATION ---\n")
    if p_value < 0.05:
        f.write("Die beste Subgruppe ist signifikant besser als Zufall.\n")
        f.write("Aber Vorsicht: P-Value knapp am Grenzwert = fragwuerdig.\n")
    else:
        f.write("Die beste Subgruppe ist NICHT signifikant besser als Zufall.\n")
        f.write("Das Ergebnis koennte durch multiples Testen entstanden sein.\n")

    if overall_p < 0.05:
        f.write(f"\nGapDn overall Drift ist signifikant positiv (P={overall_p:.4f}).\n")
        f.write("Der Richtungseffekt (GapDn -> Mean Reversion) existiert als Basis-Phaenomen.\n")
    else:
        f.write(f"\nGapDn overall Drift ist NICHT signifikant (P={overall_p:.4f}).\n")

    f.write("\n" + "=" * 80 + "\n")
    f.write("ENDE PERMUTATION TEST\n")
    f.write("=" * 80 + "\n")

print(f"Ergebnisse: {output_path}", file=sys.stderr)
