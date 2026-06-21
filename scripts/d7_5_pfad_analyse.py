"""
Durchlauf 7.5 -- Aufgabe 4: MFE/MAE Pfad-Analyse fuer Top-Zellen (IS)
Laedt 1-Minuten-Daten fuer die interessantesten Zellen.
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys, os, warnings, glob
warnings.filterwarnings('ignore')

# === LOAD METADATA ===
df = pd.read_parquet('data/metadata/metadata_v7.parquet')
h1 = df[(df['date'] >= '2021-02-21') & (df['date'] <= '2023-12-31')].copy()
h1 = h1.dropna(subset=['pm_rth5', 'rvol_5', 'gap_size_in_adr', 'full_drift', 'close_935', 'adr_10'])
h1['gap_dir'] = h1['gap_direction'].map({'up': 'GapUp', 'down': 'GapDown'})

# Buckets (3x3x3 for robustness)
def pm5_alt(x):
    if x < 0.50: return 'PM5_LO'
    elif x < 1.00: return 'PM5_MID'
    else: return 'PM5_HI'
def rv5_alt(x):
    if x < 3: return 'RV5_LO'
    elif x < 7: return 'RV5_MID'
    else: return 'RV5_HI'
def gap_bucket(x):
    if x < 1: return 'GAP_SM'
    elif x < 2: return 'GAP_MD'
    else: return 'GAP_LG'

h1['pm5'] = h1['pm_rth5'].apply(pm5_alt)
h1['rv5'] = h1['rvol_5'].apply(rv5_alt)
h1['gap'] = h1['gap_size_in_adr'].apply(gap_bucket)

# Load the 3x3x3 results to find top cells
matrix = pd.read_parquet('results/d7_5_3er_matrix_3x3x3.parquet')

# Top-5 Continuation cells (highest full_drift, N>=20)
cont_cells = matrix[matrix['N'] >= 20].nlargest(5, 'fd')
# Top-5 Fade cells (lowest full_drift, N>=20)
fade_cells = matrix[matrix['N'] >= 20].nsmallest(5, 'fd')
top_cells = pd.concat([cont_cells, fade_cells])

print(f"Top-10 Zellen fuer Pfad-Analyse:", file=sys.stderr)
for _, r in top_cells.iterrows():
    print(f"  {r['dir']} {r['pm5']} {r['rv5']} {r['gap']} N={r['N']:.0f} fd={r['fd']:+.3f}", file=sys.stderr)

# === LOAD 1-MIN DATA FOR SELECTED GAPPERS ===
def load_1min(ticker, date_str):
    """Load 1-min data for a given ticker/date."""
    path = f"data/raw_1min/{ticker}/{date_str}.parquet"
    if not os.path.exists(path):
        return None
    try:
        bars = pd.read_parquet(path)
        # Filter RTH bars (9:30 - 16:00)
        if 'datetime_et' in bars.columns:
            bars['datetime_et'] = pd.to_datetime(bars['datetime_et'])
            bars = bars[(bars['datetime_et'].dt.hour >= 9) & (bars['datetime_et'].dt.hour < 16)]
            bars = bars[~((bars['datetime_et'].dt.hour == 9) & (bars['datetime_et'].dt.minute < 30))]
            bars = bars.sort_values('datetime_et').reset_index(drop=True)
            bars['min_since_930'] = (bars['datetime_et'].dt.hour - 9) * 60 + bars['datetime_et'].dt.minute - 30
        elif 'time_et' in bars.columns:
            bars['time_str'] = bars['time_et'].astype(str)
            bars = bars[bars['time_str'] >= '09:30'].copy()
            bars = bars[bars['time_str'] < '16:00'].copy()
            bars = bars.sort_values('time_et').reset_index(drop=True)
            # Parse time
            def parse_min(t):
                parts = str(t).split(':')
                h, m = int(parts[0]), int(parts[1])
                return (h - 9) * 60 + m - 30
            bars['min_since_930'] = bars['time_et'].apply(parse_min)
        return bars
    except:
        return None

# Time points for reporting
time_points = [5, 10, 15, 20, 25, 30, 45, 60, 90, 120, 150, 180, 210, 270, 330, 385]
time_labels = ['09:35', '09:40', '09:45', '09:50', '09:55', '10:00', '10:15', '10:30',
               '11:00', '11:30', '12:00', '12:30', '13:00', '14:00', '15:00', '15:55']

out = open('results/d7_5_pfad_analyse.txt', 'w', encoding='utf-8')
def p(text=''):
    out.write(text + '\n')

p("=" * 110)
p("DURCHLAUF 7.5 -- AUFGABE 4: MFE/MAE PFAD-ANALYSE (Top-10 Zellen)")
p("=" * 110)

# Process each cell
all_path_data = []

for cell_idx, (_, cell) in enumerate(top_cells.iterrows()):
    direction = cell['dir']
    pm5_val = cell['pm5']
    rv5_val = cell['rv5']
    gap_val = cell['gap']
    cell_fd = cell['fd']
    cell_type = "CONTINUATION" if cell_fd > 0 else "FADE"

    p(f"\n{'='*110}")
    p(f"Zelle #{cell_idx+1}: {direction} | {pm5_val} | {rv5_val} | {gap_val}")
    p(f"  Typ: {cell_type} | N={cell['N']:.0f} | full_drift={cell_fd:+.3f} | rest_drift={cell['rd']:+.3f}")
    p(f"{'='*110}")

    # Select gappers for this cell
    mask = (h1['gap_dir'] == direction) & (h1['pm5'] == pm5_val) & (h1['rv5'] == rv5_val) & (h1['gap'] == gap_val)
    gappers = h1[mask].copy()
    p(f"  Gapper in Zelle: {len(gappers)}")

    # Load 1-min data for each gapper
    paths = []
    loaded = 0
    failed = 0

    for _, gapper in tqdm(gappers.iterrows(), total=len(gappers), desc=f"Cell {cell_idx+1}", file=sys.stderr):
        ticker = gapper['ticker']
        date_str = str(gapper['date'])
        close_935 = gapper['close_935']
        adr = gapper['adr_10']

        if pd.isna(close_935) or pd.isna(adr) or adr <= 0:
            failed += 1
            continue

        bars = load_1min(ticker, date_str)
        if bars is None or len(bars) < 10:
            failed += 1
            continue

        # Get close prices at each time point relative to close_935
        # Favorable = in gap direction, Adverse = against gap direction
        gap_sign = 1 if direction == 'GapUp' else -1

        path_data = {'cell_idx': cell_idx, 'ticker': ticker, 'date': date_str}

        # For each minute, compute running MFE and MAE
        for tp in time_points:
            bars_up_to = bars[bars['min_since_930'] <= tp]
            if len(bars_up_to) == 0:
                continue

            # Close at this time (use last available bar)
            bar_at = bars[bars['min_since_930'] == tp]
            if len(bar_at) > 0:
                close_at = bar_at.iloc[0]['close']
            else:
                close_at = bars_up_to.iloc[-1]['close']

            # Bars from 5 min (9:35) onwards up to this time
            bars_window = bars[(bars['min_since_930'] >= 5) & (bars['min_since_930'] <= tp)]
            if len(bars_window) == 0:
                continue

            # MFE: max favorable excursion from close_935
            highs = bars_window['high'].values
            lows = bars_window['low'].values

            if gap_sign == 1:  # GapUp, continuation = up
                mfe_price = np.max(highs)
                mae_price = np.min(lows)
            else:  # GapDown, continuation = down
                mfe_price = np.min(lows)
                mae_price = np.max(highs)

            mfe = gap_sign * (mfe_price - close_935) / adr
            mae = -gap_sign * (mae_price - close_935) / adr  # adverse = positive number
            drift_at = gap_sign * (close_at - close_935) / adr

            path_data[f'mfe_{tp}'] = mfe
            path_data[f'mae_{tp}'] = mae
            path_data[f'drift_{tp}'] = drift_at

        paths.append(path_data)
        loaded += 1

    p(f"  Loaded: {loaded}, Failed: {failed}")

    if loaded < 5:
        p("  SKIP: Nicht genug 1-min Daten")
        continue

    path_df = pd.DataFrame(paths)

    # Report path statistics
    p(f"\n  Pfad-Statistiken (in ADR, ab 9:35):")
    p(f"  {'Time':>6s} | {'MFE_mean':>8s} {'MFE_med':>8s} | {'MAE_mean':>8s} {'MAE_med':>8s} | {'Drift_mean':>10s} {'Drift_med':>10s}")
    p(f"  {'-'*75}")

    for tp, label in zip(time_points, time_labels):
        mfe_col = f'mfe_{tp}'
        mae_col = f'mae_{tp}'
        drift_col = f'drift_{tp}'
        if mfe_col not in path_df.columns:
            continue

        mfe_vals = path_df[mfe_col].dropna()
        mae_vals = path_df[mae_col].dropna()
        drift_vals = path_df[drift_col].dropna()

        if len(mfe_vals) < 3:
            continue

        p(f"  {label:>6s} | {mfe_vals.mean():>+8.3f} {mfe_vals.median():>+8.3f} | "
          f"{mae_vals.mean():>+8.3f} {mae_vals.median():>+8.3f} | "
          f"{drift_vals.mean():>+10.3f} {drift_vals.median():>+10.3f}")

    # 4c: Peak time and drift concentration
    # Find when max drift is reached (use mean drift at each time point)
    drift_by_time = {}
    for tp in time_points:
        drift_col = f'drift_{tp}'
        if drift_col in path_df.columns:
            vals = path_df[drift_col].dropna()
            if len(vals) >= 3:
                drift_by_time[tp] = vals.mean()

    if drift_by_time:
        if cell_type == "CONTINUATION":
            peak_time = max(drift_by_time, key=drift_by_time.get)
            peak_drift = drift_by_time[peak_time]
        else:  # FADE
            peak_time = min(drift_by_time, key=drift_by_time.get)
            peak_drift = drift_by_time[peak_time]

        final_drift = drift_by_time.get(385, drift_by_time.get(330, 0))
        drift_at_30 = drift_by_time.get(30, 0)  # 10:00
        drift_at_90 = drift_by_time.get(90, 0)  # 11:00

        pct_at_1000 = (drift_at_30 / final_drift * 100) if abs(final_drift) > 0.001 else 0
        pct_at_1100 = (drift_at_90 / final_drift * 100) if abs(final_drift) > 0.001 else 0

        peak_label = f"{9 + (peak_time + 30) // 60}:{(peak_time + 30) % 60:02d}"
        p(f"\n  4c Zusammenfassung:")
        p(f"    Peak-Drift-Zeit: {peak_label} ({peak_drift:+.3f} ADR)")
        p(f"    % des Tages-Drift um 10:00: {pct_at_1000:+.0f}%")
        p(f"    % des Tages-Drift um 11:00: {pct_at_1100:+.0f}%")
        if abs(drift_at_30) > abs(final_drift) * 0.5:
            p(f"    -> Drift konzentriert in den ersten 30 Min (9:35-10:00)")
        elif abs(drift_at_90) > abs(final_drift) * 0.7:
            p(f"    -> Drift konzentriert bis 11:00")
        else:
            p(f"    -> Drift verteilt sich ueber den ganzen Tag")

    # Save per cell
    path_df['cell_type'] = cell_type
    path_df['direction'] = direction
    path_df['pm5'] = pm5_val
    path_df['rv5'] = rv5_val
    path_df['gap_b'] = gap_val
    all_path_data.append(path_df)

# Save all path data
if all_path_data:
    all_paths = pd.concat(all_path_data, ignore_index=True)
    all_paths.to_parquet('results/d7_5_pfad_raw.parquet', index=False)
    p(f"\nGespeichert: {len(all_paths)} Pfade in results/d7_5_pfad_raw.parquet")

out.close()
print("Done!", file=sys.stderr)
