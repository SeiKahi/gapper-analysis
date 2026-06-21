###############################################################################
# MONTE CARLO STYLE — GAPPER OVERLAY CHARTS
#
# Plottet jeden Gapper-Tag als transparente Linie (Z-Score vs Zeit).
# Wo sich viele Linien überlagern → dunklere Farbe = wahrscheinlicherer Pfad.
#
# VWAP = 0, +1σ = 1, -1σ = -1, etc.
#
# Ausführen: cd gapper-analysis
#            .\gapper_env\Scripts\python.exe vwap_montecarlo.py
###############################################################################

import pandas as pd
import numpy as np
import glob
import matplotlib
matplotlib.use('Agg')  # Headless
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# ============================================================
# CONFIG
# ============================================================
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[0]

VWAP_DIR = str(PROJECT_ROOT / 'data' / 'vwap')
METADATA_PATH = str(PROJECT_ROOT / 'data' / 'metadata' / 'metadata_master.parquet')
OUTPUT_DIR = str(PROJECT_ROOT / 'charts')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# METADATA
# ============================================================
print("Lade Metadata...")
meta = pd.read_parquet(METADATA_PATH)
meta['key'] = meta['ticker'] + '_' + meta['date']
meta_lookup = meta.set_index('key')[[
    'gap_direction', 'gap_size_in_adr', 'rvol_at_time_30min'
]].to_dict('index')

print(f"Metadata: {len(meta)} Gapper")

# ============================================================
# DATEN SAMMELN
# ============================================================
print("\nSammle Z-Score Pfade...")
files = glob.glob(f'{VWAP_DIR}/**/*.parquet', recursive=True)

paths_gap_up = []
paths_gap_up_rvol = []
paths_gap_down = []
paths_gap_down_rvol = []

for filepath in tqdm(files, desc="Lade"):
    try:
        df = pd.read_parquet(filepath, columns=['time_et', 'z_score', 'ticker', 'date'])
        
        if len(df) < 30:
            continue
        
        ticker = df['ticker'].iloc[0]
        date = df['date'].iloc[0]
        key = f"{ticker}_{date}"
        
        info = meta_lookup.get(key)
        if info is None:
            continue
        
        gap_adr = abs(info['gap_size_in_adr']) if not pd.isna(info['gap_size_in_adr']) else 0
        rvol = info['rvol_at_time_30min'] if not pd.isna(info['rvol_at_time_30min']) else 0
        gap_dir = info['gap_direction']
        
        # Filter: Gap > 1 ADR
        if gap_adr < 1.0:
            continue
        
        # Z-Score Pfad extrahieren
        df = df[(df['time_et'] >= '09:30') & (df['time_et'] <= '16:00')].copy()
        df = df.dropna(subset=['z_score'])
        
        if len(df) < 20:
            continue
        
        z_scores = df['z_score'].values.clip(-5, 5)
        
        # Zeit als Minuten nach Open
        times = []
        for t in df['time_et'].values:
            h, m = int(t[:2]), int(t[3:5])
            minutes = (h - 9) * 60 + (m - 30)
            times.append(minutes)
        times = np.array(times)
        
        raw_path = (times, z_scores)
        
        if gap_dir == 'up':
            paths_gap_up.append(raw_path)
            if rvol >= 1.5:
                paths_gap_up_rvol.append(raw_path)
        else:
            paths_gap_down.append(raw_path)
            if rvol >= 1.5:
                paths_gap_down_rvol.append(raw_path)
    
    except Exception:
        continue

print(f"\nGesammelt:")
print(f"  Gap Up  (>1 ADR):            {len(paths_gap_up):,}")
print(f"  Gap Up  (>1 ADR, RVOL>=1.5): {len(paths_gap_up_rvol):,}")
print(f"  Gap Down (>1 ADR):           {len(paths_gap_down):,}")
print(f"  Gap Down (>1 ADR, RVOL>=1.5):{len(paths_gap_down_rvol):,}")


# ============================================================
# INTERPOLATE ALL PATHS TO COMMON GRID
# ============================================================
def interpolate_paths(paths, time_grid):
    """Interpoliert alle Pfade auf ein gemeinsames Zeitgitter."""
    n = len(paths)
    all_z = np.full((n, len(time_grid)), np.nan)
    
    for idx, (times, z_scores) in enumerate(paths):
        if len(times) < 2:
            continue
        # Numpy Interpolation (schnell)
        valid = np.isfinite(z_scores)
        if valid.sum() < 2:
            continue
        all_z[idx, :] = np.interp(
            time_grid, times[valid], z_scores[valid],
            left=np.nan, right=np.nan
        )
    
    return all_z


# ============================================================
# PLOT FUNCTION
# ============================================================

def plot_montecarlo(paths, title, filename, color='#00aaff'):
    """Plottet alle Pfade als transparente Linien mit VWAP/StdDev Levels."""
    
    n = len(paths)
    if n == 0:
        print(f"  Keine Daten für {filename}")
        return
    
    fig, ax = plt.subplots(figsize=(18, 10), dpi=150)
    
    # ──────────────────────────────────────────────────────
    # Alpha berechnen
    # ──────────────────────────────────────────────────────
    if n < 50:
        alpha = 0.15
        lw = 0.8
    elif n < 200:
        alpha = 0.06
        lw = 0.5
    elif n < 500:
        alpha = 0.025
        lw = 0.4
    elif n < 1000:
        alpha = 0.012
        lw = 0.3
    elif n < 2000:
        alpha = 0.007
        lw = 0.25
    else:
        alpha = 0.004
        lw = 0.2
    
    # ──────────────────────────────────────────────────────
    # Alle Pfade plotten
    # ──────────────────────────────────────────────────────
    for times, z_scores in paths:
        ax.plot(times, z_scores, color=color, alpha=alpha, linewidth=lw, solid_capstyle='round')
    
    # ──────────────────────────────────────────────────────
    # Statistiken berechnen
    # ──────────────────────────────────────────────────────
    time_grid = np.arange(0, 391, 1)
    all_z = interpolate_paths(paths, time_grid)
    
    with np.errstate(all='ignore'):
        median_z = np.nanmedian(all_z, axis=0)
        p25_z = np.nanpercentile(all_z, 25, axis=0)
        p75_z = np.nanpercentile(all_z, 75, axis=0)
        p10_z = np.nanpercentile(all_z, 10, axis=0)
        p90_z = np.nanpercentile(all_z, 90, axis=0)
    
    # 25-75% Band
    ax.fill_between(time_grid, p25_z, p75_z, color=color, alpha=0.12, zorder=5, label='25–75% Band')
    
    # 10/90% gestrichelt
    ax.plot(time_grid, p10_z, color=color, linewidth=1.0, linestyle=':', alpha=0.6, zorder=6, label='10/90% Perzentil')
    ax.plot(time_grid, p90_z, color=color, linewidth=1.0, linestyle=':', alpha=0.6, zorder=6)
    
    # Median als dicke Linie
    ax.plot(time_grid, median_z, color='white', linewidth=3.5, zorder=10)
    ax.plot(time_grid, median_z, color=color, linewidth=2.2, zorder=11, label='Median')
    
    # ──────────────────────────────────────────────────────
    # VWAP und StdDev Levels
    # ──────────────────────────────────────────────────────
    # Hintergrund-Zonen einfärben
    zone_alpha = 0.04
    ax.axhspan(-1, 1, color='green', alpha=zone_alpha)
    ax.axhspan(1, 2, color='yellow', alpha=zone_alpha)
    ax.axhspan(-2, -1, color='yellow', alpha=zone_alpha)
    ax.axhspan(2, 3, color='orange', alpha=zone_alpha)
    ax.axhspan(-3, -2, color='orange', alpha=zone_alpha)
    
    # Level-Linien
    ax.axhline(y=0, color='#ffdd00', linewidth=1.8, zorder=15, label='VWAP')
    
    for z, clr, lbl in [(1, '#00dd00', '+1σ'), (-1, '#00dd00', '-1σ'),
                         (2, '#ff8800', '+2σ'), (-2, '#ff8800', '-2σ'),
                         (3, '#ff2222', '+3σ'), (-3, '#ff2222', '-3σ')]:
        ax.axhline(y=z, color=clr, linewidth=1.0, linestyle='--', alpha=0.7, zorder=15)
    
    # Level-Labels rechts
    for z, label, clr in [(0, 'VWAP', '#ffdd00'), (1, '+1σ', '#00dd00'), (-1, '-1σ', '#00dd00'),
                           (2, '+2σ', '#ff8800'), (-2, '-2σ', '#ff8800'),
                           (3, '+3σ', '#ff2222'), (-3, '-3σ', '#ff2222')]:
        ax.text(393, z, f' {label}', fontsize=9, va='center', fontweight='bold',
                color=clr, zorder=20)
    
    # ──────────────────────────────────────────────────────
    # X-Achse: Uhrzeiten
    # ──────────────────────────────────────────────────────
    time_ticks = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360, 390]
    time_labels = ['9:30', '10:00', '10:30', '11:00', '11:30', '12:00', '12:30',
                   '13:00', '13:30', '14:00', '14:30', '15:00', '15:30', '16:00']
    ax.set_xticks(time_ticks)
    ax.set_xticklabels(time_labels, rotation=45, fontsize=9)
    
    # Vertikale Zeitlinien
    for t in time_ticks:
        ax.axvline(x=t, color='gray', linestyle=':', alpha=0.15, linewidth=0.5)
    
    # ──────────────────────────────────────────────────────
    # Statistik-Annotations
    # ──────────────────────────────────────────────────────
    # Wo ist der Median zu bestimmten Zeiten?
    key_times = [(30, '10:00'), (60, '10:30'), (120, '11:30'), (210, '13:00'), (390, '16:00')]
    for t_min, t_label in key_times:
        if t_min < len(median_z):
            med_val = median_z[t_min]
            if not np.isnan(med_val):
                ax.plot(t_min, med_val, 'o', color='white', markersize=4, zorder=12)
    
    # ──────────────────────────────────────────────────────
    # Styling
    # ──────────────────────────────────────────────────────
    ax.set_facecolor('#0d1117')
    fig.set_facecolor('#010409')
    
    ax.set_xlim(-5, 400)
    ax.set_ylim(-5, 5)
    ax.set_xlabel('Uhrzeit (ET)', fontsize=12, color='#c9d1d9')
    ax.set_ylabel('Z-Score (Abstand vom VWAP in StdDevs)', fontsize=12, color='#c9d1d9')
    ax.set_title(f'{title}', fontsize=16, color='white', fontweight='bold', pad=15)
    
    ax.tick_params(colors='#8b949e', labelsize=9)
    for spine in ax.spines.values():
        spine.set_color('#30363d')
    ax.grid(True, alpha=0.08, color='#8b949e')
    
    # Legend
    legend = ax.legend(loc='upper left', fontsize=9, facecolor='#161b22',
                       edgecolor='#30363d', labelcolor='#c9d1d9', ncol=2,
                       framealpha=0.9)
    
    # Info-Box
    eod_z = all_z[:, -1]
    valid_eod = eod_z[~np.isnan(eod_z)]
    above_vwap_pct = (valid_eod > 0).mean() * 100 if len(valid_eod) > 0 else 0
    
    # Wie viele erreichen +2σ / -2σ irgendwann?
    max_z = np.nanmax(all_z, axis=1)
    min_z = np.nanmin(all_z, axis=1)
    reach_plus2 = (max_z >= 2).mean() * 100
    reach_minus2 = (min_z <= -2).mean() * 100
    reach_plus3 = (max_z >= 3).mean() * 100
    reach_minus3 = (min_z <= -3).mean() * 100
    
    info_lines = [
        f'n = {n:,} Tage',
        f'',
        f'Median EOD: {median_z[-1]:+.2f}σ',
        f'Close über VWAP: {above_vwap_pct:.0f}%',
        f'',
        f'Erreicht +2σ: {reach_plus2:.0f}%',
        f'Erreicht -2σ: {reach_minus2:.0f}%',
        f'Erreicht +3σ: {reach_plus3:.0f}%',
        f'Erreicht -3σ: {reach_minus3:.0f}%',
    ]
    info_text = '\n'.join(info_lines)
    
    ax.text(0.985, 0.97, info_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#161b22', edgecolor='#30363d', alpha=0.95),
            color='#c9d1d9')
    
    plt.tight_layout()
    
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✅ {filepath}")


# ============================================================
# CHARTS GENERIEREN
# ============================================================
print("\n" + "=" * 60)
print("Generiere Charts...")
print("=" * 60)

# 1. Gap Up > 1 ADR — Alle
plot_montecarlo(
    paths_gap_up,
    'Gap Up > 1 ADR — Alle Tage',
    'montecarlo_gapup_1adr.png',
    color='#4da6ff'
)

# 2. Gap Down > 1 ADR — Alle
plot_montecarlo(
    paths_gap_down,
    'Gap Down > 1 ADR — Alle Tage',
    'montecarlo_gapdown_1adr.png',
    color='#ff5555'
)

# 3. Gap Up > 1 ADR + RVOL ≥ 1.5
plot_montecarlo(
    paths_gap_up_rvol,
    'Gap Up > 1 ADR + RVOL ≥ 1.5 — Stock in Play',
    'montecarlo_gapup_1adr_rvol15.png',
    color='#4da6ff'
)

# 4. Gap Down > 1 ADR + RVOL ≥ 1.5
plot_montecarlo(
    paths_gap_down_rvol,
    'Gap Down > 1 ADR + RVOL ≥ 1.5 — Stock in Play',
    'montecarlo_gapdown_1adr_rvol15.png',
    color='#ff5555'
)

print("\n" + "=" * 60)
print(f"✅ Alle Charts gespeichert in: {OUTPUT_DIR}")
print("=" * 60)
