###############################################################################
# ZEITFENSTER-ANALYSE v1
#
# Beantwortet: "Wie SCHNELL erreicht der Preis das Target?"
#
# Fuer die Top-Setups misst dieses Script:
#   1. Time-to-Target: % die Target in 15/30/60/120/240min erreichen
#   2. MAE (Max Adverse Excursion): Wie weit laeuft es GEGEN dich?
#   3. Pfad-Qualitaet: Direkter Move oder Hin-und-Her?
#   4. Alles aufgeschluesselt nach Gewinnern vs Verlierern
#
# Ausfuehren: cd gapper-analysis
#   .\gapper_env\Scripts\python.exe zeitfenster_analyse.py > zeitfenster.txt 2>&1
###############################################################################

import pandas as pd
import numpy as np
import glob
from tqdm import tqdm
import warnings
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[0]

VWAP_DIR = str(PROJECT_ROOT / 'data' / 'vwap')
METADATA_PATH = str(PROJECT_ROOT / 'data' / 'metadata' / 'metadata_master.parquet')

MIN_BARS_AFTER_OPEN = 20

# Zeitfenster in Minuten
TIME_WINDOWS = [5, 10, 15, 30, 60, 120, 240]

# ============================================================
# METADATA
# ============================================================
print("Lade Metadata...")
meta = pd.read_parquet(METADATA_PATH)
meta['key'] = meta['ticker'] + '_' + meta['date']

meta_lookup = meta.set_index('key')[[
    'gap_direction', 'gap_size_in_adr', 'adr_10', 'rvol_at_time_30min'
]].to_dict('index')

print(f"Metadata: {len(meta)} Gapper")

# ============================================================
# LEVEL HELPERS
# ============================================================
LEVEL_NAMES = ['-3s', '-2s', '-1s', 'VWAP', '+1s', '+2s', '+3s']

def price_to_zone(price, levels):
    for i, lvl in enumerate(levels):
        if price < lvl:
            return i
    return len(levels)

# ============================================================
# EARLY RANGE
# ============================================================
def compute_early_range(df, adr_10):
    if adr_10 <= 0:
        return 0.0, 0.0
    market = df[(df['time_et'] >= '09:30') & (df['time_et'] <= '16:00')].copy()
    if len(market) == 0:
        return 0.0, 0.0
    times = market['time_et'].values
    use_hl = 'high' in market.columns and 'low' in market.columns
    highs = market['high'].values if use_hl else market['close'].values
    lows = market['low'].values if use_hl else market['close'].values
    
    mask_15 = times < '09:45'
    range_15 = (highs[mask_15].max() - lows[mask_15].min()) / adr_10 if mask_15.any() else 0.0
    mask_30 = times < '10:00'
    range_30 = (highs[mask_30].max() - lows[mask_30].min()) / adr_10 if mask_30.any() else 0.0
    return round(range_15, 3), round(range_30, 3)

# ============================================================
# DETAILLIERTE SWING DETECTION
#   Speichert den KOMPLETTEN PFAD fuer jeden Swing
# ============================================================
def detect_swings_detailed(df, adr_10):
    swings = []
    df = df.dropna(subset=['vwap', 'std_dev']).copy()
    df = df[df['std_dev'] > 0].copy()
    
    if len(df) < MIN_BARS_AFTER_OPEN + 10:
        return swings
    
    n = len(df)
    closes = df['close'].values
    times = df['time_et'].values
    
    # Running %ADR
    running_high = np.maximum.accumulate(closes)
    running_low = np.minimum.accumulate(closes)
    pct_adr = (running_high - running_low) / adr_10 if adr_10 > 0 else np.zeros(n)
    
    # Zonen + Levels pro Bar
    zones = []
    all_levels = []
    for _, row in df.iterrows():
        lvls = [row['lower_3std'], row['lower_2std'], row['lower_1std'],
                row['vwap'], row['upper_1std'], row['upper_2std'], row['upper_3std']]
        all_levels.append(lvls)
        zones.append(price_to_zone(row['close'], lvls))
    
    for i in range(MIN_BARS_AFTER_OPEN, n - 5):
        prev_zone = zones[i - 1]
        curr_zone = zones[i]
        
        if prev_zone == curr_zone:
            continue
        
        if curr_zone > prev_zone:
            direction = 'up'
            crossed_level_idx = prev_zone
            start_zone = curr_zone
        else:
            direction = 'down'
            crossed_level_idx = curr_zone
            start_zone = curr_zone
        
        if crossed_level_idx >= len(LEVEL_NAMES):
            continue
        crossed_level = LEVEL_NAMES[crossed_level_idx]
        
        entry_price = closes[i]
        entry_time = times[i]
        
        # ===== DETAILLIERTER PFAD-TRACKING =====
        max_levels_reached = 0
        max_move_adr = 0.0
        max_adverse_adr = 0.0
        outcome = 'unresolved'
        bars_to_end = 0
        
        # Zeitpunkt wann jedes Level erreicht wird
        time_to_level = {}  # level_num -> bars_after_entry
        
        # MAE zu verschiedenen Zeitpunkten (in Bars = Minuten)
        mae_at_bar = {}     # bar_offset -> max_adverse bis dahin
        move_at_bar = {}    # bar_offset -> max_favorable bis dahin
        
        running_mae = 0.0
        running_mfe = 0.0   # Max Favorable Excursion
        
        # Pfad-Tracking: Jede Bar speichern
        path_moves = []     # Liste von (bar_offset, move_adr, adverse_adr, zone)
        
        for j in range(i + 1, n):
            bar_offset = j - i  # Minuten seit Entry (1min bars)
            price = closes[j]
            j_zone = zones[j]
            
            # Move berechnen
            if adr_10 > 0:
                if direction == 'up':
                    move = (price - entry_price) / adr_10
                    adverse = (entry_price - price) / adr_10
                else:
                    move = (entry_price - price) / adr_10
                    adverse = (price - entry_price) / adr_10
                
                running_mfe = max(running_mfe, move)
                running_mae = max(running_mae, max(0, adverse))
                max_move_adr = max(max_move_adr, move)
                max_adverse_adr = max(max_adverse_adr, max(0, adverse))
            else:
                move = 0
                adverse = 0
            
            # Snapshot bei bestimmten Zeitpunkten
            if bar_offset in TIME_WINDOWS:
                mae_at_bar[bar_offset] = round(running_mae, 4)
                move_at_bar[bar_offset] = round(running_mfe, 4)
            
            # Levels tracking
            if direction == 'up':
                levels_above = j_zone - start_zone
                if levels_above > max_levels_reached:
                    max_levels_reached = levels_above
                    time_to_level[max_levels_reached] = bar_offset
                if j_zone <= prev_zone:
                    outcome = 'rejected'
                    bars_to_end = bar_offset
                    break
            else:
                levels_below = start_zone - j_zone
                if levels_below > max_levels_reached:
                    max_levels_reached = levels_below
                    time_to_level[max_levels_reached] = bar_offset
                if j_zone >= prev_zone:
                    outcome = 'rejected'
                    bars_to_end = bar_offset
                    break
        
        if outcome != 'rejected':
            outcome = f'continued_{max_levels_reached}' if max_levels_reached > 0 else 'held_zone'
            bars_to_end = n - i
        
        # Pfad-Effizienz: Wie direkt war der Move?
        # Ratio = Netto-Move / Summe aller absoluten Bar-zu-Bar Moves
        total_path = 0.0
        for j in range(i + 1, min(i + bars_to_end + 1, n)):
            if j > i + 1:
                bar_move = abs(closes[j] - closes[j-1])
                total_path += bar_move
        net_move = abs(closes[min(i + bars_to_end, n-1)] - entry_price)
        path_efficiency = (net_move / total_path) if total_path > 0 else 0
        
        swings.append({
            'time': entry_time,
            'direction': direction,
            'crossed_level': crossed_level,
            'max_levels_continued': max_levels_reached,
            'outcome': outcome,
            'bars_to_end': bars_to_end,
            'max_move_adr': round(max_move_adr, 4),
            'max_adverse_adr': round(max_adverse_adr, 4),
            'pct_adr_used': round(pct_adr[i], 3),
            'path_efficiency': round(path_efficiency, 3),
            # Zeitpunkt der Level-Erreichung
            'bars_to_lvl1': time_to_level.get(1, np.nan),
            'bars_to_lvl2': time_to_level.get(2, np.nan),
            'bars_to_lvl3': time_to_level.get(3, np.nan),
            # MAE Snapshots
            'mae_5m': mae_at_bar.get(5, np.nan),
            'mae_10m': mae_at_bar.get(10, np.nan),
            'mae_15m': mae_at_bar.get(15, np.nan),
            'mae_30m': mae_at_bar.get(30, np.nan),
            'mae_60m': mae_at_bar.get(60, np.nan),
            'mae_120m': mae_at_bar.get(120, np.nan),
            'mae_240m': mae_at_bar.get(240, np.nan),
            # MFE Snapshots
            'mfe_5m': move_at_bar.get(5, np.nan),
            'mfe_10m': move_at_bar.get(10, np.nan),
            'mfe_15m': move_at_bar.get(15, np.nan),
            'mfe_30m': move_at_bar.get(30, np.nan),
            'mfe_60m': move_at_bar.get(60, np.nan),
            'mfe_120m': move_at_bar.get(120, np.nan),
            'mfe_240m': move_at_bar.get(240, np.nan),
        })
    
    return swings

# ============================================================
# ALLE DATEIEN VERARBEITEN
# ============================================================
print("\nVerarbeite VWAP-Dateien (Detailliert)...")
files = glob.glob(f'{VWAP_DIR}/**/*.parquet', recursive=True)
print(f"{len(files)} Dateien gefunden")

all_swings = []

for filepath in tqdm(files, desc="Analysiere"):
    try:
        df = pd.read_parquet(filepath)
        if len(df) < 30:
            continue
        
        ticker = df['ticker'].iloc[0]
        date = df['date'].iloc[0]
        key = f"{ticker}_{date}"
        
        info = meta_lookup.get(key)
        if info is None:
            continue
        
        adr_10 = info['adr_10'] if not pd.isna(info['adr_10']) else 0
        if adr_10 <= 0:
            continue
        
        first_close = df[df['time_et'] >= '09:30']['close'].iloc[0] if len(df[df['time_et'] >= '09:30']) > 0 else df['close'].iloc[0]
        adr_pct = (adr_10 / first_close * 100) if first_close > 0 else 0
        
        er_15, er_30 = compute_early_range(df, adr_10)
        swings = detect_swings_detailed(df, adr_10)
        
        gap_adr = abs(info['gap_size_in_adr']) if not pd.isna(info['gap_size_in_adr']) else 0
        rvol = info['rvol_at_time_30min'] if not pd.isna(info['rvol_at_time_30min']) else 0
        
        for sw in swings:
            sw['ticker'] = ticker
            sw['date'] = date
            sw['gap_direction'] = info['gap_direction']
            sw['gap_adr'] = gap_adr
            sw['rvol'] = rvol
            sw['early_range_15'] = er_15
            sw['early_range_30'] = er_30
            sw['adr_pct'] = round(adr_pct, 2)
        
        all_swings.extend(swings)
    except Exception:
        continue

df_all = pd.DataFrame(all_swings)
if len(df_all) == 0:
    print("FEHLER: Keine Swings gefunden!")
    sys.exit()

print(f"\n{len(df_all):,} Swings erkannt")

# ============================================================
# TIME BUCKET
# ============================================================
def time_bucket(t):
    if t <= '10:00': return '1_open30'
    elif t <= '10:30': return '2_30_60'
    elif t <= '11:30': return '3_1_2h'
    elif t <= '13:00': return '4_mittag'
    else: return '5_nachm'

df_all['time_bucket'] = df_all['time'].apply(time_bucket)

# ============================================================
# SZENARIEN (Fokus auf die staerksten)
# ============================================================
setups = [
    ('GapUp +2s Fade',   'up',   '+2s', 'down', 'GAP UP: Aktie bei +2 StdDev -> Faellt sie?'),
    ('GapUp +1s Fade',   'up',   '+1s', 'down', 'GAP UP: Aktie bei +1 StdDev -> Faellt sie zum VWAP?'),
    ('GapUp VWAP Break', 'up',   'VWAP', 'down', 'GAP UP: VWAP Break nach unten -> Faellt sie zu -1 StdDev?'),
    ('GapDn -2s Fade',   'down', '-2s', 'up',   'GAP DOWN: Aktie bei -2 StdDev -> Steigt sie?'),
    ('GapDn -1s Fade',   'down', '-1s', 'up',   'GAP DOWN: Aktie bei -1 StdDev -> Steigt sie zum VWAP?'),
    ('GapDn VWAP Break', 'down', 'VWAP', 'up',  'GAP DOWN: VWAP Break nach oben -> Steigt sie zu +1 StdDev?'),
]

# Filter-Kombinationen
filters = {
    'ALLE':                lambda df: df,
    'ADR>=3%':             lambda df: df[df['adr_pct'] >= 3],
    'ADR>=3% Open30':      lambda df: df[(df['adr_pct'] >= 3) & (df['time_bucket'] == '1_open30')],
    'ADR>=3% 30-60min':    lambda df: df[(df['adr_pct'] >= 3) & (df['time_bucket'] == '2_30_60')],
    'ADR>=3% Morgen':      lambda df: df[(df['adr_pct'] >= 3) & (df['time_bucket'].isin(['1_open30', '2_30_60']))],
    'ADR>=3% <50%ADR':     lambda df: df[(df['adr_pct'] >= 3) & (df['pct_adr_used'] < 0.50)],
    'ADR>=3% Morg+<50%':   lambda df: df[(df['adr_pct'] >= 3) & (df['time_bucket'].isin(['1_open30', '2_30_60'])) & (df['pct_adr_used'] < 0.50)],
}


# ################################################################
#                     ANALYSEN
# ################################################################

# ================================================================
# ANALYSE 1: TIME-TO-TARGET
#   Wie schnell wird jedes Level erreicht?
# ================================================================
print()
print("=" * 140)
print("ANALYSE 1: TIME-TO-TARGET")
print("  Fuer jeden Setup+Filter: Wie viel % erreichen Level 1 innerhalb von X Minuten?")
print("  Zeigt ob der Move schnell und clean ist oder langsam und unsicher.")
print("=" * 140)

for s_name, gap_dir, level, swing_dir, desc in setups:
    base = df_all[(df_all['gap_direction'] == gap_dir) & 
                  (df_all['crossed_level'] == level) & 
                  (df_all['direction'] == swing_dir)]
    if len(base) < 100:
        continue
    
    print(f"\n{'=' * 140}")
    print(f"  {desc}")
    print(f"{'=' * 140}")
    
    for f_name, f_func in filters.items():
        sub = f_func(base)
        n = len(sub)
        if n < 30:
            continue
        
        # Gewinner = erreichen Level 1
        winners = sub[sub['max_levels_continued'] >= 1]
        n_win = len(winners)
        win_pct = n_win / n * 100
        
        # Time-to-Level1 Verteilung (nur Gewinner)
        ttl1 = winners['bars_to_lvl1'].dropna()
        
        print(f"\n  {f_name} (n={n:,}, Winners={n_win} = {win_pct:.1f}%)")
        
        if len(ttl1) < 10:
            print(f"    Zu wenig Gewinner fuer Time-Analyse")
            continue
        
        # Kumulative Verteilung: Wie viel % der Gewinner erreichen Target in X min?
        print(f"    Time-to-Level-1 (nur Gewinner, n={len(ttl1)}):")
        print(f"    {'Innerhalb':>12s}  {'% Gewinner':>11s}  {'% von ALLEN':>11s}  {'Kumulativ':>10s}")
        print(f"    {'-'*12}  {'-'*11}  {'-'*11}  {'-'*10}")
        
        for minutes in [5, 10, 15, 30, 60, 120, 240]:
            reached = (ttl1 <= minutes).sum()
            pct_winners = reached / len(ttl1) * 100
            pct_all = reached / n * 100
            bar = '#' * int(pct_winners / 3)
            print(f"    {minutes:>8d} min  {pct_winners:>10.1f}%  {pct_all:>10.1f}%  {reached:>5d}/{len(ttl1):<5d}  {bar}")
        
        # Percentile
        print(f"\n    Percentile Time-to-Level-1:")
        for p in [10, 25, 50, 75, 90]:
            val = ttl1.quantile(p/100)
            print(f"      P{p}: {val:.0f} min")
        
        # Level 2 (wenn genug Daten)
        winners2 = sub[sub['max_levels_continued'] >= 2]
        ttl2 = winners2['bars_to_lvl2'].dropna()
        if len(ttl2) >= 10:
            print(f"\n    Time-to-Level-2 (n={len(ttl2)}):")
            for minutes in [15, 30, 60, 120, 240]:
                reached = (ttl2 <= minutes).sum()
                pct = reached / len(ttl2) * 100
                pct_all = reached / n * 100
                print(f"    {minutes:>8d} min  {pct:>10.1f}% Gewinner  {pct_all:>10.1f}% von Allen")
            print(f"      Median: {ttl2.median():.0f} min")


# ================================================================
# ANALYSE 2: MAX ADVERSE EXCURSION (MAE)
#   Wie weit laeuft es gegen dich, bevor es zum Target geht?
# ================================================================
print()
print("=" * 140)
print("ANALYSE 2: MAX ADVERSE EXCURSION (MAE)")
print("  Gewinner vs Verlierer: Wie weit laeuft der Preis GEGEN die Swing-Richtung?")
print("  MAE = Max Adverse Excursion in ADR")
print("=" * 140)

for s_name, gap_dir, level, swing_dir, desc in setups:
    base = df_all[(df_all['gap_direction'] == gap_dir) & 
                  (df_all['crossed_level'] == level) & 
                  (df_all['direction'] == swing_dir)]
    if len(base) < 100:
        continue
    
    print(f"\n{'=' * 140}")
    print(f"  {desc}")
    print(f"{'=' * 140}")
    
    for f_name, f_func in filters.items():
        sub = f_func(base)
        n = len(sub)
        if n < 30:
            continue
        
        winners = sub[sub['max_levels_continued'] >= 1]
        losers = sub[sub['max_levels_continued'] == 0]
        
        if len(winners) < 10 or len(losers) < 10:
            continue
        
        print(f"\n  {f_name} (n={n:,})")
        
        # MAE fuer Gewinner
        w_mae = winners['max_adverse_adr']
        l_mae = losers['max_adverse_adr']
        
        print(f"    MAE Vergleich (in ADR):")
        print(f"    {'':20s} {'Gewinner':>10s} (n={len(winners)})  {'Verlierer':>10s} (n={len(losers)})")
        print(f"    {'-'*20} {'-'*18}  {'-'*18}")
        for p in [25, 50, 75, 90, 95]:
            w_val = w_mae.quantile(p/100)
            l_val = l_mae.quantile(p/100)
            print(f"    P{p:>2d} MAE:             {w_val:>8.3f} ADR       {l_val:>8.3f} ADR")
        
        # MAE Zeitverlauf (nur Gewinner): Wie entwickelt sich MAE ueber die Zeit?
        print(f"\n    MAE-Entwicklung ueber Zeit (Median, nur Gewinner):")
        print(f"    {'Zeit':>8s}  {'Med MAE':>8s}  {'P75 MAE':>8s}  {'P90 MAE':>8s}")
        print(f"    {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
        for minutes, col in [(5, 'mae_5m'), (10, 'mae_10m'), (15, 'mae_15m'), 
                              (30, 'mae_30m'), (60, 'mae_60m'), (120, 'mae_120m')]:
            vals = winners[col].dropna()
            if len(vals) < 10:
                continue
            print(f"    {minutes:>5d}min  {vals.median():>8.3f}  {vals.quantile(0.75):>8.3f}  {vals.quantile(0.90):>8.3f}")


# ================================================================
# ANALYSE 3: MFE-ENTWICKLUNG (Favorable Excursion ueber Zeit)
#   Wie viel Gewinn ist zu welchem Zeitpunkt bereits aufgelaufen?
# ================================================================
print()
print("=" * 140)
print("ANALYSE 3: FAVORABLE EXCURSION UEBER ZEIT")
print("  Wie viel Gewinn (in ADR) ist nach X Minuten aufgelaufen?")
print("  Getrennt fuer Gewinner und alle Swings.")
print("=" * 140)

for s_name, gap_dir, level, swing_dir, desc in setups:
    base = df_all[(df_all['gap_direction'] == gap_dir) & 
                  (df_all['crossed_level'] == level) & 
                  (df_all['direction'] == swing_dir)]
    if len(base) < 100:
        continue
    
    print(f"\n{'=' * 140}")
    print(f"  {desc}")
    print(f"{'=' * 140}")
    
    for f_name, f_func in filters.items():
        sub = f_func(base)
        n = len(sub)
        if n < 30:
            continue
        
        winners = sub[sub['max_levels_continued'] >= 1]
        if len(winners) < 10:
            continue
        
        print(f"\n  {f_name} (n={n:,}, Winners={len(winners)})")
        
        print(f"    MFE-Entwicklung (Median in ADR):")
        print(f"    {'Zeit':>8s}  {'Alle Med':>9s}  {'Win Med':>8s}  {'Win P25':>8s}  {'Win P75':>8s}")
        print(f"    {'-'*8}  {'-'*9}  {'-'*8}  {'-'*8}  {'-'*8}")
        
        for minutes, col in [(5, 'mfe_5m'), (10, 'mfe_10m'), (15, 'mfe_15m'),
                              (30, 'mfe_30m'), (60, 'mfe_60m'), (120, 'mfe_120m'), (240, 'mfe_240m')]:
            all_vals = sub[col].dropna()
            win_vals = winners[col].dropna()
            if len(all_vals) < 10:
                continue
            a_med = all_vals.median()
            w_med = win_vals.median() if len(win_vals) >= 10 else np.nan
            w_25 = win_vals.quantile(0.25) if len(win_vals) >= 10 else np.nan
            w_75 = win_vals.quantile(0.75) if len(win_vals) >= 10 else np.nan
            
            w_med_s = f"{w_med:.3f}" if not np.isnan(w_med) else "n/a"
            w_25_s = f"{w_25:.3f}" if not np.isnan(w_25) else "n/a"
            w_75_s = f"{w_75:.3f}" if not np.isnan(w_75) else "n/a"
            
            print(f"    {minutes:>5d}min  {a_med:>9.3f}  {w_med_s:>8s}  {w_25_s:>8s}  {w_75_s:>8s}")


# ================================================================
# ANALYSE 4: PFAD-EFFIZIENZ
#   Geht der Move direkt oder eiert er herum?
# ================================================================
print()
print("=" * 140)
print("ANALYSE 4: PFAD-EFFIZIENZ")
print("  Effizienz = Netto-Move / Gesamte zurueckgelegte Strecke")
print("  1.0 = Perfekt geradlinig, 0.0 = Totales Hin-und-Her")
print("=" * 140)

for s_name, gap_dir, level, swing_dir, desc in setups:
    base = df_all[(df_all['gap_direction'] == gap_dir) & 
                  (df_all['crossed_level'] == level) & 
                  (df_all['direction'] == swing_dir)]
    if len(base) < 100:
        continue
    
    print(f"\n  {desc}")
    
    for f_name, f_func in filters.items():
        sub = f_func(base)
        n = len(sub)
        if n < 30:
            continue
        
        winners = sub[sub['max_levels_continued'] >= 1]
        losers = sub[sub['max_levels_continued'] == 0]
        
        if len(winners) < 10:
            continue
        
        w_eff = winners['path_efficiency']
        l_eff = losers['path_efficiency'] if len(losers) >= 10 else pd.Series()
        
        l_med = f"{l_eff.median():.3f}" if len(l_eff) >= 10 else "n/a"
        
        print(f"    {f_name:25s}  Gewinner Median={w_eff.median():.3f} P25={w_eff.quantile(0.25):.3f} P75={w_eff.quantile(0.75):.3f}  "
              f"Verlierer Median={l_med}")


# ================================================================
# ANALYSE 5: ZEITFENSTER-BASIERTE WIN RATES
#   "Erreicht +1s innerhalb von X Minuten" statt "irgendwann"
# ================================================================
print()
print("=" * 140)
print("ANALYSE 5: ZEITFENSTER-BASIERTE WIN RATES")
print("  Wie viel % erreichen Level 1 INNERHALB eines bestimmten Zeitfensters?")
print("  Das ist die Trading-relevanteste Metrik.")
print("=" * 140)

for s_name, gap_dir, level, swing_dir, desc in setups:
    base = df_all[(df_all['gap_direction'] == gap_dir) & 
                  (df_all['crossed_level'] == level) & 
                  (df_all['direction'] == swing_dir)]
    if len(base) < 100:
        continue
    
    print(f"\n{'=' * 140}")
    print(f"  {desc}")
    print(f"{'=' * 140}")
    
    print(f"\n  {'Filter':25s} {'N':>7s}", end='')
    for tw in [15, 30, 60, 120, 240]:
        print(f"  {'<'+str(tw)+'m':>8s}", end='')
    print(f"  {'Gesamt':>8s}")
    
    print(f"  {'-'*25} {'-'*7}", end='')
    for _ in [15, 30, 60, 120, 240]:
        print(f"  {'-'*8}", end='')
    print(f"  {'-'*8}")
    
    for f_name, f_func in filters.items():
        sub = f_func(base)
        n = len(sub)
        if n < 30:
            continue
        
        ttl1 = sub['bars_to_lvl1']  # NaN = Level nie erreicht
        total_winners = sub['max_levels_continued'] >= 1
        total_win_pct = total_winners.sum() / n * 100
        
        print(f"  {f_name:25s} {n:7d}", end='')
        for tw in [15, 30, 60, 120, 240]:
            # Level 1 innerhalb von tw Minuten erreicht
            reached_in_time = (ttl1 <= tw).sum()
            pct = reached_in_time / n * 100
            print(f"  {pct:>7.1f}%", end='')
        print(f"  {total_win_pct:>7.1f}%")
    
    # Dasselbe fuer Level 2
    print(f"\n  Level 2 (2 Levels weiter):")
    print(f"  {'Filter':25s} {'N':>7s}", end='')
    for tw in [30, 60, 120, 240]:
        print(f"  {'<'+str(tw)+'m':>8s}", end='')
    print(f"  {'Gesamt':>8s}")
    
    print(f"  {'-'*25} {'-'*7}", end='')
    for _ in [30, 60, 120, 240]:
        print(f"  {'-'*8}", end='')
    print(f"  {'-'*8}")
    
    for f_name, f_func in filters.items():
        sub = f_func(base)
        n = len(sub)
        if n < 30:
            continue
        
        ttl2 = sub['bars_to_lvl2']
        total_win2 = (sub['max_levels_continued'] >= 2).sum() / n * 100
        
        print(f"  {f_name:25s} {n:7d}", end='')
        for tw in [30, 60, 120, 240]:
            reached = (ttl2 <= tw).sum()
            pct = reached / n * 100
            print(f"  {pct:>7.1f}%", end='')
        print(f"  {total_win2:>7.1f}%")


# ================================================================
# ANALYSE 6: GEWINNER-PROFIL
#   Wie sieht der "typische Gewinner" aus?
# ================================================================
print()
print("=" * 140)
print("ANALYSE 6: PROFIL EINES TYPISCHEN GEWINNERS")
print("  Fuer die Top-Setups: Median-Werte der Trades die Level 1+ erreichen")
print("=" * 140)

top_setups = [
    ('GapUp +2s Fade Morgen',  'up',   '+2s', 'down', 
     lambda df: df[(df['adr_pct'] >= 3) & (df['time_bucket'].isin(['1_open30', '2_30_60']))]),
    ('GapDn -2s Fade Morgen',  'down', '-2s', 'up',
     lambda df: df[(df['adr_pct'] >= 3) & (df['time_bucket'].isin(['1_open30', '2_30_60']))]),
    ('GapUp +1s Fade Morgen',  'up',   '+1s', 'down',
     lambda df: df[(df['adr_pct'] >= 3) & (df['time_bucket'].isin(['1_open30', '2_30_60']))]),
    ('GapDn -1s Fade Morgen',  'down', '-1s', 'up',
     lambda df: df[(df['adr_pct'] >= 3) & (df['time_bucket'].isin(['1_open30', '2_30_60']))]),
    ('GapUp VWAP Brk Morgen',  'up',   'VWAP', 'down',
     lambda df: df[(df['adr_pct'] >= 3) & (df['time_bucket'].isin(['1_open30', '2_30_60']))]),
    ('GapDn VWAP Brk Morgen',  'down', 'VWAP', 'up',
     lambda df: df[(df['adr_pct'] >= 3) & (df['time_bucket'].isin(['1_open30', '2_30_60']))]),
]

for name, gap_dir, level, swing_dir, f_func in top_setups:
    base = df_all[(df_all['gap_direction'] == gap_dir) & 
                  (df_all['crossed_level'] == level) & 
                  (df_all['direction'] == swing_dir)]
    sub = f_func(base)
    
    winners = sub[sub['max_levels_continued'] >= 1]
    losers = sub[sub['max_levels_continued'] == 0]
    
    if len(winners) < 20:
        continue
    
    win_pct = len(winners) / len(sub) * 100
    
    print(f"\n  {name}")
    print(f"  {'='*60}")
    print(f"  Total: {len(sub):,}  Gewinner: {len(winners)} ({win_pct:.1f}%)  Verlierer: {len(losers)}")
    print()
    
    # Gewinner-Profil
    print(f"  GEWINNER (Level 1+ erreicht):")
    ttl1 = winners['bars_to_lvl1'].dropna()
    if len(ttl1) >= 10:
        print(f"    Zeit bis Level 1:  Median={ttl1.median():.0f}min  P25={ttl1.quantile(0.25):.0f}min  P75={ttl1.quantile(0.75):.0f}min")
    print(f"    Max Move (ADR):    Median={winners['max_move_adr'].median():.3f}")
    print(f"    Max Adverse (ADR): Median={winners['max_adverse_adr'].median():.3f}  P75={winners['max_adverse_adr'].quantile(0.75):.3f}  P90={winners['max_adverse_adr'].quantile(0.90):.3f}")
    print(f"    Pfad-Effizienz:    Median={winners['path_efficiency'].median():.3f}")
    
    # Wie viele erreichen Level 2?
    w2 = winners[winners['max_levels_continued'] >= 2]
    if len(w2) >= 10:
        cond = len(w2) / len(winners) * 100
        ttl2 = w2['bars_to_lvl2'].dropna()
        print(f"    Weiter zu Level 2: {len(w2)}/{len(winners)} = {cond:.0f}%")
        if len(ttl2) >= 10:
            print(f"      Zeit bis Level 2: Median={ttl2.median():.0f}min")
    
    # Verlierer-Profil
    if len(losers) >= 10:
        print(f"\n  VERLIERER (Level 1 nie erreicht):")
        print(f"    Max Move (ADR):    Median={losers['max_move_adr'].median():.3f}  (so weit kamen sie IN die richtige Richtung)")
        print(f"    Max Adverse (ADR): Median={losers['max_adverse_adr'].median():.3f}  P75={losers['max_adverse_adr'].quantile(0.75):.3f}")
        print(f"    Pfad-Effizienz:    Median={losers['path_efficiency'].median():.3f}")
        print(f"    Dauer bis Ende:    Median={losers['bars_to_end'].median():.0f}min")


print()
print("=" * 140)
print("FERTIG.")
print("=" * 140)
