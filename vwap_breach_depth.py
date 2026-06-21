###############################################################################
# VWAP BREACH DEPTH ANALYSE — Szenarien B und D verfeinert
#
# Frage: Wie tief muss der VWAP-Bruch sein bevor die Recovery/Failure
# ein gutes Signal wird?
#
# Szenario B: Gap Up → Aktie fällt UNTER VWAP → kommt zurück ÜBER VWAP
#   → Wie tief war sie unter VWAP (in StdDevs)?
#   → Win = erreicht +1 StdDev danach
#
# Szenario D: Gap Down → Aktie steigt ÜBER VWAP → fällt zurück UNTER VWAP
#   → Wie hoch war sie über VWAP (in StdDevs)?
#   → Win = erreicht -1 StdDev danach
#
# Ausführen: cd gapper-analysis
#            .\gapper_env\Scripts\python.exe vwap_breach_depth.py
###############################################################################

import pandas as pd
import numpy as np
import glob
from tqdm import tqdm
from collections import defaultdict

# ============================================================
# CONFIG
# ============================================================
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[0]

VWAP_DIR = str(PROJECT_ROOT / 'data' / 'vwap')
METADATA_PATH = str(PROJECT_ROOT / 'data' / 'metadata' / 'metadata_master.parquet')

MIN_BARS_AFTER_OPEN = 20
MIN_BARS_REMAINING = 10

# ============================================================
# METADATA LADEN
# ============================================================
print("Lade Metadata...")
meta = pd.read_parquet(METADATA_PATH)
meta['key'] = meta['ticker'] + '_' + meta['date']

meta['gap_adr_bucket'] = pd.cut(
    meta['gap_size_in_adr'].abs(),
    bins=[0, 0.75, 1.0, 1.5, 2.0, 100],
    labels=['<0.75_ADR', '0.75-1_ADR', '1-1.5_ADR', '1.5-2_ADR', '>2_ADR']
)
meta['rvol_bucket'] = pd.cut(
    meta['rvol_at_time_30min'],
    bins=[0, 1.0, 1.5, 2.5, 5.0, 1000],
    labels=['RVOL<1', 'RVOL_1-1.5', 'RVOL_1.5-2.5', 'RVOL_2.5-5', 'RVOL>5']
)

meta_lookup = meta.set_index('key')[[
    'gap_adr_bucket', 'rvol_bucket', 'gap_direction',
    'gap_size_in_adr', 'adr_10', 'rvol_at_time_30min'
]].to_dict('index')

print(f"Metadata: {len(meta)} Gapper")

# ============================================================
# EVENT DETECTION — Mit Breach Depth
# ============================================================

def analyze_breach_events(df, gap_direction, adr_10):
    """
    Erkennt NUR Szenarien B und D, mit zusätzlicher Info:
    - max_breach_z: Maximaler Z-Score (StdDevs) auf der "falschen" Seite
    - min_breach_z: Minimaler Z-Score auf der "falschen" Seite
    - time_below_vwap: Wie viele Minuten unter/über VWAP bevor Recovery
    """
    events = []
    
    df = df.dropna(subset=['vwap', 'std_dev']).copy()
    df = df[df['std_dev'] > 0].copy()
    
    if len(df) < MIN_BARS_AFTER_OPEN + MIN_BARS_REMAINING:
        return events
    
    closes = df['close'].values
    vwaps = df['vwap'].values
    upper1 = df['upper_1std'].values
    lower1 = df['lower_1std'].values
    upper2 = df['upper_2std'].values
    lower2 = df['lower_2std'].values
    z_scores = df['z_score'].values
    times = df['time_et'].values
    std_devs = df['std_dev'].values
    
    n = len(closes)
    open_price = closes[0]
    
    # Running high/low für %ADR
    running_high = np.maximum.accumulate(closes)
    running_low = np.minimum.accumulate(closes)
    if adr_10 > 0:
        pct_adr = (running_high - running_low) / adr_10
    else:
        pct_adr = np.zeros(n)
    
    # ──────────────────────────────────────────────────────────
    # Track Zustand: Wann ist Preis auf der falschen Seite?
    # ──────────────────────────────────────────────────────────
    
    if gap_direction == 'up':
        # Szenario B: Gap Up, Preis fällt unter VWAP, kommt zurück
        # "Falsche Seite" = unter VWAP
        # Wir tracken: wann geht er unter VWAP, wie tief, wann kommt er zurück
        
        below_vwap_since = None  # Index wo er unter VWAP ging
        max_depth_z = 0  # Tiefster Z-Score unter VWAP (negativster)
        
        for i in range(MIN_BARS_AFTER_OPEN, n - MIN_BARS_REMAINING):
            curr_z = z_scores[i]
            
            # Prüfe auf NaN
            if np.isnan(curr_z):
                continue
            
            # Unter VWAP gegangen?
            if closes[i] < vwaps[i]:
                if below_vwap_since is None:
                    below_vwap_since = i
                    max_depth_z = curr_z
                else:
                    # Tracking: tiefster Punkt
                    if curr_z < max_depth_z:
                        max_depth_z = curr_z
            
            # Zurück über VWAP! → Event!
            elif below_vwap_since is not None and closes[i] >= vwaps[i]:
                time_below = i - below_vwap_since
                breach_depth = abs(max_depth_z)  # Positiv machen (0.5 = halbe StdDev unter VWAP)
                
                # Outcome tracken
                outcome, bars, max_fav, max_adv = _track_outcome(
                    closes, vwaps, upper1, lower1, i, n,
                    favorable_direction='up', adr_10=adr_10
                )
                
                events.append({
                    'scenario': 'B_gapup_recovery',
                    'time': times[i],
                    'bar_index': i,
                    'breach_depth_z': round(breach_depth, 2),
                    'time_below_vwap': time_below,
                    'pct_adr_used': round(pct_adr[i], 2),
                    'outcome': outcome,
                    'bars_to_outcome': bars,
                    'max_favorable_adr': round(max_fav, 2),
                    'max_adverse_adr': round(max_adv, 2),
                })
                
                # Reset
                below_vwap_since = None
                max_depth_z = 0
    
    elif gap_direction == 'down':
        # Szenario D: Gap Down, Preis steigt über VWAP, fällt zurück
        # "Falsche Seite" = über VWAP
        
        above_vwap_since = None
        max_height_z = 0  # Höchster Z-Score über VWAP
        
        for i in range(MIN_BARS_AFTER_OPEN, n - MIN_BARS_REMAINING):
            curr_z = z_scores[i]
            
            if np.isnan(curr_z):
                continue
            
            # Über VWAP gegangen?
            if closes[i] > vwaps[i]:
                if above_vwap_since is None:
                    above_vwap_since = i
                    max_height_z = curr_z
                else:
                    if curr_z > max_height_z:
                        max_height_z = curr_z
            
            # Zurück unter VWAP! → Event!
            elif above_vwap_since is not None and closes[i] <= vwaps[i]:
                time_above = i - above_vwap_since
                breach_depth = abs(max_height_z)
                
                outcome, bars, max_fav, max_adv = _track_outcome(
                    closes, vwaps, upper1, lower1, i, n,
                    favorable_direction='down', adr_10=adr_10
                )
                
                events.append({
                    'scenario': 'D_gapdown_failure',
                    'time': times[i],
                    'bar_index': i,
                    'breach_depth_z': round(breach_depth, 2),
                    'time_below_vwap': time_above,
                    'pct_adr_used': round(pct_adr[i], 2),
                    'outcome': outcome,
                    'bars_to_outcome': bars,
                    'max_favorable_adr': round(max_fav, 2),
                    'max_adverse_adr': round(max_adv, 2),
                })
                
                above_vwap_since = None
                max_height_z = 0
    
    return events


def _track_outcome(closes, vwaps, upper1, lower1, start_idx, n,
                   favorable_direction, adr_10):
    entry_price = closes[start_idx]
    max_favorable = 0.0
    max_adverse = 0.0
    
    for j in range(start_idx + 1, n):
        price = closes[j]
        
        if adr_10 > 0:
            if favorable_direction == 'up':
                fav = (price - entry_price) / adr_10
                adv = (entry_price - price) / adr_10
            else:
                fav = (entry_price - price) / adr_10
                adv = (price - entry_price) / adr_10
            max_favorable = max(max_favorable, fav)
            max_adverse = max(max_adverse, adv)
        
        if price >= upper1[j]:
            return 'reached_+1std', j - start_idx, max_favorable, max_adverse
        elif price <= lower1[j]:
            return 'reached_-1std', j - start_idx, max_favorable, max_adverse
    
    return 'stayed_at_vwap', n - start_idx, max_favorable, max_adverse


# ============================================================
# ALLE DATEIEN VERARBEITEN
# ============================================================
print("\nVerarbeite VWAP-Dateien...")
files = glob.glob(f'{VWAP_DIR}/**/*.parquet', recursive=True)
print(f"{len(files)} Dateien gefunden")

all_events = []

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
        gap_dir = info['gap_direction']
        
        events = analyze_breach_events(df, gap_dir, adr_10)
        
        for ev in events:
            ev['ticker'] = ticker
            ev['date'] = date
            ev['gap_direction'] = gap_dir
            ev['gap_adr_bucket'] = info['gap_adr_bucket']
            ev['rvol_bucket'] = info['rvol_bucket']
            ev['gap_size_adr'] = info['gap_size_in_adr']
            ev['rvol'] = info['rvol_at_time_30min']
        
        all_events.extend(events)
    except Exception:
        continue

events_df = pd.DataFrame(all_events)
print(f"\n{len(events_df):,} Events erkannt")

# Buckets
def time_bucket(t):
    if t <= '10:00':
        return '1_open_30min'
    elif t <= '10:30':
        return '2_30_60min'
    elif t <= '11:30':
        return '3_1_2h'
    elif t <= '13:00':
        return '4_mittag'
    else:
        return '5_nachmittag'

events_df['time_bucket'] = events_df['time'].apply(time_bucket)

events_df['adr_used_bucket'] = pd.cut(
    events_df['pct_adr_used'],
    bins=[0, 0.5, 0.75, 1.0, 1.5, 100],
    labels=['<50%', '50-75%', '75-100%', '100-150%', '>150%']
)

events_df['breach_depth_bucket'] = pd.cut(
    events_df['breach_depth_z'],
    bins=[0, 0.25, 0.5, 0.75, 1.0, 1.5, 100],
    labels=['<0.25σ', '0.25-0.5σ', '0.5-0.75σ', '0.75-1.0σ', '1.0-1.5σ', '>1.5σ']
)

events_df['time_on_wrong_side'] = pd.cut(
    events_df['time_below_vwap'],
    bins=[0, 3, 10, 30, 60, 999],
    labels=['1-3min', '4-10min', '11-30min', '31-60min', '>60min']
)

# Zähle Events pro Tag für "nur erstes Event" Filter
events_df['day_key'] = events_df['ticker'] + '_' + events_df['date']
events_df['event_num'] = events_df.groupby(['day_key', 'scenario']).cumcount() + 1

# ============================================================
# HELPER
# ============================================================
def print_outcomes(sub, good_outcome, indent=4, min_n=30):
    n = len(sub)
    if n < min_n:
        print(f"{' ' * indent}(n={n}, zu wenig Daten)")
        return n, 0
    
    prefix = " " * indent
    good_n = (sub['outcome'] == good_outcome).sum()
    good_pct = good_n / n * 100
    
    outcomes = sub['outcome'].value_counts()
    for outcome, count in outcomes.items():
        pct = count / n * 100
        bar = '█' * int(pct / 2)
        median_bars = sub[sub['outcome'] == outcome]['bars_to_outcome'].median()
        marker = " ◄── WIN" if outcome == good_outcome else ""
        print(f"{prefix}{outcome:22s}  {count:5d} ({pct:5.1f}%)  "
              f"~{median_bars:3.0f}min  {bar}{marker}")
    
    return n, good_pct


# ################################################################
#                     ERGEBNISSE
# ################################################################

for scenario, good_outcome, title in [
    ('B_gapup_recovery', 'reached_+1std', 
     'SZENARIO B: Gap Up → Unter VWAP gefallen → Recovery über VWAP'),
    ('D_gapdown_failure', 'reached_-1std',
     'SZENARIO D: Gap Down → Über VWAP gestiegen → Zurück unter VWAP'),
]:
    sub_all = events_df[events_df['scenario'] == scenario]
    sub_first = sub_all[sub_all['event_num'] == 1]
    
    # ============================================================
    # ANALYSE 1: BREACH DEPTH
    # ============================================================
    print()
    print("=" * 80)
    print(f"  {title}")
    print(f"  Win = {good_outcome}")
    print("=" * 80)
    
    print(f"\n  GESAMT: {len(sub_all):,} Events, davon {len(sub_first):,} erste pro Tag")
    
    print(f"\n{'─' * 80}")
    print(f"  1. WIE TIEF MUSS DER BREACH SEIN? (Alle Events)")
    print(f"{'─' * 80}")
    
    for depth_b in ['<0.25σ', '0.25-0.5σ', '0.5-0.75σ', '0.75-1.0σ', '1.0-1.5σ', '>1.5σ']:
        s = sub_all[sub_all['breach_depth_bucket'] == depth_b]
        n = len(s)
        if n < 20:
            continue
        good_n = (s['outcome'] == good_outcome).sum()
        good_pct = good_n / n * 100
        bar = '▓' * int(good_pct / 2) + '░' * (25 - int(good_pct / 2))
        med_fav = s['max_favorable_adr'].median()
        med_adv = s['max_adverse_adr'].median()
        print(f"\n  Breach Depth: {depth_b} (n={n:,})")
        print(f"    Win: {good_pct:5.1f}%  {bar}  MedFav:{med_fav:+.2f} MedAdv:{med_adv:+.2f}")
        print_outcomes(s, good_outcome, min_n=10)
    
    # ============================================================
    # ANALYSE 2: BREACH DEPTH — Nur ERSTE Events
    # ============================================================
    print(f"\n{'─' * 80}")
    print(f"  2. WIE TIEF MUSS DER BREACH SEIN? (Nur ERSTE Events pro Tag)")
    print(f"{'─' * 80}")
    
    for depth_b in ['<0.25σ', '0.25-0.5σ', '0.5-0.75σ', '0.75-1.0σ', '1.0-1.5σ', '>1.5σ']:
        s = sub_first[sub_first['breach_depth_bucket'] == depth_b]
        n = len(s)
        if n < 20:
            continue
        good_n = (s['outcome'] == good_outcome).sum()
        good_pct = good_n / n * 100
        bar = '▓' * int(good_pct / 2) + '░' * (25 - int(good_pct / 2))
        print(f"\n  Breach Depth: {depth_b} (n={n:,})")
        print(f"    Win: {good_pct:5.1f}%  {bar}")
        print_outcomes(s, good_outcome, min_n=10)
    
    # ============================================================
    # ANALYSE 3: WIE LANGE AUF DER FALSCHEN SEITE?
    # ============================================================
    print(f"\n{'─' * 80}")
    print(f"  3. WIE LANGE WAR DER PREIS AUF DER FALSCHEN SEITE?")
    print(f"{'─' * 80}")
    
    for time_b in ['1-3min', '4-10min', '11-30min', '31-60min', '>60min']:
        s = sub_all[sub_all['time_on_wrong_side'] == time_b]
        n = len(s)
        if n < 20:
            continue
        good_n = (s['outcome'] == good_outcome).sum()
        good_pct = good_n / n * 100
        bar = '▓' * int(good_pct / 2) + '░' * (25 - int(good_pct / 2))
        print(f"\n  Zeit auf falscher Seite: {time_b} (n={n:,})")
        print(f"    Win: {good_pct:5.1f}%  {bar}")
        print_outcomes(s, good_outcome, min_n=10)
    
    # ============================================================
    # ANALYSE 4: BREACH DEPTH × TAGESZEIT
    # ============================================================
    print(f"\n{'─' * 80}")
    print(f"  4. BREACH DEPTH × TAGESZEIT")
    print(f"{'─' * 80}")
    
    print(f"\n  {'Breach Depth':15s} {'Tageszeit':15s} {'N':>5s}  {'Win%':>6s}  {'MedFav':>7s}  {'MedAdv':>7s}")
    print(f"  {'─'*15} {'─'*15} {'─'*5}  {'─'*6}  {'─'*7}  {'─'*7}")
    
    results = []
    for depth_b in ['<0.25σ', '0.25-0.5σ', '0.5-0.75σ', '0.75-1.0σ', '1.0-1.5σ', '>1.5σ']:
        for tb in ['1_open_30min', '2_30_60min', '3_1_2h', '4_mittag', '5_nachmittag']:
            s = sub_all[(sub_all['breach_depth_bucket'] == depth_b) & 
                        (sub_all['time_bucket'] == tb)]
            n = len(s)
            if n < 30:
                continue
            good_n = (s['outcome'] == good_outcome).sum()
            good_pct = good_n / n * 100
            med_fav = s['max_favorable_adr'].median()
            med_adv = s['max_adverse_adr'].median()
            results.append((depth_b, tb, n, good_pct, med_fav, med_adv))
    
    results.sort(key=lambda x: x[3], reverse=True)
    for depth_b, tb, n, good_pct, med_fav, med_adv in results:
        bar = '▓' * int(good_pct / 4)
        print(f"  {depth_b:15s} {tb:15s} {n:5d}  {good_pct:5.1f}%  {med_fav:+6.2f}  {med_adv:+6.2f}  {bar}")
    
    # ============================================================
    # ANALYSE 5: BREACH DEPTH × %ADR VERBRAUCHT
    # ============================================================
    print(f"\n{'─' * 80}")
    print(f"  5. BREACH DEPTH × %ADR VERBRAUCHT")
    print(f"{'─' * 80}")
    
    print(f"\n  {'Breach Depth':15s} {'%ADR':15s} {'N':>5s}  {'Win%':>6s}  {'MedFav':>7s}  {'MedAdv':>7s}")
    print(f"  {'─'*15} {'─'*15} {'─'*5}  {'─'*6}  {'─'*7}  {'─'*7}")
    
    results = []
    for depth_b in ['<0.25σ', '0.25-0.5σ', '0.5-0.75σ', '0.75-1.0σ', '1.0-1.5σ', '>1.5σ']:
        for adr_b in ['<50%', '50-75%', '75-100%', '100-150%', '>150%']:
            s = sub_all[(sub_all['breach_depth_bucket'] == depth_b) & 
                        (sub_all['adr_used_bucket'] == adr_b)]
            n = len(s)
            if n < 30:
                continue
            good_n = (s['outcome'] == good_outcome).sum()
            good_pct = good_n / n * 100
            med_fav = s['max_favorable_adr'].median()
            med_adv = s['max_adverse_adr'].median()
            results.append((depth_b, adr_b, n, good_pct, med_fav, med_adv))
    
    results.sort(key=lambda x: x[3], reverse=True)
    for depth_b, adr_b, n, good_pct, med_fav, med_adv in results:
        bar = '▓' * int(good_pct / 4)
        print(f"  {depth_b:15s} {adr_b:15s} {n:5d}  {good_pct:5.1f}%  {med_fav:+6.2f}  {med_adv:+6.02f}  {bar}")
    
    # ============================================================
    # ANALYSE 6: BREACH DEPTH × GAP-ADR
    # ============================================================
    print(f"\n{'─' * 80}")
    print(f"  6. BREACH DEPTH × GAP-ADR")
    print(f"{'─' * 80}")
    
    print(f"\n  {'Breach Depth':15s} {'Gap-ADR':15s} {'N':>5s}  {'Win%':>6s}  {'MedFav':>7s}  {'MedAdv':>7s}")
    print(f"  {'─'*15} {'─'*15} {'─'*5}  {'─'*6}  {'─'*7}  {'─'*7}")
    
    results = []
    for depth_b in ['<0.25σ', '0.25-0.5σ', '0.5-0.75σ', '0.75-1.0σ', '1.0-1.5σ', '>1.5σ']:
        for gap_b in ['<0.75_ADR', '0.75-1_ADR', '1-1.5_ADR', '1.5-2_ADR', '>2_ADR']:
            s = sub_all[(sub_all['breach_depth_bucket'] == depth_b) & 
                        (sub_all['gap_adr_bucket'] == gap_b)]
            n = len(s)
            if n < 30:
                continue
            good_n = (s['outcome'] == good_outcome).sum()
            good_pct = good_n / n * 100
            med_fav = s['max_favorable_adr'].median()
            med_adv = s['max_adverse_adr'].median()
            results.append((depth_b, gap_b, n, good_pct, med_fav, med_adv))
    
    results.sort(key=lambda x: x[3], reverse=True)
    for depth_b, gap_b, n, good_pct, med_fav, med_adv in results:
        bar = '▓' * int(good_pct / 4)
        print(f"  {depth_b:15s} {gap_b:15s} {n:5d}  {good_pct:5.1f}%  {med_fav:+6.02f}  {med_adv:+6.02f}  {bar}")
    
    # ============================================================
    # ANALYSE 7: BREACH DEPTH × RVOL
    # ============================================================
    print(f"\n{'─' * 80}")
    print(f"  7. BREACH DEPTH × RVOL")
    print(f"{'─' * 80}")
    
    print(f"\n  {'Breach Depth':15s} {'RVOL':15s} {'N':>5s}  {'Win%':>6s}  {'MedFav':>7s}  {'MedAdv':>7s}")
    print(f"  {'─'*15} {'─'*15} {'─'*5}  {'─'*6}  {'─'*7}  {'─'*7}")
    
    results = []
    for depth_b in ['<0.25σ', '0.25-0.5σ', '0.5-0.75σ', '0.75-1.0σ', '1.0-1.5σ', '>1.5σ']:
        for rvol_b in ['RVOL<1', 'RVOL_1-1.5', 'RVOL_1.5-2.5', 'RVOL_2.5-5', 'RVOL>5']:
            s = sub_all[(sub_all['breach_depth_bucket'] == depth_b) & 
                        (sub_all['rvol_bucket'] == rvol_b)]
            n = len(s)
            if n < 30:
                continue
            good_n = (s['outcome'] == good_outcome).sum()
            good_pct = good_n / n * 100
            med_fav = s['max_favorable_adr'].median()
            med_adv = s['max_adverse_adr'].median()
            results.append((depth_b, rvol_b, n, good_pct, med_fav, med_adv))
    
    results.sort(key=lambda x: x[3], reverse=True)
    for depth_b, rvol_b, n, good_pct, med_fav, med_adv in results:
        bar = '▓' * int(good_pct / 4)
        print(f"  {depth_b:15s} {rvol_b:15s} {n:5d}  {good_pct:5.1f}%  {med_fav:+6.02f}  {med_adv:+6.02f}  {bar}")
    
    # ============================================================
    # ANALYSE 8: BESTE TRIPLE-KOMBINATION
    # ============================================================
    print(f"\n{'─' * 80}")
    print(f"  8. BESTE KOMBINATION: Breach Depth × Tageszeit × %ADR")
    print(f"     (n >= 30)")
    print(f"{'─' * 80}")
    
    print(f"\n  {'Breach':10s} {'Zeit':15s} {'%ADR':10s} {'N':>5s}  {'Win%':>6s}  {'MedFav':>7s}  {'MedAdv':>7s}")
    print(f"  {'─'*10} {'─'*15} {'─'*10} {'─'*5}  {'─'*6}  {'─'*7}  {'─'*7}")
    
    results = []
    for depth_b in ['<0.25σ', '0.25-0.5σ', '0.5-0.75σ', '0.75-1.0σ', '1.0-1.5σ', '>1.5σ']:
        for tb in ['1_open_30min', '2_30_60min', '3_1_2h', '4_mittag', '5_nachmittag']:
            for adr_b in ['<50%', '50-75%', '75-100%', '100-150%', '>150%']:
                s = sub_all[(sub_all['breach_depth_bucket'] == depth_b) & 
                            (sub_all['time_bucket'] == tb) &
                            (sub_all['adr_used_bucket'] == adr_b)]
                n = len(s)
                if n < 30:
                    continue
                good_n = (s['outcome'] == good_outcome).sum()
                good_pct = good_n / n * 100
                med_fav = s['max_favorable_adr'].median()
                med_adv = s['max_adverse_adr'].median()
                results.append((depth_b, tb, adr_b, n, good_pct, med_fav, med_adv))
    
    results.sort(key=lambda x: x[4], reverse=True)
    for depth_b, tb, adr_b, n, good_pct, med_fav, med_adv in results[:25]:
        bar = '▓' * int(good_pct / 4)
        print(f"  {depth_b:10s} {tb:15s} {adr_b:10s} {n:5d}  {good_pct:5.1f}%  {med_fav:+6.02f}  {med_adv:+6.02f}  {bar}")
    
    print(f"\n  ... und die schlechtesten:")
    for depth_b, tb, adr_b, n, good_pct, med_fav, med_adv in results[-10:]:
        bar = '▓' * int(good_pct / 4)
        print(f"  {depth_b:10s} {tb:15s} {adr_b:10s} {n:5d}  {good_pct:5.1f}%  {med_fav:+6.02f}  {med_adv:+6.02f}  {bar}")

    # ============================================================
    # ANALYSE 9: TAKE PROFIT nach Breach Depth
    # ============================================================
    print(f"\n{'─' * 80}")
    print(f"  9. TAKE PROFIT: Max Favorable Excursion nach Breach Depth")
    print(f"     (Nur Winners)")
    print(f"{'─' * 80}")
    
    for depth_b in ['<0.25σ', '0.25-0.5σ', '0.5-0.75σ', '0.75-1.0σ', '1.0-1.5σ', '>1.5σ']:
        winners = sub_all[(sub_all['breach_depth_bucket'] == depth_b) & 
                          (sub_all['outcome'] == good_outcome)]
        n = len(winners)
        if n < 20:
            continue
        
        print(f"\n  Breach: {depth_b} (n={n} winners)")
        print(f"    Max Favorable (ADR):  P25={winners['max_favorable_adr'].quantile(0.25):+.2f}  "
              f"P50={winners['max_favorable_adr'].quantile(0.50):+.2f}  "
              f"P75={winners['max_favorable_adr'].quantile(0.75):+.2f}  "
              f"P90={winners['max_favorable_adr'].quantile(0.90):+.2f}")
        print(f"    Time to Win (min):    P25={winners['bars_to_outcome'].quantile(0.25):.0f}  "
              f"P50={winners['bars_to_outcome'].quantile(0.50):.0f}  "
              f"P75={winners['bars_to_outcome'].quantile(0.75):.0f}  "
              f"P90={winners['bars_to_outcome'].quantile(0.90):.0f}")


print()
print("=" * 80)
print("FERTIG.")
print("=" * 80)
