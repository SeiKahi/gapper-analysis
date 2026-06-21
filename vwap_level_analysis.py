###############################################################################
# LEVEL-TO-LEVEL VWAP ANALYSE
#
# Misst Bewegungen ZWISCHEN VWAP-Levels:
#   - Von VWAP zu ±1σ, von ±1σ zu ±2σ, von ±2σ zu ±3σ
#   - Durchbrüche: VWAP direkt zu ±2σ ohne Rücksetzer
#   - Wie oft hält ein Level vs wird durchbrochen?
#
# Für Intraday Swings: Nicht "berührt er +1σ" sondern
# "wie weit läuft der Move von Level zu Level?"
#
# Ausführen: cd gapper-analysis
#            .\gapper_env\Scripts\python.exe vwap_level_analysis.py
###############################################################################

import pandas as pd
import numpy as np
import glob
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[0]

VWAP_DIR = str(PROJECT_ROOT / 'data' / 'vwap')
METADATA_PATH = str(PROJECT_ROOT / 'data' / 'metadata' / 'metadata_master.parquet')

MIN_BARS_AFTER_OPEN = 20

# ============================================================
# METADATA
# ============================================================
print("Lade Metadata...")
meta = pd.read_parquet(METADATA_PATH)
meta['key'] = meta['ticker'] + '_' + meta['date']

meta['gap_adr_bucket'] = pd.cut(
    meta['gap_size_in_adr'].abs(),
    bins=[0, 1.0, 2.0, 100],
    labels=['<1_ADR', '1-2_ADR', '>2_ADR']
)
meta['rvol_bucket'] = pd.cut(
    meta['rvol_at_time_30min'],
    bins=[0, 1.5, 3.0, 1000],
    labels=['RVOL<1.5', 'RVOL_1.5-3', 'RVOL>3']
)

meta_lookup = meta.set_index('key')[[
    'gap_adr_bucket', 'rvol_bucket', 'gap_direction', 'adr_10'
]].to_dict('index')

print(f"Metadata: {len(meta)} Gapper")

# ============================================================
# LEVEL DEFINITIONS
# ============================================================
# Levels von unten nach oben: -3σ, -2σ, -1σ, VWAP, +1σ, +2σ, +3σ
LEVEL_NAMES = ['-3σ', '-2σ', '-1σ', 'VWAP', '+1σ', '+2σ', '+3σ']
LEVEL_IDX = {name: i for i, name in enumerate(LEVEL_NAMES)}

def get_level_values(row):
    """Gibt die 7 Level-Werte für eine Minute zurück."""
    return [
        row['lower_3std'],
        row['lower_2std'],
        row['lower_1std'],
        row['vwap'],
        row['upper_1std'],
        row['upper_2std'],
        row['upper_3std'],
    ]

def price_to_zone(price, levels):
    """
    Bestimmt in welcher Zone der Preis ist.
    Zone 0: unter -3σ
    Zone 1: -3σ bis -2σ
    Zone 2: -2σ bis -1σ
    Zone 3: -1σ bis VWAP
    Zone 4: VWAP bis +1σ
    Zone 5: +1σ bis +2σ
    Zone 6: +2σ bis +3σ
    Zone 7: über +3σ
    """
    for i, lvl in enumerate(levels):
        if price < lvl:
            return i
    return len(levels)  # über +3σ

def zone_to_label(zone):
    labels = ['unter_-3σ', '-3σ_bis_-2σ', '-2σ_bis_-1σ', '-1σ_bis_VWAP',
              'VWAP_bis_+1σ', '+1σ_bis_+2σ', '+2σ_bis_+3σ', 'über_+3σ']
    if 0 <= zone < len(labels):
        return labels[zone]
    return f'zone_{zone}'


# ============================================================
# SWING DETECTION
# ============================================================

def detect_swings(df, adr_10):
    """
    Erkennt Level-Übergänge und trackt wie weit der Move geht.
    
    Ein "Swing" startet wenn der Preis ein Level kreuzt.
    Der Swing endet wenn:
    a) Der Preis das nächste Level in Bewegungsrichtung erreicht → "Continuation"
    b) Der Preis zurück zum Start-Level geht → "Rejection"
    c) Tag endet → "Unresolved"
    
    Zusätzlich: Multi-Level Swings
    Wenn Preis von VWAP zu +1σ geht, tracken wir ob er weiter zu +2σ geht
    OHNE dass er zurück unter +1σ fällt.
    """
    swings = []
    
    df = df.dropna(subset=['vwap', 'std_dev']).copy()
    df = df[df['std_dev'] > 0].copy()
    
    if len(df) < MIN_BARS_AFTER_OPEN + 10:
        return swings
    
    n = len(df)
    rows = df.to_dict('records')
    
    # Running high/low für %ADR
    closes = df['close'].values
    running_high = np.maximum.accumulate(closes)
    running_low = np.minimum.accumulate(closes)
    if adr_10 > 0:
        pct_adr = (running_high - running_low) / adr_10
    else:
        pct_adr = np.zeros(n)
    
    # Für jeden Bar: welche Zone?
    zones = []
    all_levels = []
    for row in rows:
        lvls = get_level_values(row)
        all_levels.append(lvls)
        zones.append(price_to_zone(row['close'], lvls))
    
    times = df['time_et'].values
    
    # ──────────────────────────────────────────────────────
    # SWING DETECTION: Level-Crossing Events
    # ──────────────────────────────────────────────────────
    
    for i in range(MIN_BARS_AFTER_OPEN, n - 5):
        prev_zone = zones[i - 1]
        curr_zone = zones[i]
        
        # Keine Zone-Änderung → kein Event
        if prev_zone == curr_zone:
            continue
        
        curr_close = closes[i]
        curr_levels = all_levels[i]
        
        # Bestimme welches Level gekreuzt wurde und Richtung
        if curr_zone > prev_zone:
            # Aufwärtsbewegung: Preis hat ein Level nach oben gekreuzt
            direction = 'up'
            # Welches Level wurde gekreuzt? Das Level zwischen den Zonen
            crossed_level_idx = prev_zone  # Das obere Level der alten Zone
            if crossed_level_idx >= len(LEVEL_NAMES):
                continue
            crossed_level = LEVEL_NAMES[crossed_level_idx]
            start_zone = curr_zone
        else:
            # Abwärtsbewegung
            direction = 'down'
            crossed_level_idx = curr_zone  # Das untere Level der neuen Zone
            if crossed_level_idx >= len(LEVEL_NAMES):
                continue
            crossed_level = LEVEL_NAMES[crossed_level_idx]
            start_zone = curr_zone
        
        # ──────────────────────────────────────────────────
        # Track: Wie weit geht der Move?
        # Zähle wie viele weitere Levels in Richtung erreicht werden
        # OHNE dass das Start-Level wieder berührt wird
        # ──────────────────────────────────────────────────
        
        max_levels_reached = 0
        levels_reached_times = {}
        entry_price = curr_close
        max_move_adr = 0.0
        max_adverse_adr = 0.0
        outcome = 'unresolved'
        bars_to_end = 0
        
        for j in range(i + 1, n):
            price = closes[j]
            j_levels = all_levels[j]
            j_zone = zones[j]
            
            # Track max move in ADR
            if adr_10 > 0:
                if direction == 'up':
                    move = (price - entry_price) / adr_10
                    adverse = (entry_price - price) / adr_10
                else:
                    move = (entry_price - price) / adr_10
                    adverse = (price - entry_price) / adr_10
                max_move_adr = max(max_move_adr, move)
                max_adverse_adr = max(max_adverse_adr, adverse)
            
            if direction == 'up':
                # Wie viele Levels über Start?
                levels_above = j_zone - start_zone
                if levels_above > max_levels_reached:
                    max_levels_reached = levels_above
                    levels_reached_times[max_levels_reached] = j - i
                
                # Zurückgefallen zum/unter Start-Level?
                if j_zone <= prev_zone:
                    outcome = 'rejected'
                    bars_to_end = j - i
                    break
            else:
                levels_below = start_zone - j_zone
                if levels_below > max_levels_reached:
                    max_levels_reached = levels_below
                    levels_reached_times[max_levels_reached] = j - i
                
                if j_zone >= prev_zone:
                    outcome = 'rejected'
                    bars_to_end = j - i
                    break
        
        if outcome != 'rejected':
            if max_levels_reached > 0:
                outcome = f'continued_{max_levels_reached}_levels'
            else:
                outcome = 'held_zone'
            bars_to_end = n - i
        
        swings.append({
            'time': times[i],
            'direction': direction,
            'crossed_level': crossed_level,
            'start_zone': zone_to_label(start_zone),
            'max_levels_continued': max_levels_reached,
            'outcome': outcome,
            'bars_to_end': bars_to_end,
            'max_move_adr': round(max_move_adr, 3),
            'max_adverse_adr': round(max_adverse_adr, 3),
            'pct_adr_used': round(pct_adr[i], 2),
            'bar_index': i,
        })
    
    return swings


# ============================================================
# ALLE DATEIEN VERARBEITEN
# ============================================================
print("\nVerarbeite VWAP-Dateien...")
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
        
        swings = detect_swings(df, adr_10)
        
        for sw in swings:
            sw['ticker'] = ticker
            sw['date'] = date
            sw['gap_direction'] = info['gap_direction']
            sw['gap_adr_bucket'] = info['gap_adr_bucket']
            sw['rvol_bucket'] = info['rvol_bucket']
        
        all_swings.extend(swings)
    except Exception:
        continue

swings_df = pd.DataFrame(all_swings)
print(f"\n{len(swings_df):,} Swings erkannt")

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

swings_df['time_bucket'] = swings_df['time'].apply(time_bucket)
swings_df['adr_bucket'] = pd.cut(
    swings_df['pct_adr_used'],
    bins=[0, 0.5, 1.0, 1.5, 100],
    labels=['<50%_ADR', '50-100%_ADR', '100-150%_ADR', '>150%_ADR']
)

# ################################################################
#                     ERGEBNISSE
# ################################################################

# ============================================================
# ANALYSE 1: Level-Crossing → Wie viele weitere Levels?
# ============================================================
print()
print("=" * 80)
print("ANALYSE 1: LEVEL-CROSSING → WIE WEIT GEHT DER MOVE?")
print("  Wenn der Preis ein Level kreuzt, wie viele weitere Levels")
print("  erreicht er bevor er zurückfällt?")
print("=" * 80)

# Für jedes Level und jede Richtung
for level in ['VWAP', '+1σ', '+2σ', '-1σ', '-2σ']:
    for direction in ['up', 'down']:
        sub = swings_df[(swings_df['crossed_level'] == level) & 
                        (swings_df['direction'] == direction)]
        n = len(sub)
        if n < 100:
            continue
        
        if direction == 'up':
            desc = f"Preis kreuzt {level} nach OBEN"
        else:
            desc = f"Preis kreuzt {level} nach UNTEN"
        
        print(f"\n{'─' * 80}")
        print(f"  {desc}  (n={n:,})")
        print(f"{'─' * 80}")
        
        # Verteilung: wie viele Levels weiter?
        rejected = (sub['outcome'] == 'rejected').sum()
        rej_pct = rejected / n * 100
        
        cont_counts = {}
        for lvls in range(1, 5):
            c = (sub['max_levels_continued'] >= lvls).sum()
            cont_counts[lvls] = c
        
        print(f"    Sofort rejected (0 weitere Levels):  {rejected:5d} ({rej_pct:5.1f}%)")
        for lvls in range(1, 5):
            c = cont_counts[lvls]
            pct = c / n * 100
            bar = '█' * int(pct / 2)
            # Median ADR move für diese Gruppe
            group = sub[sub['max_levels_continued'] >= lvls]
            med_adr = group['max_move_adr'].median()
            med_time = group['bars_to_end'].median()
            print(f"    Mindestens {lvls} Level(s) weiter:      {c:5d} ({pct:5.1f}%)  "
                  f"~{med_adr:.2f} ADR  ~{med_time:.0f}min  {bar}")
        
        # Median Move
        med_move = sub['max_move_adr'].median()
        med_adv = sub['max_adverse_adr'].median()
        print(f"\n    Median Max Move:     {med_move:+.3f} ADR")
        print(f"    Median Max Adverse:  {med_adv:+.3f} ADR")

# ============================================================
# ANALYSE 2: SPEZIFISCHE LEVEL-ZU-LEVEL MOVES
# ============================================================
print()
print("=" * 80)
print("ANALYSE 2: SPEZIFISCHE LEVEL-ZU-LEVEL MOVES")
print("  Wenn Preis Level A erreicht: Wahrscheinlichkeit Level B zu erreichen")
print("  OHNE zurück unter Level A zu fallen")
print("=" * 80)

transitions = [
    ('VWAP', 'up', 'VWAP → +1σ → +2σ → +3σ (Aufwärtsschwung)'),
    ('VWAP', 'down', 'VWAP → -1σ → -2σ → -3σ (Abwärtsschwung)'),
    ('+1σ', 'up', '+1σ → +2σ → +3σ (Continuation nach oben)'),
    ('-1σ', 'down', '-1σ → -2σ → -3σ (Continuation nach unten)'),
    ('+1σ', 'down', '+1σ → VWAP → -1σ (Reversal von oben)'),
    ('-1σ', 'up', '-1σ → VWAP → +1σ (Reversal von unten)'),
]

for level, direction, desc in transitions:
    sub = swings_df[(swings_df['crossed_level'] == level) & 
                    (swings_df['direction'] == direction)]
    n = len(sub)
    if n < 100:
        continue
    
    print(f"\n{'━' * 80}")
    print(f"  {desc}  (n={n:,})")
    print(f"{'━' * 80}")
    
    # Cascade: wie viele erreichen jedes weitere Level?
    prev_n = n
    for lvls in range(0, 4):
        group = sub[sub['max_levels_continued'] >= lvls]
        c = len(group)
        if prev_n > 0:
            cascade_pct = c / n * 100
            cond_pct = c / prev_n * 100 if lvls > 0 else cascade_pct
        else:
            cascade_pct = 0
            cond_pct = 0
        
        med_adr = group['max_move_adr'].median() if len(group) > 0 else 0
        med_time = group['bars_to_end'].median() if len(group) > 0 else 0
        
        if lvls == 0:
            label = f"Kreuzt {level}"
        else:
            label = f"+{lvls} Level(s) weiter"
        
        bar = '█' * int(cascade_pct / 2)
        print(f"    {label:25s}  {c:6d} ({cascade_pct:5.1f}% von Start, "
              f"{cond_pct:5.1f}% vom Vorherigen)  ~{med_adr:.2f} ADR  ~{med_time:.0f}min  {bar}")
        
        prev_n = c

# ============================================================
# ANALYSE 3: LEVEL-TO-LEVEL nach TAGESZEIT
# ============================================================
print()
print("=" * 80)
print("ANALYSE 3: VWAP-CROSSING NACH OBEN → Wie weit? Nach TAGESZEIT")
print("=" * 80)

for tb in ['1_open_30min', '2_30_60min', '3_1_2h', '4_mittag', '5_nachmittag']:
    sub = swings_df[(swings_df['crossed_level'] == 'VWAP') & 
                    (swings_df['direction'] == 'up') &
                    (swings_df['time_bucket'] == tb)]
    n = len(sub)
    if n < 50:
        continue
    
    print(f"\n  {tb} (n={n:,})")
    
    rejected = (sub['outcome'] == 'rejected').sum()
    print(f"    Rejected: {rejected} ({rejected/n*100:.1f}%)")
    
    for lvls in range(1, 4):
        c = (sub['max_levels_continued'] >= lvls).sum()
        pct = c / n * 100
        grp = sub[sub['max_levels_continued'] >= lvls]
        med_adr = grp['max_move_adr'].median() if len(grp) > 0 else 0
        bar = '█' * int(pct / 2)
        print(f"    +{lvls} Level(s): {c:5d} ({pct:5.1f}%)  ~{med_adr:.2f} ADR  {bar}")

# ============================================================
# ANALYSE 4: LEVEL-TO-LEVEL nach %ADR VERBRAUCHT
# ============================================================
print()
print("=" * 80)
print("ANALYSE 4: VWAP-CROSSING → Wie weit? Nach %ADR VERBRAUCHT")
print("=" * 80)

for level in ['VWAP', '+1σ', '-1σ']:
    for direction in ['up', 'down']:
        desc = f"{level} nach {'oben' if direction == 'up' else 'unten'}"
        
        print(f"\n{'━' * 80}")
        print(f"  {desc}")
        print(f"{'━' * 80}")
        
        for adr_b in ['<50%_ADR', '50-100%_ADR', '100-150%_ADR', '>150%_ADR']:
            sub = swings_df[(swings_df['crossed_level'] == level) & 
                            (swings_df['direction'] == direction) &
                            (swings_df['adr_bucket'] == adr_b)]
            n = len(sub)
            if n < 50:
                continue
            
            rejected = (sub['outcome'] == 'rejected').sum()
            rej_pct = rejected / n * 100
            
            reach_1 = (sub['max_levels_continued'] >= 1).sum()
            reach_2 = (sub['max_levels_continued'] >= 2).sum()
            r1_pct = reach_1 / n * 100
            r2_pct = reach_2 / n * 100
            
            med_adr = sub['max_move_adr'].median()
            
            print(f"\n  {adr_b} (n={n:,})")
            print(f"    Rejected: {rej_pct:5.1f}%  |  +1 Level: {r1_pct:5.1f}%  |  +2 Levels: {r2_pct:5.1f}%  |  Med Move: {med_adr:.2f} ADR")

# ============================================================
# ANALYSE 5: LEVEL-TO-LEVEL nach GAP-ADR und RVOL
# ============================================================
print()
print("=" * 80)
print("ANALYSE 5: VWAP-CROSSING → Wie weit? Nach GAP-ADR und RVOL")
print("=" * 80)

for level in ['VWAP']:
    for direction in ['up', 'down']:
        desc = f"{level} nach {'oben' if direction == 'up' else 'unten'}"
        
        print(f"\n{'━' * 80}")
        print(f"  {desc}")
        print(f"{'━' * 80}")
        
        print(f"\n  {'Gap-ADR':10s} {'RVOL':10s} {'N':>6s}  {'Rej%':>6s}  {'+1Lvl%':>7s}  {'+2Lvl%':>7s}  {'MedMove':>8s}")
        print(f"  {'─'*10} {'─'*10} {'─'*6}  {'─'*6}  {'─'*7}  {'─'*7}  {'─'*8}")
        
        results = []
        for gap_b in ['<1_ADR', '1-2_ADR', '>2_ADR']:
            for rvol_b in ['RVOL<1.5', 'RVOL_1.5-3', 'RVOL>3']:
                sub = swings_df[(swings_df['crossed_level'] == level) & 
                                (swings_df['direction'] == direction) &
                                (swings_df['gap_adr_bucket'] == gap_b) &
                                (swings_df['rvol_bucket'] == rvol_b)]
                n = len(sub)
                if n < 50:
                    continue
                
                rej = (sub['outcome'] == 'rejected').mean() * 100
                r1 = (sub['max_levels_continued'] >= 1).mean() * 100
                r2 = (sub['max_levels_continued'] >= 2).mean() * 100
                med = sub['max_move_adr'].median()
                results.append((gap_b, rvol_b, n, rej, r1, r2, med))
        
        results.sort(key=lambda x: x[5], reverse=True)  # Sort by +2 Level %
        for gap_b, rvol_b, n, rej, r1, r2, med in results:
            print(f"  {gap_b:10s} {rvol_b:10s} {n:6d}  {rej:5.1f}%  {r1:6.1f}%  {r2:6.1f}%  {med:+7.3f}")

# ============================================================
# ANALYSE 6: MULTI-LEVEL SWINGS — Move-Größe in ADR
# ============================================================
print()
print("=" * 80)
print("ANALYSE 6: MOVE-GRÖßE (ADR) JE NACH ANZAHL ERREICHTER LEVELS")
print("  → Percentile der tatsächlichen ADR-Bewegung")
print("=" * 80)

for level in ['VWAP', '+1σ', '-1σ']:
    for direction in ['up', 'down']:
        sub = swings_df[(swings_df['crossed_level'] == level) & 
                        (swings_df['direction'] == direction)]
        if len(sub) < 100:
            continue
        
        desc = f"{level} nach {'oben' if direction == 'up' else 'unten'}"
        print(f"\n{'━' * 80}")
        print(f"  {desc}  (n={len(sub):,})")
        print(f"{'━' * 80}")
        
        for lvls in range(0, 4):
            if lvls == 0:
                group = sub[sub['outcome'] == 'rejected']
                label = "Rejected (0 Levels)"
            else:
                group = sub[sub['max_levels_continued'] == lvls]
                label = f"Exakt {lvls} Level(s)"
            
            n = len(group)
            if n < 20:
                continue
            
            p25 = group['max_move_adr'].quantile(0.25)
            p50 = group['max_move_adr'].quantile(0.50)
            p75 = group['max_move_adr'].quantile(0.75)
            p90 = group['max_move_adr'].quantile(0.90)
            
            t25 = group['bars_to_end'].quantile(0.25)
            t50 = group['bars_to_end'].quantile(0.50)
            t75 = group['bars_to_end'].quantile(0.75)
            
            print(f"\n  {label} (n={n:,})")
            print(f"    Move (ADR): P25={p25:+.2f}  P50={p50:+.2f}  P75={p75:+.2f}  P90={p90:+.2f}")
            print(f"    Dauer:      P25={t25:.0f}min  P50={t50:.0f}min  P75={t75:.0f}min")

# ============================================================
# ANALYSE 7: GAP-RICHTUNG vs SWING-RICHTUNG
# ============================================================
print()
print("=" * 80)
print("ANALYSE 7: SWING PRO GAP vs GEGEN GAP")
print("  → Sind Level-zu-Level Moves stärker in Gap-Richtung?")
print("=" * 80)

for level in ['VWAP', '+1σ', '-1σ']:
    sub = swings_df[swings_df['crossed_level'] == level]
    if len(sub) < 200:
        continue
    
    print(f"\n{'━' * 80}")
    print(f"  Level: {level}")
    print(f"{'━' * 80}")
    
    combos = [
        ('up', 'up', 'Gap Up + Swing Up (pro Gap)'),
        ('up', 'down', 'Gap Up + Swing Down (gegen Gap)'),
        ('down', 'up', 'Gap Down + Swing Up (gegen Gap)'),
        ('down', 'down', 'Gap Down + Swing Down (pro Gap)'),
    ]
    
    for gap_dir, swing_dir, label in combos:
        s = sub[(sub['gap_direction'] == gap_dir) & (sub['direction'] == swing_dir)]
        n = len(s)
        if n < 50:
            continue
        
        rej = (s['outcome'] == 'rejected').mean() * 100
        r1 = (s['max_levels_continued'] >= 1).mean() * 100
        r2 = (s['max_levels_continued'] >= 2).mean() * 100
        med = s['max_move_adr'].median()
        
        print(f"\n  {label} (n={n:,})")
        print(f"    Rejected: {rej:5.1f}%  |  +1 Level: {r1:5.1f}%  |  +2 Levels: {r2:5.1f}%  |  Med Move: {med:.3f} ADR")


print()
print("=" * 80)
print("FERTIG.")
print("=" * 80)
