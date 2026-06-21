###############################################################################
# LEVEL-TO-LEVEL VWAP ANALYSE v5 — VERHALTENSANALYSE
#
# Ziel: Wie verhalten sich Aktien unter bestimmten Parametern?
#   NICHT: "Hätte ein konkreter Trade funktioniert?"
#   SONDERN: "Was passiert typischerweise wenn eine Aktie bei +2σ ist?"
#
# Filter-Dimensionen:
#   1. Gap-ADR (wie groß war der Gap relativ zur ADR?)
#   2. RVOL (wie viel Volumen?)
#   3. %ADR used (wie viel hat sich die Aktie schon bewegt?)
#   4. Early Range 15min (wie viel %ADR in ersten 15 Minuten?)
#   5. Early Range 30min (wie viel %ADR in ersten 30 Minuten?)
#   6. Tageszeit
#   7. ADR% (wie volatil ist die Aktie generell? Min 2% etc.)
#
# Ausführen: cd gapper-analysis
#            .\gapper_env\Scripts\python.exe vwap_level_matrix_v5.py > ergebnisse_v5.txt 2>&1
###############################################################################

import pandas as pd
import numpy as np
import glob
from tqdm import tqdm
from itertools import product
import warnings
import sys
import io

# ============================================================
# WINDOWS UTF-8 FIX
# ============================================================
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
MIN_SAMPLE = 30

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
LEVEL_NAMES = ['-3σ', '-2σ', '-1σ', 'VWAP', '+1σ', '+2σ', '+3σ']

def price_to_zone(price, levels):
    for i, lvl in enumerate(levels):
        if price < lvl:
            return i
    return len(levels)

# ============================================================
# EARLY RANGE BERECHNUNG (nutzt High/Low wenn verfügbar)
# ============================================================
def compute_early_range(df, adr_10):
    if adr_10 <= 0:
        return 0.0, 0.0
    
    market = df[(df['time_et'] >= '09:30') & (df['time_et'] <= '16:00')].copy()
    if len(market) == 0:
        return 0.0, 0.0
    
    times = market['time_et'].values
    use_hl = 'high' in market.columns and 'low' in market.columns
    
    if use_hl:
        highs = market['high'].values
        lows = market['low'].values
    else:
        highs = market['close'].values
        lows = market['close'].values
    
    mask_15 = times < '09:45'
    if mask_15.any():
        range_15 = (highs[mask_15].max() - lows[mask_15].min()) / adr_10
    else:
        range_15 = 0.0
    
    mask_30 = times < '10:00'
    if mask_30.any():
        range_30 = (highs[mask_30].max() - lows[mask_30].min()) / adr_10
    else:
        range_30 = 0.0
    
    return round(range_15, 3), round(range_30, 3)

# ============================================================
# SWING DETECTION (Zonen-basiert, Verhaltensanalyse)
# ============================================================
def detect_swings(df, adr_10):
    swings = []
    df = df.dropna(subset=['vwap', 'std_dev']).copy()
    df = df[df['std_dev'] > 0].copy()
    
    if len(df) < MIN_BARS_AFTER_OPEN + 10:
        return swings
    
    n = len(df)
    closes = df['close'].values
    times = df['time_et'].values
    
    running_high = np.maximum.accumulate(closes)
    running_low = np.minimum.accumulate(closes)
    pct_adr = (running_high - running_low) / adr_10 if adr_10 > 0 else np.zeros(n)
    
    zones = []
    band_widths = []
    for _, row in df.iterrows():
        lvls = [row['lower_3std'], row['lower_2std'], row['lower_1std'],
                row['vwap'], row['upper_1std'], row['upper_2std'], row['upper_3std']]
        zones.append(price_to_zone(row['close'], lvls))
        bw = (row['upper_1std'] - row['lower_1std']) / adr_10 if adr_10 > 0 else 0
        band_widths.append(bw)
    
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
        
        max_levels_reached = 0
        entry_price = closes[i]
        max_move_adr = 0.0
        max_adverse_adr = 0.0
        outcome = 'unresolved'
        bars_to_end = 0
        
        for j in range(i + 1, n):
            price = closes[j]
            j_zone = zones[j]
            
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
                levels_above = j_zone - start_zone
                if levels_above > max_levels_reached:
                    max_levels_reached = levels_above
                if j_zone <= prev_zone:
                    outcome = 'rejected'
                    bars_to_end = j - i
                    break
            else:
                levels_below = start_zone - j_zone
                if levels_below > max_levels_reached:
                    max_levels_reached = levels_below
                if j_zone >= prev_zone:
                    outcome = 'rejected'
                    bars_to_end = j - i
                    break
        
        if outcome != 'rejected':
            outcome = f'continued_{max_levels_reached}_levels' if max_levels_reached > 0 else 'held_zone'
            bars_to_end = n - i
        
        swings.append({
            'time': times[i],
            'direction': direction,
            'crossed_level': crossed_level,
            'max_levels_continued': max_levels_reached,
            'outcome': outcome,
            'bars_to_end': bars_to_end,
            'max_move_adr': round(max_move_adr, 3),
            'max_adverse_adr': round(max_adverse_adr, 3),
            'pct_adr_used': round(pct_adr[i], 2),
            'band_width_adr': round(band_widths[i], 3),
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
        if adr_10 <= 0:
            continue
        
        # ADR% berechnen: ADR / Preis * 100
        first_close = df[df['time_et'] >= '09:30']['close'].iloc[0] if len(df[df['time_et'] >= '09:30']) > 0 else df['close'].iloc[0]
        adr_pct = (adr_10 / first_close * 100) if first_close > 0 else 0
        
        er_15, er_30 = compute_early_range(df, adr_10)
        swings = detect_swings(df, adr_10)
        
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
            sw['price'] = round(first_close, 2)
        
        all_swings.extend(swings)
    except Exception:
        continue

df_all = pd.DataFrame(all_swings)
if len(df_all) == 0:
    print("FEHLER: Keine Swings gefunden!")
    sys.exit()

print(f"\n{len(df_all):,} Swings erkannt")

# ============================================================
# ADR% VERTEILUNG
# ============================================================
print("\n" + "=" * 80)
print("ADR% VERTEILUNG (ADR als % des Preises)")
print("=" * 80)

day_data = df_all.drop_duplicates(subset=['ticker', 'date'])[
    ['ticker', 'date', 'adr_pct', 'price', 'early_range_15', 'early_range_30', 'gap_adr', 'rvol']
].copy()
print(f"\n  Unique Ticker-Tage: {len(day_data):,}")

print(f"\n  ADR% Verteilung:")
for pct in [1, 2, 3, 5, 7, 10, 15]:
    count = (day_data['adr_pct'] >= pct).sum()
    print(f"    ADR >= {pct:2d}%: {count:5,} ({count/len(day_data)*100:.1f}%)")

print(f"\n  ADR% Percentile:")
for p in [10, 25, 50, 75, 90]:
    val = day_data['adr_pct'].quantile(p/100)
    print(f"    P{p}: {val:.1f}%")

print(f"\n  Preis-Verteilung:")
for p in [10, 25, 50, 75, 90]:
    val = day_data['price'].quantile(p/100)
    print(f"    P{p}: ${val:.2f}")

print(f"\n  Early Range 15min:")
for pct in [0.25, 0.50, 0.75, 1.0]:
    count = (day_data['early_range_15'] >= pct).sum()
    print(f"    >= {pct*100:.0f}%: {count:,} ({count/len(day_data)*100:.1f}%)")

print(f"\n  Early Range 30min:")
for pct in [0.25, 0.50, 0.75, 1.0]:
    count = (day_data['early_range_30'] >= pct).sum()
    print(f"    >= {pct*100:.0f}%: {count:,} ({count/len(day_data)*100:.1f}%)")

print(f"\n  ADR% × Early Range 15min (Median ER15 pro ADR%-Bucket):")
for adr_lo, adr_hi, label in [(0,2,'<2%'), (2,3,'2-3%'), (3,5,'3-5%'), (5,7,'5-7%'), (7,10,'7-10%'), (10,100,'>10%')]:
    sub = day_data[(day_data['adr_pct'] >= adr_lo) & (day_data['adr_pct'] < adr_hi)]
    if len(sub) < 10:
        continue
    med_er = sub['early_range_15'].median()
    ge50 = (sub['early_range_15'] >= 0.50).mean() * 100
    print(f"    ADR {label:6s}: n={len(sub):5d}  MedianER15={med_er:.2f}  ER15>=50%: {ge50:.1f}%")

# ============================================================
# BUCKETS
# ============================================================
df_all['gap_adr_bucket'] = pd.cut(
    df_all['gap_adr'],
    bins=[0, 0.75, 1.0, 1.5, 2.0, 100],
    labels=['<0.75', '0.75-1', '1-1.5', '1.5-2', '>2']
)

df_all['rvol_bucket'] = pd.cut(
    df_all['rvol'],
    bins=[0, 1.0, 1.5, 2.5, 5.0, 1000],
    labels=['<1', '1-1.5', '1.5-2.5', '2.5-5', '>5']
)

df_all['adr_used_bucket'] = pd.cut(
    df_all['pct_adr_used'],
    bins=[0, 0.5, 0.75, 1.0, 1.5, 100],
    labels=['<50%', '50-75%', '75-100%', '100-150%', '>150%']
)

df_all['er15_bucket'] = pd.cut(
    df_all['early_range_15'],
    bins=[-0.001, 0.25, 0.50, 0.75, 1.0, 100],
    labels=['<25%', '25-50%', '50-75%', '75-100%', '>100%']
)

df_all['er30_bucket'] = pd.cut(
    df_all['early_range_30'],
    bins=[-0.001, 0.25, 0.50, 0.75, 1.0, 100],
    labels=['<25%', '25-50%', '50-75%', '75-100%', '>100%']
)

df_all['adr_pct_bucket'] = pd.cut(
    df_all['adr_pct'],
    bins=[0, 2, 3, 5, 7, 10, 100],
    labels=['<2%', '2-3%', '3-5%', '5-7%', '7-10%', '>10%']
)

def time_bucket(t):
    if t <= '10:00': return '1_open30'
    elif t <= '10:30': return '2_30_60'
    elif t <= '11:30': return '3_1_2h'
    elif t <= '13:00': return '4_mittag'
    else: return '5_nachm'

df_all['time_bucket'] = df_all['time'].apply(time_bucket)

# ============================================================
# KPI HELPER
# ============================================================
def compute_kpis(sub):
    n = len(sub)
    if n < MIN_SAMPLE:
        return None
    
    reach_1 = (sub['max_levels_continued'] >= 1).sum()
    reach_2 = (sub['max_levels_continued'] >= 2).sum()
    reach_3 = (sub['max_levels_continued'] >= 3).sum()
    
    r1_pct = reach_1 / n * 100
    r2_pct = reach_2 / n * 100
    r3_pct = reach_3 / n * 100
    
    cond_2_given_1 = (reach_2 / reach_1 * 100) if reach_1 > 0 else 0
    cond_3_given_2 = (reach_3 / reach_2 * 100) if reach_2 > 0 else 0
    
    med_move = sub['max_move_adr'].median()
    med_adv = sub['max_adverse_adr'].median()
    
    r1_sub = sub[sub['max_levels_continued'] >= 1]
    med_move_r1 = r1_sub['max_move_adr'].median() if len(r1_sub) >= 10 else np.nan
    med_time_r1 = r1_sub['bars_to_end'].median() if len(r1_sub) >= 10 else np.nan
    
    med_bw = sub['band_width_adr'].median() if 'band_width_adr' in sub.columns else np.nan
    
    return {
        'n': n,
        'reach_1_pct': r1_pct,
        'reach_2_pct': r2_pct,
        'reach_3_pct': r3_pct,
        'cond_2|1': cond_2_given_1,
        'cond_3|2': cond_3_given_2,
        'med_move_all': med_move,
        'med_move_r1': med_move_r1,
        'med_time_r1': med_time_r1,
        'med_adverse': med_adv,
        'med_band_width': med_bw,
    }

def fmt_mr1(kpi):
    return f"{kpi['med_move_r1']:.3f}" if not np.isnan(kpi['med_move_r1']) else "  n/a"

def fmt_mt1(kpi):
    return f"{kpi['med_time_r1']:.0f}m" if not np.isnan(kpi['med_time_r1']) else " n/a"

def fmt_bw(kpi):
    return f"{kpi['med_band_width']:.3f}" if not np.isnan(kpi['med_band_width']) else "n/a"

# ============================================================
# SZENARIEN
# ============================================================
scenarios = [
    ('GapUp_bei_+1σ_Fade',    'up',   '+1σ', 'down', 'GAP UP: +1σ → Fade Ri. VWAP'),
    ('GapUp_bei_+1σ_Cont',    'up',   '+1σ', 'up',   'GAP UP: +1σ → Weiter zu +2σ'),
    ('GapUp_bei_+2σ_Fade',    'up',   '+2σ', 'down', 'GAP UP: +2σ → Fade zu +1σ'),
    ('GapUp_bei_VWAP_Bounce', 'up',   'VWAP', 'up',  'GAP UP: VWAP → Bounce zu +1σ'),
    ('GapUp_bei_VWAP_Break',  'up',   'VWAP', 'down', 'GAP UP: VWAP Break → -1σ'),
    ('GapUp_bei_-1σ_Bounce',  'up',   '-1σ', 'up',   'GAP UP: -1σ → Bounce zu VWAP'),
    ('GapDn_bei_-1σ_Fade',    'down', '-1σ', 'up',   'GAP DOWN: -1σ → Fade Ri. VWAP'),
    ('GapDn_bei_-1σ_Cont',    'down', '-1σ', 'down', 'GAP DOWN: -1σ → Weiter zu -2σ'),
    ('GapDn_bei_-2σ_Fade',    'down', '-2σ', 'up',   'GAP DOWN: -2σ → Fade zu -1σ'),
    ('GapDn_bei_VWAP_Bounce', 'down', 'VWAP', 'down', 'GAP DOWN: VWAP → Bounce zu -1σ'),
    ('GapDn_bei_VWAP_Break',  'down', 'VWAP', 'up',   'GAP DOWN: VWAP Break → +1σ'),
    ('GapDn_bei_+1σ_Bounce',  'down', '+1σ', 'down',  'GAP DOWN: +1σ → Bounce zu VWAP'),
]

key_fades = [
    ('GapUp_+1σ_Fade',   'up',   '+1σ', 'down', 'GAP UP: +1σ → Fade Ri. VWAP'),
    ('GapUp_+2σ_Fade',   'up',   '+2σ', 'down', 'GAP UP: +2σ → Fade zu +1σ'),
    ('GapUp_VWAP_Break',  'up',   'VWAP', 'down', 'GAP UP: VWAP Break → -1σ'),
    ('GapDn_-1σ_Fade',   'down', '-1σ', 'up',   'GAP DOWN: -1σ → Fade Ri. VWAP'),
    ('GapDn_-2σ_Fade',   'down', '-2σ', 'up',   'GAP DOWN: -2σ → Fade zu -1σ'),
    ('GapDn_VWAP_Break',  'down', 'VWAP', 'up',  'GAP DOWN: VWAP Break → +1σ'),
]

# ################################################################
#                     ANALYSEN
# ################################################################

# ================================================================
# ANALYSE 1: ÜBERSICHT ALLER 12 SZENARIEN
# ================================================================
print()
print("=" * 130)
print("ANALYSE 1: ÜBERSICHT ALLER SZENARIEN")
print("=" * 130)

print(f"\n{'Szenario':40s} {'N':>6s}  {'->1Lvl':>7s}  {'->2Lvl':>7s}  {'->3Lvl':>7s}  "
      f"{'2|1':>5s}  {'3|2':>5s}  {'MedADR':>7s}  {'MedADR1+':>9s}  {'BandW':>6s}")
print("-" * 130)

for name, gap_dir, level, swing_dir, desc in scenarios:
    sub = df_all[(df_all['gap_direction'] == gap_dir) & 
                 (df_all['crossed_level'] == level) & 
                 (df_all['direction'] == swing_dir)]
    kpi = compute_kpis(sub)
    if kpi is None:
        continue
    print(f"{name:40s} {kpi['n']:6d}  {kpi['reach_1_pct']:6.1f}%  {kpi['reach_2_pct']:6.1f}%  "
          f"{kpi['reach_3_pct']:6.1f}%  {kpi['cond_2|1']:4.0f}%  {kpi['cond_3|2']:4.0f}%  "
          f"{kpi['med_move_all']:6.3f}  {fmt_mr1(kpi):>9s}  {fmt_bw(kpi):>6s}")

# ================================================================
# ANALYSE 2: ADR% ALS FILTER — DER NEUE VORFILTER
# ================================================================
print()
print("=" * 130)
print("ANALYSE 2: ADR% ALS VORFILTER")
print("  Hypothese: Aktien mit ADR < 2% haben enge Baender und erzeugen Rauschen")
print("=" * 130)

adr_pct_thresholds = [
    ('ALLE',          lambda df: df),
    ('ADR >= 1%',     lambda df: df[df['adr_pct'] >= 1]),
    ('ADR >= 2%',     lambda df: df[df['adr_pct'] >= 2]),
    ('ADR >= 3%',     lambda df: df[df['adr_pct'] >= 3]),
    ('ADR >= 5%',     lambda df: df[df['adr_pct'] >= 5]),
    ('ADR >= 7%',     lambda df: df[df['adr_pct'] >= 7]),
    ('ADR >= 10%',    lambda df: df[df['adr_pct'] >= 10]),
]

for name, gap_dir, level, swing_dir, desc in key_fades:
    base = df_all[(df_all['gap_direction'] == gap_dir) & 
                  (df_all['crossed_level'] == level) & 
                  (df_all['direction'] == swing_dir)]
    if len(base) < 100:
        continue
    
    print(f"\n{'=' * 130}")
    print(f"  {desc}")
    print(f"{'=' * 130}")
    print(f"  {'ADR% Filter':14s} {'N':>7s}  {'->1Lvl':>7s}  {'->2Lvl':>7s}  {'->3Lvl':>7s}  "
          f"{'2|1':>5s}  {'MedADR':>7s}  {'MedADR1+':>9s}  {'BandW':>6s}")
    print(f"  {'-'*14} {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*5}  {'-'*7}  {'-'*9}  {'-'*6}")
    
    for f_name, f_func in adr_pct_thresholds:
        sub = f_func(base)
        kpi = compute_kpis(sub)
        if kpi is None:
            continue
        bar = '#' * int(kpi['reach_1_pct'] / 2)
        print(f"  {f_name:14s} {kpi['n']:7d}  {kpi['reach_1_pct']:6.1f}%  {kpi['reach_2_pct']:6.1f}%  "
              f"{kpi['reach_3_pct']:6.1f}%  {kpi['cond_2|1']:4.0f}%  {kpi['med_move_all']:6.3f}  "
              f"{fmt_mr1(kpi):>9s}  {fmt_bw(kpi):>6s}  {bar}")

# ================================================================
# ANALYSE 3: ADR% BUCKETS — Detaillierter Blick
# ================================================================
print()
print("=" * 130)
print("ANALYSE 3: ADR% BUCKETS (Detail)")
print("=" * 130)

adr_pct_labels = ['<2%', '2-3%', '3-5%', '5-7%', '7-10%', '>10%']

for name, gap_dir, level, swing_dir, desc in key_fades:
    base = df_all[(df_all['gap_direction'] == gap_dir) & 
                  (df_all['crossed_level'] == level) & 
                  (df_all['direction'] == swing_dir)]
    if len(base) < 100:
        continue
    
    print(f"\n  {desc}")
    print(f"  {'ADR%':8s} {'N':>6s}  {'->1Lvl':>7s}  {'->2Lvl':>7s}  {'2|1':>5s}  "
          f"{'MedADR':>7s}  {'MedADR1+':>9s}  {'MedTime':>7s}  {'BandW':>6s}")
    print(f"  {'-'*8} {'-'*6}  {'-'*7}  {'-'*7}  {'-'*5}  {'-'*7}  {'-'*9}  {'-'*7}  {'-'*6}")
    
    for adr_b in adr_pct_labels:
        sub = base[base['adr_pct_bucket'] == adr_b]
        kpi = compute_kpis(sub)
        if kpi is None:
            continue
        bar = '#' * int(kpi['reach_1_pct'] / 2)
        print(f"  {adr_b:8s} {kpi['n']:6d}  {kpi['reach_1_pct']:6.1f}%  {kpi['reach_2_pct']:6.1f}%  "
              f"{kpi['cond_2|1']:4.0f}%  {kpi['med_move_all']:6.3f}  {fmt_mr1(kpi):>9s}  "
              f"{fmt_mt1(kpi):>7s}  {fmt_bw(kpi):>6s}  {bar}")

# ================================================================
# ANALYSE 4: EARLY RANGE 15min
# ================================================================
print()
print("=" * 130)
print("ANALYSE 4: SZENARIEN x EARLY RANGE 15min")
print("=" * 130)

er15_labels = ['<25%', '25-50%', '50-75%', '75-100%', '>100%']

for name, gap_dir, level, swing_dir, desc in key_fades:
    base = df_all[(df_all['gap_direction'] == gap_dir) & 
                  (df_all['crossed_level'] == level) & 
                  (df_all['direction'] == swing_dir)]
    if len(base) < 100:
        continue
    
    print(f"\n  {desc}")
    print(f"  {'ER15':8s} {'N':>6s}  {'->1Lvl':>7s}  {'->2Lvl':>7s}  {'2|1':>5s}  "
          f"{'MedADR':>7s}  {'MedADR1+':>9s}  {'MedTime':>7s}  {'BandW':>6s}")
    print(f"  {'-'*8} {'-'*6}  {'-'*7}  {'-'*7}  {'-'*5}  {'-'*7}  {'-'*9}  {'-'*7}  {'-'*6}")
    
    for er_b in er15_labels:
        sub = base[base['er15_bucket'] == er_b]
        kpi = compute_kpis(sub)
        if kpi is None:
            continue
        bar = '#' * int(kpi['reach_1_pct'] / 2)
        print(f"  {er_b:8s} {kpi['n']:6d}  {kpi['reach_1_pct']:6.1f}%  {kpi['reach_2_pct']:6.1f}%  "
              f"{kpi['cond_2|1']:4.0f}%  {kpi['med_move_all']:6.3f}  {fmt_mr1(kpi):>9s}  "
              f"{fmt_mt1(kpi):>7s}  {fmt_bw(kpi):>6s}  {bar}")

# ================================================================
# ANALYSE 5: EARLY RANGE 30min
# ================================================================
print()
print("=" * 130)
print("ANALYSE 5: SZENARIEN x EARLY RANGE 30min")
print("=" * 130)

er30_labels = ['<25%', '25-50%', '50-75%', '75-100%', '>100%']

for name, gap_dir, level, swing_dir, desc in key_fades:
    base = df_all[(df_all['gap_direction'] == gap_dir) & 
                  (df_all['crossed_level'] == level) & 
                  (df_all['direction'] == swing_dir)]
    if len(base) < 100:
        continue
    
    print(f"\n  {desc}")
    print(f"  {'ER30':8s} {'N':>6s}  {'->1Lvl':>7s}  {'->2Lvl':>7s}  {'2|1':>5s}  "
          f"{'MedADR':>7s}  {'MedADR1+':>9s}  {'MedTime':>7s}  {'BandW':>6s}")
    print(f"  {'-'*8} {'-'*6}  {'-'*7}  {'-'*7}  {'-'*5}  {'-'*7}  {'-'*9}  {'-'*7}  {'-'*6}")
    
    for er_b in er30_labels:
        sub = base[base['er30_bucket'] == er_b]
        kpi = compute_kpis(sub)
        if kpi is None:
            continue
        bar = '#' * int(kpi['reach_1_pct'] / 2)
        print(f"  {er_b:8s} {kpi['n']:6d}  {kpi['reach_1_pct']:6.1f}%  {kpi['reach_2_pct']:6.1f}%  "
              f"{kpi['cond_2|1']:4.0f}%  {kpi['med_move_all']:6.3f}  {fmt_mr1(kpi):>9s}  "
              f"{fmt_mt1(kpi):>7s}  {fmt_bw(kpi):>6s}  {bar}")

# ================================================================
# ANALYSE 6: TAGESZEIT
# ================================================================
print()
print("=" * 130)
print("ANALYSE 6: SZENARIEN x TAGESZEIT")
print("=" * 130)

time_labels = ['1_open30', '2_30_60', '3_1_2h', '4_mittag', '5_nachm']

for name, gap_dir, level, swing_dir, desc in key_fades:
    base = df_all[(df_all['gap_direction'] == gap_dir) & 
                  (df_all['crossed_level'] == level) & 
                  (df_all['direction'] == swing_dir)]
    if len(base) < 100:
        continue
    
    print(f"\n  {desc}")
    print(f"  {'Zeit':10s} {'N':>6s}  {'->1Lvl':>7s}  {'->2Lvl':>7s}  {'2|1':>5s}  "
          f"{'MedADR':>7s}  {'MedADR1+':>9s}  {'MedTime':>7s}")
    print(f"  {'-'*10} {'-'*6}  {'-'*7}  {'-'*7}  {'-'*5}  {'-'*7}  {'-'*9}  {'-'*7}")
    
    for tb in time_labels:
        sub = base[base['time_bucket'] == tb]
        kpi = compute_kpis(sub)
        if kpi is None:
            continue
        bar = '#' * int(kpi['reach_1_pct'] / 2)
        print(f"  {tb:10s} {kpi['n']:6d}  {kpi['reach_1_pct']:6.1f}%  {kpi['reach_2_pct']:6.1f}%  "
              f"{kpi['cond_2|1']:4.0f}%  {kpi['med_move_all']:6.3f}  {fmt_mr1(kpi):>9s}  "
              f"{fmt_mt1(kpi):>7s}  {bar}")

# ================================================================
# ANALYSE 7: %ADR USED
# ================================================================
print()
print("=" * 130)
print("ANALYSE 7: SZENARIEN x %ADR USED (wie viel hat sich Aktie schon bewegt)")
print("=" * 130)

adr_used_labels = ['<50%', '50-75%', '75-100%', '100-150%', '>150%']

for name, gap_dir, level, swing_dir, desc in key_fades:
    base = df_all[(df_all['gap_direction'] == gap_dir) & 
                  (df_all['crossed_level'] == level) & 
                  (df_all['direction'] == swing_dir)]
    if len(base) < 100:
        continue
    
    print(f"\n  {desc}")
    print(f"  {'%ADR':8s} {'N':>6s}  {'->1Lvl':>7s}  {'->2Lvl':>7s}  {'2|1':>5s}  "
          f"{'MedADR':>7s}  {'MedADR1+':>9s}  {'MedTime':>7s}")
    print(f"  {'-'*8} {'-'*6}  {'-'*7}  {'-'*7}  {'-'*5}  {'-'*7}  {'-'*9}  {'-'*7}")
    
    for ab in adr_used_labels:
        sub = base[base['adr_used_bucket'] == ab]
        kpi = compute_kpis(sub)
        if kpi is None:
            continue
        bar = '#' * int(kpi['reach_1_pct'] / 2)
        print(f"  {ab:8s} {kpi['n']:6d}  {kpi['reach_1_pct']:6.1f}%  {kpi['reach_2_pct']:6.1f}%  "
              f"{kpi['cond_2|1']:4.0f}%  {kpi['med_move_all']:6.3f}  {fmt_mr1(kpi):>9s}  "
              f"{fmt_mt1(kpi):>7s}  {bar}")

# ================================================================
# ANALYSE 8: GAP-ADR
# ================================================================
print()
print("=" * 130)
print("ANALYSE 8: SZENARIEN x GAP-ADR")
print("=" * 130)

gap_labels = ['<0.75', '0.75-1', '1-1.5', '1.5-2', '>2']

for name, gap_dir, level, swing_dir, desc in key_fades:
    base = df_all[(df_all['gap_direction'] == gap_dir) & 
                  (df_all['crossed_level'] == level) & 
                  (df_all['direction'] == swing_dir)]
    if len(base) < 100:
        continue
    
    print(f"\n  {desc}")
    print(f"  {'GapADR':8s} {'N':>6s}  {'->1Lvl':>7s}  {'->2Lvl':>7s}  {'2|1':>5s}  "
          f"{'MedADR':>7s}  {'MedADR1+':>9s}  {'BandW':>6s}")
    print(f"  {'-'*8} {'-'*6}  {'-'*7}  {'-'*7}  {'-'*5}  {'-'*7}  {'-'*9}  {'-'*6}")
    
    for gb in gap_labels:
        sub = base[base['gap_adr_bucket'] == gb]
        kpi = compute_kpis(sub)
        if kpi is None:
            continue
        bar = '#' * int(kpi['reach_1_pct'] / 2)
        print(f"  {gb:8s} {kpi['n']:6d}  {kpi['reach_1_pct']:6.1f}%  {kpi['reach_2_pct']:6.1f}%  "
              f"{kpi['cond_2|1']:4.0f}%  {kpi['med_move_all']:6.3f}  {fmt_mr1(kpi):>9s}  "
              f"{fmt_bw(kpi):>6s}  {bar}")

# ================================================================
# ANALYSE 9: RVOL
# ================================================================
print()
print("=" * 130)
print("ANALYSE 9: SZENARIEN x RVOL")
print("=" * 130)

rvol_labels = ['<1', '1-1.5', '1.5-2.5', '2.5-5', '>5']

for name, gap_dir, level, swing_dir, desc in key_fades:
    base = df_all[(df_all['gap_direction'] == gap_dir) & 
                  (df_all['crossed_level'] == level) & 
                  (df_all['direction'] == swing_dir)]
    if len(base) < 100:
        continue
    
    print(f"\n  {desc}")
    print(f"  {'RVOL':8s} {'N':>6s}  {'->1Lvl':>7s}  {'->2Lvl':>7s}  {'2|1':>5s}  "
          f"{'MedADR':>7s}  {'MedADR1+':>9s}")
    print(f"  {'-'*8} {'-'*6}  {'-'*7}  {'-'*7}  {'-'*5}  {'-'*7}  {'-'*9}")
    
    for rb in rvol_labels:
        sub = base[base['rvol_bucket'] == rb]
        kpi = compute_kpis(sub)
        if kpi is None:
            continue
        bar = '#' * int(kpi['reach_1_pct'] / 2)
        print(f"  {rb:8s} {kpi['n']:6d}  {kpi['reach_1_pct']:6.1f}%  {kpi['reach_2_pct']:6.1f}%  "
              f"{kpi['cond_2|1']:4.0f}%  {kpi['med_move_all']:6.3f}  {fmt_mr1(kpi):>9s}  {bar}")

# ================================================================
# ANALYSE 10: DER ULTIMATIVE VORFILTER-VERGLEICH
#   ALLE vs ADR>=2% vs ADR>=3% jeweils + ER + Tageszeit
# ================================================================
print()
print("=" * 130)
print("ANALYSE 10: KOMBINATIONS-FILTER (ADR% + Early Range + Tageszeit)")
print("  Der entscheidende Vergleich")
print("=" * 130)

combo_filters = [
    ('ALLE',                          lambda df: df),
    ('ADR>=2%',                       lambda df: df[df['adr_pct'] >= 2]),
    ('ADR>=3%',                       lambda df: df[df['adr_pct'] >= 3]),
    ('ADR>=5%',                       lambda df: df[df['adr_pct'] >= 5]),
    ('ADR>=3% + ER15>=50%',           lambda df: df[(df['adr_pct'] >= 3) & (df['early_range_15'] >= 0.50)]),
    ('ADR>=3% + ER30>=50%',           lambda df: df[(df['adr_pct'] >= 3) & (df['early_range_30'] >= 0.50)]),
    ('ADR>=3% + ER30>=75%',           lambda df: df[(df['adr_pct'] >= 3) & (df['early_range_30'] >= 0.75)]),
    ('ADR>=3% + Open30',              lambda df: df[(df['adr_pct'] >= 3) & (df['time_bucket'] == '1_open30')]),
    ('ADR>=3% + 30-60min',            lambda df: df[(df['adr_pct'] >= 3) & (df['time_bucket'] == '2_30_60')]),
    ('ADR>=3%+ER30>=50%+Open30',      lambda df: df[(df['adr_pct'] >= 3) & (df['early_range_30'] >= 0.50) & (df['time_bucket'] == '1_open30')]),
    ('ADR>=3%+ER30>=50%+30-60',       lambda df: df[(df['adr_pct'] >= 3) & (df['early_range_30'] >= 0.50) & (df['time_bucket'] == '2_30_60')]),
    ('ADR>=3%+ER30>=75%+Open30',      lambda df: df[(df['adr_pct'] >= 3) & (df['early_range_30'] >= 0.75) & (df['time_bucket'] == '1_open30')]),
    ('ADR>=5%+ER30>=50%',             lambda df: df[(df['adr_pct'] >= 5) & (df['early_range_30'] >= 0.50)]),
    ('ADR>=5%+ER30>=50%+Open30',      lambda df: df[(df['adr_pct'] >= 5) & (df['early_range_30'] >= 0.50) & (df['time_bucket'] == '1_open30')]),
    ('ADR>=5%+ER30>=75%+Open30',      lambda df: df[(df['adr_pct'] >= 5) & (df['early_range_30'] >= 0.75) & (df['time_bucket'] == '1_open30')]),
    ('ADR>=3%+<50%ADRused',           lambda df: df[(df['adr_pct'] >= 3) & (df['pct_adr_used'] < 0.50)]),
    ('ADR>=3%+ER30>=50%+<50%ADR',     lambda df: df[(df['adr_pct'] >= 3) & (df['early_range_30'] >= 0.50) & (df['pct_adr_used'] < 0.50)]),
    ('ADR>=3%+ER30>=50%+<75%ADR',     lambda df: df[(df['adr_pct'] >= 3) & (df['early_range_30'] >= 0.50) & (df['pct_adr_used'] < 0.75)]),
]

for name, gap_dir, level, swing_dir, desc in key_fades:
    base = df_all[(df_all['gap_direction'] == gap_dir) & 
                  (df_all['crossed_level'] == level) & 
                  (df_all['direction'] == swing_dir)]
    if len(base) < 200:
        continue
    
    print(f"\n{'=' * 130}")
    base_kpi = compute_kpis(base)
    print(f"  {desc}")
    if base_kpi:
        print(f"  GESAMT: n={base_kpi['n']:,}  ->1Lvl={base_kpi['reach_1_pct']:.1f}%  "
              f"->2Lvl={base_kpi['reach_2_pct']:.1f}%")
    print(f"{'=' * 130}")
    
    print(f"  {'Kombi':35s} {'N':>7s}  {'->1Lvl':>7s}  {'->2Lvl':>7s}  {'->3Lvl':>7s}  "
          f"{'2|1':>5s}  {'MedADR':>7s}  {'MedADR1+':>9s}  {'MedTime':>7s}  {'BandW':>6s}")
    print(f"  {'-'*35} {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*5}  {'-'*7}  {'-'*9}  {'-'*7}  {'-'*6}")
    
    for f_name, f_func in combo_filters:
        sub = f_func(base)
        kpi = compute_kpis(sub)
        if kpi is None:
            continue
        bar = '#' * int(kpi['reach_1_pct'] / 2)
        print(f"  {f_name:35s} {kpi['n']:7d}  {kpi['reach_1_pct']:6.1f}%  {kpi['reach_2_pct']:6.1f}%  "
              f"{kpi['reach_3_pct']:6.1f}%  {kpi['cond_2|1']:4.0f}%  {kpi['med_move_all']:6.3f}  "
              f"{fmt_mr1(kpi):>9s}  {fmt_mt1(kpi):>7s}  {fmt_bw(kpi):>6s}  {bar}")

# ================================================================
# ANALYSE 11: 2D HEATMAP — ADR% x EARLY RANGE 15min
# ================================================================
print()
print("=" * 130)
print("ANALYSE 11: 2D-MATRIX — ADR% x EARLY RANGE 15min (->1Lvl%)")
print("=" * 130)

for name, gap_dir, level, swing_dir, desc in key_fades:
    base = df_all[(df_all['gap_direction'] == gap_dir) & 
                  (df_all['crossed_level'] == level) & 
                  (df_all['direction'] == swing_dir)]
    if len(base) < 200:
        continue
    
    print(f"\n  {desc}")
    print(f"  {'':8s}", end='')
    for er_b in er15_labels:
        print(f"  {er_b:>14s}", end='')
    print()
    print(f"  {'-'*8}", end='')
    for _ in er15_labels:
        print(f"  {'-'*14}", end='')
    print()
    
    for adr_b in adr_pct_labels:
        print(f"  {adr_b:8s}", end='')
        for er_b in er15_labels:
            sub = base[(base['adr_pct_bucket'] == adr_b) & 
                       (base['er15_bucket'] == er_b)]
            kpi = compute_kpis(sub)
            if kpi is None:
                print(f"  {'---':>14s}", end='')
            else:
                val = kpi['reach_1_pct']
                if val >= 35: marker = '** '
                elif val >= 25: marker = '*  '
                elif val >= 20: marker = '+  '
                elif val >= 15: marker = 'o  '
                else: marker = '.  '
                print(f"  {val:5.1f}%{marker}n={kpi['n']:>3d}", end='')
        print()

# ================================================================
# ANALYSE 12: 2D HEATMAP — ADR% x RVOL
# ================================================================
print()
print("=" * 130)
print("ANALYSE 12: 2D-MATRIX — ADR% x RVOL (->1Lvl%)")
print("=" * 130)

for name, gap_dir, level, swing_dir, desc in key_fades:
    base = df_all[(df_all['gap_direction'] == gap_dir) & 
                  (df_all['crossed_level'] == level) & 
                  (df_all['direction'] == swing_dir)]
    if len(base) < 200:
        continue
    
    print(f"\n  {desc}")
    print(f"  {'':8s}", end='')
    for rb in rvol_labels:
        print(f"  {rb:>14s}", end='')
    print()
    print(f"  {'-'*8}", end='')
    for _ in rvol_labels:
        print(f"  {'-'*14}", end='')
    print()
    
    for adr_b in adr_pct_labels:
        print(f"  {adr_b:8s}", end='')
        for rb in rvol_labels:
            sub = base[(base['adr_pct_bucket'] == adr_b) & 
                       (base['rvol_bucket'] == rb)]
            kpi = compute_kpis(sub)
            if kpi is None:
                print(f"  {'---':>14s}", end='')
            else:
                val = kpi['reach_1_pct']
                if val >= 35: marker = '** '
                elif val >= 25: marker = '*  '
                elif val >= 20: marker = '+  '
                elif val >= 15: marker = 'o  '
                else: marker = '.  '
                print(f"  {val:5.1f}%{marker}n={kpi['n']:>3d}", end='')
        print()

# ================================================================
# ANALYSE 13: 3D-MATRIX — ADR% x ER15 x GAP-ADR (Top 25)
# ================================================================
print()
print("=" * 130)
print("ANALYSE 13: 3D-MATRIX — ADR% x EARLY RANGE 15min x GAP-ADR")
print("  Top 25 Kombis pro Szenario")
print("=" * 130)

for name, gap_dir, level, swing_dir, desc in key_fades:
    base = df_all[(df_all['gap_direction'] == gap_dir) & 
                  (df_all['crossed_level'] == level) & 
                  (df_all['direction'] == swing_dir)]
    if len(base) < 200:
        continue
    
    base_kpi = compute_kpis(base)
    print(f"\n{'=' * 130}")
    print(f"  {desc}")
    if base_kpi:
        print(f"  GESAMT: n={base_kpi['n']:,}  ->1Lvl={base_kpi['reach_1_pct']:.1f}%")
    print(f"{'=' * 130}")
    
    print(f"  {'ADR%':6s} {'ER15':8s} {'GapADR':8s} {'N':>5s}  {'->1Lvl':>7s}  {'->2Lvl':>7s}  "
          f"{'2|1':>5s}  {'MedADR1+':>9s}  {'MedTime':>7s}  {'BandW':>6s}")
    print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*5}  {'-'*7}  {'-'*7}  {'-'*5}  {'-'*9}  {'-'*7}  {'-'*6}")
    
    results = []
    for adr_b, er_b, gap_b in product(adr_pct_labels, er15_labels, gap_labels):
        sub = base[(base['adr_pct_bucket'] == adr_b) & 
                   (base['er15_bucket'] == er_b) &
                   (base['gap_adr_bucket'] == gap_b)]
        kpi = compute_kpis(sub)
        if kpi is None:
            continue
        results.append((adr_b, er_b, gap_b, kpi))
    
    results.sort(key=lambda x: x[3]['reach_1_pct'], reverse=True)
    
    for adr_b, er_b, gap_b, kpi in results[:25]:
        bar = '#' * int(kpi['reach_1_pct'] / 2.5)
        print(f"  {adr_b:6s} {er_b:8s} {gap_b:8s} {kpi['n']:5d}  {kpi['reach_1_pct']:6.1f}%  "
              f"{kpi['reach_2_pct']:6.1f}%  {kpi['cond_2|1']:4.0f}%  {fmt_mr1(kpi):>9s}  "
              f"{fmt_mt1(kpi):>7s}  {fmt_bw(kpi):>6s}  {bar}")
    
    if len(results) > 25:
        print(f"\n  ... ({len(results)} Kombis total)")
        print(f"\n  Schlechteste 5:")
        for adr_b, er_b, gap_b, kpi in results[-5:]:
            print(f"  {adr_b:6s} {er_b:8s} {gap_b:8s} {kpi['n']:5d}  {kpi['reach_1_pct']:6.1f}%  "
                  f"{kpi['reach_2_pct']:6.1f}%  {kpi['cond_2|1']:4.0f}%  {fmt_mr1(kpi):>9s}  {fmt_bw(kpi):>6s}")

# ================================================================
# ANALYSE 14: CONDITIONAL CASCADING mit ADR% Vorfilter
# ================================================================
print()
print("=" * 130)
print("ANALYSE 14: CONDITIONAL CASCADING (mit ADR>=3% Vorfilter)")
print("  Wenn Level 1 erreicht: Wie oft geht es zu Level 2, 3?")
print("=" * 130)

for name, gap_dir, level, swing_dir, desc in key_fades:
    base = df_all[(df_all['gap_direction'] == gap_dir) & 
                  (df_all['crossed_level'] == level) & 
                  (df_all['direction'] == swing_dir) &
                  (df_all['adr_pct'] >= 3)]
    if len(base) < 100:
        continue
    
    print(f"\n  {desc} [ADR>=3%]")
    
    # Nach Gap-ADR
    print(f"  {'GapADR':8s} {'N':>6s}  {'->1Lvl':>7s}  {'2|1':>5s}  {'3|2':>5s}  {'MedADR1+':>9s}")
    print(f"  {'-'*8} {'-'*6}  {'-'*7}  {'-'*5}  {'-'*5}  {'-'*9}")
    
    for gb in gap_labels:
        sub = base[base['gap_adr_bucket'] == gb]
        kpi = compute_kpis(sub)
        if kpi is None:
            continue
        print(f"  {gb:8s} {kpi['n']:6d}  {kpi['reach_1_pct']:6.1f}%  {kpi['cond_2|1']:4.0f}%  "
              f"{kpi['cond_3|2']:4.0f}%  {fmt_mr1(kpi):>9s}")
    
    # Nach ER30
    print(f"\n  {'ER30':8s} {'N':>6s}  {'->1Lvl':>7s}  {'2|1':>5s}  {'3|2':>5s}  {'MedADR1+':>9s}")
    print(f"  {'-'*8} {'-'*6}  {'-'*7}  {'-'*5}  {'-'*5}  {'-'*9}")
    
    for er_b in er30_labels:
        sub = base[base['er30_bucket'] == er_b]
        kpi = compute_kpis(sub)
        if kpi is None:
            continue
        print(f"  {er_b:8s} {kpi['n']:6d}  {kpi['reach_1_pct']:6.1f}%  {kpi['cond_2|1']:4.0f}%  "
              f"{kpi['cond_3|2']:4.0f}%  {fmt_mr1(kpi):>9s}")

print()
print("=" * 130)
print("FERTIG.")
print("=" * 130)
