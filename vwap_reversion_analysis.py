###############################################################################
# GAPPER REVERSION ANALYSE
#
# 4 Szenarien klar getrennt nach Gap-Richtung:
#   A) Gap Up  → Erster Pullback zum VWAP (normal)
#   B) Gap Up  → Durch VWAP gefallen, Recovery-Versuch (gescheiterter Gap)
#   C) Gap Down → Erster Bounce zum VWAP (normal)
#   D) Gap Down → Durch VWAP gestiegen, fällt zurück (gescheiterter Gap)
#
# Für jedes Szenario: Wohin geht der Preis danach?
# Segmentiert nach: Gap-ADR, RVOL, %ADR zum Zeitpunkt, Tageszeit
#
# Ausführen: cd gapper-analysis
#            .\gapper_env\Scripts\python.exe vwap_reversion_analysis.py
###############################################################################

import pandas as pd
import numpy as np
import glob
import os
from tqdm import tqdm
from collections import defaultdict

# ============================================================
# CONFIG
# ============================================================
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[0]

VWAP_DIR = str(PROJECT_ROOT / 'data' / 'vwap')
METADATA_PATH = str(PROJECT_ROOT / 'data' / 'metadata' / 'metadata_master.parquet')

# Erst nach X Minuten VWAP-Events zählen (Bänder brauchen Zeit)
MIN_BARS_AFTER_OPEN = 20
# Mindestens X Minuten Rest um Outcome zu messen
MIN_BARS_REMAINING = 10

# ============================================================
# METADATA LADEN
# ============================================================
print("Lade Metadata...")
meta = pd.read_parquet(METADATA_PATH)
meta['key'] = meta['ticker'] + '_' + meta['date']

# Gap-ADR Buckets
meta['gap_adr_bucket'] = pd.cut(
    meta['gap_size_in_adr'].abs(),
    bins=[0, 0.75, 1.0, 1.5, 2.0, 100],
    labels=['<0.75_ADR', '0.75-1_ADR', '1-1.5_ADR', '1.5-2_ADR', '>2_ADR']
)

# RVOL Buckets (feiner als vorher)
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
# EVENT DETECTION — 4 klare Szenarien
# ============================================================

def analyze_gapper_day(df, gap_direction, adr_10):
    """
    Analysiert einen Gapper-Tag und erkennt VWAP-Events.
    
    Gibt Liste von Events zurück, jeweils mit:
    - scenario: A/B/C/D
    - time: Zeitpunkt des Events
    - pct_adr_used: Wie viel % vom ADR wurde bis zu diesem Zeitpunkt verbraucht
    - outcome: Was passiert danach
    - bars_to_outcome: Wie viele Minuten bis Outcome
    - max_favorable_adr: Max Bewegung in "gewünschter" Richtung (in ADR)
    - max_adverse_adr: Max Bewegung gegen "gewünschte" Richtung (in ADR)
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
    times = df['time_et'].values
    
    n = len(closes)
    open_price = closes[0]  # First bar close as proxy for open
    
    # Running high/low für %ADR Berechnung
    running_high = np.maximum.accumulate(closes)
    running_low = np.minimum.accumulate(closes)
    
    # %ADR verbraucht bis zu jedem Zeitpunkt
    if adr_10 > 0:
        pct_adr = (running_high - running_low) / adr_10
    else:
        pct_adr = np.zeros(n)
    
    # Track ob wir auf der "richtigen" Seite des VWAP sind
    # Gap Up: richtig = über VWAP, Gap Down: richtig = unter VWAP
    
    # Track welche Events schon einmal aufgetreten sind pro Tag
    # (nur erstes Event pro Typ zählen, oder alle? → alle, aber markieren)
    event_count_per_type = defaultdict(int)
    
    for i in range(MIN_BARS_AFTER_OPEN, n - MIN_BARS_REMAINING):
        prev_close = closes[i - 1]
        curr_close = closes[i]
        prev_vwap = vwaps[i - 1]
        curr_vwap = vwaps[i]
        curr_upper1 = upper1[i]
        curr_lower1 = lower1[i]
        
        scenario = None
        
        # ──────────────────────────────────────────────────────
        # GAP UP Szenarien
        # ──────────────────────────────────────────────────────
        if gap_direction == 'up':
            
            # Szenario A: Gap Up, Preis fällt zum VWAP (Pullback)
            # = close war über VWAP, jetzt bei oder unter VWAP
            if prev_close > prev_vwap and curr_close <= curr_vwap:
                scenario = 'A_gapup_pullback_to_vwap'
                
                # Outcome: Bounced er zurück zu +1 StdDev oder fällt zu -1 StdDev?
                outcome, bars, max_fav, max_adv = _track_outcome_after_vwap_touch(
                    closes, vwaps, upper1, lower1, upper2, lower2, i, n,
                    favorable_direction='up', adr_10=adr_10
                )
            
            # Szenario B: Gap Up, Preis war UNTER VWAP, kommt zurück hoch
            # = close war unter VWAP, jetzt bei oder über VWAP
            elif prev_close < prev_vwap and curr_close >= curr_vwap:
                scenario = 'B_gapup_recovery_through_vwap'
                
                # Outcome: Hält er über VWAP (+1 StdDev) oder fällt er wieder (-1 StdDev)?
                outcome, bars, max_fav, max_adv = _track_outcome_after_vwap_touch(
                    closes, vwaps, upper1, lower1, upper2, lower2, i, n,
                    favorable_direction='up', adr_10=adr_10
                )
        
        # ──────────────────────────────────────────────────────
        # GAP DOWN Szenarien
        # ──────────────────────────────────────────────────────
        elif gap_direction == 'down':
            
            # Szenario C: Gap Down, Preis steigt zum VWAP (Bounce)
            # = close war unter VWAP, jetzt bei oder über VWAP
            if prev_close < prev_vwap and curr_close >= curr_vwap:
                scenario = 'C_gapdown_bounce_to_vwap'
                
                # Outcome: Wird er am VWAP abgelehnt (-1 StdDev) oder bricht durch (+1 StdDev)?
                outcome, bars, max_fav, max_adv = _track_outcome_after_vwap_touch(
                    closes, vwaps, upper1, lower1, upper2, lower2, i, n,
                    favorable_direction='down', adr_10=adr_10
                )
            
            # Szenario D: Gap Down, Preis war ÜBER VWAP, fällt zurück
            # = close war über VWAP, jetzt bei oder unter VWAP
            elif prev_close > prev_vwap and curr_close <= curr_vwap:
                scenario = 'D_gapdown_failure_through_vwap'
                
                # Outcome: Fällt er weiter (-1 StdDev) oder erholt sich (+1 StdDev)?
                outcome, bars, max_fav, max_adv = _track_outcome_after_vwap_touch(
                    closes, vwaps, upper1, lower1, upper2, lower2, i, n,
                    favorable_direction='down', adr_10=adr_10
                )
        
        if scenario is not None:
            event_count_per_type[scenario] += 1
            events.append({
                'scenario': scenario,
                'event_num': event_count_per_type[scenario],
                'time': times[i],
                'bar_index': i,
                'pct_adr_used': round(pct_adr[i], 2),
                'price_vs_open_adr': round((curr_close - open_price) / adr_10, 2) if adr_10 > 0 else 0,
                'outcome': outcome,
                'bars_to_outcome': bars,
                'max_favorable_adr': round(max_fav, 2),
                'max_adverse_adr': round(max_adv, 2),
            })
    
    return events


def _track_outcome_after_vwap_touch(closes, vwaps, upper1, lower1, upper2, lower2, 
                                      start_idx, n, favorable_direction, adr_10):
    """
    Nach einem VWAP Touch: Was wird ZUERST erreicht?
    
    Outcomes:
    - 'reached_+1std': Preis erreicht +1 StdDev
    - 'reached_-1std': Preis erreicht -1 StdDev
    - 'reached_+2std': Preis erreicht +2 StdDev
    - 'reached_-2std': Preis erreicht -2 StdDev
    - 'stayed_at_vwap': Preis bleibt zwischen ±1 StdDev
    
    Zusätzlich: Max favorable/adverse excursion in ADR
    """
    entry_price = closes[start_idx]
    max_favorable = 0.0
    max_adverse = 0.0
    
    for j in range(start_idx + 1, n):
        price = closes[j]
        
        # Track excursion
        if adr_10 > 0:
            if favorable_direction == 'up':
                fav = (price - entry_price) / adr_10
                adv = (entry_price - price) / adr_10
            else:
                fav = (entry_price - price) / adr_10
                adv = (price - entry_price) / adr_10
            max_favorable = max(max_favorable, fav)
            max_adverse = max(max_adverse, adv)
        
        # Check outcomes
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
        
        events = analyze_gapper_day(df, gap_dir, adr_10)
        
        for ev in events:
            ev['ticker'] = ticker
            ev['date'] = date
            ev['gap_direction'] = gap_dir
            ev['gap_adr_bucket'] = info['gap_adr_bucket']
            ev['rvol_bucket'] = info['rvol_bucket']
            ev['gap_size_adr'] = info['gap_size_in_adr']
        
        all_events.extend(events)
    except Exception:
        continue

events_df = pd.DataFrame(all_events)
print(f"\n{len(events_df):,} Events erkannt")

# Zeit-Buckets
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

# %ADR Buckets
events_df['adr_used_bucket'] = pd.cut(
    events_df['pct_adr_used'],
    bins=[0, 0.5, 0.75, 1.0, 1.5, 100],
    labels=['<50%_ADR', '50-75%_ADR', '75-100%_ADR', '100-150%_ADR', '>150%_ADR']
)

# Nur erste Events pro Tag? Oder alle?
# Erstmal alle, aber wir können auch filtern
events_first = events_df[events_df['event_num'] == 1].copy()

# ============================================================
# HELPER: Print Outcome Distribution
# ============================================================
def print_outcomes(sub, label="", indent=4):
    n = len(sub)
    if n == 0:
        return
    prefix = " " * indent
    outcomes = sub['outcome'].value_counts()
    for outcome, count in outcomes.items():
        pct = count / n * 100
        bar = '█' * int(pct / 2)
        median_bars = sub[sub['outcome'] == outcome]['bars_to_outcome'].median()
        median_fav = sub[sub['outcome'] == outcome]['max_favorable_adr'].median()
        median_adv = sub[sub['outcome'] == outcome]['max_adverse_adr'].median()
        print(f"{prefix}{outcome:22s}  {count:5d} ({pct:5.1f}%)  "
              f"~{median_bars:3.0f}min  MaxFav:{median_fav:+.2f} MaxAdv:{median_adv:+.2f}  {bar}")


# ################################################################
#                     ERGEBNISSE
# ################################################################

scenarios = {
    'A_gapup_pullback_to_vwap': {
        'title': 'SZENARIO A: Gap Up → Pullback zum VWAP',
        'desc': 'Aktie gapped hoch, fällt zum VWAP. Bounced sie zurück (+1 StdDev) oder fällt weiter (-1 StdDev)?',
        'good': 'reached_+1std',
        'bad': 'reached_-1std',
    },
    'B_gapup_recovery_through_vwap': {
        'title': 'SZENARIO B: Gap Up → War unter VWAP, erholt sich zurück über VWAP',
        'desc': 'Gap schon gescheitert, Aktie versucht Recovery. Schafft sie +1 StdDev oder fällt wieder (-1 StdDev)?',
        'good': 'reached_+1std',
        'bad': 'reached_-1std',
    },
    'C_gapdown_bounce_to_vwap': {
        'title': 'SZENARIO C: Gap Down → Bounce zum VWAP',
        'desc': 'Aktie gapped runter, steigt zum VWAP. Wird sie abgelehnt (-1 StdDev) oder bricht durch (+1 StdDev)?',
        'good': 'reached_-1std',
        'bad': 'reached_+1std',
    },
    'D_gapdown_failure_through_vwap': {
        'title': 'SZENARIO D: Gap Down → War über VWAP, fällt zurück unter VWAP',
        'desc': 'Bounce gescheitert, Aktie fällt zurück. Fällt sie weiter (-1 StdDev) oder erholt sich (+1 StdDev)?',
        'good': 'reached_-1std',
        'bad': 'reached_+1std',
    },
}

# ============================================================
# ANALYSE 1: ÜBERSICHT — Alle Szenarien
# ============================================================
print()
print("=" * 80)
print("ANALYSE 1: ÜBERSICHT ALLER 4 SZENARIEN")
print("=" * 80)

for scen_key, scen_info in scenarios.items():
    sub = events_df[events_df['scenario'] == scen_key]
    first = events_first[events_first['scenario'] == scen_key]
    n = len(sub)
    n_first = len(first)
    
    print(f"\n{'━' * 80}")
    print(f"  {scen_info['title']}")
    print(f"  {scen_info['desc']}")
    print(f"{'━' * 80}")
    
    print(f"\n  ALLE Events (n={n:,}):")
    print_outcomes(sub)
    
    print(f"\n  Nur ERSTE Events pro Tag (n={n_first:,}):")
    print_outcomes(first)

# ============================================================
# ANALYSE 2: NACH GAP-ADR SEGMENTIERT
# ============================================================
print()
print("=" * 80)
print("ANALYSE 2: NACH GAP-GRÖßE (ADR)")
print("  → Nur ERSTE Events pro Tag")
print("=" * 80)

for scen_key, scen_info in scenarios.items():
    print(f"\n{'━' * 80}")
    print(f"  {scen_info['title']}")
    print(f"{'━' * 80}")
    
    for gap_b in ['<0.75_ADR', '0.75-1_ADR', '1-1.5_ADR', '1.5-2_ADR', '>2_ADR']:
        sub = events_first[(events_first['scenario'] == scen_key) & 
                           (events_first['gap_adr_bucket'] == gap_b)]
        n = len(sub)
        if n < 15:
            continue
        print(f"\n  Gap: {gap_b} (n={n})")
        print_outcomes(sub)

# ============================================================
# ANALYSE 3: NACH RVOL SEGMENTIERT
# ============================================================
print()
print("=" * 80)
print("ANALYSE 3: NACH RVOL")
print("  → Nur ERSTE Events pro Tag")
print("=" * 80)

for scen_key, scen_info in scenarios.items():
    print(f"\n{'━' * 80}")
    print(f"  {scen_info['title']}")
    print(f"{'━' * 80}")
    
    for rvol_b in ['RVOL<1', 'RVOL_1-1.5', 'RVOL_1.5-2.5', 'RVOL_2.5-5', 'RVOL>5']:
        sub = events_first[(events_first['scenario'] == scen_key) & 
                           (events_first['rvol_bucket'] == rvol_b)]
        n = len(sub)
        if n < 15:
            continue
        print(f"\n  {rvol_b} (n={n})")
        print_outcomes(sub)

# ============================================================
# ANALYSE 4: NACH %ADR VERBRAUCHT
# ============================================================
print()
print("=" * 80)
print("ANALYSE 4: WIE VIEL %ADR IST SCHON VERBRAUCHT?")
print("  → Alle Events (nicht nur erste)")
print("  Frage: Macht es einen Unterschied ob die Aktie schon 50% oder 150% ADR")
print("  gemacht hat bevor sie VWAP berührt?")
print("=" * 80)

for scen_key, scen_info in scenarios.items():
    print(f"\n{'━' * 80}")
    print(f"  {scen_info['title']}")
    print(f"{'━' * 80}")
    
    for adr_b in ['<50%_ADR', '50-75%_ADR', '75-100%_ADR', '100-150%_ADR', '>150%_ADR']:
        sub = events_df[(events_df['scenario'] == scen_key) & 
                        (events_df['adr_used_bucket'] == adr_b)]
        n = len(sub)
        if n < 20:
            continue
        print(f"\n  %ADR verbraucht: {adr_b} (n={n:,})")
        print_outcomes(sub)

# ============================================================
# ANALYSE 5: NACH TAGESZEIT
# ============================================================
print()
print("=" * 80)
print("ANALYSE 5: NACH TAGESZEIT")
print("  → Alle Events")
print("=" * 80)

for scen_key, scen_info in scenarios.items():
    print(f"\n{'━' * 80}")
    print(f"  {scen_info['title']}")
    print(f"{'━' * 80}")
    
    for tb in ['1_open_30min', '2_30_60min', '3_1_2h', '4_mittag', '5_nachmittag']:
        sub = events_df[(events_df['scenario'] == scen_key) & 
                        (events_df['time_bucket'] == tb)]
        n = len(sub)
        if n < 20:
            continue
        print(f"\n  {tb} (n={n:,})")
        print_outcomes(sub)

# ============================================================
# ANALYSE 6: BESTE KOMBINATIONEN — Gap-ADR × RVOL
# ============================================================
print()
print("=" * 80)
print("ANALYSE 6: GAP-ADR × RVOL KOMBINATIONEN")
print("  → Nur ERSTE Events pro Tag")
print("  → Sortiert nach Win-Rate (Szenario-abhängig)")
print("=" * 80)

for scen_key, scen_info in scenarios.items():
    good_outcome = scen_info['good']
    
    print(f"\n{'━' * 80}")
    print(f"  {scen_info['title']}")
    print(f"  'Gut' = {good_outcome}")
    print(f"{'━' * 80}")
    
    results = []
    for gap_b in ['<0.75_ADR', '0.75-1_ADR', '1-1.5_ADR', '1.5-2_ADR', '>2_ADR']:
        for rvol_b in ['RVOL<1', 'RVOL_1-1.5', 'RVOL_1.5-2.5', 'RVOL_2.5-5', 'RVOL>5']:
            sub = events_first[(events_first['scenario'] == scen_key) & 
                               (events_first['gap_adr_bucket'] == gap_b) &
                               (events_first['rvol_bucket'] == rvol_b)]
            n = len(sub)
            if n < 10:
                continue
            
            good_n = (sub['outcome'] == good_outcome).sum()
            good_pct = good_n / n * 100
            
            med_fav = sub['max_favorable_adr'].median()
            med_adv = sub['max_adverse_adr'].median()
            med_bars = sub[sub['outcome'] == good_outcome]['bars_to_outcome'].median() if good_n > 0 else 0
            
            results.append({
                'gap': gap_b,
                'rvol': rvol_b,
                'n': n,
                'win_pct': good_pct,
                'med_favorable': med_fav,
                'med_adverse': med_adv,
                'med_time': med_bars,
            })
    
    if results:
        res = pd.DataFrame(results).sort_values('win_pct', ascending=False)
        print(f"\n  {'Gap':15s} {'RVOL':15s} {'N':>5s}  {'Win%':>6s}  {'MedFav':>7s}  {'MedAdv':>7s}  {'~Min':>5s}")
        print(f"  {'─'*15} {'─'*15} {'─'*5}  {'─'*6}  {'─'*7}  {'─'*7}  {'─'*5}")
        for _, r in res.iterrows():
            print(f"  {r['gap']:15s} {r['rvol']:15s} {r['n']:5.0f}  "
                  f"{r['win_pct']:5.1f}%  {r['med_favorable']:+6.2f}  {r['med_adverse']:+6.2f}  "
                  f"~{r['med_time']:3.0f}")

# ============================================================
# ANALYSE 7: TAKE PROFIT ANALYSE
# ============================================================
print()
print("=" * 80)
print("ANALYSE 7: TAKE PROFIT — Max Favorable Excursion nach VWAP Touch")
print("  → Wie weit läuft der Preis in die günstige Richtung (in ADR)?")
print("  → Nur ERSTE Events pro Tag")
print("=" * 80)

for scen_key, scen_info in scenarios.items():
    good_outcome = scen_info['good']
    sub = events_first[events_first['scenario'] == scen_key]
    
    if len(sub) < 20:
        continue
    
    print(f"\n{'━' * 80}")
    print(f"  {scen_info['title']}")
    print(f"{'━' * 80}")
    
    # Winners only
    winners = sub[sub['outcome'] == good_outcome]
    losers = sub[sub['outcome'] != good_outcome]
    
    if len(winners) > 0:
        print(f"\n  WINNERS ({good_outcome}, n={len(winners)}):")
        print(f"    Max Favorable Excursion (ADR):")
        for pct in [25, 50, 75, 90]:
            val = winners['max_favorable_adr'].quantile(pct/100)
            print(f"      {pct}. Perzentil:  {val:+.2f} ADR")
        print(f"    Time to Outcome (Minuten):")
        for pct in [25, 50, 75, 90]:
            val = winners['bars_to_outcome'].quantile(pct/100)
            print(f"      {pct}. Perzentil:  {val:.0f} min")
    
    if len(losers) > 0:
        print(f"\n  LOSERS (n={len(losers)}):")
        print(f"    Max Adverse Excursion (ADR):")
        for pct in [25, 50, 75, 90]:
            val = losers['max_adverse_adr'].quantile(pct/100)
            print(f"      {pct}. Perzentil:  {val:+.2f} ADR")

# ============================================================
# ANALYSE 8: ZEITBASIERT — Wann passiert der erste VWAP Test?
# ============================================================
print()
print("=" * 80)
print("ANALYSE 8: WANN PASSIERT DER ERSTE VWAP TEST?")
print("  → Verteilung der Uhrzeit des ERSTEN VWAP-Touchs")
print("=" * 80)

for scen_key, scen_info in scenarios.items():
    sub = events_first[events_first['scenario'] == scen_key]
    if len(sub) < 20:
        continue
    
    print(f"\n  {scen_info['title']} (n={len(sub)})")
    
    time_dist = sub['time_bucket'].value_counts().sort_index()
    for tb, count in time_dist.items():
        pct = count / len(sub) * 100
        bar = '█' * int(pct / 2)
        print(f"    {tb:20s}  {count:4d}  ({pct:5.1f}%)  {bar}")
    
    # Median Zeit
    print(f"    Median Uhrzeit: {sub['time'].median()}")

# ============================================================
# ANALYSE 9: PREMARKET BEDINGUNGEN — Was macht gute Setups?
# ============================================================
print()
print("=" * 80)
print("ANALYSE 9: PREMARKET-BEDINGUNGEN FÜR BESTE SETUPS")
print("  → Vergleich: Welche Gap-ADR × RVOL Kombination hat die höchste")
print("    Wahrscheinlichkeit dass der VWAP-Bounce/Pullback hält?")
print("=" * 80)

for scen_key, scen_info in scenarios.items():
    good_outcome = scen_info['good']
    sub = events_first[events_first['scenario'] == scen_key]
    
    if len(sub) < 20:
        continue
    
    print(f"\n{'━' * 80}")
    print(f"  {scen_info['title']}")
    print(f"{'━' * 80}")
    
    # Gap-ADR Effekt
    print(f"\n  Nach Gap-ADR:")
    for gap_b in ['<0.75_ADR', '0.75-1_ADR', '1-1.5_ADR', '1.5-2_ADR', '>2_ADR']:
        s = sub[sub['gap_adr_bucket'] == gap_b]
        if len(s) < 10:
            continue
        win = (s['outcome'] == good_outcome).mean() * 100
        print(f"    {gap_b:15s}  n={len(s):4d}  Win: {win:5.1f}%  "
              f"{'▓' * int(win/2)}{'░' * (25 - int(win/2))}")
    
    # RVOL Effekt
    print(f"\n  Nach RVOL:")
    for rvol_b in ['RVOL<1', 'RVOL_1-1.5', 'RVOL_1.5-2.5', 'RVOL_2.5-5', 'RVOL>5']:
        s = sub[sub['rvol_bucket'] == rvol_b]
        if len(s) < 10:
            continue
        win = (s['outcome'] == good_outcome).mean() * 100
        print(f"    {rvol_b:15s}  n={len(s):4d}  Win: {win:5.1f}%  "
              f"{'▓' * int(win/2)}{'░' * (25 - int(win/2))}")

print()
print("=" * 80)
print("FERTIG. Schick die Ergebnisse zur Interpretation.")
print("=" * 80)
