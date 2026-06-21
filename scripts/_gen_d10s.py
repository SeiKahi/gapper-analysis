# Auto-generated builder for d10s_mfe_trailing.py
import os

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

target = str(PROJECT_ROOT / 'scripts' / 'd10s_mfe_trailing.py')

Q = chr(39)  # single quote
DQ = chr(34)  # double quote
BS = chr(92)  # backslash
NL = chr(10)  # newline
EM = chr(8212)  # em dash
RA = chr(8594)  # right arrow

L = []
a = L.append

# Content will be appended in subsequent steps
a(DQ + DQ + DQ)
a('D10s: MFE + Trailing-Stop Simulation mit 1-Sekunden-Aufloesung')
a('Hybrid: 1-min Basis + 1-sec Zoom bei ambigen Bars.')
a('Repliziert D10 Aufgaben 1-5 + Synthese.')
a(DQ + DQ + DQ)
a('import pandas as pd')
a('import numpy as np')
a('from pathlib import Path')
a('import sys')
a('import os')
a('from tqdm import tqdm')
a('')
a('RAW_1MIN_DIR = Path(' + Q + 'data/raw_1min' + Q + ')')
a('RAW_1SEC_DIR = Path(' + Q + 'data/raw_1sec' + Q + ')')
a('')
a('_sec_cache = {}')
a('')
a('def load_1sec(ticker, date):')
a('    key = (ticker, date)')
a('    if key not in _sec_cache:')
a('        path = RAW_1SEC_DIR / ticker / f' + DQ + '{date}.parquet' + DQ)
a('        _sec_cache[key] = pd.read_parquet(path) if path.exists() else None')
a('    return _sec_cache[key]')

a('')
a('')
a('# ============================================================')
a('# Hybrid MFE computation')
a('# ============================================================')
a('def compute_mfe_mae_1sec(bars_rth, entry_price, sl_level, trade_dir, adr, ticker, date):')
a('    ' + DQ + DQ + DQ)
a('    Compute MFE_LIVE, MFE_FULL, MAE with 1-sec zoom.')
a('    When SL bar is ambiguous (bar has both favorable excursion and SL hit),')
a('    zoom to 1-sec to get precise MFE at SL point.')
a('    ' + DQ + DQ + DQ)
a('    sl_dist = abs(entry_price - sl_level)')
a('    if sl_dist <= 0 or pd.isna(sl_dist) or adr <= 0:')
a('        return None')
a('')
a('    post_entry = bars_rth[(bars_rth[' + Q + 'time_et' + Q + '] >= ' + Q + '09:36' + Q + ') & (bars_rth[' + Q + 'time_et' + Q + '] <= ' + Q + '15:55' + Q + ')]')
a('    if len(post_entry) == 0:')
a('        return None')
a('')
a('    sec_bars = load_1sec(ticker, date)')
a('')
a('    mfe_full = 0.0')
a('    mae = 0.0')
a('    sl_hit = False')
a('    sl_hit_time = None')
a('    mfe_at_sl = 0.0')
a('    zoom_used = False')
a('    zoom_changed = False')
a('')
a('    for _, bar in post_entry.iterrows():')
a('        bar_time = str(bar[' + Q + 'time_et' + Q + '])[:5]')
a('')
a('        if trade_dir == ' + Q + 'long' + Q + ':')
a('            fav = max(0, bar[' + Q + 'high' + Q + '] - entry_price)')
a('            adv = max(0, entry_price - bar[' + Q + 'low' + Q + '])')
a('        else:')
a('            fav = max(0, entry_price - bar[' + Q + 'low' + Q + '])')
a('            adv = max(0, bar[' + Q + 'high' + Q + '] - entry_price)')
a('')
a('        mfe_full = max(mfe_full, fav)')
a('        mae = max(mae, adv)')
a('')
a('        if not sl_hit:')
