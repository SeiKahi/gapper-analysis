"""
D14 Engine: Intra-Bar Resolution for Ambiguous 1-min Bars
==========================================================
When a 1-min bar's range spans both the SL and a new peak, we can't know
which happened first from the 1-min bar alone. This module provides:

1. 1-sec data lookup (sparse: ~2140 files in data/raw_1sec/)
2. Conservative fallback when no 1-sec data available
3. Statistics tracking for ambiguity analysis
"""
import pandas as pd
import numpy as np
from pathlib import Path


SEC_DIR = Path('data/raw_1sec')


class IntrabarResolver:
    """Stateful resolver that tracks ambiguity statistics."""

    def __init__(self, use_1sec=True):
        self.use_1sec = use_1sec
        self.stats = {
            'total_bars_processed': 0,
            'ambig_count': 0,
            'zoom_attempted': 0,
            'zoom_data_found': 0,
            'zoom_resolved_sl': 0,
            'zoom_resolved_peak': 0,
            'conservative_used': 0,
        }
        # Cache 1-sec file existence
        self._sec_index = None

    def build_index(self):
        """Build index of available 1-sec files."""
        self._sec_index = set()
        if SEC_DIR.exists():
            for f in SEC_DIR.rglob('*.parquet'):
                ticker = f.parent.name
                date = f.stem
                self._sec_index.add((ticker, date))
        return len(self._sec_index)

    def has_1sec_data(self, ticker, date):
        """Check if 1-sec data exists for this ticker/date."""
        if self._sec_index is None:
            self.build_index()
        return (ticker, str(date)) in self._sec_index

    def load_1sec_minute(self, ticker, date, minute_time):
        """
        Load 1-sec bars for a specific minute.

        Parameters:
            ticker: Stock ticker
            date: Date string 'YYYY-MM-DD'
            minute_time: String 'HH:MM' from 1-min bar

        Returns:
            DataFrame of 1-sec bars within that minute, or None
        """
        if not self.has_1sec_data(ticker, date):
            return None

        sec_path = SEC_DIR / ticker / f"{date}.parquet"
        try:
            df = pd.read_parquet(sec_path)
            # Filter to the specific minute (time_et is "HH:MM:SS")
            sec_bars = df[df['time_et'].str.startswith(minute_time)]
            return sec_bars if len(sec_bars) > 0 else None
        except Exception:
            return None

    def resolve(self, bar, sl_level, entry_price, sl_dist, trade_dir,
                trail_active, highest_r, ticker=None, date=None):
        """
        Resolve an ambiguous bar.

        Returns:
            (action, value) where action is one of:
            - 'sl_hit': value = exit_price (SL was hit first)
            - 'survived': value = new_highest_r (peak first, then SL check after trail update)
            - 'conservative_sl': value = sl_level (no 1-sec data, assume SL first)
        """
        self.stats['ambig_count'] += 1

        if not self.use_1sec or ticker is None or date is None:
            self.stats['conservative_used'] += 1
            return 'conservative_sl', sl_level

        sec_bars = self.load_1sec_minute(ticker, str(date), bar['time_et'])
        self.stats['zoom_attempted'] += 1

        if sec_bars is None:
            self.stats['conservative_used'] += 1
            return 'conservative_sl', sl_level

        self.stats['zoom_data_found'] += 1

        # Replay second by second
        local_highest_r = highest_r
        local_sl = sl_level
        local_trail = trail_active

        for _, sec in sec_bars.iterrows():
            # SL check first
            if trade_dir == 'long' and sec['low'] <= local_sl:
                self.stats['zoom_resolved_sl'] += 1
                return 'sl_hit', local_sl
            elif trade_dir == 'short' and sec['high'] >= local_sl:
                self.stats['zoom_resolved_sl'] += 1
                return 'sl_hit', local_sl

            # Peak update
            if trade_dir == 'long':
                new_r = max(0, (sec['high'] - entry_price) / sl_dist)
            else:
                new_r = max(0, (entry_price - sec['low']) / sl_dist)
            local_highest_r = max(local_highest_r, new_r)

            # Trail activation & update
            if local_highest_r >= 1.0 and not local_trail:
                local_trail = True
                local_sl = entry_price
            if local_trail:
                if trade_dir == 'long':
                    local_sl = max(local_sl, entry_price + (local_highest_r - 0.5) * sl_dist)
                else:
                    local_sl = min(local_sl, entry_price - (local_highest_r - 0.5) * sl_dist)

            # Re-check SL after trail update
            if trade_dir == 'long' and sec['low'] <= local_sl:
                self.stats['zoom_resolved_sl'] += 1
                return 'sl_hit', local_sl
            elif trade_dir == 'short' and sec['high'] >= local_sl:
                self.stats['zoom_resolved_sl'] += 1
                return 'sl_hit', local_sl

        # Survived the whole minute
        self.stats['zoom_resolved_peak'] += 1
        return 'survived', local_highest_r

    def get_stats(self):
        """Return copy of ambiguity statistics."""
        return dict(self.stats)

    def format_stats(self):
        """Format stats for logging."""
        s = self.stats
        lines = [
            "Intra-Bar Ambiguity Statistics:",
            f"  Total bars processed: {s['total_bars_processed']}",
            f"  Ambiguous bars: {s['ambig_count']}",
            f"  1-sec zoom attempted: {s['zoom_attempted']}",
            f"  1-sec data found: {s['zoom_data_found']}",
            f"  Resolved as SL hit: {s['zoom_resolved_sl']}",
            f"  Resolved as peak first: {s['zoom_resolved_peak']}",
            f"  Conservative fallback: {s['conservative_used']}",
        ]
        if s['ambig_count'] > 0:
            pct = s['ambig_count'] / max(s['total_bars_processed'], 1) * 100
            lines.append(f"  Ambiguity rate: {pct:.2f}%")
        return '\n'.join(lines)
