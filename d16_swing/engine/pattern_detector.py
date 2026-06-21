"""
D16 Engine: Streaming Pattern Detection
=========================================
State machines for S1 (Trend Continuation) and S2 (Power Breakout).
All checks are streaming: only use data up to the current bar.
"""
import numpy as np


class TrendContinuationDetector:
    """
    S1: Trend Day Continuation
    Checks after 13:00 ET whether price is in top 30% of day range,
    above VWAP, and gap not yet filled.
    """

    def __init__(self):
        self.running_high = -np.inf
        self.running_low = np.inf
        self.triggered = False
        self.entry_price = None
        self.entry_time = None

    def update(self, bar_time, bar_high, bar_low, bar_close, vwap, gap_filled):
        """
        Feed one bar. Returns True on first confirmation bar after 13:00.

        Parameters:
            bar_time: str 'HH:MM'
            bar_high, bar_low, bar_close: bar prices
            vwap: current streaming VWAP
            gap_filled: bool, whether gap has been filled so far

        Returns:
            bool: True if this bar triggers entry
        """
        # Always update running high/low
        self.running_high = max(self.running_high, bar_high)
        self.running_low = min(self.running_low, bar_low)

        if self.triggered:
            return False

        # Only check after 13:00
        if bar_time < '13:00':
            return False

        day_range = self.running_high - self.running_low
        if day_range <= 0:
            return False

        # Price in top 30% of day range
        price_pct = (bar_close - self.running_low) / day_range
        if price_pct < 0.70:  # top 30% means >= 70th percentile
            return False

        # Price above VWAP
        if bar_close <= vwap:
            return False

        # Gap NOT filled (strength signal)
        if gap_filled:
            return False

        self.triggered = True
        self.entry_price = bar_close
        self.entry_time = bar_time
        return True


class PowerBreakoutDetector:
    """
    S2: Power Breakout Hold
    Checks after 11:00 ET whether price makes a new HOD with above-average
    volume and price above VWAP.
    """

    def __init__(self):
        self.running_high = -np.inf
        self.running_low = np.inf
        self.bar_count = 0
        self.cum_vol = 0.0
        self.triggered = False
        self.entry_price = None
        self.entry_time = None
        self.am_low = np.inf  # morning low (pre-11:00)

    def update(self, bar_time, bar_high, bar_low, bar_close, bar_volume, vwap):
        """
        Feed one bar. Returns True on first breakout bar after 11:00.

        Parameters:
            bar_time: str 'HH:MM'
            bar_high, bar_low, bar_close: bar prices
            bar_volume: bar volume
            vwap: current streaming VWAP

        Returns:
            bool: True if this bar triggers entry
        """
        prev_high = self.running_high

        # Update trackers
        self.running_high = max(self.running_high, bar_high)
        self.running_low = min(self.running_low, bar_low)
        self.bar_count += 1
        self.cum_vol += bar_volume

        # Track AM low (before 11:00)
        if bar_time < '11:00':
            self.am_low = min(self.am_low, bar_low)

        if self.triggered:
            return False

        # Only check after 11:00
        if bar_time < '11:00':
            return False

        avg_vol = self.cum_vol / self.bar_count if self.bar_count > 0 else 0

        # New HOD: bar_high exceeds previous running high
        is_new_hod = bar_high > prev_high and prev_high > -np.inf

        # Volume >= 1.5x average
        vol_ok = bar_volume >= 1.5 * avg_vol if avg_vol > 0 else False

        # Price above VWAP
        above_vwap = bar_close > vwap

        if is_new_hod and vol_ok and above_vwap:
            self.triggered = True
            self.entry_price = bar_close
            self.entry_time = bar_time
            return True

        return False
