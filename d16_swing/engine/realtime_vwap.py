"""
D16 Engine: Streaming VWAP Calculator
======================================
Cumulative volume-weighted typical price. Updates bar-by-bar.
No look-ahead: VWAP at time T uses only bars up to T.
"""


class StreamingVWAP:
    __slots__ = ('cum_vol', 'cum_vp')

    def __init__(self):
        self.cum_vol = 0.0
        self.cum_vp = 0.0  # cumulative (volume * typical_price)

    def update(self, high, low, close, volume):
        """
        Feed one bar and return current VWAP.

        Parameters:
            high, low, close, volume: bar OHLCV fields

        Returns:
            float: current VWAP (or close if cum_vol == 0)
        """
        tp = (high + low + close) / 3.0
        self.cum_vp += tp * volume
        self.cum_vol += volume
        if self.cum_vol > 0:
            return self.cum_vp / self.cum_vol
        return close

    def value(self):
        """Return current VWAP without feeding a new bar."""
        if self.cum_vol > 0:
            return self.cum_vp / self.cum_vol
        return None

    def reset(self):
        self.cum_vol = 0.0
        self.cum_vp = 0.0
