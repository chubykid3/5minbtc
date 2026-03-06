"""
CandleStore — maintains rolling 1-minute OHLCV candles built from live ticks.

Each minute boundary, closes the current candle and starts a new one.
Provides get_recent_candles() for feature engineering.
"""

import time
import logging
from collections import deque
from typing import List, Dict, Optional

log = logging.getLogger(__name__)

_MINUTE = 60


class CandleStore:
    def __init__(self, max_candles: int = 120):
        self._candles: deque = deque(maxlen=max_candles)
        self._current: Optional[Dict] = None
        self._current_minute: int = 0

    def _minute_key(self, ts: float) -> int:
        return int(ts) // _MINUTE * _MINUTE

    def push_tick(self, ts: float, price: float, qty: float):
        """Add a trade tick, building and closing 1-min candles as time advances."""
        mk = self._minute_key(ts)

        if self._current is None or mk != self._current_minute:
            # Close old candle
            if self._current is not None:
                self._candles.append(dict(self._current))

            # Open new candle
            self._current_minute = mk
            self._current = {
                "t":      mk * 1000,
                "open":   price,
                "high":   price,
                "low":    price,
                "close":  price,
                "volume": qty,
            }
        else:
            c = self._current
            c["high"]   = max(c["high"], price)
            c["low"]    = min(c["low"],  price)
            c["close"]  = price
            c["volume"] += qty

    def feed_from_binance(self, binance_feed):
        """
        Ingest all buffered ticks from a BinanceFeed into the candle store.
        Called periodically from the main loop.
        """
        for tick in list(binance_feed.ticks):
            self.push_tick(tick["t"] / 1000.0, tick["p"], tick["q"])

    def get_recent_candles(self, lookback: int = 60) -> List[Dict]:
        """
        Return the last `lookback` completed candles (not including the
        currently-forming one).
        """
        candles = list(self._candles)
        return candles[-lookback:]

    def get_all(self) -> List[Dict]:
        return list(self._candles)

    def seed_from_historical(self, candle_dicts: List[Dict]):
        """Load historical candles at startup to warm up indicators."""
        for c in candle_dicts:
            self._candles.append(c)
        log.info(f"CandleStore seeded with {len(candle_dicts)} historical candles")
