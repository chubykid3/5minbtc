"""
Kraken WebSocket feed — real-time BTC/USD trades + order book depth.

Drop-in replacement for the original BinanceFeed. Uses Kraken WebSocket v2
which works globally without geo-restrictions.

Maintains the same public interface:
  - tick_buffer: list of (timestamp_ms, price, qty, side) dicts
  - bids/asks: order book snapshot
  - cvd: cumulative volume delta for current window
  - funding_rate: always 0.0 (not available on Kraken spot)
"""

import asyncio
import json
import logging
import time
from collections import deque
from datetime import datetime, timezone

import websockets

from config import (
    KRAKEN_WS_BASE,
    ORDERBOOK_DEPTH,
    MAX_TICK_BUFFER,
)

log = logging.getLogger(__name__)


class BinanceFeed:
    """Kraken-backed feed with the same interface as the original BinanceFeed."""

    def __init__(self):
        self.ticks: deque = deque(maxlen=MAX_TICK_BUFFER)
        self.bids: list = []   # [[price_str, qty_str], ...] sorted desc
        self.asks: list = []   # [[price_str, qty_str], ...] sorted asc
        self.last_price: float = 0.0
        self.last_trade_time: float = 0.0
        self.cvd: float = 0.0
        self.funding_rate: float = 0.0  # Not applicable on Kraken spot
        self._running = False
        self._tasks: list = []

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_bid_ask_imbalance(self, levels: int = 5) -> float:
        """Ratio of bid volume to ask volume at top N levels. >1 = buy pressure."""
        try:
            bid_vol = sum(float(b[1]) for b in self.bids[:levels])
            ask_vol = sum(float(a[1]) for a in self.asks[:levels])
            if ask_vol == 0:
                return 1.0
            return bid_vol / ask_vol
        except Exception:
            return 1.0

    def get_spread_pct(self) -> float:
        """Bid-ask spread as fraction of mid price."""
        try:
            best_bid = float(self.bids[0][0])
            best_ask = float(self.asks[0][0])
            mid = (best_bid + best_ask) / 2
            return (best_ask - best_bid) / mid if mid > 0 else 0.0
        except Exception:
            return 0.0

    def get_wall_distances(self, current_price: float, threshold_ratio: float = 3.0):
        """
        Find nearest significant bid/ask wall.
        Returns (bid_wall_pct_distance, ask_wall_pct_distance).
        """
        bid_wall_dist = 0.05
        ask_wall_dist = 0.05
        try:
            if self.bids:
                sizes = [float(b[1]) for b in self.bids[:10]]
                avg_size = sum(sizes) / len(sizes) if sizes else 1
                for price_str, qty_str in self.bids[:10]:
                    price = float(price_str)
                    qty = float(qty_str)
                    if qty >= threshold_ratio * avg_size:
                        bid_wall_dist = abs(current_price - price) / current_price
                        break
            if self.asks:
                sizes = [float(a[1]) for a in self.asks[:10]]
                avg_size = sum(sizes) / len(sizes) if sizes else 1
                for price_str, qty_str in self.asks[:10]:
                    price = float(price_str)
                    qty = float(qty_str)
                    if qty >= threshold_ratio * avg_size:
                        ask_wall_dist = abs(price - current_price) / current_price
                        break
        except Exception as e:
            log.debug(f"Wall distance error: {e}")
        return bid_wall_dist, ask_wall_dist

    def reset_cvd(self):
        """Call at the start of each 5-min window."""
        self.cvd = 0.0

    def get_recent_ticks(self, since_ms: float) -> list:
        """Return ticks since given timestamp (ms)."""
        return [t for t in self.ticks if t["t"] >= since_ms]

    # ── Message handlers ───────────────────────────────────────────────────────

    async def _handle_trade(self, trades: list):
        for trade in trades:
            try:
                price = float(trade["price"])
                qty   = float(trade["qty"])
                side  = 1 if trade["side"] == "buy" else -1

                ts_str = trade.get("timestamp", "")
                try:
                    dt    = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    ts_ms = dt.timestamp() * 1000
                except Exception:
                    ts_ms = time.time() * 1000

                self.last_price = price
                self.last_trade_time = ts_ms / 1000.0
                self.ticks.append({"t": ts_ms, "p": price, "q": qty, "side": side})
                self.cvd += side * qty
            except Exception as e:
                log.debug(f"Trade handler error: {e}")

    async def _handle_book(self, data: dict, msg_type: str):
        try:
            bids_raw = data.get("bids", [])
            asks_raw = data.get("asks", [])

            if msg_type == "snapshot":
                self.bids = sorted(
                    [[str(b["price"]), str(b["qty"])] for b in bids_raw],
                    key=lambda x: -float(x[0]),
                )[:ORDERBOOK_DEPTH]
                self.asks = sorted(
                    [[str(a["price"]), str(a["qty"])] for a in asks_raw],
                    key=lambda x: float(x[0]),
                )[:ORDERBOOK_DEPTH]
            else:
                for b in bids_raw:
                    ps = str(b["price"])
                    qty = float(b["qty"])
                    self.bids = [x for x in self.bids if x[0] != ps]
                    if qty > 0:
                        self.bids.append([ps, str(qty)])
                    self.bids = sorted(self.bids, key=lambda x: -float(x[0]))[:ORDERBOOK_DEPTH]
                for a in asks_raw:
                    ps = str(a["price"])
                    qty = float(a["qty"])
                    self.asks = [x for x in self.asks if x[0] != ps]
                    if qty > 0:
                        self.asks.append([ps, str(qty)])
                    self.asks = sorted(self.asks, key=lambda x: float(x[0]))[:ORDERBOOK_DEPTH]
        except Exception as e:
            log.debug(f"Book handler error: {e}")

    # ── WebSocket loop ─────────────────────────────────────────────────────────

    async def _stream(self):
        while self._running:
            try:
                async with websockets.connect(KRAKEN_WS_BASE, ping_interval=20) as ws:
                    await ws.send(json.dumps({
                        "method": "subscribe",
                        "params": {"channel": "trade", "symbol": ["BTC/USD"]},
                    }))
                    await ws.send(json.dumps({
                        "method": "subscribe",
                        "params": {
                            "channel": "book",
                            "symbol": ["BTC/USD"],
                            "depth": ORDERBOOK_DEPTH,
                        },
                    }))
                    log.info("Kraken WebSocket connected (BTC/USD)")

                    async for raw in ws:
                        if not self._running:
                            break
                        msg = json.loads(raw)
                        channel  = msg.get("channel", "")
                        msg_type = msg.get("type", "")

                        if channel == "trade" and msg_type in ("snapshot", "update"):
                            await self._handle_trade(msg.get("data", []))
                        elif channel == "book" and msg_type in ("snapshot", "update"):
                            for book_data in msg.get("data", []):
                                await self._handle_book(book_data, msg_type)

            except Exception as e:
                log.warning(f"Kraken WS error: {e}. Reconnecting in 3s...")
                await asyncio.sleep(3)

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def start(self):
        self._running = True
        self._tasks = [asyncio.create_task(self._stream())]
        log.info("KrakenFeed started")

    async def stop(self):
        self._running = False
        for t in self._tasks:
            t.cancel()
        log.info("KrakenFeed stopped")
