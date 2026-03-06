"""
Binance WebSocket feed — real-time BTC/USDT trades + order book depth.

Maintains:
  - tick_buffer: list of (timestamp_ms, price, qty) for current + prior windows
  - orderbook: {bids: [...], asks: [...]} snapshot kept live
  - cvd_series: cumulative volume delta per window
  - funding_rate: latest perpetual funding rate
"""

import asyncio
import json
import logging
import time
from collections import deque
from typing import Optional

import websockets

from config import (
    BINANCE_WS_BASE,
    BINANCE_SYMBOL_LOWER,
    ORDERBOOK_DEPTH,
    MAX_TICK_BUFFER,
    FUNDING_RATE_SYMBOL,
    BINANCE_REST_BASE,
)

log = logging.getLogger(__name__)


class BinanceFeed:
    def __init__(self):
        # Trade ticks: deque of dicts {t, p, q, side}  (t=ms, p=price, q=qty)
        self.ticks: deque = deque(maxlen=MAX_TICK_BUFFER)
        # Order book snapshot
        self.bids: list = []   # [[price, qty], ...] sorted desc
        self.asks: list = []   # [[price, qty], ...] sorted asc
        self.last_price: float = 0.0
        self.last_trade_time: float = 0.0
        # CVD (cumulative volume delta) for current window
        self.cvd: float = 0.0
        self.funding_rate: float = 0.0
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
        A 'wall' is a level with size >= threshold_ratio * average level size.
        Returns (bid_wall_pct_distance, ask_wall_pct_distance).
        """
        bid_wall_dist = 0.05   # default 5% — no wall found
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

    # ── WebSocket handlers ─────────────────────────────────────────────────────

    async def _handle_trade(self, msg: dict):
        """Process an aggTrade message."""
        try:
            price = float(msg["p"])
            qty   = float(msg["q"])
            ts_ms = float(msg["T"])
            is_buyer_maker = msg["m"]   # True = seller initiated (sell)
            side = -1 if is_buyer_maker else 1   # +1 buy, -1 sell

            self.last_price = price
            self.last_trade_time = ts_ms / 1000.0

            self.ticks.append({
                "t": ts_ms,
                "p": price,
                "q": qty,
                "side": side,
            })
            self.cvd += side * qty
        except Exception as e:
            log.debug(f"Trade handler error: {e}")

    async def _handle_depth(self, msg: dict):
        """Process a depth snapshot message."""
        try:
            bids = msg.get("bids", [])
            asks = msg.get("asks", [])
            self.bids = sorted(bids, key=lambda x: -float(x[0]))[:ORDERBOOK_DEPTH]
            self.asks = sorted(asks, key=lambda x:  float(x[0]))[:ORDERBOOK_DEPTH]
        except Exception as e:
            log.debug(f"Depth handler error: {e}")

    async def _stream_trades(self):
        stream_name = f"{BINANCE_SYMBOL_LOWER}@aggTrade"
        url = f"{BINANCE_WS_BASE}?streams={stream_name}"
        while self._running:
            try:
                async with websockets.connect(url, ping_interval=20) as ws:
                    log.info("Binance trade stream connected")
                    async for raw in ws:
                        if not self._running:
                            break
                        data = json.loads(raw)
                        msg = data.get("data", data)
                        await self._handle_trade(msg)
            except Exception as e:
                log.warning(f"Binance trade WS error: {e}. Reconnecting in 3s…")
                await asyncio.sleep(3)

    async def _stream_depth(self):
        stream_name = f"{BINANCE_SYMBOL_LOWER}@depth{ORDERBOOK_DEPTH}@100ms"
        url = f"{BINANCE_WS_BASE}?streams={stream_name}"
        while self._running:
            try:
                async with websockets.connect(url, ping_interval=20) as ws:
                    log.info("Binance depth stream connected")
                    async for raw in ws:
                        if not self._running:
                            break
                        data = json.loads(raw)
                        msg = data.get("data", data)
                        await self._handle_depth(msg)
            except Exception as e:
                log.warning(f"Binance depth WS error: {e}. Reconnecting in 3s…")
                await asyncio.sleep(3)

    async def _poll_funding_rate(self):
        """Poll perpetual funding rate every 60 seconds via REST."""
        import aiohttp
        url = f"{BINANCE_REST_BASE}/fapi/v1/fundingRate"
        while self._running:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        url,
                        params={"symbol": FUNDING_RATE_SYMBOL, "limit": 1},
                        timeout=aiohttp.ClientTimeout(total=5),
                    ) as resp:
                        data = await resp.json()
                        if data and isinstance(data, list):
                            self.funding_rate = float(data[-1].get("fundingRate", 0))
            except Exception as e:
                log.debug(f"Funding rate poll error: {e}")
            await asyncio.sleep(60)

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def start(self):
        self._running = True
        self._tasks = [
            asyncio.create_task(self._stream_trades()),
            asyncio.create_task(self._stream_depth()),
            asyncio.create_task(self._poll_funding_rate()),
        ]
        log.info("BinanceFeed started")

    async def stop(self):
        self._running = False
        for t in self._tasks:
            t.cancel()
        log.info("BinanceFeed stopped")
