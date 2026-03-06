"""
Polymarket feed — polls current BTC 5-min market implied probabilities.

Uses the Polymarket CLOB REST API to find and track the active 5-minute
BTC market, fetching the current UP/DOWN implied odds every few seconds.
Caches the active market ID so we only need to rediscover it when the
market rolls over to a new window.
"""

import asyncio
import logging
import time
from typing import Optional

import aiohttp

from config import (
    POLYMARKET_REST_BASE,
    POLYMARKET_POLL_INTERVAL,
    POLYMARKET_BTC_MARKET_TAG,
)

log = logging.getLogger(__name__)

MARKETS_ENDPOINT   = f"{POLYMARKET_REST_BASE}/markets"
ORDERBOOK_ENDPOINT = f"{POLYMARKET_REST_BASE}/book"
MIDPOINTS_ENDPOINT = f"{POLYMARKET_REST_BASE}/midpoints"


class PolymarketFeed:
    def __init__(self):
        self.implied_up_prob: float    = 0.50   # P(UP) from market
        self.implied_down_prob: float  = 0.50
        self.volume_so_far: float      = 0.0    # total volume this window
        self.prob_history: list        = []     # [(ts, prob_up)] for current window
        self._active_condition_id: Optional[str] = None
        self._up_token_id: Optional[str] = None
        self._down_token_id: Optional[str] = None
        self._last_market_fetch: float   = 0.0
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._connected = False

    # ── Public ─────────────────────────────────────────────────────────────────

    def get_prob_delta(self, lookback_seconds: float = 30.0) -> float:
        """
        Change in implied UP probability over the last `lookback_seconds`.
        Positive = probability rising (crowd becoming more bullish).
        """
        if len(self.prob_history) < 2:
            return 0.0
        now = time.time()
        cutoff = now - lookback_seconds
        old_probs = [p for ts, p in self.prob_history if ts <= cutoff]
        if not old_probs:
            return 0.0
        return self.implied_up_prob - old_probs[-1]

    def reset_window(self):
        """Call at start of each 5-min window."""
        self.prob_history.clear()
        self.volume_so_far = 0.0

    # ── Market discovery ───────────────────────────────────────────────────────

    async def _find_active_btc_market(self, session: aiohttp.ClientSession) -> bool:
        """
        Search Polymarket for the active BTC 5-minute market.
        Returns True if found and token IDs populated.
        """
        try:
            params = {
                "tag":    "btc-price",
                "closed": "false",
                "limit":  "50",
            }
            async with session.get(
                MARKETS_ENDPOINT,
                params=params,
                timeout=aiohttp.ClientTimeout(total=8),
            ) as resp:
                if resp.status != 200:
                    log.debug(f"Polymarket markets API status {resp.status}")
                    return False

                data = await resp.json()
                markets = data if isinstance(data, list) else data.get("data", [])

                for market in markets:
                    slug = (market.get("market_slug", "") or "").lower()
                    question = (market.get("question", "") or "").lower()
                    active = market.get("active", False)
                    closed = market.get("closed", True)

                    is_btc_5min = (
                        ("btc" in slug or "bitcoin" in slug) and
                        ("5-min" in slug or "5 min" in question or
                         "five minute" in question or "5minute" in slug)
                        and active and not closed
                    )

                    if is_btc_5min:
                        tokens = market.get("tokens", [])
                        for token in tokens:
                            outcome = (token.get("outcome", "") or "").lower()
                            if "up" in outcome or "higher" in outcome or "yes" in outcome:
                                self._up_token_id = token.get("token_id")
                            elif "down" in outcome or "lower" in outcome or "no" in outcome:
                                self._down_token_id = token.get("token_id")

                        self._active_condition_id = market.get("condition_id")
                        self._last_market_fetch = time.time()
                        log.info(
                            f"Polymarket BTC 5-min market found: "
                            f"{market.get('question', 'N/A')[:60]}"
                        )
                        return True

        except aiohttp.ClientError as e:
            log.debug(f"Polymarket market discovery error: {e}")
        except Exception as e:
            log.debug(f"Polymarket parse error: {e}")

        return False

    # ── Probability fetching ───────────────────────────────────────────────────

    async def _fetch_probabilities(self, session: aiohttp.ClientSession):
        """Fetch current UP/DOWN probabilities from order book midpoints."""
        try:
            if not self._up_token_id or not self._down_token_id:
                return

            params = {"token_ids": f"{self._up_token_id},{self._down_token_id}"}
            async with session.get(
                MIDPOINTS_ENDPOINT,
                params=params,
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status != 200:
                    return
                data = await resp.json()

                up_price = float(data.get(self._up_token_id, 0.50))
                dn_price = float(data.get(self._down_token_id, 0.50))

                # Normalize so they sum to 1
                total = up_price + dn_price
                if total > 0:
                    self.implied_up_prob   = up_price / total
                    self.implied_down_prob = dn_price / total
                else:
                    self.implied_up_prob   = 0.50
                    self.implied_down_prob = 0.50

                self.prob_history.append((time.time(), self.implied_up_prob))
                self._connected = True

        except Exception as e:
            log.debug(f"Polymarket prob fetch error: {e}")

    # ── Poll loop ──────────────────────────────────────────────────────────────

    async def _poll_loop(self):
        connector = aiohttp.TCPConnector(limit=5)
        async with aiohttp.ClientSession(connector=connector) as session:
            while self._running:
                try:
                    # Rediscover market every 6 minutes (windows expire)
                    if time.time() - self._last_market_fetch > 360 or not self._up_token_id:
                        found = await self._find_active_btc_market(session)
                        if not found:
                            log.debug("Polymarket: no active BTC 5-min market found")
                            await asyncio.sleep(10)
                            continue

                    await self._fetch_probabilities(session)

                except Exception as e:
                    log.debug(f"Polymarket poll error: {e}")

                await asyncio.sleep(POLYMARKET_POLL_INTERVAL)

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def start(self):
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        log.info("PolymarketFeed started")

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
        log.info("PolymarketFeed stopped")
