"""
Polymarket feed — polls current BTC 5-min market implied probabilities.

Fixes the '50% always' bug by:
  1. Using the Gamma Markets API for broader, more reliable market discovery
  2. Falling back to CLOB API if Gamma returns nothing
  3. Using the CLOB order book (best bid/ask) instead of /midpoints for real prices
  4. Rediscovering the market every 3 minutes (5-min markets expire quickly)
  5. Logging every step of discovery so you can diagnose issues
  6. Exposing token IDs publicly so the trader module can place orders
"""

import asyncio
import logging
import time
from typing import Optional

import aiohttp

from config import (
    POLYMARKET_REST_BASE,
    POLYMARKET_GAMMA_BASE,
    POLYMARKET_POLL_INTERVAL,
)

log = logging.getLogger(__name__)

GAMMA_MARKETS_URL  = f"{POLYMARKET_GAMMA_BASE}/markets"
CLOB_MARKETS_URL   = f"{POLYMARKET_REST_BASE}/markets"
ORDERBOOK_URL      = f"{POLYMARKET_REST_BASE}/book"
MIDPOINTS_URL      = f"{POLYMARKET_REST_BASE}/midpoints"

# Rediscover market every 3 min — 5-min windows expire fast
MARKET_REDISCOVER_INTERVAL = 180


class PolymarketFeed:
    def __init__(self):
        self.implied_up_prob: float   = 0.50   # P(UP) from market
        self.implied_down_prob: float = 0.50
        self.volume_so_far: float     = 0.0
        self.prob_history: list       = []      # [(ts, prob_up)] for current window

        # Public token IDs — used by trader to place orders
        self.up_token_id: Optional[str]   = None
        self.down_token_id: Optional[str] = None
        self.active_condition_id: Optional[str] = None
        self.active_question: str = "—"

        self._last_market_fetch: float = 0.0
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._connected = False
        self._no_market_logged = False   # Rate-limit "no market found" warnings

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

    # ── Market discovery — Gamma API (preferred) ───────────────────────────────

    async def _find_via_gamma(self, session: aiohttp.ClientSession) -> bool:
        """
        Search Polymarket Gamma API for active BTC 5-min market.
        Gamma API has better search/tag support than CLOB API.
        """
        search_strategies = [
            # (params_dict, description)
            ({"tag": "btc-price",  "active": "true", "closed": "false", "limit": "50"},
             "tag=btc-price"),
            ({"tag": "crypto",     "active": "true", "closed": "false", "limit": "100"},
             "tag=crypto"),
            ({"active": "true",    "closed": "false", "limit": "100"},
             "all active"),
        ]

        for params, desc in search_strategies:
            try:
                async with session.get(
                    GAMMA_MARKETS_URL,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status != 200:
                        log.debug(f"Gamma API ({desc}) status {resp.status}")
                        continue

                    data = await resp.json()
                    markets = data if isinstance(data, list) else data.get("data", data.get("markets", []))

                    log.debug(f"Gamma search ({desc}): {len(markets)} markets returned")

                    found = self._parse_markets(markets, source="Gamma")
                    if found:
                        return True

            except aiohttp.ClientError as e:
                log.debug(f"Gamma API error ({desc}): {e}")
            except Exception as e:
                log.debug(f"Gamma parse error ({desc}): {e}")

        return False

    async def _find_via_clob(self, session: aiohttp.ClientSession) -> bool:
        """Fallback: search CLOB API directly."""
        search_strategies = [
            ({"tag": "btc-price", "closed": "false", "limit": "50"}, "tag=btc-price"),
            ({"closed": "false", "limit": "100"}, "all active"),
        ]

        for params, desc in search_strategies:
            try:
                async with session.get(
                    CLOB_MARKETS_URL,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status != 200:
                        log.debug(f"CLOB API ({desc}) status {resp.status}")
                        continue

                    data = await resp.json()
                    markets = data if isinstance(data, list) else data.get("data", [])

                    log.debug(f"CLOB search ({desc}): {len(markets)} markets returned")

                    found = self._parse_markets(markets, source="CLOB")
                    if found:
                        return True

            except Exception as e:
                log.debug(f"CLOB search error ({desc}): {e}")

        return False

    def _parse_markets(self, markets: list, source: str) -> bool:
        """
        Search a list of market objects for an active BTC short-term market.
        Priority order: 5-min > 10-min > 15-min > any BTC direction market.
        """
        btc_5min   = []
        btc_other  = []

        for market in markets:
            slug     = (market.get("market_slug", "") or market.get("slug", "") or "").lower()
            question = (market.get("question", "") or "").lower()
            active   = market.get("active", True)
            closed   = market.get("closed", False)
            archived = market.get("archived", False)

            if closed or archived:
                continue

            is_btc = (
                "btc" in slug or "btc" in question or
                "bitcoin" in slug or "bitcoin" in question
            )
            if not is_btc:
                continue

            is_direction = (
                "higher" in question or "lower" in question or
                "up" in question or "down" in question or
                "above" in question or "below" in question or
                "rise" in question or "fall" in question or
                "increase" in question or "decrease" in question
            )

            is_5min = (
                "5-min" in slug or "5 min" in question or
                "5minute" in slug or "5-minute" in question or
                "five minute" in question or "five-minute" in question or
                "5 minute" in question
            )

            tokens = market.get("tokens", []) or market.get("clobTokenIds", [])

            # Log every BTC market found for diagnosis
            log.debug(
                f"[{source}] BTC market: slug={slug[:40]!r} | "
                f"q={question[:60]!r} | active={active} | "
                f"5min={is_5min} | dir={is_direction} | tokens={len(tokens)}"
            )

            if is_5min and len(tokens) >= 2:
                btc_5min.append((market, tokens))
            elif is_direction and len(tokens) >= 2:
                btc_other.append((market, tokens))

        # Try 5-min first, then fall back to any direction market
        candidates = btc_5min or btc_other
        if not candidates:
            return False

        market, tokens = candidates[0]
        slug     = market.get("market_slug") or market.get("slug", "")
        question = market.get("question", "N/A")

        up_tok = dn_tok = None
        for token in tokens:
            # Handle both Gamma format {"token_id": ..., "outcome": ...}
            # and raw string token ID lists
            if isinstance(token, dict):
                outcome = (token.get("outcome", "") or "").lower()
                tid = token.get("token_id") or token.get("tokenId")
            else:
                # Raw token ID — first = YES/UP, second = NO/DOWN by convention
                tid = str(token)
                outcome = "yes" if tokens.index(token) == 0 else "no"

            if tid:
                if any(k in outcome for k in ("up", "higher", "yes", "rise", "above")):
                    up_tok = tid
                elif any(k in outcome for k in ("down", "lower", "no", "fall", "below")):
                    dn_tok = tid
                # If outcome label is ambiguous, use position
                elif not up_tok:
                    up_tok = tid
                elif not dn_tok:
                    dn_tok = tid

        if not up_tok or not dn_tok:
            log.warning(
                f"[{source}] BTC market found but couldn't assign UP/DOWN tokens. "
                f"Tokens: {tokens}"
            )
            return False

        self.up_token_id          = up_tok
        self.down_token_id        = dn_tok
        self.active_condition_id  = market.get("condition_id") or market.get("conditionId")
        self.active_question      = question[:80]
        self._last_market_fetch   = time.time()
        self._no_market_logged    = False

        log.info(
            f"[{source}] BTC market found: {question[:70]!r} "
            f"| UP_token={up_tok[:12]}... | DN_token={dn_tok[:12]}..."
        )
        return True

    async def _find_active_btc_market(self, session: aiohttp.ClientSession) -> bool:
        """Try Gamma API first, fall back to CLOB API."""
        found = await self._find_via_gamma(session)
        if not found:
            found = await self._find_via_clob(session)
        if not found and not self._no_market_logged:
            log.warning(
                "No active BTC 5-min market found on Polymarket. "
                "Odds will show 0.50 until a market is found. "
                "This is normal between windows or if no 5-min market exists."
            )
            self._no_market_logged = True
        return found

    # ── Price fetching — order book (real prices) ──────────────────────────────

    async def _fetch_probabilities(self, session: aiohttp.ClientSession):
        """
        Fetch real UP/DOWN prices from the CLOB order book.
        Uses best-bid/best-ask mid instead of /midpoints to avoid the 0.5 bug.
        Falls back to /midpoints if order book is empty.
        """
        if not self.up_token_id or not self.down_token_id:
            return

        up_price = await self._book_mid(session, self.up_token_id)
        dn_price = await self._book_mid(session, self.down_token_id)

        # If book is empty for either, try midpoints as fallback
        if up_price <= 0 or dn_price <= 0:
            up_price, dn_price = await self._fetch_midpoints(session)

        if up_price <= 0 and dn_price <= 0:
            log.debug("No price data from order book or midpoints")
            return

        # Normalize
        total = up_price + dn_price
        if total > 0.01:
            self.implied_up_prob   = up_price / total
            self.implied_down_prob = dn_price / total
        else:
            # Both essentially 0 — market has no liquidity, skip update
            log.debug(
                f"No liquidity (up={up_price:.4f} dn={dn_price:.4f}). "
                "Keeping previous odds."
            )
            return

        self.prob_history.append((time.time(), self.implied_up_prob))
        self._connected = True

        log.debug(
            f"Polymarket odds: UP={self.implied_up_prob:.4f} "
            f"DN={self.implied_down_prob:.4f} "
            f"(raw: up={up_price:.4f} dn={dn_price:.4f})"
        )

    async def _book_mid(self, session: aiohttp.ClientSession, token_id: str) -> float:
        """
        Fetch order book for a token and compute best-bid/best-ask mid.
        Returns 0.0 if book is empty or request fails.
        """
        try:
            async with session.get(
                ORDERBOOK_URL,
                params={"token_id": token_id},
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status != 200:
                    return 0.0
                data = await resp.json()

                bids = data.get("bids", [])
                asks = data.get("asks", [])

                best_bid = float(bids[0]["price"]) if bids else 0.0
                best_ask = float(asks[0]["price"]) if asks else 0.0

                if best_bid > 0 and best_ask > 0:
                    return (best_bid + best_ask) / 2.0
                elif best_bid > 0:
                    return best_bid
                elif best_ask > 0:
                    return best_ask
                return 0.0

        except Exception as e:
            log.debug(f"Order book fetch error (token {token_id[:12]}...): {e}")
            return 0.0

    async def _fetch_midpoints(
        self, session: aiohttp.ClientSession
    ) -> tuple[float, float]:
        """Fallback: /midpoints endpoint. Returns (up_price, dn_price)."""
        try:
            params = {"token_ids": f"{self.up_token_id},{self.down_token_id}"}
            async with session.get(
                MIDPOINTS_URL,
                params=params,
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status != 200:
                    return 0.0, 0.0
                data = await resp.json()
                up = float(data.get(self.up_token_id, 0.0))
                dn = float(data.get(self.down_token_id, 0.0))
                return up, dn
        except Exception as e:
            log.debug(f"Midpoints fallback error: {e}")
            return 0.0, 0.0

    # ── Poll loop ──────────────────────────────────────────────────────────────

    async def _poll_loop(self):
        connector = aiohttp.TCPConnector(limit=5)
        async with aiohttp.ClientSession(connector=connector) as session:
            while self._running:
                try:
                    # Rediscover every 3 min — 5-min markets expire quickly
                    age = time.time() - self._last_market_fetch
                    if age > MARKET_REDISCOVER_INTERVAL or not self.up_token_id:
                        found = await self._find_active_btc_market(session)
                        if not found:
                            await asyncio.sleep(15)
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
