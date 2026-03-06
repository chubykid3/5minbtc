"""
Chainlink BTC/USD Oracle feed via web3.py polling.

Polls the on-chain aggregator every CHAINLINK_POLL_INTERVAL seconds.
Tracks:
  - latest_price: last oracle price
  - latest_updated_at: Unix timestamp of last oracle update
  - update_history: list of (timestamp, price) tuples for current window
  - staleness: seconds since last update
"""

import asyncio
import logging
import time
from collections import deque
from typing import Optional, Tuple

from config import (
    CHAINLINK_BTC_USD_ADDRESS,
    CHAINLINK_ABI,
    ETH_RPC_URLS,
    CHAINLINK_POLL_INTERVAL,
)

log = logging.getLogger(__name__)


class ChainlinkFeed:
    def __init__(self):
        self.latest_price: float = 0.0
        self.latest_updated_at: float = 0.0   # Unix epoch seconds
        self.update_history: deque = deque(maxlen=200)   # (ts, price) for current session
        self._decimals: Optional[int] = None
        self._w3 = None
        self._contract = None
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._connected = False

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def staleness(self) -> float:
        """Seconds since last oracle update."""
        if self.latest_updated_at == 0:
            return 999.0
        return time.time() - self.latest_updated_at

    def updates_in_window(self, window_start_ts: float) -> int:
        """Number of oracle updates since window_start_ts."""
        return sum(1 for ts, _ in self.update_history if ts >= window_start_ts)

    def get_lag_vs_spot(self, spot_price: float) -> float:
        """
        Chainlink vs spot delta as fraction.
        Positive = spot above oracle (upward resolution pressure).
        """
        if self.latest_price <= 0 or spot_price <= 0:
            return 0.0
        return (spot_price - self.latest_price) / self.latest_price

    # ── Web3 setup ─────────────────────────────────────────────────────────────

    def _init_web3(self) -> bool:
        try:
            from web3 import Web3
            for rpc_url in ETH_RPC_URLS:
                try:
                    w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 5}))
                    if w3.is_connected():
                        self._w3 = w3
                        checksum_addr = Web3.to_checksum_address(CHAINLINK_BTC_USD_ADDRESS)
                        self._contract = w3.eth.contract(
                            address=checksum_addr,
                            abi=CHAINLINK_ABI,
                        )
                        self._decimals = self._contract.functions.decimals().call()
                        log.info(f"Chainlink connected via {rpc_url} (decimals={self._decimals})")
                        self._connected = True
                        return True
                except Exception as e:
                    log.debug(f"RPC {rpc_url} failed: {e}")
            log.warning("All Chainlink RPC endpoints failed — oracle data unavailable")
            return False
        except ImportError:
            log.warning("web3 not installed — Chainlink oracle feed disabled")
            return False

    def _fetch_latest(self) -> Optional[Tuple[float, float]]:
        """Returns (price, updated_at) or None on error."""
        try:
            if self._contract is None:
                return None
            round_data = self._contract.functions.latestRoundData().call()
            # (roundId, answer, startedAt, updatedAt, answeredInRound)
            answer     = round_data[1]
            updated_at = round_data[3]
            price = answer / (10 ** self._decimals)
            return price, float(updated_at)
        except Exception as e:
            log.debug(f"Chainlink fetch error: {e}")
            return None

    # ── Poll loop ──────────────────────────────────────────────────────────────

    async def _poll_loop(self):
        if not self._init_web3():
            return

        while self._running:
            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, self._fetch_latest
                )
                if result is not None:
                    price, updated_at = result
                    now = time.time()

                    if updated_at != self.latest_updated_at:
                        # New oracle update
                        self.latest_price      = price
                        self.latest_updated_at = updated_at
                        self.update_history.append((now, price))
                        log.debug(
                            f"Chainlink update: ${price:,.2f} "
                            f"(chain_ts={updated_at:.0f}, lag={now - updated_at:.1f}s)"
                        )
                    else:
                        # Same round — oracle unchanged
                        pass
            except Exception as e:
                log.debug(f"Chainlink poll loop error: {e}")

            await asyncio.sleep(CHAINLINK_POLL_INTERVAL)

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def start(self):
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        log.info("ChainlinkFeed started")

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
        log.info("ChainlinkFeed stopped")
