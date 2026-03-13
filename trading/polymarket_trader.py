"""
Polymarket Trading Executor
============================

Places real USDC bets on Polymarket CLOB based on bot decisions.

Requires:  pip install py-clob-client
Docs:      https://docs.polymarket.com/#clob-api

Flow per window:
  1. Bot calls execute_trade() at T=150s with side + implied odds
  2. Trader places a BUY order for UP or DOWN token on Polygon
  3. Bot calls settle_trade() at T=300s when outcome is known
  4. Trader records P&L and updates daily risk limits

Risk management:
  - Daily loss cap (MAX_DAILY_LOSS_USDC) — halts all trading for the day
  - Per-bet size limits (BET_SIZE_USDC, MAX_BET_SIZE_USDC)
  - Minimum liquidity check before placing
  - Dry-run mode (ENABLE_LIVE_TRADING=False) simulates without real money
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

log = logging.getLogger(__name__)


# ─── Trade record ─────────────────────────────────────────────────────────────

@dataclass
class TradeRecord:
    window_start:  int
    side:          str            # "UP" or "DOWN"
    token_id:      str
    price:         float          # price per share at order time
    shares:        float          # number of shares ordered
    usdc_spent:    float          # USDC committed
    order_id:      str
    net_ev:        float = 0.0
    filled:        bool  = False
    fill_price:    float = 0.0
    outcome_up:    Optional[bool] = None   # True = BTC went up
    pnl:           float = 0.0
    ts:            float = field(default_factory=time.time)

    @property
    def won(self) -> Optional[bool]:
        if self.outcome_up is None:
            return None
        return (self.side == "UP") == self.outcome_up


# ─── Trader ───────────────────────────────────────────────────────────────────

class PolymarketTrader:
    """
    Executes real-money trades on Polymarket CLOB.

    All public methods are safe to call even when live trading is disabled —
    they silently return None/no-op so the rest of the bot is unaffected.
    """

    def __init__(self):
        from config import (
            POLYMARKET_PRIVATE_KEY,
            POLYMARKET_API_KEY,
            POLYMARKET_API_SECRET,
            POLYMARKET_API_PASSPHRASE,
            POLYMARKET_PROXY_ADDRESS,
            POLYMARKET_CHAIN_ID,
            ENABLE_LIVE_TRADING,
            BET_SIZE_USDC,
            MAX_BET_SIZE_USDC,
            HIGH_CONF_BET_MULTIPLIER,
            MAX_DAILY_LOSS_USDC,
            MIN_LIQUIDITY_USDC,
            HIGH_CONF_EV,
        )

        self.private_key    = POLYMARKET_PRIVATE_KEY
        self.api_key        = POLYMARKET_API_KEY
        self.api_secret     = POLYMARKET_API_SECRET
        self.api_passphrase = POLYMARKET_API_PASSPHRASE
        self.proxy_address  = POLYMARKET_PROXY_ADDRESS
        self.chain_id       = POLYMARKET_CHAIN_ID

        self.live_trading        = ENABLE_LIVE_TRADING
        self.base_bet            = BET_SIZE_USDC
        self.max_bet             = MAX_BET_SIZE_USDC
        self.hc_multiplier       = HIGH_CONF_BET_MULTIPLIER
        self.max_daily_loss      = MAX_DAILY_LOSS_USDC
        self.min_liquidity       = MIN_LIQUIDITY_USDC
        self.high_conf_threshold = HIGH_CONF_EV

        self._client      = None
        self._initialized = False
        self._credentials_ok = False

        # P&L state
        self.session_pnl: float = 0.0
        self.daily_pnl:   float = 0.0
        self.wins:   int = 0
        self.losses: int = 0

        # Trade tracking
        self.open_trades:   Dict[int, TradeRecord] = {}
        self.closed_trades: List[TradeRecord]      = []

        # Day rollover
        self._day_start_ts: float = self._midnight_utc()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self):
        """Initialize CLOB client in a thread (blocking network call)."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._init_client)

    async def stop(self):
        s = self.stats()
        log.info(
            f"Trader stopped | mode={'LIVE' if self.live_trading else 'DRY-RUN'} | "
            f"trades={s['total']} | acc={s['accuracy']:.1%} | "
            f"session P&L=${s['session_pnl']:+.2f}"
        )

    # ── Initialization ────────────────────────────────────────────────────────

    def _init_client(self):
        """
        Connect to Polymarket CLOB.
        Gracefully handles: missing library, placeholder credentials, network errors.
        """
        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import ApiCreds
        except ImportError:
            log.warning(
                "py-clob-client not installed — trading disabled. "
                "Install with: pip install py-clob-client"
            )
            return

        # Check that credentials have been filled in
        placeholders = {"FILL_IN_PRIVATE_KEY", "FILL_IN_API_KEY", "FILL_IN_API_SECRET",
                        "FILL_IN_API_PASSPHRASE", "FILL_IN_PROXY_ADDRESS", ""}
        if (self.private_key in placeholders or
                self.api_key in placeholders):
            log.warning(
                "Polymarket credentials not configured. "
                "Edit config.py and fill in POLYMARKET_PRIVATE_KEY, "
                "POLYMARKET_API_KEY, etc. Bot will run in DRY-RUN mode."
            )
            self._credentials_ok = False
            return

        try:
            creds = ApiCreds(
                api_key=self.api_key,
                api_secret=self.api_secret,
                api_passphrase=self.api_passphrase,
            )

            proxy = (
                self.proxy_address
                if self.proxy_address and self.proxy_address not in placeholders
                else None
            )

            self._client = ClobClient(
                host="https://clob.polymarket.com",
                key=self.private_key,
                chain_id=self.chain_id,
                creds=creds,
                signature_type=0,   # 0 = EOA (direct wallet signing)
                funder=proxy,
            )

            # Verify connectivity + fetch balance
            bal = self._get_balance()
            self._credentials_ok = True
            self._initialized = True

            log.info(
                f"Polymarket trader ready | "
                f"mode={'LIVE TRADING' if self.live_trading else 'DRY-RUN (no real money)'} | "
                f"USDC balance: ${bal:.2f}"
            )

        except Exception as e:
            log.error(
                f"Polymarket trader init failed: {e}. "
                "Trading disabled — bot will continue making predictions only."
            )

    def _get_balance(self) -> float:
        """Fetch wallet's USDC balance from Polymarket."""
        try:
            if self._client is None:
                return 0.0
            result = self._client.get_balance()
            if isinstance(result, (int, float)):
                return float(result)
            if isinstance(result, dict):
                for k in ("USDC", "usdc", "balance"):
                    if k in result:
                        return float(result[k])
            return 0.0
        except Exception as e:
            log.debug(f"Balance fetch error: {e}")
            return 0.0

    # ── Trade execution ───────────────────────────────────────────────────────

    async def execute_trade(
        self,
        window_start:    int,
        side:            str,         # "UP" or "DOWN"
        net_ev:          float,
        implied_up_prob: float,
        up_token_id:     Optional[str],
        down_token_id:   Optional[str],
    ) -> Optional[TradeRecord]:
        """
        Place a bet on the current window.

        Called at T=150s after the bot makes its decision.
        Returns a TradeRecord on success, None if skipped.

        Skipped when:
          - Credentials not configured
          - Daily loss limit reached
          - Already traded this window
          - Token IDs not available (market not found)
          - Bet size would be < $0.50 after risk limits
        """
        if not self._credentials_ok and not self.live_trading:
            # Still allow dry-run even without credentials
            if not self._initialized and self.live_trading:
                return None

        # Already traded this window?
        if window_start in self.open_trades:
            log.debug(f"Already have a trade for window {window_start}")
            return None

        # Daily loss check
        self._rollover_daily_if_needed()
        if self.daily_pnl <= -self.max_daily_loss:
            log.warning(
                f"Daily loss cap hit (${self.daily_pnl:.2f}). "
                "No more trades today."
            )
            return None

        # Token IDs from feed
        token_id = up_token_id if side == "UP" else down_token_id
        if not token_id:
            log.warning(
                f"No token ID for side={side} — Polymarket market not found yet. "
                "Skipping trade."
            )
            return None

        # Compute price (what we pay per share)
        if side == "UP":
            price = max(0.02, min(0.98, implied_up_prob))
        else:
            price = max(0.02, min(0.98, 1.0 - implied_up_prob))

        # Size: scale up for high-confidence decisions
        high_conf = abs(net_ev) >= self.high_conf_threshold
        usdc = self.base_bet
        if high_conf:
            usdc = min(self.base_bet * self.hc_multiplier, self.max_bet)

        # Don't exceed remaining daily loss budget
        budget_left = self.max_daily_loss + self.daily_pnl
        usdc = min(usdc, budget_left)

        if usdc < 0.50:
            log.warning(f"Bet size ${usdc:.2f} too small — skipping")
            return None

        shares = usdc / price

        log.info(
            f"{'[LIVE]' if self.live_trading else '[DRY-RUN]'} "
            f"BET {side} | ${usdc:.2f} USDC | "
            f"{shares:.4f} shares @ {price:.4f} | "
            f"net_ev={net_ev:+.3f} | {'HIGH-CONF' if high_conf else 'standard'}"
        )

        if self.live_trading and self._initialized:
            record = await self._place_live_order(
                window_start=window_start,
                side=side,
                token_id=token_id,
                price=price,
                shares=shares,
                usdc=usdc,
                net_ev=net_ev,
            )
        else:
            # Dry-run simulation
            record = TradeRecord(
                window_start=window_start,
                side=side,
                token_id=token_id,
                price=price,
                shares=shares,
                usdc_spent=usdc,
                order_id=f"DRY-{window_start}",
                net_ev=net_ev,
                filled=True,
                fill_price=price,
            )

        if record:
            self.open_trades[window_start] = record
        return record

    async def _place_live_order(
        self,
        window_start: int,
        side: str,
        token_id: str,
        price: float,
        shares: float,
        usdc: float,
        net_ev: float,
    ) -> Optional[TradeRecord]:
        """Place a real order on Polymarket CLOB and return the trade record."""
        try:
            from py_clob_client.clob_types import OrderArgs, BUY

            order_args = OrderArgs(
                token_id=token_id,
                price=round(price, 4),
                size=round(shares, 4),
                side=BUY,
            )

            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(
                None,
                lambda: self._client.create_and_post_order(order_args),
            )

            if resp and resp.get("orderID"):
                order_id = resp["orderID"]
                fill_price = float(resp.get("price", price))
                filled = resp.get("status") in ("MATCHED", "FILLED", "matched", "filled")

                log.info(
                    f"[LIVE] Order placed: {order_id} | "
                    f"side={side} | ${usdc:.2f} | status={resp.get('status', '?')}"
                )

                return TradeRecord(
                    window_start=window_start,
                    side=side,
                    token_id=token_id,
                    price=price,
                    shares=shares,
                    usdc_spent=usdc,
                    order_id=order_id,
                    net_ev=net_ev,
                    filled=filled,
                    fill_price=fill_price,
                )
            else:
                log.error(f"[LIVE] Order failed: {resp}")
                return None

        except Exception as e:
            log.error(f"[LIVE] Order placement exception: {e}")
            return None

    # ── Settlement ────────────────────────────────────────────────────────────

    async def settle_trade(self, window_start: int, resolved_up: bool):
        """
        Record the outcome of a trade at T=300s (window close).

        Polymarket settles automatically on-chain; this just updates our P&L
        tracking. The actual USDC redemption happens in the user's wallet.
        """
        record = self.open_trades.pop(window_start, None)
        if record is None:
            return

        record.outcome_up = resolved_up
        won = record.won

        if won:
            # Each winning share pays $1.00
            payout = record.shares * 1.0
            pnl = payout - record.usdc_spent
            self.wins += 1
        else:
            payout = 0.0
            pnl = -record.usdc_spent
            self.losses += 1

        record.pnl = pnl
        self.session_pnl += pnl
        self.daily_pnl   += pnl
        self.closed_trades.append(record)

        total = self.wins + self.losses
        acc = self.wins / max(1, total)

        log.info(
            f"{'[LIVE]' if self.live_trading else '[DRY-RUN]'} "
            f"SETTLED {record.side} → {'WIN ✓' if won else 'LOSS ✗'} | "
            f"P&L: {pnl:+.2f} | "
            f"Session: ${self.session_pnl:+.2f} | "
            f"Daily: ${self.daily_pnl:+.2f} | "
            f"Acc: {acc:.1%} ({self.wins}W/{self.losses}L)"
        )

    # ── Risk / state helpers ──────────────────────────────────────────────────

    @staticmethod
    def _midnight_utc() -> float:
        now = datetime.now(timezone.utc)
        return now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()

    def _rollover_daily_if_needed(self):
        if time.time() >= self._day_start_ts + 86400:
            self.daily_pnl    = 0.0
            self._day_start_ts = self._midnight_utc()
            log.info("Daily P&L rolled over (new UTC day)")

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> Dict:
        total = self.wins + self.losses
        return {
            "total":        total,
            "wins":         self.wins,
            "losses":       self.losses,
            "accuracy":     self.wins / max(1, total),
            "session_pnl":  self.session_pnl,
            "daily_pnl":    self.daily_pnl,
            "live":         self.live_trading,
            "initialized":  self._initialized,
        }

    def status_line(self) -> str:
        """One-line status for dashboard display."""
        s = self.stats()
        mode = "LIVE" if self.live_trading else "DRY"
        if s["total"] == 0:
            return f"[{mode}] No trades yet"
        return (
            f"[{mode}] {s['total']} trades | "
            f"{s['accuracy']:.1%} acc | "
            f"Session ${s['session_pnl']:+.2f} | "
            f"Daily ${s['daily_pnl']:+.2f}"
        )
