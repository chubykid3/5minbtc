"""
BTC 5-Minute Polymarket Decision Bot
=====================================

Entry point. Orchestrates all components:
  - Live data feeds (Binance, Chainlink, Polymarket)
  - 5-minute window tracking aligned to UTC
  - Decision at T=150s every window
  - Outcome recording at T=300s
  - Periodic model retraining
  - Rich terminal dashboard

Usage:
  python main.py                    # run live bot
  python main.py --train-only       # fetch data + train, then exit
  python main.py --backtest         # backtest and print results
  python main.py --no-train         # skip initial training (use saved models)
"""

import argparse
import asyncio
import json
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

# ── Logging setup (before any imports that use logging) ───────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
# Quiet noisy libraries
for _lib in ("websockets", "aiohttp", "asyncio", "urllib3", "web3"):
    logging.getLogger(_lib).setLevel(logging.WARNING)

log = logging.getLogger("bot")

# ── Project imports ────────────────────────────────────────────────────────────
from config import (
    WINDOW_SECONDS,
    DECISION_OFFSET,
    RETRAIN_EVERY_HOURS,
    LSTM_RETRAIN_DAYS,
    MIN_TRAINING_SAMPLES,
    DISPLAY_REFRESH,
)
from data_feeds.binance_feed    import BinanceFeed
from data_feeds.chainlink_feed  import ChainlinkFeed
from data_feeds.polymarket_feed import PolymarketFeed
from features.feature_engineer  import FeatureEngineer, feature_vector, FEATURE_NAMES
from models.ensemble            import Ensemble
from training.data_fetcher      import DataFetcher
from training.trainer           import Trainer
from candle_store               import CandleStore
from decision_logger            import DecisionLogger


# ─────────────────────────────────────────────────────────────────────────────
# Window helpers
# ─────────────────────────────────────────────────────────────────────────────

def current_window_start() -> int:
    """Unix timestamp of the start of the current 5-minute window (UTC aligned)."""
    return (int(time.time()) // WINDOW_SECONDS) * WINDOW_SECONDS


def time_in_window() -> float:
    """Seconds elapsed since the current window started."""
    return time.time() - current_window_start()


# ─────────────────────────────────────────────────────────────────────────────
# Bot
# ─────────────────────────────────────────────────────────────────────────────

class DecisionBot:
    def __init__(self, skip_train: bool = False):
        # Feeds
        self.binance    = BinanceFeed()
        self.chainlink  = ChainlinkFeed()
        self.polymarket = PolymarketFeed()

        # Core components
        self.candle_store   = CandleStore(max_candles=200)
        self.ensemble       = Ensemble()
        self.fetcher        = DataFetcher()
        self.trainer        = Trainer(self.ensemble)
        self.feature_eng    = FeatureEngineer(
            self.binance, self.chainlink, self.polymarket, self.candle_store
        )
        self.logger         = DecisionLogger(self.fetcher)

        # State
        self.skip_train     = skip_train
        self._window_history: List[Dict] = []   # past window metadata
        self._pending_decision: Optional[Dict] = None  # current window decision
        self._last_window_start = 0
        self._last_retrain_time = 0.0
        self._decision_made_this_window = False
        self._running = True

        # Reference prices per window
        self._ref_price:   float = 0.0
        self._window_open_chainlink: float = 0.0

    # ── Startup ───────────────────────────────────────────────────────────────

    async def startup(self):
        log.info("=" * 60)
        log.info("  BTC 5-Min Polymarket Decision Bot  STARTING UP")
        log.info("=" * 60)

        # Start live feeds
        await asyncio.gather(
            self.binance.start(),
            self.chainlink.start(),
            self.polymarket.start(),
        )

        # Wait briefly for feeds to connect and get initial prices
        log.info("Waiting for initial price data (10s)...")
        await asyncio.sleep(10)

        if self.binance.last_price > 0:
            log.info(f"Binance price: ${self.binance.last_price:,.2f}")
        else:
            log.warning("No Binance price yet — feeds may be slow to connect")

        # Initial training
        if not self.skip_train:
            log.info("Running initial training (fetching 30 days of data)...")
            try:
                await self.trainer.run(fetch_data=True)
            except Exception as e:
                log.error(f"Training error: {e}. Continuing with untrained models.")
        else:
            log.info("Skipping initial training (using saved models)")

        # Seed candle store with recent historical candles
        recent_candles = self.fetcher.get_candles(lookback_days=1)
        self.candle_store.seed_from_historical(recent_candles)

        # Load recent window history for regime features
        recent_windows = self.fetcher.get_5m_windows(lookback_days=7)
        self._window_history = [
            {
                "window_start": w["window_start"],
                "resolved_up":  w["resolved_up"],
                "delta_pct":    w["delta_pct"],
            }
            for w in recent_windows[-100:]
        ]

        log.info("Bot startup complete. Entering live loop.")
        log.info(f"Models trained: {self.ensemble.any_model_trained}")

    # ── Window management ─────────────────────────────────────────────────────

    def _on_window_open(self, ws: int):
        """Called at the start of each new 5-minute window."""
        current_price = self.binance.last_price

        self._ref_price = current_price
        self._window_open_chainlink = self.chainlink.latest_price
        self._decision_made_this_window = False

        # Reset per-window accumulators
        self.binance.reset_cvd()
        self.polymarket.reset_window()

        win_dt = datetime.utcfromtimestamp(ws).strftime("%H:%M:%S")
        log.info(
            f"NEW WINDOW | {win_dt} UTC | "
            f"Ref price = ${current_price:,.2f} | "
            f"Chainlink = ${self.chainlink.latest_price:,.2f}"
        )

    def _on_decision_point(self, ws: int):
        """Called at T=150s — compute and log the decision."""
        current_price = self.binance.last_price
        if current_price <= 0:
            log.warning("No price data at decision point — defaulting UP")
            current_price = self._ref_price

        # Build feature vector
        try:
            feat_dict = self.feature_eng.compute(
                window_start_ts=ws,
                reference_price=self._ref_price,
                current_price=current_price,
                window_history=self._window_history,
            )
        except Exception as e:
            log.error(f"Feature engineering error: {e}")
            feat_dict = {name: 0.0 for name in FEATURE_NAMES}

        feat_vec = feature_vector(feat_dict)

        # LSTM sequence (last N windows' feature vectors — if we had them stored)
        # For now pass None; the ensemble handles the fallback
        lstm_seq = self._build_lstm_sequence()

        # Ensemble prediction
        try:
            p_up, side, confidence, sub_probas = self.ensemble.predict(
                feature_vec=feat_vec,
                feature_dict=feat_dict,
                lstm_sequence=lstm_seq,
                implied_up_prob=self.polymarket.implied_up_prob,
            )
        except Exception as e:
            log.error(f"Ensemble predict error: {e}. Defaulting UP.")
            p_up, side, confidence = 0.5, "UP", 0.0
            sub_probas = {}

        # Store and log
        self._pending_decision = {
            "window_start":  ws,
            "side":          side,
            "p_up":          p_up,
            "confidence":    confidence,
            "sub_probas":    sub_probas,
            "ref_price":     self._ref_price,
            "price_at_decision": current_price,
            "features":      feat_dict,
        }

        self.logger.log_decision(
            window_start=ws,
            side=side,
            p_up=p_up,
            confidence=confidence,
            sub_probas=sub_probas,
            implied_up=self.polymarket.implied_up_prob,
            features=feat_dict,
        )
        self._decision_made_this_window = True

        # Console announcement
        delta_pct = (current_price - self._ref_price) / self._ref_price * 100 if self._ref_price > 0 else 0
        conf_str  = f"HIGH-CONF" if confidence >= 0.08 else "standard"
        log.info(
            f"*** DECISION: {side} ({conf_str}) ***  "
            f"P(UP)={p_up:.3f} | Δ={delta_pct:+.3f}% | "
            f"Crowd={self.polymarket.implied_up_prob:.3f}"
        )

    def _on_window_close(self, ws: int, close_price: float):
        """Called at T=300 when next window opens — record outcome."""
        ref_price   = self._ref_price
        resolved_up = close_price >= ref_price
        delta_pct   = (close_price - ref_price) / ref_price if ref_price > 0 else 0.0

        predicted_side = None
        if self._pending_decision and self._pending_decision.get("window_start") == ws:
            predicted_side = self._pending_decision["side"]

        self.logger.log_outcome(
            window_start=ws,
            ref_price=ref_price,
            close_price=close_price,
            resolved_up=resolved_up,
            predicted_side=predicted_side,
        )

        # Update window history
        self._window_history.append({
            "window_start": ws,
            "resolved_up":  resolved_up,
            "delta_pct":    delta_pct,
        })
        self._window_history = self._window_history[-100:]

        self._pending_decision = None

    def _build_lstm_sequence(self):
        """Build LSTM input sequence from recent window history metadata."""
        from config import LSTM_SEQUENCE_LENGTH
        if len(self._window_history) < LSTM_SEQUENCE_LENGTH:
            return None
        # Minimal feature: [delta_pct, resolved_up_float] per window
        # In production this would be the full feature vector per window
        # For now return None to use XGBoost as fallback
        return None

    # ── Periodic tasks ────────────────────────────────────────────────────────

    async def _candle_builder_loop(self):
        """Feed ticks from Binance into the candle store continuously."""
        while self._running:
            ticks = list(self.binance.ticks)
            for tick in ticks:
                self.candle_store.push_tick(
                    tick["t"] / 1000.0, tick["p"], tick["q"]
                )
            await asyncio.sleep(5)

    async def _retrain_loop(self):
        """Retrain models every RETRAIN_EVERY_HOURS hours."""
        await asyncio.sleep(RETRAIN_EVERY_HOURS * 3600)  # initial delay
        while self._running:
            log.info("Scheduled model retrain starting...")
            try:
                await self.trainer.run(fetch_data=True)
                self._last_retrain_time = time.time()
                log.info("Model retrain complete.")
            except Exception as e:
                log.error(f"Retrain error: {e}")
            await asyncio.sleep(RETRAIN_EVERY_HOURS * 3600)

    # ── Main loop ─────────────────────────────────────────────────────────────

    async def run(self):
        """Main async loop — manages window lifecycle."""
        await self.startup()

        # Kick off background tasks
        asyncio.create_task(self._candle_builder_loop())
        asyncio.create_task(self._retrain_loop())

        last_display = 0.0
        ws = current_window_start()
        self._last_window_start = ws
        self._on_window_open(ws)

        try:
            while self._running:
                now = time.time()
                ws  = current_window_start()
                tiw = time_in_window()

                # New window opened
                if ws != self._last_window_start:
                    close_price = self.binance.last_price
                    self._on_window_close(self._last_window_start, close_price)
                    self._last_window_start = ws
                    self._on_window_open(ws)

                # Decision point at T=148–152s (±2s tolerance)
                if (
                    148 <= tiw <= 152
                    and not self._decision_made_this_window
                    and self.binance.last_price > 0
                ):
                    self._on_decision_point(ws)

                # Terminal display update
                if now - last_display >= DISPLAY_REFRESH:
                    feeds_status = {
                        "Binance":    self.binance.last_price > 0,
                        "Chainlink":  self.chainlink._connected,
                        "Polymarket": self.polymarket._connected,
                    }
                    self.logger.print_dashboard(
                        current_price=self.binance.last_price,
                        reference_price=self._ref_price,
                        window_start=ws,
                        time_in_window=tiw,
                        feeds_status=feeds_status,
                    )
                    last_display = now

                await asyncio.sleep(0.5)

        except asyncio.CancelledError:
            pass
        except KeyboardInterrupt:
            pass
        finally:
            await self.shutdown()

    async def shutdown(self):
        self._running = False
        log.info("Shutting down feeds...")
        await asyncio.gather(
            self.binance.stop(),
            self.chainlink.stop(),
            self.polymarket.stop(),
            return_exceptions=True,
        )
        stats = self.logger.session_stats()
        log.info(
            f"Session complete. "
            f"Decisions: {stats['total']} | "
            f"Accuracy: {stats['accuracy']:.1%} | "
            f"High-conf accuracy: {stats['high_conf_acc']:.1%}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="BTC 5-Min Polymarket Decision Bot"
    )
    p.add_argument(
        "--train-only",
        action="store_true",
        help="Fetch data and train models, then exit",
    )
    p.add_argument(
        "--backtest",
        action="store_true",
        help="Run backtest on saved data and print results",
    )
    p.add_argument(
        "--no-train",
        action="store_true",
        help="Skip initial training (use saved model files)",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
        help="Log verbosity",
    )
    return p.parse_args()


async def main_async(args):
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    if args.backtest:
        log.info("Running backtest...")
        ensemble = Ensemble()
        trainer  = Trainer(ensemble)
        results  = trainer.backtest(n_windows=500)
        print("\nBacktest Results:")
        for k, v in results.items():
            print(f"  {k:<25} {v}")
        return

    if args.train_only:
        log.info("Train-only mode: fetching data and training...")
        ensemble = Ensemble()
        trainer  = Trainer(ensemble)
        await trainer.run(fetch_data=True)
        log.info("Training complete. Exiting.")
        return

    bot = DecisionBot(skip_train=args.no_train)

    # Handle Ctrl+C gracefully
    loop = asyncio.get_event_loop()

    def _signal_handler():
        log.info("Interrupt received, shutting down...")
        bot._running = False

    try:
        loop.add_signal_handler(signal.SIGINT,  _signal_handler)
        loop.add_signal_handler(signal.SIGTERM, _signal_handler)
    except NotImplementedError:
        # Windows doesn't support add_signal_handler
        pass

    await bot.run()


def main():
    args = parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\nBot stopped.")


if __name__ == "__main__":
    main()
