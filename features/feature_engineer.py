"""
Feature Engineering — builds the complete feature vector at T=150s.

Called once per 5-minute window at the decision point. Takes live data
from all feeds and produces a flat dict suitable for model inference.
"""

import math
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from config import (
    RSI_PERIOD, EMA_SHORT, EMA_LONG,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    BB_PERIOD, BB_STD, ATR_PERIOD, OBV_LOOKBACK,
    BID_ASK_LEVELS, WALL_THRESHOLD_RATIO,
    ASIA_SESSION_START, ASIA_SESSION_END,
    EU_SESSION_START, EU_SESSION_END,
    US_SESSION_START, US_SESSION_END,
    RECENT_WINDOWS_LOOKBACK,
)
from features.technical_indicators import (
    rsi as calc_rsi, ema_latest, macd as calc_macd,
    bollinger_bands, bollinger_position, atr as calc_atr,
    obv_slope, price_velocity,
)

log = logging.getLogger(__name__)

FEATURE_NAMES = [
    # Price displacement
    "delta_pct",            # % move from T=0 reference
    "delta_sign",           # +1 if positive, -1 if negative
    "abs_delta_pct",        # magnitude
    # Momentum
    "momentum_30s",         # price change last 30s
    "momentum_60s",         # price change last 60s (velocity)
    "price_velocity_60s",   # linear slope over last 60s (frac/s)
    "distance_to_zero",     # how far reference price is (abs_delta_pct again but kept separate)
    # Technical indicators (on 1-min candles)
    "rsi_14",
    "ema_cross",            # EMA(9) - EMA(21) normalized
    "macd_hist",            # MACD histogram normalized by price
    "bb_position",          # 0=lower band, 1=upper band
    "atr_pct",              # ATR as % of price (volatility regime)
    "obv_slope",            # OBV slope normalized
    # Order book
    "bid_ask_imbalance",    # bid_vol / ask_vol at top 5 levels
    "bid_wall_dist",        # nearest bid wall distance (frac of price)
    "ask_wall_dist",        # nearest ask wall distance
    "spread_pct",           # bid-ask spread as fraction
    # Oracle
    "chainlink_vs_spot",    # (spot - oracle) / spot — positive = spot above oracle
    "oracle_staleness_s",   # seconds since last oracle update (capped at 60)
    "oracle_updates_window",# number of oracle updates this window
    # Polymarket crowd
    "implied_up_prob",      # 0–1 current market probability for UP
    "prob_dev_from_50",     # |implied_up_prob - 0.5|
    "prob_delta_30s",       # change in implied prob over last 30s
    "polymarket_volume",    # total volume this window (normalized)
    # Funding / CVD
    "funding_rate",         # perpetual funding rate (fraction)
    "cvd_normalized",       # CVD / total_volume in window
    # Session
    "hour_sin",             # sin(hour * 2pi / 24)
    "hour_cos",             # cos(hour * 2pi / 24)
    "dow_sin",              # sin(dow * 2pi / 7)
    "dow_cos",              # cos(dow * 2pi / 7)
    "is_us_session",
    "is_asia_session",
    "is_eu_session",
    # Regime
    "windows_since_big_move",  # windows since last ≥0.3% candle
    "recent_up_rate",          # fraction of last 20 windows that resolved UP
    "streak",                  # consecutive UP(+) or DOWN(-) streak
]


class FeatureEngineer:
    """
    Computes the feature vector at T=150 for a 5-minute window.

    Requires references to live feed objects:
      - binance_feed: BinanceFeed
      - chainlink_feed: ChainlinkFeed
      - polymarket_feed: PolymarketFeed
      - candle_store: CandleStore (holds rolling 1-min OHLCV candles)
      - window_history: list of past window outcomes/metadata
    """

    def __init__(self, binance_feed, chainlink_feed, polymarket_feed, candle_store):
        self.binance    = binance_feed
        self.chainlink  = chainlink_feed
        self.polymarket = polymarket_feed
        self.candles    = candle_store

    def compute(
        self,
        window_start_ts: float,
        reference_price: float,
        current_price: float,
        window_history: List[Dict],
    ) -> Dict[str, float]:
        """
        Compute all features at T=150.
        Returns a dict of feature_name → float value.
        """
        feat: Dict[str, float] = {}
        now = time.time()

        # ── Price displacement ─────────────────────────────────────────────────
        if reference_price > 0:
            delta_pct = (current_price - reference_price) / reference_price
        else:
            delta_pct = 0.0

        feat["delta_pct"]         = delta_pct
        feat["delta_sign"]        = 1.0 if delta_pct >= 0 else -1.0
        feat["abs_delta_pct"]     = abs(delta_pct)
        feat["distance_to_zero"]  = abs(delta_pct)

        # ── Price momentum (from ticks) ────────────────────────────────────────
        t30_ms  = (now - 30)  * 1000
        t60_ms  = (now - 60)  * 1000
        t90_ms  = (now - 90)  * 1000
        t150_ms = window_start_ts * 1000

        ticks = list(self.binance.ticks)

        # Price 30s ago
        price_30s_ago = _price_at_time(ticks, t30_ms, current_price)
        price_60s_ago = _price_at_time(ticks, t60_ms, current_price)

        feat["momentum_30s"] = (
            (current_price - price_30s_ago) / price_30s_ago
            if price_30s_ago > 0 else 0.0
        )
        feat["momentum_60s"] = (
            (current_price - price_60s_ago) / price_60s_ago
            if price_60s_ago > 0 else 0.0
        )

        # Velocity: linear slope over last 60s
        recent_ticks_60 = [t["p"] for t in ticks if t["t"] >= t60_ms]
        feat["price_velocity_60s"] = price_velocity(recent_ticks_60, 60, 1.0)

        # ── Technical indicators on 1-min candles ──────────────────────────────
        candle_data = self.candles.get_recent_candles(lookback=60)

        if len(candle_data) >= 3:
            closes  = np.array([c["close"] for c in candle_data])
            highs   = np.array([c["high"]  for c in candle_data])
            lows    = np.array([c["low"]   for c in candle_data])
            volumes = np.array([c["volume"] for c in candle_data])

            feat["rsi_14"] = calc_rsi(closes, RSI_PERIOD) / 100.0   # 0–1

            ema_s = ema_latest(closes, EMA_SHORT)
            ema_l = ema_latest(closes, EMA_LONG)
            feat["ema_cross"] = (ema_s - ema_l) / current_price if current_price > 0 else 0.0

            _, _, macd_hist = calc_macd(closes, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
            feat["macd_hist"] = macd_hist / current_price if current_price > 0 else 0.0

            upper, mid, lower = bollinger_bands(closes, BB_PERIOD, BB_STD)
            feat["bb_position"] = bollinger_position(current_price, upper, lower)

            atr_val = calc_atr(highs, lows, closes, ATR_PERIOD)
            feat["atr_pct"] = atr_val / current_price if current_price > 0 else 0.0

            feat["obv_slope"] = obv_slope(closes, volumes, OBV_LOOKBACK)
        else:
            feat["rsi_14"]     = 0.5
            feat["ema_cross"]  = 0.0
            feat["macd_hist"]  = 0.0
            feat["bb_position"] = 0.5
            feat["atr_pct"]    = 0.002
            feat["obv_slope"]  = 0.0

        # ── Order book ─────────────────────────────────────────────────────────
        feat["bid_ask_imbalance"] = self.binance.get_bid_ask_imbalance(BID_ASK_LEVELS)
        feat["spread_pct"]        = self.binance.get_spread_pct()
        bid_wall, ask_wall        = self.binance.get_wall_distances(
            current_price, WALL_THRESHOLD_RATIO
        )
        feat["bid_wall_dist"] = bid_wall
        feat["ask_wall_dist"] = ask_wall

        # ── Chainlink oracle ───────────────────────────────────────────────────
        feat["chainlink_vs_spot"]    = self.chainlink.get_lag_vs_spot(current_price)
        feat["oracle_staleness_s"]   = min(self.chainlink.staleness, 60.0) / 60.0  # 0–1
        feat["oracle_updates_window"] = float(
            self.chainlink.updates_in_window(window_start_ts)
        )

        # ── Polymarket ─────────────────────────────────────────────────────────
        feat["implied_up_prob"]   = self.polymarket.implied_up_prob
        feat["prob_dev_from_50"]  = abs(self.polymarket.implied_up_prob - 0.5)
        feat["prob_delta_30s"]    = self.polymarket.get_prob_delta(30.0)
        feat["polymarket_volume"] = min(self.polymarket.volume_so_far / 50000.0, 2.0)

        # ── Funding rate & CVD ─────────────────────────────────────────────────
        feat["funding_rate"]    = self.binance.funding_rate   # raw fraction ~0.0001

        window_ticks = [t for t in ticks if t["t"] >= t150_ms]
        total_vol = sum(abs(t["q"]) for t in window_ticks)
        feat["cvd_normalized"] = (
            self.binance.cvd / total_vol if total_vol > 0 else 0.0
        )

        # ── Session features ───────────────────────────────────────────────────
        import datetime
        dt_utc = datetime.datetime.utcnow()
        hour   = dt_utc.hour + dt_utc.minute / 60.0
        dow    = dt_utc.weekday()   # 0=Monday

        feat["hour_sin"] = math.sin(2 * math.pi * hour / 24)
        feat["hour_cos"] = math.cos(2 * math.pi * hour / 24)
        feat["dow_sin"]  = math.sin(2 * math.pi * dow / 7)
        feat["dow_cos"]  = math.cos(2 * math.pi * dow / 7)

        h = int(hour)
        feat["is_us_session"]   = 1.0 if US_SESSION_START   <= h < US_SESSION_END   else 0.0
        feat["is_asia_session"] = 1.0 if ASIA_SESSION_START <= h < ASIA_SESSION_END else 0.0
        feat["is_eu_session"]   = 1.0 if EU_SESSION_START   <= h < EU_SESSION_END   else 0.0

        # ── Regime / history features ──────────────────────────────────────────
        feat["windows_since_big_move"] = float(
            _windows_since_big_move(window_history, threshold=0.003)
        )
        feat["recent_up_rate"] = float(
            _recent_up_rate(window_history, RECENT_WINDOWS_LOOKBACK)
        )
        feat["streak"] = float(_streak(window_history))

        return feat


# ── Helper functions ───────────────────────────────────────────────────────────

def _price_at_time(ticks: list, target_ms: float, fallback: float) -> float:
    """Find the closest tick price at or before target_ms."""
    best = None
    best_diff = float("inf")
    for tick in ticks:
        diff = target_ms - tick["t"]
        if diff >= 0 and diff < best_diff:
            best_diff = diff
            best = tick["p"]
    return best if best is not None else fallback


def _windows_since_big_move(history: List[Dict], threshold: float = 0.003) -> int:
    """Number of consecutive windows (most recent first) without a big move."""
    count = 0
    for w in reversed(history):
        delta = abs(w.get("delta_pct", 0.0))
        if delta >= threshold:
            break
        count += 1
    return min(count, 20)


def _recent_up_rate(history: List[Dict], lookback: int = 20) -> float:
    """Fraction of last `lookback` windows that resolved UP."""
    recent = history[-lookback:]
    if not recent:
        return 0.5
    ups = sum(1 for w in recent if w.get("resolved_up", None) is True)
    return ups / len(recent)


def _streak(history: List[Dict]) -> float:
    """
    Current directional streak.
    Positive = consecutive UPs, negative = consecutive DOWNs.
    Capped at ±10.
    """
    if not history:
        return 0.0
    streak_dir = None
    count = 0
    for w in reversed(history):
        result = w.get("resolved_up", None)
        if result is None:
            break
        if streak_dir is None:
            streak_dir = result
        if result == streak_dir:
            count += 1
        else:
            break
    if streak_dir is False:
        count = -count
    return float(max(-10, min(10, count)))


def feature_vector(feat_dict: Dict[str, float]) -> np.ndarray:
    """Convert feature dict to ordered numpy array matching FEATURE_NAMES."""
    return np.array([feat_dict.get(name, 0.0) for name in FEATURE_NAMES], dtype=np.float32)
