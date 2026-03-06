"""
Model D: Mean-Reversion / Momentum Bayesian model.

Rule-based structural logic — no ML training required.

Logic:
  - When |delta_pct| > MEAN_REVERSION_THRESHOLD at T=150:
      Bet AGAINST the direction (mean reversion)
  - When |delta_pct| < MOMENTUM_THRESHOLD and momentum is clear:
      Bet WITH momentum
  - Also incorporates oracle lag, RSI extremes, and order book signals.

Calibration thresholds can be updated by the trainer after backtesting.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, Any

from config import (
    MODEL_DIR,
    MEAN_REVERSION_THRESHOLD,
    MOMENTUM_THRESHOLD,
)

log = logging.getLogger(__name__)

_SAVE_PATH = MODEL_DIR / "bayesian_params.pkl"


class BayesianModel:
    """
    Regime-conditional heuristic model.
    Returns P(UP) based on structural market logic.
    """

    def __init__(self):
        self.is_trained = True   # always active, no training required

        # Calibratable thresholds (can be tuned via backtester)
        self.mean_rev_threshold  = MEAN_REVERSION_THRESHOLD
        self.momentum_threshold  = MOMENTUM_THRESHOLD
        self.rsi_overbought      = 0.70   # (rsi_14 is already 0–1 in our features)
        self.rsi_oversold        = 0.30

        # Weights for combining internal signals
        self.w_displacement   = 0.35
        self.w_momentum       = 0.20
        self.w_rsi            = 0.15
        self.w_orderbook      = 0.15
        self.w_oracle_lag     = 0.15

        self._load()

    def _load(self):
        if _SAVE_PATH.exists():
            try:
                with open(_SAVE_PATH, "rb") as f:
                    params = pickle.load(f)
                self.__dict__.update(params)
                log.info("BayesianModel params loaded from disk")
            except Exception as e:
                log.debug(f"Could not load BayesianModel params: {e}")

    def save(self):
        params = {
            "mean_rev_threshold": self.mean_rev_threshold,
            "momentum_threshold": self.momentum_threshold,
            "rsi_overbought":     self.rsi_overbought,
            "rsi_oversold":       self.rsi_oversold,
            "w_displacement":     self.w_displacement,
            "w_momentum":         self.w_momentum,
            "w_rsi":              self.w_rsi,
            "w_orderbook":        self.w_orderbook,
            "w_oracle_lag":       self.w_oracle_lag,
        }
        try:
            with open(_SAVE_PATH, "wb") as f:
                pickle.dump(params, f)
        except Exception as e:
            log.debug(f"Could not save BayesianModel params: {e}")

    def update_thresholds(self, **kwargs):
        """Update calibration parameters from backtesting results."""
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        self.save()
        log.info(f"BayesianModel thresholds updated: {kwargs}")

    def predict_proba(self, feat: Dict[str, float]) -> float:
        """
        Takes a feature dict and returns P(UP) ∈ [0, 1].

        Logic broken into 5 sub-signals, each contributing ±1 relative to 0.5.
        Final P(UP) = 0.5 + weighted sum of signals.
        """
        delta_pct    = feat.get("delta_pct", 0.0)
        abs_delta    = feat.get("abs_delta_pct", abs(delta_pct))
        mom_60       = feat.get("momentum_60s", 0.0)
        rsi          = feat.get("rsi_14", 0.5)
        imbalance    = feat.get("bid_ask_imbalance", 1.0)
        oracle_lag   = feat.get("chainlink_vs_spot", 0.0)
        oracle_stale = feat.get("oracle_staleness_s", 0.0)   # 0–1
        atr_pct      = feat.get("atr_pct", 0.002)

        # ── Signal 1: Displacement / mean-reversion ────────────────────────────
        if abs_delta > self.mean_rev_threshold:
            # Large move → bet against (mean reversion expected)
            disp_signal = -1.0 if delta_pct > 0 else +1.0
            # Scale: bigger move = stronger mean reversion signal
            strength = min(abs_delta / (self.mean_rev_threshold * 3), 1.0)
            disp_signal *= strength
        elif abs_delta < self.momentum_threshold:
            # Tiny move → momentum signal (follow the direction)
            disp_signal = +1.0 if delta_pct >= 0 else -1.0
            disp_signal *= 0.3   # weak signal when nearly flat
        else:
            # In between — neutral
            disp_signal = 0.0

        # ── Signal 2: Short-term momentum ─────────────────────────────────────
        # Normalize by ATR to be regime-aware
        if atr_pct > 0:
            mom_normalized = mom_60 / atr_pct
        else:
            mom_normalized = 0.0
        mom_signal = max(-1.0, min(1.0, mom_normalized))

        # ── Signal 3: RSI extremes ─────────────────────────────────────────────
        if rsi > self.rsi_overbought:
            rsi_signal = -1.0 * (rsi - self.rsi_overbought) / (1.0 - self.rsi_overbought)
        elif rsi < self.rsi_oversold:
            rsi_signal = +1.0 * (self.rsi_oversold - rsi) / self.rsi_oversold
        else:
            rsi_signal = 0.0

        # ── Signal 4: Order book pressure ─────────────────────────────────────
        # imbalance > 1 = more bids than asks → bullish
        log_imb = 0.0
        if imbalance > 0:
            import math
            log_imb = math.log(imbalance)   # positive = bid heavy
        ob_signal = max(-1.0, min(1.0, log_imb))

        # ── Signal 5: Oracle lag arbitrage ────────────────────────────────────
        # Positive lag = spot above oracle → oracle will snap up at resolution
        # More powerful when oracle is stale
        effective_lag = oracle_lag * oracle_stale * 5   # amplify when stale
        oracle_signal = max(-1.0, min(1.0, effective_lag * 50))   # normalize

        # ── Combine ───────────────────────────────────────────────────────────
        raw_signal = (
            self.w_displacement * disp_signal +
            self.w_momentum     * mom_signal  +
            self.w_rsi          * rsi_signal  +
            self.w_orderbook    * ob_signal   +
            self.w_oracle_lag   * oracle_signal
        )

        p_up = 0.5 + raw_signal * 0.35   # scale so max shift ≈ ±0.35
        return max(0.05, min(0.95, p_up))
