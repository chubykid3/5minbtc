"""
Ensemble — combines all 4 models into a final P(UP) prediction,
then applies an EV-aware decision rule.

Core principle:
  The model outputs P_model(UP). Polymarket has already priced the market at
  implied_up_prob based on first-half price action. These two numbers are
  different things:

    P_model(UP)      — what the model believes is the true probability
    implied_up_prob  — what you actually PAY to bet (the market price)

  The decision signal is the EDGE between them:
    edge = P_model(UP) - implied_up_prob

  Examples:
    edge = +0.12  → model thinks UP is 12pp MORE likely than market prices
                    → buy UP (market is underpricing UP)
    edge = -0.25  → model thinks UP is 25pp LESS likely than market prices
                    → buy DOWN (market is massively overpricing UP —
                      typically because a large move has already happened
                      and the crowd has chased it)
    edge ≈ 0      → no model edge over the crowd, use structural heuristics

  This matters most in large-move windows where the crowd is at 85–95% UP.
  Following that crowd by simply being "directionally bullish" at 0.61
  means buying at 85¢ and needing 87%+ accuracy just to break even.
  The correct read of P_model=0.61 vs implied=0.85 is: BET DOWN.

Fee model:
  Polymarket charges approximately FEE_RATE * min(p, 1-p) per bet.
  At 50/50: fee ≈ 1%. At 85/15: fee ≈ 0.3%. At 95/5: fee ≈ 0.1%.
  Net EV = |edge| - fee_drag — must clear MIN_EV_THRESHOLD to bet.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, NamedTuple, Optional, Tuple
import numpy as np

from config import (
    MODEL_DIR,
    ENSEMBLE_WEIGHTS,
    FEE_RATE,
    MIN_EV_THRESHOLD,
    HIGH_CONF_EV,
    CONTRA_CROWD_EXTRA,
)
from models.logistic_model import LogisticModel
from models.xgboost_model  import XGBoostModel
from models.lstm_model     import LSTMModel
from models.bayesian_model import BayesianModel

log = logging.getLogger(__name__)

_META_PATH = MODEL_DIR / "ensemble_meta.pkl"


# ── Fee helper ─────────────────────────────────────────────────────────────────

def polymarket_fee_drag(implied_prob: float) -> float:
    """
    Estimated one-way fee drag when buying at `implied_prob`.

    Polymarket's fee scales with how close to 50/50 the market is:
      fee ≈ FEE_RATE * min(p, 1-p)

    At p=0.50: fee = 0.020 * 0.50 = 0.010   (1.0%)
    At p=0.80: fee = 0.020 * 0.20 = 0.004   (0.4%)
    At p=0.95: fee = 0.020 * 0.05 = 0.001   (0.1%)
    """
    p = max(0.01, min(0.99, implied_prob))
    return FEE_RATE * min(p, 1.0 - p)


def compute_ev(p_model: float, implied_up: float) -> Tuple[float, float, str]:
    """
    Compute net expected value for both sides given model probability and
    current market implied probability.

    Returns (ev_up, ev_down, better_side):
      ev_up   = P_model(UP) - implied_up  - fee_drag_for_buying_up
      ev_down = implied_up - P_model(UP)  - fee_drag_for_buying_down
               = (1-P_model(UP)) - (1-implied_up) - fee_drag_for_buying_down

    The better_side is the side with higher net EV (could still be negative).
    """
    fee_up   = polymarket_fee_drag(implied_up)
    fee_down = polymarket_fee_drag(1.0 - implied_up)

    ev_up   = p_model       - implied_up       - fee_up
    ev_down = (1 - p_model) - (1 - implied_up) - fee_down
    # Equivalently: ev_down = implied_up - p_model - fee_down

    better = "UP" if ev_up >= ev_down else "DOWN"
    return ev_up, ev_down, better


# ── Main ensemble class ────────────────────────────────────────────────────────

class Ensemble:
    def __init__(self):
        self.logistic = LogisticModel()
        self.xgboost  = XGBoostModel()
        self.lstm     = LSTMModel()
        self.bayesian = BayesianModel()

        self.weights = dict(ENSEMBLE_WEIGHTS)
        self._meta_model = None
        self._load_meta()

    # ── Meta-learner ───────────────────────────────────────────────────────────

    def _load_meta(self):
        if _META_PATH.exists():
            try:
                with open(_META_PATH, "rb") as f:
                    data = pickle.load(f)
                self.weights     = data.get("weights", self.weights)
                self._meta_model = data.get("meta_model")
                log.info(f"Ensemble meta loaded. Weights: {self.weights}")
            except Exception as e:
                log.debug(f"Could not load ensemble meta: {e}")

    def save_meta(self):
        try:
            with open(_META_PATH, "wb") as f:
                pickle.dump({
                    "weights":    self.weights,
                    "meta_model": self._meta_model,
                }, f)
        except Exception as e:
            log.warning(f"Could not save ensemble meta: {e}")

    def train_meta(self, oof_logistic, oof_xgboost, oof_lstm, oof_bayesian, y):
        """
        Train a logistic meta-learner on out-of-fold predictions.
        Also derives optimized weighted average weights from coefficients.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        if len(y) < 100:
            return

        X_meta = np.column_stack([oof_logistic, oof_xgboost, oof_lstm, oof_bayesian])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_meta)

        meta = LogisticRegression(C=1.0, max_iter=500)
        meta.fit(X_scaled, y)

        coef = meta.coef_[0]
        coef_abs = np.abs(coef)
        coef_sum = coef_abs.sum()
        if coef_sum > 0:
            normalized = coef_abs / coef_sum
            self.weights = {
                "logistic": float(normalized[0]),
                "xgboost":  float(normalized[1]),
                "lstm":     float(normalized[2]),
                "bayesian": float(normalized[3]),
            }

        self._meta_model = {"model": meta, "scaler": scaler}
        self.save_meta()
        log.info(f"Ensemble meta trained. Optimized weights: {self.weights}")

    # ── Prediction ─────────────────────────────────────────────────────────────

    def predict(
        self,
        feature_vec:    np.ndarray,
        feature_dict:   Dict[str, float],
        lstm_sequence:  Optional[np.ndarray] = None,
        implied_up_prob: float = 0.5,
    ) -> Tuple[float, str, float, float, Dict[str, float]]:
        """
        Returns:
          (p_model_up, side, ev, net_ev, sub_probas)

          p_model_up  — ensemble P(UP), 0–1
          side        — "UP" or "DOWN"
          ev          — raw edge: P_model(UP) - implied_up (signed)
          net_ev      — ev minus fee drag on the chosen side (the actual bet value)
          sub_probas  — individual model outputs + ev breakdown
        """
        sub_probas: Dict[str, float] = {}

        # ── Sub-model predictions ──────────────────────────────────────────────
        sub_probas["logistic"] = self.logistic.predict_proba(feature_vec)
        sub_probas["xgboost"]  = self.xgboost.predict_proba(feature_vec)
        sub_probas["bayesian"] = self.bayesian.predict_proba(feature_dict)

        if lstm_sequence is not None and self.lstm.is_trained:
            sub_probas["lstm"] = self.lstm.predict_proba(lstm_sequence)
        else:
            sub_probas["lstm"] = sub_probas["xgboost"]

        # ── Ensemble ──────────────────────────────────────────────────────────
        if self._meta_model is not None:
            try:
                X_meta = np.array([[
                    sub_probas["logistic"],
                    sub_probas["xgboost"],
                    sub_probas["lstm"],
                    sub_probas["bayesian"],
                ]])
                scaler   = self._meta_model["scaler"]
                meta_mdl = self._meta_model["model"]
                p_model_up = float(meta_mdl.predict_proba(
                    scaler.transform(X_meta))[0, 1])
            except Exception:
                p_model_up = self._weighted_average(sub_probas)
        else:
            p_model_up = self._weighted_average(sub_probas)

        # ── EV-aware decision ─────────────────────────────────────────────────
        ev_up, ev_down, side, net_ev = self._decide(
            p_model_up, implied_up_prob, feature_dict
        )

        # Annotate sub_probas with EV breakdown for logging
        sub_probas["ev_up"]      = round(ev_up,   4)
        sub_probas["ev_down"]    = round(ev_down,  4)
        sub_probas["implied_up"] = round(implied_up_prob, 4)

        # raw edge = P_model - implied (signed: positive = model more bullish)
        raw_edge = p_model_up - implied_up_prob

        return p_model_up, side, raw_edge, net_ev, sub_probas

    def _weighted_average(self, sub_probas: Dict[str, float]) -> float:
        model_keys  = ["logistic", "xgboost", "lstm", "bayesian"]
        total_w = sum(self.weights.get(k, 0) for k in model_keys)
        if total_w == 0:
            return 0.5
        p_up = sum(
            self.weights.get(k, 0) * sub_probas[k]
            for k in model_keys
        ) / total_w
        return max(0.0, min(1.0, p_up))

    def _decide(
        self,
        p_model: float,
        implied_up: float,
        feat: Dict[str, float],
    ) -> Tuple[float, float, str, float]:
        """
        EV-aware decision rule.

        Returns (ev_up, ev_down, side, net_ev_of_chosen_side).

        Logic:
          1. Compute EV for buying UP vs buying DOWN given model vs market.
          2. If the better side's net EV clears MIN_EV_THRESHOLD → bet that side.
          3. If going against the crowd (contra-crowd bet), require extra margin.
          4. If no EV edge found → structural heuristics → default UP.
        """
        ev_up, ev_down, _ = compute_ev(p_model, implied_up)

        # Is the crowd leaning heavily one way?
        crowd_lean_up   = implied_up > 0.60
        crowd_lean_down = implied_up < 0.40

        # ── Check both sides ──────────────────────────────────────────────────

        # Buying UP: model thinks UP more likely than market prices it
        if ev_up > 0:
            # Going with or against crowd?
            is_contra_crowd = crowd_lean_down   # crowd leans DOWN but we want UP
            required_ev = MIN_EV_THRESHOLD + (CONTRA_CROWD_EXTRA if is_contra_crowd else 0)
            if ev_up >= required_ev:
                return ev_up, ev_down, "UP", ev_up

        # Buying DOWN: model thinks UP less likely than market prices it
        if ev_down > 0:
            # Going with or against crowd? Contra = crowd leans UP but we want DOWN
            is_contra_crowd = crowd_lean_up
            required_ev = MIN_EV_THRESHOLD + (CONTRA_CROWD_EXTRA if is_contra_crowd else 0)
            if ev_down >= required_ev:
                return ev_up, ev_down, "DOWN", ev_down

        # ── No model edge — structural heuristics ─────────────────────────────
        # At this point neither side clears the EV threshold. We still MUST
        # pick a side. Use the best available information without pretending
        # we have edge we don't.

        side = self._structural_default(p_model, implied_up, feat)
        net_ev = ev_up if side == "UP" else ev_down
        return ev_up, ev_down, side, net_ev

    def _structural_default(
        self,
        p_model: float,
        implied_up: float,
        feat: Dict[str, float],
    ) -> str:
        """
        Tiebreaker when no EV edge is found.
        Priority order:
          1. If crowd has strong conviction either way, follow them.
             (At near-50/50 the crowd is just as uncertain as us;
              at 70%+ they're aggregating info we don't have.)
          2. Oracle lag: if Chainlink is stale and spot has moved, follow spot.
          3. High-volatility regime: follow current window's momentum.
          4. Default UP (Chainlink tie goes UP, BTC positive drift).
        """
        # 1. Strong crowd conviction
        if implied_up > 0.65:
            return "UP"
        if implied_up < 0.35:
            return "DOWN"

        # 2. Oracle lag signal
        oracle_lag = feat.get("chainlink_vs_spot", 0.0)
        oracle_stale = feat.get("oracle_staleness_s", 0.0)  # 0–1
        if oracle_stale > 0.5 and abs(oracle_lag) > 0.002:
            # Oracle is stale (>30s) and spot has moved >0.2% from oracle
            return "UP" if oracle_lag > 0 else "DOWN"

        # 3. High-volatility momentum
        atr = feat.get("atr_pct", 0.002)
        delta = feat.get("delta_pct", 0.0)
        if atr > 0.003:
            return "UP" if delta >= 0 else "DOWN"

        # 4. Default UP
        return "UP"

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def any_model_trained(self) -> bool:
        return (
            self.logistic.is_trained or
            self.xgboost.is_trained or
            self.lstm.is_trained
        )
