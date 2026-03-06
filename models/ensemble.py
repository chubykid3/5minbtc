"""
Ensemble — combines all 4 models into a final P(UP) prediction.

Weights start at the theoretical defaults and are updated via a
meta-learner (stacking) trained on out-of-fold predictions from
the training pipeline.

Decision rule applied here:
  - |P - 0.5| > EDGE_THRESHOLD → confident call
  - otherwise → follow crowd (Polymarket) or default UP
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np

from config import (
    MODEL_DIR,
    ENSEMBLE_WEIGHTS,
    EDGE_THRESHOLD,
    HIGH_CONF_THRESH,
)
from models.logistic_model import LogisticModel
from models.xgboost_model  import XGBoostModel
from models.lstm_model     import LSTMModel
from models.bayesian_model import BayesianModel

log = logging.getLogger(__name__)

_META_PATH = MODEL_DIR / "ensemble_meta.pkl"


class Ensemble:
    def __init__(self):
        self.logistic = LogisticModel()
        self.xgboost  = XGBoostModel()
        self.lstm     = LSTMModel()
        self.bayesian = BayesianModel()

        self.weights = dict(ENSEMBLE_WEIGHTS)   # mutable copy
        self._meta_model = None
        self._load_meta()

    # ── Meta-learner ───────────────────────────────────────────────────────────

    def _load_meta(self):
        if _META_PATH.exists():
            try:
                with open(_META_PATH, "rb") as f:
                    data = pickle.load(f)
                self.weights    = data.get("weights", self.weights)
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
        Also updates the simple weighted average weights.
        oof_* arrays: (n_samples,) P(UP) predictions.
        y: (n_samples,) binary labels.
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
        feature_vec: np.ndarray,
        feature_dict: Dict[str, float],
        lstm_sequence: Optional[np.ndarray] = None,
        implied_up_prob: float = 0.5,
    ) -> Tuple[float, str, float, Dict[str, float]]:
        """
        Returns:
          (final_p_up, side, confidence, sub_probas)

        sub_probas dict has keys: logistic, xgboost, lstm, bayesian.
        side: "UP" or "DOWN".
        confidence: |final_p_up - 0.5|.
        """
        sub_probas: Dict[str, float] = {}

        # Gather sub-model predictions
        sub_probas["logistic"] = self.logistic.predict_proba(feature_vec)
        sub_probas["xgboost"]  = self.xgboost.predict_proba(feature_vec)
        sub_probas["bayesian"] = self.bayesian.predict_proba(feature_dict)

        if lstm_sequence is not None and self.lstm.is_trained:
            sub_probas["lstm"] = self.lstm.predict_proba(lstm_sequence)
        else:
            # LSTM unavailable — redistribute its weight to XGBoost
            sub_probas["lstm"] = sub_probas["xgboost"]

        # Weighted ensemble
        if self._meta_model is not None:
            try:
                X_meta = np.array([[
                    sub_probas["logistic"],
                    sub_probas["xgboost"],
                    sub_probas["lstm"],
                    sub_probas["bayesian"],
                ]])
                scaler    = self._meta_model["scaler"]
                meta_mdl  = self._meta_model["model"]
                X_scaled  = scaler.transform(X_meta)
                final_p_up = float(meta_mdl.predict_proba(X_scaled)[0, 1])
            except Exception:
                final_p_up = self._weighted_average(sub_probas)
        else:
            final_p_up = self._weighted_average(sub_probas)

        # ── Decision rule ──────────────────────────────────────────────────────
        side, confidence = self._decide(final_p_up, implied_up_prob, feature_dict)

        return final_p_up, side, confidence, sub_probas

    def _weighted_average(self, sub_probas: Dict[str, float]) -> float:
        total_weight = sum(self.weights.values())
        if total_weight == 0:
            return 0.5
        p_up = sum(
            self.weights.get(k, 0) * v
            for k, v in sub_probas.items()
        ) / total_weight
        return max(0.0, min(1.0, p_up))

    def _decide(
        self,
        p_up: float,
        implied_up_prob: float,
        feat: Dict[str, float],
    ) -> Tuple[str, float]:
        """Apply decision rule and return (side, confidence)."""
        confidence = abs(p_up - 0.5)

        if p_up > 0.5 + EDGE_THRESHOLD:
            return "UP", confidence

        if p_up < 0.5 - EDGE_THRESHOLD:
            return "DOWN", confidence

        # Near 50/50 — use secondary heuristics
        # 1. Follow crowd
        if implied_up_prob > 0.5 + EDGE_THRESHOLD:
            return "UP", abs(implied_up_prob - 0.5)
        if implied_up_prob < 0.5 - EDGE_THRESHOLD:
            return "DOWN", abs(implied_up_prob - 0.5)

        # 2. Structural bias
        # High volatility → follow momentum
        atr = feat.get("atr_pct", 0.002)
        delta = feat.get("delta_pct", 0.0)
        if atr > 0.003:    # high volatility regime
            return ("UP" if delta >= 0 else "DOWN"), 0.0

        # 3. Default UP (tie-break: UP wins on Chainlink, BTC has positive drift)
        return "UP", 0.0

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def any_model_trained(self) -> bool:
        return (
            self.logistic.is_trained or
            self.xgboost.is_trained or
            self.lstm.is_trained
        )
