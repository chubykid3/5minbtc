"""
Model A: Logistic Regression — baseline linear model.

Trained on the full feature vector. Serialized to disk so it
persists across restarts.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from config import MODEL_DIR
from features.feature_engineer import FEATURE_NAMES

log = logging.getLogger(__name__)

_SAVE_PATH = MODEL_DIR / "logistic_model.pkl"


class LogisticModel:
    def __init__(self):
        self._model = None
        self._scaler = None
        self.is_trained = False
        self._load()

    def _load(self):
        if _SAVE_PATH.exists():
            try:
                with open(_SAVE_PATH, "rb") as f:
                    data = pickle.load(f)
                self._model  = data["model"]
                self._scaler = data["scaler"]
                self.is_trained = True
                log.info("LogisticModel loaded from disk")
            except Exception as e:
                log.warning(f"Could not load LogisticModel: {e}")

    def save(self):
        try:
            with open(_SAVE_PATH, "wb") as f:
                pickle.dump({"model": self._model, "scaler": self._scaler}, f)
            log.info("LogisticModel saved")
        except Exception as e:
            log.warning(f"Could not save LogisticModel: {e}")

    def train(self, X: np.ndarray, y: np.ndarray):
        """
        X: (n_samples, n_features), y: (n_samples,) binary 0/1.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        if len(X) < 50:
            log.warning("LogisticModel: not enough training samples")
            return

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LogisticRegression(
            C=0.5,
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced",
        )
        model.fit(X_scaled, y)

        self._model  = model
        self._scaler = scaler
        self.is_trained = True
        self.save()
        log.info(f"LogisticModel trained on {len(X)} samples")

    def predict_proba(self, x: np.ndarray) -> float:
        """Returns P(UP) ∈ [0, 1]. Returns 0.5 if not trained."""
        if not self.is_trained or self._model is None:
            return 0.5
        try:
            x_scaled = self._scaler.transform(x.reshape(1, -1))
            proba = self._model.predict_proba(x_scaled)[0]
            # proba[1] = P(class=1 = UP)
            return float(proba[1])
        except Exception as e:
            log.debug(f"LogisticModel predict error: {e}")
            return 0.5
