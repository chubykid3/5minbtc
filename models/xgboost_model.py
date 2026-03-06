"""
Model B: XGBoost Gradient Boosting — primary model.

Non-linear feature interactions. Trained on rolling 30-day window,
retrained every 24 hours. Best performer on 5-min BTC direction in
academic literature (59–67% accuracy reported).
"""

import logging
import pickle
from pathlib import Path

import numpy as np

from config import MODEL_DIR
from features.feature_engineer import FEATURE_NAMES

log = logging.getLogger(__name__)

_SAVE_PATH = MODEL_DIR / "xgboost_model.pkl"


class XGBoostModel:
    def __init__(self):
        self._model = None
        self._scaler = None
        self.is_trained = False
        self.feature_importances: dict = {}
        self._load()

    def _load(self):
        if _SAVE_PATH.exists():
            try:
                with open(_SAVE_PATH, "rb") as f:
                    data = pickle.load(f)
                self._model  = data["model"]
                self._scaler = data.get("scaler")
                self.feature_importances = data.get("importances", {})
                self.is_trained = True
                log.info("XGBoostModel loaded from disk")
            except Exception as e:
                log.warning(f"Could not load XGBoostModel: {e}")

    def save(self):
        try:
            with open(_SAVE_PATH, "wb") as f:
                pickle.dump({
                    "model":       self._model,
                    "scaler":      self._scaler,
                    "importances": self.feature_importances,
                }, f)
            log.info("XGBoostModel saved")
        except Exception as e:
            log.warning(f"Could not save XGBoostModel: {e}")

    def train(self, X: np.ndarray, y: np.ndarray):
        """
        X: (n_samples, n_features), y: (n_samples,) binary 0/1.
        """
        try:
            import xgboost as xgb
        except ImportError:
            log.error("xgboost not installed. Run: pip install xgboost")
            return

        from sklearn.preprocessing import StandardScaler

        if len(X) < 100:
            log.warning("XGBoostModel: not enough training samples")
            return

        # Slight normalization still helps XGBoost
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Class balance
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42,
        )
        model.fit(X_scaled, y, verbose=False)

        self._model  = model
        self._scaler = scaler
        self.is_trained = True

        # Record feature importances
        imp = model.feature_importances_
        self.feature_importances = {
            FEATURE_NAMES[i]: float(imp[i])
            for i in range(min(len(FEATURE_NAMES), len(imp)))
        }

        self.save()

        top5 = sorted(self.feature_importances.items(), key=lambda x: -x[1])[:5]
        log.info(
            f"XGBoostModel trained on {len(X)} samples. "
            f"Top features: {[(k, f'{v:.3f}') for k, v in top5]}"
        )

    def predict_proba(self, x: np.ndarray) -> float:
        """Returns P(UP) ∈ [0, 1]. Returns 0.5 if not trained."""
        if not self.is_trained or self._model is None:
            return 0.5
        try:
            x_scaled = self._scaler.transform(x.reshape(1, -1))
            proba = self._model.predict_proba(x_scaled)[0]
            return float(proba[1])
        except Exception as e:
            log.debug(f"XGBoostModel predict error: {e}")
            return 0.5
