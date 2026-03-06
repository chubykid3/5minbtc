"""
Model C: LSTM — sequential pattern model.

Takes the last LSTM_SEQUENCE_LENGTH windows of feature vectors as a time
series. Captures patterns like "3 consecutive large UPs → DOWN likely."

Uses TensorFlow/Keras. Falls back gracefully to 0.5 if TF not installed.
"""

import logging
import os
import numpy as np
from pathlib import Path

from config import MODEL_DIR, LSTM_SEQUENCE_LENGTH
from features.feature_engineer import FEATURE_NAMES

log = logging.getLogger(__name__)

_SAVE_PATH = str(MODEL_DIR / "lstm_model")   # directory for TF SavedModel
_SCALER_PATH = MODEL_DIR / "lstm_scaler.pkl"

# Suppress TF verbosity
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


class LSTMModel:
    def __init__(self):
        self._model = None
        self._scaler = None
        self.is_trained = False
        self._tf_available = self._check_tf()
        if self._tf_available:
            self._load()

    def _check_tf(self) -> bool:
        try:
            import tensorflow as tf   # noqa: F401
            return True
        except ImportError:
            log.warning("TensorFlow not installed — LSTMModel disabled. "
                        "Run: pip install tensorflow")
            return False

    def _load(self):
        try:
            import tensorflow as tf
            import pickle
            if Path(_SAVE_PATH).exists():
                self._model = tf.keras.models.load_model(_SAVE_PATH)
                self.is_trained = True
                log.info("LSTMModel loaded from disk")
            if _SCALER_PATH.exists():
                with open(_SCALER_PATH, "rb") as f:
                    self._scaler = pickle.load(f)
        except Exception as e:
            log.warning(f"Could not load LSTMModel: {e}")

    def save(self):
        try:
            import pickle
            self._model.save(_SAVE_PATH)
            with open(_SCALER_PATH, "wb") as f:
                pickle.dump(self._scaler, f)
            log.info("LSTMModel saved")
        except Exception as e:
            log.warning(f"Could not save LSTMModel: {e}")

    def train(self, X_seq: np.ndarray, y: np.ndarray):
        """
        X_seq: (n_samples, LSTM_SEQUENCE_LENGTH, n_features)
        y:     (n_samples,) binary 0/1
        """
        if not self._tf_available:
            return
        if len(X_seq) < 200:
            log.warning("LSTMModel: not enough training samples")
            return

        import tensorflow as tf
        import pickle
        from sklearn.preprocessing import StandardScaler

        n_samples, seq_len, n_features = X_seq.shape

        # Fit scaler on flattened, then reshape
        X_flat = X_seq.reshape(-1, n_features)
        scaler = StandardScaler()
        X_scaled_flat = scaler.fit_transform(X_flat)
        X_scaled = X_scaled_flat.reshape(n_samples, seq_len, n_features)

        self._scaler = scaler

        # Build model
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(
                64,
                input_shape=(seq_len, n_features),
                return_sequences=True,
                dropout=0.2,
                recurrent_dropout=0.1,
            ),
            tf.keras.layers.LSTM(32, dropout=0.2),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        )

        model.fit(
            X_scaled, y,
            epochs=50,
            batch_size=64,
            validation_split=0.15,
            callbacks=[early_stop],
            verbose=0,
        )

        self._model   = model
        self.is_trained = True
        self.save()
        log.info(f"LSTMModel trained on {len(X_seq)} sequences")

    def predict_proba(self, x_seq: np.ndarray) -> float:
        """
        x_seq: (LSTM_SEQUENCE_LENGTH, n_features) — the last N windows.
        Returns P(UP) ∈ [0, 1]. Returns 0.5 if not trained.
        """
        if not self.is_trained or self._model is None or not self._tf_available:
            return 0.5
        try:
            seq_len, n_features = x_seq.shape
            X_flat = x_seq.reshape(-1, n_features)
            X_scaled_flat = self._scaler.transform(X_flat)
            X_scaled = X_scaled_flat.reshape(1, seq_len, n_features)
            proba = float(self._model.predict(X_scaled, verbose=0)[0, 0])
            return proba
        except Exception as e:
            log.debug(f"LSTMModel predict error: {e}")
            return 0.5
