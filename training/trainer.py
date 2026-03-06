"""
Trainer — orchestrates model training and retraining.

Builds feature matrices from historical candles, trains all models,
and optionally runs a walk-forward backtest.
"""

import logging
import time
import numpy as np
from collections import deque
from typing import List, Dict, Tuple, Optional

from config import (
    RETRAIN_LOOKBACK_DAYS,
    LSTM_SEQUENCE_LENGTH,
    MIN_TRAINING_SAMPLES,
)
from training.data_fetcher import DataFetcher
from features.technical_indicators import (
    rsi as calc_rsi, ema_latest, macd as calc_macd,
    bollinger_bands, bollinger_position, atr as calc_atr, obv_slope,
)
from features.feature_engineer import FEATURE_NAMES
import math

log = logging.getLogger(__name__)


class Trainer:
    def __init__(self, ensemble):
        self.ensemble = ensemble
        self.fetcher  = DataFetcher()

    # ── Feature matrix builder ─────────────────────────────────────────────────

    def _build_feature_matrix(
        self, candles_1m: List[dict], windows_5m: List[dict]
    ) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
        """
        Construct (X, y, meta) from historical data.

        For each 5-minute window, we look back at the 1-min candles up to T=150s
        (i.e., candles 0,1,2 of the window, since T=150s = 2.5 minutes).
        We compute a simplified version of the feature vector.

        Returns X (n, n_features), y (n,), meta list of window dicts.
        """
        # Build a fast lookup: minute timestamp → candle dict
        candle_map: Dict[int, dict] = {}
        for c in candles_1m:
            # Key = minute-aligned Unix timestamp (seconds)
            t_s = c["t"] // 1000
            minute_key = (t_s // 60) * 60
            candle_map[minute_key] = c

        X_rows = []
        y_rows = []
        meta   = []

        for win in windows_5m:
            ws = win["window_start"]   # Unix seconds, 5-min aligned
            ref_price   = win["ref_price"]
            close_price = win["close_price"]
            resolved_up = win["resolved_up"]
            delta_pct   = win["delta_pct"]

            # Collect candles up to T=150s (minutes 0,1,2 within window)
            # T=0: minute 0, T=60: minute 1, T=120: minute 2
            # At T=150, we have completed candles 0,1,2
            candles_in_window = []
            for offset in range(3):
                mk = ws + offset * 60
                if mk in candle_map:
                    candles_in_window.append(candle_map[mk])

            if not candles_in_window:
                continue

            # Current price at T=150 ≈ close of the 3rd candle
            current_price = candles_in_window[-1]["close"]

            # Get a rolling window of past candles for indicators
            lookback_candles = []
            for offset in range(-60, 3):
                mk = ws + offset * 60
                if mk in candle_map:
                    lookback_candles.append(candle_map[mk])

            feat = self._compute_historical_features(
                win, current_price, ref_price, lookback_candles,
                windows_5m, win
            )

            if feat is None:
                continue

            X_rows.append(feat)
            y_rows.append(1 if resolved_up else 0)
            meta.append(win)

        if not X_rows:
            return np.array([]), np.array([]), []

        return np.array(X_rows, dtype=np.float32), np.array(y_rows), meta

    def _compute_historical_features(
        self,
        win: dict,
        current_price: float,
        ref_price: float,
        lookback_candles: List[dict],
        all_windows: List[dict],
        current_win: dict,
    ) -> Optional[List[float]]:
        """Build a feature row matching FEATURE_NAMES from historical data."""
        if ref_price <= 0 or current_price <= 0:
            return None

        ws = win["window_start"]
        delta_pct = (current_price - ref_price) / ref_price

        if len(lookback_candles) < 5:
            return None

        closes  = np.array([c["close"]  for c in lookback_candles])
        highs   = np.array([c["high"]   for c in lookback_candles])
        lows    = np.array([c["low"]    for c in lookback_candles])
        volumes = np.array([c["volume"] for c in lookback_candles])

        # RSI
        rsi_val   = calc_rsi(closes, 14) / 100.0

        # EMA cross
        ema_s     = ema_latest(closes, 9)
        ema_l     = ema_latest(closes, 21)
        ema_cross = (ema_s - ema_l) / current_price if current_price > 0 else 0.0

        # MACD
        _, _, macd_hist_val = calc_macd(closes, 12, 26, 9)
        macd_hist  = macd_hist_val / current_price if current_price > 0 else 0.0

        # Bollinger
        upper, _, lower = bollinger_bands(closes, 20, 2.0)
        bb_pos = bollinger_position(current_price, upper, lower)

        # ATR
        atr_val = calc_atr(highs, lows, closes, 14)
        atr_pct = atr_val / current_price if current_price > 0 else 0.002

        # OBV slope
        obv_s = obv_slope(closes, volumes, 20)

        # Momentum (approximate from candle closes)
        mom_30 = (closes[-1] - closes[-2]) / closes[-2] if len(closes) >= 2 and closes[-2] > 0 else 0.0
        mom_60 = (closes[-1] - closes[-3]) / closes[-3] if len(closes) >= 3 and closes[-3] > 0 else 0.0

        # Session
        import datetime
        dt_utc = datetime.datetime.utcfromtimestamp(ws)
        hour   = dt_utc.hour + dt_utc.minute / 60.0
        dow    = dt_utc.weekday()
        h      = int(hour)

        # Past window history context (only windows before this one)
        prior_wins = [w for w in all_windows if w["window_start"] < ws]

        # Streak
        streak = 0.0
        streak_dir = None
        for pw in reversed(prior_wins[-10:]):
            res = pw.get("resolved_up")
            if res is None:
                break
            if streak_dir is None:
                streak_dir = res
            if res == streak_dir:
                streak += 1
            else:
                break
        if streak_dir is False:
            streak = -streak

        # Recent up rate
        last20 = prior_wins[-20:]
        recent_up_rate = sum(1 for w in last20 if w.get("resolved_up")) / len(last20) if last20 else 0.5

        # Windows since big move
        wsince = 0
        for pw in reversed(prior_wins[-20:]):
            if abs(pw.get("delta_pct", 0.0)) >= 0.003:
                break
            wsince += 1

        feat = [
            delta_pct,                          # delta_pct
            1.0 if delta_pct >= 0 else -1.0,   # delta_sign
            abs(delta_pct),                     # abs_delta_pct
            mom_30,                             # momentum_30s
            mom_60,                             # momentum_60s
            mom_60 / 60.0,                      # price_velocity_60s (approx)
            abs(delta_pct),                     # distance_to_zero
            rsi_val,                            # rsi_14
            ema_cross,                          # ema_cross
            macd_hist,                          # macd_hist
            bb_pos,                             # bb_position
            atr_pct,                            # atr_pct
            obv_s,                              # obv_slope
            1.0,                                # bid_ask_imbalance (unknown historically)
            0.01,                               # bid_wall_dist
            0.01,                               # ask_wall_dist
            0.0002,                             # spread_pct
            0.0,                                # chainlink_vs_spot
            0.5,                                # oracle_staleness_s
            3.0,                                # oracle_updates_window
            0.5,                                # implied_up_prob
            0.0,                                # prob_dev_from_50
            0.0,                                # prob_delta_30s
            0.5,                                # polymarket_volume
            0.0,                                # funding_rate
            0.0,                                # cvd_normalized
            math.sin(2 * math.pi * hour / 24), # hour_sin
            math.cos(2 * math.pi * hour / 24), # hour_cos
            math.sin(2 * math.pi * dow / 7),   # dow_sin
            math.cos(2 * math.pi * dow / 7),   # dow_cos
            1.0 if 13 <= h < 22 else 0.0,      # is_us_session
            1.0 if 0  <= h <  8 else 0.0,      # is_asia_session
            1.0 if 7  <= h < 16 else 0.0,      # is_eu_session
            float(wsince),                      # windows_since_big_move
            recent_up_rate,                     # recent_up_rate
            streak,                             # streak
        ]

        assert len(feat) == len(FEATURE_NAMES), \
            f"Feature count mismatch: {len(feat)} vs {len(FEATURE_NAMES)}"

        return feat

    # ── LSTM sequence builder ──────────────────────────────────────────────────

    def _build_lstm_sequences(
        self, X: np.ndarray, y: np.ndarray, seq_len: int = LSTM_SEQUENCE_LENGTH
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build (X_seq, y_seq) for LSTM training."""
        if len(X) < seq_len + 1:
            return np.array([]), np.array([])

        X_seq = []
        y_seq = []
        for i in range(seq_len, len(X)):
            X_seq.append(X[i - seq_len:i])
            y_seq.append(y[i])

        return np.array(X_seq), np.array(y_seq)

    # ── Main training routine ──────────────────────────────────────────────────

    async def run(self, fetch_data: bool = True):
        """
        Full training run:
          1. (Optionally) fetch historical data
          2. Build feature matrices
          3. Train all models
        """
        if fetch_data:
            await self.fetcher.fetch_historical_candles(RETRAIN_LOOKBACK_DAYS)

        candles = self.fetcher.get_candles(RETRAIN_LOOKBACK_DAYS)
        windows = self.fetcher.get_5m_windows(RETRAIN_LOOKBACK_DAYS)

        log.info(f"Training on {len(candles)} 1m candles, {len(windows)} 5m windows")

        if len(windows) < MIN_TRAINING_SAMPLES:
            log.warning(
                f"Only {len(windows)} windows available — "
                f"need {MIN_TRAINING_SAMPLES}. Training skipped."
            )
            return

        X, y, meta = self._build_feature_matrix(candles, windows)

        if len(X) < MIN_TRAINING_SAMPLES:
            log.warning(f"Only {len(X)} valid feature rows. Training skipped.")
            return

        log.info(f"Feature matrix: {X.shape}, labels: {y.sum()}/{len(y)} UP")

        # Train base models
        self.ensemble.logistic.train(X, y)
        self.ensemble.xgboost.train(X, y)

        # LSTM
        X_seq, y_seq = self._build_lstm_sequences(X, y)
        if len(X_seq) >= 200:
            self.ensemble.lstm.train(X_seq, y_seq)

        # Meta-learner (simplified — use same data, no OOF for now)
        # In production: use proper k-fold OOF predictions
        p_log = np.array([self.ensemble.logistic.predict_proba(x) for x in X])
        p_xgb = np.array([self.ensemble.xgboost.predict_proba(x) for x in X])
        p_bay = np.array([
            self.ensemble.bayesian.predict_proba(
                dict(zip(FEATURE_NAMES, x.tolist()))
            ) for x in X
        ])
        # LSTM probas (aligned to X_seq)
        if len(X_seq) >= 200:
            p_lstm = np.array([
                self.ensemble.lstm.predict_proba(x_seq)
                for x_seq in X_seq
            ])
            # Align other arrays to X_seq length
            offset = len(X) - len(X_seq)
            self.ensemble.train_meta(
                p_log[offset:], p_xgb[offset:], p_lstm, p_bay[offset:], y[offset:]
            )
        else:
            # Without LSTM just use XGBoost as placeholder
            self.ensemble.train_meta(p_log, p_xgb, p_xgb, p_bay, y)

        log.info("Training complete.")

    # ── Walk-forward backtest ──────────────────────────────────────────────────

    def backtest(self, n_windows: int = 500) -> Dict:
        """
        Simple walk-forward accuracy check on the most recent N windows.
        Returns accuracy metrics dict.
        """
        candles = self.fetcher.get_candles(RETRAIN_LOOKBACK_DAYS)
        windows = self.fetcher.get_5m_windows(RETRAIN_LOOKBACK_DAYS)

        if len(windows) < n_windows + 100:
            log.warning("Not enough windows for backtest")
            return {}

        test_windows = windows[-(n_windows):]

        X, y, meta = self._build_feature_matrix(candles, test_windows)
        if len(X) == 0:
            return {}

        correct_total  = 0
        correct_hc     = 0   # high-confidence only
        n_hc           = 0
        p_ups          = []

        for i, (x_row, label) in enumerate(zip(X, y)):
            feat_dict = dict(zip(FEATURE_NAMES, x_row.tolist()))
            p_up, side, conf, _ = self.ensemble.predict(
                x_row, feat_dict, None, 0.5
            )
            p_ups.append(p_up)
            predicted_up = side == "UP"
            if predicted_up == bool(label):
                correct_total += 1
            if abs(p_up - 0.5) > 0.08:
                n_hc += 1
                if predicted_up == bool(label):
                    correct_hc += 1

        total_acc = correct_total / len(X) if len(X) > 0 else 0.0
        hc_acc    = correct_hc / n_hc if n_hc > 0 else 0.0

        result = {
            "n_windows":       len(X),
            "total_accuracy":  round(total_acc, 4),
            "high_conf_accuracy": round(hc_acc, 4),
            "high_conf_count": n_hc,
            "up_rate":         round(float(np.mean(y)), 4),
        }
        log.info(f"Backtest results: {result}")
        return result
