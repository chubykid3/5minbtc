"""
Technical indicator calculations on OHLCV candle data.
All functions operate on plain Python lists/numpy arrays for speed.
"""

import numpy as np
from typing import List, Optional, Tuple


def rsi(closes: np.ndarray, period: int = 14) -> float:
    """Relative Strength Index. Returns latest value or 50 if insufficient data."""
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def ema(values: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average array."""
    if len(values) == 0:
        return np.array([])
    result = np.zeros(len(values))
    k = 2.0 / (period + 1)
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = values[i] * k + result[i - 1] * (1 - k)
    return result


def ema_latest(values: np.ndarray, period: int) -> float:
    """Return only the latest EMA value."""
    arr = ema(values, period)
    return float(arr[-1]) if len(arr) > 0 else float(np.mean(values)) if len(values) > 0 else 0.0


def macd(closes: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
    """
    Returns (macd_line, signal_line, histogram).
    All as latest values.
    """
    if len(closes) < slow + signal:
        return 0.0, 0.0, 0.0
    ema_fast   = ema(closes, fast)
    ema_slow   = ema(closes, slow)
    macd_line  = ema_fast - ema_slow
    sig_line   = ema(macd_line, signal)
    histogram  = macd_line[-1] - sig_line[-1]
    return float(macd_line[-1]), float(sig_line[-1]), float(histogram)


def bollinger_bands(closes: np.ndarray, period: int = 20, std_dev: float = 2.0) -> Tuple[float, float, float]:
    """
    Returns (upper_band, middle_band, lower_band) at latest close.
    """
    if len(closes) < period:
        mean = float(np.mean(closes)) if len(closes) > 0 else 0.0
        return mean, mean, mean
    window = closes[-period:]
    mid    = float(np.mean(window))
    std    = float(np.std(window, ddof=1))
    return mid + std_dev * std, mid, mid - std_dev * std


def bollinger_position(close: float, upper: float, lower: float) -> float:
    """
    Where is close within the Bollinger Bands?
    0.0 = at lower band, 1.0 = at upper band.
    """
    band_width = upper - lower
    if band_width <= 0:
        return 0.5
    return (close - lower) / band_width


def atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
    """Average True Range — latest value."""
    if len(highs) < 2:
        return 0.0
    tr_list = []
    for i in range(1, len(highs)):
        h = highs[i]
        l = lows[i]
        pc = closes[i - 1]
        tr = max(h - l, abs(h - pc), abs(l - pc))
        tr_list.append(tr)
    if not tr_list:
        return 0.0
    arr = np.array(tr_list)
    if len(arr) < period:
        return float(np.mean(arr))
    # Wilder's smoothing
    atr_val = float(np.mean(arr[:period]))
    for i in range(period, len(arr)):
        atr_val = (atr_val * (period - 1) + arr[i]) / period
    return atr_val


def obv(closes: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    """On-Balance Volume array."""
    if len(closes) < 2:
        return np.zeros(len(closes))
    obv_arr = np.zeros(len(closes))
    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            obv_arr[i] = obv_arr[i - 1] + volumes[i]
        elif closes[i] < closes[i - 1]:
            obv_arr[i] = obv_arr[i - 1] - volumes[i]
        else:
            obv_arr[i] = obv_arr[i - 1]
    return obv_arr


def obv_slope(closes: np.ndarray, volumes: np.ndarray, lookback: int = 10) -> float:
    """Linear slope of OBV over last `lookback` periods. Normalized by mean price."""
    obv_arr = obv(closes, volumes)
    if len(obv_arr) < lookback:
        lookback = len(obv_arr)
    if lookback < 2:
        return 0.0
    y = obv_arr[-lookback:]
    x = np.arange(lookback)
    slope = float(np.polyfit(x, y, 1)[0])
    # Normalize by average close to make it price-independent
    avg_close = float(np.mean(closes[-lookback:])) if len(closes) >= lookback else 1.0
    return slope / avg_close if avg_close > 0 else 0.0


def price_velocity(prices: List[float], window_seconds: int = 60, tick_interval_seconds: float = 1.0) -> float:
    """
    Linear slope of prices over the window as % change per second.
    `prices` should be equally-spaced samples at `tick_interval_seconds`.
    """
    n = int(window_seconds / tick_interval_seconds)
    if len(prices) < 2:
        return 0.0
    p = prices[-min(n, len(prices)):]
    if len(p) < 2 or p[0] == 0:
        return 0.0
    x = np.arange(len(p))
    slope = float(np.polyfit(x, p, 1)[0])
    return slope / p[0]   # fractional change per interval
