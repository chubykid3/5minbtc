"""
Historical data fetcher.

Fetches Binance historical OHLCV candles (1-minute) for the last N days
and stores them in SQLite. Used for initial model training and retraining.
"""

import asyncio
import logging
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Optional

import aiohttp

from config import (
    COINBASE_REST_BASE,
    COINBASE_PRODUCT_ID,
    DB_FILE,
    RETRAIN_LOOKBACK_DAYS,
)

log = logging.getLogger(__name__)


class DataFetcher:
    def __init__(self):
        self._db_path = str(DB_FILE)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self._db_path)
        c = conn.cursor()
        # 1-minute OHLCV candles
        c.execute("""
            CREATE TABLE IF NOT EXISTS candles_1m (
                open_time   INTEGER PRIMARY KEY,
                open        REAL,
                high        REAL,
                low         REAL,
                close       REAL,
                volume      REAL,
                close_time  INTEGER
            )
        """)
        # 5-minute aggregated window data (ground truth for training)
        c.execute("""
            CREATE TABLE IF NOT EXISTS windows_5m (
                window_start   INTEGER PRIMARY KEY,
                ref_price      REAL,
                close_price    REAL,
                resolved_up    INTEGER,
                delta_pct      REAL
            )
        """)
        # Live decision log
        c.execute("""
            CREATE TABLE IF NOT EXISTS decisions (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                window_start   INTEGER,
                decision_ts    INTEGER,
                side           TEXT,
                p_model_up     REAL,    -- ensemble P(UP), 0-1
                raw_edge       REAL,    -- P_model - implied_up (signed)
                net_ev         REAL,    -- raw_edge minus fee drag on chosen side
                bet_price      REAL,    -- price per share you're buying at
                implied_up     REAL,    -- market implied UP probability at T=150
                p_logistic     REAL,
                p_xgboost      REAL,
                p_lstm         REAL,
                p_bayesian     REAL,
                ev_up          REAL,    -- EV of buying UP side
                ev_down        REAL,    -- EV of buying DOWN side
                resolved_up    INTEGER,
                correct        INTEGER,
                realised_ev    REAL,    -- actual P&L per unit bet
                features_json  TEXT
            )
        """)
        conn.commit()
        conn.close()

    # ── Candle fetching ────────────────────────────────────────────────────────

    async def fetch_historical_candles(self, days: int = RETRAIN_LOOKBACK_DAYS):
        """
        Fetch 1-minute OHLCV candles from Coinbase Exchange (ex-Pro) for the
        last `days` days. Coinbase returns up to 300 candles per request and
        has months of 1-minute history. Works from US servers.
        Stores in SQLite. Returns count of new candles inserted.
        """
        end_ts   = int(time.time())
        start_ts = end_ts - days * 24 * 3600
        CHUNK    = 300 * 60   # 300 candles × 60 s each

        log.info(f"Fetching {days} days of 1m candles from Coinbase...")

        connector = aiohttp.TCPConnector(limit=3)
        inserted  = 0

        async with aiohttp.ClientSession(connector=connector) as session:
            current_ts = start_ts
            while current_ts < end_ts:
                chunk_end = min(current_ts + CHUNK, end_ts)
                try:
                    params = {
                        "granularity": 60,
                        "start":       current_ts,
                        "end":         chunk_end,
                    }
                    async with session.get(
                        f"{COINBASE_REST_BASE}/products/{COINBASE_PRODUCT_ID}/candles",
                        params=params,
                        timeout=aiohttp.ClientTimeout(total=15),
                    ) as resp:
                        if resp.status != 200:
                            log.error(f"Coinbase candles API error: {resp.status}")
                            await asyncio.sleep(5)
                            continue

                        data = await resp.json()
                        if not data:
                            current_ts = chunk_end
                            continue

                        rows = []
                        for k in data:
                            # Coinbase: [time, low, high, open, close, volume]
                            open_time_s = int(k[0])
                            rows.append((
                                open_time_s * 1000,        # open_time ms
                                float(k[3]),               # open
                                float(k[2]),               # high
                                float(k[1]),               # low
                                float(k[4]),               # close
                                float(k[5]),               # volume
                                (open_time_s + 59) * 1000, # close_time ms (approx)
                            ))

                        if rows:
                            conn = sqlite3.connect(self._db_path)
                            c    = conn.cursor()
                            c.executemany(
                                "INSERT OR IGNORE INTO candles_1m VALUES (?,?,?,?,?,?,?)",
                                rows,
                            )
                            inserted += c.rowcount
                            conn.commit()
                            conn.close()

                        current_ts = chunk_end
                        await asyncio.sleep(0.2)   # Coinbase rate limit is generous

                except Exception as e:
                    log.warning(f"Candle fetch error: {e}. Retrying in 5s...")
                    await asyncio.sleep(5)

        # Build 5-minute window table from 1-minute candles
        self._build_5m_windows()

        log.info(f"Fetched {inserted} new 1m candles, {days}-day window ready.")
        return inserted

    def _build_5m_windows(self):
        """
        Aggregate 1-minute candles into 5-minute windows aligned to UTC.
        Each window: open of first minute = ref_price, close of last = close_price.
        """
        conn = sqlite3.connect(self._db_path)
        c    = conn.cursor()

        c.execute("SELECT open_time, open, close FROM candles_1m ORDER BY open_time")
        rows = c.fetchall()

        if not rows:
            conn.close()
            return

        inserted = 0
        # Group into 5-minute blocks aligned to UTC
        windows: dict = {}
        for open_time_ms, open_price, close_price in rows:
            # Align to 5-minute boundary
            t_s = open_time_ms // 1000
            win_start = (t_s // 300) * 300   # floor to 5-min boundary

            if win_start not in windows:
                windows[win_start] = {
                    "ref":   open_price,   # first open
                    "close": close_price,  # last close (updated)
                }
            else:
                windows[win_start]["close"] = close_price

        inserts = []
        for win_start, d in windows.items():
            ref   = d["ref"]
            close = d["close"]
            delta = (close - ref) / ref if ref > 0 else 0.0
            resolved = 1 if close >= ref else 0
            inserts.append((win_start, ref, close, resolved, delta))

        c.executemany(
            "INSERT OR IGNORE INTO windows_5m VALUES (?,?,?,?,?)",
            inserts,
        )
        inserted = c.rowcount
        conn.commit()
        conn.close()
        log.info(f"Built {inserted} new 5-minute windows")

    # ── Data retrieval for training ───────────────────────────────────────────

    def get_candles(self, lookback_days: int = RETRAIN_LOOKBACK_DAYS) -> List[dict]:
        """Return 1-min candles as list of dicts for the last N days."""
        since_ms = int((time.time() - lookback_days * 86400) * 1000)
        conn = sqlite3.connect(self._db_path)
        c    = conn.cursor()
        c.execute(
            "SELECT open_time, open, high, low, close, volume "
            "FROM candles_1m WHERE open_time >= ? ORDER BY open_time",
            (since_ms,),
        )
        rows = c.fetchall()
        conn.close()
        return [
            {"t": r[0], "open": r[1], "high": r[2], "low": r[3],
             "close": r[4], "volume": r[5]}
            for r in rows
        ]

    def get_5m_windows(self, lookback_days: int = RETRAIN_LOOKBACK_DAYS) -> List[dict]:
        """Return 5-minute windows as list of dicts."""
        since = int(time.time()) - lookback_days * 86400
        conn  = sqlite3.connect(self._db_path)
        c     = conn.cursor()
        c.execute(
            "SELECT window_start, ref_price, close_price, resolved_up, delta_pct "
            "FROM windows_5m WHERE window_start >= ? ORDER BY window_start",
            (since,),
        )
        rows = c.fetchall()
        conn.close()
        return [
            {"window_start": r[0], "ref_price": r[1], "close_price": r[2],
             "resolved_up": bool(r[3]), "delta_pct": r[4]}
            for r in rows
        ]

    def get_recent_decisions(self, n: int = 100) -> List[dict]:
        """Return last N live decisions."""
        conn = sqlite3.connect(self._db_path)
        c    = conn.cursor()
        c.execute(
            "SELECT * FROM decisions ORDER BY id DESC LIMIT ?", (n,)
        )
        rows = c.fetchall()
        conn.close()
        cols = [
            "id", "window_start", "decision_ts", "side",
            "p_model_up", "raw_edge", "net_ev", "bet_price", "implied_up",
            "p_logistic", "p_xgboost", "p_lstm", "p_bayesian",
            "ev_up", "ev_down",
            "resolved_up", "correct", "realised_ev", "features_json",
        ]
        return [dict(zip(cols, r)) for r in rows]

    def insert_decision(self, d: dict):
        """Insert a live decision row."""
        conn = sqlite3.connect(self._db_path)
        c    = conn.cursor()
        c.execute("""
            INSERT INTO decisions (
                window_start, decision_ts, side,
                p_model_up, raw_edge, net_ev, bet_price, implied_up,
                p_logistic, p_xgboost, p_lstm, p_bayesian,
                ev_up, ev_down,
                resolved_up, correct, realised_ev, features_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            d.get("window_start"),
            d.get("decision_ts"),
            d.get("side"),
            d.get("p_model_up"),
            d.get("raw_edge"),
            d.get("net_ev"),
            d.get("bet_price"),
            d.get("implied_up"),
            d.get("p_logistic"),
            d.get("p_xgboost"),
            d.get("p_lstm"),
            d.get("p_bayesian"),
            d.get("ev_up"),
            d.get("ev_down"),
            d.get("resolved_up"),
            d.get("correct"),
            d.get("realised_ev"),
            d.get("features_json"),
        ))
        conn.commit()
        conn.close()

    def update_decision_outcome(self, window_start: int, resolved_up: bool):
        """Update the resolution outcome and realised EV for a window's decision."""
        conn = sqlite3.connect(self._db_path)
        c    = conn.cursor()
        resolved_int = 1 if resolved_up else 0
        # realised_ev: if correct → (1 - bet_price), else → (-bet_price)
        c.execute(
            """
            UPDATE decisions SET
                resolved_up = ?,
                correct = (
                    CASE WHEN (side='UP' AND ?=1) OR (side='DOWN' AND ?=0)
                         THEN 1 ELSE 0 END
                ),
                realised_ev = (
                    CASE WHEN (side='UP' AND ?=1) OR (side='DOWN' AND ?=0)
                         THEN (1.0 - COALESCE(bet_price, 0.5))
                         ELSE (-COALESCE(bet_price, 0.5)) END
                )
            WHERE window_start = ? AND resolved_up IS NULL
            """,
            (resolved_int,
             resolved_int, resolved_int,
             resolved_int, resolved_int,
             window_start),
        )
        conn.commit()
        conn.close()
