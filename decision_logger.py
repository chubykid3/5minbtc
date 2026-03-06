"""
Decision Logger + Terminal Display.

Logs every decision to file, JSONL metrics, and SQLite.
Renders a live terminal dashboard using rich.
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

from config import LOG_FILE, METRICS_FILE

log = logging.getLogger(__name__)


# ── Rich display ───────────────────────────────────────────────────────────────

def _try_import_rich():
    try:
        from rich.console import Console
        from rich.table   import Table
        from rich.panel   import Panel
        from rich.layout  import Layout
        from rich.text    import Text
        from rich         import box
        return True
    except ImportError:
        return False


HAS_RICH = _try_import_rich()


class DecisionLogger:
    def __init__(self, db_fetcher):
        self.fetcher = db_fetcher
        self._decisions: List[Dict] = []     # recent in-memory cache
        self._session_correct = 0
        self._session_total   = 0
        self._session_start   = time.time()

        # File logger setup
        _fh = logging.FileHandler(str(LOG_FILE), encoding="utf-8")
        _fh.setLevel(logging.INFO)
        _fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        _fh.setFormatter(_fmt)
        logging.getLogger().addHandler(_fh)

    # ── Logging ────────────────────────────────────────────────────────────────

    def log_decision(
        self,
        window_start:  int,
        side:          str,
        p_up:          float,
        confidence:    float,
        sub_probas:    Dict[str, float],
        implied_up:    float,
        features:      Dict[str, float],
    ):
        """Log a prediction at T=150."""
        ts = int(time.time())
        entry = {
            "window_start": window_start,
            "decision_ts":  ts,
            "side":         side,
            "p_up":         round(p_up, 4),
            "confidence":   round(confidence, 4),
            "p_logistic":   round(sub_probas.get("logistic", 0.5), 4),
            "p_xgboost":    round(sub_probas.get("xgboost",  0.5), 4),
            "p_lstm":       round(sub_probas.get("lstm",     0.5), 4),
            "p_bayesian":   round(sub_probas.get("bayesian", 0.5), 4),
            "implied_up":   round(implied_up, 4),
            "resolved_up":  None,
            "correct":      None,
            "features_json": json.dumps(
                {k: round(v, 6) for k, v in features.items()}
            ),
        }

        self._decisions.append(entry)
        if len(self._decisions) > 200:
            self._decisions = self._decisions[-200:]

        self.fetcher.insert_decision(entry)

        win_dt = datetime.utcfromtimestamp(window_start).strftime("%H:%M:%S")
        log.info(
            f"DECISION | Window {win_dt} UTC | "
            f"Side={side:<4} | P(UP)={p_up:.3f} | Conf={confidence:.3f} | "
            f"LR={sub_probas.get('logistic', 0.5):.3f} "
            f"XGB={sub_probas.get('xgboost', 0.5):.3f} "
            f"LSTM={sub_probas.get('lstm', 0.5):.3f} "
            f"BAY={sub_probas.get('bayesian', 0.5):.3f} | "
            f"Crowd={implied_up:.3f}"
        )

        # JSONL metrics
        with open(str(METRICS_FILE), "a") as f:
            f.write(json.dumps({
                "type":         "decision",
                "ts":           ts,
                "window_start": window_start,
                "side":         side,
                "p_up":         round(p_up, 4),
                "sub_probas":   {k: round(v, 4) for k, v in sub_probas.items()},
                "implied_up":   round(implied_up, 4),
                "confidence":   round(confidence, 4),
            }) + "\n")

    def log_outcome(
        self,
        window_start: int,
        ref_price:    float,
        close_price:  float,
        resolved_up:  bool,
        predicted_side: Optional[str] = None,
    ):
        """Log resolution outcome at T=300."""
        self.fetcher.update_decision_outcome(window_start, resolved_up)

        # Find matching decision in memory
        matching = [d for d in self._decisions if d["window_start"] == window_start]
        if matching:
            d = matching[0]
            d["resolved_up"] = resolved_up
            predicted_up = d["side"] == "UP"
            d["correct"]  = predicted_up == resolved_up
            if d["correct"] is not None:
                self._session_total   += 1
                self._session_correct += 1 if d["correct"] else 0

        delta_pct = (close_price - ref_price) / ref_price * 100
        win_dt = datetime.utcfromtimestamp(window_start).strftime("%H:%M:%S")
        direction = "UP  " if resolved_up else "DOWN"
        correct_str = ""
        if predicted_side is not None:
            correct = (predicted_side == "UP") == resolved_up
            correct_str = f" | {'✓ CORRECT' if correct else '✗ WRONG'}"

        log.info(
            f"OUTCOME  | Window {win_dt} UTC | "
            f"Result={direction} | Δ={delta_pct:+.4f}% | "
            f"Ref={ref_price:.2f} → {close_price:.2f}{correct_str}"
        )

        with open(str(METRICS_FILE), "a") as f:
            f.write(json.dumps({
                "type":         "outcome",
                "window_start": window_start,
                "ref_price":    round(ref_price, 2),
                "close_price":  round(close_price, 2),
                "delta_pct":    round(delta_pct, 4),
                "resolved_up":  resolved_up,
            }) + "\n")

    # ── Session statistics ─────────────────────────────────────────────────────

    def session_stats(self) -> Dict:
        total   = self._session_total
        correct = self._session_correct
        acc     = correct / total if total > 0 else 0.0
        elapsed = time.time() - self._session_start

        # Decisions with resolved outcomes
        resolved = [d for d in self._decisions if d.get("correct") is not None]
        high_conf = [d for d in resolved if d.get("confidence", 0) >= 0.08]
        hc_correct = sum(1 for d in high_conf if d.get("correct"))

        return {
            "total":          total,
            "correct":        correct,
            "accuracy":       acc,
            "high_conf_total": len(high_conf),
            "high_conf_correct": hc_correct,
            "high_conf_acc":  hc_correct / len(high_conf) if high_conf else 0.0,
            "elapsed_s":      int(elapsed),
        }

    # ── Terminal display ───────────────────────────────────────────────────────

    def print_dashboard(
        self,
        current_price:   float,
        reference_price: float,
        window_start:    int,
        time_in_window:  float,
        ensemble_state:  Optional[Dict] = None,
        feeds_status:    Optional[Dict] = None,
    ):
        """Print a rich terminal dashboard."""
        if not HAS_RICH:
            self._print_plain(current_price, reference_price,
                              window_start, time_in_window)
            return

        from rich.console import Console
        from rich.table   import Table
        from rich.panel   import Panel
        from rich.text    import Text
        from rich         import box

        console = Console()
        console.clear()

        # Header
        delta = (current_price - reference_price) / reference_price * 100 if reference_price > 0 else 0
        dir_color = "green" if delta >= 0 else "red"
        dir_sym   = "▲" if delta >= 0 else "▼"

        remaining = max(0, 300 - time_in_window)
        at_decision = abs(time_in_window - 150) < 5

        header = Text()
        header.append("  BTC 5-MIN DECISION BOT  ", style="bold white on blue")
        header.append(f"  ${current_price:,.2f}  ", style=f"bold {dir_color}")
        header.append(f"{dir_sym} {delta:+.3f}%", style=f"bold {dir_color}")
        console.print(header)
        console.print()

        # Window status
        win_dt = datetime.utcfromtimestamp(window_start).strftime("%H:%M:%S")
        mins, secs = divmod(int(time_in_window), 60)
        console.print(
            f"  Window: [cyan]{win_dt} UTC[/]  |  "
            f"T+{mins:02d}:{secs:02d}  |  "
            f"Remaining: [yellow]{int(remaining)}s[/]"
            + ("  [bold magenta]<<< DECISION POINT >>>[/]" if at_decision else "")
        )
        console.print()

        # Feeds status
        if feeds_status:
            feed_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
            feed_table.add_column()
            feed_table.add_column()
            for feed, status in feeds_status.items():
                color = "green" if status else "red"
                icon  = "●" if status else "○"
                feed_table.add_row(f"[{color}]{icon}[/] {feed}", "")
            console.print(Panel(feed_table, title="Data Feeds", width=40))
            console.print()

        # Recent decisions
        if self._decisions:
            dec_table = Table(
                title="Recent Decisions",
                box=box.SIMPLE_HEAD,
                show_header=True,
            )
            dec_table.add_column("Time", style="cyan",    width=10)
            dec_table.add_column("Side", style="bold",    width=6)
            dec_table.add_column("P(UP)", style="white",  width=7)
            dec_table.add_column("Conf",  style="white",  width=6)
            dec_table.add_column("Crowd", style="dim",    width=7)
            dec_table.add_column("Result", style="bold",  width=8)
            dec_table.add_column("Model Breakdown",       width=34)

            for d in reversed(self._decisions[-10:]):
                dt_str  = datetime.utcfromtimestamp(d["window_start"]).strftime("%H:%M")
                side    = d["side"]
                correct = d.get("correct")
                result_str = ""
                if correct is True:
                    result_str = "[green]✓ WIN[/]"
                elif correct is False:
                    result_str = "[red]✗ LOSS[/]"
                else:
                    result_str = "[dim]pending[/]"

                side_color = "green" if side == "UP" else "red"
                breakdown = (
                    f"LR={d['p_logistic']:.2f} "
                    f"XGB={d['p_xgboost']:.2f} "
                    f"LSTM={d['p_lstm']:.2f} "
                    f"BAY={d['p_bayesian']:.2f}"
                )
                dec_table.add_row(
                    dt_str,
                    f"[{side_color}]{side}[/]",
                    f"{d['p_up']:.3f}",
                    f"{d['confidence']:.3f}",
                    f"{d['implied_up']:.3f}",
                    result_str,
                    breakdown,
                )
            console.print(dec_table)
            console.print()

        # Session stats
        stats = self.session_stats()
        acc_color = "green" if stats["accuracy"] >= 0.55 else (
            "yellow" if stats["accuracy"] >= 0.50 else "red"
        )
        console.print(
            f"  Session: [bold]{stats['total']}[/] decisions | "
            f"Accuracy: [{acc_color}]{stats['accuracy']:.1%}[/] | "
            f"High-conf: {stats['high_conf_correct']}/{stats['high_conf_total']} "
            f"([{acc_color}]{stats['high_conf_acc']:.1%}[/])"
        )
        console.print()

    def _print_plain(
        self,
        current_price:   float,
        reference_price: float,
        window_start:    int,
        time_in_window:  float,
    ):
        """Fallback when rich is not installed."""
        delta = (current_price - reference_price) / reference_price * 100 if reference_price > 0 else 0
        win_dt = datetime.utcfromtimestamp(window_start).strftime("%H:%M:%S")
        print(
            f"\r  ${current_price:,.2f} ({delta:+.3f}%) | "
            f"Window {win_dt} | T+{int(time_in_window)}s",
            end="",
            flush=True,
        )
