"""
Decision Logger + Terminal Display.

Logs every decision to file, JSONL metrics, and SQLite.
Renders a live terminal dashboard using rich.

Key display principle: the primary signal shown is EDGE vs MARKET
(P_model - implied_up), not P_model vs 0.5. The market odds column
shows what you're actually paying, making it immediately obvious
whether a bet is value or not.
"""

import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional

from config import LOG_FILE, METRICS_FILE, HIGH_CONF_EV

log = logging.getLogger(__name__)


def _try_import_rich():
    try:
        from rich.console import Console  # noqa: F401
        return True
    except ImportError:
        return False


HAS_RICH = _try_import_rich()


class DecisionLogger:
    def __init__(self, db_fetcher):
        self.fetcher = db_fetcher
        self._decisions: List[Dict] = []
        self._session_correct = 0
        self._session_total   = 0
        self._session_start   = time.time()

        _fh = logging.FileHandler(str(LOG_FILE), encoding="utf-8")
        _fh.setLevel(logging.INFO)
        _fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        _fh.setFormatter(_fmt)
        logging.getLogger().addHandler(_fh)

    # ── Logging ────────────────────────────────────────────────────────────────

    def log_decision(
        self,
        window_start: int,
        side:         str,
        p_model_up:   float,
        raw_edge:     float,    # P_model(UP) - implied_up  (signed)
        net_ev:       float,    # edge - fee drag on chosen side
        sub_probas:   Dict[str, float],
        implied_up:   float,
        features:     Dict[str, float],
    ):
        """Log a prediction at T=150."""
        ts = int(time.time())

        # Derive bet price: what you're paying per share on the chosen side
        bet_price = implied_up if side == "UP" else (1.0 - implied_up)

        entry = {
            "window_start": window_start,
            "decision_ts":  ts,
            "side":         side,
            "p_model_up":   round(p_model_up, 4),
            "raw_edge":     round(raw_edge, 4),   # positive = model more bullish than crowd
            "net_ev":       round(net_ev, 4),
            "bet_price":    round(bet_price, 4),  # what you pay per share
            "implied_up":   round(implied_up, 4),
            "p_logistic":   round(sub_probas.get("logistic", 0.5), 4),
            "p_xgboost":    round(sub_probas.get("xgboost",  0.5), 4),
            "p_lstm":       round(sub_probas.get("lstm",     0.5), 4),
            "p_bayesian":   round(sub_probas.get("bayesian", 0.5), 4),
            "ev_up":        round(sub_probas.get("ev_up",   0.0), 4),
            "ev_down":      round(sub_probas.get("ev_down", 0.0), 4),
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

        win_dt   = datetime.utcfromtimestamp(window_start).strftime("%H:%M:%S")
        conf_tag = "HIGH-CONF" if abs(net_ev) >= HIGH_CONF_EV else "standard "
        edge_tag = f"edge={raw_edge:+.3f}"
        ev_tag   = f"net_ev={net_ev:+.3f}"
        log.info(
            f"DECISION | Window {win_dt} UTC | "
            f"Side={side:<4} [{conf_tag}] | {edge_tag} | {ev_tag} | "
            f"P_model={p_model_up:.3f} vs Market={implied_up:.3f} | "
            f"Buying@{bet_price:.3f} | "
            f"LR={sub_probas.get('logistic', 0.5):.3f} "
            f"XGB={sub_probas.get('xgboost', 0.5):.3f} "
            f"LSTM={sub_probas.get('lstm', 0.5):.3f} "
            f"BAY={sub_probas.get('bayesian', 0.5):.3f}"
        )

        with open(str(METRICS_FILE), "a") as f:
            f.write(json.dumps({
                "type":         "decision",
                "ts":           ts,
                "window_start": window_start,
                "side":         side,
                "p_model_up":   round(p_model_up, 4),
                "raw_edge":     round(raw_edge, 4),
                "net_ev":       round(net_ev, 4),
                "bet_price":    round(bet_price, 4),
                "implied_up":   round(implied_up, 4),
                "sub_probas":   {k: round(v, 4) for k, v in sub_probas.items()},
            }) + "\n")

    def log_outcome(
        self,
        window_start:   int,
        ref_price:      float,
        close_price:    float,
        resolved_up:    bool,
        predicted_side: Optional[str] = None,
        implied_up_at_decision: Optional[float] = None,
    ):
        """Log resolution outcome at T=300."""
        self.fetcher.update_decision_outcome(window_start, resolved_up)

        matching = [d for d in self._decisions if d["window_start"] == window_start]
        realised_ev = None
        if matching:
            d = matching[0]
            d["resolved_up"] = resolved_up
            predicted_up = d["side"] == "UP"
            d["correct"]  = predicted_up == resolved_up
            if d["correct"] is not None:
                self._session_total   += 1
                self._session_correct += 1 if d["correct"] else 0

            # Realised EV: did we actually capture the edge?
            # If correct: we won (1 - bet_price), net = (1 - bet_price) - fee
            # If wrong: we lost bet_price
            bet_price = d.get("bet_price", 0.5)
            if d["correct"]:
                realised_ev = 1.0 - bet_price
            else:
                realised_ev = -bet_price
            d["realised_ev"] = round(realised_ev, 4)

        delta_pct  = (close_price - ref_price) / ref_price * 100
        win_dt     = datetime.utcfromtimestamp(window_start).strftime("%H:%M:%S")
        direction  = "UP  " if resolved_up else "DOWN"
        correct_str = ""
        if predicted_side is not None:
            correct = (predicted_side == "UP") == resolved_up
            correct_str = f" | {'✓ CORRECT' if correct else '✗ WRONG'}"
            if realised_ev is not None:
                correct_str += f" | realised_ev={realised_ev:+.3f}"

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
                "realised_ev":  realised_ev,
            }) + "\n")

    # ── Session statistics ─────────────────────────────────────────────────────

    def session_stats(self) -> Dict:
        total   = self._session_total
        correct = self._session_correct
        acc     = correct / total if total > 0 else 0.0

        resolved = [d for d in self._decisions if d.get("correct") is not None]

        # Split by whether we bet with or against the crowd
        with_crowd   = [d for d in resolved
                        if (d["side"] == "UP"   and d.get("implied_up", 0.5) >= 0.5) or
                           (d["side"] == "DOWN"  and d.get("implied_up", 0.5) < 0.5)]
        contra_crowd = [d for d in resolved if d not in with_crowd]

        hc = [d for d in resolved if abs(d.get("net_ev", 0)) >= HIGH_CONF_EV]
        hc_correct = sum(1 for d in hc if d.get("correct"))

        # Total realised EV this session
        total_ev = sum(d.get("realised_ev", 0) for d in resolved if d.get("realised_ev") is not None)

        return {
            "total":              total,
            "correct":            correct,
            "accuracy":           acc,
            "high_conf_total":    len(hc),
            "high_conf_correct":  hc_correct,
            "high_conf_acc":      hc_correct / len(hc) if hc else 0.0,
            "with_crowd_total":   len(with_crowd),
            "with_crowd_acc":     sum(1 for d in with_crowd if d.get("correct")) / len(with_crowd) if with_crowd else 0.0,
            "contra_crowd_total": len(contra_crowd),
            "contra_crowd_acc":   sum(1 for d in contra_crowd if d.get("correct")) / len(contra_crowd) if contra_crowd else 0.0,
            "total_realised_ev":  round(total_ev, 4),
            "elapsed_s":          int(time.time() - self._session_start),
        }

    # ── Terminal display ───────────────────────────────────────────────────────

    def print_dashboard(
        self,
        current_price:   float,
        reference_price: float,
        window_start:    int,
        time_in_window:  float,
        feeds_status:    Optional[Dict] = None,
        current_implied: float = 0.5,
        trader_status:   str = "",
    ):
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

        delta = (current_price - reference_price) / reference_price * 100 if reference_price > 0 else 0
        dir_color = "green" if delta >= 0 else "red"
        dir_sym   = "▲" if delta >= 0 else "▼"
        remaining = max(0, 300 - time_in_window)
        at_decision = abs(time_in_window - 150) < 5

        # ── Header ────────────────────────────────────────────────────────────
        header = Text()
        header.append("  BTC 5-MIN DECISION BOT  ", style="bold white on blue")
        header.append(f"  ${current_price:,.2f}  ", style=f"bold {dir_color}")
        header.append(f"{dir_sym} {delta:+.3f}%  ", style=f"bold {dir_color}")
        # Show current live market odds
        crowd_pct = int(current_implied * 100)
        crowd_color = "green" if current_implied >= 0.5 else "red"
        header.append(f"Market: [{crowd_color}]{crowd_pct}% UP / {100-crowd_pct}% DOWN[/{crowd_color}]")
        console.print(header)
        console.print()

        # ── Window timer ──────────────────────────────────────────────────────
        win_dt = datetime.utcfromtimestamp(window_start).strftime("%H:%M:%S")
        mins, secs = divmod(int(time_in_window), 60)
        console.print(
            f"  Window: [cyan]{win_dt} UTC[/]  |  "
            f"T+{mins:02d}:{secs:02d}  |  "
            f"Remaining: [yellow]{int(remaining)}s[/]"
            + ("  [bold magenta blink]<<< DECISION POINT >>>[/]" if at_decision else "")
        )
        console.print()

        # ── Feed status ───────────────────────────────────────────────────────
        if feeds_status:
            parts = []
            for feed, status in feeds_status.items():
                color = "green" if status else "red"
                icon  = "●" if status else "○"
                parts.append(f"[{color}]{icon} {feed}[/]")
            console.print("  " + "   ".join(parts))

        # ── Trader status ─────────────────────────────────────────────────────
        if trader_status:
            t_color = "green" if "LIVE" in trader_status else "yellow"
            console.print(f"  [{t_color}]💰 {trader_status}[/]")
        console.print()

        # ── Recent decisions ──────────────────────────────────────────────────
        if self._decisions:
            dec_table = Table(
                title="Recent Decisions  (edge = P_model − Market odds)",
                box=box.SIMPLE_HEAD,
                show_header=True,
                title_style="bold",
            )
            dec_table.add_column("Time",    style="cyan",  width=6)
            dec_table.add_column("Side",    style="bold",  width=5)
            dec_table.add_column("P_model", style="white", width=8)
            dec_table.add_column("Market",  style="white", width=8)
            dec_table.add_column("Edge",    style="bold",  width=8)
            dec_table.add_column("Net EV",  style="bold",  width=8)
            dec_table.add_column("@Price",  style="dim",   width=7)
            dec_table.add_column("Result",  style="bold",  width=14)
            dec_table.add_column("Models (LR/XGB/LSTM/BAY)", width=28)

            for d in reversed(self._decisions[-12:]):
                dt_str  = datetime.utcfromtimestamp(d["window_start"]).strftime("%H:%M")
                side    = d["side"]
                correct = d.get("correct")
                realised_ev = d.get("realised_ev")

                if correct is True:
                    rev_str = f"[green]✓ WIN  ev={realised_ev:+.3f}[/]" if realised_ev is not None else "[green]✓ WIN[/]"
                elif correct is False:
                    rev_str = f"[red]✗ LOSS ev={realised_ev:+.3f}[/]" if realised_ev is not None else "[red]✗ LOSS[/]"
                else:
                    rev_str = "[dim]pending[/]"

                side_color  = "green" if side == "UP" else "red"
                raw_edge    = d.get("raw_edge", 0.0)
                net_ev      = d.get("net_ev", 0.0)
                # Edge colour: positive = model more bullish than crowd (regardless of direction)
                edge_color  = "green" if abs(net_ev) >= HIGH_CONF_EV else (
                              "yellow" if net_ev > 0 else "red"
                )
                bet_price   = d.get("bet_price", 0.5)
                implied_up  = d.get("implied_up", 0.5)

                breakdown = (
                    f"{d['p_logistic']:.2f}/"
                    f"{d['p_xgboost']:.2f}/"
                    f"{d['p_lstm']:.2f}/"
                    f"{d['p_bayesian']:.2f}"
                )
                dec_table.add_row(
                    dt_str,
                    f"[{side_color}]{side}[/]",
                    f"{d['p_model_up']:.3f}",
                    f"{implied_up:.3f}",
                    f"[{edge_color}]{raw_edge:+.3f}[/]",
                    f"[{edge_color}]{net_ev:+.3f}[/]",
                    f"{bet_price:.3f}",
                    rev_str,
                    breakdown,
                )
            console.print(dec_table)
            console.print()

        # ── Session stats ─────────────────────────────────────────────────────
        stats = self.session_stats()
        acc_color = "green" if stats["accuracy"] >= 0.55 else (
            "yellow" if stats["accuracy"] >= 0.50 else "red"
        )
        ev_color = "green" if stats["total_realised_ev"] > 0 else "red"
        console.print(
            f"  Session: [bold]{stats['total']}[/] bets  |  "
            f"Accuracy: [{acc_color}]{stats['accuracy']:.1%}[/]  |  "
            f"Realised EV: [{ev_color}]{stats['total_realised_ev']:+.4f}[/]  |  "
            f"HC: {stats['high_conf_correct']}/{stats['high_conf_total']} "
            f"[{acc_color}]{stats['high_conf_acc']:.1%}[/]"
        )
        if stats["contra_crowd_total"] > 0:
            cc_color = "green" if stats["contra_crowd_acc"] >= 0.5 else "red"
            console.print(
                f"  Contra-crowd: {stats['contra_crowd_total']} bets  |  "
                f"Accuracy: [{cc_color}]{stats['contra_crowd_acc']:.1%}[/]  |  "
                f"With-crowd: {stats['with_crowd_total']} bets  |  "
                f"Accuracy: {stats['with_crowd_acc']:.1%}"
            )
        console.print()

    def _print_plain(self, current_price, reference_price, window_start, time_in_window):
        delta = (current_price - reference_price) / reference_price * 100 if reference_price > 0 else 0
        win_dt = datetime.utcfromtimestamp(window_start).strftime("%H:%M:%S")
        print(
            f"\r  ${current_price:,.2f} ({delta:+.3f}%) | "
            f"Window {win_dt} | T+{int(time_in_window)}s",
            end="", flush=True,
        )
