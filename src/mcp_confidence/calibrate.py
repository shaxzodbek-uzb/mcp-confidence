"""Calibrate the confidence-gate thresholds from collected audit data.

The gate's HIGH/LOW thresholds (:class:`~mcp_confidence.config.GateConfig`, in
MEAN TOKEN LOG-PROBABILITY space) ship as PROVISIONAL GUESSES — naive probability
bands are wrong for free-text by ~1-2 orders of magnitude and are model-specific.
This module turns a JSONL log of confidence events into the numbers needed to
choose real thresholds, instead of guessing again.

Two analyses, both PURE (they return data; printing lives in
:mod:`mcp_confidence.cli`):

  1. DISTRIBUTION (always): percentiles of ``mean_logprob`` / ``score``, the band
     split, and (via ``load_events``) the logprobs-unavailable rate. Run this
     first to see where the mass of your traffic actually sits before picking any
     threshold.

  2. RISK-COVERAGE (when labels exist): if rows carry a ``human_label`` of
     "good"/"bad", :func:`risk_coverage` sweeps a (high, low) grid over the
     observed values and returns the cell minimizing ``(risk_high + false_ask)``
     subject to ``coverage_high >= min_accept_cov`` and
     ``coverage_low <= max_ask_cov``.

Each JSONL row is expected to carry at least: an event name field (``event``),
``logprobs_available``, and the metric being calibrated (``mean_logprob`` or
``score``). The risk-coverage step additionally needs ``human_label``.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path


def load_events(
    path: str | Path,
    event_name: str | None = "worker_delegated",
) -> tuple[list[dict], int, int]:
    """Read confidence events with logprobs available from a JSONL audit file.

    Args:
        path: JSONL file path.
        event_name: if given, keep only rows whose ``event`` field equals this;
            if None, accept any row regardless of (or missing) an ``event`` field.

    Returns:
        ``(rows, total, unavailable)`` where ``rows`` are the matching events
        that have a truthy ``logprobs_available``, ``total`` is the count of
        matching events (available or not), and ``unavailable`` is how many of
        those lacked logprobs. The caller can report the unavailable rate and the
        ">20% unavailable" warning.
    """
    rows: list[dict] = []
    total = 0
    unavailable = 0
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(e, dict):
                continue
            if event_name is not None and e.get("event") != event_name:
                continue
            total += 1
            if not e.get("logprobs_available"):
                unavailable += 1
                continue
            rows.append(e)
    return rows, total, unavailable


def _pct(values: list[float], p: float) -> float:
    s = sorted(values)
    if not s:
        return float("nan")
    k = max(0, min(len(s) - 1, round(p / 100 * (len(s) - 1))))
    return s[k]


def distribution(rows: list[dict], metric: str) -> dict:
    """Summary statistics of one metric across rows.

    Returns a dict with ``n``, ``min``, ``p10``, ``p25``, ``median``, ``mean``,
    ``p75``, ``p90``, ``max`` (all NaN when there is no data) and ``band_split``,
    a ``{band: count}`` mapping over the rows' ``band`` field.
    """
    vals = [r[metric] for r in rows if isinstance(r.get(metric), (int, float))]
    bands: dict[str, int] = {}
    for r in rows:
        b = r.get("band", "?")
        bands[b] = bands.get(b, 0) + 1
    if not vals:
        nan = float("nan")
        return {
            "n": 0,
            "min": nan,
            "p10": nan,
            "p25": nan,
            "median": nan,
            "mean": nan,
            "p75": nan,
            "p90": nan,
            "max": nan,
            "band_split": bands,
        }
    return {
        "n": len(vals),
        "min": min(vals),
        "p10": _pct(vals, 10),
        "p25": _pct(vals, 25),
        "median": statistics.median(vals),
        "mean": statistics.fmean(vals),
        "p75": _pct(vals, 75),
        "p90": _pct(vals, 90),
        "max": max(vals),
        "band_split": bands,
    }


def risk_coverage(
    rows: list[dict],
    metric: str,
    min_accept_cov: float,
    max_ask_cov: float,
) -> dict | None:
    """Recommend (high, low) thresholds via a labeled risk-coverage sweep.

    Considers only rows that carry a ``human_label`` of "good"/"bad" and a
    numeric ``metric``. Sweeps every observed value as a candidate boundary and
    returns the (high, low) cell that minimizes ``(risk_high + false_ask)``
    subject to ``coverage_high >= min_accept_cov`` and
    ``coverage_low <= max_ask_cov``.

    Returns a dict ``{high, low, cov_hi, cov_lo, risk_hi, false_ask, cost}`` or
    None when there are no labeled rows, no good/bad split, or no feasible cell.
    """
    labeled = [
        r
        for r in rows
        if r.get("human_label") in ("good", "bad") and isinstance(r.get(metric), (int, float))
    ]
    if not labeled:
        return None

    vals = sorted(r[metric] for r in labeled)
    n = len(labeled)
    n_bad = sum(1 for r in labeled if r["human_label"] == "bad")
    n_good = n - n_bad

    # Candidate thresholds = the observed values, rounded for a coarser grid.
    candidates = sorted(set(round(v, 2) for v in vals))
    best = None
    for hi in candidates:
        for lo in candidates:
            if lo >= hi:
                continue
            accept = [r for r in labeled if r[metric] >= hi]
            ask = [r for r in labeled if r[metric] <= lo]
            cov_hi = len(accept) / n
            cov_lo = len(ask) / n
            risk_hi = sum(1 for r in accept if r["human_label"] == "bad") / n_bad if n_bad else 0.0
            false_ask = (
                sum(1 for r in ask if r["human_label"] == "good") / n_good if n_good else 0.0
            )
            if cov_hi < min_accept_cov or cov_lo > max_ask_cov:
                continue
            cost = risk_hi + false_ask
            if best is None or cost < best["cost"]:
                best = {
                    "high": hi,
                    "low": lo,
                    "cov_hi": cov_hi,
                    "cov_lo": cov_lo,
                    "risk_hi": risk_hi,
                    "false_ask": false_ask,
                    "cost": cost,
                }
    return best
