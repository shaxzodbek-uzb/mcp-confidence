"""Pure, no-I/O confidence gate for any model's text outputs.

Turns the per-token logprob list that an OpenAI-compatible API surfaces (e.g. in
``provider_details['logprobs']`` for pydantic-ai, or ``choices[0].logprobs.content``
for a raw OpenAI chat completion) into a normalized score and a 3-band
classification — a confidence signal you can route on (accept / verify / ask a
human). Works for any model that can return token logprobs: vLLM, Ollama,
llama.cpp, TGI, or any OpenAI-compatible endpoint.

CRITICAL — threshold calibration:
  The DEFAULT thresholds (HIGH=-1.5, LOW=-3.5 in MEAN LOG-PROBABILITY space,
  nats) are PROVISIONAL GUESSES. Naive 0.95/0.60 *probability* bands are
  calibrated for SHORT structured outputs and are almost certainly WRONG for
  longer free-text on most models (MoE routing and high-frequency function words
  pull the per-token mean toward 0, so exp(mean) sits at 0.05-0.20 even for
  fluent text — hence we band on MEAN LOGPROB in nats, not on a probability). DO
  NOT enable auto-accept routing until these are recalibrated on YOUR model's
  live output via the bundled calibration tool
  (``mcp-confidence calibrate`` / :mod:`mcp_confidence.calibrate`).

Logprob dict shape (OpenAI / pydantic-ai):
  [{"token": str, "bytes": list[int]|None, "logprob": float,
    "top_logprobs": [{"token": str, "bytes": list[int]|None, "logprob": float}]}]
Only the top-level "logprob" per token is consumed here. "top_logprobs" is left
untouched for a future self-consistency / verbalized signal.

This module has ZERO third-party imports and ZERO I/O — unit-testable in isolation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum


class ConfidenceBand(str, Enum):
    HIGH = "high"  # auto-accept: use the model's output directly
    MID = "mid"  # verify: hand off to a stronger model / supervisor check
    LOW = "low"  # ask a human: pause and request clarification


@dataclass(frozen=True, slots=True)
class ConfidenceResult:
    """Immutable result of one confidence computation.

    score:        Combined scalar in log-probability space (negative reals).
                  Higher (less negative) = more confident. NOT a probability.
    band:         HIGH / MID / LOW.
    mean_logprob: Arithmetic mean per-token logprob. exp(mean_logprob) =
                  geometric-mean token probability. THIS is the field the
                  calibration tool reads.
    min_logprob:  Worst per-token logprob, clamped to min_token_floor.
    token_count:  Tokens used (after guard).
    logprobs_available: False when provider_details was None/empty; numeric
                  fields are NaN and band is MID (conservative).
    """

    score: float
    band: ConfidenceBand
    mean_logprob: float
    min_logprob: float
    token_count: int
    logprobs_available: bool


# Returned when logprobs are absent (empty .content, or a model/path that does
# not emit logprobs). band=MID is the conservative default: when routing is on, a
# missing signal goes to verify, never to blind auto-accept.
UNAVAILABLE = ConfidenceResult(
    score=float("nan"),
    band=ConfidenceBand.MID,
    mean_logprob=float("nan"),
    min_logprob=float("nan"),
    token_count=0,
    logprobs_available=False,
)


def extract_logprobs(provider_details: dict | None) -> list[float] | None:
    """Flat per-token logprob sequence from a provider_details dict.

    Returns None if provider_details is None, the 'logprobs' key is absent, the
    list is empty, or every entry has a non-numeric logprob. Some servers return
    an empty .content list, in which case the 'logprobs' key may be omitted
    entirely — that maps to None here.
    """
    if not provider_details:
        return None
    raw = provider_details.get("logprobs")
    if not raw:
        return None
    lps = [
        e["logprob"]
        for e in raw
        if isinstance(e.get("logprob"), (int, float)) and not isinstance(e.get("logprob"), bool)
    ]
    return lps if lps else None


def mean_logprob(logprobs: Sequence[float]) -> float:
    """Arithmetic mean token logprob (nats). -inf for an empty sequence."""
    if not logprobs:
        return float("-inf")
    return sum(logprobs) / len(logprobs)


def min_logprob(logprobs: Sequence[float], floor: float = -10.0) -> float:
    """Worst per-token logprob, clamped to `floor`. The floor stops a single
    ultra-rare token (e.g. a numeric ID) from collapsing the score and
    mis-classifying an otherwise fluent output. Returns floor for empty input."""
    if not logprobs:
        return floor
    return max(min(logprobs), floor)


def combined_score(mean_lp: float, min_lp: float, min_weight: float = 0.3) -> float:
    """score = (1 - min_weight) * mean_lp + min_weight * min_lp.

    Both inputs are negative reals; output is too. Raises ValueError on a
    min_weight outside [0, 1]. Propagates -inf if either input is infinite."""
    if not (0.0 <= min_weight <= 1.0):
        raise ValueError(f"min_weight must be in [0, 1], got {min_weight}")
    if math.isinf(mean_lp) or math.isinf(min_lp):
        return float("-inf")
    return (1.0 - min_weight) * mean_lp + min_weight * min_lp


def classify(
    score: float,
    high_threshold: float = -1.5,
    low_threshold: float = -3.5,
) -> ConfidenceBand:
    """Map a combined score to a band.

      HIGH: score >= high_threshold
      LOW:  score <= low_threshold
      MID:  otherwise (also for NaN / -inf — conservative)

    Raises ValueError if high_threshold <= low_threshold (misconfig guard)."""
    if high_threshold <= low_threshold:
        raise ValueError(
            f"high_threshold ({high_threshold}) must be > low_threshold ({low_threshold})"
        )
    if math.isnan(score) or math.isinf(score):
        return ConfidenceBand.MID
    if score >= high_threshold:
        return ConfidenceBand.HIGH
    if score <= low_threshold:
        return ConfidenceBand.LOW
    return ConfidenceBand.MID


def compute(
    provider_details: dict | None,
    high_threshold: float = -1.5,
    low_threshold: float = -3.5,
    min_weight: float = 0.3,
    min_token_floor: float = -10.0,
) -> ConfidenceResult:
    """Top-level entry point: provider_details -> ConfidenceResult.

    The only function a caller needs to go from a logprob-bearing
    provider_details dict to a banded result. Returns UNAVAILABLE (band=MID)
    when logprobs are absent."""
    lps = extract_logprobs(provider_details)
    if lps is None:
        return UNAVAILABLE
    mean_lp = mean_logprob(lps)
    min_lp = min_logprob(lps, floor=min_token_floor)
    score = combined_score(mean_lp, min_lp, min_weight)
    band = classify(score, high_threshold, low_threshold)
    return ConfidenceResult(
        score=score,
        band=band,
        mean_logprob=mean_lp,
        min_logprob=min_lp,
        token_count=len(lps),
        logprobs_available=True,
    )
