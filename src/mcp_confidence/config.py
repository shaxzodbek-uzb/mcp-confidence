"""Configuration for the confidence gate — a frozen, env-overridable dataclass.

GateConfig carries the thresholds and weights that :mod:`mcp_confidence.core`
applies. The defaults are the same PROVISIONAL GUESSES as core (HIGH=-1.5,
LOW=-3.5 in mean-logprob nats) and MUST be calibrated on your own model's output
before you trust auto-accept routing — see :mod:`mcp_confidence.calibrate`.

Stdlib only: ``from_env`` reads plain ``os.environ`` (no pydantic, no dotenv).
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GateConfig:
    """Immutable gate configuration.

    high_threshold:  score >= this -> HIGH (auto-accept). Default -1.5 (GUESS).
    low_threshold:   score <= this -> LOW (ask a human). Default -3.5 (GUESS).
    min_weight:      weakest-link weight in [0, 1] for the combined score.
    min_token_floor: clamp applied to the worst per-token logprob.
    top_k:           top_logprobs to request from the OpenAI-compatible server.

    The self-consistency fallback (:mod:`mcp_confidence.consistency`) has its own
    thresholds, on a completely different scale — agreement in [0, 1], not nats.
    They are separate fields precisely so nobody mistakes one for the other, and
    they are PROVISIONAL GUESSES too.

    high_agreement:  agreement >= this -> HIGH. Default 0.8 (GUESS).
    low_agreement:   agreement <= this -> LOW. Default 0.5 (GUESS).
    samples:         how many completions to sample for a consistency check.

    Raises ValueError if either high threshold is not above its low counterpart,
    if min_weight is outside [0, 1], or if samples < 2.
    """

    high_threshold: float = -1.5
    low_threshold: float = -3.5
    min_weight: float = 0.3
    min_token_floor: float = -10.0
    top_k: int = 5
    high_agreement: float = 0.8
    low_agreement: float = 0.5
    samples: int = 5

    def __post_init__(self) -> None:
        if self.high_threshold <= self.low_threshold:
            raise ValueError(
                f"high_threshold ({self.high_threshold}) must be > "
                f"low_threshold ({self.low_threshold})"
            )
        if not (0.0 <= self.min_weight <= 1.0):
            raise ValueError(f"min_weight must be in [0, 1], got {self.min_weight}")
        if self.high_agreement <= self.low_agreement:
            raise ValueError(
                f"high_agreement ({self.high_agreement}) must be > "
                f"low_agreement ({self.low_agreement})"
            )
        for name in ("high_agreement", "low_agreement"):
            value = getattr(self, name)
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if self.samples < 2:
            # One sample cannot disagree with itself; a "consistency" check over a
            # single completion would report a fabricated 1.0.
            raise ValueError(f"samples must be >= 2, got {self.samples}")

    @classmethod
    def from_env(cls, prefix: str = "MCP_CONFIDENCE_") -> GateConfig:
        """Build a GateConfig from environment variables.

        Reads ``{prefix}HIGH_THRESHOLD``, ``{prefix}LOW_THRESHOLD``,
        ``{prefix}MIN_WEIGHT``, ``{prefix}MIN_TOKEN_FLOOR``, ``{prefix}TOP_K``,
        ``{prefix}HIGH_AGREEMENT``, ``{prefix}LOW_AGREEMENT`` and
        ``{prefix}SAMPLES`` from ``os.environ``. Any missing variable falls back to
        the field default. Validation still runs via ``__post_init__``.
        """
        defaults = cls()

        def _float(name: str, fallback: float) -> float:
            raw = os.environ.get(prefix + name)
            return float(raw) if raw is not None and raw != "" else fallback

        def _int(name: str, fallback: int) -> int:
            raw = os.environ.get(prefix + name)
            return int(raw) if raw is not None and raw != "" else fallback

        return cls(
            high_threshold=_float("HIGH_THRESHOLD", defaults.high_threshold),
            low_threshold=_float("LOW_THRESHOLD", defaults.low_threshold),
            min_weight=_float("MIN_WEIGHT", defaults.min_weight),
            min_token_floor=_float("MIN_TOKEN_FLOOR", defaults.min_token_floor),
            top_k=_int("TOP_K", defaults.top_k),
            high_agreement=_float("HIGH_AGREEMENT", defaults.high_agreement),
            low_agreement=_float("LOW_AGREEMENT", defaults.low_agreement),
            samples=_int("SAMPLES", defaults.samples),
        )
