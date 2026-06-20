"""The Gate facade — a configured, reusable entry point to the core math.

A :class:`Gate` bundles a :class:`~mcp_confidence.config.GateConfig` with the
pure functions in :mod:`mcp_confidence.core`, so callers do not have to thread
thresholds through every call. It accepts logprobs in the three shapes you are
likely to have:

  * a raw ``Sequence[float]`` of per-token logprobs (``from_logprobs``),
  * a pydantic-ai ``provider_details`` dict (``from_provider_details`` /
    ``from_dict``),
  * an OpenAI chat completion, as a dict OR an SDK object (``from_openai``).

All paths apply this gate's thresholds and weights and return a
:class:`~mcp_confidence.core.ConfidenceResult`.
"""

from __future__ import annotations

from collections.abc import Sequence

from . import core
from .adapters.openai import to_provider_details
from .config import GateConfig
from .core import ConfidenceResult


class Gate:
    """A confidence gate configured once and reused across calls."""

    def __init__(self, config: GateConfig | None = None) -> None:
        self.config = config or GateConfig()

    def from_logprobs(self, logprobs: Sequence[float]) -> ConfidenceResult:
        """Compute a result directly from a per-token logprob sequence.

        Builds the mean/min/score/classify path directly (no synthetic dict).
        An empty sequence yields the conservative UNAVAILABLE-style band MID via
        the -inf propagation in the core math.
        """
        cfg = self.config
        if not logprobs:
            return core.UNAVAILABLE
        mean_lp = core.mean_logprob(logprobs)
        min_lp = core.min_logprob(logprobs, floor=cfg.min_token_floor)
        score = core.combined_score(mean_lp, min_lp, cfg.min_weight)
        band = core.classify(score, cfg.high_threshold, cfg.low_threshold)
        return ConfidenceResult(
            score=score,
            band=band,
            mean_logprob=mean_lp,
            min_logprob=min_lp,
            token_count=len(logprobs),
            logprobs_available=True,
        )

    def from_provider_details(self, pd: dict | None) -> ConfidenceResult:
        """Compute a result from a pydantic-ai ``provider_details`` dict."""
        return core.compute(
            pd,
            high_threshold=self.config.high_threshold,
            low_threshold=self.config.low_threshold,
            min_weight=self.config.min_weight,
            min_token_floor=self.config.min_token_floor,
        )

    def from_openai(self, response) -> ConfidenceResult:
        """Compute a result from an OpenAI chat completion (dict OR SDK object).

        Missing/empty logprobs yield UNAVAILABLE (band MID); never raises on a
        malformed response.
        """
        return self.from_provider_details(to_provider_details(response))

    def from_dict(self, pd: dict | None) -> ConfidenceResult:
        """Alias of :meth:`from_provider_details`."""
        return self.from_provider_details(pd)
