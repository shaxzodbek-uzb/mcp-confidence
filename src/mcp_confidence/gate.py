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

For models that never expose logprobs, :meth:`Gate.from_samples` scores repeated
samples of the same prompt instead, and :meth:`Gate.assess` combines the two: it
prefers logprobs and falls back to self-consistency only when they are missing —
which is the case the core gate could previously only answer with a blanket MID.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from . import consistency, core
from .adapters.openai import to_provider_details
from .config import GateConfig
from .consistency import ConsistencyResult
from .core import ConfidenceBand, ConfidenceResult


@dataclass(frozen=True, slots=True)
class Assessment:
    """A band plus the evidence behind it.

    The two signals live on unrelated scales — mean logprob in nats versus agreement
    in [0, 1] — so they are never averaged or compared. ``signal`` records which one
    actually decided the band, and both sub-results are kept for logging.

    band:        The routing decision: HIGH / MID / LOW.
    signal:      ``"logprobs"``, ``"self-consistency"``, or ``"none"``.
    logprobs:    The logprob result, when one was computed.
    consistency: The self-consistency result, when one was computed.
    """

    band: ConfidenceBand
    signal: str
    logprobs: ConfidenceResult | None = None
    consistency: ConsistencyResult | None = None


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

    def from_samples(self, samples: Sequence[str], *, method: str = "auto") -> ConsistencyResult:
        """Score repeated samples of one prompt with this gate's agreement thresholds.

        The fallback signal for models that don't return logprobs. Sampling is the
        caller's job: run the same prompt ``config.samples`` times at a non-zero
        temperature and pass the answers here.

        Remember what it measures — *stability*, not correctness. A model that is
        confidently wrong the same way five times scores HIGH.
        """
        return consistency.score(
            samples,
            method=method,
            high_agreement=self.config.high_agreement,
            low_agreement=self.config.low_agreement,
        )

    def assess(
        self,
        *,
        provider_details: dict | None = None,
        samples: Sequence[str] | None = None,
        method: str = "auto",
    ) -> Assessment:
        """Band a response from whichever signal is available.

        Logprobs win when present: they are free, they need one sample, and they
        reflect what the model actually computed. Self-consistency is the fallback
        for endpoints that strip logprobs — the case where the core gate can only
        return a conservative MID that tells you nothing.

        With neither signal the band is MID and ``signal`` is ``"none"``, so a caller
        can tell "the model was unsure" apart from "we never measured anything".
        """
        logprob_result = (
            self.from_provider_details(provider_details) if provider_details is not None else None
        )
        if logprob_result is not None and logprob_result.logprobs_available:
            return Assessment(band=logprob_result.band, signal="logprobs", logprobs=logprob_result)

        consistency_result = (
            self.from_samples(samples, method=method) if samples is not None else None
        )
        if consistency_result is not None and consistency_result.n_samples >= 2:
            return Assessment(
                band=consistency_result.band,
                signal="self-consistency",
                logprobs=logprob_result,
                consistency=consistency_result,
            )

        return Assessment(
            band=ConfidenceBand.MID,
            signal="none",
            logprobs=logprob_result,
            consistency=consistency_result,
        )
