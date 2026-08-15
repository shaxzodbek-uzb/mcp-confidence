"""Self-consistency: a confidence signal for models that don't return logprobs.

:mod:`mcp_confidence.core` reads the model's own token probabilities. That is the
better signal — it is free, it needs one sample, and it reflects what the model
actually computed. But plenty of endpoints never expose it: most hosted reasoning
models, several managed gateways, and any provider that strips ``logprobs``. On those
the core gate can only return the conservative ``MID``, which is safe and useless.

This module supplies the fallback: sample the same prompt ``n`` times at a non-zero
temperature and measure **how much the answers agree with each other**. A model that
returns the same answer five times out of five is telling you something; a model that
returns five different answers is telling you something else.

CRITICAL — the thresholds here are PROVISIONAL GUESSES, exactly like the logprob ones.
``HIGH_AGREEMENT=0.8`` / ``LOW_AGREEMENT=0.5`` are starting points, not validated
values. Agreement distributions depend on your model, your temperature, your ``n``, and
the shape of your task. Recalibrate before you route on them.

What this signal is **not**:

- **Not a correctness measure.** It measures *stability*, not truth. A model that is
  confidently and consistently wrong scores as HIGH. Self-consistency catches the
  model being unsure; it cannot catch the model being wrong in the same way every time.
- **Not comparable to the logprob score.** The scales are unrelated — one is agreement
  in ``[0, 1]``, the other is mean logprob in nats. Never average or compare them; pick
  one signal per decision. :meth:`~mcp_confidence.gate.Gate.assess` prefers logprobs and
  uses this only when they are absent.
- **Not free.** It costs ``n`` completions instead of one. That is the price of a
  confidence signal on a model that won't give you its logprobs.

Zero third-party imports and zero I/O — sampling is the caller's job, this module only
scores what you hand it.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher

from .core import ConfidenceBand

#: Pairwise similarity is O(n²) in the sample count; past this the cost stops being
#: worth it and the caller has almost certainly made a mistake.
MAX_SAMPLES = 50

#: In ``auto`` mode, samples at or under this many word tokens are compared exactly.
#: Short answers are classifications, extractions, dates and numbers, where "same
#: answer" is precisely the question and fuzzy similarity would call "42" and "43"
#: half-equal. Counted in tokens rather than characters because a short *sentence*
#: ("The capital of France is Paris.") is prose, not a label, and belongs in the
#: similarity branch even though it is only thirty characters long.
SHORT_ANSWER_TOKENS = 4

HIGH_AGREEMENT = 0.8  # PROVISIONAL GUESS — calibrate on your own model
LOW_AGREEMENT = 0.5  # PROVISIONAL GUESS — calibrate on your own model

_WORD = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class ConsistencyResult:
    """Immutable result of one self-consistency computation.

    agreement:    Stability in ``[0, 1]``. Higher = the samples agree more.
    band:         HIGH / MID / LOW, from the agreement thresholds.
    n_samples:    How many usable samples were scored.
    method:       ``"exact"`` or ``"similarity"`` — which comparison ran.
    modal_answer: The most common answer (``exact`` only; ``None`` otherwise).
                  Returned verbatim as first seen, not normalised.
    modal_count:  How many samples matched ``modal_answer`` (``exact`` only).
    """

    agreement: float
    band: ConfidenceBand
    n_samples: int
    method: str
    modal_answer: str | None = None
    modal_count: int = 0


#: Returned when there is nothing to measure — no samples, or all of them blank.
#: band=MID mirrors :data:`mcp_confidence.core.UNAVAILABLE`: a missing signal routes to
#: verify, never to blind auto-accept.
UNAVAILABLE = ConsistencyResult(
    agreement=float("nan"),
    band=ConfidenceBand.MID,
    n_samples=0,
    method="none",
)


def normalise(text: str) -> str:
    """Fold a sample to its comparable form for exact matching.

    Unicode NFKC, casefold, collapsed whitespace, and stripped leading/trailing
    punctuation — so ``"Positive."``, ``"positive"`` and ``" POSITIVE "`` are one
    answer. Internal punctuation is preserved: ``"3.14"`` must not become ``"314"``.
    """
    text = unicodedata.normalize("NFKC", text).casefold()
    text = " ".join(text.split())
    return text.strip(" \t\n\r.,;:!?\"'`()[]{}")


def tokenise(text: str) -> list[str]:
    """Word tokens for similarity comparison, normalised the same way."""
    return _WORD.findall(normalise(text))


def _usable(samples: Sequence[str]) -> list[str]:
    if len(samples) > MAX_SAMPLES:
        raise ValueError(
            f"self-consistency scores at most {MAX_SAMPLES} samples "
            f"(pairwise comparison is quadratic); got {len(samples)}"
        )
    return [s for s in samples if isinstance(s, str) and s.strip()]


def exact_agreement(samples: Sequence[str]) -> tuple[float, str | None, int]:
    """Share of samples equal to the most common one, after :func:`normalise`.

    Returns ``(agreement, modal_answer, modal_count)``. With one sample the
    agreement is ``1.0`` — which is honest arithmetic and meaningless as a signal;
    :func:`score` refuses a single sample for that reason.
    """
    usable = _usable(samples)
    if not usable:
        return float("nan"), None, 0
    counts = Counter(normalise(s) for s in usable)
    modal_key, modal_count = counts.most_common(1)[0]
    # Report the original text, not the folded key — the caller wants the answer.
    modal_answer = next(s for s in usable if normalise(s) == modal_key)
    return modal_count / len(usable), modal_answer, modal_count


def similarity_agreement(samples: Sequence[str]) -> float:
    """Mean pairwise similarity of the samples, in ``[0, 1]``.

    Compares **token sequences** rather than characters: it is both faster and a
    better match for the question ("did it say the same thing?" rather than "did it
    type the same letters?"). ``autojunk`` is disabled because its heuristic treats
    frequent elements as noise once a sequence passes 200 items, which silently
    distorts the ratio on longer answers.
    """
    usable = _usable(samples)
    if len(usable) < 2:
        return float("nan") if not usable else 1.0
    tokens = [tokenise(s) for s in usable]
    ratios: list[float] = []
    for i in range(len(tokens)):
        for j in range(i + 1, len(tokens)):
            ratios.append(SequenceMatcher(None, tokens[i], tokens[j], autojunk=False).ratio())
    return sum(ratios) / len(ratios)


def classify_agreement(
    agreement: float,
    high_agreement: float = HIGH_AGREEMENT,
    low_agreement: float = LOW_AGREEMENT,
) -> ConfidenceBand:
    """Map an agreement in ``[0, 1]`` to a band.

    HIGH: ``>= high_agreement``. LOW: ``<= low_agreement``. MID otherwise, and for
    NaN — conservative, matching :func:`mcp_confidence.core.classify`.

    Raises ValueError if ``high_agreement <= low_agreement``.
    """
    if high_agreement <= low_agreement:
        raise ValueError(
            f"high_agreement ({high_agreement}) must be > low_agreement ({low_agreement})"
        )
    if agreement != agreement:  # NaN
        return ConfidenceBand.MID
    if agreement >= high_agreement:
        return ConfidenceBand.HIGH
    if agreement <= low_agreement:
        return ConfidenceBand.LOW
    return ConfidenceBand.MID


def choose_method(samples: Sequence[str]) -> str:
    """Pick ``"exact"`` or ``"similarity"`` for ``auto`` mode.

    Exact when every sample is at most :data:`SHORT_ANSWER_TOKENS` words — the
    classification / extraction / numeric case, where "same answer" is exactly the
    question. Similarity otherwise, because two correct paraphrases of a sentence
    should not score as total disagreement.

    Deliberately a length rule and nothing cleverer: a caller who needs a different
    trade-off should pass ``method=`` explicitly rather than reverse-engineer a
    heuristic.
    """
    usable = _usable(samples)
    if not usable:
        return "exact"
    return "exact" if all(len(tokenise(s)) <= SHORT_ANSWER_TOKENS for s in usable) else "similarity"


def score(
    samples: Sequence[str],
    *,
    method: str = "auto",
    high_agreement: float = HIGH_AGREEMENT,
    low_agreement: float = LOW_AGREEMENT,
) -> ConsistencyResult:
    """Score how much a set of samples of the same prompt agree.

    Args:
        samples: Independently sampled answers to one prompt. Blank entries and
            non-strings are dropped. Fewer than two usable samples yields
            :data:`UNAVAILABLE` — one sample cannot disagree with itself, so a
            "1.0 agreement" there would be a fabricated signal, not a weak one.
        method: ``"auto"`` (see :func:`choose_method`), ``"exact"``, or
            ``"similarity"``.
        high_agreement: ``>=`` this is HIGH. PROVISIONAL default.
        low_agreement: ``<=`` this is LOW. PROVISIONAL default.

    Raises:
        ValueError: On an unknown ``method``, more than :data:`MAX_SAMPLES`
            samples, or inverted thresholds.
    """
    if method not in ("auto", "exact", "similarity"):
        raise ValueError(f"method must be 'auto', 'exact' or 'similarity', got {method!r}")

    usable = _usable(samples)
    if len(usable) < 2:
        return UNAVAILABLE

    resolved = choose_method(usable) if method == "auto" else method
    if resolved == "exact":
        agreement, modal_answer, modal_count = exact_agreement(usable)
    else:
        agreement, modal_answer, modal_count = similarity_agreement(usable), None, 0

    return ConsistencyResult(
        agreement=agreement,
        band=classify_agreement(agreement, high_agreement, low_agreement),
        n_samples=len(usable),
        method=resolved,
        modal_answer=modal_answer,
        modal_count=modal_count,
    )


__all__ = [
    "ConsistencyResult",
    "UNAVAILABLE",
    "MAX_SAMPLES",
    "SHORT_ANSWER_TOKENS",
    "HIGH_AGREEMENT",
    "LOW_AGREEMENT",
    "normalise",
    "tokenise",
    "exact_agreement",
    "similarity_agreement",
    "classify_agreement",
    "choose_method",
    "score",
]
