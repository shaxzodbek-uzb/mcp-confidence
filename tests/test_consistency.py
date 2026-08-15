"""Tests for the self-consistency fallback signal.

Pure and offline, like the core gate: nothing here samples a model, it only scores
samples that are handed to it.
"""

from __future__ import annotations

import math

import pytest

from mcp_confidence import ConfidenceBand, Gate, GateConfig
from mcp_confidence.consistency import (
    MAX_SAMPLES,
    SHORT_ANSWER_TOKENS,
    UNAVAILABLE,
    choose_method,
    classify_agreement,
    exact_agreement,
    normalise,
    score,
    similarity_agreement,
    tokenise,
)

CFG = GateConfig()


# -- normalisation -----------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Positive.", "positive"),
        ("  POSITIVE  ", "positive"),
        ("positive!", "positive"),
        ("the  quick   brown", "the quick brown"),
        ('"quoted"', "quoted"),
        ("(parenthetical)", "parenthetical"),
    ],
)
def test_normalise_folds_incidental_differences(raw, expected):
    assert normalise(raw) == expected


def test_normalise_preserves_internal_punctuation():
    """3.14 must not collapse to 314 — that would make different numbers equal."""
    assert normalise("3.14") == "3.14"
    assert normalise("a,b") == "a,b"


def test_normalise_applies_unicode_nfkc():
    assert normalise("ﬁt") == "fit"


def test_tokenise_splits_on_words():
    assert tokenise("The quick, brown fox!") == ["the", "quick", "brown", "fox"]


# -- exact agreement ---------------------------------------------------------


def test_unanimous_samples_agree_completely():
    agreement, modal, count = exact_agreement(["yes", "yes", "yes"])
    assert agreement == 1.0
    assert (modal, count) == ("yes", 3)


def test_modal_share_is_the_agreement():
    agreement, modal, count = exact_agreement(["yes", "yes", "no", "maybe"])
    assert agreement == 0.5
    assert (modal, count) == ("yes", 2)


def test_all_distinct_samples_give_the_floor():
    agreement, _, count = exact_agreement(["a", "b", "c", "d"])
    assert agreement == 0.25
    assert count == 1


def test_modal_answer_is_returned_verbatim_not_normalised():
    _, modal, _ = exact_agreement(["Positive.", "positive", "negative"])
    assert modal == "Positive."


def test_normalisation_merges_cosmetic_variants():
    agreement, _, count = exact_agreement(["Positive.", " positive ", "POSITIVE"])
    assert agreement == 1.0
    assert count == 3


def test_blank_samples_are_dropped():
    agreement, _, count = exact_agreement(["yes", "", "   ", "yes"])
    assert agreement == 1.0
    assert count == 2


def test_no_usable_samples_is_nan():
    agreement, modal, count = exact_agreement(["", "  "])
    assert math.isnan(agreement)
    assert (modal, count) == (None, 0)


# -- similarity agreement ----------------------------------------------------


def test_identical_prose_is_fully_similar():
    text = "The capital of France is Paris, a city on the Seine."
    assert similarity_agreement([text, text, text]) == 1.0


def test_paraphrases_score_between_the_extremes():
    agreement = similarity_agreement(
        [
            "The capital of France is Paris.",
            "France's capital is Paris.",
            "Paris is the capital of France.",
        ]
    )
    assert 0.3 < agreement < 1.0


def test_unrelated_answers_score_low():
    agreement = similarity_agreement(
        [
            "The mitochondrion is the powerhouse of the cell.",
            "Quarterly revenue rose twelve percent year over year.",
            "Rain is expected across the northern provinces tomorrow.",
        ]
    )
    assert agreement < 0.3


def test_similarity_compares_tokens_not_characters():
    """'cat' vs 'car' share letters but are different words — and different answers."""
    assert similarity_agreement(["cat", "car"]) == 0.0


def test_similarity_is_symmetric_in_sample_order():
    a = ["alpha beta gamma", "alpha beta delta", "epsilon zeta"]
    assert similarity_agreement(a) == pytest.approx(similarity_agreement(list(reversed(a))))


def test_long_answers_are_not_distorted_by_the_autojunk_heuristic():
    """SequenceMatcher's autojunk drops frequent elements past 200 items; we disable it."""
    long_text = " ".join(["word"] * 300)
    assert similarity_agreement([long_text, long_text]) == 1.0


# -- banding -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("agreement", "expected"),
    [
        (1.0, ConfidenceBand.HIGH),
        (0.8, ConfidenceBand.HIGH),
        (0.79, ConfidenceBand.MID),
        (0.51, ConfidenceBand.MID),
        (0.5, ConfidenceBand.LOW),
        (0.0, ConfidenceBand.LOW),
    ],
)
def test_classify_agreement_boundaries(agreement, expected):
    assert classify_agreement(agreement) == expected


def test_nan_agreement_bands_mid_not_high():
    """A missing signal must route to verify, never to blind auto-accept."""
    assert classify_agreement(float("nan")) == ConfidenceBand.MID


def test_inverted_agreement_thresholds_are_rejected():
    with pytest.raises(ValueError, match="must be > "):
        classify_agreement(0.9, high_agreement=0.3, low_agreement=0.7)


# -- method selection --------------------------------------------------------


@pytest.mark.parametrize(
    "samples",
    [
        ["positive", "negative", "neutral"],
        ["3.14", "3.15"],
        ["2026-08-15", "2026-08-16"],
        ["John Michael Smith", "Jane Smith"],
    ],
)
def test_label_shaped_answers_use_exact_matching(samples):
    assert choose_method(samples) == "exact"


def test_prose_uses_similarity():
    assert choose_method(["a much longer prose answer " * 5, "another one entirely"]) == (
        "similarity"
    )


def test_a_short_sentence_is_prose_not_a_label():
    """31 characters, but six words — paraphrases of it must not read as disagreement."""
    assert (
        choose_method(["The capital of France is Paris.", "Paris is the capital of France."])
        == "similarity"
    )


def test_one_long_answer_switches_the_whole_set_to_similarity():
    assert choose_method(["yes", "a much longer prose answer " * 5]) == "similarity"


# -- score() -----------------------------------------------------------------


def test_score_reports_which_method_ran():
    assert score(["yes", "yes"]).method == "exact"
    assert score(["a " * 40, "b " * 40]).method == "similarity"
    assert score(["yes", "yes"], method="similarity").method == "similarity"


def test_the_exact_similarity_boundary_is_where_it_is_documented():
    at_limit = " ".join(["w"] * SHORT_ANSWER_TOKENS)
    over_limit = " ".join(["w"] * (SHORT_ANSWER_TOKENS + 1))
    assert choose_method([at_limit, at_limit]) == "exact"
    assert choose_method([over_limit, over_limit]) == "similarity"


def test_a_single_sample_is_unavailable_not_perfect():
    """One sample cannot disagree with itself; reporting 1.0 would be fabricated."""
    result = score(["only one"])
    assert result is UNAVAILABLE
    assert result.band == ConfidenceBand.MID
    assert result.n_samples == 0


def test_no_samples_is_unavailable():
    assert score([]) is UNAVAILABLE
    assert score(["", "   "]) is UNAVAILABLE


def test_unanimous_short_answers_band_high():
    result = score(["positive", "Positive.", "POSITIVE"])
    assert result.band == ConfidenceBand.HIGH
    assert result.agreement == 1.0
    assert result.modal_count == 3


def test_scattered_answers_band_low():
    result = score(["positive", "negative", "neutral", "mixed"])
    assert result.band == ConfidenceBand.LOW


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="must be 'auto', 'exact' or 'similarity'"):
        score(["a", "b"], method="cosine")


def test_too_many_samples_is_rejected():
    with pytest.raises(ValueError, match="at most"):
        score(["x"] * (MAX_SAMPLES + 1))


def test_similarity_mode_leaves_modal_fields_empty():
    result = score(["alpha beta", "alpha gamma"], method="similarity")
    assert result.modal_answer is None
    assert result.modal_count == 0


# -- Gate integration --------------------------------------------------------


def test_gate_from_samples_uses_configured_thresholds():
    strict = Gate(GateConfig(high_agreement=0.99, low_agreement=0.98))
    assert strict.from_samples(["a", "a", "b", "b"]).band == ConfidenceBand.LOW
    assert strict.from_samples(["a", "a", "a"]).band == ConfidenceBand.HIGH


def test_assess_prefers_logprobs_when_available():
    gate = Gate(CFG)
    pd = {"logprobs": [{"logprob": -0.1}, {"logprob": -0.2}]}
    assessment = gate.assess(provider_details=pd, samples=["a", "b", "c", "d"])
    assert assessment.signal == "logprobs"
    assert assessment.band == ConfidenceBand.HIGH  # despite the scattered samples
    assert assessment.consistency is None


def test_assess_falls_back_when_logprobs_are_missing():
    """The gap this closes: no logprobs used to mean a blanket, uninformative MID."""
    gate = Gate(CFG)
    assessment = gate.assess(provider_details=None, samples=["yes", "yes", "yes"])
    assert assessment.signal == "self-consistency"
    assert assessment.band == ConfidenceBand.HIGH


def test_assess_falls_back_on_an_empty_logprobs_payload():
    gate = Gate(CFG)
    assessment = gate.assess(provider_details={"logprobs": []}, samples=["no", "no", "no"])
    assert assessment.signal == "self-consistency"
    assert assessment.logprobs is not None
    assert assessment.logprobs.logprobs_available is False


def test_assess_with_no_signal_at_all_is_distinguishable_from_an_unsure_model():
    assessment = Gate(CFG).assess()
    assert assessment.signal == "none"
    assert assessment.band == ConfidenceBand.MID


def test_assess_with_a_single_sample_reports_no_signal():
    assessment = Gate(CFG).assess(samples=["only one"])
    assert assessment.signal == "none"
    assert assessment.band == ConfidenceBand.MID


# -- config ------------------------------------------------------------------


def test_agreement_thresholds_must_be_ordered():
    with pytest.raises(ValueError, match="high_agreement"):
        GateConfig(high_agreement=0.3, low_agreement=0.7)


@pytest.mark.parametrize("kwargs", [{"high_agreement": 1.5}, {"low_agreement": -0.1}])
def test_agreement_thresholds_must_be_in_the_unit_interval(kwargs):
    with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
        GateConfig(**kwargs)


def test_samples_below_two_is_rejected():
    with pytest.raises(ValueError, match="samples must be >= 2"):
        GateConfig(samples=1)


def test_config_reads_agreement_settings_from_env(monkeypatch):
    monkeypatch.setenv("MCP_CONFIDENCE_HIGH_AGREEMENT", "0.9")
    monkeypatch.setenv("MCP_CONFIDENCE_LOW_AGREEMENT", "0.4")
    monkeypatch.setenv("MCP_CONFIDENCE_SAMPLES", "7")
    config = GateConfig.from_env()
    assert (config.high_agreement, config.low_agreement, config.samples) == (0.9, 0.4, 7)
