"""Unit tests for mcp_confidence.core — the pure, no-I/O confidence gate.

Ported verbatim (semantics) from the gate's validated production implementation:
extraction, the three logprob aggregations, banding, and the top-level compute()
entry point including the UNAVAILABLE (missing-logprobs) path.
"""

from __future__ import annotations

import math

import pytest

from mcp_confidence import ConfidenceBand, core


def _pd(logprobs):
    """Build a provider_details dict in the verified pydantic-ai shape."""
    return {
        "logprobs": [
            {"token": f"t{i}", "bytes": None, "logprob": lp, "top_logprobs": []}
            for i, lp in enumerate(logprobs)
        ],
        "finish_reason": "stop",
    }


# -- extract_logprobs --------------------------------------------------------
def test_extract_none_provider_details():
    assert core.extract_logprobs(None) is None


def test_extract_empty_dict():
    assert core.extract_logprobs({}) is None


def test_extract_missing_logprobs_key():
    assert core.extract_logprobs({"finish_reason": "stop"}) is None


def test_extract_empty_logprobs_list():
    assert core.extract_logprobs({"logprobs": []}) is None


def test_extract_normal():
    assert core.extract_logprobs(_pd([-0.5, -1.0, -2.0])) == [-0.5, -1.0, -2.0]


def test_extract_skips_non_numeric_entries():
    raw = {
        "logprobs": [
            {"token": "a", "logprob": -1.0},
            {"token": "b", "logprob": None},
            {"token": "c", "logprob": "oops"},
            {"token": "d", "logprob": -2.0},
        ]
    }
    assert core.extract_logprobs(raw) == [-1.0, -2.0]


def test_extract_excludes_bool_logprob():
    # isinstance(True, int) is True in Python — make sure booleans aren't logprobs.
    raw = {"logprobs": [{"token": "a", "logprob": True}, {"token": "b", "logprob": -1.0}]}
    assert core.extract_logprobs(raw) == [-1.0]


def test_extract_all_bad_returns_none():
    raw = {"logprobs": [{"token": "a", "logprob": None}, {"token": "b"}]}
    assert core.extract_logprobs(raw) is None


def test_extract_accepts_int_logprob():
    raw = {"logprobs": [{"token": "a", "logprob": 0}, {"token": "b", "logprob": -1.0}]}
    assert core.extract_logprobs(raw) == [0, -1.0]


# -- mean_logprob ------------------------------------------------------------
def test_mean_empty_is_neg_inf():
    assert core.mean_logprob([]) == float("-inf")


def test_mean_single():
    assert core.mean_logprob([-1.7]) == pytest.approx(-1.7)


@pytest.mark.parametrize(
    "lps,expected",
    [([-1, -2, -3], -2.0), ([0.0, -4.0], -2.0), ([-0.5], -0.5)],
)
def test_mean_values(lps, expected):
    assert core.mean_logprob(lps) == pytest.approx(expected)


# -- min_logprob -------------------------------------------------------------
def test_min_empty_returns_floor():
    assert core.min_logprob([], floor=-10.0) == -10.0


def test_min_above_floor_returned():
    assert core.min_logprob([-1.0, -3.0, -2.0]) == pytest.approx(-3.0)


def test_min_below_floor_clamped():
    assert core.min_logprob([-1.0, -20.0]) == pytest.approx(-10.0)


def test_min_custom_floor():
    assert core.min_logprob([-1.0, -8.0], floor=-5.0) == pytest.approx(-5.0)


def test_min_default_floor_is_minus_ten():
    assert core.min_logprob([-50.0]) == pytest.approx(-10.0)


# -- combined_score ----------------------------------------------------------
def test_combined_pure_mean_when_weight_zero():
    assert core.combined_score(-2.0, -5.0, min_weight=0.0) == pytest.approx(-2.0)


def test_combined_pure_min_when_weight_one():
    assert core.combined_score(-2.0, -5.0, min_weight=1.0) == pytest.approx(-5.0)


def test_combined_default_weight():
    # 0.7*-2.0 + 0.3*-5.0 = -1.4 + -1.5 = -2.9
    assert core.combined_score(-2.0, -5.0, min_weight=0.3) == pytest.approx(-2.9)


def test_combined_neg_inf_propagates_from_mean():
    assert core.combined_score(float("-inf"), -5.0) == float("-inf")


def test_combined_neg_inf_propagates_from_min():
    assert core.combined_score(-2.0, float("-inf")) == float("-inf")


def test_combined_pos_inf_propagates():
    assert core.combined_score(float("inf"), -5.0) == float("-inf")


@pytest.mark.parametrize("bad", [-0.01, 1.01, 1.5, -1.0])
def test_combined_weight_out_of_range_raises(bad):
    with pytest.raises(ValueError):
        core.combined_score(-2.0, -5.0, min_weight=bad)


@pytest.mark.parametrize("ok", [0.0, 1.0, 0.5])
def test_combined_weight_boundaries_ok(ok):
    core.combined_score(-2.0, -5.0, min_weight=ok)  # must not raise


# -- classify ----------------------------------------------------------------
def test_classify_at_high_threshold_inclusive():
    assert core.classify(-1.5, -1.5, -3.5) is ConfidenceBand.HIGH


def test_classify_above_high():
    assert core.classify(-0.5, -1.5, -3.5) is ConfidenceBand.HIGH


def test_classify_mid():
    assert core.classify(-2.5, -1.5, -3.5) is ConfidenceBand.MID


def test_classify_at_low_threshold_inclusive():
    assert core.classify(-3.5, -1.5, -3.5) is ConfidenceBand.LOW


def test_classify_below_low():
    assert core.classify(-9.0, -1.5, -3.5) is ConfidenceBand.LOW


def test_classify_nan_is_mid():
    assert core.classify(float("nan")) is ConfidenceBand.MID


def test_classify_neg_inf_is_mid():
    assert core.classify(float("-inf")) is ConfidenceBand.MID


def test_classify_pos_inf_is_mid():
    assert core.classify(float("inf")) is ConfidenceBand.MID


def test_classify_inverted_thresholds_raise():
    with pytest.raises(ValueError):
        core.classify(-2.0, high_threshold=-4.0, low_threshold=-1.0)


def test_classify_equal_thresholds_raise():
    with pytest.raises(ValueError):
        core.classify(-2.0, high_threshold=-2.0, low_threshold=-2.0)


# -- compute (top-level) -----------------------------------------------------
@pytest.mark.parametrize("pd", [None, {}, {"finish_reason": "stop"}, {"logprobs": []}])
def test_compute_unavailable_paths(pd):
    cr = core.compute(pd)
    assert cr is core.UNAVAILABLE
    assert cr.band is ConfidenceBand.MID
    assert cr.logprobs_available is False
    assert math.isnan(cr.mean_logprob)
    assert cr.token_count == 0


def test_compute_high_band():
    cr = core.compute(_pd([-0.2, -0.3, -0.1]))
    assert cr.band is ConfidenceBand.HIGH
    assert cr.logprobs_available is True
    assert cr.token_count == 3
    assert cr.mean_logprob == pytest.approx(-0.2)


def test_compute_low_band():
    cr = core.compute(_pd([-4.0, -5.0, -6.0]))
    assert cr.band is ConfidenceBand.LOW


def test_compute_mid_band():
    cr = core.compute(_pd([-2.0, -2.5, -3.0]))
    assert cr.band is ConfidenceBand.MID


def test_compute_floor_clamps_outlier():
    # One catastrophic token (-20) clamps to floor -10; with min_weight 0.3 the
    # min term contributes 0.3*-10 = -3.0 even though the mean is fine.
    cr = core.compute(_pd([-0.1, -0.1, -20.0]), min_token_floor=-10.0)
    assert cr.min_logprob == pytest.approx(-10.0)
    assert cr.logprobs_available is True


def test_compute_custom_thresholds():
    # With a generous high threshold, a mid-ish score becomes HIGH.
    cr = core.compute(_pd([-2.0, -2.0]), high_threshold=-2.5, low_threshold=-5.0)
    assert cr.band is ConfidenceBand.HIGH


def test_compute_all_fields_finite_on_valid_input():
    cr = core.compute(_pd([-1.0, -2.0]))
    assert math.isfinite(cr.mean_logprob)
    assert math.isfinite(cr.min_logprob)
    assert math.isfinite(cr.score)


def test_compute_custom_min_weight_affects_score():
    # Pure-mean weighting: a single bad token does not drag the score down.
    cr = core.compute(_pd([-0.1, -0.1, -9.0]), min_weight=0.0)
    assert cr.score == pytest.approx(cr.mean_logprob)


def test_result_is_frozen():
    cr = core.compute(_pd([-1.0]))
    with pytest.raises((AttributeError, TypeError)):
        cr.band = ConfidenceBand.LOW  # type: ignore[misc]


def test_band_serializes_as_plain_string():
    # str-Enum: audit/OTel get clean strings, not "ConfidenceBand.HIGH".
    assert ConfidenceBand.HIGH.value == "high"
    assert f"{ConfidenceBand.LOW.value}" == "low"


def test_unavailable_is_singleton_constant():
    assert core.UNAVAILABLE.band is ConfidenceBand.MID
    assert core.UNAVAILABLE.logprobs_available is False
    assert core.UNAVAILABLE.token_count == 0
    assert math.isnan(core.UNAVAILABLE.score)
    assert math.isnan(core.UNAVAILABLE.min_logprob)
