"""Tests for mcp_confidence.gate.Gate — the configured facade over core."""

from __future__ import annotations

import math

import pytest
from conftest import openai_dict, openai_object, provider_details

from mcp_confidence import ConfidenceBand, Gate, GateConfig, core


# -- from_logprobs -----------------------------------------------------------
def test_from_logprobs_high():
    r = Gate().from_logprobs([-0.2, -0.3, -0.1])
    assert r.band is ConfidenceBand.HIGH
    assert r.logprobs_available is True
    assert r.token_count == 3
    assert r.mean_logprob == pytest.approx(-0.2)


def test_from_logprobs_low():
    r = Gate().from_logprobs([-4.0, -5.0, -6.0])
    assert r.band is ConfidenceBand.LOW


def test_from_logprobs_mid():
    r = Gate().from_logprobs([-2.0, -2.5, -3.0])
    assert r.band is ConfidenceBand.MID


def test_from_logprobs_empty_is_unavailable():
    r = Gate().from_logprobs([])
    assert r is core.UNAVAILABLE
    assert r.band is ConfidenceBand.MID
    assert r.logprobs_available is False


def test_from_logprobs_single_token():
    r = Gate().from_logprobs([-0.1])
    assert r.token_count == 1
    assert r.mean_logprob == pytest.approx(-0.1)
    assert math.isfinite(r.score)


def test_from_logprobs_floor_clamps():
    r = Gate().from_logprobs([-0.1, -0.1, -50.0])
    assert r.min_logprob == pytest.approx(-10.0)


def test_from_logprobs_respects_config_thresholds():
    cfg = GateConfig(high_threshold=-2.5, low_threshold=-5.0)
    # mean ~-2.0 would be MID under defaults, HIGH under this generous config.
    r = Gate(cfg).from_logprobs([-2.0, -2.0])
    assert r.band is ConfidenceBand.HIGH


def test_from_logprobs_respects_config_min_weight():
    # min_weight 0 => score equals mean; a single bad token does not pull it down.
    cfg = GateConfig(min_weight=0.0)
    r = Gate(cfg).from_logprobs([-0.1, -0.1, -9.0])
    assert r.score == pytest.approx(r.mean_logprob)


def test_from_logprobs_respects_config_floor():
    cfg = GateConfig(min_token_floor=-5.0)
    r = Gate(cfg).from_logprobs([-0.1, -20.0])
    assert r.min_logprob == pytest.approx(-5.0)


# -- from_provider_details ---------------------------------------------------
def test_from_provider_details_high():
    r = Gate().from_provider_details(provider_details([-0.2, -0.3, -0.1]))
    assert r.band is ConfidenceBand.HIGH
    assert r.token_count == 3


def test_from_provider_details_none_unavailable():
    r = Gate().from_provider_details(None)
    assert r is core.UNAVAILABLE


def test_from_provider_details_empty_unavailable():
    r = Gate().from_provider_details({"logprobs": []})
    assert r is core.UNAVAILABLE


def test_from_provider_details_respects_config():
    cfg = GateConfig(high_threshold=-2.5, low_threshold=-5.0)
    r = Gate(cfg).from_provider_details(provider_details([-2.0, -2.0]))
    assert r.band is ConfidenceBand.HIGH


# -- from_dict alias ---------------------------------------------------------
def test_from_dict_is_alias():
    pd = provider_details([-0.2, -0.3])
    assert Gate().from_dict(pd) == Gate().from_provider_details(pd)


def test_from_dict_none_unavailable():
    assert Gate().from_dict(None) is core.UNAVAILABLE


# -- from_openai (dict + duck-typed object) ----------------------------------
def test_from_openai_dict_high():
    r = Gate().from_openai(openai_dict([-0.2, -0.3, -0.1]))
    assert r.band is ConfidenceBand.HIGH
    assert r.token_count == 3


def test_from_openai_dict_low():
    r = Gate().from_openai(openai_dict([-4.0, -5.0, -6.0]))
    assert r.band is ConfidenceBand.LOW


def test_from_openai_object_high():
    r = Gate().from_openai(openai_object([-0.2, -0.3, -0.1]))
    assert r.band is ConfidenceBand.HIGH
    assert r.token_count == 3


def test_from_openai_object_matches_dict():
    a = Gate().from_openai(openai_dict([-0.5, -1.0, -2.0]))
    b = Gate().from_openai(openai_object([-0.5, -1.0, -2.0]))
    assert a == b


def test_from_openai_none_unavailable():
    assert Gate().from_openai(None) is core.UNAVAILABLE


def test_from_openai_missing_logprobs_unavailable():
    # A response with no logprobs block -> conservative MID/unavailable.
    r = Gate().from_openai({"choices": [{"message": {"content": "hi"}}]})
    assert r is core.UNAVAILABLE
    assert r.band is ConfidenceBand.MID


def test_from_openai_empty_choices_unavailable():
    assert Gate().from_openai({"choices": []}) is core.UNAVAILABLE


def test_from_openai_respects_config():
    cfg = GateConfig(high_threshold=-2.5, low_threshold=-5.0)
    r = Gate(cfg).from_openai(openai_dict([-2.0, -2.0]))
    assert r.band is ConfidenceBand.HIGH


# -- construction ------------------------------------------------------------
def test_default_config_used_when_none():
    g = Gate()
    assert g.config == GateConfig()


def test_explicit_config_stored():
    cfg = GateConfig(top_k=9)
    assert Gate(cfg).config is cfg
