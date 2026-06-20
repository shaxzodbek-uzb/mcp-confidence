"""Tests for mcp_confidence.config.GateConfig — defaults, validation, from_env."""

from __future__ import annotations

import dataclasses

import pytest

from mcp_confidence import GateConfig


def test_defaults():
    c = GateConfig()
    assert c.high_threshold == -1.5
    assert c.low_threshold == -3.5
    assert c.min_weight == 0.3
    assert c.min_token_floor == -10.0
    assert c.top_k == 5


def test_is_frozen():
    c = GateConfig()
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        c.high_threshold = 0.0  # type: ignore[misc]


def test_custom_values_ok():
    c = GateConfig(high_threshold=-1.0, low_threshold=-4.0, min_weight=0.5)
    assert c.high_threshold == -1.0
    assert c.low_threshold == -4.0
    assert c.min_weight == 0.5


def test_high_le_low_raises():
    with pytest.raises(ValueError):
        GateConfig(high_threshold=-4.0, low_threshold=-1.0)


def test_high_equal_low_raises():
    with pytest.raises(ValueError):
        GateConfig(high_threshold=-2.0, low_threshold=-2.0)


@pytest.mark.parametrize("bad", [-0.01, 1.01, 2.0, -1.0])
def test_weight_out_of_range_raises(bad):
    with pytest.raises(ValueError):
        GateConfig(min_weight=bad)


@pytest.mark.parametrize("ok", [0.0, 0.5, 1.0])
def test_weight_boundaries_ok(ok):
    GateConfig(min_weight=ok)  # must not raise


# -- from_env ----------------------------------------------------------------
def test_from_env_all_defaults_when_unset(monkeypatch):
    for k in (
        "HIGH_THRESHOLD",
        "LOW_THRESHOLD",
        "MIN_WEIGHT",
        "MIN_TOKEN_FLOOR",
        "TOP_K",
    ):
        monkeypatch.delenv("MCP_CONFIDENCE_" + k, raising=False)
    c = GateConfig.from_env()
    assert c == GateConfig()


def test_from_env_reads_all(monkeypatch):
    monkeypatch.setenv("MCP_CONFIDENCE_HIGH_THRESHOLD", "-1.0")
    monkeypatch.setenv("MCP_CONFIDENCE_LOW_THRESHOLD", "-4.0")
    monkeypatch.setenv("MCP_CONFIDENCE_MIN_WEIGHT", "0.5")
    monkeypatch.setenv("MCP_CONFIDENCE_MIN_TOKEN_FLOOR", "-12.0")
    monkeypatch.setenv("MCP_CONFIDENCE_TOP_K", "8")
    c = GateConfig.from_env()
    assert c.high_threshold == -1.0
    assert c.low_threshold == -4.0
    assert c.min_weight == 0.5
    assert c.min_token_floor == -12.0
    assert c.top_k == 8


def test_from_env_partial_override(monkeypatch):
    for k in ("HIGH_THRESHOLD", "LOW_THRESHOLD", "MIN_WEIGHT", "MIN_TOKEN_FLOOR", "TOP_K"):
        monkeypatch.delenv("MCP_CONFIDENCE_" + k, raising=False)
    monkeypatch.setenv("MCP_CONFIDENCE_TOP_K", "10")
    c = GateConfig.from_env()
    assert c.top_k == 10
    assert c.high_threshold == -1.5  # default kept
    assert c.low_threshold == -3.5


def test_from_env_empty_string_falls_back(monkeypatch):
    monkeypatch.setenv("MCP_CONFIDENCE_HIGH_THRESHOLD", "")
    monkeypatch.setenv("MCP_CONFIDENCE_TOP_K", "")
    c = GateConfig.from_env()
    assert c.high_threshold == -1.5
    assert c.top_k == 5


def test_from_env_custom_prefix(monkeypatch):
    monkeypatch.setenv("FOO_HIGH_THRESHOLD", "-0.5")
    monkeypatch.setenv("FOO_TOP_K", "3")
    c = GateConfig.from_env(prefix="FOO_")
    assert c.high_threshold == -0.5
    assert c.top_k == 3


def test_from_env_invalid_combo_raises(monkeypatch):
    # from_env still runs __post_init__ validation.
    monkeypatch.setenv("MCP_CONFIDENCE_HIGH_THRESHOLD", "-5.0")
    monkeypatch.setenv("MCP_CONFIDENCE_LOW_THRESHOLD", "-1.0")
    with pytest.raises(ValueError):
        GateConfig.from_env()
