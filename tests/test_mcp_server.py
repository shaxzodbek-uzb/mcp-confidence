"""Tests for mcp_confidence.mcp_server.

build_confidence_payload is pure and dependency-free, so the bulk of these tests
need neither ``mcp`` nor ``openai``. Importing the module with zero extras must
not raise (lazy imports inside run()).
"""

from __future__ import annotations

import importlib

import pytest
from conftest import openai_dict, openai_object

from mcp_confidence import GateConfig, mcp_server

CFG = GateConfig()


# -- import safety -----------------------------------------------------------
def test_module_imports_without_extras():
    # Re-import to confirm no top-level mcp/openai dependency.
    importlib.reload(mcp_server)
    assert hasattr(mcp_server, "build_confidence_payload")
    assert hasattr(mcp_server, "run")


# -- build_confidence_payload: bands -----------------------------------------
def test_payload_high_band():
    p = mcp_server.build_confidence_payload("hi", openai_dict([-0.2, -0.3, -0.1]), CFG)
    assert p["band"] == "high"
    assert p["text"] == "hi"
    assert p["logprobs_available"] is True
    assert p["token_count"] == 3
    assert p["should_verify"] is False
    assert p["should_ask_human"] is False


def test_payload_mid_band_should_verify():
    p = mcp_server.build_confidence_payload("hi", openai_dict([-2.0, -2.5, -3.0]), CFG)
    assert p["band"] == "mid"
    assert p["should_verify"] is True
    assert p["should_ask_human"] is False


def test_payload_low_band_should_ask_human():
    p = mcp_server.build_confidence_payload("hi", openai_dict([-4.0, -5.0, -6.0]), CFG)
    assert p["band"] == "low"
    assert p["should_ask_human"] is True
    assert p["should_verify"] is False


def test_payload_unavailable_logprobs():
    p = mcp_server.build_confidence_payload(
        "hi", {"choices": [{"message": {"content": "hi"}}]}, CFG
    )
    assert p["band"] == "mid"
    assert p["logprobs_available"] is False
    assert p["should_verify"] is True  # missing logprobs => verify
    assert p["should_ask_human"] is False
    assert p["token_count"] == 0


def test_payload_none_response_unavailable():
    p = mcp_server.build_confidence_payload("hi", None, CFG)
    assert p["band"] == "mid"
    assert p["logprobs_available"] is False
    assert p["should_verify"] is True


def test_payload_text_preserved_verbatim():
    text = 'a multi-line\nworker answer with "quotes"'
    p = mcp_server.build_confidence_payload(text, openai_dict([-0.2]), CFG)
    assert p["text"] == text


def test_payload_keys_complete():
    p = mcp_server.build_confidence_payload("x", openai_dict([-0.2]), CFG)
    assert set(p) == {
        "text",
        "band",
        "score",
        "mean_logprob",
        "min_logprob",
        "token_count",
        "logprobs_available",
        "should_verify",
        "should_ask_human",
    }


def test_payload_object_response_matches_dict():
    a = mcp_server.build_confidence_payload("x", openai_dict([-0.5, -1.0]), CFG)
    b = mcp_server.build_confidence_payload("x", openai_object([-0.5, -1.0]), CFG)
    assert a == b


def test_payload_respects_custom_config():
    cfg = GateConfig(high_threshold=-2.5, low_threshold=-5.0)
    p = mcp_server.build_confidence_payload("x", openai_dict([-2.0, -2.0]), cfg)
    assert p["band"] == "high"
    assert p["should_verify"] is False


def test_payload_is_json_serializable():
    import json

    p = mcp_server.build_confidence_payload("x", openai_dict([-0.2, -0.3]), CFG)
    # band is a plain str via str-Enum value; the rest are floats/ints/bools/str.
    assert json.loads(json.dumps(p))["band"] == "high"


# -- real server build (only if mcp extra installed) -------------------------
def test_run_signature_exists():
    pytest.importorskip("mcp")
    pytest.importorskip("openai")
    # We don't actually start the server (it would block on stdio); just confirm
    # the lazy imports resolve when the extras are present by inspecting run.
    import inspect

    sig = inspect.signature(mcp_server.run)
    assert "transport" in sig.parameters
