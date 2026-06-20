"""Tests for mcp_confidence.adapters — openai + pydantic_ai helpers."""

from __future__ import annotations

from types import SimpleNamespace

from conftest import openai_dict, openai_object

from mcp_confidence import core
from mcp_confidence.adapters import openai as openai_adapter
from mcp_confidence.adapters import (
    openai_logprobs,
    to_provider_details,
)
from mcp_confidence.adapters import (
    provider_details as pa_provider_details,
)
from mcp_confidence.adapters import pydantic_ai as pa_adapter


# -- openai_logprobs: dict path ----------------------------------------------
def test_openai_logprobs_dict():
    assert openai_logprobs(openai_dict([-0.2, -0.5, -0.1])) == [-0.2, -0.5, -0.1]


def test_openai_logprobs_object():
    assert openai_logprobs(openai_object([-0.2, -0.5, -0.1])) == [-0.2, -0.5, -0.1]


def test_openai_logprobs_none_response():
    assert openai_logprobs(None) is None


def test_openai_logprobs_empty_choices():
    assert openai_logprobs({"choices": []}) is None


def test_openai_logprobs_missing_logprobs_key():
    assert openai_logprobs({"choices": [{"message": {"content": "hi"}}]}) is None


def test_openai_logprobs_logprobs_none():
    assert openai_logprobs({"choices": [{"logprobs": None}]}) is None


def test_openai_logprobs_empty_content():
    assert openai_logprobs({"choices": [{"logprobs": {"content": []}}]}) is None


def test_openai_logprobs_excludes_bool():
    resp = {
        "choices": [
            {
                "logprobs": {
                    "content": [
                        {"token": "a", "logprob": True},
                        {"token": "b", "logprob": -1.0},
                    ]
                }
            }
        ]
    }
    assert openai_logprobs(resp) == [-1.0]


def test_openai_logprobs_skips_non_numeric():
    resp = {
        "choices": [
            {
                "logprobs": {
                    "content": [
                        {"token": "a", "logprob": "oops"},
                        {"token": "b", "logprob": None},
                        {"token": "c", "logprob": -2.0},
                    ]
                }
            }
        ]
    }
    assert openai_logprobs(resp) == [-2.0]


def test_openai_logprobs_all_bad_returns_none():
    resp = {"choices": [{"logprobs": {"content": [{"token": "a", "logprob": None}]}}]}
    assert openai_logprobs(resp) is None


def test_openai_logprobs_object_missing_logprobs():
    obj = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="hi"), logprobs=None)]
    )
    assert openai_logprobs(obj) is None


# -- to_provider_details -----------------------------------------------------
def test_to_provider_details_dict_shape():
    pd = to_provider_details(openai_dict([-0.2, -0.5]))
    assert pd is not None
    assert "logprobs" in pd
    assert [e["logprob"] for e in pd["logprobs"]] == [-0.2, -0.5]
    # round-trips through core.extract_logprobs
    assert core.extract_logprobs(pd) == [-0.2, -0.5]


def test_to_provider_details_object_shape():
    pd = to_provider_details(openai_object([-0.2, -0.5]))
    assert pd is not None
    assert core.extract_logprobs(pd) == [-0.2, -0.5]


def test_to_provider_details_preserves_fields():
    pd = to_provider_details(openai_dict([-0.2]))
    entry = pd["logprobs"][0]
    assert set(entry) == {"token", "logprob", "bytes", "top_logprobs"}
    assert entry["token"] == "t0"


def test_to_provider_details_none():
    assert to_provider_details(None) is None


def test_to_provider_details_missing_logprobs():
    assert to_provider_details({"choices": [{"message": {"content": "hi"}}]}) is None


def test_to_provider_details_all_bad_returns_none():
    resp = {"choices": [{"logprobs": {"content": [{"token": "a", "logprob": None}]}}]}
    assert to_provider_details(resp) is None


def test_to_provider_details_skips_non_numeric_entries():
    resp = {
        "choices": [
            {
                "logprobs": {
                    "content": [
                        {"token": "a", "logprob": "oops"},
                        {"token": "b", "logprob": -2.0},
                    ]
                }
            }
        ]
    }
    pd = to_provider_details(resp)
    assert core.extract_logprobs(pd) == [-2.0]


# -- module re-exports / private helpers -------------------------------------
def test_get_helper_dict_and_object():
    assert openai_adapter._get({"a": 1}, "a") == 1
    assert openai_adapter._get(SimpleNamespace(a=2), "a") == 2
    assert openai_adapter._get(None, "a") is None
    assert openai_adapter._get({"a": 1}, "missing") is None


# -- pydantic_ai.provider_details --------------------------------------------
def test_pa_provider_details_present():
    mr = SimpleNamespace(provider_details={"logprobs": [{"logprob": -1.0}]})
    assert pa_provider_details(mr) == {"logprobs": [{"logprob": -1.0}]}


def test_pa_provider_details_absent_returns_none():
    assert pa_provider_details(SimpleNamespace()) is None


def test_pa_provider_details_explicit_none():
    mr = SimpleNamespace(provider_details=None)
    assert pa_provider_details(mr) is None


def test_pa_module_alias_same_function():
    assert pa_adapter.provider_details is pa_provider_details
