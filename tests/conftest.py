"""Shared fixtures and helpers for the mcp-confidence test suite.

All helpers are pure, stdlib-only, network-free. The OpenAI-shaped builders
produce both plain dicts (the JSON shape) and tiny duck-typed objects so the
adapters can be exercised on both code paths without importing the openai SDK.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def provider_details(logprobs):
    """Build a pydantic-ai ``provider_details`` dict in the verified shape."""
    return {
        "logprobs": [
            {"token": f"t{i}", "bytes": None, "logprob": lp, "top_logprobs": []}
            for i, lp in enumerate(logprobs)
        ],
        "finish_reason": "stop",
    }


def openai_dict(logprobs, *, content_text="hello"):
    """Build an OpenAI ChatCompletion as a plain dict.

    Shape: ``choices[0].logprobs.content[*].logprob`` plus a message so the
    server payload path has text to read.
    """
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": content_text},
                "logprobs": {
                    "content": [
                        {
                            "token": f"t{i}",
                            "logprob": lp,
                            "bytes": None,
                            "top_logprobs": [],
                        }
                        for i, lp in enumerate(logprobs)
                    ]
                },
            }
        ]
    }


def openai_object(logprobs, *, content_text="hello"):
    """Build a duck-typed OpenAI ChatCompletion using SimpleNamespace.

    Mirrors :func:`openai_dict` but uses attribute access (``.choices`` etc.)
    so the adapters' getattr branch is exercised.
    """
    content = [
        SimpleNamespace(token=f"t{i}", logprob=lp, bytes=None, top_logprobs=[])
        for i, lp in enumerate(logprobs)
    ]
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(role="assistant", content=content_text),
                logprobs=SimpleNamespace(content=content),
            )
        ]
    )


@pytest.fixture
def make_pd():
    return provider_details


@pytest.fixture
def make_openai_dict():
    return openai_dict


@pytest.fixture
def make_openai_object():
    return openai_object
