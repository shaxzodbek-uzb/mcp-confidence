"""MCP server: delegate generation to a local worker and return a confidence band.

The manager-worker use case: a cloud "director" (e.g. Claude) plans and owns the
tool-call loop, but offloads token-heavy, low-risk text work to a fast, cheap
LOCAL model served over an OpenAI-compatible endpoint (vLLM, Ollama, llama.cpp,
TGI, ...). The director has no logprobs of its own; this server runs the local
worker NON-STREAMING with logprobs enabled, computes a confidence band, and hands
the director the text PLUS a band so it can decide to accept, verify, or ask a
human.

This module imports with ZERO extras: the ``mcp`` and ``openai`` packages are
imported lazily INSIDE :func:`run`. The gate logic lives in the pure, dependency-
free :func:`build_confidence_payload`, so tests need neither ``mcp`` nor
``openai`` — a dict fixture is enough.
"""

from __future__ import annotations

import os

from .config import GateConfig
from .gate import Gate


def build_confidence_payload(text: str, openai_response, config: GateConfig) -> dict:
    """Build the director-facing payload from a worker's text + OpenAI response.

    Computes the confidence band from ``openai_response`` (a dict OR an OpenAI SDK
    object) via ``Gate(config).from_openai`` and returns a JSON-serializable dict:

      * ``text`` — the worker's output, unchanged.
      * ``band`` / ``score`` / ``mean_logprob`` / ``min_logprob`` /
        ``token_count`` / ``logprobs_available`` — the confidence signal.
      * ``should_verify`` — True when band is MID or logprobs are unavailable
        (the conservative "have a stronger model check this" route).
      * ``should_ask_human`` — True when band is LOW.

    Pure and dependency-free: no network, no ``mcp``/``openai`` import.
    """
    result = Gate(config).from_openai(openai_response)
    band = result.band.value
    return {
        "text": text,
        "band": band,
        "score": result.score,
        "mean_logprob": result.mean_logprob,
        "min_logprob": result.min_logprob,
        "token_count": result.token_count,
        "logprobs_available": result.logprobs_available,
        "should_verify": band == "mid" or not result.logprobs_available,
        "should_ask_human": band == "low",
    }


def run(transport: str = "stdio") -> None:
    """Start the MCP confidence server (requires the ``[mcp]`` extra).

    Reads :class:`GateConfig` from ``MCP_CONFIDENCE_*`` env vars and the worker
    endpoint from ``MCP_CONFIDENCE_BASE_URL`` / ``MCP_CONFIDENCE_API_KEY`` /
    ``MCP_CONFIDENCE_MODEL``. Exposes one tool, ``generate_with_confidence``,
    which calls the OpenAI-compatible chat endpoint NON-STREAMING with
    ``logprobs=True`` and ``top_logprobs=config.top_k``, then returns
    :func:`build_confidence_payload`.

    ``mcp`` and ``openai`` are imported HERE so importing this module needs no
    extras.
    """
    from mcp.server.fastmcp import FastMCP
    from openai import OpenAI

    config = GateConfig.from_env()
    base_url = os.environ.get("MCP_CONFIDENCE_BASE_URL", "http://localhost:8000/v1")
    api_key = os.environ.get("MCP_CONFIDENCE_API_KEY", "not-needed")
    model = os.environ.get("MCP_CONFIDENCE_MODEL", "local-model")

    client = OpenAI(base_url=base_url, api_key=api_key)
    server = FastMCP("mcp-confidence")

    @server.tool()
    def generate_with_confidence(prompt: str, source: str = "") -> dict:
        """Generate text with a local worker and return it with a confidence band.

        Put the instruction in ``prompt`` and any source text in ``source``. The
        worker runs non-streaming with logprobs so the reply carries a band
        (high/mid/low), ``should_verify`` and ``should_ask_human`` flags. Missing
        logprobs degrade gracefully to band MID with ``should_verify=True``.
        """
        full = prompt if not source else f"{prompt}\n\n--- Source text ---\n{source}"
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": full}],
            logprobs=True,
            top_logprobs=config.top_k,
            stream=False,
        )
        text = response.choices[0].message.content or ""
        return build_confidence_payload(text, response, config)

    server.run(transport=transport)
