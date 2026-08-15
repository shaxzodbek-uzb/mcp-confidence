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
from collections.abc import Sequence

from . import consistency, core
from .adapters.openai import to_provider_details
from .config import GateConfig
from .gate import Gate


def build_confidence_payload(
    text: str,
    openai_response,
    config: GateConfig,
    samples: Sequence[str] | None = None,
) -> dict:
    """Build the director-facing payload from a worker's text + OpenAI response.

    Bands the response from whichever signal is available — logprobs when the
    endpoint returns them, self-consistency over ``samples`` when it does not — and
    returns a JSON-serializable dict:

      * ``text`` — the worker's output, unchanged.
      * ``signal`` — ``"logprobs"``, ``"self-consistency"`` or ``"none"``, so the
        director knows *what* the band is based on rather than guessing.
      * ``band`` / ``score`` / ``mean_logprob`` / ``min_logprob`` /
        ``token_count`` / ``logprobs_available`` — the logprob signal.
      * ``agreement`` / ``n_samples`` / ``consistency_method`` — the fallback
        signal, present only when it ran.
      * ``should_verify`` — True when the band is MID, or when no signal was
        measured at all (the conservative "have a stronger model check this" route).
      * ``should_ask_human`` — True when band is LOW.

    Pure and dependency-free: no network, no ``mcp``/``openai`` import. Sampling is
    the caller's job.
    """
    gate = Gate(config)
    assessment = gate.assess(provider_details=to_provider_details(openai_response), samples=samples)
    logprob_result = assessment.logprobs or core.UNAVAILABLE
    band = assessment.band.value

    payload = {
        "text": text,
        "signal": assessment.signal,
        "band": band,
        "score": logprob_result.score,
        "mean_logprob": logprob_result.mean_logprob,
        "min_logprob": logprob_result.min_logprob,
        "token_count": logprob_result.token_count,
        "logprobs_available": logprob_result.logprobs_available,
        "should_verify": band == "mid" or assessment.signal == "none",
        "should_ask_human": band == "low",
    }
    if assessment.consistency is not None:
        payload["agreement"] = assessment.consistency.agreement
        payload["n_samples"] = assessment.consistency.n_samples
        payload["consistency_method"] = assessment.consistency.method
    return payload


def run(transport: str = "stdio") -> None:
    """Start the MCP confidence server (requires the ``[mcp]`` extra).

    Reads :class:`GateConfig` from ``MCP_CONFIDENCE_*`` env vars and the worker
    endpoint from ``MCP_CONFIDENCE_BASE_URL`` / ``MCP_CONFIDENCE_API_KEY`` /
    ``MCP_CONFIDENCE_MODEL``. Exposes two tools:

      * ``generate_with_confidence`` — calls the OpenAI-compatible chat endpoint
        NON-STREAMING with ``logprobs=True`` and ``top_logprobs=config.top_k``,
        then returns :func:`build_confidence_payload`.
      * ``score_consistency`` — bands samples the director already has. Pure.

    Set ``MCP_CONFIDENCE_CONSISTENCY_FALLBACK=1`` to re-sample when the worker
    returns no logprobs (costs ``config.samples`` completions instead of one, hence
    opt-in), with ``MCP_CONFIDENCE_SAMPLE_TEMPERATURE`` controlling the spread.

    ``mcp`` and ``openai`` are imported HERE so importing this module needs no
    extras.
    """
    from mcp.server.fastmcp import FastMCP
    from openai import OpenAI

    config = GateConfig.from_env()
    base_url = os.environ.get("MCP_CONFIDENCE_BASE_URL", "http://localhost:8000/v1")
    api_key = os.environ.get("MCP_CONFIDENCE_API_KEY", "not-needed")
    model = os.environ.get("MCP_CONFIDENCE_MODEL", "local-model")

    # Opt-in, because it multiplies token cost by config.samples. It only ever
    # triggers on a worker that returned no logprobs, where the alternative is a
    # band of MID that carries no information at all.
    fallback = os.environ.get("MCP_CONFIDENCE_CONSISTENCY_FALLBACK", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    fallback_temperature = float(os.environ.get("MCP_CONFIDENCE_SAMPLE_TEMPERATURE", "0.7"))

    client = OpenAI(base_url=base_url, api_key=api_key)
    server = FastMCP("mcp-confidence")

    def _generate(full: str, **extra) -> object:
        return client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": full}],
            logprobs=True,
            top_logprobs=config.top_k,
            stream=False,
            **extra,
        )

    @server.tool()
    def generate_with_confidence(prompt: str, source: str = "") -> dict:
        """Generate text with a local worker and return it with a confidence band.

        Put the instruction in ``prompt`` and any source text in ``source``. The
        worker runs non-streaming with logprobs so the reply carries a band
        (high/mid/low), ``should_verify`` and ``should_ask_human`` flags, plus a
        ``signal`` field naming what the band was derived from.

        If the worker returns no logprobs and ``MCP_CONFIDENCE_CONSISTENCY_FALLBACK``
        is set, the prompt is re-sampled and the band comes from how much the samples
        agree — otherwise it degrades to MID with ``should_verify=True``.
        """
        full = prompt if not source else f"{prompt}\n\n--- Source text ---\n{source}"
        response = _generate(full)
        text = response.choices[0].message.content or ""

        samples: list[str] | None = None
        if fallback and not Gate(config).from_openai(response).logprobs_available:
            samples = [text]
            for _ in range(config.samples - 1):
                extra = _generate(full, temperature=fallback_temperature)
                samples.append(extra.choices[0].message.content or "")

        return build_confidence_payload(text, response, config, samples=samples)

    @server.tool()
    def score_consistency(samples: list[str], method: str = "auto") -> dict:
        """Band answers the director sampled itself — no generation, no cost.

        Pass the same prompt's answers from any model. ``method`` is ``auto``
        (exact for short answers, similarity for prose), ``exact``, or ``similarity``.

        This measures *stability*, not correctness: a model that is confidently wrong
        the same way every time scores HIGH.
        """
        result = consistency.score(
            samples,
            method=method,
            high_agreement=config.high_agreement,
            low_agreement=config.low_agreement,
        )
        return {
            "band": result.band.value,
            "agreement": result.agreement,
            "n_samples": result.n_samples,
            "method": result.method,
            "modal_answer": result.modal_answer,
            "modal_count": result.modal_count,
            "should_verify": result.band.value == "mid" or result.n_samples < 2,
            "should_ask_human": result.band.value == "low",
        }

    server.run(transport=transport)
