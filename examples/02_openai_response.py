"""02 - Banding a real OpenAI chat completion.

``Gate.from_openai`` accepts an OpenAI ``ChatCompletion`` as EITHER a plain dict
(the raw JSON shape) OR an SDK object — it duck-types its way down
``choices[0].logprobs.content[*].logprob``. This file uses a hand-built dict so it
runs with ZERO extras and no network. The commented block at the bottom shows the
real SDK call (which needs the ``[openai]`` extra and a server).

Run it (zero extras, no network)::

    PYTHONPATH=src python3 examples/02_openai_response.py
"""

from __future__ import annotations

from mcp_confidence import Gate, GateConfig


def fake_completion(logprobs: list[float]) -> dict:
    """Build a minimal OpenAI ChatCompletion-shaped dict carrying ``logprobs``.

    Only the fields the adapter reads are populated:
    ``choices[0].logprobs.content[*].logprob``.
    """
    content = [
        {"token": f"t{i}", "logprob": lp, "bytes": None, "top_logprobs": []}
        for i, lp in enumerate(logprobs)
    ]
    return {
        "id": "chatcmpl-demo",
        "object": "chat.completion",
        "model": "local-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "...generated text..."},
                "logprobs": {"content": content},
                "finish_reason": "stop",
            }
        ],
    }


def main() -> None:
    gate = Gate(GateConfig())

    cases = {
        "confident": [-0.02, -0.10, -0.05, -0.08, -0.04],
        "hedged": [-0.40, -2.00, -1.20, -3.00, -0.90],
        "shaky": [-3.20, -4.50, -2.80, -5.00, -3.90],
    }
    for label, lps in cases.items():
        response = fake_completion(lps)
        result = gate.from_openai(response)
        print(
            f"{label:<10} band={result.band.value:<5} "
            f"score={result.score:7.3f}  tokens={result.token_count}"
        )

    # A response with no logprobs (logprobs=True was not requested) -> band MID,
    # logprobs_available=False. The gate stays conservative instead of raising.
    no_logprobs = {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}
    degraded = gate.from_openai(no_logprobs)
    print(
        f"{'no-logprobs':<10} band={degraded.band.value:<5} "
        f"logprobs_available={degraded.logprobs_available}"
    )


# --------------------------------------------------------------------------- #
# Real SDK call (needs `pip install mcp-confidence[openai]` and a running
# OpenAI-compatible server). Point base_url at vLLM / Ollama / llama.cpp / TGI.
#
#     from openai import OpenAI
#     from mcp_confidence import Gate, GateConfig
#
#     client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
#     gate = Gate(GateConfig())
#
#     response = client.chat.completions.create(
#         model="local-model",
#         messages=[{"role": "user", "content": "Summarize this in one line: ..."}],
#         logprobs=True,        # REQUIRED to get a confidence signal
#         top_logprobs=gate.config.top_k,
#         stream=False,         # logprobs only land cleanly on the non-stream path
#     )
#     result = gate.from_openai(response)   # SDK object works the same as a dict
#     print(result.band, result.score)
# --------------------------------------------------------------------------- #


if __name__ == "__main__":
    main()
