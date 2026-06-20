"""03 - Manager-worker: delegate generation to a LOCAL worker, get a band back.

The use case mcp-confidence is built for: a cloud "director" (e.g. Claude) plans
and owns the tool-call loop, but offloads token-heavy, low-risk text work to a
fast, cheap LOCAL model served over an OpenAI-compatible endpoint (vLLM, Ollama,
llama.cpp, TGI, ...). The director has no logprobs of its own, so it cannot tell a
confident answer from a guess. This worker runs the local model NON-STREAMING with
logprobs enabled and returns the text PLUS a confidence band, so the director can
decide to accept, verify, or ask a human.

That payload is exactly what the bundled MCP server hands back
(``mcp-confidence serve`` -> ``generate_with_confidence`` tool). Here we call the
same pure ``build_confidence_payload`` helper directly so the framing is visible
without running the MCP server.

This file LOADS and the offline demo RUNS with zero extras / no network — the live
call is guarded behind a flag and the ``openai`` import is done lazily inside the
function, so nothing happens unless you opt in.

Run the offline demo::

    PYTHONPATH=src python3 examples/03_local_vllm_worker.py

Run against a real local server (needs `mcp-confidence[openai]` + a server)::

    pip install "mcp-confidence[openai]"
    # start e.g. vLLM:  python -m vllm.entrypoints.openai.api_server --model <m>
    # or Ollama:        ollama serve   (base_url http://localhost:11434/v1)
    MCP_CONFIDENCE_BASE_URL=http://localhost:8000/v1 \
    MCP_CONFIDENCE_MODEL=local-model \
        PYTHONPATH=src python3 examples/03_local_vllm_worker.py --live
"""

from __future__ import annotations

import json
import os
import sys

from mcp_confidence import GateConfig
from mcp_confidence.mcp_server import build_confidence_payload


def call_local_worker(prompt: str, config: GateConfig) -> tuple[str, object]:
    """Call a local OpenAI-compatible worker NON-STREAMING with logprobs on.

    Returns ``(text, response)``. The ``openai`` import is lazy so this module
    imports fine without the ``[openai]`` extra installed.
    """
    from openai import OpenAI  # lazy: only needed for the live path

    base_url = os.environ.get("MCP_CONFIDENCE_BASE_URL", "http://localhost:8000/v1")
    api_key = os.environ.get("MCP_CONFIDENCE_API_KEY", "not-needed")
    model = os.environ.get("MCP_CONFIDENCE_MODEL", "local-model")

    client = OpenAI(base_url=base_url, api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        logprobs=True,  # REQUIRED for a confidence signal
        top_logprobs=config.top_k,
        stream=False,  # logprobs only land cleanly off-stream
    )
    text = response.choices[0].message.content or ""
    return text, response


def offline_demo(config: GateConfig) -> None:
    """Show the director-facing payload without a server, using a canned response."""
    # Stand-in for what a local worker would return for a confident answer.
    canned_response = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "The capital of France is Paris."},
                "logprobs": {
                    "content": [
                        {"token": "The", "logprob": -0.05, "bytes": None, "top_logprobs": []},
                        {"token": " capital", "logprob": -0.12, "bytes": None, "top_logprobs": []},
                        {"token": " of", "logprob": -0.03, "bytes": None, "top_logprobs": []},
                        {"token": " France", "logprob": -0.02, "bytes": None, "top_logprobs": []},
                        {"token": " is", "logprob": -0.04, "bytes": None, "top_logprobs": []},
                        {"token": " Paris", "logprob": -0.01, "bytes": None, "top_logprobs": []},
                    ]
                },
            }
        ]
    }
    text = canned_response["choices"][0]["message"]["content"]
    payload = build_confidence_payload(text, canned_response, config)
    print("offline demo (no server) — director-facing payload:")
    print(json.dumps(payload, indent=2))
    print(
        "\nThe director reads `band` / `should_verify` / `should_ask_human` to "
        "decide what to do with `text`."
    )


def main() -> None:
    config = GateConfig.from_env()  # thresholds from MCP_CONFIDENCE_* (or defaults)

    if "--live" not in sys.argv:
        offline_demo(config)
        print("\n(Use --live with a running local server to call a real worker.)")
        return

    prompt = "Summarize the manager-worker pattern in one sentence."
    text, response = call_local_worker(prompt, config)
    payload = build_confidence_payload(text, response, config)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
