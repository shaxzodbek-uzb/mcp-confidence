"""Adapter: pydantic-ai ModelResponse -> provider_details dict.

Thin pass-through that pulls ``provider_details`` off a pydantic-ai
``ModelResponse``. Feed the result to :func:`mcp_confidence.core.compute` or
:meth:`mcp_confidence.gate.Gate.from_provider_details`.

WARNING — request logprobs on the NON-STREAMING path only.
  pydantic-ai only surfaces per-token logprobs in ``provider_details['logprobs']``
  on the non-streaming ``agent.run()`` / ``model.request()`` path. The streaming
  merge clobbers all but the LAST token, so a streamed response yields a useless
  one-token signal. Request the logprobs explicitly via model settings on an
  OpenAI-compatible model::

      result = await agent.run(
          prompt,
          model_settings={"openai_logprobs": True, "openai_top_logprobs": k},
      )

  then pass the final ModelResponse's ``provider_details`` through this adapter.
"""

from __future__ import annotations


def provider_details(model_response) -> dict | None:
    """Return ``model_response.provider_details`` (or None if absent)."""
    return getattr(model_response, "provider_details", None)
