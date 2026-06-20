"""Adapters that normalize provider responses into logprobs / provider_details.

Re-exports the two most-used helpers:

  * :func:`openai_logprobs` / :func:`to_provider_details` — OpenAI chat completion
    (dict OR SDK object) -> logprobs / provider_details dict.
  * :func:`provider_details` — pydantic-ai ModelResponse -> provider_details dict.
"""

from __future__ import annotations

from .openai import openai_logprobs, to_provider_details
from .pydantic_ai import provider_details

__all__ = ["openai_logprobs", "to_provider_details", "provider_details"]
