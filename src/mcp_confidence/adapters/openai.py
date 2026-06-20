"""Adapter: OpenAI chat completion -> per-token logprobs / provider_details.

Accepts an OpenAI ``ChatCompletion`` as EITHER a plain ``dict`` (the JSON shape)
OR an SDK object (``openai.types.chat.ChatCompletion``), duck-typed via
``getattr`` and ``[]``. Every helper is defensive: any missing or malformed link
in the chain returns ``None`` instead of raising, so a logprob-less response
cleanly maps to the conservative UNAVAILABLE band downstream.

Logprob path:
  ``response.choices[0].logprobs.content`` ->
  ``[{token, logprob, bytes, top_logprobs}, ...]`` -> the per-token ``logprob``.

To request logprobs from the server, call the chat completion with
``logprobs=True`` and ``top_logprobs=k``.
"""

from __future__ import annotations


def _get(obj, key):
    """Read ``key`` from a dict OR an attribute from an object. None on miss."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _first_choice(response):
    """Return ``response.choices[0]`` for a dict or SDK object, else None."""
    choices = _get(response, "choices")
    if not choices:
        return None
    try:
        return choices[0]
    except (IndexError, KeyError, TypeError):
        return None


def _content_entries(response):
    """Return the list under ``choices[0].logprobs.content``, else None."""
    choice = _first_choice(response)
    if choice is None:
        return None
    logprobs = _get(choice, "logprobs")
    if logprobs is None:
        return None
    content = _get(logprobs, "content")
    if not content:
        return None
    return content


def openai_logprobs(response) -> list[float] | None:
    """Flat per-token logprob list from an OpenAI chat completion.

    Returns None if any link in
    ``response.choices[0].logprobs.content[*].logprob`` is missing, empty, or has
    no numeric logprob entries. Booleans are excluded (``isinstance(True, int)``).
    """
    content = _content_entries(response)
    if not content:
        return None
    lps: list[float] = []
    for entry in content:
        lp = _get(entry, "logprob")
        if isinstance(lp, (int, float)) and not isinstance(lp, bool):
            lps.append(lp)
    return lps if lps else None


def to_provider_details(response) -> dict | None:
    """Normalize an OpenAI chat completion into a provider_details dict.

    Returns ``{"logprobs": [{"token", "logprob", "bytes", "top_logprobs"}, ...]}``
    in the shape :func:`mcp_confidence.core.extract_logprobs` consumes, or None if
    the response carries no usable logprobs. Each entry copies the per-token
    ``token``, ``logprob``, ``bytes`` and ``top_logprobs`` when present.
    """
    content = _content_entries(response)
    if not content:
        return None
    out: list[dict] = []
    for entry in content:
        lp = _get(entry, "logprob")
        if not (isinstance(lp, (int, float)) and not isinstance(lp, bool)):
            continue
        out.append(
            {
                "token": _get(entry, "token"),
                "logprob": lp,
                "bytes": _get(entry, "bytes"),
                "top_logprobs": _get(entry, "top_logprobs"),
            }
        )
    if not out:
        return None
    return {"logprobs": out}
