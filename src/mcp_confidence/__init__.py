"""mcp-confidence: a drop-in confidence gate for LLM agents.

Turn any model's token logprobs into an accept / verify / ask-a-human routing
decision, with honest per-model calibration. Local & open-model first (vLLM,
Ollama, llama.cpp, TGI, any OpenAI-compatible endpoint), plus a one-command MCP
server so a cloud director can delegate generation to a local worker and get a
confidence signal back.

Quick start::

    from mcp_confidence import Gate, GateConfig

    gate = Gate(GateConfig())
    result = gate.from_logprobs([-0.2, -0.5, -0.1])
    print(result.band, result.score)

When the endpoint strips logprobs, sample the prompt a few times instead::

    result = gate.from_samples(["positive", "positive", "neutral"])
    print(result.band, result.agreement)
"""

from __future__ import annotations

from .config import GateConfig
from .consistency import (
    ConsistencyResult,
    classify_agreement,
    exact_agreement,
    similarity_agreement,
)
from .consistency import score as consistency_score
from .core import (
    UNAVAILABLE,
    ConfidenceBand,
    ConfidenceResult,
    classify,
    combined_score,
    compute,
    extract_logprobs,
    mean_logprob,
    min_logprob,
)
from .gate import Assessment, Gate

__version__ = "0.2.0"

__all__ = [
    "ConfidenceBand",
    "ConfidenceResult",
    "ConsistencyResult",
    "Assessment",
    "UNAVAILABLE",
    "consistency_score",
    "classify_agreement",
    "exact_agreement",
    "similarity_agreement",
    "compute",
    "classify",
    "combined_score",
    "mean_logprob",
    "min_logprob",
    "extract_logprobs",
    "GateConfig",
    "Gate",
    "__version__",
]
