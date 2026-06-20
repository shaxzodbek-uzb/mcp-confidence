"""01 - Banding a raw per-token logprob list.

The simplest possible use of mcp-confidence: you already have a list of
per-token logprobs (from any OpenAI-compatible server's
``choices[0].logprobs.content``) and you want a single accept / verify /
ask-a-human routing decision.

Run it (zero extras, no network)::

    PYTHONPATH=src python3 examples/01_basic_logprobs.py

The three example outputs are the worker's per-token logprobs for, respectively,
a confident answer, a hedged answer, and a shaky one. With the DEFAULT thresholds
(HIGH=-1.5, LOW=-3.5 in mean-logprob nats) they land on HIGH / MID / LOW.

NOTE: those default thresholds are PROVISIONAL GUESSES. Calibrate them on YOUR
model's real output before trusting auto-accept routing — see example 04.
"""

from __future__ import annotations

from mcp_confidence import ConfidenceBand, Gate, GateConfig

# What each band means for an agent's control flow.
ROUTE = {
    ConfidenceBand.HIGH: "ACCEPT  - use the worker's output directly",
    ConfidenceBand.MID: "VERIFY  - hand off to a stronger model / supervisor check",
    ConfidenceBand.LOW: "ASK     - pause and ask a human to clarify",
}

# Three synthetic per-token logprob lists. Higher (closer to 0) = more confident.
SAMPLES = {
    "confident answer": [-0.05, -0.10, -0.20, -0.08, -0.15],
    "hedged answer": [-0.40, -2.00, -1.20, -3.00, -0.90],
    "shaky answer": [-3.20, -4.50, -2.80, -5.00, -3.90],
}


def main() -> None:
    gate = Gate(GateConfig())  # default thresholds

    for label, logprobs in SAMPLES.items():
        result = gate.from_logprobs(logprobs)
        print(f"{label!r}")
        print(f"  band         {result.band.value}")
        print(f"  score        {result.score:.3f}   (combined, mean+min, nats)")
        print(f"  mean_logprob {result.mean_logprob:.3f}")
        print(f"  min_logprob  {result.min_logprob:.3f}   (worst token)")
        print(f"  token_count  {result.token_count}")
        print(f"  -> {ROUTE[result.band]}")
        print()

    # Empty input degrades to the conservative MID band (never blind-accept).
    empty = gate.from_logprobs([])
    print(
        f"empty logprobs -> band {empty.band.value} (logprobs_available={empty.logprobs_available})"
    )


if __name__ == "__main__":
    main()
