<h1 align="center">mcp-confidence</h1>

<p align="center">
  <strong>A drop-in confidence gate for LLM agents.</strong><br>
  Turn any model's token logprobs into an <em>accept / verify / ask-a-human</em> routing decision —<br>
  with honest, per-model calibration. Local &amp; open-model first.
</p>

<p align="center">
  <a href="https://pypi.org/project/mcp-confidence/"><img src="https://img.shields.io/pypi/v/mcp-confidence.svg" alt="PyPI version"></a>
  <a href="https://github.com/shaxzodbek-uzb/mcp-confidence/actions/workflows/ci.yml"><img src="https://github.com/shaxzodbek-uzb/mcp-confidence/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
  <img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-3776AB?logo=python&logoColor=white" alt="Python 3.10+">
</p>

---

A language model can tell you *how* surprised it was by every token it just produced — that's what `logprobs=True` returns. `mcp-confidence` reads those numbers and turns them into a routing decision your agent can act on:

- **HIGH** → auto-accept the model's output.
- **MID** → verify (hand off to a stronger model or a supervisor check).
- **LOW** → ask a human.

It's **local- and open-model-first** (vLLM, Ollama, llama.cpp, TGI, or any OpenAI-compatible endpoint), the core has **zero third-party runtime dependencies**, and it ships a one-command **MCP server** so a cloud "director" (e.g. Claude) can delegate generation to a local worker and get a confidence band back with every answer.

## Why

The naive approach — "average the token probabilities, accept above 0.95" — is a weak and *uncalibrated* signal for open-domain free text, and quietly wrong on most models. High-frequency function words and MoE routing pull the per-token mean toward zero, so `exp(mean_logprob)` sits at 0.05–0.20 even for perfectly fluent output. A fixed probability threshold tuned on short structured outputs will reject good answers and accept bad ones once you point it at real prose.

`mcp-confidence` takes four deliberate positions to make the signal usable:

1. **Band in mean-logprob (nats) space, not probability.** Thresholds live in log space where the score actually separates, instead of an `exp()`-compressed `[0, 1]` that bunches everything near zero.
2. **A weakest-link term with a floor.** The score mixes the mean with the *worst* per-token logprob (`min_weight=0.3` by default), clamped to a floor (`-10.0`) so one ultra-rare token — a numeric ID, an unusual name — can't collapse the score of an otherwise solid answer.
3. **Per-model calibration tooling.** Bundled risk-coverage analysis turns a log of real traffic into the thresholds *your* model needs, instead of you guessing again.
4. **A 3-band decision, not a number.** The output is a routing band you can branch on — accept, verify, or ask a human — with a conservative default: when logprobs are missing, the band is **MID**, never blind auto-accept.

> **The defaults are provisional guesses.** `HIGH=-1.5`, `LOW=-3.5` are starting points, not validated thresholds. Do not turn on auto-accept routing until you've recalibrated on your own model — see [The honest part](#the-honest-part-calibrate-before-you-trust-it).

## Install

```bash
pip install mcp-confidence
```

The core (gate, config, calibration, CLI, adapters) is **pure stdlib** — nothing else gets pulled in. Install extras only for what you use:

```bash
pip install "mcp-confidence[openai]"   # the OpenAI SDK, for live calls
pip install "mcp-confidence[mcp]"      # the MCP server (mcp + openai)
pip install "mcp-confidence[dev]"      # pytest + ruff
```

With [uv](https://docs.astral.sh/uv/):

```bash
uv pip install mcp-confidence
uv pip install "mcp-confidence[mcp]"
```

You don't need any extra to score logprobs you already have — only to *fetch* them from a live model or to run the server.

## Quickstart

### a) From a raw logprob list

If you already have a list of per-token logprobs, you don't need any extras or a network call:

```python
from mcp_confidence import Gate, ConfidenceBand

gate = Gate()  # default thresholds
result = gate.from_logprobs([-0.2, -0.5, -0.1])

print(result.band)          # ConfidenceBand.HIGH
print(result.score)         # -0.337  (mean/min mix, in nats — not a probability)
print(result.mean_logprob)  # -0.267

if result.band is ConfidenceBand.LOW:
    ...   # ask a human
elif result.band is ConfidenceBand.MID:
    ...   # verify with a stronger model
else:
    ...   # HIGH — accept
```

`ConfidenceResult` is a frozen dataclass: `score`, `band`, `mean_logprob`, `min_logprob`, `token_count`, `logprobs_available`.

### b) From an OpenAI chat completion

Call the API with `logprobs=True`, then hand the whole response to `from_openai`. It accepts both an SDK object and the raw JSON `dict`, and never raises on a malformed response — missing logprobs degrade to band **MID**.

```python
from openai import OpenAI
from mcp_confidence import Gate, ConfidenceBand

client = OpenAI()
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Capital of Australia?"}],
    logprobs=True,
    top_logprobs=5,
)

result = Gate().from_openai(resp)

if result.band is ConfidenceBand.LOW:
    escalate_to_human(resp)
elif result.band is ConfidenceBand.MID:
    verify_with_stronger_model(resp)
else:
    accept(resp)
```

### c) Local model (vLLM / Ollama / llama.cpp / TGI)

Any OpenAI-compatible server works — it's the same `from_openai` path, you just swap `base_url`. This is the intended sweet spot: cheap local generation, with a confidence signal so you know when to escalate.

```python
from openai import OpenAI
from mcp_confidence import Gate

# vLLM:   http://localhost:8000/v1      Ollama: http://localhost:11434/v1
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

resp = client.chat.completions.create(
    model="Qwen/Qwen3-30B-A3B",
    messages=[{"role": "user", "content": "Summarize this ticket."}],
    logprobs=True,
    top_logprobs=5,
)

result = Gate().from_openai(resp)
print(result.band.value, round(result.score, 3))
```

Tune the gate per model by passing a `GateConfig` to `Gate(...)`, or load one from the environment:

```python
from mcp_confidence import Gate, GateConfig

gate = Gate(GateConfig(high_threshold=-1.2, low_threshold=-3.0, min_weight=0.4))
# or, from MCP_CONFIDENCE_* env vars:
gate = Gate(GateConfig.from_env())
```

There's also a CLI for a quick sanity check — no extras required:

```console
$ mcp-confidence score --logprobs="-0.2,-0.5,-0.1"
band         high
score        -0.3370
mean_logprob -0.2667
min_logprob  -0.5000
token_count  3
```

Add `--config-from-env` to apply `MCP_CONFIDENCE_*` thresholds instead of the defaults.

## The honest part: calibrate before you trust it

**Confidence thresholds are model-specific.** A band that means "almost always right" on one model means "coin flip" on another. The shipped defaults are guesses; the only way to trust auto-accept routing is to calibrate on your own model's live output.

The workflow: log a JSONL row per generation carrying at least `mean_logprob`, `score`, `band`, and `logprobs_available` (an `event` field too, if you want to filter). When you can, add a `human_label` of `"good"` or `"bad"`. Then point the calibrator at it:

```bash
mcp-confidence calibrate --audit confidence.jsonl --metric mean_logprob
```

That always prints two things. First, the **distribution** — where your traffic actually sits — plus the rate of calls where logprobs were unavailable (with a loud warning if it's over 20%, because the gate is blind on those):

```
worker_delegated events: 1240  |  logprobs unavailable: 31 (3%)

-- mean_logprob distribution (n=1209) --
  min  -6.412   p10 -3.880   p25 -2.910
  med  -1.740   mean -1.902
  p75  -0.980   p90 -0.520   max -0.090
  band split (current thresholds): high=505 (42%)  low=121 (10%)  mid=583 (48%)
```

Second, when your rows carry `good`/`bad` labels, a **risk-coverage recommendation**. It sweeps every observed value as a candidate `(high, low)` pair and picks the cell that minimizes "bad answers auto-accepted + good answers needlessly escalated," subject to your coverage constraints:

```
-- risk-coverage (metric=mean_logprob) --
  RECOMMENDED: high_threshold=-1.1  low_threshold=-3.4
    auto-accept coverage 58%  (bad slipped through: 4%)
    human-ask coverage  11%  (good needlessly asked: 6%)
  Set MCP_CONFIDENCE_HIGH_THRESHOLD / MCP_CONFIDENCE_LOW_THRESHOLD to these, then enable routing.
```

Tune the constraints with `--min-accept-coverage` (default `0.70`) and `--max-ask-coverage` (default `0.15`); calibrate on `--metric score` instead of `mean_logprob` if you prefer the combined score; use `--event-name '*'` to accept any row regardless of its `event` field. Without labels the recommendation is skipped and you pick conservative bands by eye from the distribution.

The same functions are importable and pure (no I/O) if you'd rather calibrate in a notebook:

```python
from mcp_confidence.calibrate import load_events, distribution, risk_coverage

rows, total, unavailable = load_events("confidence.jsonl", event_name=None)
print(distribution(rows, "mean_logprob"))
print(risk_coverage(rows, "mean_logprob", min_accept_cov=0.70, max_ask_cov=0.15))
```

## MCP server: the manager–worker pattern

A cloud **director** (Claude, say) is great at planning and owning the tool-call loop, but it has no logprobs of its own and is expensive for token-heavy grunt work. `mcp-confidence` ships an MCP server that lets the director **delegate generation to a local worker** over an OpenAI-compatible endpoint and get the text back **with a confidence band attached** — so it can decide to accept, verify, or ask a human.

The server runs the worker **non-streaming** with `logprobs=True` and `top_logprobs=config.top_k`, then computes the gate. Configure it entirely through the environment:

```bash
export MCP_CONFIDENCE_BASE_URL="http://localhost:8000/v1"   # your vLLM/Ollama/TGI worker
export MCP_CONFIDENCE_API_KEY="not-needed"
export MCP_CONFIDENCE_MODEL="Qwen/Qwen3-30B-A3B"
# optional gate overrides (after calibration):
export MCP_CONFIDENCE_HIGH_THRESHOLD="-1.1"
export MCP_CONFIDENCE_LOW_THRESHOLD="-3.4"

mcp-confidence serve            # stdio transport
```

Register it with Claude Code / Claude Desktop:

```json
{
  "mcpServers": {
    "confidence": {
      "command": "mcp-confidence",
      "args": ["serve"],
      "env": {
        "MCP_CONFIDENCE_BASE_URL": "http://localhost:8000/v1",
        "MCP_CONFIDENCE_API_KEY": "not-needed",
        "MCP_CONFIDENCE_MODEL": "Qwen/Qwen3-30B-A3B"
      }
    }
  }
}
```

The server exposes one tool:

> **`generate_with_confidence(prompt: str, source: str = "") -> dict`**
> Generate text with the local worker and return it with a confidence band.

The returned payload is JSON-serializable and decision-ready:

```json
{
  "text": "...the worker's answer...",
  "band": "high",
  "score": -0.94,
  "mean_logprob": -0.71,
  "min_logprob": -1.48,
  "token_count": 128,
  "logprobs_available": true,
  "should_verify": false,
  "should_ask_human": false
}
```

`should_verify` is true when the band is **MID** *or* logprobs were unavailable; `should_ask_human` is true when the band is **LOW**. A worker that doesn't emit logprobs degrades gracefully to MID with `should_verify=true` — the director is never told to blindly trust a blind signal.

The gate logic for this lives in a pure, dependency-free helper, `mcp_confidence.mcp_server.build_confidence_payload(text, openai_response, config)`, so you can unit-test the whole decision with a dict fixture and no `mcp`/`openai` installed. `import mcp_confidence.mcp_server` never requires the extras — they're imported lazily inside `run()`, only when you actually serve.

## How it compares

The closest existing package is [`llm-confidence`](https://pypi.org/project/llm-confidence/) (VATBox), which is good at what it does: deriving a confidence score per **key** in an OpenAI **structured JSON** output. `mcp-confidence` is aimed at a different problem — routing **free-text** generation, with calibration and a human-in-the-loop decision.

| | `mcp-confidence` | `llm-confidence` |
|---|---|---|
| Primary target | Free-text / open-domain generation | OpenAI structured JSON outputs |
| Output | 3-band routing decision (accept / verify / ask-a-human) | Per-key confidence scores |
| Scoring | Mean **+ weakest-link** in logprob (nats) space, with a floor | Per-field probability aggregation |
| Calibration tooling | Yes — distribution + labeled risk-coverage sweep | Not included |
| Missing-logprobs handling | Conservative MID + `should_verify` | n/a |
| Local / open models | First-class (vLLM, Ollama, llama.cpp, TGI) | OpenAI-oriented |
| MCP server | Yes — manager–worker delegation | No |
| Runtime dependencies | Zero for the core | — |

Different tools for different jobs: reach for `llm-confidence` when you're scoring fields of a JSON extraction; reach for `mcp-confidence` when you're deciding whether to trust, double-check, or escalate a generated answer.

## Roadmap

Clearly future, **not yet implemented**:

- **Supervisor / verify MCP tool** — a semantic groundedness check for the MID band (does the answer actually follow from the source?), so "verify" becomes an automated second opinion rather than just a flag.
- **Self-consistency sampling** — use the `top_logprobs` already captured per token (currently untouched) to estimate agreement across alternatives.
- **Verbalized confidence** — blend the model's own stated confidence with the logprob signal.
- **Learned calibrator** — fit a small model on labeled audit logs instead of a single threshold pair.

## Contributing

Most contributions will be **adapters** — teaching the gate to read logprobs from one more inference stack — or **calibration corrections**. A wrong confidence signal is worse than none, so corrections (a band that's miscalibrated, an adapter that misreads a real response, a docstring that overstates what a logprob can tell you) are the most valuable PRs here. Keep the core stdlib-only and every claim backed by a test. See [CONTRIBUTING.md](CONTRIBUTING.md).

```bash
uv venv && uv pip install -e ".[dev]"
ruff check . && ruff format --check .
pytest -q
```

## License

[MIT](LICENSE) © 2026 Shaxzodbek Qambaraliyev / [Blaze](https://blaze.uz)
