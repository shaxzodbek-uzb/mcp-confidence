# Examples

Runnable examples for `mcp-confidence`. Examples 01–02 and the calibrate command
run with **zero extra dependencies and no network** — the gate, the CLI, and the
adapters are stdlib-only.

Run them from the repo root. If you have not installed the package, prefix with
`PYTHONPATH=src` so Python finds `mcp_confidence` under `src/`.

| # | File | What it shows | Needs |
| - | ---- | ------------- | ----- |
| 01 | [`01_basic_logprobs.py`](01_basic_logprobs.py) | Band a raw per-token logprob list into HIGH / MID / LOW and map each band to an accept / verify / ask routing decision. | nothing |
| 02 | [`02_openai_response.py`](02_openai_response.py) | Band an OpenAI `ChatCompletion` (built as a dict, no network) via `Gate.from_openai`; includes a commented real-SDK call. | nothing (live call needs `[openai]`) |
| 03 | [`03_local_vllm_worker.py`](03_local_vllm_worker.py) | The manager-worker pattern: a cloud director delegates generation to a local vLLM/Ollama worker and gets a confidence band back. Offline demo runs with no server. | nothing (live `--live` needs `[openai]` + a server) |
| 04 | [`04_calibrate/`](04_calibrate/) | Calibrate real thresholds from a JSONL audit log. Includes a 25-row sample that produces a real recommendation. | nothing |

## Run each

```bash
# 01 - basic logprob banding
PYTHONPATH=src python3 examples/01_basic_logprobs.py

# 02 - OpenAI-shaped response (offline)
PYTHONPATH=src python3 examples/02_openai_response.py

# 03 - manager-worker framing (offline demo; add --live for a real worker)
PYTHONPATH=src python3 examples/03_local_vllm_worker.py

# 04 - calibrate thresholds from the sample audit log
mcp-confidence calibrate --audit examples/04_calibrate/sample_audit.jsonl
# or without installing:
PYTHONPATH=src python3 -m mcp_confidence.cli calibrate \
    --audit examples/04_calibrate/sample_audit.jsonl
```

## A quick CLI sanity check

The `score` subcommand bands a literal logprob list — handy for a one-liner check:

```bash
mcp-confidence score --logprobs="-0.2,-0.5,-0.1"
# or:  PYTHONPATH=src python3 -m mcp_confidence.cli score --logprobs="-0.2,-0.5,-0.1"
```

> **Note on thresholds.** The default thresholds (`HIGH=-1.5`, `LOW=-3.5` in
> mean-logprob nats) are provisional guesses. They are model-specific — calibrate
> them on your own model's output (example 04) before trusting auto-accept
> routing.
