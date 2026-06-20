# Contributing to mcp-confidence

Thanks for helping make LLM agents more honest about what they don't know.
`mcp-confidence` turns a model's own token logprobs into an **accept / verify /
ask-a-human** routing decision — a small, sharp tool that an agent can lean on
instead of pretending every answer is equally trustworthy.

## Philosophy

The whole project rests on one uncomfortable truth: **a confidence number is
worthless until it's calibrated on your model, your prompts, and your traffic.**
The default thresholds (`HIGH=-1.5`, `LOW=-3.5` in mean token log-probability
space) are *provisional guesses*. They are wrong for your setup until proven
otherwise. Every line of code, every docstring, and every README sentence here
is written to keep that honesty front and centre.

So a good contribution:

- **Tells the truth about uncertainty.** Never present a band as ground truth.
  If a code path can't get logprobs, it must degrade to MID (verify), never to
  blind auto-accept. The conservative default is a feature, not a bug.
- **Keeps the core dependency-free.** `core`, `config`, `gate`, `calibrate`,
  `cli`, and the adapters are **stdlib only**. `import mcp_confidence` must work
  with zero extras installed. Heavy deps (`mcp`, `openai`) are imported *lazily*,
  inside the function that needs them — never at module top level.
- **Stays pure where it can.** The gate math and the calibration functions
  return data and have no I/O. Printing lives in `cli.py`. This is what makes
  the 50+ tests fast and network-free.
- **Backs every claim with code.** If the README says it, a test proves it.

## Adding an adapter

Most contributions will be adapters — teaching the gate to read logprobs from one
more inference stack (vLLM, Ollama, llama.cpp, TGI, a new SDK shape, ...).

1. Add `src/mcp_confidence/adapters/<name>.py`. Export a function that returns a
   flat `list[float] | None` of per-token logprobs (return `None` if any link in
   the chain is missing or empty — never raise on a shape you don't recognise).
2. Duck-type the input: accept both a plain `dict` and an SDK object via
   `getattr`/`[]`, the way `adapters/openai.py` does. People pass both.
3. Re-export your helper from `adapters/__init__.py`.
4. Add a test with a small fixture dict — no network, no live model. Cover the
   happy path *and* the missing-logprobs path (it must return `None`).

## Running the tests

We use [uv](https://docs.astral.sh/uv/):

```bash
uv venv
uv pip install -e ".[dev]"
ruff check .
ruff format --check .
pytest -q
```

CI runs exactly this on Python 3.10–3.13. Keep it green. Tests must not hit the
network or require the `mcp`/`openai` extras — use `pytest.importorskip("mcp")`
for anything that genuinely needs the server, or (preferred) test the pure
`build_confidence_payload` helper with a dict fixture.

## Reporting issues

Found a band that's miscalibrated, an adapter that misreads a real response, or a
docstring that overstates what a logprob can tell you? Open an issue with the
model, the inference server, and a minimal response fixture. A wrong confidence
signal is worse than none — corrections are the most valuable contribution here.

## License

By contributing you agree your work is released under the [MIT License](LICENSE).
