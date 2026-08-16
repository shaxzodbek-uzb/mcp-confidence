# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-08-16

### Added
- **Self-consistency scoring**, so a confidence band is still available when a
  model does not return logprobs — which is most hosted models. Sample the same
  prompt *n* times and score how much the answers agree
  (`mcp_confidence.consistency`: `score`, `exact_agreement`,
  `similarity_agreement`, `classify_agreement`).
- Two agreement methods, plus `auto`. `exact` compares normalised answers and
  suits short factual replies; `similarity` compares token overlap and suits
  prose. `auto` picks by answer length, at a threshold of 4 tokens — measured in
  tokens rather than characters, because a short *sentence* is prose and scoring
  it for exact string equality reports near-zero agreement on answers that in fact
  say the same thing.
- `Gate.from_samples()` and `Gate.assess()` — one call from raw samples to a
  band, returning an `Assessment` that also reports which method was used and how
  many samples were usable.
- CLI: `mcp-confidence consistency`, with `--method` and `--config-from-env`.
- MCP: the server can now sample the worker *n* times and return a
  consistency-based band, so a director gets a calibrated band from a
  logprob-less model.

## [0.1.0] - 2026-06-20

### Added

- Pure, stdlib-only confidence gate (`mcp_confidence.core`) that turns per-token
  logprobs into a normalized score and a HIGH / MID / LOW band — zero
  third-party runtime dependencies.
- `Gate` API (`mcp_confidence.gate`) with `from_logprobs`, `from_provider_details`,
  `from_openai`, and `from_dict` entry points, all driven by a `GateConfig`.
- `GateConfig` (`mcp_confidence.config`) with validation and `from_env` loading
  from `MCP_CONFIDENCE_*` environment variables.
- Adapters for OpenAI chat completions (`mcp_confidence.adapters.openai`) and
  pydantic-ai model responses (`mcp_confidence.adapters.pydantic_ai`).
- Calibration engine and `mcp-confidence` CLI (`calibrate`, `score`, `serve`)
  for honest, per-model threshold selection from audit logs.
- One-command MCP server (`mcp_confidence.mcp_server`) so a cloud "director" can
  delegate generation to a local OpenAI-compatible worker and receive a
  confidence band with every answer. Heavy `mcp`/`openai` extras are lazily
  imported only when serving.
