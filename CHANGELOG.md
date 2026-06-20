# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
