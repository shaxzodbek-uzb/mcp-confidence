"""Command-line interface for mcp-confidence.

Subcommands::

    mcp-confidence calibrate --audit PATH [--metric mean_logprob|score] ...
        Read a JSONL confidence log, print the metric distribution and band
        split, and (when rows carry good/bad labels) print a recommended
        (high, low) threshold pair from a risk-coverage sweep.

    mcp-confidence score --logprobs "-0.2,-0.5,-0.1" [--config-from-env]
        Quick sanity check: band + score + mean_logprob for a literal logprob
        list, using either default thresholds or GateConfig.from_env().

    mcp-confidence serve [--transport stdio]
        Start the MCP server so a cloud director can delegate generation to a
        local OpenAI-compatible worker and get a confidence band back. The heavy
        ``mcp`` / ``openai`` extras are imported lazily inside this command only.

This module imports with ZERO third-party dependencies; ``calibrate`` and
``score`` run with no extras installed. ``serve`` lazy-imports the server.
"""

from __future__ import annotations

import argparse

from . import calibrate as calib
from .config import GateConfig
from .gate import Gate


def _cmd_calibrate(args: argparse.Namespace) -> int:
    from pathlib import Path

    path = Path(args.audit)
    if not path.exists():
        print(f"audit file not found: {path}")
        return 2

    event_name = args.event_name if args.event_name != "*" else None
    rows, total, unavailable = calib.load_events(path, event_name=event_name)

    label = event_name if event_name is not None else "matching"
    if total:
        rate = unavailable / total
        print(f"{label} events: {total}  |  logprobs unavailable: {unavailable} ({rate:.0%})")
        if rate > 0.20:
            print(
                "  ! >20% unavailable — the gate is blind on too many calls. "
                "Check your server's sampling / max-logprobs setting before "
                "trusting it."
            )

    if not rows:
        print(
            "No usable events with logprobs. Make sure your log carries "
            "logprobs_available=true rows and generate traffic first."
        )
        return 0

    dist = calib.distribution(rows, args.metric)
    print(f"\n-- {args.metric} distribution (n={dist['n']}) --")
    if dist["n"] == 0:
        print("  (no data)")
    else:
        print(f"  min  {dist['min']:.3f}   p10 {dist['p10']:.3f}   p25 {dist['p25']:.3f}")
        print(f"  med  {dist['median']:.3f}   mean {dist['mean']:.3f}")
        print(f"  p75  {dist['p75']:.3f}   p90 {dist['p90']:.3f}   max {dist['max']:.3f}")
    bands = dist["band_split"]
    n_bands = sum(bands.values()) or 1
    print(
        "  band split (current thresholds): "
        + "  ".join(f"{b}={c} ({c / n_bands:.0%})" for b, c in sorted(bands.items()))
    )

    best = calib.risk_coverage(rows, args.metric, args.min_accept_coverage, args.max_ask_coverage)
    print(f"\n-- risk-coverage (metric={args.metric}) --")
    if best is None:
        labeled = [r for r in rows if r.get("human_label") in ("good", "bad")]
        if not labeled:
            print("  SKIPPED: no rows carry a 'human_label' of good/bad.")
            print("  Add labels to enable threshold recommendation. Until then use")
            print("  the distribution above to pick conservative bands by eye.")
        else:
            print(
                f"  No (high, low) cell satisfies coverage_high >= "
                f"{args.min_accept_coverage} AND coverage_low <= "
                f"{args.max_ask_coverage}. Relax the constraints or collect more data."
            )
        return 0

    print(f"  RECOMMENDED: high_threshold={best['high']}  low_threshold={best['low']}")
    print(
        f"    auto-accept coverage {best['cov_hi']:.0%}  "
        f"(bad slipped through: {best['risk_hi']:.0%})"
    )
    print(
        f"    human-ask coverage  {best['cov_lo']:.0%}  "
        f"(good needlessly asked: {best['false_ask']:.0%})"
    )
    print(
        "  Set MCP_CONFIDENCE_HIGH_THRESHOLD / MCP_CONFIDENCE_LOW_THRESHOLD to "
        "these, then enable routing."
    )
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    try:
        logprobs = [float(x) for x in args.logprobs.split(",") if x.strip() != ""]
    except ValueError:
        print(f"could not parse --logprobs as a comma-separated float list: {args.logprobs}")
        return 2
    if not logprobs:
        print("--logprobs is empty")
        return 2

    config = GateConfig.from_env() if args.config_from_env else GateConfig()
    result = Gate(config).from_logprobs(logprobs)
    print(f"band         {result.band.value}")
    print(f"score        {result.score:.4f}")
    print(f"mean_logprob {result.mean_logprob:.4f}")
    print(f"min_logprob  {result.min_logprob:.4f}")
    print(f"token_count  {result.token_count}")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    from . import mcp_server

    mcp_server.run(transport=args.transport)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mcp-confidence",
        description="A drop-in confidence gate for LLM agents.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_cal = sub.add_parser(
        "calibrate",
        help="analyze a JSONL confidence log and recommend thresholds",
    )
    p_cal.add_argument("--audit", required=True, help="audit JSONL path")
    p_cal.add_argument(
        "--metric",
        choices=["mean_logprob", "score"],
        default="mean_logprob",
        help="which confidence metric to calibrate on (default: mean_logprob)",
    )
    p_cal.add_argument(
        "--min-accept-coverage",
        type=float,
        default=0.70,
        help="minimum auto-accept coverage required (default 0.70)",
    )
    p_cal.add_argument(
        "--max-ask-coverage",
        type=float,
        default=0.15,
        help="maximum human-ask coverage allowed (default 0.15)",
    )
    p_cal.add_argument(
        "--event-name",
        default="worker_delegated",
        help="keep only rows whose 'event' equals this; use '*' for any row",
    )
    p_cal.set_defaults(func=_cmd_calibrate)

    p_score = sub.add_parser(
        "score",
        help="band/score a literal comma-separated logprob list (quick demo)",
    )
    p_score.add_argument(
        "--logprobs",
        required=True,
        help='comma-separated per-token logprobs, e.g. "-0.2,-0.5,-0.1"',
    )
    p_score.add_argument(
        "--config-from-env",
        action="store_true",
        help="read thresholds from MCP_CONFIDENCE_* env vars instead of defaults",
    )
    p_score.set_defaults(func=_cmd_score)

    p_serve = sub.add_parser(
        "serve",
        help="run the MCP confidence server (needs the [mcp] extra)",
    )
    p_serve.add_argument(
        "--transport",
        default="stdio",
        help="MCP transport (default: stdio)",
    )
    p_serve.set_defaults(func=_cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code; console_scripts wraps it."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
