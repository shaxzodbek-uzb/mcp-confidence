# 04 - Calibrating thresholds from an audit log

The default gate thresholds (`HIGH=-1.5`, `LOW=-3.5` in mean-logprob nats) are
**provisional guesses**. They are model-specific and almost certainly wrong for
your traffic. Before you trust auto-accept routing, calibrate them on a log of
your own model's real outputs.

## Run it

From the repo root (zero extras needed — the calibrator is stdlib-only):

```bash
mcp-confidence calibrate --audit examples/04_calibrate/sample_audit.jsonl
```

or, without installing the package:

```bash
PYTHONPATH=src python3 -m mcp_confidence.cli calibrate \
    --audit examples/04_calibrate/sample_audit.jsonl
```

Expected output (numbers from `sample_audit.jsonl`):

```
worker_delegated events: 24  |  logprobs unavailable: 2 (8%)

-- mean_logprob distribution (n=22) --
  min  -4.260   p10 -3.440   p25 -2.210
  med  -1.360   mean -1.622
  p75  -0.740   p90 -0.380   max -0.210
  band split (current thresholds): high=10 (45%)  low=4 (18%)  mid=8 (36%)

-- risk-coverage (metric=mean_logprob) --
  RECOMMENDED: high_threshold=-2.05  low_threshold=-4.26
    auto-accept coverage 75%  (bad slipped through: 20%)
    human-ask coverage  6%  (good needlessly asked: 0%)
  Set MCP_CONFIDENCE_HIGH_THRESHOLD / MCP_CONFIDENCE_LOW_THRESHOLD to these, then enable routing.
```

Tune the trade-off with `--min-accept-coverage` (how much traffic you insist on
auto-accepting) and `--max-ask-coverage` (the most you are willing to send to a
human). Calibrate on `score` instead of `mean_logprob` with `--metric score`.

## The audit format

`sample_audit.jsonl` is a JSONL file (one JSON object per line). Each row is one
`worker_delegated` confidence event:

| field                | purpose                                                    |
| -------------------- | ---------------------------------------------------------- |
| `event`              | event name; the loader keeps only `worker_delegated` rows  |
| `logprobs_available` | `false` rows are counted toward the "unavailable" rate     |
| `mean_logprob`       | the metric calibrated by default                           |
| `score`              | the combined metric (`--metric score`)                     |
| `band`               | the band recorded at generation time (for the band split)  |
| `human_label`        | **optional** `"good"` / `"bad"` — drives the recommendation |

## On labeling

The **distribution** (percentiles + band split + unavailable rate) needs no
labels — run it first just to see where your traffic's confidence actually sits.

The **risk-coverage recommendation** only appears when some rows carry a
`human_label` of `good` or `bad`. You produce those by sampling real worker
outputs and judging them: was this answer good enough to ship unverified, or not?
A few dozen honest labels per task type are enough to move off the default
guesses. In `sample_audit.jsonl`, 16 of the 25 rows are labeled (the rest are
unlabeled live traffic and two `logprobs_available=false` rows) — which is why a
recommendation is produced.
