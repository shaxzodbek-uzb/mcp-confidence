"""Tests for mcp_confidence.cli.main — in-process (no subprocess), no network."""

from __future__ import annotations

import json

import pytest

from mcp_confidence import cli


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


# -- score subcommand --------------------------------------------------------
def test_score_basic(capsys):
    rc = cli.main(["score", "--logprobs", "-0.2,-0.1"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "band" in out
    assert "high" in out
    assert "mean_logprob" in out
    assert "token_count  2" in out


def test_score_low_band(capsys):
    rc = cli.main(["score", "--logprobs", "-4.0,-5.0,-6.0"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "low" in out


def test_score_empty_logprobs(capsys):
    rc = cli.main(["score", "--logprobs", ""])
    out = capsys.readouterr().out
    assert rc == 2
    assert "empty" in out.lower()


def test_score_unparseable(capsys):
    rc = cli.main(["score", "--logprobs", "abc,def"])
    out = capsys.readouterr().out
    assert rc == 2
    assert "could not parse" in out.lower()


def test_score_config_from_env(capsys, monkeypatch):
    # Generous high threshold via env -> a -2.0-ish score banded HIGH.
    monkeypatch.setenv("MCP_CONFIDENCE_HIGH_THRESHOLD", "-2.5")
    monkeypatch.setenv("MCP_CONFIDENCE_LOW_THRESHOLD", "-5.0")
    rc = cli.main(["score", "--logprobs", "-2.0,-2.0", "--config-from-env"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "high" in out


def test_score_trims_trailing_comma(capsys):
    rc = cli.main(["score", "--logprobs", "-0.2,-0.1,"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "token_count  2" in out


# -- calibrate subcommand ----------------------------------------------------
def test_calibrate_missing_file(capsys, tmp_path):
    rc = cli.main(["calibrate", "--audit", str(tmp_path / "nope.jsonl")])
    out = capsys.readouterr().out
    assert rc == 2
    assert "not found" in out.lower()


def test_calibrate_distribution_only(capsys, tmp_path):
    p = tmp_path / "audit.jsonl"
    _write_jsonl(
        p,
        [
            {
                "event": "worker_delegated",
                "logprobs_available": True,
                "mean_logprob": -0.5,
                "band": "high",
            },
            {
                "event": "worker_delegated",
                "logprobs_available": True,
                "mean_logprob": -2.0,
                "band": "mid",
            },
        ],
    )
    rc = cli.main(["calibrate", "--audit", str(p)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "distribution" in out.lower()
    assert "band split" in out.lower()
    # no labels -> risk-coverage skipped
    assert "SKIPPED" in out


def test_calibrate_unavailable_warning(capsys, tmp_path):
    p = tmp_path / "audit.jsonl"
    rows = [{"event": "worker_delegated", "logprobs_available": False, "mean_logprob": -0.5}] * 3
    rows.append(
        {
            "event": "worker_delegated",
            "logprobs_available": True,
            "mean_logprob": -0.5,
            "band": "high",
        }
    )
    _write_jsonl(p, rows)
    rc = cli.main(["calibrate", "--audit", str(p)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "unavailable" in out.lower()
    assert ">20%" in out


def test_calibrate_no_usable_events(capsys, tmp_path):
    p = tmp_path / "audit.jsonl"
    _write_jsonl(
        p, [{"event": "worker_delegated", "logprobs_available": False, "mean_logprob": -0.5}]
    )
    rc = cli.main(["calibrate", "--audit", str(p)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "no usable events" in out.lower()


def test_calibrate_with_labels_recommends(capsys, tmp_path):
    p = tmp_path / "audit.jsonl"
    rows = []
    for v in [-0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8]:
        rows.append(
            {
                "event": "worker_delegated",
                "logprobs_available": True,
                "mean_logprob": v,
                "band": "high",
                "human_label": "good",
            }
        )
    for v in [-4.0, -4.5, -5.0]:
        rows.append(
            {
                "event": "worker_delegated",
                "logprobs_available": True,
                "mean_logprob": v,
                "band": "low",
                "human_label": "bad",
            }
        )
    _write_jsonl(p, rows)
    rc = cli.main(
        [
            "calibrate",
            "--audit",
            str(p),
            "--min-accept-coverage",
            "0.5",
            "--max-ask-coverage",
            "0.4",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "RECOMMENDED" in out
    assert "high_threshold" in out
    assert "MCP_CONFIDENCE_HIGH_THRESHOLD" in out


def test_calibrate_infeasible_constraints(capsys, tmp_path):
    p = tmp_path / "audit.jsonl"
    rows = []
    for v in [-0.2, -0.3, -0.4]:
        rows.append(
            {
                "event": "worker_delegated",
                "logprobs_available": True,
                "mean_logprob": v,
                "band": "high",
                "human_label": "good",
            }
        )
    for v in [-4.0, -5.0]:
        rows.append(
            {
                "event": "worker_delegated",
                "logprobs_available": True,
                "mean_logprob": v,
                "band": "low",
                "human_label": "bad",
            }
        )
    _write_jsonl(p, rows)
    rc = cli.main(
        [
            "calibrate",
            "--audit",
            str(p),
            "--min-accept-coverage",
            "0.99",
            "--max-ask-coverage",
            "0.0",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "No (high, low) cell" in out


def test_calibrate_event_name_wildcard(capsys, tmp_path):
    p = tmp_path / "audit.jsonl"
    _write_jsonl(
        p,
        [
            {"event": "anything", "logprobs_available": True, "mean_logprob": -0.5, "band": "high"},
            {"logprobs_available": True, "mean_logprob": -0.6, "band": "high"},
        ],
    )
    rc = cli.main(["calibrate", "--audit", str(p), "--event-name", "*"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "distribution" in out.lower()


def test_calibrate_metric_score(capsys, tmp_path):
    p = tmp_path / "audit.jsonl"
    _write_jsonl(
        p,
        [
            {
                "event": "worker_delegated",
                "logprobs_available": True,
                "mean_logprob": -0.5,
                "score": -0.8,
                "band": "high",
            },
            {
                "event": "worker_delegated",
                "logprobs_available": True,
                "mean_logprob": -2.0,
                "score": -2.5,
                "band": "mid",
            },
        ],
    )
    rc = cli.main(["calibrate", "--audit", str(p), "--metric", "score"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "score distribution" in out.lower()


# -- argparse-level behavior -------------------------------------------------
def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "mcp-confidence" in out
    assert "calibrate" in out
    assert "score" in out
    assert "serve" in out


def test_calibrate_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["calibrate", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--audit" in out
    assert "--metric" in out


def test_no_subcommand_errors(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main([])
    assert exc.value.code != 0


def test_score_requires_logprobs(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["score"])
    assert exc.value.code != 0
