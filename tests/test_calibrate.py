"""Tests for mcp_confidence.calibrate — pure load/distribution/risk-coverage."""

from __future__ import annotations

import json
import math

import pytest

from mcp_confidence import calibrate


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _row(mean_lp, *, event="worker_delegated", available=True, band="high", label=None, score=None):
    r = {
        "event": event,
        "logprobs_available": available,
        "mean_logprob": mean_lp,
        "band": band,
    }
    if score is not None:
        r["score"] = score
    if label is not None:
        r["human_label"] = label
    return r


# -- load_events -------------------------------------------------------------
def test_load_events_filters_by_event_and_availability(tmp_path):
    p = tmp_path / "audit.jsonl"
    _write_jsonl(
        p,
        [
            _row(-0.5),
            _row(-1.0, available=False),
            _row(-2.0, event="something_else"),
            _row(-0.7),
        ],
    )
    rows, total, unavailable = calibrate.load_events(p)
    assert total == 3  # two available + one unavailable, all worker_delegated
    assert unavailable == 1
    assert len(rows) == 2
    assert all(r["logprobs_available"] for r in rows)


def test_load_events_event_name_none_accepts_any(tmp_path):
    p = tmp_path / "audit.jsonl"
    _write_jsonl(
        p,
        [
            _row(-0.5, event="a"),
            _row(-0.6, event="b"),
            {"logprobs_available": True, "mean_logprob": -0.7},  # no event field
        ],
    )
    rows, total, unavailable = calibrate.load_events(p, event_name=None)
    assert total == 3
    assert unavailable == 0
    assert len(rows) == 3


def test_load_events_skips_blank_and_bad_json(tmp_path):
    p = tmp_path / "audit.jsonl"
    p.write_text(
        "\n".join(
            [
                json.dumps(_row(-0.5)),
                "",
                "   ",
                "{not valid json",
                json.dumps(["a", "list", "not", "dict"]),
                json.dumps(_row(-0.9)),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rows, total, unavailable = calibrate.load_events(p)
    assert total == 2
    assert len(rows) == 2


def test_load_events_all_unavailable(tmp_path):
    p = tmp_path / "audit.jsonl"
    _write_jsonl(p, [_row(-0.5, available=False), _row(-0.6, available=False)])
    rows, total, unavailable = calibrate.load_events(p)
    assert total == 2
    assert unavailable == 2
    assert rows == []


# -- distribution ------------------------------------------------------------
def test_distribution_keys_and_values():
    rows = [
        _row(v, band=b) for v, b in [(-0.5, "high"), (-1.0, "high"), (-2.0, "mid"), (-3.0, "low")]
    ]
    d = calibrate.distribution(rows, "mean_logprob")
    assert set(d) == {"n", "min", "p10", "p25", "median", "mean", "p75", "p90", "max", "band_split"}
    assert d["n"] == 4
    assert d["min"] == -3.0
    assert d["max"] == -0.5
    assert d["mean"] == pytest.approx((-0.5 - 1.0 - 2.0 - 3.0) / 4)
    assert d["median"] == pytest.approx(-1.5)
    assert d["band_split"] == {"high": 2, "mid": 1, "low": 1}


def test_distribution_empty_rows_all_nan():
    d = calibrate.distribution([], "mean_logprob")
    assert d["n"] == 0
    assert math.isnan(d["min"])
    assert math.isnan(d["mean"])
    assert d["band_split"] == {}


def test_distribution_ignores_non_numeric_metric():
    rows = [_row(-0.5), {"band": "mid", "mean_logprob": "oops"}]
    d = calibrate.distribution(rows, "mean_logprob")
    assert d["n"] == 1  # only the numeric one counted in vals
    # band split still counts all rows
    assert sum(d["band_split"].values()) == 2


def test_distribution_band_split_missing_band_field():
    rows = [{"mean_logprob": -0.5}, {"mean_logprob": -0.6}]
    d = calibrate.distribution(rows, "mean_logprob")
    assert d["band_split"] == {"?": 2}


def test_distribution_score_metric():
    rows = [_row(-0.5, score=-0.8), _row(-1.0, score=-1.6)]
    d = calibrate.distribution(rows, "score")
    assert d["n"] == 2
    assert d["min"] == -1.6
    assert d["max"] == -0.8


# -- risk_coverage -----------------------------------------------------------
def _labeled_rows():
    # Clean separable set: good outputs sit high, bad outputs sit low.
    rows = []
    for v in [-0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8]:
        rows.append(_row(v, label="good"))
    for v in [-4.0, -4.5, -5.0]:
        rows.append(_row(v, label="bad"))
    return rows


def test_risk_coverage_returns_recommendation():
    rows = _labeled_rows()
    best = calibrate.risk_coverage(rows, "mean_logprob", min_accept_cov=0.5, max_ask_cov=0.4)
    assert best is not None
    assert set(best) == {"high", "low", "cov_hi", "cov_lo", "risk_hi", "false_ask", "cost"}
    assert best["high"] > best["low"]
    # well-separated data: no bad slips into accept, no good needlessly asked
    assert best["risk_hi"] == pytest.approx(0.0)
    assert best["false_ask"] == pytest.approx(0.0)


def test_risk_coverage_none_without_labels():
    rows = [_row(-0.5), _row(-0.6)]  # no human_label
    assert calibrate.risk_coverage(rows, "mean_logprob", 0.5, 0.4) is None


def test_risk_coverage_none_when_infeasible():
    rows = _labeled_rows()
    # Demand near-total accept coverage AND near-zero ask coverage -> infeasible.
    best = calibrate.risk_coverage(rows, "mean_logprob", min_accept_cov=0.99, max_ask_cov=0.0)
    assert best is None


def test_risk_coverage_ignores_non_good_bad_labels():
    rows = [_row(-0.5, label="meh"), _row(-0.6, label="unknown")]
    assert calibrate.risk_coverage(rows, "mean_logprob", 0.5, 0.4) is None


def test_risk_coverage_cost_is_sum_of_risks():
    rows = _labeled_rows()
    best = calibrate.risk_coverage(rows, "mean_logprob", 0.5, 0.4)
    assert best["cost"] == pytest.approx(best["risk_hi"] + best["false_ask"])


def test_risk_coverage_coverage_constraints_respected():
    rows = _labeled_rows()
    best = calibrate.risk_coverage(rows, "mean_logprob", min_accept_cov=0.5, max_ask_cov=0.4)
    assert best["cov_hi"] >= 0.5
    assert best["cov_lo"] <= 0.4
