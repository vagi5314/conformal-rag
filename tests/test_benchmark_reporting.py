"""Tests for Markdown report generation."""

from benchmarks.reporting import (
    latency_percentiles,
    render_aggregate_report,
    render_run_report,
)
from benchmarks.base import LatencyTrace


def test_latency_percentiles():
    traces = [
        LatencyTrace(nli_scoring_ms=1.0, total_ms=10.0),
        LatencyTrace(nli_scoring_ms=3.0, total_ms=30.0),
        LatencyTrace(nli_scoring_ms=2.0, total_ms=20.0),
    ]
    pct = latency_percentiles(traces)
    assert pct["nli_scoring_ms"]["p50"] == 2.0
    assert pct["nli_scoring_ms"]["p99"] == 3.0
    assert pct["total_ms"]["mean"] == 20.0


def _result_dict(fdr=0.03, coverage=0.95, abstention=0.05, total=100):
    return {
        "benchmark": "hotpotqa",
        "timestamp": "2026-07-04T00:00:00+00:00",
        "config": {"alpha": 0.05, "calibration_size": 80, "test_size": 20},
        "metrics": {
            "fdr": fdr,
            "coverage": coverage,
            "abstention_rate": abstention,
            "total_samples": total,
        },
        "latency_summary": {
            "nli_scoring_ms": {"p50": 900.0, "p95": 1100.0, "p99": 1200.0, "mean": 1000.0},
            "total_ms": {"p50": 950.0, "p95": 1150.0, "p99": 1250.0, "mean": 1050.0},
        },
    }


def test_render_run_report_pass():
    md = render_run_report(_result_dict(fdr=0.03))
    assert "# Benchmark Report: `hotpotqa`" in md
    assert "PASS" in md
    assert "3.00%" in md  # fdr formatted


def test_render_run_report_fail():
    md = render_run_report(_result_dict(fdr=0.12))
    assert "FAIL" in md


def test_render_aggregate_report():
    rows = [_result_dict(fdr=0.03, total=100), _result_dict(fdr=0.12, total=50)]
    rows[1]["benchmark"] = "ragtruth"
    md = render_aggregate_report(rows)
    assert "Cross-Benchmark Aggregate" in md
    assert "hotpotqa" in md
    assert "ragtruth" in md
    assert "✅" in md and "❌" in md
