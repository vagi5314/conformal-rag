"""Tests for shared benchmark helpers."""

from benchmarks._common import (
    aggregate_results,
    build_labeled_pair,
    inject_hallucination,
    run_engine_sample,
)
from benchmarks.base import Result, Sample
from conformal_rag.engine import ConformalRAGEngine
from conformal_rag.models import ConformalResult, ScoringBreakdown


def test_inject_hallucination_uses_context_entity():
    ctx = "Paris is the capital of France. London is a large city in England."
    wrong = inject_hallucination(ctx, "Paris")
    assert wrong.lower() != "paris"
    assert wrong in ctx  # a real entity from the context


def test_inject_hallucination_fallback():
    wrong = inject_hallucination("no capitalized entities here at all", "answer")
    assert "incorrect" in wrong


def test_build_labeled_pair_tags():
    pair = build_labeled_pair("x1", "Q?", "Ctx about Paris.", "Paris")
    assert len(pair) == 2
    assert pair[0].metadata["has_hallucination"] is False
    assert pair[1].metadata["has_hallucination"] is True
    assert pair[0].ground_truth == "Paris"
    assert pair[1].ground_truth != "Paris"


def test_aggregate_results_fdr():
    results = [
        Result(sample_id="a", prediction_set=["x"], abstained=False, confidence=0.9,
               metadata={"has_hallucination": False}),
        Result(sample_id="b", prediction_set=[], abstained=False, confidence=0.9,
               metadata={"has_hallucination": True}),
        Result(sample_id="c", prediction_set=[], abstained=True, confidence=0.1,
               metadata={"has_hallucination": True}),
    ]
    agg = aggregate_results(results)
    # accepted = a, b ; false positives (accepted & hallucination) = b ; fdr = 1/2
    assert agg.fdr == 0.5
    assert agg.coverage == 2 / 3
    assert agg.abstention_rate == 1 / 3
    assert agg.total_samples == 3


def test_aggregate_results_all_abstained_fdr_zero():
    results = [
        Result(sample_id="a", prediction_set=[], abstained=True, confidence=0.1,
               metadata={"has_hallucination": True}),
    ]
    assert aggregate_results(results).fdr == 0.0


def _fake_result(abstain: bool) -> ConformalResult:
    score = 0.1 if abstain else 0.9
    return ConformalResult(
        query="q", response="r",
        risk_score=score, risk_bound=0.05, abstained=abstain,
        confidence_interval=(0.0, 1.0),
        scoring_breakdown=ScoringBreakdown(
            nli_score=score, final_score=score, nli_latency_ms=2.0
        ),
        latency_ms=3.0,
    )


def test_run_engine_sample_captures_traces(monkeypatch):
    engine = ConformalRAGEngine(alpha=0.05)
    engine._threshold = 0.45

    def fake_predict(self, query, context, response, domain=None):
        return _fake_result(abstain=False)

    monkeypatch.setattr(ConformalRAGEngine, "predict", fake_predict)

    sample = Sample(id="s", question="Q", context="C", ground_truth="A")
    result = run_engine_sample(sample, engine)
    assert result.abstained is False
    assert result.latency_trace is not None
    assert result.latency_trace.total_ms == 3.0
    assert result.latency_trace.nli_scoring_ms == 2.0
