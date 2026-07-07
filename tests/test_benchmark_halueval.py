"""Tests for HaluEval benchmark plugin (offline, mocked dataset)."""

from unittest.mock import patch

from benchmarks.halueval import HaluEvalBenchmark
from benchmarks.base import Sample
from conformal_rag.engine import ConformalRAGEngine
from conformal_rag.models import ConformalResult, ScoringBreakdown


def _fake_accept(self, query, context, response, domain=None):
    return ConformalResult(
        query=query, response=response,
        risk_score=0.9, risk_bound=0.05, abstained=False,
        confidence_interval=(0.0, 1.0),
        scoring_breakdown=ScoringBreakdown(nli_score=0.9, final_score=0.9, nli_latency_ms=1.5),
        latency_ms=2.0,
    )


_FAKE = [
    {
        "question": "What is the capital of France?",
        "knowledge": "Paris is the capital of France. London is a city in England.",
        "right_answer": "Paris",
        "answer": "London",
        "hallucinated": True,
    }
]


async def test_halueval_load_dataset_labels():
    with patch("datasets.load_dataset", return_value=_FAKE):
        samples = await HaluEvalBenchmark().load_dataset()
    assert len(samples) == 2
    faithful = [s for s in samples if not s.metadata["has_hallucination"]][0]
    hallucinated = [s for s in samples if s.metadata["has_hallucination"]][0]
    assert faithful.ground_truth == "Paris"
    assert hallucinated.ground_truth == "London"
    assert faithful.metadata["variant"] == "faithful"


async def test_halueval_run_sample_shape():
    benchmark = HaluEvalBenchmark()
    sample = Sample(id="h", question="Q", context="C", ground_truth="Paris",
                    metadata={"has_hallucination": False})
    engine = ConformalRAGEngine(alpha=0.05)
    engine._threshold = 0.45
    with patch.object(ConformalRAGEngine, "predict", _fake_accept):
        result = await benchmark.run_sample(sample, engine)
    assert result.abstained is False
    assert result.latency_trace is not None
    assert result.latency_trace.nli_scoring_ms == 1.5
