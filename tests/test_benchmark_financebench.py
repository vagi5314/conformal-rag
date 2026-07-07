"""Tests for FinanceBench benchmark plugin (offline, mocked dataset)."""

from unittest.mock import patch

from benchmarks.financebench import FinanceBenchBenchmark
from benchmarks.base import Sample
from conformal_rag.engine import ConformalRAGEngine
from conformal_rag.models import ConformalResult, ScoringBreakdown


def _fake_accept(self, query, context, response, domain=None):
    return ConformalResult(
        query=query, response=response,
        risk_score=0.9, risk_bound=0.05, abstained=False,
        confidence_interval=(0.0, 1.0),
        scoring_breakdown=ScoringBreakdown(nli_score=0.9, final_score=0.9, nli_latency_ms=1.0),
        latency_ms=1.5,
    )


_FAKE = [
    {
        "question": "What is the total revenue?",
        "context": "Company A reported revenue of 1M. Company B reported 2M.",
        "answer": "1M",
    }
]


async def test_financebench_load_dataset_pairs():
    with patch("datasets.load_dataset", return_value=_FAKE):
        samples = await FinanceBenchBenchmark().load_dataset()
    assert len(samples) == 2
    assert {s.metadata["variant"] for s in samples} == {"faithful", "hallucinated"}
    assert samples[0].ground_truth == "1M"


async def test_financebench_run_sample_shape():
    benchmark = FinanceBenchBenchmark()
    sample = Sample(id="f", question="Q", context="C", ground_truth="1M",
                    metadata={"has_hallucination": False})
    engine = ConformalRAGEngine(alpha=0.05)
    engine._threshold = 0.45
    with patch.object(ConformalRAGEngine, "predict", _fake_accept):
        result = await benchmark.run_sample(sample, engine)
    assert result.abstained is False
    assert result.latency_trace is not None
