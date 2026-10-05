"""Tests for HotpotQA benchmark plugin."""

import pytest
from unittest.mock import patch
from benchmarks.hotpotqa import HotpotQABenchmark
from benchmarks.base import Sample


@pytest.mark.asyncio
async def test_hotpotqa_load_dataset():
    benchmark = HotpotQABenchmark()
    samples = await benchmark.load_dataset()
    assert len(samples) > 0
    assert samples[0].question is not None
    assert samples[0].ground_truth is not None


@pytest.mark.asyncio
async def test_hotpotqa_sample_format():
    benchmark = HotpotQABenchmark()
    samples = await benchmark.load_dataset()
    sample = samples[0]
    assert hasattr(sample, "id")
    assert hasattr(sample, "question")
    assert hasattr(sample, "context")
    assert hasattr(sample, "ground_truth")


@pytest.mark.asyncio
async def test_hotpotqa_run_sample_with_engine():
    from conformal_rag.engine import ConformalRAGEngine
    from conformal_rag.models import ScoringBreakdown

    benchmark = HotpotQABenchmark()
    sample = Sample(
        id="test_0",
        question="What is the capital of France?",
        context="France is a country in Europe. Its capital is Paris.",
        ground_truth="Paris",
    )

    engine = ConformalRAGEngine(alpha=0.05)
    engine._threshold = 0.45

    mock_breakdown = ScoringBreakdown(
        nli_score=0.85,
        final_score=0.85,
        nli_latency_ms=10.0,
    )
    with patch.object(engine.scoring, "score", return_value=mock_breakdown) as mock_score:
        result = await benchmark.run_sample(sample, engine)

    assert result.sample_id == "test_0"
    assert isinstance(result.prediction_set, list)
    assert result.abstained is False
    assert isinstance(result.confidence, float)
    mock_score.assert_called_once_with(sample.question, sample.context, sample.ground_truth)


@pytest.mark.asyncio
async def test_hotpotqa_run_sample_abstains():
    from conformal_rag.engine import ConformalRAGEngine
    from conformal_rag.models import ScoringBreakdown

    benchmark = HotpotQABenchmark()
    sample = Sample(
        id="test_1",
        question="What is quantum entanglement?",
        context="Some vague context.",
        ground_truth="Quantum entanglement is a phenomenon.",
    )

    engine = ConformalRAGEngine(alpha=0.05)
    engine._threshold = 0.45

    mock_breakdown = ScoringBreakdown(
        nli_score=0.2,
        final_score=0.2,
        nli_latency_ms=10.0,
    )
    with patch.object(engine.scoring, "score", return_value=mock_breakdown):
        result = await benchmark.run_sample(sample, engine)

    assert result.abstained is True
    assert result.prediction_set == []
