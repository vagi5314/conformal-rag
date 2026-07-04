"""Tests for RAGTruth benchmark plugin."""

import pytest
from unittest.mock import patch
from benchmarks.ragtruth import RAGTruthBenchmark
from benchmarks.base import Sample


@pytest.mark.asyncio
async def test_ragtruth_load_dataset():
    benchmark = RAGTruthBenchmark()
    samples = await benchmark.load_dataset()
    assert len(samples) > 0
    assert samples[0].question is not None
    assert samples[0].ground_truth is not None


@pytest.mark.asyncio
async def test_ragtruth_metadata():
    benchmark = RAGTruthBenchmark()
    samples = await benchmark.load_dataset()
    sample = samples[0]
    assert "task_type" in sample.metadata or "model" in sample.metadata


@pytest.mark.asyncio
async def test_ragtruth_run_sample_with_engine():
    from conformal_rag.engine import ConformalRAGEngine
    from conformal_rag.models import ScoringBreakdown

    benchmark = RAGTruthBenchmark()
    sample = Sample(
        id="test_0",
        question="What is quantum computing?",
        context="Quantum computing uses qubits.",
        ground_truth="Quantum computing uses quantum mechanics.",
        metadata={"task_type": "summarization", "model": "gpt-4"},
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
