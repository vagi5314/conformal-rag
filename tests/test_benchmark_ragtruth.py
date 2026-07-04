"""Tests for RAGTruth benchmark plugin."""

import pytest
from benchmarks.ragtruth import RAGTruthBenchmark


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
