"""Tests for HotpotQA benchmark plugin."""

import pytest
from benchmarks.hotpotqa import HotpotQABenchmark


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
