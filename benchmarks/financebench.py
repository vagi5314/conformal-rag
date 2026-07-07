"""FinanceBench benchmark plugin.

FinanceBench provides expert-written financial QA. It does not ship a
hallucination split, so each item is stress-tested with deterministic,
offline hallucination injection (a wrong entity drawn from the same context)
alongside the faithful gold answer.
"""

from __future__ import annotations

from typing import Optional

from benchmarks import register_benchmark
from benchmarks._common import (
    aggregate_results,
    build_labeled_pair,
    run_engine_sample,
)
from benchmarks.base import BaseBenchmark, LatencyTrace, Result, Sample

MAX_SAMPLES = 1000


@register_benchmark
class FinanceBenchBenchmark(BaseBenchmark):
    """FinanceBench: financial QA with injected hallucination stress test."""

    name = "financebench"
    description = "Financial QA — faithful + injected hallucination stress test"

    async def load_dataset(self, limit: int = MAX_SAMPLES) -> list[Sample]:
        """Load FinanceBench and expand each item into a labeled pair."""
        from datasets import load_dataset

        dataset = load_dataset("PatronusAI/financebench", split=f"train[:{limit}]")
        samples: list[Sample] = []
        for i, item in enumerate(dataset):
            question = item.get("question", "")
            context = item.get("context", "")
            answer = item.get("answer") or item.get("gold_answer") or ""
            if not question or not context or not answer:
                continue
            samples.extend(
                build_labeled_pair(
                    base_id=f"financebench_{i}",
                    question=question,
                    context=context,
                    answer=answer,
                    extra_metadata={"source": "financebench"},
                )
            )
        return samples

    async def run_sample(
        self,
        sample: Sample,
        engine,
        tracer: Optional[LatencyTrace] = None,
        cost_tracker: Optional[object] = None,
    ) -> Result:
        return run_engine_sample(sample, engine, tracer=tracer, cost_tracker=cost_tracker)

    async def aggregate(self, results: list[Result]):
        return aggregate_results(results)
