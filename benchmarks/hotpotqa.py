"""HotpotQA benchmark plugin.

HotpotQA contains only correct answers, so a naive run (feeding the correct
answer as the response) yields a trivially faithful set and FDR = 0. To make
the FDR estimate meaningful we stress-test the engine with deterministic,
offline hallucination injection: each item expands into a faithful sample
(label 0) and a hard-negative hallucinated sample (label 1) built from a
different entity in the same context.
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
class HotpotQABenchmark(BaseBenchmark):
    """HotpotQA: multi-hop reasoning benchmark (with hallucination stress test)."""

    name = "hotpotqa"
    description = "Multi-hop QA — faithful + injected hallucination stress test"

    async def load_dataset(self, limit: int = MAX_SAMPLES) -> list[Sample]:
        """Load HotpotQA and expand each item into a labeled faithful/hallucinated pair."""
        from datasets import load_dataset

        dataset = load_dataset("hotpot_qa", "distractor", split=f"train[:{limit}]")
        samples: list[Sample] = []
        for i, item in enumerate(dataset):
            context = " ".join(
                " ".join(sent_group) for sent_group in item["context"]["sentences"]
            )
            samples.extend(
                build_labeled_pair(
                    base_id=f"hotpotqa_{i}",
                    question=item["question"],
                    context=context,
                    answer=item["answer"],
                    extra_metadata={"type": item["type"], "level": item["level"]},
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
        """Run the conformal engine on a single sample."""
        return run_engine_sample(sample, engine, tracer=tracer, cost_tracker=cost_tracker)

    async def aggregate(self, results: list[Result]):
        """Compute FDR, coverage, abstention rate."""
        return aggregate_results(results)
