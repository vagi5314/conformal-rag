"""MultiHop-RAG benchmark plugin.

MultiHop-RAG is a multi-document reasoning dataset. It has no native
hallucination labels, so each item is stress-tested with deterministic,
offline hallucination injection against its gold documents as context.
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
class MultiHopRAGBenchmark(BaseBenchmark):
    """MultiHop-RAG: multi-document reasoning with injected hallucination stress test."""

    name = "multihoprag"
    description = "Multi-hop multi-doc QA — faithful + injected hallucination stress test"

    async def load_dataset(self, limit: int = MAX_SAMPLES) -> list[Sample]:
        """Load MultiHop-RAG and expand each item into a labeled pair."""
        from datasets import load_dataset

        dataset = load_dataset("yixuantt/MultiHopRAG", split=f"train[:{limit}]")
        samples: list[Sample] = []
        for i, item in enumerate(dataset):
            question = item.get("question", "")
            answer = item.get("answer", "")
            gold_docs = item.get("gold_docs", []) or []
            if isinstance(gold_docs, list) and gold_docs and isinstance(gold_docs[0], dict):
                context = " ".join(d.get("text", "") for d in gold_docs)
            else:
                context = " ".join(str(d) for d in gold_docs)
            if not question or not context or not answer:
                continue
            samples.extend(
                build_labeled_pair(
                    base_id=f"multihoprag_{i}",
                    question=question,
                    context=context,
                    answer=answer,
                    extra_metadata={"source": "multihoprag"},
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
