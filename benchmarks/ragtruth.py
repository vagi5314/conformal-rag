"""RAGTruth benchmark plugin.

RAGTruth ships real LLM-generated responses with hallucination labels, making
it a natural FDR test: the model's ``output`` is passed as the response under
test and the dataset's hallucination label is the ground-truth verdict.
"""

from __future__ import annotations

from typing import Optional

from benchmarks import register_benchmark
from benchmarks._common import aggregate_results, run_engine_sample
from benchmarks.base import BaseBenchmark, LatencyTrace, Result, Sample

MAX_SAMPLES = 1000


@register_benchmark
class RAGTruthBenchmark(BaseBenchmark):
    """RAGTruth: real LLM hallucination benchmark (native labels)."""

    name = "ragtruth"
    description = "Real LLM hallucinations — worst-case for embedding detection"

    async def load_dataset(self, limit: int = MAX_SAMPLES) -> list[Sample]:
        """Load RAGTruth and map each row to a labeled sample."""
        from datasets import load_dataset

        dataset = load_dataset("wandb/RAGTruth-processed", split=f"train[:{limit}]")
        samples: list[Sample] = []
        for i, item in enumerate(dataset):
            hl = item.get("hallucination_labels_processed", {}) or {}
            has_hallucination = (
                hl.get("evident_conflict", 0) + hl.get("baseless_info", 0)
            ) > 0
            samples.append(
                Sample(
                    id=f"ragtruth_{i}",
                    question=item.get("query", ""),
                    context=item.get("context", ""),
                    ground_truth=item.get("output", ""),
                    metadata={
                        "task_type": item.get("task_type", "unknown"),
                        "model": item.get("model", "unknown"),
                        "quality": item.get("quality", "unknown"),
                        "evident_conflict": hl.get("evident_conflict", 0),
                        "baseless_info": hl.get("baseless_info", 0),
                        "has_hallucination": has_hallucination,
                    },
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
