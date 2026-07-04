"""RAGTruth benchmark plugin."""

from __future__ import annotations

from benchmarks import register_benchmark
from benchmarks.base import (
    BaseBenchmark,
    Sample,
    Result,
    Aggregation,
)


@register_benchmark
class RAGTruthBenchmark(BaseBenchmark):
    """RAGTruth: Real LLM hallucination benchmark."""

    name = "ragtruth"
    description = "Real LLM hallucinations — worst-case for embedding detection"

    async def load_dataset(self) -> list[Sample]:
        """Load RAGTruth dataset from HuggingFace."""
        from datasets import load_dataset

        dataset = load_dataset("wandb/RAGTruth-processed", split="train[:50]")
        samples = []
        for i, item in enumerate(dataset):
            hl = item.get("hallucination_labels_processed", {})
            samples.append(
                Sample(
                    id=f"ragtruth_{i}",
                    question=item["query"],
                    context=item.get("context", ""),
                    ground_truth=item.get("output", ""),
                    metadata={
                        "task_type": item.get("task_type", "unknown"),
                        "model": item.get("model", "unknown"),
                        "quality": item.get("quality", "unknown"),
                        "evident_conflict": hl.get("evident_conflict", 0),
                        "baseless_info": hl.get("baseless_info", 0),
                        "has_hallucination": (
                            hl.get("evident_conflict", 0) + hl.get("baseless_info", 0)
                        )
                        > 0,
                    },
                )
            )
        return samples

    async def run_sample(self, sample: Sample, engine) -> Result:
        """Run the conformal engine on a single sample."""
        prediction = engine.predict(sample.question, sample.context, sample.ground_truth)
        return Result(
            sample_id=sample.id,
            prediction_set=[] if prediction.abstained else [sample.ground_truth],
            abstained=prediction.abstained,
            confidence=1.0 - prediction.risk_score,
            metadata={"has_hallucination": sample.metadata.get("has_hallucination", False)},
        )

    async def aggregate(self, results: list[Result]) -> Aggregation:
        """Compute FDR, coverage, abstention rate."""
        total = len(results)
        accepted = [r for r in results if not r.abstained]
        false_positives = sum(
            1 for r in accepted if r.metadata.get("has_hallucination", False)
        )
        fdr = false_positives / len(accepted) if accepted else 0.0
        abstained = sum(1 for r in results if r.abstained)
        return Aggregation(
            fdr=fdr,
            coverage=1.0 - (abstained / total) if total > 0 else 0.0,
            abstention_rate=abstained / total if total > 0 else 0.0,
            total_samples=total,
        )
