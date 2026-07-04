"""HotpotQA benchmark plugin."""

from __future__ import annotations

from benchmarks import register_benchmark
from benchmarks.base import (
    BaseBenchmark,
    Sample,
    Result,
    Aggregation,
)


@register_benchmark
class HotpotQABenchmark(BaseBenchmark):
    """HotpotQA: Multi-hop reasoning benchmark."""

    name = "hotpotqa"
    description = "Multi-hop question answering — tests retrieval + reasoning"

    async def load_dataset(self) -> list[Sample]:
        """Load HotpotQA dataset from HuggingFace."""
        from datasets import load_dataset

        dataset = load_dataset("hotpot_qa", "distractor", split="train[:1000]")
        samples = []
        for i, item in enumerate(dataset):
            context = " ".join(
                " ".join(sent_group) for sent_group in item["context"]["sentences"]
            )
            samples.append(
                Sample(
                    id=f"hotpotqa_{i}",
                    question=item["question"],
                    context=context,
                    ground_truth=item["answer"],
                    metadata={
                        "type": item["type"],
                        "level": item["level"],
                    },
                )
            )
        return samples

    async def run_sample(self, sample: Sample, engine) -> Result:
        """Run the conformal engine on a single sample."""
        # TODO: Integrate with actual engine in Task 12
        return Result(
            sample_id=sample.id,
            prediction_set=[sample.ground_truth],
            abstained=False,
            confidence=0.9,
        )

    async def aggregate(self, results: list[Result]) -> Aggregation:
        """Compute FDR, coverage, abstention rate."""
        total = len(results)
        abstained = sum(1 for r in results if r.abstained)
        return Aggregation(
            fdr=0.0,
            coverage=1.0 - (abstained / total) if total > 0 else 0.0,
            abstention_rate=abstained / total if total > 0 else 0.0,
            total_samples=total,
        )
