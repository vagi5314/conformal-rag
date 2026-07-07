"""HaluEval benchmark plugin.

HaluEval ships paired faithful / hallucinated answers, so we use the native
labels directly: ``right_answer`` (label 0) and ``answer`` (the hallucinated
output, label 1). This is the canonical hallucination-detection setting.
"""

from __future__ import annotations

from typing import Optional

from benchmarks import register_benchmark
from benchmarks._common import aggregate_results, run_engine_sample
from benchmarks.base import BaseBenchmark, LatencyTrace, Result, Sample

MAX_SAMPLES = 1000


@register_benchmark
class HaluEvalBenchmark(BaseBenchmark):
    """HaluEval: QA hallucination detection (native faithful/hallucinated pairs)."""

    name = "halueval"
    description = "Hallucinated vs faithful QA answers — native labels"

    async def load_dataset(self, limit: int = MAX_SAMPLES) -> list[Sample]:
        """Load HaluEval (qa split) and emit labeled faithful/hallucinated pairs."""
        from datasets import load_dataset

        dataset = load_dataset("Lin-Chen/HaluEval", "qa", split=f"train[:{limit}]")
        samples: list[Sample] = []
        for i, item in enumerate(dataset):
            question = item.get("question", "")
            context = item.get("knowledge", "")
            right = item.get("right_answer") or item.get("answer", "")
            wrong = item.get("answer") or item.get("right_answer", "")
            if not right or not wrong or right == wrong:
                continue
            samples.append(
                Sample(
                    id=f"halueval_{i}_faithful",
                    question=question,
                    context=context,
                    ground_truth=right,
                    metadata={"has_hallucination": False, "variant": "faithful"},
                )
            )
            samples.append(
                Sample(
                    id=f"halueval_{i}_hallucinated",
                    question=question,
                    context=context,
                    ground_truth=wrong,
                    metadata={"has_hallucination": True, "variant": "hallucinated"},
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
