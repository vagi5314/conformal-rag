"""Shared logic for benchmark plugins: engine invocation, hallucination
injection, and FDR aggregation.

Benchmarks evaluate the conformal engine's FDR guarantee:

    P(hallucination | accepted) <= alpha

where "accepted" means the engine did *not* abstain. A sample counts as a
false positive (FDR numerator) when it is accepted AND is actually a
hallucination (label = 1). Datasets that do not ship hallucination labels
are stress-tested with deterministic, offline hallucination injection so the
FDR estimate is non-trivial.
"""

from __future__ import annotations

import re
from typing import Optional

from benchmarks.base import Aggregation, CostTrace, LatencyTrace, Result, Sample

_ENTITY_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b")


def run_engine_sample(
    sample: Sample,
    engine,
    tracer: Optional[LatencyTrace] = None,
    cost_tracker: Optional[object] = None,
) -> Result:
    """Run the conformal engine on one sample and capture traces.

    The ground-truth / reference answer is passed as the *response* under
    test. For faithful samples this is the correct answer; for hallucinated
    samples it is the injected wrong answer. The engine returns a
    faithfulness risk score; a low score triggers abstention.
    """
    prediction = engine.predict(sample.question, sample.context, sample.ground_truth)
    breakdown = prediction.scoring_breakdown

    latency = LatencyTrace(
        nli_scoring_ms=breakdown.nli_latency_ms or 0.0,
        llm_judge_ms=breakdown.llm_latency_ms or 0.0,
        total_ms=prediction.latency_ms or 0.0,
    )
    if tracer is not None:
        tracer.record("nli_scoring_ms", latency.nli_scoring_ms)
        if breakdown.used_llm_judge and latency.llm_judge_ms is not None:
            tracer.record("llm_judge_ms", latency.llm_judge_ms)
        tracer.record("total_ms", latency.total_ms)

    cost_trace: Optional[CostTrace] = None
    if (
        cost_tracker is not None
        and breakdown.used_llm_judge
        and getattr(cost_tracker, "traces", None)
    ):
        cost_trace = cost_tracker.traces[-1]

    has_hallucination = bool(sample.metadata.get("has_hallucination", False))
    return Result(
        sample_id=sample.id,
        prediction_set=[] if prediction.abstained else [sample.ground_truth],
        abstained=prediction.abstained,
        confidence=1.0 - prediction.risk_score,
        latency_trace=latency,
        cost_trace=cost_trace,
        metadata={"has_hallucination": has_hallucination},
    )


def aggregate_results(results: list[Result]) -> Aggregation:
    """Compute FDR, coverage and abstention rate from per-sample results.

    FDR = (# accepted that are hallucinations) / (# accepted).
    Coverage = (# accepted) / total. Abstention = (# abstained) / total.
    """
    total = len(results)
    accepted = [r for r in results if not r.abstained]
    false_positives = sum(
        1 for r in accepted if r.metadata.get("has_hallucination", False)
    )
    fdr = false_positives / len(accepted) if accepted else 0.0
    abstained = sum(1 for r in results if r.abstained)

    latency_traces = [r.latency_trace for r in results if r.latency_trace is not None]
    cost_traces = [r.cost_trace for r in results if r.cost_trace is not None]

    return Aggregation(
        fdr=fdr,
        coverage=(len(accepted) / total) if total else 0.0,
        abstention_rate=(abstained / total) if total else 0.0,
        total_samples=total,
        latency_traces=latency_traces,
        cost_traces=cost_traces,
    )


def inject_hallucination(context: str, answer: str) -> str:
    """Deterministic, offline hard-negative generator.

    Picks a different capitalized entity from the same context as a plausible
    but wrong answer (the hardest negative for NLI faithfulness scoring).
    Falls back to a low-faithfulness sentence when no candidate exists.
    """
    candidates = [
        m.group(1)
        for m in _ENTITY_RE.finditer(context or "")
        if len(m.group(1)) >= 4
        and m.group(1).lower() not in answer.lower()
        and answer.lower() not in m.group(1).lower()
    ]
    if candidates:
        # Longest distinct candidate => most plausible wrong answer.
        return max(set(candidates), key=len)
    return f"{answer} is incorrect and the opposite is true."


def build_labeled_pair(
    base_id: str,
    question: str,
    context: str,
    answer: str,
    extra_metadata: Optional[dict] = None,
) -> list[Sample]:
    """Expand one dataset item into a faithful + hallucinated sample pair."""
    meta = dict(extra_metadata or {})
    faithful = Sample(
        id=f"{base_id}_faithful",
        question=question,
        context=context,
        ground_truth=answer,
        metadata={**meta, "has_hallucination": False, "variant": "faithful"},
    )
    hallucinated = Sample(
        id=f"{base_id}_hallucinated",
        question=question,
        context=context,
        ground_truth=inject_hallucination(context, answer),
        metadata={**meta, "has_hallucination": True, "variant": "hallucinated"},
    )
    return [faithful, hallucinated]
