"""FDR monitoring and drift detection."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class PredictionRecord:
    """Single prediction record for monitoring."""

    predicted_hallucination: bool
    actual_hallucination: bool
    timestamp: float = field(default_factory=time.time)


class FDRMonitor:
    """Monitors False Discovery Rate in production.

    Tracks predictions and detects when FDR exceeds the guaranteed bound.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        window_size: int = 1000,
        violation_threshold: float = 1.5,
    ):
        self.alpha = alpha
        self.window_size = window_size
        self.violation_threshold = violation_threshold
        self.records: deque = deque(maxlen=window_size)

    def record(
        self,
        predicted_hallucination: bool,
        actual_hallucination: bool,
    ) -> None:
        """Record a prediction for monitoring."""
        self.records.append(
            PredictionRecord(
                predicted_hallucination=predicted_hallucination,
                actual_hallucination=actual_hallucination,
            )
        )

    def get_metrics(self) -> dict:
        """Get current monitoring metrics."""
        if not self.records:
            return {
                "total_predictions": 0,
                "fdr": 0.0,
                "coverage": 1.0,
                "abstention_rate": 0.0,
            }

        total = len(self.records)
        predicted_hallucinations = sum(
            1 for r in self.records if r.predicted_hallucination
        )
        false_positives = sum(
            1 for r in self.records
            if r.predicted_hallucination and not r.actual_hallucination
        )

        fdr = false_positives / predicted_hallucinations if predicted_hallucinations > 0 else 0.0
        coverage = 1.0 - (predicted_hallucinations / total) if total > 0 else 1.0
        abstention_rate = predicted_hallucinations / total if total > 0 else 0.0

        return {
            "total_predictions": total,
            "fdr": fdr,
            "coverage": coverage,
            "abstention_rate": abstention_rate,
        }

    def is_violating(self) -> bool:
        """Check if FDR is violating the guarantee."""
        metrics = self.get_metrics()
        return metrics["fdr"] > self.alpha * self.violation_threshold
