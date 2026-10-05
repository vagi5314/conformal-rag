"""FDR monitoring and drift detection."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class PredictionRecord:
    """Single LABELED prediction record for monitoring."""

    predicted_hallucination: bool
    actual_hallucination: bool
    timestamp: float = field(default_factory=time.time)


class FDRMonitor:
    """Monitors the false-alarm rate of the abstention rule in production.

    Unlabeled predictions (every /predict) are counted for totals and the
    abstention rate. The false-alarm rate is computed ONLY from records whose
    true label arrived via feedback — never from assumed labels.
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
        self._n_predictions = 0
        self._n_abstained = 0

    def record_prediction(self, abstained: bool) -> None:
        """Count an unlabeled prediction; the true label may arrive later."""
        self._n_predictions += 1
        if abstained:
            self._n_abstained += 1

    def record(
        self,
        predicted_hallucination: bool,
        actual_hallucination: bool,
    ) -> None:
        """Record a prediction whose true label is known."""
        self._n_predictions += 1
        if predicted_hallucination:
            self._n_abstained += 1
        self.records.append(
            PredictionRecord(
                predicted_hallucination=predicted_hallucination,
                actual_hallucination=actual_hallucination,
            )
        )

    def get_metrics(self) -> dict:
        """Get current monitoring metrics."""
        total = self._n_predictions
        if total == 0:
            return {
                "total_predictions": 0,
                "fdr": 0.0,
                "coverage": 1.0,
                "abstention_rate": 0.0,
                "labeled_feedback": 0,
            }

        predicted_hallucinations = sum(
            1 for r in self.records if r.predicted_hallucination
        )
        false_positives = sum(
            1 for r in self.records
            if r.predicted_hallucination and not r.actual_hallucination
        )

        fdr = false_positives / predicted_hallucinations if predicted_hallucinations > 0 else 0.0
        abstention_rate = self._n_abstained / total
        coverage = 1.0 - abstention_rate

        return {
            "total_predictions": total,
            "fdr": fdr,
            "coverage": coverage,
            "abstention_rate": abstention_rate,
            "labeled_feedback": len(self.records),
        }

    def is_violating(self) -> bool:
        """Check if the false-alarm rate is violating the bound."""
        metrics = self.get_metrics()
        return metrics["fdr"] > self.alpha * self.violation_threshold
