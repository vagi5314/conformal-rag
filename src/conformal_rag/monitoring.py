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
        """Monitoring metrics.

        Two different rates are reported, and they answer different questions:

        ``approved_wrong_rate``
            The quantity the CRC guarantee actually bounds:
            ``E[P(approved AND wrong)] <= alpha``. Estimated over labeled
            feedback only, because an unlabeled prediction cannot contribute a
            true label. This is the number to plot against alpha.

        ``false_alarm_rate``
            Of the answers the system flagged as risky, how many were actually
            fine. Useful, but it is *not* the guarantee. Earlier versions
            returned this value under the name "fdr" while the UI captioned it
            as the guarantee, which was misleading.
        """
        total = self._n_predictions
        if total == 0:
            return {
                "total_predictions": 0,
                "approved_wrong_rate": 0.0,
                "false_alarm_rate": 0.0,
                "coverage": 1.0,
                "abstention_rate": 0.0,
                "labeled_feedback": 0,
            }

        labeled = len(self.records)
        flagged = sum(1 for r in self.records if r.predicted_hallucination)
        approved_wrong = sum(
            1
            for r in self.records
            if not r.predicted_hallucination and r.actual_hallucination
        )
        false_alarms = sum(
            1
            for r in self.records
            if r.predicted_hallucination and not r.actual_hallucination
        )

        abstention_rate = self._n_abstained / total

        return {
            "total_predictions": total,
            "approved_wrong_rate": approved_wrong / labeled if labeled else 0.0,
            "false_alarm_rate": false_alarms / flagged if flagged else 0.0,
            "coverage": 1.0 - abstention_rate,
            "abstention_rate": abstention_rate,
            "labeled_feedback": labeled,
        }

    def is_violating(self) -> bool:
        """Whether the observed approved-and-wrong rate exceeds the promise.

        Compares the guarantee quantity to alpha, with a 1.5x band so ordinary
        small-sample noise does not trip the flag.
        """
        metrics = self.get_metrics()
        return metrics["approved_wrong_rate"] > self.alpha * self.violation_threshold
