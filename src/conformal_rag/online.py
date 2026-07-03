"""Online Adaptive Conformal Inference (ACI)."""

from __future__ import annotations

from collections import deque


class OnlineConformal:
    """Adaptive Conformal Inference for streaming data.

    Handles distribution shift by adjusting threshold each round.
    Convergence: FDR → α as t → ∞ (under exchangeability).
    """

    def __init__(
        self,
        alpha: float = 0.05,
        window_size: int = 500,
        eta: float = 0.01,
        initial_threshold: float = 0.5,
    ):
        self.alpha = alpha
        self.window_size = window_size
        self.eta = eta
        self._threshold = initial_threshold
        self.window: deque = deque(maxlen=window_size)
        self._n_updates = 0

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        self._threshold = value

    def update(self, score: float, is_correct: bool) -> None:
        """Update threshold after observing a labeled sample.

        ACI update rule:
            α_{t+1} = α_t + η(α - I(rejected) * I(correct))

        Args:
            score: Non-conformity score (higher = less conforming).
            is_correct: Whether the prediction was correct (label=0).
        """
        self.window.append((score, is_correct))
        self._n_updates += 1

        # Compute empirical FDR from window
        fdr = self._compute_fdr()

        # ACI update: adjust threshold based on FDR error
        error = fdr - self.alpha
        self.threshold += self.eta * error

        # Clamp threshold to [0, 1]
        self.threshold = max(0.0, min(1.0, self.threshold))

    def _compute_fdr(self) -> float:
        """Compute False Discovery Rate from window.

        FDR = FP / (FP + TP) among rejected samples.
        """
        if len(self.window) < 10:
            return 0.0

        scores, labels = zip(*self.window)

        # Rejected = score > threshold (predicted hallucination)
        rejected_idx = [i for i, s in enumerate(scores) if s > self.threshold]
        if len(rejected_idx) == 0:
            return 0.0

        # False positives: rejected but actually correct (label=0)
        fp = sum(1 for i in rejected_idx if labels[i] is True)
        return fp / len(rejected_idx)

    def should_abstain(self, score: float) -> bool:
        """Return True if score exceeds threshold (risk too high)."""
        return score > self.threshold

    def get_stats(self) -> dict:
        """Get current statistics."""
        return {
            "threshold": self.threshold,
            "window_size": len(self.window),
            "n_updates": self._n_updates,
            "fdr": self._compute_fdr(),
        }
