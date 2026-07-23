"""Online Adaptive Conformal Inference (ACI), quantile-level.

Implements the Gibbs & Candès (2021) ACI idea: maintain a running quantile
level lambda_t of the nonconformity-score window and nudge lambda_t so that
the long-run miscoverage approaches the target alpha. Under exchangeability
this yields approximate alpha-level control (heuristic convergence, not a
hard finite-sample guarantee).
"""

from __future__ import annotations

from collections import deque

import numpy as np


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


class OnlineConformal:
    def __init__(
        self,
        alpha: float = 0.05,
        window_size: int = 500,
        eta: float = 0.01,
        initial_level: float = 0.9,
        initial_threshold: float = 0.5,
    ):
        self.alpha = alpha
        self.window_size = window_size
        self.eta = eta
        self.level = float(initial_level)
        self._threshold = float(initial_threshold)
        self.window: deque = deque(maxlen=window_size)
        self._n_updates = 0

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        self._threshold = float(value)

    def update(self, score: float, is_correct: bool) -> None:
        self.window.append(float(score))
        self._n_updates += 1

        # Miscoverage under the CURRENT threshold.
        rejected = score > self._threshold
        miscoverage = 1.0 if (rejected and is_correct) else 0.0

        # ACI: move the quantile level toward satisfying target miscoverage = alpha.
        self.level = _clamp(self.level + self.eta * (miscoverage - self.alpha))

        # Recompute the threshold as the lambda_t quantile of the window.
        if len(self.window) >= 2:
            self._threshold = float(np.quantile(np.array(self.window), self.level))

    def should_abstain(self, score: float) -> bool:
        return score > self._threshold

    def get_stats(self) -> dict:
        return {
            "threshold": self._threshold,
            "level": self.level,
            "window_size": len(self.window),
            "n_updates": self._n_updates,
        }
