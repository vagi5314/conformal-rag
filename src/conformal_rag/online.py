"""Online Adaptive Conformal Inference (ACI), quantile-level.

Implements the Gibbs & Candès (2021) ACI idea: maintain a running quantile
level lambda_t over a window of recent scores and nudge it so the long-run
error rate approaches the target alpha.

Score convention (must match scoring.py): HIGH score = MORE faithful.
The engine abstains when score < threshold. The error ("miscoverage") event
is approving a wrong answer: score >= threshold and is_correct == False.
Under exchangeability this yields approximate alpha-level control
(heuristic convergence, not a hard finite-sample guarantee).
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
        initial_level: float = 0.1,
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

        # Error under the CURRENT threshold: we approved (score >= threshold)
        # but the answer was actually wrong.
        approved = score >= self._threshold
        error = 1.0 if (approved and not is_correct) else 0.0

        # ACI: too many approved mistakes -> raise the quantile level, which
        # raises the threshold (window scores are high=faithful), which makes
        # us abstain more. Too few mistakes -> relax.
        self.level = _clamp(self.level + self.eta * (error - self.alpha))

        # Recompute the threshold as the lambda_t quantile of the window.
        if len(self.window) >= 2:
            self._threshold = float(np.quantile(np.array(self.window), self.level))

    def should_abstain(self, score: float) -> bool:
        # Low faithfulness score -> abstain (matches engine.predict).
        return score < self._threshold

    def get_stats(self) -> dict:
        return {
            "threshold": self._threshold,
            "level": self.level,
            "window_size": len(self.window),
            "n_updates": self._n_updates,
        }
